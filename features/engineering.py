"""
Feature matrix construction for the inflation forecasting pipeline.

Takes the master DataFrame from data/loader.py and produces:
  - X: feature matrix (lags, rolling stats, calendar, dummies, interactions)
  - y: target series (cpi_headline MoM%)

All features are constructed to avoid look-ahead bias:
  - Lags use only past values
  - Rolling windows use only past values (min_periods respected)
  - Dummies are deterministic calendar indicators
"""

import numpy as np
import pandas as pd

from config import (
    LAG_CONFIG,
    LOG_DIFF_COLS,
    STRUCTURAL_BREAKS,
)
from features.transforms import log_diff_df


def build_feature_matrix(
    master_df: pd.DataFrame,
    lag_config: dict | None = None,
    target_col: str = "cpi_headline",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the full feature matrix X and target series y.

    Parameters
    ----------
    master_df   : output of data.loader.build_master_df()
    lag_config  : dict mapping column names -> list of lag integers
                  (defaults to config.LAG_CONFIG)
    target_col  : name of the target column in master_df

    Returns
    -------
    X : pd.DataFrame, feature matrix (rows with NaN from lag creation dropped)
    y : pd.Series, target (cpi_headline MoM%)
    """
    if lag_config is None:
        lag_config = LAG_CONFIG

    df = master_df.copy()

    # ------------------------------------------------------------------
    # 1. Log-difference non-stationary columns
    #    Renames: fx_usd_amd -> fx_usd_amd_ld, oil_wti -> oil_wti_ld, etc.
    # ------------------------------------------------------------------
    ld_cols = [c for c in LOG_DIFF_COLS if c in df.columns]
    df = log_diff_df(df, ld_cols)

    # Update lag_config keys to use _ld suffix where applicable
    ld_rename = {c: f"{c}_ld" for c in ld_cols}
    updated_lag_config = {}
    for col, lags in lag_config.items():
        mapped = ld_rename.get(col, col)
        if mapped in df.columns:
            updated_lag_config[mapped] = lags
        elif col in df.columns:
            updated_lag_config[col] = lags

    # ------------------------------------------------------------------
    # 2. Lag features
    # ------------------------------------------------------------------
    lag_frames = []
    for col, lags in updated_lag_config.items():
        if col not in df.columns:
            continue
        for lag in lags:
            lag_col = df[col].shift(lag)
            lag_col.name = f"{col}_lag{lag}"
            lag_frames.append(lag_col)

    # ------------------------------------------------------------------
    # 3. Rolling features
    #    3-month rolling mean of oil and wheat (smooths short-term volatility)
    #    12-month rolling std of CPI (captures volatility regime)
    # ------------------------------------------------------------------
    rolling_frames = []

    for raw_col in ["oil_wti", "wheat"]:
        ld_col = f"{raw_col}_ld"
        if ld_col in df.columns:
            roll = df[ld_col].shift(1).rolling(window=3, min_periods=2).mean()
            roll.name = f"{ld_col}_roll3m"
            rolling_frames.append(roll)

    if "cpi_headline" in df.columns:
        roll_std = df["cpi_headline"].shift(1).rolling(window=12, min_periods=6).std()
        roll_std.name = "cpi_vol_12m"
        rolling_frames.append(roll_std)

    # ------------------------------------------------------------------
    # 4. Calendar / seasonal features
    # ------------------------------------------------------------------
    month = df.index.month.astype(float)
    cal_df = pd.DataFrame(index=df.index)
    cal_df["month_sin"]  = np.sin(2 * np.pi * month / 12)
    cal_df["month_cos"]  = np.cos(2 * np.pi * month / 12)
    cal_df["is_q1"]      = (df.index.month <= 3).astype(int)
    cal_df["is_summer"]  = df.index.month.isin([6, 7, 8]).astype(int)
    cal_df["year_trend"] = df.index.year - df.index.year.min()

    # ------------------------------------------------------------------
    # 5. Structural break dummies
    # ------------------------------------------------------------------
    dummy_df = pd.DataFrame(index=df.index)
    for name, (start, end) in STRUCTURAL_BREAKS.items():
        dummy_df[name] = (
            (df.index >= pd.Timestamp(start)) &
            (df.index <= pd.Timestamp(end))
        ).astype(int)

    # ------------------------------------------------------------------
    # 6. Interaction: RUB depreciation x Ukraine shock
    #    Captures sign reversal in FX pass-through during 2022
    # ------------------------------------------------------------------
    interaction_frames = []
    rub_ld = "fx_rub_amd_ld"
    if rub_ld in df.columns and "ukraine_2022" in dummy_df.columns:
        interaction = df[rub_ld] * dummy_df["ukraine_2022"]
        interaction.name = "rub_ukraine_interaction"
        interaction_frames.append(interaction)

    # ------------------------------------------------------------------
    # 7. Combine all features
    # ------------------------------------------------------------------
    feature_parts = (
        lag_frames +
        rolling_frames +
        [cal_df, dummy_df] +
        interaction_frames
    )

    X = pd.concat(feature_parts, axis=1)
    X = X.sort_index()

    # ------------------------------------------------------------------
    # 8. Target variable
    # ------------------------------------------------------------------
    y = df[target_col].copy()
    y.name = "cpi_headline_mom"

    # ------------------------------------------------------------------
    # 9. Drop rows with NaN (from lag creation or log-diff)
    #    Keep only rows where both X and y are complete
    # ------------------------------------------------------------------
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    # Restore MS frequency if index is contiguous (lost by concat/boolean indexing)
    try:
        X.index = pd.DatetimeIndex(X.index, freq="MS")
        y.index = pd.DatetimeIndex(y.index, freq="MS")
    except ValueError:
        # Index has gaps — infer and warn
        inferred = pd.infer_freq(X.index)
        import warnings as _w
        _w.warn(f"Feature matrix index is not contiguous MS (inferred: {inferred}). "
                "Check for gaps in input data.")

    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} features "
          f"({X.index.min().strftime('%Y-%m')} -> {X.index.max().strftime('%Y-%m')})")

    return X, y


def get_exog_for_sarima(
    X: pd.DataFrame,
    exog_cols: list[str],
) -> pd.DataFrame:
    """
    Extract the SARIMAX exogenous columns from the full feature matrix.

    Handles partial matches: e.g. 'fx_usd_amd_lag1' will match
    'fx_usd_amd_ld_lag1' if the exact name is absent (due to _ld suffix).
    """
    result_cols = []
    for col in exog_cols:
        if col in X.columns:
            result_cols.append(col)
        else:
            # Try with _ld suffix inserted
            ld_col = col.replace("_lag", "_ld_lag")
            if ld_col in X.columns:
                result_cols.append(ld_col)
            else:
                import warnings
                warnings.warn(f"SARIMA exog column '{col}' not found in feature matrix.")

    return X[result_cols]
