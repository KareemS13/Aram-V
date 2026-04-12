"""
Stationarity transformations for time series features.

- log_diff: convert price levels to approximate % changes
- adf_check: Augmented Dickey-Fuller test with warning if non-stationary
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats


def log_diff(series: pd.Series, fill_na: bool = True) -> pd.Series:
    """
    Compute log first-difference of a price series.

    log_diff(x)_t = ln(x_t) - ln(x_{t-1})
                  ≈ (x_t - x_{t-1}) / x_{t-1}   (% change approximation)

    Parameters
    ----------
    series   : pd.Series of price levels (must be positive)
    fill_na  : if True, the first NaN (from differencing) is left as NaN

    Returns
    -------
    pd.Series of log-differences, same index as input
    """
    s = series.copy()
    # Guard against zero/negative values before log
    if (s <= 0).any():
        s = s.where(s > 0)
        warnings.warn(
            f"log_diff: series '{series.name}' has non-positive values; "
            "these will become NaN."
        )
    result = np.log(s).diff()
    result.name = series.name
    return result


def log_diff_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply log_diff to specified columns of a DataFrame, in-place copy."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = log_diff(df[col])
            df = df.rename(columns={col: f"{col}_ld"})
    return df


def adf_check(series: pd.Series, alpha: float = 0.05) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    Parameters
    ----------
    series : pd.Series (NaNs dropped internally)
    alpha  : significance level (default 0.05)

    Returns
    -------
    dict with keys: stationary (bool), p_value (float), adf_stat (float)
    """
    from statsmodels.tsa.stattools import adfuller

    clean = series.dropna()
    if len(clean) < 12:
        warnings.warn(f"ADF: '{series.name}' has fewer than 12 obs — test unreliable.")
        return {"stationary": None, "p_value": None, "adf_stat": None}

    result = adfuller(clean, autolag="AIC")
    adf_stat, p_value = result[0], result[1]
    stationary = p_value < alpha

    if not stationary:
        warnings.warn(
            f"ADF: '{series.name}' may be non-stationary (p={p_value:.3f}). "
            "Consider applying log_diff or differencing."
        )

    return {"stationary": stationary, "p_value": p_value, "adf_stat": adf_stat}


def check_all_stationarity(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Run ADF test on multiple columns and return a summary DataFrame.

    Returns
    -------
    pd.DataFrame with columns: feature, stationary, p_value, adf_stat
    """
    results = []
    for col in cols:
        if col in df.columns:
            r = adf_check(df[col])
            r["feature"] = col
            results.append(r)
    return pd.DataFrame(results)[["feature", "stationary", "p_value", "adf_stat"]]
