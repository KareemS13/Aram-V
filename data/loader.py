"""
Master dataset builder.

Orchestrates the three ingestion modules (CPI, FX, commodities) and
left-joins them on a shared DatetimeIndex at month-start frequency.

Usage
-----
    from data.loader import build_master_df
    df = build_master_df(fred_api_key="...")
    print(df.tail())
"""

import warnings
import pandas as pd

from config import CPI_RAW_PATH, CPI_WEIGHTS_PATH, DATA_START
from data.ingest_cpi import ArmStatCPILoader, load_cpi_headline
from data.ingest_fx import load_fx
from data.ingest_commodities import load_commodities
from data.ingest_policy_rate import load_policy_rate
from data.ingest_ppi import load_ppi
from data.ingest_comtrade import load_trade_features

# Path to the headline CPI file (longer history: 1993-present)
import os as _os
_BASE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
CPI_HEADLINE_PATH = _os.path.join(_BASE, "data_raw", "armstat_cpi_headline.xlsx")


def build_master_df(
    cpi_path: str = CPI_RAW_PATH,
    cpi_headline_path: str | None = None,
    fred_api_key: str = "",
    comtrade_api_key: str = "",
    data_start: str = DATA_START,
    date_to: str | None = None,
) -> pd.DataFrame:
    """
    Load CPI, FX, and commodity data and merge into a single DataFrame.

    Parameters
    ----------
    cpi_path     : path to ArmStatBank CPI Excel file
    fred_api_key : FRED API key for commodity + FX cross-rate fill
    data_start   : first month to include (YYYY-MM-DD)
    date_to      : last month to include (None = latest available)

    Returns
    -------
    pd.DataFrame with DatetimeIndex (freq='MS') and columns:

    CPI (month-on-month % change, as published by ArmStat):
        cpi_headline, cp01_food, cp02_alc, cp03_clothing, cp04_housing,
        cp05_furnishings, cp06_health, cp07_transport, cp08_comms,
        cp09_recreation, cp10_education, cp11_restaurants, cp12_misc

    Exchange rates (AMD per 1 foreign unit, monthly mean):
        fx_usd_amd, fx_eur_amd, fx_rub_amd

    Commodities (USD, monthly):
        oil_wti, wheat, energy_idx, commodity_idx, nat_gas

    Monetary policy:
        cba_rate (CBA refinancing rate, %)
    """
    # ------------------------------------------------------------------
    # 1. CPI
    # ------------------------------------------------------------------
    print("Loading CPI data...")
    cpi_loader = ArmStatCPILoader(cpi_path)
    cpi_df = cpi_loader.load()
    print(f"  COICOP sub-indices: {len(cpi_df)} months, {cpi_df.columns.tolist()}")

    # Merge scraper patch file if it exists (new months appended by scrape_armstat.py)
    patch_path = _os.path.join(_os.path.dirname(cpi_path), "armstat_cpi_patch.csv")
    if _os.path.exists(patch_path):
        try:
            patch = pd.read_csv(patch_path, index_col=0, parse_dates=True)
            patch.index = pd.DatetimeIndex(patch.index).to_period("M").to_timestamp("MS")
            # Only keep columns that exist in cpi_df
            patch = patch[[c for c in patch.columns if c in cpi_df.columns]]
            # Only append rows not already in cpi_df
            new_rows = patch[~patch.index.isin(cpi_df.index)]
            if not new_rows.empty:
                cpi_df = pd.concat([cpi_df, new_rows]).sort_index()
                print(f"  Patch applied: +{len(new_rows)} months from scraper "
                      f"(now {len(cpi_df)} total)")
        except Exception as e:
            warnings.warn(f"Could not apply CPI patch: {e}")

    # Use the long headline series (1993-present) to replace the cpi_headline
    # column — this gives SARIMA more history for seasonal pattern estimation.
    headline_path = cpi_headline_path or CPI_HEADLINE_PATH
    if _os.path.exists(headline_path):
        try:
            long_headline = load_cpi_headline(headline_path)
            # Overwrite the cpi_headline column with the longer series
            cpi_df["cpi_headline"] = long_headline.reindex(cpi_df.index)
            print(f"  Headline CPI replaced with long series from {headline_path}")
        except Exception as e:
            warnings.warn(f"Could not load long headline series: {e}. Using COICOP file headline.")
    else:
        warnings.warn(f"Headline file not found at {headline_path}. Using COICOP file headline.")

    # ------------------------------------------------------------------
    # 2. Exchange rates
    # ------------------------------------------------------------------
    print("Loading exchange rates...")
    fx_df = load_fx(
        date_from=data_start,
        date_to=date_to,
        fred_api_key=fred_api_key,
    )
    print(f"  FX: {len(fx_df)} months, {fx_df.columns.tolist()}")

    # ------------------------------------------------------------------
    # 3. Commodities
    # ------------------------------------------------------------------
    print("Loading commodity prices...")
    comm_df = load_commodities(
        fred_api_key=fred_api_key,
        start=data_start,
        end=date_to,
    )
    print(f"  Commodities: {len(comm_df)} months, {comm_df.columns.tolist()}")

    # ------------------------------------------------------------------
    # 4. Policy rate
    # ------------------------------------------------------------------
    print("Loading CBA policy rate...")
    try:
        rate_df = load_policy_rate(start=data_start)
        print(f"  Policy rate: {len(rate_df)} months, range "
              f"{rate_df['cba_rate'].min():.2f}%–{rate_df['cba_rate'].max():.2f}%")
    except Exception as e:
        warnings.warn(f"Could not load CBA policy rate: {e}")
        rate_df = None

    # ------------------------------------------------------------------
    # 5. Merge on shared DatetimeIndex
    # ------------------------------------------------------------------
    # Use CPI as the base (left join) — it defines the time range
    master = cpi_df.copy()
    master = master.join(fx_df, how="left")
    master = master.join(comm_df, how="left")
    if rate_df is not None:
        master = master.join(rate_df, how="left")

    # ------------------------------------------------------------------
    # 6. PPI (optional — loads from data_raw/armstat_ppi.csv if present)
    # ------------------------------------------------------------------
    try:
        ppi_df = load_ppi(start=data_start)
        if ppi_df is not None:
            master = master.join(ppi_df, how="left")
            print(f"  PPI: merged {ppi_df['ppi_mom'].notna().sum()} months")
    except Exception as e:
        warnings.warn(f"Could not load PPI: {e}")

    # ------------------------------------------------------------------
    # 7. Trade data (UN Comtrade — optional, uses cache after first fetch)
    # ------------------------------------------------------------------
    _comtrade_key = comtrade_api_key or _os.environ.get("COMTRADE_API_KEY", "")
    if _comtrade_key:
        try:
            print("Loading trade data (UN Comtrade)...")
            trade_df = load_trade_features(
                api_key=_comtrade_key,
                start="2010-01-01",   # longer history than CPI — gets trimmed at join
                use_cache=True,
            )
            if trade_df is not None:
                master = master.join(trade_df, how="left")
                # Fill NaN trade MoM% with 0 (no-change) — avoids index gaps that break
                # skforecast. Gaps occur during data-scarce periods (e.g., 2022 sanctions).
                trade_cols = [c for c in trade_df.columns if c in master.columns]
                n_filled = master[trade_cols].isna().sum().sum()
                if n_filled > 0:
                    master[trade_cols] = master[trade_cols].fillna(0)
                    import warnings as _w2
                    _w2.warn(f"Trade features: filled {n_filled} NaN values with 0 "
                             "(no-change assumption for missing months).")
                n_trade = trade_df.notna().any(axis=1).sum()
                print(f"  Trade: merged {n_trade} months, {len(trade_df.columns)} features")
        except Exception as e:
            warnings.warn(f"Could not load trade data: {e}")
    else:
        warnings.warn("Comtrade: no API key — trade features skipped. "
                      "Set COMTRADE_API_KEY in .env to enable.")

    # Apply date filter
    if data_start:
        master = master[master.index >= pd.Timestamp(data_start)]
    if date_to:
        master = master[master.index <= pd.Timestamp(date_to)]

    master = master.sort_index()
    # Ensure MS frequency is set (joining external series can lose it)
    master.index = pd.DatetimeIndex(master.index, freq="MS")

    # ------------------------------------------------------------------
    # 7. Validate
    # ------------------------------------------------------------------
    _validate(master)

    print(f"\nMaster dataset: {len(master)} months "
          f"({master.index.min().strftime('%Y-%m')} to "
          f"{master.index.max().strftime('%Y-%m')}), "
          f"{master.shape[1]} columns.")

    return master


def _validate(df: pd.DataFrame) -> None:
    """Warn about gaps, coverage issues, and unexpected nulls."""
    # Check for month gaps in index
    expected_range = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    missing_months = expected_range.difference(df.index)
    if len(missing_months) > 0:
        warnings.warn(
            f"Master dataset is missing {len(missing_months)} months: "
            f"{[str(m.date()) for m in missing_months[:5]]}"
            f"{'...' if len(missing_months) > 5 else ''}"
        )

    # Report null counts per column
    null_counts = df.isna().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        warnings.warn(
            "Master dataset has missing values:\n" +
            "\n".join(f"  {col}: {n} nulls" for col, n in null_cols.items())
        )

    # Warn if CPI headline has any nulls (it's the target — must be complete)
    if "cpi_headline" in df.columns and df["cpi_headline"].isna().any():
        n = df["cpi_headline"].isna().sum()
        warnings.warn(
            f"WARNING: cpi_headline (target variable) has {n} missing values. "
            "These rows will be dropped during model training."
        )
