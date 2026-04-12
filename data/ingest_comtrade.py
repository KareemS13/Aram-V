"""
UN Comtrade Trade Data Ingestion
=================================

Fetches monthly Armenia import/export data from the UN Comtrade API v1.
Trade flows are predictive of CPI because:
  - Energy imports (HS 27) → fuel/heating costs → cp04_housing, cp07_transport
  - Food imports (HS 02,04,10) → food price pass-through → cp01_food
  - Total import value → overall imported inflation pressure
  - Import price index (value/weight) → unit price trends

API: https://comtradeapi.un.org/data/v1/get/C/M/HS
Auth: Ocp-Apim-Subscription-Key header (free subscription)
Rate limit: 1 req/sec, 10K req/hour

Armenia reporter code: 51
"""

import time
import warnings
from pathlib import Path

import pandas as pd
import requests

BASE_DIR  = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data_raw" / "comtrade_cache"

COMTRADE_URL = "https://comtradeapi.un.org/data/v1/get/C/M/HS"

# Armenia UN reporter code
ARMENIA_CODE = "51"

# HS chapter codes → our feature names
# Each entry: (hs_code, flow, feature_name)
TRADE_SERIES = [
    ("TOTAL", "M", "imports_total"),      # Total imports (USD)
    ("TOTAL", "X", "exports_total"),      # Total exports (USD)
    ("27",    "M", "imports_energy"),     # Mineral fuels, oil, gas
    ("10",    "M", "imports_cereals"),    # Cereals (wheat, barley)
    ("02",    "M", "imports_meat"),       # Meat
    ("04",    "M", "imports_dairy"),      # Dairy
    ("31",    "M", "imports_fertilizer"),# Fertilizers → food production cost
]

# Max periods per API request (monthly data, 1 year max per call)
PERIODS_PER_CALL = 12


def _make_period_list(start: str, end: str) -> list[str]:
    """Generate list of YYYYMM period strings between start and end."""
    idx = pd.date_range(
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        freq="MS",
    )
    return [d.strftime("%Y%m") for d in idx]


def _fetch_one(
    session: requests.Session,
    api_key: str,
    cmd_code: str,
    flow_code: str,
    periods: list[str],
) -> pd.DataFrame:
    """
    Fetch one HS chapter for a batch of periods (max 12).

    Returns DataFrame with columns: period (YYYYMM), primaryValue, netWgt
    Aggregated across all partner countries (world total).
    """
    period_str = ",".join(periods)
    params = {
        "reporterCode": ARMENIA_CODE,
        "period":       period_str,
        "flowCode":     flow_code,
        "cmdCode":      cmd_code,
        "partnerCode":  "0",     # World total (partner=0)
        "includeDesc":  "false",
    }
    headers = {"Ocp-Apim-Subscription-Key": api_key}

    try:
        r = session.get(COMTRADE_URL, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Sum all sub-commodity rows per period to get the chapter total
        agg = (
            df.groupby("period")[["primaryValue", "netWgt"]]
            .sum()
            .reset_index()
        )
        return agg

    except Exception as e:
        warnings.warn(f"Comtrade fetch failed (cmd={cmd_code}, flow={flow_code}, "
                      f"periods={periods[:2]}...): {e}")
        return pd.DataFrame()


def fetch_trade_data(
    api_key: str,
    start: str = "2010-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch all trade series for Armenia from UN Comtrade.

    Returns
    -------
    pd.DataFrame with DatetimeIndex (MS freq) and columns:
        imports_total, exports_total, imports_energy,
        imports_cereals, imports_meat, imports_dairy, imports_fertilizer
        (all in USD thousands; _upi suffix = unit price index)
    """
    if end is None:
        # Don't request current month (data not yet published)
        end = (pd.Timestamp.now() - pd.DateOffset(months=2)).strftime("%Y-%m-%d")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"arm_trade_{start[:4]}_{pd.Timestamp(end).strftime('%Y%m')}.csv"

    if use_cache and cache_file.exists():
        print(f"Comtrade: loading from cache ({cache_file.name})")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp()
        return df

    all_periods = _make_period_list(start, end)
    # Split into annual batches (API limit: 12 months per call)
    batches = [all_periods[i:i+PERIODS_PER_CALL]
               for i in range(0, len(all_periods), PERIODS_PER_CALL)]

    session = requests.Session()
    session.headers["User-Agent"] = "Armenia-CPI-Pipeline/1.0"

    frames = {}
    total_calls = len(TRADE_SERIES) * len(batches)
    call_n = 0

    for cmd_code, flow_code, feat_name in TRADE_SERIES:
        series_rows = []

        for batch in batches:
            call_n += 1
            print(f"  Comtrade [{call_n}/{total_calls}]: {feat_name} "
                  f"{batch[0]}–{batch[-1]}")

            batch_df = _fetch_one(session, api_key, cmd_code, flow_code, batch)
            if not batch_df.empty:
                series_rows.append(batch_df)

            # Respect rate limit: 1 req/sec
            time.sleep(1.1)

        if series_rows:
            combined = pd.concat(series_rows, ignore_index=True)
            combined["date"] = pd.to_datetime(
                combined["period"].astype(str), format="%Y%m"
            )
            combined = combined.set_index("date").sort_index()
            frames[feat_name] = combined["primaryValue"]
            print(f"  {feat_name}: {len(combined)} months fetched")

    if not frames:
        raise RuntimeError("Comtrade: no data fetched. Check API key and connection.")

    df = pd.DataFrame(frames)
    df = df.resample("MS").sum()
    df = df[df.index >= pd.Timestamp(start)]

    # Convert to USD millions for readability
    df = df / 1_000_000
    df.columns = [c + "_musd" for c in df.columns]

    # Compute MoM% changes (these are what go into the model as features)
    for col in list(df.columns):
        base = col.replace("_musd", "")
        df[f"{base}_mom"] = (df[col] / df[col].shift(1) - 1) * 100

    # Save cache
    df.to_csv(cache_file)
    print(f"Comtrade: cached to {cache_file.name}")

    print(f"Comtrade: {len(df)} months "
          f"({df.index.min().strftime('%Y-%m')} → {df.index.max().strftime('%Y-%m')}), "
          f"{len(df.columns)} columns")
    return df


def load_trade_features(
    api_key: str,
    start: str = "2010-01-01",
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """
    Load trade features for use in the pipeline.

    Returns MoM% change columns only (level columns dropped).
    Returns None with warning if API key missing.
    """
    if not api_key:
        warnings.warn("Comtrade: no API key — skipping trade features.")
        return None

    try:
        df = fetch_trade_data(api_key=api_key, start=start, use_cache=use_cache)
        # Keep only MoM% columns for the model
        mom_cols = [c for c in df.columns if c.endswith("_mom")]
        result = df[mom_cols].copy()
        # Drop first row (NaN from differencing)
        result = result.dropna(how="all")
        return result
    except Exception as e:
        warnings.warn(f"Comtrade: failed to load trade features: {e}")
        return None


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("COMTRADE_API_KEY", "")
    df = fetch_trade_data(api_key=key, start="2010-01-01", use_cache=False)
    print(df.tail(6))
