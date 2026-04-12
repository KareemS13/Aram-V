"""
Global commodity price ingestion.

Primary source: FRED API (fredapi package).
  Series: oil (WTI), wheat, aluminum, broad commodity index.

Fallback: World Bank Pink Sheet Excel file.
  Download from:
  https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/related/CMO-Historical-Data-Monthly.xlsx
  Save to data_raw/CMO-Historical-Data-Monthly.xlsx

Both loaders return a DataFrame with DatetimeIndex at month-start frequency.
"""

import warnings
import numpy as np
import pandas as pd

from config import FRED_COMMODITY_SERIES, WB_PINKSHEET_PATH


# World Bank Pink Sheet column names → our internal names
WB_COL_MAP = {
    "Crude oil, average":            "oil_wti",
    "Crude oil, Brent":              "oil_wti",   # fallback if WTI absent
    "Wheat, US HRW":                 "wheat",
    "Wheat, US SRW":                 "wheat",
    "Aluminum":                      "aluminum",
    "Commodity Price Index, nominal":"commodity_idx",
    "Energy index":                  "commodity_idx",
}


class FREDCommodityLoader:
    """Fetch commodity prices from FRED API."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "FRED_API_KEY is required. Set it in your .env file or pass --fred-key."
            )
        self.api_key = api_key
        self._fred = None

    def _get_fred(self):
        if self._fred is None:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.api_key)
            except ImportError:
                raise ImportError("fredapi not installed. Run: pip install fredapi")
        return self._fred

    def fetch_all(
        self,
        start: str = "2017-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch all commodity series defined in config.FRED_COMMODITY_SERIES.

        Returns
        -------
        pd.DataFrame with DatetimeIndex (MS) and columns:
            oil_wti, wheat, aluminum, commodity_idx
        """
        fred = self._get_fred()
        frames = {}

        for col_name, series_id in FRED_COMMODITY_SERIES.items():
            try:
                series = fred.get_series(
                    series_id,
                    observation_start=start,
                    observation_end=end,
                )
                series.name = col_name
                frames[col_name] = series
                print(f"FRED: fetched {series_id} ({col_name}), {len(series)} obs.")
            except Exception as e:
                warnings.warn(f"FRED: could not fetch {series_id}: {e}")

        if not frames:
            raise RuntimeError("FRED: failed to fetch any commodity series.")

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)

        # FRED series may be monthly already or need resampling
        if not isinstance(df.index.freq, pd.tseries.offsets.MonthBegin):
            df = df.resample("MS").mean()

        df = df.sort_index()
        return df


class WBPinkSheetLoader:
    """Load commodity prices from World Bank Pink Sheet Excel file."""

    def __init__(self, path: str = WB_PINKSHEET_PATH):
        self.path = path

    def load(self, start: str = "2017-01-01") -> pd.DataFrame:
        """
        Parse WB Pink Sheet Excel file.

        The file has dates as row headers in 'MMM-YY' format and
        commodity names as column headers, typically starting around row 5.

        Returns
        -------
        pd.DataFrame with DatetimeIndex (MS) and columns matching our names.
        """
        try:
            raw = pd.read_excel(self.path, header=None, dtype=str)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"World Bank Pink Sheet not found at {self.path}. "
                "Download from: https://thedocs.worldbank.org/en/doc/"
                "18675f1d1639c7a34d463f59263ba0a2-0050012025/related/"
                "CMO-Historical-Data-Monthly.xlsx"
            )

        raw = raw.fillna("")

        # Find the header row (contains commodity names)
        header_row = None
        for i, row in raw.iterrows():
            vals = [str(v).strip() for v in row]
            # WB Pink Sheet has "Crude oil" or "Wheat" in the header
            if any("crude" in v.lower() or "wheat" in v.lower() for v in vals):
                header_row = i
                break

        if header_row is None:
            raise ValueError(f"Could not find commodity header row in {self.path}")

        header = [str(v).strip() for v in raw.iloc[header_row]]
        data = raw.iloc[header_row + 1:].copy().reset_index(drop=True)

        # Parse date column (first column, format 'MMM-YY')
        date_col = data.iloc[:, 0].astype(str).str.strip()
        dates = []
        for d in date_col:
            try:
                # pd.to_datetime handles 'Jan-23' style with dayfirst=False
                dates.append(pd.to_datetime(d, format="%b-%y"))
            except ValueError:
                try:
                    dates.append(pd.to_datetime(d))
                except Exception:
                    dates.append(pd.NaT)

        data.index = pd.DatetimeIndex(dates)
        data = data.iloc[:, 1:]  # drop date column
        data.columns = header[1:]

        # Map WB column names to our internal names
        rename = {}
        for wb_col, our_col in WB_COL_MAP.items():
            for col in data.columns:
                if wb_col.lower() in col.lower() and our_col not in rename.values():
                    rename[col] = our_col
                    break

        data = data.rename(columns=rename)
        keep = [c for c in data.columns if c in set(WB_COL_MAP.values())]
        data = data[keep]

        # Convert to numeric
        for col in data.columns:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(",", ""), errors="coerce"
            )

        data = data[data.index.notna()].sort_index()
        data = data.resample("MS").mean()

        if start:
            data = data[data.index >= pd.Timestamp(start)]

        print(f"WB Pink Sheet: loaded {len(data)} months from {self.path}")
        return data


def load_commodities(
    fred_api_key: str = "",
    start: str = "2017-01-01",
    end: str | None = None,
    wb_path: str = WB_PINKSHEET_PATH,
) -> pd.DataFrame:
    """
    Load commodity prices, trying FRED first and WB Pink Sheet as fallback.

    Returns
    -------
    pd.DataFrame with DatetimeIndex (MS) and columns:
        oil_wti, wheat, aluminum, commodity_idx
    """
    if fred_api_key:
        try:
            loader = FREDCommodityLoader(api_key=fred_api_key)
            df = loader.fetch_all(start=start, end=end)
            # Fill any gaps with WB Pink Sheet if available
            if df.isna().any().any():
                try:
                    wb = WBPinkSheetLoader(wb_path).load(start=start)
                    df = df.combine_first(wb)
                    print("Commodities: filled FRED gaps with WB Pink Sheet.")
                except Exception:
                    pass
            return df
        except Exception as e:
            warnings.warn(f"FRED commodity fetch failed ({e}). Trying WB Pink Sheet.")

    # Fallback: WB Pink Sheet only
    wb_loader = WBPinkSheetLoader(wb_path)
    df = wb_loader.load(start=start)
    return df
