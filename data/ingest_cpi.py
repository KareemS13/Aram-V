"""
ArmStatBank CPI loader.

Reads the Excel file downloaded from:
  statbank.armstat.am → 1. Economy and finances → 1.2 Consumer Prices
  → 1.2.1 Monthly indicators → "PRICE INDEXES by years and months,
    grow rate to the previous month (2digit level by COICOP)"

The table contains month-on-month % change values for 13 COICOP categories
(00 = headline + 01–12 sub-indices), with years as rows and months as columns
(or the transpose, depending on export). This loader handles both orientations.

Output: pd.DataFrame with DatetimeIndex (freq='MS') and one column per
COICOP category using the short codes defined in config.COICOP_MAP.
The values are already MoM % changes as published by ArmStat (no further
transformation needed).
"""

import re
import warnings
import numpy as np
import pandas as pd

from config import COICOP_MAP, COICOP_LABELS


MONTH_ABBR = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


class ArmStatCPILoader:
    """Load and parse ArmStatBank COICOP price-index Excel file."""

    def __init__(self, path: str):
        self.path = path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Returns a DataFrame with:
          - DatetimeIndex at month-start frequency ('MS')
          - Columns: cpi_headline, cp01_food, ..., cp12_misc
          - Values: month-on-month % change (as published by ArmStat)

        Handles two formats:
          1. Original ArmStat export (complex multi-header layout)
          2. Consolidated flat format (date col + COICOP columns)
             written by data/consolidate_data.py after annual merge
        """
        # Peek at the first row to detect format
        peek = pd.read_excel(self.path, header=0, nrows=2)
        if "date" in [str(c).lower().strip() for c in peek.columns]:
            # Flat consolidated format
            df = pd.read_excel(self.path, header=0, parse_dates=["date"])
            df = df.rename(columns={"date": "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
            df = df.set_index("date").sort_index()
            df.index.name = None
            return df

        # Original ArmStat format
        raw = pd.read_excel(self.path, header=None, dtype=str)
        raw = raw.fillna("")

        # Try to detect orientation: years-as-rows or years-as-columns
        df = self._parse(raw)
        df = self._clean(df)
        df = self._align_frequency(df)
        return df

    def load_weights(self, weights_path: str) -> pd.Series:
        """
        Load CPI basket weights from a separate ArmStat weights Excel file.
        Returns a pd.Series indexed by COICOP short codes, values = weight (0–100).
        """
        raw = pd.read_excel(weights_path, header=None, dtype=str).fillna("")
        weights = {}
        for _, row in raw.iterrows():
            for val in row:
                val = str(val).strip()
                # Look for rows containing a 2-digit COICOP code
                for code, col in COICOP_MAP.items():
                    if val.startswith(code) or val == code:
                        # The numeric value should be nearby in the same row
                        nums = [x for x in row if self._is_float(x)]
                        if nums:
                            weights[col] = float(nums[-1])
                        break
        return pd.Series(weights, name="weight")

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Detect table orientation and extract a tidy DataFrame:
          rows = months, columns = COICOP categories (short codes)
        """
        # Strategy 1: years in first column, months in header row
        df = self._try_years_as_rows(raw)
        if df is not None and len(df) >= 12:
            return df

        # Strategy 2: months in first column, COICOP categories in header
        df = self._try_months_as_rows(raw)
        if df is not None and len(df) >= 12:
            return df

        raise ValueError(
            f"Could not parse CPI Excel file at {self.path}. "
            "Expected either years-as-rows or months-as-rows orientation."
        )

    def _try_years_as_rows(self, raw: pd.DataFrame) -> pd.DataFrame | None:
        """
        Expected shape: years down the left, months across the top,
        one sheet per COICOP category — OR a single wide sheet with
        (year, month) row multi-index and COICOP categories as columns.

        ArmStat PX-Web exports typically produce:
          Row 0..N: metadata/headers
          Then: year | Jan | Feb | Mar | ... | Dec
        where each group of 12 values is one COICOP category (stacked vertically
        or repeated across columns).

        This handles the common case where all COICOP categories are in one sheet
        with a column for the category code.
        """
        # Find the row that contains month names
        month_row_idx = None
        for i, row in raw.iterrows():
            vals = [str(v).strip().lower() for v in row]
            matches = sum(1 for v in vals if v in MONTH_ABBR)
            if matches >= 6:
                month_row_idx = i
                break

        if month_row_idx is None:
            return None

        header = [str(v).strip() for v in raw.iloc[month_row_idx]]
        data_rows = raw.iloc[month_row_idx + 1:].copy()

        # Find the year column (column that contains 4-digit years)
        year_col = None
        for col_idx, col in enumerate(data_rows.columns):
            col_vals = data_rows[col].astype(str).str.strip()
            years = col_vals[col_vals.str.match(r"^\d{4}$")]
            if len(years) >= 3:
                year_col = col_idx
                break

        if year_col is None:
            return None

        # Find COICOP category column
        coicop_col = None
        for col_idx, col in enumerate(data_rows.columns):
            if col_idx == year_col:
                continue
            col_vals = data_rows[col].astype(str).str.strip()
            matches = sum(1 for v in col_vals if any(
                v.startswith(code) for code in COICOP_MAP
            ))
            if matches >= 5:
                coicop_col = col_idx
                break

        # Find month columns from header
        month_cols = {}
        for col_idx, h in enumerate(header):
            h_lower = h.lower()
            if h_lower in MONTH_ABBR:
                month_cols[MONTH_ABBR[h_lower]] = col_idx

        if len(month_cols) < 6:
            return None

        # Build tidy records
        records = []
        current_year = None
        current_coicop = None

        for _, row in data_rows.iterrows():
            year_val = str(row.iloc[year_col]).strip()
            if re.match(r"^\d{4}$", year_val):
                current_year = int(year_val)

            if coicop_col is not None:
                coicop_val = str(row.iloc[coicop_col]).strip()
                matched = self._match_coicop(coicop_val)
                if matched:
                    current_coicop = matched

            if current_year is None or current_coicop is None:
                continue

            for month_num, col_idx in month_cols.items():
                val = str(row.iloc[col_idx]).strip()
                if self._is_float(val):
                    records.append({
                        "date": pd.Timestamp(current_year, month_num, 1),
                        "coicop": current_coicop,
                        "value": float(val),
                    })

        if not records:
            return None

        df = pd.DataFrame(records)
        df = df.pivot_table(index="date", columns="coicop", values="value", aggfunc="first")
        df.columns.name = None
        return df

    def _try_months_as_rows(self, raw: pd.DataFrame) -> pd.DataFrame | None:
        """
        Handles the orientation where rows are (year, month) combinations
        and columns are COICOP categories. Common in the PX-Web table view export.

        Expected structure:
          Col 0: year (e.g. 2017, 2017, ..., 2018, ...)
          Col 1: month name (e.g. January, February, ...)
          Col 2+: COICOP category values
        """
        # Find header row with COICOP codes
        header_row_idx = None
        for i, row in raw.iterrows():
            vals = [str(v).strip() for v in row]
            coicop_hits = sum(1 for v in vals if any(
                v.startswith(code) for code in COICOP_MAP
            ))
            if coicop_hits >= 3:
                header_row_idx = i
                break

        if header_row_idx is None:
            # Try looking for month names as row indicators
            month_col = None
            year_col = None
            for col in raw.columns:
                col_vals = raw[col].astype(str).str.strip().str.lower()
                if col_vals.isin(MONTH_ABBR).sum() >= 6:
                    month_col = col
                elif col_vals.str.match(r"^\d{4}$").sum() >= 3:
                    year_col = col
            if month_col is None:
                return None
            header_row_idx = 0

        header = [str(v).strip() for v in raw.iloc[header_row_idx]]
        data = raw.iloc[header_row_idx + 1:].copy().reset_index(drop=True)

        # Map header columns to COICOP short codes
        col_to_coicop = {}
        for col_idx, h in enumerate(header):
            matched = self._match_coicop(h)
            if matched:
                col_to_coicop[col_idx] = matched

        if len(col_to_coicop) < 3:
            return None

        # Find year and month columns
        year_col_idx = None
        month_col_idx = None
        for col_idx in range(min(5, len(header))):
            if col_idx in col_to_coicop:
                continue
            col_vals = data.iloc[:, col_idx].astype(str).str.strip()
            if col_vals.str.match(r"^\d{4}$").sum() >= 3:
                year_col_idx = col_idx
            elif col_vals.str.lower().isin(MONTH_ABBR).sum() >= 6:
                month_col_idx = col_idx

        records = []
        current_year = None

        for _, row in data.iterrows():
            if year_col_idx is not None:
                y = str(row.iloc[year_col_idx]).strip()
                if re.match(r"^\d{4}$", y):
                    current_year = int(y)

            if month_col_idx is not None:
                m_str = str(row.iloc[month_col_idx]).strip().lower()
                month_num = MONTH_ABBR.get(m_str)
            else:
                month_num = None

            if current_year is None or month_num is None:
                continue

            date = pd.Timestamp(current_year, month_num, 1)
            row_data = {"date": date}
            for col_idx, coicop_col in col_to_coicop.items():
                val = str(row.iloc[col_idx]).strip()
                if self._is_float(val):
                    row_data[coicop_col] = float(val)

            if len(row_data) > 1:
                records.append(row_data)

        if not records:
            return None

        df = pd.DataFrame(records).set_index("date")
        df = df[~df.index.duplicated(keep="first")]
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _match_coicop(self, text: str) -> str | None:
        """Match a cell value to a COICOP short code."""
        text = text.strip()
        # Try numeric code prefix first (e.g. "01", "01 Food...")
        for code, col in COICOP_MAP.items():
            if text.startswith(code):
                return col
        # Try label substring match
        text_lower = text.lower()
        for col, label in COICOP_LABELS.items():
            if label.lower()[:10] in text_lower:
                return col
        return None

    @staticmethod
    def _is_float(val: str) -> bool:
        try:
            float(str(val).replace(",", "."))
            return True
        except (ValueError, TypeError):
            return False

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns not in COICOP_MAP values, fill small gaps."""
        valid_cols = set(COICOP_MAP.values())
        df = df[[c for c in df.columns if c in valid_cols]]

        df = df.sort_index()

        # Drop months where more than half the COICOP columns are NaN
        # (catches ArmStat template rows for unpublished future months)
        min_valid = max(1, len(df.columns) // 2)
        df = df[df.notna().sum(axis=1) >= min_valid]

        # Drop isolated rows that are non-contiguous with the main series
        # (e.g. a spurious November 2026 row when only Jan-Feb 2026 exist)
        if len(df) > 12:
            # Keep only rows within the contiguous date range from the start
            # up to the last month where cpi_headline (or majority) is present
            headline_col = "cpi_headline" if "cpi_headline" in df.columns else df.columns[0]
            last_valid = df[headline_col].last_valid_index()
            if last_valid is not None:
                # Check for gaps: if there's a gap > 2 months before last_valid,
                # truncate at the last contiguous block
                valid_idx = df[df[headline_col].notna()].index
                if len(valid_idx) >= 2:
                    diffs = pd.Series(valid_idx).diff().dt.days.dropna()
                    # Find first large gap (> 60 days = more than 2 months)
                    large_gaps = diffs[diffs > 60]
                    if not large_gaps.empty:
                        # Truncate at the last contiguous run before the big gap
                        gap_pos = large_gaps.index[0]
                        cutoff_date = valid_idx[gap_pos - 1]
                        df = df[df.index <= cutoff_date]
                        warnings.warn(
                            f"CPI loader: truncated series at {cutoff_date.strftime('%Y-%m')} "
                            f"due to gap in data (likely unpublished future months in template)."
                        )

        # Warn if any column is missing
        missing = valid_cols - set(df.columns)
        if missing:
            warnings.warn(f"CPI loader: missing COICOP columns: {sorted(missing)}")

        return df

    def _align_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DatetimeIndex at month-start frequency."""
        df.index = pd.to_datetime(df.index)
        df = df.asfreq("MS")
        return df


def load_cpi(path: str) -> pd.DataFrame:
    """Convenience wrapper."""
    return ArmStatCPILoader(path).load()


def load_cpi_headline(path: str) -> pd.Series:
    """
    Load the headline CPI index file (1993-2026, compared to previous month).

    Structure:
      - Col 0: month name (January ... December)
      - Cols 1+: years (1993, 1994, ..., 2026)
      - Values: CPI index where previous month = 100
      - MoM % change = value - 100

    Returns pd.Series named 'cpi_headline' with DatetimeIndex (freq=MS).
    Values are MoM % change.
    """
    raw = pd.read_excel(path, header=None, dtype=str).fillna("")

    # Find header row containing years
    year_row_idx = None
    for i, row in raw.iterrows():
        vals = [str(v).strip() for v in row]
        year_hits = sum(1 for v in vals if re.match(r"^\d{4}$", v))
        if year_hits >= 5:
            year_row_idx = i
            break

    if year_row_idx is None:
        raise ValueError(f"Could not find year header row in {path}")

    header = [str(v).strip() for v in raw.iloc[year_row_idx]]
    data_rows = raw.iloc[year_row_idx + 1:].copy()

    # Map column index → year
    col_to_year = {}
    for col_idx, h in enumerate(header):
        if re.match(r"^\d{4}$", h):
            col_to_year[col_idx] = int(h)

    # Find month column (col 0)
    records = []
    for _, row in data_rows.iterrows():
        month_str = str(row.iloc[0]).strip().lower()
        month_num = MONTH_ABBR.get(month_str)
        if month_num is None:
            continue

        for col_idx, year in col_to_year.items():
            val = str(row.iloc[col_idx]).strip()
            # Skip ".." (unpublished) and empty
            if val in ("..", "", "nan"):
                continue
            try:
                index_val = float(val.replace(",", "."))
                mom_pct = index_val - 100.0  # convert index to MoM%
                records.append({
                    "date":  pd.Timestamp(year, month_num, 1),
                    "value": mom_pct,
                })
            except ValueError:
                continue

    if not records:
        raise ValueError(f"No data parsed from headline CPI file: {path}")

    series = (
        pd.DataFrame(records)
        .set_index("date")["value"]
        .sort_index()
    )
    series.name = "cpi_headline"
    # Normalize to month-start
    series.index = pd.DatetimeIndex(series.index).normalize() + pd.offsets.MonthBegin(0)
    series = series[~series.index.duplicated(keep="first")]
    series = series.asfreq("MS")

    print(f"Headline CPI: loaded {series.notna().sum()} months "
          f"({series.index.min().strftime('%Y-%m')} to {series.index.max().strftime('%Y-%m')})")
    return series
