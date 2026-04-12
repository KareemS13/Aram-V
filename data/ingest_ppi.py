"""
Armenia PPI (Producer Price Index) Ingestion
=============================================

Loads the Producer Price Index from a local CSV file manually downloaded
from ArmStatBank.am.

How to get the data
-------------------
1. Open your browser and go to: https://armstatbank.am
   (browser handles the certificate; Python cannot due to SSL mismatch)
2. Navigate: ArmStatBank → 3. Prices → 3.3 Producer Price Index
   (or search "Producer Price Index" in the search box)
3. Select all months from Jan 2017 to latest available
4. Click Download → CSV
5. Save the file to:  data_raw/armstat_ppi.csv

Expected CSV format (any of these work):
  - First column: dates (YYYY-MM, YYYY-MM-DD, or MMM-YY)
  - One column with the PPI index value (base year 2017=100)
  - The loader auto-detects the value column

Output
------
pd.DataFrame with column 'ppi_mom' (MoM% change, computed from index)
DatetimeIndex at month-start frequency.
"""

import warnings
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PPI_CSV_PATH = BASE_DIR / "data_raw" / "armstat_ppi.csv"


def load_ppi(
    path: str | Path = PPI_CSV_PATH,
    start: str = "2017-01-01",
) -> pd.DataFrame | None:
    """
    Load PPI from the manually downloaded CSV and compute MoM%.

    Returns None (with a warning) if the file doesn't exist — the pipeline
    will continue without PPI rather than crashing.

    Returns
    -------
    pd.DataFrame with column 'ppi_mom' (MoM% change), or None if unavailable.
    """
    path = Path(path)

    if not path.exists():
        warnings.warn(
            f"PPI data not found at {path}.\n"
            "To add PPI to the model, download the data manually:\n"
            "  1. Open https://armstatbank.am in your browser\n"
            "  2. Navigate to: ArmStatBank → 3. Prices → Producer Price Index\n"
            "  3. Select Jan 2017 → latest, download as CSV\n"
            f"  4. Save to: {path}\n"
            "The pipeline will continue without PPI for now."
        )
        return None

    try:
        raw = pd.read_csv(path, header=0)
    except Exception as e:
        warnings.warn(f"PPI: failed to read {path}: {e}")
        return None

    # Auto-detect date column (first column usually)
    date_col = raw.columns[0]
    raw[date_col] = raw[date_col].astype(str).str.strip()

    dates = []
    for d in raw[date_col]:
        parsed = None
        for fmt in ("%Y-%m-%d", "%Y-%m", "%b-%y", "%b %Y", "%m/%Y", "%d/%m/%Y"):
            try:
                parsed = pd.to_datetime(d, format=fmt)
                break
            except (ValueError, TypeError):
                continue
        if parsed is None:
            try:
                parsed = pd.to_datetime(d)
            except Exception:
                parsed = pd.NaT
        dates.append(parsed)

    raw.index = pd.DatetimeIndex(dates)
    raw = raw.drop(columns=[date_col])
    raw = raw[raw.index.notna()].sort_index()

    # Auto-detect the PPI index column (largest-magnitude numeric column)
    numeric_cols = raw.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        # Try converting all columns
        for col in raw.columns:
            raw[col] = pd.to_numeric(
                raw[col].astype(str).str.replace(",", ""), errors="coerce"
            )
        numeric_cols = raw.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        warnings.warn(f"PPI: no numeric columns found in {path}. Check file format.")
        return None

    # Choose the column most likely to be the PPI index (values around 90-150)
    best_col = None
    for col in numeric_cols:
        vals = raw[col].dropna()
        if len(vals) > 0 and 50 <= vals.median() <= 300:
            best_col = col
            break
    if best_col is None:
        best_col = numeric_cols[0]

    ppi_index = raw[best_col].copy()
    ppi_index = ppi_index.resample("MS").mean()

    if start:
        ppi_index = ppi_index[ppi_index.index >= pd.Timestamp(start)]

    # Compute MoM% from index values
    ppi_mom = (ppi_index / ppi_index.shift(1) - 1) * 100
    ppi_mom.name = "ppi_mom"

    # Drop the first NaN row from differencing
    ppi_mom = ppi_mom.dropna()

    print(f"PPI: loaded {len(ppi_mom)} months "
          f"({ppi_mom.index.min().strftime('%Y-%m')} → "
          f"{ppi_mom.index.max().strftime('%Y-%m')}), "
          f"range {ppi_mom.min():.2f}% – {ppi_mom.max():.2f}%")

    return ppi_mom.to_frame()


if __name__ == "__main__":
    df = load_ppi()
    if df is not None:
        print(df.tail(12))
    else:
        print("No PPI data available.")
