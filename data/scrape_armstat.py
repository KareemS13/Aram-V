"""
ArmStat CPI Live Scraper
========================

Scrapes the latest monthly CPI publication from the Armenian Statistical
Committee website, extracts MoM% data for all COICOP divisions, and
appends new months to the local Excel files used by the pipeline.

How it works:
  1. Fetches the ArmStat publications page (?nid=82) to find the latest
     CPI publication link and its file URL
  2. Downloads the .7z archive using a session cookie (required)
  3. Extracts the Excel file inside
  4. Parses TABLE 4 (MoM% change) for COICOP headline and 12 divisions
  5. Checks which months are already in our local Excel — appends only new ones
  6. Also updates the headline CPI series (armstat_cpi_headline.xlsx)

Run:
    python data/scrape_armstat.py

Returns:
    0  — new data appended
    1  — already up to date
    2  — error
"""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data_raw"

COICOP_URL  = DATA_RAW / "armstat_cpi_coicop.xlsx"
HEADLINE_URL = DATA_RAW / "armstat_cpi_headline.xlsx"

ARMSTAT_BASE    = "https://www.armstat.am"
PUBLICATIONS_URL = f"{ARMSTAT_BASE}/en/?nid=82"

# COICOP 2-digit codes we track (code -> our column name)
COICOP_CODES = {
    "00": "cpi_headline",
    "01": "cp01_food",
    "02": "cp02_alc",
    "03": "cp03_clothing",
    "04": "cp04_housing",
    "05": "cp05_furnishings",
    "06": "cp06_health",
    "07": "cp07_transport",
    "08": "cp08_comms",
    "09": "cp09_recreation",
    "10": "cp10_education",
    "11": "cp11_restaurants",
    "12": "cp12_misc",
}

MONTH_MAP = {
    "I":    1,  "II":  2,  "III": 3,
    "IV":   4,  "V":   5,  "VI":  6,
    "VII":  7,  "VIII":8,  "IX":  9,
    "X":   10,  "XI": 11,  "XII": 12,
}


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


# ---------------------------------------------------------------------------
# 1. Find latest publication
# ---------------------------------------------------------------------------

def find_latest_publication(s: requests.Session) -> tuple[str, str]:
    """
    Scrape the publications listing page and return the (page_url, file_slug)
    for the most recent CPI publication.

    Returns (page_url, file_slug) where file_slug looks like 'cpi_03_2026-eng'
    """
    r = s.get(PUBLICATIONS_URL, timeout=15)
    r.raise_for_status()

    # Find CPI publication links — pattern: ?nid=82&id=XXXX
    # paired with "Consumer Price Index" text
    pub_links = re.findall(
        r'\?nid=82&amp;id=(\d+)[^"]*"[^>]*>([^<]*Consumer [Pp]rice [Ii]ndex[^<]*)<',
        r.text
    )
    if not pub_links:
        # Try simpler pattern
        pub_links = re.findall(r'\?nid=82&(?:amp;)?id=(\d+)', r.text)
        if not pub_links:
            raise RuntimeError("Could not find CPI publication links on ArmStat page.")
        # Take the first (most recent)
        pub_id = pub_links[0] if isinstance(pub_links[0], str) else pub_links[0][0]
    else:
        pub_id = pub_links[0][0]

    page_url = f"{ARMSTAT_BASE}/en/?nid=82&id={pub_id}"
    print(f"  Latest publication page: {page_url}")

    # Now fetch that page to find the .7z download link
    r2 = s.get(page_url, timeout=15)
    r2.raise_for_status()

    # Find English 7z download link
    file_links = re.findall(
        r'href=["\']([^"\']*cpi_\d+_\d{4}-eng\.7z)["\']',
        r2.text,
        re.IGNORECASE
    )
    if not file_links:
        raise RuntimeError(f"No English .7z download link found on {page_url}")

    # Resolve relative URL
    raw_link = file_links[0]
    if raw_link.startswith("../"):
        file_url = f"{ARMSTAT_BASE}/file/" + raw_link[len("../file/"):]
    elif raw_link.startswith("/"):
        file_url = ARMSTAT_BASE + raw_link
    elif raw_link.startswith("http"):
        file_url = raw_link
    else:
        file_url = f"{ARMSTAT_BASE}/file/article/" + raw_link.split("/")[-1]

    # Extract slug for logging: cpi_03_2026-eng
    slug = re.search(r'(cpi_\d+_\d{4}-eng)', file_url, re.IGNORECASE)
    slug_str = slug.group(1) if slug else file_url.split("/")[-1]
    print(f"  Download URL:  {file_url}")
    print(f"  File slug:     {slug_str}")

    return file_url, slug_str


# ---------------------------------------------------------------------------
# 2. Download and extract
# ---------------------------------------------------------------------------

def download_and_extract(s: requests.Session, file_url: str) -> Path:
    """Download .7z, extract, return path to the Excel file."""
    import py7zr

    print(f"  Downloading archive...")
    r = s.get(file_url, timeout=60, stream=True)
    r.raise_for_status()

    out_dir = Path(tempfile.mkdtemp())
    tmp_7z  = out_dir / "download.7z"

    with open(tmp_7z, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)

    print(f"  Downloaded {tmp_7z.stat().st_size / 1024:.0f} KB — extracting...")
    with py7zr.SevenZipFile(tmp_7z, "r") as z:
        z.extractall(out_dir)
    tmp_7z.unlink()

    xlsx_files = list(out_dir.glob("*.xlsx")) + list(out_dir.glob("*.xls"))
    if not xlsx_files:
        raise RuntimeError(f"No Excel file found in archive. Contents: {list(out_dir.iterdir())}")

    print(f"  Extracted: {xlsx_files[0].name}")
    return xlsx_files[0]


# ---------------------------------------------------------------------------
# 3. Parse TABLE 4 — MoM%
# ---------------------------------------------------------------------------

def parse_table4(xlsx_path: Path, slug: str) -> pd.DataFrame:
    """
    Parse TABLE 4 ('percentage change over 1 month') from the ArmStat
    CPI Excel publication.

    Returns a DataFrame with columns:
        date (DatetimeIndex, MS freq), cpi_headline, cp01_food, ..., cp12_misc
    """
    xl  = pd.ExcelFile(xlsx_path)
    raw = xl.parse("TABLE 4", header=None)

    # Extract year from slug: cpi_03_2026-eng -> 2026
    year_match = re.search(r'_(\d{4})', slug)
    year = int(year_match.group(1)) if year_match else datetime.now().year

    # Find the header row containing Roman numeral month labels
    header_row = None
    month_cols = {}   # col_index -> month_number
    for idx, row in raw.iterrows():
        vals = [str(v).strip() for v in row if pd.notna(v)]
        matches = {v: MONTH_MAP[v] for v in vals if v in MONTH_MAP}
        if matches:
            header_row = idx
            for col_i, cell in enumerate(row):
                cell_str = str(cell).strip()
                if cell_str in MONTH_MAP:
                    month_cols[col_i] = MONTH_MAP[cell_str]
            break

    if header_row is None or not month_cols:
        raise RuntimeError("Could not find month header row in TABLE 4.")

    # Find the code column (contains "00", "01", ..., "12")
    code_col = 0   # typically column 0

    records = []
    for idx in range(header_row + 1, len(raw)):
        row = raw.iloc[idx]
        code_raw = str(row.iloc[code_col]).strip()

        # Only keep 2-digit COICOP codes we track
        if code_raw not in COICOP_CODES:
            continue

        col_name = COICOP_CODES[code_raw]
        for col_i, month_num in month_cols.items():
            val = row.iloc[col_i]
            if pd.isna(val):
                continue
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue

            date = pd.Timestamp(year=year, month=month_num, day=1)
            records.append({"date": date, "col": col_name, "value": val})

    if not records:
        raise RuntimeError("No data extracted from TABLE 4.")

    df = (
        pd.DataFrame(records)
        .pivot_table(index="date", columns="col", values="value", aggfunc="first")
        .rename_axis(None, axis=1)
    )
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    df.sort_index(inplace=True)

    print(f"  Parsed {len(df)} month(s): {df.index[0].strftime('%b %Y')} "
          f"-> {df.index[-1].strftime('%b %Y')}")
    print(f"  Columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# 4. Append new months to local Excel files
# ---------------------------------------------------------------------------

def _load_existing_coicop() -> pd.DataFrame:
    """Load the existing COICOP Excel into a tidy long DataFrame."""
    from data.ingest_cpi import ArmStatCPILoader
    loader = ArmStatCPILoader(str(COICOP_URL))
    df = loader.load()
    return df


def _load_existing_headline() -> pd.Series:
    """Load the existing headline CPI series."""
    from data.ingest_cpi import load_cpi_headline
    return load_cpi_headline(str(HEADLINE_URL))


def append_new_data(new_df: pd.DataFrame) -> dict:
    """
    Compare new_df against existing data. Append only months not yet present.

    New months are written to a CSV patch file (data_raw/armstat_cpi_patch.csv)
    that the pipeline's loader merges at load time — this avoids overwriting
    the original Excel files, preserving a clean audit trail.

    Returns dict with keys: new_months (list), updated (bool)
    """
    patch_path = DATA_RAW / "armstat_cpi_patch.csv"

    # Load existing patch if any
    if patch_path.exists():
        existing_patch = pd.read_csv(patch_path, index_col=0, parse_dates=True)
        existing_patch.index = pd.to_datetime(existing_patch.index).to_period("M").to_timestamp()
    else:
        existing_patch = pd.DataFrame()

    # Load existing COICOP data to check which months are already covered
    try:
        existing = _load_existing_coicop()
        existing_dates = set(existing.index)
        if not existing_patch.empty:
            existing_dates |= set(existing_patch.index)
    except Exception:
        existing_dates = set()
        if not existing_patch.empty:
            existing_dates = set(existing_patch.index)

    truly_new = new_df[~new_df.index.isin(existing_dates)]

    if truly_new.empty:
        print("  Already up to date — no new months to append.")
        return {"new_months": [], "updated": False}

    new_months = [d.strftime("%b %Y") for d in truly_new.index]
    print(f"  New months to append: {new_months}")

    # Merge with existing patch and save
    if not existing_patch.empty:
        combined_patch = pd.concat([existing_patch, truly_new]).sort_index()
        combined_patch = combined_patch[~combined_patch.index.duplicated(keep="last")]
    else:
        combined_patch = truly_new.sort_index()

    combined_patch.index.name = "date"
    combined_patch.to_csv(patch_path)
    print(f"  Patch file saved -> {patch_path.name} ({len(combined_patch)} rows total)")

    return {"new_months": new_months, "updated": True}


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def run() -> int:
    """
    Main scrape routine. Returns exit code:
        0 = new data appended
        1 = already up to date
        2 = error
    """
    print("=" * 55)
    print("ArmStat CPI Scraper")
    print("=" * 55)

    try:
        import py7zr  # noqa: F401
    except ImportError:
        print("ERROR: py7zr not installed. Run: pip install py7zr")
        return 2

    s = _session()

    try:
        print("\n[1/4] Finding latest publication...")
        file_url, slug = find_latest_publication(s)

        print("\n[2/4] Downloading and extracting archive...")
        xlsx_path = download_and_extract(s, file_url)

        print("\n[3/4] Parsing TABLE 4 (MoM%)...")
        new_df = parse_table4(xlsx_path, slug)

        print("\n[4/4] Appending new data to local files...")
        result = append_new_data(new_df)

        # Cleanup temp files
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import shutil
            shutil.rmtree(xlsx_path.parent, ignore_errors=True)
        except Exception:
            pass

        print("\n" + "=" * 55)
        if result["new_months"]:
            print(f"Done. Appended: {', '.join(result['new_months'])}")
            return 0
        else:
            print("Done. Already up to date.")
            return 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(run())
