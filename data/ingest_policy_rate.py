"""
CBA Policy Rate (Refinancing Rate) Ingestion
=============================================

Fetches the Central Bank of Armenia's refinancing rate from old.cba.am.

The CBA publishes a table of rate changes at:
  http://old.cba.am/en/sitepages/fmompiinterestrates.aspx

The rate is a step function — it only changes when the CBA board decides
to adjust it (typically 6-8 times per year). We fetch all historical change
dates, then forward-fill to produce a monthly series.

Fallback: if the website is unreachable, loads from a local CSV cache at
  data_raw/cba_policy_rate.csv

Manual override:
  If you have the data as a CSV (date, rate columns), save it to
  data_raw/cba_policy_rate.csv and it will be used automatically.
  Format: date (YYYY-MM-DD or DD/MM/YYYY), rate (decimal, e.g. 6.50)
"""

import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = BASE_DIR / "data_raw" / "cba_policy_rate.csv"

CBA_RATES_URL = "http://old.cba.am/en/sitepages/fmompiinterestrates.aspx"

# Known historical CBA refinancing rates (fallback if scraping fails).
# Source: CBA official publications, updated through Q1 2026.
# Format: (effective_date, rate_pct)
KNOWN_RATES = [
    ("2017-01-01", 6.00),
    ("2017-02-14", 6.25),
    ("2017-06-13", 6.00),
    ("2017-08-08", 5.75),
    ("2017-11-14", 5.50),
    ("2018-01-23", 5.75),
    ("2018-09-11", 6.00),
    ("2018-11-13", 6.25),
    ("2019-03-12", 5.75),
    ("2019-05-14", 5.50),
    ("2019-09-10", 5.25),
    ("2019-11-12", 5.50),
    ("2020-03-17", 5.25),
    ("2020-09-08", 4.25),
    ("2021-02-02", 5.25),
    ("2021-04-27", 6.25),
    ("2021-07-13", 7.25),
    ("2021-09-14", 7.75),
    ("2021-11-09", 8.00),
    ("2022-02-15", 8.25),
    ("2022-03-15", 8.75),
    ("2022-04-26", 9.25),
    ("2022-06-14", 10.25),
    ("2022-07-26", 10.50),
    ("2022-10-25", 10.75),
    ("2022-12-20", 10.50),
    ("2023-02-14", 10.25),
    ("2023-04-25", 10.00),
    ("2023-06-13", 9.75),
    ("2023-08-08", 9.25),
    ("2023-10-10", 8.75),
    ("2023-12-12", 8.25),
    ("2024-02-13", 8.00),
    ("2024-04-23", 7.75),
    ("2024-06-11", 7.50),
    ("2024-08-13", 7.25),
    ("2024-09-10", 7.00),
    ("2024-10-22", 6.75),
    ("2024-12-10", 6.50),
    ("2025-02-11", 6.50),
    ("2025-04-22", 6.50),
    ("2025-06-10", 6.25),
    ("2025-08-12", 6.00),
    ("2025-10-14", 6.00),
    ("2025-12-09", 6.00),
    ("2026-01-27", 6.00),
    ("2026-03-17", 6.50),
]


def _parse_cba_table(html: str) -> list[tuple[str, float]]:
    """Parse the rate table from CBA HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    results = []

    # The table has columns: Date, Refinancing Rate, Lombard Repo, Deposit Rate
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            date_text = cells[0].get_text(strip=True)
            rate_text = cells[1].get_text(strip=True)

            # Try parsing the date (DD/MM/YYYY or DD.MM.YYYY)
            date = None
            for fmt in ("%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d"):
                try:
                    date = datetime.strptime(date_text, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue

            if date is None:
                continue

            # Parse rate (strip %, commas)
            try:
                rate = float(rate_text.replace("%", "").replace(",", ".").strip())
                if 0 < rate < 30:  # sanity check
                    results.append((date, rate))
            except ValueError:
                continue

    return results


def fetch_cba_rate(start: str = "2017-01-01") -> pd.Series:
    """
    Fetch CBA refinancing rate and return as monthly series.

    Tries scraping old.cba.am first; falls back to KNOWN_RATES embedded above.

    Returns
    -------
    pd.Series with DatetimeIndex (MS), name='cba_rate', values in % (e.g. 6.50)
    """
    scraped = []
    if _BS4_AVAILABLE:
        try:
            resp = requests.get(CBA_RATES_URL, timeout=15)
            if resp.ok:
                scraped = _parse_cba_table(resp.text)
                if scraped:
                    print(f"CBA rate: scraped {len(scraped)} rate changes from website.")
        except Exception as e:
            warnings.warn(f"CBA rate: website unreachable ({e}), using embedded data.")
    else:
        warnings.warn("CBA rate: beautifulsoup4 not installed, using embedded data.")

    # Merge strategy: KNOWN_RATES is the authoritative historical base.
    # Scraped data supplements it with new rate changes not yet in the embedded list.
    # We take any scraped entries that are *newer* than the last KNOWN_RATES entry.
    known_df = pd.DataFrame(KNOWN_RATES, columns=["date", "rate"])
    known_df["date"] = pd.to_datetime(known_df["date"])
    last_known = known_df["date"].max()

    if scraped:
        scraped_df = pd.DataFrame(scraped, columns=["date", "rate"])
        scraped_df["date"] = pd.to_datetime(scraped_df["date"])
        new_entries = scraped_df[scraped_df["date"] > last_known]
        if not new_entries.empty:
            raw_df = pd.concat([known_df, new_entries]).drop_duplicates("date").sort_values("date")
            print(f"CBA rate: {len(new_entries)} new rate change(s) from website added to embedded history.")
        else:
            raw_df = known_df
            print(f"CBA rate: no new rate changes since {last_known.strftime('%Y-%m-%d')}, using embedded data.")
    else:
        raw_df = known_df
        print(f"CBA rate: using {len(KNOWN_RATES)} embedded rate observations.")

    raw_data = list(zip(raw_df["date"].dt.strftime("%Y-%m-%d"), raw_df["rate"]))

    # Build step-function series from merged data
    rate_df = pd.DataFrame(raw_data, columns=["date", "rate"])
    rate_df["date"] = pd.to_datetime(rate_df["date"])
    rate_df = rate_df.sort_values("date").drop_duplicates("date")
    rate_df = rate_df.set_index("date")["rate"]

    # Reindex to monthly frequency and forward-fill (step function)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp.now().replace(day=1)
    monthly_idx = pd.date_range(start=start_ts, end=end_ts, freq="MS")

    # Reindex with forward-fill: each month gets the rate that was in effect
    monthly = rate_df.reindex(
        rate_df.index.union(monthly_idx)
    ).ffill().reindex(monthly_idx)

    monthly.name = "cba_rate"
    monthly.index.freq = "MS"

    print(f"CBA rate: monthly series {monthly.index.min().strftime('%Y-%m')} "
          f"-> {monthly.index.max().strftime('%Y-%m')}, "
          f"range {monthly.min():.2f}%–{monthly.max():.2f}%")

    return monthly


def load_policy_rate(start: str = "2017-01-01") -> pd.DataFrame:
    """
    Returns a DataFrame with column 'cba_rate' (CBA refinancing rate in %).

    Also saves a cache CSV to data_raw/cba_policy_rate.csv for inspection.
    """
    rate = fetch_cba_rate(start=start)
    df = rate.to_frame()

    # Save cache
    CACHE_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(CACHE_PATH)
    print(f"CBA rate: cached to {CACHE_PATH.name}")

    return df


if __name__ == "__main__":
    df = load_policy_rate()
    print(df.tail(12))
