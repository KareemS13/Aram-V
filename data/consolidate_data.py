"""
Annual Data Consolidation
=========================

Runs every January to merge the accumulated patch CSV into the main
COICOP Excel file. This keeps the primary data file up to date and
ensures the model has the full history for YoY predictions.

Over time this builds up:
  Year 1:  110 months  (2017-2026)
  Year 2:  122 months  (2017-2027)  -> YoY forecasts become fully reliable
  Year 3+: 134+ months -> model accuracy improves with more training data

Run:
    python data/consolidate_data.py
"""

import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_RAW  = BASE_DIR / "data_raw"
PATCH_CSV = DATA_RAW / "armstat_cpi_patch.csv"
COICOP_XL = DATA_RAW / "armstat_cpi_coicop.xlsx"
BACKUP_DIR = DATA_RAW / "backups"


def consolidate():
    print("=" * 55)
    print("Annual Data Consolidation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)

    if not PATCH_CSV.exists():
        print("No patch file found — nothing to consolidate.")
        return

    # Load patch
    patch = pd.read_csv(PATCH_CSV, index_col=0, parse_dates=True)
    patch.index = pd.to_datetime(patch.index).to_period("M").to_timestamp()
    print(f"Patch: {len(patch)} months "
          f"({patch.index.min().strftime('%b %Y')} -> {patch.index.max().strftime('%b %Y')})")

    # Load existing COICOP Excel via the loader
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from data.ingest_cpi import ArmStatCPILoader

    loader = ArmStatCPILoader(str(COICOP_XL))
    existing = loader.load()
    print(f"Existing Excel: {len(existing)} months "
          f"({existing.index.min().strftime('%b %Y')} -> {existing.index.max().strftime('%b %Y')})")

    # Find truly new months
    new_rows = patch[~patch.index.isin(existing.index)]
    if new_rows.empty:
        print("Excel already contains all patch months — nothing to merge.")
        return

    print(f"New months to consolidate: {[d.strftime('%b %Y') for d in new_rows.index]}")

    # Backup the original Excel before modifying
    BACKUP_DIR.mkdir(exist_ok=True)
    backup_name = f"armstat_cpi_coicop_backup_{datetime.now().strftime('%Y%m%d')}.xlsx"
    backup_path = BACKUP_DIR / backup_name
    shutil.copy2(COICOP_XL, backup_path)
    print(f"Backed up original to: backups/{backup_name}")

    # Merge and save as a clean wide-format Excel
    combined = pd.concat([existing, new_rows]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    out = combined.copy()
    out.index.name = "date"
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

    with pd.ExcelWriter(str(COICOP_XL), engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="MoM_pct", index=False)

    print(f"Saved consolidated Excel: {len(combined)} months total")

    # Archive the patch (keep it, rename with timestamp)
    archive_name = f"armstat_cpi_patch_{datetime.now().strftime('%Y%m%d')}.csv"
    archive_path = BACKUP_DIR / archive_name
    shutil.copy2(PATCH_CSV, archive_path)
    PATCH_CSV.unlink()
    print(f"Archived patch to: backups/{archive_name}")
    print("Patch file cleared — ready for next year's accumulation.")

    print("\n" + "=" * 55)
    print(f"Consolidation complete. Total history: {len(combined)} months")
    print(f"YoY coverage: {'Full 12-month rolling possible' if len(combined) >= 24 else 'Building up...'}")
    print("=" * 55)


if __name__ == "__main__":
    consolidate()
