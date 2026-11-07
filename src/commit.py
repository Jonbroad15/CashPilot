"""Step 3: validate work/new.csv, append to master.csv (with backup), archive inbox files.

Usage: python src/commit.py [--no-archive]
"""
import argparse
import datetime
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import MASTER_COLS, OWNERS, drop_known, load_yaml, path


def commit(new_csv, master_csv, inbox, archive_root, archive_files=True):
    cfg = load_yaml("config.yaml")
    new = pd.read_csv(new_csv, dtype={"category": str, "owner": str}).fillna({"category": "", "owner": ""})
    bad_cat = new[~new["category"].isin(cfg["categories"])]
    bad_own = new[~new["owner"].isin(OWNERS)]
    if len(bad_cat) or len(bad_own):
        print(bad_cat.head(20).to_string() if len(bad_cat) else bad_own.head(20).to_string())
        raise SystemExit(f"ERROR: {len(bad_cat)} rows with missing/unknown category, "
                         f"{len(bad_own)} with bad owner. Fix work/new.csv and re-run.")
    master = pd.read_csv(master_csv)
    new_rows = drop_known(new, master)  # makes commit idempotent
    if len(new_rows):
        os.makedirs(archive_root, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy(master_csv, os.path.join(archive_root, f"master.{stamp}.csv"))
        out = pd.concat([master, new_rows[MASTER_COLS]], ignore_index=True)
        out = out.sort_values("date", kind="stable")
        out.to_csv(master_csv, index=False)
    print(f"appended {len(new_rows)} rows; master now has {len(master) + len(new_rows)}")
    if archive_files and "source_file" in new:
        dest = os.path.join(archive_root, datetime.datetime.now().strftime("%Y-%m"))
        os.makedirs(dest, exist_ok=True)
        for f in new["source_file"].dropna().unique():
            src = os.path.join(inbox, f)
            if os.path.exists(src):
                shutil.move(src, os.path.join(dest, f))
        print(f"statements archived to {dest}")
    return len(new_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default=path("work", "new.csv"))
    ap.add_argument("--master", default=path("master.csv"))
    ap.add_argument("--inbox", default=path("inbox"))
    ap.add_argument("--archive", default=path("archive"))
    ap.add_argument("--no-archive", action="store_true")
    a = ap.parse_args()
    commit(a.new, a.master, a.inbox, a.archive, not a.no_archive)
