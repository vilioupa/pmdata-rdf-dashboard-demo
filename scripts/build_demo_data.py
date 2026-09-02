#!/usr/bin/env python3
"""
build_demo_data.py

Builds a small demo_data/ sample from the full dataset: keeps only a few users
and downsamples heart_rate.json to ~1 reading per minute. Read-only on data/ —
it only ever writes inside demo_data/ (and only deletes there, with --force).

Usage:
  python build_demo_data.py --dry-run   # preview, no writes
  python build_demo_data.py             # build demo_data/ (skips existing users)
  python build_demo_data.py --force     # rebuild, replacing subfolders in demo_data/
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime

# --- Configuration ---
SOURCE_DIR = "data"              # full dataset (read-only)
DEST_DIR = "demo_data"           # the sample is written only here
USERS = ["P02", "P03", "P04"]    # users featured in the thesis findings
HR_FILENAME = "heart_rate.json"
SAMPLE_SECONDS = 60              # keep ~1 reading per this many seconds (0 = no downsampling)

# Accepted dateTime formats; add your own if needed.
DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%m/%d/%y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
]


def parse_dt(s):
    """Return a datetime, or None if the format isn't recognized."""
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


def downsample_hr(records, seconds):
    """Keep ~1 record per `seconds`. For each bucket keep the record with the
    smallest timestamp (closest to the minute start), so heart-rate timestamps
    still line up with step timestamps and the dashboard's fusion page works."""
    if not seconds or seconds <= 0:
        return list(records)

    # Make sure timestamps are parseable before doing anything.
    sample = next(
        (r.get("dateTime") for r in records if isinstance(r, dict) and r.get("dateTime")),
        None,
    )
    if sample is None or parse_dt(sample) is None:
        raise ValueError(
            f"[ERROR] Unrecognized dateTime format (e.g. {sample!r}). "
            f"Add the correct format to DATETIME_FORMATS and rerun."
        )

    best = {}  # bucket -> (ts_epoch, record)
    for r in records:
        if not isinstance(r, dict):
            continue
        dt = parse_dt(r.get("dateTime"))
        if dt is None:
            continue
        ts = int(dt.timestamp())
        bucket = ts // seconds
        current = best.get(bucket)
        if current is None or ts < current[0]:
            best[bucket] = (ts, r)

    # Sort by time so the output file stays chronological.
    return [rec for _, (_, rec) in sorted(best.items())]


def build(force, dry_run):
    src_root = os.path.abspath(SOURCE_DIR)
    dst_root = os.path.abspath(DEST_DIR)

    # Safety guards
    if not os.path.isdir(src_root):
        sys.exit(f"[ERROR] Source folder not found: {src_root}")
    if dst_root == src_root or dst_root.startswith(src_root + os.sep):
        sys.exit("[SAFETY] demo_data must not live inside data/. Aborting.")

    print(f"[INFO] Source (read-only): {src_root}")
    print(f"[INFO] Destination:        {dst_root}")
    print(f"[INFO] Users:              {USERS}")
    print(f"[INFO] Downsample HR:      {('~every %ds (grid-aligned)' % SAMPLE_SECONDS) if SAMPLE_SECONDS else 'OFF'}")
    if dry_run:
        print("[DRY-RUN] Nothing will be written or deleted.\n")

    if not dry_run:
        os.makedirs(dst_root, exist_ok=True)

    for user in USERS:
        s_user = os.path.join(src_root, user)
        d_user = os.path.join(dst_root, user)
        print(f"\n=== {user} ===")

        if not os.path.isdir(s_user):
            print(f"  [WARNING] {s_user} not found. Skipping.")
            continue

        if os.path.exists(d_user):
            if not force:
                print(f"  [SKIP] {d_user} already exists. Run with --force to replace it")
                print(f"         (deletes ONLY this subfolder inside demo_data/).")
                continue
            print(f"  [WARNING] Will be deleted and rebuilt: {d_user}")
            if not dry_run:
                shutil.rmtree(d_user)  # only ever inside demo_data/

        # Copy everything except heart_rate.json (that one is written downsampled).
        if not dry_run:
            shutil.copytree(s_user, d_user, ignore=shutil.ignore_patterns(HR_FILENAME))
        else:
            print(f"  [DRY-RUN] copytree {s_user} -> {d_user} (excluding {HR_FILENAME})")

        # Downsample heart_rate.json (read from source, write to the sample).
        s_hr = os.path.join(s_user, HR_FILENAME)
        d_hr = os.path.join(d_user, HR_FILENAME)
        if os.path.isfile(s_hr):
            with open(s_hr, "r", encoding="utf-8") as f:
                records = json.load(f)
            total = len(records)
            reduced = downsample_hr(records, SAMPLE_SECONDS)
            pct = f" (~{100 * len(reduced) / total:.1f}%)" if total else ""
            print(f"  heart_rate.json: {total} -> {len(reduced)} records{pct}")
            if not dry_run:
                with open(d_hr, "w", encoding="utf-8") as f:
                    json.dump(reduced, f)
        else:
            print(f"  [WARNING] {s_hr} not found (no downsampling).")

    print("\n[OK] Done.")
    print("     The full data/, rdf_output/ and the Docker volume were left untouched.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Safely build demo_data/ (read-only on data/).")
    ap.add_argument("--force", action="store_true",
                    help="Replace existing subfolders inside demo_data/.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would happen, without writing or deleting anything.")
    args = ap.parse_args()
    build(force=args.force, dry_run=args.dry_run)