from __future__ import annotations

import os
from pathlib import Path
import argparse

from peergrading.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="PeerGrading weekly pipeline (local run).")
    parser.add_argument("--students", default=os.getenv("STUDENTS_DB_PATH"), help="Path to students_db (csv/xlsx).")
    parser.add_argument("--forms", default=os.getenv("FORMS_EXPORT_PATH"), help="Path to Forms export (xlsx/csv).")
    parser.add_argument("--week-setup", default=os.getenv("WEEK_SETUP_PATH"), help="Path to weekly presentations.csv.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    parser.add_argument("--week", default=os.getenv("WEEK_ID"), help="Override week id, e.g., 2026-W09.")
    parser.add_argument("--tz", default=os.getenv("LOCAL_TZ", "Europe/Zurich"), help="Local timezone for deadline.")
    args = parser.parse_args()

    if not args.students or not args.forms or not args.week_setup:
        raise SystemExit("Missing required paths. Set .env or pass --students/--forms/--week-setup.")

    out_path = run_pipeline(
        students_db_path=args.students,
        forms_export_path=args.forms,
        week_setup_path=args.week_setup,
        out_dir=args.out,
        week_id=args.week,
        local_tz=args.tz,
    )
    print(f"✅ Done. Outputs written to: {out_path}")

if __name__ == "__main__":
    main()
