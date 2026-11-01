import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal
from app.vehicle_specs.ingestion import import_vehicle_specifications


def main() -> int:
    parser = argparse.ArgumentParser(description="Import vehicle specifications from CSV.")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing to the database")
    args = parser.parse_args()

    csv_text = Path(args.csv).read_text(encoding="utf-8")
    db = SessionLocal()
    try:
        summary = import_vehicle_specifications(db, csv_text, dry_run=args.dry_run)
        print(summary.as_dict())
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())