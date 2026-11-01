import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal
from app.market.ingestion import import_market_listings_from_csv, import_market_listings_from_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Import comparable market listings from CSV or JSON.")
    parser.add_argument("--csv", help="Path to CSV file")
    parser.add_argument("--json", help="Path to JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing to the database")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.csv:
            text = Path(args.csv).read_text(encoding="utf-8")
            summary = import_market_listings_from_csv(db, text, dry_run=args.dry_run)
        elif args.json:
            payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
            summary = import_market_listings_from_json(db, payload, dry_run=args.dry_run)
        else:
            parser.error("Either --csv or --json is required")
            return 2
        print(summary.as_dict())
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())