import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.db.session import SessionLocal
from app.market.acquisition import (
    LocalPermittedDatasetSource,
    discover_market_listings,
    import_discovered_listings,
)
from app.pricing.market import MarketVehicleCriteria


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover and import comparable market listings from a permitted source.")
    parser.add_argument("--input", help="Path to a permitted CSV or JSON dataset")
    parser.add_argument("--format", choices=["csv", "json"], default=None)
    parser.add_argument("--make", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--fuel", default=None)
    parser.add_argument("--transmission", default=None)
    parser.add_argument("--mileage-km", type=int, default=None)
    parser.add_argument("--location", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = LocalPermittedDatasetSource(path=args.input, input_format=args.format) if args.input else None
    if source is None:
        source = LocalPermittedDatasetSource(
            path=settings.market_listings_source_path,
            input_format=settings.market_listings_source_format,
        )
    if source.path is None and source.text is None:
        parser.error("--input is required when MARKET_LISTINGS_SOURCE_PATH is not configured")
        return 2

    criteria = MarketVehicleCriteria(
        make=args.make,
        model=args.model,
        variant=args.variant,
        year=args.year,
        fuel_type=args.fuel,
        transmission=args.transmission,
        mileage_km=args.mileage_km,
        location=args.location,
    )

    listings = discover_market_listings(source, criteria)
    db = SessionLocal()
    try:
        summary = import_discovered_listings(db, listings, dry_run=args.dry_run)
        print(json.dumps({"discovered": len(listings), "import": summary.as_dict()}, indent=2))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
