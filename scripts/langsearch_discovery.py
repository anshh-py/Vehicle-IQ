"""Opt-in one-shot LangSearch discovery diagnostic; never changes market reports."""

import argparse
import json
import sys

from app.pricing.langsearch_discovery import (
    LangSearchDiscoveryClient,
    LangSearchError,
    get_langsearch_discovery_client,
)
from app.pricing.market import MarketVehicleCriteria


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make one live LangSearch request.")
    parser.add_argument("--make", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--year", type=int)
    parser.add_argument("--fuel")
    parser.add_argument("--transmission")
    parser.add_argument("--mileage-km", type=int)
    parser.add_argument("--city")
    parser.add_argument("--summary-only", action="store_true", help="Print result count only; omit page content.")
    args = parser.parse_args()
    if not args.live:
        parser.error("refusing network access without --live")
    client = get_langsearch_discovery_client()
    if client is None:
        print("LangSearch API key is not configured.", file=sys.stderr)
        return 2
    criteria = MarketVehicleCriteria(
        make=args.make,
        model=args.model,
        variant=args.variant,
        year=args.year,
        fuel_type=args.fuel,
        transmission=args.transmission,
        mileage_km=args.mileage_km,
        location=args.city,
    )
    try:
        results = client.search(criteria)
    except LangSearchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.summary_only:
        print(json.dumps({"kind": "discovery_only", "status": "completed", "result_count": len(results)}))
    else:
        print(json.dumps({"kind": "discovery_only", "results": [item.model_dump(mode="json") for item in results]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
