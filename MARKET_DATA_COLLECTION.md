

## Automated acquisition status

No Indian used-car marketplace in `app/market/sources.py` is registered as
permitting unauthenticated automated collection.  OLX, Cars24 and CarDekho
entries are explicitly marked `automated_access_permitted=False`; do not scrape
them.

The only permitted automated source is `local_permitted_dataset`, which reads a
user-provided CSV or JSON file through `LocalPermittedDatasetSource` and
`scripts/discover_market_listings.py`.  A generic `HttpJsonMarketSource` adapter
is available for a future source whose terms explicitly permit automated access;
such a source must first be added to `MARKET_SOURCE_REGISTRY` with
`automated_access_permitted=True`.

# Comparable Market Data Collection Guide

## Purpose
This guide describes how to add legally accessible, publicly available used-car listing data to Vehicle IQ without scraping restricted endpoints or collecting personal information.

## Allowed sources
- Public listing pages that permit manual research or documented reuse.
- Dealer or manufacturer certified pre-owned pages with public pricing.
- Manually transcribed asking prices from publicly visible listings.

## Prohibited collection
- Do not bypass CAPTCHAs, login walls, anti-bot checks, or rate limits.
- Do not collect seller names, phone numbers, email addresses, or account IDs.
- Do not scrape restricted endpoints.
- Do not fabricate, estimate, or normalize a listing price.
- Do not mark a record as verified unless its source and reference are independently traceable.

## CSV template
Use `market-listings-template.csv`. One row per listing. Keep the header exactly as provided.

## Field notes
- `source_name`: stable source identifier, for example `manual_public_listing`.
- `source_listing_id`: public listing ID if available; otherwise leave blank.
- `source_url`: HTTPS URL of the public listing. Required.
- `make`, `model`: use the listing's displayed manufacturer and model names.
- `variant`: use the displayed variant or trim when available.
- `manufacturing_year`: model year shown by the source.
- `fuel_type`: `petrol`, `diesel`, `cng`, `lpg`, or `electric` when known.
- `transmission`: `manual`, `automatic`, `amt`, `cvt`, or `dct` when known.
- `odometer_km`: displayed odometer reading in kilometres.
- `asking_price_inr`: displayed seller asking price in INR.
- `city`, `state`: listing location only.
- `seller_type`: `Individual`, `Dealer`, or `Trustmark Dealer` when known.
- `listed_at`: listing date with timezone, for example `2026-10-29T00:00:00+05:30`.
- `collected_at`: date and time the public listing was manually recorded.
- `raw_data`: optional JSON object containing only non-sensitive listing attributes.

## Validation and import
Validate and import with:

```powershell
python scripts/import_market_listings.py --csv market-listings-template.csv --dry-run
python scripts/import_market_listings.py --csv market-listings-template.csv
```

Repeated imports are idempotent by source identity or source URL.

## Matching rules
Comparable matching uses:
- exact normalized make and model;
- variant only when both listing and criteria provide it;
- manufacturing year within one year;
- fuel type when both values are present;
- transmission when both values are present;
- odometer within the larger of 20,000 km or 25% of the target mileage;
- location when both values are present.

Listings are excluded when:
- they are malformed or invalid;
- the asking price is zero or negative;
- they are duplicate source identities or URLs;
- they are older than 30 days by listing or retrieval date;
- they fail vehicle matching;
- they are price outliers using Tukey 1.5 IQR fences.

A comparable range is returned only when at least five inlier listings remain. Otherwise the status is `low_evidence` and no range is returned.

All returned prices are seller asking prices, not completed transaction prices. Comparable market data is separate from the XGBoost valuation.