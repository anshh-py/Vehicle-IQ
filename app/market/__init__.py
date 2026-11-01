"""Market-listing ingestion, acquisition and comparable matching helpers."""

from .acquisition import (
    AutomatedMarketListingProvider,
    HttpJsonMarketSource,
    LocalPermittedDatasetSource,
    MarketListingSource,
    MarketSourceError,
    MarketSourceRateLimitError,
    MarketSourceTimeoutError,
    discover_market_listings,
    get_automated_market_source,
    import_discovered_listings,
    market_listing_to_import_row,
    normalize_source_record,
)
from .ingestion import import_market_listings, import_market_listings_from_csv, import_market_listings_from_json
from .repository import MarketListingRepository, DatabaseMarketListingProvider
from .sources import MARKET_SOURCE_REGISTRY, MarketSourceDefinition, get_market_source, is_automated_market_source_permitted

__all__ = [
    "import_market_listings",
    "import_market_listings_from_csv",
    "import_market_listings_from_json",
    "MarketListingRepository",
    "DatabaseMarketListingProvider",
    "AutomatedMarketListingProvider",
    "HttpJsonMarketSource",
    "LocalPermittedDatasetSource",
    "MarketListingSource",
    "MarketSourceError",
    "MarketSourceRateLimitError",
    "MarketSourceTimeoutError",
    "discover_market_listings",
    "get_automated_market_source",
    "import_discovered_listings",
    "market_listing_to_import_row",
    "normalize_source_record",
    "MARKET_SOURCE_REGISTRY",
    "MarketSourceDefinition",
    "get_market_source",
    "is_automated_market_source_permitted",
]