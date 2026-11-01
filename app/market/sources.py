from dataclasses import dataclass


@dataclass(frozen=True)
class MarketSourceDefinition:
    name: str
    source_type: str
    reference: str | None
    automated_access_permitted: bool
    authentication_required: bool
    region: str
    notes: str


MARKET_SOURCE_REGISTRY: dict[str, MarketSourceDefinition] = {
    "local_permitted_dataset": MarketSourceDefinition(
        name="local_permitted_dataset",
        source_type="file",
        reference=None,
        automated_access_permitted=True,
        authentication_required=False,
        region="any",
        notes="User-provided CSV or JSON file with documented provenance. No remote access is performed.",
    ),
    "olx_india": MarketSourceDefinition(
        name="olx_india",
        source_type="website",
        reference="https://www.olx.in/",
        automated_access_permitted=False,
        authentication_required=False,
        region="India",
        notes="Public search results exist, but automated collection is not permitted by the site terms. Manual collection only.",
    ),
    "cars24_india": MarketSourceDefinition(
        name="cars24_india",
        source_type="website",
        reference="https://www.cars24.com/",
        automated_access_permitted=False,
        authentication_required=False,
        region="India",
        notes="No documented free automated API for listing reuse. Manual collection only.",
    ),
    "cardekho_india": MarketSourceDefinition(
        name="cardekho_india",
        source_type="website",
        reference="https://www.cardekho.com/",
        automated_access_permitted=False,
        authentication_required=False,
        region="India",
        notes="No documented free automated API for listing reuse. Manual collection only.",
    ),
}


def get_market_source(name: str | None) -> MarketSourceDefinition | None:
    if not name:
        return None
    return MARKET_SOURCE_REGISTRY.get(name.strip())


def is_automated_market_source_permitted(name: str | None) -> bool:
    source = get_market_source(name)
    return bool(source and source.automated_access_permitted)
