from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SourceDefinition:
    name: str
    source_type: str
    reference: str | None
    reliability: str
    approved_for_verification: bool


SOURCE_REGISTRY: dict[str, SourceDefinition] = {
    "vehiclex_candidate": SourceDefinition(
        name="vehiclex_candidate",
        source_type="api",
        reference="https://vehiclex.profilework239.workers.dev/",
        reliability="unverified",
        approved_for_verification=False,
    ),
    "curated_catalog": SourceDefinition(
        name="curated_catalog",
        source_type="manual",
        reference=None,
        reliability="low",
        approved_for_verification=False,
    ),
    "manufacturer_catalog": SourceDefinition(
        name="manufacturer_catalog",
        source_type="manufacturer",
        reference=None,
        reliability="unverified",
        approved_for_verification=False,
    ),
}


def get_source(name: str | None) -> SourceDefinition | None:
    if not name:
        return None
    return SOURCE_REGISTRY.get(name.strip())


def is_approved_source(name: str | None) -> bool:
    source = get_source(name)
    return bool(source and source.approved_for_verification)