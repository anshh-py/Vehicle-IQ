"""Explicitly verified model-specific pricing defaults.

The current captured catalog values are unverified, so no production defaults
are enabled until their sources are independently approved.
"""

MODEL_DEFAULTS: dict[tuple[str, str, str], dict[str, object]] = {}


def get_model_defaults(make: str, model: str, variant: str | None) -> dict[str, object] | None:
    """Return an explicitly supported default for an exact vehicle variant."""
    if not make or not model or not variant:
        return None
    key = (make.strip(), model.strip(), variant.strip())
    return MODEL_DEFAULTS.get(key)
