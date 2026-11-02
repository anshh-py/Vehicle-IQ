from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMReportNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vehicle_overview: str = Field(min_length=1)
    overall_condition_summary: str = Field(min_length=1)
    detected_damage_explanation: str = Field(min_length=1)
    potential_inspection_risks: str = Field(min_length=1)
    buyer_recommendations: str = Field(min_length=1)
    price_estimate_explanation: str = Field(min_length=1)
    missing_information_and_limitations: str = Field(min_length=1)
    vehicle_identity_and_lookup_summary: str = Field(
        default="Vehicle lookup and OCR evidence were not supplied.", min_length=1
    )
    market_valuation_explanation: str = Field(
        default="Market valuation evidence was not supplied.", min_length=1
    )

    @field_validator(
        "vehicle_overview",
        "overall_condition_summary",
        "detected_damage_explanation",
        "potential_inspection_risks",
        "buyer_recommendations",
        "price_estimate_explanation",
        "missing_information_and_limitations",
        "vehicle_identity_and_lookup_summary",
        "market_valuation_explanation",
        mode="before",
    )
    @classmethod
    def coerce_text(cls, value):
        if isinstance(value, list):
            return " ".join(str(item) for item in value)
        return value
