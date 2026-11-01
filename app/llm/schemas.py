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

    @field_validator(
        "vehicle_overview",
        "overall_condition_summary",
        "detected_damage_explanation",
        "potential_inspection_risks",
        "buyer_recommendations",
        "price_estimate_explanation",
        "missing_information_and_limitations",
        mode="before",
    )
    @classmethod
    def coerce_text(cls, value):
        if isinstance(value, list):
            return " ".join(str(item) for item in value)
        return value