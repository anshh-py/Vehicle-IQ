"""Local vehicle damage detection utilities."""

from app.damage.inference import (
    DamageCheckpointMissing,
    DamageDetection,
    DamageDetector,
    DamageInferenceError,
    DamageInferenceResult,
    DamageInputError,
    DamageRuntimeUnavailable,
)

__all__ = [
    "DamageCheckpointMissing",
    "DamageDetection",
    "DamageDetector",
    "DamageInferenceError",
    "DamageInferenceResult",
    "DamageInputError",
    "DamageRuntimeUnavailable",
]
