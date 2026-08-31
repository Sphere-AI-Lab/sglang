"""Immutable bundles and exact-first comparison for adapter equivalence."""

from .compare import (
    ComparisonMismatch,
    ComparisonReport,
    compare_bundles,
)
from .schema import (
    SCHEMA_VERSION,
    BaselineRepetition,
    BundleValidationError,
    CaseKey,
    ComparisonPolicy,
    NumericTolerance,
    Observation,
    PerformanceMetrics,
    RunBundle,
    ToleranceEnvelope,
    canonical_sha256,
)

__all__ = [
    "SCHEMA_VERSION",
    "BaselineRepetition",
    "BundleValidationError",
    "CaseKey",
    "ComparisonMismatch",
    "ComparisonReport",
    "ComparisonPolicy",
    "NumericTolerance",
    "Observation",
    "PerformanceMetrics",
    "RunBundle",
    "ToleranceEnvelope",
    "canonical_sha256",
    "compare_bundles",
]
