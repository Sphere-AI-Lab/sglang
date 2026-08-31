"""Immutable bundles and exact-first comparison for adapter equivalence."""

from .compare import (
    ComparisonMismatch,
    ComparisonReport,
    compare_bundles,
)
from .fixtures import (
    AdapterFixture,
    FixtureValidationError,
    MatrixCell,
    build_lora_fixture,
    build_oft_fixture,
    validate_matrix,
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
    "AdapterFixture",
    "BaselineRepetition",
    "BundleValidationError",
    "CaseKey",
    "ComparisonMismatch",
    "ComparisonReport",
    "ComparisonPolicy",
    "FixtureValidationError",
    "MatrixCell",
    "NumericTolerance",
    "Observation",
    "PerformanceMetrics",
    "RunBundle",
    "ToleranceEnvelope",
    "canonical_sha256",
    "build_lora_fixture",
    "build_oft_fixture",
    "compare_bundles",
    "validate_matrix",
]
