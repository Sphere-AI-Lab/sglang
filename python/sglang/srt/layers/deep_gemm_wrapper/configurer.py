import logging

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, is_blackwell_supported

logger = logging.getLogger(__name__)


def _compute_enable_deep_gemm():
    if not envs.SGLANG_ENABLE_JIT_DEEPGEMM.get():
        return False

    sm_version = get_device_sm()
    if sm_version < 90:
        return False

    try:
        import deep_gemm  # noqa: F401
    except Exception as exc:
        logger.warning("Disabling DeepGEMM because import failed: %s", exc)
        return False

    return True


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell_supported()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
