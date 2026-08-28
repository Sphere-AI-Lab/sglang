import logging
from typing import Type

from sglang.srt.oft.backend.base_backend import BaseOFTBackend

logger = logging.getLogger(__name__)

OFT_SUPPORTED_BACKENDS = {}


def register_oft_backend(name):
    def decorator(fn):
        OFT_SUPPORTED_BACKENDS[name] = fn
        return fn

    return decorator


@register_oft_backend("triton")
def create_triton_backend():
    from sglang.srt.oft.backend.triton_backend import TritonOFTBackend

    return TritonOFTBackend


@register_oft_backend("csgmv")
def create_triton_csgmv_backend():
    # TODO(OFT): Implement Chunked SGMV OFT backend for optimized segment Gemm
    raise NotImplementedError(
        "Chunked SGMV OFT backend is not yet implemented. Please use `torch_native` instead."
    )


@register_oft_backend("torch_native")
def create_torch_native_backend():
    from sglang.srt.oft.backend.torch_backend import TorchNativeOFTBackend

    return TorchNativeOFTBackend


@register_oft_backend("flashinfer")
def create_flashinfer_backend():
    raise ValueError(
        "FlashInfer OFT backend is not available. Please use `torch_native` instead."
    )


def get_backend_from_name(name: str) -> Type[BaseOFTBackend]:
    """
    Get corresponding backend class from backend's name
    """
    if name not in OFT_SUPPORTED_BACKENDS:
        raise ValueError(f"Invalid OFT backend: {name}")
    oft_backend = OFT_SUPPORTED_BACKENDS[name]()
    return oft_backend
