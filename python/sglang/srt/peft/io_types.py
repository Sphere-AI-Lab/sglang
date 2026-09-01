"""OFT request/output dataclasses — the serving-API seam for OFT adapters.

``LoadOFTAdapterReqInput`` (disk/HF-path based adapter loading) is defined
here; it has no counterpart in ``sglang.srt.managers.io_struct``.
``UnloadOFTAdapterReqInput``, ``LoadOFTAdapterFromTensorsReqInput``, and
``OFTUpdateOutput`` (plus its output aliases) are the canonical wire types
added directly to ``sglang.srt.managers.io_struct`` by the native
adapter-loading RPC work; this module re-exports them (rather than
redefining them) so existing
``from sglang.srt.peft.io_types import UnloadOFTAdapterReqInput``-style
imports keep working without a second, drifting copy of their field shapes.
"""

from typing import Optional

from sglang.srt.managers.io_struct import (
    BaseReq,
    LoadOFTAdapterFromTensorsReqInput,
    OFTUpdateOutput,
    UnloadOFTAdapterReqInput,
)
from sglang.srt.oft.oft_registry import OFTRef

__all__ = [
    "LoadOFTAdapterReqInput",
    "UnloadOFTAdapterReqInput",
    "LoadOFTAdapterFromTensorsReqInput",
    "OFTUpdateOutput",
    "LoadOFTAdapterReqOutput",
    "UnloadOFTAdapterReqOutput",
    "LoadOFTAdapterFromTensorsReqOutput",
]


class LoadOFTAdapterReqInput(BaseReq, kw_only=True):
    # The name of the OFT module to newly loaded.
    adapter_name: str
    # The path of loading.
    adapter_path: str
    # Whether to pin the OFT adapter in memory.
    pinned: bool = False
    # The unique identifier for the OFT adapter, which automatically generated in the `TokenizerManager`.
    adapter_id: Optional[str] = None

    def to_ref(self) -> OFTRef:
        return OFTRef(
            adapter_id=self.adapter_id,
            adapter_name=self.adapter_name,
            adapter_path=self.adapter_path,
            pinned=self.pinned,
        )


LoadOFTAdapterReqOutput = UnloadOFTAdapterReqOutput = (
    LoadOFTAdapterFromTensorsReqOutput
) = OFTUpdateOutput
