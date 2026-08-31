"""OFT request/output dataclasses — the serving-API seam for OFT adapters.

Moved out of ``sglang.srt.managers.io_struct`` so canonical OFT owns its wire
types under ``srt/oft/``. ``io_struct.py`` re-exports these during Task 6 via
``from sglang.srt.oft.io_types import *`` so existing
``from sglang.srt.managers.io_struct import LoadOFTAdapterReqInput``-style
imports keep working.
"""

from typing import Any, Dict, Optional

from sglang.srt.managers.io_struct import BaseReq
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


class UnloadOFTAdapterReqInput(BaseReq, kw_only=True):
    # The name of OFT module to unload.
    adapter_name: str
    # The unique identifier for the OFT adapter, which automatically generated in the `TokenizerManager`.
    adapter_id: Optional[str] = None

    def to_ref(self) -> OFTRef:
        return OFTRef(
            adapter_id=self.adapter_id,
            adapter_name=self.adapter_name,
        )


class LoadOFTAdapterFromTensorsReqInput(BaseReq, kw_only=True):
    adapter_name: str
    config_dict: Dict[str, Any]
    serialized_tensors: str
    pinned: bool = False
    added_tokens_config: Optional[Dict[str, Any]] = None
    adapter_id: Optional[str] = None

    def to_ref(self) -> OFTRef:
        return OFTRef(
            adapter_id=self.adapter_id,
            adapter_name=self.adapter_name,
            adapter_path="__tensor__",
            pinned=self.pinned,
        )


class OFTUpdateOutput(BaseReq, kw_only=True):
    success: bool
    error_message: Optional[str] = None
    loaded_adapters: Optional[Dict[str, OFTRef]] = None


LoadOFTAdapterReqOutput = UnloadOFTAdapterReqOutput = (
    LoadOFTAdapterFromTensorsReqOutput
) = OFTUpdateOutput
