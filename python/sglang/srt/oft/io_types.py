"""OFT request/output dataclasses — the serving-API seam for OFT adapters.

Moved out of ``sglang.srt.managers.io_struct`` so canonical OFT owns its wire
types under ``srt/oft/``. ``io_struct.py`` re-exports these during Task 6 via
``from sglang.srt.oft.io_types import *`` so existing
``from sglang.srt.managers.io_struct import LoadOFTAdapterReqInput``-style
imports keep working.
"""

from typing import Annotated, Any, Dict, List, Optional, Union

from sglang.srt.managers.io_struct import BaseReq
from sglang.srt.oft.oft_registry import OFTRef
from sglang.srt.utils.msgspec_utils import Base64Bytes

__all__ = [
    "LoadOFTAdapterReqInput",
    "UnloadOFTAdapterReqInput",
    "LoadOFTAdapterFromTensorsReqInput",
    "LoadOFTAdapterFromDistributedReqInput",
    "OFTUpdateOutput",
    "LoadOFTAdapterReqOutput",
    "UnloadOFTAdapterReqOutput",
    "LoadOFTAdapterFromTensorsReqOutput",
    "LoadOFTAdapterFromDistributedReqOutput",
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
    # The PEFT adapter_config.json, already JSON.
    config_dict: Dict[str, Any]
    # One serialized copy of the adapter tensors per TP rank; each rank
    # deserializes only its own copy.
    serialized_named_tensors: Annotated[List[bytes], Base64Bytes()]
    pinned: bool = False
    adapter_id: Optional[str] = None
    load_format: Optional[str] = None
    # If already loaded, refresh weights in place instead of failing.
    upsert: bool = False

    def to_ref(self) -> OFTRef:
        return OFTRef(
            adapter_id=self.adapter_id,
            adapter_name=self.adapter_name,
            adapter_path="__tensor__",
            pinned=self.pinned,
            reloadable=False,
        )


class LoadOFTAdapterFromDistributedReqInput(BaseReq, kw_only=True):
    adapter_name: str
    config_dict: Dict[str, Any]
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    group_name: str = "weight_update_group"
    pinned: bool = False
    adapter_id: Optional[str] = None
    # If already loaded, refresh weights in place instead of failing.
    upsert: bool = False

    def to_ref(self) -> OFTRef:
        return OFTRef(
            adapter_id=self.adapter_id,
            adapter_name=self.adapter_name,
            adapter_path="__distributed__",
            pinned=self.pinned,
            reloadable=False,
        )


class OFTUpdateOutput(BaseReq, kw_only=True):
    success: bool
    error_message: Optional[str] = None
    loaded_adapters: Optional[Dict[str, Union[str, OFTRef]]] = None


LoadOFTAdapterReqOutput = UnloadOFTAdapterReqOutput = (
    LoadOFTAdapterFromTensorsReqOutput
) = LoadOFTAdapterFromDistributedReqOutput = OFTUpdateOutput
