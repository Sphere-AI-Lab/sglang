"""OFT request/output dataclasses — the serving-API seam for OFT adapters.

``UnloadOFTAdapterReqInput``, ``LoadOFTAdapterFromTensorsReqInput``, and
``OFTUpdateOutput`` (plus its output aliases) are the canonical wire types
added directly to ``sglang.srt.managers.io_struct`` by the native
adapter-loading RPC work; this module re-exports them (rather than
redefining them) so existing
``from sglang.srt.peft.io_types import UnloadOFTAdapterReqInput``-style
imports keep working without a second, drifting copy of their field shapes.
"""

from sglang.srt.managers.io_struct import (
    LoadOFTAdapterFromTensorsReqInput,
    OFTUpdateOutput,
    UnloadOFTAdapterReqInput,
)

__all__ = [
    "UnloadOFTAdapterReqInput",
    "LoadOFTAdapterFromTensorsReqInput",
    "OFTUpdateOutput",
    "UnloadOFTAdapterReqOutput",
    "LoadOFTAdapterFromTensorsReqOutput",
]


UnloadOFTAdapterReqOutput = LoadOFTAdapterFromTensorsReqOutput = OFTUpdateOutput
