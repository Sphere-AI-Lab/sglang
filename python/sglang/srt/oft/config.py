"""OFT-owned server argument declarations, CLI registration, and validation."""

import argparse
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

from sglang.srt.arg_groups.arg_utils import A, NS
from sglang.srt.oft.oft_registry import OFTRef

logger = logging.getLogger(__name__)

OFT_BACKEND_CHOICES = ["triton", "torch_native"]
OFT_TYPE_CHOICES = ["oft", "canonical_oft"]
SUPPORTED_OFT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "gate_up_proj",
    "embed_tokens",
    "lm_head",
    "wq_a",
    "wq_b",
    "wkv",
    "wo_a",
    "wo_b",
    "w1",
    "w2",
    "w3",
]


@dataclass(kw_only=True)
class OFTArgs:
    """Canonical OFT fields mixed into ``ServerArgs`` by the runtime seam."""

    peft_method: A[Optional[str], NS("lora")] = None
    peft_paths: A[
        Optional[
            Union[
                dict[str, str],
                List[dict[str, str]],
                List[str],
                List["OFTRef"],
            ]
        ],
        NS("lora"),
    ] = None
    peft_target_modules: A[Optional[Union[set[str], List[str]]], NS("lora")] = None
    max_oft_block_size: A[Optional[int], NS("lora")] = None
    max_ofts_per_batch: A[int, NS("lora")] = 8
    max_loaded_ofts: A[Optional[int], NS("lora")] = None
    oft_backend: A[str, NS("lora")] = "triton"
    oft_dtype: A[Optional[str], NS("lora")] = None
    oft_type: A[str, NS("lora")] = "canonical_oft"
    max_oft_chunk_size: A[Optional[int], NS("lora")] = 16
    oft_drain_wait_threshold: A[float, NS("lora")] = 0.0

    @property
    def enable_oft(self) -> bool:
        return self.peft_method == "oft"


class OFTPathAction(argparse.Action):
    """Collect string or JSON adapter entries from ``--peft-paths``."""

    def __call__(self, parser, namespace, values, option_string=None):
        paths = []
        if values:
            assert isinstance(values, list), "Expected a list of OFT paths."
            for path in values:
                path = path.strip()
                if path.startswith("{") and path.endswith("}"):
                    paths.append(json.loads(path))
                else:
                    paths.append(path)
        setattr(namespace, self.dest, paths)


def register_oft_args(parser: argparse.ArgumentParser) -> None:
    """Register the canonical OFT command-line surface."""
    parser.add_argument(
        "--peft-method",
        type=str,
        default=OFTArgs.peft_method,
        choices=["oft"],
        help="Enable canonical Orthogonal Finetuning adapter serving.",
    )
    parser.add_argument(
        "--peft-paths",
        type=str,
        nargs="*",
        default=None,
        action=OFTPathAction,
        help=(
            "OFT adapters to load: PATH, NAME=PATH, or JSON with "
            "adapter_name/adapter_path/pinned fields."
        ),
    )
    parser.add_argument(
        "--peft-target-modules",
        type=str,
        nargs="*",
        default=None,
        help="OFT target-module allow-list; 'all' selects supported modules.",
    )
    parser.add_argument(
        "--max-oft-block-size",
        default=OFTArgs.max_oft_block_size,
        type=int,
        help="Maximum OFT adapter block size; inferred from adapters if omitted.",
    )
    parser.add_argument(
        "--max-ofts-per-batch",
        type=int,
        default=OFTArgs.max_ofts_per_batch,
        help="Maximum resident OFT identities in one batch, including base.",
    )
    parser.add_argument(
        "--max-loaded-ofts",
        type=int,
        default=OFTArgs.max_loaded_ofts,
        help=(
            "Maximum OFT adapters kept in the tokenizer-side registry. Must be "
            "at least --max-ofts-per-batch - 1 because slot 0 is the base model."
        ),
    )
    parser.add_argument(
        "--oft-backend",
        type=str,
        choices=OFT_BACKEND_CHOICES,
        default=OFTArgs.oft_backend,
        help="Kernel backend for canonical OFT serving.",
    )
    parser.add_argument(
        "--oft-dtype",
        type=str,
        choices=[
            "auto",
            "model",
            "float32",
            "fp32",
            "bfloat16",
            "bf16",
            "float16",
            "fp16",
        ],
        default=OFTArgs.oft_dtype,
        help="Dtype for precomputed OFT rotation buffers; defaults to model dtype.",
    )
    parser.add_argument(
        "--oft-type",
        type=str,
        choices=OFT_TYPE_CHOICES,
        default=OFTArgs.oft_type,
        help="OFT layout: canonical split rotations or legacy fused rotations.",
    )
    parser.add_argument(
        "--max-oft-chunk-size",
        type=int,
        default=OFTArgs.max_oft_chunk_size,
        choices=[16, 32, 64, 128],
        help="Maximum OFT kernel chunk size.",
    )
    parser.add_argument(
        "--oft-drain-wait-threshold",
        type=float,
        default=OFTArgs.oft_drain_wait_threshold,
        help=(
            "Maximum seconds an OFT request may wait before a running adapter "
            "is drained to make room for it. Disabled when set to 0."
        ),
    )


def _normalize_oft_refs(raw_paths):
    from sglang.srt.oft.oft_registry import OFTRef

    if raw_paths is None:
        return []
    if isinstance(raw_paths, dict):
        return [
            OFTRef(adapter_name=name, adapter_path=path, pinned=False)
            for name, path in raw_paths.items()
        ]
    if not isinstance(raw_paths, list):
        raise ValueError(
            f"Invalid type for --peft-paths: {type(raw_paths)}. "
            "Expected a list or dictionary."
        )

    refs = []
    for entry in raw_paths:
        if isinstance(entry, OFTRef):
            refs.append(entry)
        elif isinstance(entry, str):
            if "=" in entry:
                name, path = entry.split("=", 1)
            else:
                name = path = entry
            refs.append(OFTRef(adapter_name=name, adapter_path=path, pinned=False))
        elif isinstance(entry, dict):
            assert "adapter_name" in entry and "adapter_path" in entry, (
                "Each OFT path dictionary must contain adapter_name and "
                f"adapter_path; got {entry}."
            )
            refs.append(
                OFTRef(
                    adapter_name=entry["adapter_name"],
                    adapter_path=entry["adapter_path"],
                    pinned=entry.get("pinned", False),
                )
            )
        else:
            raise ValueError(
                f"Invalid --peft-paths entry type: {type(entry)}. "
                "Expected a string, dictionary, or OFTRef."
            )
    return refs


def validate_oft_args(server_args) -> None:
    """Validate and normalize the OFT-owned ``ServerArgs`` fields in place."""
    assert server_args.oft_drain_wait_threshold >= 0.0, (
        "--oft-drain-wait-threshold must be non-negative."
    )
    method = server_args.peft_method
    if method not in (None, "oft"):
        raise ValueError(f"Unsupported --peft-method {method!r}; only 'oft' is valid.")
    if server_args.enable_lora and method is not None:
        raise ValueError(
            "--enable-lora and --peft-method are mutually exclusive: native "
            "multi-tenant LoRA and canonical OFT cannot be initialized together."
        )

    assert server_args.max_ofts_per_batch > 0, "max_ofts_per_batch must be positive"
    if server_args.max_loaded_ofts is not None:
        assert server_args.max_loaded_ofts >= server_args.max_ofts_per_batch - 1, (
            "max_loaded_ofts should be greater than or equal to "
            "max_ofts_per_batch - 1 (slot 0 is reserved for the base model). "
            f"max_loaded_ofts={server_args.max_loaded_ofts}, "
            f"max_ofts_per_batch={server_args.max_ofts_per_batch}"
        )
        if server_args.peft_paths:
            assert len(server_args.peft_paths) <= server_args.max_loaded_ofts, (
                "The number of OFT paths should not exceed max_loaded_ofts. "
                f"max_loaded_ofts={server_args.max_loaded_ofts}, "
                f"peft_paths={len(server_args.peft_paths)}"
            )
    if server_args.peft_paths and method is None:
        raise ValueError("--peft-paths requires --peft-method oft.")
    if method is None:
        return

    assert server_args.oft_type in OFT_TYPE_CHOICES, (
        f"--oft-type must be one of {OFT_TYPE_CHOICES}, got "
        f"{server_args.oft_type!r}."
    )
    if server_args.speculative_algorithm not in ["NGRAM", None]:
        raise ValueError("Currently OFT is only compatible with NGRAM speculation.")

    if server_args.cuda_graph_config is not None:
        from sglang.srt.model_executor.cuda_graph_config import Backend

        if server_args.cuda_graph_config.prefill.backend != Backend.DISABLED:
            logger.warning(
                "Canonical OFT is incompatible with prefill CUDA graphs; "
                "disabling prefill capture while leaving decode graphs enabled."
            )
            server_args.cuda_graph_config.prefill.backend = Backend.DISABLED

    peft_paths = _normalize_oft_refs(server_args.peft_paths)
    target_modules = server_args.peft_target_modules
    if target_modules:
        target_modules = set(target_modules)
        if "all" in target_modules:
            assert len(target_modules) == 1, (
                "If 'all' is specified in --peft-target-modules, it must be "
                "the only module."
            )
            target_modules = set(SUPPORTED_OFT_TARGET_MODULES)
            target_modules.discard("embed_tokens")
            target_modules.discard("lm_head")

    assert peft_paths or (
        server_args.max_oft_block_size and target_modules
    ), (
        "Without initial --peft-paths, specify both --max-oft-block-size and "
        "--peft-target-modules for OFT initialization."
    )
    if server_args.max_oft_chunk_size is not None:
        chunk = server_args.max_oft_chunk_size
        assert 16 <= chunk <= 128 and chunk & (chunk - 1) == 0, (
            "--max-oft-chunk-size must be a power of 2 between 16 and 128."
        )

    server_args._late_resolution(
        "validate_oft_args",
        peft_paths=peft_paths,
        peft_target_modules=target_modules,
    )
