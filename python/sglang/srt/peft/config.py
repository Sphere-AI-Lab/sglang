"""PEFT (OFT) server-args config seam.

Owns the OFT CLI flags, their defaults, the argparse Action, and validation, so
``server_args.py`` shrinks to two call-outs — ``register_peft_args(parser)`` and
``validate_peft_args(self)`` — and ``ServerArgs`` mixes in :class:`PEFTArgs` for the
field declarations.

``PEFTArgs`` is ``kw_only`` so it can be a base of ``ServerArgs`` (which has the
required positional ``model_path``) without the "non-default follows default"
dataclass error; ``ServerArgs`` is built by keyword (``from_cli_args``), so the
keyword-only OFT fields construct fine.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

from sglang.srt.peft.lora.lora_registry import LoRARef
from sglang.srt.peft.oft.oft_registry import OFTRef

logger = logging.getLogger(__name__)

OFT_BACKEND_CHOICES = ["triton", "torch_native"]
OFT_TYPE_CHOICES = ["oft", "canonical_oft"]


@dataclass(kw_only=True)
class PEFTArgs:
    """OFT (Orthogonal Finetuning) server-arg fields, mixed into ServerArgs."""

    # Single-active PEFT method: None (off) | "oft" | "lora". This one field
    # replaces the former enable_oft / enable_peft_lora boolean pair -- two bools
    # could encode illegal states (both set) that the single-active invariant
    # forbids; a 3-valued enum makes those unrepresentable.
    peft_method: Optional[str] = None

    # Shared single-active PEFT inputs (the active method is `peft_method`):
    #   peft_paths          adapter path map; normalized to OFTRef or LoRARef by
    #                       peft_method in validate_peft_args.
    #   peft_target_modules module allow-list; method-specific normalization
    #                       ("all"/embed/lm_head handling is OFT-only) in validate.
    peft_paths: Optional[
        Union[
            dict[str, str],
            List[dict[str, str]],
            List[str],
            List[OFTRef],
            List["LoRARef"],
        ]
    ] = None
    peft_target_modules: Optional[Union[set[str], List[str]]] = None

    max_oft_block_size: Optional[int] = None
    # Default 2 covers the typical single-active-adapter workload: slot 0 holds
    # the auto-registered `None` placeholder (identity = base/reference model,
    # used for KL against base in RL), slot 1 holds the active OFT adapter.
    # This also enables the single-adapter fast-path kernel
    # (`gemm_oft_r_fwd`) since `TritonOFTBackend.single_adapter_mode = (max_ofts_per_batch <= 2)`.
    # Multi-tenant deployments serving more than one adapter concurrently
    # should override with `--max-ofts-per-batch <N>`; for the segmented
    # kernel to run, also pass `--max-ofts-per-batch >= 3`.
    max_ofts_per_batch: int = 2
    oft_backend: str = "triton"
    oft_dtype: Optional[str] = None
    # Single global signal for split(canonical)-vs-fused OFT (attention qkv,
    # dense MLP gate_up, MoE expert gate_up all derive split-vs-fused from
    # this one flag; see sglang.srt.peft.oft.utils.detect_canonical_split_active
    # and oft/mem_pool.py's _declare_expert_groups). "canonical_oft" is orbit's
    # ONLY trained variant (Megatron-Bridge canonical_oft emits per-sub-
    # projection SPLIT rotations); "oft" is the legacy shared-R fused variant.
    oft_type: str = "canonical_oft"
    max_oft_chunk_size: Optional[int] = 16

    # peft-lora (single-active method under srt/peft/lora): the low-rank dim.
    # Distinct from OFT's max_oft_block_size -- different hyperparameter, kept
    # separate. Paths/target-modules are shared (peft_paths/peft_target_modules).
    peft_max_lora_rank: Optional[int] = None

    # Double-buffer weight-sync (async RL, NCCL transport). When set, adapter
    # memory pools reserve a staging slot and the stage/activate endpoints are
    # live. Orbit sets this alongside --adapter-double-buffer. Off => in-place
    # single-active sync (IPC/colocate), byte-identical to today.
    peft_double_buffer: bool = False

    @property
    def enable_peft(self) -> bool:
        """True if a single-active peft method (LoRA or OFT) is enabled."""
        return self.peft_method is not None


class PeftPathAction(argparse.Action):
    """Collect --peft-paths entries (strings or JSON dicts). Dict entries use
    {adapter_name, adapter_path} keys (unified across our LoRA/OFT); key
    validation is deferred to validate_peft_args, which knows the active
    peft_method."""

    def __call__(self, parser, namespace, values, option_string=None):
        paths = []
        if values:
            assert isinstance(values, list), "Expected a list of peft paths."
            for path in values:
                path = path.strip()
                if path.startswith("{") and path.endswith("}"):
                    paths.append(json.loads(path))
                else:
                    paths.append(path)

        setattr(namespace, self.dest, paths)


def register_peft_args(parser: argparse.ArgumentParser) -> None:
    """Register all OFT CLI flags on ``parser`` (was the OFT block of add_cli_args)."""
    # Single-active PEFT method selector (replaces the former --enable-oft /
    # --enable-peft-lora store_true pair).
    parser.add_argument(
        "--peft-method",
        type=str,
        default=PEFTArgs.peft_method,
        choices=["oft", "lora"],
        help="Single-active PEFT method: 'oft' (Orthogonal Finetuning) or 'lora' "
        "(the srt/peft/lora single-active method). Required when --peft-paths is "
        "given. Distinct from upstream --enable-lora (multi-tenant).",
    )
    parser.add_argument(
        "--max-oft-block-size",
        default=PEFTArgs.max_oft_block_size,
        type=int,
        help="The maximum block size of OFT adapters. If not specified, it will be automatically inferred from the adapters provided in --peft-paths.",
    )
    parser.add_argument(
        "--peft-target-modules",
        type=str,
        nargs="*",
        default=None,
        help="The set of target modules where the active PEFT method is applied. "
        "If not specified, inferred from the adapters in --peft-paths. For OFT, "
        "'all' selects all supported modules (validated per-method in validate_peft_args).",
    )
    parser.add_argument(
        "--peft-paths",
        type=str,
        nargs="*",
        default=None,
        action=PeftPathAction,
        help='The list of PEFT adapters to load (requires --peft-method). Each adapter: '
        '<PATH> | <NAME>=<PATH> | JSON {"<method>_name":str,"<method>_path":str,"pinned":bool}',
    )
    parser.add_argument(
        "--max-ofts-per-batch",
        type=int,
        default=8,
        help="Maximum number of OFT adapters for a running batch, include base-only request.",
    )
    parser.add_argument(
        "--oft-backend",
        type=str,
        choices=OFT_BACKEND_CHOICES,
        default=PEFTArgs.oft_backend,
        help="Choose the kernel backend for multi-OFT serving.",
    )
    parser.add_argument(
        "--oft-dtype",
        type=str,
        choices=["auto", "model", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
        default=PEFTArgs.oft_dtype,
        help="Dtype for precomputed OFT rotation buffers. Defaults to the model dtype.",
    )
    parser.add_argument(
        "--oft-type",
        type=str,
        choices=OFT_TYPE_CHOICES,
        default=PEFTArgs.oft_type,
        help="OFT variant: 'canonical_oft' (default) uses independent per-"
        "sub-projection SPLIT rotations (orbit's only trained variant); "
        "'oft' uses the legacy shared-R FUSED rotation. Single global "
        "split-vs-fused signal for attention qkv, dense MLP gate_up, and "
        "MoE expert gate_up.",
    )
    parser.add_argument(
        "--max-oft-chunk-size",
        type=int,
        default=PEFTArgs.max_oft_chunk_size,
        choices=[16, 32, 64, 128],
        help="Maximum chunk size for the OFT backend. Choosing a larger value might improve performance.",
    )

    # peft-lora (single-active method under srt/peft/lora): paths/targets are the
    # shared --peft-paths / --peft-target-modules; only the rank is LoRA-specific.
    parser.add_argument(
        "--peft-max-lora-rank",
        type=int,
        default=None,
        help="Max LoRA rank; inferred from --peft-paths if omitted.",
    )
    parser.add_argument(
        "--peft-double-buffer",
        action="store_true",
        default=PEFTArgs.peft_double_buffer,
        help="Reserve a staging slot and enable the double-buffer stage/activate "
             "adapter endpoints (async-RL NCCL weight-sync).",
    )


def validate_peft_args(server_args) -> None:
    """Validate + normalize OFT server args in place (was check_oft_server_args)."""
    if server_args.enable_lora and server_args.peft_method is not None:
        raise ValueError(
            "--enable-lora and --peft-method are mutually exclusive: native "
            "multi-tenant LoRA and single-active PEFT cannot be initialized together."
        )

    from sglang.srt.utils.common import SUPPORTED_OFT_TARGET_MODULES

    assert server_args.max_ofts_per_batch > 0, "max_ofts_per_batch must be positive"

    # peft_paths is method-agnostic, so the method can no longer be inferred from
    # which path field was set -- require it explicitly when paths are given.
    if server_args.peft_paths and server_args.peft_method is None:
        raise ValueError(
            "--peft-paths requires --peft-method (oft|lora) to be set explicitly."
        )

    if server_args.peft_method == "oft":
        assert server_args.oft_type in OFT_TYPE_CHOICES, (
            f"--oft-type must be one of {OFT_TYPE_CHOICES}, got "
            f"{server_args.oft_type!r}."
        )

        # Double-buffer OFT sizing: OFTMemoryPool.__init__ sets active_idx=1,
        # staging_idx=max_ofts_per_batch-1. At max_ofts_per_batch==2 those
        # collide (active==staging==1), so activate()'s staging->active copy
        # would corrupt the active slot instead of promoting it. Fail loud
        # here rather than at pool-init time.
        if server_args.peft_double_buffer:
            assert server_args.max_ofts_per_batch >= 3, (
                "double-buffer OFT requires --max-ofts-per-batch >= 3 "
                "(base + active + staging); got "
                f"{server_args.max_ofts_per_batch}."
            )

        # Validate compatibility with speculative decoding
        if server_args.speculative_algorithm not in ["NGRAM", None]:
            raise ValueError(
                "Currently OFT is only compatible with NGRAM speculative decoding."
            )

        # Parse peft_paths -> List[OFTRef]
        if isinstance(server_args.peft_paths, list):
            adapter_paths = server_args.peft_paths
            server_args.peft_paths = []
            for adapter_path in adapter_paths:
                if isinstance(adapter_path, str):
                    if "=" in adapter_path:
                        name, path = adapter_path.split("=", 1)
                        oft_ref = OFTRef(
                            adapter_name=name, adapter_path=path, pinned=False
                        )
                    else:
                        oft_ref = OFTRef(
                            adapter_name=adapter_path, adapter_path=adapter_path, pinned=False
                        )
                elif isinstance(adapter_path, dict):
                    assert (
                        "adapter_name" in adapter_path and "adapter_path" in adapter_path
                    ), f"When providing OFT paths as a list of dict, each dict should contain 'adapter_name' and 'adapter_path' keys. Got: {adapter_path}"
                    oft_ref = OFTRef(
                        adapter_name=adapter_path["adapter_name"],
                        adapter_path=adapter_path["adapter_path"],
                        pinned=adapter_path.get("pinned", False),
                    )
                else:
                    raise ValueError(
                        f"Invalid type for item in --peft-paths list: {type(adapter_path)}. "
                        "Expected a string or a dictionary."
                    )
                server_args.peft_paths.append(oft_ref)
        elif isinstance(server_args.peft_paths, dict):
            server_args.peft_paths = [
                OFTRef(adapter_name=k, adapter_path=v, pinned=False)
                for k, v in server_args.peft_paths.items()
            ]
        elif server_args.peft_paths is None:
            server_args.peft_paths = []
        else:
            raise ValueError(
                f"Invalid type for --peft-paths: {type(server_args.peft_paths)}. "
                "Expected a list or a dictionary."
            )

        # Expand target modules (OFT-specific "all"/embed/lm_head handling)
        if server_args.peft_target_modules:
            server_args.peft_target_modules = set(server_args.peft_target_modules)
            if "all" in server_args.peft_target_modules:
                assert (
                    len(server_args.peft_target_modules) == 1
                ), "If 'all' is specified in --peft-target-modules, it should be the only module specified."
                server_args.peft_target_modules = set(SUPPORTED_OFT_TARGET_MODULES)

                # OFT currently only supports torch_native backend,
                # which does not support embedding / lm_head layers yet.
                logger.warning(
                    "OFT backend does not yet support embedding or lm_head layers; "
                    "dropping 'embed_tokens' and 'lm_head' from --peft-target-modules=all."
                )
                server_args.peft_target_modules.discard("embed_tokens")
                server_args.peft_target_modules.discard("lm_head")

        # Ensure sufficient information is provided for OFT initialization.
        assert server_args.peft_paths or (
            server_args.max_oft_block_size and server_args.peft_target_modules
        ), "When no initial --peft-paths is provided, you need to specify both --max-oft-block-size and --peft-target-modules for OFT initialization."

        if server_args.max_oft_chunk_size is not None:
            assert (
                16 <= server_args.max_oft_chunk_size <= 128
                and (server_args.max_oft_chunk_size & (server_args.max_oft_chunk_size - 1)) == 0
            ), "--max-oft-chunk-size must be a power of 2 between 16 and 128."

    # ------------------------------------------------------------------ #
    #  peft-lora (single-active). Mirrors the OFT block above, simpler
    #  (no overlap/spec-decode/max_loaded checks).
    # ------------------------------------------------------------------ #
    if server_args.peft_method == "lora":
        from sglang.srt.peft.lora.lora_registry import LoRARef

        # Parse peft_paths -> List[LoRARef] (mirror the OFT branch, LoRARef keys).
        if isinstance(server_args.peft_paths, list):
            adapter_paths = server_args.peft_paths
            server_args.peft_paths = []
            for adapter_path in adapter_paths:
                if isinstance(adapter_path, str):
                    if "=" in adapter_path:
                        name, path = adapter_path.split("=", 1)
                        lora_ref = LoRARef(
                            adapter_name=name, adapter_path=path, pinned=False
                        )
                    else:
                        lora_ref = LoRARef(
                            adapter_name=adapter_path,
                            adapter_path=adapter_path,
                            pinned=False,
                        )
                elif isinstance(adapter_path, dict):
                    assert (
                        "adapter_name" in adapter_path and "adapter_path" in adapter_path
                    ), f"When providing peft LoRA paths as a list of dict, each dict should contain 'adapter_name' and 'adapter_path' keys. Got: {adapter_path}"
                    lora_ref = LoRARef(
                        adapter_name=adapter_path["adapter_name"],
                        adapter_path=adapter_path["adapter_path"],
                        pinned=adapter_path.get("pinned", False),
                    )
                else:
                    raise ValueError(
                        f"Invalid type for item in --peft-paths list: {type(adapter_path)}. "
                        "Expected a string or a dictionary."
                    )
                server_args.peft_paths.append(lora_ref)
        elif isinstance(server_args.peft_paths, dict):
            server_args.peft_paths = [
                LoRARef(adapter_name=k, adapter_path=v, pinned=False)
                for k, v in server_args.peft_paths.items()
            ]
        elif server_args.peft_paths is None:
            server_args.peft_paths = []
        else:
            raise ValueError(
                f"Invalid type for --peft-paths: {type(server_args.peft_paths)}. "
                "Expected a list or a dictionary."
            )

        # Expand target modules (simple: just set(...) if provided; MVP skips
        # "all"/embed/lm_head handling).
        if server_args.peft_target_modules:
            server_args.peft_target_modules = set(server_args.peft_target_modules)

        # Ensure sufficient info (mirror OFT's assert).
        assert server_args.peft_paths or (
            server_args.peft_max_lora_rank and server_args.peft_target_modules
        ), "When no --peft-paths is provided, specify both --peft-max-lora-rank and --peft-target-modules."
