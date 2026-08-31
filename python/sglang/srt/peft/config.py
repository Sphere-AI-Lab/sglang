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

from sglang.srt.arg_groups.arg_utils import NS, A
from sglang.srt.peft.oft.oft_registry import OFTRef

logger = logging.getLogger(__name__)

OFT_BACKEND_CHOICES = ["triton", "torch_native"]
OFT_TYPE_CHOICES = ["oft", "canonical_oft"]
OFT_IMPL_CHOICES = ["peft", "sibling", "staged"]


@dataclass(kw_only=True)
class PEFTArgs:
    """OFT (Orthogonal Finetuning) server-arg fields, mixed into ServerArgs."""

    # Single-active PEFT method: None (off) | "oft". This one field replaces
    # the former enable_oft / enable_peft_lora boolean pair -- two bools could
    # encode illegal states (both set) that the single-active invariant
    # forbids. (Also used to be "lora", served by srt/peft/lora; that legacy
    # path was deleted once srt/lora + StagedLoRAManager superseded it.)
    peft_method: A[Optional[str], NS("lora")] = None

    # Shared single-active PEFT inputs (the active method is `peft_method`):
    #   peft_paths          adapter path map; normalized to OFTRef by
    #                       peft_method in validate_peft_args.
    #   peft_target_modules module allow-list; method-specific normalization
    #                       ("all"/embed/lm_head handling is OFT-only) in validate.
    peft_paths: A[
        Optional[
            Union[
                dict[str, str],
                List[dict[str, str]],
                List[str],
                List[OFTRef],
            ]
        ],
        NS("lora"),
    ] = None
    peft_target_modules: A[Optional[Union[set[str], List[str]]], NS("lora")] = None

    max_oft_block_size: A[Optional[int], NS("lora")] = None
    # Default 8 matches the CLI default (argparse resolved to 8 all along) and
    # upstream LoRA's max_loras_per_batch, so CLI and programmatic launches now
    # select the same kernel path. Slot 0 holds the auto-registered `None`
    # placeholder (identity = base/reference model, used for KL against base in
    # RL); the rest hold adapters. Values <= 2 select the single-adapter
    # fast-path kernels (`TritonOFTBackend.single_adapter_mode`); orbit's RL
    # launcher pins the value explicitly (2..4) and ignores this default.
    max_ofts_per_batch: A[int, NS("lora")] = 8
    oft_backend: A[str, NS("lora")] = "triton"
    # Which OFT implementation serves when peft_method == "oft". CUTOVER
    # 2026-08-29: default is "sibling" = srt/oft (the srt/lora-shaped mirror),
    # after the equivalence gate passed bitwise on the full parity matrix.
    # "peft" = srt/peft/oft, kept intact as the rollback lever and frozen
    # reference. "staged" = srt/oft's StagedOFTManager, an explicit
    # stage/activate transaction on top of the same srt/oft family (async-RL
    # weight sync; see srt/oft/staged_manager.py). The tokenizer-side
    # registry/ref classes follow this flag too (byte-identical twins;
    # dispatched in validate_peft_args and peft/tokenizer_hooks.py).
    oft_impl: A[str, NS("lora")] = "sibling"
    oft_dtype: A[Optional[str], NS("lora")] = None
    # Single global signal for split(canonical)-vs-fused OFT (attention qkv,
    # dense MLP gate_up, MoE expert gate_up all derive split-vs-fused from
    # this one flag; see sglang.srt.peft.oft.utils.detect_canonical_split_active
    # and oft/mem_pool.py's _declare_expert_groups). "canonical_oft" is orbit's
    # ONLY trained variant (Megatron-Bridge canonical_oft emits per-sub-
    # projection SPLIT rotations); "oft" is the legacy shared-R fused variant.
    oft_type: A[str, NS("lora")] = "canonical_oft"
    max_oft_chunk_size: A[Optional[int], NS("lora")] = 16

    # Double-buffer weight-sync (async RL, NCCL transport). When set, adapter
    # memory pools reserve a staging slot and the stage/activate endpoints are
    # live. Orbit sets this alongside --adapter-double-buffer. Off => in-place
    # single-active sync (IPC/colocate), byte-identical to today.
    peft_double_buffer: A[bool, NS("lora")] = False

    @property
    def enable_peft(self) -> bool:
        """True if the single-active peft method (OFT) is enabled."""
        return self.peft_method is not None


class PeftPathAction(argparse.Action):
    """Collect --peft-paths entries (strings or JSON dicts). Dict entries use
    {adapter_name, adapter_path} keys; key validation is deferred to
    validate_peft_args, which knows the active peft_method."""

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
        choices=["oft"],
        help="Single-active PEFT method: 'oft' (Orthogonal Finetuning). "
        "Required when --peft-paths is given. Distinct from upstream "
        "--enable-lora (multi-tenant).",
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
        '<PATH> | <NAME>=<PATH> | JSON {"adapter_name":str,"adapter_path":str,"pinned":bool}',
    )
    parser.add_argument(
        "--max-ofts-per-batch",
        type=int,
        default=PEFTArgs.max_ofts_per_batch,
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

    parser.add_argument(
        "--peft-double-buffer",
        action="store_true",
        default=PEFTArgs.peft_double_buffer,
        help="Reserve a staging slot and enable the double-buffer stage/activate "
             "adapter endpoints (async-RL NCCL weight-sync).",
    )
    parser.add_argument(
        "--oft-impl",
        type=str,
        default="sibling",
        choices=OFT_IMPL_CHOICES,
        help="Which OFT implementation serves for peft_method='oft'. "
        "'sibling' = srt/oft (default since the 2026-08-29 cutover); "
        "'peft' = srt/peft/oft (rollback lever / frozen reference); "
        "'staged' = srt/oft's StagedOFTManager (explicit stage/activate "
        "transaction for async-RL weight sync).",
    )

def validate_peft_args(server_args) -> None:
    """Validate + normalize OFT server args in place (was check_oft_server_args)."""
    if server_args.peft_method not in (None, "oft"):
        raise ValueError(
            f"--peft-method {server_args.peft_method!r} is no longer supported: "
            "srt/peft/lora was deleted (superseded by srt/lora + "
            "StagedLoRAManager; see --enable-lora for native multi-tenant "
            "LoRA). Only None and 'oft' are valid for --peft-method."
        )
    if getattr(server_args, "oft_impl", "sibling") not in OFT_IMPL_CHOICES:
        raise ValueError(
            f"Invalid --oft-impl {server_args.oft_impl!r}; choose from {OFT_IMPL_CHOICES}."
        )
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
            "--peft-paths requires --peft-method (oft) to be set explicitly."
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

        # OFT is not prefill-CUDA-graph-safe: prepare_oft_batch routes
        # EXTEND-mode prepares to the eager path (forward_mode.is_cuda_graph()
        # is False for EXTEND), which builds a fresh batch_info and rebinds
        # backend.batch_info instead of refreshing persistent buffers. Prefill
        # graph capture therefore freezes device pointers to tensors that are
        # freed (and reused) by the very next prepare call, and the first
        # replay reads garbage seg_indptr/weight_indices in the segmented OFT
        # kernels -> CUDA illegal memory access at the first prefill request.
        # Decode CUDA graphs are unaffected (decode prepares take the in-place
        # cuda_graph_batch_info path) and stay enabled. Lifting this requires
        # mirroring upstream v0.5.18's prefill-graph protocol
        # (init_prefill_cuda_graph_batch_info + can_use_prefill_cuda_graph +
        # a can_run_graph gate); see the no-mirrors note in srt/oft/__init__.py.
        if server_args.cuda_graph_config is not None:
            from sglang.srt.model_executor.cuda_graph_config import Backend

            if server_args.cuda_graph_config.prefill.backend != Backend.DISABLED:
                logger.warning(
                    "peft_method=oft is incompatible with prefill CUDA graphs "
                    "(captured OFT batch metadata goes stale before replay -> "
                    "illegal memory access); disabling the prefill CUDA graph "
                    "(was backend=%s). Decode CUDA graphs stay enabled.",
                    server_args.cuda_graph_config.prefill.backend,
                )
                server_args.cuda_graph_config.prefill.backend = Backend.DISABLED

        # Refs must be the class family of the serving implementation: the
        # sibling registry's ctor asserts its own OFTRef type (a byte-identical
        # twin of the peft one), and both the tokenizer-side registry and the
        # worker-side manager are built from these refs. "staged" serves via
        # StagedOFTManager (srt/oft), the same OFTRef family as "sibling".
        if server_args.oft_impl in ("sibling", "staged"):
            from sglang.srt.oft.oft_registry import OFTRef as _ImplOFTRef
        else:
            _ImplOFTRef = OFTRef

        # Parse peft_paths -> List[OFTRef]. Normalize through locals -- not
        # server_args.peft_paths directly -- because ServerArgs is read-only
        # once __post_init__ reaches materialize_declarations() (well before
        # this call), so writes must go through _late_resolution below.
        peft_paths = server_args.peft_paths
        if isinstance(peft_paths, list):
            adapter_paths = peft_paths
            peft_paths = []
            for adapter_path in adapter_paths:
                if isinstance(adapter_path, str):
                    if "=" in adapter_path:
                        name, path = adapter_path.split("=", 1)
                        oft_ref = _ImplOFTRef(
                            adapter_name=name, adapter_path=path, pinned=False
                        )
                    else:
                        oft_ref = _ImplOFTRef(
                            adapter_name=adapter_path, adapter_path=adapter_path, pinned=False
                        )
                elif isinstance(adapter_path, dict):
                    assert (
                        "adapter_name" in adapter_path and "adapter_path" in adapter_path
                    ), f"When providing OFT paths as a list of dict, each dict should contain 'adapter_name' and 'adapter_path' keys. Got: {adapter_path}"
                    oft_ref = _ImplOFTRef(
                        adapter_name=adapter_path["adapter_name"],
                        adapter_path=adapter_path["adapter_path"],
                        pinned=adapter_path.get("pinned", False),
                    )
                else:
                    raise ValueError(
                        f"Invalid type for item in --peft-paths list: {type(adapter_path)}. "
                        "Expected a string or a dictionary."
                    )
                peft_paths.append(oft_ref)
        elif isinstance(peft_paths, dict):
            peft_paths = [
                _ImplOFTRef(adapter_name=k, adapter_path=v, pinned=False)
                for k, v in peft_paths.items()
            ]
        elif peft_paths is None:
            peft_paths = []
        else:
            raise ValueError(
                f"Invalid type for --peft-paths: {type(peft_paths)}. "
                "Expected a list or a dictionary."
            )

        # Expand target modules (OFT-specific "all"/embed/lm_head handling)
        peft_target_modules = server_args.peft_target_modules
        if peft_target_modules:
            peft_target_modules = set(peft_target_modules)
            if "all" in peft_target_modules:
                assert (
                    len(peft_target_modules) == 1
                ), "If 'all' is specified in --peft-target-modules, it should be the only module specified."
                peft_target_modules = set(SUPPORTED_OFT_TARGET_MODULES)

                # OFT currently only supports torch_native backend,
                # which does not support embedding / lm_head layers yet.
                logger.warning(
                    "OFT backend does not yet support embedding or lm_head layers; "
                    "dropping 'embed_tokens' and 'lm_head' from --peft-target-modules=all."
                )
                peft_target_modules.discard("embed_tokens")
                peft_target_modules.discard("lm_head")

        # Ensure sufficient information is provided for OFT initialization.
        assert peft_paths or (
            server_args.max_oft_block_size and peft_target_modules
        ), "When no initial --peft-paths is provided, you need to specify both --max-oft-block-size and --peft-target-modules for OFT initialization."

        if server_args.max_oft_chunk_size is not None:
            assert (
                16 <= server_args.max_oft_chunk_size <= 128
                and (server_args.max_oft_chunk_size & (server_args.max_oft_chunk_size - 1)) == 0
            ), "--max-oft-chunk-size must be a power of 2 between 16 and 128."

        server_args._late_resolution(
            "validate_peft_args",
            peft_paths=peft_paths,
            peft_target_modules=peft_target_modules,
        )
