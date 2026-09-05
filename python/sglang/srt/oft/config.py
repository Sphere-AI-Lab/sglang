"""OFT server-args config seam.

Owns the OFT CLI flags, their defaults, the argparse Action, and validation, so
``server_args.py`` shrinks to two call-outs — ``register_oft_args(parser)`` and
``validate_oft_args(self)`` — and ``ServerArgs`` mixes in :class:`OFTArgs` for the
field declarations.

``OFTArgs`` is ``kw_only`` so it can be a base of ``ServerArgs`` (which has the
required positional ``model_path``) without the "non-default follows default"
dataclass error; ``ServerArgs`` is built by keyword (``from_cli_args``), so the
keyword-only OFT fields construct fine.
"""

import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

from sglang.srt.arg_groups.arg_utils import NS, A

logger = logging.getLogger(__name__)

OFT_BACKEND_CHOICES = ["triton", "torch_native"]
OFT_TYPE_CHOICES = ["oft", "canonical_oft"]
OFT_IMPL_CHOICES = ["sibling", "staged"]


@dataclass(kw_only=True)
class OFTArgs:
    """OFT (Orthogonal Finetuning) server-arg fields, mixed into ServerArgs."""

    # Enable single-active OFT serving, mirroring --enable-lora exactly.
    # (Used to be the string field peft_method: None | "oft" -- "oft" was its
    # only valid value, so the string added nothing a plain bool doesn't;
    # before that it also accepted "lora", served by the since-deleted
    # srt/peft/lora, superseded by srt/lora + StagedLoRAManager.)
    enable_oft: A[bool, NS("lora")] = False

    # Shared single-active OFT inputs (the active method is enable_oft):
    #   oft_target_modules  module allow-list; method-specific normalization
    #                       ("all"/embed/lm_head handling is OFT-only) in validate.
    oft_target_modules: A[Optional[Union[set[str], List[str]]], NS("lora")] = None

    max_oft_block_size: A[Optional[int], NS("lora")] = None
    # Default 8 matches the CLI default (argparse resolved to 8 all along) and
    # upstream LoRA's max_loras_per_batch, so CLI and programmatic launches now
    # select the same kernel path. Slot 0 holds the auto-registered `None`
    # placeholder (identity = base/reference model, used for KL against base in
    # RL); the rest hold adapters. Values <= 2 select the single-adapter
    # fast-path kernels (`TritonOFTBackend.single_adapter_mode`); orbit's RL
    # launcher pins the value explicitly (2..4) and ignores this default.
    max_ofts_per_batch: A[int, NS("lora")] = 8
    # Help text lives on the manual parser.add_argument in register_oft_args
    # below (like every other OFTArgs field) -- a bare-string Arg annotation
    # here would make add_cli_args_from_dataclass auto-register this field
    # too, conflicting with that manual registration.
    max_loaded_ofts: A[Optional[int], NS("lora")] = None
    oft_backend: A[str, NS("lora")] = "triton"
    # Which OFT implementation serves when enable_oft is set. CUTOVER
    # 2026-08-29: default is "sibling" = srt/oft (the srt/lora-shaped mirror),
    # after the equivalence gate passed bitwise on the full parity matrix.
    # "staged" = srt/oft's StagedOFTManager, an explicit stage/activate
    # transaction on top of the same srt/oft family (async-RL weight sync;
    # see srt/oft/staged_manager.py). (Also used to be "peft" = srt/peft/oft,
    # the frozen pre-cutover reference; that legacy path was deleted once the
    # equivalence gate passed -- see srt/lora + StagedLoRAManager's analogous
    # history for LoRA.) The tokenizer-side registry/ref class (OFTRef, from
    # srt/oft/oft_registry.py) is shared by both "sibling" and "staged" now
    # that the legacy twin is gone -- no per-oft_impl dispatch remains.
    oft_impl: A[str, NS("lora")] = "sibling"
    oft_dtype: A[Optional[str], NS("lora")] = None
    # Single global signal for split(canonical)-vs-fused OFT (attention qkv,
    # dense MLP gate_up, MoE expert gate_up all derive split-vs-fused from
    # this one flag; see sglang.srt.oft.utils.detect_canonical_split_active
    # and oft/mem_pool.py's _declare_expert_groups). "canonical_oft" is orbit's
    # ONLY trained variant (Megatron-Bridge canonical_oft emits per-sub-
    # projection SPLIT rotations); "oft" is the legacy shared-R fused variant.
    oft_type: A[str, NS("lora")] = "canonical_oft"
    max_oft_chunk_size: A[Optional[int], NS("lora")] = 16

    # Double-buffer weight-sync (async RL, NCCL transport). When set, adapter
    # memory pools reserve a staging slot and the stage/activate endpoints are
    # live. Orbit sets this alongside --adapter-double-buffer. Off => in-place
    # single-active sync (IPC/colocate), byte-identical to today.
    oft_double_buffer: A[bool, NS("lora")] = False

    # Starvation-prevention scheduling (mirrors --lora-drain-wait-threshold).
    # Help text lives on the manual parser.add_argument in register_oft_args
    # below, like every other OFTArgs field.
    oft_drain_wait_threshold: A[float, NS("lora")] = 0.0

    # Overlap the GPU-side cost of materializing a not-yet-resident adapter's
    # weights with ongoing compute (mirrors --enable-lora-overlap-loading).
    enable_oft_overlap_loading: A[bool, NS("lora")] = False


def register_oft_args(parser: argparse.ArgumentParser) -> None:
    """Register all OFT CLI flags on ``parser`` (was the OFT block of add_cli_args)."""
    parser.add_argument(
        "--enable-oft",
        action="store_true",
        default=OFTArgs.enable_oft,
        help="Enable single-active OFT (Orthogonal Finetuning) serving. "
        "Distinct from --enable-lora (multi-tenant).",
    )
    parser.add_argument(
        "--max-oft-block-size",
        default=OFTArgs.max_oft_block_size,
        type=int,
        help="The maximum block size of OFT adapters. Required (together with "
        "--oft-target-modules) for OFT initialization.",
    )
    parser.add_argument(
        "--oft-target-modules",
        type=str,
        nargs="*",
        default=None,
        help="The set of target modules where the active PEFT method is applied. "
        "'all' selects all supported modules (validated per-method in "
        "validate_oft_args).",
    )
    parser.add_argument(
        "--max-ofts-per-batch",
        type=int,
        default=OFTArgs.max_ofts_per_batch,
        help="Maximum number of OFT adapters for a running batch, include base-only request.",
    )
    parser.add_argument(
        "--max-loaded-ofts",
        type=int,
        default=OFTArgs.max_loaded_ofts,
        help="If specified, limits the maximum number of OFT adapters loaded in the tokenizer-side registry at a time. The value must be greater than or equal to `--max-ofts-per-batch - 1` (buffer slot 0 is always reserved for the base/identity placeholder, so real per-batch adapter capacity is `--max-ofts-per-batch - 1`).",
    )
    parser.add_argument(
        "--oft-backend",
        type=str,
        choices=OFT_BACKEND_CHOICES,
        default=OFTArgs.oft_backend,
        help="Choose the kernel backend for multi-OFT serving.",
    )
    parser.add_argument(
        "--oft-dtype",
        type=str,
        choices=["auto", "model", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
        default=OFTArgs.oft_dtype,
        help="Dtype for precomputed OFT rotation buffers. Defaults to the model dtype.",
    )
    parser.add_argument(
        "--oft-type",
        type=str,
        choices=OFT_TYPE_CHOICES,
        default=OFTArgs.oft_type,
        help="OFT variant: 'canonical_oft' (default) uses independent per-"
        "sub-projection SPLIT rotations (orbit's only trained variant); "
        "'oft' uses the legacy shared-R FUSED rotation. Single global "
        "split-vs-fused signal for attention qkv, dense MLP gate_up, and "
        "MoE expert gate_up.",
    )
    parser.add_argument(
        "--max-oft-chunk-size",
        type=int,
        default=OFTArgs.max_oft_chunk_size,
        choices=[16, 32, 64, 128],
        help="Maximum chunk size for the OFT backend. Choosing a larger value might improve performance.",
    )

    parser.add_argument(
        "--oft-double-buffer",
        action="store_true",
        default=OFTArgs.oft_double_buffer,
        help="Reserve a staging slot and enable the double-buffer stage/activate "
             "adapter endpoints (async-RL NCCL weight-sync).",
    )
    parser.add_argument(
        "--oft-impl",
        type=str,
        default="sibling",
        choices=OFT_IMPL_CHOICES,
        help="Which OFT implementation serves when --enable-oft is set. "
        "'sibling' = srt/oft (default since the 2026-08-29 cutover); "
        "'staged' = srt/oft's StagedOFTManager (explicit stage/activate "
        "transaction for async-RL weight sync).",
    )
    parser.add_argument(
        "--oft-drain-wait-threshold",
        type=float,
        default=OFTArgs.oft_drain_wait_threshold,
        help="When any OFT adapter request waits longer than this threshold "
        "(in seconds), the scheduler will selectively drain one running "
        "adapter to make room. This mitigates extreme tail latency under "
        "high or skewed workloads by preventing a small set of adapters "
        "from monopolizing batch slots. Set to 0 to disable draining "
        "(default).",
    )
    parser.add_argument(
        "--enable-oft-overlap-loading",
        action="store_true",
        default=OFTArgs.enable_oft_overlap_loading,
        help="Overlap the GPU-side cost of materializing a not-yet-resident "
        "OFT adapter's weights with ongoing compute, instead of stalling "
        "the forward pass that first names it.",
    )


# Attribute names the MoE architectures in this repo use for their per-token
# expert count -- the same probe Scheduler.init_moe_gemm_config
# (managers/scheduler.py) uses to decide whether a model is MoE at all.
_MOE_EXPERT_COUNT_CONFIG_ATTRS = (
    "num_experts_per_tok",
    "num_experts_per_token",
    "top_k_experts",
    "moe_top_k",
    "moe_topk",
)

# The MoE expert projection module names OFT can target. Shared with
# decode_cuda_graph_runner.py's _resolve_record_oft_variant_graph (its
# pre-model-load fallback branch) so the two never drift out of sync -- see
# the final whole-branch review's I2/I3 findings for
# 2026-09-01-oft-moe-cuda-graph-dual-capture.
MOE_EXPERT_TARGET_MODULES = frozenset(
    {"gate_up_proj", "gate_proj", "up_proj", "down_proj"}
)


def effective_oft_capacity(server_args) -> int:
    """Effective per-batch adapter capacity: buffer slot 0 is always reserved
    for the base/identity placeholder (OFTMemoryPool), so real resident-
    adapter capacity is max_ofts_per_batch - 1, not max_ofts_per_batch.
    Shared with decode_cuda_graph_runner.py's own copy of this arithmetic so
    the two can never drift (final whole-branch review's I2/I3)."""
    return server_args.max_ofts_per_batch - 1


def _model_has_moe_layers(server_args) -> bool:
    """Whether the model this server is about to load actually has MoE layers
    (and therefore FusedMoE modules for expert OFT to wrap).

    The runtime equivalents of this check -- ``OFTMemoryPool.
    _declare_expert_groups``'s ``if not moe_layers: return`` and
    ``OFTManager._install_moe_oft_wrappers``'s ``moe_names`` scan -- walk the
    built module tree, which does not exist yet at server-args validation
    time. The HF config's per-token expert count is the earliest available
    signal for the same fact.
    """
    return any(
        hasattr(server_args.get_model_config().hf_text_config, attr)
        for attr in _MOE_EXPERT_COUNT_CONFIG_ATTRS
    )


def validate_oft_args(server_args) -> None:
    """Validate + normalize OFT server args in place (was check_oft_server_args)."""
    if server_args.oft_impl not in OFT_IMPL_CHOICES:
        raise ValueError(
            f"Invalid --oft-impl {server_args.oft_impl!r}; choose from {OFT_IMPL_CHOICES}."
        )
    if server_args.enable_lora and server_args.enable_oft:
        raise ValueError(
            "--enable-lora and --enable-oft are mutually exclusive: native "
            "multi-tenant LoRA and single-active OFT cannot be initialized together."
        )

    from sglang.srt.utils.common import SUPPORTED_OFT_TARGET_MODULES

    assert server_args.max_ofts_per_batch > 0, "max_ofts_per_batch must be positive"
    assert (
        server_args.oft_drain_wait_threshold >= 0.0
    ), "--oft-drain-wait-threshold must be non-negative."

    if server_args.max_loaded_ofts is not None:
        # Buffer slot 0 is always reserved for the base/identity placeholder
        # (see OFTMemoryPool), so real per-batch adapter capacity is
        # max_ofts_per_batch - 1, not max_ofts_per_batch. Requiring
        # max_loaded_ofts >= max_ofts_per_batch (the prior bound) let the
        # minimum legal configuration already overcommit real capacity by
        # exactly one slot: with wire-loaded (non-reloadable, never-evicted)
        # adapters, loading max_loaded_ofts of them could then always fail
        # to admit the last one, since only max_ofts_per_batch - 1 real
        # slots ever exist.
        assert server_args.max_loaded_ofts >= server_args.max_ofts_per_batch - 1, (
            "max_loaded_ofts should be greater than or equal to "
            "max_ofts_per_batch - 1 (buffer slot 0 is always reserved for "
            "the base/identity placeholder, so real per-batch adapter "
            "capacity is max_ofts_per_batch - 1). "
            f"max_loaded_ofts={server_args.max_loaded_ofts}, "
            f"max_ofts_per_batch={server_args.max_ofts_per_batch}"
        )

    if server_args.enable_oft:
        assert server_args.oft_type in OFT_TYPE_CHOICES, (
            f"--oft-type must be one of {OFT_TYPE_CHOICES}, got "
            f"{server_args.oft_type!r}."
        )

        # Double-buffer OFT sizing: OFTMemoryPool.__init__ sets active_idx=1,
        # staging_idx=max_ofts_per_batch-1. At max_ofts_per_batch==2 those
        # collide (active==staging==1), so activate()'s staging->active copy
        # would corrupt the active slot instead of promoting it. Fail loud
        # here rather than at pool-init time.
        if server_args.oft_double_buffer:
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

        # Prefill CUDA graphs are supported through the prefill-graph protocol
        # mirrored from srt/lora (OFTManager.init_prefill_cuda_graph_batch_info
        # / can_use_prefill_cuda_graph and TritonOFTBackend's static
        # prefill_cuda_graph_batch_info). Configurations the protocol excludes
        # (DP attention, MoE-expert OFT, mixed base/adapter batches on the
        # single-adapter fast path) replay eagerly per batch through
        # PrefillCudaGraphRunner.can_run_graph rather than disabling the
        # backend here.

        # Expand target modules (OFT-specific "all"/embed/lm_head handling).
        # Normalize through a local -- not server_args.oft_target_modules
        # directly -- because ServerArgs is read-only once __post_init__
        # reaches materialize_declarations() (well before this call), so
        # writes must go through _late_resolution below.
        oft_target_modules = server_args.oft_target_modules
        if oft_target_modules:
            oft_target_modules = set(oft_target_modules)
            if "all" in oft_target_modules:
                assert (
                    len(oft_target_modules) == 1
                ), "If 'all' is specified in --oft-target-modules, it should be the only module specified."
                oft_target_modules = set(SUPPORTED_OFT_TARGET_MODULES)

                # OFT currently only supports torch_native backend,
                # which does not support embedding / lm_head layers yet.
                logger.warning(
                    "OFT backend does not yet support embedding or lm_head layers; "
                    "dropping 'embed_tokens' and 'lm_head' from --oft-target-modules=all."
                )
                oft_target_modules.discard("embed_tokens")
                oft_target_modules.discard("lm_head")

        # OFT is not decode-CUDA-graph-safe either, but only for one specific
        # configuration: dynamically-loaded adapters served via the native-RPC
        # ("sibling") implementation, when MoE expert modules are targeted AND
        # the model actually has MoE layers -- both halves of the same gate
        # OFTManager._install_moe_oft_wrappers and OFTMemoryPool.
        # _declare_expert_groups use to decide whether expert-OFT buffers exist
        # at all (their gate_up_proj/gate_proj/up_proj/down_proj target-module
        # check plus their non-empty FusedMoE scan, which this pre-model-load
        # check approximates via _model_has_moe_layers above). The names alone
        # are not enough: a dense model's MLP uses those same names, and
        # --oft-target-modules=all expands to include them, so checking only
        # the names would strip decode CUDA graphs from dense deployments that
        # have no expert-OFT buffers and never take the path below at all.
        #
        # In the plain native-RPC pool, buffer slot 0
        # (memory_pool.active_idx) is permanently reserved by the boot-time
        # base-request registration -- OFTMemoryPool.
        # allocate_buffer_slot_with_eviction (oft/mem_pool.py), the sole
        # admission path since the on-disk lazy admission path was retired,
        # excludes uid=None from its eviction candidates
        # outright (Task 4b review fix), so a real dynamically-loaded
        # adapter can genuinely never occupy it -- meaning
        # OFTManager._compute_moe_multi_tenant_slot_ids always takes its
        # general, per-token multi-tenant branch for any real adapter. That
        # branch allocates a FRESH routing tensor on
        # every prepare_oft_batch call, which has no pointer stability across
        # CUDA-graph capture and replay (prepare_oft_batch runs outside the
        # capture region -- see decode_cuda_graph_runner.py's
        # prepare_oft_batch/_prepare_oft_replay_batch calls): a
        # captured decode graph replays against the stale capture-time
        # tensor, silently applying an unrelated adapter's rotation (or
        # identity) instead of erroring. Same class of failure as the OFT
        # prefill-graph freeze above. The dense OFT path is unaffected (its
        # own per-token weight_indices mechanism is a different,
        # already-correct code path), and oft_impl="staged" is unaffected too
        # (its double-buffer activate() copies real adapter data in place
        # into active_idx, so the single-adapter fast path stays genuinely
        # correct under decode-graph replay there). The
        # 2026-09-01-oft-moe-cuda-graph-dual-capture plan's Tasks 1-4 added
        # the persistent-buffer + dual-capture mechanism that makes this
        # configuration safe under decode-graph replay -- but only when it
        # actually engages: decode_cuda_graph_runner.py's
        # _resolve_record_oft_variant_graph gates it on effective per-batch
        # adapter capacity (max_ofts_per_batch - 1) being >= 1, i.e. at least
        # one real (non-base) adapter buffer slot existing at all. Below that
        # (max_ofts_per_batch == 1), dual-capture never engages -- only the
        # single fast-path graph is captured -- so this guard must keep
        # disabling decode CUDA graphs for that residual case rather than
        # silently serve wrong rotations under replay.
        #
        # --enable-dp-attention is a SECOND, independent reason to keep
        # disabling: _resolve_record_oft_variant_graph also excludes it
        # outright (final whole-branch review's C1) because
        # _compute_moe_multi_tenant_slot_ids's cross-rank MoE token gathering
        # is not supported by this per-rank persistent buffer -- so
        # dual-capture never engages for a DP-attention server EITHER,
        # regardless of capacity. Without this guard also disabling decode
        # CUDA graphs for that combination, a DP-attention server would be
        # "eligible but never dual-captured": the single fast-path graph
        # alone is not safe here the way it is for the capacity-only case
        # above (a real adapter is never guaranteed to land at
        # memory_pool.active_idx), so this must unconditionally disable,
        # exactly like the pre-plan behavior for this whole configuration.
        moe_expert_target_modules = MOE_EXPERT_TARGET_MODULES & (
            oft_target_modules or set()
        )
        if (
            server_args.oft_impl == "sibling"
            and moe_expert_target_modules
            and server_args.cuda_graph_config is not None
        ):
            from sglang.srt.model_executor.cuda_graph_config import Backend

            # Effective per-batch adapter capacity (buffer slot 0 is always
            # reserved for the base/identity placeholder). Checked before the
            # expensive _model_has_moe_layers call below, mirroring
            # _resolve_record_oft_variant_graph's own capacity check exactly.
            capacity = effective_oft_capacity(server_args)
            insufficient_capacity = capacity < 1
            dp_attention_unsupported = server_args.enable_dp_attention

            # _model_has_moe_layers last: it resolves (and caches) the model
            # config, so servers whose decode graphs are already disabled --
            # and every non-MoE-targeting or sufficient-capacity,
            # non-DP-attention server above -- never pay for it.
            if (
                server_args.cuda_graph_config.decode.backend != Backend.DISABLED
                and (insufficient_capacity or dp_attention_unsupported)
                and _model_has_moe_layers(server_args)
            ):
                logger.warning(
                    "enable_oft with oft_impl=sibling targeting MoE expert "
                    "modules %s is incompatible with decode CUDA graphs "
                    "(effective per-batch adapter capacity %s%s, so the "
                    "dual-capture mechanism in docs/superpowers/plans/"
                    "2026-09-01-oft-moe-cuda-graph-dual-capture.md never "
                    "engages and only the single fast-path graph is "
                    "captured); disabling the decode CUDA graph (was "
                    "backend=%s).%s",
                    sorted(moe_expert_target_modules),
                    capacity,
                    " < 1" if insufficient_capacity else " is sufficient",
                    server_args.cuda_graph_config.decode.backend,
                    (
                        " --enable-dp-attention is not supported by this "
                        "mechanism regardless of capacity."
                        if dp_attention_unsupported
                        else " Increase --max-ofts-per-batch to >= 2 to keep "
                        "decode CUDA graphs enabled for this configuration."
                    ),
                )
                server_args.cuda_graph_config.decode.backend = Backend.DISABLED

        # Ensure sufficient information is provided for OFT initialization.
        assert (
            server_args.max_oft_block_size and oft_target_modules
        ), "You need to specify both --max-oft-block-size and --oft-target-modules for OFT initialization."

        if server_args.max_oft_chunk_size is not None:
            assert (
                16 <= server_args.max_oft_chunk_size <= 128
                and (server_args.max_oft_chunk_size & (server_args.max_oft_chunk_size - 1)) == 0
            ), "--max-oft-chunk-size must be a power of 2 between 16 and 128."

        server_args._late_resolution(
            "validate_oft_args",
            oft_target_modules=oft_target_modules,
        )
