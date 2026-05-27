from dataclasses import dataclass
from typing import Iterable, Optional, Set, Tuple, Union

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.hf_transformers_utils import AutoConfig


_MISSING_CONFIG_ATTR = object()


def get_hf_config_attr(config: AutoConfig, attr_name: str):
    value = getattr(config, attr_name, _MISSING_CONFIG_ATTR)
    if value is not _MISSING_CONFIG_ATTR:
        return value

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        value = getattr(text_config, attr_name, _MISSING_CONFIG_ATTR)
        if value is not _MISSING_CONFIG_ATTR:
            return value

    raise AttributeError(
        f"{type(config).__name__} has no attribute {attr_name!r} "
        "on the top-level config or text_config"
    )


def get_hf_text_config(config: AutoConfig) -> AutoConfig:
    return getattr(config, "text_config", config)


@dataclass
class OFTBatchInfo:
    # The forward mode is using CUDA Graph.
    use_cuda_graph: bool

    # Batch size
    bs: int

    # Number of segments. For torch_native backend, it is equal to batch size.
    num_segments: int

    # Indice pointers of each segment in shape (num_segments + 1, )
    seg_indptr: torch.Tensor

    # The index of OFT adapter used by each segment, in shape (num_segments,)
    weight_indices: torch.Tensor

    # Block sizes of each OFT adapter, in shape (oft_num,)
    # (analogous to lora_ranks in LoRA, but controls orthogonal block size)
    oft_block_sizes: torch.Tensor

    # Maximum segment length of current batch
    max_len: Optional[int]

    # Lengths of each segments in shape (num_segments,)
    seg_lens: Optional[torch.Tensor]

    # The logical (re)ordering of input rows (tokens), in shape (num_tokens,)
    permutation: Optional[torch.Tensor]


def _intermediate_for_layer(config: AutoConfig, layer_idx: int) -> int:
    """Pick MLP intermediate size for a given layer.

    DeepSeek-style MoE configs have a leading block of dense layers
    (``first_k_dense_replace``) followed by routed-MoE layers that use a
    smaller ``moe_intermediate_size``. Models without those attributes fall
    back to ``intermediate_size`` for every layer.
    """
    n_routed = getattr(config, "n_routed_experts", None)
    moe_inter = getattr(config, "moe_intermediate_size", None)
    first_dense = getattr(config, "first_k_dense_replace", 0)
    if n_routed and moe_inter and layer_idx >= first_dense:
        return moe_inter
    return config.intermediate_size


def get_hidden_dim(
    module_name: str,
    config: AutoConfig,
    base_model: torch.nn.Module,
    layer_idx: int,
    oft_added_vocab_size: int = 0,
) -> Tuple[int]:
    """
    Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
    """

    if hasattr(base_model, "get_hidden_dim"):
        return base_model.get_hidden_dim(module_name, layer_idx)
    else:
        config = get_hf_text_config(config)
        """
        WARNING: get_hidden_dim() is not defined,
        which is used to get the hidden dim for different OFT modules.
        Use the default one, but please check if it is correct for your model.
        Please implement the function in the model class if it is not.
        You can reference this function in llama.py.
        """
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        # Multi-Latent Attention (DeepSeek v2/v3/v4): factored Q/KV projections.
        # Detected by the presence of ``kv_lora_rank`` on the HF config.
        kv_lora_rank = getattr(config, "kv_lora_rank", None)
        if kv_lora_rank is not None and module_name in {
            "q_proj",
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "fused_qkv_a_proj_with_mqa",
            "o_proj",
        }:
            qk_nope = getattr(config, "qk_nope_head_dim", 0)
            qk_rope = getattr(config, "qk_rope_head_dim", 0)
            v_head = getattr(config, "v_head_dim", qk_nope)
            qk_head = qk_nope + qk_rope
            num_heads = config.num_attention_heads
            q_lora_rank = getattr(config, "q_lora_rank", None)
            if module_name == "q_proj":
                return config.hidden_size, num_heads * qk_head
            if module_name == "q_a_proj":
                return config.hidden_size, q_lora_rank
            if module_name == "q_b_proj":
                return q_lora_rank, num_heads * qk_head
            if module_name == "kv_a_proj_with_mqa":
                return config.hidden_size, kv_lora_rank + qk_rope
            if module_name == "kv_b_proj":
                return kv_lora_rank, num_heads * (qk_nope + v_head)
            if module_name == "fused_qkv_a_proj_with_mqa":
                return config.hidden_size, q_lora_rank + kv_lora_rank + qk_rope
            # o_proj differs from non-MLA: input dim is num_heads * v_head_dim.
            return num_heads * v_head, config.hidden_size
        if module_name == "qkv_proj":
            return config.hidden_size, head_dim * (
                config.num_attention_heads + config.num_key_value_heads * 2
            )
        elif module_name == "o_proj":
            return (
                head_dim * config.num_attention_heads,
                config.hidden_size,
            )
        elif module_name == "gate_up_proj":
            # MoE layers (DeepSeek): use moe_intermediate_size when the layer
            # index is past first_k_dense_replace.
            inter = _intermediate_for_layer(config, layer_idx)
            return config.hidden_size, inter * 2
        elif module_name == "down_proj":
            inter = _intermediate_for_layer(config, layer_idx)
            return inter, config.hidden_size
        elif module_name == "embed_tokens":
            # For embedding: input is vocab_size (as embedding lookup), output is hidden_size
            # if contain extra tokens will be added; otherwise is 0.
            return config.vocab_size + oft_added_vocab_size, config.hidden_size
        elif module_name == "lm_head":
            # For lm_head: input is hidden_size, output is vocab_size
            # if contain extra tokens will be added; otherwise is 0.
            return config.hidden_size, config.vocab_size + oft_added_vocab_size
        else:
            raise NotImplementedError(
                "get_hidden_dim not implemented for " + module_name
            )


def get_normalized_target_modules(
    target_modules: Union[str, Iterable[str]],
) -> set[str]:
    """
    Mapping a list of target module name to names of the normalized OFT weights.
    Handles both base module names (e.g., "gate_proj") and prefixed module names (e.g., "feed_forward.gate_proj").

    Also handles PEFT shorthand strings like "all-linear" or "all" by returning
    {"all"} as a sentinel value (the caller should check for "all" and fall
    back to the CLI --oft-target-modules to determine the concrete module set).
    """
    # Handle PEFT shorthand strings — these cannot be resolved to concrete
    # module names without inspecting the base model, so we return {"all"}
    # and let the caller fall back to the CLI --oft-target-modules.
    if isinstance(target_modules, str):
        return {"all"}

    params_mapping = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "embed_tokens": "embed_tokens",
        "vocab_emb": "embed_tokens",
        "embeddings": "embed_tokens",
        "word_embeddings": "embed_tokens",
        "lm_head": "lm_head",
        "output": "lm_head",
        "unembed_tokens": "lm_head",
    }

    result = set()
    for name in target_modules:
        base_name = name.split(".")[-1]
        normalized_name = params_mapping.get(base_name, base_name)
        result.add(normalized_name)
    return result


def get_stacked_multiply(module_name: str) -> int:
    """
    Mapping an OFT module name to its magnification at output dimension.
    """
    stacked_rank = {
        "qkv_proj": 3,
        "gate_up_proj": 2,
    }
    return stacked_rank[module_name] if module_name in stacked_rank else 1


def get_target_module_name(full_module_name: str, target_modules: Set[str]) -> str:
    """
    Get the target module name in target_modules that can match full_module_name.

    If there is a target module name in target_modules that can match full_module_name, return this name.
    Else raise ValueError.
    """
    for target_module in target_modules:
        if target_module in full_module_name:
            return target_module
    raise ValueError(
        f"Cannot find target module name for {full_module_name} in {target_modules}"
    )


EMBEDDING_NAMES = ["embed_tokens", "lm_head"]
ROW_PARALLELISM_LINEAR_OFT_NAMES = ["o_proj", "down_proj", "wo_b"]


def detect_canonical_split_active() -> bool:
    """True iff the global server args indicate canonical-split OFT will run.

    Detection rule: both 'gate_proj' AND 'up_proj' present in
    oft_target_modules, AND enable_oft is True. This mirrors the canonical-
    split signature emitted by Megatron-Bridge's
    ``OFTLinearGroupedSplitFC1UpGate`` (which writes two separate
    gate_proj.oft_R / up_proj.oft_R adapter tensors). Legacy single-R would
    emit only one of them (or use the fused gate_up_proj name).

    Called at process_weights_after_loading time, BEFORE OFT adapters have
    been loaded -- we infer the future configuration from server args.

    CONTRACT: this is the single source of truth for "is canonical-split
    active for this run?". Any quant scheme that branches on this must call
    this helper rather than re-implementing the rule.

    SGLANG_DISABLE_PREPACK_SPLIT=1 forces this to return False regardless of
    server args -- used for the launcher A/B baseline (Task 7).
    """
    from sglang.srt.utils import get_bool_env_var

    if get_bool_env_var("SGLANG_DISABLE_PREPACK_SPLIT"):
        return False
    try:
        from sglang.srt.server_args import get_global_server_args
    except ImportError:
        return False
    try:
        args = get_global_server_args()
    except Exception:
        return False
    if not getattr(args, "enable_oft", False):
        return False
    targets = getattr(args, "oft_target_modules", None)
    if not targets:
        return False
    targets_set = set(targets) if not isinstance(targets, set) else targets
    return ("gate_proj" in targets_set) and ("up_proj" in targets_set)


def assert_canonical_split_supported(scheme_name: str) -> None:
    """Fail-fast when canonical-split OFT is requested but this MoE scheme
    doesn't implement split-aware prepack/forward.

    Called from process_weights_after_loading. If canonical-split is
    detected via server args but the scheme has no split-aware path, raise
    NotImplementedError now (at model load) instead of failing at first
    forward inside the Triton MoE runner.
    """
    if not detect_canonical_split_active():
        return
    raise NotImplementedError(
        f"Canonical-split OFT (gate_proj + up_proj both in "
        f"--oft-target-modules) is not yet supported for MoE scheme "
        f"{scheme_name!r}. Only INT4 W4A16 via CompressedTensorsWNA16MoE "
        "has a split-aware kernel path at this time. "
        "For FP8/NVFP4, either use legacy single-R OFT (--oft-type oft) "
        "or wait for the follow-up plan to implement the split-aware "
        "kernel for this scheme."
    )


def generate_sequence_lengths(
    forward_batch: ForwardBatch, device: Optional[torch.device] = None
) -> torch.Tensor:

    device = torch.get_default_device() if device is None else device
    with torch.device(device):
        if forward_batch.forward_mode.is_decode_or_idle():
            seg_lens = torch.ones(forward_batch.batch_size, dtype=torch.int32)
        elif forward_batch.forward_mode.is_target_verify():
            seg_lens = torch.full(
                size=(forward_batch.batch_size,),
                fill_value=forward_batch.spec_info.draft_token_num,
                dtype=torch.int32,
            )
        elif forward_batch.forward_mode.is_extend():
            seg_lens = (
                forward_batch.extend_seq_lens
                if forward_batch.extend_seq_lens.device == device
                else torch.tensor(
                    forward_batch.extend_seq_lens_cpu,
                    dtype=torch.int32,
                )
            )
        else:
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")
    return seg_lens
