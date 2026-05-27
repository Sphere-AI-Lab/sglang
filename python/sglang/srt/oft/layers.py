import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.oft.backend.base_backend import BaseOFTBackend
from sglang.srt.oft.utils import OFTBatchInfo


import atexit as _atexit
import os as _os
import sys as _sys

_FUSED_INSTRUMENT_ENABLED = (
    _os.environ.get("SGLANG_OFT_FUSED_INSTRUMENT", "0").strip()
    not in ("0", "false", "False")
)
_FUSED_LOG_EVERY = int(_os.environ.get("SGLANG_OFT_FUSED_LOG_EVERY", "200"))
_FUSED_COUNTERS = {
    "engaged_qkv_project": 0,
    "engaged_gate_up_inputs": 0,
    "fallback_env_disabled": 0,
    "fallback_ineligible": 0,
    "fallback_not_implemented": 0,
    "fallback_runtime_error": 0,
}
_FUSED_M_HISTOGRAM = {"engaged": {}, "fallback": {}}
_QUANTIZED_SHARED_FALLBACK_WARNED = set()


def _fused_m_bucket(m: int) -> str:
    if m <= 16:
        return "le16"
    if m <= 64:
        return "le64"
    if m <= 256:
        return "le256"
    if m <= 1024:
        return "le1024"
    return "gt1024"


def _record_fused_outcome(reason: str, tokens: int) -> None:
    if not _FUSED_INSTRUMENT_ENABLED:
        return
    _FUSED_COUNTERS[reason] = _FUSED_COUNTERS.get(reason, 0) + 1
    cat = "engaged" if reason.startswith("engaged") else "fallback"
    bucket = _fused_m_bucket(tokens)
    hist = _FUSED_M_HISTOGRAM[cat]
    hist[bucket] = hist.get(bucket, 0) + 1
    total = sum(_FUSED_COUNTERS.values())
    if _FUSED_LOG_EVERY > 0 and total % _FUSED_LOG_EVERY == 0:
        _sys.stderr.write(
            f"[fused-instrument] total={total} calls={_FUSED_COUNTERS} "
            f"engaged_M={_FUSED_M_HISTOGRAM['engaged']} "
            f"fallback_M={_FUSED_M_HISTOGRAM['fallback']}\n"
        )
        _sys.stderr.flush()


@_atexit.register
def _dump_fused_instrument_final():
    if not _FUSED_INSTRUMENT_ENABLED:
        return
    _sys.stderr.write(
        f"[fused-instrument][final] calls={_FUSED_COUNTERS} "
        f"engaged_M={_FUSED_M_HISTOGRAM['engaged']} "
        f"fallback_M={_FUSED_M_HISTOGRAM['fallback']}\n"
    )
    _sys.stderr.flush()


def _is_unquantized_dense_linear(layer) -> bool:
    """Whether a base linear is plain dense (not FP8 / INT4 / NVFP4 quantized).

    The split path below applies R per slice on the BF16 weight directly;
    quantized bases need per-quant kernels and currently fail loud.
    """
    quant_method = getattr(layer, "quant_method", None)
    if quant_method is None:
        return False
    if not hasattr(layer, "weight"):
        return False
    name = quant_method.__class__.__name__
    return name in {"UnquantizedLinearMethod", "_FakeUnquantizedMethod"}


def _split_stacked_R(R: torch.Tensor, num_slices: int) -> List[torch.Tensor]:
    if R.shape[-3] % num_slices != 0:
        raise RuntimeError(
            f"OFT R has {R.shape[-3]} blocks, not divisible by num_slices={num_slices}."
        )
    per_slice_blocks = R.shape[-3] // num_slices
    return list(torch.split(R, per_slice_blocks, dim=-3))


def _apply_block_R(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """CPU/debug einsum reference for block-diagonal rotation.

    Mirrors ``OFTRotationModule.forward`` semantics: per block,
    ``out_block = x_block @ R_block``.
    """
    block = R.shape[-1]
    num_blocks = R.shape[-3]
    return torch.einsum(
        "...nb,nbc->...nc", x.view(*x.shape[:-1], num_blocks, block), R
    ).reshape_as(x)


def _first_stacked_R_slice(R: torch.Tensor, split_count: int) -> torch.Tensor:
    blocks_per_slice = R.shape[-3] // split_count
    return R[..., :blocks_per_slice, :, :].contiguous()


def _warn_quantized_shared_fallback(wrapper_name: str, quant_method_name: str) -> None:
    key = (wrapper_name, quant_method_name)
    if key in _QUANTIZED_SHARED_FALLBACK_WARNED:
        return
    _QUANTIZED_SHARED_FALLBACK_WARNED.add(key)
    logger.warning(
        "%s with quant_method=%s is using shared-input OFT rotation fallback; "
        "exact split-slice OFT is currently dense-only.",
        wrapper_name,
        quant_method_name,
    )


def _batched_equal_output_linear(
    input_slices: List[torch.Tensor],
    weight_slices: List[torch.Tensor],
    bias_slices: List[Optional[torch.Tensor]],
) -> torch.Tensor:
    """Run equal-output split linears as one strided batched GEMM.

    All input slices share shape ``(..., hidden)``. All weight slices must
    share shape ``(out, hidden)``. Output layout matches concatenating
    individual ``F.linear`` calls in slice order.
    """
    if not input_slices:
        raise ValueError("input_slices must be non-empty")
    out_features = weight_slices[0].shape[0]
    if any(weight.shape[0] != out_features for weight in weight_slices):
        raise ValueError("all weight slices must have the same output size")

    input_shape = input_slices[0].shape
    hidden = input_shape[-1]
    flat_inputs = [input_slice.reshape(-1, hidden) for input_slice in input_slices]
    stacked_input = torch.stack(flat_inputs, dim=0)
    stacked_weight = torch.stack(weight_slices, dim=0)
    out = torch.bmm(stacked_input, stacked_weight.transpose(1, 2))
    if bias_slices[0] is not None:
        out = out + torch.stack(bias_slices, dim=0)[:, None, :]
    out = out.permute(1, 0, 2).reshape(*input_shape[:-1], -1)
    return out


def split_dense_merged_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_sizes: List[int],
    R: torch.Tensor,
    oft_backend: Optional[BaseOFTBackend] = None,
) -> torch.Tensor:
    """Dense BF16 split-OFT forward for a merged projection.

    For unit/reference tests, ``R`` may be a single 3D stacked rotation tensor
    with ``len(output_sizes) * blocks_per_slice`` blocks along the block dim.
    In serving, ``R`` is the live 4D SGLang buffer
    ``(max_ofts_per_batch, stacked_blocks, block, block)``. That 4D path must
    route through the OFT backend so adapter-slot selection and batch metadata
    are honored.
    """
    W_slices = torch.split(weight, output_sizes, dim=0)
    if bias is None:
        b_slices: List[Optional[torch.Tensor]] = [None] * len(output_sizes)
    else:
        b_slices = list(torch.split(bias, output_sizes, dim=0))

    if R.dim() == 4:
        if oft_backend is None:
            raise RuntimeError("4D split OFT buffers require an OFT backend.")
        # === Fused split-OFT fast paths ===
        # QKV uses one rotate-project kernel. FC1/gate-up rotates into two
        # input tensors, then keeps the large projection GEMMs in cuBLAS.
        # Falls through to the legacy code below on miss.
        _tokens = x.shape[0]
        _fused_disabled = _os.environ.get(
            "SGLANG_OFT_DISABLE_FUSED_ROTATE_PROJECT", ""
        ).strip() not in ("", "0", "false", "False")
        if _fused_disabled:
            _record_fused_outcome("fallback_env_disabled", _tokens)
        else:
            _common_eligible = (
                x.dtype == torch.bfloat16
                and R.dtype == torch.bfloat16
                and weight.dtype == torch.bfloat16
                and x.is_contiguous()
                and weight.is_contiguous()
            )
            if (
                len(output_sizes) == 3
                and hasattr(oft_backend, "run_fused_rotate_project")
                and _common_eligible
            ):
                try:
                    _result = oft_backend.run_fused_rotate_project(
                        x, R, weight, output_sizes, bias,
                    )
                    _record_fused_outcome("engaged_qkv_project", _tokens)
                    return _result
                except NotImplementedError as _exc:
                    _record_fused_outcome("fallback_not_implemented", _tokens)
                    logger.debug("Fused rotate-project NotImplementedError: %s", _exc)
                except RuntimeError as _exc:
                    _record_fused_outcome("fallback_runtime_error", _tokens)
                    logger.debug("Fused rotate-project RuntimeError: %s", _exc)
            elif (
                len(output_sizes) == 2
                and output_sizes[0] == output_sizes[1]
                and hasattr(oft_backend, "run_fused_gate_up_inputs")
                and _common_eligible
            ):
                try:
                    x_gate, x_up = oft_backend.run_fused_gate_up_inputs(x, R)
                    _record_fused_outcome("engaged_gate_up_inputs", _tokens)
                    # Two F.linears + cat outperforms _batched_equal_output_linear
                    # at FC1 shapes: stack copies ~34MB of weight views per call,
                    # bmm with batch=2 underperforms two cuBLAS gemms, and the
                    # permute+reshape forces another contiguous copy.
                    return torch.cat(
                        [
                            F.linear(x_gate, W_slices[0], b_slices[0]),
                            F.linear(x_up, W_slices[1], b_slices[1]),
                        ],
                        dim=-1,
                    )
                except NotImplementedError as _exc:
                    _record_fused_outcome("fallback_not_implemented", _tokens)
                    logger.debug("Fused gate/up input NotImplementedError: %s", _exc)
                except RuntimeError as _exc:
                    _record_fused_outcome("fallback_runtime_error", _tokens)
                    logger.debug("Fused gate/up input RuntimeError: %s", _exc)
            else:
                _record_fused_outcome("fallback_ineligible", _tokens)
        # === End fused fast path; legacy code below is preserved verbatim ===
        if len(output_sizes) == 3:
            rotated = oft_backend.run_qkv_oft(x, R)
        elif len(output_sizes) == 2:
            rotated = oft_backend.run_gate_up_oft(x, R)
        else:
            raise RuntimeError(
                f"Unsupported split OFT slice count: {len(output_sizes)}."
            )
        input_slices = list(torch.split(rotated, x.shape[-1], dim=-1))
        if len(input_slices) != len(output_sizes):
            raise RuntimeError(
                f"Backend returned {len(input_slices)} input slices for "
                f"{len(output_sizes)} output slices."
            )
    elif R.dim() == 3:
        R_slices = _split_stacked_R(R, len(output_sizes))
        input_slices = [_apply_block_R(x, R_slice) for R_slice in R_slices]
    else:
        raise RuntimeError(f"Expected 3D or 4D OFT R buffer, got shape={tuple(R.shape)}.")

    if len(output_sizes) == 2 and output_sizes[0] == output_sizes[1]:
        return _batched_equal_output_linear(input_slices, W_slices, b_slices)

    if len(output_sizes) == 3 and output_sizes[1] == output_sizes[2]:
        q_out = F.linear(input_slices[0], W_slices[0], b_slices[0])
        kv_out = _batched_equal_output_linear(
            input_slices[1:],
            W_slices[1:],
            b_slices[1:],
        )
        return torch.cat([q_out, kv_out], dim=-1)

    outs = [
        F.linear(input_slice, W_slice, b_slice)
        for input_slice, W_slice, b_slice in zip(input_slices, W_slices, b_slices)
    ]
    return torch.cat(outs, dim=-1)


class BaseLayerWithOFT(nn.Module):
    # External callers (e.g. deepseek_v2.py reading o_proj.reduce_results,
    # gate_up_proj.output_size_per_partition, q_b_proj.quant_method) inspect
    # base-linear metadata the wrapper doesn't redefine. __getattr__ delegates
    # these specific names to base_layer; the whitelist avoids shadowing
    # nn.Module's bookkeeping for parameters/submodules (e.g. `weight`, which
    # the wrapper registers itself in __init__).
    _BASE_LAYER_PROXY_ATTRS = frozenset(
        {
            "bias",
            "gather_output",
            "input_is_parallel",
            "input_size",
            "input_size_per_partition",
            "output_size",
            "output_size_per_partition",
            "quant_method",
            "reduce_results",
            "skip_bias_add",
            "tp_rank",
            "tp_size",
        }
    )

    def __init__(
        self,
        base_layer: nn.Module,
        oft_backend: BaseOFTBackend,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        self.set_oft: bool = False
        self.oft_backend: BaseOFTBackend = oft_backend
        # Snapshot the Bridge-parity flag at construction. The flag is a
        # server-level knob fixed for the process lifetime; caching here
        # avoids a per-forward server-args lookup while keeping a single
        # source of truth (`is_parity_mode()`).
        from sglang.srt.oft.parity_dequant import is_parity_mode

        self._oft_parity_mode: bool = is_parity_mode()
        if hasattr(self.base_layer, "weight"):
            self.weight = self.base_layer.weight

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_oft_info(self, *args):
        pass

    def get_local_tp_rank(self) -> int:
        return getattr(self.base_layer, "tp_rank", 0)

    def get_oft_input_dim(self) -> int:
        if hasattr(self.base_layer, "input_size_per_partition"):
            return self.base_layer.input_size_per_partition
        if hasattr(self.base_layer, "input_size"):
            return self.base_layer.input_size
        raise NotImplementedError(
            f"Cannot infer local OFT input dim for {type(self.base_layer).__name__}"
        )

    def slice_oft_r_weights(self, R: torch.Tensor, tp_rank: int):
        pass

    def __getattr__(self, name: str):
        if name in BaseLayerWithOFT._BASE_LAYER_PROXY_ATTRS:
            try:
                base_layer = super().__getattr__("base_layer")
            except AttributeError:
                raise AttributeError(name) from None
            return getattr(base_layer, name)
        return super().__getattr__(name)


class VocabParallelEmbeddingWithOFT(BaseLayerWithOFT):
    """
    Vocab parallel embedding layer with OFT support (simplified for TP=1, no extra tokens).

    For embedding layers with OFT: output = R @ base_embedding(x)
    where R is the block-diagonal orthogonal rotation applied to the embedding output.
    Unlike LoRA which decomposes the adapter into A (embedding lookup) and B (projection),
    OFT applies a single orthogonal rotation to the base embedding output.

    Note: Embedding is special — since the input is discrete token IDs (not a continuous
    vector), OFT rotates the embedding OUTPUT rather than the input.
    """

    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
        oft_backend: BaseOFTBackend,
    ) -> None:
        super().__init__(base_layer, oft_backend)
        self.weight = base_layer.weight
        self.embed_dim = base_layer.embedding_dim
        self.vocab_size = base_layer.org_vocab_size

    def set_oft_info(
        self,
        new_embeddings_buffer: Optional[torch.Tensor],  # For extra tokens
        embedding_R_buffer: torch.Tensor,
    ):
        """Set OFT buffers for embedding layer.

        Unlike LoRA which needs separate A (embedding) and B (projection) buffers,
        OFT uses a single R buffer for the orthogonal rotation.
        """
        self.set_oft = True
        self.new_embeddings_buffer = new_embeddings_buffer
        self.embedding_R_buffer = embedding_R_buffer  # (num_ofts, num_blocks, block_size, block_size) precomputed R

    def apply_oft(self, base_output: torch.Tensor) -> torch.Tensor:
        """
        Apply OFT to base embedding output.
        Formula: output = base_embedding(input_) @ R

        For embeddings, OFT rotates the output (since input is discrete token IDs).
        This is the same rotation op as run_oft_r_sgemm — just applied to embedding output.
        """

        # Apply OFT rotation to base embedding output: rotated = base_output @ R
        rotated_output = self.oft_backend.run_oft_r_sgemm(
            x=base_output,
            weights=self.embedding_R_buffer,
        )

        return rotated_output

    def extra_token_embedding(
        self, input_: torch.Tensor, base_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Need to impl:

        Process extra tokens (tokens >= vocab_size) by looking up their embeddings
        from the new_embeddings_buffer and replacing them in base_output.

        Args:
            input_: (s,) token IDs
            base_output: (s, embed_dim) base embedding output to be modified in-place

        Returns:
            base_output: (s, embed_dim) modified input base_output (tensor[0,0,0,...]) with extra token embeddings
        """
        # return base_output
        raise NotImplementedError(
            "Error in sglang/python/sglang/srt/oft/layers.py - VocabParallelEmbeddingWithOFT \n"
            "Current SGLang codebase did not support tuned OFT with extra/added tokens. \n"
            "[TODO]: \n"
            "1. Refer to the LoRA extra token implementation for guidance \n"
            "2. And then you need to modified the en/decoder tokenizer - tokenizer_manager.py to support extra_token_embedding in-place. \n"
        )

    def forward(self, input_: torch.Tensor):
        """
        Forward pass with OFT support and CUDA graph compatibility.

        Embedding is special: OFT rotates the OUTPUT (since input is discrete token IDs).
        For all other layers, OFT rotates the INPUT before the base forward.
        """
        # Get base embedding output
        # For tokens >= vocab_size, base_layer will clamp or handle them
        # We mask them to 0 to avoid out-of-bounds access
        added_tokens_mask = input_ > self.vocab_size - 1
        base_output = self.base_layer.forward(input_.masked_fill(added_tokens_mask, 0))

        # [TODO] SGLang did not support extra/added token process; thus, self.extra_token_embedding only return original input_ now
        # Extra tokens - It will replace extra token embedding with self.new_embeddings_buffer's emb (Default is 0)
        if (
            hasattr(self, "new_embeddings_buffer")
            and self.new_embeddings_buffer is not None
        ):
            base_output = self.extra_token_embedding(input_, base_output)

        # Apply OFT if configured (rotate embedding output)
        if self.set_oft:
            base_output = self.apply_oft(base_output)

        return base_output

    def slice_oft_r_weights(self, R: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # OFT R weights are not sliced for embedding (operates on full embedding dim)
        # For TP>1, Need to modify code in: sglang/python/sglang/srt/oft/mem_pool.py
        # return R
        if tp_rank > 1:
            raise NotImplementedError(
                f"VocabParallelEmbeddingWithOFT does not support tensor parallelism > 1. "
                f"Got tp_size={tp_rank}"
            )


class ParallelLMHeadWithOFT(BaseLayerWithOFT):
    """
    Parallel LM Head layer with OFT support (simplified for TP=1).

    The LM head with OFT computes logits = (hidden_states @ R) @ W^T
    where R is the block-diagonal orthogonal rotation applied as input rotation.
    Unlike LoRA which adds: logits = hidden_states @ (W + B @ A)^T
    """

    def __init__(
        self,
        base_layer: ParallelLMHead,
        oft_backend: BaseOFTBackend,
    ) -> None:
        super().__init__(base_layer, oft_backend)
        self.weight = base_layer.weight
        self.embed_dim = base_layer.embedding_dim
        self.vocab_size = base_layer.org_vocab_size

    def set_oft_info(
        self,
        lm_head_R_buffer: torch.Tensor,
    ):
        """Set OFT buffers for LM head layer.

        Unlike LoRA which needs separate A and B buffers,
        OFT uses a single R buffer for the orthogonal rotation.
        """
        self.set_oft = True
        self.lm_head_R_buffer = lm_head_R_buffer  # (num_ofts, num_blocks, block_size, block_size) precomputed R

    def apply_oft(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply OFT input rotation to hidden_states.

        For LM head with OFT: output = (hidden @ R) @ W^T
        The backend rotates the input; the layer then applies the base linear.

        Unlike LoRA which is a parallel branch: base_output + (hidden @ A^T) @ B^T
        OFT is sequential: rotate input first, then base forward.
        """
        # Apply OFT input rotation: rotated_x = x @ R
        rotated_x = self.oft_backend.run_oft_r_sgemm(
            x=x,
            weights=self.lm_head_R_buffer,
        )
        return rotated_x

    def forward(self, hidden_states: torch.Tensor):
        # OFT: rotate input FIRST, then apply base linear
        # (Unlike LoRA which computes base output first, then adds correction)
        if self.set_oft:
            hidden_states = self.apply_oft(hidden_states)

        # Apply base linear transformation on (possibly rotated) input
        base_output = F.linear(
            hidden_states, self.weight, bias=getattr(self.base_layer, "bias", None)
        )

        return base_output

    def slice_oft_r_weights(self, R: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # For TP>1, need to modify code in: sglang/python/sglang/srt/oft/mem_pool.py
        # return R
        if tp_rank > 1:
            raise NotImplementedError(
                f"ParallelLMHeadWithOFT does not support tensor parallelism > 1. "
                f"Got tp_size={tp_rank}"
            )


class ColumnParallelLinearWithOFT(BaseLayerWithOFT):
    def __init__(
        self,
        base_layer: ColumnParallelLinear,
        oft_backend: BaseOFTBackend,
    ) -> None:
        super().__init__(base_layer, oft_backend)

    def set_oft_info(
        self,
        R_buffer: torch.Tensor,
    ):
        self.set_oft = True
        self.R_buffer = R_buffer

    def apply_oft(self, x: torch.Tensor) -> torch.Tensor:
        # OFT input rotation: rotated_x = x @ R
        # The rotated input is then fed into the base linear layer.
        # Unlike LoRA which is a parallel additive branch.
        rotated_x = self.oft_backend.run_oft_r_sgemm(
            x=x,
            weights=self.R_buffer,
        )
        return rotated_x

    def apply_input_rotation(
        self,
        x: torch.Tensor,
        *,
        transpose: bool = False,
        n_groups: int = 1,
    ) -> torch.Tensor:
        """Apply only the OFT input rotation without running the base linear.

        Some MLA serving paths algebraically absorb a ColumnParallelLinear's
        weight into attention-side batched GEMMs and never call ``forward``.
        This hook lets those paths preserve the same ``x @ R`` semantics.  When
        the absorbed algebra moves the rotation to the query side, callers can
        request ``transpose=True`` to apply ``x @ R.T``.
        """
        if not self.set_oft:
            return x
        if n_groups <= 0:
            raise ValueError(f"n_groups must be positive, got {n_groups}")

        orig_shape = x.shape
        if x.dim() != 2:
            x = x.reshape(-1, orig_shape[-1])

        weights = self.R_buffer.transpose(-1, -2) if transpose else self.R_buffer
        if n_groups == 1:
            rotated = self.oft_backend.run_oft_r_sgemm(x=x, weights=weights)
        else:
            rotated = self.oft_backend.run_grouped_oft_r_sgemm(
                x=x,
                weights=weights,
                n_groups=n_groups,
            )

        if rotated.shape != orig_shape:
            rotated = rotated.reshape(orig_shape)
        return rotated

    def forward(self, input_: torch.Tensor):
        # OFT: rotate input FIRST, then apply base forward
        if self.set_oft:
            input_ = self.apply_oft(input_)

        if self.set_oft and self._oft_parity_mode:
            from sglang.srt.oft.parity_dequant import parity_linear

            output_parallel = parity_linear(self.base_layer, input_)
        else:
            # duplicate the logic in ColumnParallelLinear
            bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
            output_parallel = self.base_layer.quant_method.apply(
                self.base_layer, input_, bias
            )

        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    def forward_quantized_shared_oft(self, input_: torch.Tensor, split_count: int):
        quant_method = getattr(self.base_layer, "quant_method", None)
        if quant_method is None or not hasattr(quant_method, "apply"):
            raise RuntimeError(
                f"{type(self).__name__} quantized shared OFT fallback requires "
                f"a quant_method with apply(); got {type(quant_method).__name__}."
            )
        _warn_quantized_shared_fallback(
            type(self).__name__,
            type(quant_method).__name__,
        )
        shared_R = _first_stacked_R_slice(self.R_buffer, split_count)
        input_ = self.oft_backend.run_oft_r_sgemm(x=input_, weights=shared_R)

        if getattr(self, "_oft_parity_mode", False):
            from sglang.srt.oft.parity_dequant import parity_linear

            output_parallel = parity_linear(self.base_layer, input_)
        else:
            bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
            output_parallel = quant_method.apply(self.base_layer, input_, bias)

        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    def slice_oft_r_weights(self, R: torch.Tensor, tp_rank: int):
        # OFT R operates on the input dimension for ColumnParallel (input rotation).
        # R is block-diagonal: shape (num_blocks, block_size, block_size) precomputed.
        # For ColumnParallel, input is replicated across TP ranks, so no slicing needed
        # (analogous to LoRA A for ColumnParallel which is also not sliced).
        return R


class MergedColumnParallelLinearWithOFT(ColumnParallelLinearWithOFT):
    """gate_up_proj OFT wrapper.

    CanonicalOFT trains independent rotations for ``gate`` and ``up``; the
    server-side R_buffer stacks both rotations along the block dim
    (``stacked_multiply == 2``). The forward splits ``W_fc1`` along output dim
    into ``[W_gate; W_up]`` and runs two BF16 GEMMs with their respective
    rotated inputs.

    Quantized bases currently fail loud — per-quant split GEMM lands in a
    follow-up plan.
    """

    def __init__(
        self,
        base_layer: MergedColumnParallelLinear,
        oft_backend: BaseOFTBackend,
    ) -> None:
        super().__init__(base_layer, oft_backend)

    def slice_oft_r_weights(self, R: torch.Tensor, tp_rank: int):
        return R

    def forward(self, input_: torch.Tensor):
        if not self.set_oft:
            return self.base_layer.forward(input_)

        if self.R_buffer.shape[-3] % 2 != 0:
            # Buffer was sized for shared-R; fall back to legacy single-R path.
            return super().forward(input_)

        if not _is_unquantized_dense_linear(self.base_layer):
            return self.forward_quantized_shared_oft(input_, split_count=2)

        output_sizes = [int(s) for s in self.base_layer.output_sizes]
        # Account for TP sharding: weight stored locally is sum/tp_size.
        tp_size = getattr(self.base_layer, "tp_size", 1)
        if tp_size > 1:
            output_sizes = [s // tp_size for s in output_sizes]
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = split_dense_merged_projection(
            input_,
            self.base_layer.weight,
            bias,
            output_sizes,
            self.R_buffer,
            oft_backend=self.oft_backend,
        )

        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias


class QKVParallelLinearWithOFT(ColumnParallelLinearWithOFT):
    """qkv_proj OFT wrapper with stacked R (q/k/v) for CanonicalOFT.

    Slice sizes come from the base layer's GQA-correct ``output_sizes``
    (``[q_proj, k_proj, v_proj]``) so this matches Megatron's
    ``split_qkv_weights`` semantics on the serving side.
    """

    def __init__(
        self,
        base_layer: QKVParallelLinear,
        oft_backend: BaseOFTBackend,
    ) -> None:
        super().__init__(base_layer, oft_backend)

    def slice_oft_r_weights(self, R: torch.Tensor, tp_rank: int):
        return R

    def forward(self, input_: torch.Tensor):
        if not self.set_oft:
            return self.base_layer.forward(input_)

        if self.R_buffer.shape[-3] % 3 != 0:
            return super().forward(input_)

        if not _is_unquantized_dense_linear(self.base_layer):
            return self.forward_quantized_shared_oft(input_, split_count=3)

        bl = self.base_layer
        tp_size = getattr(bl, "tp_size", 1)
        # output_sizes is the post-TP-multiplied total (see QKVParallelLinear.__init__).
        output_sizes = [int(s) // tp_size for s in bl.output_sizes]
        bias = bl.bias if not bl.skip_bias_add else None
        output_parallel = split_dense_merged_projection(
            input_,
            bl.weight,
            bias,
            output_sizes,
            self.R_buffer,
            oft_backend=self.oft_backend,
        )

        if bl.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = bl.bias if bl.skip_bias_add else None
        return output, output_bias


class RowParallelLinearWithOFT(BaseLayerWithOFT):
    def __init__(
        self,
        base_layer: RowParallelLinear,
        oft_backend: BaseOFTBackend,
    ) -> None:
        super().__init__(base_layer, oft_backend)

    def set_oft_info(self, R_buffer: torch.Tensor):
        self.set_oft = True
        self.R_buffer = R_buffer

    def apply_oft(self, x: torch.Tensor) -> torch.Tensor:
        # OFT input rotation: rotated_x = x @ R
        # The rotated input is then fed into the base linear layer.
        # Unlike LoRA which is a parallel additive branch.
        rotated_x = self.oft_backend.run_oft_r_sgemm(
            x=x,
            weights=self.R_buffer,
        )
        return rotated_x

    def forward(self, input_: torch.Tensor, skip_all_reduce=False):
        # duplicate the logic in RowParallelLinear
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()

        # OFT: rotate input FIRST, then apply base forward
        if self.set_oft:
            input_parallel = self.apply_oft(input_parallel)

        if self.set_oft and self._oft_parity_mode:
            from sglang.srt.oft.parity_dequant import parity_linear

            output_parallel = parity_linear(
                self.base_layer, input_parallel, apply_bias=False
            )
        else:
            output_parallel = self.base_layer.quant_method.apply(
                self.base_layer, input_parallel
            )

        if (
            self.base_layer.reduce_results
            and self.base_layer.tp_size > 1
            and not skip_all_reduce
        ):
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    def slice_oft_r_weights(self, R: torch.Tensor, tp_rank: int):
        # For RowParallel with input rotation, the input is split across TP ranks.
        # R operates on the input dimension, so it must be sliced for each rank's partition.
        # R is block-diagonal: shape (num_blocks, block_size, block_size) precomputed.
        # Each block covers block_size input features; for TP, we take the blocks
        # corresponding to this rank's input partition.
        shard_size = self.base_layer.input_size_per_partition
        input_size = self.base_layer.input_size
        if shard_size == input_size:
            # TP=1, no slicing needed
            return R
        num_blocks = R.shape[0]
        blocks_per_shard = num_blocks * shard_size // input_size
        start_block = tp_rank * blocks_per_shard
        end_block = start_block + blocks_per_shard
        return R[start_block:end_block, :]


class DeepSeekV4LinearWithOFT(BaseLayerWithOFT):
    """OFT wrapper for ``DeepSeekV4Linear`` (and its TP-aware subclasses).

    DeepSeekV4Linear is the V4 native-quant linear primitive — its forward
    dispatches through ``_quantized_linear`` keyed on ``weight.dtype`` to
    handle FP4 / FP8 / bf16 / fp32 storage natively, and returns a single
    ``Tensor`` (not the ``(out, bias)`` tuple stock SGLang ColumnParallel
    uses for TE compatibility).

    The OFT wrap therefore:
    * Rotates the input via the standard ``oft_backend.run_oft_r_sgemm``
      path, exactly like ``ColumnParallelLinearWithOFT``.
    * Calls ``base_layer.forward`` directly so the V4 quant dispatch and
      the bf16/fp32 ``_quantized_linear`` paths stay in charge.
    * Mirrors the base layer's plain-``Tensor`` return contract — so call
      sites like ``self.q_norm(self.wq_a(x))`` keep working.
    * Forwards ``.scale`` access to the base layer (in addition to the
      ``.weight`` proxy already installed by ``BaseLayerWithOFT.__init__``)
      so the V4 attention forward's direct-weight reshape at
      ``self.wo_a.weight.view(...)`` keeps resolving correctly. The
      ``apply_input_rotation`` hook below handles the ``wo_a`` path that
      bypasses ``forward``.
    """

    def __init__(self, base_layer: nn.Module, oft_backend: BaseOFTBackend) -> None:
        super().__init__(base_layer, oft_backend)
        # Forward `.scale` (and `.bias` when present) so the base layer's
        # native attribute surface stays intact under wrapping.
        if hasattr(base_layer, "scale"):
            self.scale = base_layer.scale
        if hasattr(base_layer, "bias") and base_layer.bias is not None:
            self.bias = base_layer.bias

    def set_oft_info(self, R_buffer: torch.Tensor):
        self.set_oft = True
        self.R_buffer = R_buffer

    def apply_oft(self, x: torch.Tensor) -> torch.Tensor:
        # ``run_oft_r_sgemm`` requires 2-D input ``(total_seq_len, dim)``;
        # V4 attention threads tensors in Megatron-LM ``[s, b, d]`` (or
        # arbitrarily nested) shapes, so flatten leading axes for the
        # rotation and restore them on the way out.
        orig_shape = x.shape
        if x.dim() != 2:
            x = x.reshape(-1, orig_shape[-1])
        rotated = self.oft_backend.run_oft_r_sgemm(x=x, weights=self.R_buffer)
        if rotated.shape != orig_shape:
            rotated = rotated.reshape(orig_shape)
        return rotated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.set_oft:
            x = self.apply_oft(x)
        return self.base_layer.forward(x)

    def apply_input_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply only the OFT input rotation, returning the rotated tensor
        *without* calling the base linear.

        Used by V4 attention's einsum-bypass code path on ``wo_a``: the
        attention forward reaches into ``self.wo_a.weight.view(...)``
        directly, bypassing ``forward``. Without this hook every non-
        identity wo_a adapter is silently dropped at serving time.

        wo_a is a *grouped* linear: ``base_layer.in_features`` reports
        the per-group dim (e.g. 4096 = head_local_dim_per_group on the
        V4 debug ckpt), and ``wo_a.weight.view(n_groups, o_lora_rank,
        -1)`` contracts only that per-group axis. The flattened
        attention output, however, has last-dim
        ``n_groups * head_local_dim_per_group`` (e.g. 32768). Apply the
        rotation *per-group*: reshape to ``(*, n_groups, in_features)``,
        run the OFT 2-D rotation (the kernel handles arbitrary leading
        dims via ``reshape(-1, in_features)``), then reshape back. This
        keeps the OFT R buffer at its natural per-group size and uses
        the same block-diagonal R for every group, mirroring Megatron-
        Bridge's DeepSeek V4 OFT input-rotation path.
        """
        if not self.set_oft:
            return x
        in_dim = self.base_layer.in_features
        last = x.shape[-1]
        if last == in_dim:
            return self.apply_oft(x)
        if last % in_dim != 0:
            raise ValueError(
                f"DeepSeekV4LinearWithOFT.apply_input_rotation: last-dim {last} "
                f"is not a multiple of base in_features {in_dim}"
            )
        # Per-group rotation through the selected OFT backend. The backend
        # scales token-segment metadata by ``n_groups`` before launching its
        # regular segmented kernel, so the grouped rows use the same adapter
        # slots as the logical tokens.
        n_groups = last // in_dim
        x_grouped = x.reshape(*x.shape[:-1], n_groups, in_dim).reshape(-1, in_dim)
        x_rot = self.oft_backend.run_grouped_oft_r_sgemm(
            x_grouped,
            self.R_buffer,
            n_groups=n_groups,
        )
        return x_rot.reshape(*x.shape[:-1], n_groups, in_dim).reshape(
            *x.shape[:-1], last
        )

    def get_oft_input_dim(self) -> int:
        return self.base_layer.in_features

    def slice_oft_r_weights(self, R: torch.Tensor, tp_rank: int):
        # ColumnParallel-style: input is replicated across TP ranks for V4
        # attention sublayers, so R is not sliced. RowParallel slicing lives
        # on ``DeepSeekV4RowParallelLinearWithOFT``.
        return R


class DeepSeekV4ColumnParallelLinearWithOFT(DeepSeekV4LinearWithOFT):
    """OFT wrap for ``DeepSeekV4ColumnParallelLinear``.

    Same input-rotation semantics as :class:`DeepSeekV4LinearWithOFT` — the base
    layer already shards ``out_features`` across TP and returns the local
    shard as a plain ``Tensor``. Input is replicated across TP ranks, so R
    is not sliced.
    """

    pass


class DeepSeekV4RowParallelLinearWithOFT(DeepSeekV4LinearWithOFT):
    """OFT wrap for ``DeepSeekV4RowParallelLinear``.

    Row-parallel base layers shard ``in_features`` across TP, so the OFT
    rotation must operate on the per-rank slice of R (matching the input
    partition). For TP=1 the slice is the whole R; for TP>1 the wrapper
    selects the local rank's R slice.
    """

    def get_oft_input_dim(self) -> int:
        # ``DeepSeekV4RowParallelLinear.__init__`` already divides ``in_features``
        # by the TP world before calling super, so the base layer's
        # ``in_features`` is the per-rank slice.
        return self.base_layer.in_features

    def get_local_tp_rank(self) -> int:
        tp_group = getattr(self.base_layer, "tp_group", None)
        if tp_group is not None:
            rank = getattr(tp_group, "rank_in_group", None)
            if rank is not None:
                return int(rank)
            rank = getattr(tp_group, "rank", None)
            if rank is not None:
                return int(rank)
        return super().get_local_tp_rank()

    def slice_oft_r_weights(self, R: torch.Tensor, tp_rank: int):
        # R is shaped (num_blocks, block_size, block_size). For RowParallel
        # the input is partitioned across TP, so each rank takes the slice
        # of blocks corresponding to its input partition.
        tp_group = self.base_layer.tp_group
        if tp_group is None:
            tp_world = 1
        elif hasattr(tp_group, "world_size"):
            tp_world = int(tp_group.world_size)
        else:
            tp_world = int(tp_group.size())
        if tp_world <= 1:
            return R
        num_blocks = R.shape[0]
        blocks_per_rank = num_blocks // tp_world
        start = tp_rank * blocks_per_rank
        end = start + blocks_per_rank
        return R[start:end, :]


DSV4LinearWithOFT = DeepSeekV4LinearWithOFT
DSV4ColumnParallelLinearWithOFT = DeepSeekV4ColumnParallelLinearWithOFT
DSV4RowParallelLinearWithOFT = DeepSeekV4RowParallelLinearWithOFT


def get_oft_layer(
    layer: nn.Module, oft_backend: BaseOFTBackend
) -> BaseLayerWithOFT:
    supported_layer_types = {
        # the order matters
        ParallelLMHead: ParallelLMHeadWithOFT,
        VocabParallelEmbedding: VocabParallelEmbeddingWithOFT,
        QKVParallelLinear: QKVParallelLinearWithOFT,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithOFT,
        ColumnParallelLinear: ColumnParallelLinearWithOFT,
        RowParallelLinear: RowParallelLinearWithOFT,
    }
    # Lazy-register V4 native-quant linear types — we can't import them at
    # module load time without dragging the V4 model in (and creating a
    # cycle for builds that don't need V4). The check is on the fully
    # qualified type name so subclassing inside the V4 module file still
    # works without further updates here.
    src_cls_name = type(layer).__name__
    if src_cls_name in {
        "DeepSeekV4Linear",
        "DeepSeekV4ColumnParallelLinear",
        "DeepSeekV4RowParallelLinear",
        "DSV4Linear",
        "DSV4ColumnParallelLinear",
        "DSV4RowParallelLinear",
    }:
        from sglang.srt.models.deepseek_v4 import (
            DeepSeekV4ColumnParallelLinear,
            DeepSeekV4Linear,
            DeepSeekV4RowParallelLinear,
        )

        if isinstance(layer, DeepSeekV4RowParallelLinear):
            return DeepSeekV4RowParallelLinearWithOFT(layer, oft_backend)
        if isinstance(layer, DeepSeekV4ColumnParallelLinear):
            return DeepSeekV4ColumnParallelLinearWithOFT(layer, oft_backend)
        if isinstance(layer, DeepSeekV4Linear):
            return DeepSeekV4LinearWithOFT(layer, oft_backend)
    for src_layer_type, oft_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = oft_layer_type(layer, oft_backend)
            return ret
    raise Exception(f"No corresponding OFT layer supported for {type(layer)}.")
