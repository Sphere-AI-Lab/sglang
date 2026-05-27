# Copyright 2024-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""DeepSeek V4 attention backend.

DeepSeek V4 does not use SGLang's generic paged-KV payload in the attention
hot path. The scheduler still needs request-token bookkeeping, CUDA-graph
metadata, and per-request slots, but payload state lives in the model-owned
``Attention.kv_cache`` / ``Compressor`` / ``Indexer`` buffers.

This backend owns that scheduler-facing metadata and dispatches each layer to
``DeepSeekV4Attention.forward_stateful`` or
``DeepSeekV4Attention.forward_stateful_cuda_graph``. The only sparse-attention
math implementation is the official DeepSeek V4 TileLang kernel re-exported as
``deepseek_v4_kernels.sparse_attn``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True)
class DeepSeekV4AttentionMetadata:
    """Per-forward metadata for the V4 stateful attention path.

    All tensors are on the same device the backend was constructed on.
    The struct is frozen so the backend can pass it through without
    accidentally mutating shared state across layers.

    Fields mirror the relevant subset of ``NSAMetadata`` (see
    ``nsa_backend.py:NSAMetadata``) — not the full surface, because V4
    owns its indexer and KV payload state inside the model.
    """

    # Number of requests in this forward batch.
    batch_size: int

    # Cumulative query lengths, int32, len = batch_size + 1.
    # cu_seqlens_q[i:i+1] gives the [start, end) of request i in the
    # flattened-batch q tensor (FA2 convention).
    cu_seqlens_q: torch.Tensor

    # Cumulative key lengths (same structure as cu_seqlens_q).
    cu_seqlens_k: torch.Tensor

    # Page table at page_size=1 (one token per page, fine-grained sparse
    # indexing — same as V2/V3 NSA's ``page_table_1``).
    # Shape: ``[batch_size, max_seqlen_k]``, int32.
    page_table_1: torch.Tensor

    # Per-request key-cache seqlens, int32 [batch_size]. Equivalent to
    # ``cu_seqlens_k[1:] - cu_seqlens_k[:-1]`` but stored once for fast
    # access on the hot path.
    cache_seqlens_int32: torch.Tensor

    max_seqlen_q: int
    max_seqlen_k: int

    # Compatibility flag for older metadata tests. The production path keeps
    # compressed history in model-owned state, not in the SGLang KV pool.
    compressor_recompute: bool = True

    # True iff `forward_batch.forward_mode.is_decode()`. Routes the
    # backend's `forward_extend` / `forward_decode` choice but also
    # determines `start_pos`'s arithmetic.
    is_decode: bool = False

    # Per-request `start_pos` for the V4 stateful path. For decode this
    # is `seq_lens[i] - 1` (position of the new token being generated).
    # For fresh extend it's 0. For prefix-hit extend it is the prefix
    # length, so the suffix computes at the correct absolute position
    # after the scheduler restores model-owned prefix state.
    start_pos: Optional[torch.Tensor] = None  # [batch_size] int64

    # Per-request slot index into the V4-owned per-request buffers
    # (`Attention.kv_cache`, `Compressor.kv_state`, `Indexer.kv_cache`).
    # Equals `forward_batch.req_pool_indices`. Each layer indexes its
    # state with `req_indices[i]` for the i-th request in the batch.
    req_indices: Optional[torch.Tensor] = None  # [batch_size] int64

    # CPU mirrors for Python dispatch paths. They are populated once while
    # building metadata so per-layer forward dispatch does not call `.item()`
    # and serialize the CUDA stream repeatedly.
    cu_seqlens_q_cpu: Optional[list[int]] = None
    start_pos_cpu: Optional[list[int]] = None
    req_indices_cpu: Optional[list[int]] = None
    active_mask_cpu: Optional[list[bool]] = None
    has_prefix_hit: bool = False

    # CUDA graph replay metadata. `active_mask` is false for graph-padding
    # rows so the DeepSeek V4 stateful path can leave those model-owned cache slots
    # unchanged while still replaying a fixed batch-size graph.
    is_cuda_graph: bool = False
    active_mask: Optional[torch.Tensor] = None  # [batch_size] bool


class DeepSeekV4AttentionBackend(AttentionBackend):
    """Scheduler-facing attention backend for DeepSeek V4.

    The backend is intentionally thin:
      * ``init_forward_metadata`` extracts request slots, lengths, page tables,
        and CUDA-graph metadata from ``ForwardBatch``.
      * ``forward_extend`` and ``forward_decode`` dispatch into the model-owned
        stateful attention path.
      * ``get_indexer_metadata`` returns ``None`` because V4 has its own
        indexer module inside the layer (``_DeepSeekV4Indexer``); we do
        NOT route through SGLang's NSA indexer interface.

    The actual attention math ALWAYS goes through
    ``deepseek_v4_kernels.sparse_attn``.
    """

    def __init__(
        self,
        model_runner=None,
        kv_cache_pool=None,
        *,
        model_ref=None,
        max_batch_size: Optional[int] = None,
    ):
        # ``model_runner`` is the standard SGLang plumbing handle. For
        # CPU-only unit tests it's None and the backend just carries metadata;
        # for integration paths it provides scheduler pools and OFT state.
        #
        # ``kv_cache_pool`` is retained for device discovery and older tests.
        # DeepSeek V4 payload KV is model-owned and is not read from this pool on the
        # attention hot path.
        #
        # ``model_ref`` is retained for older test harnesses. Prefix-state
        # reset/restore is handled by the scheduler before forward.
        # ``max_batch_size`` is the server-side cap for V4-owned per-request
        # buffers; ``init_forward_metadata`` rejects larger batches.
        super().__init__()
        self.model_runner = model_runner
        self.kv_cache_pool = kv_cache_pool
        if kv_cache_pool is None and model_runner is not None:
            self.kv_cache_pool = getattr(model_runner, "token_to_kv_pool", None)
        self._cached_metadata: Optional[DeepSeekV4AttentionMetadata] = None
        self._single_oft_batch_infos = None
        self._single_token_oft_batch_infos = None
        self._cached_forward_batch_for_oft = None
        self._model_ref = model_ref
        self._max_batch_size = max_batch_size
        self._cuda_graph_max_context_len = 0
        self._cuda_graph_cache_seqlens = None
        self._cuda_graph_cu_seqlens_q = None
        self._cuda_graph_cu_seqlens_k = None
        self._cuda_graph_page_table_1 = None
        self._cuda_graph_start_pos = None
        self._cuda_graph_req_indices = None
        self._cuda_graph_active_mask = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def init_forward_metadata(
        self, forward_batch: "ForwardBatch"
    ) -> DeepSeekV4AttentionMetadata:
        """Build the per-forward metadata struct.

        Reads ``forward_batch.seq_lens`` (the per-request K-cache lengths) and
        ``forward_batch.extend_seq_lens`` (the per-request Q lengths for
        prefill). If ``extend_seq_lens`` is absent, prefill uses ``seq_lens``
        and decode uses one query token per active request.
        """
        seq_lens = forward_batch.seq_lens
        if seq_lens.dim() != 1:
            raise ValueError(
                f"seq_lens must be 1-D, got {seq_lens.shape}"
            )

        bs = int(seq_lens.shape[0])
        device = seq_lens.device

        # Detect decode vs fresh extend via ``forward_batch.forward_mode``.
        # Hand-crafted SimpleNamespace tests without ``forward_mode`` default
        # to extend with start_pos=0.
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_decode = bool(forward_mode.is_decode()) if forward_mode is not None else False

        # Q lengths. For full prefill (no chunked-prefill), Q lengths
        # equal K lengths. For decode, SGLang omits extend_seq_lens; each
        # request contributes exactly one query token.
        q_lens = getattr(forward_batch, "extend_seq_lens", None)
        if q_lens is None:
            q_lens = torch.ones_like(seq_lens) if is_decode else seq_lens
        if not isinstance(q_lens, torch.Tensor):
            q_lens = torch.as_tensor(q_lens, device=device, dtype=torch.int32)
        q_lens = q_lens.to(device=device, dtype=torch.int32)
        k_lens = seq_lens.to(device=device, dtype=torch.int32)

        cu_seqlens_q = torch.zeros(bs + 1, device=device, dtype=torch.int32)
        cu_seqlens_q[1:] = torch.cumsum(q_lens, dim=0)
        cu_seqlens_k = torch.zeros(bs + 1, device=device, dtype=torch.int32)
        cu_seqlens_k[1:] = torch.cumsum(k_lens, dim=0)

        max_seqlen_q = int(q_lens.max().item()) if bs > 0 else 0
        max_seqlen_k = int(k_lens.max().item()) if bs > 0 else 0

        # Page table at page_size=1. Source it from the request-token
        # pool (``req_to_token_pool``) when the model_runner is wired in.
        # Unit tests without a model_runner can provide a pre-built
        # ``forward_batch.page_table_1`` field as a stand-in.
        page_table_1 = getattr(forward_batch, "page_table_1", None)
        if page_table_1 is None:
            req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
            if req_to_token_pool is None and self.model_runner is not None:
                req_to_token_pool = self.model_runner.req_to_token_pool
            if req_to_token_pool is None:
                raise RuntimeError(
                    "DeepSeek V4 backend cannot build page_table_1: forward_batch "
                    "exposes neither ``page_table_1`` nor "
                    "``req_to_token_pool``."
                )
            req_pool_indices = forward_batch.req_pool_indices
            page_table_1 = req_to_token_pool.req_to_token[
                req_pool_indices, :max_seqlen_k
            ].to(dtype=torch.int32)

        if page_table_1.shape != (bs, max_seqlen_k):
            raise ValueError(
                f"page_table_1 shape mismatch: expected ({bs}, {max_seqlen_k}), "
                f"got {tuple(page_table_1.shape)}"
            )

        seq_lens_i64 = seq_lens.to(dtype=torch.int64)
        active_mask = None
        if is_decode:
            # SGLang may pad decode batches to an attention/MLP parallel
            # granularity even when CUDA graph is disabled. Padding rows have
            # seq_len=0; keep their position in-bounds and let forward_decode
            # skip the row without mutating DeepSeek V4 model-owned state.
            active_mask = seq_lens_i64 > 0
            start_pos = torch.clamp(seq_lens_i64, min=1) - 1
        else:
            extend_prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)
            if extend_prefix_lens is None:
                start_pos = torch.zeros_like(seq_lens_i64)
            else:
                if not isinstance(extend_prefix_lens, torch.Tensor):
                    extend_prefix_lens = torch.as_tensor(
                        extend_prefix_lens, device=device, dtype=torch.int64
                    )
                start_pos = extend_prefix_lens.to(device=device, dtype=torch.int64)

        # Server-side batch cap. The V4-owned per-request buffers were
        # sized at model construction; reject silently-truncating batches.
        if self._max_batch_size is not None and bs > self._max_batch_size:
            raise RuntimeError(
                f"DeepSeek V4 backend received batch_size={bs} > "
                f"max_batch_size={self._max_batch_size}. Bump "
                f"`config.max_batch_size_override` before constructing "
                f"the model."
            )

        # Per-request slot indices into the V4-owned buffers.
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if req_pool_indices is None:
            # Harness path: no req_pool_indices set up. Default to contiguous
            # slots [0..bs).
            req_indices = torch.arange(bs, device=device, dtype=torch.int64)
        else:
            req_indices = req_pool_indices.to(device=device, dtype=torch.int64)

        if not is_decode and self._model_ref is not None:
            reset_fn = getattr(self._model_ref, "reset_inference_state", None)
            if reset_fn is not None:
                start_pos_cpu = start_pos.detach().cpu()
                req_indices_cpu = req_indices.detach().cpu()
                for b in range(bs):
                    if int(start_pos_cpu[b].item()) == 0:
                        reset_fn(int(req_indices_cpu[b].item()))

        cu_seqlens_q_cpu = [int(x) for x in cu_seqlens_q.detach().cpu().tolist()]
        start_pos_cpu = [int(x) for x in start_pos.detach().cpu().tolist()]
        req_indices_cpu = [int(x) for x in req_indices.detach().cpu().tolist()]
        active_mask_cpu = (
            [bool(x) for x in active_mask.detach().cpu().tolist()]
            if active_mask is not None
            else None
        )
        has_prefix_hit = (not is_decode) and any(pos > 0 for pos in start_pos_cpu)

        meta = DeepSeekV4AttentionMetadata(
            batch_size=bs,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            cache_seqlens_int32=k_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            compressor_recompute=True,
            is_decode=is_decode,
            start_pos=start_pos,
            req_indices=req_indices,
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            start_pos_cpu=start_pos_cpu,
            req_indices_cpu=req_indices_cpu,
            active_mask_cpu=active_mask_cpu,
            has_prefix_hit=has_prefix_hit,
            active_mask=active_mask,
        )
        self._single_oft_batch_infos = self._prepare_single_request_oft_batch_infos(
            forward_batch, q_lens
        )
        self._single_token_oft_batch_infos = None
        self._cached_forward_batch_for_oft = forward_batch
        self._cached_metadata = meta
        return meta

    def _device(self) -> torch.device:
        if self.model_runner is not None:
            device = getattr(self.model_runner, "device", None)
            if device is not None:
                return torch.device(device)
        if self.kv_cache_pool is not None:
            device = getattr(self.kv_cache_pool, "device", None)
            if device is not None:
                return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _max_context_len(self, fallback: int) -> int:
        if self.model_runner is not None:
            req_to_token_pool = getattr(self.model_runner, "req_to_token_pool", None)
            max_context_len = getattr(req_to_token_pool, "max_context_len", None)
            if max_context_len is not None:
                return int(max_context_len)
        model_ref = self._model_ref
        max_seq_len = getattr(model_ref, "max_seq_len", None)
        if max_seq_len is not None:
            return int(max_seq_len)
        return int(fallback)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize DeepSeek V4 decode CUDA graph metadata buffers.

        DeepSeek V4 owns stateful per-request attention/compressor buffers outside
        SGLang's normal KV pool. CUDA graph replay therefore needs stable
        tensor addresses for the request slots and absolute decode positions;
        the captured model graph must read these buffers after replay copies
        runtime metadata into them.
        """
        del max_num_tokens  # Decode-only graph: one token per batch row.
        device = self._device()
        max_context_len = self._max_context_len(max_bs)
        self._cuda_graph_max_context_len = max_context_len
        self._cuda_graph_cache_seqlens = torch.ones(
            max_bs, dtype=torch.int32, device=device
        )
        self._cuda_graph_cu_seqlens_q = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=device
        )
        self._cuda_graph_cu_seqlens_k = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=device
        )
        self._cuda_graph_page_table_1 = torch.zeros(
            max_bs, max_context_len, dtype=torch.int32, device=device
        )
        self._cuda_graph_start_pos = torch.zeros(
            max_bs, dtype=torch.int64, device=device
        )
        self._cuda_graph_req_indices = torch.zeros(
            max_bs, dtype=torch.int64, device=device
        )
        self._cuda_graph_active_mask = torch.zeros(
            max_bs, dtype=torch.bool, device=device
        )

    def _require_cuda_graph_state(self):
        if self._cuda_graph_cache_seqlens is None:
            raise RuntimeError(
                "DeepSeek V4 CUDA graph metadata is not initialized; call "
                "init_cuda_graph_state before capture or replay."
            )

    def _set_cuda_graph_metadata(
        self,
        *,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> DeepSeekV4AttentionMetadata:
        self._require_cuda_graph_state()
        device = self._cuda_graph_cache_seqlens.device
        seq_lens_i32 = seq_lens[:bs].to(device=device, dtype=torch.int32)
        req_indices_i64 = req_pool_indices[:bs].to(device=device, dtype=torch.int64)
        active = active_mask[:bs].to(device=device, dtype=torch.bool)
        if bs > 0:
            slot_cap = int(self._max_batch_size or self._cuda_graph_req_indices.shape[0])
            candidate_slots = torch.arange(slot_cap, device=device, dtype=torch.int64)
            active_req = torch.where(
                active,
                req_indices_i64,
                torch.full_like(req_indices_i64, -1),
            )
            slot_available = ~torch.any(
                candidate_slots[:, None] == active_req[None, :],
                dim=1,
            )
            scratch_idx = torch.argmax(slot_available.to(torch.int64))
            scratch = scratch_idx.reshape(1).expand_as(req_indices_i64)
            req_indices_i64 = torch.where(active, req_indices_i64, scratch)

        cache_seqlens = self._cuda_graph_cache_seqlens[:bs]
        cache_seqlens.copy_(torch.clamp(seq_lens_i32, min=1))

        cu_seqlens_k = self._cuda_graph_cu_seqlens_k[: bs + 1]
        cu_seqlens_k[0].zero_()
        cu_seqlens_k[1:].copy_(torch.cumsum(cache_seqlens, dim=0))

        start_pos = self._cuda_graph_start_pos[:bs]
        start_pos.copy_(torch.clamp(seq_lens_i32.to(torch.int64), min=1) - 1)

        req_indices = self._cuda_graph_req_indices[:bs]
        req_indices.copy_(req_indices_i64)

        active_buf = self._cuda_graph_active_mask[:bs]
        active_buf.copy_(active)

        page_table_1 = self._cuda_graph_page_table_1[:bs]
        if self.model_runner is not None:
            req_to_token_pool = getattr(self.model_runner, "req_to_token_pool", None)
            req_to_token = getattr(req_to_token_pool, "req_to_token", None)
            if req_to_token is not None:
                page_table_1[:, : self._cuda_graph_max_context_len].copy_(
                    req_to_token[
                        req_indices,
                        : self._cuda_graph_max_context_len,
                    ].to(dtype=torch.int32)
                )

        meta = DeepSeekV4AttentionMetadata(
            batch_size=bs,
            cu_seqlens_q=self._cuda_graph_cu_seqlens_q[: bs + 1],
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            cache_seqlens_int32=cache_seqlens,
            max_seqlen_q=1,
            max_seqlen_k=self._cuda_graph_max_context_len,
            compressor_recompute=True,
            is_decode=True,
            start_pos=start_pos,
            req_indices=req_indices,
            cu_seqlens_q_cpu=None,
            start_pos_cpu=None,
            req_indices_cpu=None,
            active_mask_cpu=None,
            has_prefix_hit=False,
            is_cuda_graph=True,
            active_mask=active_buf,
        )
        self._cached_metadata = meta
        return meta

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode,
        spec_info,
    ):
        del num_tokens, encoder_lens, spec_info
        if not forward_mode.is_decode_or_idle():
            raise ValueError(
                "DeepSeek V4 CUDA graph currently supports decode capture only."
            )
        # Capture with inactive rows to avoid mutating DeepSeek V4 model-owned
        # state during graph warmup/capture. Replay copies a real active
        # mask into the same tensor storage before graph execution.
        active_mask = torch.zeros(bs, dtype=torch.bool, device=seq_lens.device)
        return self._set_cuda_graph_metadata(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=torch.clamp(seq_lens, min=1),
            active_mask=active_mask,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        del seq_lens_sum, encoder_lens, spec_info, seq_lens_cpu
        if not forward_mode.is_decode_or_idle():
            raise ValueError(
                "DeepSeek V4 CUDA graph currently supports decode replay only."
            )
        # Graph padding rows are filled with seq_len == 1. A real decode row
        # has already consumed at least one prompt token and the current token,
        # so seq_len > 1 marks rows that may update DeepSeek V4 state.
        active_mask = seq_lens[:bs] > 1
        return self._set_cuda_graph_metadata(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            active_mask=active_mask,
        )

    @property
    def metadata(self) -> DeepSeekV4AttentionMetadata:
        if self._cached_metadata is None:
            raise RuntimeError(
                "DeepSeek V4 backend metadata not yet built; "
                "call init_forward_metadata first."
            )
        return self._cached_metadata

    def _prepare_single_request_oft_batch_infos(
        self,
        forward_batch: "ForwardBatch",
        q_lens: torch.Tensor,
    ):
        oft_ids = getattr(forward_batch, "oft_ids", None)
        if not oft_ids:
            return None
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is not None and forward_mode.is_idle():
            return None
        if self.model_runner is None:
            return None
        oft_manager = getattr(self.model_runner, "oft_manager", None)
        if oft_manager is None:
            return None
        oft_backend = getattr(oft_manager, "oft_backend", None)
        if oft_backend is None:
            return None
        if getattr(oft_backend, "_use_single_adapter_fast_path", False):
            return None

        old_batch_info = getattr(oft_backend, "batch_info", None)
        device = getattr(forward_batch.seq_lens, "device", None)
        if device is None:
            device = torch.device("cuda")
        single_infos = []
        try:
            for batch_idx, uid in enumerate(oft_ids):
                token_len = int(q_lens[batch_idx].item())
                single_forward_batch = SimpleNamespace(
                    batch_size=1,
                    forward_mode=forward_batch.forward_mode,
                    oft_ids=[uid],
                    extend_seq_lens=torch.tensor(
                        [token_len], dtype=torch.int32, device=device
                    ),
                    extend_seq_lens_cpu=[token_len],
                    spec_info=getattr(forward_batch, "spec_info", None),
                )
                oft_manager.prepare_oft_batch(single_forward_batch)
                single_infos.append(oft_backend.batch_info)
        finally:
            oft_backend.batch_info = old_batch_info
        return single_infos

    def _call_with_single_request_oft_batch(
        self,
        batch_idx: int,
        fn,
        token_len: Optional[int] = None,
    ):
        """Run a sliced DeepSeek V4 stateful call with matching OFT batch metadata.

        DeepSeek V4's stateful path splits SGLang's flat multi-request forward into
        one request at a time so each request can advance its own compressor
        state. The OFT backend batch_info is normally built for the original
        global flat batch; using that metadata while rotating a per-request
        slice makes the OFT kernel read past the slice. Temporarily narrow the
        OFT batch_info to the current request and restore it after the layer
        call.
        """
        single_infos = self._single_oft_batch_infos
        if not single_infos:
            return fn()
        if self.model_runner is None:
            return fn()
        oft_manager = getattr(self.model_runner, "oft_manager", None)
        if oft_manager is None:
            return fn()
        oft_backend = getattr(oft_manager, "oft_backend", None)
        if oft_backend is None:
            return fn()

        old_batch_info = getattr(oft_backend, "batch_info", None)
        batch_infos = single_infos
        if token_len == 1 and getattr(single_infos[batch_idx], "max_len", None) != 1:
            if self._single_token_oft_batch_infos is None:
                forward_batch = self._cached_forward_batch_for_oft
                if forward_batch is None:
                    return fn()
                one_lens = torch.ones(
                    len(single_infos),
                    dtype=torch.int32,
                    device=self.metadata.cu_seqlens_q.device,
                )
                self._single_token_oft_batch_infos = (
                    self._prepare_single_request_oft_batch_infos(
                        forward_batch, one_lens
                    )
                )
            batch_infos = self._single_token_oft_batch_infos

        oft_backend.batch_info = batch_infos[batch_idx]
        try:
            return fn()
        finally:
            oft_backend.batch_info = old_batch_info

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward_extend(
        self,
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer,
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """V4 prefill via the model-owned stateful path.

        Dispatches to ``layer.forward_stateful(hidden_states, start_pos,
        req_idx)`` — the V4-owned per-layer ``Attention.kv_cache`` and
        Compressor / Indexer state buffers carry the per-request KV; the
        SGLang ``MHATokenToKVPool`` is allocated by the ModelRunner but
        not read on the hot path.

        ``q`` / ``k`` / ``v`` are accepted for the ``AttentionBackend`` ABC
        but ignored: V4 builds its own q / kv inside ``forward_stateful``
        from ``hidden_states``.
        """
        del q, k, v, save_kv_cache, kwargs  # Built inside forward_stateful.

        meta = self.metadata
        if hidden_states is None:
            raise RuntimeError(
                "DeepSeek V4 backend forward_extend requires the "
                "`hidden_states` kwarg from `DeepSeekV4Attention.forward`. "
                "Callers passing only q/k/v are no longer supported."
            )

        bsz = meta.batch_size
        start_pos_cpu = meta.start_pos_cpu
        req_indices_cpu = meta.req_indices_cpu
        cu_seqlens_q_cpu = meta.cu_seqlens_q_cpu
        if start_pos_cpu is None or req_indices_cpu is None or cu_seqlens_q_cpu is None:
            raise RuntimeError(
                "DeepSeek V4 forward_extend requires CPU metadata mirrors built by "
                "init_forward_metadata."
            )
        if bsz == 1:
            return layer.forward_stateful(
                hidden_states,
                start_pos=start_pos_cpu[0],
                req_idx=req_indices_cpu[0],
            )

        # Multi-request prefill: per-request loop. SGLang's scheduler
        # hands extend tokens to the model as a flat stream
        # `[total_tokens, 1, d]`; use `cu_seqlens_q` to recover request
        # boundaries. Older harnesses may still pass rectangular
        # `[s, b, d]`, so keep the batch-dim slicing path for that shape.
        # Per-request calls preserve byte-identity because each call has
        # bsz=1; the kernel batch-precision floor does not apply.
        outputs: list[torch.Tensor] = []
        if hidden_states.shape[1] == 1:
            total_q = cu_seqlens_q_cpu[-1]
            if hidden_states.shape[0] < total_q:
                raise RuntimeError(
                    "DeepSeek V4 flat prefill hidden_states length does not match "
                    f"cu_seqlens_q total: {hidden_states.shape[0]} vs {total_q}."
                )
            padded_tail = hidden_states[total_q:]
            hidden_states = hidden_states[:total_q]
            for b in range(bsz):
                start = cu_seqlens_q_cpu[b]
                end = cu_seqlens_q_cpu[b + 1]
                hs_b = hidden_states[start:end, :, :].contiguous()
                out_b = self._call_with_single_request_oft_batch(
                    b,
                    lambda hs_b=hs_b, b=b: layer.forward_stateful(
                        hs_b,
                        start_pos=start_pos_cpu[b],
                        req_idx=req_indices_cpu[b],
                    ),
                )
                outputs.append(out_b)
            out = torch.cat(outputs, dim=0)
            if padded_tail.shape[0] > 0:
                out = torch.cat([out, torch.zeros_like(padded_tail)], dim=0)
            return out

        if hidden_states.shape[1] != bsz:
            raise RuntimeError(
                "DeepSeek V4 multi-request prefill expected hidden_states shaped "
                f"[total_tokens, 1, d] or [s, {bsz}, d], got "
                f"{tuple(hidden_states.shape)}."
            )

        for b in range(bsz):
            hs_b = hidden_states[:, b : b + 1, :].contiguous()
            out_b = self._call_with_single_request_oft_batch(
                b,
                lambda hs_b=hs_b, b=b: layer.forward_stateful(
                    hs_b,
                    start_pos=start_pos_cpu[b],
                    req_idx=req_indices_cpu[b],
                ),
            )
            outputs.append(out_b)
        return torch.cat(outputs, dim=1)

    def forward_decode(
        self,
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer,
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """V4 single-token decode via the model-owned stateful path.

        Mirrors ``forward_extend``: ignores ``q`` / ``k`` / ``v`` (V4 builds
        its own q / kv inside ``forward_stateful`` from ``hidden_states``)
        and dispatches to ``layer.forward_stateful(hidden_states, start_pos,
        req_idx)`` with ``start_pos > 0`` (decode contract). The
        ``Compressor`` / ``Indexer`` / ``Attention`` per-request KV slots
        are advanced by the call.
        """
        del q, k, v, save_kv_cache, kwargs  # Built inside forward_stateful.

        meta = self.metadata
        if hidden_states is None:
            raise RuntimeError(
                "DeepSeek V4 backend forward_decode requires the "
                "`hidden_states` kwarg from `DeepSeekV4Attention.forward`."
            )
        if not meta.is_decode:
            raise RuntimeError(
                "DeepSeek V4 backend forward_decode invoked on extend metadata."
            )
        if meta.is_cuda_graph:
            if not hasattr(layer, "forward_stateful_cuda_graph"):
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph decode requires "
                    "`forward_stateful_cuda_graph` on the attention layer."
                )
            return layer.forward_stateful_cuda_graph(
                hidden_states,
                start_pos=meta.start_pos,
                req_indices=meta.req_indices,
                active_mask=meta.active_mask,
            )

        bsz = meta.batch_size
        start_pos_cpu = meta.start_pos_cpu
        req_indices_cpu = meta.req_indices_cpu
        active_mask_cpu = meta.active_mask_cpu
        if start_pos_cpu is None or req_indices_cpu is None:
            raise RuntimeError(
                "DeepSeek V4 forward_decode requires CPU metadata mirrors built by "
                "init_forward_metadata when CUDA graph replay is not active."
            )
        if bsz == 1:
            if active_mask_cpu is not None and not active_mask_cpu[0]:
                return torch.zeros_like(hidden_states)
            return layer.forward_stateful(
                hidden_states,
                start_pos=start_pos_cpu[0],
                req_idx=req_indices_cpu[0],
            )

        # Multi-request decode: per-request loop. `hidden_states` is
        # `[1, b, d]` (decode = single token per request). Per-request
        # bsz=1 calls preserve byte-identity (kernel batch-precision floor
        # doesn't apply).
        outputs: list[torch.Tensor] = []
        for b in range(bsz):
            hs_b = hidden_states[:, b : b + 1, :].contiguous()
            if active_mask_cpu is not None and not active_mask_cpu[b]:
                outputs.append(torch.zeros_like(hs_b))
                continue
            out_b = self._call_with_single_request_oft_batch(
                b,
                lambda hs_b=hs_b, b=b: layer.forward_stateful(
                    hs_b,
                    start_pos=start_pos_cpu[b],
                    req_idx=req_indices_cpu[b],
                ),
            )
            outputs.append(out_b)
        return torch.cat(outputs, dim=1)

    def get_indexer_metadata(self, layer_id: int, forward_batch: "ForwardBatch"):
        """V4 has its own indexer inside the layer; SGLang's NSA indexer
        path is unused. Returning None tells SGLang's plumbing to skip
        the indexer hand-off."""
        return None

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def support_triton(self) -> bool:
        # The official sparse_attn kernel is TileLang. Return true so the
        # CUDA graph / metadata plumbing uses the GPU-capable code path.
        return True
