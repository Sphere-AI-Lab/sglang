import os
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

_DSV4_FP4_GEMM_BACKEND_ENV = "DSV4_FP4_GEMM_BACKEND"
_VALID_DSV4_FP4_GEMM_BACKENDS = ("auto", "deepgemm", "tilelang")
_DSV4_FP4_GEMM_BACKEND = (
    os.environ.get(_DSV4_FP4_GEMM_BACKEND_ENV, "auto").strip().lower() or "auto"
)
if _DSV4_FP4_GEMM_BACKEND not in _VALID_DSV4_FP4_GEMM_BACKENDS:
    raise ValueError(
        f"{_DSV4_FP4_GEMM_BACKEND_ENV} must be one of "
        f"{', '.join(_VALID_DSV4_FP4_GEMM_BACKENDS)}, got "
        f"{_DSV4_FP4_GEMM_BACKEND!r}"
    )

_deep_gemm_official_import_error = None
if _DSV4_FP4_GEMM_BACKEND == "tilelang":
    _deep_gemm_official = None
else:
    try:
        import deep_gemm_official as _deep_gemm_official
    except Exception as exc:
        _deep_gemm_official_import_error = exc
        _deep_gemm_official = None


tilelang.set_log_level("WARNING")

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

_FP8 = "float8_e4m3"
_FP4 = "float4_e2m1fn"
_FE8M0 = "float8_e8m0fnu"
_BF16 = "bfloat16"
_FP32 = "float32"
_INT32 = "int32"

_DEEP_GEMM_OFFICIAL_GROUPED_FP8_FP4_GEMM = (
    getattr(_deep_gemm_official, "m_grouped_fp8_fp4_gemm_nt_contiguous", None)
    if _deep_gemm_official is not None
    else None
)
if (
    _DSV4_FP4_GEMM_BACKEND == "deepgemm"
    and _DEEP_GEMM_OFFICIAL_GROUPED_FP8_FP4_GEMM is None
):
    if _deep_gemm_official_import_error is not None:
        raise RuntimeError(
            f"{_DSV4_FP4_GEMM_BACKEND_ENV}=deepgemm requires importing "
            "deep_gemm_official"
        ) from _deep_gemm_official_import_error
    raise RuntimeError(
        f"{_DSV4_FP4_GEMM_BACKEND_ENV}=deepgemm requires "
        "deep_gemm_official.m_grouped_fp8_fp4_gemm_nt_contiguous"
    )

def has_deep_gemm_official_fp8_fp4() -> bool:
    return _DEEP_GEMM_OFFICIAL_GROUPED_FP8_FP4_GEMM is not None


def _align_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _empty_deep_gemm_scale_layout(
    rows: int,
    scale_cols: int,
    device: torch.device,
) -> torch.Tensor:
    aligned_rows = _align_to(rows, 4)
    aligned_scale_cols = _align_to(scale_cols, 4)
    return torch.empty(
        (aligned_scale_cols // 4, aligned_rows),
        device=device,
        dtype=torch.int32,
    ).mT[:rows, :]


@triton.jit
def _dsv4_deep_gemm_act_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    m: tl.constexpr,
    k: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yk: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sk: tl.constexpr,
    num_scale_cols: tl.constexpr,
    block_m: tl.constexpr,
    group_size: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_pack = tl.program_id(1)

    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, group_size)
    row_mask = rows < m
    packed = tl.zeros((block_m,), dtype=tl.uint32)

    for pack_idx in tl.static_range(0, 4):
        scale_col = pid_pack * 4 + pack_idx
        k_offsets = scale_col * group_size + cols
        valid_group = scale_col < num_scale_cols
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + k_offsets[None, :] * stride_xk,
            mask=row_mask[:, None] & valid_group,
            other=0.0,
        ).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1.0e-4)
        scale_unrounded = amax * (1.0 / 448.0)
        bits = scale_unrounded.to(tl.uint32, bitcast=True)
        mantissa = bits & 0x7FFFFF
        exp = ((bits >> 23) & 0xFF).to(tl.int32) - 127
        exp_rounded = exp + (mantissa != 0).to(tl.int32)
        scale_bits = ((exp_rounded + 127) & 0xFF).to(tl.uint32) << 23
        scale = scale_bits.to(tl.float32, bitcast=True)

        y = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0)
        tl.store(
            y_ptr + rows[:, None] * stride_ym + k_offsets[None, :] * stride_yk,
            y.to(tl.float8e4nv),
            mask=row_mask[:, None] & valid_group,
        )

        scale_byte = ((exp_rounded + 127) & 0xFF).to(tl.uint32)
        scale_byte = tl.where(valid_group, scale_byte, 0)
        packed = packed | (scale_byte << (pack_idx * 8))

    tl.store(
        s_ptr + rows * stride_sm + pid_pack * stride_sk,
        packed.to(tl.int32),
        mask=row_mask,
    )


def deepseek_v4_deep_gemm_act_quant(
    x: torch.Tensor,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations and emit DeepGEMM's packed E8M0 scale layout."""

    assert block_size == 128, "DeepGEMM official DeepSeek V4 act quant uses 128-wide blocks"
    assert x.is_contiguous(), "input activations must be contiguous"
    k = x.size(-1)
    assert k % block_size == 0, "activation K must be divisible by 128"
    m = x.numel() // k
    x_2d = x.view(m, k)
    y = torch.empty_like(x_2d, dtype=torch.float8_e4m3fn)
    scale = _empty_deep_gemm_scale_layout(m, k // block_size, x.device)

    if m > 0:
        grid = (triton.cdiv(m, 32), triton.cdiv(k // block_size, 4))
        _dsv4_deep_gemm_act_quant_kernel[grid](
            x_2d,
            y,
            scale,
            m,
            k,
            x_2d.stride(0),
            x_2d.stride(1),
            y.stride(0),
            y.stride(1),
            scale.stride(0),
            scale.stride(1),
            k // block_size,
            block_m=32,
            group_size=block_size,
            num_warps=4,
            num_stages=1,
        )
    return y.view_as(x), scale


def _pack_fp8_e8m0_scale_for_deep_gemm_torch(scale_u8: torch.Tensor) -> torch.Tensor:
    num_groups, mn, scale_k = scale_u8.shape
    aligned_mn = _align_to(mn, 4)
    aligned_scale_k = _align_to(scale_k, 4)
    padded = torch.zeros(
        (num_groups, aligned_mn, aligned_scale_k),
        device=scale_u8.device,
        dtype=torch.uint8,
    )
    padded[:, :mn, :scale_k] = scale_u8
    packed = (
        padded.view(-1)
        .view(dtype=torch.int32)
        .view(num_groups, aligned_mn, aligned_scale_k // 4)
    )
    tma_layout = torch.empty(
        (num_groups, aligned_scale_k // 4, aligned_mn),
        device=scale_u8.device,
        dtype=torch.int32,
    ).mT
    tma_layout[:, :, :] = packed
    return tma_layout[:, :mn, :]


def pack_fp8_e8m0_scale_for_deep_gemm(scale: torch.Tensor) -> torch.Tensor:
    """Pack FP8 E8M0 scales into DeepGEMM's TMA-aligned int32 layout."""

    assert scale.dtype == torch.float8_e8m0fnu, (
        "DeepSeek V4 DeepGEMM FP4 scale packer expects float8_e8m0fnu scales"
    )
    assert scale.dim() in (2, 3), "scale must be [mn, k] or [groups, mn, k]"

    remove_dim = scale.dim() == 2
    scale_u8 = scale.contiguous().view(torch.uint8)
    if remove_dim:
        scale_u8 = scale_u8.unsqueeze(0)

    packed_scale = _pack_fp8_e8m0_scale_for_deep_gemm_torch(scale_u8)
    return packed_scale.squeeze(0) if remove_dim else packed_scale


_CLAMP_SILU_MUL_CPP_SRC = r"""
#include <torch/extension.h>

void dsv4_clamp_silu_mul_topk_forward_cuda(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor topk_weights,
    torch::Tensor pos_to_token_topk,
    torch::Tensor out,
    torch::Tensor act,
    double swiglu_limit);
std::vector<torch::Tensor> dsv4_clamp_silu_mul_topk_backward_cuda(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor topk_weights,
    torch::Tensor pos_to_token_topk,
    torch::Tensor act,
    torch::Tensor grad_out,
    double swiglu_limit);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_forward_out", &dsv4_clamp_silu_mul_topk_forward_cuda);
  m.def("topk_backward", &dsv4_clamp_silu_mul_topk_backward_cuda);
}
"""


_CLAMP_SILU_MUL_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace {

__device__ __forceinline__ float load_bf16(const __nv_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}

__device__ __forceinline__ float load_weight(const __nv_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}

__device__ __forceinline__ float load_weight(const float* p, int64_t i) {
  return p[i];
}

__device__ __forceinline__ __nv_bfloat16 store_bf16(float v) {
  return __float2bfloat16_rn(v);
}

__device__ __forceinline__ float clamp_gate(float x, float limit) {
  return limit > 0.0f ? fminf(x, limit) : x;
}

__device__ __forceinline__ float clamp_up(float x, float limit) {
  return limit > 0.0f ? fminf(fmaxf(x, -limit), limit) : x;
}

__device__ __forceinline__ float signed_zero_for_unweighted_act(
    float g,
    float u,
    float limit) {
  float gc = clamp_gate(g, limit);
  float uc = clamp_up(u, limit);
  uint32_t sign = (__float_as_uint(gc) ^ __float_as_uint(uc)) & 0x80000000u;
  return __uint_as_float(sign);
}

__device__ __forceinline__ float silu_exact(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_for_silu_backward(float x) {
  float e = expf(-x);
  return 1.0f / (1.0f + e);
}

__device__ __forceinline__ bool gate_clamp_allows_grad(float x, float limit) {
  return limit <= 0.0f || x <= limit;
}

__device__ __forceinline__ bool up_clamp_allows_grad(float x, float limit) {
  return limit <= 0.0f || (x >= -limit && x <= limit);
}

__device__ __forceinline__ void store_weight_grad(__nv_bfloat16* p, int64_t i, float v) {
  p[i] = store_bf16(v);
}

__device__ __forceinline__ void store_weight_grad(float* p, int64_t i, float v) {
  p[i] = v;
}

template <typename weight_t>
__global__ void topk_forward_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    const weight_t* __restrict__ topk_weights,
    const int32_t* __restrict__ pos_to_token_topk,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ act_out,
    int64_t total,
    int64_t hidden,
    int64_t num_topk_slots,
    float limit) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / hidden;
  int32_t slot = pos_to_token_topk[row];
  float g = load_bf16(gate, idx);
  float u = load_bf16(up, idx);
  if (slot < 0 || slot >= num_topk_slots) {
    float zero = signed_zero_for_unweighted_act(g, u, limit);
    act_out[idx] = zero;
    out[idx] = store_bf16(zero);
    return;
  }
  float weight = load_weight(topk_weights, slot);
  float gc = clamp_gate(g, limit);
  float uc = clamp_up(u, limit);
  float act = silu_exact(gc) * uc;
  act_out[idx] = act;
  out[idx] = store_bf16(act * weight);
}

template <typename weight_t>
__global__ void topk_backward_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    const weight_t* __restrict__ topk_weights,
    const int32_t* __restrict__ pos_to_token_topk,
    const float* __restrict__ act,
    const __nv_bfloat16* __restrict__ grad_out,
    __nv_bfloat16* __restrict__ grad_gate,
    __nv_bfloat16* __restrict__ grad_up,
    weight_t* __restrict__ grad_weights,
    int64_t rows,
    int64_t hidden,
    int64_t num_topk_slots,
    float limit) {
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  int64_t row = blockIdx.x;
  int32_t slot = pos_to_token_topk[row];
  bool valid_slot = slot >= 0 && slot < num_topk_slots;
  float weight = valid_slot ? load_weight(topk_weights, slot) : 0.0f;
  float weight_grad = 0.0f;

  for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    int64_t idx = row * hidden + col;
    float g = load_bf16(gate, idx);
    float u = load_bf16(up, idx);
    float go = load_bf16(grad_out, idx);
    float gc = clamp_gate(g, limit);
    float uc = clamp_up(u, limit);
    float silu = silu_exact(gc);
    float common = go * weight;
    float grad_silu = common * uc;
    float sigmoid = sigmoid_for_silu_backward(gc);
    float gg = (grad_silu * sigmoid) * (1.0f + gc * (1.0f - sigmoid));
    float gu = common * silu;
    if (!gate_clamp_allows_grad(g, limit)) {
      gg = 0.0f;
    }
    if (!up_clamp_allows_grad(u, limit)) {
      gu = 0.0f;
    }
    grad_gate[idx] = store_bf16(gg);
    grad_up[idx] = store_bf16(gu);
    weight_grad += go * act[idx];
  }

  float weight_sum = BlockReduce(reduce_storage).Sum(weight_grad);
  if (threadIdx.x == 0 && valid_slot) {
    store_weight_grad(grad_weights, slot, weight_sum);
  }
}

template <typename weight_t>
void launch_topk_forward(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor topk_weights,
    torch::Tensor pos_to_token_topk,
    torch::Tensor out,
    torch::Tensor act,
    double swiglu_limit) {
  int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream(gate.get_device());
  int64_t hidden = gate.size(1);
  int64_t total = gate.numel();
  int blocks = (total + threads - 1) / threads;
  topk_forward_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(up.data_ptr<at::BFloat16>()),
      reinterpret_cast<const weight_t*>(topk_weights.data_ptr()),
      pos_to_token_topk.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      act.data_ptr<float>(),
      total,
      hidden,
      topk_weights.numel(),
      static_cast<float>(swiglu_limit));
}

}  // namespace

void dsv4_clamp_silu_mul_topk_forward_cuda(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor topk_weights,
    torch::Tensor pos_to_token_topk,
    torch::Tensor out,
    torch::Tensor act,
    double swiglu_limit) {
  if (topk_weights.scalar_type() == at::ScalarType::BFloat16) {
    launch_topk_forward<__nv_bfloat16>(
        gate, up, topk_weights, pos_to_token_topk, out, act, swiglu_limit);
    return;
  }
  if (topk_weights.scalar_type() == at::ScalarType::Float) {
    launch_topk_forward<float>(
        gate, up, topk_weights, pos_to_token_topk, out, act, swiglu_limit);
    return;
  }
  TORCH_CHECK(false, "topk_weights must be bfloat16 or float32");
}

template <typename weight_t>
std::vector<torch::Tensor> launch_topk_backward(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor topk_weights,
    torch::Tensor pos_to_token_topk,
    torch::Tensor act,
    torch::Tensor grad_out,
    double swiglu_limit) {
  auto grad_gate = torch::empty_like(gate);
  auto grad_up = torch::empty_like(up);
  auto grad_weights = torch::empty_like(topk_weights);
  int64_t rows = gate.size(0);
  int64_t hidden = gate.size(1);
  int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream(gate.get_device());
  C10_CUDA_CHECK(cudaMemsetAsync(
      grad_weights.data_ptr(),
      0,
      grad_weights.nbytes(),
      stream));
  topk_backward_kernel<<<rows, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(up.data_ptr<at::BFloat16>()),
      reinterpret_cast<const weight_t*>(topk_weights.data_ptr()),
      pos_to_token_topk.data_ptr<int32_t>(),
      act.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(grad_gate.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(grad_up.data_ptr<at::BFloat16>()),
      reinterpret_cast<weight_t*>(grad_weights.data_ptr()),
      rows,
      hidden,
      topk_weights.numel(),
      static_cast<float>(swiglu_limit));
  return {grad_gate, grad_up, grad_weights};
}

std::vector<torch::Tensor> dsv4_clamp_silu_mul_topk_backward_cuda(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor topk_weights,
    torch::Tensor pos_to_token_topk,
    torch::Tensor act,
    torch::Tensor grad_out,
    double swiglu_limit) {
  if (topk_weights.scalar_type() == at::ScalarType::BFloat16) {
    return launch_topk_backward<__nv_bfloat16>(
        gate, up, topk_weights, pos_to_token_topk, act, grad_out, swiglu_limit);
  }
  if (topk_weights.scalar_type() == at::ScalarType::Float) {
    return launch_topk_backward<float>(
        gate, up, topk_weights, pos_to_token_topk, act, grad_out, swiglu_limit);
  }
  TORCH_CHECK(false, "topk_weights must be bfloat16 or float32");
}
"""


@lru_cache(maxsize=1)
def _dsv4_clamp_silu_mul_ext():
    return load_inline(
        name="dsv4_clamp_silu_mul_topk_ext",
        cpp_sources=_CLAMP_SILU_MUL_CPP_SRC,
        cuda_sources=_CLAMP_SILU_MUL_CUDA_SRC,
        with_cuda=True,
        extra_cuda_cflags=[],
        verbose=False,
    )


def _torch_clamp_silu_mul_topk(
    gate: torch.Tensor,
    up: torch.Tensor,
    topk_weights: torch.Tensor,
    pos_to_token_topk: torch.Tensor,
    swiglu_limit: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    gate_f = gate.float()
    up_f = up.float()
    if swiglu_limit and swiglu_limit > 0:
        up_f = torch.clamp(up_f, min=-swiglu_limit, max=swiglu_limit)
        gate_f = torch.clamp(gate_f, max=swiglu_limit)
    y = F.silu(gate_f) * up_f
    flat_weights = F.pad(topk_weights.reshape(-1), (1, 0))
    slots = pos_to_token_topk.to(torch.int64)
    valid = (slots >= 0) & (slots < flat_weights.numel() - 1)
    gather_idx = torch.where(valid, slots + 1, torch.zeros_like(slots))
    y = y * flat_weights.gather(0, gather_idx).unsqueeze(-1)
    return y.to(out_dtype)


class _DSV4ClampSiluMulTopK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        topk_weights: torch.Tensor,
        pos_to_token_topk: torch.Tensor,
        swiglu_limit: float,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        gate_c = gate.contiguous()
        up_c = up.contiguous()
        weights_c = topk_weights.contiguous()
        pos_c = pos_to_token_topk.contiguous().to(torch.int32)
        out = torch.empty_like(gate_c)
        act = torch.empty(gate_c.shape, device=gate_c.device, dtype=torch.float32)
        _dsv4_clamp_silu_mul_ext().topk_forward_out(
            gate_c,
            up_c,
            weights_c,
            pos_c,
            out,
            act,
            float(swiglu_limit),
        )
        ctx.save_for_backward(gate_c, up_c, weights_c, pos_c, act)
        ctx.swiglu_limit = float(swiglu_limit)
        return out.reshape_as(gate)

    @staticmethod
    def backward(ctx, grad_out):
        gate, up, topk_weights, pos_to_token_topk, act = ctx.saved_tensors
        grad_gate, grad_up, grad_weights = _dsv4_clamp_silu_mul_ext().topk_backward(
            gate,
            up,
            topk_weights,
            pos_to_token_topk,
            act,
            grad_out.contiguous(),
            ctx.swiglu_limit,
        )
        return grad_gate, grad_up, grad_weights, None, None, None


def deepseek_v4_clamp_silu_mul_topk(
    gate: torch.Tensor,
    up: torch.Tensor,
    topk_weights: torch.Tensor,
    pos_to_token_topk: torch.Tensor,
    swiglu_limit: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    assert pos_to_token_topk is not None, "pos_to_token_topk is required"
    if (
        not gate.is_cuda
        or gate.dtype != torch.bfloat16
        or up.dtype != torch.bfloat16
        or out_dtype != torch.bfloat16
        or topk_weights.dtype not in (torch.bfloat16, torch.float32)
    ):
        return _torch_clamp_silu_mul_topk(
            gate, up, topk_weights, pos_to_token_topk, swiglu_limit, out_dtype
        )
    return _DSV4ClampSiluMulTopK.apply(
        gate,
        up,
        topk_weights,
        pos_to_token_topk,
        swiglu_limit,
        out_dtype,
    )


def deepseek_v4_clamp_silu_mul_preexpanded(
    gate: torch.Tensor,
    up: torch.Tensor,
    preexpanded_weights: torch.Tensor,
    swiglu_limit: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    flat_weights = preexpanded_weights.reshape(-1)
    rows = gate.numel() // gate.shape[-1] if gate.dim() > 0 else 0
    if flat_weights.numel() != rows:
        raise ValueError(
            "DeepSeek V4 preexpanded activation weights must have one value per row: "
            f"got {flat_weights.numel()} weights for {rows} rows"
        )
    pos_to_token_topk = torch.arange(
        rows,
        device=gate.device,
        dtype=torch.int32,
    )
    return deepseek_v4_clamp_silu_mul_topk(
        gate,
        up,
        flat_weights,
        pos_to_token_topk,
        swiglu_limit,
        out_dtype,
    )


dsv4_deep_gemm_act_quant = deepseek_v4_deep_gemm_act_quant
dsv4_clamp_silu_mul_topk = deepseek_v4_clamp_silu_mul_topk
dsv4_clamp_silu_mul_preexpanded = deepseek_v4_clamp_silu_mul_preexpanded


@lru_cache(maxsize=None)
@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _reduce_fused_topk_fp32_kernel(
    hidden: int,
    num_topk: int,
    in_dtype: str,
):
    num_expanded_tokens = T.dynamic("num_expanded_tokens")
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_expanded_tokens, hidden), in_dtype],
        token_topk_to_pos: T.Tensor[(num_tokens, num_topk), _INT32],
        out: T.Tensor[(num_tokens, hidden), _FP32],
    ):
        with T.Kernel(num_tokens, threads=128) as (pid_token,):
            reduced = T.alloc_fragment((hidden,), _FP32)
            topk_to_pos = T.alloc_fragment((num_topk,), _INT32)

            T.clear(reduced)
            T.copy(token_topk_to_pos[pid_token, :], topk_to_pos)
            for k in T.unroll(num_topk):
                pos = topk_to_pos[k]
                T.assume(pos < num_expanded_tokens)
                if pos >= 0:
                    for i in T.Parallel(hidden):
                        reduced[i] += T.Cast(_FP32, x[pos, i])

            for i in T.Parallel(hidden):
                out[pid_token, i] = reduced[i]

    return kernel


def reduce_fused_topk_fp32(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
) -> torch.Tensor:
    assert x.is_contiguous(), "expanded expert outputs must be contiguous"
    assert token_topk_to_pos.is_contiguous(), "token_topk_to_pos must be contiguous"
    assert x.dim() == 2, "expanded expert outputs must be [expanded_tokens, hidden]"
    assert token_topk_to_pos.dim() == 2, "token_topk_to_pos must be [tokens, topk]"
    hidden = x.size(-1)
    num_tokens = token_topk_to_pos.size(0)
    out = torch.empty((num_tokens, hidden), dtype=torch.float32, device=x.device)
    if num_tokens > 0:
        kernel = _reduce_fused_topk_fp32_kernel(
            hidden,
            token_topk_to_pos.size(1),
            T.dtype(x.dtype),
        )
        kernel(x, token_topk_to_pos.to(torch.int32), out)
    return out


@lru_cache(maxsize=None)
@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _grouped_fp4_gemm_kernel(
    num_experts: int,
    n: int,
    k_dim: int,
    out_dtype: str = _BF16,
    accum_dtype: str = _FP32,
    scale_dtype: str = _FE8M0,
):
    """Expert-major FP8 activation x FP4 weight GEMM.

    This is the official DeepSeek V4 fp4_gemm tile shape with the expert and output
    axes flattened on the weight tensors. Rows must be grouped by expert in
    32-row aligned segments, which is exactly what TileKernels' fused MoE
    mapping produces when called with alignment >= 32. Padding rows use
    expert id -1 and are written as zero.
    """

    m = T.symbolic("m")
    act_group_size = 128
    weight_group_size = 32
    block_m = 32
    block_n = 128
    block_k = 32
    n_sub = act_group_size // block_k

    @T.prim_func
    def kernel(
        a: T.Tensor[(m, k_dim), _FP8],
        b: T.Tensor[(num_experts * n, k_dim), _FP4],
        c: T.Tensor[(m, n), out_dtype],
        scales_a: T.Tensor[(m, T.ceildiv(k_dim, act_group_size)), scale_dtype],
        scales_b: T.Tensor[
            (num_experts * n, T.ceildiv(k_dim, weight_group_size)), scale_dtype
        ],
        pos_to_expert: T.Tensor[(m,), _INT32],
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=128) as (
            bx,
            by,
        ):
            a_shared = T.alloc_shared((block_m, block_k), _FP8)
            b_fp4_shared = T.alloc_shared((block_n, block_k), _FP4)
            b_shared = T.alloc_shared((block_n, block_k), _FP8)
            c_shared = T.alloc_shared((block_m, block_n), out_dtype)
            c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
            c_local_accum = T.alloc_fragment((block_m, block_n), accum_dtype)
            scale_a_frag = T.alloc_fragment((block_m,), _FP32)
            scale_b_frag = T.alloc_fragment((block_n,), _FP32)

            T.use_swizzle(panel_size=10)
            expert = pos_to_expert[by * block_m]
            T.clear(c_local)
            T.clear(c_local_accum)

            if expert >= 0:
                k_iters = T.ceildiv(k_dim, block_k)
                for kk in T.Pipelined(k_iters, num_stages=2):
                    T.copy(a[by * block_m, kk * block_k], a_shared)
                    T.copy(b[expert * n + bx * block_n, kk * block_k], b_fp4_shared)
                    for i, j in T.Parallel(block_n, block_k):
                        b_shared[i, j] = T.Cast(
                            _FP8, T.Cast(_FP32, b_fp4_shared[i, j])
                        )

                    for i in T.Parallel(block_n):
                        scale_b_frag[i] = T.Cast(
                            _FP32, scales_b[expert * n + bx * block_n + i, kk]
                        )
                    for i in T.Parallel(block_m):
                        scale_a_frag[i] = T.Cast(
                            _FP32, scales_a[by * block_m + i, kk // n_sub]
                        )

                    T.gemm(a_shared, b_shared, c_local, transpose_B=True)

                    for i, j in T.Parallel(block_m, block_n):
                        c_local_accum[i, j] += (
                            c_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
                        )
                    T.clear(c_local)

            T.copy(c_local_accum, c_shared)
            T.copy(c_shared, c[by * block_m, bx * block_n])

    return kernel


def grouped_fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    pos_to_expert: torch.Tensor,
    scale_dtype: Optional[torch.dtype] = torch.float8_e8m0fnu,
) -> torch.Tensor:
    assert a.is_contiguous() and b.is_contiguous(), "input tensors must be contiguous"
    assert b.dim() == 3, "grouped FP4 weights must be [experts, out, in//2]"
    assert pos_to_expert.is_contiguous(), "pos_to_expert must be contiguous"
    k_dim = a.size(-1)
    m = a.numel() // k_dim
    n = b.size(1)
    c = a.new_empty(*a.size()[:-1], n, dtype=torch.bfloat16)
    if has_deep_gemm_official_fp8_fp4():
        grouped_layout = pos_to_expert.view(-1)
        if grouped_layout.dtype != torch.int32:
            grouped_layout = grouped_layout.to(torch.int32)
        _DEEP_GEMM_OFFICIAL_GROUPED_FP8_FP4_GEMM(
            (a.view(m, k_dim), a_s),
            (b.view(torch.int8), b_s),
            c.view(m, n),
            grouped_layout,
            recipe_a=(1, 128),
            recipe_b=(1, 32),
            compiled_dims="nk",
            disable_ue8m0_cast=True,
        )
        return c

    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "scaling factor tensors must be contiguous"
    )
    assert b_s.dim() == 3, "grouped FP4 scales must be [experts, out, in//32]"
    tl_scale_dtype = _FE8M0 if scale_dtype == torch.float8_e8m0fnu else _FP32
    kernel = _grouped_fp4_gemm_kernel(b.size(0), n, k_dim, scale_dtype=tl_scale_dtype)
    kernel(
        a.view(m, k_dim),
        b.view(b.size(0) * b.size(1), b.size(2)),
        c.view(m, n),
        a_s.view(m, -1),
        b_s.view(b_s.size(0) * b_s.size(1), b_s.size(2)),
        pos_to_expert.view(-1).to(torch.int32),
    )
    return c
