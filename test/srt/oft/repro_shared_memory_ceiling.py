"""Minimal reproduction: the fused QKV kernel cannot launch above BS=128.

The kernel stages the whole BS x BS rotation block in shared memory, so its
footprint is 6*BS*(BS+128) bytes against sm_90's 232,448 B per-block limit:

    BS=128 ->   196,608 B  (fits, 85% of budget)
    BS=256 ->   589,824 B
    BS=512 -> 1,966,080 B
    BS=1024-> 7,077,888 B

Above 128 it fails inside Triton as `OutOfResources`, after the SGLang server
has already started -- which is how it presents in production.

The reference is fp32 (`x.float() @ W.float().T`), so a passing size reports
~1.2e-04 rather than exactly zero -- that is bf16 rounding in the kernel's own
accumulate-and-cast, not an error. The parity bar for this kernel is 2e-3.

Run: python test/srt/oft/repro_shared_memory_ceiling.py
"""

import torch

from sglang.srt.peft.oft.triton_ops.fused_rotate_project import fused_rotate_project_qkv

# Llama-3.1-8B fused QKV: hidden 4096 in, (32 + 8 + 8) * 128 = 6144 out.
K, OUT, M = 4096, [4096, 1024, 1024], 64
dev, dt = "cuda", torch.bfloat16

x = (torch.randn(M, K, device=dev, dtype=dt) * 0.01).contiguous()
W = (torch.randn(sum(OUT), K, device=dev, dtype=dt) * 0.02).contiguous()

for BS in (16, 32, 64, 128, 256, 512, 1024):
    blocks = 3 * (K // BS)
    # Identity rotation: the fused result must equal a plain projection, which
    # makes any mismatch the kernel's rather than a drifting reference's.
    R = torch.eye(BS, device=dev, dtype=dt).expand(blocks, BS, BS).contiguous()
    try:
        out = fused_rotate_project_qkv(x, R, W, OUT)
        torch.cuda.synchronize()
        err = (out.float() - (x.float() @ W.float().T)).abs().max().item()
        print(f"BS={BS:>5}  OK    max|out-ref|={err:.2e}")
    except Exception as exc:  # noqa: BLE001 -- report whatever Triton raises
        print(f"BS={BS:>5}  FAIL  {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
    del R
    torch.cuda.empty_cache()
