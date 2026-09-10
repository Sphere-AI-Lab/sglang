"""Execute host dispatch decisions without loading CUDA kernels."""

import ast
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _function(relative_path, name, **namespace):
    path = Path(__file__).resolve().parents[4] / "python/sglang/srt" / relative_path
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "active,capturing", [(False, False), (True, False), (False, True)]
)
def test_inactive_batch_bypasses_rotations_except_during_capture(
    monkeypatch, active, capturing
):
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.oft.oft_moe_runners",
        SimpleNamespace(OFTInfo=SimpleNamespace),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.runner",
        SimpleNamespace(get_is_capture_mode=lambda: capturing),
    )
    get_info = _function("oft/layers.py", "_get_oft_info")
    rotations = object()
    layer = SimpleNamespace(
        oft_backend=SimpleNamespace(
            batch_info=SimpleNamespace(moe_oft_info=object(), has_active_oft=active),
            max_ofts_per_batch=3,
        ),
        base_layer=SimpleNamespace(num_experts=3),
        w13_oft_r=rotations,
        w1_oft_r=None,
        w3_oft_r=None,
        w2_oft_r=rotations,
    )
    info = get_info(layer)
    if not active and not capturing:
        assert info is None
    else:
        # Capture must retain pool buffers even when no adapter is active now.
        assert info.w13_oft_r is rotations
        assert info.w2_oft_r is rotations


def test_absent_oft_info_keeps_native_invoker():
    build = _function("oft/oft_moe_runners.py", "make_oft_invoke")

    def native(*args, **kwargs):
        pass

    assert build(SimpleNamespace(), native, None) is native


class _AtGateUp(Exception):
    pass


@pytest.mark.parametrize(
    "adapter,expected_sorted,expected_rows",
    [("base", True, 16), ("lora", False, 4), ("oft", False, 4)],
)
def test_adapter_gemm_uses_route_major_intermediates(
    adapter, expected_sorted, expected_rows
):
    def native(*args, **kwargs):
        # Actual kernel call boundary: check layout selection and buffer size.
        assert kwargs["c_sorted"] is expected_sorted
        assert args[3].shape == (expected_rows, 8)
        raise _AtGateUp

    sequence = _function(
        "layers/moe/moe_runner/triton_utils/fused_moe.py",
        "_fused_moe_kernel_sequence",
        invoke_fused_moe_kernel=native,
        torch=SimpleNamespace(
            bfloat16="bf16",
            empty=lambda shape, **kwargs: SimpleNamespace(shape=shape),
        ),
        tl=SimpleNamespace(bfloat16="bf16", float16="fp16"),
        get_exec=lambda: SimpleNamespace(
            moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=False)
        ),
    )
    # Unused optional quantization/bias arguments are None. Set every input
    # consumed before the first GEMM explicitly, with hand-derived shapes:
    # 2 tokens * top_k 2 = 4 route rows; with 5 experts, TMA adds
    # min(4, 6) * (4 - 1) = 12 padding rows, for 16 rows total.
    kwargs = {
        name: None
        for name, p in inspect.signature(sequence).parameters.items()
        if p.default is inspect.Parameter.empty
    }
    kwargs.update(
        hidden_states=SimpleNamespace(shape=(2, 4), dtype="bf16", device="cpu"),
        w1=SimpleNamespace(shape=(5, 8, 4)),
        w2=SimpleNamespace(shape=(5, 4, 4)),
        topk_ids=SimpleNamespace(shape=(2, 2)),
        config={"BLOCK_SIZE_M": 4},
        down_moe_use_tma=True,
        up_moe_use_tma=False,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        no_combine=False,
        inplace=True,
        apply_router_weight_on_input=False,
    )
    if adapter == "lora":
        kwargs["hooks"] = SimpleNamespace(after_gate_up=lambda: None, after_down=None)
    elif adapter == "oft":
        kwargs["invoke"] = lambda *args, **kw: native(*args, **kw)
    with pytest.raises(_AtGateUp):
        sequence(**kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
