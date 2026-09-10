"""Execute model entrypoints up to projection admission, without model weights."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Tensor:
    def __init__(self, dtype):
        self.dtype = dtype

    def to(self, dtype):
        return _Tensor(dtype)


class _ProjectionReached(Exception):
    pass


def _forward(model, force_dense):
    filename, class_name = (
        ("qwen3.py", "Qwen3Attention")
        if model == "attention"
        else ("qwen2.py", "Qwen2MLP")
    )
    path = Path(__file__).resolve().parents[4] / "python/sglang/srt/models" / filename
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    torch = SimpleNamespace(
        **{
            name: name
            for name in (
                "float16",
                "bfloat16",
                "float32",
                "float64",
            )
        }
    )
    namespace = {
        "torch": torch,
        "should_force_bfloat16_dense_tensor_math": lambda: force_dense,
        "_is_npu": False,
    }
    exec("from __future__ import annotations\n" + ast.unparse(method), namespace)
    return namespace["forward"]


@pytest.mark.parametrize("model", ["attention", "mlp"])
@pytest.mark.parametrize("force_dense", [False, True])
@pytest.mark.parametrize(
    "weight_dtype,expected_dtype",
    [
        ("float8_e4m3fn", "bfloat16"),
        ("int8", "bfloat16"),
        ("uint8", "bfloat16"),
        ("float16", "float16"),
        ("float32", "float32"),
    ],
)
def test_projection_input_is_not_cast_to_quantized_storage(
    model, force_dense, weight_dtype, expected_dtype
):
    observed = []

    def project(x):
        observed.append(x.dtype)
        raise _ProjectionReached

    projection = SimpleNamespace(weight=_Tensor(weight_dtype))
    if model == "attention":
        layer = SimpleNamespace(
            qkv_proj=projection,
            use_fused_qk_norm_mrope=False,
            forward_prepare_native=lambda positions, hidden_states: project(
                hidden_states
            ),
        )
        args = (None, _Tensor("bfloat16"), SimpleNamespace())
    else:

        class Projection:
            weight = projection.weight

            def __call__(self, x):
                return project(x)

        layer = SimpleNamespace(gate_up_proj=Projection())
        args = (_Tensor("bfloat16"),)
    with pytest.raises(_ProjectionReached):
        _forward(model, force_dense)(layer, *args)
    assert observed == [expected_dtype]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
