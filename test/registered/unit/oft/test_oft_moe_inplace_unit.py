"""Host constructor contract; this does not exercise CUDA output writes."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "backend_name,oft_type", [("triton", "canonical_oft"), ("marlin", "oft")]
)
def test_runner_receives_non_inplace_shared_config(monkeypatch, backend_name, oft_type):
    # Execute the unchanged production constructor without importing CUDA/Torch.
    # Only module initialization and the runner/quantization dependencies are
    # replaced: the config mutation and runner argument selection remain real.
    path = Path(__file__).resolve().parents[4] / "python/sglang/srt/oft/layers.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "FusedMoEWithOFT"
    )
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    ]
    namespace = {"nn": SimpleNamespace(Module=object)}
    exec("from __future__ import annotations\n" + ast.unparse(cls), namespace)

    config = SimpleNamespace(inplace=True)
    backend = SimpleNamespace(
        is_auto=lambda: False,
        is_triton=lambda: backend_name == "triton",
        is_marlin=lambda: backend_name == "marlin",
    )

    def create_runner(runner_backend, runner_config, *, peft_enabled):
        # Fail at construction, before a runner could snapshot an unsafe value.
        assert runner_config is config
        assert runner_config.inplace is False
        return SimpleNamespace(config=runner_config)

    for name, module in {
        "sglang.srt.layers.moe": SimpleNamespace(MoeRunnerBackend=object),
        "sglang.srt.layers.moe.moe_runner.runner": SimpleNamespace(
            MoeRunner=create_runner
        ),
        "sglang.srt.layers.moe.utils": SimpleNamespace(
            get_moe_runner_backend=lambda: backend
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    base_layer = SimpleNamespace(
        moe_runner_config=config,
        quant_method=SimpleNamespace(
            get_triton_quant_info=lambda layer: None,
            get_marlin_quant_info=lambda layer: None,
        ),
        dispatcher=None,
        num_experts=3,
        num_local_experts=3,
        top_k=2,
        should_fuse_routed_scaling_factor_in_topk=False,
    )
    layer = namespace["FusedMoEWithOFT"](base_layer, SimpleNamespace(), oft_type)

    assert layer.moe_runner_config is base_layer.moe_runner_config
    assert base_layer.moe_runner_config.inplace is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
