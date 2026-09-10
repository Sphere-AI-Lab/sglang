"""Reject unsupported MoE adapter verification before graph initialization."""

import ast
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
spec = importlib.util.spec_from_file_location(
    "ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_initializer(method, scope):
    if method == "LoRA":
        path = SRT / "model_executor/model_runner.py"
        cls = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
        )
        function = next(
            node
            for node in cls.body
            if getattr(node, "name", "") == "init_lora_manager"
        )
    else:
        path = SRT / "oft/integration.py"
        function = next(
            node
            for node in ast.parse(path.read_text()).body
            if getattr(node, "name", "") == "_init_oft_manager"
        )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)
    return scope[function.name]


@pytest.mark.parametrize("method", ["LoRA", "OFT"])
@pytest.mark.parametrize("is_moe", [False, True])
@pytest.mark.parametrize("algorithm", [None, "NGRAM"])
@pytest.mark.parametrize("graphs_disabled", [False, True])
def test_ngram_guard_uses_actual_adapter_targets(
    monkeypatch, method, is_moe, algorithm, graphs_disabled
):
    # The supplied record may differ from the resolved runtime configuration.
    args = NS(
        speculative_algorithm="NGRAM" if algorithm is None else None,
        max_ofts_per_batch=2,
        oft_backend="triton",
        max_oft_block_size=4,
        peft_target_modules=[],
        peft_paths=[],
        enable_weights_cpu_backup=False,
    )
    manager = NS(
        lora_backend=NS(is_moe_lora=is_moe),
        oft_backend=NS(is_moe_oft=is_moe),
    )

    # Manager construction is the GPU boundary. Its completed backend tells
    # us whether any MoE adapter wrappers were actually installed.
    def factory(**kwargs):
        return manager

    staged_module = ModuleType("sglang.srt.oft.staged_manager")
    staged_module.StagedOFTManager = factory
    monkeypatch.setitem(sys.modules, staged_module.__name__, staged_module)
    runner = NS(
        model=object(),
        model_config=NS(hf_config=None),
        load_config=None,
        dtype=None,
        server_args=args,
        ps=NS(tp_size=1, tp_rank=0),
        memory_saver_adapter=None,
        _get_lora_manager_class=lambda: factory,
    )
    graph_calls = []
    scope = dict(
        logger=logging.getLogger(__name__),
        get_spec=lambda: NS(speculative_algorithm=algorithm),
        get_lora=lambda: NS(
            max_loras_per_batch=2,
            lora_backend="triton",
            max_lora_rank=8,
            lora_target_modules=[],
            lora_paths=[],
        ),
        cuda_graph_fully_disabled=lambda: graphs_disabled,
        init_lora_cuda_graph_moe_buffers=lambda **kwargs: graph_calls.append(kwargs),
    )
    initialize = _load_initializer(method, scope)
    call_args = (runner,) if method == "LoRA" else (runner, args)
    if is_moe and algorithm == "NGRAM":
        with pytest.raises(ValueError, match=f"NGRAM.*MoE {method}"):
            initialize(*call_args)
        assert not graph_calls
    else:
        initialize(*call_args)
        assert getattr(runner, f"{method.lower()}_manager") is manager


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
