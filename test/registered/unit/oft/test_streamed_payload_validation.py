"""Validate lazy admission with production resolver bodies and CPU tensors."""

import ast
import importlib.util
import logging
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
spec = importlib.util.spec_from_file_location(
    "validation_ci", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def functions(path):
    tree = ast.parse((SRT / path).read_text())
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id.startswith("_") for t in n.targets)
        )
    ]
    tree.body.insert(
        0,
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
    )
    scope = dict(torch=torch, re=re, os=os, logger=logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(tree), str(SRT / path), "exec"), scope)
    return scope


@pytest.fixture
def runtime(monkeypatch):
    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    def layer_id(name):
        match = re.search(r"layers\.(\d+)\.", name)
        return int(match[1]) if match else None

    stub("sglang.srt.layers.utils", get_layer_id=layer_id)
    stub("sglang.srt.oft._streamed_audit", record_expert_partition=lambda *a, **k: None)
    stub(
        "sglang.srt.oft.mem_pool",
        normalize_merged_oft_weights=lambda weights, **kw: weights,
    )
    stub(
        "sglang.srt.oft.oft_manager",
        _get_fused_moe_weight_device=lambda moe: torch.device("cpu"),
    )
    pool = NS(
        tp_rank=0,
        R_buffer={"qkv_proj": [torch.full((2, 6, 4, 4), 7.0)]},
        embedding_R_buffer={"embed_tokens": torch.full((2, 2, 4, 4), 7.0)},
        lm_head_R_buffer={"lm_head": torch.full((2, 2, 4, 4), 7.0)},
        _groups={},
    )
    pool._resolve_oft_tensor_plan = lambda *a: ("qkv_proj", None, False, None, 1)
    pool._slice_oft_compact_weight = lambda value, module: value
    pool.slot = lambda group, layer, slot: pool._groups[group][layer][slot]
    moe = NS(
        num_local_experts=2,
        moe_tp_rank=0,
        moe_tp_size=2,
        _map_global_expert_id_to_local_expert_id=lambda i: i if i < 2 else -1,
    )
    manager = NS(
        memory_pool=pool,
        adapter_modules=[],
        oft_type="canonical_oft",
        oft_r_dtype=torch.float32,
        _find_fused_moe_modules=lambda: {0: moe},
    )
    pool._groups = {
        group: {0: torch.full((2, 2, 2, 4, 4), 7.0)}
        for group in ("w1_oft_r", "w2_oft_r", "w3_oft_r", "w13_oft_r")
    }
    resolve = functions("oft/streamed_weight_loader.py")[
        "_resolve_streamed_oft_tensor_groups"
    ]
    return manager, pool, lambda tensors, block=4: resolve(manager, tensors, block)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros(6),
        torch.zeros(2, 2),
        torch.zeros(2, 6, dtype=torch.int64),
        torch.zeros(7, 6),
    ],
)
def test_rejects_bad_dense_compact_before_publication(runtime, tensor):
    _, pool, resolve = runtime
    before = pool.R_buffer["qkv_proj"][0].clone()
    plan, error = resolve([("model.layers.0.q_proj.oft_R", tensor)])
    assert plan is None and error
    assert torch.equal(before, pool.R_buffer["qkv_proj"][0])


@pytest.mark.parametrize(
    "slice_index,split_count,blocks", [(1, 3, 3), (0, 4, 1), (3, 3, 1), (0, 1, 1)]
)
def test_rejects_split_capacity_and_plan_errors(
    runtime, slice_index, split_count, blocks
):
    _, pool, resolve = runtime
    pool._resolve_oft_tensor_plan = lambda *a: (
        "qkv_proj",
        None,
        False,
        slice_index,
        split_count,
    )
    assert resolve([("model.layers.0.q_proj.oft_R", torch.zeros(blocks, 6))])[0] is None


@pytest.mark.parametrize("name", ["embed_tokens", "lm_head"])
def test_rejects_optional_buffer_overflow(runtime, name):
    _, _, resolve = runtime
    assert resolve([(f"model.{name}.oft_R", torch.zeros(3, 6))])[0] is None


def test_rejects_block_size_exceeding_buffer(runtime):
    _, _, resolve = runtime
    assert (
        resolve([("model.layers.0.q_proj.oft_R", torch.zeros(1, 15))], block=6)[0]
        is None
    )


@pytest.mark.parametrize("blocks,width", [(1, 1), (2, 6), (6, 8), (0, 6)])
def test_preserves_shared_partial_padded_and_empty_dense_payloads(
    runtime, blocks, width
):
    _, _, resolve = runtime
    plan, error = resolve([("model.layers.0.q_proj.oft_R", torch.zeros(blocks, width))])
    assert plan is not None and not error


def test_row_capacity_checked_after_tp_slice(runtime):
    _, pool, resolve = runtime
    pool._resolve_oft_tensor_plan = lambda *a: ("qkv_proj", object(), True, None, 1)
    pool._slice_oft_compact_weight = lambda value, module: value[:6]
    assert resolve([("model.layers.0.o_proj.oft_R", torch.zeros(12, 6))])[0] is not None


def test_rejects_independent_expert_up_in_shared_layout(runtime):
    manager, _, resolve = runtime
    manager.oft_type = "oft"
    assert (
        resolve([("model.layers.0.mlp.experts.0.up_proj.oft_R", torch.zeros(2, 6))])[0]
        is None
    )


@pytest.mark.parametrize("projection,blocks", [("gate_proj", 3), ("down_proj", 3)])
def test_rejects_expert_buffer_shape_and_tp_divisibility(runtime, projection, blocks):
    _, _, resolve = runtime
    assert (
        resolve(
            [
                (
                    f"model.layers.0.mlp.experts.0.{projection}.oft_R",
                    torch.zeros(blocks, 6),
                )
            ]
        )[0]
        is None
    )


def test_rejects_expert_scatter_inconsistent_blocks(runtime):
    _, _, resolve = runtime
    assert (
        resolve(
            [
                (f"model.layers.0.mlp.experts.{i}.gate_proj.oft_R", torch.zeros(n, 6))
                for i, n in [(0, 2), (1, 3)]
            ]
        )[0]
        is None
    )


def test_accepts_expert_tp_and_nonlocal_payloads_without_mutation(runtime):
    _, pool, resolve = runtime
    before = pool._groups["w2_oft_r"][0].clone()
    plan, error = resolve(
        [
            (f"model.layers.0.mlp.experts.{i}.down_proj.oft_R", torch.zeros(4, 6))
            for i in (0, 5)
        ]
    )
    assert plan is not None and not error
    assert torch.equal(before, pool._groups["w2_oft_r"][0])


def test_expert_secondary_dtype_is_converted_by_writer(runtime):
    _, _, resolve = runtime
    plan, error = resolve(
        [
            ("model.layers.0.mlp.experts.0.gate_proj.oft_R", torch.zeros(2, 6)),
            (
                "model.layers.0.mlp.experts.1.gate_proj.oft_R",
                torch.zeros(2, 6, dtype=torch.int64),
            ),
        ]
    )
    assert plan is not None and not error


def test_expert_scalar_rejected_without_raising(runtime):
    _, _, resolve = runtime
    plan, error = resolve(
        [("model.layers.0.mlp.experts.0.gate_proj.oft_R", torch.tensor(1.0))]
    )
    assert plan is None and error


@pytest.mark.parametrize("width", [1, 6, 8])
def test_accepted_widths_work_with_production_cayley(width):
    scope = functions("oft/torch_ops/oft_ops.py")
    scope["HAS_TRITON"] = False
    result = scope["precompute_oft_r"](torch.zeros(2, width), 4)
    assert torch.equal(result, torch.eye(4).repeat(2, 1, 1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
