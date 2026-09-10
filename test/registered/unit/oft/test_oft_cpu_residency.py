"""Exercise CPU ownership and GPU-slot lifecycle with production method bodies.

GPU writes are represented by small CPU tensors. AST loading avoids importing
GPU runtimes; adapter/pool/update control flow is not reimplemented.
"""

import ast
import importlib.util
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"


def load_class(path, name, methods=None, bases=(), **scope):
    node = next(
        n
        for n in ast.parse((SRT / path).read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    node.bases = [ast.Name(id=f"base{i}", ctx=ast.Load()) for i in range(len(bases))]
    scope.update({f"base{i}": base for i, base in enumerate(bases)})
    if methods is not None:
        node.body = [n for n in node.body if getattr(n, "name", None) in methods]
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(SRT / path), "exec"), scope)
    return scope[name]


spec = importlib.util.spec_from_file_location(
    "cpu_ci", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@dataclass(frozen=True)
class Ref:
    adapter_id: str
    adapter_name: str
    adapter_path: str = "__tensors__"
    adapter_version: int = 1
    pinned: bool = False
    reloadable: bool = False


class Config:
    @staticmethod
    def from_dict(value):
        return NS(block_size=value["oft_block_size"], oft_added_tokens_size=0)


class Adapter:
    def __init__(self, uid, config, *args):
        self.uid = uid
        self.config = config
        self.block_size = config.block_size
        self.weights = {}

    def _process_weight(self, name, value):
        self.weights[name] = value

    def _normalize_weights(self):
        pass


Adapter.initialize_weights_from_tensors = load_class(
    "oft/oft.py", "OFTAdapter", {"initialize_weights_from_tensors"}, torch=torch
).initialize_weights_from_tensors


@pytest.fixture
def runtime(monkeypatch):
    base = load_class(
        "oft/base/manager.py",
        "AdapterManager",
        {"validate_batch", "_make_update_result"},
    )
    manager_type = load_class(
        "oft/oft_manager.py",
        "OFTManager",
        {
            "load_adapter_from_tensors",
            "_try_transactional_streamed_upsert",
            "_is_expected_previous_streamed_ref",
            "_prepare_mem_pool_batch",
            "_restore_streamed_oft",
            "unload_streamed_adapter",
            "create_oft_update_result",
            "register_streamed_adapter",
        },
        bases=(base,),
        OFTConfig=Config,
        OFTAdapter=Adapter,
        torch=torch,
        logger=logging.getLogger(__name__),
        EMPTY_SLOT="empty",
    )
    pool_base = load_class(
        "oft/base/mem_pool.py",
        "AdapterMemPool",
        {"_acquire_buffer_slot"},
        EMPTY_SLOT="empty",
        logger=logging.getLogger(__name__),
    )
    pool_type = load_class(
        "oft/mem_pool.py",
        "OFTMemoryPool",
        {"prepare_oft_batch"},
        EMPTY_SLOT="empty",
        logger=logging.getLogger(__name__),
        bases=(pool_base,),
    )
    manager = manager_type()
    manager.refs = {}
    manager.adapters = {}
    manager.configs = {}
    manager.num_pinned = 0
    manager.max_adapters_per_batch = 2
    manager.base_hf_config = manager.load_config = manager.oft_backend = None
    manager.adapter_modules = []
    manager.embed_tokens_module = manager.lm_head_module = None
    manager._set_expert_oft = Mock()
    manager.device = torch.device("cpu")
    manager._update_output_cls = lambda: NS
    pool = pool_type()
    pool.max_adapters_per_batch = pool.max_ofts_per_batch = 2
    pool.max_oft_block_size = 4
    pool.staging_idx = 2
    pool._active_versions = {}
    pool.staged_identity = lambda: None
    pool.uid_to_buffer_id = {}
    pool.buffer_id_to_uid = ["empty", "empty"]
    pool.values = torch.zeros(3)
    pool.reset_buffer_slot_to_identity = lambda slot: pool.values[slot].zero_()
    pool.copy_supported_buffer_slot = lambda src, dst: pool.values[dst].copy_(
        pool.values[src]
    )
    pool.eviction_policy = NS(
        mark_used=lambda uid: None,
        remove=lambda uid: None,
        select_victim=lambda candidates: sorted(candidates, key=str)[0],
    )
    pool.load_oft_weight_to_buffer = lambda uid, slot, *args: pool.values[slot].zero_()
    manager.memory_pool = pool
    loader = ModuleType("sglang.srt.oft.streamed_weight_loader")
    loader._resolve_streamed_oft_tensor_groups = lambda manager, tensors, block: (
        tuple(tensors),
        "",
    )

    def commit(manager, tensors, plan, slot, *args):
        pool.values[slot] = sum(float(t.sum()) for _, t in tensors)
        return True, ""

    loader._commit_streamed_oft_tensor_groups = commit
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    return manager, pool, loader


def load(manager, name, value, *, version=1, upsert=False):
    ref = Ref(name, name, adapter_version=version)
    result = manager.load_adapter_from_tensors(
        ref, {"weight": torch.tensor([value])}, {"oft_block_size": 4}, upsert=upsert
    )
    assert result.success, result.error_message
    return ref


def test_cpu_snapshot_does_not_alias_sender():
    adapter = Adapter("a", NS(block_size=4))
    source = torch.tensor([2.0], requires_grad=True)
    adapter.initialize_weights_from_tensors({"weight": source})
    with torch.no_grad():
        source.fill_(9)
    assert adapter.weights["weight"].item() == 2
    assert not adapter.weights["weight"].requires_grad


def test_native_load_is_cpu_backed_and_does_not_need_a_serving_slot(runtime):
    manager, pool, _ = runtime
    for name in ("a", "b", "c"):
        load(manager, name, 2)
    assert set(manager.adapters) == {"a", "b", "c"}
    assert not pool.uid_to_buffer_id
    assert all(not ref.reloadable for ref in manager.refs.values())


def test_eviction_restores_native_cpu_weights_and_keeps_registration(runtime):
    manager, pool, _ = runtime
    load(manager, "a", 2)
    load(manager, "b", 3)
    manager._prepare_mem_pool_batch({None, "a"})
    assert manager.validate_batch({None, "b"})
    manager._prepare_mem_pool_batch({None, "b"})
    assert "a" not in pool.uid_to_buffer_id
    assert "a" in manager.adapters and "a" in manager.refs
    manager._prepare_mem_pool_batch({None, "a"})
    assert pool.values[pool.uid_to_buffer_id["a"]].item() == 2


def test_nonresident_update_restores_latest_version(runtime):
    manager, pool, _ = runtime
    load(manager, "a", 2)
    load(manager, "a", 7, version=2, upsert=True)
    manager._prepare_mem_pool_batch({"a"})
    assert pool.values[pool.uid_to_buffer_id["a"]].item() == 7
    assert manager.refs["a"].adapter_version == 2


def test_resident_update_publishes_cpu_copy_only_after_gpu_success(runtime):
    manager, pool, loader = runtime
    ref = load(manager, "a", 2)
    manager._prepare_mem_pool_batch({"a"})
    old_cpu = manager.adapters["a"]

    def fail(manager, tensors, plan, slot, *args):
        pool.values[slot] = 99
        return False, "injected GPU failure"

    loader._commit_streamed_oft_tensor_groups = fail
    result = manager.load_adapter_from_tensors(
        replace(ref, adapter_version=2),
        {"weight": torch.tensor([7.0])},
        {"oft_block_size": 4},
        upsert=True,
    )
    assert not result.success and result.previous_adapter_preserved
    assert manager.adapters["a"] is old_cpu
    assert manager.refs["a"] is ref
    assert pool.values[pool.uid_to_buffer_id["a"]].item() == 2


def test_resident_update_survives_later_eviction(runtime):
    manager, pool, _ = runtime
    load(manager, "a", 2)
    manager._prepare_mem_pool_batch({None, "a"})
    load(manager, "a", 7, version=2, upsert=True)
    load(manager, "b", 3)
    manager._prepare_mem_pool_batch({None, "b"})
    manager._prepare_mem_pool_batch({None, "a"})
    assert pool.values[pool.uid_to_buffer_id["a"]].item() == 7


def test_pinned_cpu_backing_does_not_make_adapter_evictable(runtime):
    manager, pool, _ = runtime
    ref = load(manager, "a", 2)
    manager.refs["a"] = replace(ref, pinned=True)
    manager.num_pinned = 1
    load(manager, "b", 3)
    manager._prepare_mem_pool_batch({None, "a"})
    assert not manager.validate_batch({None, "b"})
    with pytest.raises(ValueError):
        manager._prepare_mem_pool_batch({None, "b"})


def test_cpu_snapshot_canonicalizes_lm_head_alias():
    adapter = Adapter("a", NS(block_size=4))
    adapter.initialize_weights_from_tensors(
        {"model.unembed_tokens.oft_R": torch.ones(1)}
    )
    assert adapter.streamed_named_tensors[0][0] == "model.lm_head.oft_R"


def test_nonresident_staged_activation_advances_version_before_native_update(runtime):
    manager, pool, _ = runtime
    ref = load(manager, "a", 2)
    pending_adapter = Adapter("a", NS(block_size=4))
    pending_adapter.initialize_weights_from_tensors({"weight": torch.tensor([5.0])})
    manager._pending_oft_stage = NS(
        uid="a",
        version=2,
        config=pending_adapter.config,
        adapter=pending_adapter,
        ref=replace(ref, adapter_version=2),
    )
    pool.discard_stage = Mock()
    activate = load_class(
        "oft/staged_manager.py", "StagedOFTManager", {"activate_adapter"}
    ).activate_adapter
    assert activate(manager, "a", 2, "a").success
    load(manager, "a", 7, version=3, upsert=True)
    manager._prepare_mem_pool_batch({"a"})
    assert pool.values[pool.uid_to_buffer_id["a"]].item() == 7


def test_failed_restoration_leaves_cpu_copy_available_for_retry(runtime):
    manager, pool, loader = runtime
    load(manager, "a", 2)
    commit = loader._commit_streamed_oft_tensor_groups
    loader._commit_streamed_oft_tensor_groups = lambda *args: (
        False,
        "injected failure",
    )
    with pytest.raises(ValueError, match="injected failure"):
        manager._prepare_mem_pool_batch({"a"})
    assert not pool.uid_to_buffer_id
    assert "a" in manager.adapters
    loader._commit_streamed_oft_tensor_groups = commit
    manager._prepare_mem_pool_batch({"a"})
    assert pool.values[pool.uid_to_buffer_id["a"]].item() == 2


def test_staged_snapshot_uses_same_writer_and_preserves_old_cpu_until_activation(
    runtime,
):
    manager, pool, _ = runtime
    stage_type = load_class(
        "oft/staged_manager.py",
        "StagedOFTManager",
        {"stage_adapter", "activate_adapter"},
        OFTConfig=Config,
        OFTAdapter=Adapter,
        OFTRef=Ref,
        PendingOFTStage=NS,
    )
    manager._pending_oft_stage = None
    pool.stage = lambda uid, version, weights, **kwargs: pool.values[
        pool.staging_idx
    ].zero_()
    pool.discard_stage = Mock()
    pool.activate = lambda uid, version, destination: pool.values[destination].copy_(
        pool.values[pool.staging_idx]
    )
    load(manager, "a", 2)
    manager._prepare_mem_pool_batch({None, "a"})
    old_cpu = manager.adapters["a"]
    payload = torch.tensor([5.0])
    result = stage_type.stage_adapter(
        manager, [("weight", payload)], {"oft_block_size": 4}, "a", 2, "a"
    )
    assert result.success, result.error_message
    assert manager.adapters["a"] is old_cpu
    assert pool.values[pool.uid_to_buffer_id["a"]].item() == 2
    payload.fill_(99)
    assert stage_type.activate_adapter(manager, "a", 2, "a").success
    assert pool.values[pool.uid_to_buffer_id["a"]].item() == 5
    load(manager, "b", 3)
    manager._prepare_mem_pool_batch({None, "b"})
    manager._prepare_mem_pool_batch({None, "a"})
    assert pool.values[pool.uid_to_buffer_id["a"]].item() == 5


def test_native_pins_cannot_reserve_every_serving_slot(runtime):
    manager, pool, _ = runtime
    first = Ref("a", "a", pinned=True)
    config = {"oft_block_size": 4}
    assert manager.load_adapter_from_tensors(first, {}, config).success
    result = manager.load_adapter_from_tensors(Ref("b", "b", pinned=True), {}, config)
    assert not result.success
    assert manager.num_pinned == 1
    assert "b" not in manager.refs
    assert manager.load_adapter_from_tensors(
        replace(first, adapter_version=2), {}, config, upsert=True
    ).success
    load(manager, "b", 3)
    result = manager.load_adapter_from_tensors(
        Ref("b", "b", pinned=True, adapter_version=2), {}, config, upsert=True
    )
    assert not result.success and result.previous_adapter_preserved
    assert not manager.refs["b"].pinned


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
