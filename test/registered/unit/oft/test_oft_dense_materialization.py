# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");

"""Focused CPU tests for single-adapter OFT dense materialization."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.managers.io_struct import OFTUpdateOutput
from sglang.srt.oft import layers as oft_layers
from sglang.srt.oft import oft_manager as manager_module
from sglang.srt.oft.backend.triton_backend import TritonOFTBackend
from sglang.srt.oft.oft_manager import OFTManager
from sglang.srt.oft.oft_registry import OFTRef
from sglang.srt.oft.staged_manager import StagedOFTManager
from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _orthogonal_blocks(num_blocks: int, block_size: int) -> torch.Tensor:
    return torch.stack(
        [torch.linalg.qr(torch.randn(block_size, block_size))[0] for _ in range(num_blocks)]
    )


def _rotate(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "...nb,nbc->...nc", x.reshape(*x.shape[:-1], r.shape[0], r.shape[-1]), r
    ).reshape_as(x)


def _batch_manager(materialized_id):
    manager = OFTManager.__new__(OFTManager)
    manager.dense_oft_materialization = True
    manager._dense_materialized_oft_id = materialized_id
    manager.max_ofts_per_batch = 2
    slots = {None: 0, "orbit_oft": 1}
    manager.memory_pool = SimpleNamespace(
        uid_to_buffer_id=slots, get_buffer_id=slots.__getitem__, active_idx=1
    )
    manager.adapters = {}
    manager.configs = {"orbit_oft": SimpleNamespace(block_size=4)}
    manager._moe_modules = {}
    manager.oft_backend = Mock()
    return manager


class TestDenseOFTWeightMaterialization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_single_rotation_matches_runtime_input_rotation(self):
        x = torch.randn(7, 8)
        weight = torch.randn(11, 8)
        r = _orthogonal_blocks(num_blocks=2, block_size=4)

        expected = F.linear(_rotate(x, r), weight)
        materialized = oft_layers.materialize_dense_oft_weight(weight, r)
        actual = F.linear(x, materialized)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_split_rotations_match_qkv_style_runtime_projection(self):
        x = torch.randn(5, 8)
        output_sizes = [9, 3, 3]
        weight = torch.randn(sum(output_sizes), 8)
        stacked_r = torch.cat(
            [_orthogonal_blocks(num_blocks=2, block_size=4) for _ in output_sizes]
        )

        expected = torch.cat(
            [
                F.linear(_rotate(x, r), weight_slice)
                for r, weight_slice in zip(
                    torch.chunk(stacked_r, len(output_sizes), dim=0),
                    torch.split(weight, output_sizes, dim=0),
                )
            ],
            dim=-1,
        )
        materialized = oft_layers.materialize_dense_oft_weight(
            weight, stacked_r, output_sizes=output_sizes
        )

        torch.testing.assert_close(
            F.linear(x, materialized), expected, atol=1e-5, rtol=1e-5
        )

    def test_destination_copy_preserves_captured_weight_pointer(self):
        weight = torch.randn(11, 8)
        original = weight.clone()
        pointer = weight.data_ptr()
        r = _orthogonal_blocks(num_blocks=2, block_size=4)

        oft_layers.materialize_dense_oft_weight(original, r, destination=weight)

        self.assertEqual(weight.data_ptr(), pointer)
        torch.testing.assert_close(
            weight, oft_layers.materialize_dense_oft_weight(original, r)
        )


class TestDenseOFTManagerActivation(unittest.TestCase):
    def test_sibling_activation_materializes_after_promoting_staging_slot(self):
        events = []
        manager = SimpleNamespace(
            dense_oft_materialization=True,
            memory_pool=SimpleNamespace(
                active_idx=1,
                activate=lambda version: events.append(("activate", version)),
            ),
            _materialize_dense_oft_slot=lambda slot: events.append(("materialize", slot)),
            _bump_ref_version=lambda name, version: events.append(("bump", name, version)),
        )

        result = OFTManager.activate_adapter(manager, "orbit_oft", 7)

        self.assertIsNone(result)  # Preserve this sibling API's native result type.
        self.assertEqual(
            events,
            [("activate", 7), ("materialize", 1), ("bump", "orbit_oft", 7)],
        )

    def test_prepare_rejects_base_request_after_materialization(self):
        manager = _batch_manager("orbit_oft")
        batch = SimpleNamespace(oft_ids=[None], batch_size=1)

        with self.assertRaisesRegex(RuntimeError, "only serves adapter 'orbit_oft'"):
            manager.prepare_oft_batch(batch)

    def test_prepare_skips_runtime_slot_metadata_after_materialization(self):
        manager = _batch_manager("orbit_oft")
        batch = SimpleNamespace(oft_ids=["orbit_oft"] * 4, batch_size=4)

        manager.prepare_oft_batch(batch)

        manager.oft_backend.prepare_oft_batch.assert_not_called()

    def test_prepare_skips_runtime_slot_metadata_before_first_activation(self):
        manager = _batch_manager(None)
        batch = SimpleNamespace(oft_ids=[None] * 4, batch_size=4)

        manager.prepare_oft_batch(batch)

        manager.oft_backend.prepare_oft_batch.assert_not_called()


def _linear_wrapper(kind, *, dtype=torch.bfloat16, tp_size=1):
    kwargs = dict(params_dtype=dtype, tp_rank=0, tp_size=tp_size, bias=False)
    if kind == "column":
        base = ColumnParallelLinear(8, 12, **kwargs)
        wrapper_cls = oft_layers.ColumnParallelLinearWithOFT
    elif kind == "merged":
        base = MergedColumnParallelLinear(8, [12, 12], **kwargs)
        wrapper_cls = oft_layers.MergedColumnParallelLinearWithOFT
    elif kind == "qkv":
        base = QKVParallelLinear(8, 4, 4, total_num_kv_heads=1, **kwargs)
        wrapper_cls = oft_layers.QKVParallelLinearWithOFT
    else:
        base = RowParallelLinear(8 * tp_size, 12, reduce_results=False, **kwargs)
        wrapper_cls = oft_layers.RowParallelLinearWithOFT
    with torch.no_grad():
        base.weight.normal_(std=0.2)
    backend = TritonOFTBackend(max_ofts_per_batch=2, device=torch.device("cpu"))
    return wrapper_cls(base, backend)


@pytest.mark.parametrize("kind", ["column", "merged", "qkv", "row"])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_bf16_wrapper_repeated_fold_uses_frozen_base_and_stable_live_pointer(kind, tp_size):
    torch.manual_seed(2026)
    module = _linear_wrapper(kind, tp_size=tp_size)
    original = module.base_layer.weight.detach().clone()
    live_weight = module.base_layer.weight
    pointer = live_weight.data_ptr()
    if kind in ("merged", "qkv"):
        output_sizes = [size // tp_size for size in module.base_layer.output_sizes]
    else:
        output_sizes = [original.shape[0]]
    rotations = [
        torch.cat([_orthogonal_blocks(2, 4) for _ in output_sizes]).bfloat16()
        for _ in range(2)
    ]
    module.set_oft_info(torch.stack(rotations))
    assert module.oft_active  # The default path still applies runtime rotations.
    module.enable_dense_materialization()
    assert not module.oft_active
    x = torch.randn(5, 8).bfloat16()

    for rotation in rotations:
        module.materialize_dense_weight(rotation)
        actual, _ = module(x)
        expected = torch.cat(
            [
                F.linear(_rotate(x.float(), r.float()), weight.float())
                for r, weight in zip(
                    torch.chunk(rotation, len(output_sizes), dim=0),
                    torch.split(original, output_sizes, dim=0),
                )
            ],
            dim=-1,
        )
        torch.testing.assert_close(actual.float(), expected, atol=0.015, rtol=0.015)
        assert module.base_layer.weight is live_weight
        assert live_weight.data_ptr() == pointer
        torch.testing.assert_close(module._oft_dense_base_weight, original, rtol=0, atol=0)


CONFIG = {"peft_type": "OFT", "target_modules": ["q_proj"], "oft_block_size": 4}
UID = "server-generated-oft-id"
NAME = "orbit_oft"


def _raw_weights(value):
    return [("model.layers.0.self_attn.q_proj.oft_R", torch.full((2, 6), value).bfloat16())]


def _dense_manager(*, dense=True, configure=True, double_buffer=True):
    """Real CPU wrapper, backend, pool and update methods; bypass model boot."""
    torch.manual_seed(2026)
    module = _linear_wrapper("column")
    manager = StagedOFTManager.__new__(StagedOFTManager)
    manager.base_model = torch.nn.ModuleList([module])
    manager.base_hf_config = SimpleNamespace(num_hidden_layers=1, hidden_size=8)
    manager.load_config = LoadConfig()
    manager.oft_backend = module.oft_backend
    manager.oft_modules = [{"model.layers.0.self_attn.q_proj": module}]
    manager.target_modules = {"q_proj"}
    manager.max_ofts_per_batch = manager.max_adapters_per_batch = 2
    manager.dtype = manager.oft_r_dtype = torch.bfloat16
    manager.tp_size = 1
    manager.tp_rank = 0
    manager.max_oft_block_size = 4
    manager.oft_type = "canonical_oft"
    manager.oft_double_buffer = double_buffer
    manager.eviction_policy = "lru"
    manager.oft_added_tokens_size = 0
    manager.memory_saver_adapter = None
    manager.memory_saver_cpu_backup = False
    manager.embed_tokens_module = manager.lm_head_module = None
    manager.configs = {}
    manager.adapters = {}
    manager.oft_refs = {}
    manager.num_pinned = 0
    manager._moe_modules = {}
    manager._pending_oft_stage = None
    manager.dense_oft_materialization = dense
    manager._dense_materialized_oft_id = None
    manager.init_memory_pool()
    manager.update_oft_info()
    if configure:
        with patch.dict(os.environ, {"SGLANG_OFT_REQUIRE_UNIFORM_BATCH": "1"}):
            manager._configure_dense_oft_materialization(n_expert_wrapped=0)
    return manager, module


def _assert_active_fold(manager, module, original, payload, version, pointer):
    slot = manager.memory_pool.uid_to_buffer_id[UID]
    assert slot != manager.memory_pool.staging_idx
    assert manager.memory_pool.buffer_id_to_uid[slot] == UID
    assert manager._dense_materialized_oft_id == UID
    assert manager.oft_refs[UID].version == version
    assert module.base_layer.weight.data_ptr() == pointer
    rotation = precompute_oft_r(payload[0][1], 4)
    x = torch.randn(5, 8).bfloat16()
    actual, _ = module(x)
    expected = F.linear(_rotate(x.float(), rotation.float()), original.float())
    torch.testing.assert_close(actual.float(), expected, atol=0.015, rtol=0.015)
    torch.testing.assert_close(module._oft_dense_base_weight, original, rtol=0, atol=0)


def test_staged_first_and_repeated_activation_fold_during_paused_update():
    manager, module = _dense_manager()
    original = module.base_layer.weight.detach().clone()
    pointer = module.base_layer.weight.data_ptr()
    previous = original.clone()
    for version, value in [(1, 0.025), (2, -0.04)]:
        payload = _raw_weights(value)
        staged = manager.stage_adapter(payload, CONFIG, NAME, version, oft_id=UID)
        assert isinstance(staged, OFTUpdateOutput)
        assert staged.success, staged.error_message
        torch.testing.assert_close(module.base_layer.weight, previous, atol=0, rtol=0)

        activated = manager.activate_adapter(NAME, version, oft_id=UID)

        assert isinstance(activated, OFTUpdateOutput)
        assert activated.success, activated.error_message
        assert activated.loaded_adapters == {NAME: "__distributed__"}
        assert manager.memory_pool.active_version_for(UID) == version
        assert manager.memory_pool.staged_identity() is None
        assert manager._pending_oft_stage is None
        assert UID in manager.adapters and UID in manager.configs
        _assert_active_fold(manager, module, original, payload, version, pointer)
        previous = module.base_layer.weight.detach().clone()


def test_direct_first_and_repeated_upsert_fold_before_rpc_returns():
    manager, module = _dense_manager(double_buffer=False)
    original = module.base_layer.weight.detach().clone()
    pointer = module.base_layer.weight.data_ptr()
    for version, value in [(1, 0.025), (2, -0.04)]:
        ref = OFTRef(UID, NAME, "__distributed__", pinned=False, reloadable=False, version=version)
        payload = _raw_weights(value)

        result = manager.load_adapter_from_tensors(ref, dict(payload), CONFIG, upsert=version > 1)

        assert isinstance(result, OFTUpdateOutput)
        assert result.success, result.error_message
        assert result.loaded_adapters == {NAME: "__distributed__"}
        _assert_active_fold(manager, module, original, payload, version, pointer)


@pytest.mark.parametrize("update_mode", ["direct", "staged"])
def test_second_logical_adapter_rejected_before_live_weight_or_slot_changes(update_mode):
    manager, module = _dense_manager()
    payload = _raw_weights(0.025)
    first = OFTRef(UID, NAME, "__distributed__", pinned=False, reloadable=False, version=1)
    assert manager.load_adapter_from_tensors(first, payload, CONFIG).success
    live_before = module.base_layer.weight.detach().clone()
    slot_before = manager.memory_pool.R_buffer["q_proj"][0].clone()

    if update_mode == "direct":
        second = OFTRef("other-id", "other", "__distributed__", pinned=False, reloadable=False, version=2)
        result = manager.load_adapter_from_tensors(second, _raw_weights(-0.04), CONFIG)
    else:
        result = manager.stage_adapter(_raw_weights(-0.04), CONFIG, "other", 2, oft_id="other-id")

    assert not result.success
    assert "single-adapter" in result.error_message
    assert set(manager.oft_refs) == {UID}
    assert manager.memory_pool.staged_identity() is None
    torch.testing.assert_close(module.base_layer.weight, live_before, atol=0, rtol=0)
    torch.testing.assert_close(manager.memory_pool.R_buffer["q_proj"][0], slot_before, atol=0, rtol=0)


def test_dense_materialization_defaults_off(monkeypatch):
    monkeypatch.delenv("SGLANG_OFT_MATERIALIZE_DENSE", raising=False)
    assert not manager_module._dense_oft_materialization_requested()
    monkeypatch.setenv("SGLANG_OFT_MATERIALIZE_DENSE", "1")
    assert manager_module._dense_oft_materialization_requested()


def test_dense_materialization_requires_explicit_uniform_batch_contract(monkeypatch):
    manager, _ = _dense_manager(configure=False)
    monkeypatch.delenv("SGLANG_OFT_REQUIRE_UNIFORM_BATCH", raising=False)
    with pytest.raises(ValueError, match="SGLANG_OFT_REQUIRE_UNIFORM_BATCH=1"):
        manager._configure_dense_oft_materialization(n_expert_wrapped=0)


@pytest.mark.parametrize("unsupported", ["moe", "embedding", "quantized", "dtype"])
def test_dense_materialization_retains_supported_target_restrictions(unsupported, monkeypatch):
    manager, module = _dense_manager(configure=False)
    monkeypatch.setenv("SGLANG_OFT_REQUIRE_UNIFORM_BATCH", "1")
    if unsupported == "embedding":
        manager.embed_tokens_module = torch.nn.Embedding(4, 8)
    elif unsupported == "quantized":
        module.base_layer.quant_method = object()
    elif unsupported == "dtype":
        module.R_buffer = module.R_buffer.float()
    with pytest.raises(ValueError):
        manager._configure_dense_oft_materialization(n_expert_wrapped=int(unsupported == "moe"))


def test_dense_graph_replay_validates_real_requests_without_adapter_padding():
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner

    manager = _batch_manager("orbit_oft")
    runner = SimpleNamespace(
        model_runner=SimpleNamespace(
            server_args=SimpleNamespace(enable_oft=True), oft_manager=manager
        )
    )
    ids = ["orbit_oft"] * 3
    batch = SimpleNamespace(oft_ids=ids, batch_size=3)

    DecodeCudaGraphRunner._prepare_oft_replay_batch(runner, batch, bs=4, raw_bs=3)

    assert batch.oft_ids is ids
    assert batch.batch_size == 3
    manager.oft_backend.prepare_oft_batch.assert_not_called()
    batch.oft_ids = ["orbit_oft", None, "orbit_oft"]
    with pytest.raises(RuntimeError, match="only serves adapter 'orbit_oft'"):
        DecodeCudaGraphRunner._prepare_oft_replay_batch(runner, batch, bs=4, raw_bs=3)


if __name__ == "__main__":
    unittest.main()
