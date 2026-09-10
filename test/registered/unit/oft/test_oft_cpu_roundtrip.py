"""Numerical OFT slot restoration through both public worker loading paths."""

from dataclasses import replace

import pytest
import torch
from test_oft_staging_backend import _manager, _slot_snapshot

from sglang.srt.oft.oft_registry import OFTRef
from sglang.srt.oft.torch_ops.oft_ops import precompute_oft_r
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("load_path", ["native", "staged"])
def test_split_and_optional_weights_roundtrip_through_eviction(load_path):
    manager = _manager(
        max_ofts_per_batch=2,
        target_modules={"qkv_proj", "embed_tokens", "lm_head"},
    )
    manager.device = torch.device("cpu")
    pool = manager.memory_pool
    # Three LM-head blocks expose the identity tail when two are supplied.
    pool.lm_head_R_buffer["lm_head"] = torch.zeros(3, 3, 4, 4)
    config = {
        "peft_type": "oft",
        "oft_block_size": 4,
        "target_modules": ["q_proj", "embed_tokens", "lm_head"],
    }
    q = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 100
    head = q + 0.01
    embedding = q[:1] + 0.02
    payload = [
        ("model.layers.0.self_attn.q_proj.oft_R", q),
        ("model.unembed_tokens.oft_R", head),
        ("model.embed_tokens.oft_R", embedding),
    ]
    ref = OFTRef(
        adapter_id="a",
        adapter_name="a",
        adapter_path="__tensors__",
        adapter_version=1,
        reloadable=False,
    )
    if load_path == "native":
        result = manager.load_adapter_from_tensors(ref, payload, config)
    else:
        result = manager.stage_adapter(payload, config, "a", 1, "a")
        assert result.success, result.error_message
        result = manager.activate_adapter("a", 1, "a")
    assert result.success, result.error_message
    manager._prepare_mem_pool_batch({None, "a"})
    slot = pool.uid_to_buffer_id["a"]
    expected = _slot_snapshot(pool, slot)
    torch.testing.assert_close(
        pool.R_buffer["qkv_proj"][0][slot, :2], precompute_oft_r(q, 4)
    )
    torch.testing.assert_close(
        pool.R_buffer["qkv_proj"][0][slot, 2:], torch.eye(4).expand(4, 4, 4)
    )
    torch.testing.assert_close(
        pool.lm_head_R_buffer["lm_head"][slot, :2], precompute_oft_r(head, 4)
    )
    torch.testing.assert_close(pool.lm_head_R_buffer["lm_head"][slot, 2], torch.eye(4))
    torch.testing.assert_close(
        pool.embedding_R_buffer["embed_tokens"][slot],
        precompute_oft_r(embedding, 4).expand(2, 4, 4),
    )
    # Transport/client buffers can be reused without altering the CPU snapshot.
    for _, tensor in payload:
        tensor.fill_(99)
    other = replace(ref, adapter_id="b", adapter_name="b")
    assert manager.load_adapter_from_tensors(other, [], config).success
    manager._prepare_mem_pool_batch({None, "b"})
    assert "a" not in pool.uid_to_buffer_id
    assert "a" in manager.adapters and not manager.refs["a"].reloadable
    manager._prepare_mem_pool_batch({None, "a"})
    restored = _slot_snapshot(pool, pool.uid_to_buffer_id["a"])
    assert restored.keys() == expected.keys()
    for key in expected:
        assert torch.equal(restored[key], expected[key]), key


@pytest.mark.parametrize("load_path", ["native", "staged"])
def test_expert_aliases_and_tp_slices_roundtrip(load_path):
    from types import SimpleNamespace

    manager = _manager(max_ofts_per_batch=2)
    manager.device = torch.device("cpu")
    manager.oft_type = "canonical_oft"
    manager.oft_r_dtype = torch.float32
    manager._moe_modules = {
        0: SimpleNamespace(
            w13_weight=torch.zeros(1),
            num_local_experts=2,
            moe_tp_rank=1,
            moe_tp_size=2,
            _map_global_expert_id_to_local_expert_id=lambda uid: uid,
        )
    }
    pool = manager.memory_pool
    for family in ("w1_oft_r", "w3_oft_r", "w2_oft_r"):
        pool._groups[family] = {0: torch.zeros(3, 2, 2, 4, 4)}
    config = {"peft_type": "oft", "oft_block_size": 4, "target_modules": ["q_proj"]}
    gate = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 100
    up = gate + 0.01
    down = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 100
    payload = [
        (f"model.layers.0.ffn.experts.0.{proj}.oft_R", value)
        for proj, value in (("w1", gate), ("w3", up), ("w2", down))
    ]
    ref = OFTRef(
        adapter_id="a",
        adapter_name="a",
        adapter_path="__tensors__",
        adapter_version=1,
        reloadable=False,
    )
    if load_path == "native":
        result = manager.load_adapter_from_tensors(ref, payload, config)
    else:
        result = manager.stage_adapter(payload, config, "a", 1, "a")
        assert result.success, result.error_message
        result = manager.activate_adapter("a", 1, "a")
    assert result.success, result.error_message
    manager._prepare_mem_pool_batch({None, "a"})
    slot = pool.uid_to_buffer_id["a"]
    expected = _slot_snapshot(pool, slot)
    for family, value in (("w1_oft_r", gate), ("w3_oft_r", up), ("w2_oft_r", down[2:])):
        tensor = pool.slot(family, 0, slot)
        torch.testing.assert_close(tensor[0], precompute_oft_r(value, 4))
        torch.testing.assert_close(tensor[1], torch.eye(4).expand(2, 4, 4))
    assert manager.load_adapter_from_tensors(
        replace(ref, adapter_id="b", adapter_name="b"), [], config
    ).success
    manager._prepare_mem_pool_batch({None, "b"})
    manager._prepare_mem_pool_batch({None, "a"})
    restored = _slot_snapshot(pool, pool.uid_to_buffer_id["a"])
    for key in expected:
        assert torch.equal(restored[key], expected[key]), key


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
