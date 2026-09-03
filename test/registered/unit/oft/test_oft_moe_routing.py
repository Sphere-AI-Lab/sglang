import torch

from sglang.srt.oft.backend.base_backend import _compute_moe_oft_info


def test_builds_moe_mapping_for_base_and_two_adapters():
    info = _compute_moe_oft_info(
        num_tokens=5,
        seg_indptr=torch.tensor([0, 2, 3, 5], dtype=torch.int32),
        weight_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_ofts=3,
    )

    assert info.adapter_enabled.tolist() == [0, 1, 1]
    assert info.token_oft_mapping.tolist() == [0, 0, 1, 2, 2]
    assert info.num_tokens == 5


def test_base_only_batch_has_no_active_oft():
    info = _compute_moe_oft_info(
        num_tokens=3,
        seg_indptr=torch.tensor([0, 3], dtype=torch.int32),
        weight_indices=torch.tensor([0], dtype=torch.int32),
        max_ofts=3,
    )

    assert info.adapter_enabled.tolist() == [0, 0, 0]
    assert info.token_oft_mapping.tolist() == [0, 0, 0]
