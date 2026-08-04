from collections.abc import Iterable


# NOTE: "qkv_proj" is allowed but a no-op for Kimi K2.5 -- the text model uses
# MLA, so no module name ends in ".qkv_proj" and
# is_kimi_dense_first_oft_module never matches. We accept it so verl-style
# rollouts that pass "qkv_proj" do not error out.
#
# The first MLA q/kv projections (q_a_proj, kv_a_proj_with_mqa) ARE allowed
# here. SGLang fuses them into one ReplicatedLinear
# (fused_qkv_a_proj_with_mqa) in the Kimi serving graph, while Megatron trains
# separate OFT rotations for those two projections. ReplicatedLinearWithOFT
# now applies those per-branch rotations to the fused module's split slices
# (merged single-R or split stacked-R, matching how the adapter was trained),
# so both the fused name and its split constituents are accepted as target
# modules.
KIMI_DENSE_FIRST_OFT_ALLOWED_SUFFIXES = frozenset(
    {
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "qkv_proj",
        "q_b_proj",
        "kv_b_proj",
        "q_a_proj",
        "kv_a_proj_with_mqa",
        "fused_qkv_a_proj_with_mqa",
    }
)


def is_kimi_dense_first_oft_module(module_name: str) -> bool:
    suffix = module_name.rsplit(".", 1)[-1]
    if suffix not in KIMI_DENSE_FIRST_OFT_ALLOWED_SUFFIXES:
        return False
    if ".experts." in module_name and ".shared_experts." not in module_name:
        return False
    return True


def get_kimi_dense_first_unsupported_targets(
    target_modules: Iterable[str],
) -> set[str]:
    return set(target_modules) - KIMI_DENSE_FIRST_OFT_ALLOWED_SUFFIXES
