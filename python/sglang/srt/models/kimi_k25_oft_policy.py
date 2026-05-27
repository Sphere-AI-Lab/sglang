from collections.abc import Iterable


# NOTE: "qkv_proj" is allowed but a no-op for Kimi K2.5 -- the text model uses
# MLA, so no module name ends in ".qkv_proj" and
# is_kimi_dense_first_oft_module never matches. We accept it so verl-style
# rollouts that pass "qkv_proj" do not error out.
#
# The first MLA q/kv projections are intentionally not allowed here. SGLang
# fuses them into one ReplicatedLinear in the Kimi serving graph, while Megatron
# trains separate OFT rotations for those two projections. Until SGLang has
# dedicated per-branch OFT support for that fused module, accepting those target
# suffixes would leave the rollout path on a different adapter surface.
KIMI_DENSE_FIRST_OFT_ALLOWED_SUFFIXES = frozenset(
    {
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "qkv_proj",
        "q_b_proj",
        "kv_b_proj",
    }
)

KIMI_DENSE_FIRST_OFT_UNSUPPORTED_FUSED_INPUT_SUFFIXES = frozenset(
    {
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
