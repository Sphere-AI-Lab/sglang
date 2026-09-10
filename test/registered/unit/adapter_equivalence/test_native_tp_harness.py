"""CPU checks of the actual GPU TP harness's launch and identity helpers."""

import ast
import json
import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))
from adapter_equivalence.server import (
    AdapterIdentity,
    ServerSpec,
    normalize_control_result,
)

SOURCE = Path(__file__).resolve().parents[2] / "lora/test_lora_staged_update_tp.py"


def helper(name, namespace):
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "NativeTPHarness"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    # Engine is beyond the launch-validation boundary; no GPU runtime import.
    method.body = [n for n in method.body if not isinstance(n, ast.ImportFrom)]
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace[name]


def test_dynamic_lora_shape_is_declared_before_engine_kwargs(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 8, "target_modules": ["q_proj", "lm_head"]})
    )

    class LaunchReached(Exception):
        pass

    def engine_kwargs(spec):
        assert spec.max_lora_rank == 8
        assert spec.lora_target_modules == ("all",)
        raise LaunchReached

    init = helper(
        "__init__",
        dict(
            Path=Path,
            json=json,
            ServerSpec=ServerSpec,
            MODEL_PATH="test/model",
            engine_kwargs=engine_kwargs,
        ),
    )
    with pytest.raises(LaunchReached):
        init(SimpleNamespace(), None, "native_lora", tmp_path, tmp_path / "fault.json")


@pytest.mark.parametrize(
    "state,expected",
    [
        (
            {
                "active": None,
                "staged": {"name": "policy-a", "id": "allocated-stage"},
                "registered": [],
            },
            "allocated-stage",
        ),
        (
            {
                "active": None,
                "staged": None,
                "registered": [{"name": "policy-a", "id": "allocated-registry"}],
            },
            "allocated-registry",
        ),
        (
            {
                "active": {"name": "policy-a", "id": "allocated-active"},
                "staged": None,
                "registered": [],
            },
            "allocated-active",
        ),
        (
            {
                "active": {"name": "other", "id": "unrelated"},
                "staged": None,
                "registered": [],
            },
            None,
        ),
        ({"active": None, "staged": None, "registered": []}, None),
    ],
)
def test_identity_uses_allocated_policy_id_or_requests_initial_allocation(
    state, expected
):
    identity = helper("identity", dict(AdapterIdentity=AdapterIdentity))
    harness = SimpleNamespace(control=SimpleNamespace(inspect_state=lambda: state))
    result = identity(harness, 3)
    assert result == AdapterIdentity("policy-a", expected, "3")


def _activation_harness(published_state, *, success=True):
    requested = AdapterIdentity("policy-a", "allocated-stage", "3")

    class Control:
        def __init__(self):
            self.state = {
                "active": None,
                "staged": {
                    "name": requested.name,
                    "id": requested.adapter_id,
                    "version": requested.version,
                },
                "registered": [],
            }
            self.calls = []

        def inspect_state(self):
            return self.state

        def activate(self, identity):
            self.calls.append(identity)
            self.state = published_state
            # This is the actual tokenizer API return contract, normalized
            # by the same helper used by NativeLoRAControl/NativeOFTControl.
            return normalize_control_result((success, "activation result"))

    harness = SimpleNamespace(testcase=unittest.TestCase(), control=Control())
    harness.identity = MethodType(
        helper("identity", dict(AdapterIdentity=AdapterIdentity)), harness
    )

    def require_success(result):
        harness.testcase.assertTrue(result.success, result)
        return result

    harness.success = require_success
    return harness, requested


def _published_state():
    return {
        "active": {"name": "policy-a", "id": "allocated-stage", "version": "3"},
        "staged": None,
        "registered": [],
    }


def test_activate_verifies_publication_with_tuple_return_contract():
    harness, requested = _activation_harness(_published_state())

    helper("activate", {})(harness, 3)

    assert harness.control.calls == [requested]


@pytest.mark.parametrize(
    "defect",
    ["missing-active", "wrong-name", "wrong-id", "wrong-version", "uncleared-stage"],
)
def test_activate_rejects_incorrect_published_state(defect):
    state = _published_state()
    if defect == "missing-active":
        state["active"] = None
    elif defect == "uncleared-stage":
        state["staged"] = dict(state["active"])
    else:
        field = {"wrong-name": "name", "wrong-id": "id", "wrong-version": "version"}[
            defect
        ]
        state["active"][field] = "unexpected"
    harness, requested = _activation_harness(state)

    with pytest.raises(AssertionError):
        helper("activate", {})(harness, 3)

    assert harness.control.calls == [requested]


def test_activate_rejects_failed_tuple_even_with_matching_published_state():
    harness, requested = _activation_harness(_published_state(), success=False)

    with pytest.raises(AssertionError):
        helper("activate", {})(harness, 3)

    assert harness.control.calls == [requested]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
