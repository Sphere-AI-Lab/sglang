"""Failed concurrent-output checks retain exact evidence without relaxing equality."""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))

from adapter_equivalence.scenarios import (
    ScenarioContractError,
    validate_lifecycle_observations,
)
from adapter_equivalence.schema import Observation
from test_harness_contract import complete_native_observations

STREAM = "concurrent.stream"
NON_STREAM = "concurrent.non-stream"
MESSAGE = "concurrent output changed between stream modes"


def _observations():
    observations = complete_native_observations("native_oft")
    batch = replace(
        observations[STREAM],
        output_ids=(101, 102, 103),
        text='["first", "second"]',
        token_logprobs=(-0.25, -0.5, -0.75),
        selected_logits={
            f"decode.{index:03d}.top_logprobs": (-0.5, -1.0) for index in range(3)
        },
        selected_token_ids={
            f"decode.{index:03d}.top_logprobs": (10, 20) for index in range(3)
        },
        request_output_lengths=(1, 2),
        request_texts=("first", "second"),
    )
    batch.validate()
    observations[STREAM] = observations[NON_STREAM] = batch
    return observations


def _changed(observation, field):
    if field == "selected_logits":
        values = dict(observation.selected_logits)
        values["decode.001.top_logprobs"] = (math.nextafter(-0.5, 0.0), -1.0)
    elif field == "selected_token_ids":
        values = dict(observation.selected_token_ids)
        values["decode.000.top_logprobs"] = (11, 20)
    else:
        values = {
            "output_ids": (999, 102, 103),
            "text": '["changed", "second"]',
            "token_logprobs": (math.nextafter(-0.25, 0.0), -0.5, -0.75),
            "request_output_lengths": (2, 1),
            "request_texts": ("changed", "second"),
        }[field]
    result = replace(observation, **{field: values})
    result.validate()
    return result


@pytest.mark.parametrize(
    "field",
    (
        "output_ids",
        "text",
        "token_logprobs",
        "selected_logits",
        "selected_token_ids",
        "request_output_lengths",
        "request_texts",
    ),
)
def test_each_concurrent_difference_keeps_complete_payloads_and_exact_verdict(field):
    observations = _observations()
    observations[NON_STREAM] = _changed(observations[NON_STREAM], field)
    before = {name: value.to_dict() for name, value in observations.items()}

    with pytest.raises(ScenarioContractError, match=MESSAGE) as failure:
        validate_lifecycle_observations(observations)

    assert str(failure.value) == MESSAGE
    details = failure.value.details
    assert details["kind"] == "concurrent-output-mismatch"
    assert details["transitions"] == [STREAM, NON_STREAM]
    assert details["changed_fields"] == [field]
    assert details["observations"] == {
        STREAM: before[STREAM],
        NON_STREAM: before[NON_STREAM],
    }
    # Standard failure-artifact JSON must retain every float, row, request
    # boundary and runtime identity, including one-ULP score differences.
    decoded = json.loads(json.dumps(details, allow_nan=False))
    assert decoded == details
    for name in (STREAM, NON_STREAM):
        assert (
            Observation.from_dict(decoded["observations"][name]) == observations[name]
        )
    details["observations"][STREAM]["selected_logits"]["decode.000.top_logprobs"][
        0
    ] = 0.0
    details["observations"][STREAM]["adapter_state"]["active"]["id"] = "mutated"
    assert {name: value.to_dict() for name, value in observations.items()} == before


def test_concurrent_details_report_all_different_fields():
    observations = _observations()
    for field in ("output_ids", "token_logprobs", "selected_token_ids"):
        observations[NON_STREAM] = _changed(observations[NON_STREAM], field)
    with pytest.raises(ScenarioContractError, match=MESSAGE) as failure:
        validate_lifecycle_observations(observations)
    assert failure.value.details["changed_fields"] == [
        "output_ids",
        "token_logprobs",
        "selected_token_ids",
    ]


def test_equal_concurrent_outputs_still_pass():
    validate_lifecycle_observations(_observations())


def test_unrelated_contract_errors_preserve_message_without_output_details():
    failure = ScenarioContractError("existing contract failure")
    assert str(failure) == "existing contract failure"
    assert failure.args == ("existing contract failure",)
    assert failure.details is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
