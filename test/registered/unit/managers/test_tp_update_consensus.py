"""Unit tests for TP-wide control-plane update consensus."""

import unittest
from unittest.mock import Mock

from sglang.srt.managers.scheduler_components.tp_update_consensus import (
    gather_tp_update_result,
    run_tp_adapter_activation,
    run_tp_adapter_update,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeDistributed:
    def __init__(self, remote_results):
        self.remote_results = remote_results

    def get_world_size(self, *, group):
        return len(self.remote_results) + 1

    def all_gather_object(self, outputs, local_result, *, group):
        outputs[:] = [local_result, *self.remote_results]


class TestTpUpdateConsensus(CustomTestCase):
    def test_remote_failure_fails_the_whole_tp_group(self):
        """A failure hidden on a non-sender TP rank must fail the request."""
        distributed = _FakeDistributed([(False, "rank-local staging failed", None)])

        success, message, version = gather_tp_update_result(
            distributed=distributed,
            group=object(),
            success=True,
            message="rank 0 staged",
            version="4",
        )

        self.assertFalse(success)
        self.assertIn("TP rank 1", message)
        self.assertIn("rank-local staging failed", message)
        self.assertIsNone(version)

    def test_version_disagreement_fails_the_whole_tp_group(self):
        """Ranks activating different versions must not report success."""
        distributed = _FakeDistributed([(True, "rank 1 activated", "5")])

        success, message, version = gather_tp_update_result(
            distributed=distributed,
            group=object(),
            success=True,
            message="rank 0 activated",
            version="4",
        )

        self.assertFalse(success)
        self.assertIn("TP ranks reported different versions", message)
        self.assertIn("rank 0='4'", message)
        self.assertIn("rank 1='5'", message)
        self.assertIsNone(version)

    def test_remote_stage_failure_prevents_activation(self):
        """Partial staging must stop every rank before any activation begins."""
        distributed = _FakeDistributed([(False, "rank-local staging failed", None)])
        activate = Mock(return_value=(True, "activated"))

        success, _, staged_version, active_version = run_tp_adapter_update(
            distributed=distributed,
            group=object(),
            version="4",
            activate_immediately=True,
            stage=lambda: (True, "rank 0 staged"),
            activate=activate,
        )

        self.assertFalse(success)
        self.assertIsNone(staged_version)
        self.assertIsNone(active_version)
        activate.assert_not_called()

    def test_local_stage_exception_is_reported_before_activation(self):
        """A handled local exception must join consensus instead of stranding peers."""
        distributed = _FakeDistributed([(True, "rank 1 staged", "4")])
        activate = Mock(return_value=(True, "activated"))

        success, message, staged_version, active_version = run_tp_adapter_update(
            distributed=distributed,
            group=object(),
            version="4",
            activate_immediately=True,
            stage=Mock(side_effect=RuntimeError("rank 0 staging exception")),
            activate=activate,
        )

        self.assertFalse(success)
        self.assertIn("TP rank 0", message)
        self.assertIn("rank 0 staging exception", message)
        self.assertIsNone(staged_version)
        self.assertIsNone(active_version)
        activate.assert_not_called()

    def test_unanimous_stage_and_activation_succeed(self):
        """Consensus must not turn an all-rank success into a failure."""
        distributed = _FakeDistributed([(True, "rank 1 succeeded", "4")])
        activate = Mock(return_value=(True, "rank 0 activated"))

        success, _, staged_version, active_version = run_tp_adapter_update(
            distributed=distributed,
            group=object(),
            version="4",
            activate_immediately=True,
            stage=lambda: (True, "rank 0 staged"),
            activate=activate,
        )

        self.assertTrue(success)
        self.assertEqual(staged_version, "4")
        self.assertEqual(active_version, "4")
        activate.assert_called_once_with()

    def test_remote_activation_failure_fails_the_whole_tp_group(self):
        """A failure after partial activation must reach the tokenizer."""
        distributed = _FakeDistributed([(False, "rank-local activation failed", None)])

        success, message, active_version = run_tp_adapter_activation(
            distributed=distributed,
            group=object(),
            version="4",
            activate=lambda: (True, "rank 0 activated"),
        )

        self.assertFalse(success)
        self.assertIn("TP rank 1", message)
        self.assertIn("rank-local activation failed", message)
        self.assertIsNone(active_version)


if __name__ == "__main__":
    unittest.main()
