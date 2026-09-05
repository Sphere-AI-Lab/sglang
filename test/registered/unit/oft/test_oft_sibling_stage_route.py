"""The distributed stage/activate routes for the sibling (double-buffer) OFT
manager must see a result object, not ``None``.

weight_updater.stage_adapter / activate_adapter_version read ``.success`` off
whatever the manager returns. StagedOFTManager returns an OFTUpdateOutput; the
sibling OFTManager used to return None, so every ``/update_adapter_from_distributed``
and ``/activate_adapter_version`` against a ``--oft-impl sibling --oft-double-buffer``
engine failed with "'NoneType' object has no attribute 'success'" (HTTP 400).
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.oft.oft_manager import OFTManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sibling_manager():
    """A bare OFTManager with only the state stage_adapter/activate_adapter touch."""
    manager = OFTManager.__new__(OFTManager)
    manager.oft_refs = {}
    manager.memory_pool = SimpleNamespace(activate=mock.Mock())
    manager._stage_fill = mock.Mock()
    manager._bump_ref_version = mock.Mock()
    manager._make_streamed_ref = mock.Mock(
        side_effect=lambda name, version, oft_id, config: SimpleNamespace(
            oft_id=oft_id or name, oft_name=name, oft_path=name
        )
    )
    return manager


def _updater(manager):
    """The updater state the routes touch, for a sibling double-buffer engine.
    WeightUpdater is a frozen dataclass, so the routes are called unbound."""
    return SimpleNamespace(
        _model_update_group={"g": object()},
        device="cpu",
        get_model_runner=lambda: SimpleNamespace(
            server_args=SimpleNamespace(
                enable_lora_staging=False, oft_impl="sibling", enable_oft=True
            ),
            oft_manager=manager,
        ),
    )


class TestSiblingManagerReturnsUpdateResults(unittest.TestCase):
    def test_stage_adapter_reports_success(self):
        manager = _sibling_manager()
        result = manager.stage_adapter([], {"oft_block_size": 32}, "orbit_oft", 1)
        self.assertTrue(result.success)
        self.assertIn("orbit_oft", result.loaded_adapters)

    def test_activate_adapter_reports_success(self):
        manager = _sibling_manager()
        manager.stage_adapter([], {"oft_block_size": 32}, "orbit_oft", 1)
        result = manager.activate_adapter("orbit_oft", 1)
        self.assertTrue(result.success)
        manager.memory_pool.activate.assert_called_once_with(1)
        manager._bump_ref_version.assert_called_once_with("orbit_oft", 1)


class TestSiblingDistributedRoutes(unittest.TestCase):
    """Drive the weight updater's sibling routes with the real manager methods.
    No tensors are broadcast (``names=[]``), so no process group is needed."""

    def test_stage_route_succeeds(self):
        updater = _updater(_sibling_manager())
        success, message = WeightUpdater.stage_adapter(
            updater,
            names=[],
            dtypes=[],
            shapes=[],
            group_name="g",
            load_format="oft_adapter",
            adapter_config={"oft_block_size": 32},
            adapter_name="orbit_oft",
            adapter_version="1",
            double_buffer=True,
        )
        self.assertTrue(success, message)

    def test_activate_route_succeeds(self):
        manager = _sibling_manager()
        manager.stage_adapter([], {"oft_block_size": 32}, "orbit_oft", 1)
        updater = _updater(manager)
        success, message = WeightUpdater.activate_adapter_version(
            updater,            adapter_name="orbit_oft", adapter_id=None, adapter_version="1"
        )
        self.assertTrue(success, message)


if __name__ == "__main__":
    unittest.main()
