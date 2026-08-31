"""Unit tests for Scheduler's max_loras_per_batch == 1 LoRA admission fast path.

Regression target: StagedLoRAManager at max_loras_per_batch=1 (the
single-active RL use case) was measurably slower than the legacy
single-adapter path because Scheduler._get_new_batch_prefill_raw built and
queried a `running_loras` set for every waiting request on every scheduling
step, even though at capacity 1 the admission decision is always a scalar
comparison: is req.lora_id the one adapter identity currently active (or is
nothing active yet)?

These tests pin down that the O(1) fast path (_resolve_active_lora_fast /
_can_schedule_lora_req_fast) is exactly equivalent to the general set-based
logic (_can_schedule_lora_req) for every admission outcome at capacity 1 --
including the reasoning that validate_lora_batch degenerates to a
size-<=1 check because no adapter can ever be pinned when
max_loras_per_batch == 1 (validate_new_adapter rejects any pinned adapter
once max_loras_per_batch - 1 == 0) -- and that the fast path still respects
the LoRA drainer and overlap-loading, both of which remain meaningfully
engageable at capacity 1.
"""

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_drainer import LoRADrainer
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(lora_id, *, finished=False):
    req = SimpleNamespace(lora_id=lora_id, finished=lambda: finished)
    return cast(Req, req)


def _make_scheduler(
    *,
    max_loras_per_batch=1,
    enable_lora_overlap_loading=False,
    lora_drainer=None,
):
    s = Scheduler.__new__(Scheduler)
    s.max_loras_per_batch = max_loras_per_batch
    s.enable_lora_overlap_loading = enable_lora_overlap_loading
    s.lora_drainer = lora_drainer
    s.lora_overlap_loader = MagicMock()

    lora_manager = MagicMock()
    # Mirror the real LoRAManager.validate_lora_batch at max_loras_per_batch == 1
    # with no pinned adapters loaded: admits iff the union of ids is <= capacity.
    lora_manager.validate_lora_batch.side_effect = (
        lambda lora_ids: len(lora_ids) <= max_loras_per_batch
    )
    s.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(lora_manager=lora_manager)
    )
    return s


class TestResolveActiveLoraFast(CustomTestCase):
    def test_no_running_and_no_can_run_list_reports_no_active_identity(self):
        s = _make_scheduler()
        running_batch = SimpleNamespace(reqs=[])

        has_active, active_id = s._resolve_active_lora_fast(running_batch, [])

        self.assertFalse(has_active)
        self.assertIsNone(active_id)

    def test_non_finished_running_req_is_the_active_identity(self):
        s = _make_scheduler()
        running_batch = SimpleNamespace(
            reqs=[_make_req("A", finished=True), _make_req("B", finished=False)]
        )

        has_active, active_id = s._resolve_active_lora_fast(running_batch, [])

        self.assertTrue(has_active)
        self.assertEqual(active_id, "B")

    def test_falls_back_to_can_run_list_when_running_batch_is_idle(self):
        s = _make_scheduler()
        running_batch = SimpleNamespace(reqs=[_make_req("A", finished=True)])

        has_active, active_id = s._resolve_active_lora_fast(
            running_batch, [_make_req("C")]
        )

        self.assertTrue(has_active)
        self.assertEqual(active_id, "C")


class TestCanScheduleLoraReqFast(CustomTestCase):
    def test_admits_first_request_when_nothing_is_active(self):
        s = _make_scheduler()
        req = _make_req("A")

        self.assertTrue(
            s._can_schedule_lora_req_fast(
                req, has_active_lora=False, active_lora_id=None
            )
        )

    def test_admits_request_matching_the_active_identity(self):
        s = _make_scheduler()
        req = _make_req("A")

        self.assertTrue(
            s._can_schedule_lora_req_fast(req, has_active_lora=True, active_lora_id="A")
        )

    def test_rejects_request_for_a_different_identity_while_one_is_active(self):
        s = _make_scheduler()
        req = _make_req("B")

        self.assertFalse(
            s._can_schedule_lora_req_fast(req, has_active_lora=True, active_lora_id="A")
        )

    def test_drainer_veto_is_respected_even_when_identity_matches(self):
        # The drainer can mark the single active adapter as "draining" to make
        # room for a starving one even at capacity 1 (LoRADrainer's own
        # draining trigger only requires len(running_adapter_ids) to have
        # reached max_loras_per_batch, which is 1 here) -- so the fast path
        # must not bypass it just because req.lora_id matches active_lora_id.
        drainer = MagicMock(spec=LoRADrainer)
        drainer.can_schedule.return_value = False
        s = _make_scheduler(lora_drainer=drainer)
        req = _make_req("A")

        self.assertFalse(
            s._can_schedule_lora_req_fast(req, has_active_lora=True, active_lora_id="A")
        )
        drainer.can_schedule.assert_called_once_with(req)

    def test_overlap_loading_is_consulted_for_a_new_identity(self):
        s = _make_scheduler(enable_lora_overlap_loading=True)
        s.lora_overlap_loader.try_overlap_load_lora.return_value = True
        req = _make_req("A")

        result = s._can_schedule_lora_req_fast(
            req, has_active_lora=False, active_lora_id=None
        )

        self.assertTrue(result)
        s.lora_overlap_loader.try_overlap_load_lora.assert_called_once_with("A", set())

    def test_overlap_loading_receives_the_active_identity_as_running_loras(self):
        s = _make_scheduler(enable_lora_overlap_loading=True)
        s.lora_overlap_loader.try_overlap_load_lora.return_value = False
        req = _make_req("B")

        result = s._can_schedule_lora_req_fast(
            req, has_active_lora=True, active_lora_id="A"
        )

        self.assertFalse(result)
        s.lora_overlap_loader.try_overlap_load_lora.assert_called_once_with("B", {"A"})


class TestFastPathMatchesGeneralAdmission(CustomTestCase):
    """Derived-property guard: at max_loras_per_batch == 1, the O(1) fast path
    must return exactly what the general set-based _can_schedule_lora_req
    would, for every admission outcome, including base-model (lora_id=None)
    requests and the "no adapter is currently active" identity."""

    CASES = [
        # (has_active_lora, active_lora_id, req_lora_id)
        (False, None, "A"),  # nothing active yet
        (False, None, None),  # nothing active yet, base-model request
        (True, "A", "A"),  # matches the active identity
        (True, "A", "B"),  # a different identity while one is active
        (True, None, None),  # active identity is "no adapter" (base model)
        (True, None, "A"),  # switching away from "no adapter"
    ]

    def test_fast_and_general_paths_agree_without_overlap_loading(self):
        for has_active, active_id, req_id in self.CASES:
            with self.subTest(
                has_active=has_active, active_id=active_id, req_id=req_id
            ):
                s = _make_scheduler(enable_lora_overlap_loading=False)
                req = _make_req(req_id)
                running_loras = {active_id} if has_active else set()

                fast_result = s._can_schedule_lora_req_fast(
                    req, has_active_lora=has_active, active_lora_id=active_id
                )
                general_result = s._can_schedule_lora_req(req, running_loras)

                self.assertEqual(fast_result, general_result)


if __name__ == "__main__":
    unittest.main()
