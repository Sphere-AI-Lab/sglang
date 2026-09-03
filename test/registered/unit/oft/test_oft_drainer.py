import unittest
from types import SimpleNamespace
from unittest import mock


MOCK_START_TIME = 1000.0
DRAIN_THRESHOLD = 3.0


def make_req(adapter_id, wait_queue_entry_time, max_new_tokens, output_len=0):
    return SimpleNamespace(
        adapter_id=adapter_id,
        time_stats=SimpleNamespace(wait_queue_entry_time=wait_queue_entry_time),
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
        output_ids=[0] * output_len,
    )


class TestOFTDrainer(unittest.TestCase):
    def test_starving_adapter_drains_shortest_running_adapter(self):
        from sglang.srt.oft.oft_drainer import OFTDrainer

        with mock.patch("time.monotonic", return_value=MOCK_START_TIME):
            drainer = OFTDrainer(
                max_ofts_per_batch=2, max_wait_time_secs=DRAIN_THRESHOLD
            )
            waiting = [
                make_req("waiting-a", MOCK_START_TIME - 4.0, 10),
                make_req("waiting-b", MOCK_START_TIME - 3.5, 10),
            ]
            running = [
                make_req("short", MOCK_START_TIME, 5),
                make_req("long", MOCK_START_TIME, 100),
            ]

            drainer.update_draining_state(waiting, running)

        self.assertEqual(
            drainer.adapter_to_stats["short"].is_draining_for, "waiting-a"
        )
        self.assertEqual(
            drainer.adapter_to_stats["long"].is_draining_for, "waiting-b"
        )

    def test_draining_adapter_rejects_requests_that_extend_its_tail(self):
        from sglang.srt.oft.oft_drainer import OFTDrainer

        with mock.patch("time.monotonic", return_value=MOCK_START_TIME):
            drainer = OFTDrainer(
                max_ofts_per_batch=1, max_wait_time_secs=DRAIN_THRESHOLD
            )
            drainer.update_draining_state(
                [make_req("waiting", MOCK_START_TIME - 4.0, 10)],
                [make_req("running", MOCK_START_TIME, 15)],
            )

        self.assertTrue(
            drainer.can_schedule(make_req("running", 0.0, max_new_tokens=10))
        )
        self.assertFalse(
            drainer.can_schedule(make_req("running", 0.0, max_new_tokens=20))
        )

    def test_oft_admission_honors_draining_before_capacity(self):
        from sglang.srt.oft.oft_drainer import OFTDrainer
        from sglang.srt.oft.integration import maybe_admit_request

        drainer = OFTDrainer(max_ofts_per_batch=1)
        drainer.adapter_to_stats["running"].is_draining_for = "waiting"
        drainer.adapter_to_stats["running"].max_remaining_tokens = 10

        class CapacityMustNotRun:
            def validate_oft_batch(self, _adapter_ids):
                raise AssertionError("capacity check ran for a draining adapter")

        scheduler = SimpleNamespace(
            oft_drainer=drainer,
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(oft_manager=CapacityMustNotRun())
            ),
        )
        req = make_req("running", 0.0, max_new_tokens=20)

        self.assertFalse(maybe_admit_request(scheduler, req, {"running"}))

    def test_scheduler_initializes_the_configured_oft_drainer(self):
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.oft.oft_drainer import OFTDrainer

        scheduler = object.__new__(Scheduler)
        scheduler.max_ofts_per_batch = 3

        with mock.patch(
            "sglang.srt.managers.scheduler.get_lora",
            return_value=SimpleNamespace(oft_drain_wait_threshold=2.5),
        ):
            scheduler.init_oft_drainer()

        self.assertIsInstance(scheduler.oft_drainer, OFTDrainer)
        self.assertEqual(scheduler.oft_drainer.max_ofts_per_batch, 3)
        self.assertEqual(scheduler.oft_drainer.max_wait_time_secs, 2.5)


if __name__ == "__main__":
    unittest.main()
