import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import (  # isort: skip
    Scheduler,
    _resolve_exact_scoring_suffix_boundary,
    _should_allow_auto_truncate,
)
from sglang.srt.managers.utils import validate_input_length  # isort: skip

register_cpu_ci(est_time=2, suite="base-b-test-cpu")


def _recv(input_ids, suffix_len=2):
    return SimpleNamespace(
        input_ids=array("q", input_ids),
        scoring_suffix_len=suffix_len,
    )


def _req(input_ids, *, streaming_session=False):
    session = SimpleNamespace(streaming=True) if streaming_session else None
    return SimpleNamespace(origin_input_ids=array("q", input_ids), session=session)


class TestExactScoringSuffixSchedulerBoundary(unittest.TestCase):
    def test_derives_boundary_after_model_specific_multimodal_expansion(self):
        recv_req = _recv([10, 99, 11, 41, 42])
        req = _req([10, 901, 902, 903, 11, 41, 42])

        boundary, error = _resolve_exact_scoring_suffix_boundary(req, recv_req)

        self.assertIsNone(error)
        self.assertEqual(boundary, 4)

    def test_derives_boundary_after_non_streaming_session_history(self):
        recv_req = _recv([10, 11, 41, 42])
        req = _req([1, 2, 3, 10, 11, 41, 42])

        boundary, error = _resolve_exact_scoring_suffix_boundary(req, recv_req)

        self.assertIsNone(error)
        self.assertEqual(boundary, 4)

    def test_rejects_final_sequence_that_does_not_preserve_suffix_tail(self):
        recv_req = _recv([10, 11, 41, 42])
        req = _req([10, 11, 34070])

        boundary, error = _resolve_exact_scoring_suffix_boundary(req, recv_req)

        self.assertIsNone(boundary)
        self.assertIn("did not preserve scoring_suffix_ids", error)

    def test_rejects_streaming_session(self):
        recv_req = _recv([10, 11, 41, 42])
        req = _req([10, 11, 41, 42], streaming_session=True)

        boundary, error = _resolve_exact_scoring_suffix_boundary(req, recv_req)

        self.assertIsNone(boundary)
        self.assertIn("not supported for streaming sessions", error)

    @patch("sglang.srt.managers.scheduler.Req")
    def test_rejects_streaming_session_before_composition(self, req_cls):
        session = SimpleNamespace(
            streaming=True,
            close_on_finish=False,
            _inflight=False,
            multimodal_inputs=object(),
            create_req=Mock(side_effect=AssertionError("must not compose session")),
        )
        original_multimodal_inputs = session.multimodal_inputs
        queued_req = SimpleNamespace(tokenizer=None, set_finish_with_abort=Mock())
        req_cls.return_value = queued_req
        scheduler = SimpleNamespace(
            session_controller={"session-id": session},
            model_config=SimpleNamespace(vocab_size=128),
            tokenizer=object(),
            init_req_max_new_tokens=Mock(),
            _add_request_to_queue=Mock(),
        )
        recv_req = SimpleNamespace(
            session_params=SimpleNamespace(id="session-id"),
            session_id=None,  # v0.5.15 top-level field read by the radix-native session route
            scoring_suffix_len=2,
            rid="request-id",
            input_text=None,
            input_ids=array("q", [10, 11, 41, 42]),
            sampling_params=object(),
            http_worker_ipc=None,
        )

        Scheduler.handle_generate_request(scheduler, recv_req)

        session.create_req.assert_not_called()
        self.assertFalse(session._inflight)
        self.assertIs(session.multimodal_inputs, original_multimodal_inputs)
        queued_req.set_finish_with_abort.assert_called_once()
        scheduler.init_req_max_new_tokens.assert_called_once_with(queued_req)
        scheduler._add_request_to_queue.assert_called_once_with(queued_req)

    def test_non_exact_request_does_not_override_normal_boundary_logic(self):
        recv_req = _recv([10, 11], suffix_len=None)
        req = _req([10, 11])

        boundary, error = _resolve_exact_scoring_suffix_boundary(req, recv_req)

        self.assertIsNone(boundary)
        self.assertIsNone(error)

    def test_exact_suffix_disables_scheduler_auto_truncation(self):
        req = _req([10, 11, 41, 42])

        error = validate_input_length(
            req,
            max_req_input_len=3,
            allow_auto_truncate=_should_allow_auto_truncate(True, 2),
        )

        self.assertIsNotNone(error)
        self.assertEqual(list(req.origin_input_ids), [10, 11, 41, 42])

    def test_non_exact_request_keeps_configured_auto_truncation(self):
        req = _req([10, 11, 12, 13])

        error = validate_input_length(
            req,
            max_req_input_len=3,
            allow_auto_truncate=_should_allow_auto_truncate(True, None),
        )

        self.assertIsNone(error)
        self.assertEqual(list(req.origin_input_ids), [10, 11, 12])


if __name__ == "__main__":
    unittest.main()
