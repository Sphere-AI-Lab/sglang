import asyncio
import inspect
import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.EngineBase import EngineBase  # isort: skip
from sglang.srt.entrypoints.engine import Engine  # isort: skip
from sglang.srt.entrypoints.http_server_engine import (  # isort: skip
    HttpServerEngineAdapter,
)

register_cpu_ci(est_time=2, suite="base-b-test-cpu")


class _CapturingTokenizerManager:
    def __init__(self):
        self.requests = []

    def generate_request(self, request, _raw_request):
        self.requests.append(request)

        async def result():
            yield {"ok": True}

        return result()


class TestExactScoringSuffixPublicApi(unittest.TestCase):
    def test_all_public_generate_signatures_expose_suffix(self):
        for generate in (
            EngineBase.generate,
            Engine.generate,
            Engine.async_generate,
            HttpServerEngineAdapter.generate,
        ):
            with self.subTest(generate=generate):
                self.assertIn(
                    "scoring_suffix_ids", inspect.signature(generate).parameters
                )

    def test_engine_sync_and_async_generate_forward_suffix(self):
        engine = object.__new__(Engine)
        engine.tokenizer_manager = _CapturingTokenizerManager()
        engine.loop = asyncio.new_event_loop()
        try:
            self.assertEqual(
                engine.generate(prompt="prompt", scoring_suffix_ids=[41, 42]),
                {"ok": True},
            )
        finally:
            engine.loop.close()

        self.assertEqual(
            engine.tokenizer_manager.requests[-1].scoring_suffix_ids, [41, 42]
        )

        self.assertEqual(
            asyncio.run(
                engine.async_generate(
                    prompt=["prompt-a", "prompt-b"],
                    scoring_suffix_ids=[[41, 42], [51, 52]],
                )
            ),
            {"ok": True},
        )
        self.assertEqual(
            engine.tokenizer_manager.requests[-1].scoring_suffix_ids,
            [[41, 42], [51, 52]],
        )

    def test_http_adapter_forwards_suffix_in_generate_payload(self):
        adapter = object.__new__(HttpServerEngineAdapter)
        adapter._make_request = MagicMock(return_value={"ok": True})

        self.assertEqual(
            adapter.generate(prompt="prompt", scoring_suffix_ids=[41, 42]),
            {"ok": True},
        )

        endpoint, payload = adapter._make_request.call_args.args
        self.assertEqual(endpoint, "generate")
        self.assertEqual(payload["scoring_suffix_ids"], [41, 42])


if __name__ == "__main__":
    unittest.main()
