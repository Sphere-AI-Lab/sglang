"""Unit coverage for versioned native LoRA request and radix identity."""

import asyncio
import unittest

import msgspec

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import _extend_lora_extra_key
from sglang.srt.sampling.sampling_params import SamplingParams


class TestLoRARefVersion(CustomTestCase):
    def test_old_array_payload_defaults_version_to_zero(self):
        old = ["id-a", "adapter-a", "/adapter/a", False, True]
        decoded = msgspec.json.decode(msgspec.json.encode(old), type=LoRARef)
        self.assertEqual(decoded.version, 0)

    def test_acquire_snapshots_id_and_version_under_one_lock(self):
        registry = LoRARegistry()
        ref = LoRARef(
            lora_id="id-a",
            lora_name="adapter-a",
            lora_path="/adapter/a",
            pinned=False,
            version=7,
        )
        asyncio.run(registry.register(ref))
        self.assertEqual(
            asyncio.run(registry.acquire_with_version("adapter-a")),
            ("id-a", 7),
        )
        asyncio.run(registry.release("id-a"))

    def test_batch_acquire_returns_parallel_ids_and_versions(self):
        async def run():
            registry = LoRARegistry()
            await registry.register(
                LoRARef(lora_id="id-a", lora_name="a", version=2)
            )
            await registry.register(
                LoRARef(lora_id="id-b", lora_name="b", version=5)
            )
            ids, versions = await registry.acquire_with_version(["a", None, "b"])
            self.assertEqual(ids, ["id-a", None, "id-b"])
            self.assertEqual(versions, [2, None, 5])
            await registry.release(["id-a", "id-b"])

        asyncio.run(run())

    def test_staged_refresh_can_preserve_existing_pinned_state(self):
        async def run():
            registry = LoRARegistry()
            await registry.register(
                LoRARef(lora_id="id-a", lora_name="a", pinned=True, version=1)
            )
            resolved, reused = await registry.register_or_reuse(
                LoRARef(lora_name="a", pinned=False, version=2),
                upsert=True,
                preserve_pinned=True,
            )
            self.assertTrue(reused)
            self.assertEqual(resolved.lora_id, "id-a")
            self.assertTrue(resolved.pinned)
            self.assertEqual(resolved.version, 2)

        asyncio.run(run())

    def test_tokenized_version_fields_are_wire_compatible_trailing_fields(self):
        self.assertEqual(
            msgspec.structs.fields(TokenizedGenerateReqInput)[-1].name,
            "lora_version",
        )

        request = TokenizedEmbeddingReqInput(
            input_text=None,
            input_ids=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(max_new_tokens=0),
            lora_version=7,
        )
        current_payload = msgspec.json.decode(msgspec.json.encode(request))
        old_payload = current_payload[:-2]
        decoded = msgspec.json.decode(
            msgspec.json.encode(old_payload), type=TokenizedEmbeddingReqInput
        )

        self.assertEqual(decoded.lora_version, 7)
        self.assertIsNone(decoded.adapter_id)
        self.assertIsNone(decoded.adapter_version)


class TestLoRARadixIdentity(CustomTestCase):
    def test_base_key_is_unchanged(self):
        self.assertEqual(_extend_lora_extra_key("tenant", None, None), "tenant")

    def test_lora_key_contains_id_and_version(self):
        self.assertEqual(
            _extend_lora_extra_key("tenant", "id-a", 3),
            "tenant|lora:id-a:v3",
        )

    def test_versions_never_share_a_key(self):
        self.assertNotEqual(
            _extend_lora_extra_key(None, "id-a", 3),
            _extend_lora_extra_key(None, "id-a", 4),
        )


if __name__ == "__main__":
    unittest.main()
