import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestAdapterStagingBackendIsAbstract(unittest.TestCase):
    def test_cannot_instantiate_the_bare_interface(self):
        from sglang.srt.adapter_sync.tokenizer_backend import AdapterStagingBackend

        with self.assertRaises(TypeError):
            AdapterStagingBackend()

    def test_lora_backend_implements_the_full_interface(self):
        from sglang.srt.adapter_sync.tokenizer_backend import AdapterStagingBackend
        from sglang.srt.lora.staged_manager import LoRAStagingBackend

        self.assertTrue(issubclass(LoRAStagingBackend, AdapterStagingBackend))


class TestStagingBackendSelection(unittest.TestCase):
    def test_selects_lora_backend_for_native_staged_update(self):
        from types import SimpleNamespace

        from sglang.srt.adapter_sync.tokenizer_backend import get_staging_backend
        from sglang.srt.lora.staged_manager import LoRAStagingBackend

        tm = SimpleNamespace(server_args=SimpleNamespace(enable_lora_staging=True))
        obj = SimpleNamespace(load_format="lora_adapter")
        self.assertIsInstance(get_staging_backend(tm, obj), LoRAStagingBackend)

    def test_selects_no_backend_when_lora_staging_is_disabled(self):
        from types import SimpleNamespace

        from sglang.srt.adapter_sync.tokenizer_backend import get_staging_backend

        tm = SimpleNamespace(server_args=SimpleNamespace(enable_lora_staging=False))
        obj = SimpleNamespace(load_format="lora_adapter")
        self.assertIsNone(get_staging_backend(tm, obj))

    def test_selects_no_backend_for_non_lora_payload(self):
        from types import SimpleNamespace

        from sglang.srt.adapter_sync.tokenizer_backend import get_staging_backend

        tm = SimpleNamespace(server_args=SimpleNamespace(enable_lora_staging=True))
        obj = SimpleNamespace(load_format="oft_adapter")
        self.assertIsNone(get_staging_backend(tm, obj))
