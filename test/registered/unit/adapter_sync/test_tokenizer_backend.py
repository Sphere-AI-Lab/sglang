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
    def test_selects_oft_backend_once_staged_impl_chosen(self):
        from types import SimpleNamespace

        from sglang.srt.adapter_sync.tokenizer_backend import get_staging_backend
        from sglang.srt.oft.staged_manager import OFTStagingBackend

        tm = SimpleNamespace(
            server_args=SimpleNamespace(enable_lora_staging=False, oft_impl="staged")
        )
        obj = SimpleNamespace(load_format="oft_adapter")
        self.assertIsInstance(get_staging_backend(tm, obj), OFTStagingBackend)

    def test_selects_no_backend_for_the_plain_sibling_impl(self):
        from types import SimpleNamespace

        from sglang.srt.adapter_sync.tokenizer_backend import get_staging_backend

        tm = SimpleNamespace(
            server_args=SimpleNamespace(enable_lora_staging=False, oft_impl="sibling")
        )
        obj = SimpleNamespace(load_format="oft_adapter")
        self.assertIsNone(get_staging_backend(tm, obj))
