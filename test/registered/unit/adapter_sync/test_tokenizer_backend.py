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
