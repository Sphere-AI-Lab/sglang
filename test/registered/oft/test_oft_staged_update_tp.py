"""Native OFT TP2 optional-buffer, transport, rollback and consensus coverage."""

import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Import the module, not its TestCase: unittest must collect each suite once.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lora"))
import test_lora_staged_update_tp as tp

register_cuda_ci(est_time=900, stage="extra-b", runner_config="4-gpu-h100")


class TestStagedOFTUpdateTP(CustomTestCase):
    def test_tp2_native_lifecycle(self):
        config = tp.AutoConfig.from_pretrained(tp.MODEL_PATH).to_dict()
        with tempfile.TemporaryDirectory(prefix="oft-tp2-") as directory:
            root = Path(directory)
            v1, v2 = [
                tp._complete_fixture(root / str(v), config, "native_oft", v)
                for v in (1, 2)
            ]
            harness = tp.NativeTPHarness(self, "native_oft", v1, root / "fault.json")
            try:
                base = harness.generate()
                expected = {}
                for version, fixture in ((1, v1), (2, v2)):
                    for mechanism in ("path", "tensor", "distributed"):
                        with self.subTest(version=version, mechanism=mechanism):
                            if mechanism == "path":
                                result = harness.control.load_path(
                                    "policy-a", str(fixture)
                                )
                            elif mechanism == "tensor":
                                result = harness.control.load_tensors(
                                    "policy-a",
                                    tp.load_file(
                                        str(fixture / "adapter_model.safetensors")
                                    ),
                                    tp.json.loads(
                                        (fixture / "adapter_config.json").read_text()
                                    ),
                                )
                            else:
                                result = harness.transfer(fixture)
                            harness.success(result)
                            self.assertIsNone(harness.control.inspect_state()["staged"])
                            tokens = harness.generate("policy-a")
                            self.assertNotEqual(tokens, base)
                            if version in expected:
                                self.assertEqual(tokens, expected[version])
                            else:
                                expected[version] = tokens
                            self.assertEqual(harness.generate(), base)
                            harness.success(harness.control.unload("policy-a"))
                            self.assertEqual(harness.generate(), base)
                self.assertNotEqual(expected[1], expected[2])

                harness.success(harness.transfer(v1, version=1))
                self.assertIsNone(harness.control.inspect_state()["active"])
                self.assertEqual(harness.generate(), base)
                harness.activate(1)
                self.assertEqual(harness.generate("policy-a"), expected[1])
                harness.success(harness.transfer(v2, version=2))
                self.assertEqual(harness.generate("policy-a"), expected[1])
                self.assertEqual(harness.generate(), base)
                harness.activate(2)
                self.assertEqual(harness.generate("policy-a"), expected[2])
                self.assertEqual(harness.generate(), base)

                for rank in (0, 1):
                    with self.subTest(stage_failure_rank=rank):
                        harness.assert_stage_rollback(v1, rank, base, expected[2])
                harness.success(harness.transfer(v1, version=3))
                harness.activate(3)
                self.assertEqual(harness.generate("policy-a"), expected[1])
                self.assertEqual(harness.generate(), base)

                for rank in (0, 1):
                    with self.subTest(unload_failure_rank=rank):
                        old = harness.control.inspect_state()["registered"][0]
                        with harness.fail_rank("unload", rank):
                            result = harness.control.unload("policy-a")
                        self.assertFalse(result.success, result)
                        self.assertIn(
                            f"TP rank {rank}: injected unload failure on rank {rank}",
                            result.message,
                        )
                        state = harness.control.inspect_state()
                        self.assertEqual(state["registered"], [])
                        self.assertIsNone(state["staged"])
                        self.assertIsNone(state["active"])
                        self.assertEqual(
                            state["tombstoned"],
                            [{k: v for k, v in old.items() if k != "registry_slot"}],
                        )
                        self.assertEqual(harness.generate(), base)
                        harness.success(harness.control.unload("policy-a"))
                        self.assertEqual(
                            harness.control.inspect_state()["tombstoned"], []
                        )
                        self.assertEqual(harness.generate(), base)
                        if rank == 0:
                            harness.success(
                                harness.control.load_path("policy-a", str(v2))
                            )
                            self.assertEqual(harness.generate("policy-a"), expected[2])
                # Native controls must expose a non-sender rank's failure.
                with harness.fail_rank("native_path", 1):
                    result = harness.control.load_path("failed-path", str(v1))
                self.assertFalse(result.success, result)
                self.assertIn("TP rank 1", result.message)
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    harness.generate("failed-path")

                config_dict = tp.json.loads((v1 / "adapter_config.json").read_text())
                harness.success(
                    harness.control.load_tensors(
                        "failed-tensor",
                        tp.load_file(str(v1 / "adapter_model.safetensors")),
                        config_dict,
                    )
                )
                self.assertEqual(harness.generate("failed-tensor"), expected[1])
                with harness.fail_rank("native_tensor", 1):
                    result = harness.control.load_tensors(
                        "failed-tensor",
                        tp.load_file(str(v2 / "adapter_model.safetensors")),
                        config_dict,
                        upsert=True,
                    )
                self.assertFalse(result.success, result)
                self.assertIn("TP rank 1", result.message)
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    harness.generate("failed-tensor")

                with harness.fail_rank("native_distributed", 1):
                    result = harness.transfer(v1)
                self.assertFalse(result.success, result)
                self.assertIn("TP rank 1", result.message)
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    harness.generate("policy-a")
                self.assertEqual(harness.generate(), base)
            finally:
                harness.close()


if __name__ == "__main__":
    unittest.main()
