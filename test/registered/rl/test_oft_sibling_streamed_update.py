"""GPU coverage for the sibling (non-staged) streamed OFT adapter path --
``load_streamed_oft_adapter`` / ``_ensure_streaming_oft_adapter_slot``
(python/sglang/srt/oft/streamed_weight_loader.py), reached through
``UpdateWeightsFromTensorReqInput`` with ``load_format="oft_adapter"`` when
the server boots with ``--oft-impl sibling`` (the default). This is the RL
weight-sync path (orbit/verl repeatedly pushing updated policy-adapter
weights into a live server) and, before this file, had zero test coverage
anywhere in the repo.

Uses ``sgl.Engine`` in-process, mirroring test_lora_load_from_tensor.py's
pattern -- but that file's ``load_lora_adapter_from_tensors`` is LoRA's own
dedicated RPC and is NOT the mechanism under test here. Engine.update_weights_
from_tensor's public wrapper does not expose adapter_config/adapter_name/
adapter_id, and its private ``_serialize_tensors_per_rank`` does not special-
case load_format="oft_adapter" either (verified empirically: it falls through
to generic MultiprocessingSerializer.serialize of the raw tensor list, which
``normalize_oft_weight_payload`` then rejects). Real streamed-OFT callers
(orbit, verl) instead build the wire payload with
``serialize_flattened_oft_payload`` directly (see that function's docstring)
and hand it to a manually-constructed ``UpdateWeightsFromTensorReqInput`` --
that is what ``_do_streamed_update`` below does.

``update_weights_from_tensor`` also asserts an open ``begin_weight_update()``
session (scheduler_components/weight_updater.py), so every update below is
wrapped in begin/end.

REGRESSION TEST for a since-fixed bug: ``_ensure_streaming_oft_adapter_slot``
raises ``ValueError`` for both the block-size-mismatch guard and the other-
resident-adapter guard, but ``load_streamed_oft_adapter`` used to only catch
``RuntimeError`` around the call:

    try:
        buffer_id, block_size = _ensure_streaming_oft_adapter_slot(...)
    except RuntimeError as exc:
        return False, str(exc)

Since ``ValueError`` is not a ``RuntimeError``, both guards' exceptions
escaped uncaught instead of becoming a graceful ``(False, message)`` -- they
propagated through the scheduler's request-dispatch loop, which has no
handler either, so the scheduler's outer ``except Exception`` in
``run_scheduler_process`` caught it, logged "Scheduler hit an exception", and
sent SIGQUIT to its parent process, whose ``running_phase_sigquit_handler``
then called ``kill_process_tree(os.getpid())`` -- i.e. instead of rejecting
one bad update, THE ENTIRE ENGINE PROCESS WAS KILLED. Verified on real GPU
hardware before the fix. Fixed by widening the ``except`` clause to
``(RuntimeError, ValueError)`` and converting a third, separate unguarded
``raise ValueError`` (the DSV4-to-FusedMoE conversion's no-target fallback)
into the function's own established ``return False, message`` convention.

``test_block_size_mismatch_guard`` and ``test_other_resident_adapter_guard``
still run the guard-triggering update inside an isolated subprocess (so a
future regression of this same shape can't take down the pytest process
itself) and assert the graceful ``success=False`` behavior directly.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import AutoConfig

import sglang as sgl
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.oft.streamed_weight_loader import serialize_flattened_oft_payload
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-large")

MODEL_PATH = "Qwen/Qwen3-0.6B"
TARGET_MODULE = "down_proj"
BLOCK_SIZE = 32
PROMPT = "Hello, my name is"
MAX_NEW_TOKENS = 8

REPO_ROOT = "/workspace/sglang-spherelab"


def _oft_named_tensors(num_layers: int, intermediate_size: int, seed: int) -> dict:
    """Compact OFT-R tensors for TARGET_MODULE across every layer, in the
    same (checkpoint-name -> tensor) format load_streamed_oft_adapter
    expects. Adapted from test/registered/lora/test_oft_staged_update.py's
    helper of the same name (no real small trained OFT HF adapter repo
    exists anywhere in this codebase to reuse instead -- see that file's
    module docstring)."""
    assert intermediate_size % BLOCK_SIZE == 0
    num_blocks = intermediate_size // BLOCK_SIZE
    n_elements = BLOCK_SIZE * (BLOCK_SIZE - 1) // 2
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer_idx in range(num_layers):
        name = f"model.layers.{layer_idx}.mlp.{TARGET_MODULE}.oft_R"
        tensors[name] = (
            torch.randn((num_blocks, n_elements), generator=generator) * 0.02
        )
    return tensors


def _adapter_config_dict(block_size: int = BLOCK_SIZE) -> dict:
    return {
        "peft_type": "OFT",
        "target_modules": [TARGET_MODULE],
        "oft_block_size": block_size,
    }


def _do_streamed_update(engine, tensors, adapter_config, adapter_name):
    """Round-trips synthetic OFT tensors through the same wire path
    WeightUpdater.update_weights_from_tensor uses for load_format=
    "oft_adapter": serialize_flattened_oft_payload builds the
    FlattenedOFTTensorPayload real streaming callers (orbit/verl) send (see
    that function's docstring and normalize_oft_weight_payload's). Engine's
    own _serialize_tensors_per_rank does not special-case "oft_adapter" (it
    falls through to a plain MultiprocessingSerializer.serialize of the raw
    tensor list, which the receiving side rejects), so it is bypassed here.
    """
    named_tensors = list(tensors.items())
    payload_bytes = serialize_flattened_oft_payload(named_tensors)
    serialized_named_tensors = [payload_bytes] * engine.server_args.tp_size
    obj = UpdateWeightsFromTensorReqInput(
        serialized_named_tensors=serialized_named_tensors,
        load_format="oft_adapter",
        adapter_config=adapter_config,
        adapter_name=adapter_name,
    )
    engine.begin_weight_update()
    try:
        return engine.loop.run_until_complete(
            engine.tokenizer_manager.update_weights_from_tensor(obj, None)
        )
    finally:
        engine.end_weight_update()


# Standalone child-process script for the two guard scenarios. Booting a
# fresh engine per subprocess call is deliberate: _ensure_streaming_oft_
# adapter_slot's guards raise a ValueError that (see this file's module
# docstring) is NOT caught anywhere on the path back to the caller, so
# triggering it in-process would take down the pytest process itself via
# SIGQUIT -> kill_process_tree. Running it in a subprocess confines that
# blast radius to the subprocess; the parent only ever observes the
# subprocess's exit code and (if it managed to write one) its result file.
_GUARD_CHILD_SCRIPT = f"""
import json
import sys

import torch
from transformers import AutoConfig

import sglang as sgl
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.oft.streamed_weight_loader import serialize_flattened_oft_payload

MODEL_PATH = {MODEL_PATH!r}
TARGET_MODULE = {TARGET_MODULE!r}
BLOCK_SIZE = {BLOCK_SIZE!r}
PROMPT = {PROMPT!r}
MAX_NEW_TOKENS = {MAX_NEW_TOKENS!r}


def _oft_named_tensors(num_layers, intermediate_size, seed):
    num_blocks = intermediate_size // BLOCK_SIZE
    n_elements = BLOCK_SIZE * (BLOCK_SIZE - 1) // 2
    generator = torch.Generator().manual_seed(seed)
    tensors = {{}}
    for layer_idx in range(num_layers):
        name = f"model.layers.{{layer_idx}}.mlp.{{TARGET_MODULE}}.oft_R"
        tensors[name] = torch.randn((num_blocks, n_elements), generator=generator) * 0.02
    return tensors


def _adapter_config_dict(block_size=BLOCK_SIZE):
    return {{"peft_type": "OFT", "target_modules": [TARGET_MODULE], "oft_block_size": block_size}}


def _do_streamed_update(engine, tensors, adapter_config, adapter_name):
    named_tensors = list(tensors.items())
    payload_bytes = serialize_flattened_oft_payload(named_tensors)
    serialized_named_tensors = [payload_bytes] * engine.server_args.tp_size
    obj = UpdateWeightsFromTensorReqInput(
        serialized_named_tensors=serialized_named_tensors,
        load_format="oft_adapter",
        adapter_config=adapter_config,
        adapter_name=adapter_name,
    )
    engine.begin_weight_update()
    try:
        return engine.loop.run_until_complete(
            engine.tokenizer_manager.update_weights_from_tensor(obj, None)
        )
    finally:
        engine.end_weight_update()


def main():
    scenario = sys.argv[1]
    result_path = sys.argv[2]
    engine = sgl.Engine(
        model_path=MODEL_PATH,
        peft_method="oft",
        oft_impl="sibling",
        max_oft_block_size=BLOCK_SIZE,
        peft_target_modules=[TARGET_MODULE],
        mem_fraction_static=0.6,
        log_level="error",
    )
    hf_config = AutoConfig.from_pretrained(MODEL_PATH)
    num_layers = hf_config.num_hidden_layers
    intermediate_size = hf_config.intermediate_size

    tensors_a = _oft_named_tensors(num_layers, intermediate_size, seed=1)
    _do_streamed_update(engine, tensors_a, _adapter_config_dict(), "policy-a")

    if scenario == "block_size_mismatch":
        result = list(
            _do_streamed_update(
                engine, tensors_a, _adapter_config_dict(block_size=16), "policy-a"
            )
        )
    elif scenario == "other_adapter_resident":
        tensors_b = _oft_named_tensors(num_layers, intermediate_size, seed=2)
        result = list(
            _do_streamed_update(engine, tensors_b, _adapter_config_dict(), "policy-b")
        )
        if not result[0]:
            # Guard correctly rejected the update -- confirm it didn't
            # silently corrupt the resident "policy-a" adapter on the way.
            out = engine.generate(
                prompt=[PROMPT],
                sampling_params={{
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature": 0.0,
                    "ignore_eos": True,
                }},
                adapter_path=["policy-a"],
            )
            result.append(out[0]["text"])
    else:
        raise ValueError(f"unknown scenario {{scenario}}")

    with open(result_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
"""


class TestOFTSiblingStreamedUpdate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path=MODEL_PATH,
            peft_method="oft",
            oft_impl="sibling",
            max_oft_block_size=BLOCK_SIZE,
            peft_target_modules=[TARGET_MODULE],
            mem_fraction_static=0.6,
            log_level="error",
        )
        hf_config = AutoConfig.from_pretrained(MODEL_PATH)
        cls.num_layers = hf_config.num_hidden_layers
        cls.intermediate_size = hf_config.intermediate_size

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()

    def _tensors(self, seed):
        return _oft_named_tensors(self.num_layers, self.intermediate_size, seed)

    def test_register_generate_and_hot_swap(self):
        """Covers scenarios 1 and 2 as one continuous adapter lifecycle
        (deliberately NOT split into separate test methods): a fresh
        streamed OFT adapter registers successfully and is usable in a real
        generate request (happy path), then a second streamed update for the
        SAME adapter name -- the actual intended use case, an RL loop
        repeatedly pushing updated weights for one policy adapter -- also
        succeeds and its new weights actually take effect (hot-swap).

        Deliberately a single adapter name throughout: any second streamed
        update naming a DIFFERENT adapter while this one is resident trips
        the other-resident-adapter guard, whose ValueError is not caught
        (see this file's module docstring) and crashes the shared
        class-level engine -- confirmed empirically when this test was
        first split into two independently-named methods on cls.engine.
        """
        result_v1 = _do_streamed_update(
            self.engine, self._tensors(1), _adapter_config_dict(), "policy-a"
        )
        self.assertTrue(result_v1[0], f"Expected success, got {result_v1}")

        sampling_params = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        out_v1 = self.engine.generate(
            prompt=[PROMPT],
            sampling_params=sampling_params,
            adapter_path=["policy-a"],
        )
        self.assertEqual(len(out_v1), 1)
        self.assertEqual(len(out_v1[0]["output_ids"]), MAX_NEW_TOKENS)

        result_v2 = _do_streamed_update(
            self.engine, self._tensors(11), _adapter_config_dict(), "policy-a"
        )
        self.assertTrue(result_v2[0], f"Expected success, got {result_v2}")

        out_v2 = self.engine.generate(
            prompt=[PROMPT],
            sampling_params=sampling_params,
            adapter_path=["policy-a"],
        )
        self.assertEqual(len(out_v2[0]["output_ids"]), MAX_NEW_TOKENS)
        self.assertNotEqual(
            out_v1[0]["text"],
            out_v2[0]["text"],
            "Hot-swapped adapter weights should change generation output",
        )

    def _run_isolated_guard_scenario(self, scenario: str):
        script_path = os.path.join(self._tmp_dir, f"{scenario}_child.py")
        result_path = os.path.join(self._tmp_dir, f"{scenario}_result.json")
        with open(script_path, "w") as f:
            f.write(_GUARD_CHILD_SCRIPT)

        env = dict(os.environ)
        env["PYTHONPATH"] = "python"
        try:
            proc = subprocess.run(
                [sys.executable, script_path, scenario, result_path],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=240,
            )
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            returncode, stdout, stderr = None, exc.stdout, exc.stderr

        self.assertTrue(
            os.path.exists(result_path),
            f"Guard scenario {scenario!r} child process did not return a "
            f"result (returncode={returncode}); it likely crashed instead of "
            "gracefully returning success=False -- see this file's module "
            f"docstring.\n--- child stdout ---\n{stdout}\n"
            f"--- child stderr ---\n{stderr}",
        )
        with open(result_path) as f:
            return json.load(f)

    def setUp(self):
        self._tmp_dir_obj = tempfile.TemporaryDirectory(prefix="oft_sibling_test_")
        self._tmp_dir = self._tmp_dir_obj.name

    def tearDown(self):
        self._tmp_dir_obj.cleanup()

    def test_block_size_mismatch_guard(self):
        """A streamed update whose adapter_config carries a different
        oft_block_size than --max-oft-block-size must be rejected gracefully
        (success=False), not crash the server. Regression test for the
        ValueError-escapes-uncaught bug fixed in streamed_weight_loader.py's
        load_streamed_oft_adapter (see git history for that fix)."""
        result = self._run_isolated_guard_scenario("block_size_mismatch")
        self.assertFalse(result[0], f"Expected success=False, got {result}")
        self.assertIn("--max-oft-block-size", result[1])
        self.assertIn("block_size=16", result[1])

    def test_other_resident_adapter_guard(self):
        """A streamed update for a differently-named adapter while another
        is resident must be rejected gracefully (success=False), and the
        resident adapter must remain intact/functional afterward. Regression
        test for the same ValueError-escapes-uncaught bug."""
        result = self._run_isolated_guard_scenario("other_adapter_resident")
        self.assertFalse(result[0], f"Expected success=False, got {result}")
        self.assertIn("other adapters are resident", result[1])
        self.assertIn("policy-a", result[1])
        self.assertEqual(len(result), 3, "Expected policy-a generation output appended")
        self.assertTrue(len(result[2]) > 0)


if __name__ == "__main__":
    unittest.main()
