"""GPU regression test for the retired ``load_format="oft_adapter"`` path in
``WeightUpdater.update_weights_from_tensor`` (python/sglang/srt/model_executor/
model_runner_components/weight_updater.py).

Context: Task 9 of the 2026-08-31-oft-native-adapter-rpc plan retired the old
srt/peft streamed-loader mechanism (``maybe_load_adapter_format`` ->
``load_streamed_oft_adapter``) that used to handle ``load_format="oft_adapter"``
here. Without an explicit graceful reject, that ``load_format`` value falls
through to the generic ``else: raise NotImplementedError(...)`` branch at the
bottom of ``update_weights_from_tensor`` -- and there is no try/except
anywhere in the scheduler's request-dispatch path for
``update_weights_from_tensor`` (``managers/scheduler_components/
weight_updater.py``), unlike its ``update_weights_from_distributed`` sibling,
which explicitly wraps the equivalent call in ``try/except Exception``. An
uncaught ``NotImplementedError`` there propagates through
``run_scheduler_process``'s outer ``except Exception``, which logs "Scheduler
hit an exception" and sends SIGQUIT to the parent -- i.e. instead of one bad
request failing, THE ENTIRE ENGINE PROCESS IS KILLED.

This is the exact same failure-mode class as the ``ValueError``-escaping-
uncaught bug fixed earlier in this plan for
``_ensure_streaming_oft_adapter_slot`` (see the now-deleted
``test_oft_sibling_streamed_update.py``'s module docstring, removed in the
same Task 9 commit that retired this path) -- except reintroduced via
``NotImplementedError``, specifically for the one ``load_format`` value most
likely to have surviving external callers (see Task 9's report). Task 9's
follow-up fix adds an explicit early ``return False, "..."`` for
``load_format == "oft_adapter"`` before that generic branch; this test
guards that fix.

Run in an isolated subprocess, same discipline as the deleted file's guard
tests (``test_block_size_mismatch_guard`` / ``test_other_resident_adapter_
guard``): this asserts a "used to crash, now must not" property, so if the
graceful-reject fix ever regresses, the failure must show up as a subprocess
exit code / missing result file, not a crash of the pytest process itself.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="extra-a", runner_config="1-gpu-large")

MODEL_PATH = "Qwen/Qwen3-0.6B"
REPO_ROOT = "/workspace/sglang-spherelab"

# Standalone child-process script, mirroring test_oft_sibling_streamed_
# update.py's _GUARD_CHILD_SCRIPT pattern: booting a fresh engine in a
# subprocess confines the blast radius of a regression (the code path under
# test used to SIGQUIT-kill its own engine process) to the subprocess: the
# parent only ever observes its exit code and (if it managed to write one)
# its result file.
_CHILD_SCRIPT = f"""
import json
import sys

import sglang as sgl
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.utils import MultiprocessingSerializer

MODEL_PATH = {MODEL_PATH!r}


def main():
    result_path = sys.argv[1]
    engine = sgl.Engine(
        model_path=MODEL_PATH,
        mem_fraction_static=0.6,
        log_level="error",
    )

    # The graceful reject fires on load_format alone, before any tensor
    # payload is interpreted -- an empty list is enough to survive
    # MultiprocessingSerializer.deserialize on the receiving end.
    payload_bytes = MultiprocessingSerializer.serialize([])
    obj = UpdateWeightsFromTensorReqInput(
        serialized_named_tensors=[payload_bytes] * engine.server_args.tp_size,
        load_format="oft_adapter",
        adapter_config={{
            "peft_type": "OFT",
            "target_modules": ["down_proj"],
            "oft_block_size": 32,
        }},
        adapter_name="policy-a",
    )
    engine.begin_weight_update()
    try:
        success, message = engine.loop.run_until_complete(
            engine.tokenizer_manager.update_weights_from_tensor(obj, None)
        )
    finally:
        engine.end_weight_update()

    # The engine must still be alive and servable afterwards -- the graceful
    # reject must not have corrupted anything.
    out = engine.generate(
        prompt=["Hello, my name is"],
        sampling_params={{"max_new_tokens": 4, "temperature": 0.0, "ignore_eos": True}},
    )

    with open(result_path, "w") as f:
        json.dump([success, message, len(out[0]["output_ids"])], f)


if __name__ == "__main__":
    main()
"""


class TestOFTAdapterFormatRetired(CustomTestCase):
    def setUp(self):
        self._tmp_dir_obj = tempfile.TemporaryDirectory(
            prefix="oft_adapter_retired_test_"
        )
        self._tmp_dir = self._tmp_dir_obj.name

    def tearDown(self):
        self._tmp_dir_obj.cleanup()

    def test_oft_adapter_load_format_fails_gracefully(self):
        """update_weights_from_tensor(load_format="oft_adapter", ...) must
        return (False, <clear message>) instead of taking down the engine
        process, and the engine must remain usable afterward."""
        script_path = os.path.join(self._tmp_dir, "child.py")
        result_path = os.path.join(self._tmp_dir, "result.json")
        with open(script_path, "w") as f:
            f.write(_CHILD_SCRIPT)

        env = dict(os.environ)
        env["PYTHONPATH"] = "python"
        try:
            proc = subprocess.run(
                [sys.executable, script_path, result_path],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            returncode, stdout, stderr = None, exc.stdout, exc.stderr

        self.assertTrue(
            os.path.exists(result_path),
            f"Child process did not return a result (returncode={returncode}); "
            "it likely crashed instead of gracefully rejecting "
            "load_format='oft_adapter' -- see this file's module docstring."
            f"\n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}",
        )
        with open(result_path) as f:
            success, message, num_output_ids = json.load(f)

        self.assertFalse(success, f"Expected success=False, got message={message!r}")
        self.assertIn("oft_adapter", message)
        self.assertIn("no longer supported", message)
        self.assertGreater(
            num_output_ids,
            0,
            "Engine should remain servable after the graceful reject",
        )


if __name__ == "__main__":
    unittest.main()
