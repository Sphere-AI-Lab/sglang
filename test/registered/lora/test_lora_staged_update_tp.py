"""TP and MoE coverage for native two-phase LoRA updates."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from test_lora_staged_update import (
    MODEL_PATH,
    StagedLoRATestHarness,
    _versioned_tensors,
)
from transformers import AutoConfig

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import (
    MOE_BASE_MODEL_PATH,
    MOE_LORA_PATH,
    MOE_LORA_TEST_PROMPTS,
)
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "manual"))

from adapter_equivalence.distributed_sender import DistributedSession
from adapter_equivalence.fixtures import build_lora_fixture, build_oft_fixture
from adapter_equivalence.run_case import (
    build_target_shapes,
    distributed_payload_for_fixture,
)
from adapter_equivalence.server import (
    AdapterIdentity,
    ServerSpec,
    engine_kwargs,
    make_adapter_control,
)

register_cuda_ci(est_time=600, stage="base-c", runner_config="4-gpu-h100")


def _complete_fixture(destination, config, mode, version):
    """v2 changes only optional buffers, making their activation observable."""
    builder = build_lora_fixture if mode == "native_lora" else build_oft_fixture
    fixture = builder(
        destination,
        adapter_id="policy-a",
        architecture="dense",
        seed=1729,
        target_shapes=build_target_shapes(config, "dense"),
        **({"shared": True} if mode == "native_oft" else {}),
    )
    tensor_path = fixture.path / "adapter_model.safetensors"
    tensors = load_file(str(tensor_path))
    for name in tensors:
        optional = ".embed_tokens." in name or ".lm_head." in name
        scale = 5 if mode == "native_lora" else 30
        tensors[name] = tensors[name] * scale * (3 if version == 2 and optional else 1)
    save_file(tensors, str(tensor_path))
    hashes = {
        name: hashlib.sha256((fixture.path / name).read_bytes()).hexdigest()
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }
    (fixture.path / "sha256.json").write_text(json.dumps(hashes))
    return fixture.path


def _rank_failure_boundary(original, fault, used, **kwargs):
    """Keep the real rank collective; change only one local result at its edge."""
    if fault and fault["operation"] in ("stage", "unload"):
        rank = kwargs["distributed"].get_rank(group=kwargs["group"])
        selected = (
            kwargs["version"] == "3"
            if fault["operation"] == "stage"
            else kwargs["version"] is None
        )
        if selected and rank == fault["rank"] and fault["nonce"] not in used:
            used.add(fault["nonce"])
            if kwargs["success"]:
                kwargs.update(
                    success=False,
                    message=f"injected {fault['operation']} failure on rank {rank}",
                    version=None,
                )
    return original(**kwargs)


def _run_fault_scheduler(*args, fault_path, **kwargs):
    """Spawn-importable hook; never patch tokenizer's already-merged replies."""
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.managers.scheduler_components import tp_update_consensus

    original = tp_update_consensus.gather_tp_update_result
    used = set()

    def boundary(**values):
        path = Path(fault_path)
        fault = json.loads(path.read_text()) if path.exists() else None
        return _rank_failure_boundary(original, fault, used, **values)

    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.oft.io_types import OFTUpdateOutput

    def native_boundary(original, operation):
        def load(worker, request):
            path = Path(fault_path)
            fault = json.loads(path.read_text()) if path.exists() else None
            rank = torch.distributed.get_rank()
            if (
                fault
                and fault["operation"] == operation
                and fault["rank"] == rank
                and fault["nonce"] not in used
            ):
                used.add(fault["nonce"])
                # Broadcast receivers must consume their payload before an
                # injected failure, otherwise the sender/peers would deadlock.
                if operation == "native_distributed":
                    original(worker, request)
                return OFTUpdateOutput(
                    success=False,
                    error_message=f"injected {operation} failure on rank {rank}",
                    previous_adapter_preserved=(
                        operation == "native_tensor" and request.upsert
                    ),
                    inconsistent_update=operation == "native_distributed",
                )
            return original(worker, request)

        return load

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(tp_update_consensus, "gather_tp_update_result", boundary)
        )
        for method, operation in (
            ("load_oft_adapter", "native_path"),
            ("load_oft_adapter_from_tensors", "native_tensor"),
            ("load_oft_adapter_from_distributed", "native_distributed"),
        ):
            stack.enter_context(
                patch.object(
                    TpModelWorker,
                    method,
                    native_boundary(getattr(TpModelWorker, method), operation),
                )
            )
        run_scheduler_process(*args, **kwargs)


class NativeTPHarness:
    """Offline native APIs with an external sender on GPU 0 and TP2 on 1–2."""

    def __init__(self, testcase, mode, fixture, fault_path):
        from sglang.srt.entrypoints.engine import Engine

        self.testcase, self.mode = testcase, mode
        self.fault_path = Path(fault_path)
        self.nonce = 0
        config = json.loads((fixture / "adapter_config.json").read_text())
        spec = ServerSpec(
            "candidate",
            MODEL_PATH,
            mode,
            30000,
            2,
            1,
            True,
            max_lora_rank=8 if mode == "native_lora" else None,
            lora_target_modules=("all",) if mode == "native_lora" else (),
            max_oft_block_size=config.get("oft_block_size"),
            peft_target_modules=(
                tuple(config["target_modules"]) if mode == "native_oft" else ()
            ),
            base_gpu_id=1,
            mem_fraction_static=0.6,
        )
        kwargs = engine_kwargs(spec)
        if mode == "native_lora":
            kwargs.update(max_loras_per_batch=1)
        else:
            kwargs.update(oft_type="oft", max_ofts_per_batch=2, max_loaded_ofts=1)
        hook = partial(_run_fault_scheduler, fault_path=str(self.fault_path))
        with patch.object(Engine, "run_scheduler_process_func", staticmethod(hook)):
            self.engine = Engine(**kwargs)
        self.session = None
        self.control = make_adapter_control(mode, self.engine)
        try:
            self.session = DistributedSession.open(self.engine, tp_size=2, timeout=300)
        except BaseException:
            self.engine.shutdown()
            raise

    def generate(self, adapter=None):
        kwargs = {
            "lora_path" if self.mode == "native_lora" else "adapter_path": adapter
        }
        body = self.engine.generate(
            "Hello, my name is",
            sampling_params={"temperature": 0, "max_new_tokens": 24},
            **kwargs,
        )
        self.testcase.assertTrue(body["output_ids"], body)
        return body["output_ids"]

    def success(self, result):
        self.testcase.assertTrue(result.success, result)
        return result

    def transfer(self, fixture, *, version=None):
        self.nonce += 1
        payload = distributed_payload_for_fixture(fixture)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.session.sender.broadcast_fixture,
                f"transfer-{self.nonce}-{version}",
                fixture,
                timeout=300,
            )
            if version is None:
                result = self.control.load_distributed(
                    "policy-a", payload, self.session.group_name
                )
            else:
                result = self.control.stage(
                    self.identity(version), payload, self.session.group_name
                )
            self.testcase.assertEqual(future.result(timeout=300), payload)
        return result

    def identity(self, version):
        state = self.control.inspect_state()
        records = [state["staged"], state["active"], *state["registered"]]
        for record in records:
            if record is not None and record["name"] == "policy-a":
                return AdapterIdentity("policy-a", record["id"], str(version))
        # A first stage asks the runtime to allocate its identity. Subsequent
        # activation must use that staged ID even before an adapter is active.
        return AdapterIdentity("policy-a", None, str(version))

    def activate(self, version):
        identity = self.identity(version)
        self.success(self.control.activate(identity))
        # The tokenizer API returns (success, message). Verify the identity
        # it published after checking every worker's active version.
        state = self.control.inspect_state()
        self.testcase.assertEqual(
            state["active"],
            {"name": identity.name, "id": identity.adapter_id, "version": str(version)},
        )
        self.testcase.assertIsNone(state["staged"])

    @contextmanager
    def fail_rank(self, operation, rank):
        self.nonce += 1
        self.fault_path.write_text(
            json.dumps(dict(operation=operation, rank=rank, nonce=self.nonce))
        )
        try:
            yield
        finally:
            self.fault_path.unlink()

    def assert_stage_rollback(self, fixture, rank, base, adapter):
        before = self.control.inspect_state()
        with self.fail_rank("stage", rank):
            result = self.transfer(fixture, version=3)
        self.testcase.assertFalse(result.success, result)
        self.testcase.assertIn(
            f"TP rank {rank}: injected stage failure on rank {rank}", result.message
        )
        self.testcase.assertIsNone(result.active_version)
        self.testcase.assertEqual(self.control.inspect_state(), before)
        rejected = self.control.activate(self.identity(3))
        self.testcase.assertFalse(rejected.success, rejected)
        self.testcase.assertEqual(self.control.inspect_state(), before)
        self.testcase.assertEqual(self.generate("policy-a"), adapter)
        self.testcase.assertEqual(self.generate(), base)

    def close(self):
        try:
            if self.session is not None:
                self.session.close(timeout=60)
        finally:
            self.engine.shutdown()


def _load_adapter(repo_id):
    adapter_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
    )
    tensors = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    with open(
        os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8"
    ) as config_file:
        config = json.load(config_file)
    return tensors, config


class TestStagedLoRAUpdateTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.cuda.set_device(0)

    def test_tp2(self):
        config = AutoConfig.from_pretrained(MODEL_PATH).to_dict()
        with tempfile.TemporaryDirectory(prefix="lora-tp2-") as directory:
            root = Path(directory)
            v1, v2 = [
                _complete_fixture(root / str(v), config, "native_lora", v)
                for v in (1, 2)
            ]
            harness = NativeTPHarness(self, "native_lora", v1, root / "fault.json")
            try:
                base = harness.generate()
                harness.success(harness.transfer(v1, version=1))
                self.assertEqual(harness.generate(), base)
                harness.activate(1)
                before = harness.generate("policy-a")
                self.assertNotEqual(before, base)
                harness.success(harness.transfer(v2, version=2))
                self.assertEqual(harness.generate("policy-a"), before)
                self.assertEqual(harness.generate(), base)
                harness.activate(2)
                after = harness.generate("policy-a")
                self.assertNotEqual(after, before)
                self.assertEqual(harness.generate(), base)
                for rank in (0, 1):
                    with self.subTest(rank=rank):
                        harness.assert_stage_rollback(v1, rank, base, after)
                # Both failed stages must release the real worker hidden slot.
                harness.success(harness.transfer(v1, version=3))
                harness.activate(3)
                self.assertEqual(harness.generate("policy-a"), before)
                self.assertEqual(harness.generate(), base)
                harness.success(harness.control.unload("policy-a"))
                self.assertEqual(harness.generate(), base)
                # Same TP2 layout, but an independent immediate path is the oracle.
                harness.success(harness.control.load_path("policy-a", str(v2)))
                self.assertEqual(harness.generate("policy-a"), after)
                harness.success(harness.control.unload("policy-a"))
                self.assertEqual(harness.generate(), base)
            finally:
                harness.close()

    def test_moe_sharded_placement(self):
        adapter, adapter_config = _load_adapter(MOE_LORA_PATH)
        v1 = _versioned_tensors(adapter, 1)
        v2 = _versioned_tensors(adapter, 2)
        prompt = MOE_LORA_TEST_PROMPTS[0]
        harness = StagedLoRATestHarness(
            self,
            model_path=MOE_BASE_MODEL_PATH,
            base_gpu_id=1,
            tp_size=2,
            max_loras_per_batch=1,
        )
        try:
            harness.stage("moe-policy", 1, v1, adapter_config)
            harness.activate("moe-policy", 1)
            before = harness.generate(adapter="moe-policy", prompt=prompt)
            harness.stage("moe-policy", 2, v2, adapter_config)
            self.assertEqual(
                harness.generate(adapter="moe-policy", prompt=prompt), before
            )
            harness.activate("moe-policy", 2)
            self.assertNotEqual(
                harness.generate(adapter="moe-policy", prompt=prompt), before
            )
        finally:
            harness.close()


if __name__ == "__main__":
    unittest.main()
