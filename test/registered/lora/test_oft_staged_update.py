"""End-to-end coverage for native two-phase OFT (staged) updates.

Mirrors test_lora_staged_update.py + test_lora_staged_update_tp.py (merged
into one file here, per Task 7b Step 1's instruction to create a single
test_oft_staged_update.py covering the base scenarios AND TP>1
consistency), adapted for StagedOFTManager (python/sglang/srt/oft/
staged_manager.py) instead of StagedLoRAManager.

Two structural differences from the LoRA sibling, both discovered by
reading the landed OFT staging code (not copied from an existing test --
flagged prominently in this task's report):

1. No established real small OFT (Orthogonal Finetuning) HF adapter repo
   was found anywhere in this codebase to reuse the way the LoRA test reuses
   "charent/self_cognition_Alice". Adapter checkpoints are synthesized here
   instead (see _oft_named_tensors/_write_local_oft_adapter) -- shape-correct
   and deterministic, but not a real trained adapter.

2. StagedOFTManager.activate_adapter hard-fails ("No serving slot is
   reserved for adapter uid=...") when memory_pool.uid_to_buffer_id has no
   entry for the target uid yet -- unlike StagedLoRAManager.activate_adapter,
   which tolerates a brand-new uid by registering self.configs/self.loras
   without touching the pool, deferring the physical admission to the next
   scheduler-driven fetch_new_loras (which loads straight from the
   already-fully-updated self.loras[uid] object). OFT has no such lazy
   fallback, so a brand-new adapter must already have a real buffer_id
   before any stage()/activate() call for it can succeed. The only way to
   get one (short of the legacy srt/peft/oft's /load_oft_adapter endpoint,
   which is hardcoded to the OLD OFTRef class and not wired for
   oft_impl=staged) is: boot the adapter via --peft-paths (a real on-disk
   checkpoint, admitted into self.configs/self.adapters/self.refs at boot)
   and then send one real /generate request naming it, which drives the
   scheduler's fetch_new_ofts/prepare_oft_batch admission and assigns the
   real buffer_id. See StagedOFTTestHarness.__init__'s warmup loop below.
"""

import json
import os
import socket
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from safetensors.torch import save_file
from transformers import AutoConfig

# Match SGLang's established distributed-weight test setup. CUDA's cuMem and
# NVLS transports are not valid for this trainer-plus-inference rank layout on
# every CI/Slurm node, and can fail the first NCCL broadcast with
# ncclUnhandledCudaError before any adapter code runs.
os.environ["NCCL_CUMEM_ENABLE"] = "0"
os.environ["NCCL_NVLS_ENABLE"] = "0"

from sglang.srt.utils import init_custom_process_group
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import terminate_process

# One file covering both the base (TP=1) scenarios and the TP=2 case (unlike
# the LoRA sibling, which splits those across test_lora_staged_update.py
# "2-gpu-large" and test_lora_staged_update_tp.py "4-gpu-h100"), so
# runner_config is sized for the larger TP=2 requirement throughout.
register_cuda_ci(est_time=900, stage="base-b", runner_config="4-gpu-h100")

MODEL_PATH = "Qwen/Qwen3-0.6B"
GROUP_NAME = "test_oft_stage_group"
PROMPT = "Hello, my name is"
# Single target module, kept deliberately simple: down_proj is row-parallel
# under TP (srt/oft/utils.py's ROW_PARALLELISM_LINEAR_OFT_NAMES), so it also
# exercises the TP-aware compact-weight slicing path
# (StagedOFTMemoryPool/_partition_and_precompute's is_row_parallel branch,
# and the regular admission path's load_oft_weight_to_buffer) for the TP=2
# test below, without needing per-head-dim bookkeeping that q/k/v/o_proj
# would require.
TARGET_MODULE = "down_proj"
BLOCK_SIZE = 32


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _adapter_config_dict() -> dict:
    return {
        "peft_type": "OFT",
        "target_modules": [TARGET_MODULE],
        "oft_block_size": BLOCK_SIZE,
    }


def _oft_named_tensors(num_layers: int, intermediate_size: int, seed: int) -> dict:
    """Raw checkpoint-name -> compact-OFT-weight tensors for TARGET_MODULE
    across every layer -- the SAME format StagedOFTManager.stage_adapter's
    named_tensors argument takes (see staged_manager.py's module docstring)
    and the SAME format a real adapter_model.safetensors file holds.

    Shape per layer is (num_blocks, block_size*(block_size-1)//2) -- the
    compact upper-triangular generator format precompute_oft_r (srt/oft/
    torch_ops/oft_ops.py) consumes -- with num_blocks derived from
    down_proj's INPUT dimension (intermediate_size), matching
    OFTMemoryPool.get_oft_R_shape's "OFT input rotation" convention. This is
    the FULL, unsharded size regardless of --tp-size: both
    load_oft_weight_to_buffer (regular admission) and
    _partition_and_precompute (staged updates) slice a row-parallel
    module's compact weight down to each TP rank's shard internally
    (module.slice_oft_r_weights), so the raw tensor supplied here -- on disk
    or over the staging NCCL broadcast -- must always be the unsharded
    original.

    Not a real trained OFT adapter -- see this file's module docstring for
    why (no reusable real small OFT HF repo was found).
    """
    assert intermediate_size % BLOCK_SIZE == 0, (
        f"intermediate_size={intermediate_size} must be a multiple of "
        f"BLOCK_SIZE={BLOCK_SIZE} for this fixture's compact OFT tensors to "
        "have a well-defined block count."
    )
    num_blocks = intermediate_size // BLOCK_SIZE
    n_elements = BLOCK_SIZE * (BLOCK_SIZE - 1) // 2
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer_idx in range(num_layers):
        name = f"model.layers.{layer_idx}.mlp.{TARGET_MODULE}.oft_R"
        # Small values: OFT's Cayley transform is only Neumann-approximated
        # (precompute_oft_r, num_neumann_terms=5) for small skew-symmetric
        # generators, matching real trained-adapter magnitudes.
        tensors[name] = (
            torch.randn((num_blocks, n_elements), generator=generator) * 0.02
        )
    return tensors


def _write_local_oft_adapter(root_dir: str, name: str, tensors: dict) -> str:
    """Write a real on-disk OFT adapter checkpoint (adapter_config.json +
    adapter_model.safetensors) -- the same file layout OFTConfig/
    OFTAdapter.initialize_weights read for a real HF PEFT-OFT checkpoint,
    just built locally instead of via snapshot_download (see this file's
    module docstring: no reusable real OFT repo exists to download)."""
    adapter_dir = os.path.join(root_dir, name)
    os.makedirs(adapter_dir, exist_ok=True)
    with open(
        os.path.join(adapter_dir, "adapter_config.json"), "w", encoding="utf-8"
    ) as config_file:
        json.dump(_adapter_config_dict(), config_file)
    save_file(tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))
    return adapter_dir


def _stage_payload(name, version, tensors, *, double_buffer=True):
    return {
        "names": list(tensors),
        "dtypes": [str(t.dtype).removeprefix("torch.") for t in tensors.values()],
        "shapes": [list(t.shape) for t in tensors.values()],
        "group_name": GROUP_NAME,
        "weight_version": str(version),
        "adapter_version": str(version),
        "load_format": "oft_adapter",
        "adapter_config": _adapter_config_dict(),
        "adapter_name": name,
        "double_buffer": double_buffer,
    }


class StagedOFTTestHarness:
    def __init__(
        self,
        testcase,
        *,
        adapters,
        model_path=MODEL_PATH,
        base_gpu_id=1,
        tp_size=1,
        max_ofts_per_batch=2,
        disable_cuda_graph=False,
        url=DEFAULT_URL_FOR_TEST,
    ):
        """``adapters`` maps adapter_name -> a local on-disk checkpoint
        directory (see _write_local_oft_adapter), admitted via --peft-paths
        at boot. ``max_ofts_per_batch`` must cover len(adapters) + 1 (the
        auto-registered base/identity uid=None slot, admitted eagerly at
        boot by StagedOFTManager.init_memory_pool's fetch_new_ofts({None}))
        or the two can evict each other via the pool's ordinary LRU
        admission (out of scope here -- see staged_manager.py's own
        docstring: eviction is "B1's existing multi-tenant admission and
        eviction (unaffected by this file)").
        """
        self.testcase = testcase
        self.model_path = model_path
        self.base_gpu_id = base_gpu_id
        self.tp_size = tp_size
        self.url = url
        self.master_port = _free_port()
        self.group = None

        args = [
            "--base-gpu-id",
            str(base_gpu_id),
            "--tp-size",
            str(tp_size),
            "--peft-method",
            "oft",
            "--oft-impl",
            "staged",
            # Deliberate: keeps the OFT rotation buffers in float32
            # regardless of the model's own bf16 dtype, so the bitwise
            # comparisons below (staged-update-then-activate vs. a fresh
            # boot reaching the same version) aren't also comparing bf16
            # rounding taken through two different code paths (the regular
            # boot-time disk-load path vs. the staged NCCL-load path).
            "--oft-dtype",
            "float32",
            "--max-oft-block-size",
            str(BLOCK_SIZE),
            "--peft-target-modules",
            TARGET_MODULE,
            "--max-ofts-per-batch",
            str(max_ofts_per_batch),
            "--mem-fraction-static",
            "0.6",
            "--log-level",
            "error",
        ]
        if adapters:
            args.append("--peft-paths")
            args.extend(f"{name}={path}" for name, path in adapters.items())
        if disable_cuda_graph:
            args.append("--disable-cuda-graph")
        self.process = popen_launch_server(
            model_path,
            url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=tuple(args),
        )
        self._init_group()

        # Warmup: admits each boot-loaded adapter into a real serving slot.
        # See this file's module docstring (point 2) for why this is
        # required for OFT but has no analogue in test_lora_staged_update.py.
        for name in adapters:
            self.generate(adapter=name)

    def _post(self, path, payload):
        response = requests.post(self.url + path, json=payload, timeout=300)
        response.raise_for_status()
        return response

    def _init_group(self):
        world_size = self.tp_size + 1
        payload = {
            "master_address": "127.0.0.1",
            "master_port": str(self.master_port),
            "rank_offset": self.base_gpu_id,
            "world_size": world_size,
            "group_name": GROUP_NAME,
            "backend": "nccl",
        }
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._post, "/init_weights_update_group", payload
            )
            self.group = init_custom_process_group(
                backend="nccl",
                init_method=f"tcp://127.0.0.1:{self.master_port}",
                world_size=world_size,
                rank=0,
                group_name=GROUP_NAME,
            )
            result = future.result().json()
        self.testcase.assertTrue(result["success"], result)

    def stage(self, name, version, tensors, *, double_buffer=True):
        payload = _stage_payload(name, version, tensors, double_buffer=double_buffer)
        tensors_cuda = {
            key: tensor.detach().clone().to("cuda:0") for key, tensor in tensors.items()
        }
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._post, "/update_adapter_from_distributed", payload
            )
            for tensor in tensors_cuda.values():
                torch.distributed.broadcast(tensor, src=0, group=self.group)
            torch.cuda.synchronize()
            response = future.result()
        body = response.json()
        self.testcase.assertTrue(body["success"], body)
        self.testcase.assertEqual(body["staged_adapter_version"], str(version))
        if not double_buffer:
            self.testcase.assertEqual(body["active_adapter_version"], str(version))
        return body

    def activate(self, name, version):
        body = self._post(
            "/activate_adapter_version",
            {
                "adapter_name": name,
                "adapter_version": str(version),
                "load_format": "oft_adapter",
            },
        ).json()
        self.testcase.assertTrue(body["success"], body)
        self.testcase.assertEqual(body["active_adapter_version"], str(version))
        return body

    def generate(self, *, adapter=None, prompt=PROMPT):
        payload = {
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": 24},
        }
        if adapter is not None:
            # OFT's per-request adapter field (single-active PEFT's
            # "adapter_path", not native LoRA's "lora_path" -- see
            # peft/tokenizer_hooks.py's _request_peft_path and
            # GenerateReqInput's adapter_path field; the value passed is the
            # adapter's --peft-paths NAME, not its on-disk path, since
            # tm.peft_ref_cache is keyed by name).
            payload["adapter_path"] = adapter
        body = self._post("/generate", payload).json()
        return body["output_ids"]

    def close(self):
        if self.group is not None:
            torch.distributed.destroy_process_group(self.group)
            self.group = None
        terminate_process(self.process)
        time.sleep(2)


class TestStagedOFTUpdate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.cuda.set_device(0)
        cls.tmp_dir = tempfile.mkdtemp(prefix="oft_staged_test_")
        hf_config = AutoConfig.from_pretrained(MODEL_PATH)
        cls.num_layers = hf_config.num_hidden_layers
        cls.intermediate_size = hf_config.intermediate_size
        assert cls.intermediate_size % (BLOCK_SIZE * 2) == 0, (
            f"{MODEL_PATH}'s intermediate_size={cls.intermediate_size} must "
            f"be a multiple of BLOCK_SIZE*2={BLOCK_SIZE * 2} so a TP=2 "
            "shard still divides evenly by BLOCK_SIZE."
        )

    def _tensors(self, seed):
        return _oft_named_tensors(self.num_layers, self.intermediate_size, seed)

    def _write_adapter(self, name, seed):
        return _write_local_oft_adapter(
            self.tmp_dir, f"{name}-{seed}", self._tensors(seed)
        )

    def _run_update(self, *, disable_cuda_graph=False, fresh_check=False):
        v1_dir = self._write_adapter("policy-a", 1)
        v2_tensors = self._tensors(2)
        harness = StagedOFTTestHarness(
            self,
            adapters={"policy-a": v1_dir},
            disable_cuda_graph=disable_cuda_graph,
        )
        try:
            before_a = harness.generate(adapter="policy-a")

            harness.stage("policy-a", 2, v2_tensors)
            self.assertEqual(harness.generate(adapter="policy-a"), before_a)

            harness.activate("policy-a", 2)
            after_a = harness.generate(adapter="policy-a")
            self.assertNotEqual(after_a, before_a)
        finally:
            harness.close()

        if fresh_check:
            fresh = StagedOFTTestHarness(self, adapters={"policy-a": v1_dir})
            try:
                fresh.stage("policy-a", 2, v2_tensors)
                fresh.activate("policy-a", 2)
                self.assertEqual(fresh.generate(adapter="policy-a"), after_a)
            finally:
                fresh.close()

    def test_single_gpu(self):
        self._run_update(fresh_check=True)

    def test_decode_graph_on_and_off(self):
        self._run_update(disable_cuda_graph=False)
        self._run_update(disable_cuda_graph=True)

    def test_hidden_slot_never_evicts(self):
        """Covers both: the hidden staging slot never counts against
        available_serving_slots() (StagedOFTMemoryPool.available_
        serving_slots() always returns max_ofts_per_batch, never +1 -- see
        Task 2's unit coverage in test_oft_staging_backend.py), and a
        second, unrelated resident adapter's output is unaffected by
        another adapter's stage/activate cycle (the actual multi-tenancy +
        staging bug this whole effort exists to fix).

        No HTTP endpoint exposes available_serving_slots() directly (unlike
        the pool-internal unit test), so this is exercised black-box: with
        max_ofts_per_batch=3 set EXACTLY to the number of concurrently
        resident uids needed (base/None + policy-a + policy-b, no slack),
        staging+activating policy-a must not disturb policy-b or the base
        output -- if the hidden slot were incorrectly borrowing one of
        those 3 serving slots, admitting all three upfront would already
        have evicted one of them.
        """
        a_dir = self._write_adapter("policy-a", 1)
        b_dir = self._write_adapter("policy-b", 1)
        v2_tensors = self._tensors(2)
        harness = StagedOFTTestHarness(
            self,
            adapters={"policy-a": a_dir, "policy-b": b_dir},
            max_ofts_per_batch=3,
        )
        try:
            before_a = harness.generate(adapter="policy-a")
            before_b = harness.generate(adapter="policy-b")
            before_base = harness.generate()

            harness.stage("policy-a", 2, v2_tensors)
            self.assertEqual(harness.generate(adapter="policy-a"), before_a)
            self.assertEqual(harness.generate(adapter="policy-b"), before_b)
            self.assertEqual(harness.generate(), before_base)

            harness.activate("policy-a", 2)
            after_a = harness.generate(adapter="policy-a")
            self.assertNotEqual(after_a, before_a)
            self.assertEqual(harness.generate(adapter="policy-b"), before_b)
            self.assertEqual(harness.generate(), before_base)
        finally:
            harness.close()

    def test_tp2(self):
        """TP>1 activation consistency: every TP rank must reach the same
        activated version, or the update must fail cleanly (no partial /
        inconsistent state across ranks). Mirrors test_lora_staged_update_
        tp.py's test_tp2: a second TP=2 boot reaching v2 directly is the
        correctness oracle (not a TP=1 boot -- the synthetic v2 weights can
        amplify tiny TP-vs-non-TP rounding differences that are irrelevant
        to the staging behavior under test), so any rank landing on a
        stale/mismatched version after activate() would show up as a
        divergent generation vs. this oracle.
        """
        v1_dir = self._write_adapter("policy-a", 1)
        v2_tensors = self._tensors(2)
        tp2 = StagedOFTTestHarness(
            self,
            adapters={"policy-a": v1_dir},
            base_gpu_id=1,
            tp_size=2,
        )
        try:
            before = tp2.generate(adapter="policy-a")
            tp2.stage("policy-a", 2, v2_tensors)
            self.assertEqual(tp2.generate(adapter="policy-a"), before)
            tp2.activate("policy-a", 2)
            tp2_v2 = tp2.generate(adapter="policy-a")
            self.assertNotEqual(tp2_v2, before)
        finally:
            tp2.close()

        reference = StagedOFTTestHarness(
            self,
            adapters={"policy-a": v1_dir},
            base_gpu_id=1,
            tp_size=2,
        )
        try:
            reference.stage("policy-a", 2, v2_tensors)
            reference.activate("policy-a", 2)
            self.assertEqual(reference.generate(adapter="policy-a"), tp2_v2)
        finally:
            reference.close()


if __name__ == "__main__":
    unittest.main()
