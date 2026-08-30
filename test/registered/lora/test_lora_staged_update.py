"""End-to-end coverage for native two-phase LoRA updates."""

import json
import os
import socket
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from sglang.srt.utils import init_custom_process_group
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import terminate_process

register_cuda_ci(est_time=600, stage="base-b", runner_config="2-gpu-large")

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_REPO = "charent/self_cognition_Alice"
GROUP_NAME = "test_lora_stage_group"
PROMPT = "Hello, my name is"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _versioned_tensors(adapter, version: int):
    source = adapter.state_dict() if hasattr(adapter, "state_dict") else adapter
    tensors = {
        name: tensor.detach().clone().to("cuda:0")
        for name, tensor in source.items()
    }
    if version == 2:
        for name in sorted(tensors):
            if "lora_B" in name:
                tensors[name].add_(0.125)
    return tensors


def _stage_payload(name, version, tensors, adapter_config, *, double_buffer=True):
    return {
        "names": list(tensors),
        "dtypes": [
            str(t.dtype).removeprefix("torch.") for t in tensors.values()
        ],
        "shapes": [list(t.shape) for t in tensors.values()],
        "group_name": GROUP_NAME,
        "weight_version": str(version),
        "adapter_version": str(version),
        "load_format": "lora_adapter",
        "adapter_config": adapter_config,
        "adapter_name": name,
        "double_buffer": double_buffer,
    }


class StagedLoRATestHarness:
    def __init__(
        self,
        testcase,
        *,
        model_path=MODEL_PATH,
        base_gpu_id=1,
        tp_size=1,
        max_loras_per_batch=2,
        disable_cuda_graph=False,
        url=DEFAULT_URL_FOR_TEST,
    ):
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
            "--enable-lora",
            "--enable-lora-staging",
            "--max-lora-rank",
            "64",
            "--lora-target-modules",
            "all",
            "--max-loras-per-batch",
            str(max_loras_per_batch),
            "--mem-fraction-static",
            "0.6",
            "--log-level",
            "error",
        ]
        if disable_cuda_graph:
            args.append("--disable-cuda-graph")
        self.process = popen_launch_server(
            model_path,
            url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=tuple(args),
        )
        self._init_group()

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

    def stage(self, name, version, tensors, adapter_config, *, double_buffer=True):
        payload = _stage_payload(
            name,
            version,
            tensors,
            adapter_config,
            double_buffer=double_buffer,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._post, "/update_adapter_from_distributed", payload
            )
            for tensor in tensors.values():
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
                "load_format": "lora_adapter",
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
            payload["lora_path"] = adapter
        body = self._post("/generate", payload).json()
        return body["output_ids"]

    def close(self):
        if self.group is not None:
            torch.distributed.destroy_process_group(self.group)
            self.group = None
        terminate_process(self.process)
        time.sleep(2)


class TestStagedLoRAUpdate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.cuda.set_device(0)
        adapter_dir = snapshot_download(
            repo_id=LORA_REPO,
            allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
        )
        cls.adapter = load_file(
            os.path.join(adapter_dir, "adapter_model.safetensors")
        )
        with open(
            os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8"
        ) as config_file:
            cls.adapter_config = json.load(config_file)

    def _run_update(self, *, disable_cuda_graph=False, fresh_check=False):
        v1 = _versioned_tensors(self.adapter, 1)
        v2 = _versioned_tensors(self.adapter, 2)
        harness = StagedLoRATestHarness(
            self, disable_cuda_graph=disable_cuda_graph
        )
        try:
            harness.stage("policy-b", 1, v1, self.adapter_config)
            harness.activate("policy-b", 1)
            harness.stage("policy-a", 1, v1, self.adapter_config)
            harness.activate("policy-a", 1)

            before_a = harness.generate(adapter="policy-a")
            before_b = harness.generate(adapter="policy-b")
            before_base = harness.generate()

            harness.stage("policy-a", 2, v2, self.adapter_config)
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

        if fresh_check:
            fresh = StagedLoRATestHarness(self)
            try:
                fresh.stage("policy-a", 2, v2, self.adapter_config)
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
        v1 = _versioned_tensors(self.adapter, 1)
        v2 = _versioned_tensors(self.adapter, 2)
        harness = StagedLoRATestHarness(self, max_loras_per_batch=2)
        try:
            for name in ("policy-a", "policy-b"):
                harness.stage(name, 1, v1, self.adapter_config)
                harness.activate(name, 1)
            before_a = harness.generate(adapter="policy-a")
            before_b = harness.generate(adapter="policy-b")
            harness.stage("policy-a", 2, v2, self.adapter_config)
            self.assertEqual(harness.generate(adapter="policy-a"), before_a)
            self.assertEqual(harness.generate(adapter="policy-b"), before_b)
        finally:
            harness.close()


if __name__ == "__main__":
    unittest.main()
