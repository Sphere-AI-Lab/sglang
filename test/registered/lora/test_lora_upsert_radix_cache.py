"""Regression coverage for an in-place LoRA upsert over the plain
/load_lora_adapter_from_distributed RPC route (the RL weight-sync path, as
opposed to the native staged double-buffer protocol covered by
test_lora_staged_update.py).

lora_registry.register_or_reuse used to never bump ``version`` on an upsert
reuse, so schedule_batch's ``_extend_lora_extra_key`` (which folds
``lora_version`` into the radix cache key) rendered the same key before and
after an in-place weight refresh of the same adapter name. A prompt re-issued
after the upsert could then be served from a KV prefix cached under the
pre-upsert weights.
"""

import json
import os
import socket
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

# Match the established distributed-weight test setup: cuMem/NVLS transports
# are not valid for this trainer-plus-inference rank layout on every CI/Slurm
# node and can fail the first NCCL broadcast with ncclUnhandledCudaError.
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

register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_REPO = "charent/self_cognition_Alice"
GROUP_NAME = "test_lora_upsert_distributed_group"
PROMPT = "Hello, my name is"
ADAPTER_NAME = "policy_cache_check"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _adapter_tensors(adapter, *, bump: bool):
    """The base adapter tensors, nudged on lora_B so a real upsert changes
    the adapter's output deterministically (mirrors
    test_lora_staged_update.py's _versioned_tensors)."""
    tensors = {
        name: tensor.detach().clone().to("cuda:0") for name, tensor in adapter.items()
    }
    if bump:
        for name in tensors:
            if "lora_B" in name:
                tensors[name].add_(0.5)
    return tensors


class TestLoRAUpsertInvalidatesRadixCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        adapter_dir = snapshot_download(
            repo_id=LORA_REPO,
            allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
        )
        cls.adapter = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
        with open(
            os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8"
        ) as config_file:
            cls.adapter_config = json.load(config_file)

    def test_upsert_invalidates_radix_cache(self):
        """Regression test: reusing the SAME prompt across an in-place
        upsert of the SAME adapter name (over /load_lora_adapter_from_
        distributed) must be a full cache miss for the prompt -- not served
        from a prefix cached under the pre-upsert weights."""
        print(
            "[Test]Testing that an in-place LoRA upsert (from_distributed) "
            "invalidates the radix cache..."
        )
        torch.cuda.set_device(0)
        master_port = _free_port()

        process = popen_launch_server(
            MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--base-gpu-id",
                "1",
                "--enable-lora",
                "--max-lora-rank",
                "64",
                "--lora-target-modules",
                "all",
                "--mem-fraction-static",
                "0.6",
                "--log-level",
                "error",
            ),
        )
        group = None
        try:
            world_size = 2
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    requests.post,
                    DEFAULT_URL_FOR_TEST + "/init_weights_update_group",
                    json={
                        "master_address": "127.0.0.1",
                        "master_port": str(master_port),
                        "rank_offset": 1,
                        "world_size": world_size,
                        "group_name": GROUP_NAME,
                        "backend": "nccl",
                    },
                    timeout=300,
                )
                group = init_custom_process_group(
                    backend="nccl",
                    init_method=f"tcp://127.0.0.1:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=GROUP_NAME,
                )
                init_response = future.result()
            init_response.raise_for_status()
            self.assertTrue(init_response.json()["success"], init_response.json())

            def _load(tensors, upsert):
                payload = {
                    "lora_name": ADAPTER_NAME,
                    "config_dict": self.adapter_config,
                    "names": list(tensors),
                    "dtypes": [
                        str(t.dtype).removeprefix("torch.") for t in tensors.values()
                    ],
                    "shapes": [list(t.shape) for t in tensors.values()],
                    "group_name": GROUP_NAME,
                    "upsert": upsert,
                }
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        requests.post,
                        DEFAULT_URL_FOR_TEST + "/load_lora_adapter_from_distributed",
                        json=payload,
                        timeout=300,
                    )
                    for tensor in tensors.values():
                        torch.distributed.broadcast(tensor, src=0, group=group)
                    torch.cuda.synchronize()
                    response = future.result()
                response.raise_for_status()
                return response.json()

            def _generate():
                return requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "text": PROMPT,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                        "lora_path": ADAPTER_NAME,
                    },
                    timeout=300,
                ).json()

            v1 = _adapter_tensors(self.adapter, bump=False)
            result = _load(v1, upsert=False)
            self.assertTrue(result["success"], result)

            body_v1 = _generate()
            self.assertTrue(body_v1["text"], "Generation before upsert produced no text")

            v2 = _adapter_tensors(self.adapter, bump=True)
            result = _load(v2, upsert=True)
            self.assertTrue(result["success"], result)

            # SAME prompt, SAME adapter name -- if lora_version weren't bumped
            # on upsert, the radix cache key would be unchanged, so the
            # scheduler's prefix match would hit the prompt's KV cached under
            # the pre-upsert weights instead of recomputing it fresh. This is
            # the direct, deterministic signature of the bug: with the fix,
            # the key changes and the prompt must be a full cache miss.
            body_v2 = _generate()
            self.assertTrue(body_v2["text"], "Generation after upsert produced no text")
            self.assertEqual(
                body_v2["meta_info"]["cached_tokens"],
                0,
                "Re-issuing the SAME prompt after an in-place LoRA upsert hit "
                "the radix cache for some prompt tokens -- the scheduler "
                "served (part of) the pre-upsert weights' cached KV prefix "
                "instead of recomputing it under the new weights (i.e. "
                "lora_version was not bumped on upsert).",
            )
        finally:
            if group is not None:
                torch.distributed.destroy_process_group(group)
            terminate_process(process)


if __name__ == "__main__":
    unittest.main()
