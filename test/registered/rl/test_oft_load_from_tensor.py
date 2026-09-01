"""GPU integration tests for the native OFT adapter RPC
(load_oft_adapter_from_tensors / upsert / LRU eviction / multi-adapter
residency), added end-to-end across Tasks 1-7 of the
2026-08-31-oft-native-adapter-rpc plan: tokenizer-manager handlers ->
scheduler/tp_worker dispatch -> OFTManager/ModelRunner GPU-side admission ->
Engine surface.

Modeled directly on test/registered/rl/test_lora_load_from_tensor.py (LoRA's
equivalent native-RPC test template). Tensor/config synthesis is adapted from
test/registered/lora/test_oft_staged_update.py's _oft_named_tensors /
_adapter_config_dict helpers: per that file's own module docstring, no
reusable small real trained OFT HF adapter repo exists to download the way
the LoRA test downloads "charent/self_cognition_Alice".

NOT the same mechanism as test/registered/rl/test_oft_sibling_streamed_update.py
(that file covers the OLD legacy update_weights_from_tensor(load_format=
"oft_adapter") streamed path). This file exercises the NEW native RPC surface
added in Task 7 (engine.load_oft_adapter_from_tensors / unload_oft_adapter),
which -- unlike the legacy single-active path -- allows multiple OFT adapters
to be concurrently resident (Task 6's removal of the single-active
restriction).

KNOWN BUGS (see this task's report for full evidence/tracebacks -- reported
to the controller, not fixed here per this task's charter):
  1. OFTManager.load_adapter_from_tensors (srt/oft/oft_manager.py) passes a
     raw dict where the shared _write_streamed_oft_tensors/
     _partition_expert_oft_tensors (srt/oft/streamed_weight_loader.py)
     expect List[Tuple[str, Tensor]] and iterate it directly -- every call
     currently fails with "ValueError: too many values to unpack (expected
     2)". This is the first failure every scenario below hits.
  2. There is no release counterpart to peft/tokenizer_hooks.py's
     `tm.peft_registry.acquire_with_version(path)` anywhere in the codebase
     (unlike LoRA's TokenizerManager._finalize_lora_lease / lora_registry
     .release). Once any /generate request names an adapter_path,
     peft_registry.wait_for_unload() -- called by both unload_oft_adapter
     and the tokenizer-side max_loaded_ofts LRU-eviction loop -- blocks
     forever (confirmed by direct reproduction + py-spy). Deliberately
     NEVER call engine.unload_oft_adapter() after a generate() call in this
     file, to keep this suite from hanging CI indefinitely regardless of
     bug #1's fix status; cls.engine's teardown is a process kill, not a
     graceful unload.
  3. OFTMemoryPool permanently reserves buffer slot 0 for the base/identity
     (uid=None) placeholder at boot, leaving only max_ofts_per_batch - 1
     slots for real adapters via allocate_buffer_slot() (srt/oft/mem_pool.py),
     which -- unlike the regular scheduler-driven batch-admission path --
     has no eviction fallback and raises once full. With this file's
     max_ofts_per_batch=4/max_loaded_ofts=4 config, the pool physically
     exhausts on the 4th real adapter, before the tokenizer-side
     max_loaded_ofts=4 registry cap is ever reached -- so
     test_lru_eviction_past_max_loaded_ofts's 5th-load eviction can never
     be exercised as specified.

Despite bug #1 blocking every scenario below in the pristine tree, a
temporary local-only patch around bug #1 (reverted, never committed) was
used to verify past it: test_multi_adapter_concurrent_residency's two
adapters serve correctly and independently, and test_upsert_refresh's core
mechanism -- the production RL use case of repeatedly refreshing one
adapter name in place (resolve_or_reuse/refresh) -- is correct: adapter_id
stays stable across rounds, the registry never grows, and each round's new
weights visibly change generation output. See this task's report for the
patched run transcripts.
"""

import unittest

import torch
from transformers import AutoConfig

import sglang as sgl
from sglang.srt.managers.io_struct import LoadOFTAdapterFromTensorsReqInput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-large")

MODEL_PATH = "Qwen/Qwen3-0.6B"
# Single target module, row-parallel under TP (srt/oft/utils.py's
# ROW_PARALLELISM_LINEAR_OFT_NAMES) but that doesn't matter at tp_size=1
# here; kept for consistency with test_oft_staged_update.py's fixture.
TARGET_MODULE = "down_proj"
BLOCK_SIZE = 32
TEST_PROMPT = "Hello, my name is"
MAX_NEW_TOKENS = 16

# Engine boot config specified by this task's brief: native sibling OFT,
# block size 32, single target module, and small loaded/per-batch caps so the
# LRU-eviction and multi-adapter-residency scenarios below can actually
# exercise their limits without booting an oversized pool.
ENGINE_KWARGS = dict(
    model_path=MODEL_PATH,
    peft_method="oft",
    oft_impl="sibling",
    max_oft_block_size=BLOCK_SIZE,
    peft_target_modules=[TARGET_MODULE],
    max_loaded_ofts=4,
    max_ofts_per_batch=4,
    mem_fraction_static=0.6,
    log_level="error",
)


def _adapter_config_dict() -> dict:
    """Adapted from test_oft_staged_update.py's _adapter_config_dict."""
    return {
        "peft_type": "OFT",
        "target_modules": [TARGET_MODULE],
        "oft_block_size": BLOCK_SIZE,
    }


def _oft_named_tensors(num_layers: int, intermediate_size: int, seed: int) -> dict:
    """Adapted from test_oft_staged_update.py's _oft_named_tensors: raw
    checkpoint-name -> compact-OFT-weight tensors for TARGET_MODULE across
    every layer, in the same format a real adapter_model.safetensors file
    holds. Not a real trained adapter -- see this file's module docstring."""
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
        tensors[name] = (
            torch.randn((num_blocks, n_elements), generator=generator) * 0.02
        )
    return tensors


class TestOFTLoadFromTensor(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        hf_config = AutoConfig.from_pretrained(MODEL_PATH)
        cls.num_layers = hf_config.num_hidden_layers
        cls.intermediate_size = hf_config.intermediate_size
        cls.engine = sgl.Engine(**ENGINE_KWARGS)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def _tensors(self, seed: int) -> dict:
        return _oft_named_tensors(self.num_layers, self.intermediate_size, seed)

    @staticmethod
    def _generate(engine, adapter_name=None) -> str:
        output = engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0},
            adapter_path=[adapter_name] if adapter_name is not None else None,
        )
        return output[0]["text"]

    def test_fresh_load_and_generate(self):
        print("[Test]Testing fresh OFT adapter load + generate...")
        name = "oft_fresh_load"
        result = self.engine.load_oft_adapter_from_tensors(
            adapter_name=name,
            tensors=self._tensors(seed=1),
            config_dict=_adapter_config_dict(),
        )
        self.assertTrue(
            result.success, f"Failed to load OFT adapter from tensors: {result.error_message}"
        )
        # No unload_oft_adapter call here -- see the module docstring: it
        # hangs forever (peft_registry.wait_for_unload never returns) after
        # any generate() call names this adapter, a confirmed separate bug.
        # cls.engine's teardown (process kill, not a graceful unload) cleans
        # this adapter up at tearDownClass.
        base_text = self._generate(self.engine)
        adapter_text = self._generate(self.engine, adapter_name=name)
        print(f"[Without OFT] {base_text}")
        print(f"[With OFT]    {adapter_text}")
        self.assertTrue(
            adapter_text, "Generation with the freshly-loaded OFT adapter produced no text"
        )
        self.assertNotEqual(
            base_text,
            adapter_text,
            "OFT-adapted generation should differ from the unadapted base output "
            "(random-but-nonzero OFT rotation weights should perturb decoding)",
        )

    def _upsert(self, engine, name: str, seed: int):
        """Send one in-place-refresh LoadOFTAdapterFromTensorsReqInput
        (upsert=True) directly to the tokenizer-manager method, exactly
        what engine.load_oft_adapter_from_tensors() builds internally.

        engine.load_oft_adapter_from_tensors() (srt/entrypoints/engine.py)
        has no `upsert` parameter, unlike the LoadOFTAdapterFromTensorsReqInput
        it builds internally (and unlike the tokenizer-manager/oft_manager
        route underneath, which fully supports upsert=True on this route --
        confirmed by reading tokenizer_control_mixin.load_oft_adapter_from_tensors
        and oft/oft_manager.py's load_adapter_from_tensors; this differs from
        LoRA's from_tensors route, which explicitly *rejects* upsert). This
        looks like a small Engine-surface API gap left over from Task 7 --
        flagged in this task's report -- so the request is constructed
        directly here to exercise upsert=True end to end.
        """
        serialized = engine._serialize_tensors_per_rank(
            _oft_named_tensors(self.num_layers, self.intermediate_size, seed), None
        )
        req = LoadOFTAdapterFromTensorsReqInput(
            adapter_name=name,
            config_dict=_adapter_config_dict(),
            serialized_named_tensors=serialized,
            upsert=True,
        )
        return engine.loop.run_until_complete(
            engine.tokenizer_manager.load_oft_adapter_from_tensors(req, None)
        )

    def test_upsert_refresh(self):
        """The real production RL use case: a training loop repeatedly
        pushes new weight versions for the SAME adapter name in place (e.g.
        "policy" refreshed step after step via upsert=True) -- this matters
        more than the LRU-eviction/multi-named-adapter scenarios, which
        cover a different (secondary) use case. Runs several in-place
        rounds (not just one load-then-upsert pair) and, for every round,
        verifies: the update actually changes generation output, the
        adapter_id stays stable (resolve_or_reuse/refresh reusing the same
        identity, never minting a new one), and the tokenizer-side registry
        never grows past a single entry. Uses a dedicated engine with the
        tightest config the current implementation's own invariants allow
        (max_loaded_ofts=max_ofts_per_batch=2 -- validate_peft_args asserts
        max_loaded_ofts >= max_ofts_per_batch, and bug #3 in the module
        docstring means max_ofts_per_batch=1 would leave zero real-adapter
        slots even for the very first load, so 2 is the practical floor, not
        1) to prove this growth-free in-place-update path needs no capacity
        headroom beyond a single adapter slot and never depends on (or
        trips) LRU eviction.
        """
        print("[Test]Testing repeated in-place OFT adapter upserts (RL training-loop pattern)...")
        name = "policy"
        NUM_ROUNDS = 4  # initial load + 3 further in-place refreshes
        # No base_gpu_id override -- this suite is registered on a
        # "1-gpu-large" CI runner_config (matching test_lora_load_from_tensor
        # .py's registration exactly, per the brief), so this dedicated
        # engine must share GPU 0 with cls.engine, not assume a 2nd GPU. A
        # lower mem_fraction_static than ENGINE_KWARGS's 0.6 keeps the two
        # engines' combined static reservation on one device well under 1.0.
        #
        # Deliberately never calls engine.shutdown(): Engine.shutdown() calls
        # kill_process_tree(os.getpid(), ...) (srt/utils/common.py), which
        # kills EVERY child process of the current Python process, not just
        # this engine's own subprocesses -- so it would also kill cls.engine's
        # scheduler/detokenizer, silently breaking every later test in this
        # class (confirmed by direct reproduction: cls.engine's scheduler
        # showed up defunct, and the next test hung forever inside
        # run_until_complete waiting on a reply from a dead scheduler).
        # test_lora_load_from_tensor.py's test_lora_lru_eviction has the
        # exact same shape (a second same-process engine) and likewise never
        # calls test_engine.shutdown() for this reason; this engine's
        # process is reaped when cls.engine's tearDownClass shutdown runs.
        engine = sgl.Engine(
            **{
                **ENGINE_KWARGS,
                "max_loaded_ofts": 2,
                "max_ofts_per_batch": 2,
                "mem_fraction_static": 0.15,
            }
        )
        prev_text = self._generate(engine)  # base (no adapter) output
        prev_adapter_id = None
        for round_idx in range(NUM_ROUNDS):
            seed = 200 + round_idx
            if round_idx == 0:
                result = engine.load_oft_adapter_from_tensors(
                    adapter_name=name,
                    tensors=_oft_named_tensors(
                        self.num_layers, self.intermediate_size, seed
                    ),
                    config_dict=_adapter_config_dict(),
                )
            else:
                result = self._upsert(engine, name, seed)
            self.assertTrue(
                result.success,
                f"Round {round_idx}: "
                f"{'initial load' if round_idx == 0 else 'in-place upsert'} "
                f"failed: {result.error_message}",
            )

            all_adapters = engine.tokenizer_manager.peft_registry.get_all_adapters()
            self.assertEqual(
                list(all_adapters.keys()),
                [name],
                f"Round {round_idx}: registry should hold exactly one entry "
                f"({name!r}) throughout repeated in-place updates -- growth "
                f"would mean this path isn't actually update-in-place -- "
                f"got {list(all_adapters)}",
            )
            adapter_id = all_adapters[name].adapter_id
            if prev_adapter_id is None:
                prev_adapter_id = adapter_id
            else:
                self.assertEqual(
                    adapter_id,
                    prev_adapter_id,
                    f"Round {round_idx}: in-place upsert must reuse the same "
                    f"adapter_id via resolve_or_reuse/refresh, got a new id "
                    f"{adapter_id!r} != {prev_adapter_id!r}",
                )

            text = self._generate(engine, adapter_name=name)
            print(f"[Round {round_idx}] {text!r}")
            self.assertTrue(text, f"Round {round_idx}: generation produced no text")
            self.assertNotEqual(
                text,
                prev_text,
                f"Round {round_idx}: this update's new weight values produced "
                f"no observable change in generation output vs. the previous "
                f"round -- the in-place refresh may not have actually taken "
                f"effect on the serving weights",
            )
            prev_text = text

    def test_lru_eviction_past_max_loaded_ofts(self):
        print("[Test]Testing OFT adapter LRU eviction past --max-loaded-ofts...")
        max_loaded_ofts = ENGINE_KWARGS["max_loaded_ofts"]
        # Dedicated engine (sharing GPU 0 with cls.engine -- this suite is
        # registered on a "1-gpu-large" CI runner_config, so no 2nd GPU is
        # assumed) so this capacity-boundary scenario is fully isolated from
        # whatever the other tests in this class have loaded/unloaded on the
        # shared cls.engine -- mirrors test_lora_load_from_tensor.py's
        # test_lora_lru_eviction, which likewise spins up its own test_engine
        # (also without a base_gpu_id override) for this reason. Lower
        # mem_fraction_static than ENGINE_KWARGS's 0.6 so this doesn't starve
        # cls.engine's KV pool while both are resident on GPU 0.
        #
        # Deliberately never calls engine.shutdown() -- see test_upsert_
        # refresh's identical note: it kills every child process of the
        # current Python process (kill_process_tree(os.getpid(), ...)), not
        # just this engine's own subprocesses, so it would take down
        # cls.engine too and hang every later test. test_lora_lru_eviction
        # has the same shape and likewise never shuts its test_engine down.
        engine = sgl.Engine(**{**ENGINE_KWARGS, "mem_fraction_static": 0.15})
        names = [f"oft_lru_{i}" for i in range(max_loaded_ofts + 1)]
        for i, name in enumerate(names):
            print(f"[Test]Loading OFT adapter {i + 1}/{len(names)}: {name}")
            result = engine.load_oft_adapter_from_tensors(
                adapter_name=name,
                tensors=_oft_named_tensors(
                    self.num_layers, self.intermediate_size, seed=100 + i
                ),
                config_dict=_adapter_config_dict(),
            )
            self.assertTrue(
                result.success, f"Failed to load OFT adapter {name}: {result.error_message}"
            )

        all_adapters = engine.tokenizer_manager.peft_registry.get_all_adapters()
        print(f"[Test]Adapters resident after loading {len(names)}: {list(all_adapters)}")
        self.assertLessEqual(
            len(all_adapters),
            max_loaded_ofts,
            f"OFT registry should never exceed max_loaded_ofts={max_loaded_ofts}, "
            f"got {len(all_adapters)}: {list(all_adapters)}",
        )
        self.assertNotIn(
            names[0],
            all_adapters,
            f"Least-recently-used adapter {names[0]!r} should have been evicted",
        )
        for name in names[1:]:
            self.assertIn(
                name, all_adapters, f"Adapter {name!r} should still be resident"
            )

    def test_multi_adapter_concurrent_residency(self):
        print("[Test]Testing concurrent residency of multiple OFT adapters...")
        name_a, name_b = "oft_multi_a", "oft_multi_b"
        result_a = self.engine.load_oft_adapter_from_tensors(
            adapter_name=name_a,
            tensors=self._tensors(seed=20),
            config_dict=_adapter_config_dict(),
        )
        self.assertTrue(result_a.success, f"Failed to load {name_a}: {result_a.error_message}")
        result_b = self.engine.load_oft_adapter_from_tensors(
            adapter_name=name_b,
            tensors=self._tensors(seed=21),
            config_dict=_adapter_config_dict(),
        )
        self.assertTrue(result_b.success, f"Failed to load {name_b}: {result_b.error_message}")

        # Neither adapter is unloaded -- both must stay resident and
        # generatable at once (the capability Task 6 newly unlocked by
        # removing the old single-active restriction), and see the module
        # docstring's note on why this test never calls unload_oft_adapter.
        text_a = self._generate(self.engine, adapter_name=name_a)
        text_b = self._generate(self.engine, adapter_name=name_b)
        print(f"[Adapter A] {text_a}")
        print(f"[Adapter B] {text_b}")
        self.assertTrue(text_a, f"Generation with {name_a} produced no text")
        self.assertTrue(text_b, f"Generation with {name_b} produced no text")

        # Re-issuing against A after serving B must reproduce the same
        # output: B's residency/serving must not have disturbed A.
        text_a_again = self._generate(self.engine, adapter_name=name_a)
        self.assertEqual(
            text_a,
            text_a_again,
            f"{name_a}'s output changed after serving {name_b} concurrently -- "
            "adapters are not independently resident",
        )


if __name__ == "__main__":
    unittest.main()
