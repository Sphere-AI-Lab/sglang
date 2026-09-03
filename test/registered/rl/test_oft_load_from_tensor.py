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

NOT the same mechanism as the OLD legacy update_weights_from_tensor(load_format=
"oft_adapter") streamed path (retired in Task 9; formerly covered by
test/registered/rl/test_oft_sibling_streamed_update.py, deleted in that same
task). This file exercises the NEW native RPC surface added in Task 7
(engine.load_oft_adapter_from_tensors / unload_oft_adapter), which -- unlike
the legacy single-active path -- allows multiple OFT adapters to be
concurrently resident (Task 6's removal of the single-active restriction).

HISTORY: this file's docstring used to document three known bugs against the
native RPC path (a dict/list mismatch in OFTManager.load_adapter_from_tensors,
a missing release counterpart for oft_registry.acquire_with_version that made
unload_oft_adapter hang forever after any generate() call, and
OFTMemoryPool's eviction-free hard-fail once its buffer pool filled). All
three were fixed and reviewed earlier in this branch's history: the dict/list
normalization landed in OFTManager.load_adapter_from_tensors,
OFTTokenizerMixin.finalize_oft_lease now releases every request's
adapter lease on every terminal path, and allocate_buffer_slot_with_eviction
added LRU eviction to the native admission path. Per that last fix, tests
below now call engine.unload_oft_adapter() after generate() (see
test_fresh_load_and_generate) and it completes promptly instead of hanging.
"""

import unittest

import torch
from transformers import AutoConfig

import sglang as sgl
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
    enable_oft=True,
    oft_impl="sibling",
    max_oft_block_size=BLOCK_SIZE,
    oft_target_modules=[TARGET_MODULE],
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

        # Regression guard: this used to hang forever (oft_registry.
        # wait_for_unload never returned) after any generate() call named
        # this adapter, because nothing released the request's adapter
        # lease. OFTTokenizerMixin.finalize_oft_lease now releases it on
        # every terminal request path, so this must now complete promptly.
        unload_result = self.engine.unload_oft_adapter(name)
        self.assertTrue(
            unload_result.success,
            f"Failed to unload OFT adapter after generate(): {unload_result.error_message}",
        )

    def _upsert(self, engine, name: str, seed: int):
        """One in-place-refresh load via engine.load_oft_adapter_from_tensors
        (upsert=True). OFT's from_tensors route fully supports upsert=True
        (unlike LoRA's, which explicitly rejects it)."""
        return engine.load_oft_adapter_from_tensors(
            adapter_name=name,
            tensors=_oft_named_tensors(self.num_layers, self.intermediate_size, seed),
            config_dict=_adapter_config_dict(),
            upsert=True,
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
        (max_loaded_ofts=max_ofts_per_batch=2 -- buffer slot 0 is always the
        identity placeholder, so max_ofts_per_batch=1 would leave zero
        real-adapter slots even for the very first load, making 2 the
        practical floor, not 1) to prove this growth-free in-place-update
        path needs no capacity headroom beyond a single adapter slot and
        never depends on (or trips) LRU eviction.
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

            all_adapters = engine.tokenizer_manager.oft_registry.get_all_adapters()
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

    def test_upsert_invalidates_radix_cache(self):
        """Regression test for C2: an in-place upsert must bump
        adapter_version so its radix cache key changes -- otherwise
        re-issuing the SAME prompt after an upsert can be served from a
        stale KV prefix cached under the pre-upsert weights' key (see
        weight_updater.py's documented invariant: KV produced under version
        k lives under a different radix key than requests at k+1,
        specifically so this can never happen).

        Unlike test_upsert_refresh (which uses a different prompt each
        round, so a stale-cache hit would never be observable there), this
        reuses the exact same prompt/adapter_name across both rounds, so a
        cache hit is directly observable as unchanged output despite the
        weights differing.
        """
        print("[Test]Testing that an in-place OFT upsert invalidates the radix cache...")
        name = "policy_cache_check"
        # Dedicated engine, mirroring test_upsert_refresh's rationale for
        # using its own engine (never shut down -- see that test's note on
        # why: Engine.shutdown() kills every child process of the current
        # Python process, not just this engine's own subprocesses).
        engine = sgl.Engine(
            **{
                **ENGINE_KWARGS,
                "max_loaded_ofts": 2,
                "max_ofts_per_batch": 2,
                "mem_fraction_static": 0.15,
            }
        )
        result = engine.load_oft_adapter_from_tensors(
            adapter_name=name,
            tensors=_oft_named_tensors(self.num_layers, self.intermediate_size, seed=400),
            config_dict=_adapter_config_dict(),
        )
        self.assertTrue(result.success, f"Failed to load OFT adapter: {result.error_message}")
        text_v1 = self._generate(engine, adapter_name=name)
        self.assertTrue(text_v1, "Generation before upsert produced no text")

        upsert_result = self._upsert(engine, name, seed=401)
        self.assertTrue(upsert_result.success, f"Upsert failed: {upsert_result.error_message}")

        # SAME prompt, SAME adapter name -- if the radix cache key didn't
        # change across the upsert, this could be served from the
        # pre-upsert KV prefix instead of reflecting the new weights.
        text_v2 = self._generate(engine, adapter_name=name)
        self.assertTrue(text_v2, "Generation after upsert produced no text")
        self.assertNotEqual(
            text_v1,
            text_v2,
            "Re-issuing the SAME prompt after an in-place OFT upsert produced "
            "identical output -- the radix cache may have served a stale KV "
            "prefix cached under the pre-upsert adapter_version (i.e. "
            "adapter_version was not bumped on upsert).",
        )

    def test_lru_eviction_past_max_loaded_ofts(self):
        """C1 fix: adapters loaded over the wire (this native RPC path) have
        no CPU-side artifact to re-page from, so allocate_buffer_slot_with_
        eviction now never selects one as a GPU-side eviction victim --
        unlike before the fix, which would silently GPU-evict the LRU
        wire-loaded adapter once real per-batch capacity (max_ofts_per_batch
        - 1; slot 0 is always the identity placeholder) was exceeded, while
        the tokenizer-side registry still considered it resident. A later
        /generate or tokenizer-side max_loaded_ofts LRU eviction naming that
        silently-evicted adapter then hit an assert/crash instead of a clean
        rejection or unload -- this is the branch's own test config
        (max_ofts_per_batch=4, max_loaded_ofts=4) the review reproduced this
        crash against.

        Note this also makes the tokenizer-side max_loaded_ofts LRU-eviction
        loop unreachable for a pure wire-adapter workload in a *successful*
        load: validate_oft_args now requires max_loaded_ofts >=
        max_ofts_per_batch - 1, so the tokenizer-side cap can never bind
        before the GPU-side one does for adapters that (unlike disk-backed
        ones) can never be paged out once resident -- attempting to exceed
        real GPU capacity fails at the GPU layer first, before
        num_registered_ofts could ever exceed max_loaded_ofts. This test
        confirms that overflow load now fails gracefully (instead of
        crashing the engine), and that the already-resident adapters remain
        fully intact, still generate correctly, and can still be unloaded
        promptly afterward.
        """
        print("[Test]Testing that loading past real GPU OFT capacity fails gracefully...")
        # Real per-batch adapter capacity is max_ofts_per_batch - 1 (slot 0
        # is always the identity/base-model placeholder).
        real_capacity = ENGINE_KWARGS["max_ofts_per_batch"] - 1
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
        names = [f"oft_lru_{i}" for i in range(real_capacity + 1)]
        for i, name in enumerate(names[:real_capacity]):
            print(f"[Test]Loading OFT adapter {i + 1}/{real_capacity}: {name}")
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

        # One more, beyond real capacity: every resident adapter is
        # wire-loaded (non-reloadable) and none is pinned, so there is no
        # evictable candidate -- this must fail gracefully, not silently
        # evict an unrecoverable adapter (the pre-fix behavior) or crash.
        overflow_name = names[real_capacity]
        overflow_result = engine.load_oft_adapter_from_tensors(
            adapter_name=overflow_name,
            tensors=_oft_named_tensors(
                self.num_layers, self.intermediate_size, seed=999
            ),
            config_dict=_adapter_config_dict(),
        )
        self.assertFalse(
            overflow_result.success,
            f"Loading past real GPU capacity ({real_capacity}) with no evictable "
            "resident adapter should fail gracefully, not succeed by silently "
            "evicting an unrecoverable wire-loaded adapter.",
        )
        print(f"[Test]Overflow load correctly rejected: {overflow_result.error_message}")

        all_adapters = engine.tokenizer_manager.oft_registry.get_all_adapters()
        self.assertNotIn(
            overflow_name,
            all_adapters,
            "The rejected overflow load must not leave a registered-but-not-"
            "actually-resident name behind",
        )
        for name in names[:real_capacity]:
            self.assertIn(
                name,
                all_adapters,
                f"Resident adapter {name!r} must not have been evicted by the "
                "failed overflow load",
            )

        # The already-resident adapters must be untouched by the rejected
        # overflow load: they still generate correctly, and unloading them
        # after generate() must complete promptly rather than hang (bug #2,
        # fixed by finalize_peft_lease releasing the request's adapter lease
        # on completion).
        for name in names[:real_capacity]:
            text = self._generate(engine, adapter_name=name)
            self.assertTrue(
                text, f"Generation with resident adapter {name!r} produced no text"
            )
            unload_result = engine.unload_oft_adapter(name)
            self.assertTrue(
                unload_result.success,
                f"Failed to unload adapter {name!r} after generate(): "
                f"{unload_result.error_message}",
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

        # Neither adapter is unloaded here -- both must stay resident and
        # generatable at once (the capability Task 6 newly unlocked by
        # removing the old single-active restriction). The unload path
        # itself is covered by test_fresh_load_and_generate and
        # test_lru_eviction_past_max_loaded_ofts.
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
