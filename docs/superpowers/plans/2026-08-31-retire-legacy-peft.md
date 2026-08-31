# Retire Legacy srt/peft Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Produce an Impossible SGLang branch that combines the pinned Radix baseline, Impossible-specific behavior, canonical Sphere OFT, shared adapter synchronization, and native staged LoRA without ever landing the legacy srt/peft implementation.

**Architecture:** Merge Radix commit 5a0132a7 into the Impossible branch, resolve conflicts by preserving Impossible behavior on the newer Radix APIs, then selectively port only Sphere's canonical srt/oft, srt/adapter_sync, and staged native-LoRA changes. Run the unchanged Sphere branch as the legacy/canonical oracle and compare it with the Impossible candidate through immutable, manifest-driven bundles.

**Tech Stack:** Python 3.12, PyTorch, SGLang, pytest/unittest, msgspec, CUDA, NCCL, FlashInfer/Triton, NVIDIA H100/B200, HTCondor on mpi3.

**Spec:** docs/superpowers/specs/2026-08-31-retire-legacy-peft-design.md

## Global Constraints

- Work only in /data/home/zeju/miles-orbit-dev/impossible/sglang/.worktrees/retire-legacy-peft on branch codex/retire-legacy-peft.
- The Impossible starting point is f3863a3f12c4e4db92374158bb1b843ed7cf6ed9.
- Merge exactly Radix 5a0132a7623b0da58f98540f1edcca6cf154c72c; do not incorporate its two later commits.
- The reviewed Sphere source is a89f20362e2f6dd6ba135caa558b4fabdff9812c; record the final orbit-main-corrected SHA immediately before porting and use that immutable SHA thereafter.
- Never copy python/sglang/srt/peft into the Impossible candidate. It exists only in the source checkout used for oracle generation.
- Keep --peft-method oft. Remove --oft-impl and reject --peft-method lora.
- Keep native LoRA on --enable-lora; staged updates remain optional through --enable-lora-staging.
- Mandatory models: Qwen3-4B-Instruct-2507 dense and Qwen3-30B-A3B MoE.
- Mandatory precisions: BF16, FP8, and NVFP4. NVFP4 runs only on B200.
- DeepSeek and Kimi end-to-end validation are explicitly deferred, but their imports must point to canonical OFT and remain importable.
- Deterministic execution is required wherever supported. Exact comparison is preferred; tolerance exceptions must be derived from unchanged-baseline repetition and recorded by manifest hash.
- Median throughput and peak GPU memory may not regress by more than 5%.
- GPU work runs through HTCondor on mpi3, never on a login node. Bid 100 is authorized for multiple concurrent jobs.
- Before every submission wave, refresh condor_free; do not touch the user's pre-existing jobs or sessions.
- Every remote run writes durable provenance, stdout, stderr, event log, result bundle, and completion status under the remote-cluster run-store contract.

---

### Task 1: Consolidate the pinned Radix baseline with Impossible

**Files:**
- Modify while resolving the merge: the recorded 51-file conflict manifest, including python/pyproject.toml, python/sglang/srt/server_args.py, native LoRA, scheduler, weight-update, Qwen, multimodal, and distributed-runtime files.
- Create: docs/superpowers/plans/artifacts/radix-5a0132a7-conflicts.txt
- Test: existing Impossible tests changed between merge base 8905cbd4 and f3863a3f.

**Interfaces:**
- Consumes: Impossible f3863a3f and Radix 5a0132a7.
- Produces: one merge commit whose first parent preserves Impossible history and whose second parent is exactly 5a0132a7.

- [ ] **Step 1: Record immutable merge inputs and the conflict list**

Run:

~~~bash
git status --short
git merge-base --is-ancestor f3863a3f12c4e4db92374158bb1b843ed7cf6ed9 HEAD
test "$(git diff --name-only f3863a3f12c4e4db92374158bb1b843ed7cf6ed9..HEAD -- python test | wc -l)" -eq 0
git cat-file -e 5a0132a7623b0da58f98540f1edcca6cf154c72c^{commit}
git merge-base f3863a3f12c4e4db92374158bb1b843ed7cf6ed9 5a0132a7623b0da58f98540f1edcca6cf154c72c
~~~

Expected merge base: 8905cbd42f4a27dbecad6487cd3904278756fef8. Save the read-only git merge-tree textual-conflict list. The expected list contains 51 files and includes native LoRA, server_args.py, scheduler/weight-update seams, Qwen model files, and three add/add conflicts.

- [ ] **Step 2: Start the pinned merge**

~~~bash
git merge --no-ff --no-commit 5a0132a7623b0da58f98540f1edcca6cf154c72c
~~~

Expected: Git stops for the recorded conflict set; no unrecorded source is introduced.

- [ ] **Step 3: Resolve infrastructure and API conflicts from the Radix side first**

Use Radix as the structural base for dependency metadata, argument groups, distributed state, current model APIs, and scheduler interfaces. Reapply Impossible behavior rather than restoring obsolete v0.5.16 files wholesale.

Resolve these groups independently:

~~~text
Dependency/API:
  python/pyproject.toml
  python/sglang/srt/arg_groups/overrides.py
  python/sglang/srt/distributed/parallel_state.py
  python/sglang/srt/configs/cohere2_moe.py

Entrypoints/protocol:
  python/sglang/srt/entrypoints/engine.py
  python/sglang/srt/entrypoints/openai/serving_chat.py
  python/sglang/srt/entrypoints/sidecar.py
  python/sglang/srt/function_call/function_call_parser.py

Multimodal:
  python/sglang/srt/layers/attention/vision.py
  python/sglang/srt/multimodal/processors/base_processor.py
  python/sglang/srt/multimodal/processors/qwen_vl.py
  test/registered/vlm/test_token_id_retokenize_e2e.py
~~~

Preserve Impossible's bare-metal CUDA dependency flavor, exact-scoring suffix, pretokenized-ID handling, and Kimi tool-call recovery while adopting Radix signatures.

- [ ] **Step 4: Resolve on-policy, weight-update, LoRA, and Qwen conflicts**

Start from each Radix file and reapply the behavior represented by Impossible commits [1/27] through [27/27], multi-LoRA redesign, local-checkpoint reseeding, pause-safe graph pools, FP8 tuple-aware rows, and Qwen GDN adapters.

Critical files:

~~~text
python/sglang/srt/lora/backend/base_backend.py
python/sglang/srt/lora/layers.py
python/sglang/srt/lora/lora_manager.py
python/sglang/srt/lora/mem_pool.py
python/sglang/srt/managers/io_struct.py
python/sglang/srt/managers/scheduler.py
python/sglang/srt/managers/scheduler_components/weight_updater.py
python/sglang/srt/managers/tokenizer_control_mixin.py
python/sglang/srt/managers/tokenizer_manager.py
python/sglang/srt/managers/tp_worker.py
python/sglang/srt/model_executor/forward_batch_info.py
python/sglang/srt/model_executor/model_runner.py
python/sglang/srt/model_executor/model_runner_components/weight_updater.py
python/sglang/srt/model_loader/loader.py
python/sglang/srt/models/qwen2.py
python/sglang/srt/models/qwen2_moe.py
python/sglang/srt/models/qwen3.py
python/sglang/srt/server_args.py
python/sglang/srt/weight_sync/local_checkpoint.py
~~~

Do not add Sphere OFT or staged-LoRA code during merge resolution.

- [ ] **Step 5: Verify the merged tree before committing**

~~~bash
git diff --check
git grep -n '^<<<<<<<\|^=======\|^>>>>>>>' -- .
python -m compileall -q python/sglang
python -m pytest -q test/registered/unit
python -m pytest -q test/registered/vlm/test_token_id_retokenize_e2e.py
~~~

Expected: no conflict markers, compile success, and selected CPU tests pass. Record any environment-only failure with command, traceback, and proof that it reproduces on both parents.

- [ ] **Step 6: Commit the consolidation**

~~~bash
git add -A
git commit -m "merge: consolidate radix 5a0132a7 with sglang-miles"
git show -s --format='%P' HEAD
~~~

Expected: second parent 5a0132a7623b0da58f98540f1edcca6cf154c72c.

---

### Task 2: Build the immutable equivalence-bundle schema and comparator

**Files:**
- Create: test/manual/adapter_equivalence/__init__.py
- Create: test/manual/adapter_equivalence/schema.py
- Create: test/manual/adapter_equivalence/compare.py
- Create: test/registered/unit/adapter_equivalence/test_compare.py

**Interfaces:**
- Produces: CaseKey, Observation, RunBundle, ToleranceEnvelope, compare_bundles().
- Consumes later: JSON bundle paths produced by server scenario shards.

- [ ] **Step 1: Write failing exact-comparison tests**

Cover identical bundles, token mismatch, shape/dtype mismatch, adapter-state mismatch, error mismatch, missing provenance, and a performance regression. Use this public shape:

~~~python
@dataclass(frozen=True)
class CaseKey:
    model: str
    architecture: Literal["dense", "moe"]
    precision: Literal["bf16", "fp8", "nvfp4"]
    revision: str
    mode: Literal["base", "legacy_oft", "canonical_oft", "legacy_lora", "native_lora"]
    cuda_graph: bool
    scenario: str

@dataclass(frozen=True)
class Observation:
    output_ids: tuple[int, ...]
    text: str
    token_logprobs: tuple[float, ...]
    selected_logits: dict[str, tuple[float, ...]]
    adapter_state: dict[str, object]
    error: dict[str, object] | None

def compare_bundles(
    expected: RunBundle,
    actual: RunBundle,
    envelope: ToleranceEnvelope,
) -> ComparisonReport:
    raise RuntimeError("red phase: comparator not implemented")
~~~

- [ ] **Step 2: Run and confirm failure**

~~~bash
python -m pytest -q test/registered/unit/adapter_equivalence/test_compare.py
~~~

Expected: import failure because the harness modules do not exist.

- [ ] **Step 3: Implement exact-first comparison**

Compare in this order: provenance hashes; token IDs/shapes/dtypes/state/errors exactly; logits/logprobs exactly for deterministic paths; named math.isclose tolerance only for recorded exceptions; performance medians and peak memory with a 1.05 ratio. Every mismatch includes CaseKey, request ID, token/state position, expected, actual, and envelope.

- [ ] **Step 4: Add comparator self-protection tests**

Mutate one token, adapter version, logprob, and peak-memory value and assert each fails for the expected reason. Widening tolerance without a changed envelope manifest hash must invalidate comparison.

- [ ] **Step 5: Run and commit**

~~~bash
python -m pytest -q test/registered/unit/adapter_equivalence/test_compare.py
git add test/manual/adapter_equivalence test/registered/unit/adapter_equivalence
git commit -m "test: add immutable adapter equivalence comparator"
~~~

---

### Task 3: Define the deterministic six-cell matrix and adapter fixtures

**Files:**
- Create: test/manual/adapter_equivalence/matrix.json
- Create: test/manual/adapter_equivalence/prompts.jsonl
- Create: test/manual/adapter_equivalence/fixtures.py
- Create: test/manual/adapter_equivalence/preflight.py
- Create: test/registered/unit/adapter_equivalence/test_fixtures.py

**Interfaces:**
- Produces: MatrixCell, AdapterFixture, build_oft_fixture(), build_lora_fixture(), validate_matrix().

- [ ] **Step 1: Write the matrix manifest**

~~~json
[
  {"id":"qwen3-4b-bf16","model":"Qwen/Qwen3-4B-Instruct-2507","architecture":"dense","precision":"bf16","gpu":"H100","tp":1,"ep":1},
  {"id":"qwen3-4b-fp8","model":"Qwen/Qwen3-4B-Instruct-2507-FP8","architecture":"dense","precision":"fp8","gpu":"H100","tp":1,"ep":1,"quantization":"fp8"},
  {"id":"qwen3-4b-nvfp4","model":"Qwen3-4B-Instruct-2507-NVFP4","architecture":"dense","precision":"nvfp4","gpu":"B200","tp":1,"ep":1,"quantization":"modelopt_fp4"},
  {"id":"qwen3-30b-a3b-bf16","model":"Qwen/Qwen3-30B-A3B","architecture":"moe","precision":"bf16","gpu":"H100","tp":4,"ep":4},
  {"id":"qwen3-30b-a3b-fp8","model":"Qwen/Qwen3-30B-A3B-FP8","architecture":"moe","precision":"fp8","gpu":"H100","tp":4,"ep":4,"quantization":"fp8","moe_runner":"triton"},
  {"id":"qwen3-30b-a3b-nvfp4","model":"nvidia/Qwen3-30B-A3B-NVFP4","architecture":"moe","precision":"nvfp4","gpu":"B200","tp":4,"ep":4,"quantization":"modelopt_fp4","moe_runner":"flashinfer_cutlass"}
]
~~~

Preflight resolves IDs to absolute snapshots and hashes config.json, safetensors index, and every shard.

- [ ] **Step 2: Add deterministic prompts**

Use fixed short factual, arithmetic, code-completion, long repeated-prefix, uneven mixed-batch, and graph-bucket prompts. Record tokenized inputs. Include 1-, 2-, 8-, and 32-request batches with temperature 0.

- [ ] **Step 3: Write failing fixture tests**

Seeds 1729 and 2718 must produce stable, distinct adapters A/B; repeated generation is byte-identical; OFT is identity plus deterministic skew-symmetric perturbation; LoRA rank is 8.

- [ ] **Step 4: Implement fixtures**

Dense OFT targets q_proj, o_proj, gate_proj, up_proj, down_proj with block size 128. MoE OFT adds expert gate/up/down with block size 32. LoRA rank is 8 and alpha 16. Write adapter_config.json, adapter_model.safetensors, and sha256.json; reject unresolved target names.

- [ ] **Step 5: Prepare dense NVFP4 if absent**

Use NVIDIA Model Optimizer's official unified-HF PTQ workflow
(https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md)
and run on B200:

~~~bash
python hf_ptq.py \
  --pyt_ckpt_path Qwen/Qwen3-4B-Instruct-2507 \
  --qformat nvfp4 \
  --kv_cache_qformat fp8_cast \
  --calib_size 512 \
  --batch_size 8 \
  --export_path /lustre/home/zqiu/model-cache/Qwen3-4B-Instruct-2507-NVFP4
~~~

Hash and reuse the immutable export. Use nvidia/Qwen3-30B-A3B-NVFP4 for MoE.

- [ ] **Step 6: Run and commit**

~~~bash
python -m pytest -q test/registered/unit/adapter_equivalence/test_fixtures.py
git add test/manual/adapter_equivalence test/registered/unit/adapter_equivalence
git commit -m "test: define deterministic Qwen adapter matrix"
~~~

---

### Task 4: Port shared adapter synchronization and native staged LoRA

**Files:**
- Create from final Sphere SHA: python/sglang/srt/adapter_sync/ excluding __pycache__.
- Modify: python/sglang/srt/lora/lora_registry.py
- Modify: python/sglang/srt/entrypoints/http_server.py
- Modify: python/sglang/srt/managers/io_struct.py
- Modify: python/sglang/srt/managers/scheduler.py
- Modify: python/sglang/srt/managers/scheduler_components/weight_updater.py
- Modify: python/sglang/srt/managers/tokenizer_control_mixin.py
- Modify: python/sglang/srt/managers/tokenizer_manager.py
- Modify: python/sglang/srt/managers/tp_worker.py
- Modify: python/sglang/srt/model_executor/model_runner.py
- Modify: python/sglang/srt/model_executor/model_runner_components/weight_updater.py
- Modify: python/sglang/srt/server_args.py
- Test: test/registered/unit/adapter_sync/
- Test: test/registered/unit/lora/test_lora_staging_control.py
- Test: test/registered/unit/lora/test_lora_versioning.py
- Test: test/registered/lora/test_lora_staged_update.py
- Test: test/registered/lora/test_lora_staged_update_tp.py

**Interfaces:**
- Produces: VersionedStaging, AdapterMemPool, AdapterManager, StagedLoRAManager.
- Preserves: merged Impossible multi-LoRA and MoE-LoRA behavior.

- [ ] **Step 1: Port unit tests first and verify failure**

~~~bash
python -m pytest -q test/registered/unit/adapter_sync test/registered/unit/lora/test_lora_staging_control.py test/registered/unit/lora/test_lora_versioning.py
~~~

Expected: import/attribute failures for the absent staging core.

- [ ] **Step 2: Port the shared state machine**

Port versioning.py, mem_pool.py, manager.py, utils.py, and backends/lora.py. Preserve:

~~~text
stage(version, uid) writes only the inactive slot for uid
activate(version, uid) rejects mismatched uid/version
activate copies only the selected adapter
active_version(uid) is independent per adapter
native LoRA buffer layout is not reorganized
staging slots never appear in serving lookup tables
~~~

- [ ] **Step 3: Port native-LoRA registry version identity**

Append version: int = 0 to LoRARef, preserve array-like wire compatibility, and include version in radix-cache identity without changing stable lora_id.

- [ ] **Step 4: Wire staged update and activation routing**

When enable_lora_staging and load_format == "lora_adapter", route update and activation to StagedLoRAManager; otherwise preserve Impossible's weight-update and native LoRA paths. Do not keep the Sphere fallback to srt.peft.integration; Task 6 supplies canonical OFT routing.

- [ ] **Step 5: Verify CPU and GPU tests**

~~~bash
python -m pytest -q test/registered/unit/adapter_sync test/registered/unit/lora/test_lora_staging_control.py test/registered/unit/lora/test_lora_versioning.py
python -m pytest -q test/registered/lora/test_lora_staged_update.py
python -m pytest -q test/registered/lora/test_lora_staged_update_tp.py
~~~

The final two commands run inside allocated GPU jobs.

- [ ] **Step 6: Commit**

~~~bash
git add python/sglang/srt/adapter_sync python/sglang/srt/lora python/sglang/srt/entrypoints/http_server.py python/sglang/srt/managers python/sglang/srt/model_executor python/sglang/srt/server_args.py test/registered
git commit -m "feat(lora): port native staged adapter updates"
~~~

---

### Task 5: Port canonical OFT without the legacy package

**Files:**
- Create from final Sphere SHA: python/sglang/srt/oft/ excluding __pycache__.
- Create: python/sglang/srt/oft/config.py
- Create: python/sglang/srt/oft/integration.py
- Create: python/sglang/srt/oft/io_types.py
- Create: python/sglang/srt/oft/tokenizer_hooks.py
- Create: python/sglang/srt/oft/tokenizer_mixin.py
- Test: test/srt/oft/
- Create: test/registered/unit/oft/test_oft_config.py

**Interfaces:**
- Produces: OFTArgs, register_oft_args(), validate_oft_args(), canonical manager/registry/layers/kernels, OFT request types, and tokenizer control.
- Must not consume: any symbol under sglang.srt.peft.

- [ ] **Step 1: Port tests first**

Copy the eight Sphere test/srt/oft tests and retarget all legacy imports to sglang.srt.oft. Move the PEFT config test to test/registered/unit/oft/test_oft_config.py and allow only None and "oft".

~~~bash
python -m pytest -q test/registered/unit/oft/test_oft_config.py test/srt/oft/test_tiny_block_validation.py
~~~

Expected: failure because canonical OFT/config are absent.

- [ ] **Step 2: Port the canonical implementation**

Copy Sphere srt/oft at the frozen source SHA. Internal imports may refer only to sglang.srt.oft and sglang.srt.adapter_sync. Keep canonical oft_moe_runner_marlin.py, oft_moe_runners.py, and base/*. Do not port legacy marlin_runner.py, moe_invoke.py, or srt/peft/base.

- [ ] **Step 3: Convert the live control layer to OFT-only modules**

Use Sphere top-level srt/peft files only as source material. Rename:

~~~text
PEFTArgs                 -> OFTArgs
register_peft_args       -> register_oft_args
validate_peft_args       -> validate_oft_args
PEFTTokenizerMixin       -> OFTTokenizerMixin
init_tokenizer_peft      -> init_tokenizer_oft
register_peft_ref        -> register_oft_ref
bump_peft_version        -> bump_oft_version
resolve_peft_path        -> resolve_oft_path
maybe_resolve_peft_path  -> maybe_resolve_oft_path
maybe_init_peft_manager  -> maybe_init_oft_manager
maybe_prepare_peft_batch -> maybe_prepare_oft_batch
~~~

Remove every peft_method == "lora" branch, every oft_impl branch, legacy imports, peft_max_lora_rank, and peft_double_buffer. Keep OFT paths, block settings, streamed staging, request types, and versioned radix identity.

- [ ] **Step 4: Preserve lazy imports**

Merge control symbols into srt/oft/__init__.py using its PEP 562 lazy map. Importing sglang.srt.oft must not eagerly import model runner, CUDA kernels, or torch.distributed.

- [ ] **Step 5: Test and commit**

~~~bash
python -m pytest -q test/registered/unit/oft/test_oft_config.py test/srt/oft/test_tiny_block_validation.py test/srt/oft/test_split_dense_merged_projection_dispatch.py
python -m pytest -q test/srt/oft
git add python/sglang/srt/oft test/srt/oft test/registered/unit/oft
git commit -m "feat(oft): port canonical adapter serving"
~~~

The full test/srt/oft command runs inside an allocated H100 job.

---

### Task 6: Retarget runtime seams and remove legacy selectors

**Files:**
- Modify: python/sglang/srt/server_args.py
- Modify: python/sglang/srt/managers/io_struct.py
- Modify: python/sglang/srt/managers/schedule_batch.py
- Modify: python/sglang/srt/managers/scheduler.py
- Modify: python/sglang/srt/managers/tokenizer_control_mixin.py
- Modify: python/sglang/srt/managers/tokenizer_manager.py
- Modify: python/sglang/srt/model_executor/forward_batch_info.py
- Modify: python/sglang/srt/model_executor/model_runner.py
- Modify: python/sglang/srt/model_executor/model_runner_components/weight_updater.py
- Modify: python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
- Modify: python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py
- Modify: python/sglang/srt/layers/moe/moe_runner/runner.py
- Modify: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
- Modify: python/sglang/srt/models/kimi_k25.py

**Interfaces:**
- Consumes: Task 4 staged LoRA and Task 5 canonical OFT.
- Produces: no runtime import, dispatch, or selector for legacy PEFT.

- [ ] **Step 1: Write failing selector-removal tests**

Assert peft_method="oft" is valid, peft_method="lora" raises ValueError, CLI --peft-method lora exits 2, and --oft-impl peft is unrecognized.

- [ ] **Step 2: Change argument ownership**

Make ServerArgs inherit OFTArgs, call register_oft_args() and validate_oft_args(), retain peft_method choices ["oft"], and remove oft_impl. Keep native LoRA args in the native group.

- [ ] **Step 3: Retarget OFT integration calls**

Import sglang.srt.oft.integration as oft. Use OFT-named helpers for admission, batch preparation, cache keys, manager initialization, staged updates, and graph capture/replay. Remove maybe_prepare_lora_batch() from generic integration; native LoRA continues through lora_manager.prepare_lora_batch().

- [ ] **Step 4: Retarget model-specific imports**

Use sglang.srt.oft.oft_moe_runner_marlin.MarlinOFTRunnerCore and sglang.srt.oft.oft_moe_runners.make_oft_invoke without a selector. Point DeepSeek and Kimi helpers at sglang.srt.oft; no expensive e2e test is added.

- [ ] **Step 5: Verify and commit**

~~~bash
python -c 'from sglang.srt.server_args import ServerArgs; print(ServerArgs(model_path="Qwen/Qwen3-0.6B", peft_method="oft").peft_method)'
python -m pytest -q test/registered/unit/oft test/registered/unit/lora test/registered/unit/adapter_sync
git add python/sglang/srt test/registered/unit
git commit -m "refactor: route adapters without legacy peft"
~~~

Expected first output: oft.

---

### Task 7: Add permanent absence guards and CPU regression coverage

**Files:**
- Create: test/registered/unit/oft/test_no_legacy_peft.py
- Modify: migrated OFT/adapter tests and user-facing examples that advertise removed selectors.

**Interfaces:**
- Produces: a gate preventing reintroduction of the package or selectors.

- [ ] **Step 1: Write the static guard**

Scan python/, executable tests, and user-facing docs. Reject sglang.srt.peft, python/sglang/srt/peft, oft_impl, --oft-impl, --peft-method lora, and peft_method == "lora". Exclude only the guard's literal fixtures and historical design/plan documents. Assert python/sglang/srt/peft does not exist.

- [ ] **Step 2: Test error and rollback contracts**

Add CPU tests for duplicate/stale versions, wrong adapter ID, malformed config, failed stage leaving active state untouched, unload restoring identity, and independent A/B versions.

- [ ] **Step 3: Run the CPU gate**

~~~bash
python -m pytest -q \
  test/registered/unit/adapter_sync \
  test/registered/unit/lora \
  test/registered/unit/oft \
  test/srt/oft/test_tiny_block_validation.py \
  test/srt/oft/test_split_dense_merged_projection_dispatch.py
python -m compileall -q python/sglang
git diff --check
~~~

- [ ] **Step 4: Commit**

~~~bash
git add python test docs
git commit -m "test: prevent legacy peft regression"
~~~

---

### Task 8: Implement scenario execution and freeze unchanged-source oracles

**Files:**
- Create: test/manual/adapter_equivalence/server.py
- Create: test/manual/adapter_equivalence/scenarios.py
- Create: test/manual/adapter_equivalence/run_case.py
- Create: test/manual/adapter_equivalence/README.md
- Modify: test/registered/unit/adapter_equivalence/test_compare.py

**Interfaces:**
- Produces: one immutable RunBundle per cell, revision, mode, graph setting, and scenario group.
- Consumes: unchanged Sphere source SHA and deterministic fixtures.

- [ ] **Step 1: Implement the state machine**

Each shard runs: base inference; startup adapter; dynamic load/infer/unload/base; A/B/A; mixed base/A/B; concurrent stream/non-stream; short/long prefill and decode; stage v1/activate/stage v2/activate; duplicate/stale rejection; invalid ID/config and rollback; restart from the same manifest. Record state around every transition. Post-unload output must exactly equal initial base output.

- [ ] **Step 2: Support source and candidate modes**

Source:

~~~text
base:          no adapter flag
legacy_oft:    --peft-method oft --oft-impl peft
canonical_oft: --peft-method oft --oft-impl sibling
legacy_lora:   --peft-method lora
native_lora:   --enable-lora --enable-lora-staging
~~~

Candidate:

~~~text
base:          no adapter flag
canonical_oft: --peft-method oft
native_lora:   --enable-lora --enable-lora-staging
~~~

- [ ] **Step 3: Prove harness sensitivity**

Run Qwen3-4B BF16 source smoke, mutate one output token and adapter version in a copied bundle, and require both comparisons to fail before submitting the matrix.

- [ ] **Step 4: Transfer immutable revisions to mpi3**

Create Git bundles for frozen Sphere source and candidate, snapshot through the Mac control plane, and clone into separate /lustre/home/zqiu worktrees. Record bundle SHA256. Never transfer uncommitted source.

- [ ] **Step 5: Submit 60 source shards in parallel**

~~~text
6 cells x 5 modes x 2 CUDA-graph settings = 60 shards
~~~

Use H100 for BF16/FP8 and B200 for NVFP4. Dense processes may split a full node; MoE TP4/EP4 pairs use disjoint four-GPU groups.

- [ ] **Step 6: Validate and freeze bundles**

Require schema, checkpoint/adapter/code/environment hashes, and completion marker. Make validated directories read-only and write a CaseKey-to-bundle-hash index.

- [ ] **Step 7: Commit harness**

~~~bash
git add test/manual/adapter_equivalence test/registered/unit/adapter_equivalence
git commit -m "test: add adapter lifecycle oracle runner"
~~~

---

### Task 9: Run GPU components and 36 candidate functional shards

**Files:**
- Artifacts only unless a test identifies a defect; fixes return to the owning task with a regression test.

**Interfaces:**
- Consumes: candidate, frozen source bundles, tolerance envelopes.
- Produces: component and end-to-end evidence.

- [ ] **Step 1: Submit component shards in parallel**

~~~bash
python -m pytest -q test/srt/oft
python -m pytest -q test/registered/lora/test_lora_staged_update.py
python -m pytest -q test/registered/lora/test_lora_staged_update_tp.py
~~~

BF16/FP8 on H100; NVFP4 kernel paths on B200.

- [ ] **Step 2: Submit candidate wave**

~~~text
6 cells x 3 modes x 2 CUDA-graph settings = 36 shards
~~~

Never compare different GPU models for one cell.

- [ ] **Step 3: Establish envelopes from unchanged-source repetition**

Repeat each non-exact source case three times:

~~~python
atol = max(pairwise_max_abs_diff)
rtol = max(pairwise_max_relative_diff)
~~~

Exact source cases keep atol=rtol=0. Candidate limits are the measured envelope; token divergence always fails.

- [ ] **Step 4: Compare canonical pre/post**

For all six cells and both graph modes:

~~~text
source canonical_oft <-> candidate canonical_oft
source native_lora   <-> candidate native_lora
source base          <-> candidate base
~~~

- [ ] **Step 5: Compare common legacy behavior**

For single-adapter scenarios:

~~~text
source legacy_oft  <-> source canonical_oft
source legacy_lora <-> source native_lora
~~~

Multi-adapter staged scenarios compare source canonical/native to candidate canonical/native.

- [ ] **Step 6: Require complete aggregate report**

List all 96 functional shards, component shards, bundle hashes, comparison modes, and mismatches. Absent completion markers fail the gate.

---

### Task 10: Run distributed stress and performance gates

**Files:**
- Artifacts only unless a defect requires code and a regression test.

**Interfaces:**
- Consumes: functionally equivalent candidate.
- Produces: TP/EP, leak, concurrency, throughput, latency, and memory evidence.

- [ ] **Step 1: Run MoE distributed equivalence**

For BF16, FP8, and NVFP4, run canonical OFT and native LoRA with TP4/EP4. Pair source/candidate on identical GPU classes. Include mixed adapters, version activation, graph replay, and unload.

- [ ] **Step 2: Run lifecycle stress**

Every cell runs 100 A/B load-switch-unload cycles and 1,000 mixed base/A/B requests. Assert no failed request, stale version, leaked registry entry, or monotonic allocated-memory growth after synchronization/cache cleanup.

- [ ] **Step 3: Run controlled performance**

For base, canonical OFT, and native LoRA in six cells: two warm-ups, three measurements, fixed workload, no co-tenant on measured GPUs, and placement swap when equivalent placements exist.

- [ ] **Step 4: Enforce gates**

~~~text
candidate_throughput >= 0.95 * source_throughput
candidate_peak_gpu_memory <= 1.05 * source_peak_gpu_memory
~~~

Record latency and investigate median movement over 5%. Never average away a functional failure.

- [ ] **Step 5: Preserve results**

Snapshot authoritative mpi3 run directories once to matching Mac run-store suffixes. Preserve job IDs, bids, node/GPU identities, commands, manifests, stdout/stderr, and reports.

---

### Task 11: Final audit, documentation, and self-review

**Files:**
- Modify: user-facing OFT/native-LoRA docs and examples.
- Create: docs/superpowers/plans/artifacts/retire-legacy-peft-verification.md
- Review: origin/sglang-miles...HEAD.

**Interfaces:**
- Produces: review-ready branch and evidence index; does not push or create a PR without separate authorization.

- [ ] **Step 1: Run absence and cleanliness audit**

~~~bash
test ! -e python/sglang/srt/peft
git grep -n 'sglang\.srt\.peft' -- python test ':!test/registered/unit/oft/test_no_legacy_peft.py'
git grep -n -E 'oft_impl|--oft-impl|peft_method *== .*lora' -- python test
git diff --check origin/sglang-miles...HEAD
~~~

Expected: first succeeds; both grep commands produce no output.

- [ ] **Step 2: Run final CPU suite**

~~~bash
python -m compileall -q python/sglang
python -m pytest -q test/registered/unit/adapter_sync test/registered/unit/lora test/registered/unit/oft
~~~

- [ ] **Step 3: Write verification index**

Record exact source/candidate SHAs, merge parents, commands, pass counts, Condor IDs, H100/B200 types, 96 functional results, distributed/stress results, performance medians, and artifact paths.

- [ ] **Step 4: Review commit by commit**

Confirm Radix merge preserves Impossible behavior; no commit adds srt/peft; OFT control is OFT-owned; native LoRA bypasses OFT integration; DeepSeek/Kimi imports work; docs state intentional CLI removals.

- [ ] **Step 5: Commit documentation**

~~~bash
git add docs test python
git commit -m "docs: record canonical adapter migration verification"
~~~

- [ ] **Step 6: Stop for PR self-review**

Invoke the pull-request review workflow against origin/sglang-miles...HEAD. Fix every blocking finding and rerun the affected gate before requesting authorization to push or open the PR.
