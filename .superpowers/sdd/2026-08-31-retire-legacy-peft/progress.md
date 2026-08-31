# SDD ledger — plan: docs/superpowers/plans/2026-08-31-retire-legacy-peft.md

## Preflight task/interface overlap scan

| Producer / earlier task | Consumer / later task | Shared files or interface | Ruling |
|---|---|---|---|
| Task 1 | Tasks 2–11 | Entire merged source baseline and first-parent history | Task 1 is a hard dependency for every later task. No later task starts until the pinned merge is committed and reviewed. |
| Task 2 | Tasks 3, 8, 9, 10, 11 | `RunBundle`, `CaseKey`, manifest hashes, comparator and report schema | Freeze the schema in Task 2. Later tasks may extend scenario content but may not silently change identity or comparison semantics; schema changes return to Task 2 with regression coverage. |
| Task 3 | Tasks 8, 9, 10, 11 | Six-cell matrix, deterministic prompts, adapter fixtures, dense NVFP4 checkpoint | Fixture manifests are immutable inputs after Task 3. Every later artifact records their hashes. |
| Task 4 | Tasks 5, 6, 7, 8, 9, 10, 11 | `srt/adapter_sync`, staged native-LoRA manager, server/worker update seams | Task 4 owns native staged LoRA and shared synchronization. Task 5 may consume `adapter_sync` but must not duplicate it; Task 6 may only retarget integration seams while preserving Task 4 behavior. |
| Task 4 | Task 6 | `server_args.py`, scheduler/tokenizer/TP-worker/model-runner/weight-updater seams | Shared-file edits are sequential. Task 6 begins from Task 4’s reviewed commit and must keep native `--enable-lora --enable-lora-staging` behavior intact. |
| Task 5 | Tasks 6, 7, 8, 9, 10, 11 | Canonical `srt/oft`, OFT config/integration/request/tokenizer APIs and OFT tests | Task 5 owns the canonical OFT API. Task 6 retargets consumers to it; later work may fix defects only through an owning-task regression loop. |
| Task 5 | Task 6 | `server_args.py`, scheduler/model-runner/CUDA-graph/MoE/model import seams | Task 5 lands the provider first; Task 6 updates every consumer and deletes all legacy selectors/imports. Do not combine these tasks into an unreviewable port. |
| Task 6 | Task 7 | Public CLI/API and runtime import surface | Task 7 codifies Task 6’s absence guarantees. Guards must reject `--oft-impl`, reject `--peft-method lora`, and forbid candidate `srt/peft` files/imports. |
| Tasks 4–7 | Task 8 | Candidate modes and lifecycle behavior used by scenario runner | Task 8 treats reviewed runtime code as fixed. Harness defects stay in Task 8; runtime defects route back to the owning implementation task with a regression test. |
| Tasks 2–3 | Task 8 | Bundle writer/comparator plus immutable scenario inputs | Task 8 must use, not fork, the Task 2 schema and Task 3 manifests. Harness-sensitivity mutations operate only on copied bundles. |
| Task 8 | Task 9 | Frozen source bundles, CaseKey index, tolerance input, candidate harness | Source oracle directories become read-only after validation. Candidate comparison never regenerates or edits a source oracle in place. |
| Task 9 | Task 10 | Validated candidate functionality and common environment provenance | Distributed/performance testing starts only after the complete functional matrix passes. Performance pairs use matching GPU model and environment. |
| Tasks 2–10 | Task 11 | All commits, test outputs, Condor IDs, bundle hashes, performance evidence | Task 11 is audit/documentation only unless it finds a defect; defects route to their owning task and are re-reviewed. |

All unlisted task pairs have no direct file/interface overlap beyond the reviewed branch state and final audit provenance.

## Preflight rulings

Task 1: Ruling: The plan’s compile and pytest commands are project workloads and must not run on a Slurm or mpi3 login node. Run them in bid-100 HTCondor allocations on mpi3 (CPU-only checks may still use a GPU allocation) and preserve logs under the run-store contract. Cost if wrong: slower queueing and use of a GPU for CPU-heavy validation; benefit is compliance with the cluster safety boundary.

Task 1: Ruling: Fix the reproducible Condor setup defect by exporting a valid `CUDA_HOME`, but do not mutate source dependencies, install an AMD-only runtime on H100, or filter the required unit suite to hide optional-stack collection failures. Run the exact compile/unit/VLM commands against the merged tree and both merge parents in parallel under one validated dependency manifest. Residual `nixl`/`mori` collection failures may be recorded as environment-only only if their tracebacks reproduce on both parents with no additional merged-only collection or assertion failures; otherwise they block Task 1. Cost if wrong: Task 1 may accept less full-suite assertion coverage than an all-optional-dependencies environment could provide, so the exact missing coverage and parent evidence must remain visible in the report.

Task 3: Ruling: The generated dense NVFP4 checkpoint is a versioned test fixture, not source code. Its exact ModelOpt revision, source checkpoint revision, calibration manifest, output hash, and B200 job ID must be recorded before Tasks 8–10 consume it. Cost if wrong: regeneration may be required if the manifest is incomplete.

Task 8: Ruling: `legacy_lora` and `legacy_oft` are source-only oracle modes, never candidate/public modes. Run them for every mandatory cell the unchanged source backend genuinely supports; if the legacy implementation cannot initialize a cell, record `unsupported_by_legacy` with the unchanged-source traceback and still require base, canonical OFT, and native staged LoRA comparisons for that cell. Cost if wrong: a legacy-only unsupported precision will not block retirement, but its legacy-to-canonical equivalence will remain unproven for that cell.

Task 9: Ruling: The mandatory 96 count is interpreted as 60 source plus 36 candidate shards. Common-legacy comparisons are evaluations over the frozen source bundles, not additional candidate modes or public compatibility promises. Cost if wrong: reporting may need adjustment, but no mandatory canonical/native execution is omitted.

Task 10: Ruling: Throughput and peak-memory gates compare matching hardware, model revision, precision, graph setting, tensor/expert parallel layout, prompts, and adapter fixture. A mismatched pair is invalid evidence rather than a tolerance exception. Cost if wrong: some runs may need resubmission to obtain a valid pair.

## Task progress

Task 1: complete (commits bf3d1aaf5..4d2dc0c3d, review clean)

Task 2: in progress (base 4d2dc0c3d)

Task 2: fix round 1/5 (3 findings addressed, 2 Important open; commits 422141cfd..53d1e3222)

Task 2: fix round 2/5 (2 findings addressed, 0 open; commits 53d1e3222..517d05d3a)

Task 2: complete (commits 4d2dc0c3d..517d05d3a, review clean)

Task 3: in progress (base 517d05d3a)

Task 3: Ruling: The brief’s NVIDIA Model Optimizer repository/path has moved. Use the current official NVIDIA/Model-Optimizer `examples/hf_ptq` workflow pinned at commit `029c67f27e67088fb19ac0a9af241dc2bc740650`, after verifying its current CLI, while preserving the brief’s source model, NVFP4 format, FP8 KV-cache format, calibration size 512, batch size 8, B200 requirement, immutable staging, and full provenance. Cost if wrong: the generated dense NVFP4 artifact may differ from an older ModelOpt implementation, so the exact source revision, CLI, calibration manifest, and all export hashes must remain recorded and later comparisons must use this immutable artifact only.

Task 3: Ruling: mpi3 home quota rejects even small execution-copy writes and `/lustre/scratch/zqiu` is mounted only on compute nodes. Use the active agent-owned allocation to seed a contract-equivalent run store and immutable checkpoint staging directly under compute-mounted scratch, submit subsequent jobs without duplicating Hugging Face caches in home, and mirror compact final provenance/manifests/logs to Slurm before task completion. Cost if wrong: scratch may have a different retention policy, so no task may consume an artifact until its full hash manifest and durable evidence have been mirrored outside scratch; immutable model payloads remain on scratch only by explicit recorded path.

Task 3: Ruling: The pinned official dense FP8 snapshot is intentionally unsharded: it contains exactly one `model.safetensors` and no `model.safetensors.index.json`. Preserve the upstream snapshot byte-for-byte and extend fail-closed preflight to accept exactly two layouts: (a) an index whose complete contiguous numbered shard set is present and hashed, or (b) exactly one regular `model.safetensors`, no index, and no other safetensors files. Record the layout explicitly and hash the single file; never synthesize an index. Cost if wrong: a downstream consumer that assumes an index may need adaptation, so tests must prove ambiguous/multiple unindexed files and incomplete indexed sets are rejected before matrix execution.

Task 3: Ruling (supersedes the generated-dense-NVFP4 and ModelOpt/PTQ rulings above): use the directly downloadable `OPENZEKA/Qwen3-4B-Instruct-2507-NVFP4` snapshot pinned at revision `7009563e02c47b3ce728ecdc8cab2f0d9cd52ee4`. All six dense/MoE × BF16/FP8/NVFP4 cells are upstream snapshots with pinned revisions; Task 3 performs no local quantization or checkpoint generation. Cost if wrong: a third-party quantized snapshot may not match a locally generated NVIDIA export byte-for-byte, so its repository, revision, checkpoint hash, and runtime compatibility must remain explicit evidence.

Task 3: direct-model TDD update complete in bid-100 Condor job `17497109.0`: accepted RED was 3 focused failures in 0.53s against the old local-NVFP4 behavior; GREEN was the full focused file, 28 passed in 0.67s. Static compilation and JSON/JSONL parsing passed in job `17497087.0`. The authoritative Slurm files exactly match tested bundle SHA-256 `6c08836763484603e945e756dd4d83f063fd342c420b548ef4fabe7ed9adad2f`.

Task 3: all six checkpoints are promoted at immutable revision paths. Dense FP8 manifest hash is `2555ca55f5e0cb2c58e3014c8356c01660aeeaffe2b9388186ea2036008c9762`; dense NVFP4 manifest hash is `d009978597b1a807de5c479562e1f2dbed7d4038861250ed8d67762473caa5c3`. The integrated all-six real-checkpoint preflight passed in bid-100 Condor job `17497109.0`; manifest SHA-256 is `6823ad50ef71880f4b0e60e14a151c0c3209b4028a420bb664a83a17d36d5e7e`, with evidence under `20260831T092125Z-126d11/precision-downloads`.

Task 3: complete (commit `86fd769df`, self-review clean; focused suite 28 passed; all-six real-checkpoint preflight passed)

Task 4: complete (base `9e7cbf1e1`; tests `601d8faded`, `adb8e72f3`; implementation `a3ea65690`; review clean)

Task 4: Ruling: freeze the Sphere source endpoint at `20e7efe8ecff9979063a2d8efe12fa95eae3b0a8`. Port generic `srt/adapter_sync`, place native staging in `srt/lora/staged_manager.py`, and register only native LoRA in the tokenizer staging backend. Exclude OFT staging, `--oft-impl`, and all legacy `srt/peft` fallbacks; Tasks 5–6 own those surfaces. Preserve existing Impossible multi-LoRA/MoE-LoRA behavior whenever `--enable-lora-staging` is disabled. Cost if wrong: Task 4 would either duplicate Task 5 provider work or silently change the existing native LoRA serving path before equivalence testing.

Task 4: accepted RED was bid-100 Condor job `17497214.0`, exit 2, failing only on the intentionally missing `sglang.srt.adapter_sync`, staged request types, `_extend_lora_extra_key`, and `sglang.srt.lora.staged_manager`. First GREEN was job `17497232.0`: 64 passed, 15 warnings, 76.31s. The exact implementation archive SHA-256 is `c2243c5963c30e6e8f07be53faab277171afc9941e15759eb3e2393df50600f5`.

Task 4: GPU validation on the exact source archive: job `17497238.0` passed 3 native staged-update/CUDA-graph/hidden-slot cases in 700.82s; A100 job `17497239.0` passed TP=2 before the MoE setup hit the home-directory download quota; H100 job `17497258.0` passed the focused TP=2 case in 575.99s. Post-commit job `17497295.0` passed the 64-test unit suite in 75.59s with commit `a3ea656906e9e381c1c70fab1a6e155514d192e3` recorded in provenance.

Task 4: MoE validation required a task-specific `/fast/zqiu/huggingface-task4/` cache because the 28.6 GB Qwen1.5-MoE checkpoint could not be reconstructed within the home quota. Jobs `17497278.0` and `17497302.0` are retained as negative environment evidence: the first exhausted server-start time while downloading; the second correctly failed closed because seven indexed shards were still `.incomplete`. Dedicated pinned download job `17497369.0` verified all 8 shards at revision `1a758c50ecb6350748b9ce0a99d2352fd9fc11c9`, totaling 28,632,144,944 bytes. Offline three-B200 rerun `17497811.0` then failed before assertions because the retained CCCL include path pointed one directory above the actual `cub/cub.cuh`; no source change was made. Corrected three-GPU retry `17498235.0` reached TP initialization but failed with an `i203` NCCL system error before model loading. Two-GPU diagnostic `17498316.0` confirmed that the fixture's base-GPU-1 plus TP=2 layout requires three visible devices and is not acceptance evidence. Corrected three-GPU job `17498496.0` passed the sharded MoE placement test on `i104`: 1 passed, 15 warnings, 376.90s. Redundant H100-pool job `17498553.0` eventually passed without intervention: 1 passed, 15 warnings, 375.34s.

Task 4: final acceptance is green. The exact hashed implementation passed 64 focused unit tests, 3 single-node staged-update/CUDA-graph/hidden-slot cases, focused TP=2, and focused sharded-MoE placement. Boundary scan and self-review are clean; no OFT or legacy-PEFT runtime path was introduced.

Task 4: compact source-excluding evidence archive `/lustre/home/zqiu/task4-complete-evidence-a3ea65690.tar.gz`, mirrored to the local run-state store, SHA-256 `3d2a7c92dbc3290f908c91c2589e34e81ed2493978b12e211bb6bc53106e9ef3`.

Task 5: complete (base `29d47c5853`; tests `9f605b6ba`, `a777f304f`, `4a2a622ce`; implementation/fixes `03866d199`, `aa48ad335`, `61f64c5ad`; lazy-import contract `7a01e2a4e`; self-review clean)

Task 5: Ruling: reuse the Task 4 Sphere pin `20e7efe8ecff9979063a2d8efe12fa95eae3b0a8`. Port every canonical `python/sglang/srt/oft` provider file, relocate the OFT-only control surface into `srt/oft`, and add the OFT row to the existing shared adapter-sync registry. Exclude runtime consumer rewiring, `--oft-impl`, single-active LoRA, legacy `sglang.srt.peft` imports, and all legacy double-buffer selectors. Task 6 owns consumer rewiring; Task 7 owns removal guards and deletion. Cost if wrong: combining consumer rewiring with the provider port would obscure ownership and could regress Task 4 native staged LoRA before the OFT provider contract is independently verified.

Task 5: exact source accounting is 37 frozen canonical OFT Python files, 43 target files, zero missing files, and exactly six OFT-owned additions: `base/eviction_policy.py`, `config.py`, `integration.py`, `io_types.py`, `tokenizer_hooks.py`, and `tokenizer_mixin.py`. The final boundary scan is empty for `sglang.srt.peft`, `oft_impl`, `peft_max_lora_rank`, and `peft_double_buffer` inside `srt/oft`.

Task 5: accepted initial RED was job `17498702.0`, failing because `sglang.srt.oft` did not exist. First implementation job `17498907.0` produced 50 passed and 2 targeted failures; isolated fixes passed in jobs `17498921.0` and `17498967.0`. Commit `7a01e2a4e` then passed the 52-test focused set in job `17499003.0` and all 187 `test/srt/oft` cases on H100 in job `17499004.0`.

Task 5: self-review found an unresolved `OFTRef` forward reference that would break the existing ServerArgs `get_type_hints` consumer. Regression commit `4a2a622ce` produced the intended RED in job `17499249.0`: `NameError: name 'OFTRef' is not defined`, with 1 failed and 5 passed. Job `17499079.0` was an identical earlier submission that matched `g136` but remained stalled before runner startup with zero user CPU and no test output; it was not cancelled and is not acceptance evidence.

Task 5: final commit `61f64c5ad6b4a1009f0301d996febb0c3031fe27` imports `OFTRef` into the config namespace. Exact archive SHA-256 `4f01ba6a356e92e66f613e08694635b0aea6f44754c85c089004278deb1fec8a` passed 53 focused tests in bid-100 job `17499284.0` and all 187 `test/srt/oft` tests on H100 host `g198` in job `17499286.0`. Final run records are under `20260831T143630Z-task5-final-61f64c/` and mirrored source-excluding to the local run store.

Task 6: complete (base `38858aba3`; tests `0c83aba36`, `42778d86f`; implementation/fixes `a51d26d55`, `1573428a5`; self-review clean)

Task 6: Ruling: make canonical OFT the only `--peft-method` value, keep native multi-tenant LoRA solely under `--enable-lora`, and route ServerArgs, request identity/versioning, scheduler admission, tokenizer staging, weight updates, model runner, CUDA graphs, MoE, DeepSeek MLA, and Kimi policy directly through `srt/oft`. The scheduler weight-updater metadata forwarder is included because the canonical streamed-weight seam consumes adapter config/name/ID. No legacy `srt/peft` import, `oft_impl` selector, or `peft_method == "lora"` branch is introduced. Cost if wrong: Task 7 could delete a still-live legacy call path, so the changed-path boundary scan and runtime import tests remain acceptance evidence.

Task 6: accepted selector RED was bid-100 job `17499407.0`: the new real-CLI and IPC-export tests failed in the expected three places while both rejection-only cases already passed. Implementation job `17499474.0` then produced 10 passed and one test-fixture failure because a dummy `ServerArgs` instance omitted `served_model_name`; commit `42778d86f` corrected only that fixture. Focused rerun `17499480.0` passed all 11 CLI/config/import tests in 30.20s.

Task 6: the first combined full-directory run, job `17499511.0`, passed 209 tests plus 12 subtests and exposed two real Task 6 routing regressions together with 13 environment/order or inherited LoRA failures. Root-cause analysis found that the new OFT route had removed native LoRA's early eligibility guard without replacing it with an explicit `peft_method == "oft"` check. The content-equivalent one-file fix (patch SHA-256 `29de6ab2963b85c18b4f2695bd959bb06059d11e1b015e2de4a65e8d436d062c`) passed 30 focused routing/OFT tests in job `17499523.0`, then landed as `1573428a5`.

Task 6: exact final archive SHA-256 `07d81736d2870cae645d9a9a49568c2c76730b7c9e91182ed2bd2324d9beed45` was tested in job `17499534.0`: all 33 registered OFT tests and all 14 adapter-sync tests passed; LoRA produced 166 passed, 12 passed subtests, and 11 failures. Pre-Task-6 base comparator job `17499537.0` produced the identical 11 failure node IDs and identical 166-passed/12-subtest result, proving no remaining LoRA regression in the Task 6 delta. The inherited failures are confined to tokenizer-worker parallel-context fixtures, the existing MoE in-place fixture, and attention-TP memory-pool sizing. They are recorded but are outside Task 6's changed paths.

Task 6: final acceptance is green for the Task 6 delta. Canonical OFT CLI acceptance and legacy-selector rejection pass; runtime imports, IPC compatibility exports, staged update routing, request cache identity/versioning, graph/MoE/model seams, OFT tests, and adapter-sync tests are covered. The changed-path scan and `git diff --check` are clean, and the authoritative worktree is clean at `1573428a55c92ceb35d707e7f818f5cfb622ec22` before this ledger update.

Pre-Task-7 LoRA baseline cleanup: complete (implementation `bbde5bf91dc7c3e8197916dd6e898999afc1f25a`; review and exact-commit validation clean)

Pre-Task-7 LoRA baseline cleanup: the 11 failures inherited by Task 6 were repaired before legacy deletion. The fix publishes the real parallel context in tokenizer-manager unit fixtures, keeps `server_args.dp_size` in the distributed-upsert fixture because that path still consumes it directly, replaces the MoE fake wrapper with a real `torch.nn.Module`, and makes LoRA memory-pool fallback sizing use the captured attention TP size for attention projections rather than the outer TP size.

Pre-Task-7 LoRA baseline cleanup: canonical OFT was used as the reference implementation. OFT confirmed that adapter buffers should follow the wrapped module's local partition metadata and that wrapper contracts assume a real `nn.Module`. The comparison also exposed the current mixed topology seam: LoRA `from_tensors` consumes the published runtime context, while its distributed path still reads `server_args.dp_size`; both representations remain intentional until that path is unified.

Pre-Task-7 LoRA baseline cleanup: diagnostic job `17499594.0` removed all original 11 failures but exposed six distributed-upsert fixture failures after `server_args.dp_size` was removed too aggressively. Corrected job `17499601.0` passed the full LoRA directory (`177 passed`, `12` passed subtests). Exact implementation archive SHA-256 `ced0f434d1f052e8e368b076d35291e5be0963ff973e9486fa8fe51e2dc719f1` then passed in bid-100 job `17499623.0`, with separate Python processes: OFT `33 passed`, LoRA `177 passed` plus `12` passed subtests, adapter-sync `14 passed`, and suite exit code zero. The authoritative worktree was clean at `bbde5bf91dc7c3e8197916dd6e898999afc1f25a` before this ledger update.

Task 7: complete (base `1244e0f63`; implementation/tests `d9fee6a0e`; self-review and exact-commit validation clean)

Task 7: the candidate never contained `python/sglang/srt/peft`, so no physical deletion was required. A new repository-wide guard now asserts the package remains absent and scans active `python/`, `test/`, and user-facing `docs/` text for legacy imports, package paths, `oft_impl`, `--oft-impl`, `--peft-method lora`, and `peft_method == "lora"`; only the guard's own fixtures and historical plan/spec directories are excluded. Existing negative CLI tests construct retired spellings from fragments so they continue proving rejection without weakening the source scan.

Task 7: lifecycle audit found five of six planned CPU rollback contracts already covered: duplicate/stale versions, wrong adapter ID, malformed config, failed stage preserving active state, and independent A/B versions. The missing streamed-unload contract exposed two real defects. OFT unload released routing metadata without resetting the dense slot to mathematical identity or removing its active version, and its non-staged MoE rotations remained bound directly on FusedMoE modules. The final implementation restores the slot to identity, clears streamed expert bindings, drops version metadata, and only then releases the slot mapping.

Task 7: accepted RED job `17499656.0` produced exactly two failures: five forbidden literals in negative tests and stale dense/version state after unload. Focused job `17499662.0` passed both after the first correction. Self-review extended the same unload contract to expert state; job `17499671.0` produced the intended single failure and job `17499676.0` passed after clearing the streamed MoE bindings. Diagnostic full gate `17499667.0` passed `245` tests plus `12` subtests before the expert extension.

Task 7: exact commit `d9fee6a0e81c9831f1d126ac461619f52f031b2d` archive SHA-256 `cc5c453c50d06c2e9279d33fc22213daaa88d3234219c965fc5bbe2c16fcf650` passed bid-100 job `17499681.0`: `python -m compileall -q python/sglang` passed, the exact Task 7 CPU gate passed `245` tests plus `12` subtests with `15` warnings in `78.57s`, and the final marker is `compile=0 download=0 cpu_gate=0`, exit code zero. Source-excluding evidence is mirrored locally under `20260831T165644Z-d9fee6/task7-final/`. The authoritative worktree was clean at `d9fee6a0e81c9831f1d126ac461619f52f031b2d` before this ledger update.
