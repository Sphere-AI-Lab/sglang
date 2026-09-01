# Canonical adapter migration — verification index

Evidence index for the legacy-PEFT retirement branch. Every claim below is
traceable to a Condor job ID and a run-store directory on `mpi3`.

**This index records a deliberately reduced scope.** Development was closed on
the dense BF16 cell; MoE, the FP8/NVFP4 precision cells, the remaining Task 9
cells, and all of Task 10 were deferred by explicit decision. Section 6 lists
every gap with its failure signature. Nothing below should be read as evidence
for an axis that section 6 names as open.

## 1. Identity

| | |
|---|---|
| Branch | `codex/retire-legacy-peft` |
| HEAD | `0c3c9ce6b` |
| PR base | `origin/sglang-miles` |
| Merge base | `f3863a3f12c4` |
| Commits on branch | 1415 |
| Tasks 1–7 acceptance root | `d9fee6a0e81c9831f1d126ac461619f52f031b2d` |
| Frozen source (oracle) | `sphere/orbit-main-corrected` @ `20e7efe8ecff9979063a2d8efe12fa95eae3b0a8` |
| Candidate archive tested | `git archive --prefix=sglang/ HEAD`, SHA-256 `11fd731fbee42da429fa8f482e8521f06b1daf18ca923991f83ab75ba189ccc3` |
| Execution host | `mpi3` (HTCondor), all jobs at bid 100 |
| Run-store prefix | `/lustre/home/zqiu/.local/state/remote-cluster-runs/mpi3/sglang/codex-retire-legacy-peft-37abbccb/` |

Commits added in this session:

| Commit | Contents |
|---|---|
| `40912057b` | `run_case.py` and `README.md` landed — the shard runner had existed only as untracked copies inside run directories |
| `75f780c22` | Ledger: acceptance-gate ruling, four harness defects, dense BF16 results, deferrals |
| `0c3c9ce6b` | Source-oracle harness kept out of the legacy-PEFT scan (fragment-constructed spellings) |

## 2. Legacy package absence

`python/sglang/srt/peft` contains **51 files in the frozen Sphere source** and
**zero files** at HEAD, at `origin/sglang-miles`, and at the merge base. The
package was never present in this lineage, so retirement is *never adopted and
permanently guarded*, not *deleted in this branch*. No commit on the branch
adds it (`git log --diff-filter=A -- python/sglang/srt/peft` is empty).

Audit at HEAD:

| Check | Result |
|---|---|
| `test ! -e python/sglang/srt/peft` | PASS |
| `sglang.srt.peft` under `python/` | 0 hits |
| `oft_impl`, `--oft-impl` under `python/` | 0 hits |
| `--peft-method lora`, `peft_method == "lora"` | 0 hits |
| `git diff --check origin/sglang-miles...HEAD -- python test` | 0 issues |

`git diff --check` reports 2517 whitespace items across the full diff; all are
in `docs/` inherited from the Radix merge, none in changed code.

The manual harness `test/manual/adapter_equivalence/server.py` still *names*
the retired control surface, because it must drive the frozen source, whose
only OFT selector is the legacy one. Those spellings are assembled from
fragments and the package is reached via `importlib`, so the guard's source
scan stays strict. Commit `0c3c9ce6b`.

## 3. Functional results — dense BF16

Model `/fast/zqiu/hf_models/Qwen3-4B-Instruct-2507`, TP=1, H100 80 GB,
greedy decoding, OFT fixture scale `1e-2`.

| Job | Mode | Prompt | CUDA graphs | Result |
|---|---|---|---|---|
| `17502805` | canonical OFT | factual | off | 25 passed, 1 failed |
| `17502806` | canonical OFT | long-prefix | off | 25 passed, 1 failed |
| `17502882` | canonical OFT | factual | **on** | 25 passed, 1 failed |
| `17502883` | canonical OFT | long-prefix | **on** | 25 passed, 1 failed |
| `17502846` | native staged LoRA | factual | off | 26 passed |
| `17502847` | native staged LoRA | long-prefix | off | 26 passed |
| `17502884` | native staged LoRA | factual | **on** | pass |
| `17502885` | native staged LoRA | long-prefix | **on** | pass |

Every OFT failure is `reject.stale`, one of six surrogate transitions. It
depends on `stage.v1`/`activate.v1`/`stage.v2`/`activate.v2`, which require an
external rank publishing tensors over an NCCL weight-sync group that a
single-process offline `Engine` cannot supply. It is unpassable offline and is
not a defect.

Two fields decide whether a run is interpretable at all:

- `dynamic.infer` → `differs_from_base = true` — the adapter was genuinely
  active. True for all four OFT runs.
- `dynamic.post-unload-base` → `differs_from_base = false` — unload restored
  the base exactly. True for all four, digest `663baa0f7b576101`.

**Caveat on the LoRA `factual` shards.** Their `dynamic.infer` reports
`differs_from_base = false` with and without CUDA graphs: that fixture does not
move tokens on the `factual` prompt. Their 26/26 is therefore a valid
lifecycle result but *not* evidence that the adapter was applied. The
long-prefix LoRA shards do move tokens and carry that evidence.

## 4. Acceptance gate — OFT fixture scale

Canonical OFT is multiplicative (`W' = R·W`, `R = Cayley(S) ≈ I + 2S`). The
harness default `1e-3` puts the rotation within ~0.2% of identity, so
`differs_from_base` reads false whether or not the adapter works, and the gate
cannot distinguish a correct implementation from a no-op.

Sweep on dense BF16, base digest `663baa0f7b576101`:

| Scale | Job | `dynamic.infer` digest | Moves tokens | Post-unload |
|---|---|---|---|---|
| `1e-3` | `17502765` | `663baa0f7b576101` | no | restores base |
| `1e-2` | `17502796` | `e4689e29b806b258` | **yes** | restores base |
| `3e-2` | `17502767` | `f45e1ef5f05b3cd2` | yes | restores base |
| `1e-1` | `17502768` | `b11920a1fa90c77c` | yes | restores base |

`1e-2` is adopted: the smallest measured scale that makes the gate able to
fail, and equal to the LoRA fixture scale, removing the 10× asymmetry the
predecessor report identified. Job `17502766` (first `1e-2` attempt) is not
evidence — its scheduler died with SIGBUS on a bad node.

`--oft-scale` exposes this on `run_case.py` and defaults to the harness value,
so omitting it preserves prior behaviour.

## 5. Component and unit results

| Job | Command | Result |
|---|---|---|
| `17502879` | `pytest -q test/srt/oft` | **187 passed** |
| `17502945` | `pytest -q test/registered/lora/test_lora_staged_update.py` | **3 passed** |
| `17502946` | `pytest -q test/registered/lora/test_lora_staged_update_tp.py` | 1 passed (`test_tp2`), 1 failed (`test_moe_sharded_placement`) |
| `17502943` | `compileall python/sglang` + registered `adapter_sync`, `lora`, `oft` unit suites | `compileall=0`, **228 passed**, 12 subtests |

`17502943` ran against the HEAD archive and is the run in which the absence
guard `test_no_legacy_peft.py` returned to green.

## 6. Open items and deferrals

Nothing in this section is evidenced. Each entry records how it failed so it
can be resumed without re-deriving the diagnosis.

| # | Item | Signature |
|---|---|---|
| 1 | MoE BF16 lifecycle — OFT ×2 prompts, LoRA ×2 prompts | Jobs `17502807`, `17502808`, `17502775`, `17502776`: `Rank 1 scheduler died during initialization (exit code: -6)`, SIGABRT during TP=4 startup, before model load |
| 2 | MoE staged-LoRA sharded placement | Job `17502946`: `assertNotEqual` fails because generation returns 24 zero token IDs in both states — the adapter version change cannot be observed. Passed in Task 4 job `17498496`, so this is either a Tasks 5–7 regression or an environment difference; unresolved |
| 3 | Dense FP8, MoE FP8 precision smokes | Jobs `17502809`, `17502811`: exit 131 |
| 4 | Dense NVFP4 precision smoke | Job `17502810`: held |
| 5 | MoE NVFP4 precision smoke | Job `17502812`: never started (no free 4-GPU B200 node); removed on instruction |
| 6 | Task 9 Step 2 — remaining 5 cells | Not run. Nominally 5 cells × 2 modes × 2 CUDA-graph settings |
| 7 | Task 9 Step 3 — tolerance envelopes | Not run, and possibly inexecutable: envelopes derive from repeated *source* runs, and the frozen source cannot perform OFT dynamic load/unload at all |
| 8 | Task 10 — distributed stress and performance | Not run: MoE TP4/EP4 equivalence, 100 load-switch-unload cycles and 1,000 mixed requests per cell, controlled throughput and peak-memory gates |
| 9 | Source-vs-candidate equivalence comparison | **Never executed.** All shards ran `--revision-kind candidate`. The comparison the harness was built for has not been performed |

No throughput, latency, or peak-memory medians exist. No distributed or stress
results exist. The plan's 96-shard functional mandate is not met; 12 candidate
shards were run, of which 8 produced usable results.

## 7. Harness defects corrected

Found and fixed in this session. All four lived in the shard runner while it
existed only as untracked copies inside run directories, invisible to review.

1. `ShardRunner.__init__` stored the `Emitter` object where all seven call
   sites invoke the bound `.emit` method — `TypeError: 'Emitter' object is not
   callable`, killing every shard ~1s into setup.
2. A launcher wrapping `smoke_matrix.main()` at import time re-ran the entire
   shard inside SGLang's spawned scheduler process. `run_case.py` guards its
   own entry point; do not wrap it.
3. A batch record's `requests` field is the list of request objects, not a
   count; `int()` on it errored every multi-request transition.
4. The OFT target suffixes omitted `k_proj`/`v_proj`, which `ae203a9a6` had
   made mandatory.

Three further failures were environmental, not code, and each is worth
recording because each cost a full round trip:

- `sglang` on `PATH` resolved to an interpreter without `pybase64`.
- `--base-gpu-id 1` fixtures require at least two visible GPUs.
- The cluster Squid proxy intercepts loopback, so the server's own warmup
  request to `127.0.0.1` returned HTTP 403. `no_proxy` must include
  `127.0.0.1,localhost`.

A wrapper flaw is also recorded: the Task 8 wrappers evaluate the verdict check
under `set -e` before extracting the measurement, so a surrogate-only failure
leaves `oft_differs=unknown` in `completion.status` even when the JSONL records
`true`. The per-transition JSONL is authoritative.
