# Native adapter qualification

Executable modes are exactly `base`, `native_lora`, and `native_oft`. Both
immutable checkouts must expose the selected native implementation or preflight
fails. Native OFT means `sglang.srt.oft` with `oft_type="oft"`, using shared
rotations for q/k/v and gate/up. `--mode native_oft` selects
`peft_method="oft"` and `oft_type="oft"` without an additional user flag.
Its fixture builder emits fused `qkv_proj`/`gate_up_proj` compact tensors and
the shared expert representation. Canonical split OFT is outside this gate.

Registered CPU contract tests and LoRA/OFT TP component tests run in CI. The full
matrix needs separate H200 allocations. This README describes a procedure and
does not assert that GPU qualification has already run.

## Semantics and evidence

Immediate path loads read an adapter directory; tensor loads use the Engine's
serialized-tensor API; distributed immediate loads use an external sender and
the native load-from-distributed endpoint. Each publishes immediately. Staging
uses `double_buffer=True` and a hidden slot; only activation of the exact
adapter name, ID and version publishes it. Every declared full-lifecycle action
must execute and produce its required result, including rollback, unload retry,
cancellation, eviction, restart and exact base restoration.

Visible GPU 0 belongs to the sender. Model ranks start at GPU 1: dense matrix
cells use TP1 on GPU 1, dense component tests use TP2 on GPUs 1–2, and MoE
matrix cells use TP4/EP4 on GPUs 1–4. MoE therefore needs at least five visible
GPUs. NCCL carries tensor broadcasts over the weight-update group: sender rank 0
plus model ranks 1…TP. Adapter-result consensus uses the model's separate TP CPU
group with Gloo. Compare identical topologies and hardware classes. Component
failures are injected on physical TP ranks 0 and 1 immediately before that Gloo
result-consensus collective; harness reply-position injection alone cannot prove
physical-rank behavior.

Observations require output IDs, text, finite token logprobs, configured selected
scores and shapes, exact adapter state, and expected normalized errors. Missing
evidence fails closed. Tokens, text, shapes, request order, lifecycle state,
errors and restored base tokens compare exactly. Numeric envelopes derive only
from three immutable reference bundles. Each bundle includes three post-warm-up
performance repetitions from fresh measurement engines. Median candidate
throughput must be at least 95% of reference; median allocated and reserved
CUDA peaks must each be at most 105%. Peaks come from every scheduler TP rank.
Startup/latency are recorded evidence, without additional acceptance thresholds.

## Freeze inputs and build the manifest

Use final clean committed checkouts, pinned complete checkpoints, tokenized
`prompts.jsonl`, and immutable complete-target fixtures. The Python APIs
`build_preflight_manifest` (`preflight.py`) and `build_fixture_set` (`run_case.py`)
build checkpoint and fixture inputs; neither is a fixture-generation CLI.
The fixture manifest maps `policy-a`, `policy-b`, `1`, and `2` to absolute
directories, each containing `adapter_config.json`, `adapter_model.safetensors`
and `sha256.json`. Cover q/k/v/o, gate/up/down, embedding and LM head, including
experts for MoE. Apply the same reviewed memory-measurement boundary to both
checkouts before freezing SHAs.

From the checkout with this harness, in Bash, substitute real values:

```bash
HARNESS=test/manual/adapter_equivalence
REFERENCE_SHA=REPLACE_WITH_EXACT_40_HEX_REFERENCE_SHA
CANDIDATE_SHA=REPLACE_WITH_EXACT_40_HEX_CANDIDATE_SHA
RUN=/absolute/durable/run-store
python "$HARNESS/qualification.py" \
  --reference-sha "$REFERENCE_SHA" --candidate-sha "$CANDIDATE_SHA" \
  --artifact-root "$RUN/shards" --output "$RUN/qualification.json"
```

The matrix is four H200 model cells (dense/MoE × BF16/FP8) × two native modes ×
graph off/on × (three reference + one candidate) = **64 shards: 48 reference and
16 candidate**. `--case-id qwen3-4b-bf16.native_lora.graph-0` builds an explicitly
non-qualifying subset. `base` is diagnostic; every native lifecycle contains base
requests. NVFP4/B200 dense and MoE are deferred outside the active manifest and
explicitly **unqualified**, with no H200 substitution.

## Run a shard

This candidate BF16 dense LoRA example uses the current parser exactly. Dispatch
each manifest shard from its actual revision's environment and checkout. Source
uses repetitions 0/1/2; candidate uses 0. Select the appropriate mode, fixtures,
checkpoint, graph flag and topology. FP8 adds `--quantization fp8`; MoE uses
`--architecture moe --tp-size 4 --ep-size 4`; MoE FP8 adds `--moe-runner triton`.
OFT uses `--mode native_oft` with its matching shared-rotation fixture manifest.
Run on the selected checkout in an approved Slurm H200 allocation. The allocation
must expose the sender and every model GPU; `SLURM_JOB_ID` below identifies that
existing allocation. Do not start the final matrix before the candidate is frozen.

```bash
CASE=qwen3-4b-bf16.native_lora.graph-0
SHARD="$RUN/shards/$CASE/candidate/rep-0"
mkdir -p "$SHARD"
ARGS=(
  --mode native_lora --case-id "$CASE"
  --revision-kind candidate --revision-sha "$CANDIDATE_SHA"
  --architecture dense --precision bf16 --cuda-graph off
  --model-path /absolute/checkpoints/Qwen3-4B-Instruct-2507
  --checkpoint-manifest /absolute/inputs/checkpoint-manifest.json
  --prompts-file /absolute/inputs/prompts.jsonl
  --fixture-manifest /absolute/inputs/lora-fixtures.json
  --tp-size 1 --ep-size 1 --base-gpu-id 1 --port 30000 --repetition 0
)
set -o pipefail
srun --jobid "$SLURM_JOB_ID" --ntasks=1 \
  python "$HARNESS/run_case.py" "${ARGS[@]}" --selection full \
  --bundle-output "$SHARD/bundle.json" \
  --completion-output "$SHARD/complete.json" \
  2>"$SHARD/stderr.log" | tee "$SHARD/events.jsonl" >"$SHARD/stdout.log"
```

Run `run_case.py` directly: its guard is necessary for scheduler spawn. The pipe
records diagnostic stdout in two separate regular files. Never reuse completed
shard directories or overwrite evidence. Exact paths are
`shards/<case-id>/<source|candidate>/rep-<n>/{bundle.json,complete.json,events.jsonl,stdout.log,stderr.log}`;
`blocked.json` is reserved in that same directory. The validated bundle is
published atomically and exclusively after teardown, then its completion marker
(`status: complete`, `bundle_hash`). A bundle without a matching marker is
incomplete. Missing bundles/markers, unexpected paths, symlinks and hardlinks
fail inventory checks; bundle identities and completion digests must validate.
The three log paths are allowed diagnostics, not required qualification evidence
or content-hashed substitutes for a bundle.

Runner exit codes are 0 success, 1 failure, and 2 recorded timeout; parser usage
errors also exit 2. `--selection smoke` is a diagnostic subset and publishes no
qualifying bundle or completion marker, even on exit 0. Use a separate directory
for smoke. JSONL, smoke success and TP component success alone never qualify.

## Stress is a separate invocation

`--selection full` does not run stress. `stress_case.py` has no standalone CLI;
`StressSpec`, `StressResult` and `run_stress` are exported Python interfaces.
`ShardRunner.run_stress` connects them to the real native Engine and sender.
Save this guarded operator driver outside the shard root as `$RUN/stress_driver.py`:

```python
import json
from dataclasses import asdict
from adapter_equivalence.run_case import ShardRunner, build_parser, inputs_from_args

def main():
    args = build_parser().parse_args()
    spec, prompts, batches, fixtures = inputs_from_args(args)
    runner = ShardRunner(spec, prompts=prompts, batches=batches, fixtures=fixtures)
    try:
        result = runner.run_stress(job_timeout=3600.0)
    finally:
        runner.close()
    print(json.dumps(asdict(result), sort_keys=True))

if __name__ == "__main__":
    main()
```

Run with the input arguments above and separate diagnostic paths:

```bash
mkdir -p "$RUN/stress"
PYTHONPATH="$PWD/test/manual${PYTHONPATH:+:$PYTHONPATH}" \
  srun --jobid "$SLURM_JOB_ID" --ntasks=1 \
  python "$RUN/stress_driver.py" "${ARGS[@]}" \
  --bundle-output "$RUN/stress/unused-bundle.json" \
  --completion-output "$RUN/stress/unused-complete.json" \
  >"$RUN/stress/stdout.log" 2>"$RUN/stress/stderr.log"
```

This runs 100 cycles/1,000 requests, distributed in-place updates,
every-tenth-cycle cancellation and exact final cleanup. The driver exits 0 only
after completion/teardown, and nonzero on exceptions. It publishes no RunBundle;
stress diagnostics are supporting evidence kept outside qualification inventory.

## Aggregate and blocked cases

```bash
python "$HARNESS/aggregate.py" --manifest "$RUN/qualification.json" \
  --output "$RUN/report.json"
```

Manifest/report outputs must be outside the artifact root and are exclusive.
Manifest creation exits 0 on success, 1 on validation failure. Aggregation exits
1 for failed, blocked, missing or invalid evidence; 0 means comparisons passed.
A subset can exit 0 with `qualification_passed: false`; only the full manifest
can qualify.

Valid blocked reasons are `required_h200_unavailable`, or `reference_unavailable`
for source shards only. A `blocked.json` has exactly `status` (`blocked`),
`reason`, nonempty evidence `detail`, `qualification_hash` (manifest hash),
`shard` (exact manifest shard object), and `record_hash` (`canonical_sha256` of
the other five fields). It cannot coexist with a bundle or completion marker in
its shard; diagnostic logs may remain alongside it.
Missing data without that record fails. Other runtime errors are not relabeled
as blocked. Blocked active cells make aggregation nonzero; deferred B200 cells
are not blocked records.

Tasks 12–13 use the existing Slurm worktree and durable run store. Validate the
real H200 execution path first, then freeze the clean final candidate after all
remaining review fixes and dispatch all shards. Diagnostic smoke cannot satisfy
Task 12's separate bundle-validation requirement: that requires real
`--selection full` source/candidate shards. GPU/TP/NCCL/stress execution remains
unverified here. Preserve remote jobs, sessions, logs and artifacts unless
cleanup is separately authorized.
