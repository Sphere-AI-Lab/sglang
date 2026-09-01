# Adapter equivalence harness

Manual GPU harness for Task 8 of the legacy-PEFT retirement: it drives one
SGLang `Engine` through the full adapter lifecycle and records an immutable
result line per transition, so a candidate build can be compared against the
frozen Sphere source.

Nothing here runs in CI. Every entry point needs a GPU allocation.

## Layout

| File | Purpose |
|---|---|
| `run_case.py` | Shard runner. One invocation = one cell x mode x prompt x transition set. |
| `scenarios.py` | Transition definitions shared with the registered unit tests. |
| `server.py` | Engine construction and lifecycle helpers. |
| `fixtures.py` | Deterministic OFT/LoRA adapter fixtures built from a seed. |
| `compare.py` | Bundle comparison and tolerance evaluation. |
| `policy.py`, `schema.py` | `RunBundle` / `CaseKey` identity and comparison semantics (Task 2). |
| `preflight.py` | Fail-closed checkpoint and manifest validation (Task 3). |
| `matrix.json` | The six cells: dense/MoE x BF16/FP8/NVFP4, with tp/ep and GPU class. |
| `prompts.jsonl` | Deterministic prompts and batch definitions. |

## Running one shard

```bash
python run_case.py RUN_DIR MODEL_PATH \
  --mode canonical_oft --prompt factual --transitions full \
  --harness-dir /path/to/test/manual/adapter_equivalence \
  --revision-kind candidate --architecture dense \
  --tp-size 1 --ep-size 1 --base-gpu-id 0 --port 31010 \
  --oft-scale 1e-2 --log-name run.jsonl
```

`RUN_DIR` must already exist and is written to, not read from. Results stream
as one JSON object per line to stdout and to `RUN_DIR/<log-name>`.

Invoke `run_case.py` **directly**. Do not wrap it in a launcher script that
calls `main()` at import time: SGLang's `Engine` starts its scheduler with
multiprocessing's *spawn* start method, which re-imports the main module in the
child, and an unguarded wrapper therefore re-runs the entire shard inside the
scheduler process. `run_case.py` guards its own entry point.

## Transition sets

`--transitions full` runs all 26 lifecycle transitions; `expert` runs the
subset that touches MoE expert bindings; `short` runs
`base.initial -> dynamic.load -> dynamic.infer -> dynamic.unload ->
dynamic.post-unload-base`, which is the smallest set that exercises the
acceptance gate.

Six transitions run as **surrogates** and are flagged `"surrogate": true` in
their output. `stage.v1/v2` and `activate.v1/v2` need an external rank
publishing tensors over an NCCL weight-sync group, which a single-process
offline `Engine` cannot supply; `reject.stale` depends on them; and
`startup.adapter` cannot use an adapter that is also dynamically loaded. A
`fail` on `reject.stale` in an offline run is expected and is not a defect.

## Reading a result

Two fields decide whether a run means anything:

- `differs_from_base` on `dynamic.infer` must be `true`. It proves the adapter
  was genuinely active. If it is `false`, the run cannot distinguish a working
  adapter from no adapter at all, and every other result in that run is
  uninterpretable.
- `differs_from_base` on `dynamic.post-unload-base` must be `false`, i.e. the
  output returns to the exact base tokens. This is the acceptance gate, and it
  is the one comparison the runner actually enforces (via `contract_ok`).

## `--oft-scale`, and why it exists

`build_oft_fixture` generates the skew-symmetric parameters `S` at magnitude
`1e-3`. Canonical OFT is multiplicative — `W' = R·W` with `R = Cayley(S)`,
which for small values is approximately `I + 2S` — so at `1e-3` the rotation
sits within ~0.2% of identity and cannot change which token the model picks.
LoRA's fixture is additive at `1e-2` with alpha/r = 2, which is why LoRA moves
tokens and OFT did not.

Measured on dense BF16, Qwen3-4B-Instruct-2507, greedy decoding:

| `--oft-scale` | `dynamic.infer` moves tokens | post-unload restores base |
|---|---|---|
| `1e-3` (harness default) | no | yes |
| `1e-2` | yes | yes |
| `3e-2` | yes | yes |
| `1e-1` | yes | yes |

`1e-2` is the smallest measured scale that makes the gate able to fail, and it
matches the LoRA fixture scale. Omitting the flag preserves the harness
default, so existing behaviour is unchanged unless the flag is passed.

## Defects fixed in this file

Recorded because each one cost GPU time before `run_case.py` was version
controlled, and all four were invisible to review while it lived only inside
run directories:

1. `ShardRunner.__init__` stored the `Emitter` object where every call site
   expected the bound `.emit` method — `TypeError: 'Emitter' object is not
   callable`, killing each shard ~1s in.
2. A batch record's `requests` field is the *list* of requests, not a count;
   `int()` on it errored every multi-request transition.
3. The OFT target suffixes omitted `k_proj`/`v_proj`. Canonical OFT rotates the
   input side of the fused `qkv_proj` and `normalize_merged_oft_weights()`
   skips a group unless every sibling leaf is present, so the whole q/k/v group
   is required.
4. The OFT fixture scale was not adjustable — see `--oft-scale` above.
