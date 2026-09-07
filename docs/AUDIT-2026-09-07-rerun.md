# exex repo audit — re-run after the remediation pass (2026-09-07, evening)

Baseline: [`AUDIT-2026-09-07.md`](AUDIT-2026-09-07.md) (main @ c5d906f, overall **5/10**).
This re-run: main @ 26c366d+ (16 PRs merged today: #23 #38 #39 #40 #41 #42 #43 #44 #45 #46 #47 #48 #49 #50 + docs).
Same rubric, same reviewer, same harshness.

## Feature matrix delta

| Feature | Before | Now | Evidence |
|---|---|---|---|
| Training CLI | partial | **done** | pad mask (#25), `--clone_from` exclusive (#30), accumulation / warmup / decay / clipping / seed / shuffle / periodic eval / metrics.jsonl / run.json (#31); 24 CLI tests |
| Trainer router handling | caveated | **done** | only target rows + scales train by default; `--train_full_router` escape hatch; extract→merge is tensor-exact; two-expert composability test (#26) |
| Cartridge as the artefact | implicit | **done** | cartridge-first saving; `--cartridge` in-memory install for evals; verified bit-exact at 26B (341 MB) |
| Pruning "reap" | mislabelled | **done** | real router-gate × activation-norm saliency via experts hook; old proxy renamed `gate_weight_norm`; 7 tests incl. manual recomputation |
| Routing analyzer | broken | **done** | hook-based, vectorised, 9 tests; `generate_report.py` real + mock paths share it |
| Packaging | missing | **done** | `exex.cli.*`, 9 console scripts, dynamic `__version__`, `datasets` required, wheel built and inspected |
| Active-param knob | — | **new** | `set_top_k` / `--top_k` eval override |
| Clone memory + divergence | — | **new** | single-allocation grow + cache release; seeded `router_noise` |
| LICENSE | missing | **done** | Apache-2.0 text |
| CI | disabled | **dropped by owner** | token lacks `workflow` scope; Markus: "screw ci now" |
| Backends (#6) | stub | stub, honestly re-scoped | tracking issue only |
| Memory work (#5) | planned | re-scoped | measured: opt state ~80 MB, activations are the cost; checkpointing is the one real item |
| Variant-agnostic trainer (#32) | leak | re-scoped | router ops now deliberately Gemma-4-specific; `RouterOps` protocol proposed |

## Debug sweep delta

* Tests: 54 → **110**, all green on CPU (~70 s). New coverage: padding invariance, cartridge exactness + composability, accumulation/schedule/clipping, top-k override, clone noise, REAP saliency, analyzer, and end-to-end CLI smoke for every entry point.
* Static: ruff clean on `src`, `scripts`, `tests` with the project config (E,F,W,B,UP,SIM); 0 findings.
* Bugs confirmed in the morning, status now: pad-token loss — fixed; lossy cartridge — fixed by design; analyzer — fixed; fused-kernel assert — minimal repro on GH200, `batched_mm` and `eager` both pass; eager default kept, timing job 2176230 quantifies the tax.
* Science guard: every pre-fix number is flagged in cluster STATE.md; README and #10 corrected (the −11 % was format learning); fixed-stack rerun and raw-text arms are running.

## Quality rating (1–10, same rubric)

| Axis | Before | Now | One line |
|---|---|---|---|
| Correctness | 6 | **8** | Every confirmed bug fixed with a regression test; cartridge exactness is now a tested invariant; the remaining risk is the Gemma-4-specific router re-implementation (#32) and an untested 4-bit path. |
| Test coverage | 5 | **8** | 110 tests spanning units, invariants and all CLIs on a tiny fixture; missing: GPU/26B tests (manual repro only) and a coverage report. |
| API design | 6 | **7** | Clean `exex.cli`, lazy cartridge install, explicit top-k knob; trainer still mutates `model.load_state_dict` and reaches into router internals. |
| Docs | 5 | **7** | README truthful (including what the first experiment did *not* show), audit docs in-tree, issues carry rationale; no API reference, no hparam guide. |
| Reproducibility | 4 | **7** | `--seed`, metrics.jsonl, run.json with args, cartridge artefacts, STATE.md with job IDs; no lockfile, no CI (owner's call). |
| Packaging | 3 | **7** | pip-installable with 9 entry points, LICENSE, dynamic version, correct deps; not on PyPI, no lockfile. |
| **Overall** | **5** | **7.5** | A tool a second person can install, run and trust the numbers of; not yet a library with a stable, variant-agnostic API. |

## What would move it to 9

1. `RouterOps` adapter (#32) and a second MoE family exercised in tests.
2. Gradient checkpointing + measured VRAM floor (#5).
3. `uv.lock`, coverage in the test run, and CI once the token has the scope (owner deferred).
4. A raw-text domain result that survives its controls — the science, not the code, is now the bottleneck.
