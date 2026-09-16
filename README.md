# exex — Expert Exchange

> Train, merge, prune and exchange MoE experts — built around [Gemma 4 MoE](https://ai.google.dev/gemma/docs/core/model_card_4), variant-agnostic by design.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

## Overview

**exex** (expert exchange) is a toolkit for surgery on Mixture-of-Experts models with fused expert tensors (the Gemma 4 layout). It lets you:

- 🏋️ **Train** individual experts in place — full-rank, memory-efficient, no LoRA required
- 📦 **Extract** experts into portable single-file *cartridges*
- 🔀 **Merge** cartridge experts into any compatible checkpoint — replace, blend, or grow new slots
- ✂️ **Prune** low-value experts to slim models for deployment
- 🔍 **Analyze** routing behavior: expert–domain specialization, co-activation, cross-layer pipelines

All architecture parameters (expert count, active experts per token, MoE dimensions, which layers carry MoE blocks) are derived from the model config at runtime — nothing assumes the 26B A4B's 128 experts / top-8 specifically, and expert counts can be dialed up and down freely.

## How it works

### Training: view-based surgery, not adapters

`prepare_expert_for_training` freezes the whole model, then creates trainable `nn.Parameter` **views** into the fused 3D expert tensors — zero weight duplication — and patches the expert forward so only target experts route through trainable parameters. A single expert's matrices are small, so full-rank training of them is cheap; LoRA is unnecessary at this granularity.

`ExpertTrainer` co-trains the selected experts **and their router rows**, regularized by KL divergence against a snapshot of the pretrained router to prevent routing collapse. Router inputs are captured by forward hooks, so the KL term uses exactly what the router saw. Only the target experts' router rows and per-expert scales train by default; the shared router scale and every other row stay frozen, so a trained expert is *exactly* its cartridge (`--train_full_router` opts out and makes cartridges approximate).

### Cartridges: the exchange format

A cartridge is one safetensors file holding one or more experts (per-layer `gate_up` / `down` weights, router row, per-expert scale) plus a metadata header: manifest with names and labels, source model, format version, and an architecture fingerprint so dimension-incompatible transplants fail loudly. Expert count deliberately isn't part of the fingerprint — cartridges move freely between models with different numbers of experts.

## Installation

```bash
git clone https://github.com/marksverdhei/exex.git
cd exex
pip install -e .            # core
pip install -e .[analysis]  # + plots for the routing analyzer
```

Installing also puts the CLIs on your `PATH` as console scripts: `exex-train`,
`exex-eval`, `exex-router-stats`, `exex-extract`, `exex-merge`, `exex-prune`,
`exex-manage`, `exex-analyze`, `exex-report`. The `python scripts/*.py` forms
below keep working (they are thin shims over `exex.cli`).

## Quickstart

### Train an expert

```bash
python scripts/train_expert.py \
  --model_path google/gemma-4-26B-A4B \
  --dataset your/domain-dataset \
  --expert_indices 42 --label medical \
  --max_steps 500 --batch_size 2 --grad_accum 4 \
  --eval_dataset heldout.jsonl --eval_every 100 \
  --output_dir ./runs/medical
```

The run directory gets `cartridge.safetensors` (the trained expert + its router row, ~340 MB at 26B), `metrics.jsonl` (loss, KL, lr, grad norm, periodic held-out perplexity) and `run.json`. The full model is only written with `--save_full_model`. Use `--clone_from N` instead of `--expert_indices` to grow a fresh slot cloned from expert N and train that (expert extension). Other knobs: `--warmup_steps`, `--lr_decay {linear,cosine}`, `--max_grad_norm`, `--seed`, `--kl_weight`.

### Evaluate base + cartridge without merging to disk

```bash
python scripts/eval_perplexity.py --model_path google/gemma-4-26B-A4B \
  --cartridge ./runs/medical/cartridge.safetensors --dataset heldout.jsonl
python scripts/router_stats.py --model_path google/gemma-4-26B-A4B \
  --cartridge ./runs/medical/cartridge.safetensors:medical:new --dataset heldout.jsonl
```

`--cartridge path[:expert[:target_index|new]]` installs in memory before the forward passes; the target defaults to the expert's original slot, `new` grows one.

### Extract experts into a cartridge

```bash
python scripts/extract_experts.py \
  --model_path ./checkpoints/domain_expert \
  --experts '{"medical": 42}' \
  --labels '{"medical": ["medicine"]}' \
  --output medical.safetensors
```

### Merge a cartridge into another checkpoint

```bash
python scripts/merge_experts.py \
  --model_path google/gemma-4-26B-A4B \
  --cartridge medical.safetensors \
  --expert medical \
  --output_dir ./checkpoints/merged
```

Omit `--target_index` to grow a new slot; set `--alpha 0.5` to blend with the incumbent instead of replacing it. Batch installs via `--merge_config merges.json`:

```json
[
  {"cartridge": "medical.safetensors", "expert": "medical", "target_index": 42},
  {"cartridge": "legal.safetensors",   "expert": "legal"}
]
```

### Prune experts

```bash
python scripts/prune_experts.py \
  --model_path ./checkpoints/merged \
  --strategy reap \
  --calibration_dataset your/calibration-data \
  --num_prune 16 \
  --output_dir ./checkpoints/pruned
```

Strategies: `utilisation` (routing frequency), `magnitude` (weight norm), `reap` (Router-weighted Expert Activation Pruning, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999): mean over routed calibration tokens of router weight × L2 norm of the expert's output), `gate_weight_norm` (the cheaper router gate mass × weight norm proxy that was previously labelled `reap`). `--mode zero` zeroes weights in place for sparse runtimes instead of shrinking the model.

### Analyze routing

```bash
python scripts/run_analysis.py --model_path ... --dataset_path your/multidomain-data
python scripts/generate_report.py --model_path ... --output_dir report/   # or --model_path mock
```

Produces expert–domain activation maps, co-occurrence heatmaps, cross-layer expert pipelines, and suggested expert labels.

## Repository layout

```
src/exex/
├── arch.py       # Config-driven architecture descriptor (the variant-agnostic core)
├── surgery.py    # Trainable views into fused expert tensors
├── trainer.py    # KL-regularized expert + router co-training
├── manager.py    # Clone / remove / label experts (router resizing included)
├── cartridge.py  # Expert cartridge format v0
├── merger.py     # Install cartridge experts: replace, blend, grow
├── pruner.py     # Calibration stats, scoring, remove/zero pruning
├── analyzer.py   # Router behavior analysis
└── cli/          # One module per CLI, each exposing main(argv=None) (exex-* entry points)
scripts/          # Thin shims over exex.cli for running from a checkout
tests/            # CPU-only suite on a tiny Gemma 4 MoE config
```

## Memory notes

Only the trained expert views and router carry gradients and optimizer state; everything else is frozen. Loading the frozen base in 8-/4-bit (`--load_in_4bit`) is supported for the GPU-poor but never required. Numbers for real hardware land with the first end-to-end benchmark ([#10](https://github.com/marksverdhei/exex/issues/10)).

## Status

**Validated end-to-end on Gemma 4 26B A4B** (single GH200, bf16): the toolkit trains one expert in place, saves it as a 341 MB cartridge that reproduces the checkpoint bit-exactly, merges it into a fresh base, grows a 129th slot, and evaluates — all in one GPU-hour. Peak training VRAM ~58 GB at batch size 1, seq 512, full bf16 (no quantization). Details in [#10](https://github.com/marksverdhei/exex/issues/10).

**What single-expert training actually learns** (batch-2 readout, 2026-09-08, clean training stack — pad-loss #25, NaN/KL #52, right-padding #54):

- *Templated data → format learning.* The original −11 % on templated PubMedQA **replicates** on the fixed stack (5.300 → 4.703), but the same model gains 6.4 % on non-medical SQuAD in the same template and is **flat (+0.1 %) on raw PubMed abstracts**. The template wrapper, not medical knowledge, carries most of the gain; the medical-specific residual within-template is ≈ 2 pp.
- *Raw-text training → real but modest domain learning.* Training on raw PubMed abstracts improves held-out raw medical text by **−1.4 %** PPL; the control trained on raw SQuAD contexts improves its own domain by **−4.6 %** while leaving medical text flat (+0.1 %) — so gains are domain-specific, not generic. Costs exist: the medical arm regressed wikitext-2 by +2.4 %. One expert at a 0.5 M-token budget buys percent-level domain gains, not the headline number.
- *Isolation is solvable — but the gain and the leak were the same thing.* The routing-separation recipe (`exex.separation`: domain-minus-general centroid row init, logit calibration to the top-k boundary, contrastive push-down) gives a fresh slot a ~300:1 Norwegian-vs-English routing split with a **clean** regression panel. Doing so also removed the −2.2 % own-domain gain: in-place expert training's domain gains and its cross-domain regressions are both edits to *shared* capacity. An isolated expert must first pay back an insertion cost (~+8 % own-domain at step 0, from displacing a top-k incumbent) before accumulating value; whether it goes net-positive with more tokens is under test.
- *In-model isolation is unsolved for free; deployment-side isolation works today.* A cloned 129th slot with a noised router row leaked exactly like in-place training (batch-4): the copied row stays a *routing twin* of its source, so out-of-domain tokens flow through the domain-shifted slot. The honest current capability is **cartridges as runtime domain adapters**: install the cartridge for domain traffic (−2.2 % own-domain PPL for the Norwegian expert), serve the untouched base otherwise — zero cross-domain cost *by construction*, and removal is bit-identical (guaranteed by `tests/test_runtime_adapter.py`). Training-side separation (domain-centroid row init + contrastive routing push-down, `exex.separation`) is under evaluation.
- *Routing barely moves on raw text* (trained-expert frequency 0.0491 → 0.0489): the gain lives in the expert's weights, not in routing shifts. Training with router top-k 12 instead of 8 adds nothing (−0.1 % in-domain, worse general).
- *Scaling a single expert saturates* (batch-3, 2026-09-16): 4× tokens moves the own-domain gain only −1.4 → −1.8 %, while general-text regression grows monotonically (+1.7 → +3.5 % wikitext-2) and overtakes the gain by ~step 750–1000. The PPL gains are also behaviorally inert on 0-shot MC benchmarks (all deltas < 1 se); a Norwegian arm showed a weak but prompt-consistent commonsense/comma positive. Because routing stays anchored, a trained expert modifies *shared* capacity — isolation (clone slots, multi-expert budgets) is the open direction, not more tokens.

A candid state-of-the-repo audit lives in [`docs/AUDIT-2026-09-07.md`](docs/AUDIT-2026-09-07.md).

Inference note: the fused grouped-GEMM MoE kernel currently asserts on Hopper for `no_grad` forwards with unaligned per-expert token counts; eval/calibration CLIs default to `--experts_impl eager` ([#21](https://github.com/marksverdhei/exex/pull/21)).

See the [issue tracker](https://github.com/marksverdhei/exex/issues) for the roadmap (backends, deeper memory optimization, expansion training).

## License

[Apache 2.0](LICENSE) — the same license as Gemma 4 itself.
