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

`ExpertTrainer` co-trains the selected experts **and the router**, regularized by KL divergence against a snapshot of the pretrained router to prevent routing collapse. Router inputs are captured by forward hooks, so the KL term uses exactly what the router saw.

### Cartridges: the exchange format

A cartridge is one safetensors file holding one or more experts (per-layer `gate_up` / `down` weights, router row, per-expert scale) plus a metadata header: manifest with names and labels, source model, format version, and an architecture fingerprint so dimension-incompatible transplants fail loudly. Expert count deliberately isn't part of the fingerprint — cartridges move freely between models with different numbers of experts.

## Installation

```bash
git clone https://github.com/marksverdhei/exex.git
cd exex
pip install -e .            # core
pip install -e .[analysis]  # + plots for the routing analyzer
```

## Quickstart

### Train an expert

```bash
python scripts/train_expert.py \
  --model_path google/gemma-4-26B-A4B \
  --dataset your/domain-dataset \
  --expert_indices 42 \
  --output_dir ./checkpoints/domain_expert
```

Use `--clone_from N` to grow a fresh slot cloned from expert N and train that instead of overwriting a pretrained expert.

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

Strategies: `utilisation` (routing frequency), `magnitude` (weight norm), `reap` (router gate mass × weight norm, after [arXiv:2510.13999](https://arxiv.org/abs/2510.13999)). `--mode zero` zeroes weights in place for sparse runtimes instead of shrinking the model.

### Analyze routing

```bash
python scripts/run_analysis.py --model_path ... --dataset_path your/multidomain-data
python scripts/generate_report.py --model_path ... --output_dir report/
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
└── analyzer.py   # Router behavior analysis
scripts/          # CLIs for each of the above
tests/            # CPU-only suite on a tiny Gemma 4 MoE config
```

## Memory notes

Only the trained expert views and router carry gradients and optimizer state; everything else is frozen. Loading the frozen base in 8-/4-bit (`--load_in_4bit`) is supported for the GPU-poor but never required. Numbers for real hardware land with the first end-to-end benchmark ([#10](https://github.com/marksverdhei/exex/issues/10)).

## Status

**Validated end-to-end on Gemma 4 26B A4B** (single GH200, bf16): training expert 42 for 500 steps on PubMedQA cut held-out domain perplexity by **11.3%** with general perplexity flat (−0.09% on wikitext-2), stable-to-rising router allocation for the trained expert, and KL ≈ 1e-4 throughout — no routing collapse. Details in [#10](https://github.com/marksverdhei/exex/issues/10). Peak training VRAM ~58 GB at batch size 1, seq 512, full bf16 (no quantization).

Inference note: the fused grouped-GEMM MoE kernel currently asserts on Hopper for `no_grad` forwards with unaligned per-expert token counts; eval/calibration CLIs default to `--experts_impl eager` ([#21](https://github.com/marksverdhei/exex/pull/21)).

See the [issue tracker](https://github.com/marksverdhei/exex/issues) for the roadmap (backends, deeper memory optimization, expansion training).

## License

[Apache 2.0](LICENSE) — the same license as Gemma 4 itself.
