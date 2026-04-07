# exex — Expert Exchange

> Memory-efficient training, merging, and pruning of domain-specific experts for [Gemma 4 26B MoE](https://ai.google.dev/gemma/docs/core/model_card_4).

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Model: Gemma 4 26B MoE](https://img.shields.io/badge/Model-Gemma%204%2026B%20MoE-orange)](https://ai.google.dev/gemma/docs/core/model_card_4)

---

## Overview

**exex** (expert exchange) is a toolkit built around **Gemma 4 26B A4B** — Google's open-weight Mixture-of-Experts model released in 2025 under the Apache 2.0 license. exex lets you:

- 🏋️ **Train** new, domain-specific MoE experts with minimal GPU memory
- 🔀 **Merge** experts from independently fine-tuned checkpoints into a single model
- ✂️ **Prune** redundant or low-utilization experts to slim down the model for deployment

The name is a double meaning: *expert exchange* (swapping experts between models) and the file extension `.ex` (as in executables — things that *run*).

---

## Gemma 4 26B MoE Architecture

| Property | Value |
|---|---|
| Total parameters | ~26 B |
| Active parameters per token | ~3.8 B |
| Total experts | 128 + 1 shared |
| Active experts per token | 8 |
| Context window | 256 K tokens |
| Attention | Hybrid sliding-window + global |
| Modalities | Text + Image |
| License | Apache 2.0 |
| VRAM (fp16, 256K ctx) | ~23 GB |

Each transformer block in Gemma 4 26B contains a standard MLP **plus** a set of MoE blocks whose outputs are summed. A learned router selects 8 of the 128 specialist experts for each token; the shared expert always fires. This means the model has the capability of a 26 B dense model while only performing ~3.8 B parameters of compute per forward pass.

exex targets this expert layer: training, combining, and removing individual experts without touching the rest of the model weights.

---

## Features

### Memory-Efficient Expert Training

- **LoRA on individual experts** — fine-tune a single expert's feed-forward layers with rank-decomposed adapters, leaving all other weights frozen.
- **Gradient checkpointing** — trades compute for memory so you can run on a single 24 GB GPU.
- **8-bit / 4-bit quantized base model** — use [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) to load the frozen base in NF4 or Int8 while training adapters in full precision.
- **Expert-only forward pass** — the training loop can optionally skip non-expert layers to further reduce activation memory.

### Expert Merging

Combine experts trained on different domains into a single checkpoint without any additional training:

- **Direct weight copy** — transplant a trained expert's weights from one checkpoint into the target model at a chosen expert index.
- **Weighted linear interpolation** — blend two expert weight matrices by a configurable α to merge capabilities smoothly.
- **Task-vector merging** — compute the delta from the base model and add it to any target checkpoint (works across different expert indices).

### Expert Pruning

Identify and remove underutilised experts to reduce serving costs:

- **Utilisation-based pruning** — collect router logits over a calibration dataset and drop experts whose average activation probability falls below a threshold.
- **Weight-magnitude pruning** — remove experts whose L2 norm is smallest (often correlated with redundancy).
- **Structured zeroing** — zero out pruned expert weights (for sparse inference runtimes) rather than physically removing them.

---

## Repository Layout

```
exex/
├── scripts/
│   ├── train_expert.py       # Fine-tune a single domain expert
│   ├── merge_experts.py      # Merge experts from different checkpoints
│   └── prune_experts.py      # Prune low-utilisation experts
├── src/
│   └── exex/
│       ├── __init__.py
│       ├── trainer.py        # Expert trainer (LoRA + quant helpers)
│       ├── merger.py         # Expert merging utilities
│       ├── pruner.py         # Expert pruning utilities
│       └── utils.py          # Router analysis, calibration, etc.
├── examples/
│   └── finetune_medical.sh   # End-to-end example: medical expert
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/marksverdhei/exex.git
cd exex
pip install -r requirements.txt
```

> **Requirements:** Python 3.10+, CUDA 12+, a GPU with ≥ 24 GB VRAM (e.g. RTX 3090, RTX 4090, A100).

---

## Quickstart

### 1. Train a domain-specific expert

```bash
python scripts/train_expert.py \
  --base_model google/gemma-4-26b-moe \
  --dataset path/to/domain_data \
  --expert_index 42 \
  --lora_rank 16 \
  --output_dir ./checkpoints/medical_expert
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--base_model` | — | HuggingFace model ID or local path |
| `--dataset` | — | Path or HF dataset name |
| `--expert_index` | `0` | Which expert slot to fine-tune (0–127) |
| `--lora_rank` | `16` | LoRA rank |
| `--load_in_4bit` | `False` | Load base in NF4 (saves ~50 % VRAM) |
| `--gradient_checkpointing` | `True` | Enable gradient checkpointing |
| `--max_steps` | `1000` | Training steps |
| `--output_dir` | `./output` | Where to save the adapter |

### 2. Merge experts into a single checkpoint

Suppose you have trained a medical expert and a legal expert separately and want to combine them:

```bash
python scripts/merge_experts.py \
  --base_model google/gemma-4-26b-moe \
  --donor_checkpoint ./checkpoints/medical_expert \
  --donor_expert_index 42 \
  --target_expert_index 42 \
  --output_dir ./checkpoints/merged_model
```

To merge multiple experts in one pass, pass a JSON config:

```bash
python scripts/merge_experts.py \
  --base_model google/gemma-4-26b-moe \
  --merge_config merge_config.json \
  --output_dir ./checkpoints/merged_model
```

Example `merge_config.json`:

```json
[
  {"donor": "./checkpoints/medical_expert", "donor_idx": 42, "target_idx": 42},
  {"donor": "./checkpoints/legal_expert",   "donor_idx": 7,  "target_idx": 7}
]
```

### 3. Prune low-utilisation experts

```bash
python scripts/prune_experts.py \
  --model_path ./checkpoints/merged_model \
  --calibration_dataset path/to/calibration_data \
  --strategy utilisation \
  --threshold 0.01 \
  --output_dir ./checkpoints/pruned_model
```

| Strategy | Description |
|---|---|
| `utilisation` | Drop experts activated < `--threshold` fraction of tokens |
| `magnitude` | Drop experts with the smallest L2 weight norm |
| `zero` | Zero weights in-place instead of removing (for sparse runtimes) |

---

## Examples

### Fine-tune a medical expert on PubMedQA

```bash
bash examples/finetune_medical.sh
```

This script:
1. Downloads `qiaojin/PubMedQA` from the HuggingFace Hub
2. Loads Gemma 4 26B MoE in 4-bit with LoRA targeting expert 42
3. Trains for 500 steps on a single 24 GB GPU
4. Saves the LoRA adapter to `./checkpoints/medical_expert`

---

## Memory Footprint

| Configuration | VRAM |
|---|---|
| Base model, fp16 | ~52 GB |
| Base model, Int8 | ~26 GB |
| Base model, NF4 (4-bit) | ~13 GB |
| NF4 base + LoRA (rank 16) training | ~18–22 GB |

With NF4 quantisation and gradient checkpointing you can train a new expert on most 24 GB consumer GPUs.

---

## Contributing

Contributions are welcome! Please open an issue or pull request for:

- New merging / pruning strategies
- Support for other MoE architectures
- Efficiency improvements and benchmarks

---

## License

[Apache 2.0](LICENSE) — the same license as Gemma 4 itself.
