#!/usr/bin/env bash
# examples/finetune_medical.sh
# End-to-end example: fine-tune a medical expert on PubMedQA
#
# Requirements: a GPU with >= 24 GB VRAM, Python env with requirements.txt installed.
# The base model is streamed from HuggingFace; set HF_TOKEN if needed.

set -euo pipefail

BASE_MODEL="google/gemma-4-26b-moe"
DATASET="qiaojin/PubMedQA"
EXPERT_INDEX=42
OUTPUT_DIR="./checkpoints/medical_expert"

echo "=== exex: fine-tuning medical expert ==="
echo "Base model  : $BASE_MODEL"
echo "Dataset     : $DATASET"
echo "Expert index: $EXPERT_INDEX"
echo "Output dir  : $OUTPUT_DIR"
echo ""

python scripts/train_expert.py \
  --base_model "$BASE_MODEL" \
  --dataset "$DATASET" \
  --dataset_split "train" \
  --text_column "long_answer" \
  --expert_index "$EXPERT_INDEX" \
  --lora_rank 16 \
  --lora_alpha 32 \
  --load_in_4bit \
  --gradient_checkpointing \
  --max_seq_length 1024 \
  --max_steps 500 \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --learning_rate 2e-4 \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "=== Training complete. Adapter saved to $OUTPUT_DIR ==="
