"""train_expert.py — Fine-tune a single domain-specific expert in Gemma 4 26B MoE.

Usage:
    python scripts/train_expert.py \
        --base_model google/gemma-4-26b-moe \
        --dataset path/to/domain_data \
        --expert_index 42 \
        --lora_rank 16 \
        --output_dir ./checkpoints/my_expert
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a single MoE expert")
    p.add_argument("--base_model", required=True, help="HF model ID or local path")
    p.add_argument("--dataset", required=True, help="HF dataset name or local path")
    p.add_argument("--dataset_split", default="train", help="Dataset split to use")
    p.add_argument("--text_column", default="text", help="Column containing training text")
    p.add_argument(
        "--expert_index",
        type=int,
        default=0,
        help="Which expert slot to fine-tune (0-127 for Gemma 4 26B MoE)",
    )
    p.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--load_in_4bit", action="store_true", help="Load base model in NF4")
    p.add_argument("--load_in_8bit", action="store_true", help="Load base model in Int8")
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    p.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    p.add_argument("--max_steps", type=int, default=1000, help="Number of training steps")
    p.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size")
    p.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--output_dir", default="./output", help="Directory to save the adapter")
    p.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps")
    return p.parse_args()


def build_quantization_config(args: argparse.Namespace) -> BitsAndBytesConfig | None:
    if args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if args.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def build_lora_target_modules(expert_index: int) -> list[str]:
    """Return LoRA target module patterns for a specific expert index."""
    # Gemma 4 MoE expert layers follow a naming pattern like:
    #   model.layers.{layer}.mlp.experts.{expert_index}.{gate,up,down}_proj
    return [
        f"experts.{expert_index}.gate_proj",
        f"experts.{expert_index}.up_proj",
        f"experts.{expert_index}.down_proj",
    ]


def main() -> None:
    args = parse_args()

    print(f"Loading tokenizer from {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = build_quantization_config(args)

    print(f"Loading base model from {args.base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if quant_config is None else None,
        device_map="auto",
        trust_remote_code=True,
    )

    target_modules = build_lora_target_modules(args.expert_index)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    print(f"Wrapping model with LoRA (rank={args.lora_rank}, expert={args.expert_index}) ...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading dataset from {args.dataset} ...")
    if os.path.exists(args.dataset):
        dataset = load_dataset("json", data_files=args.dataset, split=args.dataset_split)
    else:
        dataset = load_dataset(args.dataset, split=args.dataset_split)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        dataset_text_field=args.text_column,
    )

    print("Starting training ...")
    trainer.train()

    print(f"Saving adapter to {args.output_dir} ...")
    trainer.save_model(args.output_dir)

    metadata = {
        "base_model": args.base_model,
        "expert_index": args.expert_index,
        "lora_rank": args.lora_rank,
    }
    with open(os.path.join(args.output_dir, "expert_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
