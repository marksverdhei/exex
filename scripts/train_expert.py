#!/usr/bin/env python3
"""CLI for training domain-specific experts on Gemma4 MoE."""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from exex.trainer import ExpertTrainer
from exex.manager import ExpertManager


def main():
    parser = argparse.ArgumentParser(
        description="Train domain-specific experts on Gemma4 MoE"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        help="HF dataset name or local path")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--expert_indices", type=int, nargs="+", required=True,
                        help="Expert indices to train")
    parser.add_argument("--clone_from", type=int, default=None,
                        help="Clone this expert to a new slot before training")
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--router_lr_scale", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--load_in_4bit", action="store_true")

    args = parser.parse_args()

    # Load model
    load_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Optionally clone expert to new slot
    expert_indices = list(args.expert_indices)
    if args.clone_from is not None:
        manager = ExpertManager.from_model(model)
        new_idx = manager.clone_expert(source_idx=args.clone_from)
        expert_indices = [new_idx]
        print(f"Cloned expert {args.clone_from} -> new slot {new_idx}")

    # Create trainer
    trainer = ExpertTrainer(
        model=model,
        target_expert_indices=expert_indices,
        kl_weight=args.kl_weight,
        lr=args.lr,
        router_lr_scale=args.router_lr_scale,
    )

    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")

    # Training loop
    print(f"Training experts {expert_indices} for {args.max_steps} steps...")
    step = 0
    for epoch in range(100):  # enough epochs to reach max_steps
        for i in range(0, len(dataset), args.batch_size):
            if step >= args.max_steps:
                break

            batch_texts = [
                dataset[j][args.text_column]
                for j in range(i, min(i + args.batch_size, len(dataset)))
            ]
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                padding=True,
            ).to(model.device)

            labels = encodings.input_ids.clone()
            metrics = trainer.train_step(
                input_ids=encodings.input_ids, labels=labels
            )

            step += 1
            if step % args.log_every == 0:
                print(
                    f"Step {step}/{args.max_steps} | "
                    f"task_loss={metrics['task_loss']:.4f} | "
                    f"kl_loss={metrics['kl_loss']:.6f} | "
                    f"total_loss={metrics['total_loss']:.4f}"
                )

        if step >= args.max_steps:
            break

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
