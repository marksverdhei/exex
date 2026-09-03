#!/usr/bin/env python3
"""Prune low-scoring experts from an MoE checkpoint."""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exex.pruner import (
    collect_router_stats,
    score_experts,
    select_prune_candidates,
    prune_experts,
)


def calibration_batches(dataset_name, tokenizer, text_column, max_samples, max_length):
    from datasets import load_dataset
    if os.path.isfile(dataset_name):
        dataset = load_dataset("json", data_files=dataset_name, split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")
    for i in range(min(max_samples, len(dataset))):
        yield tokenizer(
            dataset[i][text_column], return_tensors="pt",
            truncation=True, max_length=max_length,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--strategy", default="reap",
                        choices=["utilisation", "magnitude", "reap"])
    parser.add_argument("--calibration_dataset",
                        help="HF dataset (needed for utilisation/reap)")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_prune", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--protect", type=int, nargs="*", default=[],
                        help="Expert indices never pruned")
    parser.add_argument("--mode", default="remove", choices=["remove", "zero"])
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    stats = None
    if args.strategy in ("utilisation", "reap"):
        if not args.calibration_dataset:
            parser.error(f"--strategy {args.strategy} requires --calibration_dataset")
        print("Collecting router statistics...")
        stats = collect_router_stats(
            model,
            calibration_batches(
                args.calibration_dataset, tokenizer,
                args.text_column, args.max_samples, args.max_length,
            ),
        )

    scores = score_experts(model, args.strategy, stats=stats)
    candidates = select_prune_candidates(
        scores, num_prune=args.num_prune, threshold=args.threshold,
        protected=args.protect,
    )
    print(f"Pruning {len(candidates)} experts ({args.mode}): {candidates}")
    arch = prune_experts(model, candidates, mode=args.mode)
    print(f"Experts remaining: {arch.num_experts}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved pruned model to {args.output_dir}")


if __name__ == "__main__":
    main()
