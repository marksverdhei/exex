"""Prune low-scoring experts from an MoE checkpoint."""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exex.pruner import (
    STRATEGIES,
    collect_router_stats,
    prune_experts,
    score_experts,
    select_prune_candidates,
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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--strategy", default="reap",
                        choices=list(STRATEGIES),
                        help="reap: router-weighted expert activation norm "
                             "(arXiv:2510.13999); gate_weight_norm: gate mass "
                             "x weight norm proxy; utilisation: routing "
                             "frequency; magnitude: weight norm")
    parser.add_argument("--calibration_dataset",
                        help="HF dataset or local JSON file (needed for "
                             "utilisation/reap/gate_weight_norm)")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_prune", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--protect", type=int, nargs="*", default=[],
                        help="Expert indices never pruned")
    parser.add_argument("--mode", default="remove", choices=["remove", "zero"])
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Load dtype; bf16 default matches native Gemma 4 checkpoints")
    parser.add_argument("--output_dir", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=getattr(torch, args.dtype), device_map="cpu",
        # eager: fused grouped-GEMM asserts on unaligned per-expert token
        # counts during no_grad calibration forwards
        experts_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    stats = None
    if args.strategy in ("utilisation", "reap", "gate_weight_norm"):
        if not args.calibration_dataset:
            parser.error(f"--strategy {args.strategy} requires --calibration_dataset")
        print("Collecting router statistics...")
        stats = collect_router_stats(
            model,
            calibration_batches(
                args.calibration_dataset, tokenizer,
                args.text_column, args.max_samples, args.max_length,
            ),
            activation_norms=args.strategy == "reap",
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
    return candidates


if __name__ == "__main__":
    main()
