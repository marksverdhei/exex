#!/usr/bin/env python3
"""Dump per-expert routing statistics (selection frequency, gate mass) as JSON."""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exex.arch import MoEArch
from exex.pruner import collect_router_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True, help="Local json(l) file or HF name")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output", default=None, help="Write JSON result here")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=getattr(torch, args.dtype), device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    from datasets import load_dataset
    if os.path.isfile(args.dataset):
        ds = load_dataset("json", data_files=args.dataset, split="train")
    else:
        ds = load_dataset(args.dataset, split="train")

    def batches():
        for i in range(min(args.max_samples, len(ds))):
            yield tokenizer(ds[i][args.text_column], return_tensors="pt",
                            truncation=True, max_length=args.max_length)

    stats = collect_router_stats(model, batches())
    arch = MoEArch.from_model(model)

    freq = stats.selection_freq            # [L, E]
    gate = stats.gate_mass / max(stats.tokens, 1)
    result = {
        "model": args.model_path,
        "dataset": args.dataset,
        "tokens": stats.tokens,
        "num_experts": arch.num_experts,
        "selection_freq_mean": freq.mean(dim=0).tolist(),
        "gate_mass_mean": gate.mean(dim=0).tolist(),
        "selection_freq_per_layer": freq.tolist(),
    }
    print(json.dumps({k: v for k, v in result.items()
                      if k != "selection_freq_per_layer"}, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
