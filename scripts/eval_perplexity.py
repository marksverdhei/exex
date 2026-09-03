#!/usr/bin/env python3
"""Compute token-level perplexity of a causal LM over a text dataset."""

import argparse
import json
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def perplexity(model, tokenizer, texts, max_length=512):
    """Mean per-token NLL over texts -> (ppl, total_tokens)."""
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length).to(model.device)
        if enc.input_ids.shape[1] < 2:
            continue
        out = model(input_ids=enc.input_ids, labels=enc.input_ids)
        n_tokens = enc.input_ids.shape[1] - 1
        total_nll += out.loss.float().item() * n_tokens
        total_tokens += n_tokens
    return math.exp(total_nll / max(total_tokens, 1)), total_tokens


def load_texts(dataset, text_column, max_samples):
    from datasets import load_dataset
    if os.path.isfile(dataset):
        ds = load_dataset("json", data_files=dataset, split="train")
    else:
        name, _, split = dataset.partition("@")
        ds = load_dataset(name, split=split or "test")
    n = min(max_samples, len(ds))
    return [ds[i][text_column] for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True,
                        help="Local json(l) file, or HF name (optionally name@split)")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--experts_impl", default="eager",
                        help="MoE experts implementation; eager avoids fused "
                             "grouped-GEMM kernels that assert on unaligned "
                             "per-expert token counts under no_grad")
    parser.add_argument("--output", default=None, help="Write JSON result here")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=getattr(torch, args.dtype), device_map="auto",
        experts_implementation=args.experts_impl,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    texts = load_texts(args.dataset, args.text_column, args.max_samples)
    ppl, n_tokens = perplexity(model, tokenizer, texts, max_length=args.max_length)

    result = {
        "model": args.model_path,
        "dataset": args.dataset,
        "num_texts": len(texts),
        "num_tokens": n_tokens,
        "perplexity": ppl,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
