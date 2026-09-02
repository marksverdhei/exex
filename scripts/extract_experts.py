#!/usr/bin/env python3
"""Extract experts from an MoE checkpoint into a cartridge file."""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoModelForCausalLM

from exex.cartridge import save_cartridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--experts", required=True,
                        help='JSON dict {"name": index} or comma-separated indices')
    parser.add_argument("--labels", default=None,
                        help='JSON dict {"name": ["label", ...]}')
    parser.add_argument("--output", required=True, help="Cartridge .safetensors path")
    args = parser.parse_args()

    try:
        experts = json.loads(args.experts)
    except json.JSONDecodeError:
        experts = [int(i) for i in args.experts.split(",")]
    labels = json.loads(args.labels) if args.labels else None

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    cart = save_cartridge(
        model, experts, args.output, source_model=args.model_path, labels=labels
    )
    print(f"Wrote {args.output} with experts: {cart.expert_names}")


if __name__ == "__main__":
    main()
