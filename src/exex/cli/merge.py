"""Install cartridge experts into a target MoE checkpoint."""

import argparse
import contextlib
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exex.merger import apply_merge_config, install_expert


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cartridge", help="Cartridge file (single-install mode)")
    parser.add_argument("--expert", help="Expert name in the cartridge")
    parser.add_argument("--target_index", type=int, default=None,
                        help="Slot to overwrite; omit to grow a new slot")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="1.0 = replace, 0.5 = average with incumbent")
    parser.add_argument("--merge_config", help="JSON file with a list of installs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Load dtype; bf16 default matches native Gemma 4 "
                             "checkpoints (fp16 casts lose precision / overflow)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if bool(args.merge_config) == bool(args.cartridge):
        parser.error("Provide either --merge_config or --cartridge + --expert")

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=getattr(torch, args.dtype), device_map="cpu"
    )

    if args.merge_config:
        with open(args.merge_config) as f:
            landed = apply_merge_config(model, json.load(f))
    else:
        landed = [install_expert(
            model, args.cartridge, args.expert,
            target_index=args.target_index, alpha=args.alpha,
        )]
    print(f"Installed experts at indices: {landed}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    with contextlib.suppress(Exception):
        AutoTokenizer.from_pretrained(args.model_path).save_pretrained(args.output_dir)
    print(f"Saved merged model to {args.output_dir}")
    return landed


if __name__ == "__main__":
    main()
