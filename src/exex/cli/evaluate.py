"""Compute token-level perplexity of a causal LM over a text dataset."""

import argparse
import json

from exex.evaluate import load_texts, perplexity
from exex.loading import load_model


def build_parser():
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
    parser.add_argument("--cartridge", action="append", default=[],
                        help="Install before eval: path[:expert[:target_index|new]]; repeatable")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Override active experts per token at eval (A4B -> AxB knob)")
    parser.add_argument("--output", default=None, help="Write JSON result here")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    model, tokenizer = load_model(args.model_path, dtype=args.dtype,
                                  experts_impl=args.experts_impl, cartridges=args.cartridge,
                                  top_k=args.top_k)

    texts = load_texts(args.dataset, args.text_column, args.max_samples)
    ppl, n_tokens = perplexity(model, tokenizer, texts, max_length=args.max_length)

    result = {
        "model": args.model_path,
        "cartridges": args.cartridge,
        "top_k": args.top_k,
        "dataset": args.dataset,
        "num_texts": len(texts),
        "num_tokens": n_tokens,
        "perplexity": ppl,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    main()
