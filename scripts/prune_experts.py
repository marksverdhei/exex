"""prune_experts.py — Prune low-utilisation or low-magnitude experts from a Gemma 4 MoE model.

Strategies:
  utilisation  — collect router logits over a calibration dataset, drop experts
                 whose mean activation probability is below --threshold.
  magnitude    — drop the N experts with the smallest L2 weight norms.
  zero         — zero out selected expert weights in-place (for sparse runtimes).

Usage:
    python scripts/prune_experts.py \
        --model_path ./checkpoints/merged_model \
        --calibration_dataset path/to/calibration_data \
        --strategy utilisation \
        --threshold 0.01 \
        --output_dir ./checkpoints/pruned_model
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune MoE experts")
    p.add_argument("--model_path", required=True, help="Path to the model checkpoint")
    p.add_argument(
        "--calibration_dataset",
        default=None,
        help="HF dataset name or local path (required for utilisation strategy)",
    )
    p.add_argument("--calibration_split", default="train", help="Dataset split")
    p.add_argument("--calibration_samples", type=int, default=512, help="Number of samples")
    p.add_argument("--text_column", default="text", help="Column with text")
    p.add_argument(
        "--strategy",
        choices=["utilisation", "magnitude", "zero"],
        default="utilisation",
        help="Pruning strategy",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Activation-probability threshold for utilisation pruning",
    )
    p.add_argument(
        "--n_prune",
        type=int,
        default=None,
        help="Number of experts to prune (used with magnitude strategy)",
    )
    p.add_argument("--output_dir", required=True, help="Directory to save pruned model")
    p.add_argument("--max_seq_length", type=int, default=512, help="Max tokens per sample")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Router-hook helpers
# ---------------------------------------------------------------------------

_router_activations: dict[str, list[torch.Tensor]] = defaultdict(list)


def _make_router_hook(name: str):
    def hook(module, inputs, output):
        # output is typically (hidden_states, router_logits) or just router_logits
        logits = output if output.dim() == 2 else output[1]
        probs = torch.softmax(logits.float(), dim=-1)
        _router_activations[name].append(probs.detach().cpu())

    return hook


def register_router_hooks(model) -> list:
    handles = []
    for name, module in model.named_modules():
        if "router" in name.lower() and hasattr(module, "forward"):
            handles.append(module.register_forward_hook(_make_router_hook(name)))
    return handles


def compute_mean_utilisation(num_experts: int) -> dict[int, float]:
    """Average per-expert activation probability across all collected router outputs."""
    if not _router_activations:
        return {}

    all_probs = []
    for tensors in _router_activations.values():
        all_probs.append(torch.cat(tensors, dim=0))  # [N_tokens, num_experts_or_more]

    combined = torch.cat(all_probs, dim=0)  # [total_tokens, experts]
    mean_probs = combined.mean(dim=0)  # [experts]
    return {i: float(mean_probs[i]) for i in range(min(num_experts, mean_probs.shape[0]))}


# ---------------------------------------------------------------------------
# Weight-magnitude helpers
# ---------------------------------------------------------------------------

def compute_expert_magnitudes(
    state_dict: dict[str, torch.Tensor], num_experts: int
) -> dict[int, float]:
    magnitudes: dict[int, float] = {}
    for expert_idx in range(num_experts):
        prefix = f".experts.{expert_idx}."
        tensors = [v for k, v in state_dict.items() if prefix in k]
        if tensors:
            total_norm = sum(t.float().norm().item() ** 2 for t in tensors) ** 0.5
            magnitudes[expert_idx] = total_norm
    return magnitudes


# ---------------------------------------------------------------------------
# Pruning operations
# ---------------------------------------------------------------------------

def zero_expert(state_dict: dict[str, torch.Tensor], expert_idx: int) -> None:
    prefix = f".experts.{expert_idx}."
    for key in state_dict:
        if prefix in key:
            state_dict[key].zero_()


def remove_expert_keys(state_dict: dict[str, torch.Tensor], expert_idx: int) -> dict[str, torch.Tensor]:
    prefix = f".experts.{expert_idx}."
    return {k: v for k, v in state_dict.items() if prefix not in k}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    state_dict = model.state_dict()

    # Detect total expert count from state dict keys
    expert_indices = set()
    for key in state_dict:
        if ".experts." in key:
            parts = key.split(".experts.")
            if len(parts) > 1:
                idx_str = parts[1].split(".")[0]
                if idx_str.isdigit():
                    expert_indices.add(int(idx_str))
    num_experts = max(expert_indices) + 1 if expert_indices else 128
    print(f"Detected {num_experts} experts in model.")

    experts_to_prune: list[int] = []

    if args.strategy == "utilisation":
        if not args.calibration_dataset:
            raise ValueError("--calibration_dataset is required for utilisation pruning.")

        print(f"Loading calibration dataset: {args.calibration_dataset} ...")
        if os.path.exists(args.calibration_dataset):
            dataset = load_dataset("json", data_files=args.calibration_dataset, split=args.calibration_split)
        else:
            dataset = load_dataset(args.calibration_dataset, split=args.calibration_split)

        samples = dataset.select(range(min(args.calibration_samples, len(dataset))))

        print("Registering router hooks ...")
        handles = register_router_hooks(model)

        print("Running calibration forward passes ...")
        with torch.inference_mode():
            for sample in samples:
                text = sample[args.text_column]
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_seq_length,
                ).to(model.device)
                model(**inputs)

        for h in handles:
            h.remove()

        util = compute_mean_utilisation(num_experts)
        print("Expert utilisation (mean activation probability):")
        for idx, prob in sorted(util.items(), key=lambda x: x[1]):
            marker = " ← PRUNE" if prob < args.threshold else ""
            print(f"  Expert {idx:4d}: {prob:.4f}{marker}")

        experts_to_prune = [idx for idx, prob in util.items() if prob < args.threshold]

    elif args.strategy in ("magnitude", "zero"):
        magnitudes = compute_expert_magnitudes(state_dict, num_experts)
        print("Expert weight magnitudes (L2 norm):")
        sorted_by_mag = sorted(magnitudes.items(), key=lambda x: x[1])
        n_prune = args.n_prune or max(1, num_experts // 10)
        for idx, mag in sorted_by_mag:
            marker = " ← PRUNE" if idx in [i for i, _ in sorted_by_mag[:n_prune]] else ""
            print(f"  Expert {idx:4d}: {mag:.4f}{marker}")
        experts_to_prune = [idx for idx, _ in sorted_by_mag[:n_prune]]

    print(f"\nPruning {len(experts_to_prune)} experts: {experts_to_prune}")

    if args.strategy == "zero":
        for idx in experts_to_prune:
            zero_expert(state_dict, idx)
    else:
        for idx in experts_to_prune:
            state_dict = remove_expert_keys(state_dict, idx)

    model.load_state_dict(state_dict, strict=False)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving pruned model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "strategy": args.strategy,
        "experts_pruned": experts_to_prune,
        "threshold": args.threshold if args.strategy == "utilisation" else None,
        "n_pruned": len(experts_to_prune),
    }
    with open(os.path.join(args.output_dir, "pruning_report.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Pruning report saved to {os.path.join(args.output_dir, 'pruning_report.json')}")
    print("Done.")


if __name__ == "__main__":
    main()
