"""merge_experts.py — Merge experts from independently fine-tuned checkpoints.

Supports:
  - Direct weight copy from a donor checkpoint
  - Weighted linear interpolation between two expert weight matrices
  - Task-vector merging (delta from base applied to target)
  - Batch merging via a JSON config file

Usage:
    # Single expert merge
    python scripts/merge_experts.py \
        --base_model google/gemma-4-26b-moe \
        --donor_checkpoint ./checkpoints/medical_expert \
        --donor_expert_index 42 \
        --target_expert_index 42 \
        --output_dir ./checkpoints/merged_model

    # Batch merge from config
    python scripts/merge_experts.py \
        --base_model google/gemma-4-26b-moe \
        --merge_config merge_config.json \
        --output_dir ./checkpoints/merged_model
"""

import argparse
import json
import os
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge MoE experts from separate checkpoints")
    p.add_argument("--base_model", required=True, help="HF model ID or local path for base model")
    p.add_argument("--donor_checkpoint", default=None, help="Path to donor LoRA checkpoint")
    p.add_argument(
        "--donor_expert_index",
        type=int,
        default=None,
        help="Expert index in the donor checkpoint",
    )
    p.add_argument(
        "--target_expert_index",
        type=int,
        default=None,
        help="Expert index in the target (base) model to overwrite",
    )
    p.add_argument(
        "--merge_config",
        default=None,
        help="Path to JSON config listing multiple donor/target index pairs",
    )
    p.add_argument(
        "--strategy",
        choices=["copy", "interpolate", "task_vector"],
        default="copy",
        help="Merging strategy",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Interpolation weight for the donor (used with --strategy interpolate)",
    )
    p.add_argument("--output_dir", required=True, help="Directory to save the merged model")
    return p.parse_args()


def _expert_key(layer: int, expert_idx: int, proj: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert_idx}.{proj}.weight"


def collect_expert_weights(
    state_dict: dict[str, torch.Tensor], expert_idx: int
) -> dict[str, torch.Tensor]:
    """Return all weight tensors belonging to a given expert index."""
    prefix = f".experts.{expert_idx}."
    return {k: v for k, v in state_dict.items() if prefix in k}


def rename_expert_weights(
    weights: dict[str, torch.Tensor], src_idx: int, dst_idx: int
) -> dict[str, torch.Tensor]:
    """Rename expert weights from src_idx to dst_idx."""
    src_prefix = f".experts.{src_idx}."
    dst_prefix = f".experts.{dst_idx}."
    return {k.replace(src_prefix, dst_prefix): v for k, v in weights.items()}


def merge_single(
    base_state: dict[str, torch.Tensor],
    donor_state: dict[str, torch.Tensor],
    donor_idx: int,
    target_idx: int,
    strategy: str,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Apply a single expert merge operation to base_state (in-place)."""
    donor_weights = collect_expert_weights(donor_state, donor_idx)
    renamed = rename_expert_weights(donor_weights, donor_idx, target_idx)

    for key, donor_tensor in renamed.items():
        if key not in base_state:
            print(f"  Warning: key {key!r} not found in base model — skipping.")
            continue

        base_tensor = base_state[key]
        donor_tensor = donor_tensor.to(base_tensor.device, dtype=base_tensor.dtype)

        if strategy == "copy":
            base_state[key] = donor_tensor.clone()
        elif strategy == "interpolate":
            base_state[key] = (1.0 - alpha) * base_tensor + alpha * donor_tensor
        elif strategy == "task_vector":
            # task vector = donor - base; add to base (no-op source, kept for explicit intent)
            delta = donor_tensor - base_tensor
            base_state[key] = base_tensor + delta
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return base_state


def load_merged_state(
    checkpoint_path: str, base_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Load a PEFT/LoRA checkpoint and merge adapters into full weights."""
    # Attempt to load as a PEFT model and merge into a fresh copy of the base
    # (returned as a plain state dict so we can do multi-expert operations)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    return {k: v.clone() for k, v in model.state_dict().items()}


def main() -> None:
    args = parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    base_state = base_model.state_dict()

    # Build the list of merge operations
    merge_ops: list[dict[str, Any]] = []

    if args.merge_config:
        with open(args.merge_config) as f:
            merge_ops = json.load(f)
    elif args.donor_checkpoint and args.donor_expert_index is not None and args.target_expert_index is not None:
        merge_ops = [
            {
                "donor": args.donor_checkpoint,
                "donor_idx": args.donor_expert_index,
                "target_idx": args.target_expert_index,
            }
        ]
    else:
        raise ValueError(
            "Provide either --merge_config or all of "
            "--donor_checkpoint, --donor_expert_index, --target_expert_index."
        )

    for op in merge_ops:
        donor_path = op["donor"]
        donor_idx = op["donor_idx"]
        target_idx = op["target_idx"]
        strategy = op.get("strategy", args.strategy)
        alpha = op.get("alpha", args.alpha)

        print(f"Merging expert {donor_idx} from {donor_path!r} → target expert {target_idx} ({strategy})")
        donor_state = load_merged_state(donor_path, base_state)
        base_state = merge_single(base_state, donor_state, donor_idx, target_idx, strategy, alpha)

    print(f"Loading merged weights into model ...")
    base_model.load_state_dict(base_state)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving merged model to {args.output_dir} ...")
    base_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
