"""
Expert merging: install cartridge experts into a target model.

Supports transplanting into an existing slot (with optional alpha blending
against the incumbent weights) or growing a fresh slot via ExpertManager.
"""

import torch

from exex.arch import MoEArch, iter_moe_layers
from exex.cartridge import Cartridge, load_cartridge
from exex.manager import ExpertManager


def install_expert(model, cartridge, expert_name, target_index=None, alpha=1.0,
                   check_fingerprint=True):
    """Install one cartridge expert into a model.

    Args:
        model: target MoE causal LM
        cartridge: Cartridge instance or path to a cartridge file
        expert_name: which expert in the cartridge to install
        target_index: slot to overwrite; None grows a new slot at the end
        alpha: blend factor — 1.0 replaces the target entirely,
               0.5 averages donor and incumbent, etc. Applied to expert
               weights and the router row alike.
        check_fingerprint: refuse dimension-incompatible transplants loudly

    Returns:
        the expert index the donor landed in
    """
    if not isinstance(cartridge, Cartridge):
        cartridge = load_cartridge(cartridge)
    if expert_name not in cartridge.manifest:
        raise KeyError(
            f"Expert {expert_name!r} not in cartridge "
            f"(has: {cartridge.expert_names})"
        )

    arch = MoEArch.from_model(model)
    if check_fingerprint and cartridge.fingerprint != arch.fingerprint():
        raise ValueError(
            "Cartridge/model architecture mismatch: "
            f"cartridge fingerprint {cartridge.fingerprint} != "
            f"model fingerprint {arch.fingerprint()}. "
            "Pass check_fingerprint=False to override."
        )

    if target_index is None:
        # Grow a new slot (cloned from expert 0, immediately overwritten)
        manager = ExpertManager.from_model(model)
        target_index = manager.clone_expert(0)
        arch = manager.arch
    elif not 0 <= target_index < arch.num_experts:
        raise IndexError(
            f"target_index {target_index} out of range "
            f"(num_experts={arch.num_experts})"
        )

    for layer_idx, layer in iter_moe_layers(model):
        gate_up, down, router_row, router_scale = cartridge.expert_tensors(
            expert_name, layer_idx
        )
        _blend_(layer.experts.gate_up_proj.data[target_index], gate_up, alpha)
        _blend_(layer.experts.down_proj.data[target_index], down, alpha)
        _blend_(layer.router.proj.weight.data[target_index], router_row, alpha)
        _blend_(layer.router.per_expert_scale.data[target_index], router_scale, alpha)

    labels = cartridge.manifest[expert_name].get("labels") or []
    if labels:
        # expert_labels values are always plain strings
        manager = ExpertManager.from_model(model)
        manager.label_expert(target_index, ",".join(labels))

    return target_index


def _blend_(dst, src, alpha):
    src = src.to(device=dst.device, dtype=dst.dtype)
    if alpha >= 1.0:
        dst.copy_(src)
    else:
        dst.mul_(1.0 - alpha).add_(src, alpha=alpha)


def apply_merge_config(model, merge_config):
    """Apply a batch of installs described by a JSON-style list.

    Each entry: {"cartridge": path, "expert": name,
                 "target_index": int | null, "alpha": float}

    Returns list of landed expert indices, in order.
    """
    landed = []
    for entry in merge_config:
        landed.append(
            install_expert(
                model,
                entry["cartridge"],
                entry["expert"],
                target_index=entry.get("target_index"),
                alpha=entry.get("alpha", 1.0),
            )
        )
    return landed
