"""
Model surgery for view-based expert training on Gemma4 MoE.

Creates trainable nn.Parameter views into the fused 3D expert tensors,
sharing memory with the original (no duplication). Patches the expert
forward to route target experts through trainable views.
"""

import torch
import torch.nn as nn

from exex.arch import iter_moe_layers


def prepare_expert_for_training(model, target_expert_indices):
    """
    Freeze all model parameters, then create trainable view parameters
    for the specified expert indices. Patches expert forward methods.

    Args:
        model: MoE causal LM instance (Gemma 4 fused-tensor layout)
        target_expert_indices: list of int, which expert slots to make trainable
    """
    if isinstance(target_expert_indices, int):
        target_expert_indices = [target_expert_indices]

    # Freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # For each MoE layer, create trainable views and patch forward
    for _, layer in iter_moe_layers(model):
        experts = layer.experts

        for idx in target_expert_indices:
            if not 0 <= idx < experts.num_experts:
                raise IndexError(
                    f"expert index {idx} out of range (num_experts={experts.num_experts})"
                )

        for idx in target_expert_indices:
            # View into fused tensor — shares memory, no copy
            gate_up_view = nn.Parameter(experts.gate_up_proj.data[idx])
            down_view = nn.Parameter(experts.down_proj.data[idx])

            setattr(experts, f"_train_gate_up_{idx}", gate_up_view)
            setattr(experts, f"_train_down_{idx}", down_view)

        experts._train_indices = set(target_expert_indices)
        experts.forward = _make_patched_forward(experts, target_expert_indices)


def prepare_router_rows_for_training(model, target_expert_indices):
    """Make only the target experts' router rows trainable.

    For every MoE layer, creates trainable view parameters into
    ``router.proj.weight[idx]`` and ``router.per_expert_scale[idx]`` (shared
    storage, no copy) and patches the router forward to compose the frozen
    weight with those views. ``router.scale`` and every other row stay frozen,
    so the trained delta is exactly what a cartridge carries (exex#26).
    """
    if isinstance(target_expert_indices, int):
        target_expert_indices = [target_expert_indices]
    for _, layer in iter_moe_layers(model):
        router = layer.router
        for param in router.parameters():
            param.requires_grad_(False)
        for idx in target_expert_indices:
            setattr(router, f"_router_row_{idx}",
                    nn.Parameter(router.proj.weight.data[idx]))
            setattr(router, f"_router_scale_{idx}",
                    nn.Parameter(router.per_expert_scale.data[idx]))
        router._train_indices = set(target_expert_indices)
        router.forward = _make_patched_router_forward(router, target_expert_indices)


def composed_router_params(router):
    """Return (proj_weight, per_expert_scale) with trainable rows spliced in.

    Falls back to the raw parameters when the router is not row-patched, so
    callers (forward, KL) can use it unconditionally.
    """
    indices = sorted(getattr(router, "_train_indices", ()))
    weight = router.proj.weight
    scale = router.per_expert_scale
    if not indices:
        return weight, scale
    idx = torch.tensor(indices, device=weight.device)
    rows = torch.stack([getattr(router, f"_router_row_{i}") for i in indices])
    scales = torch.stack([getattr(router, f"_router_scale_{i}") for i in indices])
    return weight.index_copy(0, idx, rows), scale.index_copy(0, idx, scales)


def _make_patched_router_forward(router, target_indices):
    """Router forward (Gemma 4 layout) using the composed trainable rows."""
    top_k = router.config.top_k_experts

    def patched_forward(hidden_states):
        weight, per_expert_scale = composed_router_params(router)
        hidden_states = router.norm(hidden_states)
        hidden_states = hidden_states * router.scale * router.scalar_root_size
        expert_scores = nn.functional.linear(hidden_states, weight)
        router_probabilities = nn.functional.softmax(
            expert_scores, dim=-1, dtype=torch.float32
        )
        top_k_weights, top_k_index = torch.topk(router_probabilities, k=top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * per_expert_scale[top_k_index]
        return router_probabilities, top_k_weights, top_k_index

    return patched_forward


def finalize_expert_training(model):
    """
    Remove trainable view parameters and restore the original expert forward.

    The views share storage with the fused expert tensors, so all trained
    updates are already present in the base weights; after finalizing, the
    model has no aliased parameters and saves cleanly with safetensors.
    """
    for _, layer in iter_moe_layers(model):
        experts = layer.experts
        if not hasattr(experts, "_train_indices"):
            continue
        for idx in experts._train_indices:
            for name in (f"_train_gate_up_{idx}", f"_train_down_{idx}"):
                if hasattr(experts, name):
                    delattr(experts, name)
        del experts._train_indices
        if "forward" in experts.__dict__:
            del experts.forward  # fall back to the class forward
    for _, layer in iter_moe_layers(model):
        router = layer.router
        if not hasattr(router, "_train_indices"):
            continue
        for idx in router._train_indices:
            for name in (f"_router_row_{idx}", f"_router_scale_{idx}"):
                if hasattr(router, name):
                    delattr(router, name)
        del router._train_indices
        if "forward" in router.__dict__:
            del router.forward


def _make_patched_forward(experts_module, target_indices):
    """
    Create a patched forward that routes target experts through
    trainable view parameters, all others through the frozen fused tensor.
    """
    target_set = set(target_indices)
    frozen_gate_up = experts_module.gate_up_proj
    frozen_down = experts_module.down_proj
    act_fn = experts_module.act_fn
    num_experts = experts_module.num_experts

    # Collect references to trainable params
    train_params = {}
    for idx in target_indices:
        train_params[idx] = (
            getattr(experts_module, f"_train_gate_up_{idx}"),
            getattr(experts_module, f"_train_down_{idx}"),
        )

    def patched_forward(hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = nn.functional.one_hot(
                top_k_index, num_classes=num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx_t in expert_hit:
            expert_idx = expert_idx_t[0].item()
            if expert_idx >= num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if len(token_idx) == 0:
                continue

            current_state = hidden_states[token_idx]

            if expert_idx in target_set:
                gate_up_w, down_w = train_params[expert_idx]
            else:
                gate_up_w = frozen_gate_up[expert_idx]
                down_w = frozen_down[expert_idx]

            gate, up = nn.functional.linear(current_state, gate_up_w).chunk(2, dim=-1)
            current_hidden = act_fn(gate) * up
            current_hidden = nn.functional.linear(current_hidden, down_w)
            current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]

            final_hidden_states.index_add_(
                0, token_idx, current_hidden.to(final_hidden_states.dtype)
            )

        return final_hidden_states

    return patched_forward
