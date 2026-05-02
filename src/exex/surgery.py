"""
Model surgery for view-based expert training on Gemma4 MoE.

Creates trainable nn.Parameter views into the fused 3D expert tensors,
sharing memory with the original (no duplication). Patches the expert
forward to route target experts through trainable views.
"""

import torch
import torch.nn as nn


def prepare_expert_for_training(model, target_expert_indices):
    """
    Freeze all model parameters, then create trainable view parameters
    for the specified expert indices. Patches expert forward methods.

    Args:
        model: Gemma4ForCausalLM model instance
        target_expert_indices: list of int, which expert slots to make trainable
    """
    if isinstance(target_expert_indices, int):
        target_expert_indices = [target_expert_indices]

    # Freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # For each MoE layer, create trainable views and patch forward
    for layer in model.model.layers:
        if not hasattr(layer, "experts"):
            continue

        experts = layer.experts

        for idx in target_expert_indices:
            # View into fused tensor — shares memory, no copy
            gate_up_view = nn.Parameter(experts.gate_up_proj.data[idx])
            down_view = nn.Parameter(experts.down_proj.data[idx])

            setattr(experts, f"_train_gate_up_{idx}", gate_up_view)
            setattr(experts, f"_train_down_{idx}", down_view)

        experts._train_indices = set(target_expert_indices)
        experts.forward = _make_patched_forward(experts, target_expert_indices)


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
