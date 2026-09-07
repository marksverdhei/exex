"""
Expert pruning: score experts on a calibration set, then remove or zero them.

Strategies:
    utilisation      — mean routing selection frequency (drop the rarely-picked)
    magnitude        — L2 norm of expert weights (drop the smallest)
    reap             — Router-weighted Expert Activation Pruning
                       (arXiv:2510.13999): mean over the calibration tokens
                       routed to expert e of g_e(x) * ||f_e(x)||_2, where g_e
                       is the (normalised, per-expert-scaled) router weight and
                       f_e(x) the expert's output for that token. Needs
                       ``collect_router_stats(..., activation_norms=True)``.
    gate_weight_norm — router gate mass x expert weight norm (the cheap proxy
                       that used to be called "reap")
"""

import torch

from exex.arch import MoEArch, iter_moe_layers
from exex.manager import ExpertManager

STRATEGIES = ("utilisation", "magnitude", "reap", "gate_weight_norm")
_NEEDS_STATS = ("utilisation", "reap", "gate_weight_norm")


class RouterStats:
    """Per-layer routing statistics accumulated over calibration batches."""

    def __init__(self, num_layers, num_experts):
        self.counts = torch.zeros(num_layers, num_experts)
        self.gate_mass = torch.zeros(num_layers, num_experts)
        # sum over routed tokens of g_e(x) * ||f_e(x)||_2 (REAP saliency);
        # only filled when collected with ``activation_norms=True``
        self.activation_saliency = torch.zeros(num_layers, num_experts)
        self.has_activation_saliency = False
        self.tokens = 0

    @property
    def selection_freq(self):
        """Fraction of tokens that routed to each expert, per layer [L, E]."""
        return self.counts / max(self.tokens, 1)

    @property
    def mean_saliency(self):
        """Mean REAP saliency per routed token, per layer [L, E]."""
        return self.activation_saliency / self.counts.clamp(min=1)


def expert_output_norms(experts, hidden_states, expert_idx):
    """L2 norm of expert ``expert_idx``'s output for each row of
    ``hidden_states`` [T, H] -> [T]. Recomputed from the raw fused tensors so
    it also works when the experts' forward has been patched by surgery."""
    gate_up = experts.gate_up_proj.data[expert_idx]
    down = experts.down_proj.data[expert_idx]
    x = hidden_states.to(gate_up.dtype)
    gate, up = torch.nn.functional.linear(x, gate_up).chunk(2, dim=-1)
    out = torch.nn.functional.linear(experts.act_fn(gate) * up, down)
    return out.float().norm(dim=-1)


def _activation_saliency(experts, hidden_states, top_k_index, top_k_weights):
    """Per-expert sum over routed tokens of g_e(x) * ||f_e(x)||_2 -> [E]."""
    num_experts = experts.gate_up_proj.shape[0]
    saliency = torch.zeros(num_experts)
    for expert_idx in torch.unique(top_k_index).tolist():
        token_idx, top_k_pos = torch.where(top_k_index == expert_idx)
        norms = expert_output_norms(experts, hidden_states[token_idx], expert_idx)
        gate = top_k_weights[token_idx, top_k_pos].float()
        saliency[expert_idx] = (gate * norms).sum().cpu()
    return saliency


@torch.no_grad()
def collect_router_stats(model, batches, activation_norms=False):
    """Run calibration batches and capture routing decisions via hooks.

    Args:
        batches: iterable of dicts with at least ``input_ids``
        activation_norms: also accumulate the REAP activation saliency
            g_e(x) * ||f_e(x)||_2 per expert (needs an extra expert forward
            per hit expert; required by the ``"reap"`` strategy)
    """
    arch = MoEArch.from_model(model)
    moe_layers = list(iter_moe_layers(model))
    stats = RouterStats(len(moe_layers), arch.num_experts)
    stats.has_activation_saliency = activation_norms

    handles = []
    for pos, (_, layer) in enumerate(moe_layers):
        def hook(module, args, output, _pos=pos):
            _, top_k_weights, top_k_index = output
            flat_idx = top_k_index.reshape(-1)
            flat_w = top_k_weights.reshape(-1).float().cpu()
            stats.counts[_pos].index_add_(
                0, flat_idx.cpu(), torch.ones_like(flat_w)
            )
            stats.gate_mass[_pos].index_add_(0, flat_idx.cpu(), flat_w)

        handles.append(layer.router.register_forward_hook(hook))

        if activation_norms:
            def experts_hook(module, args, output, _pos=pos):
                hidden_states, top_k_index, top_k_weights = args
                stats.activation_saliency[_pos] += _activation_saliency(
                    module, hidden_states, top_k_index, top_k_weights
                )

            handles.append(layer.experts.register_forward_hook(experts_hook))

    model.eval()
    try:
        for batch in batches:
            input_ids = batch["input_ids"].to(model.device)
            model(input_ids=input_ids)
            stats.tokens += input_ids.numel()
    finally:
        for h in handles:
            h.remove()

    return stats


def weight_norms(model):
    """Per-expert L2 norm of expert weights, summed across layers [E]."""
    arch = MoEArch.from_model(model)
    norms = torch.zeros(arch.num_experts)
    for _, layer in iter_moe_layers(model):
        norms += layer.experts.gate_up_proj.data.float().norm(dim=(1, 2)).cpu()
        norms += layer.experts.down_proj.data.float().norm(dim=(1, 2)).cpu()
    return norms


def score_experts(model, strategy, stats=None):
    """Score every expert; lower score == better prune candidate. Returns [E]."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r} (choose from {STRATEGIES})")
    if strategy in _NEEDS_STATS and stats is None:
        raise ValueError(f"strategy {strategy!r} requires collected RouterStats")

    if strategy == "utilisation":
        return stats.selection_freq.mean(dim=0)
    if strategy == "magnitude":
        return weight_norms(model)
    if strategy == "reap":
        if not stats.has_activation_saliency:
            raise ValueError(
                "strategy 'reap' requires RouterStats collected with "
                "activation_norms=True"
            )
        return stats.mean_saliency.mean(dim=0)
    # gate_weight_norm: mean gate mass per token, weighted by expert weight norm
    gate = stats.gate_mass.sum(dim=0) / max(stats.tokens, 1)
    return gate * weight_norms(model)


def select_prune_candidates(scores, num_prune=None, threshold=None,
                            protected=(), min_keep=1):
    """Pick expert indices to prune from scores (lowest first)."""
    if (num_prune is None) == (threshold is None):
        raise ValueError("Specify exactly one of num_prune / threshold")
    order = torch.argsort(scores)
    candidates = [
        int(i) for i in order
        if int(i) not in set(protected)
        and (threshold is None or scores[i] < threshold)
    ]
    if threshold is None:
        candidates = candidates[:num_prune]
    max_prunable = len(scores) - max(min_keep, 1)
    return candidates[:max_prunable]


def prune_experts(model, indices, mode="remove"):
    """Prune experts by index. ``remove`` slices them out (router shrinks);
    ``zero`` zeroes weights in place for sparse runtimes."""
    indices = sorted(set(indices))
    if mode == "remove":
        manager = ExpertManager.from_model(model)
        if manager.arch.num_experts - len(indices) < manager.arch.top_k:
            raise ValueError(
                f"Cannot remove {len(indices)} experts: would leave fewer "
                f"than top_k={manager.arch.top_k}"
            )
        for idx in sorted(indices, reverse=True):
            manager.remove_expert(idx)
        return manager.arch
    if mode == "zero":
        with torch.no_grad():
            for _, layer in iter_moe_layers(model):
                for idx in indices:
                    layer.experts.gate_up_proj.data[idx].zero_()
                    layer.experts.down_proj.data[idx].zero_()
        return MoEArch.from_model(model)
    raise ValueError(f"Unknown mode {mode!r} (choose 'remove' or 'zero')")
