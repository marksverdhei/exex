"""
Expert pruning: score experts on a calibration set, then remove or zero them.

Strategies:
    utilisation — mean routing selection frequency (drop the rarely-picked)
    magnitude   — L2 norm of expert weights (drop the smallest)
    reap        — router gate mass x expert weight norm (REAP-style proxy;
                  see arXiv:2510.13999)
"""

import torch

from exex.arch import MoEArch, iter_moe_layers
from exex.manager import ExpertManager

STRATEGIES = ("utilisation", "magnitude", "reap")


class RouterStats:
    """Per-layer routing statistics accumulated over calibration batches."""

    def __init__(self, num_layers, num_experts):
        self.counts = torch.zeros(num_layers, num_experts)
        self.gate_mass = torch.zeros(num_layers, num_experts)
        self.tokens = 0

    @property
    def selection_freq(self):
        """Fraction of tokens that routed to each expert, per layer [L, E]."""
        return self.counts / max(self.tokens, 1)


@torch.no_grad()
def collect_router_stats(model, batches):
    """Run calibration batches and capture routing decisions via hooks.

    Args:
        batches: iterable of dicts with at least ``input_ids``
    """
    arch = MoEArch.from_model(model)
    moe_layers = list(iter_moe_layers(model))
    stats = RouterStats(len(moe_layers), arch.num_experts)

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
    if strategy in ("utilisation", "reap") and stats is None:
        raise ValueError(f"strategy {strategy!r} requires collected RouterStats")

    if strategy == "utilisation":
        return stats.selection_freq.mean(dim=0)
    if strategy == "magnitude":
        return weight_norms(model)
    # reap: mean gate mass per token, weighted by expert weight norm
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
