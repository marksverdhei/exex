"""
Routing-separation utilities (batch-4 separation arm).

The clone-and-noise experiment showed a copied router row stays a routing
twin of its source: the new slot fires on the same distribution and domain
deltas leak into general traffic. These tools attack that directly:

- ``centroid_router_init``: point the new slot's row at the domain itself —
  the centroid of the domain tokens' (normed, scaled) router inputs.
- a contrastive push-down objective lives in ``ExpertTrainer``
  (``contrastive_slot``): negative-batch routing probability for the slot
  is minimized, gradients reaching only the trainable router rows.
"""

import torch
import torch.nn.functional as F

from exex.arch import iter_moe_layers


@torch.no_grad()
def centroid_router_init(model, slot_idx, batches):
    """Re-initialize ``slot_idx``'s router row in every MoE layer to the
    centroid of the domain batches' router inputs (after the router's own
    norm/scale), rescaled to the layer's mean row norm.

    Args:
        model: MoE causal LM (row-patched or not; trainable row views that
            share storage with ``router.proj.weight`` see the update).
        slot_idx: expert slot whose row is re-initialized
        batches: iterable of dicts with ``input_ids`` (domain text)

    Returns:
        list of per-layer cosine similarities between the new row and the
        old one — near 1.0 means the centroid didn't move the row (twin
        risk persists); log these.
    """
    layers = [layer for _, layer in iter_moe_layers(model)]
    sums = [None] * len(layers)
    counts = [0] * len(layers)
    hooks = []
    for i, layer in enumerate(layers):
        def hook(module, args, output, _i=i):
            router = layers[_i].router
            hs = args[0].detach()
            z = (router.norm(hs) * router.scale * router.scalar_root_size).float()
            sums[_i] = z.sum(0) if sums[_i] is None else sums[_i] + z.sum(0)
            counts[_i] += z.shape[0]
        hooks.append(layer.router.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    try:
        for batch in batches:
            model(input_ids=batch["input_ids"].to(model.device))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    cosines = []
    for i, layer in enumerate(layers):
        router = layer.router
        if counts[i] == 0:
            cosines.append(float("nan"))
            continue
        centroid = sums[i] / counts[i]
        weight = router.proj.weight.data
        target_norm = weight.float().norm(dim=1).mean()
        new_row = centroid / centroid.norm().clamp_min(1e-8) * target_norm
        cosines.append(
            F.cosine_similarity(new_row, weight[slot_idx].float(), dim=0).item()
        )
        # In-place write: any trainable row view sharing this storage sees it
        weight[slot_idx].copy_(new_row.to(weight.dtype))
        view = getattr(router, f"_router_row_{slot_idx}", None)
        if view is not None and view.data.data_ptr() != weight[slot_idx].data_ptr():
            view.data.copy_(new_row.to(view.dtype))
    return cosines
