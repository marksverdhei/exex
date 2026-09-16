"""
Routing-separation utilities (batch-4 separation arm).

The clone-and-noise experiment showed a copied router row stays a routing
twin of its source. The first centroid attempt showed the opposite failure:
a row *aligned with the input centroid* dominates routing everywhere
(slot prob 0.65 on all text, 26B job 2268845) because (a) norm-matching
ignores alignment — logits scale with cos(z, row), and trained rows are
nearly orthogonal to z while the centroid is not — and (b) the raw centroid
is mostly the shared anisotropic direction common to *all* text, so it does
not discriminate the domain.

``centroid_router_init`` therefore:
- uses the **difference of centroids** (domain − general) as the direction
  when negative batches are given, cancelling the shared component;
- **calibrates the magnitude** so the slot's mean logit on domain tokens
  equals the mean top-k-th logit of the existing experts — the slot competes
  at the top-k boundary on domain text instead of dominating it.

The contrastive push-down objective lives in ``ExpertTrainer``
(``contrastive_slot``).
"""

import torch
import torch.nn.functional as F

from exex.arch import MoEArch, iter_moe_layers


@torch.no_grad()
def _collect_router_stats(model, batches, want_kth):
    """One pass: per-layer input-centroid sums and (optionally) the mean
    top-k-th logit of the existing experts."""
    layers = [layer for _, layer in iter_moe_layers(model)]
    k = MoEArch.from_model(model).top_k
    sums = [None] * len(layers)
    kth_sums = [0.0] * len(layers)
    counts = [0] * len(layers)
    hooks = []
    for i, layer in enumerate(layers):
        def hook(module, args, output, _i=i):
            router = layers[_i].router
            hs = args[0].detach()
            z = (router.norm(hs) * router.scale * router.scalar_root_size).float()
            sums[_i] = z.sum(0) if sums[_i] is None else sums[_i] + z.sum(0)
            if want_kth:
                logits = F.linear(z, router.proj.weight.float())
                kth = torch.kthvalue(logits, logits.shape[-1] - k + 1, dim=-1).values
                kth_sums[_i] += kth.sum().item()
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

    centroids = [s / max(c, 1) if s is not None else None
                 for s, c in zip(sums, counts)]
    kth_means = [ks / max(c, 1) for ks, c in zip(kth_sums, counts)]
    return centroids, kth_means


@torch.no_grad()
def centroid_router_init(model, slot_idx, batches, neg_batches=None):
    """Re-initialize ``slot_idx``'s router row in every MoE layer.

    Direction: the domain centroid of (normed, scaled) router inputs — or,
    when ``neg_batches`` is given, the domain-minus-general centroid
    difference (recommended: the raw centroid is dominated by the shared
    direction of all text and does not discriminate).

    Magnitude: calibrated so the slot's mean logit on domain tokens equals
    the existing experts' mean top-k-th logit — the slot competes at the
    top-k boundary on domain text rather than dominating all routing.

    Returns per-layer dicts: cosine(new,old), slot mean domain logit,
    top-k-th mean logit — log these.
    """
    pos_centroids, kth_means = _collect_router_stats(model, batches, want_kth=True)
    neg_centroids = [None] * len(pos_centroids)
    if neg_batches is not None:
        neg_centroids, _ = _collect_router_stats(model, neg_batches, want_kth=False)

    diagnostics = []
    for i, (_, layer) in enumerate(iter_moe_layers(model)):
        router = layer.router
        pos = pos_centroids[i]
        if pos is None:
            diagnostics.append(None)
            continue
        direction = pos if neg_centroids[i] is None else pos - neg_centroids[i]
        direction = direction / direction.norm().clamp_min(1e-8)
        # slot logit on a domain token ~ z . row; mean over domain = pos . row
        proj = torch.dot(pos, direction)
        if proj.abs().item() < 1e-6:
            scale = torch.tensor(1.0, device=proj.device)
        else:
            scale = kth_means[i] / proj
        new_row = direction * scale

        weight = router.proj.weight.data
        old = weight[slot_idx].float()
        diagnostics.append({
            "cosine_to_old": F.cosine_similarity(new_row, old, dim=0).item(),
            "slot_domain_logit": torch.dot(pos, new_row).item(),
            "kth_logit": kth_means[i],
        })
        weight[slot_idx].copy_(new_row.to(weight.dtype))
        view = getattr(router, f"_router_row_{slot_idx}", None)
        if view is not None and view.data.data_ptr() != weight[slot_idx].data_ptr():
            view.data.copy_(new_row.to(view.dtype))
    return diagnostics
