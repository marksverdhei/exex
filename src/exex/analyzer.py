"""
Router behaviour analysis via forward hooks on each MoE layer's ``router``.

Works on transformers 5, which no longer accepts ``output_router_logits`` for
Gemma 4. Instead we register a forward hook on every ``layer.router`` (the same
pattern as :func:`exex.pruner.collect_router_stats`) and read the routing
decision the model actually made -- ``(router_probabilities, top_k_weights,
top_k_index)`` -- so per-expert scaling and any other router-internal logic are
respected rather than re-derived from a plain softmax.
"""

from collections import defaultdict
import itertools

import numpy as np
import torch
from tqdm import tqdm

from exex.arch import MoEArch, iter_moe_layers


def _load_dtype(device):
    """bf16 on CUDA (matches the checkpoints), fp32 elsewhere. Never fp16."""
    return torch.bfloat16 if str(device).startswith("cuda") else torch.float32


class RoutingAnalyzer:
    """Collect per-domain expert activation and co-occurrence statistics.

    Args:
        model_path_or_model: HF hub id / local path, or an already-loaded model
            exposing per-layer ``router`` / ``experts`` modules.
        device: target device; defaults to cuda if available. Ignored when an
            already-loaded model is passed (its own device is used).
        tokenizer: optional tokenizer-like callable
            ``tok(text, return_tensors="pt", truncation=..., max_length=...)``
            returning a mapping with ``input_ids``. Loaded from the model path
            when omitted; required when passing a model object.
        max_length: truncation length for each sample.
    """

    def __init__(self, model_path_or_model, device=None, tokenizer=None, max_length=512):
        self.max_length = max_length
        if isinstance(model_path_or_model, str):
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path_or_model,
                device_map=self.device,
                dtype=_load_dtype(self.device),
            )
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_path_or_model)
        else:
            self.model = model_path_or_model
            self.device = getattr(self.model, "device", None) or device or "cpu"
            if tokenizer is None:
                raise ValueError("tokenizer is required when passing an already-loaded model")
            self.tokenizer = tokenizer

        self.config = self.model.config
        self.arch = MoEArch.from_model(self.model)
        self.moe_layers = list(iter_moe_layers(self.model))
        # Result matrices are indexed by MoE-layer *position*, not decoder index.
        self.num_moe_layers = len(self.moe_layers)
        self.num_layers = self.arch.num_layers
        self.num_experts = self.arch.num_experts
        self.top_k = self.arch.top_k
        self._captured = {}

    # ------------------------------------------------------------------ hooks

    def _make_hook(self, pos):
        def hook(module, args, output, _pos=pos):
            _, top_k_weights, top_k_index = output
            k = top_k_index.shape[-1]
            self._captured[_pos] = (
                top_k_index.reshape(-1, k).detach().cpu(),
                top_k_weights.reshape(-1, k).detach().float().cpu(),
            )

        return hook

    def _register_hooks(self):
        return [
            layer.router.register_forward_hook(self._make_hook(pos))
            for pos, (_, layer) in enumerate(self.moe_layers)
        ]

    # --------------------------------------------------------------- analysis

    @torch.no_grad()
    def analyze_dataset(self, dataset, text_col="text", domain_col="domain",
                        max_samples_per_domain=100, verbose=True):
        """Analyze router behaviour over a multi-domain dataset.

        Returns a dict with:
            domain_expert_activations: {domain: [num_moe_layers, E]} selection counts
            co_occurrence: [num_moe_layers, E, E] symmetric within-token pair counts
                (zero diagonal)
            cross_layer_co: [num_moe_layers - 1, E, E] counts of
                (expert at layer l, expert at l+1) for the same token
            top_k_weights: {domain: 1-D array} router weight of every selection
            total_tokens_per_domain: {domain: int}
        """
        L, E, k = self.num_moe_layers, self.num_experts, self.top_k
        activations = defaultdict(lambda: torch.zeros(L, E, dtype=torch.float64))
        co_occurrence = torch.zeros(L, E, E, dtype=torch.float64)
        cross_layer_co = torch.zeros(max(L - 1, 0), E, E, dtype=torch.float64)
        top_k_weights = defaultdict(list)
        total_tokens = defaultdict(int)
        pair_offsets = list(itertools.combinations(range(k), 2))

        domain_data = defaultdict(list)
        for row in dataset:
            domain = row[domain_col]
            if len(domain_data[domain]) < max_samples_per_domain:
                domain_data[domain].append(row[text_col])

        self.model.eval()
        handles = self._register_hooks()
        try:
            for domain, texts in domain_data.items():
                iterator = tqdm(texts, desc=f"Processing {domain}") if verbose else texts
                for text in iterator:
                    inputs = self.tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=self.max_length
                    )
                    input_ids = inputs["input_ids"].to(self.device)
                    self._captured = {}
                    self.model(input_ids=input_ids)
                    total_tokens[domain] += input_ids.numel()

                    prev_idx = None
                    for pos in range(L):
                        idx, w = self._captured[pos]  # [T, k] each
                        activations[domain][pos] += torch.bincount(
                            idx.reshape(-1), minlength=E
                        ).double()
                        top_k_weights[domain].append(w.reshape(-1))
                        for i, j in pair_offsets:
                            pairs = idx[:, i] * E + idx[:, j]
                            co_occurrence[pos] += torch.bincount(
                                pairs, minlength=E * E
                            ).double().view(E, E)
                        if prev_idx is not None:
                            # every (prev expert, current expert) pair per token
                            pairs = (prev_idx.unsqueeze(2) * E + idx.unsqueeze(1)).reshape(-1)
                            cross_layer_co[pos - 1] += torch.bincount(
                                pairs, minlength=E * E
                            ).double().view(E, E)
                        prev_idx = idx
        finally:
            for h in handles:
                h.remove()
            self._captured = {}

        # Pair counts were accumulated one-directionally; make symmetric.
        co_occurrence = co_occurrence + co_occurrence.transpose(1, 2)

        return {
            "domain_expert_activations": {d: a.numpy() for d, a in activations.items()},
            "co_occurrence": co_occurrence.numpy(),
            "cross_layer_co": cross_layer_co.numpy(),
            "top_k_weights": {
                d: torch.cat(ws).numpy() if ws else np.zeros(0)
                for d, ws in top_k_weights.items()
            },
            "total_tokens_per_domain": dict(total_tokens),
        }

    # ---------------------------------------------------------------- reports

    def print_associations(self, results, top_n=3):
        """Print the experts most associated with each domain, per MoE layer."""
        activations = results["domain_expert_activations"]
        tokens = results["total_tokens_per_domain"]

        print("\n--- Expert-Domain Associations ---")
        for domain, counts in activations.items():
            avg_act = counts / max(tokens[domain], 1)
            print(f"\nDomain: {domain} (Tokens: {tokens[domain]})")
            for layer in range(counts.shape[0]):
                top_experts = np.argsort(avg_act[layer])[::-1][:top_n]
                print(
                    f"  MoE layer {layer:02d} Top Experts: "
                    + ", ".join(f"E{e} ({avg_act[layer, e]:.3f}/tok)" for e in top_experts)
                )

    def find_correlations(self, results, threshold_ratio=0.8):
        """Print block-wise expert correlations; returns [(layer, i, j, kind)]."""
        print("\n--- Block-wise Expert Correlations ---")
        co_occurrence = results["co_occurrence"]
        total_activations = sum(results["domain_expert_activations"].values())
        found = []

        for layer in range(co_occurrence.shape[0]):
            layer_co = co_occurrence[layer]
            layer_act = total_activations[layer]
            lines = []

            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    if layer_co[i, j] == 0:
                        continue
                    p_j_given_i = layer_co[i, j] / layer_act[i] if layer_act[i] > 0 else 0
                    p_i_given_j = layer_co[i, j] / layer_act[j] if layer_act[j] > 0 else 0

                    if p_j_given_i >= threshold_ratio and p_i_given_j >= threshold_ratio:
                        lines.append(f"  [1-to-1] E{i} <-> E{j} "
                                     f"(P(j|i)={p_j_given_i:.2f}, P(i|j)={p_i_given_j:.2f})")
                        found.append((layer, i, j, "1-to-1"))
                    elif p_j_given_i >= threshold_ratio:
                        lines.append(f"  [Dependency] E{i} -> E{j} (P(j|i)={p_j_given_i:.2f})")
                        found.append((layer, i, j, "dependency"))
                    elif p_i_given_j >= threshold_ratio:
                        lines.append(f"  [Dependency] E{j} -> E{i} (P(i|j)={p_i_given_j:.2f})")
                        found.append((layer, j, i, "dependency"))

            if lines:
                print(f"\nMoE layer {layer:02d}:")
                print("\n".join(lines))
        return found
