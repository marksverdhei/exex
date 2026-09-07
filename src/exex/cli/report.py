"""
Generate a Markdown + plots routing report for an MoE model.

Both the real-model path and ``--model_path mock`` run through
``exex.analyzer.RoutingAnalyzer`` (forward hooks on each ``layer.router``), so
the report code is exercised identically in both cases. Plotting deps
(matplotlib / seaborn, the ``analysis`` extra) are imported lazily.
"""

import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from exex.analyzer import RoutingAnalyzer


def create_nordic_dataset():
    """
    Creates a dataset representative of Norwegian and Nordic languages
    along with English, STEM, and Coding domains.
    """
    return [
        {"text": "Hei, hvordan går det? Norge er et vakkert land med mange fjell og fjorder.", "domain": "norwegian_bokmaal"},
        {"text": "Oslo er hovedstaden i Norge. Vi spiser ofte brunost på brødskiva.", "domain": "norwegian_bokmaal"},
        {"text": "Kva heiter du? Eg kjem frå Noreg og trivst godt her.", "domain": "norwegian_nynorsk"},
        {"text": "Det er mange fine stader på Vestlandet. Nynorsk er eit vakkert skriftspråk.", "domain": "norwegian_nynorsk"},
        {"text": "Hej, hur mår du? Sverige har många sjöar och skogar.", "domain": "swedish"},
        {"text": "Stockholm är Sveriges huvudstad. Vi älskar att fika med kanelbullar.", "domain": "swedish"},
        {"text": "Hej, hvordan har du det? Danmark er et fladt land med mange cykler.", "domain": "danish"},
        {"text": "København er en dejlig by. Smørrebrød er en klassisk dansk ret.", "domain": "danish"},
        {"text": "The quick brown fox jumps over the lazy dog. English is a global language.", "domain": "english"},
        {"text": "Photosynthesis is the process by which plants use sunlight to synthesize nutrients.", "domain": "stem"},
        {"text": "The mitochondria is the powerhouse of the cell.", "domain": "stem"},
        {"text": "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[0]\n    return quicksort([x for x in arr[1:] if x < pivot]) + [pivot] + quicksort([x for x in arr[1:] if x >= pivot])", "domain": "coding"},
        {"text": "import torch\nimport torch.nn as nn\nclass MoE(nn.Module):\n    def __init__(self):\n        super().__init__()", "domain": "coding"},
    ] * 20  # Repeat to get a decent number of samples


# --------------------------------------------------------------------- mock


class MockRouter(nn.Module):
    """Mimics the Gemma4 router contract: returns (probs, top_k_weights, top_k_index)."""

    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, logits):
        probs = torch.softmax(logits, dim=-1)
        top_w, top_i = torch.topk(probs, self.top_k, dim=-1)
        return probs, top_w, top_i


class MockMoELayer(nn.Module):
    def __init__(self, num_experts, top_k, hidden, inter):
        super().__init__()
        self.router = MockRouter(top_k)
        # Only the shapes matter: MoEArch.from_model reads them.
        self.experts = SimpleNamespace(
            gate_up_proj=torch.zeros(num_experts, 2 * inter, hidden),
            down_proj=torch.zeros(num_experts, inter, hidden),
        )


class MockMoEModel(nn.Module):
    """Fake MoE model with artificial domain / pair / cross-layer correlations.

    Structurally compatible with ``exex.arch`` (``.model.layers[i].router`` and
    ``.experts``), so ``RoutingAnalyzer`` can hook it like a real model.
    """

    def __init__(self, num_layers=4, num_experts=16, top_k=2, hidden=8, inter=4):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_experts=num_experts,
            top_k_experts=top_k,
            hidden_size=hidden,
            moe_intermediate_size=inter,
            enable_moe_block=True,
        )
        self.model = SimpleNamespace(
            layers=nn.ModuleList(
                MockMoELayer(num_experts, top_k, hidden, inter) for _ in range(num_layers)
            )
        )
        self.num_experts = num_experts

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids, **kwargs):
        batch, seq_len = input_ids.shape
        E = self.num_experts
        domain_hash = input_ids[0, 0].item() % E
        for l, layer in enumerate(self.model.layers):
            logits = torch.randn(batch, seq_len, E)
            # 1. Domain bias: one expert strongly tied to the domain
            main_expert = (domain_hash + l * 2) % E
            logits[:, :, main_expert] += 8.0
            # 2. Within-layer pair: main_expert co-fires with its neighbour
            logits[:, :, (main_expert + 1) % E] += 6.0
            # 3. Cross-layer pipeline emerges from the deterministic l*2 progression
            layer.router(logits)
        return None


class MockTokenizer:
    def __call__(self, text, return_tensors="pt", **kwargs):
        # Hash the text so that different domains map to different first tokens
        hashed = sum(ord(c) for c in text)
        return {"input_ids": torch.tensor([[hashed, hashed + 1, hashed + 2, hashed + 3]])}


# ------------------------------------------------------------------ report


def find_pipelines(cross_layer_co, layer_marginals, total_tokens, threshold=0.6, top=5):
    """Trace chains of experts across layers where P(e_{l+1} | e_l) > threshold."""
    num_layers, num_experts = layer_marginals.shape
    paths = []
    for l in range(num_layers - 1):
        for e in range(num_experts):
            if layer_marginals[l, e] < total_tokens * 0.01:  # skip very rare experts
                continue
            for next_e in range(num_experts):
                prob = cross_layer_co[l, e, next_e] / layer_marginals[l, e]
                if prob > threshold:
                    paths.append({"start_layer": l, "path": [e, next_e], "prob": prob})

    for p in paths:
        layer = p["start_layer"] + len(p["path"]) - 1
        expert = p["path"][-1]
        while layer < num_layers - 1:
            row = cross_layer_co[layer, expert] / max(1, layer_marginals[layer, expert])
            best_next = int(np.argmax(row))
            if row[best_next] > threshold:
                p["path"].append(best_next)
                expert = best_next
                layer += 1
            else:
                break

    paths.sort(key=lambda x: len(x["path"]), reverse=True)
    return paths[:top]


def tag_experts(domain_counts, domain_tokens, threshold=0.1):
    """Label each expert with the domain that activates it most (per token)."""
    num_layers, num_experts = next(iter(domain_counts.values())).shape
    tags = {l: {} for l in range(num_layers)}
    for l in range(num_layers):
        for e in range(num_experts):
            best_domain, best_val = None, 0
            for domain, counts in domain_counts.items():
                val = counts[l, e] / max(1, domain_tokens[domain])
                if val > best_val:
                    best_val, best_domain = val, domain
            if best_val > threshold:
                tags[l][e] = best_domain
    return tags


def make_plots(results, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    domain_counts = results["domain_expert_activations"]
    domain_tokens = results["total_tokens_per_domain"]
    within = results["co_occurrence"]
    cross = results["cross_layer_co"]
    num_layers, num_experts = within.shape[:2]
    total_tokens = sum(domain_tokens.values())

    print("Generating Bar Plots...")
    domains = list(domain_counts)
    cols = 4
    rows = (len(domains) + cols - 1) // cols
    plt.figure(figsize=(15, 4 * rows))
    for i, domain in enumerate(domains):
        plt.subplot(rows, cols, i + 1)
        avg_act = domain_counts[domain].sum(axis=0) / (domain_tokens[domain] * num_layers)
        plt.bar(range(num_experts), avg_act)
        plt.title(f"Domain: {domain}")
        plt.xlabel("Expert Index")
        plt.ylabel("Activation Freq")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "domain_bar_plots.png"))
    plt.close()

    print("Generating Density Plots...")
    plt.figure(figsize=(10, 6))
    for domain in domains:
        sns.kdeplot(results["top_k_weights"][domain], label=domain, fill=True, alpha=0.3,
                    clip=(0.0, 1.0))
    plt.title("Density of Top-K Expert Routing Weights by Domain")
    plt.xlabel("Router Weight")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "density_plots.png"))
    plt.close()

    print("Generating Heatmaps...")
    plt.figure(figsize=(16, 4))
    for l in range(min(4, num_layers)):
        plt.subplot(1, 4, l + 1)
        sns.heatmap(within[l] / max(1, total_tokens), cmap="YlGnBu", cbar=False)
        plt.title(f"Layer {l} Correlations")
        plt.xlabel("Expert")
        plt.ylabel("Expert")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmaps.png"))
    plt.close()

    print("Generating Cross-Layer Heatmaps...")
    plt.figure(figsize=(16, 4))
    for l in range(min(4, num_layers - 1)):
        plt.subplot(1, 4, l + 1)
        sns.heatmap(cross[l] / max(1, total_tokens), cmap="YlOrRd", cbar=False)
        plt.title(f"Layer {l} -> {l + 1} Correlations")
        plt.xlabel(f"Expert L+{l + 1}")
        plt.ylabel(f"Expert L+{l}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_layer_heatmaps.png"))
    plt.close()


def write_report(results, model_path, output_dir, top_paths, tags):
    domain_tokens = results["total_tokens_per_domain"]
    total_tokens = sum(domain_tokens.values())
    num_layers = results["co_occurrence"].shape[0]

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("# MoE Expert Analysis Report\n\n")
        f.write(f"**Model:** `{model_path}`\n")
        f.write(f"**Total Tokens Processed:** {total_tokens}\n")
        f.write(f"**MoE Layers:** {num_layers}\n\n")

        f.write("## 1. Takeaways\n")
        f.write("- **Domain Specialization:** Certain experts show strong specialization for specific languages and domains.\n")
        f.write("- **Block-wise Correlations:** Experts often fire in pairs within a layer.\n")
        f.write("- **Consecutive Expert Paths:** 'Expert pipelines' route the same token through a predictable sequence of experts across layers.\n\n")

        f.write("## 2. Longest Experts (Cross-Layer Pipelines)\n")
        f.write("Sequences of experts strongly correlated across consecutive MoE layers (activating Expert A in layer L strongly predicts Expert B in layer L+1).\n\n")
        if top_paths:
            for i, p in enumerate(top_paths):
                path_str = " -> ".join(f"E{e}" for e in p["path"])
                f.write(f"- **Pipeline {i + 1}** (Length {len(p['path'])}): Starts at Layer {p['start_layer']} | Sequence: `{path_str}`\n")
        else:
            f.write("- No strong multi-layer pipelines detected with threshold P > 0.6.\n")
        f.write("\n")

        f.write("## 3. Visualizations\n\n")
        f.write("### Domain Activations (Bar Plots)\n")
        f.write("Which experts are most frequently activated for each domain across all layers.\n\n")
        f.write("![Domain Bar Plots](./domain_bar_plots.png)\n\n")
        f.write("### Routing Weights (Density Plots)\n")
        f.write("Distribution of the router weights of selected experts. A bimodal distribution indicates strong certainty in routing.\n\n")
        f.write("![Density Plots](./density_plots.png)\n\n")
        f.write("### Layer Correlations (Heatmaps)\n")
        f.write("Co-occurrence matrices for the first 4 MoE layers. Bright spots off the diagonal indicate experts that consistently activate together.\n\n")
        f.write("![Correlation Heatmaps](./correlation_heatmaps.png)\n\n")
        f.write("### Cross-Layer Correlations\n")
        f.write("Transition counts from layer L to layer L+1. Bright spots indicate strong predictive flow between experts.\n\n")
        f.write("![Cross-Layer Heatmaps](./cross_layer_heatmaps.png)\n\n")

        f.write("## 4. Expert Tags (Top Specialists)\n")
        f.write("Based on relative per-token activation frequencies:\n\n")
        for l in range(min(2, num_layers)):
            f.write(f"### Layer {l}\n")
            if not tags[l]:
                f.write("- No distinct specialists found.\n")
            else:
                for e, domain in tags[l].items():
                    f.write(f"- **Expert {e}**: Tagged as `{domain}` specialist.\n")
            f.write("\n")
    return report_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, default="mock", help="Path to model or 'mock'")
    parser.add_argument("--output_dir", type=str, default="./report", help="Directory to save report")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples per domain")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_path == "mock":
        print("Using Mock MoE Model")
        analyzer = RoutingAnalyzer(
            MockMoEModel(num_layers=8, num_experts=32, top_k=4), tokenizer=MockTokenizer()
        )
    else:
        analyzer = RoutingAnalyzer(args.model_path, device=args.device)

    print("Running prompt processing on dataset...")
    results = analyzer.analyze_dataset(
        create_nordic_dataset(), max_samples_per_domain=args.max_samples
    )

    domain_counts = results["domain_expert_activations"]
    domain_tokens = results["total_tokens_per_domain"]
    total_tokens = sum(domain_tokens.values())

    make_plots(results, args.output_dir)

    print("Finding Longest Experts...")
    layer_marginals = sum(domain_counts.values())
    top_paths = find_pipelines(results["cross_layer_co"], layer_marginals, total_tokens)

    tags = tag_experts(domain_counts, domain_tokens)

    print("Writing Report...")
    report_path = write_report(results, args.model_path, args.output_dir, top_paths, tags)
    print(f"Report generated successfully at {report_path}")
    return report_path


# Backwards-compatible alias for the old scripts/generate_report.py entry point.
generate_report = main


if __name__ == "__main__":
    main()
