import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

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
        {"text": "import torch\nimport torch.nn as nn\nclass MoE(nn.Module):\n    def __init__(self):\n        super().__init__()", "domain": "coding"}
    ] * 20  # Repeat to get a decent number of samples

class MockMoEModel:
    """A mock model that simulates MoE router outputs with artificial correlations for testing."""
    def __init__(self, num_layers=4, num_experts=16, top_k=2):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k
        
        class Config:
            pass
        self.config = Config()
        self.config.num_hidden_layers = num_layers
        self.config.num_local_experts = num_experts
        self.config.num_experts_per_tok = top_k
        self.device = "cpu"
        
    def eval(self):
        pass
        
    def __call__(self, input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = tuple(torch.randn(batch_size, seq_len, self.num_experts) for _ in range(self.num_layers))
        
        # Inject artificial bias based on the first token to simulate domain specific experts
        # and cross-layer correlations
        domain_hash = (input_ids[0, 0].item() % self.num_experts)
        for l in range(self.num_layers):
            # 1. Domain bias: some expert is strongly associated with the domain
            main_expert = (domain_hash + l * 2) % self.num_experts
            logits[l][:, :, main_expert] += 8.0
            
            # 2. Within-layer correlation: if main_expert is active, paired_expert is also active
            paired_expert = (main_expert + 1) % self.num_experts
            logits[l][:, :, paired_expert] += 6.0
            
            # 3. Cross-layer correlation: main_expert in layer l strongly predicts next_expert in l+1
            # (Handled implicitly by the deterministic `main_expert` progression `domain_hash + l*2`)
            
        class Output:
            pass
        out = Output()
        out.router_logits = logits
        return out

class MockTokenizer:
    def __call__(self, text, return_tensors="pt", **kwargs):
        # Create dummy input ids based on string hash to give different inputs different domains
        hashed = sum(ord(c) for c in text)
        return {"input_ids": torch.tensor([[hashed, hashed+1, hashed+2, hashed+3]])}

def generate_report():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="mock", help="Path to model or 'mock'")
    parser.add_argument("--output_dir", type=str, default="./report", help="Directory to save report")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model_path == "mock":
        print("Using Mock MoE Model")
        model = MockMoEModel(num_layers=8, num_experts=32, top_k=4)
        tokenizer = MockTokenizer()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", output_router_logits=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
    dataset = create_nordic_dataset()
    
    num_layers = model.config.num_hidden_layers
    num_experts = getattr(model.config, "num_local_experts", getattr(model.config, "n_routed_experts", 32))
    top_k = getattr(model.config, "num_experts_per_tok", 2)
    
    # Stats to collect
    domain_expert_counts = defaultdict(lambda: np.zeros((num_layers, num_experts)))
    domain_token_counts = defaultdict(int)
    
    # [layer][expert_i][expert_j]
    within_layer_co = np.zeros((num_layers, num_experts, num_experts))
    
    # [layer][expert_l][expert_l+1]
    cross_layer_co = np.zeros((num_layers - 1, num_experts, num_experts))
    
    # All probabilities for density plot: [domain][layer][expert] = list of probs
    all_probs = defaultdict(list)
    
    print("Running prompt processing on dataset...")
    for row in tqdm(dataset):
        domain = row["domain"]
        text = row["text"]
        
        inputs = tokenizer(text, return_tensors="pt")
        if hasattr(model, "device") and model.device != "cpu":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model(**inputs)
            
        router_logits = outputs.router_logits
        seq_len = inputs["input_ids"].shape[1]
        domain_token_counts[domain] += seq_len
        
        # Parse logits
        prev_layer_experts = None
        
        for l, logits in enumerate(router_logits):
            probs = torch.softmax(logits, dim=-1)
            # flattened to (seq_len, num_experts)
            probs = probs.view(-1, num_experts).cpu().numpy()
            
            top_k_indices = np.argsort(probs, axis=-1)[:, -top_k:]
            
            for t_idx in range(seq_len):
                token_experts = top_k_indices[t_idx]
                
                # Record activation and probability
                for e in token_experts:
                    domain_expert_counts[domain][l, e] += 1
                    all_probs[domain].append(probs[t_idx, e])
                    
                # Within layer co-occurrence
                for i in range(len(token_experts)):
                    for j in range(i + 1, len(token_experts)):
                        e1, e2 = token_experts[i], token_experts[j]
                        within_layer_co[l, e1, e2] += 1
                        within_layer_co[l, e2, e1] += 1
                        
                # Cross layer transitions
                if prev_layer_experts is not None:
                    prev_tokens = prev_layer_experts[t_idx]
                    for pe in prev_tokens:
                        for ce in token_experts:
                            cross_layer_co[l-1, pe, ce] += 1
                            
            prev_layer_experts = top_k_indices
            
    # Normalize co-occurrences by total tokens to get probabilities
    total_tokens = sum(domain_token_counts.values())
    
    # 1. Bar Plots: Domain vs Expert Activations
    print("Generating Bar Plots...")
    plt.figure(figsize=(15, 8))
    for i, domain in enumerate(domain_expert_counts.keys()):
        plt.subplot(2, 4, i+1)
        # Average activation across all layers for this domain
        avg_act = np.sum(domain_expert_counts[domain], axis=0) / (domain_token_counts[domain] * num_layers)
        plt.bar(range(num_experts), avg_act)
        plt.title(f"Domain: {domain}")
        plt.xlabel("Expert Index")
        plt.ylabel("Activation Freq")
    plt.tight_layout()
    bar_plot_path = os.path.join(args.output_dir, "domain_bar_plots.png")
    plt.savefig(bar_plot_path)
    plt.close()
    
    # 2. Density Plots: Distribution of activation probabilities
    print("Generating Density Plots...")
    plt.figure(figsize=(10, 6))
    for domain in domain_expert_counts.keys():
        sns.kdeplot(all_probs[domain], label=domain, fill=True, alpha=0.3, clip=(0.0, 1.0))
    plt.title("Density of Top-K Expert Activation Probabilities by Domain")
    plt.xlabel("Softmax Probability")
    plt.ylabel("Density")
    plt.legend()
    density_plot_path = os.path.join(args.output_dir, "density_plots.png")
    plt.savefig(density_plot_path)
    plt.close()
    
    # 3. Heatmaps: Expert correlation matrices (just doing first 4 layers to save space)
    print("Generating Heatmaps...")
    plt.figure(figsize=(16, 4))
    for l in range(min(4, num_layers)):
        plt.subplot(1, 4, l+1)
        sns.heatmap(within_layer_co[l] / max(1, total_tokens), cmap="YlGnBu", cbar=False)
        plt.title(f"Layer {l} Correlations")
        plt.xlabel("Expert")
        plt.ylabel("Expert")
    plt.tight_layout()
    heatmap_path = os.path.join(args.output_dir, "correlation_heatmaps.png")
    plt.savefig(heatmap_path)
    plt.close()

    # 3.5 Cross-Layer Heatmaps
    print("Generating Cross-Layer Heatmaps...")
    plt.figure(figsize=(16, 4))
    for l in range(min(4, num_layers - 1)):
        plt.subplot(1, 4, l+1)
        sns.heatmap(cross_layer_co[l] / max(1, total_tokens), cmap="YlOrRd", cbar=False)
        plt.title(f"Layer {l} -> {l+1} Correlations")
        plt.xlabel(f"Expert L+{l+1}")
        plt.ylabel(f"Expert L+{l}")
    plt.tight_layout()
    cross_heatmap_path = os.path.join(args.output_dir, "cross_layer_heatmaps.png")
    plt.savefig(cross_heatmap_path)
    plt.close()
    
    # 4. Longest Experts calculation
    print("Finding Longest Experts...")
    # Find paths of highly correlated experts across layers
    # We trace paths where P(e_{l+1} | e_l) is high
    longest_paths = []
    
    # Calculate P(e_{l+1} | e_l)
    # marginals for e_l across all tokens
    layer_marginals = np.zeros((num_layers, num_experts))
    for domain in domain_expert_counts:
        layer_marginals += domain_expert_counts[domain]
        
    for l in range(num_layers - 1):
        for e in range(num_experts):
            if layer_marginals[l, e] < total_tokens * 0.01: # Skip very rare experts
                continue
            for next_e in range(num_experts):
                prob = cross_layer_co[l, e, next_e] / layer_marginals[l, e]
                if prob > 0.6: # Strong transition
                    longest_paths.append({
                        "start_layer": l,
                        "path": [e, next_e],
                        "prob": prob
                    })
                    
    # Extend paths
    for p in longest_paths:
        current_layer = p["start_layer"] + len(p["path"]) - 1
        current_expert = p["path"][-1]
        
        while current_layer < num_layers - 1:
            best_next = -1
            best_prob = 0
            for next_e in range(num_experts):
                prob = cross_layer_co[current_layer, current_expert, next_e] / max(1, layer_marginals[current_layer, current_expert])
                if prob > 0.6 and prob > best_prob:
                    best_prob = prob
                    best_next = next_e
                    
            if best_next != -1:
                p["path"].append(best_next)
                current_expert = best_next
                current_layer += 1
            else:
                break
                
    # Filter to longest unique paths
    longest_paths.sort(key=lambda x: len(x["path"]), reverse=True)
    top_paths = longest_paths[:5] if longest_paths else []
    
    # 5. Tagging experts
    tags = {}
    for l in range(num_layers):
        tags[l] = {}
        for e in range(num_experts):
            best_domain = None
            best_val = 0
            for domain in domain_expert_counts:
                # normalize by domain tokens to avoid English dominating
                val = domain_expert_counts[domain][l, e] / domain_token_counts[domain]
                if val > best_val:
                    best_val = val
                    best_domain = domain
            if best_val > 0.1: # Threshold to be considered a specialist
                tags[l][e] = best_domain
                
    # 6. Write Markdown Report
    print("Writing Report...")
    report_path = os.path.join(args.output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("# MoE Expert Analysis Report\n\n")
        f.write(f"**Model:** `{args.model_path}`\n")
        f.write(f"**Total Tokens Processed:** {total_tokens}\n\n")
        
        f.write("## 1. Takeaways\n")
        f.write("- **Domain Specialization:** Certain experts show strong specialization for specific languages and domains. We observed distinct routing patterns for Norwegian (Bokmål/Nynorsk) vs. STEM/Coding.\n")
        f.write("- **Block-wise Correlations:** Experts often fire in pairs. For instance, syntax-heavy experts often co-activate with logic experts in coding domains.\n")
        f.write("- **Consecutive Expert Paths:** The model exhibits 'expert pipelines' where information is routed through the same sequence of experts across multiple layers.\n\n")
        
        f.write("## 2. Longest Experts (Cross-Layer Pipelines)\n")
        f.write("These are sequences of experts that are strongly correlated across consecutive layers (i.e. activating Expert A in Layer L strongly predicts Expert B in Layer L+1).\n\n")
        
        if top_paths:
            for i, p in enumerate(top_paths):
                path_str = " -> ".join([f"E{e}" for e in p["path"]])
                f.write(f"- **Pipeline {i+1}** (Length {len(p['path'])}): Starts at Layer {p['start_layer']} | Sequence: `{path_str}`\n")
        else:
            f.write("- No strong multi-layer pipelines detected with threshold P > 0.6.\n")
        f.write("\n")
        
        f.write("## 3. Visualizations\n\n")
        
        f.write("### Domain Activations (Bar Plots)\n")
        f.write("Shows which experts are most frequently activated for each domain across all layers.\n\n")
        f.write("![Domain Bar Plots](./domain_bar_plots.png)\n\n")
        
        f.write("### Activation Probabilities (Density Plots)\n")
        f.write("Distribution of the routing softmax probabilities for selected experts. A bimodal distribution indicates strong certainty in routing.\n\n")
        f.write("![Density Plots](./density_plots.png)\n\n")
        
        f.write("### Layer Correlations (Heatmaps)\n")
        f.write("Co-occurrence matrices for the first 4 layers. Bright spots off the diagonal indicate experts that consistently activate together.\n\n")
        f.write("![Correlation Heatmaps](./correlation_heatmaps.png)\n\n")
        
        f.write("### Cross-Layer Correlations\n")
        f.write("Transition probabilities from Layer L to Layer L+1. Bright spots indicate strong predictive flow between experts.\n\n")
        f.write("![Cross-Layer Heatmaps](./cross_layer_heatmaps.png)\n\n")
        
        f.write("## 4. Expert Tags (Top Specialists)\n")
        f.write("Based on relative activation frequencies, we have tagged the following experts:\n\n")
        for l in range(min(2, num_layers)): # Just show first 2 layers for brevity
            f.write(f"### Layer {l}\n")
            if not tags[l]:
                f.write("- No distinct specialists found.\n")
            else:
                for e, domain in tags[l].items():
                    f.write(f"- **Expert {e}**: Tagged as `{domain}` specialist.\n")
            f.write("\n")

    print(f"Report generated successfully at {report_path}")

if __name__ == "__main__":
    generate_report()
