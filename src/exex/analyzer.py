import torch
import numpy as np
from collections import defaultdict
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class RoutingAnalyzer:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map=device,
            output_router_logits=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.config = self.model.config

        # Architecture parameters derived from the model itself
        from exex.arch import MoEArch
        self.arch = MoEArch.from_model(self.model)
        self.num_layers = self.arch.num_layers
        self.num_experts = self.arch.num_experts
        self.top_k = self.arch.top_k
        
    @torch.no_grad()
    def analyze_dataset(self, dataset, text_col="text", domain_col="domain", max_samples_per_domain=100):
        """
        Analyzes router behavior over a multi-domain dataset.
        Returns matrices for expert-domain association and expert co-occurrence.
        """
        domain_counts = defaultdict(int)
        
        # We track how many times each expert was selected per layer per domain
        # shape: (num_layers, num_experts)
        domain_expert_activations = defaultdict(lambda: np.zeros((self.num_layers, self.num_experts)))
        
        # We track co-occurrence of experts within the same token routing decision
        # shape: (num_layers, num_experts, num_experts)
        co_occurrence = np.zeros((self.num_layers, self.num_experts, self.num_experts))
        total_tokens_per_domain = defaultdict(int)
        
        self.model.eval()
        
        # Organize dataset by domain to easily enforce max_samples_per_domain
        domain_data = defaultdict(list)
        for row in dataset:
            domain = row[domain_col]
            if len(domain_data[domain]) < max_samples_per_domain:
                domain_data[domain].append(row[text_col])
                
        for domain, texts in domain_data.items():
            print(f"Analyzing domain: {domain}")
            for text in tqdm(texts, desc=f"Processing {domain}"):
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # router_logits is a tuple of (batch_size, sequence_length, num_experts)
                router_logits = outputs.router_logits
                
                seq_len = inputs.input_ids.shape[1]
                total_tokens_per_domain[domain] += seq_len
                
                for layer_idx, logits in enumerate(router_logits):
                    # probabilities -> (1, seq_len, num_experts)
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Top-k indices -> (1, seq_len, top_k)
                    top_k_indices = torch.topk(probs, self.top_k, dim=-1).indices
                    
                    # Flatten batch and seq length -> (seq_len, top_k)
                    top_k_indices = top_k_indices.view(-1, self.top_k).cpu().numpy()
                    
                    for token_experts in top_k_indices:
                        for exp_idx in token_experts:
                            domain_expert_activations[domain][layer_idx, exp_idx] += 1
                            
                        # Update co-occurrence matrix for pairs (combinations of 2)
                        for exp_i, exp_j in itertools.combinations(token_experts, 2):
                            co_occurrence[layer_idx, exp_i, exp_j] += 1
                            co_occurrence[layer_idx, exp_j, exp_i] += 1

        results = {
            "domain_expert_activations": dict(domain_expert_activations),
            "co_occurrence": co_occurrence,
            "total_tokens_per_domain": dict(total_tokens_per_domain)
        }
        return results
        
    def print_associations(self, results):
        """Identifies which experts are most associated with which domains."""
        activations = results["domain_expert_activations"]
        tokens = results["total_tokens_per_domain"]
        
        print("\n--- Expert-Domain Associations ---")
        for domain in activations.keys():
            # Average activations per token for this domain
            avg_act = activations[domain] / tokens[domain]
            
            print(f"\nDomain: {domain} (Tokens: {tokens[domain]})")
            for layer in range(self.num_layers):
                # Find top 3 experts for this layer and domain
                top_experts = np.argsort(avg_act[layer])[::-1][:3]
                print(f"  Layer {layer:02d} Top Experts: " + 
                      ", ".join([f"E{e} ({avg_act[layer, e]:.3f}/tok)" for e in top_experts]))
                      
    def find_correlations(self, results, threshold_ratio=0.8):
        """
        Identifies block-wise expert correlations.
        Looks for one-to-one, one-to-many, etc.
        """
        print("\n--- Block-wise Expert Correlations ---")
        co_occurrence = results["co_occurrence"]
        
        # We need the marginal activations to calculate P(A|B) and P(B|A)
        # We can approximate marginals from the diagonal of co-occurrence if we tracked it,
        # or we can reconstruct it from domain_expert_activations.
        total_activations = np.zeros((self.num_layers, self.num_experts))
        for counts in results["domain_expert_activations"].values():
            total_activations += counts
            
        for layer in range(self.num_layers):
            layer_co = co_occurrence[layer]
            layer_act = total_activations[layer]
            
            correlations_found = False
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    if layer_co[i, j] == 0:
                        continue
                        
                    # Probability of seeing j given i is selected
                    p_j_given_i = layer_co[i, j] / layer_act[i] if layer_act[i] > 0 else 0
                    # Probability of seeing i given j is selected
                    p_i_given_j = layer_co[i, j] / layer_act[j] if layer_act[j] > 0 else 0
                    
                    if p_j_given_i >= threshold_ratio and p_i_given_j >= threshold_ratio:
                        if not correlations_found:
                            print(f"\nLayer {layer:02d}:")
                            correlations_found = True
                        print(f"  [1-to-1] E{i} <-> E{j} (P(j|i)={p_j_given_i:.2f}, P(i|j)={p_i_given_j:.2f})")
                    elif p_j_given_i >= threshold_ratio:
                        if not correlations_found:
                            print(f"\nLayer {layer:02d}:")
                            correlations_found = True
                        print(f"  [Many-to-1 / Dependency] E{i} -> E{j} (P(j|i)={p_j_given_i:.2f})")
                    elif p_i_given_j >= threshold_ratio:
                        if not correlations_found:
                            print(f"\nLayer {layer:02d}:")
                            correlations_found = True
                        print(f"  [Many-to-1 / Dependency] E{j} -> E{i} (P(i|j)={p_i_given_j:.2f})")
