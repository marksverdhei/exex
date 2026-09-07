import torch
import torch.nn as nn

from exex.arch import MoEArch, iter_moe_layers


class ExpertManager:
    def __init__(self, model_path=None):
        if model_path is not None:
            from transformers import AutoModelForCausalLM, AutoConfig
            self.config = AutoConfig.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="cpu"
            )
            self.arch = MoEArch.from_model(self.model)
        # Fields set by from_model() when model_path is None

    @classmethod
    def from_model(cls, model):
        """Create an ExpertManager from an already-loaded model."""
        instance = cls.__new__(cls)
        instance.model = model
        instance.config = model.config
        instance.arch = MoEArch.from_model(model)
        return instance

    def label_expert(self, expert_index, label):
        if not hasattr(self.config, "expert_labels"):
            self.config.expert_labels = {}
        self.config.expert_labels[str(expert_index)] = label

    def get_labels(self):
        return getattr(self.config, "expert_labels", {})

    def clone_expert(self, source_idx, label=None):
        """Clone an existing expert to a new slot at the end."""
        if not 0 <= source_idx < self.arch.num_experts:
            raise IndexError(
                f"source_idx {source_idx} out of range (num_experts={self.arch.num_experts})"
            )
        for _, layer in iter_moe_layers(self.model):
            experts = layer.experts
            router = layer.router

            # Grow each fused tensor by one slot. Allocate the new tensor,
            # copy, drop the old one and release its cached CUDA blocks
            # before the next layer: with the old cat() pattern every freed
            # block was too small to be reused, so reserved memory doubled
            # (89 GB vs 68 GB at 26B, job 2175641).
            def grown(param):
                old = param.data
                new = old.new_empty((old.shape[0] + 1, *old.shape[1:]))
                new[:-1].copy_(old)
                new[-1].copy_(old[source_idx])
                return nn.Parameter(new, requires_grad=param.requires_grad)

            experts.gate_up_proj = grown(experts.gate_up_proj)
            experts.down_proj = grown(experts.down_proj)
            experts.num_experts += 1

            new_weight = grown(router.proj.weight)
            router.proj = nn.Linear(new_weight.shape[1], new_weight.shape[0], bias=False,
                                    device="meta")
            router.proj.weight = new_weight
            router.per_expert_scale = grown(router.per_expert_scale)
            if torch.cuda.is_available() and new_weight.is_cuda:
                torch.cuda.empty_cache()

        self.arch.num_experts += 1
        self.arch.sync_config(self.config)
        new_idx = self.arch.num_experts - 1

        if label:
            self.label_expert(new_idx, label)

        return new_idx

    def remove_expert(self, expert_idx, output_dir=None):
        """Remove an expert by slicing it out of the fused tensors and router."""
        if not 0 <= expert_idx < self.arch.num_experts:
            raise IndexError(
                f"expert_idx {expert_idx} out of range (num_experts={self.arch.num_experts})"
            )
        if self.arch.num_experts - 1 < self.arch.top_k:
            raise ValueError(
                f"Cannot remove: {self.arch.num_experts - 1} experts would be "
                f"fewer than top_k={self.arch.top_k}"
            )
        for _, layer in iter_moe_layers(self.model):
            experts = layer.experts

            # Remove from fused tensors
            experts.gate_up_proj = nn.Parameter(torch.cat([
                experts.gate_up_proj.data[:expert_idx],
                experts.gate_up_proj.data[expert_idx+1:],
            ], dim=0))
            experts.down_proj = nn.Parameter(torch.cat([
                experts.down_proj.data[:expert_idx],
                experts.down_proj.data[expert_idx+1:],
            ], dim=0))
            experts.num_experts -= 1

            # Shrink router
            router = layer.router
            new_weight = torch.cat([
                router.proj.weight.data[:expert_idx],
                router.proj.weight.data[expert_idx+1:],
            ], dim=0)
            router.proj = nn.Linear(new_weight.shape[1], new_weight.shape[0], bias=False)
            router.proj.weight = nn.Parameter(new_weight)

            router.per_expert_scale = nn.Parameter(torch.cat([
                router.per_expert_scale.data[:expert_idx],
                router.per_expert_scale.data[expert_idx+1:],
            ], dim=0))

        self.arch.num_experts -= 1
        self.arch.sync_config(self.config)

        # Update labels (shift indices down)
        if hasattr(self.config, "expert_labels"):
            new_labels = {}
            for k, v in self.config.expert_labels.items():
                idx = int(k)
                if idx < expert_idx:
                    new_labels[str(idx)] = v
                elif idx > expert_idx:
                    new_labels[str(idx - 1)] = v
            self.config.expert_labels = new_labels

        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.config.save_pretrained(output_dir)
