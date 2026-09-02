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

            # Expand fused expert tensors
            source_gate_up = experts.gate_up_proj.data[source_idx:source_idx+1].clone()
            experts.gate_up_proj = nn.Parameter(
                torch.cat([experts.gate_up_proj.data, source_gate_up], dim=0)
            )
            source_down = experts.down_proj.data[source_idx:source_idx+1].clone()
            experts.down_proj = nn.Parameter(
                torch.cat([experts.down_proj.data, source_down], dim=0)
            )
            experts.num_experts += 1

            # Expand router
            router = layer.router
            source_row = router.proj.weight.data[source_idx:source_idx+1].clone()
            new_weight = torch.cat([router.proj.weight.data, source_row], dim=0)
            router.proj = nn.Linear(new_weight.shape[1], new_weight.shape[0], bias=False)
            router.proj.weight = nn.Parameter(new_weight)

            source_scale = router.per_expert_scale.data[source_idx:source_idx+1].clone()
            router.per_expert_scale = nn.Parameter(
                torch.cat([router.per_expert_scale.data, source_scale], dim=0)
            )

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
