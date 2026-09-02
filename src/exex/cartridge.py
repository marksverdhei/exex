"""
Expert cartridge format v0.

A cartridge is a single safetensors file holding one or more experts
extracted from an MoE checkpoint, portable between compatible models.

Tensor naming (one set per expert per MoE layer):
    experts.{name}.layers.{i}.gate_up      [2 * moe_intermediate, hidden]
    experts.{name}.layers.{i}.down         [hidden, moe_intermediate]
    experts.{name}.layers.{i}.router_row   [hidden]
    experts.{name}.layers.{i}.router_scale []  (per-expert scale scalar)

The safetensors ``__metadata__`` header (strings only; nested values are
JSON-encoded) carries:
    format_version   "0"
    source_model     free-form model id/path
    fingerprint      MoEArch.fingerprint() of the source model
    arch             JSON dict of source MoEArch
    experts          JSON manifest {name: {"source_index": int, "labels": [...]}}
"""

import json

import torch
from safetensors.torch import save_file, load_file

from exex.arch import MoEArch, iter_moe_layers

FORMAT_VERSION = "0"


class Cartridge:
    """In-memory representation of a loaded or extracted cartridge."""

    def __init__(self, tensors, metadata):
        self.tensors = tensors
        self.metadata = metadata

    @property
    def manifest(self):
        return json.loads(self.metadata["experts"])

    @property
    def expert_names(self):
        return list(self.manifest.keys())

    @property
    def fingerprint(self):
        return self.metadata["fingerprint"]

    def expert_tensors(self, name, layer_idx):
        """Return (gate_up, down, router_row, router_scale) for one layer."""
        prefix = f"experts.{name}.layers.{layer_idx}."
        return (
            self.tensors[prefix + "gate_up"],
            self.tensors[prefix + "down"],
            self.tensors[prefix + "router_row"],
            self.tensors[prefix + "router_scale"],
        )

    def save(self, path):
        save_file(self.tensors, path, metadata=self.metadata)

    @classmethod
    def load(cls, path):
        from safetensors import safe_open
        tensors = load_file(path)
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata() or {}
        version = metadata.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported cartridge format_version={version!r} "
                f"(supported: {FORMAT_VERSION})"
            )
        return cls(tensors, metadata)


def extract_cartridge(model, experts, source_model="", labels=None):
    """Extract experts from a model into a Cartridge.

    Args:
        model: MoE causal LM (Gemma 4 fused-tensor layout)
        experts: dict {name: expert_index} or list of indices
                 (named ``expert_{idx}`` automatically)
        source_model: free-form identifier recorded in metadata
        labels: optional dict {name: list_of_labels}
    """
    if not isinstance(experts, dict):
        experts = {f"expert_{idx}": idx for idx in experts}
    labels = labels or {}

    arch = MoEArch.from_model(model)
    tensors = {}
    manifest = {}

    for name, idx in experts.items():
        if not 0 <= idx < arch.num_experts:
            raise IndexError(
                f"expert index {idx} out of range (num_experts={arch.num_experts})"
            )
        for layer_idx, layer in iter_moe_layers(model):
            prefix = f"experts.{name}.layers.{layer_idx}."
            tensors[prefix + "gate_up"] = (
                layer.experts.gate_up_proj.data[idx].detach().cpu().contiguous()
            )
            tensors[prefix + "down"] = (
                layer.experts.down_proj.data[idx].detach().cpu().contiguous()
            )
            tensors[prefix + "router_row"] = (
                layer.router.proj.weight.data[idx].detach().cpu().contiguous()
            )
            tensors[prefix + "router_scale"] = (
                layer.router.per_expert_scale.data[idx].detach().cpu().clone()
            )
        manifest[name] = {
            "source_index": idx,
            "labels": labels.get(name, []),
        }

    metadata = {
        "format_version": FORMAT_VERSION,
        "source_model": str(source_model),
        "fingerprint": arch.fingerprint(),
        "arch": json.dumps(arch.to_dict()),
        "experts": json.dumps(manifest),
    }
    return Cartridge(tensors, metadata)


def save_cartridge(model, experts, path, source_model="", labels=None):
    """Extract experts and write a cartridge file. Returns the Cartridge."""
    cart = extract_cartridge(model, experts, source_model=source_model, labels=labels)
    cart.save(path)
    return cart


def load_cartridge(path):
    return Cartridge.load(path)
