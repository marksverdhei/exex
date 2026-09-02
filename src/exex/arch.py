"""
Config-driven MoE architecture descriptor.

Every exex module derives expert counts, top-k, and dimensions from here
instead of hardcoding a particular Gemma 4 variant. Supports any model that
exposes per-layer ``router`` / ``experts`` modules with fused 3D expert
tensors (Gemma 4 MoE naming), at any scale.
"""

from dataclasses import dataclass, asdict
import hashlib
import json


# Config keys tried in order, to tolerate naming differences across variants.
_NUM_EXPERTS_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
_TOP_K_KEYS = ("top_k_experts", "num_experts_per_tok", "moe_top_k")
_MOE_INTERMEDIATE_KEYS = ("moe_intermediate_size", "moe_ffn_hidden_size")


def _first_attr(config, keys, default=None):
    for key in keys:
        value = getattr(config, key, None)
        if value is not None:
            return value, key
    return default, None


@dataclass
class MoEArch:
    """Structural description of an MoE model, derived from its config."""

    num_layers: int
    num_experts: int
    top_k: int
    hidden_size: int
    moe_intermediate_size: int
    moe_layer_indices: list
    has_shared_mlp: bool
    num_experts_key: str = "num_experts"
    top_k_key: str = "top_k_experts"

    @classmethod
    def from_config(cls, config):
        config = getattr(config, "text_config", None) or config
        num_experts, ne_key = _first_attr(config, _NUM_EXPERTS_KEYS)
        top_k, tk_key = _first_attr(config, _TOP_K_KEYS)
        moe_intermediate, _ = _first_attr(config, _MOE_INTERMEDIATE_KEYS)
        if num_experts is None or top_k is None:
            raise ValueError(
                "Config does not describe an MoE model: "
                f"missing one of {_NUM_EXPERTS_KEYS} / {_TOP_K_KEYS}"
            )
        num_layers = config.num_hidden_layers
        return cls(
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=config.hidden_size,
            moe_intermediate_size=moe_intermediate,
            # Without a model instance, assume every layer carries MoE when
            # the config enables it globally. from_model() refines this.
            moe_layer_indices=list(range(num_layers))
            if getattr(config, "enable_moe_block", True)
            else [],
            has_shared_mlp=True,
            num_experts_key=ne_key or "num_experts",
            top_k_key=tk_key or "top_k_experts",
        )

    @classmethod
    def from_model(cls, model):
        """Derive from config, then refine by walking the actual layers."""
        arch = cls.from_config(model.config)
        arch.moe_layer_indices = [
            i for i, layer in enumerate(_decoder_layers(model))
            if hasattr(layer, "experts") and hasattr(layer, "router")
        ]
        first = next(iter_moe_layers(model), None)
        if first is not None:
            experts = first[1].experts
            arch.num_experts = experts.gate_up_proj.shape[0]
            arch.moe_intermediate_size = experts.gate_up_proj.shape[1] // 2
            arch.hidden_size = experts.gate_up_proj.shape[2]
            arch.has_shared_mlp = hasattr(first[1], "mlp")
        return arch

    def sync_config(self, config):
        """Write current expert count back to the config (after grow/shrink)."""
        config = getattr(config, "text_config", None) or config
        setattr(config, self.num_experts_key, self.num_experts)

    def fingerprint(self):
        """Hash of the dimensions that must match for expert transplants.

        Deliberately excludes num_experts and top_k: cartridges may move
        between models with different expert counts, but never between
        models with different tensor shapes.
        """
        payload = json.dumps(
            {
                "hidden_size": self.hidden_size,
                "moe_intermediate_size": self.moe_intermediate_size,
                "num_moe_layers": len(self.moe_layer_indices),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self):
        return asdict(self)


def _decoder_layers(model):
    inner = getattr(model, "model", model)
    return inner.layers


def iter_moe_layers(model):
    """Yield (layer_index, layer) for every layer with a router and experts."""
    for i, layer in enumerate(_decoder_layers(model)):
        if hasattr(layer, "experts") and hasattr(layer, "router"):
            yield i, layer
