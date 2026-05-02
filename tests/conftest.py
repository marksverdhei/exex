import pytest
import torch
from transformers import Gemma4ForCausalLM
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig


@pytest.fixture(autouse=True)
def seed():
    """Deterministic RNG for reproducible routing decisions."""
    torch.manual_seed(42)


@pytest.fixture
def tiny_gemma4_moe():
    """Tiny Gemma4 MoE model for testing. 4 experts, 2 layers, runs on CPU."""
    config = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        num_hidden_layers=2,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=32,
        max_position_embeddings=64,
        hidden_activation="gelu_pytorch_tanh",
    )
    model = Gemma4ForCausalLM(config)
    model.train()
    return model


@pytest.fixture
def sample_batch():
    """Minimal input batch for forward passes."""
    return {
        "input_ids": torch.randint(0, 256, (2, 16)),
        "labels": torch.randint(0, 256, (2, 16)),
    }
