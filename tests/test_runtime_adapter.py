"""Runtime-adapter guarantee: installing a cartridge into a grown slot is
fully reversible — removing the slot returns the base model bit-identical.
This is the deployment-side isolation story: serve base for general traffic,
install the domain cartridge only for domain traffic (zero cross-domain cost
by construction)."""

import torch

from exex.cartridge import extract_cartridge
from exex.manager import ExpertManager
from exex.merger import install_expert


class TestRuntimeAdapter:
    def test_grow_install_remove_is_bit_identical(self, tiny_gemma4_moe, sample_batch):
        base_state = {n: p.detach().clone()
                      for n, p in tiny_gemma4_moe.state_dict().items()}
        with torch.no_grad():
            base_logits = tiny_gemma4_moe(**sample_batch).logits.clone()

        # Fake "domain" cartridge: expert 1 with perturbed weights
        cart = extract_cartridge(tiny_gemma4_moe, {"domain": 1})
        for key, tensor in cart.tensors.items():
            cart.tensors[key] = tensor + 0.05

        # Install into a grown slot ("cartridge on")
        landed = install_expert(tiny_gemma4_moe, cart, "domain", target_index=None,
                                check_fingerprint=False)
        assert landed == 4
        with torch.no_grad():
            on_logits = tiny_gemma4_moe(**sample_batch).logits
        assert not torch.equal(on_logits, base_logits)  # adapter has an effect

        # Remove the slot ("cartridge off") -> bit-identical base
        ExpertManager.from_model(tiny_gemma4_moe).remove_expert(landed)
        post_state = tiny_gemma4_moe.state_dict()
        assert set(post_state) == set(base_state)
        for name, tensor in base_state.items():
            assert torch.equal(post_state[name], tensor), name
        with torch.no_grad():
            off_logits = tiny_gemma4_moe(**sample_batch).logits
        assert torch.equal(off_logits, base_logits)
