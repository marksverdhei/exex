"""Cartridge exactness and composability (exex#26).

With the default trainer (target router rows only), a trained expert is fully
described by its cartridge: extract -> merge into a fresh base reproduces the
trained model tensor-for-tensor, and two experts trained separately merge
into one base without touching each other or any shared parameter.
"""
import copy

import torch

from exex.cartridge import extract_cartridge
from exex.merger import install_expert
from exex.trainer import ExpertTrainer


def _train(base, idx, batch, steps=3, **kw):
    model = copy.deepcopy(base)
    trainer = ExpertTrainer(model, [idx], lr=0.05, kl_weight=0.1, **kw)
    for _ in range(steps):
        trainer.train_step(**batch)
    trainer.finalize()
    return model


def _changed_keys(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sorted(k for k in sa if not torch.equal(sa[k], sb[k]))


class TestCartridgeExactness:
    def test_only_target_expert_and_its_router_row_change(self, tiny_gemma4_moe, sample_batch):
        base = tiny_gemma4_moe
        trained = _train(base, 1, sample_batch)
        changed = _changed_keys(base, trained)
        assert changed, "training did nothing"
        for k in changed:
            assert k.endswith(("experts.gate_up_proj", "experts.down_proj",
                               "router.proj.weight", "router.per_expert_scale")), k
        # Shared router scale and non-target rows are untouched
        sb, st = base.state_dict(), trained.state_dict()
        for k in st:
            if k.endswith("router.scale"):
                assert torch.equal(sb[k], st[k]), k
            if k.endswith(("router.proj.weight", "router.per_expert_scale",
                           "experts.gate_up_proj", "experts.down_proj")):
                keep = [i for i in range(sb[k].shape[0]) if i != 1]
                assert torch.equal(sb[k][keep], st[k][keep]), k

    def test_extract_merge_roundtrip_is_bit_exact(self, tiny_gemma4_moe, sample_batch):
        base = tiny_gemma4_moe
        trained = _train(base, 1, sample_batch)
        cart = extract_cartridge(trained, {"e1": 1})
        merged = copy.deepcopy(base)
        install_expert(merged, cart, "e1", target_index=1)
        assert _changed_keys(trained, merged) == []

    def test_two_experts_trained_separately_compose(self, tiny_gemma4_moe, sample_batch):
        base = tiny_gemma4_moe
        other = {k: torch.randint(0, 256, v.shape) for k, v in sample_batch.items()}
        t1 = _train(base, 1, sample_batch)
        t2 = _train(base, 2, other)
        c1 = extract_cartridge(t1, {"e1": 1})
        c2 = extract_cartridge(t2, {"e2": 2})
        merged = copy.deepcopy(base)
        install_expert(merged, c1, "e1", target_index=1)
        install_expert(merged, c2, "e2", target_index=2)
        sm, s1, s2, sb = merged.state_dict(), t1.state_dict(), t2.state_dict(), base.state_dict()
        for k in sm:
            if k.endswith(("experts.gate_up_proj", "experts.down_proj",
                           "router.proj.weight", "router.per_expert_scale")):
                assert torch.equal(sm[k][1], s1[k][1]), k
                assert torch.equal(sm[k][2], s2[k][2]), k
                rest = [i for i in range(sm[k].shape[0]) if i not in (1, 2)]
                assert torch.equal(sm[k][rest], sb[k][rest]), k
            else:
                assert torch.equal(sm[k], sb[k]), k

    def test_full_router_escape_hatch_moves_shared_scale(self, tiny_gemma4_moe, sample_batch):
        base = tiny_gemma4_moe
        trained = _train(base, 1, sample_batch, steps=5, train_full_router=True)
        changed = _changed_keys(base, trained)
        assert any(k.endswith("router.scale") for k in changed)
