import pytest
import torch

from exex.arch import iter_moe_layers
from exex.separation import centroid_router_init
from exex.trainer import ExpertTrainer


def _batches(n=4, seq=16):
    for _ in range(n):
        yield {"input_ids": torch.randint(0, 256, (1, seq))}


class TestCentroidInit:
    def test_rows_move_views_synced_and_logit_calibrated(self, tiny_gemma4_moe):
        from exex.surgery import prepare_router_rows_for_training
        prepare_router_rows_for_training(tiny_gemma4_moe, [2])
        old_rows = [layer.router.proj.weight.data[2].clone()
                    for _, layer in iter_moe_layers(tiny_gemma4_moe)]

        diags = centroid_router_init(tiny_gemma4_moe, 2, _batches(),
                                     neg_batches=_batches(seq=8))

        layers = list(iter_moe_layers(tiny_gemma4_moe))
        assert len(diags) == len(layers)
        for (_, layer), old, d in zip(layers, old_rows, diags):
            new = layer.router.proj.weight.data[2]
            assert torch.isfinite(new).all()
            assert not torch.equal(new, old)
            # trainable view shares the storage -> identical values
            view = getattr(layer.router, "_router_row_2")
            assert torch.equal(view.data, new)
            # calibrated: mean domain logit sits at the top-k boundary,
            # not far above it (the c3 dominance failure)
            assert d["slot_domain_logit"] == pytest.approx(d["kth_logit"], rel=0.05)

    def test_no_routing_dominance_after_init(self, tiny_gemma4_moe, sample_batch):
        """Slot must compete, not capture the router (26B job 2268845)."""
        from exex.pruner import collect_router_stats
        centroid_router_init(tiny_gemma4_moe, 2, _batches(),
                             neg_batches=_batches(seq=8))
        stats = collect_router_stats(tiny_gemma4_moe, [sample_batch])
        # 4 experts, top-2: uniform selection freq = 0.5; dominance would be ~1.0
        assert stats.selection_freq.mean(dim=0)[2].item() < 0.85


class TestContrastive:
    def test_push_down_reduces_slot_probability(self, tiny_gemma4_moe, sample_batch):
        trainer = ExpertTrainer(
            tiny_gemma4_moe, [1], contrastive_slot=1, contrastive_weight=5.0,
            lr=1e-2, router_lr_scale=1.0,
        )
        neg = torch.randint(0, 256, (2, 16))
        before = trainer.compute_contrastive_loss(neg).item()
        for _ in range(8):
            m = trainer.train_step(neg_input_ids=neg, **sample_batch)
        assert m["contrastive_loss"] is not None
        after = trainer.compute_contrastive_loss(neg).item()
        assert after < before

    def test_no_neg_batch_means_no_aux(self, tiny_gemma4_moe, sample_batch):
        trainer = ExpertTrainer(tiny_gemma4_moe, [1], contrastive_slot=1)
        m = trainer.train_step(**sample_batch)
        assert m["contrastive_loss"] is None


class TestRowWarmup:
    def test_expert_frozen_during_warmup_then_trains(self, tiny_gemma4_moe, sample_batch):
        trainer = ExpertTrainer(tiny_gemma4_moe, [1], row_warmup_steps=2, lr=1e-2)
        experts0 = tiny_gemma4_moe.model.layers[0].experts
        w_before = experts0.gate_up_proj.data[1].clone()
        row_before = tiny_gemma4_moe.model.layers[0].router.proj.weight.data[1].clone()

        for _ in range(2):  # warmup steps
            trainer.train_step(**sample_batch)
        assert torch.equal(experts0.gate_up_proj.data[1], w_before)
        assert not torch.equal(
            tiny_gemma4_moe.model.layers[0].router.proj.weight.data[1], row_before
        )

        trainer.train_step(**sample_batch)  # first post-warmup step
        assert not torch.equal(experts0.gate_up_proj.data[1], w_before)
