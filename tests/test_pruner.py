import pytest
import torch

from exex.arch import MoEArch
from exex.pruner import (
    collect_router_stats,
    score_experts,
    select_prune_candidates,
    prune_experts,
)


@pytest.fixture
def stats(tiny_gemma4_moe, sample_batch):
    return collect_router_stats(tiny_gemma4_moe, [sample_batch])


class TestCollect:
    def test_counts_shape_and_mass(self, tiny_gemma4_moe, stats):
        assert stats.counts.shape == (2, 4)
        assert stats.tokens == 32
        # top-2 routing: selections per layer == tokens * k
        assert stats.counts.sum(dim=1).tolist() == [64.0, 64.0]
        assert (stats.gate_mass >= 0).all()


class TestScores:
    def test_all_strategies(self, tiny_gemma4_moe, stats):
        for strategy in ("utilisation", "magnitude", "reap"):
            scores = score_experts(tiny_gemma4_moe, strategy, stats=stats)
            assert scores.shape == (4,)
            assert torch.isfinite(scores).all()

    def test_stats_required(self, tiny_gemma4_moe):
        with pytest.raises(ValueError):
            score_experts(tiny_gemma4_moe, "utilisation")


class TestSelect:
    def test_num_prune_and_protect(self):
        scores = torch.tensor([0.4, 0.1, 0.3, 0.2])
        assert select_prune_candidates(scores, num_prune=2) == [1, 3]
        assert select_prune_candidates(scores, num_prune=2, protected=[1]) == [3, 2]

    def test_threshold(self):
        scores = torch.tensor([0.4, 0.1, 0.3, 0.2])
        assert select_prune_candidates(scores, threshold=0.25) == [1, 3]

    def test_exactly_one_criterion(self):
        with pytest.raises(ValueError):
            select_prune_candidates(torch.ones(4))


class TestPrune:
    def test_remove(self, tiny_gemma4_moe, sample_batch):
        arch = prune_experts(tiny_gemma4_moe, [3, 1], mode="remove")
        assert arch.num_experts == 2
        assert tiny_gemma4_moe.config.num_experts == 2
        layer = tiny_gemma4_moe.model.layers[0]
        assert layer.experts.gate_up_proj.shape[0] == 2
        assert layer.router.proj.weight.shape[0] == 2
        with torch.no_grad():
            out = tiny_gemma4_moe(**sample_batch)
        assert torch.isfinite(out.loss)

    def test_remove_respects_top_k(self, tiny_gemma4_moe):
        with pytest.raises(ValueError, match="top_k"):
            prune_experts(tiny_gemma4_moe, [0, 1, 2], mode="remove")

    def test_zero(self, tiny_gemma4_moe, sample_batch):
        arch = prune_experts(tiny_gemma4_moe, [2], mode="zero")
        assert arch.num_experts == 4
        layer = tiny_gemma4_moe.model.layers[0]
        assert layer.experts.gate_up_proj.data[2].abs().sum() == 0
        with torch.no_grad():
            out = tiny_gemma4_moe(**sample_batch)
        assert torch.isfinite(out.loss)
