import pytest
import torch

from exex.pruner import (
    collect_router_stats,
    expert_output_norms,
    score_experts,
    select_prune_candidates,
    prune_experts,
)


@pytest.fixture
def stats(tiny_gemma4_moe, sample_batch):
    return collect_router_stats(
        tiny_gemma4_moe, [sample_batch], activation_norms=True
    )


class TestCollect:
    def test_counts_shape_and_mass(self, tiny_gemma4_moe, stats):
        assert stats.counts.shape == (2, 4)
        assert stats.tokens == 32
        # top-2 routing: selections per layer == tokens * k
        assert stats.counts.sum(dim=1).tolist() == [64.0, 64.0]
        assert (stats.gate_mass >= 0).all()

    def test_activation_saliency_shape_and_sign(self, stats):
        assert stats.has_activation_saliency
        assert stats.activation_saliency.shape == (2, 4)
        assert (stats.activation_saliency >= 0).all()
        assert (stats.activation_saliency > 0).any()
        assert stats.mean_saliency.shape == (2, 4)

    def test_saliency_off_by_default(self, tiny_gemma4_moe, sample_batch):
        s = collect_router_stats(tiny_gemma4_moe, [sample_batch])
        assert not s.has_activation_saliency
        assert (s.activation_saliency == 0).all()

    def test_zeroed_expert_has_zero_saliency(self, tiny_gemma4_moe, sample_batch):
        with torch.no_grad():
            for layer in tiny_gemma4_moe.model.layers:
                layer.experts.gate_up_proj.data[1].zero_()
                layer.experts.down_proj.data[1].zero_()
        s = collect_router_stats(
            tiny_gemma4_moe, [sample_batch], activation_norms=True
        )
        # zeroing weights does not touch the router: expert 1 is still routed to
        assert (s.counts[:, 1] > 0).all()
        assert (s.activation_saliency[:, 1] == 0).all()
        others = [e for e in range(4) if e != 1]
        assert (s.activation_saliency[:, others] > 0).any()

    def test_saliency_matches_manual_recomputation(self, tiny_gemma4_moe, sample_batch):
        model = tiny_gemma4_moe
        captured = []

        def grab(module, args, output):
            captured.append(tuple(a.detach().clone() for a in args))

        handles = [
            layer.experts.register_forward_hook(grab)
            for layer in model.model.layers
        ]
        try:
            s = collect_router_stats(model, [sample_batch], activation_norms=True)
        finally:
            for h in handles:
                h.remove()
        assert len(captured) == 2

        for pos, (hidden, top_k_index, top_k_weights) in enumerate(captured):
            experts = model.model.layers[pos].experts
            expected = torch.zeros(4)
            expected_counts = torch.zeros(4)
            for t in range(hidden.shape[0]):
                for k in range(top_k_index.shape[1]):
                    e = int(top_k_index[t, k])
                    g = top_k_weights[t, k].float()
                    x = hidden[t]
                    gate, up = (experts.gate_up_proj[e] @ x).chunk(2)
                    f = experts.down_proj[e] @ (experts.act_fn(gate) * up)
                    expected[e] += g * f.norm()
                    expected_counts[e] += 1
                    # helper agrees with the by-hand expert output norm
                    torch.testing.assert_close(
                        expert_output_norms(experts, x[None], e)[0], f.norm(),
                        rtol=1e-4, atol=1e-5,
                    )
            torch.testing.assert_close(
                s.activation_saliency[pos], expected, rtol=1e-4, atol=1e-5
            )
            torch.testing.assert_close(s.counts[pos], expected_counts)


class TestScores:
    def test_all_strategies(self, tiny_gemma4_moe, stats):
        for strategy in ("utilisation", "magnitude", "reap", "gate_weight_norm"):
            scores = score_experts(tiny_gemma4_moe, strategy, stats=stats)
            assert scores.shape == (4,)
            assert torch.isfinite(scores).all()

    def test_stats_required(self, tiny_gemma4_moe):
        for strategy in ("utilisation", "reap", "gate_weight_norm"):
            with pytest.raises(ValueError):
                score_experts(tiny_gemma4_moe, strategy)

    def test_reap_requires_activation_norms(self, tiny_gemma4_moe, sample_batch):
        s = collect_router_stats(tiny_gemma4_moe, [sample_batch])
        with pytest.raises(ValueError, match="activation_norms"):
            score_experts(tiny_gemma4_moe, "reap", stats=s)

    def test_reap_is_mean_saliency_over_layers(self, tiny_gemma4_moe, stats):
        scores = score_experts(tiny_gemma4_moe, "reap", stats=stats)
        expected = (stats.activation_saliency / stats.counts.clamp(min=1)).mean(dim=0)
        torch.testing.assert_close(scores, expected)
        assert (scores >= 0).all()

    def test_reap_differs_from_gate_weight_norm(self, tiny_gemma4_moe, stats):
        reap = score_experts(tiny_gemma4_moe, "reap", stats=stats)
        proxy = score_experts(tiny_gemma4_moe, "gate_weight_norm", stats=stats)
        assert not torch.allclose(reap, proxy)
        # also not merely a rescaling of the proxy
        assert not torch.allclose(reap / reap.sum(), proxy / proxy.sum())


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

    def test_remove_dedupes_indices(self, tiny_gemma4_moe):
        arch = prune_experts(tiny_gemma4_moe, [1, 1, 3], mode="remove")
        assert arch.num_experts == 2

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
