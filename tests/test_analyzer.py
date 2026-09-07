import numpy as np
import pytest
import torch

from exex.analyzer import RoutingAnalyzer


class StubTokenizer:
    """Deterministic fake tokenizer: text length -> token count (clamped)."""

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
        n = min(max(len(text.split()), 1), max_length)
        return {"input_ids": torch.randint(0, self.vocab_size, (1, n))}


DATASET = [
    {"text": "one two three four", "domain": "a"},
    {"text": "five six", "domain": "a"},
    {"text": "seven eight nine", "domain": "b"},
    {"text": "ten", "domain": "b"},
    {"text": "this sample exceeds max_samples and is skipped", "domain": "b"},
]


@pytest.fixture
def analyzer(tiny_gemma4_moe):
    return RoutingAnalyzer(tiny_gemma4_moe, tokenizer=StubTokenizer())


@pytest.fixture
def results(analyzer):
    return analyzer.analyze_dataset(DATASET, max_samples_per_domain=2, verbose=False)


def test_requires_tokenizer_for_model_object(tiny_gemma4_moe):
    with pytest.raises(ValueError):
        RoutingAnalyzer(tiny_gemma4_moe)


def test_arch_derived(analyzer):
    assert analyzer.num_moe_layers == 2
    assert analyzer.num_experts == 4
    assert analyzer.top_k == 2
    assert analyzer.device == torch.device("cpu")


def test_total_tokens_per_domain(results):
    assert results["total_tokens_per_domain"] == {"a": 6, "b": 4}


def test_activation_shapes_and_topk_conservation(results):
    acts = results["domain_expert_activations"]
    assert set(acts) == {"a", "b"}
    for domain, tokens in results["total_tokens_per_domain"].items():
        assert acts[domain].shape == (2, 4)
        # each token selects exactly top_k=2 experts per layer
        np.testing.assert_array_equal(acts[domain].sum(axis=1), [2 * tokens, 2 * tokens])


def test_co_occurrence_symmetric_zero_diagonal(results):
    co = results["co_occurrence"]
    assert co.shape == (2, 4, 4)
    np.testing.assert_array_equal(co, co.transpose(0, 2, 1))
    for layer in range(2):
        assert np.all(np.diag(co[layer]) == 0)
        # one unordered pair per token per layer, counted in both directions
        assert co[layer].sum() == 2 * 10


def test_cross_layer_co(results):
    cross = results["cross_layer_co"]
    assert cross.shape == (1, 4, 4)
    # k * k (prev, cur) pairs per token
    assert cross.sum() == 4 * 10


def test_top_k_weights(results):
    for domain, tokens in results["total_tokens_per_domain"].items():
        w = results["top_k_weights"][domain]
        assert w.shape == (2 * 2 * tokens,)
        assert np.all(w >= 0) and np.all(w <= 1)


def test_hooks_removed_after_analysis(analyzer, results):
    for _, layer in analyzer.moe_layers:
        assert not layer.router._forward_hooks


def test_report_helpers_run(analyzer, results, capsys):
    analyzer.print_associations(results)
    found = analyzer.find_correlations(results, threshold_ratio=0.0)
    out = capsys.readouterr().out
    assert "Expert-Domain Associations" in out
    assert "Block-wise Expert Correlations" in out
    assert isinstance(found, list)
