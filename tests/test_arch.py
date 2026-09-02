import pytest

from exex.arch import MoEArch, iter_moe_layers


class TestMoEArch:
    def test_from_model_matches_fixture(self, tiny_gemma4_moe):
        arch = MoEArch.from_model(tiny_gemma4_moe)
        assert arch.num_experts == 4
        assert arch.top_k == 2
        assert arch.hidden_size == 64
        assert arch.moe_intermediate_size == 32
        assert arch.moe_layer_indices == [0, 1]
        assert arch.has_shared_mlp

    def test_iter_moe_layers(self, tiny_gemma4_moe):
        layers = list(iter_moe_layers(tiny_gemma4_moe))
        assert len(layers) == 2
        for i, layer in layers:
            assert hasattr(layer, "router") and hasattr(layer, "experts")

    def test_fingerprint_ignores_expert_count(self, tiny_gemma4_moe):
        from exex.manager import ExpertManager
        arch_before = MoEArch.from_model(tiny_gemma4_moe)
        ExpertManager.from_model(tiny_gemma4_moe).clone_expert(0)
        arch_after = MoEArch.from_model(tiny_gemma4_moe)
        assert arch_after.num_experts == arch_before.num_experts + 1
        assert arch_after.fingerprint() == arch_before.fingerprint()

    def test_sync_config(self, tiny_gemma4_moe):
        arch = MoEArch.from_model(tiny_gemma4_moe)
        arch.num_experts = 7
        arch.sync_config(tiny_gemma4_moe.config)
        assert tiny_gemma4_moe.config.num_experts == 7

    def test_non_moe_config_raises(self):
        class Dummy:
            num_hidden_layers = 2
            hidden_size = 8
        with pytest.raises(ValueError):
            MoEArch.from_config(Dummy())
