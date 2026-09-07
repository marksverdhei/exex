import torch
from exex.manager import ExpertManager


class TestExpertManagerGemma4:
    def test_clone_expert_to_new_slot(self, tiny_gemma4_moe):
        """Cloning an expert should expand the fused tensor and router."""
        model = tiny_gemma4_moe
        manager = ExpertManager.from_model(model)

        original_num_experts = model.config.num_experts
        source_idx = 1

        manager.clone_expert(source_idx=source_idx)

        # Fused tensors should have grown by 1 in dim 0
        for layer in model.model.layers:
            if hasattr(layer, "experts"):
                assert layer.experts.gate_up_proj.shape[0] == original_num_experts + 1
                assert layer.experts.down_proj.shape[0] == original_num_experts + 1
                # New expert should equal source
                assert torch.equal(
                    layer.experts.gate_up_proj.data[-1],
                    layer.experts.gate_up_proj.data[source_idx],
                )

    def test_clone_expert_expands_router(self, tiny_gemma4_moe):
        """Cloning should add a new row to the router projection."""
        model = tiny_gemma4_moe
        manager = ExpertManager.from_model(model)

        original_num_experts = model.config.num_experts
        manager.clone_expert(source_idx=0)

        for layer in model.model.layers:
            if hasattr(layer, "router"):
                assert layer.router.proj.weight.shape[0] == original_num_experts + 1
                assert layer.router.per_expert_scale.shape[0] == original_num_experts + 1

    def test_remove_expert_shrinks_fused_tensor(self, tiny_gemma4_moe):
        """Removing an expert should shrink the fused tensor."""
        model = tiny_gemma4_moe
        manager = ExpertManager.from_model(model)

        original_num_experts = model.config.num_experts
        manager.remove_expert(expert_idx=2)

        for layer in model.model.layers:
            if hasattr(layer, "experts"):
                assert layer.experts.gate_up_proj.shape[0] == original_num_experts - 1


class TestCloneRouterNoise:
    def test_verbatim_clone_ties_with_source(self, tiny_gemma4_moe):
        from exex.manager import ExpertManager
        m = ExpertManager.from_model(tiny_gemma4_moe)
        new = m.clone_expert(1)
        for layer in tiny_gemma4_moe.model.layers:
            w = layer.router.proj.weight.data
            assert torch.equal(w[new], w[1])

    def test_noise_breaks_tie_but_keeps_expert_weights(self, tiny_gemma4_moe):
        from exex.manager import ExpertManager
        m = ExpertManager.from_model(tiny_gemma4_moe)
        new = m.clone_expert(1, router_noise=0.02, seed=3)
        for layer in tiny_gemma4_moe.model.layers:
            w = layer.router.proj.weight.data
            rel = (w[new] - w[1]).norm() / w[1].norm()
            assert 0.005 < rel.item() < 0.05
            assert torch.equal(layer.experts.gate_up_proj.data[new], layer.experts.gate_up_proj.data[1])
            assert torch.equal(layer.experts.down_proj.data[new], layer.experts.down_proj.data[1])

    def test_noise_is_deterministic(self, tiny_gemma4_moe):
        import copy
        from exex.manager import ExpertManager
        a = copy.deepcopy(tiny_gemma4_moe)
        b = copy.deepcopy(tiny_gemma4_moe)
        ExpertManager.from_model(a).clone_expert(1, router_noise=0.02, seed=7)
        ExpertManager.from_model(b).clone_expert(1, router_noise=0.02, seed=7)
        for la, lb in zip(a.model.layers, b.model.layers, strict=True):
            assert torch.equal(la.router.proj.weight.data, lb.router.proj.weight.data)
