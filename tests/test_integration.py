import torch
from exex.manager import ExpertManager
from exex.trainer import ExpertTrainer


class TestFullPipeline:
    def test_clone_then_train(self, tiny_gemma4_moe, sample_batch):
        """Full pipeline: clone expert to new slot, train it, verify coexistence."""
        model = tiny_gemma4_moe
        original_num_experts = model.config.num_experts
        source_idx = 1

        # Clone expert 1 to a new slot
        manager = ExpertManager.from_model(model)
        new_idx = manager.clone_expert(source_idx=source_idx)
        assert new_idx == original_num_experts  # appended at end

        # Snapshot source expert weights (should stay frozen)
        source_weights = {}
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                source_weights[li] = layer.experts.gate_up_proj.data[source_idx].clone()

        # Train the NEW expert (not the source)
        trainer = ExpertTrainer(
            model=model,
            target_expert_indices=[new_idx],
            kl_weight=0.1,
            lr=0.01,
        )

        for _ in range(5):
            trainer.train_step(**sample_batch)

        # Source expert should be unchanged
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                assert torch.equal(
                    layer.experts.gate_up_proj.data[source_idx],
                    source_weights[li],
                ), f"Layer {li}: source expert was modified!"

        # New expert should have diverged from source
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                assert not torch.equal(
                    layer.experts.gate_up_proj.data[new_idx],
                    source_weights[li],
                ), f"Layer {li}: new expert didn't change!"
