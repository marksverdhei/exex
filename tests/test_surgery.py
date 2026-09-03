import pytest
import torch

from exex.surgery import finalize_expert_training
from exex.trainer import ExpertTrainer


class TestFinalize:
    def test_trained_weights_persist_and_save_works(
        self, tiny_gemma4_moe, sample_batch, tmp_path
    ):
        trainer = ExpertTrainer(tiny_gemma4_moe, [1])
        trainer.train_step(**sample_batch)
        trained = tiny_gemma4_moe.model.layers[0].experts.gate_up_proj.data[1].clone()

        trainer.finalize()

        experts = tiny_gemma4_moe.model.layers[0].experts
        # No aliased view params, patched forward removed
        assert not hasattr(experts, "_train_gate_up_1")
        assert not hasattr(experts, "_train_indices")
        assert "forward" not in experts.__dict__
        assert not any("_train_" in n for n, _ in tiny_gemma4_moe.named_parameters())
        # Trained values live in the fused tensor
        assert torch.equal(experts.gate_up_proj.data[1], trained)
        # safetensors save no longer trips on shared tensors
        tiny_gemma4_moe.save_pretrained(tmp_path)
        assert (tmp_path / "model.safetensors").exists()
        # Model still runs with the restored class forward
        with torch.no_grad():
            out = tiny_gemma4_moe(**sample_batch)
        assert torch.isfinite(out.loss)

    def test_finalize_idempotent(self, tiny_gemma4_moe):
        from exex.surgery import prepare_expert_for_training
        prepare_expert_for_training(tiny_gemma4_moe, [0])
        finalize_expert_training(tiny_gemma4_moe)
        finalize_expert_training(tiny_gemma4_moe)  # no-op second time
from exex.surgery import prepare_expert_for_training


class TestPrepareExpertForTraining:
    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    def test_view_shares_memory(self, tiny_gemma4_moe):
        """Trainable view params must share data_ptr with fused tensor."""
        model = tiny_gemma4_moe
        prepare_expert_for_training(model, target_expert_indices=[1])

        for layer in model.model.layers:
            if not hasattr(layer, "experts"):
                continue
            experts = layer.experts
            assert experts.gate_up_proj.data[1].data_ptr() == experts._train_gate_up_1.data.data_ptr()
            assert experts.down_proj.data[1].data_ptr() == experts._train_down_1.data.data_ptr()

    def test_only_target_expert_trainable(self, tiny_gemma4_moe):
        """Only target expert view params should have requires_grad=True."""
        model = tiny_gemma4_moe
        prepare_expert_for_training(model, target_expert_indices=[1])

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert all("_train_" in n for n in trainable)
        assert len(trainable) > 0

    def test_frozen_params_no_grad(self, tiny_gemma4_moe, sample_batch):
        """Frozen params must not accumulate gradients after backward."""
        model = tiny_gemma4_moe
        prepare_expert_for_training(model, target_expert_indices=[1])

        outputs = model(**sample_batch)
        outputs.loss.backward()

        frozen = [p for n, p in model.named_parameters() if not p.requires_grad]
        grads_on_frozen = sum(1 for p in frozen if p.grad is not None and p.grad.abs().sum() > 0)
        assert grads_on_frozen == 0

    def test_trainable_params_get_grad(self, tiny_gemma4_moe, sample_batch):
        """Trainable view params must receive nonzero gradients."""
        model = tiny_gemma4_moe
        prepare_expert_for_training(model, target_expert_indices=[1])

        outputs = model(**sample_batch)
        outputs.loss.backward()

        trainable = [p for n, p in model.named_parameters() if p.requires_grad]
        for p in trainable:
            assert p.grad is not None and p.grad.abs().sum() > 0

    def test_optimizer_updates_fused_tensor(self, tiny_gemma4_moe, sample_batch):
        """Optimizer step on view params must update the fused tensor via shared storage."""
        model = tiny_gemma4_moe
        target_idx = 1

        original = {}
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                original[li] = layer.experts.gate_up_proj.data[target_idx].clone()

        prepare_expert_for_training(model, target_expert_indices=[target_idx])

        outputs = model(**sample_batch)
        outputs.loss.backward()

        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=0.01
        )
        optimizer.step()

        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                assert not torch.equal(
                    layer.experts.gate_up_proj.data[target_idx], original[li]
                ), f"Layer {li}: fused tensor not updated"

    def test_non_target_experts_unchanged(self, tiny_gemma4_moe, sample_batch):
        """Non-target expert weights must remain exactly unchanged after training step."""
        model = tiny_gemma4_moe
        target_idx = 1

        original = {}
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                original[li] = layer.experts.gate_up_proj.data.clone()

        prepare_expert_for_training(model, target_expert_indices=[target_idx])

        outputs = model(**sample_batch)
        outputs.loss.backward()

        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=0.01
        )
        optimizer.step()

        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                for eidx in range(4):
                    if eidx == target_idx:
                        continue
                    assert torch.equal(
                        layer.experts.gate_up_proj.data[eidx], original[li][eidx]
                    )

    def test_multiple_target_experts(self, tiny_gemma4_moe, sample_batch):
        """Support training multiple experts simultaneously."""
        model = tiny_gemma4_moe
        prepare_expert_for_training(model, target_expert_indices=[0, 2])

        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        # Should have view params for experts 0 and 2, across 2 layers, gate_up + down each
        assert len(trainable_names) == 8  # 2 experts * 2 layers * 2 projections

        outputs = model(**sample_batch)
        outputs.loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and p.grad.abs().sum() > 0
