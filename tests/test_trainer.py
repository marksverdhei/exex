import pytest
import torch
from exex.trainer import ExpertTrainer


class TestExpertTrainer:
    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    def test_init_snapshots_router(self, tiny_gemma4_moe):
        """Trainer must snapshot pretrained router state for KL reference."""
        trainer = ExpertTrainer(
            model=tiny_gemma4_moe,
            target_expert_indices=[1],
            kl_weight=0.1,
            lr=1e-3,
        )
        assert len(trainer._ref_router_params) > 0
        # Reference params should be detached clones
        for ref in trainer._ref_router_params:
            assert not ref["proj_weight"].requires_grad

    def test_model_prepared_after_init(self, tiny_gemma4_moe):
        """After init, model should have trainable view params and frozen base."""
        trainer = ExpertTrainer(
            model=tiny_gemma4_moe,
            target_expert_indices=[1],
            kl_weight=0.1,
            lr=1e-3,
        )
        trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
        assert len(trainable) > 0
        # Should include router params and expert view params
        has_router = any("router" in n for n in trainable)
        has_expert = any("_train_" in n for n in trainable)
        assert has_router
        assert has_expert

    def test_kl_loss_is_zero_before_training(self, tiny_gemma4_moe, sample_batch):
        """KL loss should be ~0 before any optimizer steps (router hasn't changed)."""
        trainer = ExpertTrainer(
            model=tiny_gemma4_moe,
            target_expert_indices=[1],
            kl_weight=1.0,
            lr=1e-3,
        )
        _, kl_loss = trainer.compute_loss(**sample_batch)
        assert kl_loss.item() < 1e-5

    def test_train_step_reduces_task_loss(self, tiny_gemma4_moe, sample_batch):
        """A training step should reduce the task loss."""
        trainer = ExpertTrainer(
            model=tiny_gemma4_moe,
            target_expert_indices=[1],
            kl_weight=0.1,
            lr=0.01,
        )
        task_loss_before, _ = trainer.compute_loss(**sample_batch)
        trainer.train_step(**sample_batch)
        task_loss_after, _ = trainer.compute_loss(**sample_batch)
        assert task_loss_after.item() < task_loss_before.item()

    def test_kl_loss_increases_after_training(self, tiny_gemma4_moe, sample_batch):
        """After training steps, KL loss should increase (router has diverged)."""
        trainer = ExpertTrainer(
            model=tiny_gemma4_moe,
            target_expert_indices=[1],
            kl_weight=0.01,  # low weight so router can move
            lr=0.01,
        )
        for _ in range(5):
            trainer.train_step(**sample_batch)
        _, kl_loss = trainer.compute_loss(**sample_batch)
        assert kl_loss.item() > 1e-6

    def test_high_kl_weight_constrains_router(self, tiny_gemma4_moe, sample_batch):
        """High KL weight should keep router close to pretrained baseline."""
        trainer_low = ExpertTrainer(
            model=tiny_gemma4_moe,
            target_expert_indices=[1],
            kl_weight=0.001,
            lr=0.01,
        )
        trainer_high = ExpertTrainer(
            # Need a fresh model for fair comparison
            model=type(tiny_gemma4_moe).from_config(tiny_gemma4_moe.config),
            target_expert_indices=[1],
            kl_weight=100.0,
            lr=0.01,
        )
        # Copy weights so both start identical
        trainer_high.model.load_state_dict(tiny_gemma4_moe.state_dict(), strict=False)

        for _ in range(10):
            trainer_low.train_step(**sample_batch)
            trainer_high.train_step(**sample_batch)

        _, kl_low = trainer_low.compute_loss(**sample_batch)
        _, kl_high = trainer_high.compute_loss(**sample_batch)
        # High KL weight should result in less router divergence
        assert kl_high.item() < kl_low.item()
