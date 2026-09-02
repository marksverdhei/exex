import pytest
import torch

from exex.backends import get_backend, TorchBackend
from exex.trainer import ExpertTrainer


class TestRegistry:
    def test_torch_backend(self):
        backend = get_backend("torch")
        assert isinstance(backend, TorchBackend)

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nope")

    def test_unsloth_stub_fails_informatively(self):
        with pytest.raises((ImportError, NotImplementedError)):
            get_backend("unsloth")


class TestTrainerIntegration:
    def test_explicit_torch_backend_trains(self, tiny_gemma4_moe, sample_batch):
        trainer = ExpertTrainer(tiny_gemma4_moe, [1], backend="torch")
        metrics = trainer.train_step(**sample_batch)
        assert torch.isfinite(torch.tensor(metrics["total_loss"]))

    def test_backend_instance_accepted(self, tiny_gemma4_moe, sample_batch):
        trainer = ExpertTrainer(tiny_gemma4_moe, [1], backend=TorchBackend())
        metrics = trainer.train_step(**sample_batch)
        assert torch.isfinite(torch.tensor(metrics["total_loss"]))
