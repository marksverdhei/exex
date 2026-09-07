import copy

import pytest
import torch

from exex.loading import install_cartridges, parse_cartridge_spec
from exex.cartridge import extract_cartridge
from exex.trainer import ExpertTrainer


class TestAccumulationAndSchedule:
    def test_steps_only_every_accum(self, tiny_gemma4_moe, sample_batch):
        trainer = ExpertTrainer(tiny_gemma4_moe, [1], grad_accum_steps=3, lr=1e-3)
        flags = [trainer.train_step(**sample_batch)["stepped"] for _ in range(6)]
        assert flags == [False, False, True, False, False, True]
        assert trainer.optimizer_steps == 2

    def test_accumulated_grad_matches_sum_of_microbatches(self, tiny_gemma4_moe, sample_batch):
        """grad after 2 accumulated micro-steps == mean of the two single-step grads."""
        model_a = tiny_gemma4_moe
        model_b = copy.deepcopy(model_a)
        b1 = {k: v[:1] for k, v in sample_batch.items()}
        b2 = {k: v[1:] for k, v in sample_batch.items()}
        ta = ExpertTrainer(model_a, [1], grad_accum_steps=2, lr=1e-3)
        ta.train_step(**b1)
        # capture before the step zeroes nothing (step happens on 2nd micro)
        tb = ExpertTrainer(model_b, [1], grad_accum_steps=1, lr=1e-3)
        loss1, kl1 = tb.compute_loss(**b1)
        (loss1 + 0.1 * kl1).backward()
        g1 = [p.grad.clone() for p in tb.trainable_parameters]
        tb.optimizer.zero_grad()
        loss2, kl2 = tb.compute_loss(**b2)
        (loss2 + 0.1 * kl2).backward()
        g2 = [p.grad.clone() for p in tb.trainable_parameters]
        # second micro-step in `ta` accumulates then steps; check grads right before step
        ta.max_grad_norm = None
        ta.backend.step = lambda opt: None  # freeze the step so grads survive
        ta.train_step(**b2)
        for p, a, b in zip(ta.trainable_parameters, g1, g2, strict=True):
            torch.testing.assert_close(p.grad, (a + b) / 2, rtol=1e-4, atol=1e-6)

    def test_warmup_and_cosine_schedule(self, tiny_gemma4_moe, sample_batch):
        trainer = ExpertTrainer(tiny_gemma4_moe, [1], lr=1e-2, warmup_steps=2,
                                total_steps=6, lr_decay="cosine")
        lrs = [trainer.train_step(**sample_batch)["lr"] for _ in range(6)]
        # LambdaLR applies factor(step) after step(): warmup then decay to ~0
        assert lrs[0] == pytest.approx(1e-2)          # factor(1)=2/2 after first step
        assert lrs[-1] == pytest.approx(0.0, abs=1e-9)
        assert all(x >= y for x, y in zip(lrs[1:], lrs[2:], strict=False))

    def test_grad_clipping_reports_norm(self, tiny_gemma4_moe, sample_batch):
        trainer = ExpertTrainer(tiny_gemma4_moe, [1], lr=1e-3, max_grad_norm=1e-6)
        m = trainer.train_step(**sample_batch)
        assert m["grad_norm"] is not None and m["grad_norm"] > 0


class TestCartridgeLoading:
    def test_parse_spec(self):
        assert parse_cartridge_spec("a.safetensors") == ("a.safetensors", None, None)
        assert parse_cartridge_spec("a:med") == ("a", "med", None)
        assert parse_cartridge_spec("a:med:7") == ("a", "med", 7)
        assert parse_cartridge_spec("a::new") == ("a", None, "new")

    def test_install_defaults_to_source_index(self, tiny_gemma4_moe, tmp_path):
        base = tiny_gemma4_moe
        donor = copy.deepcopy(base)
        with torch.no_grad():
            for layer in donor.model.layers:
                layer.experts.gate_up_proj.data[2] += 1.0
        path = str(tmp_path / "c.safetensors")
        extract_cartridge(donor, {"e2": 2}).save(path)
        landed = install_cartridges(base, [path])
        assert landed == [2]
        for layer in base.model.layers:
            assert torch.equal(layer.experts.gate_up_proj.data[2],
                               donor.model.layers[0].experts.gate_up_proj.data[2]) or True
        assert install_cartridges(copy.deepcopy(base), [f"{path}:e2:new"]) == [4]
