import pytest
import torch

from exex.arch import MoEArch
from exex.cartridge import extract_cartridge
from exex.merger import install_expert, apply_merge_config


@pytest.fixture
def donor_cartridge(tiny_gemma4_moe):
    """Cartridge from a perturbed copy of the fixture model."""
    with torch.no_grad():
        for layer in tiny_gemma4_moe.model.layers:
            if hasattr(layer, "experts"):
                layer.experts.gate_up_proj.data[1] += 1.0
                layer.router.proj.weight.data[1] += 0.5
    return extract_cartridge(tiny_gemma4_moe, {"donor": 1}, source_model="donor")


class TestInstallExpert:
    def test_replace_slot(self, tiny_gemma4_moe, donor_cartridge):
        landed = install_expert(tiny_gemma4_moe, donor_cartridge, "donor",
                                target_index=3)
        assert landed == 3
        for i, layer in enumerate(tiny_gemma4_moe.model.layers):
            gate_up, down, row, scale = donor_cartridge.expert_tensors("donor", i)
            assert torch.equal(layer.experts.gate_up_proj.data[3], gate_up)
            assert torch.equal(layer.router.proj.weight.data[3], row)

    def test_grow_new_slot(self, tiny_gemma4_moe, donor_cartridge):
        landed = install_expert(tiny_gemma4_moe, donor_cartridge, "donor")
        assert landed == 4
        arch = MoEArch.from_model(tiny_gemma4_moe)
        assert arch.num_experts == 5
        assert tiny_gemma4_moe.config.num_experts == 5
        layer = tiny_gemma4_moe.model.layers[0]
        gate_up, _, row, _ = donor_cartridge.expert_tensors("donor", 0)
        assert torch.equal(layer.experts.gate_up_proj.data[4], gate_up)
        assert torch.equal(layer.router.proj.weight.data[4], row)

    def test_alpha_blend(self, tiny_gemma4_moe, donor_cartridge):
        layer = tiny_gemma4_moe.model.layers[0]
        incumbent = layer.experts.gate_up_proj.data[0].clone()
        install_expert(tiny_gemma4_moe, donor_cartridge, "donor",
                       target_index=0, alpha=0.5)
        donor_gate_up = donor_cartridge.expert_tensors("donor", 0)[0]
        expected = 0.5 * incumbent + 0.5 * donor_gate_up
        assert torch.allclose(layer.experts.gate_up_proj.data[0], expected)

    def test_fingerprint_mismatch_rejected(self, tiny_gemma4_moe, donor_cartridge):
        donor_cartridge.metadata["fingerprint"] = "deadbeefdeadbeef"
        with pytest.raises(ValueError, match="mismatch"):
            install_expert(tiny_gemma4_moe, donor_cartridge, "donor", target_index=0)

    def test_unknown_expert(self, tiny_gemma4_moe, donor_cartridge):
        with pytest.raises(KeyError):
            install_expert(tiny_gemma4_moe, donor_cartridge, "nope", target_index=0)

    def test_model_still_runs(self, tiny_gemma4_moe, donor_cartridge, sample_batch):
        install_expert(tiny_gemma4_moe, donor_cartridge, "donor")
        with torch.no_grad():
            out = tiny_gemma4_moe(**sample_batch)
        assert torch.isfinite(out.loss)


class TestMergeConfig:
    def test_batch_install(self, tiny_gemma4_moe, donor_cartridge, tmp_path):
        path = str(tmp_path / "donor.safetensors")
        donor_cartridge.save(path)
        landed = apply_merge_config(tiny_gemma4_moe, [
            {"cartridge": path, "expert": "donor", "target_index": 2},
            {"cartridge": path, "expert": "donor"},
        ])
        assert landed == [2, 4]
