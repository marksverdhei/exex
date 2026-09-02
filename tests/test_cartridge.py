import json

import pytest
import torch

from exex.cartridge import extract_cartridge, save_cartridge, load_cartridge


class TestCartridge:
    def test_extract_names_and_manifest(self, tiny_gemma4_moe):
        cart = extract_cartridge(
            tiny_gemma4_moe, {"medical": 1}, source_model="tiny",
            labels={"medical": ["medicine"]},
        )
        assert cart.expert_names == ["medical"]
        entry = cart.manifest["medical"]
        assert entry["source_index"] == 1
        assert entry["labels"] == ["medicine"]
        assert cart.metadata["source_model"] == "tiny"

    def test_extract_list_form(self, tiny_gemma4_moe):
        cart = extract_cartridge(tiny_gemma4_moe, [0, 2])
        assert set(cart.expert_names) == {"expert_0", "expert_2"}

    def test_tensors_match_model(self, tiny_gemma4_moe):
        cart = extract_cartridge(tiny_gemma4_moe, {"e": 3})
        layer = tiny_gemma4_moe.model.layers[0]
        gate_up, down, row, scale = cart.expert_tensors("e", 0)
        assert torch.equal(gate_up, layer.experts.gate_up_proj.data[3])
        assert torch.equal(down, layer.experts.down_proj.data[3])
        assert torch.equal(row, layer.router.proj.weight.data[3])
        assert torch.equal(scale, layer.router.per_expert_scale.data[3])

    def test_roundtrip(self, tiny_gemma4_moe, tmp_path):
        path = str(tmp_path / "cart.safetensors")
        saved = save_cartridge(tiny_gemma4_moe, {"e": 2}, path)
        loaded = load_cartridge(path)
        assert loaded.manifest == saved.manifest
        assert loaded.fingerprint == saved.fingerprint
        for key, tensor in saved.tensors.items():
            assert torch.equal(loaded.tensors[key], tensor)

    def test_bad_version_rejected(self, tiny_gemma4_moe, tmp_path):
        path = str(tmp_path / "cart.safetensors")
        cart = extract_cartridge(tiny_gemma4_moe, [0])
        cart.metadata["format_version"] = "999"
        cart.save(path)
        with pytest.raises(ValueError, match="format_version"):
            load_cartridge(path)

    def test_out_of_range_index(self, tiny_gemma4_moe):
        with pytest.raises(IndexError):
            extract_cartridge(tiny_gemma4_moe, [99])
