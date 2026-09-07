"""In-process smoke tests for every ``exex-*`` CLI (``exex.cli.<mod>.main(argv)``).

One tiny Gemma4 MoE checkpoint (4 experts, 2 layers, fp32) is saved once per
session together with a tiny BPE tokenizer and a jsonl corpus; every CLI runs
against it with ``--dtype float32`` and a couple of optimizer steps. Each
tiny checkpoint is ~270 MB (per-layer input embeddings), so outputs that are
not needed by later tests are removed as soon as they have been checked and
the whole workspace is deleted at session end.
"""

import json
import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from exex.cli import analyze, evaluate, extract, manage, merge, prune, router_stats, train

SENTENCES = [
    ("The quick brown fox jumps over the lazy dog.", "english"),
    ("A journey of a thousand miles begins with a single step.", "english"),
    ("To be or not to be, that is the question.", "english"),
    ("All that glitters is not gold, my friend.", "english"),
    ("def add(a, b):\n    return a + b", "code"),
    ("for i in range(10):\n    print(i * i)", "code"),
    ("import os\nprint(os.getcwd())", "code"),
    ("class Foo:\n    pass", "code"),
]

COMMON = ["--dtype", "float32", "--max_length", "16"]
TRAIN_ARGS = COMMON + ["--max_steps", "2", "--batch_size", "2", "--log_every", "1"]


def _tiny_config():
    return Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        num_hidden_layers=2,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=32,
        max_position_embeddings=64,
        hidden_activation="gelu_pytorch_tanh",
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )


def _tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=200, special_tokens=["[PAD]", "[UNK]", "<bos>", "<eos>"]
    )
    tok.train_from_iterator([t for t, _ in SENTENCES], trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
        bos_token="<bos>", eos_token="<eos>",
    )


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _num_experts(model_dir):
    cfg = _read_json(os.path.join(model_dir, "config.json"))
    return (cfg.get("text_config") or cfg)["num_experts"]


def _rm(path):
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def ws(tmp_path_factory):
    """Session workspace; removed wholesale at the end (tiny checkpoints are big)."""
    root = tmp_path_factory.mktemp("exex_cli")
    yield root
    _rm(root)


@pytest.fixture(scope="session")
def tiny_model_dir(ws):
    """The ONE saved tiny model + tokenizer for the whole session."""
    torch.manual_seed(0)
    from transformers import Gemma4ForCausalLM

    model = Gemma4ForCausalLM(_tiny_config())
    out = ws / "tiny_model"
    model.save_pretrained(out)
    _tiny_tokenizer().save_pretrained(out)
    assert (out / "model.safetensors").exists()
    return str(out)


@pytest.fixture(scope="session")
def jsonl_path(ws):
    path = ws / "corpus.jsonl"
    with open(path, "w") as f:
        for text, domain in SENTENCES:
            f.write(json.dumps({"text": text, "domain": domain}) + "\n")
    return str(path)


@pytest.fixture(scope="session")
def trained_run(ws, tiny_model_dir, jsonl_path):
    """Train expert 1 in place once; shared by the train/eval/router_stats tests."""
    out = ws / "run_inplace"
    train.main([
        "--model_path", tiny_model_dir, "--dataset", jsonl_path,
        "--expert_indices", "1", "--output_dir", str(out), *TRAIN_ARGS,
    ])
    return str(out)


# ------------------------------------------------------------------- train


def test_train_in_place_writes_cartridge_not_model(trained_run):
    files = set(os.listdir(trained_run))
    assert {"cartridge.safetensors", "metrics.jsonl", "run.json", "config.json"} <= files
    assert "model.safetensors" not in files
    run = _read_json(os.path.join(trained_run, "run.json"))
    assert run["expert_indices"] == [1]
    assert run["num_experts"] == 4
    assert run["optimizer_steps"] == 2
    with open(os.path.join(trained_run, "metrics.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows and rows[-1]["step"] == 2


def test_train_save_full_model(ws, tiny_model_dir, jsonl_path):
    out = ws / "run_full"
    train.main([
        "--model_path", tiny_model_dir, "--dataset", jsonl_path,
        "--expert_indices", "0", "--output_dir", str(out), "--save_full_model", *TRAIN_ARGS,
    ])
    try:
        assert (out / "model.safetensors").exists()
        assert (out / "cartridge.safetensors").exists()
        assert (out / "tokenizer.json").exists()
    finally:
        _rm(out)


def test_train_clone_from_grows_expert(ws, tiny_model_dir, jsonl_path):
    out = ws / "run_clone"
    train.main([
        "--model_path", tiny_model_dir, "--dataset", jsonl_path,
        "--clone_from", "1", "--label", "clone", "--output_dir", str(out), *TRAIN_ARGS,
    ])
    try:
        run = _read_json(out / "run.json")
        assert run["num_experts"] == 5
        assert run["expert_indices"] == [4]
        assert _num_experts(out) == 5
    finally:
        _rm(out)


# -------------------------------------------------------------------- eval


def test_eval_with_cartridge(ws, tiny_model_dir, jsonl_path, trained_run):
    out = ws / "eval.json"
    cart = os.path.join(trained_run, "cartridge.safetensors")
    result = evaluate.main([
        "--model_path", tiny_model_dir, "--dataset", jsonl_path,
        "--cartridge", cart, "--output", str(out), *COMMON,
    ])
    saved = _read_json(out)
    assert set(saved) == set(result)
    assert "perplexity" in saved and "cartridges" in saved
    assert saved["cartridges"] == [cart]
    assert saved["perplexity"] > 0 and saved["num_tokens"] > 0


def test_eval_top_k_override(ws, tiny_model_dir, jsonl_path):
    out = ws / "eval_topk.json"
    evaluate.main([
        "--model_path", tiny_model_dir, "--dataset", jsonl_path,
        "--top_k", "3", "--output", str(out), *COMMON,
    ])
    saved = _read_json(out)
    assert saved["top_k"] == 3
    assert saved["perplexity"] > 0


# ------------------------------------------------------------ router_stats


def test_router_stats_cartridge_as_new_slot(ws, tiny_model_dir, jsonl_path, trained_run):
    out = ws / "router_stats.json"
    cart = os.path.join(trained_run, "cartridge.safetensors")
    router_stats.main([
        "--model_path", tiny_model_dir, "--dataset", jsonl_path,
        "--cartridge", f"{cart}:expert_1:new", "--output", str(out), *COMMON,
    ])
    saved = _read_json(out)
    assert saved["num_experts"] == 5
    assert len(saved["selection_freq_mean"]) == 5
    assert len(saved["selection_freq_per_layer"]) == 2
    assert saved["tokens"] > 0


# --------------------------------------------------------- extract / merge


def test_extract_then_merge_roundtrip(ws, tiny_model_dir):
    cart = ws / "extracted.safetensors"
    merged = ws / "merged"
    extract.main([
        "--model_path", tiny_model_dir, "--experts", '{"e1": 1}',
        "--labels", '{"e1": ["test"]}', "--output", str(cart), "--dtype", "float32",
    ])
    assert cart.exists()
    landed = merge.main([
        "--model_path", tiny_model_dir, "--cartridge", str(cart), "--expert", "e1",
        "--target_index", "1", "--output_dir", str(merged), "--dtype", "float32",
    ])
    assert landed == [1]
    try:
        src = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32)
        dst = AutoModelForCausalLM.from_pretrained(str(merged), dtype=torch.float32)
        for s_layer, d_layer in zip(src.model.layers, dst.model.layers):
            torch.testing.assert_close(
                d_layer.experts.gate_up_proj[1], s_layer.experts.gate_up_proj[1]
            )
            torch.testing.assert_close(d_layer.experts.down_proj[1], s_layer.experts.down_proj[1])
        assert (merged / "tokenizer.json").exists()
    finally:
        _rm(merged)


def test_merge_requires_exactly_one_mode(tiny_model_dir, tmp_path):
    with pytest.raises(SystemExit):
        merge.main(["--model_path", tiny_model_dir, "--output_dir", str(tmp_path / "x")])


# ------------------------------------------------------------------- prune


def test_prune_magnitude_zero_mode(ws, tiny_model_dir):
    out = ws / "pruned_zero"
    candidates = prune.main([
        "--model_path", tiny_model_dir, "--strategy", "magnitude", "--num_prune", "1",
        "--mode", "zero", "--output_dir", str(out), "--dtype", "float32",
    ])
    try:
        assert len(candidates) == 1
        assert _num_experts(out) == 4  # zero mode keeps the slot
        assert (out / "model.safetensors").exists()
    finally:
        _rm(out)


def test_prune_reap_removes_expert(ws, tiny_model_dir, jsonl_path):
    out = ws / "pruned_reap"
    prune.main([
        "--model_path", tiny_model_dir, "--strategy", "reap",
        "--calibration_dataset", jsonl_path, "--num_prune", "1",
        "--output_dir", str(out), *COMMON,
    ])
    try:
        assert _num_experts(out) == 3
        model = AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.float32)
        assert model.model.layers[0].experts.gate_up_proj.shape[0] == 3
    finally:
        _rm(out)


# ------------------------------------------------------------------ manage


def test_manage_add_clone(ws, tiny_model_dir):
    out = ws / "managed_add"
    manage.main([
        "--model_path", tiny_model_dir, "--output_dir", str(out),
        "add", "--clone_from", "0", "--label", "cloned",
    ])
    try:
        assert _num_experts(out) == 5
        cfg = _read_json(out / "config.json")
        labels = (cfg.get("text_config") or cfg).get("expert_labels", {})
        assert labels.get("4") == "cloned"
    finally:
        _rm(out)


def test_manage_remove_expert(ws, tiny_model_dir):
    out = ws / "managed_remove"
    manage.main([
        "--model_path", tiny_model_dir, "--output_dir", str(out),
        "remove", "--expert_index", "3",
    ])
    try:
        assert _num_experts(out) == 3
        assert (out / "model.safetensors").exists()
    finally:
        _rm(out)


# ----------------------------------------------------------------- analyze


def test_analyze_on_jsonl(tiny_model_dir, jsonl_path, capsys):
    results = analyze.main([
        "--model_path", tiny_model_dir, "--dataset_path", jsonl_path,
        "--max_samples", "4", "--max_length", "16", "--device", "cpu",
    ])
    assert set(results["domain_expert_activations"]) == {"english", "code"}
    assert results["co_occurrence"].shape == (2, 4, 4)
    assert all(n > 0 for n in results["total_tokens_per_domain"].values())
    assert "english" in capsys.readouterr().out


def test_analyze_dummy_dataset(tiny_model_dir):
    results = analyze.main([
        "--model_path", tiny_model_dir, "--max_samples", "2", "--max_length", "16",
        "--device", "cpu",
    ])
    assert set(results["domain_expert_activations"]) == {"eng", "stem", "coding", "multilingual"}


# ------------------------------------------------------------------ report


def test_report_mock(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from exex.cli import report

    out = tmp_path / "report"
    path = report.main(["--model_path", "mock", "--output_dir", str(out), "--max_samples", "5"])
    assert os.path.basename(path) == "report.md"
    assert (out / "report.md").exists()
    for png in ("domain_bar_plots", "density_plots", "correlation_heatmaps",
                "cross_layer_heatmaps"):
        assert (out / f"{png}.png").exists()
    text = (out / "report.md").read_text()
    assert "# MoE Expert Analysis Report" in text and "`mock`" in text


# ------------------------------------------------------------------- shims


@pytest.mark.parametrize("shim,module", [
    ("train_expert", "train"), ("eval_perplexity", "evaluate"),
    ("router_stats", "router_stats"), ("extract_experts", "extract"),
    ("merge_experts", "merge"), ("prune_experts", "prune"),
    ("manage_expert", "manage"), ("run_analysis", "analyze"),
    ("generate_report", "report"),
])
def test_script_shims_import_cli_main(shim, module):
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(shim, os.path.join(root, "scripts", f"{shim}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import importlib

    assert mod.main is importlib.import_module(f"exex.cli.{module}").main



def test_train_with_top_k_override(ws, tiny_model_dir, jsonl_path):
    out = ws / "run_topk"
    train.main([
        "--model_path", tiny_model_dir, "--dataset", jsonl_path,
        "--expert_indices", "1", "--top_k", "3", "--output_dir", str(out), *TRAIN_ARGS,
    ])
    run = _read_json(os.path.join(out, "run.json"))
    assert run["top_k"] == 3
    assert _read_json(os.path.join(out, "config.json"))["top_k_experts"] == 3
    _rm(out)
