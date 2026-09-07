"""Model loading helpers shared by the CLIs.

``load_model`` loads a causal LM and optionally installs cartridge experts
in memory before returning, so evals can run against ``base + cartridge``
without ever materialising a merged checkpoint on disk.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exex.merger import install_expert


def parse_cartridge_spec(spec):
    """``path[:expert[:target_index]]`` -> (path, expert_name|None, target|None).

    ``expert`` defaults to the cartridge's only expert; ``target_index``
    defaults to the expert's recorded ``source_index``. Use ``new`` as the
    target to grow a fresh slot.
    """
    parts = spec.split(":")
    path = parts[0]
    expert = parts[1] if len(parts) > 1 and parts[1] else None
    target = None
    if len(parts) > 2 and parts[2]:
        target = "new" if parts[2] == "new" else int(parts[2])
    return path, expert, target


def install_cartridges(model, specs):
    """Install every cartridge spec into ``model``; returns landed indices."""
    from exex.cartridge import load_cartridge

    landed = []
    for spec in specs:
        path, expert, target = parse_cartridge_spec(spec) if isinstance(spec, str) else spec
        cart = load_cartridge(path)
        if expert is None:
            if len(cart.expert_names) != 1:
                raise ValueError(
                    f"{path} holds {cart.expert_names}; specify which expert"
                )
            expert = cart.expert_names[0]
        if target is None:
            target = cart.manifest[expert]["source_index"]
        elif target == "new":
            target = None
        landed.append(install_expert(model, cart, expert, target_index=target))
    return landed


def load_model(model_path, dtype="bfloat16", device_map="auto",
               experts_impl=None, cartridges=(), eval_mode=True):
    """Load model (+tokenizer) and install cartridges. Returns (model, tokenizer)."""
    kwargs = {"dtype": getattr(torch, dtype) if isinstance(dtype, str) else dtype,
              "device_map": device_map}
    if experts_impl:
        kwargs["experts_implementation"] = experts_impl
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if cartridges:
        with torch.no_grad():
            landed = install_cartridges(model, cartridges)
        print(f"Installed cartridge experts at indices: {landed}")
    if eval_mode:
        model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
