"""Perplexity evaluation shared by the eval CLI and the trainer's periodic eval."""

import json
import math
import os

import torch


@torch.no_grad()
def perplexity(model, tokenizer, texts, max_length=512):
    """Token-weighted mean NLL over texts -> (ppl, total_tokens)."""
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length).to(model.device)
        if enc.input_ids.shape[1] < 2:
            continue
        out = model(input_ids=enc.input_ids, labels=enc.input_ids)
        n_tokens = enc.input_ids.shape[1] - 1
        total_nll += out.loss.float().item() * n_tokens
        total_tokens += n_tokens
    if was_training:
        model.train()
    return math.exp(total_nll / max(total_tokens, 1)), total_tokens


def load_texts(dataset, text_column="text", max_samples=500, split=None):
    """Texts from a local json(l) file or an HF dataset (``name@split``)."""
    from datasets import load_dataset
    if os.path.isfile(dataset):
        ds = load_dataset("json", data_files=dataset, split="train")
    else:
        name, _, at_split = dataset.partition("@")
        ds = load_dataset(name, split=at_split or split or "test")
    n = min(max_samples, len(ds))
    return [ds[i][text_column] for i in range(n)]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
