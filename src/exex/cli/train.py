"""Train one or more MoE experts (in place, or in a freshly grown slot).

Default artefact is a cartridge (the trained experts + their router rows,
a few hundred MB at 26B); the full model is written only with
--save_full_model. Metrics go to <output_dir>/metrics.jsonl, run metadata
to <output_dir>/run.json.
"""

import argparse
import json
import os
import random
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from exex.cartridge import save_cartridge
from exex.evaluate import load_texts, perplexity
from exex.manager import ExpertManager
from exex.trainer import ExpertTrainer


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset", required=True, help="HF dataset name or local json(l)")
    p.add_argument("--text_column", default="text")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--expert_indices", type=int, nargs="+",
                        help="Existing expert slots to train in place")
    target.add_argument("--clone_from", type=int, default=None,
                        help="Grow a new slot cloned from this expert and train "
                             "that instead (expert extension)")
    p.add_argument("--clone_router_noise", type=float, default=0.0,
                   help="With --clone_from: relative Gaussian noise on the new slot's "
                        "router row so it does not tie with its source (0 = verbatim copy)")
    p.add_argument("--label", default=None,
                   help="Label recorded in the cartridge manifest / config")
    # optimisation
    p.add_argument("--kl_weight", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--router_lr_scale", type=float, default=0.1)
    p.add_argument("--train_full_router", action="store_true",
                   help="Train every router parameter (shared scale, all rows). "
                        "Default trains only the target experts' rows so "
                        "cartridges stay exact.")
    p.add_argument("--max_steps", type=int, default=1000,
                   help="Optimizer steps (micro-batches = max_steps * grad_accum)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Clip total grad norm; <=0 disables")
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--lr_decay", default="none", choices=["none", "linear", "cosine"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_shuffle", action="store_true",
                   help="Iterate the dataset in file order instead of a seeded shuffle")
    # periodic eval
    p.add_argument("--eval_dataset", default=None, help="Held-out json(l) for PPL")
    p.add_argument("--eval_every", type=int, default=0, help="Optimizer steps between evals")
    p.add_argument("--eval_samples", type=int, default=100)
    # output
    p.add_argument("--output_dir", required=True)
    p.add_argument("--save_full_model", action="store_true",
                   help="Also write the full model with save_pretrained (large)")
    p.add_argument("--no_cartridge", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Load dtype; bf16 default (fp16 overflows on natively-bf16 checkpoints)")
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = getattr(torch, args.dtype)
    load_kwargs = {"dtype": dtype, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype
        )
    print(f"Loading model from {args.model_path}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    manager = ExpertManager.from_model(model)
    if args.clone_from is not None:
        new_idx = manager.clone_expert(source_idx=args.clone_from, label=args.label,
                                       router_noise=args.clone_router_noise, seed=args.seed)
        expert_indices = [new_idx]
        print(f"Cloned expert {args.clone_from} -> new slot {new_idx} "
              f"(num_experts now {manager.arch.num_experts})", flush=True)
    else:
        expert_indices = list(args.expert_indices)
        if args.label:
            for idx in expert_indices:
                manager.label_expert(idx, args.label)

    trainer = ExpertTrainer(
        model=model,
        target_expert_indices=expert_indices,
        kl_weight=args.kl_weight,
        lr=args.lr,
        router_lr_scale=args.router_lr_scale,
        train_full_router=args.train_full_router,
        grad_accum_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
    )
    n_trainable = sum(p.numel() for p in trainer.trainable_parameters)

    print(f"Loading dataset {args.dataset}...", flush=True)
    if os.path.isfile(args.dataset):
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    else:
        dataset = load_dataset(args.dataset, split="train")
    eval_texts = None
    if args.eval_dataset and args.eval_every > 0:
        eval_texts = load_texts(args.eval_dataset, args.text_column, args.eval_samples)

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    # Kept open across the whole training loop (closed after the final eval).
    metrics_f = open(metrics_path, "w")  # noqa: SIM115

    def log(row):
        metrics_f.write(json.dumps(row) + "\n")
        metrics_f.flush()

    def run_eval(step):
        ppl, n_tok = perplexity(model, tokenizer, eval_texts, max_length=args.max_length)
        print(f"[eval] step {step} | ppl={ppl:.4f} over {n_tok} tokens", flush=True)
        log({"step": step, "eval_ppl": ppl, "eval_tokens": n_tok})
        return ppl

    print(f"Training experts {expert_indices} ({n_trainable:,} trainable params) "
          f"for {args.max_steps} optimizer steps x {args.grad_accum} micro-batches "
          f"of {args.batch_size}...", flush=True)
    order = list(range(len(dataset)))
    t0 = time.time()
    tokens_seen = 0
    step = 0
    epoch = 0
    eval_history = []
    if eval_texts:
        eval_history.append(run_eval(0))
    while step < args.max_steps:
        if not args.no_shuffle:
            random.Random(args.seed + epoch).shuffle(order)
        for i in range(0, len(order), args.batch_size):
            if step >= args.max_steps:
                break
            batch_texts = [dataset[j][args.text_column] for j in order[i:i + args.batch_size]]
            enc = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                            max_length=args.max_length, padding=True).to(model.device)
            labels = enc.input_ids.clone()
            labels[enc.attention_mask == 0] = -100
            m = trainer.train_step(input_ids=enc.input_ids,
                                   attention_mask=enc.attention_mask, labels=labels)
            tokens_seen += int(enc.attention_mask.sum())
            if not m["stepped"]:
                continue
            step += 1
            if step % args.log_every == 0 or step == args.max_steps:
                row = {"step": step, "epoch": epoch, "tokens_seen": tokens_seen,
                       "elapsed_s": round(time.time() - t0, 1),
                       **{k: m[k] for k in ("task_loss", "kl_loss", "total_loss", "lr", "grad_norm")}}
                log(row)
                print(f"Step {step}/{args.max_steps} | task_loss={m['task_loss']:.4f} | "
                      f"kl_loss={m['kl_loss']:.6f} | lr={m['lr']:.2e}"
                      + (f" | gnorm={m['grad_norm']:.3f}" if m["grad_norm"] is not None else ""),
                      flush=True)
            if eval_texts and step % args.eval_every == 0 and step < args.max_steps:
                eval_history.append(run_eval(step))
        epoch += 1
    if eval_texts:
        eval_history.append(run_eval(step))
    metrics_f.close()

    trainer.finalize()
    artefacts = {}
    if not args.no_cartridge:
        cart_path = os.path.join(args.output_dir, "cartridge.safetensors")
        names = {(args.label or f"expert_{idx}") if len(expert_indices) == 1
                 else f"{args.label or 'expert'}_{idx}": idx for idx in expert_indices}
        labels = {n: [args.label] for n in names} if args.label else None
        save_cartridge(model, names, cart_path, source_model=args.model_path, labels=labels)
        artefacts["cartridge"] = cart_path
        print(f"Wrote cartridge {cart_path} ({os.path.getsize(cart_path) / 1e6:.0f} MB): "
              f"{list(names)}", flush=True)
    if args.save_full_model:
        print(f"Saving full model to {args.output_dir}...", flush=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        artefacts["model"] = args.output_dir
    else:
        # config alone is enough to know the resulting expert count / labels
        model.config.save_pretrained(args.output_dir)

    run = {"args": vars(args), "expert_indices": expert_indices,
           "num_experts": manager.arch.num_experts, "trainable_params": n_trainable,
           "optimizer_steps": step, "tokens_seen": tokens_seen,
           "elapsed_s": round(time.time() - t0, 1), "eval_ppl_history": eval_history,
           "artefacts": artefacts}
    with open(os.path.join(args.output_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=2)
    print("Done.", flush=True)
    return run


if __name__ == "__main__":
    main()
