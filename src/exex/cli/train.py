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
    p.add_argument("--top_k", type=int, default=None,
                   help="Train (and periodically eval) with this many active experts per "
                        "token instead of the checkpoint default; recorded in run.json")
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
    p.add_argument("--attention_mask", default="off", choices=["on", "off"],
                   help="Pass the padding attention mask to the model. Off by default: "
                        "with right padding the causal mask already keeps real tokens "
                        "correct, and Gemma 4 bf16 forwards with a mask produced "
                        "non-finite pad-row activations that poisoned expert grads "
                        "(26B, 2026-09-07). Pad labels are always -100 either way.")
    p.add_argument("--allow_nonfinite", action="store_true",
                   help="Skip non-finite steps instead of aborting the run")
    p.add_argument("--no_shuffle", action="store_true",
                   help="Iterate the dataset in file order instead of a seeded shuffle")
    # periodic eval
    # routing separation (batch-4 arm: centroid init + contrastive push-down)
    p.add_argument("--row_init", default="clone", choices=["clone", "centroid"],
                   help="With --clone_from: 'centroid' re-inits the new slot's "
                        "router row from the domain-token centroid instead of "
                        "keeping the (noised) copy of the source row")
    p.add_argument("--row_init_samples", type=int, default=64,
                   help="Domain texts used for the centroid estimate")
    p.add_argument("--row_warmup_steps", type=int, default=0,
                   help="Optimizer steps training only router rows before "
                        "expert weights unfreeze")
    p.add_argument("--neg_dataset", default=None,
                   help="Out-of-domain json(l); enables the contrastive "
                        "push-down of the trained slot's routing probability")
    p.add_argument("--neg_weight", type=float, default=1.0)
    p.add_argument("--neg_samples", type=int, default=2000,
                   help="Negative texts cycled during training")
    p.add_argument("--eval_dataset", default=None, help="Held-out json(l) for PPL")
    p.add_argument("--eval_every", type=int, default=0, help="Optimizer steps between evals")
    p.add_argument("--eval_samples", type=int, default=100)
    p.add_argument("--snapshot_every", type=int, default=0,
                   help="Optimizer steps between mid-training cartridge "
                        "snapshots under <output_dir>/snapshots/ (0=off); "
                        "lets multi-set PPL curves be evaluated post-hoc "
                        "without full-model saves")
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
    # Right-pad for training: with a causal mask, real tokens then never see a
    # pad token even when no attention mask is passed. Gemma 4 tokenizers
    # default to LEFT padding (generation-friendly), which silently let real
    # tokens attend to pads in every batch>1 run before 2026-09-07.
    if tokenizer.padding_side != "right":
        print(f"tokenizer.padding_side {tokenizer.padding_side!r} -> 'right' for training", flush=True)
        tokenizer.padding_side = "right"

    manager = ExpertManager.from_model(model)
    if args.top_k is not None:
        from exex.loading import set_top_k
        prev = set_top_k(model, args.top_k)
        print(f"Router top-k for training: {prev} -> {args.top_k}", flush=True)
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
        abort_on_nonfinite=not args.allow_nonfinite,
        row_warmup_steps=args.row_warmup_steps,
        contrastive_slot=expert_indices[0] if args.neg_dataset else None,
        contrastive_weight=args.neg_weight,
    )
    n_trainable = sum(p.numel() for p in trainer.trainable_parameters)

    neg_texts = None
    if args.neg_dataset:
        neg_texts = load_texts(args.neg_dataset, args.text_column, args.neg_samples)
        print(f"[contrastive] {len(neg_texts)} negative texts, "
              f"weight {args.neg_weight}, slot {expert_indices[0]}", flush=True)

    if args.row_init == "centroid":
        if args.clone_from is None:
            raise SystemExit("--row_init centroid requires --clone_from")
        from exex.separation import centroid_router_init

        def _tok_batches(texts):
            for t in texts:
                yield tokenizer(t, return_tensors="pt", truncation=True,
                                max_length=args.max_length)
        cen_texts = load_texts(args.dataset, args.text_column, args.row_init_samples)
        neg_cen = _tok_batches(neg_texts[:args.row_init_samples]) if neg_texts else None
        diags = [d for d in centroid_router_init(
            model, expert_indices[0], _tok_batches(cen_texts), neg_batches=neg_cen,
        ) if d]
        cos = [d["cosine_to_old"] for d in diags]
        print(f"[row_init] centroid-diff over {len(cen_texts)} pos"
              f"{' + neg' if neg_texts else ''} texts; cosine(new,old) "
              f"min/mean/max = {min(cos):.3f}/{sum(cos)/len(cos):.3f}/{max(cos):.3f}; "
              f"slot-vs-kth logit (layer0) = {diags[0]['slot_domain_logit']:.3f} "
              f"vs {diags[0]['kth_logit']:.3f}", flush=True)

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

    cart_names = {(args.label or f"expert_{idx}") if len(expert_indices) == 1
                  else f"{args.label or 'expert'}_{idx}": idx for idx in expert_indices}
    cart_labels = {n: [args.label] for n in cart_names} if args.label else None
    snapshots = []

    def save_snapshot(step):
        # Views alias the fused tensors, so mid-training extraction sees the
        # current weights without finalizing.
        snap_dir = os.path.join(args.output_dir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        path = os.path.join(snap_dir, f"cartridge_step{step}.safetensors")
        save_cartridge(model, cart_names, path,
                       source_model=args.model_path, labels=cart_labels)
        snapshots.append({"step": step, "path": path})
        print(f"[snapshot] step {step} -> {path}", flush=True)

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
            step_kwargs = {"input_ids": enc.input_ids, "labels": labels}
            if args.attention_mask == "on":
                step_kwargs["attention_mask"] = enc.attention_mask
            if neg_texts:
                lo = (step * args.batch_size) % len(neg_texts)
                neg = neg_texts[lo:lo + args.batch_size] or neg_texts[:args.batch_size]
                neg_enc = tokenizer(neg, return_tensors="pt", truncation=True,
                                    max_length=args.max_length,
                                    padding=True).to(model.device)
                step_kwargs["neg_input_ids"] = neg_enc.input_ids
                step_kwargs["neg_attention_mask"] = neg_enc.attention_mask
            m = trainer.train_step(**step_kwargs)
            tokens_seen += int(enc.attention_mask.sum())
            if not m["stepped"]:
                continue
            step += 1
            if step % args.log_every == 0 or step == args.max_steps:
                row = {"step": step, "epoch": epoch, "tokens_seen": tokens_seen,
                       "elapsed_s": round(time.time() - t0, 1),
                       **{k: m[k] for k in ("task_loss", "kl_loss", "total_loss",
                                            "contrastive_loss", "lr", "grad_norm")}}
                log(row)
                print(f"Step {step}/{args.max_steps} | task_loss={m['task_loss']:.4f} | "
                      f"kl_loss={m['kl_loss']:.6f} | lr={m['lr']:.2e}"
                      + (f" | gnorm={m['grad_norm']:.3f}" if m["grad_norm"] is not None else ""),
                      flush=True)
            if eval_texts and step % args.eval_every == 0 and step < args.max_steps:
                eval_history.append(run_eval(step))
            if args.snapshot_every and step % args.snapshot_every == 0 \
                    and step < args.max_steps:
                save_snapshot(step)
        epoch += 1
    if eval_texts:
        eval_history.append(run_eval(step))
    metrics_f.close()

    trainer.finalize()
    artefacts = {}
    if not args.no_cartridge:
        cart_path = os.path.join(args.output_dir, "cartridge.safetensors")
        save_cartridge(model, cart_names, cart_path,
                       source_model=args.model_path, labels=cart_labels)
        artefacts["cartridge"] = cart_path
        print(f"Wrote cartridge {cart_path} ({os.path.getsize(cart_path) / 1e6:.0f} MB): "
              f"{list(cart_names)}", flush=True)
    if args.save_full_model:
        print(f"Saving full model to {args.output_dir}...", flush=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        artefacts["model"] = args.output_dir
    else:
        # config alone is enough to know the resulting expert count / labels
        model.config.save_pretrained(args.output_dir)

    run = {"args": vars(args), "expert_indices": expert_indices,
           "num_experts": manager.arch.num_experts, "top_k": args.top_k,
           "trainable_params": n_trainable,
           "optimizer_steps": step, "tokens_seen": tokens_seen,
           "nonfinite_steps": trainer.nonfinite_steps,
           "elapsed_s": round(time.time() - t0, 1), "eval_ppl_history": eval_history,
           "snapshots": snapshots, "artefacts": artefacts}
    with open(os.path.join(args.output_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=2)
    print("Done.", flush=True)
    return run


if __name__ == "__main__":
    main()
