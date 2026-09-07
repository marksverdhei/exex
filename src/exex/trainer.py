"""
KL-regularized expert trainer for Gemma4 MoE.

Co-trains selected expert weights (via view-based surgery) and the router,
using KL divergence against the pretrained router as regularization to
prevent routing collapse.
"""

import torch
import torch.nn.functional as F

from exex.arch import iter_moe_layers
from exex.backends import get_backend
from exex.surgery import (
    composed_router_params,
    finalize_expert_training,
    prepare_expert_for_training,
    prepare_router_rows_for_training,
)


class ExpertTrainer:
    """
    Trains selected MoE experts with KL-regularized router co-training.

    Args:
        model: Gemma4ForCausalLM instance
        target_expert_indices: list of expert indices to train
        kl_weight: weight for KL divergence regularization on the router
        lr: learning rate
        router_lr_scale: router learning rate = lr * router_lr_scale
        train_full_router: if False (default) only the target experts' router
            rows and per-expert scales train, so a cartridge reproduces the
            trained model exactly. If True every router parameter trains
            (shared ``scale``, all rows); cartridges then become approximate.
        grad_accum_steps: micro-batches per optimizer step
        max_grad_norm: clip total grad norm before each step (None = off)
        warmup_steps: linear LR warmup over this many optimizer steps
        total_steps: optimizer steps for the decay schedule (None = constant)
        lr_decay: "none" | "linear" | "cosine" after warmup
        weight_decay: AdamW weight decay (0 keeps frozen-row semantics exact)
    """

    def __init__(
        self,
        model,
        target_expert_indices,
        kl_weight=0.1,
        lr=1e-4,
        router_lr_scale=0.1,
        backend="torch",
        train_full_router=False,
        grad_accum_steps=1,
        max_grad_norm=None,
        warmup_steps=0,
        total_steps=None,
        lr_decay="none",
        weight_decay=0.0,
    ):
        self.backend = get_backend(backend) if isinstance(backend, str) else backend
        self.model = model
        self.target_expert_indices = (
            [target_expert_indices]
            if isinstance(target_expert_indices, int)
            else target_expert_indices
        )
        self.kl_weight = kl_weight
        self.train_full_router = train_full_router

        # Step 1: Snapshot pretrained router for KL reference (before any freezing)
        self._ref_router_params = self._snapshot_routers()

        # Step 2: Prepare model (freeze all, create expert views, patch forward)
        prepare_expert_for_training(model, self.target_expert_indices)

        # Step 3: Make router trainable — target rows only, or everything
        if train_full_router:
            self._unfreeze_routers()
        else:
            prepare_router_rows_for_training(model, self.target_expert_indices)

        # Step 4: Install forward hooks to capture router inputs for KL
        self._install_router_hooks()

        # Step 5: Patch model.load_state_dict so reference snapshot stays in sync
        # when external code loads weights after trainer construction.
        _orig_load = model.load_state_dict

        def _patched_load_state_dict(state_dict, strict=True, **kwargs):
            result = _orig_load(state_dict, strict=strict, **kwargs)
            self._ref_router_params = self._snapshot_routers()
            return result

        model.load_state_dict = _patched_load_state_dict

        # Step 6: Build optimizer with param groups
        expert_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "_train_" in n and "router" not in n
        ]
        router_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "router" in n
        ]
        self.optimizer = self.backend.create_optimizer([
            {"params": expert_params, "lr": lr},
            {"params": router_params, "lr": lr * router_lr_scale},
        ], weight_decay=weight_decay)
        self.grad_accum_steps = max(int(grad_accum_steps), 1)
        self.max_grad_norm = max_grad_norm
        self._micro_step = 0
        self.optimizer_steps = 0
        self.scheduler = self._build_scheduler(warmup_steps, total_steps, lr_decay)

    def _build_scheduler(self, warmup_steps, total_steps, lr_decay):
        if warmup_steps <= 0 and lr_decay == "none":
            return None
        import math

        def factor(step):
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            if lr_decay == "none" or not total_steps:
                return 1.0
            progress = min(
                max(step - warmup_steps, 0) / max(total_steps - warmup_steps, 1), 1.0
            )
            if lr_decay == "linear":
                return max(1.0 - progress, 0.0)
            if lr_decay == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            raise ValueError(f"Unknown lr_decay {lr_decay!r}")

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)

    @property
    def trainable_parameters(self):
        return [p for g in self.optimizer.param_groups for p in g["params"]]

    @property
    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _snapshot_routers(self):
        """Clone router parameters as frozen reference for KL computation."""
        refs = []
        for _, layer in iter_moe_layers(self.model):
            refs.append({
                "proj_weight": layer.router.proj.weight.data.clone().detach(),
                "scale": layer.router.scale.data.clone().detach(),
                "scalar_root_size": layer.router.scalar_root_size,
            })
        return refs

    def _unfreeze_routers(self):
        """Unfreeze all router parameters for co-training."""
        for _, layer in iter_moe_layers(self.model):
            for param in layer.router.parameters():
                param.requires_grad_(True)

    def _install_router_hooks(self):
        """
        Install forward hooks on each router to capture the actual input
        hidden states. This avoids the mismatch between output_hidden_states
        (post-layer) and the router's actual input (pre-MoE residual).
        """
        self._router_inputs = {}
        self._hooks = []

        for idx, (_, layer) in enumerate(iter_moe_layers(self.model)):

            def hook_fn(module, args, output, _idx=idx):
                # args[0] is the hidden_states input to the router
                self._router_inputs[_idx] = args[0].detach()

            handle = layer.router.register_forward_hook(hook_fn)
            self._hooks.append(handle)

    def _compute_kl_loss(self):
        """
        Compute KL divergence between current and pretrained router distributions.

        Uses captured router inputs (from forward hooks) and frozen parameter
        snapshot to compute reference logits, then KL(current || ref).
        """
        device = next(self.model.parameters()).device
        total_kl = torch.tensor(0.0, device=device)

        router_idx = 0
        moe_layers = [layer for _, layer in iter_moe_layers(self.model)]
        for layer, ref in zip(moe_layers, self._ref_router_params, strict=True):

            if router_idx not in self._router_inputs:
                router_idx += 1
                continue

            # Get the actual input the router received during this forward pass
            hs_flat = self._router_inputs[router_idx]

            router = layer.router

            # Current router logits (recompute — these are in the grad graph)
            weight, _ = composed_router_params(router)
            normed = router.norm(hs_flat)
            scaled = normed * router.scale * router.scalar_root_size
            current_logits = F.linear(scaled, weight)

            # Reference router logits (using frozen snapshot, no grad)
            with torch.no_grad():
                ref_logits = F.linear(
                    normed * ref["scale"].to(device) * ref["scalar_root_size"],
                    ref["proj_weight"].to(device),
                )

            # fp32 for the divergence itself: at KL ~ 0 (early training),
            # bf16/fp16 rounding yields small negative values
            current_log_probs = F.log_softmax(current_logits.float(), dim=-1)
            ref_probs = F.softmax(ref_logits.float(), dim=-1)

            kl = F.kl_div(current_log_probs, ref_probs, reduction="batchmean")
            total_kl = total_kl + kl.clamp_min(0.0)
            router_idx += 1

        return total_kl / max(len(self._ref_router_params), 1)

    def compute_loss(self, input_ids, labels, **kwargs):
        """
        Compute task loss and KL regularization loss.

        Router inputs are captured via forward hooks installed during __init__,
        ensuring we use the exact hidden states the router actually received.

        Returns:
            task_loss: cross-entropy language modeling loss
            kl_loss: KL divergence between current and pretrained router
        """
        self._router_inputs = {}  # clear from previous call

        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            **kwargs,
        )
        task_loss = outputs.loss
        kl_loss = self._compute_kl_loss()

        return task_loss, kl_loss

    def finalize(self):
        """
        End training: remove router hooks and the aliased view parameters.

        Trained weights live in the fused tensors (views share storage), so
        call this before save_pretrained — safetensors refuses aliased params.
        """
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._router_inputs = {}
        finalize_expert_training(self.model)

    def train_step(self, input_ids, labels, **kwargs):
        """
        One micro-step: forward + backward; optimizer step every
        ``grad_accum_steps`` micro-steps (with clipping and LR schedule).

        Returns:
            dict with task_loss, kl_loss, total_loss, lr, stepped
        """
        self.model.train()
        if self._micro_step % self.grad_accum_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        task_loss, kl_loss = self.compute_loss(input_ids, labels, **kwargs)
        total_loss = task_loss + self.kl_weight * kl_loss
        self.backend.backward(total_loss / self.grad_accum_steps)
        self._micro_step += 1

        stepped = self._micro_step % self.grad_accum_steps == 0
        grad_norm = None
        if stepped:
            if self.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.trainable_parameters, self.max_grad_norm
                ).item()
            self.backend.step(self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer_steps += 1

        return {
            "task_loss": task_loss.detach().item(),
            "kl_loss": kl_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
            "lr": self.current_lr,
            "grad_norm": grad_norm,
            "stepped": stepped,
        }
