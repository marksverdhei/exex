"""
KL-regularized expert trainer for Gemma4 MoE.

Co-trains selected expert weights (via view-based surgery) and the router,
using KL divergence against the pretrained router as regularization to
prevent routing collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Gemma4ForCausalLM

from exex.surgery import prepare_expert_for_training

# Patch from_config onto Gemma4ForCausalLM if not present (uses _from_config internally)
if not hasattr(Gemma4ForCausalLM, "from_config"):
    Gemma4ForCausalLM.from_config = classmethod(
        lambda cls, config, **kwargs: cls._from_config(config, **kwargs)
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
    """

    def __init__(
        self,
        model,
        target_expert_indices,
        kl_weight=0.1,
        lr=1e-4,
        router_lr_scale=0.1,
    ):
        self.model = model
        self.target_expert_indices = (
            [target_expert_indices]
            if isinstance(target_expert_indices, int)
            else target_expert_indices
        )
        self.kl_weight = kl_weight

        # Step 1: Snapshot pretrained router for KL reference (before any freezing)
        self._ref_router_params = self._snapshot_routers()

        # Step 2: Prepare model (freeze all, create expert views, patch forward)
        prepare_expert_for_training(model, self.target_expert_indices)

        # Step 3: Unfreeze router parameters
        self._unfreeze_routers()

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
            if p.requires_grad and "_train_" in n
        ]
        router_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "router" in n
        ]
        self.optimizer = torch.optim.Adam([
            {"params": expert_params, "lr": lr},
            {"params": router_params, "lr": lr * router_lr_scale},
        ])

    def _snapshot_routers(self):
        """Clone router parameters as frozen reference for KL computation."""
        refs = []
        for layer in self.model.model.layers:
            if not hasattr(layer, "router"):
                continue
            refs.append({
                "proj_weight": layer.router.proj.weight.data.clone().detach(),
                "scale": layer.router.scale.data.clone().detach(),
                "scalar_root_size": layer.router.scalar_root_size,
            })
        return refs

    def _unfreeze_routers(self):
        """Unfreeze all router parameters for co-training."""
        for layer in self.model.model.layers:
            if not hasattr(layer, "router"):
                continue
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

        router_idx = 0
        for layer in self.model.model.layers:
            if not hasattr(layer, "router"):
                continue
            idx = router_idx  # capture for closure

            def hook_fn(module, args, output, _idx=idx):
                # args[0] is the hidden_states input to the router
                self._router_inputs[_idx] = args[0].detach()

            handle = layer.router.register_forward_hook(hook_fn)
            self._hooks.append(handle)
            router_idx += 1

    def _compute_kl_loss(self):
        """
        Compute KL divergence between current and pretrained router distributions.

        Uses captured router inputs (from forward hooks) and frozen parameter
        snapshot to compute reference logits, then KL(current || ref).
        """
        device = next(self.model.parameters()).device
        total_kl = torch.tensor(0.0, device=device)

        router_idx = 0
        for layer, ref in zip(self.model.model.layers, self._ref_router_params):
            if not hasattr(layer, "router"):
                continue

            if router_idx not in self._router_inputs:
                router_idx += 1
                continue

            # Get the actual input the router received during this forward pass
            hs_flat = self._router_inputs[router_idx]

            router = layer.router

            # Current router logits (recompute — these are in the grad graph)
            normed = router.norm(hs_flat)
            scaled = normed * router.scale * router.scalar_root_size
            current_logits = router.proj(scaled)

            # Reference router logits (using frozen snapshot, no grad)
            with torch.no_grad():
                ref_logits = F.linear(
                    normed * ref["scale"].to(device) * ref["scalar_root_size"],
                    ref["proj_weight"].to(device),
                )

            current_log_probs = F.log_softmax(current_logits, dim=-1)
            ref_probs = F.softmax(ref_logits, dim=-1)

            kl = F.kl_div(current_log_probs, ref_probs, reduction="batchmean")
            total_kl = total_kl + kl
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

    def train_step(self, input_ids, labels, **kwargs):
        """
        Single training step: forward, backward, optimizer step.

        Returns:
            dict with task_loss, kl_loss, total_loss
        """
        self.model.train()
        self.optimizer.zero_grad()

        task_loss, kl_loss = self.compute_loss(input_ids, labels, **kwargs)
        total_loss = task_loss + self.kl_weight * kl_loss

        total_loss.backward()
        self.optimizer.step()

        return {
            "task_loss": task_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
        }
