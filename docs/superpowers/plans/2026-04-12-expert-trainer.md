# Expert Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement memory-efficient expert training for Gemma4 MoE using view-based parameter isolation and KL-regularized router co-training.

**Architecture:** View-based training extracts expert weight slices as `nn.Parameter` views into the fused 3D tensor (no memory duplication). The router is co-trained with KL divergence regularization against the pretrained baseline to prevent routing collapse. A patched expert forward routes the target expert through the trainable view while all others use the frozen fused tensor.

**Tech Stack:** PyTorch, HuggingFace Transformers (Gemma4ForCausalLM), Gemma4TextConfig

---

## File Structure

```
src/exex/
├── __init__.py          # Create — package init, public API exports
├── surgery.py           # Create — model preparation: freeze, view params, forward patching
├── trainer.py           # Create — training loop: KL router loss, optimizer, train step
├── analyzer.py          # Existing — no changes
├── manager.py           # Modify — update for Gemma4 fused tensor layout
tests/
├── conftest.py          # Create — shared tiny model fixture
├── test_surgery.py      # Create — tests for view-based model surgery
├── test_trainer.py      # Create — tests for KL-regularized training
├── test_manager.py      # Create — tests for updated manager
scripts/
├── train_expert.py      # Create — CLI entry point for expert training
```

**Key design decisions:**
- `surgery.py` owns all model mutation: freezing, view creation, forward patching. Stateless functions.
- `trainer.py` owns the training loop and KL loss computation. Stateful `ExpertTrainer` class.
- Manager updated to work with Gemma4 fused tensors (3D `gate_up_proj`/`down_proj`) instead of `ModuleList`.
- All tests use a shared tiny Gemma4 fixture (4 experts, 2 layers, 64 hidden) to run fast on CPU.

---

### Task 1: Package setup and test fixture

**Files:**
- Create: `src/exex/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create package init**

```python
# src/exex/__init__.py
```

- [ ] **Step 2: Create shared test fixture**

```python
# tests/conftest.py
import pytest
import torch
from transformers import Gemma4ForCausalLM
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig


@pytest.fixture
def tiny_gemma4_moe():
    """Tiny Gemma4 MoE model for testing. 4 experts, 2 layers, runs on CPU."""
    config = Gemma4TextConfig(
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
    )
    model = Gemma4ForCausalLM(config)
    model.train()
    return model


@pytest.fixture(autouse=True)
def seed():
    """Deterministic RNG for reproducible routing decisions."""
    torch.manual_seed(42)


@pytest.fixture
def sample_batch():
    """Minimal input batch for forward passes."""
    return {
        "input_ids": torch.randint(0, 256, (2, 16)),
        "labels": torch.randint(0, 256, (2, 16)),
    }
```

- [ ] **Step 3: Verify fixture loads**

Run: `pytest tests/conftest.py --co -v`
Expected: Collects fixtures without error.

- [ ] **Step 4: Commit**

```bash
git add src/exex/__init__.py tests/conftest.py
git commit -m "feat: add package init and shared test fixture"
```

---

### Task 2: Model surgery — view-based expert preparation

**Files:**
- Create: `tests/test_surgery.py`
- Create: `src/exex/surgery.py`

- [ ] **Step 1: Write failing tests for surgery**

```python
# tests/test_surgery.py
import pytest
import torch
from exex.surgery import prepare_expert_for_training


class TestPrepareExpertForTraining:
    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    def test_view_shares_memory(self, tiny_gemma4_moe):
        """Trainable view params must share data_ptr with fused tensor."""
        model = tiny_gemma4_moe
        prepare_expert_for_training(model, target_expert_indices=[1])

        for layer in model.model.layers:
            if not hasattr(layer, "experts"):
                continue
            experts = layer.experts
            assert experts.gate_up_proj.data[1].data_ptr() == experts._train_gate_up_1.data.data_ptr()
            assert experts.down_proj.data[1].data_ptr() == experts._train_down_1.data.data_ptr()

    def test_only_target_expert_trainable(self, tiny_gemma4_moe):
        """Only target expert view params should have requires_grad=True."""
        model = tiny_gemma4_moe
        prepare_expert_for_training(model, target_expert_indices=[1])

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert all("_train_" in n for n in trainable)
        assert len(trainable) > 0

    def test_frozen_params_no_grad(self, tiny_gemma4_moe, sample_batch):
        """Frozen params must not accumulate gradients after backward."""
        model = tiny_gemma4_moe
        prepare_expert_for_training(model, target_expert_indices=[1])

        outputs = model(**sample_batch)
        outputs.loss.backward()

        frozen = [p for n, p in model.named_parameters() if not p.requires_grad]
        grads_on_frozen = sum(1 for p in frozen if p.grad is not None and p.grad.abs().sum() > 0)
        assert grads_on_frozen == 0

    def test_trainable_params_get_grad(self, tiny_gemma4_moe, sample_batch):
        """Trainable view params must receive nonzero gradients."""
        model = tiny_gemma4_moe
        prepare_expert_for_training(model, target_expert_indices=[1])

        outputs = model(**sample_batch)
        outputs.loss.backward()

        trainable = [p for n, p in model.named_parameters() if p.requires_grad]
        for p in trainable:
            assert p.grad is not None and p.grad.abs().sum() > 0

    def test_optimizer_updates_fused_tensor(self, tiny_gemma4_moe, sample_batch):
        """Optimizer step on view params must update the fused tensor via shared storage."""
        model = tiny_gemma4_moe
        target_idx = 1

        # Snapshot original
        original = {}
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                original[li] = layer.experts.gate_up_proj.data[target_idx].clone()

        prepare_expert_for_training(model, target_expert_indices=[target_idx])

        outputs = model(**sample_batch)
        outputs.loss.backward()

        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=0.01
        )
        optimizer.step()

        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                assert not torch.equal(
                    layer.experts.gate_up_proj.data[target_idx], original[li]
                ), f"Layer {li}: fused tensor not updated"

    def test_non_target_experts_unchanged(self, tiny_gemma4_moe, sample_batch):
        """Non-target expert weights must remain exactly unchanged after training step."""
        model = tiny_gemma4_moe
        target_idx = 1

        original = {}
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                original[li] = layer.experts.gate_up_proj.data.clone()

        prepare_expert_for_training(model, target_expert_indices=[target_idx])

        outputs = model(**sample_batch)
        outputs.loss.backward()

        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=0.01
        )
        optimizer.step()

        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                for eidx in range(4):
                    if eidx == target_idx:
                        continue
                    assert torch.equal(
                        layer.experts.gate_up_proj.data[eidx], original[li][eidx]
                    )

    def test_multiple_target_experts(self, tiny_gemma4_moe, sample_batch):
        """Support training multiple experts simultaneously."""
        model = tiny_gemma4_moe
        prepare_expert_for_training(model, target_expert_indices=[0, 2])

        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        # Should have view params for experts 0 and 2, across 2 layers, gate_up + down each
        assert len(trainable_names) == 8  # 2 experts * 2 layers * 2 projections

        outputs = model(**sample_batch)
        outputs.loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and p.grad.abs().sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_surgery.py -v`
Expected: ImportError — `exex.surgery` does not exist yet.

- [ ] **Step 3: Implement surgery.py**

```python
# src/exex/surgery.py
"""
Model surgery for view-based expert training on Gemma4 MoE.

Creates trainable nn.Parameter views into the fused 3D expert tensors,
sharing memory with the original (no duplication). Patches the expert
forward to route target experts through trainable views.
"""

import torch
import torch.nn as nn


def prepare_expert_for_training(model, target_expert_indices):
    """
    Freeze all model parameters, then create trainable view parameters
    for the specified expert indices. Patches expert forward methods.

    Args:
        model: Gemma4ForCausalLM model instance
        target_expert_indices: list of int, which expert slots to make trainable
    """
    if isinstance(target_expert_indices, int):
        target_expert_indices = [target_expert_indices]

    # Freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # For each MoE layer, create trainable views and patch forward
    for layer in model.model.layers:
        if not hasattr(layer, "experts"):
            continue

        experts = layer.experts

        for idx in target_expert_indices:
            # View into fused tensor — shares memory, no copy
            gate_up_view = nn.Parameter(experts.gate_up_proj.data[idx])
            down_view = nn.Parameter(experts.down_proj.data[idx])

            setattr(experts, f"_train_gate_up_{idx}", gate_up_view)
            setattr(experts, f"_train_down_{idx}", down_view)

        experts._train_indices = set(target_expert_indices)
        experts.forward = _make_patched_forward(experts, target_expert_indices)


def _make_patched_forward(experts_module, target_indices):
    """
    Create a patched forward that routes target experts through
    trainable view parameters, all others through the frozen fused tensor.
    """
    target_set = set(target_indices)
    frozen_gate_up = experts_module.gate_up_proj
    frozen_down = experts_module.down_proj
    act_fn = experts_module.act_fn
    num_experts = experts_module.num_experts

    # Collect references to trainable params
    train_params = {}
    for idx in target_indices:
        train_params[idx] = (
            getattr(experts_module, f"_train_gate_up_{idx}"),
            getattr(experts_module, f"_train_down_{idx}"),
        )

    def patched_forward(hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = nn.functional.one_hot(
                top_k_index, num_classes=num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx_t in expert_hit:
            expert_idx = expert_idx_t[0].item()
            if expert_idx >= num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if len(token_idx) == 0:
                continue

            current_state = hidden_states[token_idx]

            if expert_idx in target_set:
                gate_up_w, down_w = train_params[expert_idx]
            else:
                gate_up_w = frozen_gate_up[expert_idx]
                down_w = frozen_down[expert_idx]

            gate, up = nn.functional.linear(current_state, gate_up_w).chunk(2, dim=-1)
            current_hidden = act_fn(gate) * up
            current_hidden = nn.functional.linear(current_hidden, down_w)
            current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]

            final_hidden_states.index_add_(
                0, token_idx, current_hidden.to(final_hidden_states.dtype)
            )

        return final_hidden_states

    return patched_forward
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_surgery.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/exex/surgery.py tests/test_surgery.py
git commit -m "feat: view-based expert surgery for Gemma4 MoE"
```

---

### Task 3: KL-regularized expert trainer

**Files:**
- Create: `tests/test_trainer.py`
- Create: `src/exex/trainer.py`

- [ ] **Step 1: Write failing tests for trainer**

```python
# tests/test_trainer.py
import pytest
import torch
from exex.trainer import ExpertTrainer


class TestExpertTrainer:
    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    def test_init_snapshots_router(self, tiny_gemma4_moe):
        """Trainer must snapshot pretrained router state for KL reference."""
        trainer = ExpertTrainer(
            model=tiny_gemma4_moe,
            target_expert_indices=[1],
            kl_weight=0.1,
            lr=1e-3,
        )
        assert len(trainer._ref_router_params) > 0
        # Reference params should be detached clones
        for ref in trainer._ref_router_params:
            assert not ref["proj_weight"].requires_grad

    def test_model_prepared_after_init(self, tiny_gemma4_moe):
        """After init, model should have trainable view params and frozen base."""
        trainer = ExpertTrainer(
            model=tiny_gemma4_moe,
            target_expert_indices=[1],
            kl_weight=0.1,
            lr=1e-3,
        )
        trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
        assert len(trainable) > 0
        # Should include router params and expert view params
        has_router = any("router" in n for n in trainable)
        has_expert = any("_train_" in n for n in trainable)
        assert has_router
        assert has_expert

    def test_kl_loss_is_zero_before_training(self, tiny_gemma4_moe, sample_batch):
        """KL loss should be ~0 before any optimizer steps (router hasn't changed)."""
        trainer = ExpertTrainer(
            model=tiny_gemma4_moe,
            target_expert_indices=[1],
            kl_weight=1.0,
            lr=1e-3,
        )
        _, kl_loss = trainer.compute_loss(**sample_batch)
        assert kl_loss.item() < 1e-5

    def test_train_step_reduces_task_loss(self, tiny_gemma4_moe, sample_batch):
        """A training step should reduce the task loss."""
        trainer = ExpertTrainer(
            model=tiny_gemma4_moe,
            target_expert_indices=[1],
            kl_weight=0.1,
            lr=0.01,
        )
        task_loss_before, _ = trainer.compute_loss(**sample_batch)
        trainer.train_step(**sample_batch)
        task_loss_after, _ = trainer.compute_loss(**sample_batch)
        assert task_loss_after.item() < task_loss_before.item()

    def test_kl_loss_increases_after_training(self, tiny_gemma4_moe, sample_batch):
        """After training steps, KL loss should increase (router has diverged)."""
        trainer = ExpertTrainer(
            model=tiny_gemma4_moe,
            target_expert_indices=[1],
            kl_weight=0.01,  # low weight so router can move
            lr=0.01,
        )
        for _ in range(5):
            trainer.train_step(**sample_batch)
        _, kl_loss = trainer.compute_loss(**sample_batch)
        assert kl_loss.item() > 1e-6

    def test_high_kl_weight_constrains_router(self, tiny_gemma4_moe, sample_batch):
        """High KL weight should keep router close to pretrained baseline."""
        trainer_low = ExpertTrainer(
            model=tiny_gemma4_moe,
            target_expert_indices=[1],
            kl_weight=0.001,
            lr=0.01,
        )
        trainer_high = ExpertTrainer(
            # Need a fresh model for fair comparison
            model=type(tiny_gemma4_moe).from_config(tiny_gemma4_moe.config),
            target_expert_indices=[1],
            kl_weight=100.0,
            lr=0.01,
        )
        # Copy weights so both start identical
        trainer_high.model.load_state_dict(tiny_gemma4_moe.state_dict(), strict=False)

        for _ in range(10):
            trainer_low.train_step(**sample_batch)
            trainer_high.train_step(**sample_batch)

        _, kl_low = trainer_low.compute_loss(**sample_batch)
        _, kl_high = trainer_high.compute_loss(**sample_batch)
        # High KL weight should result in less router divergence
        assert kl_high.item() < kl_low.item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_trainer.py -v`
Expected: ImportError — `exex.trainer` does not exist yet.

- [ ] **Step 3: Implement trainer.py**

```python
# src/exex/trainer.py
"""
KL-regularized expert trainer for Gemma4 MoE.

Co-trains selected expert weights (via view-based surgery) and the router,
using KL divergence against the pretrained router as regularization to
prevent routing collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from exex.surgery import prepare_expert_for_training


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

        # Step 5: Build optimizer with param groups
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_trainer.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/exex/trainer.py tests/test_trainer.py
git commit -m "feat: KL-regularized expert trainer with router co-training"
```

---

### Task 4: Update manager for Gemma4 fused tensors

**Files:**
- Create: `tests/test_manager.py`
- Modify: `src/exex/manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_manager.py
import torch
from exex.manager import ExpertManager


class TestExpertManagerGemma4:
    def test_clone_expert_to_new_slot(self, tiny_gemma4_moe):
        """Cloning an expert should expand the fused tensor and router."""
        model = tiny_gemma4_moe
        manager = ExpertManager.from_model(model)

        original_num_experts = model.config.num_experts
        source_idx = 1

        manager.clone_expert(source_idx=source_idx)

        # Fused tensors should have grown by 1 in dim 0
        for layer in model.model.layers:
            if hasattr(layer, "experts"):
                assert layer.experts.gate_up_proj.shape[0] == original_num_experts + 1
                assert layer.experts.down_proj.shape[0] == original_num_experts + 1
                # New expert should equal source
                assert torch.equal(
                    layer.experts.gate_up_proj.data[-1],
                    layer.experts.gate_up_proj.data[source_idx],
                )

    def test_clone_expert_expands_router(self, tiny_gemma4_moe):
        """Cloning should add a new row to the router projection."""
        model = tiny_gemma4_moe
        manager = ExpertManager.from_model(model)

        original_num_experts = model.config.num_experts
        manager.clone_expert(source_idx=0)

        for layer in model.model.layers:
            if hasattr(layer, "router"):
                assert layer.router.proj.weight.shape[0] == original_num_experts + 1
                assert layer.router.per_expert_scale.shape[0] == original_num_experts + 1

    def test_remove_expert_shrinks_fused_tensor(self, tiny_gemma4_moe):
        """Removing an expert should shrink the fused tensor."""
        model = tiny_gemma4_moe
        manager = ExpertManager.from_model(model)

        original_num_experts = model.config.num_experts
        manager.remove_expert(expert_idx=2)

        for layer in model.model.layers:
            if hasattr(layer, "experts"):
                assert layer.experts.gate_up_proj.shape[0] == original_num_experts - 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_manager.py -v`
Expected: FAIL — `ExpertManager.from_model` and `clone_expert` don't exist yet.

- [ ] **Step 3: Update manager.py**

Modify `src/exex/manager.py` — add `from_model` classmethod and Gemma4-aware `clone_expert` and `remove_expert` methods. Key changes:

1. Add `from_model(cls, model)` classmethod that works with an already-loaded model (no re-loading from path).
2. Add `clone_expert(source_idx, label=None)` that expands fused tensors + router.
3. Replace `remove_expert` with Gemma4-compatible version using tensor slicing.

```python
import torch
import torch.nn as nn


class ExpertManager:
    def __init__(self, model_path=None):
        if model_path is not None:
            from transformers import AutoModelForCausalLM, AutoConfig
            self.config = AutoConfig.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="cpu"
            )
            self.layers = self.model.model.layers
        # Fields set by from_model() when model_path is None

    @classmethod
    def from_model(cls, model):
        """Create an ExpertManager from an already-loaded model."""
        instance = cls.__new__(cls)
        instance.model = model
        instance.config = model.config
        instance.layers = model.model.layers
        return instance

    def label_expert(self, expert_index, label):
        if not hasattr(self.config, "expert_labels"):
            self.config.expert_labels = {}
        self.config.expert_labels[str(expert_index)] = label

    def get_labels(self):
        return getattr(self.config, "expert_labels", {})

    def clone_expert(self, source_idx, label=None):
        """Clone an existing expert to a new slot at the end."""
        for layer in self.layers:
            if not hasattr(layer, "experts"):
                continue
            experts = layer.experts

            # Expand fused expert tensors
            source_gate_up = experts.gate_up_proj.data[source_idx:source_idx+1].clone()
            experts.gate_up_proj = nn.Parameter(
                torch.cat([experts.gate_up_proj.data, source_gate_up], dim=0)
            )
            source_down = experts.down_proj.data[source_idx:source_idx+1].clone()
            experts.down_proj = nn.Parameter(
                torch.cat([experts.down_proj.data, source_down], dim=0)
            )
            experts.num_experts += 1

            # Expand router
            if hasattr(layer, "router"):
                router = layer.router
                source_row = router.proj.weight.data[source_idx:source_idx+1].clone()
                new_weight = torch.cat([router.proj.weight.data, source_row], dim=0)
                router.proj = nn.Linear(new_weight.shape[1], new_weight.shape[0], bias=False)
                router.proj.weight = nn.Parameter(new_weight)

                source_scale = router.per_expert_scale.data[source_idx:source_idx+1].clone()
                router.per_expert_scale = nn.Parameter(
                    torch.cat([router.per_expert_scale.data, source_scale], dim=0)
                )

        if hasattr(self.config, "num_experts"):
            self.config.num_experts += 1
        new_idx = self.config.num_experts - 1

        if label:
            self.label_expert(new_idx, label)

        return new_idx

    def remove_expert(self, expert_idx, output_dir=None):
        """Remove an expert by slicing it out of the fused tensors and router."""
        for layer in self.layers:
            if not hasattr(layer, "experts"):
                continue
            experts = layer.experts

            # Remove from fused tensors
            experts.gate_up_proj = nn.Parameter(torch.cat([
                experts.gate_up_proj.data[:expert_idx],
                experts.gate_up_proj.data[expert_idx+1:],
            ], dim=0))
            experts.down_proj = nn.Parameter(torch.cat([
                experts.down_proj.data[:expert_idx],
                experts.down_proj.data[expert_idx+1:],
            ], dim=0))
            experts.num_experts -= 1

            # Shrink router
            if hasattr(layer, "router"):
                router = layer.router
                new_weight = torch.cat([
                    router.proj.weight.data[:expert_idx],
                    router.proj.weight.data[expert_idx+1:],
                ], dim=0)
                router.proj = nn.Linear(new_weight.shape[1], new_weight.shape[0], bias=False)
                router.proj.weight = nn.Parameter(new_weight)

                router.per_expert_scale = nn.Parameter(torch.cat([
                    router.per_expert_scale.data[:expert_idx],
                    router.per_expert_scale.data[expert_idx+1:],
                ], dim=0))

        if hasattr(self.config, "num_experts"):
            self.config.num_experts -= 1

        # Update labels (shift indices down)
        if hasattr(self.config, "expert_labels"):
            new_labels = {}
            for k, v in self.config.expert_labels.items():
                idx = int(k)
                if idx < expert_idx:
                    new_labels[str(idx)] = v
                elif idx > expert_idx:
                    new_labels[str(idx - 1)] = v
            self.config.expert_labels = new_labels

        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.config.save_pretrained(output_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_manager.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/exex/manager.py tests/test_manager.py
git commit -m "feat: update manager for Gemma4 fused tensor layout"
```

---

### Task 5: CLI training script

**Files:**
- Create: `scripts/train_expert.py`

- [ ] **Step 1: Write CLI script**

```python
#!/usr/bin/env python3
"""CLI for training domain-specific experts on Gemma4 MoE."""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from exex.trainer import ExpertTrainer
from exex.manager import ExpertManager


def main():
    parser = argparse.ArgumentParser(
        description="Train domain-specific experts on Gemma4 MoE"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        help="HF dataset name or local path")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--expert_indices", type=int, nargs="+", required=True,
                        help="Expert indices to train")
    parser.add_argument("--clone_from", type=int, default=None,
                        help="Clone this expert to a new slot before training")
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--router_lr_scale", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--load_in_4bit", action="store_true")

    args = parser.parse_args()

    # Load model
    load_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Optionally clone expert to new slot
    expert_indices = list(args.expert_indices)
    if args.clone_from is not None:
        manager = ExpertManager.from_model(model)
        new_idx = manager.clone_expert(source_idx=args.clone_from)
        expert_indices = [new_idx]
        print(f"Cloned expert {args.clone_from} -> new slot {new_idx}")

    # Create trainer
    trainer = ExpertTrainer(
        model=model,
        target_expert_indices=expert_indices,
        kl_weight=args.kl_weight,
        lr=args.lr,
        router_lr_scale=args.router_lr_scale,
    )

    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")

    # Training loop
    print(f"Training experts {expert_indices} for {args.max_steps} steps...")
    step = 0
    for epoch in range(100):  # enough epochs to reach max_steps
        for i in range(0, len(dataset), args.batch_size):
            if step >= args.max_steps:
                break

            batch_texts = [
                dataset[j][args.text_column]
                for j in range(i, min(i + args.batch_size, len(dataset)))
            ]
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                padding=True,
            ).to(model.device)

            labels = encodings.input_ids.clone()
            metrics = trainer.train_step(
                input_ids=encodings.input_ids, labels=labels
            )

            step += 1
            if step % args.log_every == 0:
                print(
                    f"Step {step}/{args.max_steps} | "
                    f"task_loss={metrics['task_loss']:.4f} | "
                    f"kl_loss={metrics['kl_loss']:.6f} | "
                    f"total_loss={metrics['total_loss']:.4f}"
                )

        if step >= args.max_steps:
            break

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses args**

Run: `PYTHONPATH=src python scripts/train_expert.py --help`
Expected: Prints usage with all arguments.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_expert.py
git commit -m "feat: CLI script for expert training"
```

---

### Task 6: Integration test — full pipeline

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
import torch
from exex.manager import ExpertManager
from exex.trainer import ExpertTrainer


class TestFullPipeline:
    def test_clone_then_train(self, tiny_gemma4_moe, sample_batch):
        """Full pipeline: clone expert to new slot, train it, verify coexistence."""
        model = tiny_gemma4_moe
        original_num_experts = model.config.num_experts
        source_idx = 1

        # Clone expert 1 to a new slot
        manager = ExpertManager.from_model(model)
        new_idx = manager.clone_expert(source_idx=source_idx)
        assert new_idx == original_num_experts  # appended at end

        # Snapshot source expert weights (should stay frozen)
        source_weights = {}
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                source_weights[li] = layer.experts.gate_up_proj.data[source_idx].clone()

        # Train the NEW expert (not the source)
        trainer = ExpertTrainer(
            model=model,
            target_expert_indices=[new_idx],
            kl_weight=0.1,
            lr=0.01,
        )

        for _ in range(5):
            trainer.train_step(**sample_batch)

        # Source expert should be unchanged
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                assert torch.equal(
                    layer.experts.gate_up_proj.data[source_idx],
                    source_weights[li],
                ), f"Layer {li}: source expert was modified!"

        # New expert should have diverged from source
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "experts"):
                assert not torch.equal(
                    layer.experts.gate_up_proj.data[new_idx],
                    source_weights[li],
                ), f"Layer {li}: new expert didn't change!"
```

- [ ] **Step 2: Run full test suite**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration test for clone-then-train pipeline"
```

---

### Task 7: Clean up smoke test

**Files:**
- Remove: `scripts/test_view_training.py` (superseded by proper test suite)

- [ ] **Step 1: Remove old smoke test**

```bash
git rm scripts/test_view_training.py
git commit -m "chore: remove smoke test superseded by test suite"
```
