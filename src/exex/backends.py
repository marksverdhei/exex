"""
Pluggable training backends.

exex's surgery/expert-selection layer is backend-agnostic; a backend owns
the optimizer and the forward/backward/step machinery. The bare-PyTorch
backend is the always-working reference. Optimized backends (unsloth,
ht-unsloth, ...) register themselves here and are imported lazily, so a
missing dependency never breaks the core.
"""

import importlib

import torch

_REGISTRY = {}


def register_backend(name):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_backend(name="torch"):
    """Instantiate a backend by name. Raises with guidance if unavailable."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend {name!r} (available: {sorted(_REGISTRY)})"
        )
    return _REGISTRY[name]()


@register_backend("torch")
class TorchBackend:
    """Reference backend: plain PyTorch Adam + eager forward/backward."""

    def create_optimizer(self, param_groups):
        return torch.optim.Adam(param_groups)

    def backward_and_step(self, loss, optimizer):
        loss.backward()
        optimizer.step()


class _LazyImportBackend:
    """Base for backends wrapping an optional third-party package."""

    package = None  # override

    def __init__(self):
        try:
            self._module = importlib.import_module(self.package)
        except ImportError as e:
            raise ImportError(
                f"Backend requires the {self.package!r} package, which is "
                f"not installed. Install it or use backend='torch'."
            ) from e
        self._check_support()

    def _check_support(self):
        raise NotImplementedError


@register_backend("unsloth")
class UnslothBackend(_LazyImportBackend):
    """Placeholder adapter: fails informatively until expert-subset training
    lands upstream. Tracked in exex issue #6."""

    package = "unsloth"

    def _check_support(self):
        raise NotImplementedError(
            "unsloth does not yet expose MoE expert-subset training; "
            "this adapter is a stub. Use backend='torch'."
        )
