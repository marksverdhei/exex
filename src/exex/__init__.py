__version__ = "0.1.0"

from exex.analyzer import RoutingAnalyzer
from exex.arch import MoEArch, iter_moe_layers
from exex.backends import get_backend, register_backend
from exex.cartridge import Cartridge, extract_cartridge, load_cartridge, save_cartridge
from exex.evaluate import load_texts, perplexity
from exex.loading import install_cartridges, load_model, parse_cartridge_spec, set_top_k
from exex.manager import ExpertManager
from exex.merger import apply_merge_config, install_expert
from exex.pruner import (
    STRATEGIES,
    RouterStats,
    collect_router_stats,
    prune_experts,
    score_experts,
    select_prune_candidates,
)
from exex.surgery import prepare_expert_for_training
from exex.trainer import ExpertTrainer

__all__ = [
    "__version__",
    # arch / backends
    "MoEArch",
    "iter_moe_layers",
    "get_backend",
    "register_backend",
    # experts lifecycle
    "ExpertManager",
    "prepare_expert_for_training",
    "ExpertTrainer",
    # cartridges
    "Cartridge",
    "extract_cartridge",
    "save_cartridge",
    "load_cartridge",
    # merging
    "install_expert",
    "apply_merge_config",
    # pruning
    "STRATEGIES",
    "RouterStats",
    "collect_router_stats",
    "score_experts",
    "select_prune_candidates",
    "prune_experts",
    # loading / eval
    "load_model",
    "install_cartridges",
    "parse_cartridge_spec",
    "set_top_k",
    "perplexity",
    "load_texts",
    # analysis
    "RoutingAnalyzer",
]
