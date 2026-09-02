# exex — feat/core-v0 work list

- [x] 1. Branch `feat/core-v0`; merge analyzer from `feat/expert-trainer`; drop generated artifacts + plan docs (#9)
- [x] 2. Dev env: uv venv with CPU torch + transformers (Gemma4) + safetensors + pytest
- [x] 3. Scaffolding: `pyproject.toml` (src layout, extras), `.gitignore`, light CPU-only CI workflow
- [x] 4. `arch.py`: config-driven architecture descriptor (num experts, top-k, shared experts, moe dims) — no hardcoded 128/8 (#4)
- [x] 5. Refactor `manager.py` / `surgery.py` / `trainer.py` / `analyzer.py` onto the descriptor (#4)
- [x] 6. `cartridge.py`: expert cartridge v0 — one safetensors file, multi-expert, `__metadata__` header (manifest, source model, version, config fingerprint)
- [x] 7. `merger.py` + `scripts/merge_experts.py`: transplant from cartridge/checkpoint, alpha-blend, JSON batch config (#7)
- [x] 8. `pruner.py` + `scripts/prune_experts.py`: utilisation + router-gate-weighted activation-norm criteria, remove or zero modes (#8)
- [x] 9. Tests for arch/cartridge/merger/pruner on tiny fixture; full suite green
- [x] 10. README rewrite to match reality; full-rank view training as default story (#3, #11)
- [ ] 11. Push branch, open PR, comment on affected issues

Deferred (blocked or out of scope for this branch):
- #10 end-to-end 26B validation — needs real GPU (batched for Markus)
- #6 backend adapters (unsloth/ht-unsloth) — only the seam exists after refactor
- #5 deeper memory work (expert-only recompute, VRAM bound docs)
