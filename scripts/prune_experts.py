#!/usr/bin/env python3
"""Thin shim; the implementation lives in ``exex.cli.prune`` (``exex-prune``)."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from exex.cli.prune import main  # noqa: E402

if __name__ == "__main__":
    main()
