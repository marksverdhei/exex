#!/usr/bin/env python3
"""Thin shim; the implementation lives in ``exex.cli.report`` (``exex-report``)."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from exex.cli.report import main  # noqa: E402

if __name__ == "__main__":
    main()
