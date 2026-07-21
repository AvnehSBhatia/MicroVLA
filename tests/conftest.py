"""Pytest configuration: make ``import microvla`` work without installation.

Adds the repository root (parent of this ``tests/`` directory) to ``sys.path``
so the test suite runs against the in-tree package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
