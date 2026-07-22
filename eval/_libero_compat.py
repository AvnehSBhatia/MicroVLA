"""Compatibility shims for running LIBERO under modern torch / headless boxes.

Call :func:`prepare_libero` BEFORE importing anything from ``libero`` (both
``eval/env_smoke.py`` and ``eval/libero_eval.py``'s real path do). It fixes two
frictions that have nothing to do with MicroVLA:

1. ``torch>=2.6`` flipped ``torch.load(weights_only=...)`` to default ``True``,
   which rejects LIBERO's numpy-pickle data files (init states, etc.). LIBERO
   is a trusted source, so we default those (flag-less) loads back to
   ``weights_only=False``. Loads that pass the flag explicitly (e.g. MicroVLA's
   own checkpoint loads, which pass ``weights_only=True``) are untouched.
2. LIBERO writes a config on first import and prompts interactively for a
   dataset path; pointing ``LIBERO_CONFIG_PATH`` at a temp dir keeps runs
   non-interactive after the first.
"""

from __future__ import annotations

import os


def prepare_libero(config_path: str = "/tmp/libero_home") -> None:
    """Patch torch.load default + set LIBERO config path. Idempotent."""
    import torch

    if not getattr(torch.load, "_libero_patched", False):
        _orig = torch.load

        def _load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig(*args, **kwargs)

        _load._libero_patched = True
        torch.load = _load

    os.environ.setdefault("LIBERO_CONFIG_PATH", config_path)
