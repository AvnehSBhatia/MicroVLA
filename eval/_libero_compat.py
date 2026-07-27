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
import pathlib


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
    _seed_libero_config(os.environ["LIBERO_CONFIG_PATH"])


def _libero_root() -> "pathlib.Path | None":
    """Directory holding LIBERO's bddl_files/init_files/assets.

    Prefers a source checkout, because the PyPI wheel ships WITHOUT `assets/`
    and its HuggingFace fallback repo returns 401 — env construction then dies
    on a missing scene XML. $LIBERO_ROOT wins; then a sibling checkout; then the
    installed package, which is correct on machines where assets are present.
    """
    import os as _os

    cands = []
    if _os.environ.get("LIBERO_ROOT"):
        cands.append(pathlib.Path(_os.environ["LIBERO_ROOT"]) / "libero" / "libero")
    here = pathlib.Path(__file__).resolve().parent.parent
    cands += [here / ".libero_src" / "libero" / "libero",
              pathlib.Path("/root/LIBERO/libero/libero")]
    for c in cands:
        if (c / "assets").is_dir() and (c / "bddl_files").is_dir():
            return c
    try:
        import importlib.util

        spec = importlib.util.find_spec("libero.libero")
        if spec and spec.origin:
            return pathlib.Path(spec.origin).parent
    except Exception:
        pass
    return None


def _seed_libero_config(config_path: str) -> None:
    """Write LIBERO's config.yaml if absent, so no import ever prompts.

    Setting ``LIBERO_CONFIG_PATH`` alone was not enough. LIBERO's package
    ``__init__`` calls ``input()`` when the config file is missing, so pointing
    at an empty directory guaranteed an interactive prompt — and every eval run
    is headless, where that is an immediate ``EOFError`` before a single env is
    built. The docstring's "non-interactive after the first" had no reachable
    first: the prompt cannot be answered in the environments this runs in.

    Paths are derived from the INSTALLED libero package rather than hardcoded,
    so this works on any machine without a checked-out LIBERO tree.
    """
    cfg_dir = pathlib.Path(config_path)
    cfg_file = cfg_dir / "config.yaml"
    if cfg_file.exists():
        return
    root = _libero_root()
    if root is None:
        return                          # libero not installed; mock paths only

    import yaml

    cfg_dir.mkdir(parents=True, exist_ok=True)
    # `datasets` points at the raw HDF5 demos. Only needed for dataset loading,
    # not for stepping the simulator, so a missing directory is harmless here —
    # bddl_files and init_states are what env construction actually reads.
    datasets = os.environ.get("LIBERO_DATASETS", str(root.parent.parent / "datasets"))
    cfg_file.write_text(yaml.dump({
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": datasets,
        "assets": str(root / "assets"),
    }))
