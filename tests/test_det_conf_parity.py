"""The bake and the robot must threshold the detector identically (defect 26).

``det_conf`` decides which boxes exist at all, and every surviving box carries
its confidence into fusion's ``box_weight`` fade. A threshold that differs
between the corpus and the robot therefore does two things at once: it changes
which objects the policy can see, and it puts evidence-weight mass at
deployment exactly where training had none. The bake was taking
``YoloWorldPerception``'s class default (0.10) while ``eval/policy.py`` passed
0.02 — and the asymmetry was written down in a docstring rather than
reconciled.

These tests read the SOURCE, because the real detector needs ultralytics and a
GPU and the tests must stay CPU-only, mock-only, no network.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from microvla.config import DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parents[1]


def _default_of(func, name):
    sig = inspect.signature(func)
    return sig.parameters[name].default


def test_config_owns_the_threshold():
    assert 0.0 < DEFAULT_CONFIG.det_conf < 1.0


def test_eval_policy_defaults_to_the_config_threshold():
    from eval.policy import MicroVLAPolicy, _build_real_perception

    assert _default_of(_build_real_perception, "det_conf") == DEFAULT_CONFIG.det_conf
    assert _default_of(MicroVLAPolicy.__init__, "det_conf") == DEFAULT_CONFIG.det_conf


def _constructor_kwargs(path: Path, callee: str) -> list[set[str]]:
    """Keyword names passed to every ``callee(...)`` call in a source file."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = getattr(f, "id", None) or getattr(f, "attr", None)
            if name == callee:
                out.append({kw.arg for kw in node.keywords if kw.arg})
    return out


def test_every_real_detector_construction_passes_det_conf():
    """No construction site may fall back to the detector's class default.

    Three sites build the real detector — the bake, ``JEPALoop.build_real``,
    and ``eval/policy.py``. Each one that omits ``det_conf`` silently
    reintroduces defect 26.
    """
    sites = {
        "preprocess/common.py": None,
        "microvla/jepa/loop.py": None,
        "eval/policy.py": None,
    }
    for rel in sites:
        calls = _constructor_kwargs(ROOT / rel, "YoloWorldPerception")
        assert calls, f"{rel}: expected a YoloWorldPerception construction"
        for kwargs in calls:
            assert "det_conf" in kwargs, (
                f"{rel} builds YoloWorldPerception without det_conf; it would "
                f"take the class default instead of cfg.det_conf"
            )


def test_bake_records_its_thresholds_in_the_manifest():
    """A corpus must say what it was built with, or a mismatch is unfindable."""
    src = (ROOT / "preprocess/common.py").read_text()
    assert '"provenance"' in src
    for field in ("det_conf", "real_frame_hz", "max_objects"):
        assert f'"{field}"' in src, f"provenance should record {field}"


def test_libero_converter_records_the_camera_pairing():
    src = (ROOT / "preprocess/libero.py").read_text()
    assert '"eval_camera"' in src, (
        "the corpus must name the live camera key a deployment has to use"
    )


def test_two_view_bake_is_refused_not_silently_wrong():
    """``--detect-camera`` must not produce a corpus the guard calls clean.

    ``preprocess/common.py`` makes ONE ``perceive()`` call per frame, on the
    DETECT view, so ``frame_embs`` come from there too — the opposite of the
    flag's documented contract. Worse, ``manifest.json``'s ``eval_camera`` is
    derived from ``--camera``, the view that never reached the encoder, so the
    provenance check of ``test_provenance.py`` would report zero mismatches for
    a corpus whose every latent is from the other camera.
    """
    import pytest

    from preprocess.libero import main

    with pytest.raises(SystemExit) as e:
        main(["/nonexistent", "/tmp/out", "--camera", "eye_in_hand_rgb",
              "--detect-camera", "agentview_rgb"])
    assert "detect-camera" in str(e.value)
