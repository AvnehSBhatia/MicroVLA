"""Compare the conditions a corpus was baked under against how the robot is run.

Of the 26 defects in ``paper.md``, four are the same sentence with a different
noun: *the deployment used a different <camera / detector threshold / render
size / perception period> than the corpus was built with*. Each cost days,
because none of them raises anything — the policy runs, the trainer's metrics
stay healthy, and closed-loop success is 0.000 for a reason the telemetry does
not name.

They share a root cause that is not a bug in any single file: the ``.npz``
corpus did not record what produced it, so no consumer could check. ``run_conversion``
now writes a ``provenance`` block into ``manifest.json``, and this module is the
consumer side — one function, called by the eval harness, that turns "silently
mismatched" into a line of output.

Pure stdlib + numpy-free: this is on the deployment path.
"""
from __future__ import annotations

import json
import os
from typing import Any


#: Deployment knob -> provenance key it must agree with. Only knobs that change
#: what the FROZEN perception front-end emits belong here; anything the policy
#: merely reads (batch size, workers, seeds) does not.
_CHECKS = {
    "camera": "eval_camera",
    "det_conf": "det_conf",
    "render_size": "detect_frame_hw",
    "perception_period": "_stride",
}


def load(corpus_or_manifest: str | os.PathLike) -> dict[str, Any]:
    """Reads a corpus's ``provenance`` block.

    Args:
        corpus_or_manifest: A corpus directory, its ``manifest.json``, or the
            ``norm_stats.json`` beside it — eval knows the last of these, so
            accepting all three keeps the call site honest instead of making
            it reconstruct paths.

    Returns:
        The provenance dict, or ``{}`` when the corpus predates provenance
        (every corpus baked before 2026-07-30) or cannot be read. Absence is
        NOT an error: it means "unknown", and an unknown cannot be checked.
    """
    p = str(corpus_or_manifest)
    if p.endswith(".json"):
        p = os.path.join(os.path.dirname(p), "manifest.json")
    else:
        p = os.path.join(p, "manifest.json")
    try:
        with open(p) as f:
            return dict(json.load(f).get("provenance") or {})
    except (OSError, ValueError):
        return {}


def mismatches(prov: dict[str, Any], **deployment: Any) -> list[str]:
    """Human-readable descriptions of every disagreement found.

    Args:
        prov: The corpus provenance, from :func:`load`. Empty means unknown;
            an empty result is returned rather than a false all-clear.
        **deployment: Any of ``camera``, ``det_conf``, ``render_size``,
            ``perception_period``. Pass only what the caller actually knows;
            ``None`` values are skipped.

    Returns:
        One string per mismatch, empty when everything agrees or nothing is
        knowable. Each string names both values, because "mismatch" without
        the numbers is not actionable at 3am.
    """
    if not prov:
        return []
    out: list[str] = []
    for knob, key in _CHECKS.items():
        have = deployment.get(knob)
        if have is None:
            continue
        if knob == "perception_period":
            # The corpus records its sampling rate, not a tick count. A 10 Hz
            # corpus of a 20 Hz source has stride 2 -- NOT the Pi loop's
            # 30/2 = 15 default, which is what paper.md 5d was measured at.
            hz = prov.get("real_frame_hz")
            src = prov.get("source_hz", 20.0)
            if not hz:
                continue
            want = max(1, int(round(float(src) / float(hz))))
            if int(have) != want:
                out.append(
                    f"perception_period={have} but the corpus samples "
                    f"{hz:g} Hz from {float(src):g} Hz, i.e. stride {want}"
                )
            continue
        want = prov.get(key)
        if want is None:
            continue
        if knob == "render_size":
            # detect_frame_hw is [H, W] of the pixels the detector actually saw.
            try:
                h, w = int(want[0]), int(want[1])
            except (TypeError, ValueError, IndexError):
                continue
            if int(have) != h or int(have) != w:
                out.append(
                    f"render_size={have} but the corpus's detector frames were "
                    f"{h}x{w}; the two upscale differently into the detector's "
                    f"512-px short side"
                )
            continue
        if knob == "det_conf":
            if abs(float(have) - float(want)) > 1e-9:
                out.append(f"det_conf={have} but the corpus was baked at {want}")
            continue
        if str(have) != str(want):
            out.append(f"{knob}={have!r} but the corpus was baked for {want!r}")
    return out


def describe(prov: dict[str, Any]) -> str:
    """One-line summary of a provenance block, for logs and results JSON."""
    if not prov:
        return "corpus provenance: (absent — baked before 2026-07-30)"
    keys = ("camera", "eval_camera", "det_conf", "real_frame_hz",
            "detect_frame_hw", "deflip", "max_objects", "grid_size")
    parts = [f"{k}={prov[k]}" for k in keys if k in prov]
    return "corpus provenance: " + ", ".join(parts)
