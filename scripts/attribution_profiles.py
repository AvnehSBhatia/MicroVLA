"""Substitution attribution for the grasp heads --- and the probe-only counterexample.

Regenerates the figure the manuscript previously shipped without a generator, a
reproducibility gap its own caption admitted. The measurement is a substitution
probe: take the two most-separated recorded ticks, swap ONE input channel from
one into the other, and record how far the predicted grasp point moves. A head
that reads a channel moves when that channel is swapped; a head that ignores it
does not.

The point of running it over several heads at once is the counterexample the
paper leans on: ``v2`` and ``v2.1`` are trained the same way on the same
10-episode corpus and produce near-identical attribution profiles, yet score
0.000 and 0.700 deployed. If that holds when measured rather than remembered,
then an on-manifold probe cannot certify off-manifold behaviour, and probe
evidence must be paired with behavioural randomization.

Everything is offline: heads are 0.24M parameters and the inputs come from the
recorded corpus, so no simulator, no detector and no GPU are involved.

Usage:
    python scripts/attribution_profiles.py --out results/attribution_profiles.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: name -> checkpoint. The deployed lineage; see the head-lineage table.
HEADS = {
    "v2": "results_backup/weights/goal_heads_v2_partial10.pt",
    "v2.1": "results_backup/weights/goal_heads_v21.pt",
    "v3": "results_backup/weights/goal_heads_v3.pt",
    "v5 (released)": "models/goal_heads_v5.pt",
}
#: The channels a substitution can target, named as the head sees them.
CHANNELS = ["uv", "proprio", "box_emb", "frame_emb"]


def load_ticks(corpus: Path, limit: int) -> list[dict]:
    """Per-tick raw inputs from recorded episodes (the head's own input diet)."""
    from microvla.utils.embedding import standardize

    ticks: list[dict] = []
    for f in sorted(corpus.glob("*.npz"))[:limit]:
        d = np.load(f)
        n = len(d["source_centers"])
        for i in range(n):
            if float(d["box_weights"][i][0]) <= 0:
                continue                      # no source detection on this tick
            ticks.append({
                "uv": np.asarray(d["source_centers"][i], dtype=np.float32),
                "conf": float(d["box_weights"][i][0]),
                "proprio": np.asarray(d["proprio"][i], dtype=np.float32),
                "box_emb": standardize(torch.as_tensor(
                    d["source_box_embs"][i])).numpy().astype(np.float32),
                "frame_emb": standardize(torch.as_tensor(
                    d["frame_embs"][i])).numpy().astype(np.float32),
            })
    return ticks


def predict(head, t: dict) -> np.ndarray:
    """The head's OFFSET, not its absolute point.

    ``GraspPointHead`` emits ``xy = eef_xy + delta + lever_bias``. Substituting
    proprioception therefore moves the absolute output by the whole end-effector
    displacement no matter what the head has learned --- measuring that would
    report ~24 cm of "proprioception attribution" for every head including the
    repaired one, which is an artifact of the parameterisation and not a
    shortcut. The question the probe is asking is whether a channel changes
    where the head thinks the OBJECT is, so we subtract the anchor and profile
    the residual.
    """
    from microvla.control import build_grasp_features

    feats = build_grasp_features(uv=t["uv"], conf=t["conf"], proprio=t["proprio"],
                                 box_emb=t["box_emb"], frame_emb=t["frame_emb"])
    with torch.no_grad():
        p = head(feats)
    xy = p["xy"][0].cpu().numpy().astype(np.float64)
    eef = feats["eef_xy"][0].cpu().numpy().astype(np.float64)
    return xy - eef


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/libero_object_v8")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", default="results/attribution_profiles.json")
    a = ap.parse_args()

    from microvla.control import load_goal_heads

    ticks = load_ticks(REPO / a.corpus, a.episodes)
    if len(ticks) < 2:
        raise RuntimeError(f"only {len(ticks)} usable ticks; refusing to profile")

    # "Most separated" is defined on the INPUTS, once, and reused for every head
    # so the substitution is identical across heads. Picking a different pair per
    # head would make the profiles incomparable, which is the whole measurement.
    uv = np.stack([t["uv"] for t in ticks])
    d2 = ((uv[:, None, :] - uv[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
    print(f"{len(ticks)} ticks; most-separated pair {i},{j} "
          f"(uv distance {np.sqrt(d2[i, j]):.4f})")

    out = {"corpus": a.corpus, "n_ticks": len(ticks), "pair": [int(i), int(j)],
           "channels": CHANNELS, "heads": {}}
    for name, ck in HEADS.items():
        p = REPO / ck
        if not p.exists():
            print(f"  {name:<14} MISSING {ck} -- reported absent, not zero")
            out["heads"][name] = None
            continue
        grasp, _place, _meta = load_goal_heads(str(p))
        grasp.eval()
        base = predict(grasp, ticks[i])
        prof = {}
        for ch in CHANNELS:
            swapped = dict(ticks[i])
            swapped[ch] = ticks[j][ch]
            if ch == "uv":                    # confidence travels with the box
                swapped["conf"] = ticks[j]["conf"]
            moved = float(np.linalg.norm(predict(grasp, swapped) - base)) * 100
            prof[ch] = round(moved, 4)
        out["heads"][name] = prof
        print(f"  {name:<14} " + "  ".join(f"{c}={prof[c]:.2f}cm" for c in CHANNELS))

    # The counterexample, computed rather than asserted.
    a2, b2 = out["heads"].get("v2"), out["heads"].get("v2.1")
    if a2 and b2:
        diffs = [abs(a2[c] - b2[c]) for c in CHANNELS]
        out["v2_vs_v21_max_channel_diff_cm"] = round(max(diffs), 4)
        out["v2_vs_v21_mean_channel_diff_cm"] = round(float(np.mean(diffs)), 4)
        print(f"\nv2 vs v2.1: max channel difference "
              f"{max(diffs):.2f} cm, mean {np.mean(diffs):.2f} cm "
              f"-- deployed 0.000 vs 0.700")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
