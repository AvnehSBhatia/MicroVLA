"""Does the grasp head get worse at tail uv? Measured OFF the machine's trajectory.

Breaks the circularity of the deployment measurement: here uv is set by the
TEACHER's trajectory in the recorded corpus, not by the head's own error, so
binning the head's accuracy by uv cannot be an artifact of the head being
wrong. Offline, no simulator, no rollout.
"""
import sys
from pathlib import Path
sys.path.insert(0, "/root/MicroVLA")
import numpy as np
import torch
from train.train_goal import _episode_samples
from microvla.control.goal_head import load_goal_heads

grasp, place, extra = load_goal_heads("checkpoints/goal_heads_v8.pt", map_location="cpu")
grasp.eval()

CORPORA = [("cream", "data/cream_rand"), ("soup", "data/teacher_rand"),
           ("butter", "data/butter_rand")]

for name, d in CORPORA:
    files = sorted(p for p in Path(d).glob("*.npz") if not p.stem.endswith("_state"))
    U, E = [], []
    for f in files:
        s = _episode_samples(f, min_hold=3)
        if not s:
            continue
        with torch.no_grad():
            pred = grasp({k: s[k] for k in ("geom", "box_emb", "frame_emb", "eef_xy")})
        err = (pred["xy"] - s["label_xy"]).norm(dim=-1).numpy()
        uv = s["geom"][:, :2].numpy()
        keep = s["geom"][:, 2].numpy() > 0        # detected ticks only
        U.append(uv[keep]); E.append(err[keep])
    if not U:
        print(f"{name}: no samples"); continue
    U = np.concatenate(U); E = np.concatenate(E)
    print(f"\n{name}: {len(E)} detected ticks over {len(files)} episodes")
    print(f"  overall mean xy error {E.mean():.4f} m")
    # bin by max(u,v) distance from frame centre -- 'how far into the tail'
    ecc = np.maximum(np.abs(U[:, 0] - 0.5), np.abs(U[:, 1] - 0.5))
    qs = np.quantile(ecc, [0, .25, .5, .75, .9, 1.0])
    for lo, hi, lbl in zip(qs[:-1], qs[1:], ["0-25%", "25-50%", "50-75%", "75-90%", "90-100%"]):
        m = (ecc >= lo) & (ecc <= hi)
        if m.sum():
            print(f"    eccentricity {lbl:<8s} (|uv-0.5|max {lo:.3f}-{hi:.3f}): "
                  f"n={m.sum():4d}  mean err {E[m].mean():.4f} m")
print("\nUVOFFLINE_DONE", flush=True)
