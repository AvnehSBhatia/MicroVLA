"""Build per-object crop-embedding BANKS for discriminative role binding.

A single mean prototype throws away the crops' multimodality: measured
leave-episode-out 3-way identity accuracy is 0.613 for the mean but
**0.902 for 1-NN** over the same vectors (0.82 on the hardest object;
`scripts/knn_sep.py`, log 2026-08-06). So ship the bank, not the mean.

    python scripts/build_role_bank.py --obj cream=data/cream_rand \
        --obj butter=data/butter_rand,data/butter_rand2 \
        --obj soup=data/teacher_rand_full --out checkpoints/role_bank.pt

Vectors are unit-normalized and centered on the corpus-wide common
direction (raw crop embeddings are ~0.99 collinear across objects; the
common component carries no identity). Consumers score a proposal by
``max cos(e, bank[target]) - max cos(e, bank[other])``.
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def load(dirs: list[str], conf_floor: float) -> np.ndarray:
    vecs = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            if f.stem.endswith("_state"):
                continue
            with np.load(f) as data:
                if "source_box_embs" not in data or "box_weights" not in data:
                    continue
                emb = np.asarray(data["source_box_embs"], dtype=np.float64)
                conf = np.asarray(data["box_weights"], dtype=np.float64)[:, 0]
            keep = conf >= conf_floor
            if keep.any():
                e = emb[keep]
                vecs.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8))
    if not vecs:
        raise SystemExit(f"no usable episodes under {dirs}")
    return np.concatenate(vecs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", action="append", required=True, help="name=dir[,dir]")
    ap.add_argument("--conf-floor", type=float, default=0.02)
    ap.add_argument("--stride", type=int, default=2,
                    help="subsample stride over ticks (bank size control)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = {}
    for spec in a.obj:
        name, dirs = spec.split("=", 1)
        raw[name] = load(dirs.split(","), a.conf_floor)
        print(f"{name}: {raw[name].shape[0]} ticks")

    g = np.concatenate(list(raw.values())).mean(axis=0)
    g = g / (np.linalg.norm(g) + 1e-8)

    out = {"_global_mean": torch.as_tensor(g, dtype=torch.float32)}
    for name, v in raw.items():
        c = v - (v @ g)[:, None] * g
        c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
        c = c[::max(1, a.stride)]
        out[name] = torch.as_tensor(c, dtype=torch.float32)
        print(f"{name}: bank {tuple(out[name].shape)}")
    torch.save(out, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
