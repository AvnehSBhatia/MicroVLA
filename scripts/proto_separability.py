"""Is object identity recoverable from the frozen detector's crop embeddings?

Measures, on the corpora's own grasped-box embeddings: (a) raw and
mean-centered prototype cosines, (b) leave-episode-out nearest-prototype
classification accuracy (3-way, chance 1/3). This decides whether the
multi-object binding boundary is the detector's TEXT tower alone (fixable
by visual binding) or its crop FEATURE SPACE (a stack-level bound).
"""
import argparse
from pathlib import Path

import numpy as np


def load(dirs: list[str], conf_floor: float):
    eps = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            if f.stem.endswith("_state"):
                continue
            with np.load(f) as data:
                if "source_box_embs" not in data:
                    continue
                emb = np.asarray(data["source_box_embs"], dtype=np.float64)
                conf = np.asarray(data["box_weights"], dtype=np.float64)[:, 0]
            keep = conf >= conf_floor
            if keep.any():
                e = emb[keep]
                eps.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8))
    return eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", action="append", required=True,
                    help="name=dir[,dir...]")
    ap.add_argument("--conf-floor", type=float, default=0.02)
    a = ap.parse_args()

    data = {}
    for spec in a.obj:
        name, dirs = spec.split("=", 1)
        data[name] = load(dirs.split(","), a.conf_floor)
        print(f"{name}: {len(data[name])} episodes, "
              f"{sum(len(e) for e in data[name])} ticks")

    names = list(data)
    allticks = np.concatenate([np.concatenate(v) for v in data.values()])
    gmean = allticks.mean(axis=0)
    gmean /= np.linalg.norm(gmean) + 1e-8

    def proto(eps, center):
        m = np.concatenate(eps).mean(axis=0)
        if center:
            m = m - (m @ gmean) * gmean
        return m / (np.linalg.norm(m) + 1e-8)

    for center in (False, True):
        tag = "centered" if center else "raw"
        P = {n: proto(v, center) for n, v in data.items()}
        print(f"\n[{tag}] pairwise prototype cosines")
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                print(f"  {x} vs {y}: {float(P[x] @ P[y]):+.4f}")

        # leave-one-episode-out nearest prototype, per tick
        correct = total = 0
        for n in names:
            for i, ep in enumerate(data[n]):
                rest = [e for j, e in enumerate(data[n]) if j != i]
                if not rest:
                    continue
                Q = dict(P)
                Q[n] = proto(rest, center)
                v = ep - (ep @ gmean)[:, None] * gmean if center else ep
                v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
                scores = np.stack([v @ Q[m] for m in names], axis=1)
                pred = np.asarray(names)[scores.argmax(axis=1)]
                correct += int((pred == n).sum())
                total += len(pred)
        print(f"[{tag}] leave-episode-out tick accuracy: "
              f"{correct}/{total} = {correct / max(1, total):.3f} "
              f"(chance {1 / len(names):.3f})")


if __name__ == "__main__":
    main()
