"""Build per-object source-box visual prototypes from a recorded corpus.

The frozen detector's TEXT tower cannot separate look-alike objects
("cream cheese" matches several boxes; measured uv std 0.33 with and
without crop-CLIP reranking, log 2026-08-06). The corpora already contain
the answer: every recorded episode stores the ROIAlign embedding of the
box the teacher actually grasped. Their mean is a discriminative visual
prototype for that object — zero new episodes, zero training.

    python scripts/build_prototypes.py --name cream --data-dir data/cream_rand \
        --name butter --data-dir data/butter_rand --data-dir data/butter_rand2 \
        --out checkpoints/role_prototypes.pt

Prototypes are unit-normalized; the consumer scores proposals by cosine.
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def _ticks(dirs: list[str], conf_floor: float) -> np.ndarray:
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
    return np.concatenate(vecs) if vecs else np.zeros((0, 1))


def prototype(dirs: list[str], conf_floor: float) -> tuple[torch.Tensor, int, int]:
    vecs, n_eps = [], 0
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
            if not keep.any():
                continue
            e = emb[keep]
            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
            vecs.append(e)
            n_eps += 1
    if not vecs:
        raise SystemExit(f"no usable episodes under {dirs}")
    allv = np.concatenate(vecs, axis=0)
    m = allv.mean(axis=0)
    m = m / (np.linalg.norm(m) + 1e-8)
    return torch.as_tensor(m, dtype=torch.float32), n_eps, allv.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", required=True,
                    help="object name; repeat, pairing with --data-dir groups "
                         "separated by --next")
    ap.add_argument("--data-dir", action="append", required=True,
                    help="corpus dir for the CURRENT --name (repeatable)")
    ap.add_argument("--next", action="append", default=[],
                    help=argparse.SUPPRESS)
    ap.add_argument("--conf-floor", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # One name per invocation group: names[i] owns the dirs listed after it.
    # Simple contract: equal counts means 1:1; a single name owns all dirs.
    protos: dict[str, torch.Tensor] = {}
    if len(a.name) == 1:
        vec, n_eps, n_ticks = prototype(a.data_dir, a.conf_floor)
        protos[a.name[0]] = vec
        print(f"{a.name[0]}: {n_eps} episodes, {n_ticks} ticks")
    else:
        if len(a.name) != len(a.data_dir):
            raise SystemExit("with multiple --name, pass one --data-dir each")
        for nm, d in zip(a.name, a.data_dir):
            vec, n_eps, n_ticks = prototype([d], a.conf_floor)
            protos[nm] = vec
            print(f"{nm}: {n_eps} episodes, {n_ticks} ticks")

    # The crop embeddings share a dominant common component (measured: raw
    # prototype cosines ~0.99 across three objects, while the SAME vectors
    # centered on that component separate to -0.08..-0.77 and classify
    # object identity at 0.613 vs 0.333 chance, leave-episode-out —
    # scripts/proto_separability.py). Ship the common direction so the
    # consumer can project it out of both sides before scoring.
    allticks = np.concatenate([_ticks(a.data_dir, a.conf_floor)]) \
        if len(a.name) == 1 else \
        np.concatenate([_ticks([d], a.conf_floor) for d in a.data_dir])
    g = allticks.mean(axis=0)
    g = g / (np.linalg.norm(g) + 1e-8)
    gt = torch.as_tensor(g, dtype=torch.float32)
    cen = {}
    for k, v in protos.items():
        w = v - float((v * gt).sum()) * gt
        cen[k] = w / (w.norm() + 1e-8)

    names = list(protos)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            print(f"cosine({x}, {y}) raw={float((protos[x] * protos[y]).sum()):+.4f} "
                  f"centered={float((cen[x] * cen[y]).sum()):+.4f}")
    out = dict(cen)
    out["_global_mean"] = gt
    torch.save(out, a.out)
    print("wrote", a.out, "(centered prototypes + _global_mean)")


if __name__ == "__main__":
    main()
