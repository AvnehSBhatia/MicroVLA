"""Train the stage-1 learned gates from self-play scaffold-state sidecars.

Labels come from the machine's own recorded decisions (``machine_state``
sidecars written by ``preprocess/teacher_rollouts.py`` when the structured
policy drives): the close trigger fires at descend->grasp transitions
(phase col 0: 1 -> 2), the hold check's label is whether the grasp that
just completed led to lift (2 -> 4) rather than rise (2 -> 3).

Usage: python -m train.train_gates --data-dir data/selfplay_v5 \
    --out checkpoints/gates_v1.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from microvla.control.gate_heads import (CloseTriggerGate, HoldCheckGate,
                                         save_gates)
from microvla.control.machine import PROBE_XY

APPROACH, DESCEND, GRASP, RISE, LIFT = 0, 1, 2, 3, 4


def episode_samples(ep_npz: Path):
    """(close_X, close_y, hold_X, hold_y) arrays from one episode + sidecar."""
    side = ep_npz.with_name(f"{ep_npz.stem}_state.npz")
    if not side.exists():
        return None
    with np.load(side) as sd:
        if "machine_state" not in sd:
            return None
        ms = np.asarray(sd["machine_state"], dtype=np.float64)
    with np.load(ep_npz) as d:
        if "proprio" not in d:
            return None
        pro = np.asarray(d["proprio"], dtype=np.float64)
    # Sidecars are PER-ENV-TICK; the baked npz is sampled (~2 Hz). Prefer a
    # per-tick proprio if the sidecar carries one (newer corpora); otherwise
    # stride-align the machine state down to the sampled grid — transitions
    # survive (the shortest phase spans 12+ env ticks vs stride ~10), at the
    # cost of "fire within one sample" label smearing.
    with np.load(side) as sd:
        if "proprio" in sd:
            pro = np.asarray(sd["proprio"], dtype=np.float64)
    if len(ms) > len(pro) * 2:
        stride = max(1, round(len(ms) / len(pro)))
        ms = ms[::stride]
    T = min(len(ms), len(pro))
    ms, pro = ms[:T], pro[:T]
    phase, attempt, latched = ms[:, 0], ms[:, 1], ms[:, 2]
    bx, by, close_z = ms[:, 3], ms[:, 4], ms[:, 5]
    cX, cy, hX, hy = [], [], [], []
    descend_n = 0
    for t in range(T - 1):
        if phase[t] == DESCEND and latched[t] > 0:
            descend_n += 1
            pi = int(min(attempt[t], len(PROBE_XY) - 1))
            ax = bx[t] + PROBE_XY[pi][0]
            ay = by[t] + PROBE_XY[pi][1]
            lat = max(abs(ax - pro[t, 0]), abs(ay - pro[t, 1]))
            cX.append([pro[t, 2], close_z[t], lat, descend_n / 100.0])
            cy.append(1.0 if phase[t + 1] == GRASP else 0.0)
        else:
            descend_n = 0
        if phase[t] == GRASP and phase[t + 1] in (LIFT, RISE):
            hX.append([abs(pro[t, 7]), abs(pro[t, 8]), 1.0])
            hy.append(1.0 if phase[t + 1] == LIFT else 0.0)
    if not cX:
        return None
    return (np.array(cX, dtype=np.float32), np.array(cy, dtype=np.float32),
            np.array(hX, dtype=np.float32) if hX else np.zeros((0, 3), np.float32),
            np.array(hy, dtype=np.float32) if hy else np.zeros((0,), np.float32))


def _train_binary(model, X, y, epochs=400, lr=5e-3, pos_weight=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.as_tensor(X)
    yt = torch.as_tensor(y)
    crit = torch.nn.BCEWithLogitsLoss(
        pos_weight=None if pos_weight is None else torch.tensor(pos_weight))
    for _ in range(epochs):
        loss = crit(model(Xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = torch.sigmoid(model(Xt)) > 0.5
        acc = float((pred.float() == yt).float().mean())
        pos_recall = (float(pred[yt > 0.5].float().mean())
                      if (yt > 0.5).any() else float("nan"))
    return acc, pos_recall


def main(argv=None) -> None:
    from microvla.utils.signals import ignore_sigterm
    ignore_sigterm()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", action="append", required=True)
    ap.add_argument("--out", default="checkpoints/gates_v1.pt")
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args(argv)

    cX, cy, hX, hy = [], [], [], []
    n_eps = 0
    for d in args.data_dir:
        for f in sorted(Path(d).glob("*.npz")):
            if f.stem.endswith("_state"):
                continue
            s = episode_samples(f)
            if s is None:
                continue
            n_eps += 1
            cX.append(s[0]); cy.append(s[1]); hX.append(s[2]); hy.append(s[3])
    cX = np.concatenate(cX); cy = np.concatenate(cy)
    hX = np.concatenate(hX); hy = np.concatenate(hy)
    print(f"[gates] {n_eps} episodes: close {len(cy)} ticks "
          f"({int(cy.sum())} fire), hold {len(hy)} decisions "
          f"({int(hy.sum())} held)")

    close = CloseTriggerGate()
    # Fire ticks are rare (one per descend) — reweight.
    pw = float((len(cy) - cy.sum()) / max(1.0, cy.sum()))
    c_acc, c_rec = _train_binary(close, cX, cy, args.epochs, pos_weight=pw)
    hold = HoldCheckGate()
    h_acc, h_rec = (float("nan"), float("nan"))
    if len(hy) >= 4:
        h_acc, h_rec = _train_binary(hold, hX, hy, args.epochs)
    meta = {"episodes": n_eps, "close_acc": c_acc, "close_fire_recall": c_rec,
            "hold_acc": h_acc, "hold_recall": h_rec,
            "n_close": int(len(cy)), "n_hold": int(len(hy))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_gates(args.out, close, hold, meta=meta)
    print("[gates] saved", args.out, meta)


if __name__ == "__main__":
    main()
