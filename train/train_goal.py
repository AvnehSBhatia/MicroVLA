"""Offline trainer for the structured-control goal heads (microvla/control/).

Replaces monolithic BC for the *task-content* quantities with two small,
well-posed supervised regressions over the EXISTING teacher corpus — no
re-recording, no simulator:

  * GraspPointHead: every pre-grasp tick with a visible source box maps
    (uv, conf, proprio, box emb, frame emb) -> the eef position at the
    episode's FINAL close onset (where the hand actually was when the grasp
    that stuck began). Labels contain the hand-eye lever arm by construction.
  * PlaceHead: command text embedding -> eef xy at the release onset.

Label derivation from npz keys (train/dataset.py schema):
  grip command = ``pwm_targets[:, 0, -1]`` (normalized, symmetric: close>0).
  The FINAL close onset is the last open->close transition whose closed run
  persists >= ``--min-hold`` samples (probe-retry closes at 2 Hz sampling are
  ~1 sample; the grasp that sticks stays closed through lift+transport).
  Release onset = the first open sample after it.

Usage (pod):
  python train/train_goal.py --data-dir data/teacher_grid2 \
      --data-dir data/teacher_dagger_soup_grid --out checkpoints/goal_heads.pt

The val report (cm error by altitude band + sigma calibration + lever bias)
is the offline go/no-go: median xy error under ~3 cm at altitude means the
servo shell's ±6 cm probe search will convert estimates into grasps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from microvla.control.goal_head import (FEATURE_VERSION, GraspPointHead,
                                        PlaceHead, build_grasp_features,
                                        save_goal_heads)


def derive_labels(ep: dict, min_hold: int = 3) -> dict | None:
    """Grasp/place labels + supervisable tick mask from one episode's arrays.

    Returns ``None`` when the episode has no persistent close (no grasp to
    learn from). ``place_xy`` may be ``None`` when the episode never reopens.
    """
    grip = np.asarray(ep["pwm_targets"], dtype=np.float64)[:, 0, -1]
    closed = grip > 0.0
    T = closed.shape[0]
    onsets = [t for t in range(T) if closed[t] and (t == 0 or not closed[t - 1])]
    final_close = None
    for t in reversed(onsets):
        run = 0
        while t + run < T and closed[t + run]:
            run += 1
        if run >= min_hold:
            final_close = t
            break
    if final_close is None:
        return None
    proprio = np.asarray(ep["proprio"], dtype=np.float64)
    if proprio[final_close, 9] < 0.5:
        return None                     # no valid proprio at the label tick
    grasp_xyz = proprio[final_close, :3].copy()
    if grasp_xyz[2] > 0.20:
        return None                     # a "grasp" nowhere near the table: junk
    release = None
    for t in range(final_close + 1, T):
        if not closed[t]:
            release = t
            break
    place_xy = None
    if release is not None and proprio[release, 9] > 0.5:
        place_xy = proprio[release, :2].copy()
    # Supervisable ticks: strictly before the final close, source box seen,
    # gripper not mid-close (probe attempts), valid proprio.
    mask = np.zeros(T, dtype=bool)
    w = np.asarray(ep["box_weights"], dtype=np.float64)[:, 0]
    for t in range(final_close):
        if w[t] > 0.01 and not closed[t] and proprio[t, 9] > 0.5:
            mask[t] = True
    if not mask.any():
        return None
    return {"grasp_xyz": grasp_xyz, "place_xy": place_xy,
            "tick_mask": mask, "final_close": final_close}


def _episode_samples(path: Path, min_hold: int) -> dict | None:
    with np.load(path) as data:
        keys = ("pwm_targets", "proprio", "box_weights", "source_centers",
                "source_box_embs", "frame_embs", "text_tokens")
        if any(k not in data for k in keys):
            return None
        if "success" in data and float(np.asarray(data["success"]).reshape(-1)[0]) < 0.5:
            return None
        ep = {k: data[k] for k in keys}
    lab = derive_labels(ep, min_hold=min_hold)
    if lab is None:
        return None
    feats = []
    for t in np.nonzero(lab["tick_mask"])[0]:
        feats.append(build_grasp_features(
            uv=ep["source_centers"][t], conf=ep["box_weights"][t, 0],
            proprio=ep["proprio"][t], box_emb=ep["source_box_embs"][t],
            frame_emb=ep["frame_embs"][t]))
    out = {k: torch.cat([f[k] for f in feats], dim=0) for k in feats[0]}
    n = out["geom"].shape[0]
    out["label_xy"] = torch.as_tensor(
        np.tile(lab["grasp_xyz"][:2], (n, 1)), dtype=torch.float32)
    out["label_z"] = torch.as_tensor(
        np.tile(lab["grasp_xyz"][2:3], (n, 1)), dtype=torch.float32)
    out["command_emb"] = torch.as_tensor(ep["text_tokens"][0], dtype=torch.float32)
    out["place_xy"] = (None if lab["place_xy"] is None else
                       torch.as_tensor(lab["place_xy"], dtype=torch.float32))
    return out


def load_corpus(dirs: list[str], min_hold: int, val_frac: float, seed: int):
    files: list[Path] = []
    for d in dirs:
        files.extend(sorted(Path(d).glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"no .npz episodes under {dirs}")
    rng = np.random.default_rng(seed)
    eps, skipped = [], 0
    for f in files:
        s = _episode_samples(f, min_hold)
        if s is None:
            skipped += 1
        else:
            eps.append(s)
    if not eps:
        raise RuntimeError(f"no usable episodes ({skipped} skipped) in {dirs}")
    order = rng.permutation(len(eps))
    n_val = max(1, int(len(eps) * val_frac))
    val_idx = set(order[:n_val].tolist())
    train = [e for i, e in enumerate(eps) if i not in val_idx]
    val = [e for i, e in enumerate(eps) if i in val_idx]
    print(f"[goal] corpus: {len(eps)} episodes usable ({skipped} skipped), "
          f"{sum(e['geom'].shape[0] for e in eps)} grasp ticks, "
          f"{sum(1 for e in eps if e['place_xy'] is not None)} place labels, "
          f"split {len(train)}/{len(val)}")
    return train, val


def _stack(eps: list[dict], device: str) -> dict:
    out = {k: torch.cat([e[k] for e in eps], dim=0).to(device)
           for k in ("geom", "box_emb", "frame_emb", "eef_xy",
                     "label_xy", "label_z")}
    return out


def _grasp_metrics(head: GraspPointHead, batch: dict) -> dict:
    with torch.no_grad():
        pred = head(batch)
        err = (pred["xy"] - batch["label_xy"]).norm(dim=-1)      # metres
        ez = (pred["z"] - batch["label_z"]).abs().reshape(-1)
        z_eef = batch["geom"][:, 5]
        sig = GraspPointHead.sigma(pred)[:, :2].max(dim=-1).values
        cover = ((pred["xy"] - batch["label_xy"]).abs()
                 <= 2.0 * GraspPointHead.sigma(pred)[:, :2]).all(dim=-1)
        bands = {"alt(z>0.25)": z_eef > 0.25,
                 "mid(0.15-0.25)": (z_eef > 0.15) & (z_eef <= 0.25),
                 "low(z<=0.15)": z_eef <= 0.15}
        m = {}
        for name, sel in bands.items():
            if sel.any():
                e = err[sel]
                m[name] = {"n": int(sel.sum()),
                           "xy_med_cm": float(e.median() * 100),
                           "xy_p90_cm": float(e.quantile(0.9) * 100)}
        m["all"] = {"n": int(err.shape[0]),
                    "xy_med_cm": float(err.median() * 100),
                    "xy_p90_cm": float(err.quantile(0.9) * 100),
                    "z_med_cm": float(ez.median() * 100),
                    "sigma_med_cm": float(sig.median() * 100),
                    "cover_2sigma": float(cover.float().mean())}
        return m


def main(argv=None) -> None:
    from microvla.utils.signals import ignore_sigterm
    ignore_sigterm()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", action="append", required=True)
    ap.add_argument("--out", default="checkpoints/goal_heads.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--min-hold", type=int, default=3)
    ap.add_argument("--sigma-epochs", type=int, default=600,
                    help="post-pass training ONLY the lv head (lr 1e-2): the "
                         "log-variance converges ~10x slower than the mean "
                         "(measured: sigma_med 8.8 cm vs true err 1.3 cm at "
                         "300 joint epochs), and an over-conservative sigma "
                         "starves the machine's latch gate.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    train_eps, val_eps = load_corpus(args.data_dir, args.min_hold,
                                     args.val_frac, args.seed)
    tr, va = _stack(train_eps, args.device), _stack(val_eps, args.device)

    grasp = GraspPointHead().to(args.device)
    opt = torch.optim.Adam(grasp.parameters(), lr=args.lr)
    n = tr["geom"].shape[0]
    best_val, best_state = float("inf"), None
    for ep in range(args.epochs):
        grasp.train()
        perm = torch.randperm(n, device=args.device)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch = {k: v[idx] for k, v in tr.items()}
            loss = GraspPointHead.loss(grasp(batch), batch["label_xy"],
                                       batch["label_z"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * idx.shape[0]
        grasp.eval()
        with torch.no_grad():
            vloss = float(GraspPointHead.loss(
                grasp(va), va["label_xy"], va["label_z"]))
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in grasp.state_dict().items()}
        if ep % 25 == 0 or ep == args.epochs - 1:
            m = _grasp_metrics(grasp, va)["all"]
            print(f"[goal] epoch {ep:3d} train {tot / n:.4f} val {vloss:.4f} "
                  f"| val xy med {m['xy_med_cm']:.2f} cm p90 {m['xy_p90_cm']:.2f} "
                  f"| lever {grasp.lever_bias.detach().cpu().numpy().round(4)}")
    grasp.load_state_dict(best_state)
    # Sigma refinement: lv head only, mean frozen by optimizer selection.
    # Full-batch; the NLL on detached residuals is exactly the calibration
    # objective, so this can only improve sigma, never move the mean.
    if args.sigma_epochs > 0:
        sopt = torch.optim.Adam(grasp.lv_head.parameters(), lr=1e-2)
        grasp.train()
        for i in range(args.sigma_epochs):
            loss = GraspPointHead.loss(grasp(tr), tr["label_xy"], tr["label_z"])
            sopt.zero_grad()
            loss.backward()
            sopt.step()
        with torch.no_grad():
            m = _grasp_metrics(grasp, va)["all"]
        print(f"[goal] sigma refined ({args.sigma_epochs} ep): "
              f"sigma_med {m['sigma_med_cm']:.2f} cm "
              f"cover_2sigma {m['cover_2sigma']:.3f}")
    grasp.eval()

    # ---- PlaceHead: one (command_emb -> place_xy) sample per episode -------
    place = PlaceHead().to(args.device)
    p_tr = [(e["command_emb"], e["place_xy"]) for e in train_eps
            if e["place_xy"] is not None]
    p_va = [(e["command_emb"], e["place_xy"]) for e in val_eps
            if e["place_xy"] is not None]
    place_metrics = {"n_train": len(p_tr), "n_val": len(p_va)}
    if p_tr:
        pc = torch.stack([a for a, _ in p_tr]).to(args.device)
        pl = torch.stack([b for _, b in p_tr]).to(args.device)
        popt = torch.optim.Adam(place.parameters(), lr=args.lr)
        for _ in range(min(2000, args.epochs * 10)):
            loss = PlaceHead.loss(place(pc), pl)
            popt.zero_grad()
            loss.backward()
            popt.step()
        place.eval()
        with torch.no_grad():
            if p_va:
                vc = torch.stack([a for a, _ in p_va]).to(args.device)
                vl = torch.stack([b for _, b in p_va]).to(args.device)
                perr = (place(vc)["xy"] - vl).norm(dim=-1)
                place_metrics["val_med_cm"] = float(perr.median() * 100)
            with torch.no_grad():
                mean_pred = place(pc)["xy"].mean(dim=0)
            place_metrics["mean_pred"] = [float(v) for v in mean_pred]

    metrics = {"grasp_val": _grasp_metrics(grasp, va),
               "place": place_metrics,
               "lever_bias": [float(v) for v in grasp.lever_bias.detach().cpu()],
               "feature_version": FEATURE_VERSION,
               "data_dirs": args.data_dir, "best_val_nll": best_val}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_goal_heads(out, grasp.cpu(), place.cpu(), meta=metrics)
    print("[goal] saved", out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
