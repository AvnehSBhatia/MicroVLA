"""Batched MicroVLA training — for a real GPU (CUDA / AMD ROCm, e.g. MI300X).

The Mac trainer (train_full.py) runs batch=1 because the drift encoder is
stateful. But every module already accepts a batch dimension, so on a real GPU
we batch by **bucketing episodes by exact length T** (Bridge/LIBERO T ranges
~10-17, a handful of buckets) — within a bucket all episodes are the same
length, so batching needs no padding or masking. Everything else (objective,
scheduled-horizon rollout, early stopping, best-checkpoint) matches train_full
exactly, so results are comparable.

VRAM cap: ``--max-vram-gb`` hard-limits the process via
``torch.cuda.set_per_process_memory_fraction`` (ROCm honors the cuda API), so
on a 192 GB MI300X asked for 50 GB, the process physically cannot exceed 50 GB
— it OOMs inside the cap rather than eating the whole card. Peak usage is
printed after epoch 1 so you can tune ``--batch-size``.

Data is preloaded into RAM once (the whole corpus is <1 GB), so epochs are
GPU-bound, not disk-bound.

Example (MI300X, ROCm):
    python train/train_batched.py --data-dir data/bridge --data-dir data/libero \\
        --device cuda --batch-size 64 --max-vram-gb 50 \\
        --stage-a-epochs 30 --warmup-epochs 4 --max-horizon 6 --patience 3 \\
        --stage-b-epochs 3
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.utils.embedding import standardize
from train.dataset import EPISODE_KEYS, EpisodeDataset
from train.losses import planner_bc_loss, smoothness_loss, split_planner_loss, total_planner_loss
from train.train_full import _scheduled_horizon, _tagged_name, save
from train.train_planner import resolve_device
from TRM import RecursiveTRM, spec_loss


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", action="append", required=True)
    p.add_argument("--batch-size", type=int, default=64,
                   help="episodes per batch (within a length bucket).")
    p.add_argument("--max-vram-gb", type=float, default=50.0,
                   help="hard cap on GPU memory for this process (cuda/ROCm).")
    p.add_argument("--stage-a-epochs", type=int, default=30, help="hard cap (early stop usually halts first)")
    p.add_argument("--warmup-epochs", type=int, default=4)
    p.add_argument("--max-horizon", type=int, default=6)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr-patience", type=int, default=2,
                   help="halve LR after this many at-max-horizon epochs without val "
                        "improvement (< --patience, so LR drops before early stop).")
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--stage-b-epochs", type=int, default=3)
    p.add_argument("--segments-per-episode", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--trm-d", type=int, default=1024)
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--ablate-grounding", action="store_true")
    p.add_argument("--load-stage-a", type=str, default=None,
                   help="path to a trained full_stageA.pt: load the world model and skip "
                        "stage A, retraining ONLY the planner (e.g. after a planner change).")
    return p.parse_args(argv)


def cap_vram(device: torch.device, max_gb: float) -> None:
    """Hard-caps this process's GPU memory (cuda/ROCm)."""
    if device.type != "cuda":
        return
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    frac = min(1.0, max_gb / total)
    # set_per_process_memory_fraction requires an explicit device index;
    # torch.device("cuda") has none, so resolve to the current device.
    idx = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(frac, idx)
    print(f"VRAM cap: {max_gb:.0f} GB of {total:.0f} GB total (fraction {frac:.3f})", flush=True)


def preload_buckets(data_dirs, val_frac, seed, device):
    """Loads every episode into RAM and buckets each split by length T.

    Returns (train_buckets, val_buckets), each a dict[T] -> dict of stacked
    tensors on ``device`` with a leading batch dim: frame_embs [N, T, 512],
    pwm_targets [N, T, 5, 7], text_tokens [N, 3, 512], etc.
    """
    sets = [EpisodeDataset(d) for d in data_dirs]
    index = [(i, j) for i, ds in enumerate(sets) for j in range(len(ds))]
    rng = random.Random(seed)
    rng.shuffle(index)
    n_val = max(1, int(len(index) * val_frac)) if val_frac > 0 else 0
    splits = {"val": index[:n_val], "train": index[n_val:]}

    out = {}
    for name, idx in splits.items():
        by_T = defaultdict(list)
        for i, j in idx:
            ep = sets[i][j]
            by_T[ep["frame_embs"].shape[0]].append(ep)
        buckets = {}
        for T, eps in by_T.items():
            buckets[T] = {k: torch.stack([e[k] for e in eps]).to(device) for k in EPISODE_KEYS}
        out[name] = buckets
    return out["train"], out["val"]


def _boxes(batch, idx, fade, cfg, ablate):
    """Held box evidence at timestep idx for the whole batch (or zeros if ablated)."""
    B = batch["frame_embs"].shape[0]
    if ablate:
        z = batch["frame_embs"].new_zeros(B, cfg.vis_dim)
        z2 = batch["frame_embs"].new_zeros(B, 2)
        return z, z, z2, z2, batch["frame_embs"].new_zeros(B, 2)
    return (batch["source_box_embs"][:, idx], batch["target_box_embs"][:, idx],
            batch["source_centers"][:, idx], batch["target_centers"][:, idx],
            batch["box_weights"][:, idx] * fade)


def real_paths(batch, fusion, drift, cfg, ablate):
    """Batched grounded fused matrices + drift codes for every timestep.

    Returns fused_all[t] -> [B, 32, 5] and delta_all[t] -> [B, 256], lists over
    t, with grad. Drift runs sequentially over T (batched over episodes).
    """
    B, T = batch["frame_embs"].shape[:2]
    text = batch["text_tokens"]
    drift.reset()
    fused_all, delta_all = [], []
    zeros_act = batch["pwm_targets"].new_zeros(B, batch["pwm_targets"].shape[-1])
    for t in range(T):
        last_action = batch["pwm_targets"][:, t - 1, 0] if t > 0 else zeros_act
        sbe, tbe, sc, tc, bw = _boxes(batch, t, 1.0, cfg, ablate)
        fused_all.append(fusion(text, batch["frame_embs"][:, t], sbe, tbe, sc, tc,
                                box_weight=bw, last_action=last_action))
        delta_all.append(drift(batch["frame_embs"][:, t]))
    return fused_all, delta_all


def rollout(batch, t, fused_t, delta_t, fusion, trm, cfg, H, gamma, ablate):
    """Batched H-step data-rate rollout loss (mean over batch + discounted steps)."""
    text = batch["text_tokens"]
    frames = batch["frame_embs"]
    T = frames.shape[1]
    latent = frames[:, t]
    ctx = [latent]
    fused_k, delta_k = fused_t, delta_t
    loss = torch.zeros((), device=frames.device)
    wsum = 0.0
    for k in range(1, H + 1):
        context = torch.stack(ctx[-cfg.context_window:], dim=1)  # [B, K, 512]
        pred = trm(fused_k, delta_k, latent, context=context)
        w = gamma ** (k - 1)
        loss = loss + w * spec_loss(pred, frames[:, t + k])
        wsum += w
        if k == H:
            break
        latent = standardize(pred)
        ctx.append(latent.detach())
        sbe, tbe, sc, tc, bw = _boxes(batch, t, cfg.staleness_decay ** k, cfg, ablate)
        act_idx = min(t + k, T - 1)
        last_action = batch["pwm_targets"][:, act_idx, 0]
        fused_k = fusion(text, latent, sbe, tbe, sc, tc, box_weight=bw, last_action=last_action)
    return loss / wsum


def persistence(batch, t, H, gamma) -> float:
    frames = batch["frame_embs"]
    cur = frames[:, t]
    loss, wsum = 0.0, 0.0
    for k in range(1, H + 1):
        w = gamma ** (k - 1)
        loss += w * float(spec_loss(cur, frames[:, t + k]))
        wsum += w
    return loss / wsum


def iter_batches(buckets, H, batch_size, rng, need=1):
    """Yields (T, batch) over length-buckets with T >= H+need, batched by size."""
    order = list(buckets.keys())
    rng.shuffle(order)
    for T in order:
        if T < H + need:
            continue
        b = buckets[T]
        N = b["frame_embs"].shape[0]
        perm = list(range(N)); rng.shuffle(perm)
        for s in range(0, N, batch_size):
            sel = perm[s:s + batch_size]
            yield T, {k: v[sel] for k, v in b.items()}


@torch.no_grad()
def evaluate(val_buckets, fusion, drift, trm, cfg, H, gamma, ablate, batch_size, rng):
    fusion.eval(); drift.eval(); trm.eval()
    vs, ps, n = 0.0, 0.0, 0
    for T, batch in iter_batches(val_buckets, H, batch_size, rng, need=1):
        fused_all, delta_all = real_paths(batch, fusion, drift, cfg, ablate)
        for t in range(0, T - H, max(1, (T - H) // 4)):
            vs += float(rollout(batch, t, fused_all[t], delta_all[t], fusion, trm, cfg, H, gamma, ablate))
            ps += persistence(batch, t, H, gamma)
            n += 1
    return vs / max(n, 1), ps / max(n, 1)


def stage_a(args, cfg, train_b, val_b, fusion, drift, trm, device):
    params = [*fusion.parameters(), *drift.parameters(), *trm.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    # Halve the LR when val plateaus (at max horizon), so a high initial LR
    # gets fast early progress then settles to a finer minimum instead of
    # oscillating. Pairs with early stopping: LR reduces before patience trips.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-5)
    rng = random.Random(args.seed)
    ckpt = _tagged_name("full_stageA.pt", args.tag)
    best, stale = float("inf"), 0

    for epoch in range(1, args.stage_a_epochs + 1):
        H = _scheduled_horizon(epoch, args.warmup_epochs, args.max_horizon)
        at_max = H >= args.max_horizon
        fusion.train(); drift.train(); trm.train()
        run, nb, t0 = 0.0, 0, time.time()
        for T, batch in iter_batches(train_b, H, args.batch_size, rng, need=1):
            fused_all, delta_all = real_paths(batch, fusion, drift, cfg, args.ablate_grounding)
            ts = list(range(T - H)); rng.shuffle(ts); ts = ts[: args.segments_per_episode]
            if not ts:
                continue
            opt.zero_grad()
            loss = torch.zeros((), device=device)
            for t in ts:
                loss = loss + rollout(batch, t, fused_all[t], delta_all[t], fusion, trm,
                                      cfg, H, args.gamma, args.ablate_grounding)
            loss = loss / len(ts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run += float(loss.detach()); nb += 1

        val, pers = evaluate(val_b, fusion, drift, trm, cfg, H, args.gamma,
                             args.ablate_grounding, args.batch_size, rng)
        verdict = "BEATS persistence" if val < pers else "not yet below persistence"
        tag = ""
        if not at_max:
            save(args, cfg, ckpt, fusion=fusion, drift=drift, trm=trm)
        elif val < best - args.min_delta:
            best, stale = val, 0
            save(args, cfg, ckpt, fusion=fusion, drift=drift, trm=trm); tag = " *best*"
        else:
            stale += 1; tag = f" (no improve {stale}/{args.patience})"
        if at_max:
            sched.step(val)  # only at fixed horizon (val rises during warmup by design)
        lr_now = opt.param_groups[0]["lr"]
        peak = (f" | peakVRAM {torch.cuda.max_memory_allocated(device)/1024**3:.1f}GB"
                if device.type == "cuda" else "")
        print(f"[stage A] epoch {epoch} | H={H} | lr {lr_now:.1e} | train {run/max(nb,1):.4f} "
              f"| val {val:.4f} vs persistence {pers:.4f} ({verdict}){tag} "
              f"| {time.time()-t0:.0f}s{peak}", flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if at_max and args.patience > 0 and stale >= args.patience:
            print(f"[stage A] early stop at H={args.max_horizon}, best val {best:.4f}", flush=True)
            break

    if Path(args.checkpoint_dir, ckpt).exists():
        st = torch.load(Path(args.checkpoint_dir, ckpt), map_location=device, weights_only=True)
        fusion.load_state_dict(st["fusion"]); drift.load_state_dict(st["drift"]); trm.load_state_dict(st["trm"])


def stage_b(args, cfg, train_b, val_b, fusion, drift, trm, planner, device):
    for m in (fusion, drift, trm):
        for p in m.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(planner.parameters(), lr=args.lr, weight_decay=1e-2)
    rng = random.Random(args.seed + 1)
    ckpt = _tagged_name("full_stageB.pt", args.tag)

    for epoch in range(1, args.stage_b_epochs + 1):
        planner.train(); fusion.eval(); drift.eval(); trm.eval()
        run = 0.0; grip_acc = 0.0; nb = 0; t0 = time.time()
        for T, batch in iter_batches(train_b, 1, args.batch_size, rng, need=1):
            with torch.no_grad():
                fused_all, delta_all = real_paths(batch, fusion, drift, cfg, args.ablate_grounding)
            preds, grips = [], []
            for t in range(T):
                cur = batch["frame_embs"][:, t]
                next_emb = trm(fused_all[t], delta_all[t], cur)
                plan, grip = planner(next_emb, current_emb=cur, state_delta=delta_all[t],
                                     fused=fused_all[t], return_aux=True)
                preds.append(plan); grips.append(grip)
            preds = torch.stack(preds, dim=1)          # [B, T, 5, 7]
            grips = torch.stack(grips, dim=1)          # [B, T, 5]
            target = batch["pwm_targets"]               # [B, T, 5, 7]
            P = preds.reshape(-1, *preds.shape[2:]); G = grips.reshape(-1, grips.shape[-1])
            Y = target.reshape(-1, *target.shape[2:])
            loss = split_planner_loss(P, G, Y, smooth_weight=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                run += float(loss)
                # gripper decision accuracy vs the demo (are we learning to close?)
                grip_acc += float(((G > 0) == (Y[..., -1] > 0)).float().mean())
            nb += 1
        print(f"[stage B] epoch {epoch}/{args.stage_b_epochs} | loss {run/max(nb,1):.4f} "
              f"| grip_acc {grip_acc/max(nb,1):.3f} | {time.time()-t0:.0f}s", flush=True)
        save(args, cfg, ckpt, fusion=fusion, drift=drift, trm=trm, planner=planner)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = DEFAULT_CONFIG
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    cap_vram(device, args.max_vram_gb)
    print(f"batched training on {device} | batch {args.batch_size} | data {args.data_dir}", flush=True)

    train_b, val_b = preload_buckets(args.data_dir, args.val_frac, args.seed, device)
    n_train = sum(v["frame_embs"].shape[0] for v in train_b.values())
    n_val = sum(v["frame_embs"].shape[0] for v in val_b.values())
    print(f"episodes: train {n_train} ({len(train_b)} length-buckets), val {n_val}", flush=True)

    fusion = SlotResonanceFusion(cfg).to(device)
    drift = AnchoredDriftEncoder(cfg).to(device)
    trm = RecursiveTRM(cfg, d=args.trm_d).to(device)
    planner = ChronoQueryPlanner(cfg).to(device)

    if args.load_stage_a:
        # Retrain ONLY the policy: load the trained world model, skip stage A.
        st = torch.load(args.load_stage_a, map_location=device, weights_only=True)
        fusion.load_state_dict(st["fusion"]); drift.load_state_dict(st["drift"]); trm.load_state_dict(st["trm"])
        print(f"loaded world model from {args.load_stage_a}; skipping stage A", flush=True)
        args.stage_a_epochs = 0

    if args.stage_a_epochs > 0:
        stage_a(args, cfg, train_b, val_b, fusion, drift, trm, device)
    if args.stage_b_epochs > 0:
        stage_b(args, cfg, train_b, val_b, fusion, drift, trm, planner, device)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
