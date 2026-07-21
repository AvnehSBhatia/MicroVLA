"""Full MicroVLA training: world model (TRM) first, then the policy.

Two stages, matching the training order documented in TRM_SPEC.md:

Stage A — WORLD (fusion + drift + TRM, self-supervised):
    For consecutive real frames (0.5 s apart at the stored 2 Hz), the TRM is
    unrolled for exactly ``round(tick_hz / real_frame_hz)`` = 15 calls —
    deployment-exact: the first call sees grounded evidence, the remaining 14
    are dream ticks whose predictions feed back through fusion with held
    boxes at staleness-decayed weights and re-standardized latents. The loss
    (``TRM.spec_loss``: cosine + raw MSE) compares the final prediction with
    the actual next real frame embedding. Nothing else supervises the world
    model — it trains on any converted episode, labeled or not.

Stage B — POLICY (planner, behavior cloning):
    With the world model frozen (default), the planner maps each real tick's
    TRM prediction to the recorded action chunk (``pwm_targets``), with the
    smoothness penalty. ``--joint`` unfreezes the world model at 0.1x lr for
    a final polish.

Data: one or more converted episode directories (see preprocess/) —
BridgeData V2 for pretraining, LIBERO for fine-tune. Multiple ``--data-dir``
flags concatenate datasets.

Example (24 GB M-series MacBook, MPS):

    python train/train_full.py --data-dir data/bridge --data-dir data/libero \\
        --stage-a-epochs 2 --stage-b-epochs 3 --device auto

Checkpoints: ``checkpoints/full_stageA.pt`` / ``full_stageB.pt`` (all module
state_dicts + config + normalization pointer). ``--resume`` reloads them.
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.utils.embedding import standardize
from train.dataset import EpisodeDataset
from train.losses import planner_bc_loss, smoothness_loss, total_planner_loss
from train.train_planner import resolve_device
from TRM import RecursiveTRM, spec_loss


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", action="append", required=True,
                   help="converted episode dir; repeat to concatenate datasets")
    p.add_argument("--stage-a-epochs", type=int, default=2)
    p.add_argument("--stage-b-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--joint", action="store_true",
                   help="stage B also finetunes fusion/drift/TRM at 0.1x lr")
    p.add_argument("--segments-per-episode", type=int, default=8,
                   help="stage A: rollout segments sampled per episode per epoch")
    p.add_argument("--ticks-per-meas", type=int, default=None,
                   help="TRM calls between real frames (default round(tick_hz/real_frame_hz)=15)")
    p.add_argument("--episodes-cap", type=int, default=None,
                   help="cap episodes per epoch (subsampled each epoch)")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--trm-d", type=int, default=1024, help="RecursiveTRM width")
    return p.parse_args(argv)


class _MultiDataset:
    """Concatenation of EpisodeDatasets with a deterministic val split."""

    def __init__(self, dirs: list[str], val_frac: float, seed: int) -> None:
        self.sets = [EpisodeDataset(d) for d in dirs]
        index = [(i, j) for i, ds in enumerate(self.sets) for j in range(len(ds))]
        rng = random.Random(seed)
        rng.shuffle(index)
        n_val = max(1, int(len(index) * val_frac)) if val_frac > 0 else 0
        self.val_index = index[:n_val]
        self.train_index = index[n_val:]

    def get(self, key) -> dict[str, torch.Tensor]:
        i, j = key
        return self.sets[i][j]


def _episode_real_paths(episode, fusion, drift, device):
    """Per-t grounded fused matrices + drift codes for one episode.

    Drift is stateful/sequential, so this always sweeps the full episode in
    order. Returns per-step tensors on ``device`` (batch dim 1).
    """
    T = episode["frame_embs"].shape[0]
    text = episode["text_tokens"].unsqueeze(0)
    drift.reset()
    fused_all, delta_all = [], []
    for t in range(T):
        last_action = (
            episode["pwm_targets"][t - 1, 0].unsqueeze(0)
            if t > 0 else episode["pwm_targets"].new_zeros(1, episode["pwm_targets"].shape[-1])
        )
        fused_all.append(fusion(
            text,
            episode["frame_embs"][t].unsqueeze(0),
            episode["source_box_embs"][t].unsqueeze(0),
            episode["target_box_embs"][t].unsqueeze(0),
            episode["source_centers"][t].unsqueeze(0),
            episode["target_centers"][t].unsqueeze(0),
            box_weight=episode["box_weights"][t].unsqueeze(0),
            last_action=last_action,
        ))
        delta_all.append(drift(episode["frame_embs"][t].unsqueeze(0)))
    return fused_all, delta_all


def _rollout(episode, t, fused_t, delta_t, fusion, trm, cfg, ticks: int):
    """Deployment-exact latent rollout from real frame t to a prediction of t+1.

    Call 1 uses the grounded fused matrix; calls 2..ticks re-fuse the fed-back
    (standardized) prediction with the HELD boxes of frame t at
    staleness-decayed evidence weights — exactly ``JEPALoop``'s dream path.
    """
    text = episode["text_tokens"].unsqueeze(0)
    last_action = episode["pwm_targets"][t, 0].unsqueeze(0)  # executing this chunk
    latent = episode["frame_embs"][t].unsqueeze(0)
    ctx = [latent.squeeze(0)]

    pred = trm(fused_t, delta_t, latent,
               context=torch.stack(ctx, 0).unsqueeze(0))
    for k in range(1, ticks):
        latent = standardize(pred)
        ctx.append(latent.squeeze(0).detach())
        fade = cfg.staleness_decay ** k
        fused_dream = fusion(
            text,
            latent,
            episode["source_box_embs"][t].unsqueeze(0),
            episode["target_box_embs"][t].unsqueeze(0),
            episode["source_centers"][t].unsqueeze(0),
            episode["target_centers"][t].unsqueeze(0),
            box_weight=episode["box_weights"][t].unsqueeze(0) * fade,
            last_action=last_action,
        )
        pred = trm(fused_dream, delta_t, latent,
                   context=torch.stack(ctx[-cfg.context_window:], 0).unsqueeze(0))
    return pred


def stage_a(args, cfg, data, fusion, drift, trm, device) -> None:
    """World-model training: rollout prediction of the next real embedding."""
    ticks = args.ticks_per_meas or int(round(cfg.tick_hz / cfg.real_frame_hz))
    params = [*fusion.parameters(), *drift.parameters(), *trm.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    rng = random.Random(args.seed)

    for epoch in range(1, args.stage_a_epochs + 1):
        fusion.train(); drift.train(); trm.train()
        order = list(data.train_index)
        rng.shuffle(order)
        if args.episodes_cap:
            order = order[: args.episodes_cap]
        run_loss, n_seg, t0 = 0.0, 0, time.time()

        for key in order:
            episode = {k: v.to(device) for k, v in data.get(key).items()}
            T = episode["frame_embs"].shape[0]
            if T < 2:
                continue
            fused_all, delta_all = _episode_real_paths(episode, fusion, drift, device)
            ts = list(range(T - 1))
            rng.shuffle(ts)
            ts = ts[: args.segments_per_episode]

            opt.zero_grad()
            loss = torch.zeros((), device=device)
            for t in ts:
                pred = _rollout(episode, t, fused_all[t], delta_all[t], fusion, trm, cfg, ticks)
                loss = loss + spec_loss(pred, episode["frame_embs"][t + 1].unsqueeze(0))
            loss = loss / max(len(ts), 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run_loss += float(loss.detach()); n_seg += 1

        val = evaluate_world(args, cfg, data, fusion, drift, trm, device, ticks)
        print(f"[stage A] epoch {epoch}/{args.stage_a_epochs} | train {run_loss / max(n_seg,1):.4f} "
              f"| val {val:.4f} | {time.time()-t0:.0f}s | ticks/meas={ticks}", flush=True)
        save(args, cfg, "full_stageA.pt", fusion=fusion, drift=drift, trm=trm)


@torch.no_grad()
def evaluate_world(args, cfg, data, fusion, drift, trm, device, ticks) -> float:
    fusion.eval(); drift.eval(); trm.eval()
    losses = []
    for key in data.val_index[:64]:
        episode = {k: v.to(device) for k, v in data.get(key).items()}
        T = episode["frame_embs"].shape[0]
        if T < 2:
            continue
        fused_all, delta_all = _episode_real_paths(episode, fusion, drift, device)
        for t in range(0, T - 1, max(1, (T - 1) // 4)):
            pred = _rollout(episode, t, fused_all[t], delta_all[t], fusion, trm, cfg, ticks)
            losses.append(float(spec_loss(pred, episode["frame_embs"][t + 1].unsqueeze(0))))
    return sum(losses) / max(len(losses), 1)


def stage_b(args, cfg, data, fusion, drift, trm, planner, device) -> None:
    """Policy training: behavior cloning through the (frozen) world model."""
    world = [*fusion.parameters(), *drift.parameters(), *trm.parameters()]
    for p in world:
        p.requires_grad_(args.joint)
    groups = [{"params": list(planner.parameters()), "lr": args.lr}]
    if args.joint:
        groups.append({"params": world, "lr": args.lr * 0.1})
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    rng = random.Random(args.seed + 1)

    for epoch in range(1, args.stage_b_epochs + 1):
        planner.train()
        fusion.train(args.joint); drift.train(args.joint); trm.train(args.joint)
        order = list(data.train_index)
        rng.shuffle(order)
        if args.episodes_cap:
            order = order[: args.episodes_cap]
        run_bc = run_sm = 0.0; n = 0; t0 = time.time()

        for key in order:
            episode = {k: v.to(device) for k, v in data.get(key).items()}
            T = episode["frame_embs"].shape[0]
            if T < 2:
                continue
            fused_all, delta_all = _episode_real_paths(episode, fusion, drift, device)
            preds = []
            for t in range(T):
                next_emb = trm(fused_all[t], delta_all[t],
                               episode["frame_embs"][t].unsqueeze(0))
                preds.append(planner(next_emb).squeeze(0))
            preds = torch.stack(preds, 0)
            loss = total_planner_loss(preds, episode["pwm_targets"], smooth_weight=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], args.grad_clip)
            opt.step()
            with torch.no_grad():
                run_bc += float(planner_bc_loss(preds, episode["pwm_targets"]))
                run_sm += float(smoothness_loss(preds))
            n += 1

        print(f"[stage B] epoch {epoch}/{args.stage_b_epochs} | bc {run_bc / max(n,1):.4f} "
              f"| smooth {run_sm / max(n,1):.4f} | {time.time()-t0:.0f}s", flush=True)
        save(args, cfg, "full_stageB.pt", fusion=fusion, drift=drift, trm=trm, planner=planner)


def save(args, cfg, name, **modules) -> None:
    out = Path(args.checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"cfg": dataclasses.asdict(cfg),
         "trm_d": args.trm_d,
         **{k: m.state_dict() for k, m in modules.items()}},
        out / name,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg: MicroVLAConfig = DEFAULT_CONFIG
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    print(f"training on {device} | data: {args.data_dir}")

    data = _MultiDataset(args.data_dir, args.val_frac, args.seed)
    print(f"episodes: train {len(data.train_index)}, val {len(data.val_index)}")

    fusion = SlotResonanceFusion(cfg).to(device)
    drift = AnchoredDriftEncoder(cfg).to(device)
    trm = RecursiveTRM(cfg, d=args.trm_d).to(device)
    planner = ChronoQueryPlanner(cfg).to(device)

    ckpt_a = Path(args.checkpoint_dir) / "full_stageA.pt"
    if args.resume and ckpt_a.exists():
        state = torch.load(ckpt_a, map_location=device, weights_only=True)
        fusion.load_state_dict(state["fusion"])
        drift.load_state_dict(state["drift"])
        trm.load_state_dict(state["trm"])
        print(f"resumed world model from {ckpt_a}")
    if args.stage_a_epochs > 0 and not (args.resume and ckpt_a.exists()):
        stage_a(args, cfg, data, fusion, drift, trm, device)
    if args.stage_b_epochs > 0:
        stage_b(args, cfg, data, fusion, drift, trm, planner, device)
    print("done.")


if __name__ == "__main__":
    main()
