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

    # Early-stopping run (recommended): horizon ramps 1->6 over 4 warmup epochs,
    # then holds at 6 and stops when val stops improving; best checkpoint kept.
    python train/train_full.py --data-dir data/bridge --data-dir data/libero \\
        --stage-a-epochs 30 --warmup-epochs 4 --max-horizon 6 --patience 3 \\
        --stage-b-epochs 3 --device auto

Ablations (paper.md E6/E7; ``--tag`` keeps their checkpoints from clobbering
the main run):

    python train/train_full.py --data-dir data/bridge --data-dir data/libero \\
        --ablate-evidence-fade --tag nofade --stage-a-epochs 2 --stage-b-epochs 3 --device auto

    python train/train_full.py --data-dir data/bridge --data-dir data/libero \\
        --ablate-grounding --tag noground --stage-a-epochs 2 --stage-b-epochs 3 --device auto

Checkpoints: ``checkpoints/full_stageA.pt`` / ``full_stageB.pt`` (all module
state_dicts + config + normalization pointer), or
``full_stageA_<tag>.pt`` / ``full_stageB_<tag>.pt`` when ``--tag`` is given.
``--resume`` reloads them (honoring ``--tag``).
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
    p.add_argument("--stage-a-epochs", type=int, default=2,
                   help="HARD CAP on stage-A epochs (safety). With early stopping on, "
                        "training usually stops well before this; set high (e.g. 30) for "
                        "an early-stop run. Set 0 to skip stage A.")
    p.add_argument("--warmup-epochs", type=int, default=4,
                   help="stage A: epochs over which the rollout horizon ramps 1->max-horizon; "
                        "after warmup, horizon holds at max and early stopping is armed.")
    p.add_argument("--patience", type=int, default=3,
                   help="stage A early stop: halt if val rollout loss does not improve by "
                        "--min-delta for this many consecutive AT-MAX-HORIZON epochs. 0 "
                        "disables early stopping (train the full --stage-a-epochs cap).")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="stage A early stop: minimum val-loss improvement to reset patience.")
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
    p.add_argument("--max-horizon", type=int, default=6,
                   help="stage A: max data-rate rollout horizon (grows 1->max across "
                        "epochs per TRM_SPEC section 5). Episodes shorter than horizon+1 "
                        "are skipped.")
    p.add_argument("--gamma", type=float, default=0.9,
                   help="stage A: discount for the per-step rollout loss sum_k gamma^(k-1)*L_k.")
    p.add_argument("--ablate-grounding", action="store_true",
                   help="E7 (paper.md Claim 6): zero box_weight and centers everywhere in "
                        "_episode_real_paths, training the frame-only variant (no boxes, no "
                        "geometry) at matched params.")
    p.add_argument("--ablate-evidence-fade", action="store_true",
                   help="E6 (paper.md Claim 4): sets cfg.modality_dropout=0 (via "
                        "dataclasses.replace) before building fusion, training without the "
                        "evidence-fade dream-regime regularizer.")
    p.add_argument("--tag", type=str, default="",
                   help="Suffix appended to checkpoint filenames (full_stageA_<tag>.pt / "
                        "full_stageB_<tag>.pt) so ablation runs do not clobber the main run.")
    return p.parse_args(argv)


def _tagged_name(name: str, tag: str) -> str:
    """Inserts an optional ``--tag`` suffix before a checkpoint's ``.pt`` extension.

    Args:
        name: Base checkpoint filename, e.g. ``"full_stageA.pt"``.
        tag: ``args.tag``; empty string leaves ``name`` unchanged.

    Returns:
        ``name`` with ``_<tag>`` inserted before the extension when ``tag`` is set.
    """
    if not tag:
        return name
    stem, ext = name.rsplit(".", 1)
    return f"{stem}_{tag}.{ext}"


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


def _episode_real_paths(episode, fusion, drift, device, ablate_grounding: bool = False):
    """Per-t grounded fused matrices + drift codes for one episode.

    Drift is stateful/sequential, so this always sweeps the full episode in
    order. Returns per-step tensors on ``device`` (batch dim 1).

    Args:
        ablate_grounding: E7 (paper.md Claim 6) ablation -- when True, the
            box_weight and both box centers are zeroed before fusion, so the
            box + geometry tokens contribute nothing and fusion trains on
            the frame embedding (+ text + last action) alone.
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
        box_weight = episode["box_weights"][t].unsqueeze(0)
        source_center = episode["source_centers"][t].unsqueeze(0)
        target_center = episode["target_centers"][t].unsqueeze(0)
        if ablate_grounding:
            box_weight = torch.zeros_like(box_weight)
            source_center = torch.zeros_like(source_center)
            target_center = torch.zeros_like(target_center)
        fused_all.append(fusion(
            text,
            episode["frame_embs"][t].unsqueeze(0),
            episode["source_box_embs"][t].unsqueeze(0),
            episode["target_box_embs"][t].unsqueeze(0),
            source_center,
            target_center,
            box_weight=box_weight,
            last_action=last_action,
        ))
        delta_all.append(drift(episode["frame_embs"][t].unsqueeze(0)))
    return fused_all, delta_all


def _rollout(episode, t, fused_t, delta_t, fusion, trm, cfg, horizon: int,
             gamma: float = 0.9, ablate_grounding: bool = False):
    """Data-rate multi-step rollout: predict frames[t+1..t+H], each supervised.

    This is the TRM_SPEC.md section-5 rollout objective — the one that actually
    tests and trains against COMPOUNDING error (an earlier version dreamed many
    internal ticks toward the single target frame[t+1] with no intermediate
    supervision, which drifted and never beat persistence). Here each step is
    one real-data (2 Hz) frame:

    * Step 1 is grounded (the real boxes/drift at t, ``fused_t``/``delta_t``).
    * Steps 2..H are open-loop dreams: the fed-back standardized prediction
      becomes the frame token, boxes are HELD from t at staleness-decayed
      weight (deployment has no future boxes), the executed action is the
      recorded ``pwm_targets`` (teacher-forced), and the drift code is held
      (no new measurement during a rollout — matches the deployment loop).

    Returns the discounted mean spec_loss over the H supervised steps. Caller
    guarantees ``t + horizon < T``.
    """
    text = episode["text_tokens"].unsqueeze(0)
    frames = episode["frame_embs"]
    T = frames.shape[0]

    def _boxes(idx, fade):
        z2 = frames.new_zeros(1, 2)
        if ablate_grounding:
            return (frames.new_zeros(1, cfg.vis_dim), frames.new_zeros(1, cfg.vis_dim),
                    z2, z2, frames.new_zeros(1, 2))
        return (episode["source_box_embs"][idx].unsqueeze(0),
                episode["target_box_embs"][idx].unsqueeze(0),
                episode["source_centers"][idx].unsqueeze(0),
                episode["target_centers"][idx].unsqueeze(0),
                episode["box_weights"][idx].unsqueeze(0) * fade)

    latent = frames[t].unsqueeze(0)
    ctx = [latent.squeeze(0)]
    fused_k, delta_k = fused_t, delta_t
    loss = torch.zeros((), device=frames.device)
    wsum = 0.0
    for k in range(1, horizon + 1):
        pred = trm(fused_k, delta_k, latent,
                   context=torch.stack(ctx[-cfg.context_window:], 0).unsqueeze(0))
        w = gamma ** (k - 1)
        loss = loss + w * spec_loss(pred, frames[t + k].unsqueeze(0))
        wsum += w
        if k == horizon:
            break
        # Feed back for the next step (open-loop dream).
        latent = standardize(pred)
        ctx.append(latent.squeeze(0).detach())
        sbe, tbe, sc, tc, bw = _boxes(t, cfg.staleness_decay ** k)
        act_idx = min(t + k, T - 1)
        last_action = episode["pwm_targets"][act_idx, 0].unsqueeze(0)
        fused_k = fusion(text, latent, sbe, tbe, sc, tc, box_weight=bw, last_action=last_action)
        # delta_k held: no measurement during the rollout.
    return loss / wsum


def _horizon_for_epoch(epoch: int, epochs: int, max_horizon: int) -> int:
    """Scheduled rollout horizon: grow 1 -> max_horizon across the epochs.

    TRM_SPEC.md section 5 mandates a curriculum (start 1-step, grow) so the
    world model learns single-step prediction before being asked to control
    compounding error over a long open-loop rollout.
    """
    # NOTE: "epochs" here is the WARMUP length (see _scheduled_horizon), not the
    # total run length, so the ramp is independent of early stopping.
    if epochs <= 1:
        return max_horizon
    if epoch >= epochs:
        return max_horizon
    frac = (epoch - 1) / (epochs - 1)
    return max(1, round(1 + (max_horizon - 1) * frac))


def _scheduled_horizon(epoch: int, warmup_epochs: int, max_horizon: int) -> int:
    """Rollout horizon for an epoch: ramp 1->max over warmup, then hold at max.

    Decoupled from the total run length so early stopping can add epochs at the
    max horizon without changing the curriculum.
    """
    return _horizon_for_epoch(epoch, warmup_epochs, max_horizon)


def _persistence_loss(episode, t, horizon, cfg, gamma) -> float:
    """Discounted 'predict no change' baseline over the same H-step horizon.

    Uses the exact discounted-mean normalization :func:`_rollout` uses, so the
    two numbers are directly comparable. The world model must beat this to
    have learned any scene dynamics.
    """
    frames = episode["frame_embs"]
    cur = frames[t].unsqueeze(0)
    loss, wsum = 0.0, 0.0
    for k in range(1, horizon + 1):
        w = gamma ** (k - 1)
        loss += w * float(spec_loss(cur, frames[t + k].unsqueeze(0)))
        wsum += w
    return loss / wsum


def stage_a(args, cfg, data, fusion, drift, trm, device) -> None:
    """World-model training: scheduled-horizon rollout (TRM_SPEC S5) + early stop.

    Horizon ramps 1->max over ``--warmup-epochs``, then holds at max. Once at
    max horizon, validation loss is monitored: the BEST checkpoint is kept, and
    training halts after ``--patience`` epochs without a ``--min-delta``
    improvement (or at the ``--stage-a-epochs`` hard cap). The saved
    ``full_stageA.pt`` is always the best-val model, not the last.
    """
    params = [*fusion.parameters(), *drift.parameters(), *trm.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    rng = random.Random(args.seed)
    ckpt_name = _tagged_name("full_stageA.pt", args.tag)

    best_val = float("inf")
    stale = 0
    for epoch in range(1, args.stage_a_epochs + 1):
        H = _scheduled_horizon(epoch, args.warmup_epochs, args.max_horizon)
        at_max = H >= args.max_horizon
        fusion.train(); drift.train(); trm.train()
        order = list(data.train_index)
        rng.shuffle(order)
        if args.episodes_cap:
            order = order[: args.episodes_cap]
        run_loss, n_seg, t0 = 0.0, 0, time.time()

        for key in order:
            episode = {k: v.to(device) for k, v in data.get(key).items()}
            T = episode["frame_embs"].shape[0]
            if T < H + 1:
                continue
            fused_all, delta_all = _episode_real_paths(
                episode, fusion, drift, device, ablate_grounding=args.ablate_grounding
            )
            ts = list(range(T - H))
            rng.shuffle(ts)
            ts = ts[: args.segments_per_episode]
            if not ts:
                continue

            opt.zero_grad()
            loss = torch.zeros((), device=device)
            for t in ts:
                loss = loss + _rollout(episode, t, fused_all[t], delta_all[t], fusion, trm,
                                       cfg, H, args.gamma, args.ablate_grounding)
            loss = loss / len(ts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run_loss += float(loss.detach()); n_seg += 1

        val, persistence = evaluate_world(args, cfg, data, fusion, drift, trm, device, H)
        verdict = "BEATS persistence" if val < persistence else "not yet below persistence"

        # Best-checkpoint + early stopping are only meaningful at the fixed max
        # horizon (during warmup, val rises as the task gets harder by design).
        tag = ""
        if not at_max:
            save(args, cfg, ckpt_name, fusion=fusion, drift=drift, trm=trm)  # keep last during ramp
        else:
            if val < best_val - args.min_delta:
                best_val = val
                stale = 0
                save(args, cfg, ckpt_name, fusion=fusion, drift=drift, trm=trm)  # keep BEST
                tag = " *best*"
            else:
                stale += 1
                tag = f" (no improve {stale}/{args.patience})"

        print(f"[stage A] epoch {epoch} | H={H} | train {run_loss / max(n_seg,1):.4f} "
              f"| val {val:.4f} vs persistence {persistence:.4f} ({verdict}){tag} "
              f"| {time.time()-t0:.0f}s", flush=True)

        if at_max and args.patience > 0 and stale >= args.patience:
            print(f"[stage A] early stop: no val improvement for {args.patience} epochs "
                  f"at H={args.max_horizon}. Best val {best_val:.4f}. Restoring best checkpoint.",
                  flush=True)
            break

    # Ensure the in-memory modules reflect the BEST saved checkpoint before
    # stage B trains on top of them.
    if Path(args.checkpoint_dir, ckpt_name).exists():
        st = torch.load(Path(args.checkpoint_dir, ckpt_name), map_location=device, weights_only=True)
        fusion.load_state_dict(st["fusion"]); drift.load_state_dict(st["drift"]); trm.load_state_dict(st["trm"])


@torch.no_grad()
def evaluate_world(args, cfg, data, fusion, drift, trm, device, horizon) -> tuple[float, float]:
    """Val H-step rollout loss AND the discounted persistence baseline.

    The pass/fail bar for "the world model learned anything": val loss must
    drop BELOW the persistence baseline — otherwise the TRM is just leaning
    on its residual skip connection.
    """
    fusion.eval(); drift.eval(); trm.eval()
    losses, base = [], []
    for key in data.val_index[:64]:
        episode = {k: v.to(device) for k, v in data.get(key).items()}
        T = episode["frame_embs"].shape[0]
        if T < horizon + 1:
            continue
        fused_all, delta_all = _episode_real_paths(
            episode, fusion, drift, device, ablate_grounding=args.ablate_grounding
        )
        for t in range(0, T - horizon, max(1, (T - horizon) // 4)):
            losses.append(float(_rollout(episode, t, fused_all[t], delta_all[t], fusion, trm,
                                         cfg, horizon, args.gamma, args.ablate_grounding)))
            base.append(_persistence_loss(episode, t, horizon, cfg, args.gamma))
    n = max(len(losses), 1)
    return sum(losses) / n, sum(base) / n


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
            fused_all, delta_all = _episode_real_paths(
                episode, fusion, drift, device, ablate_grounding=args.ablate_grounding
            )
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
        save(args, cfg, _tagged_name("full_stageB.pt", args.tag),
             fusion=fusion, drift=drift, trm=trm, planner=planner)


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
    if args.ablate_evidence_fade:
        # E6 (paper.md Claim 4): train without the evidence-fade dream-regime
        # regularizer. Must happen before SlotResonanceFusion is constructed
        # -- it reads cfg.modality_dropout once, at __init__.
        cfg = dataclasses.replace(cfg, modality_dropout=0.0)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    print(f"training on {device} | data: {args.data_dir}"
          f"{' | ablate-grounding' if args.ablate_grounding else ''}"
          f"{' | ablate-evidence-fade' if args.ablate_evidence_fade else ''}"
          f"{f' | tag={args.tag}' if args.tag else ''}")

    data = _MultiDataset(args.data_dir, args.val_frac, args.seed)
    print(f"episodes: train {len(data.train_index)}, val {len(data.val_index)}")

    fusion = SlotResonanceFusion(cfg).to(device)
    drift = AnchoredDriftEncoder(cfg).to(device)
    trm = RecursiveTRM(cfg, d=args.trm_d).to(device)
    planner = ChronoQueryPlanner(cfg).to(device)

    ckpt_a = Path(args.checkpoint_dir) / _tagged_name("full_stageA.pt", args.tag)
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
