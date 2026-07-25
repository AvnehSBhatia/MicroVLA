"""Wind-tunnel benchmark: sub-second per-episode fidelity evals, no simulator.

LIBERO closed-loop is the ground truth but costs minutes per episode (mujoco +
osmesa). This bench is the fast inner loop: it replays baked episodes through
the FULL stage-B forward (fusion -> drift -> TRM -> planner, teacher-forced)
and scores the metrics that predicted every closed-loop failure so far:

  * std_ratio   emitted-action std / demo std (pose dims). THE collapse metric
                (~0.12 across v4-v6 = the diagnosed 8x timidity; healthy ~1.0).
  * pose_mae    per-step action error in normalized units.
  * corr        mean per-dim correlation with the demo actions (direction).
  * grip_acc    gripper open/close agreement.
  * wm_margin   world-model H-step rollout improvement over persistence
                (positive = the TRM predicts real dynamics).

Each episode is one "eval": ~25 ms/step on an M-series CPU -> ~0.3-0.5 s per
episode, 30 episodes in ~10-15 s, zero sim deps. Use it to iterate on
architecture/training and reserve LIBERO (--workers N) for confirmation runs.

    # real data (box, or wherever npzs live):
    python -m eval.bench --checkpoint checkpoints/full_stageB.pt --data-dir data/libero

    # no data present (fresh Mac clone): deterministic synthetic episodes
    python -m eval.bench --checkpoint none --synthetic 30

Honest caveat: this is OPEN-LOOP fidelity, not task success — it cannot see
compounding closed-loop error. It is a necessary-not-sufficient gate: a policy
that fails here will fail LIBERO; one that passes has earned a sim run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.utils.embedding import standardize


def _episode_metrics(ep: dict, mods: dict, cfg: MicroVLAConfig, horizon: int) -> dict:
    """Replays one episode through the stage-B forward; returns its scorecard."""
    from TRM import spec_loss

    fusion, drift, trm, planner = mods["fusion"], mods["drift"], mods["trm"], mods["planner"]
    frames = ep["frame_embs"]
    T = frames.shape[0]
    text = ep["text_tokens"].unsqueeze(0)
    pwm = ep["pwm_targets"]

    emitted, demo, grip_hits = [], [], []
    wm_loss = pers_loss = 0.0
    n_wm = 0
    with torch.no_grad():
        drift.reset()
        fused_all, delta_all = [], []
        for i in range(T):
            cur = frames[i].unsqueeze(0)
            last_action = pwm[i - 1, 0].unsqueeze(0) if i > 0 else pwm.new_zeros(1, cfg.num_servos)
            fused = fusion(text, cur, ep["source_box_embs"][i].unsqueeze(0),
                           ep["target_box_embs"][i].unsqueeze(0),
                           ep["source_centers"][i].unsqueeze(0),
                           ep["target_centers"][i].unsqueeze(0),
                           box_weight=ep["box_weights"][i].unsqueeze(0),
                           last_action=last_action)
            delta = drift(cur)
            fused_all.append(fused); delta_all.append(delta)

            wm = trm.forward_full(fused, delta, cur)
            next_emb, next_box = wm["next_emb"], wm["next_box"]
            geom = torch.cat([ep["source_centers"][i], ep["target_centers"][i],
                              ep["box_weights"][i]]).unsqueeze(0)
            plan, grip_logit = planner(
                next_emb, current_emb=cur, state_delta=delta, fused=fused,
                pred_box_emb=next_box, geometry=geom,
                proprio=ep["proprio"][i].unsqueeze(0), wm_msg=wm["msg"],
                return_aux=True,
            )
            emitted.append(plan[0, 0].numpy())
            demo.append(pwm[i, 0].numpy())
            grip_hits.append(float((grip_logit[0, 0] > 0) == (pwm[i, 0, -1] > 0)))

        # World-model margin: H-step open-loop rollout vs persistence, a few
        # anchors per episode (matches the training objective's data-rate form).
        H = min(horizon, T - 1)
        if H >= 1:
            for t0 in range(0, T - H, max(1, (T - H) // 3)):
                latent = frames[t0].unsqueeze(0)
                fused_k, delta_k = fused_all[t0], delta_all[t0]
                ctx = [latent.squeeze(0)]
                for k in range(1, H + 1):
                    pred = trm(fused_k, delta_k, latent,
                               context=torch.stack(ctx[-cfg.context_window:], 0).unsqueeze(0))
                    wm_loss += float(spec_loss(pred, frames[t0 + k].unsqueeze(0)))
                    pers_loss += float(spec_loss(frames[t0].unsqueeze(0),
                                                 frames[t0 + k].unsqueeze(0)))
                    n_wm += 1
                    latent = standardize(pred)
                    ctx.append(latent.squeeze(0))

    E, D = np.stack(emitted), np.stack(demo)
    pose_e, pose_d = E[:, :-1], D[:, :-1]
    d_std = pose_d.std(axis=0)
    ratio = pose_e.std(axis=0)[d_std > 1e-6] / d_std[d_std > 1e-6]
    corrs = []
    for i in range(pose_e.shape[1]):
        if pose_e[:, i].std() > 1e-6 and pose_d[:, i].std() > 1e-6:
            corrs.append(float(np.corrcoef(pose_e[:, i], pose_d[:, i])[0, 1]))
    return {
        "T": int(T),
        "std_ratio": float(np.median(ratio)) if ratio.size else float("nan"),
        "pose_mae": float(np.abs(pose_e - pose_d).mean()),
        "corr": float(np.mean(corrs)) if corrs else float("nan"),
        "grip_acc": float(np.mean(grip_hits)),
        "wm_margin": float((pers_loss - wm_loss) / pers_loss) if pers_loss > 0 else float("nan"),
    }


def _episode_sensitivity(ep: dict, mods: dict, cfg: MicroVLAConfig) -> dict:
    """ON-DISTRIBUTION input sensitivity: mean |Δplan| when withholding inputs.

    The interpretability probe (random inputs) found fused 7x dominant with
    geometry/next_emb functionally dead. This is the same measurement on real
    episode data, re-runnable after every retrain: for each step, compute the
    plan with all inputs, then with each input withheld (optional inputs ->
    None; next_emb -> current_emb, i.e. a persistence 'no-prediction'), and
    average the plan change. Healthy training (planner input-dropout) should
    pull geometry/next_emb well off zero.
    """
    fusion, drift, trm, planner = mods["fusion"], mods["drift"], mods["trm"], mods["planner"]
    frames = ep["frame_embs"]
    T = frames.shape[0]
    text = ep["text_tokens"].unsqueeze(0)
    pwm = ep["pwm_targets"]
    deltas: dict[str, list[float]] = {k: [] for k in
        ("fused", "current_emb", "state_delta", "geometry", "proprio",
         "pred_box_emb", "wm_msg", "next_emb->cur")}
    with torch.no_grad():
        drift.reset()
        for i in range(T):
            cur = frames[i].unsqueeze(0)
            last_action = pwm[i - 1, 0].unsqueeze(0) if i > 0 else pwm.new_zeros(1, cfg.num_servos)
            fused = fusion(text, cur, ep["source_box_embs"][i].unsqueeze(0),
                           ep["target_box_embs"][i].unsqueeze(0),
                           ep["source_centers"][i].unsqueeze(0),
                           ep["target_centers"][i].unsqueeze(0),
                           box_weight=ep["box_weights"][i].unsqueeze(0),
                           last_action=last_action)
            delta = drift(cur)
            wm = trm.forward_full(fused, delta, cur)
            next_emb, next_box = wm["next_emb"], wm["next_box"]
            geom = torch.cat([ep["source_centers"][i], ep["target_centers"][i],
                              ep["box_weights"][i]]).unsqueeze(0)
            prop = ep["proprio"][i].unsqueeze(0)
            kw = dict(current_emb=cur, state_delta=delta, fused=fused,
                      pred_box_emb=next_box, geometry=geom, proprio=prop,
                      wm_msg=wm["msg"])
            base = planner(next_emb, **kw)
            for name in ("fused", "current_emb", "state_delta", "geometry",
                         "proprio", "pred_box_emb", "wm_msg"):
                alt = planner(next_emb, **{**kw, name: None})
                deltas[name].append(float((alt - base).abs().mean()))
            alt = planner(cur, **kw)  # no-prediction: next_emb := current_emb
            deltas["next_emb->cur"].append(float((alt - base).abs().mean()))
    return {k: float(np.mean(v)) for k, v in deltas.items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", default="checkpoints/full_stageB.pt",
                    help="stage-B checkpoint, or 'none' for fresh (untrained) modules")
    ap.add_argument("--data-dir", action="append", default=[],
                    help="baked npz dir(s); omit to use --synthetic")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="use N deterministic synthetic episodes (no data needed)")
    ap.add_argument("--episodes", type=int, default=30, help="max episodes to bench")
    ap.add_argument("--horizon", type=int, default=6, help="world-model rollout depth")
    ap.add_argument("--sensitivity", action="store_true",
                    help="also report on-distribution planner input sensitivity "
                         "(mean |dPlan| per withheld input) over the benched episodes")
    ap.add_argument("--out", default="eval_results/bench.json")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    cfg = DEFAULT_CONFIG
    mods = {
        "fusion": SlotResonanceFusion(cfg), "drift": AnchoredDriftEncoder(cfg),
        "planner": ChronoQueryPlanner(cfg),
    }
    if str(args.checkpoint).lower() != "none":
        from eval.policy import _load_relaxed
        from TRM import RecursiveTRM

        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "cfg" in state:
            cfg = MicroVLAConfig(**state["cfg"])
        mods["trm"] = RecursiveTRM(cfg, d=state.get("trm_d", 1024))
        for name in ("fusion", "drift", "trm", "planner"):
            if name in state or name in mods:
                target = mods.get(name)
                if target is None:
                    continue
                if name in state:
                    _load_relaxed(target, state[name], name)
    else:
        from microvla.trm.mock_trm import MockTRM

        mods["trm"] = MockTRM(cfg)
    if "trm" not in mods:  # checkpoint without trm key (shouldn't happen)
        from microvla.trm.mock_trm import MockTRM
        mods["trm"] = MockTRM(cfg)
    for m in mods.values():
        m.eval()

    # Episodes: real npz dirs, else synthetic.
    eps: list[tuple[str, dict]] = []
    if args.data_dir:
        from train.dataset import EpisodeDataset

        for d in args.data_dir:
            ds = EpisodeDataset(d)
            for k in range(len(ds)):
                if len(eps) >= args.episodes:
                    break
                eps.append((ds.files[k].name, ds[k]))
    n_syn = args.synthetic if args.synthetic else (args.episodes if not eps else 0)
    if n_syn:
        from train.dataset import make_synthetic_episode

        for s in range(n_syn):
            ep = {k: torch.as_tensor(v, dtype=torch.float32)
                  for k, v in make_synthetic_episode(14, cfg, seed=s).items()}
            eps.append((f"synthetic_{s}", ep))
    eps = eps[: args.episodes]

    rows = []
    t0 = time.time()
    for name, ep in eps:
        t1 = time.time()
        m = _episode_metrics(ep, mods, cfg, args.horizon)
        m["episode"] = name
        m["sec"] = round(time.time() - t1, 3)
        rows.append(m)
    wall = time.time() - t0

    def med(k):
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.median(vals)) if vals else float("nan")

    agg = {"episodes": len(rows), "wall_sec": round(wall, 2),
           "sec_per_eval": round(wall / max(len(rows), 1), 3),
           "std_ratio": med("std_ratio"), "pose_mae": med("pose_mae"),
           "corr": med("corr"), "grip_acc": med("grip_acc"),
           "wm_margin": med("wm_margin")}

    print(f"\n{'episode':44s} {'std_ratio':>9s} {'mae':>6s} {'corr':>6s} "
          f"{'grip':>5s} {'wm+%':>6s} {'sec':>5s}")
    for r in rows:
        print(f"{r['episode'][:44]:44s} {r['std_ratio']:>9.3f} {r['pose_mae']:>6.3f} "
              f"{r['corr']:>6.2f} {r['grip_acc']:>5.2f} {100*r['wm_margin']:>6.1f} {r['sec']:>5.2f}")
    print(f"\nAGGREGATE (median): std_ratio {agg['std_ratio']:.3f} | mae {agg['pose_mae']:.3f} "
          f"| corr {agg['corr']:.2f} | grip {agg['grip_acc']:.2f} "
          f"| wm_margin {100*agg['wm_margin']:.1f}% | {agg['sec_per_eval']:.2f}s/eval")
    print("read: std_ratio ~1.0 = healthy magnitude (collapse shows as ~0.1); "
          "wm_margin > 0 = world model beats persistence.")

    sens = None
    if args.sensitivity:
        per_ep = [_episode_sensitivity(ep, mods, cfg) for _, ep in eps[: min(len(eps), 10)]]
        sens = {k: float(np.mean([e[k] for e in per_ep])) for k in per_ep[0]} if per_ep else {}
        print("\nplanner input sensitivity (mean |dPlan| when withheld; on-distribution):")
        for k, v in sorted(sens.items(), key=lambda kv: -kv[1]):
            print(f"  {k:16s} {v:.4f}")
        print("read: dead paths sit at ~0; planner-input-dropout training should "
              "lift geometry / next_emb->cur / pred_box_emb off the floor.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"aggregate": agg, "episodes": rows, "sensitivity": sens}, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
