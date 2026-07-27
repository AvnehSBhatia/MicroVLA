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
from microvla.utils.signals import ignore_sigterm
from microvla.utils.waypoint import long_horizon_targets, waypoint_targets


def _rel_tokens(mods, ep, i, next_emb, cfg, last_action):
    """Relational tokens for one bench step, or None on a v7 stack.

    Prefers the baked class-agnostic scene (``obj_*``) and falls back to the two
    role slots, exactly as the trainer does — scoring a v8 checkpoint on
    different evidence than it trained on is the same class of error as the
    camera mismatch.
    """
    rel = mods.get("relational")
    if rel is None:
        return None
    import torch as _t

    from microvla.v8 import pack_objects

    if "obj_embs" in ep:
        obj = ep["obj_embs"][i].unsqueeze(0)
        ctr = ep["obj_centers"][i].unsqueeze(0)
        w = ep["obj_weights"][i].unsqueeze(0)
    else:
        obj, ctr, w = pack_objects(
            ep["source_box_embs"][i].unsqueeze(0), ep["target_box_embs"][i].unsqueeze(0),
            ep["source_centers"][i].unsqueeze(0), ep["target_centers"][i].unsqueeze(0),
            ep["box_weights"][i].unsqueeze(0), cfg)
    return rel(next_emb, obj, ctr, w, ep["text_tokens"].unsqueeze(0),
               last_action=last_action)


def _spatial_at(ep: dict, i: int, mods: dict) -> dict | None:
    """TQSA output for step ``i``, or None when the bench runs spatial-free.

    Mirrors the deployment path (``JEPALoop``): frozen backbone map -> trainable
    adapter, conditioned on the episode's text tokens.
    """
    tqsa, backbone = mods.get("tqsa"), mods.get("backbone")
    if tqsa is None or backbone is None or "wrist_frames" not in ep:
        return None
    frame = ep["wrist_frames"][i].cpu().numpy()          # [H, W, 3] RGB uint8
    fmap = backbone.feature_maps([frame[..., ::-1]]).to(ep["frame_embs"].device)
    return tqsa(fmap, ep["text_tokens"].unsqueeze(0))


def _episode_metrics(ep: dict, mods: dict, cfg: MicroVLAConfig, horizon: int) -> dict:
    """Replays one episode through the stage-B forward; returns its scorecard."""
    from TRM import spec_loss

    fusion, drift, trm, planner = mods["fusion"], mods["drift"], mods["trm"], mods["planner"]
    frames = ep["frame_embs"]
    T = frames.shape[0]
    text = ep["text_tokens"].unsqueeze(0)
    pwm = ep["pwm_targets"]

    emitted, demo, grip_hits = [], [], []
    wp_pred, wp_true = [], []
    # Score the head against the supervision it was TRAINED on. A long-horizon
    # head predicts 0.5-2.5 s of displacement; comparing that to native-spaced
    # 0.05-0.20 s targets reports a ~10x scale error as if it were prediction
    # error (observed: wp_std_ratio 3.95, wp_mae 116 mm on a working head).
    wp_tgt_all = wp_mask_all = None
    if "eef_pos_chunk" in ep:
        if getattr(cfg, "waypoint_long", False):
            wp_tgt_all, wp_mask_all = long_horizon_targets(
                ep["eef_pos_chunk"].unsqueeze(0), cfg.plan_steps, cfg.waypoint_range)
            wp_tgt_all, wp_mask_all = wp_tgt_all[0], wp_mask_all[0]   # [T, rows, 3], [T, rows]
        else:
            tg, rm = waypoint_targets(ep["eef_pos_chunk"], cfg.plan_steps, cfg.waypoint_range)
            wp_tgt_all, wp_mask_all = tg, rm.expand(tg.shape[0], -1)
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
            plan, grip_logit, wp = planner(
                next_emb, current_emb=cur, state_delta=delta, fused=fused,
                pred_box_emb=next_box, geometry=geom,
                proprio=ep["proprio"][i].unsqueeze(0), wm_msg=wm["msg"],
                wm_latent=wm.get("latent"), spatial=_spatial_at(ep, i, mods),
                relational=_rel_tokens(mods, ep, i, next_emb, cfg, last_action),
                return_wp=True,
            )
            emitted.append(plan[0, 0].cpu().numpy())
            demo.append(pwm[i, 0].cpu().numpy())
            grip_hits.append(float((grip_logit[0, 0] > 0) == (pwm[i, 0, -1] > 0)))
            # v7.2: the waypoint head's own fidelity, in METRES — no normalizer
            # and no gain needed, so it isolates the design claim (positions
            # regress with less shrinkage than noisy teleop actions). Skipped
            # for steps with no real proprio (zero-filled -> a fake "no motion").
            if (wp is not None and wp_tgt_all is not None
                    and float(ep["proprio"][i, -1]) > 0.5):
                # Score the row the ACTUATOR actually servoes toward, which
                # WaypointActuator derives from cfg.waypoint_horizon and clamps
                # to the last SUPERVISED row. Scoring row 0 while the controller
                # used row 3 described a prediction nobody executes.
                row = max(0, min(cfg.waypoint_horizon, cfg.plan_steps - 1) - 1)
                if float(wp_mask_all[i, row]) > 0:
                    wp_pred.append(wp[0, row].cpu().numpy() * cfg.waypoint_range)
                    wp_true.append(wp_tgt_all[i, row].cpu().numpy() * cfg.waypoint_range)

        # World-model margin: H-step open-loop rollout vs persistence, a few
        # anchors per episode — TRAINING-PROTOCOL-MATCHED (the predicted latent
        # feeds back through fusion with held boxes at staleness-faded weight,
        # exactly like train_batched.rollout / the deployment dream path).
        # Holding fused frozen at the anchor (the old bench) is a harsher
        # protocol than anything the model was trained or deployed under.
        H = min(horizon, T - 1)
        if H >= 1:
            text = ep["text_tokens"].unsqueeze(0)
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
                    if k == H:
                        break
                    latent = standardize(pred)
                    ctx.append(latent.squeeze(0))
                    fade = cfg.staleness_decay ** k
                    act_idx = min(t0 + k, T - 1)
                    fused_k = fusion(
                        text, latent,
                        ep["source_box_embs"][t0].unsqueeze(0),
                        ep["target_box_embs"][t0].unsqueeze(0),
                        ep["source_centers"][t0].unsqueeze(0),
                        ep["target_centers"][t0].unsqueeze(0),
                        box_weight=ep["box_weights"][t0].unsqueeze(0) * fade,
                        last_action=pwm[act_idx, 0].unsqueeze(0),
                    )

    E, D = np.stack(emitted), np.stack(demo)
    pose_e, pose_d = E[:, :-1], D[:, :-1]
    d_std = pose_d.std(axis=0)
    ratio = pose_e.std(axis=0)[d_std > 1e-6] / d_std[d_std > 1e-6]
    corrs = []
    for i in range(pose_e.shape[1]):
        if pose_e[:, i].std() > 1e-6 and pose_d[:, i].std() > 1e-6:
            corrs.append(float(np.corrcoef(pose_e[:, i], pose_d[:, i])[0, 1]))
    out = {
        "T": int(T),
        "std_ratio": float(np.median(ratio)) if ratio.size else float("nan"),
        "pose_mae": float(np.abs(pose_e - pose_d).mean()),
        "corr": float(np.mean(corrs)) if corrs else float("nan"),
        "grip_acc": float(np.mean(grip_hits)),
        "wm_margin": float((pers_loss - wm_loss) / pers_loss) if pers_loss > 0 else float("nan"),
        "wp_std_ratio": float("nan"),
        "wp_mae_mm": float("nan"),
    }
    if len(wp_pred) > 1:
        WP, WT = np.stack(wp_pred), np.stack(wp_true)
        t_std = WT.std(axis=0)
        r = WP.std(axis=0)[t_std > 1e-9] / t_std[t_std > 1e-9]
        out["wp_std_ratio"] = float(np.median(r)) if r.size else float("nan")
        out["wp_mae_mm"] = float(np.abs(WP - WT).mean() * 1000.0)
    return out


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
         "pred_box_emb", "wm_msg", "wm_latent", "spatial", "relational",
         "next_emb->cur", "next_emb->stale")}
    grip_flips: dict[str, list[float]] = {k: [] for k in deltas}
    prev_next_emb = None
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
                      wm_msg=wm["msg"], wm_latent=wm.get("latent"),
                      spatial=_spatial_at(ep, i, mods),
                      relational=_rel_tokens(mods, ep, i, next_emb, cfg, last_action))
            base = planner(next_emb, **kw)

            def _delta(alt):
                """Pose-only |dplan| and the gripper-flip RATE, separately.

                The plan's last column is a hard +/-1, so one flipped gripper
                decision at every step contributes exactly
                plan_steps*2/(plan_steps*num_servos) = 0.2857 to a whole-plan
                mean — the same magnitude as the largest sensitivity ever
                measured here (state_delta 0.2740). Averaging a discrete bit
                together with continuous pose made those two indistinguishable,
                so they are reported apart.
                """
                pose = float((alt[..., :-1] - base[..., :-1]).abs().mean())
                flip = float((alt[..., -1] != base[..., -1]).float().mean())
                return pose, flip
            probes = ["fused", "current_emb", "state_delta", "geometry",
                      "proprio", "pred_box_emb", "wm_msg", "wm_latent"]
            if kw["spatial"] is not None:
                probes.append("spatial")
            # v8's relational tokens carry ALL object evidence, so withholding
            # them is the direct test of whether the redesign is used at all.
            if kw.get("relational") is not None:
                probes.append("relational")
            for name in probes:
                pose, flip = _delta(planner(next_emb, **{**kw, name: None}))
                deltas[name].append(pose)
                grip_flips[name].append(flip)
            pose, flip = _delta(planner(cur, **kw))  # no-prediction: next_emb := cur
            deltas["next_emb->cur"].append(pose)
            grip_flips["next_emb->cur"].append(flip)
            # The ->cur probe zeroes the TRM's RESIDUAL, which is small next to
            # ||current_emb||, so a near-zero reading is partly an amplitude
            # artifact. ->stale swaps in the PREVIOUS tick's prediction: a
            # full-magnitude, in-distribution wrong answer. If that is also ~0,
            # the predicted-embedding path really is dead.
            if prev_next_emb is not None:
                pose, flip = _delta(planner(prev_next_emb, **kw))
                deltas["next_emb->stale"].append(pose)
                grip_flips["next_emb->stale"].append(flip)
            prev_next_emb = next_emb
    return ({k: float(np.mean(v)) for k, v in deltas.items() if v},
            {k: float(np.mean(v)) for k, v in grip_flips.items() if v})


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
    ap.add_argument("--sensitivity-episodes", type=int, default=10,
                    help="episodes used for the sensitivity probe (it costs 9 extra "
                         "planner forwards per step, hence a smaller default than --episodes).")
    ap.add_argument("--sensitivity", action="store_true",
                    help="also report on-distribution planner input sensitivity "
                         "(mean |dPlan| per withheld input) over the benched episodes")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default) or cuda:0 — the d=1024 TRM dominates cost; "
                         "GPU cuts a contended-CPU 75s/eval to ~1s")
    ap.add_argument("--tqsa", action="store_true",
                    help="run the v7 spatial adapter, so a TQSA-trained checkpoint is "
                         "scored with the observation it was TRAINED on (the planner takes "
                         "~27%% of its memory tokens from it). Needs ultralytics and baked "
                         "wrist_frames, and runs the frozen backbone once per step — far "
                         "slower than the default wind-tunnel pass.")
    ap.add_argument("--out", default="eval_results/bench.json")
    args = ap.parse_args(argv)
    ignore_sigterm()

    dev = torch.device(args.device)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # Resolve cfg from the checkpoint BEFORE constructing anything. cfg now
    # carries fields that change ARCHITECTURE — `waypoint_action` builds the
    # waypoint head, `planner_inputs` selects which memory projections exist —
    # so building from DEFAULT_CONFIG and loading afterwards silently drops
    # whatever the checkpoint actually trained (observed: `dropped=
    # ['wp_disp_head.weight', 'wp_disp_head.bias']`, and wp_std_ratio nan).
    cfg = DEFAULT_CONFIG
    state: dict = {}
    if str(args.checkpoint).lower() != "none":
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "cfg" in state:
            cfg = MicroVLAConfig(**state["cfg"])

    # v8 checkpoints carry a `relational` state_dict and were trained with the
    # HRM/evidence adapters in the fusion and drift slots. Detected from the
    # checkpoint's own keys, not a flag: building the v7 modules against v8
    # weights would load nothing and score a randomly-initialised stack, which
    # is exactly the class of silent mis-build that voided the TQSA numbers.
    is_v8 = bool(state) and "relational" in state
    if is_v8:
        from microvla.relational import RelationalHead
        from microvla.v8 import DriftAdapter, FusionAdapter

        mods = {"fusion": FusionAdapter(cfg), "drift": DriftAdapter(cfg),
                "planner": ChronoQueryPlanner(cfg),
                "relational": RelationalHead(cfg)}
    else:
        mods = {
            "fusion": SlotResonanceFusion(cfg), "drift": AnchoredDriftEncoder(cfg),
            "planner": ChronoQueryPlanner(cfg),
        }
    if state:
        from eval.policy import _load_relaxed
        from TRM import RecursiveTRM

        mods["trm"] = RecursiveTRM(cfg, d=state.get("trm_d", 1024))
        for name in ("fusion", "drift", "trm", "planner", "relational"):
            if name in state and name in mods:
                _load_relaxed(mods[name], state[name], name)
        if is_v8:
            print(f"v8 checkpoint: relational head loaded "
                  f"({sum(p.numel() for p in mods['relational'].parameters()):,} params)",
                  flush=True)
    else:
        from microvla.trm.mock_trm import MockTRM

        mods["trm"] = MockTRM(cfg)
    # The planner takes 22 of its ~82 memory tokens from TQSA, and this bench
    # never passed `spatial=` — so a TQSA-TRAINED checkpoint has been scored
    # with a quarter of its observation withheld, sensitivity ranking included.
    # Loud, because the affected numbers are already written down.
    if args.tqsa:
        from microvla.perception.spatial_adapter import TextQueriedSpatialAdapter
        from microvla.perception.yolo_world import YoloWorldPerception

        tqsa = TextQueriedSpatialAdapter(cfg)
        if "tqsa" in state:
            from eval.policy import _load_relaxed
            _load_relaxed(tqsa, state["tqsa"], "tqsa")
        else:
            print("WARNING: --tqsa but this checkpoint has no TQSA weights — "
                  "the adapter is at RANDOM INIT and these numbers are noise.")
        mods["tqsa"] = tqsa
        mods["backbone"] = YoloWorldPerception(device=str(dev))
    elif "tqsa" in state:
        print("NOTE: this checkpoint has TRAINED TQSA weights but bench is "
              "running WITHOUT the spatial adapter — the planner is missing "
              "~27% of its memory tokens, so these numbers UNDERSTATE it. "
              "Pass --tqsa (needs ultralytics + baked wrist_frames) for the "
              "honest figure; without it, only compare against other "
              "no-spatial runs.")

    for name, m in mods.items():
        if name != "backbone":
            m.to(dev).eval()

    # Episodes: real npz dirs, else synthetic.
    eps: list[tuple[str, dict]] = []
    if args.data_dir:
        from train.dataset import EpisodeDataset

        for d in args.data_dir:
            ds = EpisodeDataset(d, load_frames=args.tqsa)
            for k in range(len(ds)):
                if len(eps) >= args.episodes:
                    break
                # Frames stay uint8 on CPU (the backbone takes numpy); every
                # other key goes to the eval device.
                item = {kk: (vv if kk == "wrist_frames" else vv.to(dev))
                        for kk, vv in ds[k].items()}
                if not args.tqsa:
                    item.pop("wrist_frames", None)
                eps.append((ds.files[k].name, item))
    n_syn = args.synthetic if args.synthetic else (args.episodes if not eps else 0)
    if n_syn:
        from train.dataset import make_synthetic_episode

        for s in range(n_syn):
            ep = {k: torch.as_tensor(v, dtype=torch.float32).to(dev)
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
           "wm_margin": med("wm_margin"),
           "wp_std_ratio": med("wp_std_ratio"), "wp_mae_mm": med("wp_mae_mm")}

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
    if np.isfinite(agg["wp_std_ratio"]):
        print(f"waypoint head (v7.2): std_ratio {agg['wp_std_ratio']:.3f} | "
              f"mae {agg['wp_mae_mm']:.1f} mm  —  compare against the action "
              f"std_ratio above: the lever only pays if positions regress with "
              f"less shrinkage than actions do.")

    sens = None
    if args.sensitivity:
        per_ep = [_episode_sensitivity(ep, mods, cfg)
                  for _, ep in eps[: min(len(eps), args.sensitivity_episodes)]]
        pose = [a for a, _ in per_ep]
        grip = [b for _, b in per_ep]
        sens = {k: float(np.mean([e[k] for e in pose if k in e])) for k in pose[0]} if pose else {}
        gflip = {k: float(np.mean([e[k] for e in grip if k in e])) for k in grip[0]} if grip else {}
        print("\nplanner input sensitivity when the input is WITHHELD "
              f"({len(per_ep)} episodes):")
        print(f"  {'input':16s} {'POSE |dplan|':>12s} {'grip flip %':>12s}")
        for k, v in sorted(sens.items(), key=lambda kv: -kv[1]):
            print(f"  {k:16s} {v:12.4f} {100*gflip.get(k, float('nan')):11.1f}%")
        print("read: POSE and the GRIPPER are reported SEPARATELY on purpose. The plan's\n"
              "last column is a hard +/-1, so one flipped gripper decision per step adds\n"
              f"exactly {cfg.plan_steps*2/(cfg.plan_steps*cfg.num_servos):.4f} to a whole-plan mean — "
              "indistinguishable from the largest\npose effect ever measured here. Earlier combined "
              "readings (paper.md 4g) mixed\nthe two and must be re-measured with this build.")
        sens = {"pose": sens, "grip_flip_rate": gflip}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"aggregate": agg, "episodes": rows, "sensitivity": sens}, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
