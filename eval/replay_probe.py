"""Do the trained heads reproduce the demo actions on IN-DISTRIBUTION inputs?

The closed-loop eval fails with the arm drifting into a wall. That is either
(a) the heads never learned to reproduce the demo actions (BC / no-proprioception
drift), or (b) the heads are fine but LIVE eval perception is out-of-distribution
vs the baked training embeddings. This probe separates them WITHOUT the sim:

  * load one baked training episode (.npz of embeddings + recorded pwm_targets),
  * run the exact Stage-B forward (fusion -> drift -> TRM -> planner) on those
    in-distribution embeddings,
  * compare the emitted executed action (plan row 0) to the recorded demo action
    at every step, in the normalized space both live in.

Read the output:
  * low per-dim MAE + emitted mean ~= demo mean  -> heads REPRODUCE the demo on
    in-distribution inputs. The wall is LIVE PERCEPTION (train/eval mismatch).
  * emitted actions have a constant bias / don't track -> the heads DRIFT even
    in-distribution: BC-collapse / missing proprioception. No perception fix
    will help; the action interface / conditioning is the problem.

    python -m eval.replay_probe --checkpoint checkpoints/full_stageB.pt \
        --episode data/libero/<some_episode>.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.planner.chrono_planner import ChronoQueryPlanner


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", default="checkpoints/full_stageB.pt")
    ap.add_argument("--episode", required=True, help="a baked .npz training episode")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    from eval.policy import _load_relaxed  # tolerant loader (arch may have moved)
    from TRM import RecursiveTRM

    dev = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    cfg = MicroVLAConfig(**state["cfg"]) if "cfg" in state else DEFAULT_CONFIG

    fusion = SlotResonanceFusion(cfg).to(dev).eval()
    drift = AnchoredDriftEncoder(cfg).to(dev).eval()
    trm = RecursiveTRM(cfg, d=state.get("trm_d", 1024)).to(dev).eval()
    planner = ChronoQueryPlanner(cfg).to(dev).eval()
    _load_relaxed(fusion, state["fusion"], "fusion")
    _load_relaxed(drift, state["drift"], "drift")
    _load_relaxed(trm, state["trm"], "trm")
    if "planner" not in state:
        raise SystemExit("checkpoint has no planner (stage A only); pass a stage-B ckpt")
    _load_relaxed(planner, state["planner"], "planner")

    ep = np.load(args.episode)
    def t(k):
        return torch.as_tensor(ep[k], dtype=torch.float32, device=dev)
    frames = t("frame_embs")            # [T, 512]
    T = frames.shape[0]
    text = t("text_tokens").unsqueeze(0)  # [1, 3, 512]
    pwm = t("pwm_targets")              # [T, 5, 7]

    names = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "grip"]
    emitted, demo = [], []
    with torch.no_grad():
        drift.reset()
        # Precompute drift over the episode (teacher-forced last action, like Stage B).
        for i in range(T):
            cur = frames[i].unsqueeze(0)
            last_action = pwm[i - 1, 0].unsqueeze(0) if i > 0 else pwm.new_zeros(1, cfg.num_servos)
            fused = fusion(text, cur, t("source_box_embs")[i].unsqueeze(0),
                           t("target_box_embs")[i].unsqueeze(0),
                           t("source_centers")[i].unsqueeze(0),
                           t("target_centers")[i].unsqueeze(0),
                           box_weight=t("box_weights")[i].unsqueeze(0),
                           last_action=last_action)
            delta = drift(cur)
            next_emb, next_box = trm(fused, delta, cur, return_box=True)
            geom = torch.cat([t("source_centers")[i], t("target_centers")[i],
                              t("box_weights")[i]]).unsqueeze(0)  # [1, 6]
            plan = planner(next_emb, current_emb=cur, state_delta=delta,
                           fused=fused, pred_box_emb=next_box, geometry=geom)  # [1,5,7]
            emitted.append(plan[0, 0].cpu().numpy())   # executed action (row 0)
            demo.append(pwm[i, 0].cpu().numpy())        # demo action at this step

    E = np.stack(emitted)   # [T, 7]
    D = np.stack(demo)      # [T, 7]
    mae = np.abs(E - D).mean(axis=0)

    print(f"\nepisode {Path(args.episode).name}: {T} steps  (normalized action space)\n")
    print(f"{'dim':8s} {'emit_mean':>10s} {'demo_mean':>10s} {'emit_std':>9s} "
          f"{'demo_std':>9s} {'MAE':>7s} {'corr':>7s}")
    for i, nm in enumerate(names):
        e, d = E[:, i], D[:, i]
        corr = float(np.corrcoef(e, d)[0, 1]) if e.std() > 1e-6 and d.std() > 1e-6 else float("nan")
        print(f"{nm:8s} {e.mean():>10.3f} {d.mean():>10.3f} {e.std():>9.3f} "
              f"{d.std():>9.3f} {mae[i]:>7.3f} {corr:>7.3f}")

    pose_mae = float(mae[:6].mean())
    # A constant emitted mean far from the demo mean, with near-zero corr, = drift/bias.
    bias = float(np.abs(E[:, :6].mean(axis=0) - D[:, :6].mean(axis=0)).mean())
    print("\n=== verdict ===")
    if pose_mae < 0.15 and bias < 0.1:
        print("  HEADS REPRODUCE THE DEMO on in-distribution baked embeddings "
              f"(pose MAE {pose_mae:.3f}, bias {bias:.3f}). The heads are fine -> "
              "the closed-loop failure is LIVE PERCEPTION being out-of-distribution "
              "vs the baked training embeddings. Fix the train/eval perception gap.")
    else:
        print(f"  HEADS DO NOT TRACK THE DEMO even in-distribution (pose MAE "
              f"{pose_mae:.3f}, mean bias {bias:.3f}). The action interface / "
              "conditioning is the problem (BC-collapse / no proprioception / "
              "normalization bias), NOT perception. Compare emit_mean vs demo_mean "
              "above: a constant offset with low corr is the drift you see in sim.")


if __name__ == "__main__":
    main()
