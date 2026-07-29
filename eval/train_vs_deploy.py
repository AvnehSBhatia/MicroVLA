"""A/B one corpus episode through the TRAINER path and the DEPLOYMENT path.

paper.md 4u narrowed the closed-loop failure to the one thing left after inputs
and weights were both verified clean: the planner CALL ASSEMBLY. The trainer
(``train/train_batched.py``) and the loop (``microvla/jepa/loop.py``) each build
that call themselves, from the same modules, and only their agreement was never
tested — which is where all five previously-found defects in this project lived.

This drives ONE baked episode through both paths with perception held identical
(the loop is handed a replay perception that returns the corpus's own embeddings
and boxes, so the detector is not a variable), and compares the planner's inputs
tensor by tensor plus the resulting gripper logit.

Read the output as: any input group with a large relative difference is the
divergence. ``grip_logit`` is reported for both paths because it is the quantity
that actually disagrees in deployment — the trainer's stage-B validation puts it
above zero on ~52% of steps (``grip_acc 0.934``) while the deployed policy holds
it at or below zero on 100%.

Usage::

    python -m eval.train_vs_deploy --checkpoint checkpoints/full_stageB_v8_act.pt \\
        --episode data/libero_object_v8/<ep>.npz
"""
from __future__ import annotations

import argparse

import numpy as np


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episode", required=True, help="one baked .npz")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    import torch

    from microvla.config import DEFAULT_CONFIG
    from microvla.jepa.loop import JEPALoop
    from microvla.perception.yolo_world import BoxObs, Perception
    from microvla.utils.embedding import standardize

    dev = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", DEFAULT_CONFIG)
    if isinstance(cfg, dict):
        import dataclasses
        cfg = dataclasses.replace(DEFAULT_CONFIG, **{
            k: v for k, v in cfg.items() if k in {f.name for f in dataclasses.fields(DEFAULT_CONFIG)}})

    ep = np.load(args.episode)
    T = ep["frame_embs"].shape[0]
    t = lambda k: torch.as_tensor(ep[k], dtype=torch.float32, device=dev)

    # ---- build the modules once; both paths share them ----------------------
    from eval.policy import MicroVLAPolicy  # reuses the deployed builder

    import os
    norm = os.path.join(os.path.dirname(args.episode), "norm_stats.json")
    wp = os.path.join(os.path.dirname(args.episode), "waypoint_stats.json")
    policy = MicroVLAPolicy(checkpoint=args.checkpoint, norm_stats=norm,
                            waypoint_stats=wp if os.path.exists(wp) else None,
                            device=args.device, heads_device=args.device)
    loop = policy.loop
    fusion, drift, trm, planner = loop.fusion, loop.drift, loop.trm, loop.planner
    relational = getattr(loop, "relational", None)

    # ---- TRAINER path -------------------------------------------------------
    from train.train_batched import real_paths, _boxes, _relational

    batch = {k: t(k).unsqueeze(0) for k in
             ("frame_embs", "source_box_embs", "target_box_embs", "source_centers",
              "target_centers", "box_weights", "pwm_targets", "proprio",
              "obj_embs", "obj_centers", "obj_weights") if k in ep.files}
    batch["text_tokens"] = t("text_tokens").unsqueeze(0)
    batch["has_objects"] = torch.ones(1, 1, device=dev)

    train_in, train_grip = [], []
    with torch.no_grad():
        fused_all, delta_all = real_paths(batch, fusion, drift, cfg, ablate=False)
        for i in range(T):
            cur = batch["frame_embs"][:, i]
            sbe, tbe, sc, tc, bw = _boxes(batch, i, 1.0, cfg, False)
            wm = trm.forward_full(fused_all[i], delta_all[i], cur)
            geom = torch.cat([sc, tc, bw], dim=-1)
            rel = _relational(relational, wm["next_emb"], batch, i, i, 1.0, cfg)
            kw = dict(current_emb=cur, state_delta=delta_all[i], fused=fused_all[i],
                      pred_box_emb=wm["next_box"], geometry=geom,
                      proprio=batch["proprio"][:, i], wm_msg=wm["msg"],
                      wm_latent=wm.get("latent"), relational=rel)
            _plan, grip, _wp = planner(wm["next_emb"], return_wp=True, **kw)
            train_in.append({k: v for k, v in kw.items() if torch.is_tensor(v)}
                            | {"next_emb": wm["next_emb"]})
            train_grip.append(float(grip.mean()))

    # ---- DEPLOYMENT path, same perception -----------------------------------
    class Replay:
        """Returns the corpus's own perception for tick i — no detector."""

        def __init__(self):
            self.i = 0

        def set_role_prompts(self, source, target=None):
            pass

        def set_classes(self, names):
            pass

        def perceive(self, _frame_bgr):
            i = min(self.i, T - 1)
            self.i += 1
            box = lambda e, c, w: BoxObs(
                emb=torch.as_tensor(ep[e][i], dtype=torch.float32),
                center=torch.as_tensor(ep[c][i], dtype=torch.float32),
                xyxy=torch.zeros(4), confidence=float(ep["box_weights"][i][w]))
            props = tuple(
                BoxObs(emb=torch.as_tensor(ep["obj_embs"][i][k], dtype=torch.float32),
                       center=torch.as_tensor(ep["obj_centers"][i][k], dtype=torch.float32),
                       xyxy=torch.zeros(4), confidence=float(ep["obj_weights"][i][k]))
                for k in range(ep["obj_embs"].shape[1])
                if float(ep["obj_weights"][i][k]) > 0)
            return Perception(
                frame_emb=torch.as_tensor(ep["frame_embs"][i], dtype=torch.float32),
                source=box("source_box_embs", "source_centers", 0),
                target=box("target_box_embs", "target_centers", 1),
                proposals=props)

    loop.perception = Replay()
    hooked = []

    def hook(_m, _a, kwargs):
        hooked.append({k: v.detach().clone() for k, v in kwargs.items()
                       if torch.is_tensor(v)})
        return None

    h = planner.register_forward_pre_hook(hook, with_kwargs=True)
    loop.set_task("pick up the alphabet soup and place it in the basket")
    loop.perception = Replay()          # set_task may have reset it
    # Hold TEXT identical too. The loop re-encodes the task with CLIP while the
    # trainer uses the tokens baked into the episode; that is a real deployment
    # difference but it is not the one under test here, and leaving it in would
    # dirty every text-consuming group (fused, relational). Override with the
    # baked tokens so the only remaining variable is the assembly itself.
    tt = t("text_tokens")
    loop._task.command_emb, loop._task.source_emb, loop._task.target_emb = (
        tt[0], tt[1], tt[2])
    deploy_grip = []
    dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(T):
        # Force a REAL tick every step so the comparison is like-for-like with
        # the trainer's real-step assembly (dream ticks are a separate question).
        loop._tick_index = 0 if hasattr(loop, "_tick_index") else None
        res = loop.tick(dummy, proprio=ep["proprio"][i])
        deploy_grip.append(float(res.plan[0, -1]))
    h.remove()

    # ---- compare ------------------------------------------------------------
    print(f"\nepisode {args.episode}  T={T}\n")
    print(f"{'planner input':14s} {'train mean/std':>22} {'deploy mean/std':>22} {'rel diff':>10}")
    n = min(len(train_in), len(hooked))
    keys = sorted(set(train_in[0]) | set(hooked[0]) if n else [])
    for k in keys:
        a = [x[k] for x in train_in[:n] if k in x]
        b = [x[k] for x in hooked[:n] if k in x]
        if not a or not b:
            print(f"{k:14s} {'PRESENT' if a else 'ABSENT':>22} "
                  f"{'PRESENT' if b else 'ABSENT':>22} {'<== ONLY ONE SIDE':>10}")
            continue
        A = torch.cat([x.reshape(-1) for x in a]).float().numpy()
        B = torch.cat([x.reshape(-1) for x in b]).float().numpy()
        if A.shape != B.shape:
            print(f"{k:14s} {str(a[0].shape):>22} {str(b[0].shape):>22} {'<== SHAPE':>10}")
            continue
        d = float(np.abs(A - B).mean()) / max(float(np.abs(A).mean()), 1e-9)
        flag = "  <== DIVERGES" if d > 0.05 else ""
        print(f"{k:14s} {A.mean():10.4f}/{A.std():<10.4f} "
              f"{B.mean():10.4f}/{B.std():<10.4f} {d:10.4f}{flag}")

    # Raw values for the first ticks: an aggregate rel-diff says THAT a group
    # differs, not HOW, and geometry is small enough to just read.
    print("\ngeometry, first 3 ticks (src_center, tgt_center, box_weights)")
    for i in range(min(3, n)):
        a = train_in[i].get("geometry"); b = hooked[i].get("geometry")
        if a is not None and b is not None:
            print(f"  t{i} train  {np.round(a.reshape(-1).float().numpy(), 4).tolist()}")
            print(f"     deploy {np.round(b.reshape(-1).float().numpy(), 4).tolist()}")

    print(f"\ngrip: trainer logit mean {np.mean(train_grip):+.4f} "
          f"(>0 on {100*np.mean(np.array(train_grip) > 0):.0f}% of steps)")
    print(f"      deployed plan gripper >0 on {100*np.mean(np.array(deploy_grip) > 0):.0f}% "
          f"of steps, unique {np.unique(np.round(deploy_grip, 3)).tolist()[:6]}")
    tgt = ep["pwm_targets"][:, 0, -1]
    print(f"      corpus target closes on {100*float((tgt > 0).mean()):.0f}% of steps")


if __name__ == "__main__":
    main()
