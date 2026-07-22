"""Evaluate a stage-A checkpoint's rollout loss vs persistence across horizons.

This is the direct early-evidence figure for paper Claim 2 (perception-rate
decoupling): a good world model's advantage over "assume no change" should GROW
the further ahead it must predict. We hold the checkpoint fixed and sweep the
rollout horizon H, reporting val spec_loss, the discounted persistence
baseline, and the margin at each H — every number logged to the durable store.

Runs on CPU by default so it never contends with an MPS training job.

    python -m experiments.horizon_curve --checkpoint checkpoints/full_stageA.pt \\
        --data-dir data/bridge --data-dir data/libero \\
        --horizons 1 2 3 4 5 6 8 --n-episodes 48 --device cpu
"""

from __future__ import annotations

import argparse

import torch

from experiments.tracker import log, report
from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import MicroVLAConfig
from microvla.fusion.slot_fusion import SlotResonanceFusion
from train.train_full import _MultiDataset, _episode_real_paths, _persistence_loss, _rollout
from TRM import RecursiveTRM, spec_loss  # noqa: F401 (spec_loss used via _rollout)


def load_stage_a(checkpoint: str, device: torch.device):
    """Loads fusion/drift/trm from a stage-A checkpoint dict."""
    st = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = MicroVLAConfig(**st["cfg"]) if isinstance(st["cfg"], dict) else st["cfg"]
    fusion = SlotResonanceFusion(cfg).to(device); fusion.load_state_dict(st["fusion"]); fusion.eval()
    drift = AnchoredDriftEncoder(cfg).to(device); drift.load_state_dict(st["drift"])
    trm = RecursiveTRM(cfg, d=st.get("trm_d", 1024)).to(device); trm.load_state_dict(st["trm"]); trm.eval()
    return cfg, fusion, drift, trm


@torch.no_grad()
def curve(checkpoint: str, data_dirs: list[str], horizons: list[int],
          n_episodes: int, gamma: float, device: str, val_frac: float, seed: int,
          run_id: str) -> list[dict]:
    """Computes and logs the (val, persistence, margin) curve over horizons."""
    dev = torch.device(device)
    cfg, fusion, drift, trm = load_stage_a(checkpoint, dev)
    data = _MultiDataset(data_dirs, val_frac, seed)

    class _A:
        pass
    args = _A(); args.ablate_grounding = False; args.gamma = gamma

    rows = []
    ckpt_name = checkpoint.split("/")[-1]
    for H in horizons:
        vals, base, used = [], [], 0
        for key in data.val_index:
            if used >= n_episodes:
                break
            episode = {k: v.to(dev) for k, v in data.get(key).items()}
            T = episode["frame_embs"].shape[0]
            if T < H + 1:
                continue
            fused_all, delta_all = _episode_real_paths(episode, fusion, drift, dev)
            for t in range(0, T - H, max(1, (T - H) // 4)):
                vals.append(float(_rollout(episode, t, fused_all[t], delta_all[t],
                                           fusion, trm, cfg, H, gamma)))
                base.append(_persistence_loss(episode, t, H, cfg, gamma))
            used += 1
        v = sum(vals) / max(len(vals), 1)
        p = sum(base) / max(len(base), 1)
        margin = (p - v) / p * 100 if p else 0.0
        rec = log({"run_id": run_id, "kind": "horizon_curve", "checkpoint": ckpt_name,
                   "horizon": H, "val_loss": round(v, 5), "persistence": round(p, 5),
                   "margin_pct": round(margin, 1), "n_episodes": used})
        rows.append(rec)
        print(f"H={H:>2} | val {v:.5f} | persistence {p:.5f} | "
              f"margin {margin:+5.1f}% | {used} eps", flush=True)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", action="append", required=True, dest="data_dirs")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--n-episodes", type=int, default=48)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default="horizon_curve")
    args = ap.parse_args(argv)
    curve(args.checkpoint, args.data_dirs, args.horizons, args.n_episodes,
          args.gamma, args.device, args.val_frac, args.seed, args.run_id)
    print("wrote", report())


if __name__ == "__main__":
    main()
