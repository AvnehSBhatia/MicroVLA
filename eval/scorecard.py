"""Offline scorecard: rollout error, innovation norms, planner BC (paper.md E2, prep for E6/E7).

Loads a checkpoint per the SHARED CONTRACT (``torch.save`` dict with keys
``{cfg: dict, trm_d: int, fusion/drift/trm: state_dicts, planner:
state_dict (optional)}`` -- ``full_stageA.pt`` has no planner,
``full_stageB.pt`` has all four) and scores it against the *val* split of one
or more converted episode directories, reusing
``train.train_full._MultiDataset`` with the same ``--val-frac``/``--seed``
defaults ``train_full.py`` uses so the split lines up with training. Because
scoring only depends on the checkpoint (not on how it was produced), this
same tool scores the E6 (``--ablate-evidence-fade``) and E7
(``--ablate-grounding``) ablation checkpoints from ``train_full.py`` -- just
point ``--checkpoint`` at ``full_stageB_<tag>.pt``.

Four sections, mirroring ``train_full``'s own eval code so results are
reproducible outside a training run:

  (a) k-step open-loop rollout ``spec_loss`` for k in {1, 5, 10, 15} vs the
      persistence baseline at each k -- ``train_full._rollout`` run with
      ``ticks=k`` from each sampled real frame t, compared against the
      actual next real frame's embedding (``train_full._episode_real_paths``
      supplies the grounded per-t fused matrices/drift codes).
  (b) innovation-norm distribution (``||pred - actual next real emb||``) vs
      persistence's, at the deployment-exact rollout length (``k =
      max(1, 5, 10, 15) = 15`` at the default config, i.e. a full real-frame
      interval) -- this is exactly the quantity ``InnovationCorrector``
      consumes at inference.
  (c) planner behavior-cloning loss + smoothness on val (mirrors
      ``train_full.stage_b``'s per-episode forward pass, without the
      backward pass); skipped with a note if the checkpoint has no planner.
  (d) a one-screen text report, plus the same numbers written to
      ``eval_results/scorecard.json``.

Usage::

    python eval/scorecard.py --checkpoint checkpoints/full_stageB.pt \\
        --data-dir data/bridge --data-dir data/libero --device cpu

CPU-only by default; pass ``--device auto`` to let
``train.train_planner.resolve_device`` pick MPS when available.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder  # noqa: E402
from microvla.config import MicroVLAConfig  # noqa: E402
from microvla.fusion.slot_fusion import SlotResonanceFusion  # noqa: E402
from microvla.planner.chrono_planner import ChronoQueryPlanner  # noqa: E402
from train.losses import planner_bc_loss, smoothness_loss  # noqa: E402
from train.train_full import _MultiDataset, _episode_real_paths, _rollout  # noqa: E402
from train.train_planner import resolve_device  # noqa: E402
from TRM import RecursiveTRM, spec_loss  # noqa: E402

#: k-step open-loop rollout lengths scored in section (a); paper.md E2.
ROLLOUT_KS: tuple[int, ...] = (1, 5, 10, 15)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True,
                   help="path to full_stageA.pt / full_stageB.pt (or a tagged ablation "
                        "checkpoint) -- the SHARED CONTRACT torch.save dict")
    p.add_argument("--data-dir", action="append", required=True,
                   help="converted episode dir; repeat to concatenate datasets")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="must match the --val-frac train_full.py was run with, so the val "
                        "split (via _MultiDataset) lines up with the one used in training")
    p.add_argument("--seed", type=int, default=0,
                   help="must match train_full.py's --seed for the same split")
    p.add_argument("--max-val-episodes", type=int, default=64,
                   help="cap episodes scored (train_full.evaluate_world uses the same 64 "
                        "default); each episode contributes up to 4 sampled rollout anchors")
    p.add_argument("--device", default="cpu",
                   help="'cpu' (default) or 'auto' (train_planner.resolve_device: MPS if "
                        "available, else CPU) or an explicit torch device string")
    p.add_argument("--out", default="eval_results/scorecard.json",
                   help="where to write the JSON report")
    return p.parse_args(argv)


def load_checkpoint(path: str, device: torch.device):
    """Loads a SHARED CONTRACT checkpoint into eval-mode modules on ``device``.

    Args:
        path: Checkpoint file (``{cfg, trm_d, fusion, drift, trm, planner?}``).
        device: Torch device to place the modules on.

    Returns:
        ``(cfg, fusion, drift, trm, planner)`` -- ``planner`` is ``None`` when
        the checkpoint has no ``"planner"`` key (a stage-A checkpoint).
    """
    state = torch.load(path, map_location=device, weights_only=True)
    cfg = MicroVLAConfig(**state["cfg"])

    fusion = SlotResonanceFusion(cfg).to(device)
    fusion.load_state_dict(state["fusion"])
    drift = AnchoredDriftEncoder(cfg).to(device)
    drift.load_state_dict(state["drift"])
    trm = RecursiveTRM(cfg, d=state["trm_d"]).to(device)
    trm.load_state_dict(state["trm"])
    fusion.eval(); drift.eval(); trm.eval()

    planner = None
    if "planner" in state:
        planner = ChronoQueryPlanner(cfg).to(device)
        planner.load_state_dict(state["planner"])
        planner.eval()

    return cfg, fusion, drift, trm, planner


def _dist_stats(values: list[float]) -> dict:
    """Summarizes a distribution: mean/std/median/p90/p99/min/max/n."""
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "p90": float("nan"), "p99": float("nan"), "min": float("nan"), "max": float("nan")}
    xs = sorted(values)
    n = len(xs)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, round(p * (n - 1))))
        return xs[idx]

    return {
        "n": n,
        "mean": statistics.fmean(xs),
        "std": statistics.pstdev(xs) if n > 1 else 0.0,
        "median": statistics.median(xs),
        "min": xs[0],
        "max": xs[-1],
        "p90": pct(0.90),
        "p99": pct(0.99),
    }


@torch.no_grad()
def score_rollout(cfg, data, fusion, drift, trm, device, ks, max_episodes):
    """Sections (a) + (b): k-step rollout spec_loss + innovation-norm distribution.

    For each scored episode, samples up to 4 anchor points t (same density as
    ``train_full.evaluate_world``) and, at each, runs
    ``train_full._rollout(..., ticks=k)`` for every ``k`` in ``ks`` against
    the actual next real frame embedding, alongside the persistence baseline
    (``spec_loss``/L2-norm of "predict next = current"). The innovation-norm
    distribution (b) reuses the ``k = max(ks)`` rollout -- the
    deployment-exact full real-frame interval, matching what
    ``InnovationCorrector`` sees at inference.

    Returns:
        ``(per_k, innovation_trm, innovation_persistence, n_episodes)`` where
        ``per_k[k]`` is ``{"trm": [...], "persistence": [...]}`` spec_loss
        samples and the innovation lists hold raw L2 norms.
    """
    from train.train_full import _persistence_loss

    max_k = max(ks)
    gamma = 0.9
    per_k = {k: {"trm": [], "persistence": []} for k in ks}
    innovation_trm: list[float] = []
    innovation_persistence: list[float] = []
    n_episodes = 0

    for key in data.val_index[:max_episodes]:
        episode = {k: v.to(device) for k, v in data.get(key).items()}
        T = episode["frame_embs"].shape[0]
        if T < 2:
            continue
        fused_all, delta_all = _episode_real_paths(episode, fusion, drift, device)
        n_episodes += 1

        # v3 _rollout returns the discounted H-step loss (not a prediction) and
        # requires t + k < T, so anchors are bounded per k.
        for k in ks:
            for t in range(0, max(T - k, 0), max(1, (T - 1) // 4)):
                per_k[k]["trm"].append(
                    float(_rollout(episode, t, fused_all[t], delta_all[t], fusion, trm, cfg, k, gamma)))
                per_k[k]["persistence"].append(_persistence_loss(episode, t, k, cfg, gamma))

        # Innovation-norm distribution (b): 1-step prediction error vs
        # persistence's, matching what InnovationCorrector sees at each real tick.
        for t in range(0, T - 1, max(1, (T - 1) // 4)):
            current = episode["frame_embs"][t].unsqueeze(0)
            target = episode["frame_embs"][t + 1].unsqueeze(0)
            pred1 = trm(fused_all[t], delta_all[t], current)
            innovation_trm.append(float(torch.linalg.vector_norm(pred1 - target, dim=-1).squeeze(0)))
            innovation_persistence.append(
                float(torch.linalg.vector_norm(current - target, dim=-1).squeeze(0)))

    return per_k, innovation_trm, innovation_persistence, n_episodes


@torch.no_grad()
def score_planner(cfg, data, fusion, drift, trm, planner, device, max_episodes):
    """Section (c): planner BC loss + smoothness on val (mirrors ``train_full.stage_b``).

    Returns:
        ``None`` if ``planner`` is ``None`` (stage-A checkpoint); otherwise a
        dict with ``bc_loss``, ``smoothness_loss``, ``n_episodes``.
    """
    if planner is None:
        return None

    bc_losses, smooth_losses = [], []
    n_episodes = 0
    for key in data.val_index[:max_episodes]:
        episode = {k: v.to(device) for k, v in data.get(key).items()}
        T = episode["frame_embs"].shape[0]
        if T < 2:
            continue
        fused_all, delta_all = _episode_real_paths(episode, fusion, drift, device)
        preds = []
        for t in range(T):
            next_emb = trm(fused_all[t], delta_all[t], episode["frame_embs"][t].unsqueeze(0))
            preds.append(planner(next_emb).squeeze(0))
        preds = torch.stack(preds, 0)
        bc_losses.append(float(planner_bc_loss(preds, episode["pwm_targets"])))
        smooth_losses.append(float(smoothness_loss(preds)))
        n_episodes += 1

    return {
        "bc_loss": statistics.fmean(bc_losses) if bc_losses else float("nan"),
        "smoothness_loss": statistics.fmean(smooth_losses) if smooth_losses else float("nan"),
        "n_episodes": n_episodes,
    }


def render_report(result: dict) -> str:
    """Renders section (d): the one-screen text report."""
    lines = []
    rule = "=" * 74
    lines.append(rule)
    lines.append(f"MicroVLA offline scorecard -- {result['checkpoint']}")
    lines.append(f"data: {', '.join(result['data_dirs'])}")
    lines.append(
        f"val episodes scored: {result['n_val_episodes']} of {result['n_val_total']} "
        f"(val-frac={result['val_frac']}, seed={result['seed']})"
    )
    lines.append("-" * 74)
    lines.append("(a) k-step open-loop rollout spec_loss vs persistence")
    lines.append(f"{'k':>4} | {'TRM spec_loss':>14} | {'persistence':>12} | {'beats?':>7} | {'n':>6}")
    for k in sorted(result["rollout"], key=int):
        row = result["rollout"][k]
        beats = "YES" if row["trm_spec_loss"] < row["persistence_spec_loss"] else "no"
        lines.append(
            f"{k:>4} | {row['trm_spec_loss']:>14.4f} | {row['persistence_spec_loss']:>12.4f} "
            f"| {beats:>7} | {row['n_samples']:>6}"
        )
    lines.append("-" * 74)
    innov = result["innovation_norm"]
    lines.append(f"(b) innovation-norm distribution (||pred - actual next real emb||, k={innov['k']})")
    for name in ("trm", "persistence"):
        d = innov[name]
        lines.append(
            f"  {name:>11}: mean {d['mean']:.4f} | std {d['std']:.4f} | median {d['median']:.4f} "
            f"| p90 {d['p90']:.4f} | p99 {d['p99']:.4f} | n={d['n']}"
        )
    lines.append("-" * 74)
    lines.append("(c) planner BC loss + smoothness (val)")
    if result["planner"] is None:
        lines.append("  NOT EVALUATED -- checkpoint has no 'planner' state_dict (stage-A checkpoint)")
    else:
        p = result["planner"]
        lines.append(
            f"  bc_loss {p['bc_loss']:.4f} | smoothness {p['smoothness_loss']:.4f} "
            f"(n={p['n_episodes']} episodes)"
        )
    lines.append(rule)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    device = resolve_device(args.device)

    cfg, fusion, drift, trm, planner = load_checkpoint(args.checkpoint, device)
    data = _MultiDataset(args.data_dir, args.val_frac, args.seed)

    per_k, innov_trm, innov_persistence, n_episodes = score_rollout(
        cfg, data, fusion, drift, trm, device, ROLLOUT_KS, args.max_val_episodes
    )
    planner_result = score_planner(
        cfg, data, fusion, drift, trm, planner, device, args.max_val_episodes
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "data_dirs": list(args.data_dir),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "n_val_total": len(data.val_index),
        "n_val_episodes": n_episodes,
        "rollout": {
            str(k): {
                "trm_spec_loss": statistics.fmean(v["trm"]) if v["trm"] else float("nan"),
                "persistence_spec_loss": statistics.fmean(v["persistence"]) if v["persistence"] else float("nan"),
                "n_samples": len(v["trm"]),
            }
            for k, v in per_k.items()
        },
        "innovation_norm": {
            "k": max(ROLLOUT_KS),
            "trm": _dist_stats(innov_trm),
            "persistence": _dist_stats(innov_persistence),
        },
        "planner": planner_result,
    }

    report = render_report(result)
    print(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")
    return result


if __name__ == "__main__":
    main()
