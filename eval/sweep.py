"""Perception-rate sweep: paper.md E4 (Claim 2) + E5 (Claim 5), the central result.

Sweeps ``perception_period`` (ticks between REAL YOLO-World perceptions, the
30 Hz control loop's dream-window knob) across ``{1, 2, 5, 10, 15, 20}`` for
three conditions:

* ``ours``        — the trained ``TRM.py::RecursiveTRM`` (or whatever
                     ``MicroVLAPolicy`` builds from ``--checkpoint``, or
                     fresh untrained modules if ``--checkpoint`` is omitted).
* ``persistence``  — :class:`eval.baselines.PersistenceTRM`, the "hold-last
                     observation / no world model" foil. Kill bar (paper.md):
                     if ``ours`` degrades identically to this as
                     ``perception_period`` grows, the world model adds
                     nothing.
* ``linear``       — :class:`eval.baselines.LinearExtrapolationTRM`, the
                     "cheap dreamer" foil the TRM must beat to be more than
                     decoration.

For each (condition, perception_period) cell, builds a
``MicroVLAPolicy`` (``trm=`` overridden per condition; ``None`` for
``ours`` so the policy's own checkpoint-loaded TRM is used) and calls
``eval.libero_eval.run_eval`` (``mock_env`` passed straight through, so
``--mock-env`` makes the whole sweep runnable today with no sim deps and no
network, per ``eval.libero_eval``'s ``MockLiberoEnv``).

Output: ``eval_results/sweep.json`` — ``{"rows": [...], "auroc": {...},
"meta": {...}}`` — plus a plain-text table printed to stdout. Each row is
``{condition, perception_period, mean_success, mean_trust, n_trials,
per_task, telemetry_path}``. ``auroc`` maps each condition to the pure-Python
(no sklearn) AUROC of "does this episode's trust telemetry predict it
failed" (paper.md Claim 5 / E5): label = episode failure, score = -(min
trust observed during the episode) so higher score means higher predicted
failure risk, aggregated across every perception_period run for that
condition.

Telemetry format: ``eval.libero_eval.run_eval`` writes ``telemetry_path`` as
JSONL (one JSON object per env step), each line merging ``{suite, task,
trial, success, step}`` with the policy's own per-tick record (``{tick_index,
is_real, trust, plan_norm}``) -- ``_parse_telemetry`` below groups lines by
``(task, trial)`` to recover one ``(success, min_trust)`` pair per episode
for the AUROC label. It also tolerates a plain JSON array or a
``{"episodes": [...]}`` nesting (defensive, since it primarily targets
that JSONL shape but costs nothing to accept the alternatives too).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.trm.interface import TRMBase

from eval.baselines import LinearExtrapolationTRM, PersistenceTRM

DEFAULT_PERCEPTION_PERIODS: tuple[int, ...] = (1, 2, 5, 10, 15, 20)
DEFAULT_CONDITIONS: tuple[str, ...] = ("ours", "persistence", "linear")


def _build_condition_trm(condition: str, cfg: MicroVLAConfig) -> TRMBase | None:
    """Maps a sweep condition name to its ``trm=`` override for ``MicroVLAPolicy``.

    Args:
        condition: One of ``DEFAULT_CONDITIONS``.
        cfg: Config passed through to the foil's constructor.

    Returns:
        ``None`` for ``"ours"`` (use the policy's own checkpoint-loaded TRM,
        or its fresh-module default), else a stateless zero-param foil.

    Raises:
        ValueError: Unknown condition name.
    """
    if condition == "ours":
        return None
    if condition == "persistence":
        return PersistenceTRM(cfg)
    if condition == "linear":
        return LinearExtrapolationTRM(cfg)
    raise ValueError(f"unknown sweep condition {condition!r}; expected one of {DEFAULT_CONDITIONS}")


def _write_identity_norm_stats(path: Path, num_servos: int) -> Path:
    """Synthesizes an identity ``norm_stats.json`` (last-resort fallback).

    ``ActionNormalizer(q_low=-1, q_high=1)`` makes ``inverse()`` the identity
    map on ``[-1, 1]`` planner outputs — meaningless in real action units,
    fine for exercising the harness end to end. Only used if the repo-shipped
    ``eval/identity_norm_stats.json`` (see :func:`_default_norm_stats_path`)
    is somehow missing.

    Args:
        path: Output path (created, including parent dirs).
        num_servos: Action dimensionality (``cfg.num_servos``).

    Returns:
        ``path``, for chaining.
    """
    from preprocess.common import ActionNormalizer

    path.parent.mkdir(parents=True, exist_ok=True)
    ActionNormalizer(q_low=[-1.0] * num_servos, q_high=[1.0] * num_servos).save(path)
    return path


def _default_norm_stats_path(fallback_dir: Path, num_servos: int) -> Path:
    """Resolves ``--norm-stats`` when omitted.

    Prefers the repo-shipped ``eval/identity_norm_stats.json`` (the same
    file ``eval.libero_eval``'s own CLI defaults to for smoke/mock runs with
    no trained action distribution yet); synthesizes one under
    ``fallback_dir`` only if that shipped file is somehow missing.
    """
    shipped = Path(__file__).resolve().parent / "identity_norm_stats.json"
    if shipped.exists():
        return shipped
    return _write_identity_norm_stats(fallback_dir / "_identity_norm_stats.json", num_servos)


def _episode_key(tick: dict):
    """Groups a flat per-tick telemetry record into its episode.

    Prefers an explicit ``episode``/``episode_index`` id; otherwise falls
    back to ``(task, trial)`` -- ``eval.libero_eval.run_eval`` restarts
    ``trial`` at 0 for every task, so ``trial`` alone would collide across
    tasks.
    """
    for key in ("episode", "episode_index"):
        if key in tick:
            return tick[key]
    task, trial = tick.get("task"), tick.get("trial", tick.get("trial_index"))
    if task is not None and trial is not None:
        return (task, trial)
    return trial if trial is not None else "_single_episode_"


def _parse_telemetry(telemetry_path: str | None) -> tuple[list[float], list[dict]]:
    """Tolerantly recovers trust values + per-episode (success, min_trust) pairs.

    Targets the JSONL shape ``eval.libero_eval.run_eval`` writes (see the
    module docstring's "Telemetry format" section); also accepts a plain
    JSON array of the same flat per-tick records, or a
    ``{"episodes": [{"success": bool, "telemetry"|"steps": [tick, ...]}, ...]}``
    nesting, in case the shape drifts.

    Args:
        telemetry_path: ``run_eval(...)["telemetry_path"]``, or ``None``.

    Returns:
        ``(all_trust_values, episodes)`` where ``episodes`` is a list of
        ``{"success": bool, "min_trust": float}`` dicts, one per episode
        recoverable from the file (``episodes`` is empty, but
        ``all_trust_values`` may still be non-empty, if no per-tick record
        carries a ``success`` flag).
    """
    if not telemetry_path:
        return [], []
    path = Path(telemetry_path)
    if not path.exists():
        return [], []
    text = path.read_text()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict) and isinstance(data.get("episodes"), list):
        items, nested = data["episodes"], True
    elif isinstance(data, list):
        items, nested = data, False
    else:
        # Not a single JSON document -> JSONL, one object per line (the
        # shape eval.libero_eval.run_eval actually writes).
        items, nested = [], False
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    all_trust: list[float] = []
    episodes: list[dict] = []

    if nested:
        for ep in items:
            if not isinstance(ep, dict):
                continue
            steps = ep.get("telemetry") or ep.get("steps") or ep.get("ticks") or []
            trusts = [s["trust"] for s in steps if isinstance(s, dict) and "trust" in s]
            all_trust.extend(trusts)
            success = ep.get("success", ep.get("is_success", ep.get("solved")))
            if success is not None and trusts:
                episodes.append({"success": bool(success), "min_trust": min(trusts)})
    else:
        grouped: dict = {}
        for tick in items:
            if not isinstance(tick, dict) or "trust" not in tick:
                continue
            all_trust.append(tick["trust"])
            g = grouped.setdefault(_episode_key(tick), {"trusts": [], "success": None})
            g["trusts"].append(tick["trust"])
            success = tick.get("success", tick.get("is_success"))
            if success is not None:
                g["success"] = bool(success)
        for g in grouped.values():
            if g["success"] is not None and g["trusts"]:
                episodes.append({"success": g["success"], "min_trust": min(g["trusts"])})

    return all_trust, episodes


def _rank_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Pure-Python AUROC (Mann-Whitney rank-sum), no sklearn.

    ``AUROC = P(score of a random positive > score of a random negative)``,
    with tied scores split via average ranks.

    Args:
        labels: ``0``/``1`` per sample.
        scores: Higher = more likely positive, same order as ``labels``.

    Returns:
        AUROC in ``[0, 1]``, or ``float("nan")`` if either class is empty.
    """
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed, averaged over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    sum_pos_ranks = sum(r for r, label in zip(ranks, labels) if label == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # numpy/torch scalars
        return obj.item()
    return str(obj)


def run_sweep(
    perception_periods: Sequence[int] = DEFAULT_PERCEPTION_PERIODS,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    checkpoint: str | None = None,
    norm_stats: str | None = None,
    suite: str = "libero_spatial",
    n_trials: int = 5,
    max_steps: int = 200,
    camera: str = "robot0_eye_in_hand_image",
    mock_env: bool = True,
    seed: int = 0,
    device: str = "cpu",
    cfg: MicroVLAConfig | None = None,
    out_path: str | Path = "eval_results/sweep.json",
) -> dict:
    """Runs the full (condition x perception_period) grid and writes the results.

    Args:
        perception_periods: Sweep values for ``MicroVLAPolicy(perception_period=...)``.
        conditions: Subset/order of ``DEFAULT_CONDITIONS`` to run.
        checkpoint: ``checkpoints/full_stageB.pt``-style path, or ``None``
            for fresh (untrained) modules — ``MicroVLAPolicy``'s smoke path.
        norm_stats: ``norm_stats.json`` path paired with ``checkpoint``; if
            ``None``, defaults to the repo-shipped
            ``eval/identity_norm_stats.json`` (see
            :func:`_default_norm_stats_path`).
        suite: LIBERO suite name, passed to ``run_eval``.
        n_trials: Trials per task, passed to ``run_eval``.
        max_steps: Episode step cap, passed to ``run_eval``.
        camera: Camera key, passed to ``run_eval``.
        mock_env: Use ``libero_eval``'s built-in ``MockLiberoEnv`` (no sim
            deps, deterministic) instead of a real LIBERO/robosuite env.
        seed: Eval seed, passed to ``run_eval``.
        device: Torch device for policy modules. ``"mps"`` is refused — a
            dataset conversion job owns the MPS device on this machine.
        cfg: Optional ``MicroVLAConfig`` override, forwarded to every
            ``MicroVLAPolicy`` verbatim. Only used to construct the
            ``persistence``/``linear`` foils (functionally cfg-agnostic —
            see ``eval/baselines.py``); left ``None`` by default so
            ``MicroVLAPolicy`` resolves ``cfg`` itself (the checkpoint's
            saved config when one is given, else ``DEFAULT_CONFIG``) rather
            than this sweep silently overriding a trained checkpoint's
            architecture.
        out_path: Where to write the JSON results.

    Returns:
        ``{"rows": [...], "auroc": {condition: float}, "meta": {...}}`` —
        the same dict written to ``out_path``.

    Raises:
        ValueError: ``device == "mps"``, or an unknown condition name.
    """
    if device == "mps":
        raise ValueError(
            "eval.sweep refuses device='mps': a dataset conversion job owns the "
            "MPS device on this machine right now (hard constraint). Use 'cpu'."
        )

    # Imported here (not at module scope) so importing eval.sweep for its
    # pure-Python helpers (AUROC, baselines) never requires eval.policy /
    # eval.libero_eval to already exist.
    from eval.libero_eval import run_eval
    from eval.policy import MicroVLAPolicy

    # cfg_for_baselines: a concrete config to hand PersistenceTRM/
    # LinearExtrapolationTRM's constructor (they ignore it functionally).
    # cfg_for_policy: what gets forwarded to MicroVLAPolicy(cfg=...) --
    # stays None (checkpoint/DEFAULT_CONFIG auto-resolution) unless the
    # caller explicitly asked for an override.
    cfg_for_baselines = cfg or DEFAULT_CONFIG
    cfg_for_policy = cfg
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm_stats_path = (
        Path(norm_stats) if norm_stats
        else _default_norm_stats_path(out_path.parent, cfg_for_baselines.num_servos)
    )

    rows: list[dict] = []
    episodes_by_condition: dict[str, list[dict]] = {c: [] for c in conditions}

    for condition in conditions:
        trm_override = _build_condition_trm(condition, cfg_for_baselines)
        for period in perception_periods:

            def policy_factory(_period=period, _trm=trm_override):
                # mock_env only selects MockLiberoEnv vs a real sim env; it
                # says nothing about perception. Mirror
                # eval.libero_eval._make_policy_factory's own convention so
                # a mock_env=True sweep never triggers MicroVLAPolicy's real
                # (heavy-import, weight-loading) YoloWorldPerception default.
                perception = task_encoder = None
                if mock_env:
                    from microvla.perception.text_encoder import MockTaskEncoder
                    from microvla.perception.yolo_world import MockYoloWorldPerception

                    perception = MockYoloWorldPerception()
                    task_encoder = MockTaskEncoder()
                return MicroVLAPolicy(
                    checkpoint=checkpoint,
                    norm_stats=str(norm_stats_path),
                    cfg=cfg_for_policy,
                    perception_period=_period,
                    trm=_trm,
                    device=device,
                    perception=perception,
                    task_encoder=task_encoder,
                )

            result = run_eval(
                policy_factory,
                suite=suite,
                n_trials=n_trials,
                max_steps=max_steps,
                camera=camera,
                mock_env=mock_env,
                seed=seed,
            )

            all_trust, episodes = _parse_telemetry(result.get("telemetry_path"))
            mean_trust = sum(all_trust) / len(all_trust) if all_trust else float("nan")
            episodes_by_condition[condition].extend(episodes)

            rows.append({
                "condition": condition,
                "perception_period": period,
                "mean_success": result["mean_success"],
                "mean_trust": mean_trust,
                "n_trials": result.get("n_trials", n_trials),
                "per_task": result.get("per_task", {}),
                "telemetry_path": result.get("telemetry_path"),
            })

    auroc_by_condition: dict[str, float] = {}
    for condition, episodes in episodes_by_condition.items():
        labels = [1 if not ep["success"] else 0 for ep in episodes]  # 1 = failure
        risk = [-ep["min_trust"] for ep in episodes]  # low trust -> high risk
        auroc_by_condition[condition] = _rank_auroc(labels, risk)

    out = {
        "rows": rows,
        "auroc": auroc_by_condition,
        "meta": {
            "suite": suite,
            "n_trials": n_trials,
            "max_steps": max_steps,
            "camera": camera,
            "mock_env": mock_env,
            "seed": seed,
            "device": device,
            "checkpoint": checkpoint,
            "norm_stats": str(norm_stats_path),
            "perception_periods": list(perception_periods),
            "conditions": list(conditions),
        },
    }
    out_path.write_text(json.dumps(out, indent=2, default=_json_default))
    return out


def _format_table(rows: list[dict], auroc_by_condition: dict[str, float]) -> str:
    """Renders ``run_sweep``'s rows + AUROC dict as a fixed-width text table."""
    headers = ["condition", "perception_period", "mean_success", "mean_trust", "n_trials"]
    str_rows = []
    for r in rows:
        mt = r["mean_trust"]
        str_rows.append({
            "condition": str(r["condition"]),
            "perception_period": str(r["perception_period"]),
            "mean_success": f"{r['mean_success']:.3f}",
            "mean_trust": f"{mt:.3f}" if mt == mt else "n/a",  # nan != nan
            "n_trials": str(r["n_trials"]),
        })
    widths = {h: max(len(h), *(len(sr[h]) for sr in str_rows)) if str_rows else len(h) for h in headers}

    lines = [
        "  ".join(h.ljust(widths[h]) for h in headers),
        "  ".join("-" * widths[h] for h in headers),
    ]
    lines.extend("  ".join(sr[h].ljust(widths[h]) for h in headers) for sr in str_rows)

    lines.append("")
    lines.append("tau -> failure AUROC per condition (risk = -min(trust) over the episode, "
                  "label = episode failure; aggregated across perception_period):")
    for condition, auc in auroc_by_condition.items():
        auc_str = f"{auc:.3f}" if auc == auc else "n/a (no episode-success labels, or only one outcome class, in telemetry)"
        lines.append(f"  {condition:<12s} AUROC={auc_str}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--suite", default="libero_spatial", help="LIBERO suite name")
    p.add_argument("--n-trials", type=int, default=5, help="trials per task")
    p.add_argument("--max-steps", type=int, default=200, help="episode step cap")
    p.add_argument("--camera", default="robot0_eye_in_hand_image")
    p.add_argument(
        "--mock-env", action="store_true",
        help="use eval.libero_eval's built-in MockLiberoEnv (no sim deps, deterministic); "
             "omit to run the real LIBERO/robosuite env",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="never 'mps' -- refused, see hard constraints")
    p.add_argument(
        "--checkpoint", default=None,
        help="checkpoints/full_stageB.pt; omit for fresh (untrained) modules (smoke runs)",
    )
    p.add_argument(
        "--norm-stats", default=None,
        help="norm_stats.json paired with --checkpoint; omit to use the repo-shipped "
             "eval/identity_norm_stats.json (smoke runs)",
    )
    p.add_argument("--periods", type=int, nargs="+", default=list(DEFAULT_PERCEPTION_PERIODS))
    p.add_argument(
        "--conditions", nargs="+", default=list(DEFAULT_CONDITIONS),
        choices=list(DEFAULT_CONDITIONS),
    )
    p.add_argument("--out", default="eval_results/sweep.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.device == "mps":
        print("eval.sweep: refusing --device mps (hard constraint on this machine); "
              "use --device cpu.", file=sys.stderr)
        return 2

    result = run_sweep(
        perception_periods=args.periods,
        conditions=args.conditions,
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        suite=args.suite,
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        camera=args.camera,
        mock_env=args.mock_env,
        seed=args.seed,
        device=args.device,
        out_path=args.out,
    )
    print(_format_table(result["rows"], result["auroc"]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
