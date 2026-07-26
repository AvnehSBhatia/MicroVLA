"""Summarize closed-loop telemetry — is the policy actually commanding motion?

Every closed-loop failure in this project so far has been an INTERFACE defect
between individually correct components (asymmetric normalization, trust-hold
momentum, the geometry bottleneck, inference-mode tensors, bucket
frame-stripping, actuator units), and none of them were visible in the
aggregate success rate. Per-step telemetry found each one. This makes that
check a command instead of a recalled one-liner::

    python -m eval.telemetry_probe                    # newest non-empty file
    python -m eval.telemetry_probe --all              # every worker of the run
    python -m eval.telemetry_probe --file path.jsonl

What to look for:

* ``|cmd| max`` at the clip (1.0) means the waypoint actuator is SATURATING —
  it is commanding faster than the arm's fitted response, so the trajectory is
  bang-bang rather than tracked.
* ``|cmd| mean`` near zero means the actuator is emitting nothing and the arm
  will sit still; suspect the gain, the anchor, or missing proprio.
* ``real ticks`` well below ``steps / perception_period`` means perception is
  not firing on schedule.
* ``trust`` collapsing toward 0 means the corrector thinks the world model has
  diverged, and in ``delta`` action space that BRAKES the plan.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from microvla.utils.signals import ignore_sigterm


def summarize(records: list[dict]) -> dict:
    """Reduces one episode/worker's per-step records to the diagnostic numbers.

    Args:
        records: Parsed telemetry lines (see ``eval.libero_eval.run_eval``).

    Returns:
        Dict of summary statistics; ``{"steps": 0}`` when there is nothing to
        summarize (a worker that has not finished its first trial yet writes an
        empty file — the telemetry path is created BEFORE the policy build so
        that its existence proves the worker started).
    """
    if not records:
        return {"steps": 0}
    out: dict = {"steps": len(records)}
    trials = {(r.get("task"), r.get("trial")) for r in records if "task" in r}
    if trials:
        out["episodes"] = len(trials)
        out["successes"] = len({t for t in trials
                                if any(r.get("success") and (r.get("task"), r.get("trial")) == t
                                       for r in records)})
    real = [r for r in records if r.get("is_real")]
    out["real_ticks"] = len(real)

    cmds = np.array([r["waypoint_cmd"] for r in records if r.get("waypoint_cmd")],
                    dtype=np.float64)
    out["waypoint_steps"] = int(cmds.shape[0])
    if cmds.size:
        out["cmd_abs_mean"] = float(np.abs(cmds).mean())
        out["cmd_abs_max"] = float(np.abs(cmds).max())
        out["cmd_saturated_frac"] = float((np.abs(cmds) >= 0.999).mean())
        out["cmd_per_axis_mean"] = [float(v) for v in np.abs(cmds).mean(axis=0)]
    for key in ("plan_norm", "trust"):
        vals = [r[key] for r in records if key in r]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_min"] = float(np.min(vals))
    return out


def _report(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    s = summarize(records)
    print(f"\n{path.name}")
    if not s["steps"]:
        print("  (empty — the worker has not finished a trial yet)")
        return s
    ep = f"{s.get('successes', '?')}/{s.get('episodes', '?')} succeeded"
    print(f"  steps {s['steps']} | {ep} | real ticks {s['real_ticks']}")
    if s["waypoint_steps"]:
        sat = s["cmd_saturated_frac"]
        flag = "  <-- SATURATED" if sat > 0.05 else ""
        print(f"  waypoint |cmd| mean {s['cmd_abs_mean']:.4f} max {s['cmd_abs_max']:.4f} "
              f"| clipped {100*sat:.1f}% of steps{flag}")
        print(f"  per-axis |cmd| (x, y, z): "
              f"{', '.join(f'{v:.4f}' for v in s['cmd_per_axis_mean'])}")
    else:
        print("  waypoint actuation OFF (no waypoint_cmd in telemetry)")
    if "plan_norm_mean" in s:
        print(f"  plan_norm mean {s['plan_norm_mean']:.3f}", end="")
    if "trust_mean" in s:
        print(f" | trust mean {s['trust_mean']:.3f} min {s['trust_min']:.3f}", end="")
    print()
    return s


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="eval_results", help="telemetry directory")
    ap.add_argument("--file", default=None, help="one specific .jsonl")
    ap.add_argument("--all", action="store_true",
                    help="every non-empty telemetry file of the newest run "
                         "(default: only the most recently written one)")
    args = ap.parse_args(argv)
    ignore_sigterm()

    if args.file:
        _report(Path(args.file))
        return
    # Newest FIRST by mtime, and skip the empties — a worker that just started
    # has created its file but written nothing.
    files = sorted((Path(p) for p in glob.glob(f"{args.dir}/*telemetry.jsonl")),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    live = [p for p in files if p.stat().st_size > 0]
    if not live:
        raise SystemExit(
            f"no non-empty telemetry in {args.dir}/ "
            f"({len(files)} file(s) present). A worker writes its first records "
            f"when its first trial finishes — wait for a DONE line.")
    for p in (live if args.all else live[:1]):
        _report(p)


if __name__ == "__main__":
    main()
