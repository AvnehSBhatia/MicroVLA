"""Durable, append-only experiment tracking for the MicroVLA paper.

Every training epoch, eval, and provenance fact is one JSON record appended to
``results/metrics.jsonl`` — never overwritten (unlike stdout logs, which a
re-run clobbers). Each record is stamped with a wall-clock ISO time and the
current git SHA so any number in the paper is traceable to exact code + data.

Record schema (loose; ``kind`` selects the fields that matter):
    {ts, git, run_id, kind, ...}
  kind="train_epoch": stage, epoch, epochs, horizon, train_loss, val_loss,
                      persistence, margin_pct, duration_s, device, recipe, note
  kind="horizon_curve": horizon, val_loss, persistence, margin_pct, checkpoint,
                        n_episodes
  kind="provenance":  dataset composition, config, checkpoint hashes, notes
  kind="eval":        suite, condition, perception_period, mean_success, ...

CLI:
    python -m experiments.tracker report      # regenerate results/RESULTS.md
    python -m experiments.tracker ingest LOG RUN_ID [--recipe R] [--device D]
        # parse a train_full stdout log's epoch lines into the store
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STORE = RESULTS_DIR / "metrics.jsonl"
REPORT = RESULTS_DIR / "RESULTS.md"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=RESULTS_DIR.parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def log(record: dict) -> dict:
    """Appends one record (stamped with ts + git SHA) to the store."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamped = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "git": _git_sha(), **record}
    with STORE.open("a") as f:
        f.write(json.dumps(stamped) + "\n")
    return stamped


def load() -> list[dict]:
    """Returns every stored record (empty if the store does not exist)."""
    if not STORE.exists():
        return []
    return [json.loads(line) for line in STORE.read_text().splitlines() if line.strip()]


_EPOCH_RE = re.compile(
    r"\[stage (?P<stage>[AB])\] epoch (?P<epoch>\d+)/(?P<epochs>\d+)"
    r"(?: \| H=(?P<H>\d+))?"
    r" \| train (?P<train>[\d.]+)"
    r"(?: \| val (?P<val>[\d.]+) vs persistence (?P<pers>[\d.]+))?"
    r"(?: \| bc (?P<bc>[\d.]+) \| smooth (?P<smooth>[\d.]+))?"
    r".*?\| (?P<dur>\d+)s"
)


def ingest(log_path: str | Path, run_id: str, recipe: str = "", device: str = "") -> int:
    """Parses a train_full stdout log's epoch lines into the store.

    Idempotency is the caller's job (dedupe by run_id if re-ingesting); this
    appends whatever it parses.

    Returns the number of epoch records written.
    """
    text = Path(log_path).read_text()
    n = 0
    for m in _EPOCH_RE.finditer(text):
        d = m.groupdict()
        rec = {"run_id": run_id, "kind": "train_epoch",
               "stage": d["stage"], "epoch": int(d["epoch"]), "epochs": int(d["epochs"]),
               "duration_s": int(d["dur"]), "device": device, "recipe": recipe}
        if d["H"]:
            rec["horizon"] = int(d["H"])
        rec["train_loss"] = float(d["train"])
        if d["val"]:
            v, p = float(d["val"]), float(d["pers"])
            rec.update(val_loss=v, persistence=p,
                       margin_pct=round((p - v) / p * 100, 1))
        if d["bc"]:
            rec.update(bc_loss=float(d["bc"]), smooth=float(d["smooth"]))
        log(rec)
        n += 1
    return n


def report() -> Path:
    """Regenerates results/RESULTS.md from the store."""
    recs = load()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# MicroVLA — Results (auto-generated from results/metrics.jsonl)", "",
             f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
             f"{len(recs)} records · **do not hand-edit** (regenerate: "
             "`python -m experiments.tracker report`)", ""]

    prov = [r for r in recs if r.get("kind") == "provenance"]
    if prov:
        lines += ["## Provenance", ""]
        for r in prov:
            lines.append(f"- `{r['git']}` {r['ts']} — {r.get('note', '')}")
        lines.append("")

    epochs = [r for r in recs if r.get("kind") == "train_epoch"]
    if epochs:
        lines += ["## Stage-A world model (rollout loss vs persistence)", "",
                  "| run | recipe | ep | H | train | val | persistence | margin | s |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in epochs:
            if r["stage"] != "A":
                continue
            m = f"{r['margin_pct']:+.0f}%" if "margin_pct" in r else "—"
            lines.append(
                f"| {r['run_id']} | {r.get('recipe', '')} | {r['epoch']}/{r['epochs']} "
                f"| {r.get('horizon', '—')} | {r.get('train_loss', '—')} "
                f"| {r.get('val_loss', '—')} | {r.get('persistence', '—')} | {m} "
                f"| {r['duration_s']} |")
        lines.append("")
        stageb = [r for r in epochs if r["stage"] == "B"]
        if stageb:
            lines += ["## Stage-B policy (behavior cloning)", "",
                      "| run | ep | bc | smooth | s |", "|---|---|---|---|---|"]
            for r in stageb:
                lines.append(f"| {r['run_id']} | {r['epoch']}/{r['epochs']} "
                             f"| {r.get('bc_loss', '—')} | {r.get('smooth', '—')} "
                             f"| {r['duration_s']} |")
            lines.append("")

    hc = [r for r in recs if r.get("kind") == "horizon_curve"]
    if hc:
        lines += ["## Horizon curve (Claim 2 early evidence — margin vs rollout depth)", "",
                  "| checkpoint | H | val | persistence | margin |",
                  "|---|---|---|---|---|"]
        for r in hc:
            lines.append(f"| {r.get('checkpoint', '')} | {r['horizon']} | {r['val_loss']} "
                         f"| {r['persistence']} | {r['margin_pct']:+.0f}% |")
        lines.append("")

    ev = [r for r in recs if r.get("kind") == "eval"]
    if ev:
        lines += ["## Closed-loop / sweep eval", "",
                  "| suite | condition | period | success | trust |",
                  "|---|---|---|---|---|"]
        for r in ev:
            lines.append(f"| {r.get('suite', '')} | {r.get('condition', '')} "
                         f"| {r.get('perception_period', '')} | {r.get('mean_success', '')} "
                         f"| {r.get('mean_trust', '')} |")
        lines.append("")

    REPORT.write_text("\n".join(lines))
    return REPORT


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report")
    ing = sub.add_parser("ingest")
    ing.add_argument("log"); ing.add_argument("run_id")
    ing.add_argument("--recipe", default=""); ing.add_argument("--device", default="")
    args = ap.parse_args(argv)
    if args.cmd == "report":
        print("wrote", report())
    elif args.cmd == "ingest":
        n = ingest(args.log, args.run_id, args.recipe, args.device)
        report()
        print(f"ingested {n} epoch records from {args.log}; report regenerated")


if __name__ == "__main__":
    main()
