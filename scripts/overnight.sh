#!/usr/bin/env bash
# Overnight paper batch: every stage-B arm with error bars, every bench, the
# closed-loop number, and a paper-ready summary table.
#
#   cd /root/MicroVLA && git pull && bash scripts/overnight.sh
#
# Design notes, all of them learned the hard way this session:
#
#  * RESUMABLE. Every step skips itself if its output already exists, so a
#    re-run continues instead of redoing ~4 hours. Kill it and restart freely.
#  * NEVER exits on error. One dead arm must not cost the whole batch; failures
#    are recorded and the run continues. So NO `set -e`.
#  * SIGTERM is ignored by every CLI here AND by this script (the host reaps
#    jobs; a SIGTERM to the bash wrapper killed an earlier batch mid-run). So
#    `kill` will not stop it. To stop: `touch STOP` (checked between steps), or
#    `kill -9 -<pgid>` — the pgid is printed at startup.
#  * All output goes to logs/overnight/00_progress.log, not the terminal, so
#    losing the terminal cannot break the run. Watch it with `tail -f`.
#  * SEEDS BEFORE ARMS. Measured run-to-run variance on an IDENTICAL command
#    spans std_ratio 0.022-0.245 — larger than any effect this project has
#    claimed from an architecture change (paper.md 4l). So the core arms run at
#    3 seeds and the exploratory ones at 1, and a single-seed number is labelled
#    as one sample in the summary.
#
# Total: ~14 stage-B runs (~20 min each) + benches + one closed-loop eval.
# Budget ~5-6 h on an uncontended card; longer if the box is shared.

cd "$(dirname "$0")/.." || exit 1

# --- survive the host reaper AND a dying terminal ----------------------------
# The Python CLIs each ignore SIGTERM (microvla/utils/signals.py), but THIS
# wrapper had no protection, so a SIGTERM to bash killed the whole batch mid-run
# — observed. `trap ''` sets SIG_IGN, which children also inherit, so the shield
# is now end-to-end.
trap '' TERM HUP
mkdir -p logs/overnight
_LOG="logs/overnight/00_progress.log"
echo "overnight batch: pid $$  pgid $(ps -o pgid= -p $$ | tr -d ' ')"
echo "  watch:  tail -f $_LOG"
echo "  stop :  touch STOP        (checked between steps)"
echo "  force:  kill -9 -\$(ps -o pgid= -p $$ | tr -d ' ')   # SIGTERM is ignored"
# Everything from here goes to the log, so losing the terminal cannot break the
# run's output (a `tee` to a dead tty exits and takes the pipeline with it).
exec >> "$_LOG" 2>&1

export TORCH_BLAS_PREFER_HIPBLASLT=0
export PFX_OSMESA="PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa PYTHONPATH=/root/LIBERO"

DATA="--data-dir data/bridge --data-dir data/libero_v7"
LIBERO_ONLY="--data-dir data/libero_v7"
STAGE_A="checkpoints/full_stageA_wrist_v72.pt"
NORM="data/libero_v7/norm_stats.json"
WPSTATS="data/libero_v7/waypoint_stats.json"
LOGS="logs/overnight"; mkdir -p "$LOGS" eval_results results
SUMMARY="$_LOG"

COMMON="$DATA --device cuda --batch-size 64 --lr 5e-4 --max-vram-gb 50 \
  --load-stage-a $STAGE_A --stage-b-epochs 40 --stage-b-patience 4 \
  --dream-frac 0.25 --waypoint-weight 1.0"

say() { echo "[$(date +%H:%M:%S)] $*"; }
stop_requested() { [ -f STOP ] && { say "STOP file present — exiting cleanly."; return 0; }; return 1; }

# --- preflight ---------------------------------------------------------------
say "=== overnight batch starting: $(git rev-parse --short HEAD) ==="
for f in "$STAGE_A" "$NORM"; do
  [ -f "$f" ] || { say "FATAL: missing $f"; exit 1; }
done
[ -f "$WPSTATS" ] || { say "fitting waypoint gain"; python -m preprocess.fit_waypoint_gain data/libero_v7 2>&1 | tee -a "$LOGS/gain.log"; }
say "episodes: $(ls data/libero_v7/*.npz | wc -l | tr -d ' ') libero, $(ls data/bridge/*.npz | wc -l | tr -d ' ') bridge"

# --- helpers ----------------------------------------------------------------
# train_arm <tag> <extra flags...>   — idempotent, one retry, logged.
train_arm() {
  local tag="$1"; shift
  local ckpt="checkpoints/full_stageB_${tag}.pt"
  if [ -f "$ckpt" ]; then say "SKIP train $tag (checkpoint exists)"; return 0; fi
  stop_requested && return 1
  local try
  for try in 1 2; do
    say "TRAIN $tag (attempt $try)"
    python train/train_batched.py $COMMON "$@" --tag "$tag" \
      > "$LOGS/train_${tag}.log" 2>&1
    local rc=$?
    if [ -f "$ckpt" ]; then
      say "  done: $(grep -c 'stage B\] epoch' "$LOGS/train_${tag}.log") epochs, \
$(grep -o 'best val [0-9.]*' "$LOGS/train_${tag}.log" | tail -1)"
      return 0
    fi
    say "  FAILED (rc=$rc): $(tail -3 "$LOGS/train_${tag}.log" | tr '\n' ' ' | cut -c1-160)"
  done
  return 1
}

# train_arm_data <tag> <data flags> <extra...> — for a different corpus.
train_arm_data() {
  local tag="$1" data="$2"; shift 2
  local ckpt="checkpoints/full_stageB_${tag}.pt"
  if [ -f "$ckpt" ]; then say "SKIP train $tag (checkpoint exists)"; return 0; fi
  stop_requested && return 1
  say "TRAIN $tag (corpus: $data)"
  python train/train_batched.py $data --device cuda --batch-size 64 --lr 5e-4 \
    --max-vram-gb 50 --load-stage-a "$STAGE_A" --stage-b-epochs 40 \
    --stage-b-patience 4 --dream-frac 0.25 --waypoint-weight 1.0 "$@" --tag "$tag" \
    > "$LOGS/train_${tag}.log" 2>&1
  local rc=$?
  [ -f "$ckpt" ] || { say "  FAILED (rc=$rc): $(tail -3 "$LOGS/train_${tag}.log" | tr '\n' ' ' | cut -c1-160)"; return 1; }
  say "  done"
}

# bench_ckpt <name> <checkpoint> [extra flags...] — idempotent.
bench_ckpt() {
  local name="$1" ckpt="$2"; shift 2
  local out="eval_results/bench_${name}.json"
  [ -f "$out" ] && { say "SKIP bench $name"; return 0; }
  [ -f "$ckpt" ] || { say "SKIP bench $name (no $ckpt)"; return 0; }
  stop_requested && return 1
  say "BENCH $name"
  python -m eval.bench --checkpoint "$ckpt" --data-dir data/libero_v7 \
    --sensitivity --episodes 30 --device cuda:0 --out "$out" "$@" \
    > "$LOGS/bench_${name}.log" 2>&1
  grep -m1 AGGREGATE "$LOGS/bench_${name}.log" \
    || say "  FAILED bench $name: $(tail -3 "$LOGS/bench_${name}.log" | tr '\n' ' ' | cut -c1-160)"
}

# ============================================================================
# 1. VARIANCE ERROR BARS — the highest-value measurement. 3 seeds each for the
#    two configurations every comparison in paper.md rests on.
# ============================================================================
say "--- phase 1: error bars (6 runs) ---"
for S in 0 1 2; do
  train_arm "native_s$S"  --seed "$S"
  train_arm "longh_s$S"   --waypoint-long --seed "$S"
done

# ============================================================================
# 2. THE ARMS — one seed each; the summary labels them as single samples.
# ============================================================================
say "--- phase 2: arms (6 runs) ---"
train_arm "longh_pregrasp" --waypoint-long --pre-grasp-weight 3.0
train_arm "longh_sdfade"   --waypoint-long --planner-drop-rate 'state_delta=0.4'
train_arm "longh_all"      --waypoint-long --pre-grasp-weight 3.0 \
                           --planner-drop-rate 'state_delta=0.4'
train_arm "longh_novis"    --waypoint-long --planner-input-dropout 0.0
train_arm "longh_tqsa"     --waypoint-long --tqsa
train_arm_data "longh_liberoonly" "$LIBERO_ONLY" --waypoint-long

# ============================================================================
# 3. BENCH EVERYTHING, including the older checkpoints — they were measured on
#    the COMBINED sensitivity instrument, which mixed pose with the discrete
#    gripper bit (paper.md 4i). Re-benching them here is the only way the
#    cross-arm comparison is like-for-like.
# ============================================================================
say "--- phase 3: bench all ---"
for S in 0 1 2; do
  bench_ckpt "native_s$S" "checkpoints/full_stageB_native_s$S.pt"
  bench_ckpt "longh_s$S"  "checkpoints/full_stageB_longh_s$S.pt"
done
for T in longh_pregrasp longh_sdfade longh_all longh_novis longh_liberoonly; do
  bench_ckpt "$T" "checkpoints/full_stageB_${T}.pt"
done
bench_ckpt "longh_tqsa"    "checkpoints/full_stageB_longh_tqsa.pt" --tqsa
# The 2-minute run paper.md 4b owes: the SAME TQSA checkpoint scored WITHOUT
# spatial, which attributes its gripper collapse to the input vs the head.
bench_ckpt "longh_tqsa_nospatial" "checkpoints/full_stageB_longh_tqsa.pt"
# Any pre-existing checkpoints, on the new instrument.
for f in checkpoints/full_stageB.pt checkpoints/full_stageB_longh.pt \
         checkpoints/full_stageB_wristwp.pt; do
  [ -f "$f" ] && bench_ckpt "legacy_$(basename "$f" .pt)" "$f"
done

# ============================================================================
# 4. CLOSED LOOP on the best arm by std_ratio — the number the paper is missing.
# ============================================================================
say "--- phase 4: closed loop ---"
BEST=$(python - <<'PY'
import glob, json
best = (None, -1)
for f in glob.glob("eval_results/bench_*.json"):
    try:
        a = json.load(open(f))["aggregate"]
        sr = a.get("std_ratio")
        if sr and sr == sr and a.get("grip_acc", 0) > 0.7 and sr > best[1]:
            best = (f, sr)
    except Exception:
        pass
name = best[0].split("bench_")[1][:-5] if best[0] else ""
print(name)
PY
)
if [ -n "$BEST" ] && [ -f "checkpoints/full_stageB_${BEST}.pt" ]; then
  say "best arm by std_ratio (with grip > 0.7): $BEST"
  if [ ! -f "$LOGS/closedloop_${BEST}.log" ]; then
    env $PFX_OSMESA python -m eval.libero_eval --suite libero_object \
      --n-trials 5 --max-steps 300 \
      --checkpoint "checkpoints/full_stageB_${BEST}.pt" \
      --norm-stats "$NORM" --waypoint-stats "$WPSTATS" \
      --device cuda:0 --heads-device cuda:0 \
      --workers 5 --stagger 10 --worker-timeout 3600 \
      > "$LOGS/closedloop_${BEST}.log" 2>&1
    grep -E '"mean_success"|tasks_completed' "$LOGS/closedloop_${BEST}.log"
    python -m eval.telemetry_probe --all > "$LOGS/telemetry_${BEST}.log" 2>&1
    tail -20 "$LOGS/telemetry_${BEST}.log"
  else
    say "SKIP closed loop (log exists)"
  fi
else
  say "no benched arm with grip > 0.7 — skipping closed loop"
fi

# ============================================================================
# 5. PAPER TABLE
# ============================================================================
say "--- phase 5: summary ---"
python - <<'PY' | tee results/PAPER_TABLE.md
import glob, json, os, statistics as st

rows = {}
for f in sorted(glob.glob("eval_results/bench_*.json")):
    name = os.path.basename(f)[len("bench_"):-len(".json")]
    try:
        d = json.load(open(f))
    except Exception:
        continue
    a = d.get("aggregate", {})
    sens = d.get("sensitivity") or {}
    pose = sens.get("pose", sens if "fused" in sens else {})
    rows[name] = (a, pose)

def g(a, k):
    v = a.get(k)
    return v if isinstance(v, (int, float)) and v == v else None

def fmt(v, n=3):
    return "—" if v is None else f"{v:.{n}f}"

print("# MicroVLA — overnight batch results\n")
print("Auto-generated by `scripts/overnight.sh`. `paper.md` is the narrative;")
print("`results/metrics.jsonl` is the durable store.\n")

print("## Bench\n")
print("| arm | std_ratio | wp_std_ratio | corr | grip | pose_mae | wm_margin |")
print("|---|---|---|---|---|---|---|")
for n, (a, _) in rows.items():
    print(f"| {n} | {fmt(g(a,'std_ratio'))} | {fmt(g(a,'wp_std_ratio'))} | "
          f"{fmt(g(a,'corr'),2)} | {fmt(g(a,'grip_acc'),2)} | "
          f"{fmt(g(a,'pose_mae'))} | {fmt(g(a,'wm_margin_pct') or (g(a,'wm_margin') or 0)*100,1)}% |")

print("\n## Seed spread — the error bar every single-run A/B needs\n")
for base in ("native", "longh"):
    vals = {k: g(a, "std_ratio") for k, (a, _) in rows.items()
            if k.startswith(base + "_s") and g(a, "std_ratio") is not None}
    if len(vals) >= 2:
        v = list(vals.values())
        print(f"**{base}** std_ratio over {len(v)} seeds: "
              f"mean {st.mean(v):.3f}, sd {st.stdev(v):.3f}, "
              f"range {min(v):.3f}-{max(v):.3f}   `{vals}`\n")
    elif vals:
        print(f"**{base}**: only {len(vals)} seed — single sample, no error bar.\n")

print("\n## Planner input sensitivity (POSE only; the gripper bit is excluded)\n")
keys = ["proprio", "state_delta", "fused", "geometry", "wm_msg", "wm_latent",
        "current_emb", "pred_box_emb", "next_emb->stale"]
present = [n for n, (_, p) in rows.items() if p]
if present:
    print("| arm | " + " | ".join(keys) + " | phase:vision |")
    print("|---" * (len(keys) + 2) + "|")
    for n in present:
        p = rows[n][1]
        phase = (p.get("proprio", 0) or 0) + (p.get("state_delta", 0) or 0)
        vis = (p.get("fused", 0) or 0) + (p.get("geometry", 0) or 0)
        ratio = f"{phase/vis:.1f}:1" if vis > 1e-9 else "—"
        print(f"| {n} | " + " | ".join(fmt(p.get(k), 4) for k in keys) + f" | {ratio} |")

print("\n## Within-run claim: does displacement regress less shrunk than action?\n")
print("| arm | action std_ratio | waypoint std_ratio | ratio |")
print("|---|---|---|---|")
for n, (a, _) in rows.items():
    s, w = g(a, "std_ratio"), g(a, "wp_std_ratio")
    if s and w:
        print(f"| {n} | {s:.3f} | {w:.3f} | **{w/s:.1f}x** |")
print("\nThis comparison is measured WITHIN one forward pass, so the cross-run")
print("variance that undermines the arm rankings does not apply to it.")
PY

say "=== done. table: results/PAPER_TABLE.md | logs: $LOGS ==="
say "failures (if any):"; grep -h FAILED "$SUMMARY" || true
