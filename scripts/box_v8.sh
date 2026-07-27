#!/usr/bin/env bash
# ============================================================================
# v8 on the box: re-bake LIBERO with SIGHTED perception, then train big.
#
# WHY A RE-BAKE IS MANDATORY. data/libero_v7 on this machine was baked with
# set_classes([source, target]), and YOLO-World-S returns EXACTLY 0.000 for
# every LIBERO product name ("alphabet soup", "bbq sauce", ...). Measured: the
# source object is detected on 0.0% of frames in BOTH camera views, so
# box_weight is 0 everywhere and fusion fades all box and geometry evidence to
# nothing. Every model this project has trained had no object input at all
# (paper.md 4n). Training v8 on that corpus would give the relational head two
# identically-zero slots to reason over.
#
# After the fix (prompt fallback chain + class-agnostic proposals), a fresh
# libero_object bake detects the source on 45.1% of steps.
#
# Reaper-proof: SIGTERM/HUP ignored, output via exec (a tee to a dead tty exits
# and takes the pipeline with it). `touch STOP` to halt; `kill` will not work.
# ============================================================================
trap '' TERM HUP
set -u
cd "$(dirname "$0")/.." || exit 1

mkdir -p logs/box_v8 data checkpoints eval_results
LOG=logs/box_v8/00_progress.log
echo "box v8: pid $$  pgid $(ps -o pgid= -p $$ | tr -d ' ')"
echo "  watch:  tail -f $LOG"
echo "  stop :  touch STOP"
echo "  force:  kill -9 -$(ps -o pgid= -p $$ | tr -d ' ')"
exec >> "$LOG" 2>&1

say() { echo "[$(date +%H:%M:%S)] $*"; }
stopped() { [ -f STOP ] && { say "STOP present — exiting."; return 0; }; return 1; }

SUITES="${SUITES:-libero_object libero_spatial libero_goal}"
RAW="${RAW:-/root/libero_raw}"
DEV="${DEV:-cuda}"
# Detection-rate floor. Below this the corpus is effectively blind and training
# on it is what produced every previous null result.
GATE_PCT="${GATE_PCT:-20}"

say "=== box v8 start: $(git rev-parse --short HEAD) | suites: $SUITES ==="
say "disk: $(df -h . | tail -1 | awk '{print $4}') free"

# ---------------------------------------------------------------------------
# 1. Per suite: download -> bake (wrist) -> GATE -> delete raw.
#    ONE SUITE RESIDENT AT A TIME: preprocess/libero.py globs recursively, and
#    all three raw suites together are ~21 GB.
# ---------------------------------------------------------------------------
BAKED=""
for S in $SUITES; do
  stopped && break
  OUT="data/${S}_v8"
  if [ -d "$OUT" ] && [ "$(ls "$OUT"/*.npz 2>/dev/null | wc -l)" -gt 100 ]; then
    say "SKIP $S (already baked: $(ls "$OUT"/*.npz | wc -l) episodes)"
    BAKED="$BAKED --data-dir $OUT"; continue
  fi

  say "--- $S: download ---"
  mkdir -p "$RAW"
  if [ ! -d "$RAW/$S" ] || [ "$(ls "$RAW/$S"/*.hdf5 2>/dev/null | wc -l)" -eq 0 ]; then
    if [ -f /root/LIBERO/benchmark_scripts/download_libero_datasets.py ]; then
      yes n | python /root/LIBERO/benchmark_scripts/download_libero_datasets.py \
        --datasets "$S" --download-dir "$RAW" 2>&1 | tail -3
    else
      # Direct HuggingFace fallback — verified working.
      mkdir -p "$RAW/$S"
      BASE="https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/$S"
      for f in $(curl -sS "https://huggingface.co/api/datasets/yifengzhu-hf/LIBERO-datasets" \
                 | python -c "import json,sys;print('\n'.join(x['rfilename'] for x in json.load(sys.stdin)['siblings'] if x['rfilename'].startswith('$S/')))"); do
        n=$(basename "$f")
        [ -s "$RAW/$S/$n" ] || curl -sSL --retry 3 -o "$RAW/$S/$n" "$BASE/$n"
      done
    fi
  fi
  N_RAW=$(ls "$RAW/$S"/*.hdf5 2>/dev/null | wc -l)
  say "  raw: $N_RAW files, $(du -sh "$RAW/$S" 2>/dev/null | cut -f1)"
  [ "$N_RAW" -eq 0 ] && { say "  DOWNLOAD FAILED for $S — skipping"; continue; }

  say "--- $S: bake (wrist view, sighted prompts, class-agnostic proposals) ---"
  # --camera eye_in_hand_rgb is REQUIRED: eval reads robot0_eye_in_hand_image,
  # and baking agentview trains the policy on a view it never sees (4f).
  python -m preprocess.libero "$RAW/$S" "$OUT" \
    --camera eye_in_hand_rgb --device "$DEV" 2>&1 | tail -4
  say "  baked $(ls "$OUT"/*.npz 2>/dev/null | wc -l) episodes, $(du -sh "$OUT" 2>/dev/null | cut -f1)"

  say "--- $S: GATE (is the corpus sighted?) ---"
  python - "$OUT" "$GATE_PCT" <<'PY'
import glob, sys
import numpy as np
out, floor = sys.argv[1], float(sys.argv[2])
fs = sorted(glob.glob(f"{out}/*.npz"))[:120]
if not fs:
    print("GATE FAILED: no episodes"); sys.exit(1)
w = np.concatenate([np.load(f)["box_weights"] for f in fs])
c = np.concatenate([np.load(f)["source_centers"] for f in fs])
src, tgt = float((w[:, 0] > 0).mean()) * 100, float((w[:, 1] > 0).mean()) * 100
o = np.load(fs[0])
print(f"  source {src:.1f}% | target {tgt:.1f}% | weight mean {w.mean():.4f} "
      f"| center std {c.std(0).round(3).tolist()} | obj_* {'obj_embs' in o.files}")
if src < floor:
    print(f"GATE FAILED: source detection {src:.1f}% < {floor}% — this corpus is "
          f"BLIND and training on it reproduces the null result.")
    sys.exit(1)
print("GATE PASSED")
PY
  if [ $? -ne 0 ]; then say "  GATE FAILED for $S — not adding it to the corpus"; continue; fi

  BAKED="$BAKED --data-dir $OUT"
  say "--- $S: delete raw ($(du -sh "$RAW/$S" | cut -f1)) ---"
  rm -rf "${RAW:?}/${S:?}"
  say "  disk: $(df -h . | tail -1 | awk '{print $4}') free"
done

[ -z "$BAKED" ] && { say "NO SIGHTED CORPUS — aborting."; exit 1; }
say "=== corpus:$BAKED ==="

# ---------------------------------------------------------------------------
# 2. One shared normalizer + waypoint gain across the suites.
# ---------------------------------------------------------------------------
NDIRS=$(echo "$BAKED" | grep -o '\-\-data-dir' | wc -l)
if [ "$NDIRS" -gt 1 ]; then
  say "=== unify norm stats across $NDIRS suites ==="
  python -m preprocess.unify_norm_stats $BAKED 2>&1 | tail -4
fi
FIRST=$(echo "$BAKED" | awk '{print $2}')
say "=== fit waypoint gain on $FIRST ==="
python -m preprocess.fit_waypoint_gain "$FIRST" \
  --out "$FIRST/waypoint_stats.json" 2>&1 | tail -5

# ---------------------------------------------------------------------------
# 3. Train v8. Bigger budget than the Mac run, and the stop criterion fixed:
#    --stage-b-select bc is the only term on a scale shared across arms, and
#    --stage-b-min-epochs stops a noisy plateau from ending a run at epoch 8
#    (the confound that made every bench metric track epochs-survived, 4m).
# ---------------------------------------------------------------------------
stopped || {
  say "=== train v8 (stage A 60 / stage B 100) ==="
  python -u train/train_batched.py $BAKED \
    --device "$DEV" --batch-size 64 --lr 5e-4 --max-vram-gb 50 \
    --stage-a-epochs 60 --warmup-epochs 5 --max-horizon 6 --patience 5 \
    --stage-b-epochs 100 --stage-b-patience 6 \
    --stage-b-select bc --stage-b-min-epochs 30 \
    --dream-frac 0.25 --waypoint-weight 1.0 --waypoint-long \
    --v8 --tag v8_big 2>&1 | tail -200
}

CKPT=checkpoints/full_stageB_v8_big.pt
[ -f "$CKPT" ] || { say "NO STAGE-B CHECKPOINT — stopping before eval."; exit 1; }

# ---------------------------------------------------------------------------
# 4. Bench, then the number this project has never had: closed-loop success.
# ---------------------------------------------------------------------------
say "=== bench ==="
python -u -m eval.bench --checkpoint "$CKPT" --data-dir "$FIRST" \
  --sensitivity --episodes 30 --device "${DEV}:0" \
  --out eval_results/bench_v8_big.json 2>&1 | tail -30

say "=== closed loop (real sim) ==="
for S in $SUITES; do
  stopped && break
  [ -d "data/${S}_v8" ] || continue
  say "--- closed loop: $S ---"
  env PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa PYTHONPATH=/root/LIBERO \
  python -u -m eval.libero_eval --suite "$S" --n-trials 5 --max-steps 300 \
    --checkpoint "$CKPT" \
    --norm-stats "$FIRST/norm_stats.json" \
    --waypoint-stats "$FIRST/waypoint_stats.json" \
    --device "${DEV}:0" --heads-device "${DEV}:0" \
    --workers 5 --stagger 10 --worker-timeout 3600 \
    > "logs/box_v8/closedloop_${S}.log" 2>&1
  grep -E '"mean_success"|tasks_completed' "logs/box_v8/closedloop_${S}.log" | head -2
done

say "=== DONE. bench: eval_results/bench_v8_big.json | logs: logs/box_v8 ==="
