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
echo "  log  :  $LOG   (also printed here, live)"
echo "  stop :  touch STOP"
echo "  force:  kill -9 -$(ps -o pgid= -p $$ | tr -d ' ')"

# Keep the ORIGINAL terminal on fd 3 before redirecting everything to the log,
# so progress can go to both. Deliberately not `tee`: a tee whose reader dies
# (closed terminal, dropped ssh) exits and takes the whole pipeline with it,
# which is how a previous overnight run died. Every write to fd 3 is guarded, so
# losing the terminal costs the live view and nothing else.
exec 3>&1
exec >> "$LOG" 2>&1

#: Print to the log AND the terminal.
say() {
  local line="[$(date +%H:%M:%S)] $*"
  echo "$line"
  echo "$line" >&3 2>/dev/null || true
}

#: Run a command, streaming its output to BOTH a log file and the terminal.
#: Sets RC to the command's own exit status (not the loop's).
run_logged() {
  local logfile="$1"; shift
  "$@" > >(while IFS= read -r l; do
             printf '%s\n' "$l" >> "$logfile"
             printf '  %s\n' "$l" >&3 2>/dev/null || true
           done) 2>&1
  RC=$?
  wait 2>/dev/null || true
}
stopped() { [ -f STOP ] && { say "STOP present — exiting."; return 0; }; return 1; }

SUITES="${SUITES:-libero_object libero_spatial libero_goal}"
RAW="${RAW:-/root/libero_raw}"
DEV="${DEV:-cuda}"
# Detection-rate floor. Below this the corpus is effectively blind and training
# on it is what produced every previous null result.
GATE_PCT="${GATE_PCT:-20}"

# A leftover STOP from a previous run aborts instantly and looks like a bug.
if [ -f STOP ]; then
  say "removing stale STOP file from a previous run"
  rm -f STOP
fi
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
  MIN_EP="${MIN_EP:-300}"
  HAVE=$(ls "$OUT"/*.npz 2>/dev/null | wc -l | tr -d ' ')
  # Threshold matches the partial-bake gate below. A fixed 100 here would have
  # made any partial bake of >100 episodes permanently "done" and never retried.
  if [ "$HAVE" -ge "$MIN_EP" ]; then
    say "SKIP $S (already baked: $HAVE episodes)"
    BAKED="$BAKED --data-dir $OUT"; continue
  fi
  [ "$HAVE" -gt 0 ] && say "  $S has $HAVE episodes (< $MIN_EP) — re-baking"

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
  run_logged "logs/box_v8/bake_${S}.log" \
    python -u -m preprocess.libero "$RAW/$S" "$OUT" \
    --camera eye_in_hand_rgb --device "$DEV"
  N_EP=$(ls "$OUT"/*.npz 2>/dev/null | wc -l | tr -d ' ')
  say "  baked $N_EP episodes, $(du -sh "$OUT" 2>/dev/null | cut -f1)"
  # A LIBERO suite is ~500 episodes (10 tasks x ~50 demos). preprocess/libero.py
  # can die partway through a suite and leave a valid-looking directory, so a
  # count far below that means a partial bake, not a small suite. Training on
  # 85 episodes instead of 500 is what produced a stage A that never beat
  # persistence: with H=6 rollouts the val set is a handful of episodes.
  if [ "$N_EP" -lt "$MIN_EP" ]; then
    say "  PARTIAL BAKE: $N_EP < $MIN_EP episodes for $S."
    say "  Raw kept at $RAW/$S for a retry; see logs/box_v8/bake_${S}.log."
    say "  Re-run to resume, or set MIN_EP=<n> to accept a smaller suite."
    continue
  fi

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
  say "  $S ACCEPTED: $N_EP episodes"
  say "--- $S: delete raw ($(du -sh "$RAW/$S" | cut -f1)) ---"
  rm -rf "${RAW:?}/${S:?}"
  say "  disk: $(df -h . | tail -1 | awk '{print $4}') free"
done

[ -z "$BAKED" ] && { say "NO SIGHTED CORPUS — aborting."; exit 1; }
TOTAL_EP=0
for d in $(echo "$BAKED" | tr ' ' '\n' | grep -v '^--data-dir$' | grep -v '^$'); do
  TOTAL_EP=$((TOTAL_EP + $(ls "$d"/*.npz 2>/dev/null | wc -l | tr -d ' ')))
done
say "=== corpus:$BAKED  ($TOTAL_EP episodes) ==="
MIN_TOTAL="${MIN_TOTAL:-400}"
if [ "$TOTAL_EP" -lt "$MIN_TOTAL" ]; then
  say "CORPUS TOO SMALL ($TOTAL_EP < $MIN_TOTAL). Stage A on ~140 episodes never"
  say "beat persistence; that is a data problem, not an architecture one."
  say "Fix the bakes and re-run, or set MIN_TOTAL=<n> to override."
  exit 1
fi

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

# --- 3. BLIND CONTROL: same frames/actions/architecture, evidence removed. ---
# This is what ATTRIBUTES the result. Built by zeroing weights rather than
# re-baking, so the two corpora differ in exactly one thing.
BLIND=""
for d in $(echo "$BAKED" | tr ' ' '\n' | grep '_v8$'); do
  bd="${d%_v8}_blind"; BLIND="$BLIND --data-dir $bd"
  [ "$(ls "$bd"/*.npz 2>/dev/null | wc -l)" -gt 0 ] && { say "SKIP blind $bd"; continue; }
  say "=== blind control $d -> $bd ==="; mkdir -p "$bd"
  python - "$d" "$bd" <<'PY2'
import glob, pathlib, sys
import numpy as np
src, dst = sys.argv[1], pathlib.Path(sys.argv[2]); n = 0
for f in sorted(glob.glob(f"{src}/*.npz")):
    d = dict(np.load(f))
    d["box_weights"] = np.zeros_like(d["box_weights"])
    d["source_centers"] = np.full_like(d["source_centers"], 0.5)
    d["target_centers"] = np.full_like(d["target_centers"], 0.5)
    if "obj_weights" in d:
        d["obj_weights"] = np.zeros_like(d["obj_weights"])
        d["obj_centers"] = np.full_like(d["obj_centers"], 0.5)
    np.savez_compressed(dst / pathlib.Path(f).name, **d); n += 1
print(f"  wrote {n} blind episodes")
PY2
  cp "$d/norm_stats.json" "$d/waypoint_stats.json" "$bd/" 2>/dev/null || true
done

# --- 4. ARMS, ordered by paper value so a short night still yields the claim ---
# expandable_segments cuts fragmentation, which is what turned "9.6 GB held on a
# 192 GB card" into an OOM while other tenants owned the rest.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
BATCHES="${BATCHES:-64 32 16 8}"
COMMON="--device $DEV --lr 5e-4 --max-vram-gb ${VRAM_GB:-40} --dream-frac 0.25 --waypoint-weight 1.0 --waypoint-long --stage-b-select bc --stage-b-min-epochs 30"
SA="--stage-a-epochs 60 --warmup-epochs 5 --max-horizon 6 --patience 5"
SB="--stage-b-epochs 100 --stage-b-patience 6"

# Retry on OOM at a smaller batch. The box is SHARED: the previous run died with
# "0 bytes free" on a 192 GB card while holding only 9.6 GB, i.e. other tenants
# owned the rest. A fixed batch size cannot be right on a machine whose free
# memory is somebody else's variable.
_train_retry() {
  local tag="$1" ck="$2"; shift 2
  local b
  for b in $BATCHES; do
    stopped && return 1
    say "  $tag: trying batch $b"
    run_logged "logs/box_v8/train_${tag}.log" \
      python -u train/train_batched.py --batch-size "$b" "$@" --tag "$tag"
    [ -f "$ck" ] && { say "  $tag OK at batch $b"; return 0; }
    if grep -qi "out of memory" "logs/box_v8/train_${tag}.log"; then
      # Distinguish OUR footprint from the box being full. "0 bytes is free"
      # while we hold ~2 GB of a 192 GB card means other tenants own it, and
      # shrinking our batch cannot help — it just burns another run. Observed:
      # batch 32 OOM'd at peakVRAM 5.2 GB, batch 16 at 2.2 GB, both on a 32 MB
      # allocation.
      if grep -q "0 bytes is free" "logs/box_v8/train_${tag}.log"; then
        say "  $tag: card is FULL (0 bytes free, we held only a few GB) —"
        say "  external contention, not our batch. Waiting ${WAIT_S:-600}s at batch $b."
        BATCHES="$b $BATCHES"        # retry this size first once memory frees
        sleep "${WAIT_S:-600}"
        WAITS=$((${WAITS:-0} + 1))
        if [ "$WAITS" -ge "${MAX_WAITS:-6}" ]; then
          say "  $tag: still contended after $WAITS waits — giving up on this arm."
          return 1
        fi
        continue
      fi
      say "  $tag OOM at batch $b — retrying smaller"; continue
    fi
    say "  $tag rc=$RC, not an OOM — see logs/box_v8/train_${tag}.log"; return 1
  done
  say "  $tag FAILED at every batch size ($BATCHES)"; return 1
}

arm_full() {
  local tag="$1"; shift; local ck="checkpoints/full_stageB_${tag}.pt"
  [ -f "$ck" ] && { say "SKIP arm $tag"; return 0; }; stopped && return 1
  say "=== ARM $tag (stage A + B) ==="
  _train_retry "$tag" "$ck" $COMMON $SA $SB "$@"
}

arm_stageb() {
  local tag="$1" sa="$2"; shift 2; local ck="checkpoints/full_stageB_${tag}.pt"
  [ -f "$ck" ] && { say "SKIP arm $tag"; return 0; }
  [ -f "$sa" ] || { say "SKIP arm $tag (no $sa)"; return 1; }; stopped && return 1
  # A stage A that died during the horizon ramp is WORSE than persistence and
  # poisons every arm built on it — the previous run measured wm_margin -46.8%
  # across three arms that all loaded such a checkpoint. Refuse it.
  if [ -f logs/box_v8/train_v8_s0.log ] && \
     ! grep -q "BEATS persistence" logs/box_v8/train_v8_s0.log; then
    say "  SKIP arm $tag: $sa never beat persistence (see train_v8_s0.log)."
    say "  Every arm built on it inherits a broken world model."
    return 1
  fi
  say "=== ARM $tag (stage B only) ==="
  _train_retry "$tag" "$ck" $COMMON $SB --load-stage-a "$sa" "$@"
}

SA_MAIN=checkpoints/full_stageA_v8_s0.pt

arm_full   v8_s0     $BAKED --v8 --seed 0
arm_full   v8_blind  $BLIND --v8 --seed 0
arm_stageb v8_s1 "$SA_MAIN" $BAKED --v8 --seed 1
arm_stageb v8_s2 "$SA_MAIN" $BAKED --v8 --seed 2
arm_full   v7_arch  $BAKED --seed 0
arm_stageb v8_norel "$SA_MAIN" $BAKED --v8 --seed 0 --planner-drop relational

# --- 5. bench every arm (blind arm scored on the blind corpus) ---
say "=== bench ==="
for tag in v8_s0 v8_blind v8_s1 v8_s2 v7_arch v8_norel; do
  ck="checkpoints/full_stageB_${tag}.pt"; out="eval_results/bench_${tag}.json"
  [ -f "$ck" ] || continue; [ -f "$out" ] && continue; stopped && break
  bdir="$FIRST"; case "$tag" in v8_blind) bdir="${FIRST%_v8}_blind";; esac
  run_logged "logs/box_v8/bench_${tag}.log" python -u -m eval.bench --checkpoint "$ck" \
    --data-dir "$bdir" --sensitivity --episodes 30 --device "${DEV}:0" --out "$out"
done

# --- 6. closed loop, main + control first ---
say "=== closed loop ==="
for tag in v8_s0 v8_blind v7_arch v8_s1 v8_s2 v8_norel; do
  ck="checkpoints/full_stageB_${tag}.pt"; [ -f "$ck" ] || continue
  for S in $SUITES; do
    [ -d "data/${S}_v8" ] || continue
    lg="logs/box_v8/cl_${tag}_${S}.log"; [ -f "$lg" ] && continue
    stopped && break 2
    say "--- closed loop $tag / $S ---"
    env PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa PYTHONPATH=/root/LIBERO \
    python -u -m eval.libero_eval --suite "$S" --n-trials 5 --max-steps 300 \
      --checkpoint "$ck" --norm-stats "$FIRST/norm_stats.json" \
      --waypoint-stats "$FIRST/waypoint_stats.json" \
      --device "${DEV}:0" --heads-device "${DEV}:0" \
      --workers 5 --stagger 10 --worker-timeout 3600 > "$lg" 2>&1
    say "  $tag/$S -> $(grep -m1 '"mean_success"' "$lg" || echo none)"
  done
done

# --- 7. paper table ---
say "=== summary -> results/V8_TABLE.md ==="
mkdir -p results
python scripts/v8_table.py > results/V8_TABLE.md 2>/dev/null && cat results/V8_TABLE.md
say "=== DONE. table: results/V8_TABLE.md | logs: logs/box_v8 ==="
