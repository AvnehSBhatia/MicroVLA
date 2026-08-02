#!/usr/bin/env bash
# ============================================================================
# unaided_push.sh — get to mean_success > 0 with NO assist flags, and keep
# collecting data / training until we do (or rounds exhaust).
#
# Diagnosis this attacks:
#   bc3 reaches the object (~5–8 cm) but never closes (grip_close_rate=0).
#   Old dagger used bc2 (stalled far) → mostly open-jaw labels → poisoned grasp.
#   Fix: DAgger with the CURRENT approach student so close labels land on the
#   states the policy actually visits; retrain; unaided eval; repeat.
#
# Usage on the pod:
#   cd /root/MicroVLA && nohup bash scripts/unaided_push.sh > logs/unaided_push.log 2>&1 &
#   tail -f logs/unaided_push.log
#   touch STOP   # clean exit between phases
#
# Outputs:
#   checkpoints/full_stageB_teacher_bc{5,6,7}.pt
#   eval_results/unaided_v{5,6,7}/
#   eval_results/prox_bc3/          (assisted diagnostic — not scored as unaided)
#   watch_videos/unaided_best/
#   /tmp/UNAIDED_PUSH_READY + SCORECARD.md when done
# ============================================================================
cd /root/MicroVLA || exit 1
export MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="${PYTHONPATH:-}:/root/LIBERO"
trap '' TERM HUP

mkdir -p logs eval_results watch_videos data
LOG="logs/unaided_push.log"
# If launched via nohup redirect, don't double-redirect; still tee progress.
exec >>"$LOG" 2>&1

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
stop_requested() { [ -f STOP ] && { say "STOP present — exiting cleanly"; return 0; }; return 1; }

mean_success() {
  # $1 = eval out-dir → prints mean_success or empty
  python3 - <<PY
import json, sys
from pathlib import Path
d = Path("$1")
js = sorted(d.glob("*results.json"))
if not js:
    print("")
    sys.exit(0)
print(json.load(open(js[-1])).get("mean_success", ""))
PY
}

UNAIDED_FLAGS=(
  --suite libero_object --task-ids 0 --max-steps 600
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12
  --device cuda:0 --heads-device cpu --workers 1
)

TEACHER_FLAGS=(
  --checkpoint checkpoints/full_stageB_rec_fix.pt
  --norm-stats data/libero_object_grid/norm_stats.json
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12
  --ibvs-phase --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.4
  --ibvs-descend-hyst 0.50 --ibvs-target-uv 0.5,0.60
  --ibvs-grasp-offset 0.08,-0.05 --ibvs-close-z 0.045 --ibvs-press 0.2
  --ibvs-retry-rise 8 --ibvs-gate-z 0.10 --ibvs-approach-z 0.12
  --ibvs-place-at=-0.010,0.255 --ibvs-drop-z 0.18
  --device cuda:0 --heads-device cpu
)

student_flags() {
  # $1 = checkpoint path
  echo "--checkpoint $1 --norm-stats data/teacher_grid2/norm_stats.json --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 --device cuda:0 --heads-device cpu"
}

run_unaided_eval() {
  # $1=ckpt $2=out_dir $3=n_trials
  local ckpt="$1" out="$2" n="${3:-10}"
  mkdir -p "$out"
  if ls "$out"/*results.json >/dev/null 2>&1; then
    say "skip eval $out (results exist): $(mean_success "$out")"
    return 0
  fi
  say "UNAIDED eval $ckpt → $out (n=$n)"
  python -m eval.libero_eval "${UNAIDED_FLAGS[@]}" \
    --n-trials "$n" --checkpoint "$ckpt" \
    --norm-stats data/teacher_grid2/norm_stats.json \
    --out-dir "$out" || say "WARN: eval failed for $out"
}

film_ckpt() {
  # $1=ckpt $2=out_dir
  local ckpt="$1" out="$2"
  mkdir -p "$out"
  say "film $ckpt → $out"
  python -m eval.record_mp4 \
    --suite libero_object --task-ids 0,1,2 --n-videos 3 --max-steps 600 \
    --checkpoint "$ckpt" --norm-stats data/teacher_grid2/norm_stats.json \
    --camera robot0_eye_in_hand_image --res 256 --perception-period 2 \
    --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
    --device cuda:0 --heads-device cpu --out-dir "$out" \
    || say "WARN: film failed"
}

train_round() {
  # $1=tag $2=init_ckpt $3+= --data-dir args...
  local tag="$1" init="$2"; shift 2
  local out="checkpoints/full_stageB_${tag}.pt"
  if [ -f "$out" ]; then
    say "skip train $tag (ckpt exists)"
    return 0
  fi
  say "TRAIN $tag from $init  data=$*"
  python -u train/train_batched.py \
    "$@" --v8 --tqsa --seed 0 \
    --batch-size 8 --device cuda --lr 1e-4 --reserve-vram-gb 0 --max-vram-gb 0 \
    --no-cache-spatial \
    --load-stage-a "$init" --resume-stage-b \
    --stage-a-epochs 0 --stage-b-epochs 20 --stage-b-patience 6 --stage-b-select bc \
    --stage-b-min-epochs 6 \
    --dream-frac 0.0 --planner-input-dropout 0.0 --phase-dropout 0.0 \
    --waypoint-weight 0.0 --actuation-weight 0.0 --smooth-weight 0.05 --wm-aux-weight 0.0 \
    --grip-weight 3.0 --row0-weight 2.0 --pre-grasp-weight 2.0 \
    --recovery-noise 0.02 --variance-weight 0.3 \
    --magnitude-weight 0.8 --gain-magnitude-weight 0.3 \
    --centering-weight 0.5 --centering-uv 0.5,0.60 --centering-sign 1,-1 \
    --depth-weight 0.5 --depth-descend -0.3 \
    --action-token-sampling 0.5 \
    --tag "$tag" || { say "FATAL: train $tag failed"; return 1; }
}

dagger_round() {
  # $1=name $2=student_ckpt $3=n_success $4=init_offset $5=beta
  local name="$1" student="$2" n="$3" off="$4" beta="$5"
  local raw="data/dagger_raw_${name}" grid="data/dagger_grid_${name}"
  if [ -d "$grid" ] && [ "$(ls "$grid"/*.npz 2>/dev/null | wc -l)" -ge 10 ]; then
    say "skip dagger $name (grid has $(ls "$grid"/*.npz | wc -l) eps)"
    return 0
  fi
  stop_requested && return 1
  say "DAGGER record $name student=$student n=$n beta=$beta offset=$off"
  rm -rf "$raw"
  python -m preprocess.teacher_rollouts record \
    --suite libero_object --task-id 0 --n-success "$n" --max-attempts $((n + 20)) \
    --init-offset "$off" --raw-dir "$raw" --max-steps 400 \
    --dagger-beta "$beta" \
    --dagger-student-flags "$(student_flags "$student")" \
    -- \
    "${TEACHER_FLAGS[@]}" || { say "WARN: dagger record $name failed"; return 1; }

  say "DAGGER convert $name"
  rm -rf "$grid"
  python -m preprocess.teacher_rollouts convert \
    --raw-dir "$raw" --out "$grid" --spatial-grid 4 \
    --camera eye_in_hand_rgb --device cuda:0 --det-conf 0.02 \
    --role-disjoint-iou 0.1 --purge-raw || { say "WARN: convert $name failed"; return 1; }

  python3 - <<PY
import numpy as np
from pathlib import Path
n=n_close=0
for p in Path("$grid").glob("*.npz"):
    g=np.load(p)["pwm_targets"][:,0,-1]; n+=1
    if (g>0).any(): n_close+=1
print(f"[dagger $name] eps={n} with_close={n_close}")
PY
}

# -----------------------------------------------------------------------------
say "=== UNAIDED PUSH START pid=$$ pgid=$(ps -o pgid= -p $$ | tr -d ' ') ==="
say "touch STOP to halt between phases; kill -9 -\$pgid to force"

# 0) Kill GPU contention (exact PIDs / name prefixes — never pkill -f SSH)
say "phase0: free GPU"
for pat in 'eval.libero_eval' 'train.train_batched' 'teacher_rollouts' 'eval.record_mp4' \
           'round5_bc3_dagger' 'film_bc4' 'unaided_v2r' 'round4_aggregate'; do
  for pid in $(pgrep -f "$pat" 2>/dev/null || true); do
    # Don't kill ourselves
    [ "$pid" = "$$" ] && continue
    # Don't kill the parent unaided_push if re-entered
    case "$(ps -o args= -p "$pid" 2>/dev/null || true)" in
      *unaided_push.sh*) continue ;;
    esac
    say "  kill $pid ($(ps -o args= -p "$pid" 2>/dev/null | head -c 120))"
    kill -9 "$pid" 2>/dev/null || true
  done
done
sleep 3
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null || true
df -h / | tail -1

# Preflight
for f in checkpoints/full_stageB_teacher_bc3.pt checkpoints/full_stageB_rec_fix.pt \
         data/teacher_grid2/norm_stats.json; do
  [ -f "$f" ] || { say "FATAL: missing $f"; exit 1; }
done
N_TEACHER=$(ls data/teacher_grid2/*.npz 2>/dev/null | wc -l)
say "teacher_grid2 episodes: $N_TEACHER"

# 1) Proximity smoke (ASSISTED — proves approach+close can pick; not unaided)
if stop_requested; then exit 0; fi
PROX=eval_results/prox_bc3
if ! ls "$PROX"/*results.json >/dev/null 2>&1; then
  rm -f "$PROX"/*telemetry* 2>/dev/null || true
  say "phase1: proximity smoke on bc3 (ASSISTED diagnostic)"
  python -m eval.libero_eval "${UNAIDED_FLAGS[@]}" --n-trials 5 \
    --checkpoint checkpoints/full_stageB_teacher_bc3.pt \
    --norm-stats data/teacher_grid2/norm_stats.json \
    --grip-close-dist 0.07 --grip-close-lift 0.2 \
    --out-dir "$PROX" || say "WARN: prox smoke failed"
fi
say "prox_bc3 mean_success=$(mean_success "$PROX")  (assisted — not unaided)"

# -----------------------------------------------------------------------------
# Rounds: dagger(student) → train → unaided eval. Escalate student each round.
# -----------------------------------------------------------------------------
BEST_CKPT="checkpoints/full_stageB_teacher_bc3.pt"
BEST_SUCC="0"
declare -a ROUND_TAGS=(teacher_bc5 teacher_bc6 teacher_bc7)
declare -a ROUND_STUDENTS=(
  checkpoints/full_stageB_teacher_bc3.pt
  checkpoints/full_stageB_teacher_bc5.pt
  checkpoints/full_stageB_teacher_bc6.pt
)
declare -a ROUND_OFFSETS=(200 280 360)
declare -a ROUND_BETAS=(0.5 0.5 0.4)
declare -a ROUND_N=(50 40 40)

for i in 0 1 2; do
  stop_requested && break
  tag="${ROUND_TAGS[$i]}"
  student="${ROUND_STUDENTS[$i]}"
  name="r$((i+5))"
  # Fall back student if previous round didn't produce a ckpt
  [ -f "$student" ] || student="$BEST_CKPT"
  say "======== ROUND $((i+5)) tag=$tag student=$student ========"

  dagger_round "$name" "$student" "${ROUND_N[$i]}" "${ROUND_OFFSETS[$i]}" "${ROUND_BETAS[$i]}" \
    || say "WARN: dagger $name incomplete — training on whatever we have"

  grid="data/dagger_grid_${name}"
  DATA_ARGS=(--data-dir data/teacher_grid2)
  if [ -d "$grid" ] && [ "$(ls "$grid"/*.npz 2>/dev/null | wc -l)" -ge 5 ]; then
    DATA_ARGS+=(--data-dir "$grid")
  else
    say "WARN: $grid thin/missing — teacher-only train"
  fi
  # Keep prior dagger that has closes (soup grid) as light mix if present
  if [ -d data/teacher_dagger_soup_grid ] && [ "$i" -eq 0 ]; then
    # Only episodes that close — soft: include whole dir; grip weight handles it
    :
  fi

  train_round "$tag" "$student" "${DATA_ARGS[@]}" || continue
  ckpt="checkpoints/full_stageB_${tag}.pt"
  [ -f "$ckpt" ] || continue

  out="eval_results/unaided_v$((i+5))"
  run_unaided_eval "$ckpt" "$out" 10
  succ="$(mean_success "$out")"
  say "ROUND $((i+5)) unaided mean_success=${succ:-?}"

  # Track best by success, then by existence
  python3 - <<PY
succ = "${succ}" or "0"
best = "${BEST_SUCC}" or "0"
try:
    s, b = float(succ), float(best)
except Exception:
    s, b = 0.0, 0.0
open("/tmp/_succ_cmp","w").write("1" if s > b else "0")
print("succ", succ, "best", best)
PY
  if [ "$(cat /tmp/_succ_cmp 2>/dev/null)" = "1" ]; then
    BEST_CKPT="$ckpt"
    BEST_SUCC="${succ:-0}"
    say "NEW BEST $BEST_CKPT success=$BEST_SUCC"
  fi

  # Early exit on any unaided success — then expand evidence
  if python3 -c "import sys; sys.exit(0 if float('${succ:-0}' or 0)>0 else 1)"; then
    say "SUCCESS — unaided mean_success=$succ on $ckpt"
    run_unaided_eval "$ckpt" "eval_results/unaided_${tag}_n20" 20
    film_ckpt "$ckpt" "watch_videos/unaided_best"
    # Optional GRAM fine-tune on the winning student (extra data / diversity)
    if stop_requested; then break; fi
    if [ ! -f checkpoints/full_stageB_${tag}_gram.pt ]; then
      say "GRAM fine-tune on winning $tag"
      python -u train/train_batched.py \
        --data-dir data/teacher_grid2 --data-dir "$grid" --v8 --tqsa --seed 1 \
        --batch-size 8 --device cuda --lr 5e-5 --reserve-vram-gb 0 --max-vram-gb 0 \
        --no-cache-spatial --gram-hrm --gram-planner --gram-n-samples 4 \
        --load-stage-a "$ckpt" --resume-stage-b \
        --stage-a-epochs 0 --stage-b-epochs 12 --stage-b-patience 4 --stage-b-select bc \
        --stage-b-min-epochs 4 \
        --dream-frac 0.0 --grip-weight 3.0 --row0-weight 2.0 --pre-grasp-weight 2.0 \
        --magnitude-weight 0.8 --gain-magnitude-weight 0.3 \
        --centering-weight 0.5 --depth-weight 0.5 \
        --action-token-sampling 0.5 \
        --tag "${tag}_gram" || say "WARN: gram fine-tune failed"
      if [ -f "checkpoints/full_stageB_${tag}_gram.pt" ]; then
        run_unaided_eval "checkpoints/full_stageB_${tag}_gram.pt" \
          "eval_results/unaided_${tag}_gram" 10
      fi
    fi
    break
  fi

  # Free space between rounds
  rm -rf "data/dagger_raw_${name}" 2>/dev/null || true
  df -h / | tail -1
done

# If still zero: one more hail-mary — teacher-only from bc3 with very high grip + GRAM
if ! python3 -c "import sys; sys.exit(0 if float('${BEST_SUCC:-0}' or 0)>0 else 1)"; then
  stop_requested || {
    say "hail-mary: teacher-only + GRAM from bc3"
    train_round teacher_bc_gram checkpoints/full_stageB_teacher_bc3.pt \
      --data-dir data/teacher_grid2 --gram-hrm --gram-planner --gram-n-samples 4 || true
    if [ -f checkpoints/full_stageB_teacher_bc_gram.pt ]; then
      run_unaided_eval checkpoints/full_stageB_teacher_bc_gram.pt eval_results/unaided_gram 10
      s="$(mean_success eval_results/unaided_gram)"
      say "gram unaided mean_success=$s"
      if python3 -c "import sys; sys.exit(0 if float('${s:-0}' or 0)>0 else 1)"; then
        BEST_CKPT=checkpoints/full_stageB_teacher_bc_gram.pt
        BEST_SUCC="$s"
        film_ckpt "$BEST_CKPT" watch_videos/unaided_best
      fi
    fi
  }
fi

# Always film the best we have
if [ ! -d watch_videos/unaided_best ] || [ -z "$(ls watch_videos/unaided_best/*.mp4 2>/dev/null)" ]; then
  film_ckpt "$BEST_CKPT" watch_videos/unaided_best
fi

# Scorecard
python3 - <<'PY'
import json
from pathlib import Path
lines = ["# Unaided push scorecard", "", f"generated on pod", ""]
best = ("", -1.0)
for d in sorted(Path("eval_results").glob("unaided*")):
    js = list(d.glob("*results.json"))
    if not js:
        lines.append(f"- `{d.name}`: (no results yet)")
        continue
    r = json.load(open(js[-1]))
    s = float(r.get("mean_success") or 0)
    inter = r.get("intermediates") or {}
    lines.append(f"- `{d.name}`: mean_success={s:.3f}  intermediates={inter}")
    if s > best[1]:
        best = (d.name, s)
lines += ["", f"**best**: `{best[0]}` = {best[1]:.3f}", ""]
# prox assisted
for d in Path("eval_results").glob("prox*"):
    js = list(d.glob("*results.json"))
    if js:
        r = json.load(open(js[-1]))
        lines.append(f"- ASSISTED `{d.name}`: mean_success={r.get('mean_success')} (not unaided)")
Path("SCORECARD_UNAIDED.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

# Pack videos for pull
if ls watch_videos/unaided_best/*.mp4 >/dev/null 2>&1; then
  tar -czf /tmp/unaided_best_vids.tgz -C watch_videos unaided_best SCORECARD_UNAIDED.md 2>/dev/null \
    || tar -czf /tmp/unaided_best_vids.tgz -C watch_videos unaided_best
  cp -f SCORECARD_UNAIDED.md /tmp/SCORECARD_UNAIDED.md 2>/dev/null || true
fi
touch /tmp/UNAIDED_PUSH_READY
say "=== UNAIDED PUSH DONE best=$BEST_CKPT success=$BEST_SUCC ==="
say "READY: /tmp/UNAIDED_PUSH_READY  scorecard: SCORECARD_UNAIDED.md  vids: /tmp/unaided_best_vids.tgz"
