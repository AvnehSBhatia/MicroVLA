#!/usr/bin/env bash
# unaided_push2 — recover approach from bc3, keep r5 dagger close labels.
#
# Round-5 postmortem:
#   dagger_grid_r5: 47/50 eps HAVE close labels (good — student was bc3 near object)
#   teacher_bc5 train mixed 1:1 teacher:dagger → approach REGRESSED 6cm→22cm,
#   grip came back only ~6%. Prox assist never scored (pick≠place; gate also
#   used telemetry that ignores the override).
#
# Fix: retrain from bc3 with TEACHER-HEAVY mix (2× teacher_grid2 + dagger_r5),
# lower LR, high pre-grasp + grip, keep magnitude so approach doesn't shrink.
set -u
cd /root/MicroVLA
export MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 PYOPENGL_PLATFORM=osmesa
trap '' TERM HUP
mkdir -p logs eval_results watch_videos
exec >>logs/unaided_push2.log 2>&1

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
mean_success() {
  python3 - <<PY
import json
from pathlib import Path
js=sorted(Path("$1").glob("*results.json"))
print(json.load(open(js[-1])).get("mean_success","") if js else "")
PY
}

say "=== UNAIDED PUSH2 START ==="

# Wait for any leftover eval (ceiling) to free the GPU, up to ~40 min
for i in $(seq 1 80); do
  if pgrep -f 'eval.libero_eval' >/dev/null 2>&1; then
    say "waiting for GPU eval... ($i)"
    sleep 30
  else
    break
  fi
done
# hard-clear stragglers
for pid in $(pgrep -f 'eval.libero_eval|train.train_batched' 2>/dev/null || true); do
  say "kill leftover $pid"; kill -9 "$pid" 2>/dev/null || true
done
sleep 2

if ls eval_results/ceiling_ibvs3/*results.json >/dev/null 2>&1; then
  say "ceiling_ibvs3 mean_success=$(mean_success eval_results/ceiling_ibvs3)"
fi

[ -d data/dagger_grid_r5 ] || { say "FATAL: need data/dagger_grid_r5"; exit 1; }
[ -f checkpoints/full_stageB_teacher_bc3.pt ] || { say "FATAL: need bc3"; exit 1; }

# --- train bc5b: teacher-heavy, from bc3 ---
TAG=teacher_bc5b
CKPT=checkpoints/full_stageB_${TAG}.pt
if [ ! -f "$CKPT" ]; then
  say "TRAIN $TAG (2× teacher_grid2 + dagger_r5) from bc3"
  python -u train/train_batched.py \
    --data-dir data/teacher_grid2 --data-dir data/teacher_grid2 \
    --data-dir data/dagger_grid_r5 \
    --v8 --tqsa --seed 0 \
    --batch-size 8 --device cuda --lr 5e-5 --reserve-vram-gb 0 --max-vram-gb 0 \
    --no-cache-spatial \
    --load-stage-a checkpoints/full_stageB_teacher_bc3.pt --resume-stage-b \
    --stage-a-epochs 0 --stage-b-epochs 16 --stage-b-patience 5 --stage-b-select bc \
    --stage-b-min-epochs 6 \
    --dream-frac 0.0 --planner-input-dropout 0.0 --phase-dropout 0.0 \
    --waypoint-weight 0.0 --actuation-weight 0.0 --smooth-weight 0.05 --wm-aux-weight 0.0 \
    --grip-weight 4.0 --row0-weight 2.0 --pre-grasp-weight 3.0 \
    --recovery-noise 0.01 --variance-weight 0.2 \
    --magnitude-weight 1.0 --gain-magnitude-weight 0.4 \
    --centering-weight 0.8 --centering-uv 0.5,0.60 --centering-sign 1,-1 \
    --depth-weight 0.8 --depth-descend -0.3 \
    --action-token-sampling 0.5 \
    --tag "$TAG" || { say "FATAL train $TAG"; exit 1; }
else
  say "skip train $TAG (exists)"
fi

OUT=eval_results/unaided_v5b
say "UNAIDED eval $CKPT → $OUT"
python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 10 --max-steps 600 \
  --checkpoint "$CKPT" --norm-stats data/teacher_grid2/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir "$OUT" || say "WARN eval failed"
SUCC=$(mean_success "$OUT")
say "unaided_v5b mean_success=$SUCC"

# Analyze approach/grip
python3 - <<'PY'
import json, numpy as np
from pathlib import Path
tels=list(Path('eval_results/unaided_v5b').glob('*telemetry.jsonl'))
if not tels: raise SystemExit
rows=[json.loads(l) for l in open(tels[0])]
by={}
for r in rows: by.setdefault(r['trial'], []).append(r)
emins=[]; grips=[]
for t,rs in by.items():
  d=np.array([r['eef_obj_dist'] for r in rs], float)
  a=np.array([r['action'] for r in rs], float)
  emins.append(d.min()); grips.append(float((a[:,-1]>0).mean()))
print(f"eef_min mean={np.mean(emins):.3f} median={np.median(emins):.3f} grip_rate mean={np.mean(grips):.3f}")
PY

# If still 0: train grip-heavy bc5c from bc3 on dagger_r5 ONLY (approach from init, grip from labels)
if ! python3 -c "import sys; sys.exit(0 if float('${SUCC:-0}' or 0)>0 else 1)"; then
  say "v5b still 0 — train teacher_bc5c dagger-only grip focus from bc3"
  TAG=teacher_bc5c
  CKPT=checkpoints/full_stageB_${TAG}.pt
  if [ ! -f "$CKPT" ]; then
    python -u train/train_batched.py \
      --data-dir data/dagger_grid_r5 \
      --v8 --tqsa --seed 1 \
      --batch-size 8 --device cuda --lr 3e-5 --reserve-vram-gb 0 --max-vram-gb 0 \
      --no-cache-spatial \
      --load-stage-a checkpoints/full_stageB_teacher_bc3.pt --resume-stage-b \
      --stage-a-epochs 0 --stage-b-epochs 12 --stage-b-patience 4 --stage-b-select bc \
      --stage-b-min-epochs 4 \
      --dream-frac 0.0 \
      --grip-weight 6.0 --row0-weight 1.5 --pre-grasp-weight 1.0 \
      --magnitude-weight 0.3 --gain-magnitude-weight 0.1 \
      --smooth-weight 0.05 --recovery-noise 0.02 \
      --centering-weight 1.0 --depth-weight 1.0 \
      --action-token-sampling 0.5 \
      --tag "$TAG" || say "WARN train 5c failed"
  fi
  if [ -f "$CKPT" ]; then
    OUT=eval_results/unaided_v5c
    python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 10 --max-steps 600 \
      --checkpoint "$CKPT" --norm-stats data/teacher_grid2/norm_stats.json \
      --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
      --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
      --device cuda:0 --heads-device cpu --workers 1 \
      --out-dir "$OUT" || true
    SUCC=$(mean_success "$OUT")
    say "unaided_v5c mean_success=$SUCC"
  fi
fi

# If success: film + expand
BEST="$CKPT"
BEST_S="${SUCC:-0}"
for d in unaided_v5b unaided_v5c; do
  s=$(mean_success "eval_results/$d")
  python3 -c "import sys; sys.exit(0 if float('${s:-0}' or 0)>float('${BEST_S:-0}' or 0) else 1)" && {
    BEST_S="$s"
    case "$d" in
      unaided_v5b) BEST=checkpoints/full_stageB_teacher_bc5b.pt ;;
      unaided_v5c) BEST=checkpoints/full_stageB_teacher_bc5c.pt ;;
    esac
  }
done
say "BEST=$BEST success=$BEST_S"

if python3 -c "import sys; sys.exit(0 if float('${BEST_S:-0}' or 0)>0 else 1)"; then
  say "SUCCESS — film + n=20"
  python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 20 --max-steps 600 \
    --checkpoint "$BEST" --norm-stats data/teacher_grid2/norm_stats.json \
    --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
    --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
    --device cuda:0 --heads-device cpu --workers 1 \
    --out-dir eval_results/unaided_best_n20 || true
  mkdir -p watch_videos/unaided_best
  python -m eval.record_mp4 --suite libero_object --task-ids 0,1,2 --n-videos 3 --max-steps 600 \
    --checkpoint "$BEST" --norm-stats data/teacher_grid2/norm_stats.json \
    --camera robot0_eye_in_hand_image --res 256 --perception-period 2 \
    --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
    --device cuda:0 --heads-device cpu --out-dir watch_videos/unaided_best || true
fi

# Assisted ceiling with soup_v1 constants (NOT cream 0.08,-0.05 — that miss
# was ceiling_ibvs3's 0/3 at eef_min~13cm). soup_v1 hit 0.750 with these.
say "ceiling_soup_v1: rec_fix + soup_v1 IBVS constants"
python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 4 --max-steps 600 \
  --checkpoint checkpoints/full_stageB_rec_fix.pt \
  --norm-stats data/libero_object_grid/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --ibvs-phase --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.4 \
  --ibvs-descend-hyst 0.50 --ibvs-target-uv 0.5,0.60 \
  --ibvs-grasp-offset 0.09,-0.186 --ibvs-close-z 0.045 --ibvs-press 0.2 \
  --ibvs-retry-rise 8 --ibvs-gate-z 0.10 --ibvs-approach-z 0.12 \
  --ibvs-place-at=-0.006,0.260 --ibvs-drop-z 0.25 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/ceiling_soup_v1 || true
say "ceiling_soup_v1 mean_success=$(mean_success eval_results/ceiling_soup_v1)"

say "hybrid: bc3 + soup_v1 IBVS"
python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 4 --max-steps 600 \
  --checkpoint checkpoints/full_stageB_teacher_bc3.pt \
  --norm-stats data/teacher_grid2/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --ibvs-phase --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.4 \
  --ibvs-descend-hyst 0.50 --ibvs-target-uv 0.5,0.60 \
  --ibvs-grasp-offset 0.09,-0.186 --ibvs-close-z 0.045 --ibvs-press 0.2 \
  --ibvs-retry-rise 8 --ibvs-gate-z 0.10 --ibvs-approach-z 0.12 \
  --ibvs-place-at=-0.006,0.260 --ibvs-drop-z 0.25 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/hybrid_bc3_ibvs || true
say "hybrid_bc3_ibvs mean_success=$(mean_success eval_results/hybrid_bc3_ibvs)"

python3 - <<'PY'
import json
from pathlib import Path
lines=["# Unaided push2 scorecard", ""]
for d in sorted(Path("eval_results").glob("*")):
  if not any(x in d.name for x in ("unaided","prox","ceiling","hybrid")): continue
  js=list(d.glob("*results.json"))
  if not js: continue
  r=json.load(open(js[-1]))
  lines.append(f"- `{d.name}`: mean_success={r.get('mean_success')} intermediates={r.get('intermediates')}")
Path("SCORECARD_UNAIDED.md").write_text("\n".join(lines)+"\n")
print("\n".join(lines))
PY

if ls watch_videos/unaided_best/*.mp4 >/dev/null 2>&1; then
  tar -czf /tmp/unaided_best_vids.tgz -C watch_videos unaided_best
fi
cp -f SCORECARD_UNAIDED.md /tmp/SCORECARD_UNAIDED.md
touch /tmp/UNAIDED_PUSH_READY
say "=== PUSH2 DONE BEST=$BEST success=$BEST_S ==="
