#!/usr/bin/env bash
# Fine-tune rec_fix with enforced center + depth (IBVS-shaped) losses.
# Starts from the wrist policy that got closest to completion, keeps the
# rec_fix recovery/variance/action-token recipe, adds the lateral+descend
# prior that IBVS showed was missing from MSE-BC.
set -euo pipefail
trap '' TERM HUP
cd /root/MicroVLA
git fetch origin && git reset --hard origin/main

D=data/libero_object_grid
CK_IN=checkpoints/full_stageB_rec_fix.pt
TAG=rec_ctr
LOG=logs/${TAG}_train.log
mkdir -p logs eval_results/${TAG}

echo "==== ${TAG} $(date -u) ====" | tee -a "$LOG"
[ -f "$CK_IN" ] || { echo "missing $CK_IN"; exit 1; }
[ -f "$D/norm_stats.json" ] || { echo "missing $D"; exit 1; }

python -u train/train_batched.py \
  --data-dir "$D" --v8 --tqsa --seed 0 --batch-size 8 \
  --device cuda --lr 3e-4 --reserve-vram-gb 0 --max-vram-gb 0 \
  --load-stage-a "$CK_IN" --resume-stage-b --stage-a-epochs 0 \
  --stage-b-epochs 24 --stage-b-patience 6 --stage-b-select bc \
  --stage-b-min-epochs 10 \
  --dream-frac 0.0 --planner-input-dropout 0.0 --phase-dropout 0.0 \
  --waypoint-weight 0.0 --actuation-weight 0.0 --smooth-weight 0.0 \
  --wm-aux-weight 0.0 \
  --grip-weight 2.0 --row0-weight 2.0 \
  --recovery-noise 0.01 --variance-weight 0.1 --action-token-sampling 0.5 \
  --centering-weight 1.0 --centering-gain 0.5 --centering-sign 1,-1 \
  --centering-uv 0.5,0.55 --centering-window 2 --centering-conf-floor 0.1 \
  --depth-weight 1.0 --depth-descend -0.3 --depth-tol 0.2 \
  --tag "$TAG" 2>&1 | tee -a "$LOG" | tail -60

CK=checkpoints/full_stageB_${TAG}.pt
[ -f "$CK" ] || { echo "NO CKPT"; tail -30 "$LOG"; exit 1; }

echo "==== closed-loop + IBVS $(date -u) ====" | tee -a "$LOG"
python -m eval.libero_eval \
  --suite libero_object --n-trials 3 --max-steps 400 --task-ids 0,1,2 \
  --checkpoint "$CK" --norm-stats "$D/norm_stats.json" \
  --perception-period 2 --det-conf 0.02 --render-size 256 --no-brake \
  --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.3 \
  --ibvs-conf-floor 0.1 \
  --device cuda:0 --heads-device cuda:0 --workers 1 \
  --out-dir "eval_results/${TAG}" 2>&1 | tee -a "$LOG" | grep -aE 'mean_success|src_detect|grip_close|eef_obj|tasks_completed|====' || true

echo "==== record salad+soup $(date -u) ====" | tee -a "$LOG"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/root/LIBERO
python -m eval.record_mp4 \
  --suite libero_object --n-videos 2 --task-ids 2,0 --max-steps 500 \
  --res 256 --fps 17 \
  --checkpoint "$CK" --norm-stats "$D/norm_stats.json" \
  --camera robot0_eye_in_hand_image \
  --perception-period 2 --det-conf 0.02 --no-brake \
  --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.3 \
  --device cuda:0 --heads-device cpu \
  --out-dir "eval_results/${TAG}/videos" 2>&1 | tee -a "$LOG" | tail -20

echo "REC_CTR_DONE $(date -u)" | tee -a "$LOG"
