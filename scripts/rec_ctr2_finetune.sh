#!/usr/bin/env bash
# Tighten the last-cm lateral miss on rec_ctr:
#   - stronger centering residual (gain 1.5, weight 2)
#   - stricter depth gate (tol 0.10) so z only engages when better centred
#   - short fine-tune from rec_ctr, then high-gain IBVS eval + salad/soup vids
set -euo pipefail
trap '' TERM HUP
cd /root/MicroVLA
git fetch origin && git reset --hard origin/main

D=data/libero_object_grid
CK_IN=checkpoints/full_stageB_rec_ctr.pt
[ -f "$CK_IN" ] || CK_IN=checkpoints/full_stageB_rec_fix.pt
TAG=rec_ctr2
LOG=logs/${TAG}_train.log
mkdir -p logs eval_results/${TAG}

echo "==== ${TAG} from $CK_IN $(date -u) ====" | tee "$LOG"

python -u train/train_batched.py \
  --data-dir "$D" --v8 --tqsa --seed 0 --batch-size 8 \
  --device cuda --lr 2e-4 --reserve-vram-gb 0 --max-vram-gb 0 \
  --load-stage-a "$CK_IN" --resume-stage-b --stage-a-epochs 0 \
  --stage-b-epochs 16 --stage-b-patience 5 --stage-b-select bc \
  --stage-b-min-epochs 6 \
  --dream-frac 0.0 --planner-input-dropout 0.0 --phase-dropout 0.0 \
  --waypoint-weight 0.0 --actuation-weight 0.0 --smooth-weight 0.0 \
  --wm-aux-weight 0.0 \
  --grip-weight 3.0 --row0-weight 2.0 \
  --recovery-noise 0.01 --variance-weight 0.1 --action-token-sampling 0.5 \
  --centering-weight 2.0 --centering-gain 1.5 --centering-sign 1,-1 \
  --centering-uv 0.5,0.55 --centering-window 3 --centering-conf-floor 0.05 \
  --depth-weight 1.0 --depth-descend -0.4 --depth-tol 0.10 \
  --tag "$TAG" 2>&1 | tee -a "$LOG"

CK=checkpoints/full_stageB_${TAG}.pt
[ -f "$CK" ] || { echo "NO CKPT"; tail -40 "$LOG"; exit 1; }

echo "==== closed-loop + strong IBVS $(date -u) ====" | tee -a "$LOG"
python -m eval.libero_eval \
  --suite libero_object --n-trials 3 --max-steps 400 --task-ids 0,1,2 \
  --checkpoint "$CK" --norm-stats "$D/norm_stats.json" \
  --perception-period 2 --det-conf 0.02 --render-size 256 --no-brake \
  --ibvs-gain 1.0 --ibvs-sign 1,-1,0 --ibvs-descend -0.4 \
  --ibvs-conf-floor 0.05 \
  --device cuda:0 --heads-device cuda:0 --workers 1 \
  --out-dir "eval_results/${TAG}" 2>&1 | tee -a "$LOG" | grep -aE 'mean_success|src_detect|grip_close|eef_obj|====' || true

echo "==== record salad+soup $(date -u) ====" | tee -a "$LOG"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/root/LIBERO
python -m eval.record_mp4 \
  --suite libero_object --n-videos 2 --task-ids 2,0 --max-steps 500 \
  --res 256 --fps 17 \
  --checkpoint "$CK" --norm-stats "$D/norm_stats.json" \
  --camera robot0_eye_in_hand_image \
  --perception-period 2 --det-conf 0.02 --no-brake \
  --ibvs-gain 1.0 --ibvs-sign 1,-1,0 --ibvs-descend -0.4 \
  --device cuda:0 --heads-device cpu \
  --out-dir "eval_results/${TAG}/videos" 2>&1 | tee -a "$LOG" | tail -25

echo "REC_CTR2_DONE $(date -u)" | tee -a "$LOG"
