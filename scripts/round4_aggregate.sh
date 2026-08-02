#!/bin/bash
set -u
cd /root/MicroVLA
export MUJOCO_GL=osmesa PYTHONUNBUFFERED=1
trap "" TERM HUP

echo "[r4] train teacher_bc4 (AGGREGATE 100 teacher + 40 dagger) $(date -u)"
python -u train/train_batched.py \
  --data-dir data/teacher_grid2 --data-dir data/teacher_dagger_soup_grid --v8 --tqsa --seed 0 \
  --batch-size 8 --device cuda --lr 2e-4 --reserve-vram-gb 0 --max-vram-gb 0 \
  --no-cache-spatial \
  --load-stage-a checkpoints/full_stageB_rec_fix.pt --resume-stage-b \
  --stage-a-epochs 0 --stage-b-epochs 24 --stage-b-patience 8 --stage-b-select bc \
  --stage-b-min-epochs 8 \
  --dream-frac 0.0 --planner-input-dropout 0.0 --phase-dropout 0.0 \
  --waypoint-weight 0.0 --actuation-weight 0.0 --smooth-weight 0.05 --wm-aux-weight 0.0 \
  --grip-weight 2.0 --row0-weight 2.0 \
  --recovery-noise 0.01 --variance-weight 0.3 \
  --magnitude-weight 0.8 --gain-magnitude-weight 0.3 \
  --action-token-sampling 0.5 \
  --tag teacher_bc4 > logs/teacher_bc4_train.log 2>&1 || exit 1

echo "[r4] unaided_v4 eval $(date -u)"
python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 10 --max-steps 600 \
  --checkpoint checkpoints/full_stageB_teacher_bc4.pt \
  --norm-stats data/teacher_grid2/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/unaided_v4 > logs/unaided_v4.log 2>&1

echo "[r4] formal v2r + diag evals $(date -u)"
bash scripts/evals_v2r_diag.sh
echo "[r4] DONE $(date -u)"
