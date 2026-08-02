#!/usr/bin/env bash
set -euo pipefail
cd /root/MicroVLA
export PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa PYTHONPATH=/root/LIBERO PYTHONUNBUFFERED=1
trap '' TERM HUP

STUDENT_FLAGS='--checkpoint checkpoints/full_stageB_teacher_bc2.pt --norm-stats data/teacher_grid2/norm_stats.json --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 --device cuda:0 --heads-device cpu --workers 1'

echo "[chain] DAgger record start $(date)"
python -m preprocess.teacher_rollouts record \
  --suite libero_object --task-id 0 --n-success 40 --max-attempts 80 \
  --init-offset 50 --raw-dir data/teacher_dagger_soup --max-steps 600 \
  --dagger-beta 0.3 \
  --dagger-student-flags "$STUDENT_FLAGS" \
  -- \
  --checkpoint checkpoints/full_stageB_rec_fix.pt \
  --norm-stats data/libero_object_grid/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --ibvs-phase --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.4 \
  --ibvs-descend-hyst 0.50 --ibvs-press 0.2 --ibvs-retry-rise 8 \
  --ibvs-target-uv 0.5,0.60 --ibvs-grasp-offset 0.09,-0.186 \
  --ibvs-close-z 0.045 --ibvs-gate-z 0.10 --ibvs-approach-z 0.12 \
  --ibvs-place-at=-0.006,0.260 --ibvs-drop-z 0.25 \
  --device cuda:0 --heads-device cpu --workers 1 --max-steps 600

echo "[chain] convert $(date)"
python -m preprocess.teacher_rollouts convert \
  --raw-dir data/teacher_dagger_soup --out data/teacher_dagger_soup_grid \
  --spatial-grid 4 --camera eye_in_hand_rgb --device cuda:0 \
  --det-conf 0.02 --role-disjoint-iou 0.1 --purge-raw

echo "[chain] train teacher_bc3 $(date)"
python -u train/train_batched.py \
  --data-dir data/teacher_dagger_soup_grid --v8 --tqsa --seed 0 --batch-size 8 \
  --device cuda --lr 2e-4 --reserve-vram-gb 0 --max-vram-gb 0 --no-cache-spatial \
  --load-stage-a checkpoints/full_stageB_rec_fix.pt --resume-stage-b \
  --stage-a-epochs 0 --stage-b-epochs 24 --stage-b-patience 8 --stage-b-select bc \
  --stage-b-min-epochs 8 \
  --dream-frac 0.0 --planner-input-dropout 0.0 --phase-dropout 0.0 \
  --waypoint-weight 0.0 --actuation-weight 0.0 --smooth-weight 0.05 --wm-aux-weight 0.0 \
  --grip-weight 2.0 --row0-weight 2.0 \
  --recovery-noise 0.01 --variance-weight 0.3 \
  --magnitude-weight 0.8 --gain-magnitude-weight 0.3 \
  --action-token-sampling 0.5 \
  --tag teacher_bc3

echo "[chain] unaided_v3 eval $(date)"
python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 10 --max-steps 600 \
  --checkpoint checkpoints/full_stageB_teacher_bc3.pt \
  --norm-stats data/teacher_dagger_soup_grid/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/unaided_v3

echo "[chain] DONE $(date)"
