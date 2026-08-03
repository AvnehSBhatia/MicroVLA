#!/bin/bash
# UNAIDED_PLAN round 5 — DAgger iteration 2. Student = teacher_bc4 (reaches
# 6-10 cm), teacher = soup PhasedIBVS. beta=0.5 so the teacher completes
# grasps from student-visited near-grasp states — the labels round 4 lacked.
# Waits for the running v2r/diag evals to finish, reruns the OOM-killed
# unaided_v4 for its formal n=10 first, then records/trains/evals bc5.
set -u
cd /root/MicroVLA
export MUJOCO_GL=osmesa PYTHONUNBUFFERED=1
trap "" TERM HUP

while pgrep -f "evals_v2r_diag|libero_eval" > /dev/null 2>&1; do sleep 60; done

echo "[r5] rerun unaided_v4 formal n=10 $(date -u)"
python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 10 --max-steps 600 \
  --checkpoint checkpoints/full_stageB_teacher_bc4.pt \
  --norm-stats data/teacher_grid2/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/unaided_v4r > logs/unaided_v4r.log 2>&1

echo "[r5] DAgger2 record start $(date -u)"
python -m preprocess.teacher_rollouts record \
  --suite libero_object --task-id 0 --n-success 40 --max-attempts 60 \
  --init-offset 200 --raw-dir data/dagger2_raw --max-steps 600 \
  --dagger-beta 0.5 \
  --dagger-student-flags "--checkpoint checkpoints/full_stageB_teacher_bc4.pt --norm-stats data/teacher_grid2/norm_stats.json --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 --device cuda:0 --heads-device cpu --workers 1" \
  -- \
  --checkpoint checkpoints/full_stageB_rec_fix.pt \
  --norm-stats data/libero_object_grid/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --ibvs-phase --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.4 \
  --ibvs-descend-hyst 0.50 --ibvs-press 0.2 --ibvs-retry-rise 8 \
  --ibvs-target-uv 0.5,0.60 --ibvs-grasp-offset 0.08,-0.05 \
  --ibvs-close-z 0.045 --ibvs-gate-z 0.10 --ibvs-approach-z 0.12 \
  --ibvs-place-at=-0.010,0.255 --ibvs-drop-z 0.18 \
  --device cuda:0 --heads-device cpu --workers 1 --max-steps 600 \
  > logs/dagger2_rec.log 2>&1 || exit 1

echo "[r5] convert $(date -u)"
python -m preprocess.teacher_rollouts convert \
  --raw-dir data/dagger2_raw --out data/dagger2_grid --spatial-grid 4 \
  --camera eye_in_hand_rgb --device cuda:0 --det-conf 0.02 \
  --role-disjoint-iou 0.1 --purge-raw > logs/dagger2_convert.log 2>&1 || exit 1

echo "[r5] train teacher_bc5 (100 teacher + 40 dagger1 + 40 dagger2) $(date -u)"
python -u train/train_batched.py \
  --data-dir data/teacher_grid2 data/teacher_dagger_soup_grid data/dagger2_grid \
  --v8 --tqsa --seed 0 --batch-size 8 --device cuda \
  --reserve-vram-gb 0 --max-vram-gb 0 --no-cache-spatial --lr 2e-4 \
  --load-stage-a checkpoints/full_stageB_rec_fix.pt --resume-stage-b \
  --stage-a-epochs 0 --stage-b-epochs 24 --stage-b-patience 8 --stage-b-select bc \
  --stage-b-min-epochs 8 \
  --dream-frac 0.0 --planner-input-dropout 0.0 --phase-dropout 0.0 \
  --waypoint-weight 0.0 --actuation-weight 0.0 --smooth-weight 0.05 --wm-aux-weight 0.0 \
  --grip-weight 3.0 --row0-weight 2.0 \
  --recovery-noise 0.01 --variance-weight 0.3 \
  --magnitude-weight 0.8 --gain-magnitude-weight 0.3 \
  --action-token-sampling 0.5 \
  --tag teacher_bc5 > logs/teacher_bc5_train.log 2>&1 || exit 1

echo "[r5] unaided_v5 eval $(date -u)"
python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 10 --max-steps 600 \
  --checkpoint checkpoints/full_stageB_teacher_bc5.pt \
  --norm-stats data/teacher_grid2/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/unaided_v5 > logs/unaided_v5.log 2>&1
echo "[r5] DONE $(date -u)"
