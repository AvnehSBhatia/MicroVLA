#!/bin/bash
cd /root/MicroVLA
export MUJOCO_GL=osmesa
python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 10 --max-steps 600 \
  --checkpoint checkpoints/full_stageB_teacher_bc2.pt --norm-stats data/teacher_grid2/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/unaided_v2r > logs/unaided_v2r.log 2>&1
python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 5 --max-steps 600 \
  --checkpoint checkpoints/full_stageB_teacher_bc2.pt --norm-stats data/teacher_grid2/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --action-gain 3.0 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/diag_gain3 > logs/diag_gain3.log 2>&1
