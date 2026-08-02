#!/bin/bash
# UNAIDED_PLAN round 3 — DAgger: student (teacher_bc2) drives, PhasedIBVS
# soup-v1 teacher labels every visited state. Aggregate with the round-2
# 100-episode teacher corpus, retrain stage B, eval unaided_v3.
# Waits for the in-flight evals (unaided_v2r + diag_gain3) to free the CPU.
set -u
cd /root/MicroVLA
export MUJOCO_GL=osmesa

# ---- wait for both evals to publish mean_success (poll, no PID coupling)
while true; do
  a=$(grep -c "mean_success" logs/unaided_v2r.log 2>/dev/null || true)
  b=$(grep -c "mean_success" logs/diag_gain3.log 2>/dev/null || true)
  [ "${a:-0}" -ge 1 ] && [ "${b:-0}" -ge 1 ] && break
  sleep 60
done

# ---- 1. DAgger recording: 60 episodes, inits 150+, failures KEPT
python -m preprocess.teacher_rollouts record \
  --suite libero_object --task-id 0 --n-success 60 --max-attempts 70 \
  --init-offset 150 --raw-dir data/dagger_raw3 --max-steps 400 \
  --dagger-beta 0.3 \
  --dagger-student-flags "--checkpoint checkpoints/full_stageB_teacher_bc2.pt --norm-stats data/teacher_grid2/norm_stats.json --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 --device cuda:0 --heads-device cpu" \
  -- \
  --checkpoint checkpoints/full_stageB_rec_fix.pt \
  --norm-stats data/libero_object_grid/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --ibvs-phase --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.4 \
  --ibvs-descend-hyst 0.50 --ibvs-target-uv 0.5,0.60 \
  --ibvs-grasp-offset 0.08,-0.05 --ibvs-close-z 0.045 --ibvs-press 0.2 \
  --ibvs-retry-rise 8 --ibvs-gate-z 0.10 --ibvs-approach-z 0.12 \
  --ibvs-place-at=-0.010,0.255 --ibvs-drop-z 0.18 \
  --device cuda:0 --heads-device cpu \
  > logs/dagger_rec3.log 2>&1 || exit 1

# ---- 2. convert (fresh norm stats; same quantile scheme as teacher_grid2)
python -m preprocess.teacher_rollouts convert \
  --raw-dir data/dagger_raw3 --out data/dagger_grid3 --spatial-grid 4 \
  --camera eye_in_hand_rgb --device cuda:0 --det-conf 0.02 \
  --role-disjoint-iou 0.1 --purge-raw > logs/dagger_convert3.log 2>&1 || exit 1

# ---- 3. train teacher_bc3 on AGGREGATED corpus (100 teacher + 60 dagger)
python -u train/train_batched.py \
  --data-dir data/teacher_grid2 data/dagger_grid3 --v8 --tqsa --seed 0 \
  --batch-size 8 --device cuda --lr 2e-4 --reserve-vram-gb 0 --max-vram-gb 0 \
  --no-cache-spatial \
  --load-stage-a checkpoints/full_stageB_rec_fix.pt --resume-stage-b \
  --stage-a-epochs 0 --stage-b-epochs 24 --stage-b-patience 8 \
  --stage-b-select bc --stage-b-min-epochs 8 \
  --dream-frac 0.0 --planner-input-dropout 0.0 --phase-dropout 0.0 \
  --waypoint-weight 0.0 --actuation-weight 0.0 --smooth-weight 0.05 \
  --wm-aux-weight 0.0 --grip-weight 2.0 --row0-weight 2.0 \
  --recovery-noise 0.01 --variance-weight 0.1 --action-token-sampling 0.5 \
  --tag teacher_bc3 > logs/teacher_bc3_train.log 2>&1 || exit 1

# ---- 4. unaided_v3 eval (NO assist flags)
python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 10 \
  --max-steps 600 \
  --checkpoint checkpoints/full_stageB_teacher_bc3.pt \
  --norm-stats data/teacher_grid2/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/unaided_v3 > logs/unaided_v3.log 2>&1
echo "round3 chain done"
