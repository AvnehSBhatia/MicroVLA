#!/usr/bin/env bash
# Binding push: source_max_area + role_disjoint on the best wrist policy,
# plus agentview phased IBVS falsifier. No more gain cranking.
set -euo pipefail
trap '' TERM HUP
cd /root/MicroVLA
git fetch origin && git reset --hard origin/main
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/root/LIBERO

CK=checkpoints/full_stageB_rec_fix.pt
NS=data/libero_object_grid/norm_stats.json
LOG=logs/bind_push.log
mkdir -p logs eval_results/bind_wrist eval_results/bind_agent_phase
echo "==== bind_push $(date -u) ====" | tee "$LOG"

echo "==== 1) wrist rec_fix + disj + area + mild IBVS ====" | tee -a "$LOG"
python -m eval.libero_eval \
  --suite libero_object --task-ids 0,1,2 --n-trials 3 --max-steps 400 \
  --checkpoint "$CK" --norm-stats "$NS" \
  --camera robot0_eye_in_hand_image --render-size 256 \
  --perception-period 2 --det-conf 0.02 --no-brake \
  --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.3 \
  --ibvs-conf-floor 0.1 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/bind_wrist 2>&1 | tee -a "$LOG" \
  | grep -aE 'mean_success|src_detect|grip_close|eef_obj|====|DONE' || true

echo "==== 2) agentview phased + disj + area (binding falsifier) ====" | tee -a "$LOG"
python -m eval.libero_eval \
  --suite libero_object --task-ids 0,1,2 --n-trials 3 --max-steps 400 \
  --checkpoint "$CK" --norm-stats "$NS" \
  --camera agentview_image --render-size 128 \
  --perception-period 2 --det-conf 0.02 --no-brake \
  --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --ibvs-phase --ibvs-gain 0.5 --ibvs-sign 1,1,0 --ibvs-descend -0.3 \
  --ibvs-conf-floor 0.005 --ibvs-track-gate 0.15 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/bind_agent_phase 2>&1 | tee -a "$LOG" \
  | grep -aE 'mean_success|src_detect|grip_close|eef_obj|====|DONE' || true

echo "==== 3) record salad+soup under best of the two (wrist mild) ====" | tee -a "$LOG"
python -m eval.record_mp4 \
  --suite libero_object --n-videos 2 --task-ids 2,0 --max-steps 500 \
  --res 256 --fps 17 \
  --checkpoint "$CK" --norm-stats "$NS" \
  --camera robot0_eye_in_hand_image \
  --perception-period 2 --det-conf 0.02 --no-brake \
  --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.3 \
  --device cuda:0 --heads-device cpu \
  --out-dir eval_results/bind_wrist/videos 2>&1 | tee -a "$LOG" | tail -20

echo "==== 4) record agentview phase salad ====" | tee -a "$LOG"
python -m eval.record_mp4 \
  --suite libero_object --n-videos 1 --task-ids 2 --max-steps 500 \
  --res 256 --fps 17 \
  --checkpoint "$CK" --norm-stats "$NS" \
  --camera agentview_image \
  --perception-period 2 --det-conf 0.02 --no-brake \
  --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --ibvs-phase --ibvs-gain 0.5 --ibvs-sign 1,1,0 --ibvs-descend -0.3 \
  --ibvs-conf-floor 0.005 --ibvs-track-gate 0.15 \
  --device cuda:0 --heads-device cpu \
  --out-dir eval_results/bind_agent_phase/videos 2>&1 | tee -a "$LOG" | tail -20

echo "BIND_PUSH_DONE $(date -u)" | tee -a "$LOG"
