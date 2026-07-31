#!/usr/bin/env bash
# 6h autonomous experiment runner on the pod.
# Advances through EXPERIMENTS when GPU is idle; never idle more than ~2 min.
# State: /root/MicroVLA/logs/auton_state.txt
set -euo pipefail
trap '' TERM HUP
cd /root/MicroVLA
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/root/LIBERO
STATE=logs/auton_state.txt
LOG=logs/auton_6h.log
CK=checkpoints/full_stageB_rec_fix.pt
NS=data/libero_object_grid/norm_stats.json
mkdir -p logs eval_results/auton

BUSY='train_batched|libero_eval|record_mp4|train_full'
echo "==== auton_6h start $(date -u) ====" | tee -a "$LOG"
echo "tick=0" > "$STATE"

# Experiment queue (one line each). Runner pops the first pending.
# Format: name|bash-command
QUEUE=logs/auton_queue.txt
if [ ! -f "$QUEUE" ]; then
  cat > "$QUEUE" <<'EOF'
tool_loose|python -m eval.libero_eval --suite libero_object --task-ids 0,1,2 --n-trials 3 --max-steps 500 --checkpoint checkpoints/full_stageB_rec_fix.pt --norm-stats data/libero_object_grid/norm_stats.json --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 --tool-phase --tool-gain 1.5 --tool-center-tol 0.12 --ibvs-sign 1,-1,0 --ibvs-descend -0.5 --ibvs-conf-floor 0.05 --device cuda:0 --heads-device cpu --workers 1 --out-dir eval_results/auton/tool_loose
tool_loose_vids|python -m eval.record_mp4 --suite libero_object --n-videos 2 --task-ids 2,0 --max-steps 500 --res 256 --fps 17 --checkpoint checkpoints/full_stageB_rec_fix.pt --norm-stats data/libero_object_grid/norm_stats.json --camera robot0_eye_in_hand_image --perception-period 2 --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 --tool-phase --tool-gain 1.5 --tool-center-tol 0.12 --ibvs-sign 1,-1,0 --ibvs-descend -0.5 --ibvs-conf-floor 0.05 --device cuda:0 --heads-device cpu --out-dir eval_results/auton/tool_loose/videos
tool_agent|python -m eval.libero_eval --suite libero_object --task-ids 0,1,2 --n-trials 3 --max-steps 500 --checkpoint checkpoints/full_stageB_rec_fix.pt --norm-stats data/libero_object_grid/norm_stats.json --camera agentview_image --render-size 128 --perception-period 2 --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.08 --source-min-aspect 1.15 --tool-phase --tool-gain 1.2 --tool-center-tol 0.10 --ibvs-sign 1,1,0 --ibvs-descend -0.4 --ibvs-conf-floor 0.05 --device cuda:0 --heads-device cpu --workers 1 --out-dir eval_results/auton/tool_agent
tool_agent_vids|python -m eval.record_mp4 --suite libero_object --n-videos 1 --task-ids 2 --max-steps 500 --res 256 --fps 17 --checkpoint checkpoints/full_stageB_rec_fix.pt --norm-stats data/libero_object_grid/norm_stats.json --camera agentview_image --perception-period 2 --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.08 --source-min-aspect 1.15 --tool-phase --tool-gain 1.2 --tool-center-tol 0.10 --ibvs-sign 1,1,0 --ibvs-descend -0.4 --ibvs-conf-floor 0.05 --device cuda:0 --heads-device cpu --out-dir eval_results/auton/tool_agent/videos
rec_mild_baseline|python -m eval.libero_eval --suite libero_object --task-ids 0,1,2 --n-trials 3 --max-steps 400 --checkpoint checkpoints/full_stageB_rec_fix.pt --norm-stats data/libero_object_grid/norm_stats.json --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 --det-conf 0.02 --no-brake --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.3 --ibvs-conf-floor 0.1 --device cuda:0 --heads-device cpu --workers 1 --out-dir eval_results/auton/rec_mild
hybrid_note|echo HYBRID_PLACEHOLDER — agent will inject after tool results
EOF
fi

END_EPOCH=$(( $(date +%s) + 6*3600 ))
while [ "$(date +%s)" -lt "$END_EPOCH" ]; do
  if pgrep -f "$BUSY" >/dev/null 2>&1; then
    echo "$(date -u +%H:%M:%S) busy; sleep 90" | tee -a "$LOG"
    sleep 90
    continue
  fi
  # pop next experiment
  if [ ! -s "$QUEUE" ]; then
    echo "$(date -u +%H:%M:%S) queue empty — idle heartbeat" | tee -a "$LOG"
    echo "AUTON_IDLE $(date -u)" | tee -a "$LOG"
    sleep 180
    continue
  fi
  LINE=$(head -1 "$QUEUE")
  NAME=${LINE%%|*}
  CMD=${LINE#*|}
  # remove first line portably
  tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
  echo "$(date -u +%H:%M:%S) START $NAME" | tee -a "$LOG"
  echo "AUTON_START $NAME $(date -u)" | tee -a "$LOG"
  set +e
  bash -c "$CMD" >> "logs/auton_${NAME}.log" 2>&1
  RC=$?
  set -e
  echo "$(date -u +%H:%M:%S) DONE $NAME rc=$RC" | tee -a "$LOG"
  echo "AUTON_DONE $NAME rc=$RC $(date -u)" | tee -a "$LOG"
  # scrape headline metrics if present
  grep -aE "mean_success|grip_close|eef_obj_dist_min" "logs/auton_${NAME}.log" | tail -5 | tee -a "$LOG" || true
done
echo "==== auton_6h end $(date -u) ====" | tee -a "$LOG"
echo "AUTON_FINISHED $(date -u)" | tee -a "$LOG"
