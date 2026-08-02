#!/usr/bin/env bash
# Record a watch-pack of MP4s for the latest agentview arms.
# Runs with low VRAM heads on CPU if needed; safe-ish next to a 4GB train.
set -euo pipefail
trap '' TERM HUP
cd /root/MicroVLA
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/root/LIBERO
OUT=eval_results/watch_pack
mkdir -p "$OUT"/{agent,agentpd,ibvs_phase_agentview,rec_fix_wrist}
LOG=logs/watch_pack.log
exec >>"$LOG" 2>&1
echo "==== watch_pack $(date -u) ===="

rec() {
  local name="$1"; shift
  echo "### $name $(date -u +%H:%M)"
  python -m eval.record_mp4 "$@" || echo "FAIL $name"
}

# Agentview policy (arm 1) — tasks 0,1,2
if [ -f checkpoints/full_stageB_agent.pt ]; then
  rec agent \
    --suite libero_object --n-videos 3 --task-ids 0,1,2 \
    --checkpoint checkpoints/full_stageB_agent.pt \
    --norm-stats data/libero_object_agent/norm_stats.json \
    --camera agentview_image --res 128 --max-steps 400 \
    --perception-period 2 --det-conf 0.02 --no-brake \
    --device cuda:0 --heads-device cpu \
    --out-dir "$OUT/agent"
fi

# Phase-dropout arm
if [ -f checkpoints/full_stageB_agentpd.pt ]; then
  NS=$(ls data/libero_object_agent*/norm_stats.json 2>/dev/null | head -1)
  # try agent norm first, then agentpd data dir
  [ -f data/libero_object_agentpd/norm_stats.json ] && NS=data/libero_object_agentpd/norm_stats.json
  [ -f data/libero_object_agent/norm_stats.json ] && NS=data/libero_object_agent/norm_stats.json
  rec agentpd \
    --suite libero_object --n-videos 3 --task-ids 0,1,2 \
    --checkpoint checkpoints/full_stageB_agentpd.pt \
    --norm-stats "$NS" \
    --camera agentview_image --res 128 --max-steps 400 \
    --perception-period 2 --det-conf 0.02 --no-brake \
    --device cuda:0 --heads-device cpu \
    --out-dir "$OUT/agentpd"
fi

# Phased IBVS falsifier (binding story) on agentview
rec ibvs_phase \
  --suite libero_object --n-videos 3 --task-ids 0,1,2 \
  --checkpoint checkpoints/full_stageB_agent.pt \
  --norm-stats data/libero_object_agent/norm_stats.json \
  --camera agentview_image --res 128 --max-steps 400 \
  --perception-period 2 --det-conf 0.02 --no-brake \
  --ibvs-phase --ibvs-gain 0.5 --ibvs-sign 1,1,0 --ibvs-descend -0.3 \
  --ibvs-conf-floor 0.005 --ibvs-track-gate 0.15 \
  --device cuda:0 --heads-device cpu \
  --out-dir "$OUT/ibvs_phase_agentview"

# Wrist rec_fix honest protocol — tasks 0,1,2
if [ -f checkpoints/full_stageB_rec_fix.pt ] && [ -f data/libero_object_grid/norm_stats.json ]; then
  rec rec_fix_wrist \
    --suite libero_object --n-videos 3 --task-ids 0,1,2 \
    --checkpoint checkpoints/full_stageB_rec_fix.pt \
    --norm-stats data/libero_object_grid/norm_stats.json \
    --camera robot0_eye_in_hand_image --res 256 --max-steps 400 \
    --perception-period 2 --det-conf 0.02 --no-brake \
    --device cuda:0 --heads-device cpu \
    --out-dir "$OUT/rec_fix_wrist"
fi

echo "==== done $(date -u) ===="
find "$OUT" -name '*.mp4' -ls
