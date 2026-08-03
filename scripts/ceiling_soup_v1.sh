#!/usr/bin/env bash
# Re-baseline assisted soup with the ACTUAL soup_v1 constants (0.750 historically).
# ceiling_ibvs3 wrongly used cream offsets (0.08,-0.05) → 0/3 at eef_min~13cm.
set -u
cd /root/MicroVLA
export MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 PYOPENGL_PLATFORM=osmesa
mkdir -p logs eval_results
exec >>logs/ceiling_soup_v1.log 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START ceiling_soup_v1"

# Wait for unaided_push2 (and any GPU hog) to finish first
for i in $(seq 1 180); do
  if pgrep -f 'scripts/unaided_push2.sh' >/dev/null 2>&1 \
     || pgrep -f 'eval.libero_eval' >/dev/null 2>&1 \
     || pgrep -f 'train/train_batched|train.train_batched' >/dev/null 2>&1; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting push2/GPU ($i)"
    sleep 30
  else
    break
  fi
done

python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 4 --max-steps 600 \
  --checkpoint checkpoints/full_stageB_rec_fix.pt \
  --norm-stats data/libero_object_grid/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --ibvs-phase --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.4 \
  --ibvs-descend-hyst 0.50 --ibvs-target-uv 0.5,0.60 \
  --ibvs-grasp-offset 0.09,-0.186 --ibvs-close-z 0.045 --ibvs-press 0.2 \
  --ibvs-retry-rise 8 --ibvs-gate-z 0.10 --ibvs-approach-z 0.12 \
  --ibvs-place-at=-0.006,0.260 --ibvs-drop-z 0.25 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/ceiling_soup_v1

python3 - <<'PY'
import json,glob
fs=sorted(glob.glob('eval_results/ceiling_soup_v1/*results.json'))
print('ceiling_soup_v1', json.load(open(fs[-1]))['mean_success'] if fs else 'none')
PY
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE"
touch /tmp/CEILING_SOUP_V1_READY
