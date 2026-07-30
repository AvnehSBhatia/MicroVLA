#!/bin/bash
set -euo pipefail
cd /root/MicroVLA
git fetch origin && git reset --hard origin/main
mkdir -p logs eval_results/ibvs_clip_rerank
LOG=logs/ibvs_clip_rerank.log
exec > >(tee -a "$LOG") 2>&1
echo "==== ibvs_clip_rerank $(date -u) ===="
python -m eval.libero_eval \
  --suite libero_object \
  --n-trials 3 \
  --max-steps 400 \
  --checkpoint checkpoints/full_stageB_rec_fix.pt \
  --norm-stats data/libero_object_grid/norm_stats.json \
  --perception-period 2 \
  --det-conf 0.02 \
  --render-size 256 \
  --task-ids 0,1,2 \
  --device cuda:0 \
  --heads-device cuda:0 \
  --workers 1 \
  --no-brake \
  --ibvs-phase \
  --ibvs-gain 0.5 \
  --ibvs-sign 1,1,0 \
  --ibvs-descend -0.3 \
  --ibvs-conf-floor 0.005 \
  --ibvs-track-gate 0.15 \
  --ibvs-clip-rerank \
  --out-dir eval_results/ibvs_clip_rerank
echo "==== done $(date -u) ===="
