#!/bin/bash
set -euo pipefail
cd /root/MicroVLA
mkdir -p logs eval_results/night_sighted
LOG=logs/night_paper.log
exec >>"$LOG" 2>&1
echo "==== night_paper $(date -u) ===="

TRAIN_PID=$(pgrep -f "train_batched.py.*tag rec_fix" || true)
if [ -n "${TRAIN_PID:-}" ]; then
  echo "waiting for train pid $TRAIN_PID ..."
  while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 30; done
  echo "train finished $(date -u)"
else
  echo "no rec_fix train running"
fi

CKPT=checkpoints/full_stageB_rec_fix.pt
if [ ! -f "$CKPT" ]; then
  echo "missing $CKPT"; exit 1
fi
echo "evaluating $CKPT ($(ls -la $CKPT))"

python -m eval.libero_eval \
  --suite libero_object \
  --n-trials 5 \
  --max-steps 400 \
  --checkpoint "$CKPT" \
  --norm-stats data/libero_object_grid/norm_stats.json \
  --perception-period 2 \
  --det-conf 0.02 \
  --render-size 256 \
  --task-ids 0,1,2,3,4 \
  --device cuda:0 \
  --heads-device cuda:0 \
  --workers 1 \
  --no-brake \
  --out-dir eval_results/night_sighted

echo "==== done $(date -u) ===="
python - <<'PY'
import json, glob
paths = sorted(glob.glob("eval_results/night_sighted/**/*.json", recursive=True))
for p in paths[-8:]:
    try:
        d = json.load(open(p))
    except Exception:
        continue
    print(p, "mean_success=", d.get("mean_success"),
          "intermediates=", d.get("intermediates"))
PY
