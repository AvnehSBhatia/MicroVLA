#!/bin/bash
# UNAIDED round 5 — fix the grip collapse.
#
# Diagnosis (bc3/bc4 telemetry):
#   bc3 reaches eef_obj_min ~0.05–0.08 m but grip_close_rate = 0 (always open).
#   bc4 barely closes in closed loop either; aggregate dagger polluted labels
#   (25/40 dagger eps never close; those eps are longer than teacher).
#   Closed-loop failure is covariate shift: the jaw never sees grasp-shaped
#   states with close labels on the approach distribution bc3 actually visits.
#
# Fix: DAgger with bc3 as student (already near the object) + PhasedIBVS teacher
# so close labels land on near-object student states; retrain stage B from bc3
# on teacher_grid2 + new dagger; eval unaided (no assist flags).
set -u
cd /root/MicroVLA
export MUJOCO_GL=osmesa PYTHONUNBUFFERED=1
trap "" TERM HUP

mkdir -p logs eval_results/prox_bc3 eval_results/unaided_v5 data

echo "[r5] kill stale eval/film if any $(date -u)"
# Prefer exact PIDs over pkill -f (SSH cmdline self-match).
for pid in $(pgrep -f 'eval.libero_eval.*unaided_v4' || true); do
  kill "$pid" 2>/dev/null || true
done
for pid in $(pgrep -f 'film_bc4' || true); do
  kill "$pid" 2>/dev/null || true
done
sleep 2

echo "[r5] proximity smoke on bc3 (ASSISTED diagnostic) $(date -u)"
python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 5 --max-steps 600 \
  --checkpoint checkpoints/full_stageB_teacher_bc3.pt \
  --norm-stats data/teacher_grid2/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --grip-close-dist 0.07 --grip-close-lift 0.2 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/prox_bc3 > logs/prox_bc3.log 2>&1 || true
python3 - <<'PY'
import json
from pathlib import Path
js=list(Path('eval_results/prox_bc3').glob('*results.json'))
print('[r5] prox_bc3 results', js[0] if js else None)
if js:
  d=json.load(open(js[0]))
  print(json.dumps({k:d.get(k) for k in ('mean_success','per_task','intermediates')}, indent=2))
PY

echo "[r5] DAgger record: student=bc3 teacher=PhasedIBVS $(date -u)"
rm -rf data/dagger_raw5 data/dagger_grid5
python -m preprocess.teacher_rollouts record \
  --suite libero_object --task-id 0 --n-success 50 --max-attempts 60 \
  --init-offset 200 --raw-dir data/dagger_raw5 --max-steps 400 \
  --dagger-beta 0.5 \
  --dagger-student-flags "--checkpoint checkpoints/full_stageB_teacher_bc3.pt --norm-stats data/teacher_grid2/norm_stats.json --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 --device cuda:0 --heads-device cpu" \
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
  > logs/dagger_rec5.log 2>&1 || exit 1

echo "[r5] convert + purge raw $(date -u)"
python -m preprocess.teacher_rollouts convert \
  --raw-dir data/dagger_raw5 --out data/dagger_grid5 --spatial-grid 4 \
  --camera eye_in_hand_rgb --device cuda:0 --det-conf 0.02 \
  --role-disjoint-iou 0.1 --purge-raw > logs/dagger_convert5.log 2>&1 || exit 1

python3 - <<'PY'
import numpy as np
from pathlib import Path
n=0; n_close=0
for p in Path('data/dagger_grid5').glob('*.npz'):
  g=np.load(p)['pwm_targets'][:,0,-1]; n+=1
  if (g>0).any(): n_close+=1
print(f'[r5] dagger_grid5 eps={n} with_close={n_close}')
PY

echo "[r5] train teacher_bc5 from bc3 on teacher_grid2 + dagger_grid5 $(date -u)"
python -u train/train_batched.py \
  --data-dir data/teacher_grid2 --data-dir data/dagger_grid5 --v8 --tqsa --seed 0 \
  --batch-size 8 --device cuda --lr 1e-4 --reserve-vram-gb 0 --max-vram-gb 0 \
  --no-cache-spatial \
  --load-stage-a checkpoints/full_stageB_teacher_bc3.pt --resume-stage-b \
  --stage-a-epochs 0 --stage-b-epochs 20 --stage-b-patience 6 --stage-b-select bc \
  --stage-b-min-epochs 6 \
  --dream-frac 0.0 --planner-input-dropout 0.0 --phase-dropout 0.0 \
  --waypoint-weight 0.0 --actuation-weight 0.0 --smooth-weight 0.05 --wm-aux-weight 0.0 \
  --grip-weight 3.0 --row0-weight 2.0 --pre-grasp-weight 2.0 \
  --recovery-noise 0.02 --variance-weight 0.3 \
  --magnitude-weight 0.8 --gain-magnitude-weight 0.3 \
  --centering-weight 0.5 --centering-uv 0.5,0.60 --centering-sign 1,-1 \
  --depth-weight 0.5 --depth-descend -0.3 \
  --action-token-sampling 0.5 \
  --tag teacher_bc5 > logs/teacher_bc5_train.log 2>&1 || exit 1

echo "[r5] unaided_v5 eval $(date -u)"
python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 10 --max-steps 600 \
  --checkpoint checkpoints/full_stageB_teacher_bc5.pt \
  --norm-stats data/teacher_grid2/norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --out-dir eval_results/unaided_v5 > logs/unaided_v5.log 2>&1 || exit 1

python3 - <<'PY'
import json
from pathlib import Path
js=list(Path('eval_results/unaided_v5').glob('*results.json'))
print('[r5] unaided_v5', js[0] if js else None)
if js:
  print(json.dumps(json.load(open(js[0])), indent=2)[:2000])
PY
echo "[r5] DONE $(date -u)"
touch /tmp/teacher_bc5_READY
