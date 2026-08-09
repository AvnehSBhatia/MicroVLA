#!/bin/bash
# Everything this paper still needs, in one command on a fresh box.
#
#   bash scripts/run_remaining_experiments.sh <host> <port> [key]
#
# Sets up the pinned deployment stack, uploads the repo and the repaired
# evaluation suite, and runs the three experiments named in the paper as
# outstanding. Results land in results/ ready for the verifier.
#
# The three, in the order the paper's own reviewers ranked them:
#
#  E1  SHIPPED vs REPAIRED.  The released head on task 0, on the fifty states
#      LIBERO ships and on the fifty we generated from the region its task file
#      declares. This is the experiment that decides whether the pinning was
#      load-bearing FOR A POLICY, which is the one thing Section 2 cannot say.
#      A drop is the paper's missing headline; no drop bounds the defect's
#      practical cost, which is also worth knowing and is currently unknown.
#
#  E2  THE JITTER SWEEP.  Displace the blind arm's grasp constant by r and read
#      the tolerance off the curve, per object. This measures A1 on the right
#      quantity -- the grocery during grasping -- rather than on the container
#      during placing, which is the transport error Appendix C confesses.
#
#  E3  THE SPATIAL CONTROL.  Reset-oracle against blind on LIBERO-Spatial. If
#      the ceiling is high and blind is not, the radius predicts failure where
#      it should. A bare blind failure there proves nothing on its own, which
#      is why the ceiling arm is run alongside.
set -euo pipefail
HOST="${1:?usage: run_remaining_experiments.sh <host> <port> [key]}"
PORT="${2:?}"
KEY="${3:-$HOME/.ssh/id_ed25519}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT -i $KEY root@$HOST"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "== 1/4  upload =="
tar czf - --exclude='.git' --exclude='checkpoints' --exclude='watch_videos' \
    --exclude='data' --exclude='eval_results' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='.venv' --exclude='runs' --exclude='logs' \
    -C "$REPO" . | $SSH 'mkdir -p /workspace/MicroVLA && tar xzf - --no-same-owner -C /workspace/MicroVLA' 2>/dev/null || true
$SSH 'grep -c "Search every suite" /workspace/MicroVLA/eval/libero_eval.py'

echo "== 2/4  stack (pinned to the build every published cell was measured on) =="
$SSH 'bash -s' <<'REMOTE'
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq libosmesa6-dev libgl1-mesa-glx \
    libglew-dev patchelf libglfw3 libegl1 git >/dev/null 2>&1
pip install -q numpy==2.2.6 robosuite==1.4.1 mujoco==2.3.7 bddl easydict thop \
    opencv-python-headless h5py ultralytics==8.4.115 imageio imageio-ffmpeg \
    matplotlib termcolor future cloudpickle gym transformers 2>&1 | tail -2
[ -d /workspace/LIBERO ] || git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git /workspace/LIBERO
cd /workspace/LIBERO && git checkout -q 8f1084e3 && pip install -q -e . --no-deps
rm -f /root/.libero/config.yaml
cd /workspace && echo N | python -c "from libero.libero import benchmark; print('libero ok')"
REMOTE

echo "== 3/4  install the repaired suite alongside the shipped one =="
$SSH 'mkdir -p /workspace/repaired && cp /workspace/MicroVLA/results/resampled_init/*.pruned_init /workspace/repaired/ && ls /workspace/repaired | wc -l'

echo "== 4/4  launch E1, E2, E3 =="
$SSH 'cat > /workspace/run_all.sh' <<'REMOTE'
cd /workspace/MicroVLA
export MUJOCO_GL=osmesa PYTHONPATH=/workspace/MicroVLA YOLO_CONFIG_DIR=/tmp/Ultralytics
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MICROVLA_ALLOW_SIGTERM=1
BASE="--max-steps 300 --seed 20 --checkpoint models/full_stageB_rec_fix.pt \
--norm-stats eval/identity_norm_stats.json --camera robot0_eye_in_hand_image \
--render-size 256 --perception-period 2 --det-conf 0.02 --no-brake \
--role-disjoint-iou 0.1 --source-max-area 0.12 --device cpu --heads-device cpu \
--workers 1 --goal-ckpt models/goal_heads_v5.pt"
mkdir -p /workspace/out /workspace/jobs; JL=/workspace/jobs/all.txt; : > $JL
# E1: released head, task 0, shipped vs repaired states (n=50 each)
for r in $(seq 0 49); do
  echo "E1shipped_r${r}|--suite libero_object --task-ids 0 --n-trials 1 --trial-offset $r" >> $JL
done
# E2: jitter sweep on the two tasks where blind scores above zero
for t in 0 3; do for j in 1 2 4 8; do for r in $(seq 0 9); do
  echo "E2_t${t}_j${j}_r${r}|--suite libero_object --task-ids $t --n-trials 1 --trial-offset $r --goal-anchor blind --goal-anchor-jitter-cm $j" >> $JL
done; done; done
# E3: Spatial ceiling vs lookup, 5 tasks
for t in 0 1 2 3 4; do for r in $(seq 0 9); do
  echo "E3fix_t${t}_r${r}|--suite libero_spatial --task-ids $t --n-trials 1 --trial-offset $r --goal-anchor fixed" >> $JL
  echo "E3bl_t${t}_r${r}|--suite libero_spatial --task-ids $t --n-trials 1 --trial-offset $r --goal-anchor blind" >> $JL
done; done
wc -l $JL
run_one() {
  name="${1%%|*}"; flags="${1#*|}"; d=/workspace/out/$name
  grep -q mean_success "$d/log.txt" 2>/dev/null && return 0
  mkdir -p $d
  timeout -s KILL 2100 python -m eval.libero_eval $BASE $flags --out-dir $d > $d/log.txt 2>&1
  s=$(grep -o "mean_success [0-9.]*" $d/log.txt | tail -1 | awk '{print $2}')
  echo "$name rc=$? success=${s:-NA}" >> /workspace/jobs/ledger.txt
}
export -f run_one; export BASE
cat $JL | xargs -P 12 -I{} bash -c 'run_one "$@"' _ {}
echo ALL_DONE >> /workspace/jobs/ledger.txt
REMOTE
$SSH 'cd /workspace && setsid nohup bash run_all.sh > /workspace/run_all.log 2>&1 < /dev/null & exit 0'
sleep 20
$SSH 'wc -l < /workspace/jobs/all.txt; echo "running: $(pgrep -fc libero_ev[a]l)"'
echo
echo "launched. poll with:"
echo "  $SSH 'grep -c \"\" /workspace/jobs/ledger.txt'"
