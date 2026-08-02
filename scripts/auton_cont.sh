#!/usr/bin/env bash
# Continuous experiment runner — no deadline. Pops logs/auton_queue.txt forever.
trap '' TERM HUP
set -euo pipefail
cd /root/MicroVLA
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/root/LIBERO
LOG=logs/auton_cont.log
QUEUE=logs/auton_queue.txt
mkdir -p logs eval_results/auton
echo "==== auton_cont start $(date -u) ====" | tee -a "$LOG"
echo "AUTON_CONT_START $(date -u)" | tee -a "$LOG"

# Only count real python workers (SSH bash -c lines often mention module names).
is_busy() {
  ps -C python,python3 -o args= 2>/dev/null \
    | grep -qE 'eval\.libero_eval|eval\.record_mp4|eval\.record_demo_mp4|eval\.record_soup_angles|train\.|preprocess\.'
}

while true; do
  if is_busy; then
    echo "$(date -u +%H:%M:%S) busy; sleep 90" | tee -a "$LOG"
    sleep 90
    continue
  fi
  if [ ! -s "$QUEUE" ]; then
    echo "$(date -u +%H:%M:%S) queue empty — idle heartbeat" | tee -a "$LOG"
    echo "AUTON_IDLE $(date -u)" | tee -a "$LOG"
    sleep 180
    continue
  fi
  LINE=$(python3 - <<'PY'
import os, fcntl
from pathlib import Path
q = Path("logs/auton_queue.txt")
fd = os.open(str(q), os.O_RDWR | os.O_CREAT)
fcntl.flock(fd, fcntl.LOCK_EX)
try:
    lines = os.read(fd, 1 << 20).decode().splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        print("")
    else:
        print(lines[0])
        rest = "\n".join(lines[1:])
        if rest:
            rest += "\n"
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, 0)
        os.write(fd, rest.encode())
finally:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
PY
)
  if [ -z "$LINE" ]; then
    continue
  fi
  NAME=${LINE%%|*}
  CMD=${LINE#*|}
  echo "$(date -u +%H:%M:%S) START $NAME" | tee -a "$LOG"
  echo "AUTON_START $NAME $(date -u)" | tee -a "$LOG"
  set +e
  bash -c "$CMD" >> "logs/auton_${NAME}.log" 2>&1
  RC=$?
  set -e
  echo "$(date -u +%H:%M:%S) DONE $NAME rc=$RC" | tee -a "$LOG"
  echo "AUTON_DONE $NAME rc=$RC $(date -u)" | tee -a "$LOG"
  grep -aE "mean_success|grip_close|eef_obj_dist_min" "logs/auton_${NAME}.log" \
    | tail -5 | tee -a "$LOG" || true
done
