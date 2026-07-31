# Autonomous 6h run — agent loop checklist

On every wake:
1. SSH pod: check GPU, running jobs, `logs/auton_6h.log` / `AUTON_*` lines.
2. If `tool_phase` finished: pull metrics + videos to `watch_videos/tool_phase/`;
   decide next: if grip_close==0 tighten/loosen tool tols and append to
   `logs/auton_queue.txt`; if success>0 celebrate and record more.
3. If queue empty and GPU idle: invent next experiment (tool tol sweep,
   hybrid policy→tool handoff, mild IBVS A/B, fine-tune if a win appears).
   Append to `logs/auton_queue.txt`. Prefer eval-only until a metric moves.
4. Pull any new mp4s under `eval_results/auton/**/videos` locally.
5. Commit/push only when code changed; never force-push.
6. Stop after 6h from start (`AUTON_FINISHED`) or user says stop.
7. Do NOT crank IBVS gain >1.0 on wrong binds. Prefer tools / binding.
8. Keep GPU busy: if idle >3 min with empty queue, enqueue something.
