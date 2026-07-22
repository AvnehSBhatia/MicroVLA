# Closed-loop LIBERO eval on the GPU box

Turns the trained checkpoint into the real number: **task success rate**. Done
in stages so failures surface in the cheapest place first. Run each stage;
only move on when it passes.

The box is Linux with disk/network to spare, so the macOS blockers in
`SIM_SETUP.md` don't apply — the only real requirement is `robosuite==1.4.0`
(LIBERO's env code doesn't work on 1.5.x) plus LIBERO's 3D assets (they ship
inside the cloned repo).

## Stage 0 — install the sim stack (separate venv, to protect the ROCm torch)

The training venv has ROCm torch; don't risk it. Use a dedicated eval venv:

```bash
cd ~
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
python -m venv ~/eval_venv && source ~/eval_venv/bin/activate

# LIBERO + its pinned robosuite (1.4.0) + assets (in the repo):
cd ~/LIBERO
pip install -e .                         # pulls robosuite 1.4.0, bddl, mujoco, etc.

# MicroVLA + its perception stack (the policy runs YOLO-World per real tick):
cd ~/MicroVLA
pip install -e ".[perception]"           # ultralytics, opencv, torchvision
pip install torch numpy                   # CPU or ROCm torch — the model is tiny, CPU is fine for eval
```

If `pip install -e .` in LIBERO tries to drag in a conflicting torch, install
LIBERO's deps first and torch last, or `pip install -e . --no-deps` then add
`robosuite==1.4.0 bddl mujoco==3.3.0 gymnasium` by hand.

## Stage 1 — bare env smoke (NO policy) — the highest-risk step

```bash
cd ~/MicroVLA
MUJOCO_GL=egl python -m eval.env_smoke --suite libero_object --task 0
```
Expect: it enumerates the suite, builds the env, steps 5 random actions, prints
the `robot0_eye_in_hand_image` shape `(128,128,3) uint8`, and saves
`eval_results/env_smoke_wrist.png`. **Open that PNG and compare to a training
frame** (a `data/libero/*.npz` was made from the same wrist view). If it's
upside-down or mirrored vs training, tell me — the eval needs a matching flip,
or perception will be grounding a scene the model never saw. Common LIBERO
gotcha: images may need a vertical flip.

If Stage 1 errors, paste it — this is where install/API/version issues show up,
and it's cheap to iterate here.

## Stage 2 — one policy episode (mock still available as a fallback)

```bash
# sanity: the harness itself works end-to-end with mocks (no sim):
python -m eval.libero_eval --mock-env --suite libero_object --n-trials 1 --checkpoint checkpoints/full_stageB.pt

# then ONE real episode with the trained policy:
MUJOCO_GL=egl python -m eval.libero_eval --suite libero_object --n-trials 1 --max-steps 300 \
    --checkpoint checkpoints/full_stageB.pt \
    --norm-stats data/libero/norm_stats.json
```
Watch for: does it run 300 steps without crashing, and does the telemetry
trust/plan look sane? Success on one episode is a bonus; not crashing is the bar.

## Stage 3 — the real number

```bash
MUJOCO_GL=egl python -m eval.libero_eval --suite libero_object --n-trials 20 --max-steps 300 \
    --checkpoint checkpoints/full_stageB.pt --norm-stats data/libero/norm_stats.json
# repeat for --suite libero_spatial and libero_goal
```
Output: per-task + mean success rate, telemetry JSONL in `eval_results/`. That
mean success is the paper's Claim-1 number, and it decides whether the planner
needs the multimodal-head upgrade.

## Notes / likely iteration points

- **Camera match**: training used LIBERO `eye_in_hand_rgb` un-rotated (the
  agentview needed 180°, the wrist did not). The eval reads
  `robot0_eye_in_hand_image`. Stage 1's PNG confirms whether they align.
- **Perception cost**: the policy runs YOLO-World every real tick
  (`perception_period`, default matches the 2 Hz training). On the box that's
  fine; if slow, it's the detector, not the sim.
- **Action space**: the policy denormalizes plan[0] via `norm_stats.json` back
  to LIBERO's raw 7-DoF action space, so the numbers round-trip — but this is
  the other thing to sanity-check if the robot moves erratically.
- **Success detection**: the harness tries `info['success']` then
  `env.check_success()`; Stage 1 reports which LIBERO exposes.
