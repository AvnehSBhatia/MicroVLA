# HANDOFF — MicroVLA session state (updated 2026-07-25, session 2)

Read this top-to-bottom before touching anything. CLAUDE.md + DESIGN.md are the
binding contracts; this file is the *live state*: what's trained, what's
measured, what's mid-flight, and exactly where the last session stopped.

## 0. The one urgent thing — still `mean_success`, now instrumented

**Status: the eval harness was rebuilt to be diagnosable; the number is still
unobtained** (Mac-side session — nothing was run on the box).

What hung, and what was done about it. Symptoms were: 20 min, 10 workers alive
holding ~2.5 GB VRAM each, 0% GPU, zero telemetry files. A 13-agent adversarial
diagnosis narrowed the stall to ONE window — between `YoloWorldPerception`'s
`.to(device)` and the telemetry-file open — and could not separate three
candidates that share that window and emit an identical external signature:

1. **10-way concurrent HIP context init inside `.to('cuda:0')`** (leading: the
   2.5 GB IS the context being allocated; 0% util because no kernel ever ran).
2. `_real_tasks()` re-run per worker (libero→robosuite→mujoco import ×10).
3. Thread oversubscription — "crawling", not deadlocked.

Ruled OUT by ordering: CLIP `set_classes` download, osmesa/mujoco env creation
(both live inside the trial loop, after the file open), and worker-death
masking (a respawned `mp.Pool` worker holds 0 GB, not 2.5 GB). Also learned:
**0% GPU is the NORMAL state of this eval** — 14 of 15 ticks are dream ticks
that never touch the detector, and the heads were pinned to CPU regardless of
`--device`. That symptom was a red herring.

The harness now self-localizes rather than needing another guess (see §10 for
what changed). First moves on the box, in order:

```bash
source /root/eval_venv/bin/activate && cd /root/MicroVLA && git pull
export PFX="PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa PYTHONPATH=/root/LIBERO"

# 1) SERIAL CANARY (~3 min). Must print: building policy -> policy ready (Ns)
#    -> N task(s) -> START/DONE.
env $PFX python -m eval.libero_eval --suite libero_object --n-trials 1 \
  --max-steps 100 --checkpoint checkpoints/full_stageB.pt \
  --norm-stats data/libero_v7/norm_stats.json --device cuda:0 --workers 1

# 2) TWO WORKERS with a watchdog. The last heartbeat printed names the cause:
#    "building policy..." then silence -> cause 1 (HIP init)
#    "policy ready (Ns)" then silence  -> cause 2 (libero import)
#    all heartbeats, slow START->DONE  -> cause 3 (oversubscription)
env $PFX python -m eval.libero_eval --suite libero_object --n-trials 2 \
  --max-steps 250 --checkpoint checkpoints/full_stageB.pt \
  --norm-stats data/libero_v7/norm_stats.json --device cuda:0 \
  --workers 2 --stagger 10 --worker-timeout 1800 2>&1 | tee /tmp/eval2.log

# 3) SCALE once 2 workers finish clean: --workers 5 (then 10), same flags.
```

If a worker stalls anyway, it now self-reports without py-spy:
`kill -USR1 <pid>` dumps its stack (every worker prints its pid on startup),
and a stuck worker auto-dumps every 600 s. Isolate cause 1 with no LIBERO and
no policy in the loop:
```bash
python -c "
import multiprocessing as mp, time
def b(i):
    t=time.time(); import torch; torch.zeros(1, device='cuda:0'); return (i, round(time.time()-t,1))
if __name__=='__main__':
    for n in (1,2,10):
        t=time.time()
        with mp.get_context('spawn').Pool(n) as p: print(n, p.map(b, range(n)), 'wall %.1f'%(time.time()-t), flush=True)"
# 1 fast but 10 >> 10x slower (or hangs) => cause 1 confirmed.
```
Last-resort fallback, now supported directly — N independent single-task
processes, immune to every in-process concurrency hazard:
```bash
for T in $(seq 0 9); do
  env $PFX python -m eval.libero_eval --suite libero_object --task-ids $T \
    --n-trials 2 --max-steps 250 --checkpoint checkpoints/full_stageB.pt \
    --norm-stats data/libero_v7/norm_stats.json --device cuda:0 &
done; wait
# merge the per-task results: eval_results/*_results.json
```

## 1. Machines, envs, invariants

- **Mac** (this repo, `~/Code/MicroVLA`): dev + tests. `.venv` = torch/numpy/
  pytest ONLY (import microvla must work with just those). 149 tests:
  `.venv/bin/python -m pytest tests -q`. Not a git remote problem: user pushes
  from Mac, pulls on box.
- **Box** = MI300X, ROCm, container `96d8d89bde39`, repo at `/root/MicroVLA`.
  - `source /root/eval_venv/bin/activate` — main env (torch-rocm, ultralytics,
    h5py, LIBERO deps). Python 3.12.
  - Real-LIBERO commands need the prefix:
    `PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa PYTHONPATH=/root/LIBERO` (EGL is
    broken in the container; osmesa = CPU rendering = the wall-clock cost).
  - ROCm quirks (both already baked into code): YOLO must run fp32
    (`half=False` in perception), trainers set
    `TORCH_BLAS_PREFER_HIPBLASLT=0` (hipBLASLt segfaults on some GEMMs) — but
    KEEP prefixing train commands with it anyway, belt-and-braces.
  - `/root/tinyvla_venv310` — Python 3.10 env for the TinyVLA teacher
    (see §6). Separate on purpose.
- **Disk budget**: 10 GB rule applies to DATA (downloads/episodes). Suites are
  downloaded → converted → deleted, one at a time.

## 2. Architecture state (v7.1 — all committed, 149 tests green)

Deployed stack (see DESIGN.md for contracts): frozen YOLO-World-S (+ CLIP text
tower, role-ordered spatial grounding prompts, real-tick miss-hold) →
SlotResonanceFusion [B,32,5] → AnchoredDriftEncoder [B,256] →
**RecursiveTRM 9.97M** (root `TRM.py`, user authorized direct edits;
`forward_full()` → `{next_emb, next_box, msg[B,32]}`) → **ChronoQueryPlanner
1.76M** (two-stage decode: 3D waypoint tanh-cumsum → orientation+gripper
conditioned on waypoints; BCE gripper; inputs: next_emb, current_emb,
state_delta, fused, pred_box_emb, geometry[6], proprio[10], spatial{TQSA},
wm_msg[32]) + **TQSA 132K** (trainable text-queried attention on the frozen
SPPF map: per-role heatmaps 8×8, pooled role vecs, 4×4 token grid; trains in
stage B; loop runs it on real ticks, holds across dreams).

Key semantics locked in (do NOT regress):
- Actions are SYMMETRICALLY normalized (0 ⇔ no motion). `data/libero_v7/
  norm_stats.json` is symmetric; converter now bakes symmetric natively
  (`fit_symmetric`). Old asymmetric stats caused the drift-into-wall bug.
- `cfg.action_space="delta"` → progressive trust brake
  `min(1, τ/brake_trust)·raw` (τ never fell below 0.5 in telemetry ⇒ scale 1).
- Stage-B freeze policy: fusion/drift frozen; TRM core frozen but
  **msg_head trainable** (planner gradient shapes it); `--unfreeze-trm` exists
  (0.1× LR + WM-aux rollout loss), never yet used.
- Training regularizers (defaults ON): planner-input-dropout 0.15 (withholds
  fused/current_emb — cured the 7× fused dominance), drift-dropout 0.1
  (stage A — TRM delta was 0.63–0.88 drift-code-driven), dream-frac 0.25,
  row0-weight 2.0, smooth 0.05.

## 3. Data state (IMPORTANT)

- `data/libero_v7` = **libero_object ONLY** (~500 eps), v7 schema: 8 classic
  keys + `wrist_frames [T,128,128,3]u8` + `proprio [T,10]` +
  `eef_pos_chunk [T,5,3]`, symmetric stats. **Do NOT bake more suites into
  this dir incrementally** — each bake overwrites norm_stats.json fitted on
  its own input, so the episodes already in the dir no longer match the stats
  file. One dir per suite, then `preprocess/unify_norm_stats.py` to put them
  on one shared symmetric scale (commands in §5). That ordering also keeps
  peak disk at ONE raw suite instead of three.
- `data/bridge` = old schema (no frames/proprio; zero-filled by the dataset
  loader, proprio valid-flag 0). Fine as-is for stage A.
- `data/libero` = OLD asymmetric-renormed object+spatial+goal bake (pre-v7,
  patched with proprio for object only). Superseded; keep until v7_full lands.
- Trainer buckets are keyed **(T, has_frames)** — mixed buckets silently
  stripped frames once (`TQSA 0/37` bug, fixed `5b4a821`).

## 4. Results so far (the evidence trail)

**v7 pilot checkpoint** = `checkpoints/full_stageB.pt` on box (stage A: beat
persistence 0.0106 vs 0.0111 at epoch 18 WITH drift-dropout, early-stop; stage
B: resumed +TQSA after the bucket fix, early-stopping run with val tracking).

Bench (`eval/bench.py`, object data, protocol-matched, --device cuda:0):
- **std_ratio 0.369** (was ~0.12 in v4–v6 — the 8× action-magnitude collapse
  is broken; healthy ≈ 1.0, so ~⅓ vigor remains the gap)
- **grip_acc 0.93**, corr 0.49, pose_mae 0.20, **wm_margin +1.7%** (fair
  protocol; the earlier −5.3% was the bench holding fused frozen — fixed)
- Sensitivity (on-distribution |Δplan| when input withheld): proprio 0.291 ≫
  state_delta 0.075 > wm_msg 0.031 > current_emb 0.025 ≈ fused 0.023 >
  pred_box 0.013 > geometry 0.004 > next_emb→cur 0.001.
  → proprio = the phase signal (the correct diagnosis all along); fused
  dominance cured; msg channel alive; **geometry/next_emb/pred_box are
  measured-dead → pruning candidates** after the full-data run.

**Closed-loop `mean_success`: still unknown** (every earlier eval was 0/10
under now-fixed interface bugs; the v7 attempt hung — §0).

History of root-caused bugs (paper.md "Action-interface diagnosis" has the
full chain): asymmetric norm drift, delta-HOLD trust momentum, geometry
bottleneck, no proprio, TQSA-on-4×4-maps train/eval mismatch, inference-mode
tensors, bucket frame-stripping, flat-τ brake tax, parallel-eval opacity.

## 5. The staged plan (in order)

1. **Get `mean_success` for the pilot** (§0 canary → workers 5). Any nonzero =
   proof-of-life headline. Zero-with-purposeful-motion → next lever.
2. **Full 3-suite bake + full retrain** (~1.5 h bake + ~40 min train).
   ONE SUITE AT A TIME — `preprocess/libero.py` globs its root recursively, so
   baking all three under one norm_stats would need all three raw suites on
   disk at once (~10-12 GB = the whole budget). Download → bake → delete each,
   then put the three dirs on one shared normalizer:
   ```bash
   # the downloader is INTERACTIVE: `yes n` answers n/n = use Hugging Face,
   # don't overwrite an already-complete suite.
   for S in libero_object libero_spatial libero_goal; do
     yes n | python /root/LIBERO/benchmark_scripts/download_libero_datasets.py \
       --datasets $S --download-dir /root/libero_raw
     python -m preprocess.libero /root/libero_raw/$S data/${S}_v7 \
       --camera eye_in_hand_rgb --device cuda:0   # MUST match eval's camera
     rm -rf /root/libero_raw/$S            # BEFORE the next download
   done
   python -m preprocess.unify_norm_stats \
     --data-dir data/libero_object_v7 --data-dir data/libero_spatial_v7 \
     --data-dir data/libero_goal_v7
   LIB="--data-dir data/libero_object_v7 --data-dir data/libero_spatial_v7 \
        --data-dir data/libero_goal_v7"
   TORCH_BLAS_PREFER_HIPBLASLT=0 python train/train_batched.py \
     --data-dir data/bridge $LIB \
     --device cuda --batch-size 64 --lr 5e-4 --max-vram-gb 50 \
     --stage-a-epochs 30 --warmup-epochs 4 --max-horizon 6 --patience 3 \
     --stage-b-epochs 40 --stage-b-patience 4 --dream-frac 0.25 --tqsa
   python -m eval.bench --checkpoint checkpoints/full_stageB.pt \
     --data-dir data/libero_object_v7 --sensitivity --device cuda:0
   python -m preprocess.fit_waypoint_gain data/libero_object_v7 \
     data/libero_spatial_v7 data/libero_goal_v7
   ```
   Cheap A/Bs on top — BOTH levers are stage-B-only, so reuse the trained world
   model with `--load-stage-a` (~15 min each, not 40). Do NOT add
   `--resume-stage-b`: it does a strict `load_state_dict` into the planner and
   both flags change the planner's architecture.
   ```bash
   COMMON="--data-dir data/bridge $LIB --device cuda --batch-size 64 --lr 5e-4 \
     --max-vram-gb 50 --load-stage-a checkpoints/full_stageA.pt \
     --stage-b-epochs 40 --stage-b-patience 4 --dream-frac 0.25 --tqsa"
   # the std_ratio lever — read wp_std_ratio vs std_ratio in the bench output
   TORCH_BLAS_PREFER_HIPBLASLT=0 python train/train_batched.py $COMMON \
     --waypoint-weight 1.0 --tag wp
   # the pruning candidates, decided on FULL-data sensitivity, not the pilot's
   TORCH_BLAS_PREFER_HIPBLASLT=0 python train/train_batched.py $COMMON \
     --planner-drop geometry,pred_box_emb --tag pruned
   ```
3. **Remaining levers, ranked** (fire based on where the numbers stall):
   - std_ratio stuck ≈0.4 → **waypoint-absolute action head** (predict
     displacement to `eef_pos_chunk` waypoints from measured EEF each replan;
     data already baked; needs per-dim gain fit from (action, Δeef) pairs).
   - closed-loop ≪ bench → longer stage B via `--resume-stage-b`, higher
     dream-frac, or `--unfreeze-trm`.
   - after full-data validation → **prune measured-dead planner inputs**
     (geometry, next_emb path, pred_box) and re-bench; params → budget.
4. **TinyVLA distillation** (§6) — optional A/B, not a rescue: only worth it
   if bench shows label noise is the ceiling.

## 6. TinyVLA teacher status (side quest, parked)

Reality: the repo (github.com/liyaxuanliyaxuan/TinyVLA) publishes **base VLMs
only** (`/root/ckpts/Llava-Pythia-400M` downloaded ✓) — no trained policy
exists anywhere; we must train it ourselves.
- Env `/root/tinyvla_venv310` (py3.10, torch-2.13-rocm): requirements
  installed after cmake fix; `policy_heads` + `llava-pythia` need
  `pip install -e . --no-deps`; `pip uninstall -y bitsandbytes` was the last
  suggested fix for the import gate (CUDA-only bnb crashes on ROCm) — **gate
  pass unconfirmed**.
- Our side is DONE: `preprocess/libero_to_tinyvla.py` (LIBERO→ALOHA-style hdf5,
  `--demos-per-task`, `--action-10d` for their Franka 10-dim head),
  `TinyVLATeacher` wired to their real API (load_pretrained_model, pythia conv
  template, stats-pickle denorm, 6D-rot→euler, proprio as `states`), CLI
  `--teacher-base/--teacher-stats`.
- Their side remaining: constants.py task entry (converter prints it),
  train.sh edits (OUTPUT with 'llava_pythia'+'lora', task_name libero_all,
  num_gpus=1, bs 8), `bash scripts/train.sh` (hours; deepspeed-on-ROCm risk),
  `process_ckpts.sh`, then bake `data/libero_tv` with `--teacher tinyvla`
  (smoke `--limit 2` first).

## 7. Fast iteration loop (use it)

- **`eval/bench.py`** — wind tunnel: 0.3–0.7 s/eval with `--device cuda:0`,
  no sim. `--sensitivity` = on-distribution per-input |Δplan|. Gate: bench
  before every sim run; sim (`--workers N`) only to confirm.
- `eval/replay_probe.py` — single-episode fidelity table (self-describing
  about proprio/ckpt version).
- `eval/lang_probe.py` — language responsiveness (verdict: RESPONSIVE ✓).
- `eval/record_mp4.py` — dual-cam (3rd person | wrist) MP4s, `--res 128`
  cheap, needs the osmesa prefix. `eval/rollout_video.py` — PNG montage +
  action stats.
- Trainer extras: `--load-stage-a X --resume-stage-b` continues planner+TQSA
  from a stage-B ckpt; `--stage-b-patience N` = val + early stop + best-keep.

## 8. Gotchas that already burned us once

- Always pair a checkpoint with ITS norm_stats (v7 ⇒ `data/libero_v7/...`).
- Never `rm -rf checkpoints` + `git add -A` (deleted box checkpoints once).
- npz tmp files must end `.npz` (numpy appends it — atomic-replace bug).
- `_load_relaxed` (eval/policy.py) loads old checkpoints across arch growth
  (prefix-copies grown tables, warns loudly, new heads at init).
- Mock tests: CPU-only, no network, no cv2 — keep it that way (149 passing).
- LIBERO downloader flag is `--download-dir` (dash), `hf download` not
  `huggingface-cli`.
- Paper trail lives in `paper.md` (world-model result + full diagnosis
  chain); `experiments/tracker.py` exists for durable metrics.

## 9a. What changed this session (Mac-side only; nothing run on the box)

All committed, 187 tests green (was 149).

**Eval harness (`eval/libero_eval.py`) — §0's instrument.**
- `ProcessPoolExecutor` replaces `mp.Pool`: a dead worker now raises
  `BrokenProcessPool` in <1 s instead of hanging the parent forever.
- Heartbeats through the whole silent window: `spawned (pid N)` →
  `staggering Ns` → `building policy...` → `policy ready (Ns)` →
  `N task(s) (Ns)` → per-trial `START`/`DONE`. The telemetry file is now
  opened BEFORE the policy build, so its existence proves the worker started.
- Tasks are enumerated ONCE in the parent and shipped to workers — removes the
  10-way concurrent libero import (candidate 2) by construction.
- Thread caps (`OMP/MKL/OPENBLAS/LP_NUM_THREADS` = cores/workers, plus
  `torch.set_num_threads`) set **in the parent**, because `eval/__init__.py`
  imports torch — setting them inside a worker is too late for libgomp.
  `TORCH_BLAS_PREFER_HIPBLASLT=0` now matches the trainers.
- `--worker-timeout` watchdog (kills + reports + keeps partial results, clearly
  marked partial), `--task-ids` manual sharding, `--stagger` now a flag,
  `faulthandler` + SIGUSR1 stack dumps, and finished shards are scavenged off
  disk when the pool aborts.
- `--workers` works under `--mock-env`, so the parallel path has real tests
  (CPU-only, no sim) for the first time.
- **`--heads-device`**: `--device cuda:0` was only ever moving the DETECTOR;
  fusion/drift/TRM/planner/TQSA stayed on CPU and they run every tick while the
  detector runs 1 in 15. Opt-in (default unchanged); verified end-to-end on MPS.

**Planner input gating (§5.3 pruning).** `cfg.planner_inputs` selects which
memory groups are built; a disabled input is ignored by `forward`, so ablating
one is a config change, not a call-site change. `train_batched.py
--planner-drop geometry,pred_box_emb`; the choice rides in the checkpoint cfg.
Defaults are UNCHANGED — the v7 sensitivity read is from libero_object only,
and §5's own plan is to prune after the full-data run. `eval.bench
--sensitivity` gained `next_emb->stale` (a full-magnitude wrong prediction),
because `next_emb->cur` only zeroes the TRM's residual and therefore reads low
by construction — do not prune `next_emb` on the old number alone.

**Waypoint-absolute head (§5.3, the std_ratio lever).** Opt-in
`cfg.waypoint_action` / `--waypoint-weight 1.0`. Predicts metric EEF
displacement (supervised from `eef_pos_chunk`, row/validity masked), and at
eval a proportional move toward that position measured against LIVE proprio
replaces the regressed translation dims. Gain fitted by
`preprocess/fit_waypoint_gain.py` → `waypoint_stats.json` (pair it with its
checkpoint like norm_stats). `eval.bench` reports `wp_std_ratio` / `wp_mae_mm`.
See DESIGN.md for the full contract. UNTRAINED and UNMEASURED — the next
retrain is what tells us whether positions really regress with less shrinkage
than actions.

## 9. Where the headline stands

Validated + publishable now: world model beats persistence under the
deployment-matched protocol (thin but honest +1.7% on object; +11–13% on the
older mix), the perception-rate-decoupling JEPA loop, and a fully-instrumented
diagnosis chain (probe → fix → measured recovery: 0.12 → 0.37 std_ratio,
0.73 → 0.93 grip). Missing for the second headline: a nonzero closed-loop
`mean_success`. That number is one working parallel eval away — start at §0.
