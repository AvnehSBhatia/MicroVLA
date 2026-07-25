# HANDOFF — MicroVLA session state (written 2026-07-25)

Read this top-to-bottom before touching anything. CLAUDE.md + DESIGN.md are the
binding contracts; this file is the *live state*: what's trained, what's
measured, what's mid-flight, and exactly where the last session stopped.

## 0. The one urgent thing (where the last session stopped)

**The 10-worker parallel LIBERO eval hung**: 20 min, all 10 workers holding
~2.5 GB VRAM each, **0% GPU utilization, zero telemetry files created**. The
closed-loop `mean_success` for the v7 checkpoint was therefore NEVER obtained.
An observability patch was committed (`6dbfddc`: per-trial START/DONE prints,
telemetry flush, 5 s/worker startup stagger) but is **unverified on the box** —
it may or may not fix the hang itself (stagger addresses the most likely
cause: 10-way simultaneous CUDA/ultralytics/mujoco context init contention).

Next session, first moves (box, `source /root/eval_venv/bin/activate`):
```bash
git pull
# 1) serial canary — MUST work and print [main] START/DONE lines (~3 min):
PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa PYTHONPATH=/root/LIBERO \
python -m eval.libero_eval --suite libero_object --n-trials 1 --max-steps 100 \
  --checkpoint checkpoints/full_stageB.pt --norm-stats data/libero_v7/norm_stats.json \
  --device cuda:0 --workers 1
# 2) if canary OK, scale gently (5 workers, watch for [wN] prints):
#    same command with --n-trials 2 --max-steps 250 --workers 5
```
If workers still stall silently despite heartbeats: suspect mujoco/osmesa init
under concurrency; try `--workers 2`, or per-worker `MUJOCO_GL` re-export, or
run 10 sequential single-task processes via a shell loop as fallback.

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
  its own input → silently inconsistent normalization. Bake all three suites
  in ONE run into a fresh dir (`data/libero_v7_full`), commands in §5.
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
2. **Full 3-suite bake + full retrain** (~1.5 h bake + ~40 min train):
   ```bash
   for S in libero_object libero_spatial libero_goal; do
     python /root/LIBERO/benchmark_scripts/download_libero_datasets.py --datasets $S --download-dir /root/libero_raw
   done
   python -m preprocess.libero /root/libero_raw data/libero_v7_full --device cuda:0
   rm -rf /root/libero_raw
   TORCH_BLAS_PREFER_HIPBLASLT=0 python train/train_batched.py \
     --data-dir data/bridge --data-dir data/libero_v7_full \
     --device cuda --batch-size 64 --lr 5e-4 --max-vram-gb 50 \
     --stage-a-epochs 30 --warmup-epochs 4 --max-horizon 6 --patience 3 \
     --stage-b-epochs 40 --stage-b-patience 4 --dream-frac 0.25 --tqsa
   python -m eval.bench --checkpoint checkpoints/full_stageB.pt \
     --data-dir data/libero_v7_full --sensitivity --device cuda:0
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

## 9. Where the headline stands

Validated + publishable now: world model beats persistence under the
deployment-matched protocol (thin but honest +1.7% on object; +11–13% on the
older mix), the perception-rate-decoupling JEPA loop, and a fully-instrumented
diagnosis chain (probe → fix → measured recovery: 0.12 → 0.37 std_ratio,
0.73 → 0.93 grip). Missing for the second headline: a nonzero closed-loop
`mean_success`. That number is one working parallel eval away — start at §0.
