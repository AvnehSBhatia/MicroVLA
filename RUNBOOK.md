# RUNBOOK — every command, in order

Flags verified against the code at the commit that added this file. `handoff.md`
is live state and `paper.md` is the evidence trail; this is the mechanics.

Two machines. **Mac** = dev + tests, `.venv` has torch/numpy/pytest only.
**Box** = MI300X/ROCm, `/root/MicroVLA`, `source /root/eval_venv/bin/activate`.
The box is SHARED with other users' jobs — wall-clock numbers vary by 5x and are
not hardware claims.

**SIGTERM is ignored by default in every CLI here.** The host reaps jobs from
outside the container (exit 143, no dmesg entry, no visible reaper), so every
entry point installs `microvla/utils/signals.py::ignore_sigterm` at startup.
Consequences: `kill <pid>` does nothing — use `kill -9`; Ctrl-C still works;
`MICROVLA_ALLOW_SIGTERM=1` opts out. If a process still dies with **exit 137**
the reaper escalated to SIGKILL, and `train_batched.py --resume-stage-a` (which
banks progress every epoch) is the remaining defence.

Real-LIBERO commands need this prefix (EGL is broken in the container):

```bash
export PFX="PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa PYTHONPATH=/root/LIBERO"
```

---

## 0. Mac — tests. Run before every commit.

```bash
.venv/bin/python -m pytest tests -q                      # CPU-only, mock-only, no network/cv2
.venv/bin/python -m pytest tests/test_waypoint.py -q     # one file
.venv/bin/python -m pytest tests/test_shapes.py -k Planner -q
.venv/bin/python -m microvla.utils.param_audit           # 9M cap + per-module caps
.venv/bin/python -m eval.bench --checkpoint none --synthetic 30      # harness smoke, no data
.venv/bin/python -m eval.libero_eval --mock-env --suite s --n-trials 1 \
  --max-steps 10 --checkpoint none --workers 2 --stagger 0           # parallel path, no sim
.venv/bin/python train/train_planner.py --epochs 2 --episodes 4      # smoke train
.venv/bin/python -m microvla.planner.chrono_planner      # module self-test
python TRM.py                                            # TRM self-test
```

---

## 1. Box — data

`--camera` is REQUIRED. It must be `eye_in_hand_rgb`, the view
`eval/libero_eval.py` reads (`robot0_eye_in_hand_image`). Baking `agentview_rgb`
trains the policy on a viewpoint it never sees at deployment — see paper.md §4f
for what that cost. Rotation follows the camera automatically; the bake logs
`baking camera=... rotate_180=...`, so read that line.

**ONE SUITE AT A TIME.** `preprocess/libero.py` globs its root recursively, so
having all three raw suites resident is ~13 GB against a 10 GB data budget.

```bash
for S in libero_object libero_spatial libero_goal; do
  yes n | python /root/LIBERO/benchmark_scripts/download_libero_datasets.py \
    --datasets $S --download-dir /root/libero_raw      # yes n = use HF, don't overwrite
  ls /root/libero_raw/$S/*.hdf5 | wc -l                # expect ~10; 0 => download failed
  python -m preprocess.libero /root/libero_raw/$S data/${S}_wrist \
    --camera eye_in_hand_rgb --device cuda:0
  ls data/${S}_wrist/*.npz | wc -l                     # expect ~500; a bake CAN die partway
  rm -rf /root/libero_raw/$S                           # BEFORE the next download
done
```

One shared normalizer across the per-suite dirs (skip for a single dir):

```bash
python -m preprocess.unify_norm_stats \
  --data-dir data/libero_object_wrist \
  --data-dir data/libero_spatial_wrist \
  --data-dir data/libero_goal_wrist
```

Verify the bake is really the wrist view — the check that would have caught §4f:

```bash
python -c "
import numpy as np, glob
from PIL import Image
f=sorted(glob.glob('data/libero_object_wrist/*.npz'))[0]
with np.load(f) as z: Image.fromarray(z['wrist_frames'][0]).save('/tmp/baked.png')
print(f, z['wrist_frames'].shape if False else '')"
env $PFX python -m eval.env_smoke --suite libero_object --task 0
# compare /tmp/baked.png with eval_results/env_smoke_wrist.png — same view, same way up
```

Actuation gain for the waypoint head (data only, no checkpoint; pair it with the
checkpoint like norm_stats):

```bash
python -m preprocess.fit_waypoint_gain data/libero_object_wrist \
  data/libero_spatial_wrist data/libero_goal_wrist
# read the per-axis R2: below 0.5 means that axis does not respond linearly
```

Other data tooling: `preprocess/renorm_symmetric.py --data-dir D` (retrofit an
old asymmetric bake), `preprocess/patch_proprio.py` (add proprio without a YOLO
re-bake), `preprocess/shard_pipeline.py` (BudgetGuard download→convert→delete).

---

## 2. Box — train

Validation is INTERNAL: `--val-frac 0.05` holds out a split, stage A early-stops
on it (`--patience`, `--lr-patience`), stage B early-stops on it when
`--stage-b-patience > 0` and keeps the best checkpoint. There is no separate val
command. Stage B prints `val bc X wp Y` — `bc` is the same quantity in every
arm; the waypoint term is reported beside it, never folded in.

Always pass `--tag`, or the run overwrites untagged `full_stageA.pt` /
`full_stageB.pt`.

### Both stages in one go

```bash
LIB="--data-dir data/libero_object_wrist --data-dir data/libero_spatial_wrist \
     --data-dir data/libero_goal_wrist"

TORCH_BLAS_PREFER_HIPBLASLT=0 python train/train_batched.py \
  --data-dir data/bridge $LIB \
  --device cuda --batch-size 64 --lr 5e-4 --max-vram-gb 50 \
  --stage-a-epochs 40 --warmup-epochs 4 --max-horizon 6 \
  --patience 6 --lr-patience 2 --min-delta 1e-4 \
  --stage-b-epochs 40 --stage-b-patience 4 --dream-frac 0.25 \
  --waypoint-weight 1.0 --tag wristwp
```

`--patience 6 --lr-patience 2` matters: with `--patience 3` the LR halving lands
on the same epoch the early stop fires, so the schedule never gets to act.

### Stage A only (world model), then reuse it

```bash
# ... same flags ... --stage-b-epochs 0 --tag wrist
cp checkpoints/full_stageA_wrist.pt checkpoints/full_stageA_wrist_keep.pt
```

### Stage B only — this is how you A/B cheaply (~15 min each)

Both levers are stage-B-only, so every arm shares one frozen world model.
Do NOT add `--resume-stage-b`: it does a strict `load_state_dict` into the
planner, and these flags change the planner's architecture.

```bash
COMMON="--data-dir data/bridge $LIB --device cuda --batch-size 64 --lr 5e-4 \
  --max-vram-gb 50 --load-stage-a checkpoints/full_stageA_wrist.pt \
  --stage-b-epochs 40 --stage-b-patience 4 --dream-frac 0.25"

# baseline
TORCH_BLAS_PREFER_HIPBLASLT=0 python train/train_batched.py $COMMON --tag base
# waypoint auxiliary — the magnitude lever (paper.md §4c/§4d)
TORCH_BLAS_PREFER_HIPBLASLT=0 python train/train_batched.py $COMMON \
  --waypoint-weight 1.0 --tag wp
# trainable spatial perception. First epoch runs the frozen backbone ONCE over
# every framed timestep (~19 min, ~8 GB RAM cached); later epochs are free.
TORCH_BLAS_PREFER_HIPBLASLT=0 python train/train_batched.py $COMMON --tqsa --tag tqsa
# LIBERO only — bridge is ~75% of episodes with no proprio and no frames
TORCH_BLAS_PREFER_HIPBLASLT=0 python train/train_batched.py \
  $LIB --device cuda --batch-size 64 --lr 5e-4 --max-vram-gb 50 \
  --load-stage-a checkpoints/full_stageA_wrist.pt \
  --stage-b-epochs 40 --stage-b-patience 4 --dream-frac 0.25 \
  --waypoint-weight 1.0 --tag wp_liberoonly
# ablate a planner input (evidence: eval.bench --sensitivity, on FULL data)
TORCH_BLAS_PREFER_HIPBLASLT=0 python train/train_batched.py $COMMON \
  --planner-drop geometry,pred_box_emb --tag pruned
```

Continue a stage-B run (planner + TQSA) from a stage-B checkpoint — the one
place `--resume-stage-b` is correct, because the architecture is unchanged:

```bash
TORCH_BLAS_PREFER_HIPBLASLT=0 python train/train_batched.py $COMMON \
  --load-stage-a checkpoints/full_stageB_wp.pt --resume-stage-b --tag wp2
```

Other knobs: `--unfreeze-trm` (trains the whole TRM at 0.1x LR with a
world-model auxiliary; never yet used), `--ablate-grounding` (E7 frame-only),
`--no-cache-spatial`, `--trm-d`, `--row0-weight`, `--smooth-weight`,
`--planner-input-dropout`, `--drift-dropout`, `--box-loss-weight`.

Mac equivalent (batch=1, slow — the drift encoder is stateful):
`python train/train_full.py --data-dir data/libero_v7 --device mps`

---

## 3. Box — test

### 3a. Bench: the gate. Never spend a sim run before this passes.

```bash
python -m eval.bench --checkpoint checkpoints/full_stageB_wp.pt \
  --data-dir data/libero_object_wrist --sensitivity --device cuda:0 \
  --out eval_results/bench_wp.json
```

Add `--tqsa` for a TQSA-trained checkpoint, or its numbers are measured with
~27% of the planner's memory tokens withheld (paper.md §0). Bench SAYS SO when
the checkpoint has TQSA weights and the flag is absent.

Read:

| metric | meaning |
|---|---|
| `std_ratio` | emitted-action vigor / demo vigor. ~1.0 healthy, ~0.1 collapsed |
| `wp_std_ratio` / `wp_mae_mm` | the same for the waypoint head, in metres |
| `corr` | per-dim direction agreement |
| `grip_acc` | per-step gripper agreement — HIGH EVEN IF the close TIMING is wrong |
| `wm_margin` | world-model H-step rollout vs persistence; > 0 = it predicts real dynamics |
| sensitivity | mean \|Δplan\| per withheld input — which inputs the policy actually uses |

`geometry` near zero with no TQSA present means grounding is not reaching the
planner. Read `next_emb->stale` (full-magnitude wrong prediction) alongside
`next_emb->cur` (which only zeroes the TRM residual and reads low by
construction).

### 3b. Closed-loop LIBERO: the real number

```bash
# canary first — serial, short. Must print every heartbeat.
env $PFX python -m eval.libero_eval --suite libero_object --n-trials 1 \
  --max-steps 100 --checkpoint checkpoints/full_stageB_wp.pt \
  --norm-stats data/libero_object_wrist/norm_stats.json \
  --waypoint-stats data/libero_object_wrist/waypoint_stats.json \
  --device cuda:0 --heads-device cuda:0 --workers 1
# success=False at 100 steps is BY DESIGN — 5 s of robot time, tasks need 150-300.

# the number
env $PFX python -m eval.libero_eval --suite libero_object --n-trials 20 \
  --max-steps 300 --checkpoint checkpoints/full_stageB_wp.pt \
  --norm-stats data/libero_object_wrist/norm_stats.json \
  --waypoint-stats data/libero_object_wrist/waypoint_stats.json \
  --device cuda:0 --heads-device cuda:0 \
  --workers 5 --stagger 10 --worker-timeout 3600
# repeat with --suite libero_spatial and --suite libero_goal
```

`--heads-device cuda:0` is worth **16x** (3.75 -> 0.23 s/step): the heads run
every tick, the detector 1 in 15, and `--device` only ever moved the detector.

Pair each checkpoint with ITS OWN `norm_stats.json` and `waypoint_stats.json`.

Ablations and fallbacks:

```bash
--waypoint-no-brake        # do not scale the waypoint command by corrector trust
--task-ids 0,3,7           # manual sharding; N single-task processes is the
                           # fallback immune to in-process concurrency hazards
--perception-period 1      # E4 perception-rate sweep knob
--mock-env                 # no sim at all; works with --workers
```

If a worker stalls: every worker prints its pid, `kill -USR1 <pid>` dumps its
stack, and a stuck worker auto-dumps every 600 s. `--worker-timeout` kills and
reports, keeping partial results clearly marked partial.

### 3c. Diagnosis — what aggregate scores cannot see

Both closed-loop root causes this project has found (actuator units, camera
mismatch) were invisible to every aggregate metric and visible in per-step
telemetry and video. Use these BEFORE theorising.

```bash
# per-step telemetry: is the arm commanded to move, does the gripper commit,
# is the detector finding the object?
python -c "
import json,glob,os,numpy as np
f=max(glob.glob('eval_results/*telemetry.jsonl'),key=os.path.getmtime)
r=[json.loads(l) for l in open(f)]
a=np.array([x['action'] for x in r]); g=a[:,6]
w=np.array([x['waypoint_cmd'] for x in r if x.get('waypoint_cmd')])
e=np.array([x['eef'] for x in r if x.get('eef')])
t=np.array([x['trust'] for x in r])
sc=[x['src_conf'] for x in r if 'src_conf' in x]
cl=np.flatnonzero(g>0)
print('steps',len(r),'| gripper closed',int((g>0).sum()),'first close at',cl[0] if cl.size else 'NEVER')
if w.size: print('waypoint |cmd| mean',w.__abs__().mean().round(4),
                 '| clipped',round(100*float((abs(w)>=0.999).mean()),1),'% of steps')
if e.size: print('eef z: start',e[0,2].round(3),'min',e[:,2].min().round(3),'max',e[:,2].max().round(3))
print('trust mean',t.mean().round(3),'| below brake 0.5:',round(100*float((t<0.5).mean()),1),'%')
if sc: print('src detection conf: mean',round(float(np.mean(sc)),3),'| missed',
             round(100*float(np.mean(np.array(sc)==0)),1),'% of real ticks')"

# watch it. dual-cam MP4; --waypoint-stats is REQUIRED or you film a different policy
env $PFX python -m eval.record_mp4 --suite libero_object --n-videos 2 \
  --max-steps 300 --checkpoint checkpoints/full_stageB_wp.pt \
  --norm-stats data/libero_object_wrist/norm_stats.json \
  --waypoint-stats data/libero_object_wrist/waypoint_stats.json \
  --device cuda:0 --heads-device cuda:0 --res 128

# single-episode teacher-forced fidelity table
python -m eval.replay_probe --checkpoint checkpoints/full_stageB_wp.pt \
  --episode data/libero_object_wrist/<episode>.npz --device cuda:0
# does the policy respond to language at all?
python -m eval.lang_probe --checkpoint checkpoints/full_stageB_wp.pt \
  --norm-stats data/libero_object_wrist/norm_stats.json --device cuda:0
# bare env, no policy — highest-risk install step
env $PFX python -m eval.env_smoke --suite libero_object --task 0
# PNG montage + action stats
env $PFX python -m eval.rollout_video --suite libero_object --task 0 \
  --checkpoint checkpoints/full_stageB_wp.pt \
  --norm-stats data/libero_object_wrist/norm_stats.json
# perception-rate x baseline grid (E4)
env $PFX python -m eval.sweep --suite libero_object --n-trials 5 \
  --periods 1,5,15 --device cuda:0
# per-module scorecard on a val split
python -m eval.scorecard --checkpoint checkpoints/full_stageB_wp.pt \
  --data-dir data/libero_object_wrist --device cuda:0
```

---

## 4. Record it

```bash
python -m experiments.tracker report        # regenerate results/RESULTS.md
```

Numbers go in `results/metrics.jsonl` via `experiments.tracker.log(...)` and the
narrative in `paper.md`. Every number quoted in the paper should be traceable to
a record with its git SHA.
