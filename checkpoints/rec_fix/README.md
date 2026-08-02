# rec_fix — the checkpoint behind the first completed pick-and-places, and the reason that is not a policy number

Stage B trained 2026-07-29 on `data/libero_object_grid` (wrist view), tag
`rec_fix`, as the re-run of stage B after defect 24 (recovery-noise proprio
never reached the planner) was fixed. Recipe of record:
`--recovery-noise 0.01 --variance-weight 0.1 --action-token-sampling 0.5`,
v8 architecture, TQSA on, loading `full_stageA_grid10.pt`
(`paper/paper.md` §5l, `DESIGN.md` §v9).

| file | contents |
|---|---|
| `full_stageB_rec_fix.pt` | the whole deployed stack in one file: `EvidenceEncoder` + `HRMBackbone` + `RecursiveTRM` + `RelationalHead` + `TextQueriedSpatialAdapter` + `ChronoQueryPlanner` — 16,583,525 params across 211 tensors, all float32 |
| `norm_stats.json` | action normalizer fitted on `libero_object_grid`, symmetric (`q_low = -q_high` exactly) |
| `night_sighted_results.json` | the only raw closed-loop artifact for this checkpoint that exists anywhere outside the pod. It is a zero. It is here on purpose |

There is **no stage A in this bundle.** Its parent world model,
`full_stageA_grid10.pt`, is not on the machine this bundle was assembled from
and is named in exactly one line of the record (`paper/paper.md` §5l). Unlike
`v8_seed/` and `v8_pod/`, this bundle cannot be used as a warm start and its
lineage cannot be reproduced from what is published here. Stage B is
self-contained for **inference** — no stage-A file is read at load time, and
loading emits zero key-coverage warnings — so everything below still runs.

There is also **no `waypoint_stats.json`**, and that is correct rather than
missing: this checkpoint's embedded config has `waypoint_action=False`.
Passing a waypoint file from another bundle only logs a warning and leaves the
actuator at `None`. Do not copy one in to make the directory look like the
others.

`norm_stats.json` is not optional. A normalizer from another corpus silently
rescales every command.

## Measured, unaided — this is the policy number

`eval.libero_eval` on this checkpoint, no `--ibvs-phase`, no `--tool-phase`,
no grasp offsets, no place-at: `--det-conf 0.02 --render-size 256 --no-brake
--task-ids 0,1,2 --n-trials 3`, completed 2026-07-30.

| metric | value |
|---|---|
| **mean_success** | **0.000** (0/9 trials; 95% CI [0.000, 0.336]) |
| src_detect_rate | 0.719 |
| src_conf_mean | 0.061 |
| grip_close_rate | 0.653 |
| eef_obj_dist_min | 0.132 m |
| eef_obj_dist_final | 0.750 m — it diverges after closest approach |

That is the whole unaided result. It is also every other checkpoint's unaided
result: unaided closed-loop success on LIBERO object tasks is 0.000 across
every configuration this project has ever run (n > 300 real evaluations;
`results/IBVS_AUTON_SCORECARD.tsv` is 85 runs / 387 trials and every row is
`succ=0.00`).

Open-loop, from the training record: best `val bc` **0.0881**, `grip` 0.975 at
epoch 20; training stopped at epoch 23 for eval budget, so the patience-6
window from the epoch-20 best was never exhausted — this checkpoint is not
converged. Three other arms beat it on `val bc` (`teacher_bc` 0.0448,
`center_frame` 0.0733, `pregrasp3` 0.0856) and every one of them measured
*worse* on unaided proximity. Ranking on `val bc` in this project picks the
worse deployed policy. There is no `bench_*.json` for this checkpoint; the
0.0881 rests on the prose training table, not on an artifact shipped here.

## The numbers people will want to quote, and what they actually are

On the **assisted / calibrated** track — `--ibvs-phase`, where a hand-tuned
state machine with zero learned parameters owns the action — this checkpoint
is the one that was loaded when the project's first completed pick-and-places
happened:

| task | best run | mean_success | n | 95% CI |
|---|---|---|---|---|
| alphabet soup (t0) | `soup_v1` — per-object constants, no verify, no mask | 0.750 | 8 | [0.349, 0.968] |
| cream cheese (t1) | `handeye_v5cream` — align+probe+jaw fix+gate-verify, max-steps 600 | 0.300 | 10 | [0.067, 0.652] |
| salad dressing (t2) | v1–v3 | 0.000 | 8×3 | [0.000, 0.336] |

**These are not MicroVLA policy success and may not be cited as such.** That
is a binding rule of this project, written before these runs existed
(`paper/paper.md` §"What this study does not show"): "Phased IBVS and
`--tool-phase` are diagnostic / assisted controllers on top of `rec_fix`. They
may not be cited as MicroVLA policy success." Citing them as policy success is
the same class of defect as reporting the `--mock-env` 1.000.

## How much of 0.300 / 0.750 did these weights earn? Measured: none of it

Under `--ibvs-phase`, `eval/policy.py` replaces the emitted action wholesale
with `PhasedIBVS.step(...)`, and `eval/ibvs_phase.py` builds that action from
`np.zeros(action_dim)`. The machine's only inputs are the frozen YOLO-World
detections, proprio, and frozen CLIP text embeddings. There is no channel from
these weights to the environment.

Measured on the deterministic mock path with the exact winning flags: the
emitted actions are **bit-identical (max abs difference 0.0)** between this
checkpoint, `--checkpoint none` (untrained modules + `MockTRM`), and
`v8_pod/full_stageB_v8_s0.pt`. Positive control, same three checkpoints with
`--ibvs-phase` removed and nothing else changed: they differ (maxdiff 0.859
and 0.360) and episode lengths diverge. The zero is a property of the assisted
path, not of the harness.

So: the assisted track has no power to rank checkpoints, and the ablation that
would attribute 0.300/0.750 to these weights in the real simulator **was never
run**. Anyone loading this bundle expecting the assisted numbers to be a
property of the weights will be wrong.

Two further honesty notes on those numbers. The "+0.10 from gate-verify"
(0.200 → 0.300) is one extra success out of ten; Fisher exact 2/10 vs 3/10
gives p = 1.0. Sibling arms of the same machine scored 0/10 (cream v6) and 0/9
(soup v2), with CIs that overlap the winners. And the raw telemetry for every
nonzero run above lives on the training pod — it is not in this repository,
and it was not verified against an artifact when this bundle was assembled.
The one measured artifact shipped here is the 0.000.

## What the weights say about themselves

* `drift.hrm.gain_head.weight` (3×256) and `.bias` are **all zero** — the only
  all-zero tensors in the file. The HRM's learned per-axis control gains never
  received an effective gradient. This is not specific to `rec_fix`: they are
  all-zero in the published `v8_seed` and `v8_pod` checkpoints too.
* Planner input ablation puts ~97% of the plan's dependence on the
  `RelationalHead` tokens; `fused`, `spatial`, `pred_box_emb` and `geometry`
  measure exactly 0.0000 impact. Four of nine declared planner inputs are dead.
* The text path is effectively a task-ID lookup: `tqsa.t_proj` has effective
  rank 28.5 of 128.
* Action magnitude shrink survives here. LIBERO's passing band is ~1.0±0.05 of
  demo scale; ground-truth demo actions replayed at 0.8× magnitude solve 0/4.
  No checkpoint in this project has reached that band.
* The file is clean: 0 NaN, 0 Inf, global max |·| = 4.5236, loads under
  `torch.load(..., weights_only=True)`, and contains no epoch, args, optimizer,
  git rev, path or timestamp — only `cfg`, `trm_d` and six state dicts.
  Provenance for the training recipe is documentary, not embedded.

## Run it

Requires `pip install -e ".[perception]"`, a LIBERO/robosuite install, repo
root on `sys.path` (the real TRM is `TRM.py` at root), and the frozen detector
`yolov8s-worldv2.pt` in the working directory — that file is an upstream
Ultralytics artifact and is deliberately not redistributed here; fetch it from
Ultralytics. `proprio` is 10-dimensional; passing 9 raises a shape error on the
first call.

Unaided — this is the configuration whose number is a policy number:

```bash
python -m eval.libero_eval --suite libero_object --task-ids 0,1,2 --n-trials 3 \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake \
  --checkpoint checkpoints/rec_fix/full_stageB_rec_fix.pt \
  --norm-stats checkpoints/rec_fix/norm_stats.json --device cpu
```

Expected: `mean_success 0.000`. Reproducing that is the point.

Assisted (cream cheese, the `handeye_v5cream` config) — reproduces a machine,
not a policy:

```bash
python -m eval.libero_eval --suite libero_object --task-ids 1 --n-trials 10 \
  --max-steps 600 --camera robot0_eye_in_hand_image --render-size 256 \
  --perception-period 2 --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 \
  --source-max-area 0.12 \
  --checkpoint checkpoints/rec_fix/full_stageB_rec_fix.pt \
  --norm-stats checkpoints/rec_fix/norm_stats.json \
  --ibvs-phase --ibvs-gate-verify --ibvs-gain 0.5 --ibvs-sign 1,-1,0 \
  --ibvs-descend -0.4 --ibvs-descend-hyst 0.50 --ibvs-target-uv 0.5,0.60 \
  --ibvs-grasp-offset 0.08,-0.05 --ibvs-close-z 0.01 --ibvs-press 0.2 \
  --ibvs-retry-rise 8 --ibvs-place-at=-0.005,0.257 --ibvs-drop-z 0.18 \
  --device cpu --workers 1
```

`--ibvs-place-at=-0.005,0.257` needs the `=` form (the value leads with a
dash). For alphabet soup use `soup_v1`: **drop** `--ibvs-gate-verify` (it
collapsed soup from 0.750 to 0/9) and swap the object-geometry constants —
`--ibvs-close-z 0.045 --ibvs-gate-z 0.10 --ibvs-approach-z 0.12
--ibvs-grasp-offset 0.09,-0.186 --ibvs-place-at=-0.006,0.260 --ibvs-drop-z
0.25`. All constants were calibrated offline from demo statistics and logged
runs; none were tuned against eval ground truth at runtime.

Three detector knobs the run depends on are **not** stored in the checkpoint
and must be supplied on the CLI: `--det-conf 0.02 --role-disjoint-iou 0.1
--source-max-area 0.12`. The code defaults for the last two are 0.0, so
omitting them silently changes the run.

`data/libero_object_grid/` has no `manifest.json`, so every provenance check
(camera, det_conf, render_size, perception_period, role_disjoint_iou) is
silently skipped — including under `--strict-provenance`. The winning runs had
no corpus/deployment agreement check at all.

## Why this checkpoint is published

Not because it is the best policy. No checkpoint in this project has been
shown better than any other on any closed-loop measurement; unaided success is
0.000 for all of them, `wm_margin` does not discriminate stage-B arms (16 bench
files share the value to 16 digits, and its sign is a camera artifact), and
`val bc` is anti-correlated with unaided proximity across the record.

It is published because it is the architecture of record — the file the v9
`DESIGN.md` documents layer by layer, the only checkpoint carrying both the
TQSA weights and the `wm_delta_proj` planner head that today's deployment path
expects, strict-loading all six modules with no NaN and no missing keys. That
is fitness for deployment and documentary value. It is not a performance
ranking, and this README should not be read as one.
