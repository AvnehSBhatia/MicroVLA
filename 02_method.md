## 2. Method

This section fixes what was built, how it was trained, and — at equal length, deliberately —
what each instrument can and cannot see. Which instrument produced a number turns out to
determine whether the number can be read at all (§2.7), so the instruments are part of the
method, not an appendix to it.

All dimensions below are read from `microvla/config.py` (`MicroVLAConfig`), the single source
of truth for every quantity that crosses a module boundary.

### 2.1 The deployment loop

**Text, once per task.** `parse_command` splits the instruction into an ordered
(verb, source, target) triple; CLIP embeddings for the three phrases are harvested from
YOLO-World's own text tower at `set_classes` time, giving `3 x 512` text tokens
(`n_text_tokens=3`, `text_dim=512`). There is no separate text model, and nothing text-side
is resident on-device after the task is set.

**Perception, at 2 Hz.** `tick_hz=30.0`, `real_frame_hz=2.0`, so every 15th tick is real and
`dream_ticks_per_real=14`. On a real tick the frozen YOLO-World-S detector supplies
`frame_emb [512]` (GAP of the hooked SPPF map) plus, per role, the best box's ROIAlign
embedding `[512]`, center `[2]` and confidence. Every embedding is **standardized** (zero
mean, unit std per vector) at the perception boundary; that canonical space is what fusion,
drift, the TRM and the corrector all assume, and the loop re-standardizes corrected dream
latents before feeding them back.

**Dream ticks.** On the other 14 ticks the frame token is the corrected TRM prediction and
the boxes are those of the last real tick, at weight `confidence * staleness_decay**k`
(`staleness_decay=0.9`, k = dream ticks since the last real frame). A role whose detector
missed on a real tick holds its last box at `miss_decay**age` (`miss_decay=0.7`) rather than
resetting to a (0.5, 0.5) fallback.

**Fusion** (`SlotResonanceFusion`): 8 role-tagged tokens at `d_model=384` —
`[command | source | target | frame | source-box | target-box | geometry | last action]`.
FiLM from the command embedding modulates the three visual tokens; the geometry token is a
Fourier encoding (`n_fourier=16`) of both centers and their displacement. 32 learned slots
cross-attend over the tokens for `n_fusion_blocks=3` rounds, then a shared per-slot head
emits `fused [32, 5]` (`fused_rows=32`, `fused_cols=5`).

**Evidence fade is one shared path.** A per-role `box_weight` in [0, 1] scales the box tokens
and (by its mean) the geometry token; dream staleness, detector misses (weight 0) and the
train-time `modality_dropout=0.3` fade all move the *same* weight. There is no binary zeroing
and no dream flag. The action token is never faded. This is a design commitment, not a
measured one: the ablation that would test it (fade vs binary zeroing vs no dropout) was
never run.

**Drift** (`AnchoredDriftEncoder`): anchored on the episode's first real frame. Per real tick
it builds one token per reference — the anchor plus each lag in `drift_horizons=(1, 2, 4, 8)`
real frames (0.5–4 s at 2 Hz) against a `context_window=8` deque of real-frame embeddings —
attention-pools them, gates, and accumulates in a `GRUCell(256, 256)`, emitting
`state_delta [256]` (`state_dim=256`). It steps on **real ticks only**; the loop holds its
code across dreams. The first call after `reset()` returns an exactly-zero code without
stepping, and all runtime state lives in plain attributes so no checkpoint carries episode
state.

**World model** (`TRM.py::RecursiveTRM`): weight-tied recursion with FiLM drift conditioning,
**9,968,976 parameters** at `d=1024` (counted by instantiating the module).
`forward(fused [B,32,5], state_delta [B,256], current_emb [B,512], context [B,K,512]) ->
next_emb [B,512]` under the **residual convention** (`current_emb + delta`). `forward_full`
additionally returns `next_box [512]`, `msg [32]` and the pooled belief state
`latent [1024]` (`wm_latent_dim=1024`). The module is stateless; the context window of recent
tick latents is caller-owned. Inference runs one refinement pass; deep supervision is
training-only.

**Planner** (`ChronoQueryPlanner`): builds memory tokens from the groups listed in
`cfg.planner_inputs` (`next_emb`, `current_emb`, `fused`, `state_delta`, `pred_box_emb`,
`geometry`, `proprio`, `spatial`, `wm_msg`, `wm_latent`); an input not listed gets no
projection and its argument is ignored, so ablating one is a config change and every caller
keeps passing everything it has. `plan_steps=5` learned time queries plus a fixed sinusoidal
step encoding cross-attend the memory for `n_planner_blocks=3` rounds at `d_plan=256`; a
per-step head predicts deltas and `plan = tanh(cumsum(deltas))` gives `[5, 7]` in [-1, 1] —
**rows are 5 timesteps, columns are `num_servos=7`**. The gripper column is a hard +/-1 from
a separate logit. Only row 0 is ever executed: the loop replans every tick.

**Innovation corrector** (no learned parameters). At each real tick the innovation
`e = real - pred` updates an EMA accumulator `c <- 0.7c + 0.3e` (`correction_beta=0.7`);
dream tick k applies `pred + correction_decay**k * c` (`correction_decay=0.9`). Trust is a
self-calibrating *ratio*, not an absolute cosine: `err_bar` is an EMA of `||e||`,
`ratio = ||e|| / err_bar`, and `tau = exp(-0.5 * ratio^2 * trust_temperature / 4)` with
`trust_temperature=4.0`, so a typical-sized error gives `tau ~= 0.61` and the TRM is judged
against its own recent accuracy. Trust is **action-space aware** (`cfg.action_space`): for
`"delta"` actions (LIBERO, Bridge — zero means no motion) low trust *brakes*,
`scale = min(1, tau / brake_trust)` with `brake_trust=0.5`, full magnitude while tracking and
linear attenuation to a stop below the threshold; for `"absolute"` PWM targets (the Pi rig —
zero is servo mid-range) low trust hold-blends toward the previous plan and never scales
toward zero. Every result in this paper is in `"delta"` mode.

### 2.2 Budgets and the parameter ledger

Trainable heads, reproduced by `python -m microvla.utils.param_audit` at the time of writing:

| module | params | per-module cap |
|---|---|---|
| `SlotResonanceFusion` | 4,460,165 | 5,000,000 |
| `AnchoredDriftEncoder` | 724,993 | 1,500,000 |
| `ChronoQueryPlanner` | 1,803,527 (1,804,298 with the waypoint head) | 2,500,000 |
| **trainable total** | **6,988,685** | **9,000,000** (`cfg.trainable_param_budget`) |

Not in that budget: the frozen YOLO-World-S detector (13,000,000, a documented constant in
`param_audit`, not a measurement here) and the TRM slot (reserved 10,000,000; the trained
model is 9,968,976). The CLIP text tower runs once per task and is not resident. The audit's
deployed ledger sums the three *caps* — 13,000,000 frozen + 10,000,000 reserved +
9,000,000 trainable — to a headline **32,000,000**; paper.md quotes the deployed stack as
"~30M deployed params", which is the same ledger with the trained totals substituted. Quote
whichever, but not as if they were the same measurement. The budget is enforced by
`tests/test_param_budget.py` and the audit, and held across every change in this document.

**Disk: 10 GB total, ever**, including transient download and extraction state. All data
tooling runs download -> convert -> delete under a `BudgetGuard`
(`preprocess/shard_pipeline.py`); episodes are stored as compressed `.npz`.

### 2.3 Corpus

LIBERO suites were downloaded, converted and deleted one at a time — all three resident at
once is ~10-12 GB, the entire budget. Reconstructed sizes: `libero_object` 7.44 GB,
`libero_spatial` 6.24 GB. `preprocess/unify_norm_stats.py` rescales the per-suite normalizers
onto the per-dim **max** of their symmetric scales (max, never mean, so nothing clips) and
writes one shared `norm_stats.json`. Actions are symmetrically normalized so that **0 means
zero motion**; the original asymmetric quantile mapping made a neutral output a constant
drift command.

The v7.2 training corpus is **6023 train / 316 val episodes across 60 length buckets** keyed
`(T, has_frames)`. **23 of 60 buckets carry `wrist_frames`** — the ~1500 LIBERO episodes; the
~4500 Bridge episodes are frameless and proprio-less (validity flag 0) and train planner-only
with `spatial=None`.

One corpus property must travel with every number: the v7.2 three-suite bake was made from
`agentview_rgb` (rotated 180 degrees) while eval reads `robot0_eye_in_hand_image`, and the
npz key is named `wrist_frames` either way. That silent train/deploy mismatch invalidated one
three-suite bake, one stage A, three stage-B trainings, four bench runs and two closed-loop
evals, and was visible in no aggregate metric. Every measurement below is therefore tagged
with the corpus and camera it came from.

### 2.4 Two-stage training

**Stage A — world model** (`train/train_batched.py`, fusion + drift + TRM jointly). Training
unrolls the *deployment* dream path: the prediction is re-standardized and fed back through
fusion with the boxes held at `staleness_decay**k` and the executed action token advanced,
for H steps. Per-step loss is `spec_loss = (1 - cos) + 0.5 * MSE` on raw standardized vectors
— no LayerNorm inside the loss, because the space is already canonical and the loss must stay
scale-honest — discounted by `gamma**(k-1)` with `gamma=0.9` and normalized by the discount
sum. H ramps 1 -> `--max-horizon` over `--warmup-epochs` then holds (6 and 4 in the runs
reported here). An optional box term (`--box-loss-weight 0.5`) supervises `next_box`; it is
kept out of the validation objective so val stays frame-only.

The baseline is **persistence** — predict no change — evaluated under the identical
discounted normalization, so the reported val figure is always the pair (model, persistence)
and their relative margin, `wm_margin`. `ReduceLROnPlateau` (factor 0.5, `--lr-patience 2`)
fires before early stopping (`--patience 6`), and episodes are batched by exact length bucket
so no padding or masking is needed.

**Stage B — behaviour cloning through the frozen world model.** Fusion and drift are frozen;
the TRM core is frozen with only `msg_head` trainable, so the planner's gradient shapes the
32-d belief message while the world model stays provably intact (`--unfreeze-trm` trains the
whole TRM at 0.1x LR with a world-model auxiliary rollout loss). The loss is weighted pose
MSE + gripper BCE + a smoothness (jerk) penalty (`--smooth-weight 0.05`), with
`--row0-weight 2.0` on the only row deployment executes (mean-1 normalized, so it is not a
disguised learning-rate change), an optional pre-grasp step weighting, and
`--waypoint-weight W` times the displacement loss of §2.5. `--dream-frac` trains a fraction
of steps in the dream regime — the regime the planner actually occupies on 14 of every 15
deployment ticks (`--dream-frac 0.25` in the arms of §3). Planner input dropout is applied as
a **graded per-sample fade of a group's projected
content**, not deletion: deleting `fused` removes 32 of ~68 memory tokens and puts the
attention softmax in a regime deployment never occupies.

**Early stopping, and the confound it caused.** Stage B halts on a validation quantity
compared against an **absolute** `--min-delta` of 1e-4. Until 2026-07-26 that quantity was
`val_bc + waypoint_weight * val_wp`. `val_wp` is ~10x larger under `--waypoint-long`
(0.5-2.5 s displacement targets against 0.05-0.20 s), and a larger term carries larger noise,
so a fixed absolute threshold is cleared less often and long-horizon arms ran a harsher
effective patience. Arms early-stopped anywhere from **8 to 28 epochs**, and across n = 9
mixed-corpus arms every bench metric is a monotone function of epochs survived:

| metric | Pearson vs epochs | Spearman |
|---|---|---|
| `wp_std_ratio` | 0.891 | 0.924 |
| `grip_acc` | 0.901 | 0.907 |
| `std_ratio` | 0.770 | 0.866 |
| `pose_mae` | -0.793 | -0.865 |
| `corr` | 0.915 | 0.840 |

*Provenance note.* The correlation table above, the seed folds and pooled spread quoted in
§2.7, and the 50-trial closed-loop run all come from the 12-arm overnight batch of paper.md
§4m. That batch has no backing record in `results/metrics.jsonl` — the store ends before the
batch started — and the `results/PAPER_TABLE.md` it cites is absent from the repo. The
figures are quoted from paper.md; their artifacts should be recovered from the training box
before publication. [UNVERIFIED against this repo's artifacts: the entire §4m batch.]

The protocol is now `--stage-b-select {bc,total}`, defaulting to `bc` — the behaviour-cloning
term alone, the only one on a scale shared by every arm — plus `--stage-b-min-epochs` to
floor the run length; `tests/test_stage_b_selection.py` pins both, and `total` is retained
solely to reproduce the invalidated batch. Consequently **no result in this paper is a
comparison between separately trained arms.**

### 2.5 Target parameterization: displacement instead of action

The BC head regresses normalized 7-dim action commands. The alternative head regresses
**metric end-effector displacement**: `wp_disp_head = Linear(d_plan, 3)` — **771 parameters**
— reading the same features `h`, with the same `tanh(cumsum(.))` structure, emitting
`[5, 3]` in units of `cfg.waypoint_range`. It costs zero parameters at inference when unused
(the planner returns `None`) and it never touches the plan: bench metrics are bit-identical
with the head loaded and dropped.

Supervision (`microvla/utils/waypoint.py`, `train/losses.py::waypoint_loss`):

* **Native spacing.** Row k is supervised against
  `(eef_pos_chunk[k+1] - eef_pos_chunk[0]) / waypoint_range`. The bake carries `plan_steps`
  rows, so the last row has no target: 4 of 5 rows are supervised, always including row 0. At
  LIBERO's 20 Hz control rate, 5 rows span 0.25 s, so a row is 0.05-0.20 s of displacement.
* **Long horizon** (`--waypoint-long`, zero new parameters, no re-bake). Row k becomes
  `traj[t+k+1] - traj[t]` over the **sampled 2 Hz** EEF trajectory already present as the
  leading column of every npz — 0.5-2.5 s of displacement. Two unit companions are mandatory
  and each is a silent train/deploy mismatch if missed: `waypoint_range` must grow (0.15 m
  clamps a real reach; `--waypoint-long` defaults it to 0.5), and `waypoint_row_stride` must
  be `source_hz / real_frame_hz` = 10 for LIBERO. Cost of the tail: row k is unsupervised for
  the last k+1 timesteps (~8% of rows at T~30), masked per (timestep, row).

Both variants also mask samples whose proprio validity flag is 0 — a zero-filled episode
looks exactly like "the arm never moves", which is the collapse this head exists to fix.

**Actuation** (`WaypointActuator`, eval-side because it needs raw action units). The absolute
target is `measured EEF + predicted displacement` at the servoed row, re-anchored every tick,
and the command is `gain_scale * (target - eef) / (gain * steps_left)` clipped to +/-1. It
replaces only the 3 translation dims; orientation and gripper stay on the BC head, and the
delta-mode trust brake still scales the result. `gain` (metres of EEF travel per unit action
per control step) is fitted per axis from demo (action, delta-eef) pairs by
`preprocess/fit_waypoint_gain.py` over **82,844 pairs from 1500 episodes, 0 skipped**:

| axis | gain (m per unit action per step) | R^2 |
|---|---|---|
| x | 0.01056 | 0.870 |
| y | 0.01200 | 0.938 |
| z | 0.01085 | 0.866 |

so LIBERO's OSC translation response is per-axis linear and the inversion is sound. A
full-scale command moves ~1.1 cm/step, and a 5-step chunk spans ~5.5 cm — the scale against
which the head's millimetre-level errors should be judged. The gain file must be paired with
its checkpoint exactly like `norm_stats.json`; a gain fitted under a different action
normalization is meaningless. [UNVERIFIED: §4g restates this fit as "R^2 0.88/0.99/0.94",
which does not match the fit above or the metrics store; the 0.99 figure has no source.]

Two control-law defects were found here and are recorded because the mechanism generalizes:
dividing by `gain` alone over-commands by exactly the horizon h (5 at the default), and
holding the target across a whole perception period idles the arm two thirds of the time at a
15:5 ratio. Both fixed in `e362d2c`.

**Why this pair is the paper's cleanest comparison.** The two heads read the same `feats` in
the same forward pass of the same checkpoint and differ only in what they are supervised
against. Their ratio is therefore invariant to every quantity that differs across training
runs — seed, stop epoch, corpus, camera — which is exactly what §2.7 says the rest of the
measurements are not.

### 2.6 Instruments

**(a) `eval.bench` — open-loop, teacher-forced.** Replays baked episodes through the full
stage-B forward (fusion -> drift -> TRM -> planner) with no simulator. Defaults: 30 episodes,
rollout horizon 6, sensitivity probe on 10 episodes; **every reported metric is the median
over episodes**, with per-episode ranges also recorded. All pose statistics use the first 6 of
7 plan columns; the gripper is scored separately. Definitions, comparing emitted plan row 0 to
the demo action at the same step:

* **`std_ratio`** = median over pose dims of `std_t(emitted_d) / std_t(demo_d)`, dropping dims
  whose demo std is <= 1e-6. It is a **magnitude** statistic, not an error statistic: it
  compares the spread of the policy's commands over an episode to the spread of the
  demonstrator's. **1.0 is healthy** — the policy moves with the demonstrator's vigor; it is
  not bounded above, and > 1 would mean over-driving. **Near 0 is collapse**: a policy emitting
  nearly the same action at every step has `std_ratio` near 0 while its per-step MAE can still
  look reasonable, which is why `pose_mae` alone never caught this failure. The proposed
  mechanism is conditional-mean shrinkage — MSE converges to `E[a | obs]`, whose variance is
  only the *explained* part of the target's, so the fit shrinks in proportion to irreducible
  target noise. (Proposed, not established: no direct measurement of target-noise variance
  exists in this project.) A single reading is an interval rather than a point — the same
  fixed command has produced `std_ratio` anywhere in 0.022-0.245.
* **`pose_mae`** = mean `|emitted - demo|` over pose dims, in normalized action units.
* **`corr`** = mean over pose dims of the Pearson correlation between the emitted and demo
  series (dims with either std <= 1e-6 dropped). Direction without magnitude: a fully
  collapsed policy can retain `corr`.
* **`grip_acc`** = fraction of steps where the sign of the gripper logit matches the demo's.
* **`wm_margin`** = `(persistence_loss - model_loss) / persistence_loss` over H-step rollouts
  anchored at ~4 points per episode, with the model's predictions fed back through fusion at
  staleness-faded weight — **training-protocol-matched**, not the harsher frozen-fusion
  protocol. Positive means the TRM beats "predict no change". It scores stage A alone and
  never touches the planner, so across arms sharing one stage A it is a single number
  replicated, and disagreement between two such rows is a harness bug.
* **`wp_std_ratio` / `wp_mae_mm`** = the same magnitude statistic and the mean absolute error
  in **millimetres** for the displacement head, scored against the supervision it was trained on
  (native or long) and at the row `WaypointActuator` actually servoes
  (`cfg.waypoint_horizon`, clamped to the last supervised row). It needs neither the action
  normalizer nor the fitted gain, which is what makes it comparable to `std_ratio` within one
  forward pass.
* **`--sensitivity`** = on-distribution mean `|delta plan|` when one input is withheld
  (optional groups -> `None`; `next_emb` -> `current_emb` as a no-prediction foil, and ->
  the previous tick's prediction as a full-magnitude *wrong* answer, since the former only
  zeroes the TRM's residual and reads low by construction). **Pose and gripper are reported
  separately**: the gripper column is a hard +/-1, so one flipped decision per step
  contributes exactly `plan_steps*2 / (plan_steps*num_servos)` = 0.2857 to a whole-plan mean
  — the same size as the largest sensitivity ever recorded here (`state_delta` 0.2740).
  Combined-metric readings from before that split are not comparable to pose-split ones.

Reported cost ranges from 0.45 to 1.53 s/eval across the benched arms. **Caveats.** It is
teacher-forced, so it cannot see compounding closed-loop error, and it scores the
*prediction*, so it is structurally blind to actuator defects: the ~5x over-command of §2.5
was invisible to every bench metric and obvious in one line of per-step telemetry.

**(b) The closed-loop LIBERO harness.** `eval/libero_eval.py::run_eval` drives a duck-typed
policy (`reset(instruction)` / `act(frame) -> action`) through every task of a suite for
`--n-trials` seeded episodes, reporting per-task success and `mean_success`; success is the
environment's own check. LIBERO control runs at 20 Hz; `--max-steps` defaults to 300, and a
short cap reports failure by construction (100 steps is 5 s of robot time, and a
pick-and-place needs 150-300). `--perception-period 15` preserves the deployment ratio of one
real tick in 15 (§2.1: `dream_ticks_per_real=14`),
`--camera` defaults to `robot0_eye_in_hand_image`. One JSON telemetry record per env step
(the harness context merged with the policy's own per-tick record) is the diagnostic of
record. `--workers` shards tasks across processes; `--heads-device` moves fusion, TRM and
planner off CPU and is worth **3.75 s/step -> 0.23 s/step (16x)**, because `--device` only ever
moved the detector, which runs 1 tick in 15 while the d=1024 TRM runs on all of them. A
`--mock-env` path (`MockLiberoEnv`) keeps the whole harness runnable with no sim, no network
and no cv2.

The closed-loop result reported in this paper is **`mean_success` 0.000 over 50 trials**
(5 trials x 10 tasks, `libero_object`), all 10 tasks completed, 0 scavenged, no failed
workers. [UNVERIFIED: the results artifact for that specific run is not present in this
repo's `results/metrics.jsonl` or `eval_results/`; the figure is quoted from paper.md §4m and
should be recovered from the training box before publication.] The provenance gap is about
the artifact, not the outcome: paper.md records that the first closed-loop LIBERO evals
"scored 0/10 (object AND spatial)" and that "previous zeros were single small runs", so the
50-trial run is the first zero sampled well enough to state as a result, not the first zero.
[UNVERIFIED: any count of how many earlier real-environment runs scored 0.0 — paper.md gives
no number.] The one closed-loop record that does exist in `results/metrics.jsonl` reports
`mean_success` `null` and is marked VOID (the actuator over-commanded by 5x and clipped), so
it is not a competing measurement.

**(c) Within-checkpoint probes.** Weight- and activation-level measurements on one frozen
checkpoint — how much of fusion's output is determined by box embeddings versus the evidence
weight, how much of the TRM's predicted residual survives zeroing `fused`, the effective rank
of the `msg` channel. These share the property of §2.5's head pair: one checkpoint, one
forward pass, no cross-run comparison.

### 2.7 What each instrument licenses

Every measurement in this paper is tagged with the level at which it was made.

* **L1 — within one forward pass of one checkpoint.** Two heads on shared features; one input
  withheld from a fixed set of weights; a weight-level probe. Nothing that varies across
  training runs can enter the comparison.
* **L2 — within one training run.** A validation curve; a horizon sweep on one checkpoint.
  Valid as a statement about that run.
* **L3 — across checkpoints from separate training runs.** In this project these were not
  interpretable at n = 1. Three independent reasons compound: the stopping-criterion confound
  of §2.4 (Spearman >= 0.84 between every bench metric and epochs survived); run-to-run spread
  at fixed command that pools to mean 0.084, sd 0.097, an **11.1x fold** in `std_ratio`; and
  per-input sensitivity seed folds up to **134x** (`geometry`) and **76x** (`proprio`), which
  makes most single-run sensitivity rankings unreadable. `fused` pose-sensitivity is the
  exception that is stable across seeds. (The spread and fold figures carry the §2.4
  provenance note.)

The practical rule the rest of the paper follows: **report L1 and L2 as measurements, and
report no L3 comparison as a result** — not until a scale-invariant stopping criterion and at
least three seeds per configuration are in place, both of which are now implemented but not
yet run.
