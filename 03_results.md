# 3. Results

Results in §3.1–§3.3 carry an evidence tag from §2.5. **[L1]** = measured within a
single forward pass of a single checkpoint. **[L2]** = measured within a single
training run. **[L3]** = measured across checkpoints from separate training runs.
§3.4 and §3.5 are a closed-loop outcome and an infrastructure ledger and are untagged.
§3.3 establishes why the tag is load-bearing: in this project every L3 comparison
examined turned out to be uninterpretable, for a mechanical reason rather than a
statistical one.

Provenance note. The source of record for numbers is `paper.md` and
`results/metrics.jsonl` (100 records, last entry `ts 2026-07-26T08:29:37+00:00`).
The 12-arm overnight batch of §4m ran after that last record, and its artifacts —
`results/PAPER_TABLE.md` and `logs/overnight/` — are not present in this repo. Its
numbers are reproduced verbatim from `paper.md` and are flagged **[§4m, no
artifact in store]** at first use in each subsection. §3.3 rests on §4m
throughout; §3.1 draws its 19-arm ratio range and its two `longh_tqsa` rows from
it, and §3.4 its closed-loop sample size. Every one of those is flagged in place.

---

## 3.1 Target parameterization: displacement regresses less shrunk than action [L1]

`std_ratio` is the predicted per-dim output std divided by the demo action std;
healthy is ~1.0. The original quantification of the collapse was a teacher-forced
replay probe: per-dim output std ~8x smaller than the demo action std (0.03–0.09
against 0.36–0.59) at directional correlation ~0.7 — regression to the conditional
mean, not an optimization failure.

The waypoint head (`wp_disp_head`, `Linear(d_plan, 3)`, **771** parameters at train
time and **zero** at inference) and the BC action head read the *same* `feats` in the
*same* forward pass and differ only in what they are supervised against: metric
end-effector displacement versus normalized action commands. The comparison is
therefore Level 1. Its independence from the plan is verified rather than asserted:
a bench run that accidentally dropped `wp_disp_head` (§4c, fixed in `fb5f5df`)
produced BC metrics bit-identical to the run with the head loaded (§4d).

| checkpoint | camera / corpus | supervision span | action `std_ratio` | `wp_std_ratio` | ratio | `wp_mae` |
|---|---|---|---|---|---|---|
| `full_stageB_wp.pt` ep18 (§4d) | agentview, 3 suites + bridge | 0.05–0.20 s | 0.237 | **0.787** | **3.3x** | 3.0 mm |
| `full_stageB_wristwp.pt` (§4g) | wrist, `libero_v7` + bridge | 0.05–0.20 s | 0.126 | **0.604** | **4.8x** | 4.8 mm |
| long-horizon A rerun (§4l) | wrist, `libero_v7` + bridge | 0.5–2.5 s | 0.245 | **0.799** | **3.3x** | 58.2 mm |
| phase-dropout 0.3 (§4i) | wrist, `libero_v7` + bridge | 0.05–0.20 s | 0.071 | 0.654 | not reported | 5.3 mm |
| `longh_tqsa`, spatial on (§4m) | wrist | 0.5–2.5 s | 0.072 | 0.913 | not reported | — |
| `longh_tqsa`, spatial off (§4m) | wrist | 0.5–2.5 s | 0.075 | 0.739 | not reported | — |

Each row is a separate training run, so only the within-row ratio is claimed. The
columns are not an arm comparison and no row is better than another: §3.3 shows
that a cross-row difference in `std_ratio` or `wp_std_ratio` tracks how long the
run survived, not its configuration.

Across the full batch the ratio is **3.0x–29.1x over 19 arms, median ~8x**
[§4m, no artifact in store]. Read as shrinkage: at the §4d checkpoint the action
head is **76% shrunk** and the waypoint head **21% shrunk**.

Two independent framings of what the absolute error should be judged against, both
from `paper.md` and not the same quantity: §4d compares 3.0 mm to "the ~10–20 mm the
end-effector covers per 5-step chunk" (typical motion), while §4e derives from the
fitted gain that a *full-scale* command moves ~1.1 cm/step, so a 5-step chunk spans
~5.5 cm. The 58.2 mm figure is over a 2.5 s horizon and is not comparable to the
4.8 mm figure over 0.2 s.

**Mechanism (hypothesis, not measured).** MSE converges to the conditional mean,
whose magnitude is suppressed in proportion to the irreducible noise in the target.
Human teleop action commands at 20 Hz are noisy; the positions they produce are
smooth. Same network, same features, same loss family, different target. No direct
measurement of target-noise variance exists in this project
[UNVERIFIED: any measurement of the conditional variance of the two targets].

**Why it survives §3.3.** Both heads are evaluated on one checkpoint in one forward
pass, so neither the 11.1x seed spread nor the stop-timing confound can move the
ratio: whatever training length or seed produced the checkpoint produced both
numbers.

**Limits.** Open-loop and teacher-forced; bench scores the prediction and cannot see
compounding closed-loop error. Actuating the prediction requires inverting a fitted
per-axis gain (§3.4), so the claim that 0.787 of demo vigor reaches the robot is a
separate map that bench does not score. The actuator replaces only the 3 translation
dims; orientation and gripper remain on the BC head at `std_ratio` 0.237.

### Corrections (retracted or suspended, retained because they are instructive)

* **VOID.** `wp_std_ratio` 3.946 and `wp_mae` 116.1 mm (§4k) are not measurements of
  the head: bench scored a long-horizon head (0.5–2.5 s) against native-spaced
  targets (0.05–0.20 s), reporting a ~10x scale difference as prediction error.
  Bench now selects spacing from the checkpoint's cfg, with a test asserting the two
  spacings do not report the same error.
* **Row misalignment.** Bench scored the waypoint head at row 0 while
  `WaypointActuator` servoes toward the row derived from `cfg.waypoint_horizon`
  (clamped to the last supervised row, row 3). The 0.787 and 0.604 figures therefore
  describe a prediction the controller did not execute (§4i). Now aligned.
* **SUSPENDED.** The cross-run corollary that the auxiliary loss alone lifts the
  action head 0.175 → 0.237 at identical architecture, data and frozen world model
  (§4c) is a single-run A/B. It is smaller than the 0.022–0.245 gap the same command
  produces at the same seed (§3.3), and therefore carries no information.

---

## 3.2 World model [L2 for the training curve, L1 for the weight probes]

`wm_margin` is a property of the stage-A checkpoint alone — it scores a latent
rollout and never touches the planner. All 19 arms of the §4m batch share one frozen
stage A, so **`wm_margin` +19.8% is one number replicated across 19 rows, not 19
measurements**. It read identically (−7.3%) across all three v7.2 stage-B arms in
§4b/§4c for the same reason; that invariance is the harness's internal consistency
check, not a result.

| stage A | camera / corpus | best val | persistence | margin | bench `wm_margin` |
|---|---|---|---|---|---|
| v7 pilot (patience 3, ep 18) | wrist, `libero_v7` | 0.0106 (§2/§4g; no `metrics.jsonl` record) | 0.0111 | +4.5% | +1.7% |
| `v72_stageA_full` (ep 20 of 22) | agentview, 3 suites + bridge | 0.0109 | 0.0115 | +5.22% | −7.3% |
| `v72_stageA_wrist` (ep 34) | wrist, `libero_v7` + bridge | 0.0097 (note: "best 0.0098") | 0.0111 | +12.6% | **+19.8%** |

The −7.3% / +19.8% split is a property of the viewpoint, not of the model: a fixed
third-person camera is nearly static frame-to-frame, so persistence is a strong
baseline; a wrist camera moves with the arm, so persistence is weak. The v7 pilot was
already wrist-baked, so the gain from +4.5% to +12.6% on the same data is the
schedule, not the camera — `--patience 3` stops stage A on the same epoch the LR
halving fires, so the schedule never acts.

**Margin widens with rollout depth [L2, one checkpoint, horizon swept].** The
cleanest form of this measurement is a sweep over a single frozen checkpoint
(`full_stageA_ep3_backup.pt`, `n_episodes 40`), which removes the epoch/horizon
confound present in the scheduled-horizon training curve:

| H | val | persistence | margin |
|---|---|---|---|
| 1 | 0.00712 | 0.00753 | +5.5% |
| 2 | 0.00906 | 0.01013 | +10.6% |
| 3 | 0.01092 | 0.0132 | +17.3% |
| 4 | 0.0119 | 0.01466 | +18.8% |
| 5 | 0.01255 | 0.0158 | +20.5% |
| 6 | 0.0138 | 0.01715 | +19.5% |
| 8 | 0.01497 | 0.01821 | +17.8% |

n=1 per configuration; the schedule, corpus and viewpoint confounds are named, not
resolved. The training-curve version of the same claim (ep1 H=1 0.0084/0.0082 = −2%;
ep2 H=3 0.0117/0.0132 = +11%; ep3 H=4 0.0119/0.0147 = +19%) confounds horizon with
epoch and is not evidence for the widening on its own.

**Where the prediction gets its information [L1, weight probes on
`full_stageA_wrist_v72.pt`].** Change in `fused` when an input is perturbed, as a
percentage of mean |fused|: box embeddings **47.0%**, `box_weight → 0` (full evidence
fade) **43.0%**, frame embedding 38.3%, last action 31.5%, text tokens 13.8%, box
centers 12.0%. Against the residual the TRM actually predicts
(‖next_emb − cur‖ / ‖cur‖ = **0.0366**): `fused → 0` destroys **89.3%** of the
residual against **65.5%** for `state_delta → 0`, and by direction
cos(residual, residual | fused=0) = **0.634** against 0.706 for `state_delta`.
Removing the grounded observation destroys more of the residual direction than
removing the drift code, so the +19.8% margin is grounded prediction.

Two channel-level findings from the same probe. `msg` is degenerate at the source,
not ignored: mean |msg| 0.4786, constant part (batch mean) norm **3.3154** against
varying part **0.2682**, across-batch std / mean |msg| 0.100, **0 / 32** dead dims,
effective rank of the varying part **6.08 / 32**. A near-constant input is absorbable
into the consumer's bias, so a planner sensitivity of 0.0006 to `wm_msg` is the
correct response, not negligence. And `next_emb→cur` is amplitude-limited by
construction — the residual is 3.66% of ‖cur‖ — which is why `next_emb→stale` (a
full-magnitude, in-distribution *wrong* prediction) was added to separate a dead
path from a small perturbation. It read 0.0059 on both v7.2 agentview arms
(§4/§4b), the checkpoints it was introduced on.

The probes also carry a method correction that inverted a conclusion: a first pass fed
the TRM synthetic `fused` at mean |.| ≈ 1.0 when the real value is **9.0**, and at
1/9 scale the TRM looked nearly insensitive to `fused` (0.7% of next_emb), supporting
the opposite reading. Synthetic probe inputs must be scale-matched to the module that
produces them.

---

## 3.3 Run-to-run variance and the stopping confound — why no arm is ranked [L3]

**The same command at the same seed spans an order of magnitude.** §4k and §4l are the
identical stage-B command (`--waypoint-long --waypoint-weight 1.0`, same frozen stage
A, same seed):

| metric | §4k | §4l rerun |
|---|---|---|
| `std_ratio` | 0.022 | 0.245 |
| `corr` | 0.02 | 0.45 |
| `grip_acc` | 0.50 | 0.88 |
| `pose_mae` | 0.257 | 0.212 |

That gap is larger than every effect this project has claimed from an architecture or
regularizer change. Seed spread, corrected for a prefix-matching bug that had pulled
`longh_sdfade` into the `longh` error bar [§4m, no artifact in store]:

| config | n | mean `std_ratio` | sd | range | fold |
|---|---|---|---|---|---|
| `native` | 3 | 0.112 | 0.076 | 0.035–0.187 | 5.4x |
| `longh` | 3 | 0.051 | 0.048 | 0.022–0.106 | 4.7x |
| `longh`, all samples ever run | 5 | 0.084 | 0.097 | {0.022, 0.245, 0.022, 0.106, 0.024} | **11.1x** |

**Every bench metric tracked training length.** Arms in the batch early-stopped
between **8 and 28 epochs**, and across n = 9 mixed-corpus arms
[§4m, no artifact in store]:

| metric | Pearson vs epochs | Spearman |
|---|---|---|
| `wp_std_ratio` | 0.891 | **0.924** |
| `grip_acc` | 0.901 | **0.907** |
| `std_ratio` | 0.770 | **0.866** |
| `pose_mae` | −0.793 | **−0.865** |
| `corr` | 0.915 | **0.840** |

Sorted by epochs — *this is the confound, not a ranking; the arm names are labels for
runs of different length, and no row is claimed to be better than another*:

| epochs | best val | `std_ratio` | `corr` | `grip` | `pose_mae` | arm |
|---|---|---|---|---|---|---|
| 8 | 0.8200 | 0.024 | 0.09 | 0.50 | 0.250 | `longh_s2` |
| 8 | 0.7683 | 0.020 | 0.07 | 0.59 | 0.253 | `longh_sdfade` |
| 9 | 0.7523 | 0.022 | 0.02 | 0.50 | 0.257 | `longh_s0` |
| 14 | 0.6829 | 0.044 | 0.24 | 0.73 | 0.250 | `longh_pregrasp` |
| 15 | 0.6952 | 0.070 | 0.13 | 0.88 | 0.247 | `longh_novis` |
| 22 | 0.6138 | 0.114 | 0.46 | 0.87 | 0.233 | `native_s1` |
| 22 | 0.6281 | 0.106 | 0.31 | 0.88 | 0.243 | `longh_s1` |
| 24 | 0.6450 | 0.187 | 0.45 | 0.93 | 0.227 | `native_s2` |
| 28 | 0.6299 | 0.072 | 0.39 | 0.91 | 0.242 | `longh_tqsa` |

Grouped: ≤ 9 epochs → `std_ratio` 0.022, `corr` 0.06, `grip` 0.53; ≥ 22 epochs →
0.120, 0.40, 0.90. An under-trained planner emits near-constant actions, which is
exactly what a low `std_ratio` measures.

**Mechanism — a bug, not variance alone.** `train/train_batched.py` gated early
stopping on `val_loss = val_bc + waypoint_weight * val_wp`, compared against an
**absolute** `--min-delta` of 1e-4. `val_wp` is ~10x larger under `--waypoint-long`
(0.5–2.5 s targets versus 0.05–0.20 s), and a larger term carries larger noise, so a
fixed absolute threshold is cleared less often and staleness accrues faster.
Long-horizon arms ran a harsher effective patience than native ones. Best-val totals
confirm the scale split: longh 0.75–0.82 against native 0.61–0.65, consistent with the
reported decomposition `val bc 0.6924` + `wp 0.1107` = 0.803. Fixed by
`--stage-b-select {bc,total}` (defaulting to `bc`, the only term on a scale shared by
every arm) and `--stage-b-min-epochs`, both pinned by
`tests/test_stage_b_selection.py`; `total` is retained solely to reproduce the invalid
batch.

**The sensitivity instrument is unmeasurable at n=1 for most inputs.** Same config,
seed the only difference [§4m, no artifact in store]:

| input | `native`, 3 seeds | fold | `longh`, 3 seeds | fold |
|---|---|---|---|---|
| `fused` | 0.0605, 0.0479, 0.0596 | **1x** | 0.0967, 0.0327, 0.0626 | 3x |
| `state_delta` | 0.0072, 0.0234, 0.0419 | 6x | 0.0305, 0.1068, 0.0119 | 9x |
| `proprio` | 0.0020, 0.1216, 0.1511 | **76x** | 0.0020, 0.0919, 0.0543 | 46x |
| `geometry` | 0.0, 0.0134, 0.0064 | **134x** | 0.0, 0.0035, 0.0046 | 46x |

The phase:vision ratio is therefore unusable at n=1: it spans 0.2–2.9 across native
seeds and 0.3–5.5 across longh seeds. What survives is the weaker,
instrument-supported statement: **`fused` pose-sensitivity is 0.03–0.10** and is the
largest single visual contribution; its size relative to proprioception cannot be
established from one run.

**Four instrument defects, each of which inverted or voided a recorded conclusion.**

1. Bench never passed `spatial=` to the planner, so every number recorded for a
   TQSA-trained checkpoint was produced with **22 of ~82** cross-attention memory
   tokens withheld (16 spatial + 3 pooled roles + 3 heatmaps), ~27% of the planner's
   observation. Same root cause on the deployment side: `eval/policy.py` wired a TQSA
   unconditionally and only *warned* when the checkpoint lacked weights, so a
   non-TQSA checkpoint would have been fed 22 tokens of random-init perception.
2. `--sensitivity` reported mean |Δplan| over the whole plan, whose last column is a
   hard ±1 gripper decision. One flipped gripper decision at every step contributes
   exactly `plan_steps * 2 / (plan_steps * num_servos)` = **0.2857** — against a
   largest-ever recorded sensitivity of `state_delta` **0.2740**. A reading that size
   is equally consistent with "strongly shapes the pose trajectory" and "flips the
   gripper and leaves pose untouched". Bench now reports pose-only |Δplan| and
   gripper-flip rate in separate columns; readings from the two builds are not
   comparable.
3. The long-horizon-versus-native-target scoring error (§3.1 corrections).
4. The unscaled synthetic probe (§3.2), mean |.| ≈ 1.0 against a real 9.0.

**Retracted by this section.** "Vision finally dominates the planner" (§4k), which
rested on `phase:vision = 2.0:1` — three seeds of that exact config give 0.3:1, 1.0:1
and 5.5:1. "Best arm yet" (§4l) — 0.245 is the top of a 5-sample distribution of the
same config. The three-arm ranking waypoint > no-TQSA > TQSA (§4c) — three
single-run A/Bs, one sample each (§4l), on the agentview corpus later found to be
the wrong camera (§4f). `paper.md` attributes the stop-timing confound to the
12-arm batch "and several earlier single-run A/Bs" without naming them, so
[UNVERIFIED: that this particular ranking specifically measured stop timing]; the
one-sample objection is sufficient to retract it either way.

And the TQSA question, asked both ways, answers differently. The across-runs
version — `spatial` 0.0688, third-strongest, with `grip_acc` 0.93 → 0.52 (§4b) —
is an A/B between two separately trained single runs and so carries no
information under this section's own rule; it is quoted only as the claim that
motivated the within-checkpoint test. Benched with and without the spatial
pathway on ONE checkpoint, it reads 0.072 / 0.913 / 0.39 / 0.91 at 0.74 s per eval
against 0.075 / 0.739 / 0.38 / 0.87 at 0.45 s — within noise on every action metric
at 1.6x the inference cost, with withheld `spatial` moving pose by 0.0487
[§4m, no artifact in store].

---

## 3.4 Closed loop: a well-sampled zero

**`mean_success` is 0.000.** `libero_object`, the arm picked by `std_ratio` with
`grip > 0.7` (a selection rule for which checkpoint to run, not a claim that it is
the better arm — §3.3), **5 trials × 10 tasks = 50/50 failures**, all 10 tasks
completed, 0 scavenged, no failed workers [§4m, no artifact in store; the largest
real-environment result JSON in `eval_results/` is `n_trials 3 × 10` tasks at
`mean_success` 0.0, which predates both the actuator fix `e362d2c` and the camera
fix (§4f; `paper.md` records no commit hash for it)]. Earlier
closed-loop runs were 0/10 on `libero_object` and 0/10 on `libero_spatial`. This
crosses the project's own pre-registered kill bar for Claim 1 — "< 30% absolute where
big models exceed 80%" — so Claim 1 is dropped, by the rule written before the
experiment.

The zero is not attributable to any one arm: the batch that produced the benched
candidates is the confounded one (§3.3).

**The diagnosis chain is per-step telemetry, not aggregate score.** Bench scores the
prediction; the actuator is a separate map from prediction to command that only
closed-loop exercises.

*The fitted gain* (`preprocess/fit_waypoint_gain.py`, all three suites, **82,844**
(action, Δeef) pairs from **1500** episodes, **0** skipped): x 0.01056 m per unit
action per step at R² 0.870, y 0.01200 at R² 0.938, z 0.01085 at R² 0.866 — all far
above the 0.5 usability threshold, so LIBERO's OSC translation response is per-axis
linear and the inversion is sound. (§4g restates this fit as "R² 0.88/0.99/0.94",
which matches neither §4e nor the record. [UNVERIFIED: the 0.99 figure has no
source.])

*The defect.* Per-step telemetry over one completed 300-step episode: `steps 300 |
waypoint cmds 300`, `|cmd| mean 0.5301 max 1.0000`, `plan_norm mean 2.418`. Two
stacked control-law bugs. (i) Units: `gain` is metres per unit action per *one*
control step, but a horizon-*h* waypoint spans *h* steps; dividing by `gain` alone
over-commands by exactly *h* (5 at the default horizon). The correct law is the
per-step rate closing the remaining error over the remaining steps,
`error / (gain * steps_left)`. (ii) Duty cycle: the target was held across a whole
perception period, valid only when the period is no longer than the horizon; at the
deployment 15:5 ratio the arm reaches the held target in ~5 ticks and idles for 10.
Together these predict the observed profile — saturate for ~5 ticks, coast for 10.
Fixed in `e362d2c`. That run is **void as a policy measurement** and reports no
`mean_success`; the harness itself validated in it (canary clean, env built, 100
steps executed, no stall, `--workers 5 --stagger 10` sharded and reported normally).
`success=False` at `--max-steps 100` is by construction: 100 steps is 5 s of robot
time at 20 Hz and a LIBERO pick-and-place needs 150–300.

*After the fix*, `|cmd|` mean **0.4533** (was 0.5301), verified by a direct probe: a
3 cm / 5-step waypoint yields `cmd_x = 0.568` under `error/(gain*steps)` where the old
law asked for 2.84 and clipped to 1.0. Working backwards, 0.4533 implies ~2.4 cm over
5 steps (4.8 mm/step), an equivalent per-step action of ~0.45 against a demo magnitude
of ~0.58 — the §3.1 ratio arriving intact at the actuator. **Retracted in place:** an
earlier prediction of ~0.10–0.11 and the "still saturating" diagnosis built on it were
wrong; the prediction came from a guessed demo action magnitude, not a measured one,
and `max 1.0` alone does not distinguish tracking from bang-bang — the fraction of
steps at the clip does, and the first analysis never computed it.

*The train/deploy camera mismatch.* `preprocess/libero.py --camera` defaulted to
`agentview_rgb` (third-person, rotated 180°) while `eval/libero_eval.py` reads
`robot0_eye_in_hand_image`; the npz key is named `wrist_frames` either way, which hid
it for a session. Cost: **1 three-suite bake, 1 stage A, 3 stage-B trainings, 4 bench
runs, 2 closed-loop evals**. No aggregate metric flagged it — stage A converged, three
stage-B arms trained cleanly, bench produced internally coherent numbers. It surfaced
from watching the robot ("the basket is reached, the object is never approached") and
then reading the sensitivity table as a statement about which inputs the policy could
physically be using.

*Ceiling that remains.* The waypoint actuator replaces only the 3 translation dims;
orientation and gripper still come from the BC head at `std_ratio` 0.237, so 4 of 7
dims are untreated. A failure to align the wrist or close the fingers is the expected
next bottleneck, not evidence against §3.1.

---

## 3.5 Infrastructure and budgets

**Frozen-backbone spatial-map precompute.** Stage B with `--tqsa` re-ran YOLO-World
over every framed timestep every epoch at a 128→512 px upscale: ~6.1 s/batch, 105
batches in 644 s, ~18 min/epoch, ~12 h for 40 epochs. The backbone never trains, so
the maps are identical across all 40 epochs. Precomputed once: 19,680 train + 1,031
val frames, map shape **(512, 20, 20)** = 400 KB/frame at fp16, **7.5 GB train +
0.4 GB val = 7.9 GB** resident, one pass **151 s** (137 s train at 154–165 frames/s +
14 s val, 160 frames/s overall). Per-epoch cost **1080 s → 130 s, 8.3x** — the same
cost as the TQSA-free arm (~146 s), i.e. 12 h → ~90 min. Not bit-identical: fp16
storage is ~2e-4 relative and chunked batching ~1e-6, both far below the signal.
Rejected after measurement: lowering `min_side` saves **0.6%, not 4x** (512, 256 and
128 all letterbox to 640×640 and yield the same 20×20 map). Available but not taken:
truncating the forward at SPPF, measured **maxdiff 0.0, 1.8–2.0x**.

**Device placement.** Heads on CPU cost **3.75 s/step** (375 s per 300-step episode)
against **0.23 s/step** with `--heads-device cuda:0` at 5 workers (~70 s) — **16.3x**.
`--device` only ever moved the detector, which runs 1 tick in 15, while the d=1024
TRM, fusion and planner run every tick. At `--n-trials 20 --max-steps 300` this is
~60 hours against under 4.

**Contention.** Seven processes on GPU 1 totalling ~154 GB of 192 GB. Identical
stage-A epochs ran **96 s uncontended against 496 s contended (5.2x)**. Every
per-epoch second reported in this paper is contended and is not a hardware claim.

**Parameter ledger** (`microvla.utils.param_audit`). v7.2: fusion **4,460,165** ·
drift **724,993** · planner **1,770,247** · total **6,955,405** of **9,000,000**.
After v7.4: planner **1,803,527** (1,804,298 with the waypoint head), total
**6,988,685** [§4j, no record; re-runnable via the audit]. The headline result's head
costs **771** of them, `((d_plan+1)*3)`. Planner ablation deltas: `geometry` −1,792 ·
`pred_box_emb` −16,640 · both −18,432 · plus `next_emb` −35,072 · `spatial` →
1,720,583.

**Data pipeline under a 10 GB total cap.** LIBERO suites were downloaded, converted
and deleted one at a time because all three resident at once is ~10–12 GB, the entire
project budget: `libero_object` 5.82 GB download / 7.44 GB reconstructed / 500
episodes; `libero_spatial` 3.47 GB / 6.24 GB / ~500; `libero_goal` — / — / ~500.
Resulting corpus: **6023 train episodes across 60 length-buckets, 316 val**; **23 of
60** buckets carry `wrist_frames` (the ~1500 LIBERO episodes), while the ~4500 bridge
episodes are frameless and proprio-less (validity flag 0) and train planner-only with
`spatial=None`.

**Parallel-eval failure modes.** `mp.Pool` reaps and replaces a segfaulting or
OOM-killed worker without re-dispatching its chunk, so the parent blocks forever with
no error: reproduced as 10 live workers indefinitely, against `ProcessPoolExecutor`
raising `BrokenProcessPool` in **0.4 s**. Also: `eval/__init__.py` imports torch at
package import, so `os.environ` writes inside a spawned worker land after libgomp init
and are no-ops (moved to the parent); and the overnight wrapper had no SIGTERM shield
while all 21 Python CLIs did, and logged every failure as `rc=0` because `rc=$?` was
captured after a `[ -f ]` test.

**Tests.** 149 → 231 passing, CPU-only, mock-only, no network, no cv2 [§4j/§5, no
record; re-runnable via `pytest tests -q`].
