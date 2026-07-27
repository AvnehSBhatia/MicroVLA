## Discussion

### What the closed-loop zero diagnoses

Closed-loop `mean_success` is **0.000 over 50 trials** (5 trials x 10 tasks,
`libero_object`, all 10 tasks completed, 0 scavenged, no failed workers, §4m).
The harness is not the explanation: the same harness was validated end to end
(canary clean, env built, 100 steps executed, no stall, `--workers 5 --stagger 10`
sharding and reporting normally, §4e), the control law had been corrected
(`e362d2c`), and the corpus had been re-baked from the camera the eval actually
reads (§4f). This is the first zero in the project that is sampled well enough
to state as a result rather than as a symptom.

Three observations constrain what it can mean. Only the first is checkable
against `paper.md` / `results/metrics.jsonl`.

**1. On-distribution the displacement head varies normally.** On the one
checkpoint benched with and without its spatial pathway, `wp_std_ratio` reads
**0.913** against an action `std_ratio` of **0.072** in the same forward pass
(§4m). Whatever is wrong closed-loop is not a head that predicts a constant
on its training distribution. [UNVERIFIED: the batch-wide statement that
`wp_std_ratio` sits at 0.75-0.94 across the 19 benched checkpoints — this range
is quoted in `DESIGN.md` from `results/PAPER_TABLE.md`, which is not in the
repo; §4m itself quotes only the two values above.]

**2. The command was reported as near-constant in direction, with a direction
that differs between runs — and that reading cannot be checked against the
source of record.** [UNVERIFIED: the per-axis `|cmd|` means for the 0/50 run and
for an earlier run, the ratio between the strongest and the weakest axis, the
number of steps it was sustained over, and the clipped-step fraction. Those
figures appear only in `DESIGN.md` ("Motivating evidence"), which attributes
them to §4m; §4m contains no per-axis telemetry, and neither does any other part
of `paper.md` or `results/metrics.jsonl`.] What the source of record does hold
are scalar magnitudes: the pre-fix VOID run at `|cmd|` mean **0.5301**, max
**1.0000**, 300 of 300 steps commanded (§4e), and the corrected-law measurement
`|cmd|` mean **0.4533** (§4e). Neither is per-axis and neither is from the 0/50
run, so the constant-direction observation enters here as an unverified input,
not as a measurement.

**3. If (2) holds, the pair is the signature of exposure bias rather than
underfitting.** Every open-loop number in this paper is teacher-forced: bench
replays baked episodes, so the policy is placed back on the demo manifold at
every step and never has to survive its own error. Closed-loop, one step's error
selects the next observation. An MSE-BC head is a conditional-mean estimator;
off the support it was fit on, its output has no reason to be anything but its
mean — a fixed vector, which is exactly a constant commanded direction. The
competing story is actuator saturation, and §4e already names the diagnostic
that separates the two: the FRACTION of steps at the clip, not `max`. That
fraction is not reported for the 0/50 run anywhere in the source of record
[UNVERIFIED: the clipped-step fraction during the 0/50 run], so saturation is
not excluded here. Exposure bias is the leading hypothesis for the zero; it is
not established by anything in `paper.md` or `results/metrics.jsonl`.

This also bounds what the corrector can do. In `"delta"` mode low trust brakes
(`plan = min(1, tau/cfg.brake_trust) * raw`) — it scales magnitude toward zero
and leaves direction untouched. Braking a wrong constant direction produces a
slower wrong constant direction, which is not a recovery mechanism. [UNVERIFIED:
no measurement of tau during the 0/50 run is reported anywhere in the document;
Claim 5's AUROC was never computed.]

The instrumentation consequence is another instance of this project's most
repeated lesson. `std_ratio`, `corr`, `pose_mae`, `wp_std_ratio` and `wp_mae_mm`
are all computed under teacher forcing and *cannot in principle* register
compounding error. The two defects that were caught — the 5x over-command (§4e)
and the camera mismatch (§4f) — were localized by per-step telemetry and by
watching the robot, never by an aggregate score, and the constant-direction
reading of (2), whatever its provenance, is per-step telemetry as well.
Aggregate scores cannot see a train/deploy interface defect because both sides
are individually self-consistent.

### Why this benchmark under-rewards the thing that is missing

The grounding failure is not upstream. Weight-level probes on the +19.8%
stage-A checkpoint show fusion is **47.0%** determined by box embeddings and
**43.0%** by the evidence-fade weight; zeroing `fused` destroys **89.3%** of the
TRM's predicted residual against **65.5%** for the drift code, and removing the
observation destroys more residual *direction* than removing drift
(cos **0.634** vs **0.706**, §4h). Those probes are run on one fixed checkpoint
and do not depend on which stage-B run is in front of them. §4h's conclusion —
"vision is available and discarded, not absent" — therefore locates the failure
in the planner. The planner-side reading it was drawn from (`fused` **0.0178**
against **0.464** for the two phase inputs summed, §4g) is a single run on the
combined instrument §4i later showed to be ambiguous, and rests on `proprio` and
`state_delta`, which §4m measures at 46-76x seed fold. It fixes the location; it
does not license a ratio.

Two structural reasons make discarding it the loss-minimizing choice.

*Supervision horizon.* At LIBERO's 20 Hz with `plan_steps=5`, the plan spans
**0.25 s** and the native waypoint targets **0.05-0.20 s** (§4j). Over 0.2 s,
"keep doing what you are doing" is a near-sufficient statistic for the demo
action; object position is a second-order correction. MSE-BC consumes variance
in descending order, so it takes the phase term and leaves the vision residual.

*Task structure.* In `libero_object` the placement target is at a fixed location
in every episode; only the pick target moves. After the grasp the trajectory is
transport to the same place every time. Grounding is therefore priced into the
BC loss over a minority of each episode — the pre-grasp approach — and nowhere
else. The video signature matches exactly: **basket reached perfectly, object
never approached** (§4f). A policy that has learned the transport phase and not
the approach phase is heavily rewarded by the training objective and scores
zero at execution, because the task is gated on the phase the loss barely
weights.

Neither reason is measured, and no phase:vision ratio is quotable in support of
them. The wrist native arm read **0.464** for phase against **0.040** for
vision, **12:1** (§4g), but §4i withdrew that whole class of reading — the
combined instrument mixes a discrete gripper bit with continuous pose, and one
flipped gripper decision contributes exactly `5*2/35 = 0.2857` against a
largest-ever recorded sensitivity of **0.2740** — and §4m then showed the ratio
is unmeasurable at n=1, spanning **0.2-2.9** across three native seeds and
**0.3-5.5** across three longh seeds. The two reasons above are mechanisms
proposed for the video signature and for the weight-level probes, not
conclusions read off a ratio.

Both interventions aimed at this are inconclusive, and both are single-run A/Bs
of the kind §4l suspended. Taxing the shortcut (`--phase-dropout 0.3`) was
recorded as moving phase:vision 2.3x with `proprio` *rising* 0.1904 -> 0.2255
and `grip_acc` collapsing **0.93 -> 0.50** (§4i) — one run per arm, on the
combined instrument, on the two inputs §4m measures at 46-76x seed fold, so none
of it is a measurement of the intervention. What carries over is the mechanism
§4i gives for the gripper collapse — `proprio` carries the arm's gripper state,
the single best predictor of the gripper command — which is why §4j replaced the
coarse flag with per-input drop rates. Removing the shortcut's sufficiency
(`--waypoint-long`, 0.5-2.5 s targets) produced an apparent inversion that three
seeds of the identical config could not reproduce (phase:vision 0.3:1, 1.0:1,
5.5:1, §4m). What survives at instrument strength is only: **`fused`
pose-sensitivity is 0.03-0.10** and is stable across seeds; its size relative to
proprioception is not measurable from one run.

The one live hypothesis is corpus dilution. Bridge is ~75% of mixed-corpus
stage-B steps and supplies neither proprioception nor frames. `longh_liberoonly`
reached `std_ratio` **0.253**, **+2.96 prediction-sd** above the training-length
trend fitted on the 9 mixed-corpus arms, while its `corr` and `grip` residuals
(+0.57 and −0.14 sd) are entirely explained by training length. It ran 26 epochs,
so it sits in the well-trained group and the stop-timing confound does not
dispose of it by itself; its best val (0.5724) is on a different corpus and is
not comparable to the other arms at all. Against the strongest single comparator
it is 1.35x but only **0.87 seed-sd** (§4m) — which is not a ranking and cannot
become one at n=1. That is a hypothesis with one supporting number, not a result.

### Why the surviving result survives

The displacement-vs-action ratio (**3.0x-29.1x** across 19 arms, median ~8x;
replications 3.3x, 4.8x, 3.3x) is the only quantity in this document that no
confound reaches, and the reason is structural rather than lucky: both heads
read the same `feats` in the same forward pass of the same checkpoint. A
stop-timing confound shifts both numerators and denominators together; seed
spread shifts the level of each arm, not the within-pass ratio. The *levels*
move a great deal: `std_ratio` spans **0.022-0.245** across five runs of one
identical command (mean 0.084, sd 0.097, **11.1x fold**, §4m), and the
`wp_std_ratio` figures recorded in this document run from **0.604** (§4g) to
**0.913** (§4m) across different arms. None of those levels is a claim. The
ratio is.

## Limitations

* **The headline hypothesis was never tested.** Claim 2, the perception-rate
  sweep (30/5/2/1/0.5 Hz, ours vs hold-last vs oracle), is the experiment the
  paper was designed around and E4 was never run on real data. Nothing here
  bears on it.
* **Claim 1 is dropped by its own pre-registered rule.** The kill bar was
  "< 30% absolute where big models exceed 80%"; measured success is 0.000. No
  competitiveness claim against larger models is made or implied.
* **n = 1 nearly everywhere.** Exactly two configurations have three seeds. At
  n=1 the sensitivity instrument is unusable for `proprio` (**76x** and **46x**
  seed fold) and `geometry` (**134x** and **46x**); only `fused` is stable
  (**1x** and **3x**).
* **No arm ranking is possible from any data in this document.** Arms
  early-stopped between 8 and 28 epochs and every bench metric tracks
  epochs-survived (Spearman `wp_std_ratio` 0.924, `grip_acc` 0.907, `std_ratio`
  0.866, `pose_mae` −0.865, `corr` 0.840, n=9). The fix
  (`--stage-b-select bc`, `--stage-b-min-epochs`) is landed and pinned by
  `tests/test_stage_b_selection.py`, but the batch has not been re-run.
* **Everything except `mean_success` is open-loop and teacher-forced**, and by
  construction blind to compounding error. Compounding error is the leading
  hypothesis for the zero, not a measured attribution: nothing in `paper.md` or
  `results/metrics.jsonl` isolates it.
* **4 of 7 action dimensions are untreated.** The waypoint actuator replaces
  only the 3 translation dims; orientation and gripper still come from the
  collapsed BC head.
* **The waypoint numbers describe a row the controller did not execute.** Bench
  scored row 0 while `WaypointActuator` servoes the row derived from
  `cfg.waypoint_horizon` (clamped to the last supervised row, row 3). Aligned
  now; the 0.787 and 0.604 figures predate the alignment.
* **The actuation half of the target-parameterization claim is inferential.**
  0.787 arrives at the robot only to the extent the fitted per-axis gain holds
  (x 0.01056 R² 0.870, y 0.01200 R² 0.938, z 0.01085 R² 0.866), and the
  supporting telemetry is a single post-fix magnitude (`|cmd|` mean **0.4533**).
* **`wm_margin` is a property of the camera as much as of the model**: +1.7%
  (pilot, wrist), −7.3% (agentview), +19.8% (wrist, better-trained). Quoting it
  without the viewpoint is meaningless.
* **The proposed mechanism for the headline result is unmeasured.** Target noise
  is a plausible explanation for why displacement shrinks less than action, and
  no direct measurement of target-noise variance appears anywhere in this work.
* **No edge deployment.** Claim 3 (Raspberry Pi 5 + AI HAT, 30 Hz control, 2 Hz
  NPU perception, single-digit watts) is deferred; nothing was measured on the
  target hardware.
* **Provenance gap for the §4m batch.** `results/metrics.jsonl` ends at
  `2026-07-26T08:29:37+00:00` and the batch ran 08:34-11:10; `results/PAPER_TABLE.md`
  and `logs/overnight/` are not in the repository. Every §4m figure quoted above
  — including the Spearman table that invalidates all arm rankings and the
  50-trial zero — must be recovered from the training box before publication.
  [UNVERIFIED: every §4m number against this repository's artifacts, including
  the 50-trial zero itself. The direction is consistent with the rest of the
  source of record — the first closed-loop LIBERO evals "scored 0/10 (object AND
  spatial)" (2026-07-23 diagnosis) and the §4e run reports no `mean_success` at
  all — but no artifact in this repository contains the 50-trial run.]

## Corrections

Nine recorded conclusions in this project were later withdrawn or voided. They
are listed here rather than deleted because the pattern in them *is* the
methodological result: in every case the withdrawn claim was a comparison made
across training runs or through an unvalidated instrument, and in every case
the replacement is either a within-run measurement or nothing.

| # | Claim as recorded | Why it fell | What replaced it |
|---|---|---|---|
| 1 | v7 pilot bench column and the entire v7 sensitivity ranking (`std_ratio` 0.369, `grip_acc` 0.93, `corr` 0.49, `pose_mae` 0.20, `wm_margin` +1.7%) | `eval/bench.py` never passed `spatial=`; 22 of ~82 planner memory tokens (~27% of the observation) withheld from every TQSA checkpoint | No comparable replacement. `eval.bench --tqsa` added; the flagless path warns. Pilot numbers are quotable only with this caveat |
| 2 | "Ranking after three arms: waypoint > no-TQSA > TQSA" (§4c) | §4m: every bench metric is a monotone function of epochs survived (Spearman >= 0.84, n=9), and the arms ran 8-28 epochs | No ranking. The stopping criterion was fixed and the batch must be re-run |
| 3 | "TQSA carries real signal — `spatial` is the third-strongest planner input" (0.0688, §4b) | Combined instrument, cross-run comparison | Same-checkpoint head-to-head (§4m): `std_ratio` 0.072 vs 0.075, `corr` 0.39 vs 0.38, `grip` 0.91 vs 0.87, at **1.6x** inference cost (0.74 vs 0.45 s/eval). Withholding `spatial` moves pose 0.0487. Within noise on every action metric |
| 4 | "Vision finally dominates the planner" — phase:vision 2.0:1 (§4k, §4l) | Three seeds of the identical config gave 0.3:1, 1.0:1 and 5.5:1 | `fused` pose-sensitivity is 0.03-0.10 and stable across seeds; its ratio to proprioception is not measurable at n=1 |
| 5 | "Best arm yet" (§4l, `std_ratio` 0.245) | 0.245 is the top of a 5-sample distribution of the same command: mean 0.084, sd 0.097, **11.1x fold**. The paired run at the same seed gave 0.022 | No best arm. A single stage-B run is not a measurement of a configuration |
| 6 | Long-horizon action-head collapse explained by loss imbalance (target RMS ~0.163 vs ~0.047, "~12x the MSE") | A long-horizon run at weight 1.0 reported `val bc 0.6924` against `val wp 0.1107` — BC is 6x larger, so the waypoint term never dominated. The arithmetic was right and the reasoning wrong | First re-attributed to run-to-run variance (§4l), then to stop timing: the matching row in §4m's table (`longh_s0`, `std_ratio` 0.022 / `corr` 0.02 / `pose_mae` 0.257) early-stopped at 9 epochs |
| 7 | Long-horizon waypoint accuracy `wp_std_ratio` **3.946**, `wp_mae` **116.1 mm** (§4k) | Bench scored a 0.5-2.5 s head against 0.05-0.20 s targets; a ~10x scale mismatch reported as prediction error | Void. Re-measured after the fix: 0.799 and 58.2 mm over 2.5 s. A test now asserts the two spacings do not report the same error |
| 8 | "The world model ignores the observation" (first §4h probe pass) | Synthetic `fused` at mean \|·\| ~1.0 against a real 9.0 — an out-of-distribution artifact | Inverted with real inputs: zeroing `fused` destroys **89.3%** of the TRM residual, and cos 0.634 vs 0.706 says removing the observation costs more residual direction than removing drift |
| 9 | First closed-loop waypoint run (§4e); and the follow-up prediction that post-fix `\|cmd\|` would be ~0.10-0.11 with the policy "still saturating" | Actuator over-commanded by exactly the horizon (5x) and held targets across a whole perception period (2/3 idle): `\|cmd\|` mean 0.5301, max 1.0000. The 0.10-0.11 prediction came from a guessed, not measured, demo action magnitude | Control law corrected to `error / (gain * steps_left)` (`e362d2c`); measured post-fix `\|cmd\|` mean **0.4533**. The diagnostic that distinguishes tracking from bang-bang is the *fraction* of steps at the clip, which the first analysis never computed |

Three further items are superseded rather than retracted — measured correctly,
but the conclusion drawn from them no longer holds.

* **"The magnitude collapse is feature starvation."** The v4-v7 input-side chain
  (direct geometry, miss-hold, proprioception, text-queried spatial attention,
  dream-consistent training) moved `std_ratio` 0.12 -> 0.175 -> 0.237; changing
  what is regressed moved it to 0.787 in one step, and §4h established vision is
  *available and discarded*, not absent. (The input-side chain is itself three
  separate single runs and spans less than the 0.022-0.245 same-command gap, so
  it carries no information either way.)
* **"The world model's channel into control came back from the dead"**
  (`wm_msg` 0.0007 -> 0.2394, §4c). It did not reproduce: `wm_msg` read 0.0006
  in §4g, 400x down, on a *better* world model. §4h then showed `msg` is 92% a
  fixed vector (constant norm 3.315 vs varying 0.268) at effective rank
  **6.08/32** — a near-constant input is absorbable into the consumer's bias, so
  a sensitivity of 0.0006 is the correct response to it, not negligence.
* **"Every closed-loop failure in this project has been an interface defect
  between correct components, not a model failure."** Recorded in §4e as true of
  every closed-loop failure up to that point, and written before `mean_success`
  0.000 was obtained on a validated harness, a corrected control law and the
  correct camera. It is no longer available as an explanation.

One correction ran the other way. The early stage-A evidence that the world
model's margin over persistence widens with rollout horizon was recorded from an
epoch table (H=1 −2%, H=3 +11%, H=4 +19%) in which horizon is confounded with
training epoch. The clean version — one fixed checkpoint
(`full_stageA_ep3_backup.pt`), horizon swept — carries no such confound:
**+5.5% at H=1, +10.6%, +17.3%, +18.8%, +20.5% at H=5, +19.5% at H=6, +17.8% at
H=8** (`results/metrics.jsonl`, `kind: horizon_curve`, 40 episodes). `paper.md`
does not quote it.
