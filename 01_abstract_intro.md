# Regress Displacement, Not Actions: One Result That Survived, and Four Reasons the Others Did Not

## Abstract

We report a vision-language-action stack built to hard budgets — ~30M deployed
parameters of which **6,988,685** are trainable (fusion 4,460,165 · drift
724,993 · planner 1,803,527, against a 9,000,000 cap), a 10 GB total disk
ceiling including transient download state, and a Raspberry Pi 5 as the
deployment target — and its closed-loop result is **`mean_success` 0.000 over 50
LIBERO-object trials** (5 trials × 10 tasks; all 10 tasks completed, 0
scavenged, no failed workers). That zero is the honest headline and we do not
soften it.

One positive result survives every subsequent audit. At fixed architecture,
data, frozen world model and inputs, a head regressing **metric end-effector
displacement** is far less magnitude-shrunk than the head regressing
**normalized action commands** on the same features: the ratio
`wp_std_ratio`/`std_ratio` is **3.0x–29.1x across 19 benched checkpoints (median
~8x)** [UNVERIFIED against local artifacts], with individual replications of
0.787 vs 0.237 (3.3x), 0.604 vs 0.126 (4.8x) and 0.799 vs 0.245 (3.3x) — each
of which is in `results/metrics.jsonl` — spanning two cameras and two
supervision horizons. Healthy is ~1.0, so the action head is **76% shrunk** and
the displacement head **21% shrunk** on the same forward pass. The ablation
costs **771 parameters at training time and zero at inference**. The proposed
mechanism is target noise rather than model capacity: MSE converges to a
conditional mean whose magnitude is suppressed in proportion to irreducible
target noise, and 20 Hz human teleop *action commands* are noisy where the
*positions* they produce are smooth.

It survives because of *how* it was measured, and that is the paper's second
result. A 12-arm overnight batch was confounded by its own stopping rule: early
stopping gated on `val_bc + waypoint_weight * val_wp` against an **absolute**
`--min-delta` of 1e-4, while `val_wp` is ~10x larger under long-horizon
supervision. Arms stopped between **8 and 28 epochs**, and across n = 9
mixed-corpus arms every bench metric tracks epochs-survived — Spearman
`wp_std_ratio` **0.924**, `grip_acc` **0.907**, `std_ratio` **0.866**,
`pose_mae` **−0.865**, `corr` **0.840** [UNVERIFIED against local artifacts].
**No single-run arm comparison in this project is a valid result, and none is
reported as one.** Four further instrument defects and one silent train/deploy
camera mismatch each inverted or voided a recorded conclusion. The rule that
follows, and that we recommend to anyone measuring small VLA policies:
**comparisons made within a single forward
pass of a single checkpoint are trustworthy; comparisons made across
checkpoints from separate training runs are not, absent a scale-invariant
stopping criterion and >= 3 seeds.** This is a negative-result and methodology
paper with one positive core result. It ranks no arms, claims no architecture
win, and reports no competitiveness against larger models.

---

## 1. Introduction

### 1.1 The system and the constraints that shape every number in it

MicroVLA is a micro vision-language-action stack whose deployment target is a
Raspberry Pi 5 driving a 7-servo rig. Text is parsed into ordered
source/target phrases and embedded by **YOLO-World's own CLIP text tower** —
there is no separate language model. At 2 Hz a frozen **YOLO-World-S** supplies
a frame embedding plus per-role box embeddings and centers. These feed
`SlotResonanceFusion` (→ `[B, 32, 5]`) and an `AnchoredDriftEncoder` (→
`[B, 256]`), both of which feed a weight-tied recursive TRM that predicts the
**next frame embedding** `[B, 512]`; a `ChronoQueryPlanner` decodes that into a
`[5, 7]` plan. The 30 Hz control loop runs real perception every 15th tick and
**dreams** the other 14, feeding the corrected prediction back through fusion,
with a parameter-free `InnovationCorrector` doing drift correction and
trust-scaling the emitted plan.

Three constraints are not background; they determine what could be measured at
all. **Parameters:** a 9,000,000 hard cap on the trainable heads, at 6,988,685
after v7.4 and 6,955,405 before it (planner 1,770,247 → 1,803,527). **Disk:** 10 GB total, ever,
including transient download and extraction state — LIBERO's three suites were
downloaded, converted and deleted one at a time because all three resident at
once is "~10-12 GB, the entire project disk budget." The resulting corpus is
**6023 train episodes across 60 length-buckets and 316 val**, of which only **23
of 60 buckets carry frames** (the ~1500 LIBERO episodes; the ~4500 Bridge
episodes are frameless and proprio-less). **Compute:** an M-series laptop on MPS
and a contended shared GPU box, where identical stage-A epochs ran **96 s
uncontended vs 496 s contended (5.2x)**.

### 1.2 The outcome

Closed-loop success is **0.000 over 50 trials**. The run was well-formed — all
10 tasks completed, 0 scavenged, no failed workers — on a validated harness with
a corrected control law and a camera-corrected corpus. Previous zeros in this
project were single small runs; this one is sampled well enough to state as a
result. It crosses the project's own pre-registered kill bar for competence at
small scale (*"< 30% absolute where big models exceed 80% — then Claim 1 is
dropped"*), and we report it as such rather than reframing it.

The world model itself is not the failure point, and we say so with the same
directness. On the wrist-camera corpus, stage A reaches **val 0.0098 against a
persistence baseline of 0.0111 (+12.6%)** and **+19.8% `wm_margin`** at bench.
Weight-level probes on that checkpoint show fusion output is **47.0%**
determined by box embeddings and **43.0%** by the evidence-fade weight, and that
zeroing the grounded observation destroys **89.3%** of the TRM's predicted
residual against **65.5%** for the drift code — removing vision costs more
residual *direction* than removing drift (cos 0.634 vs 0.706). The prediction is
grounded. The planner receives that same `fused` matrix directly and weights it
at a pose-sensitivity of **0.03–0.10** — the largest single visual contribution,
and one of the few sensitivity readings stable enough to quote from a single run
(seed spread [UNVERIFIED against local artifacts]).
**Vision is available and largely discarded, not absent**, and the discarding
happens in one module. How that compares against proprioception we cannot say:
`proprio` and `geometry` vary **46–134x across seeds**, so the
phase-versus-vision ratio this project repeatedly quoted is withdrawn (§8.2).

### 1.3 The reframe: what the attempt to measure this actually produced

The interesting content of this project is not the system. It is that four
independent invalidation events, each found and documented after conclusions had
already been drawn from the affected numbers, together draw a boundary around
which measurements of a small VLA can be believed.

1. **A scale-mismatched stopping criterion made every bench metric a monotone
   function of training length** (Spearman 0.840–0.924 vs epochs survived, n =
   9, arms spanning 8–28 epochs) [UNVERIFIED against local artifacts]. An
   under-trained planner emits near-constant actions, which is exactly what a
   low `std_ratio` measures.
2. **Run-to-run variance at fixed command and fixed seed spans `std_ratio` 0.022
   to 0.245.** Pooling all five recorded samples of the same `longh`
   configuration gives mean 0.084, sd 0.097 — an **11.1x fold**. That gap is
   larger than every effect this project had claimed from an architecture or
   regularizer change. Per-input sensitivity is worse: `proprio` varies **76x**
   and `geometry` **134x** across three seeds of the `native` configuration, and
   **46x** each across three seeds of `longh` [UNVERIFIED against local
   artifacts].
3. **Four instrument defects each inverted or voided a recorded conclusion.**
   Bench never passed `spatial=` to the planner, withholding **22 of ~82
   cross-attention memory tokens (~27% of the planner's observation)** from every
   TQSA checkpoint ever scored. The sensitivity metric averaged a hard ±1
   gripper bit worth exactly **0.2857** into a statistic whose largest-ever
   reading was **0.2740**, making the two indistinguishable. Bench scored
   long-horizon waypoint predictions against native-spaced targets. And an
   unscaled synthetic probe (mean |.| ~1.0 against a real 9.0) produced the
   *opposite* conclusion about whether the world model uses vision.
4. **A silent train/deploy camera mismatch.** LIBERO was baked from
   `agentview_rgb` while eval reads `robot0_eye_in_hand_image`; the npz key is
   named `wrist_frames` either way, which is what hid it for a full session. Cost:
   one three-suite bake, one stage A, three stage-B trainings, four bench runs and
   two closed-loop evals. Not one aggregate metric flagged it — stage A converged,
   three arms trained cleanly and ranked sensibly, and bench produced coherent,
   internally consistent numbers. It surfaced only from watching the robot go to
   the basket perfectly and never pick anything up.

Defects (3) and (4) share a method lesson that we state once and rely on
throughout: **aggregate scores cannot see an interface defect, because both
sides of the interface are individually self-consistent.** Per-step telemetry
localized every one of them, usually in a single command.

The organizing thesis is therefore a **three-level reading protocol**, defined in
§2.5 and tagged inline on every number thereafter. **Level 1** — measured within
a single forward pass of a single checkpoint (two heads on shared features; an
input withheld from a fixed set of weights). **Level 2** — measured within a
single training run (a val curve; a horizon sweep on one checkpoint).
**Level 3** — measured across checkpoints from separate training runs. In this
project Level 3 was uninterpretable in every instance examined, for reasons that
were mechanical rather than statistical alone. Two questions were asked at more
than one level — text-queried spatial attention, and grounding — and in both the
across-run answer was later withdrawn while the within-run answer held.

The headline result is the cleanest illustration. Four generations (v4–v7) of
**input-side** fixes (direct box geometry, miss-hold, proprioception,
text-queried spatial attention, dream-consistent training) moved `std_ratio`
0.12 → 0.175 → 0.237 across three separate training
runs — a chain that, under (2) above, carries no information at all, since one
fixed configuration spans 0.022–0.245. In the *same forward pass* of a *single*
checkpoint, changing what is regressed moves it from 0.237 to 0.787. The
measurement that survived is not the one with the larger effect; it is the one
whose comparison never crossed a training run.

### 1.4 Contributions

1. **Target parameterization (the positive result).** Regressing metric EEF
   displacement instead of normalized action commands reduces conditional-mean
   magnitude collapse by **3.0x–29.1x across 19 benched checkpoints (median
   ~8x)** [UNVERIFIED against local artifacts], replicated at 3.3x / 4.8x / 3.3x
   on three checkpoints spanning two cameras and two supervision horizons.
   Absolute fidelity: **wp_mae 3.0 mm** over a
   ~0.2 s horizon, 4.8 mm on a second corpus, and 58.2 mm over a 2.5 s horizon.
   The ablation is exact — one auxiliary loss term, **771 params** at train time
   and **zero at inference** — and it is Level 1, which is why it survives every
   confound listed above (§3).
2. **A validity boundary for small-VLA measurement, backed by four documented
   invalidation events**, with the generalizable rule stated in §1.3 and the
   evidence in §8.
3. **A well-sampled negative closed-loop result with a partial diagnosis.**
   `mean_success` 0.000 over 50 trials, plus the diagnosis chain that preceded
   it: the v5 interface fixes (symmetric action normalization, delta-mode trust
   braking, direct box geometry, detection-miss hold, dream-consistent stage B),
   the two waypoint control-law defects, then the camera mismatch. Every
   closed-loop failure *diagnosed to date* has been an interface defect between
   individually correct components rather than a model failure; the 0/50 zero
   itself has no such diagnosis yet, and is not attributable to any one arm
   (§6, §7).
4. **Within-checkpoint weight-level localization of the grounding failure** —
   fusion output **47.0%** determined by box embeddings and **43.0%** by the
   evidence-fade weight; the TRM's predicted residual **89.3%** dependent on the
   observation against **65.5%** on the drift code — so the world model's
   prediction is grounded and the failure to use vision for *control* is
   downstream of it. Plus the demonstration that the world-model→planner message
   channel is **degenerate at the source, not ignored**: 92% a fixed vector
   (constant norm 3.315 vs varying 0.268) at effective rank **6.08 / 32**, which
   makes a planner sensitivity of 0.0006 the correct response to it rather than
   negligence (§5).
5. **A stage-A world model that beats persistence, with the margin rising with
   rollout depth to a peak at H=5 and falling back by H=8**: best val 0.0098 vs
   0.0111 (**+12.6%**) on the wrist corpus.
   Separately, sweeping horizon *within one earlier checkpoint* — the only
   version of this measurement free of the epoch/horizon confound in the training
   schedule — gives +5.5% at H=1, +10.6%, +17.3%, +18.8%, **+20.5% at H=5**,
   +19.5% at H=6 and +17.8% at H=8. Reported at n=1 per configuration, with the
   corpus, schedule and viewpoint confounds named rather than resolved (§4).
6. **Systems results under hard budgets.** Frozen-backbone map precompute at
   **~8x per epoch** (12 h → ~90 min for 40 epochs) at 7.9 GB resident, with two
   alternatives rejected after measurement and one left untaken. Device
   placement worth **16x** in closed-loop wall clock (3.75 s/step with heads on
   CPU vs 0.23 s/step on GPU), because `--device` only ever moved the detector
   that runs 1 tick in 15. A 10 GB-capped download→convert→delete pipeline. A
   catalogue of parallel-eval failure modes (§9).
7. **Pre-registration, kill bars, and a protocol fix pinned by tests.** The
   claims and bars were written before the experiments and are reported against
   verbatim, including the Claim 1 kill bar, which fired. The stopping confound
   is fixed by
   `--stage-b-select {bc,total}` (defaulting to `bc`, the only term on a scale
   shared by every arm) and `--stage-b-min-epochs`, both pinned by
   `tests/test_stage_b_selection.py`; `total` is retained solely to reproduce the
   invalid batch. Test suite **149 → 231**, CPU-only, mock-only, no network
   (§10).

### 1.5 What this paper does not claim

It **ranks no arms.** Every architecture and regularizer A/B in the project is a
single stage-B run, and §8 shows single stage-B runs do not measure
configurations. It claims **no architecture win** — including for the
text-queried spatial pathway, whose within-checkpoint head-to-head is within
noise on every action metric at **1.6x inference cost**. It reports **no
competitiveness against larger models**; the kill bar fired.

It also does not test the hypothesis the project was designed around.
**Perception-rate decoupling** — control quality bottlenecked by prediction
quality rather than perception rate, the pre-registered claim carrying the note
"THE paper" — required a 30/5/2/1/0.5 Hz sweep against a hold-last baseline
(experiment E4). **E4 was never run.** Neither were the evidence-fade ablation
(E6), the grounding ablation (E7), the recursion-depth Pareto (E8), the
bottleneck sweep (E9), the trust-AUROC figure (E5), or the Raspberry Pi
end-to-end demonstration (E10). The pre-registered claim set appears in §10 as
pre-registration, not as findings.

### 1.6 Provenance, and how to check any number

Every claim in this paper carries a level tag ([L1]/[L2]/[L3]) and appears in the
status ledger in §11 with one of: LIVE (measured, not retracted), CAVEAT (valid
only under a stated condition), SUPERSEDED (measured correctly, conclusion
withdrawn), or VOID (retracted as a measurement). Retracted claims are kept, in
labelled corrections subsections, because several of them were wrong in
instructive ways — but none reappears as a live claim.

One provenance gap is disclosed here rather than buried. `results/metrics.jsonl`
holds exactly 100 records and ends at `ts 2026-07-26T08:29:37+00:00`; the 12-arm
batch that supplies the confound statistics, the seed-spread folds, the
19-checkpoint displacement ratio and the closed-loop zero ran **08:34–11:10**, so **no record
in the local store post-dates it**. `results/PAPER_TABLE.md`, cited as that
batch's full table, is not in the repository, and none of the batch's arm tags
appear in the store. The zero is consistent with the earlier real-environment
runs that scored **0/10 on libero_object and libero_spatial** — single small
runs, which is exactly why they were not stated as a result — and with the one
closed-loop record that is in the store, which reports `mean_success` **null**
because it was voided as a policy measurement. The displacement ratio is
corroborated by four within-run pairs that *are* in the store (0.787/0.237,
0.604/0.126, 0.799/0.245, 0.654/0.071). The batch's own raw artifacts must be
recovered from the training box before publication, and are flagged as
[UNVERIFIED against local artifacts] at each point of use.
