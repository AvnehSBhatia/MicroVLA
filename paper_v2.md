# Regress Displacement, Not Actions: One Result That Survived, and Four Reasons the Others Did Not

**A micro vision-language-action stack scores `mean_success` 0.000 over 50 closed-loop
LIBERO trials; the single measurement that survives audit — a displacement head that
shrinks far less than an action head in the same forward pass — survives precisely
because its comparison never crossed a training run.**

---

### Reading conventions

*Evidence levels.* Every measurement carries a tag defined in §2.7: **[L1]** measured
within one forward pass of one checkpoint, **[L2]** within one training run, **[L3]**
across checkpoints from separate training runs.

*Provenance pointers.* References of the form **§4d**, **§4g**, **§4m** point into
`paper.md`, this project's raw lab record — **not** into this document. Section
numbers of the form §3.1, §2.4 are internal.

*Numbers have one home.* The abstract and introduction are a summary layer and restate
headline figures deliberately. Within the body (§2–§6), each figure is stated in full
once, in the section where it belongs, and cross-referenced elsewhere rather than
repeated.

*Unverified claims.* `[UNVERIFIED: ...]` marks a statement with no backing artifact in
this repository. The marker is deliberate. `[§4m, no artifact in store]` marks figures
quoted from the 12-arm overnight batch, whose raw records post-date the local metrics
store (§1.6).

---

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
`wp_std_ratio`/`std_ratio` is **3.0x–29.1x across 19 arms (median
~8x)** [§4m, no artifact in store], with individual replications of
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
`pose_mae` **−0.865**, `corr` **0.840** [§4m, no artifact in store].
**No single-run arm comparison in this project is a valid result, and none is
reported as one.** Four further instrument defects and one silent train/deploy
camera mismatch each inverted or voided a recorded conclusion. The rule that
follows, and that we recommend to anyone measuring small VLA policies:
**comparisons made within a single forward pass of a single checkpoint are
trustworthy; comparisons made across checkpoints from separate training runs are
not, absent a scale-invariant stopping criterion and >= 3 seeds.** This is a
negative-result and methodology paper with one positive core result. It ranks no
arms, claims no architecture win, and reports no competitiveness against larger
models.

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
after v7.4 and 6,955,405 before it (planner 1,770,247 → 1,803,527). **Disk:**
10 GB total, ever, including transient download and extraction state — LIBERO's
three suites were downloaded, converted and deleted one at a time because all
three resident at once is "~10-12 GB, the entire project disk budget." The
resulting corpus is **6023 train episodes across 60 length-buckets and 316 val**,
of which only **23 of 60 buckets carry frames** (the ~1500 LIBERO episodes; the
~4500 Bridge episodes are frameless and proprio-less). **Compute:** an M-series
laptop on MPS and a contended shared GPU box, where identical stage-A epochs ran
**96 s uncontended vs 496 s contended (5.2x)**.

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
and one of the few sensitivity readings stable enough to quote from a single run.
**Vision is available and largely discarded, not absent**, and the discarding
happens in one module. How that compares against proprioception we cannot say:
`proprio` and `geometry` vary **46–134x across seeds**, so the
phase-versus-vision ratio this project repeatedly quoted is withdrawn (§4.2,
Appendix A #4).

### 1.3 The reframe: what the attempt to measure this actually produced

The interesting content of this project is not the system. It is that four
independent invalidation events, each found and documented after conclusions had
already been drawn from the affected numbers, together draw a boundary around
which measurements of a small VLA can be believed.

1. **A scale-mismatched stopping criterion made every bench metric a monotone
   function of training length** (Spearman 0.840–0.924 vs epochs survived, n =
   9, arms spanning 8–28 epochs) [§4m, no artifact in store]. An
   under-trained planner emits near-constant actions, which is exactly what a
   low `std_ratio` measures.
2. **Run-to-run variance at fixed command and fixed seed spans `std_ratio` 0.022
   to 0.245.** Pooling all five recorded samples of the same `longh`
   configuration gives mean 0.084, sd 0.097 — an **11.1x fold**. That gap is
   larger than every effect this project had claimed from an architecture or
   regularizer change. Per-input sensitivity is worse: `proprio` varies **76x**
   and `geometry` **134x** across three seeds of the `native` configuration, and
   **46x** each across three seeds of `longh` [§4m, no artifact in store].
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
   named `wrist_frames` either way, which is what hid it for a full session.
   Cost: one three-suite bake, one stage A, three stage-B trainings, four bench
   runs and two closed-loop evals. Not one aggregate metric flagged it — stage A
   converged, three arms trained cleanly and ranked sensibly, and bench produced
   coherent, internally consistent numbers. It surfaced only from watching the
   robot go to the basket perfectly and never pick anything up.

Defects (3) and (4) share a method lesson that we state once and rely on
throughout: **aggregate scores cannot see an interface defect, because both
sides of the interface are individually self-consistent.** Per-step telemetry
localized every one of them, usually in a single command.

The organizing thesis is therefore a **three-level reading protocol**, defined in
§2.7 and tagged inline on every number thereafter. **Level 1** — measured within
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
   magnitude collapse by **3.0x–29.1x across 19 arms (median
   ~8x)** [§4m, no artifact in store], replicated at 3.3x / 4.8x / 3.3x
   on three checkpoints spanning two cameras and two supervision horizons.
   Absolute fidelity: **wp_mae 3.0 mm** over a
   ~0.2 s horizon, 4.8 mm on a second corpus, and 58.2 mm over a 2.5 s horizon.
   The ablation is exact — one auxiliary loss term, **771 params** at train time
   and **zero at inference** — and it is Level 1, which is why it survives every
   confound listed above (§3.1).
2. **A validity boundary for small-VLA measurement, backed by four documented
   invalidation events**, with the generalizable rule stated in §1.3 and the
   evidence in §3.3.
3. **A well-sampled negative closed-loop result with a partial diagnosis.**
   `mean_success` 0.000 over 50 trials, plus the diagnosis chain that preceded
   it: the v5 interface fixes (symmetric action normalization, delta-mode trust
   braking, direct box geometry, detection-miss hold, dream-consistent stage B),
   the two waypoint control-law defects, then the camera mismatch. Every
   closed-loop failure *diagnosed to date* has been an interface defect between
   individually correct components rather than a model failure; the 0/50 zero
   itself has no such diagnosis yet, and is not attributable to any one arm
   (§3.4, §4.1).
4. **Within-checkpoint weight-level localization of the grounding failure** —
   the fusion and TRM probe percentages of §1.2, measured on one frozen
   checkpoint (full figures in §3.2) — so the world model's prediction is
   grounded and the failure to use vision for *control* is downstream of it.
   Plus the demonstration that the world-model→planner message channel is
   **degenerate at the source, not ignored**: 92% a fixed vector (constant norm
   3.315 vs varying 0.268) at effective rank **6.08 / 32**, which makes a planner
   sensitivity of 0.0006 the correct response to it rather than negligence
   (§3.2).
5. **A stage-A world model that beats persistence, with the margin rising with
   rollout depth to a peak at H=5 and falling back by H=8**: best val 0.0098 vs
   0.0111 (**+12.6%**) on the wrist corpus.
   Separately, sweeping horizon *within one earlier checkpoint* — the only
   version of this measurement free of the epoch/horizon confound in the training
   schedule — gives +5.5% at H=1 rising to **+20.5% at H=5** and falling back to
   +17.8% at H=8 (full curve in §3.2). Reported at n=1 per configuration, with the
   corpus, schedule and viewpoint confounds named rather than resolved.
6. **Systems results under hard budgets.** Frozen-backbone map precompute at
   **~8x per epoch** (12 h → ~90 min for 40 epochs) at 7.9 GB resident, with two
   alternatives rejected after measurement and one left untaken. Device
   placement worth **16x** in closed-loop wall clock (3.75 s/step with heads on
   CPU vs 0.23 s/step on GPU), because `--device` only ever moved the detector
   that runs 1 tick in 15. A 10 GB-capped download→convert→delete pipeline. A
   catalogue of parallel-eval failure modes (§3.5).
7. **Pre-registration, kill bars, and a protocol fix pinned by tests.** The
   claims and bars were written before the experiments and are reported against
   verbatim, including the Claim 1 kill bar, which fired. The stopping confound
   is fixed by
   `--stage-b-select {bc,total}` (defaulting to `bc`, the only term on a scale
   shared by every arm) and `--stage-b-min-epochs`, both pinned by
   `tests/test_stage_b_selection.py`; `total` is retained solely to reproduce the
   invalid batch. Test suite **149 → 231**, CPU-only, mock-only, no network.

### 1.5 What this paper does not claim

It **ranks no arms.** Every architecture and regularizer A/B in the project is a
single stage-B run, and §3.3 shows single stage-B runs do not measure
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
end-to-end demonstration (E10). The pre-registered claim set is pre-registration,
not findings.

### 1.6 Provenance, and how to check any number

Every claim in this paper carries a level tag ([L1]/[L2]/[L3]) and one of the
following statuses: LIVE (measured, not retracted), CAVEAT (valid only under a
stated condition), SUPERSEDED (measured correctly, conclusion withdrawn), or VOID
(retracted as a measurement). Retracted claims are kept, in labelled corrections
subsections (§3.1 corrections and Appendix A), because several of them were wrong
in instructive ways — but none reappears as a live claim.

One provenance gap is disclosed here rather than buried. `results/metrics.jsonl`
holds exactly 100 records and ends at `ts 2026-07-26T08:29:37+00:00`; the 12-arm
batch that supplies the confound statistics, the seed-spread folds, the
19-arm displacement ratio and the closed-loop zero ran **08:34–11:10**, so **no
record in the local store post-dates it**. `results/PAPER_TABLE.md`, cited as
that batch's full table, is not in the repository, and none of the batch's arm
tags appear in the store. The zero is consistent with the earlier
real-environment runs that scored **0/10 on libero_object and libero_spatial** —
single small runs, which is exactly why they were not stated as a result — and
with the one closed-loop record that is in the store, which reports
`mean_success` **null** because it was voided as a policy measurement. The
displacement ratio is corroborated by four within-run pairs that *are* in the
store (0.787/0.237, 0.604/0.126, 0.799/0.245, 0.654/0.071). The batch's own raw
artifacts must be recovered from the training box before publication, and are
flagged **[§4m, no artifact in store]** at each point of use.

---

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
9,000,000 trainable — to a headline **32,000,000**; `paper.md` quotes the deployed stack as
"~30M deployed params", which is the same ledger with the trained totals substituted. Quote
whichever, but not as if they were the same measurement. The budget is enforced by
`tests/test_param_budget.py` and the audit, and held across every change in this document.
The v7.2-vs-v7.4 split and the per-input ablation deltas are in §3.5.

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
with `spatial=None`. Download and reconstruction sizes per suite are in §3.5.

One corpus property must travel with every number: the v7.2 three-suite bake was made from
`agentview_rgb` (rotated 180 degrees) while eval reads `robot0_eye_in_hand_image`, and the
npz key is named `wrist_frames` either way. That silent train/deploy mismatch invalidated a
bake, a stage A, three stage-B trainings, four bench runs and two closed-loop evals, and was
visible in no aggregate metric (the full cost ledger and how it surfaced are in §3.4). Every
measurement below is therefore tagged with the corpus and camera it came from.

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
mixed-corpus arms every bench metric is a monotone function of epochs survived (Spearman
0.840–0.924; the full correlation table, the per-arm epoch table and the best-val scale split
are in §3.3).

*Provenance note.* That correlation table, the seed folds and pooled spread quoted in
§2.7, and the 50-trial closed-loop run all come from the 12-arm overnight batch of `paper.md`
§4m. That batch has no backing record in `results/metrics.jsonl` — the store ends before the
batch started — and the `results/PAPER_TABLE.md` it cites is absent from the repo. The
figures are quoted from `paper.md`; their artifacts should be recovered from the training box
before publication. [§4m, no artifact in store: the entire batch.]

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

so LIBERO's OSC translation response is per-axis linear and the inversion is sound. All three
are far above the 0.5 usability threshold. A full-scale command moves ~1.1 cm/step, and a
5-step chunk spans ~5.5 cm — the scale against which the head's millimetre-level errors should
be judged. The gain file must be paired with its checkpoint exactly like `norm_stats.json`; a
gain fitted under a different action normalization is meaningless. [UNVERIFIED: §4g restates
this fit as "R^2 0.88/0.99/0.94", which matches neither the fit above nor the metrics store;
the 0.99 figure has no source.]

Two control-law defects were found here and are recorded because the mechanism generalizes:
dividing by `gain` alone over-commands by exactly the horizon h (5 at the default), and
holding the target across a whole perception period idles the arm two thirds of the time at a
15:5 ratio. Both fixed in `e362d2c`; the telemetry that localized them is in §3.4.

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
  fixed command has produced `std_ratio` anywhere in 0.022-0.245 (§3.3).
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
  contributes exactly `plan_steps*2 / (plan_steps*num_servos)` = **0.2857** to a whole-plan
  mean — the same size as the largest sensitivity ever recorded here (`state_delta`
  **0.2740**). Combined-metric readings from before that split are not comparable to
  pose-split ones.

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
planner off CPU and is worth the 16x wall-clock factor of §3.5, because `--device` only ever
moved the detector, which runs 1 tick in 15 while the d=1024 TRM runs on all of them. A
`--mock-env` path (`MockLiberoEnv`) keeps the whole harness runnable with no sim, no network
and no cv2.

The closed-loop result this harness produced — `mean_success` 0.000 over 50 trials — is
reported in full in §3.4, with its provenance gap. The one closed-loop record that does exist
in `results/metrics.jsonl` reports `mean_success` `null` and is marked VOID (the actuator
over-commanded by 5x and clipped), so it is not a competing measurement.
[UNVERIFIED: any count of how many earlier real-environment runs scored 0.0 — `paper.md`
gives no number.]

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
  interpretable at n = 1. Three independent reasons compound, all quantified in §3.3: the
  stopping-criterion confound of §2.4 (Spearman >= 0.84 between every bench metric and epochs
  survived); run-to-run spread at fixed command (an 11.1x fold in `std_ratio`); and per-input
  sensitivity seed folds up to 134x, which makes most single-run sensitivity rankings
  unreadable. `fused` pose-sensitivity is the exception that is stable across seeds. (Those
  figures carry the §2.4 provenance note.)

The practical rule the rest of the paper follows: **report L1 and L2 as measurements, and
report no L3 comparison as a result** — not until a scale-invariant stopping criterion and at
least three seeds per configuration are in place, both of which are now implemented but not
yet run.

---

## 3. Results

Results in §3.1–§3.3 carry an evidence tag from §2.7. §3.4 and §3.5 are a closed-loop
outcome and an infrastructure ledger and are untagged. §3.3 establishes why the tag is
load-bearing: in this project every L3 comparison examined turned out to be
uninterpretable, for a mechanical reason rather than a statistical one.

Provenance note. The source of record for numbers is `paper.md` and
`results/metrics.jsonl` (100 records, last entry `ts 2026-07-26T08:29:37+00:00`).
The 12-arm overnight batch of §4m ran after that last record, and its artifacts —
`results/PAPER_TABLE.md` and `logs/overnight/` — are not present in this repo. Its
numbers are reproduced verbatim from `paper.md` and are flagged **[§4m, no
artifact in store]** at first use in each subsection. §3.3 rests on §4m
throughout; §3.1 draws its 19-arm ratio range and its two `longh_tqsa` rows from
it, and §3.4 its closed-loop sample size. Every one of those is flagged in place.

### 3.1 Target parameterization: displacement regresses less shrunk than action [L1]

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
fitted gain of §2.5 that a *full-scale* command moves ~1.1 cm/step, so a 5-step chunk
spans ~5.5 cm. The 58.2 mm figure is over a 2.5 s horizon and is not comparable to the
4.8 mm figure over 0.2 s.

**Mechanism (hypothesis, not measured).** MSE converges to the conditional mean,
whose magnitude is suppressed in proportion to the irreducible noise in the target.
Human teleop action commands at 20 Hz are noisy; the positions they produce are
smooth. Same network, same features, same loss family, different target. No direct
measurement of target-noise variance exists in this project
[UNVERIFIED: any measurement of the conditional variance of the two targets].

**Why it survives §3.3.** Both heads are evaluated on one checkpoint in one forward
pass, so neither the seed spread nor the stop-timing confound can move the
ratio: whatever training length or seed produced the checkpoint produced both
numbers.

**Limits.** Open-loop and teacher-forced; bench scores the prediction and cannot see
compounding closed-loop error. Actuating the prediction requires inverting the fitted
per-axis gain of §2.5, so the claim that 0.787 of demo vigor reaches the robot is a
separate map that bench does not score (§3.4). The actuator replaces only the 3
translation dims; orientation and gripper remain on the BC head at `std_ratio` 0.237.

#### Corrections to §3.1 (retracted or suspended, retained because they are instructive)

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

### 3.2 World model [L2 for the training curve, L1 for the weight probes]

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

### 3.3 Run-to-run variance and the stopping confound — why no arm is ranked [L3]

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
**absolute** `--min-delta` of 1e-4 (§2.4). Best-val totals confirm the scale split:
longh 0.75–0.82 against native 0.61–0.65, consistent with the reported decomposition
`val bc 0.6924` + `wp 0.1107` = 0.803. Fixed by `--stage-b-select {bc,total}`
(defaulting to `bc`) and `--stage-b-min-epochs`, both pinned by
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
   hard ±1 gripper decision, at the exact magnitude collision quantified in §2.6.
   A reading that size is equally consistent with "strongly shapes the pose
   trajectory" and "flips the gripper and leaves pose untouched". Bench now reports
   pose-only |Δplan| and gripper-flip rate in separate columns; readings from the two
   builds are not comparable.
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

### 3.4 Closed loop: a well-sampled zero

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
(action, Δeef) pairs from **1500** episodes, **0** skipped) is tabulated in §2.5;
all three axes sit far above the 0.5 usability threshold, so LIBERO's OSC translation
response is per-axis linear and the inversion is sound.

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

### 3.5 Infrastructure and budgets

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

**Parameter ledger** (`microvla.utils.param_audit`). The v7.4 totals are tabulated in
§2.2. The v7.2 predecessor differs in the planner alone: planner **1,770,247**, total
**6,955,405** of **9,000,000**; fusion and drift are unchanged
[§4j, no record; re-runnable via the audit]. The headline result's head costs **771**
of them, `((d_plan+1)*3)`. Planner ablation deltas: `geometry` −1,792 ·
`pred_box_emb` −16,640 · both −18,432 · plus `next_emb` −35,072 · `spatial` →
1,720,583.

**Data pipeline under a 10 GB total cap.** LIBERO suites were downloaded, converted
and deleted one at a time because all three resident at once is ~10–12 GB, the entire
project budget: `libero_object` 5.82 GB download / 7.44 GB reconstructed / 500
episodes; `libero_spatial` 3.47 GB / 6.24 GB / ~500; `libero_goal` — / — / ~500. The
resulting corpus composition is in §2.3.

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

---

## 4. Discussion

### 4.1 What the closed-loop zero diagnoses

The subject of this section is §3.4's `mean_success` **0.000 over 50 trials**.
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
**0.913** against an action `std_ratio` of **0.072** in the same forward pass —
the `longh_tqsa` rows of §3.1. Whatever is wrong closed-loop is not a head that
predicts a constant on its training distribution. [UNVERIFIED: the batch-wide
statement that `wp_std_ratio` sits at 0.75-0.94 across the 19 benched checkpoints
— this range is quoted in `DESIGN.md` from `results/PAPER_TABLE.md`, which is not
in the repo; §4m itself quotes only the two values above.]

**2. The command was reported as near-constant in direction, with a direction
that differs between runs — and that reading cannot be checked against the
source of record.** [UNVERIFIED: the per-axis `|cmd|` means for the 0/50 run and
for an earlier run, the ratio between the strongest and the weakest axis, the
number of steps it was sustained over, and the clipped-step fraction. Those
figures appear only in `DESIGN.md` ("Motivating evidence"), which attributes
them to §4m; §4m contains no per-axis telemetry, and neither does any other part
of `paper.md` or `results/metrics.jsonl`.] What the source of record does hold
are the scalar magnitudes of §3.4 — the pre-fix VOID run's `|cmd|` mean and max,
and the corrected-law measurement. Neither is per-axis and neither is from the
0/50 run, so the constant-direction observation enters here as an unverified
input, not as a measurement.

**3. If (2) holds, the pair is the signature of exposure bias rather than
underfitting.** Every open-loop number in this paper is teacher-forced: bench
replays baked episodes, so the policy is placed back on the demo manifold at
every step and never has to survive its own error. Closed-loop, one step's error
selects the next observation. An MSE-BC head is a conditional-mean estimator;
off the support it was fit on, its output has no reason to be anything but its
mean — a fixed vector, which is exactly a constant commanded direction. The
competing story is actuator saturation, and §3.4 already names the diagnostic
that separates the two: the FRACTION of steps at the clip, not `max`. That
fraction is not reported for the 0/50 run anywhere in the source of record
[UNVERIFIED: the clipped-step fraction during the 0/50 run], so saturation is
not excluded here. Exposure bias is the leading hypothesis for the zero; it is
not established by anything in `paper.md` or `results/metrics.jsonl`.

This also bounds what the corrector can do. In `"delta"` mode low trust brakes
(`plan = min(1, tau/cfg.brake_trust) * raw`, §2.1) — it scales magnitude toward
zero and leaves direction untouched. Braking a wrong constant direction produces
a slower wrong constant direction, which is not a recovery mechanism.
[UNVERIFIED: no measurement of tau during the 0/50 run is reported anywhere in
the document; Claim 5's AUROC was never computed.]

The instrumentation consequence is another instance of this project's most
repeated lesson. `std_ratio`, `corr`, `pose_mae`, `wp_std_ratio` and `wp_mae_mm`
are all computed under teacher forcing and *cannot in principle* register
compounding error. The two defects that were caught — the 5x over-command and
the camera mismatch, both in §3.4 — were localized by per-step telemetry and by
watching the robot, never by an aggregate score, and the constant-direction
reading of (2), whatever its provenance, is per-step telemetry as well.
Aggregate scores cannot see a train/deploy interface defect because both sides
are individually self-consistent.

### 4.2 Why this benchmark under-rewards the thing that is missing

The grounding failure is not upstream. The weight-level probes of §3.2, run on
the +19.8% stage-A checkpoint, show fusion is dominated by box embeddings and the
evidence-fade weight, and that zeroing `fused` costs more of the TRM's predicted
residual — and more of its *direction* — than zeroing the drift code. Those
probes are run on one fixed checkpoint and do not depend on which stage-B run is
in front of them. §4h's conclusion — "vision is available and discarded, not
absent" — therefore locates the failure in the planner. The planner-side reading
it was drawn from (`fused` **0.0178** against **0.464** for the two phase inputs
summed, §4g) is a single run on the combined instrument §4i later showed to be
ambiguous, and rests on `proprio` and `state_delta`, which §4m measures at 46-76x
seed fold (§3.3). It fixes the location; it does not license a ratio.

Two structural reasons make discarding it the loss-minimizing choice.

*Supervision horizon.* At LIBERO's 20 Hz with `plan_steps=5`, the plan spans
**0.25 s** and the native waypoint targets **0.05-0.20 s** (§2.5). Over 0.2 s,
"keep doing what you are doing" is a near-sufficient statistic for the demo
action; object position is a second-order correction. MSE-BC consumes variance
in descending order, so it takes the phase term and leaves the vision residual.

*Task structure.* In `libero_object` the placement target is at a fixed location
in every episode; only the pick target moves. After the grasp the trajectory is
transport to the same place every time. Grounding is therefore priced into the
BC loss over a minority of each episode — the pre-grasp approach — and nowhere
else. The video signature matches exactly: **basket reached perfectly, object
never approached** (§3.4). A policy that has learned the transport phase and not
the approach phase is heavily rewarded by the training objective and scores
zero at execution, because the task is gated on the phase the loss barely
weights.

Neither reason is measured, and no phase:vision ratio is quotable in support of
them. The wrist native arm read **0.464** for phase against **0.040** for
vision, **12:1** (§4g), but §4i withdrew that whole class of reading — the
combined instrument mixes a discrete gripper bit with continuous pose, at the
magnitude collision quantified in §2.6 — and §4m then showed the ratio is
unmeasurable at n=1, spanning the seed ranges of §3.3. The two reasons above are
mechanisms proposed for the video signature and for the weight-level probes, not
conclusions read off a ratio.

Both interventions aimed at this are inconclusive, and both are single-run A/Bs
of the kind §3.1 suspended. Taxing the shortcut (`--phase-dropout 0.3`) was
recorded as moving phase:vision 2.3x with `proprio` *rising* 0.1904 -> 0.2255
and `grip_acc` collapsing **0.93 -> 0.50** (§4i) — one run per arm, on the
combined instrument, on the two inputs §4m measures at 46-76x seed fold, so none
of it is a measurement of the intervention. What carries over is the mechanism
§4i gives for the gripper collapse — `proprio` carries the arm's gripper state,
the single best predictor of the gripper command — which is why §4j replaced the
coarse flag with per-input drop rates. Removing the shortcut's sufficiency
(`--waypoint-long`, 0.5-2.5 s targets) produced an apparent inversion that three
seeds of the identical config could not reproduce (§3.3). What survives at
instrument strength is only: **`fused` pose-sensitivity is 0.03-0.10** and is
stable across seeds; its size relative to proprioception is not measurable from
one run.

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

### 4.3 Why the surviving result survives

The displacement-vs-action ratio of §3.1 is the only quantity in this document
that no confound reaches, and the reason is structural rather than lucky: both
heads read the same `feats` in the same forward pass of the same checkpoint. A
stop-timing confound shifts both numerators and denominators together; seed
spread shifts the level of each arm, not the within-pass ratio. The *levels*
move a great deal: the 11.1x fold across five runs of one identical command
(§3.3), and `wp_std_ratio` levels running from 0.604 (§4g) to 0.913 (§4m) across
different arms. None of those levels is a claim. The ratio is.

---

## 5. Limitations

* **The headline hypothesis was never tested.** Claim 2, the perception-rate
  sweep (30/5/2/1/0.5 Hz, ours vs hold-last vs oracle), is the experiment the
  paper was designed around and E4 was never run on real data. Nothing here
  bears on it.
* **Claim 1 is dropped by its own pre-registered rule.** The kill bar was
  "< 30% absolute where big models exceed 80%"; measured success is 0.000. No
  competitiveness claim against larger models is made or implied.
* **n = 1 nearly everywhere.** Exactly two configurations have three seeds. At
  n=1 the sensitivity instrument is unusable for `proprio` and `geometry`; only
  `fused` is stable (§3.3 for the folds).
* **No arm ranking is possible from any data in this document.** Arms
  early-stopped between 8 and 28 epochs and every bench metric tracks
  epochs-survived (§3.3 for the correlation table). The fix
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
  0.787 arrives at the robot only to the extent the fitted per-axis gain of §2.5
  holds, and the supporting telemetry is a single post-fix magnitude (§3.4).
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

---

## 6. Reproducing

Two environments are involved and they are not interchangeable. The **local CPU
environment** (`CLAUDE.md`) runs the test suite, the parameter audit and the
synthetic wind tunnel — no network, no sim, no cv2. The **training box**
(`scripts/overnight.sh`) runs the GPU batch that produced §3.3 and §3.4.

### 6.1 Local: tests, audit, smoke runs

Fresh setup if `.venv` is missing:

```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy pytest
# or: pip install -e ".[dev]"
```

```bash
.venv/bin/python -m pytest tests -q            # full suite (CPU-only, mocks, no network)
.venv/bin/python -m pytest tests/test_jepa_loop.py -q                  # one file
.venv/bin/python -m pytest tests/test_shapes.py::TestChronoQueryPlanner -q  # one class/test (-k works too)
.venv/bin/python -m microvla.utils.param_audit # asserts the 9M cap + per-module caps
.venv/bin/python train/train_planner.py --epochs 2 --episodes 4   # smoke train
.venv/bin/python -m eval.bench --checkpoint none --synthetic 30   # wind tunnel: <0.1s/eval, no sim
```

`param_audit` reproduces the §2.2 ledger; `pytest tests -q` reproduces the
149 → 231 test count of §3.5.

### 6.2 Training box: the overnight batch

```bash
cd /root/MicroVLA && git pull && bash scripts/overnight.sh
```

Controls and invariants of that script:

```bash
SUFFIX=_fx bash scripts/overnight.sh   # namespace every tag; do not reuse old checkpoints
MIN_EPOCHS=20                          # default; the fix for the stop-timing confound (§2.4)
tail -f logs/overnight/00_progress.log # all output goes here, not the terminal
touch STOP                             # stop cleanly, checked between steps
kill -9 -<pgid>                        # SIGTERM is trapped end-to-end; pgid printed at startup
```

Environment the script exports:

```bash
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PFX_OSMESA="PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa PYTHONPATH=/root/LIBERO"
```

Shared stage-B flags (`$COMMON`), on the frozen stage A
`checkpoints/full_stageA_wrist_v72.pt`:

```bash
--data-dir data/bridge --data-dir data/libero_v7 \
  --device cuda --batch-size 64 --lr 5e-4 --max-vram-gb 50 \
  --load-stage-a checkpoints/full_stageA_wrist_v72.pt \
  --stage-b-epochs 40 --stage-b-patience 4 \
  --stage-b-select bc --stage-b-min-epochs 20 \
  --dream-frac 0.25 --waypoint-weight 1.0
```

Phase 0 — the gain fit of §2.5, if `data/libero_v7/waypoint_stats.json` is absent:

```bash
python -m preprocess.fit_waypoint_gain data/libero_v7
```

Phase 1 — seed error bars (6 runs; each is `python train/train_batched.py $COMMON ... --tag <tag>`):

```bash
# for S in 0 1 2:
--seed $S                        --tag native_s$S
--waypoint-long --seed $S        --tag longh_s$S
```

Phase 2 — the arms, one seed each, labelled as single samples:

```bash
--waypoint-long --pre-grasp-weight 3.0                                    --tag longh_pregrasp
--waypoint-long --planner-drop-rate 'state_delta=0.4'                     --tag longh_sdfade
--waypoint-long --pre-grasp-weight 3.0 --planner-drop-rate 'state_delta=0.4' --tag longh_all
--waypoint-long --planner-input-dropout 0.0                               --tag longh_novis
--waypoint-long --tqsa                                                    --tag longh_tqsa
# different corpus (--data-dir data/libero_v7 only), otherwise the same flags:
--waypoint-long                                                           --tag longh_liberoonly
```

Phase 3 — bench every checkpoint on the pose/grip-split instrument (§2.6):

```bash
python -m eval.bench --checkpoint checkpoints/full_stageB_<tag>.pt \
  --data-dir data/libero_v7 --sensitivity --episodes 30 --device cuda:0 \
  --out eval_results/bench_<tag>.json
# the TQSA checkpoint is benched twice: once with --tqsa, once without.
```

Phase 4 — closed loop on the arm with the highest `std_ratio` at `grip_acc > 0.7`
(a selection rule, not a ranking — §3.4):

```bash
env $PFX_OSMESA python -m eval.libero_eval --suite libero_object \
  --n-trials 5 --max-steps 300 \
  --checkpoint checkpoints/full_stageB_<best>.pt \
  --norm-stats data/libero_v7/norm_stats.json \
  --waypoint-stats data/libero_v7/waypoint_stats.json \
  --device cuda:0 --heads-device cuda:0 \
  --workers 5 --stagger 10 --worker-timeout 3600
python -m eval.telemetry_probe --all
```

`--heads-device` is the 16.3x factor of §3.5; omitting it and passing `--device`
alone moves only the detector, which runs 1 tick in 15.

Phase 5 — the summary table, written to `results/PAPER_TABLE.md`. That file is the
artifact missing from this repository (§1.6, §5); regenerating it is what closes the
provenance gap.

Every step is idempotent: it skips itself if its output exists, so a re-run continues
rather than redoing the batch. The script deliberately has no `set -e` — one dead arm
must not cost the whole batch.

---

## Appendix A. Corrections

Nine recorded conclusions in this project were later withdrawn or voided. They
are listed here rather than deleted because the pattern in them *is* the
methodological result: in every case the withdrawn claim was a comparison made
across training runs or through an unvalidated instrument, and in every case
the replacement is either a within-run measurement or nothing. None of the
left-hand column is a live claim anywhere in this paper.

| # | Claim as recorded | Why it fell | What replaced it |
|---|---|---|---|
| 1 | v7 pilot bench column and the entire v7 sensitivity ranking (`std_ratio` 0.369, `grip_acc` 0.93, `corr` 0.49, `pose_mae` 0.20, `wm_margin` +1.7%) | `eval/bench.py` never passed `spatial=`; 22 of ~82 planner memory tokens (~27% of the observation) withheld from every TQSA checkpoint | No comparable replacement. `eval.bench --tqsa` added; the flagless path warns. Pilot numbers are quotable only with this caveat |
| 2 | "Ranking after three arms: waypoint > no-TQSA > TQSA" (§4c) | §4m: every bench metric is a monotone function of epochs survived (Spearman >= 0.84, n=9), and the arms ran 8-28 epochs | No ranking. The stopping criterion was fixed and the batch must be re-run |
| 3 | "TQSA carries real signal — `spatial` is the third-strongest planner input" (0.0688, §4b) | Combined instrument, cross-run comparison | Same-checkpoint head-to-head (§3.3): within noise on every action metric, at **1.6x** inference cost. Withholding `spatial` moves pose 0.0487 |
| 4 | "Vision finally dominates the planner" — phase:vision 2.0:1 (§4k, §4l) | Three seeds of the identical config gave 0.3:1, 1.0:1 and 5.5:1 | `fused` pose-sensitivity is 0.03-0.10 and stable across seeds; its ratio to proprioception is not measurable at n=1 |
| 5 | "Best arm yet" (§4l, `std_ratio` 0.245) | 0.245 is the top of a 5-sample distribution of the same command (§3.3). The paired run at the same seed gave 0.022 | No best arm. A single stage-B run is not a measurement of a configuration |
| 6 | Long-horizon action-head collapse explained by loss imbalance (target RMS ~0.163 vs ~0.047, "~12x the MSE") | A long-horizon run at weight 1.0 reported `val bc 0.6924` against `val wp 0.1107` — BC is 6x larger, so the waypoint term never dominated. The arithmetic was right and the reasoning wrong | First re-attributed to run-to-run variance (§4l), then to stop timing: the matching row in §3.3's table (`longh_s0`) early-stopped at 9 epochs |
| 7 | Long-horizon waypoint accuracy `wp_std_ratio` **3.946**, `wp_mae` **116.1 mm** (§4k) | Bench scored a 0.5-2.5 s head against 0.05-0.20 s targets; a ~10x scale mismatch reported as prediction error | Void. Re-measured after the fix: 0.799 and 58.2 mm over 2.5 s (§3.1). A test now asserts the two spacings do not report the same error |
| 8 | "The world model ignores the observation" (first §4h probe pass) | Synthetic `fused` at mean \|·\| ~1.0 against a real 9.0 — an out-of-distribution artifact | Inverted with real inputs (§3.2): zeroing `fused` destroys **89.3%** of the TRM residual, and cos 0.634 vs 0.706 says removing the observation costs more residual direction than removing drift |
| 9 | First closed-loop waypoint run (§4e); and the follow-up prediction that post-fix `\|cmd\|` would be ~0.10-0.11 with the policy "still saturating" | Actuator over-commanded by exactly the horizon (5x) and held targets across a whole perception period (2/3 idle) | Control law corrected to `error / (gain * steps_left)` (`e362d2c`); measured post-fix magnitude in §3.4. The diagnostic that distinguishes tracking from bang-bang is the *fraction* of steps at the clip, which the first analysis never computed |

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
  fixed vector at effective rank **6.08/32** (§3.2) — a near-constant input is
  absorbable into the consumer's bias, so a sensitivity of 0.0006 is the correct
  response to it, not negligence.
* **"Every closed-loop failure in this project has been an interface defect
  between correct components, not a model failure."** Recorded in §4e as true of
  every closed-loop failure up to that point, and written before `mean_success`
  0.000 was obtained on a validated harness, a corrected control law and the
  correct camera. It is no longer available as an explanation.

One correction ran the other way. The early stage-A evidence that the world
model's margin over persistence widens with rollout horizon was recorded from an
epoch table (H=1 −2%, H=3 +11%, H=4 +19%) in which horizon is confounded with
training epoch. The clean version — one fixed checkpoint
(`full_stageA_ep3_backup.pt`), horizon swept — carries no such confound and is
tabulated in §3.2 (`results/metrics.jsonl`, `kind: horizon_curve`, 40 episodes).
`paper.md` does not quote it.
