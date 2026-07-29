# MicroVLA — Paper Plan

Working document: the claims, experiments, pass/fail bars, and release goals.
The ambition level is explicit: engineer the conditions under which this could
become a *field-defining* paper, while pre-registering the honest bars that
tell us which tier we actually landed in.

## Why "Attention Is All You Need" worked — and what that demands of us

AIAYN was not a benchmark paper. It won because it had:

1. **One nameable primitive** (self-attention replacing recurrence) that
   others could lift out of the paper and reuse anywhere.
2. **A general claim**, not a system description — the primitive mattered
   beyond translation.
3. **Radical simplicity** — the reader could reimplement it in a weekend.
4. **An artifact** people actually adopted.

Mapped onto us, the primitive is **latent-rollout control**: *a control loop
does not need perception at control rate — it needs a world model good enough
to dream between measurements, a filter that knows when the dream has
diverged, and a policy trained in the dream regime it will deploy in.* The
nameable pieces: **dream ticks**, **evidence fade** (train-time dropout ≡
inference-time staleness), and the **innovation-gated corrector**.

The title should state the law, not the system. Candidates:

- *Perception Is Not the Clock: Latent-Rollout Control for Vision-Language-Action Models*
- *Dreaming Between Frames: 30 Hz Robot Control from 2 Hz Perception*
- *A 30M-Parameter VLA* (understatement framing, only if Claim 1 lands hard)

---

## The claims (ordered by scientific weight)

### Claim 2 — Perception-rate decoupling (THE paper)
Run the detector at 30 → 5 → 2 → 1 → 0.5 Hz and plot closed-loop success:

- **with** the JEPA rollout + corrector (ours),
- **without** (hold-last-observation baseline),
- **oracle** (full-rate perception; the ceiling).

Target finding: success degrades **gracefully** with the world model and
**collapses** without it → *control quality is bottlenecked by prediction
quality, not perception rate.* This is a transferable law: it applies to any
robot whose perception is slower than its actuation (which is every edge
robot). The trust telemetry gives the companion figure free: τ at measurement
k predicts failure within the next second (report AUROC).

**Pass bar (landmark tier):** at 2 Hz perception, ours ≥ 85% of its own
30 Hz-perception score while hold-last ≤ 50% of it, consistently across ≥ 3
task families. **Kill bar:** if ours and hold-last degrade identically, the
world model adds nothing — the paper is not this paper.

### Claim 1 — Competence at 1/200th the scale
LIBERO-spatial/object/goal success at ~30M deployed params vs OpenVLA (7B,
~230×), SmolVLA (450M, ~15×), TinyVLA.

**Bar for "good":** within 15–25 points of the 7B models' published numbers.
**Bar for "landmark":** parity on ≥ 1 suite. **Kill bar:** < 30% absolute
where big models exceed 80% — then Claim 1 is dropped and the paper stands on
Claims 2 + 3 (efficiency at *matched* modest success is still a result; a
gap this large is not "striking distance" and we will not spin it).

### Claim 3 — The edge demonstration
Full closed-loop stack on a Raspberry Pi 5 + AI HAT: 30 Hz control, real
perception at 2 Hz on the NPU, single-digit watts, measured end-to-end
latency per tick, vs a quantized small-VLA baseline achieving ~1–2 Hz on the
same board. Report watts/success and $/unit. Industry citations live here.

### Claim 4 — Training–inference alignment as a recipe (evidence fade)
Dream mode is not a hack bolted on at inference: box evidence is weighted by
confidence × freshness, and training-time modality dropout fades the *same*
weights. Ablation: train with binary zeroing vs evidence fade vs no dropout;
evaluate dream-window success. Target: fade > zeroing > none, with the gap
widening as perception rate drops. Generalizes to any policy that must act on
stale observations (network robots, multi-camera time-slicing).

### Claim 5 — Self-calibrating trust (the safety figure)
The corrector's error-ratio τ (a) predicts task failure before it happens
(AUROC vs post-hoc labels), (b) gates action via hold-blend, cutting
catastrophic motions during divergence at negligible success cost. "The robot
knows when its imagination is wrong" — reviewers and safety teams both cite
this.

### Claim 6 — Structure buys back scale (the bitter-lesson counterpoint)
The param ledger as an argument: a frozen open-vocab detector supplies vision
AND language grounding (its own CLIP tower, once per task); the *learned*
core is ~17M. Ablation: replace grounded dual-box inputs with raw frame
embedding only (no boxes, no geometry, no parser) at matched params →
quantify how many "free" points the structured grounding is worth.

### Claim 7 — Recursion is a compute knob at constant parameters
Unique to the weight-tied TRM: quality vs recursion depth (T, n_inner,
n_sup_infer) at *fixed* 9.5M params, plotted against wall-clock on Pi-class
hardware. Anytime inference: more think-time when the tick budget allows,
graceful shedding under load. Nobody has shown an anytime world model on a
robot control loop.

### Claim 8 — The bottleneck scaling curve
Success and rollout error vs fused-matrix size (8×5 → 32×5 → 64×5) and
state-code width (128 → 256 → 512). Either a clean scaling curve (a mini
scaling law for world-model interfaces) or a plateau proving 160 floats
suffice — both are findings.

---

## Experiment matrix

| ID | Experiment | Claim | Status |
|----|-----------|-------|--------|
| E1 | Stage A/B training on Bridge+LIBERO (running) | prereq | in progress |
| E2 | Open-loop rollout error vs persistence baseline, 1–15 ticks | 2,4 | after E1 |
| E3 | LIBERO closed-loop eval harness + success rates, 3 suites | 1 | next build |
| E4 | Perception-rate sweep ×{ours, hold-last, oracle} | 2 | after E3 |
| E5 | τ→failure AUROC + hold-blend safety ablation | 5 | free with E4 |
| E6 | Evidence-fade ablation (fade/zero/none) | 4 | 3 training runs |
| E7 | Grounding ablation (dual-box vs frame-only) | 6 | 2 training runs |
| E8 | Recursion-depth × latency Pareto (Mac + Pi) | 7 | cheap, post-E1 |
| E9 | Bottleneck sweep (8×5/32×5/64×5) | 8 | 3 training runs |
| E10 | Pi 5 + AI HAT end-to-end: Hz, watts, latency, vs quantized baseline | 3 | deferred (deploy phase) |
| E11 | Rig transfer: oracle-sim demos, then TinyVLA-teacher distill ablation | generality | after rig sim |

Priority order if compute-constrained: **E1 → E3 → E4 → E2/E5 → E6 → E8 →
E7 → E9 → E10 → E11.** E4 is the paper; everything else is supporting cast.

## Baselines (all at our data budget, honestly tuned)

- Hold-last-observation at each perception rate (the Claim-2 foil).
- Linear box-motion extrapolation (the "cheap dreamer" foil — must beat it,
  else the TRM is decoration).
- Published OpenVLA / SmolVLA / TinyVLA LIBERO numbers (cited) + quantized
  SmolVLA on-device for E10.
- ACT-style chunked BC at matched trainable params (does the world model beat
  a plain policy of the same size?).

## Release goals (adoption is what makes a paper a landmark)

- Code + weights + converted-episode recipe, one-command LIBERO reproduction.
- A **single-file reference implementation** (~500 lines: fusion + corrector
  + loop with a pluggable world model) — the "lift the primitive out" artifact.
- The Pi image + wiring doc for the full demo; a 90-second video of the
  physical rig obeying novel commands. Videos recruit citations.
- Pre-registered bars (this file, versioned) — reviewers reward it and it
  keeps us honest.

## Tier calibration (pre-registered, no self-deception)

| Outcome | Tier |
|---|---|
| E4 graceful-vs-collapse + E3 within 25 pts of 7B models + E10 | Landmark attempt: CoRL/RSS oral, arXiv splash, the AIAYN-style shot |
| E4 clean + E3 respectable (within ~35 pts) | Strong main-conference paper |
| E4 muddy but E2/E8/E10 solid | Systems/efficiency paper (ICRA) or strong workshop |
| TRM never beats persistence (E2 fails) | Stop. Diagnose or redesign; no paper spin. |

A real AIAYN-level outcome additionally requires what no plan can guarantee:
the *law* in E4 holding beyond our stack (other labs reproducing it on other
robots). The single-file artifact and pre-registered bars are how we maximize
the probability someone tries.

## Timeline hooks (auto-updated as stages complete)

- [x] Datasets streamed + converted under 10 GB cap (in progress)
- [ ] E1 Stage A world model — val spec_loss must beat persistence baseline
- [ ] E1 Stage B policy — val BC loss reported
- [ ] E3 LIBERO harness (robosuite on macOS ARM; fallback: Linux box/cloud eval)
- [ ] E4 the sweep
- [ ] Deploy phase (ONNX → Hailo → int8 QAT → E10)

## Training log & known issues (E1)

**Stage A (world model) — VALIDATED.** Bridge+LIBERO, scheduled-horizon rollout
(TRM_SPEC S5), MPS. The pre-registered "beats persistence" bar cleared, and the
margin WIDENS with horizon — the signature of a real dynamics model and direct
early evidence for Claim 2:

| epoch | H | val spec_loss | persistence | margin |
|---|---|---|---|---|
| 1 | 1 | 0.0084 | 0.0082 | -2% (parity) |
| 2 | 3 | 0.0117 | 0.0132 | +11% |
| 3 | 4 | 0.0119 | 0.0147 | **+19%** |

The pilot's pre-fix recipe (fixed multi-dream toward a single target, no
intermediate supervision) scored 2.6x WORSE than persistence — so the
scheduled-horizon data-rate objective is itself load-bearing (a free ablation
for Claim 4's family).

**Epoch-4 (H=6) interruption — RESOLVED (not a code issue).** The 4th stage-A epoch appeared to stall ~10x; root cause was the laptop LID CLOSING, which sleeps the Mac — `etime` counted ~3.9h of wall-clock sleep, not compute. No MPS/algorithm problem. Re-run under `caffeinate -s` with the lid open completes the full 1->6 curriculum. The epoch-3 checkpoint (+19% at H=4) is preserved as full_stageA_ep3_backup.pt.
## Action-interface diagnosis (E3 closed-loop, 2026-07-23) — the v5 redesign

First closed-loop LIBERO evals scored 0/10 (object AND spatial) with a
signature failure: the arm drifts diagonally and slams into the wall while the
scene sits untouched. The diagnosis chain (each step evidence-backed, all
tooling in-repo) is itself paper material — a case study in why VLA evals fail
for interface reasons before model reasons:

1. **Language exonerated** (`eval/lang_probe.py`): text embeddings AND emitted
   actions vary per instruction — no CLIP-harvest bug, language is wired.
2. **Policy collapse quantified** (`eval/replay_probe.py`): on in-distribution
   baked embeddings, teacher-forced, the planner's per-dim output std is ~8x
   smaller than the demo action std (0.03–0.09 vs 0.36–0.59) with decent
   directional correlation (~0.7) — classic MSE-BC regression-to-the-mean,
   pointing at feature starvation, not optimization failure.
3. **Drift mechanism #1 — asymmetric normalization**: quantile min-max maps a
   NEUTRAL normalized action to the (nonzero) range midpoint: a collapsed
   policy constantly commands +dx/+dy/+yaw. Fixed by symmetric re-normalization
   (`preprocess/renorm_symmetric.py`): 0 <=> zero motion.
4. **Drift mechanism #2 — trust HOLD on deltas is momentum**: `tau*raw +
   (1-tau)*last_plan` was designed for absolute PWM; for DELTA actions a held
   plan is a *continued motion*, so low trust perpetuated the drift. Fixed:
   action-space-aware trust (`cfg.action_space`), "delta" mode BRAKES (tau*raw).
5. **Feature starvation — geometry bottleneck**: the planner received box
   centers only through fusion's 160-float matrix (trained for FRAME
   prediction). For a wrist camera the target's frame position IS the
   visual-servo error vector; v5 hands (src_center, tgt_center, weights)
   to the planner directly.
6. **Perception gap at the grasp moment**: real-tick detection misses reset
   geometry to a (0.5,0.5)/weight-0 fallback exactly when the object fills or
   leaves the wrist view; v5 holds the last-known box per role at
   `miss_decay**age` weight.
7. **Train/deploy regime mismatch**: stage B trained the planner ONLY on
   real-tick features, yet at 30 Hz deployment 14/15 executed actions come from
   planner(dream features). v5 adds dream-consistent stage B (`--dream-frac`).

Known open item (needs a re-bake through the BudgetGuard pipeline): NO
PROPRIOCEPTION — the policy emits deltas with zero knowledge of arm state, so
residual drift cannot self-correct; adding baked EEF pose is the next
structural fix if v5 closed-loop numbers stay at zero.

---

# v7 / v7.2 full-data results (2026-07-25)

Everything below is measured, not projected. Where a number is a median over
episodes it says so. Two checkpoints are compared throughout:

* **v7 pilot** — `full_stageB.pt` trained on `data/bridge` + `data/libero_v7`
  (libero_object only, ~500 episodes), TQSA on.
* **v7.2 full** — trained on `data/bridge` + all three baked LIBERO suites
  (`libero_object_v7`, `libero_spatial_v7`, `libero_goal_v7`).

## 0. CORRECTION — the v7 pilot bench numbers were measured blind

`eval/bench.py` never passed `spatial=` to the planner (verified: `grep -n
spatial eval/bench.py` returned zero hits before 2026-07-25). The planner draws
22 of its ~82 cross-attention memory tokens from TQSA (16 spatial tokens + 3
pooled roles + 3 heatmaps). **Every bench number recorded for a TQSA-trained
checkpoint was therefore produced with ~27% of the planner's observation
withheld**, including:

    std_ratio 0.369 | grip_acc 0.93 | corr 0.49 | pose_mae 0.20 | wm_margin +1.7%

and the entire v7 sensitivity ranking. These figures **understate** the pilot by
an unknown amount and are NOT comparable to a checkpoint trained without TQSA.
`eval.bench --tqsa` (added 2026-07-25) runs the frozen backbone + adapter the
way the deployment loop does; the flagless path now prints a warning when the
checkpoint carries TQSA weights. Any pilot number quoted without `--tqsa` must
carry this caveat.

Related deployment-side defect, same root: `eval/policy.py` built and wired a
TQSA unconditionally and only *warned* when the checkpoint had no weights for
it — so a checkpoint trained without `--tqsa` would have fed the planner 22
tokens of RANDOM-INIT perception at eval. It now runs without the adapter.

## 1. Data — three-suite bake

Raw LIBERO, downloaded → converted → deleted one suite at a time (all three
resident at once is ~10-12 GB, the entire project disk budget):

| suite | download | reconstructed | episodes baked |
|---|---|---|---|
| libero_object  | 5.82 GB | 7.44 GB | 500 |
| libero_spatial | 3.47 GB | 6.24 GB | ~500 |
| libero_goal    | —       | —       | ~500 |

`preprocess/libero.py` globs its root recursively, so per-suite baking gives one
normalizer per suite; `preprocess/unify_norm_stats.py` rescales all three onto
the per-dim MAX of their symmetric scales (max, never mean, so nothing clips)
and writes one shared `norm_stats.json`.

Resulting training corpus: **6023 train episodes across 60 length-buckets, 316
val**. Buckets are keyed `(T, has_frames)`; **23 of 60 carry `wrist_frames`** —
those are the ~1500 LIBERO episodes; the ~4500 bridge episodes are frameless
(and proprio-less, validity flag 0), so they train planner-only with
`spatial=None`.

## 2. Stage A — world model, full mix

`--lr 5e-4 --batch-size 64 --max-horizon 6 --warmup-epochs 4 --patience 6
--lr-patience 2`, loaded nothing, 4 data dirs. Persistence baseline at H=6 is
**0.0115**.

| ep | H | lr | train | val | verdict | s | peakVRAM |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 5.0e-4 | 0.0964 | 0.0072 | (pers 0.0064) | 77 | 5.4 GB |
| 2 | 3 | 5.0e-4 | 0.0667 | 0.0120 | (pers 0.0091) | 177 | 10.1 GB |
| 3 | 4 | 5.0e-4 | 0.0654 | 0.0154 | (pers 0.0102) | 144 | 12.5 GB |
| 4 | 6 | 5.0e-4 | 0.0674 | 0.0163 | best | 218 | 17.2 GB |
| 5 | 6 | 5.0e-4 | 0.0638 | 0.0166 | 1/6 | 222 | 17.2 GB |
| 6 | 6 | 5.0e-4 | 0.0628 | 0.0157 | best | 165 | 17.2 GB |
| 7 | 6 | 5.0e-4 | 0.0596 | 0.0172 | 1/6 | 171 | 17.2 GB |
| 8 | 6 | 5.0e-4 | 0.0570 | 0.0132 | best | 168 | 17.2 GB |
| 9 | 6 | 5.0e-4 | 0.0528 | 0.0154 | 1/6 | 262 | 17.2 GB |
| 10 | 6 | 5.0e-4 | 0.0541 | 0.0133 | 2/6 | 382 | 17.2 GB |
| 11 | 6 | 5.0e-4 | 0.0539 | 0.0129 | best | 487 | 17.2 GB |
| 12 | 6 | 5.0e-4 | 0.0498 | 0.0167 | 1/6 | 496 | 17.2 GB |
| 13 | 6 | 5.0e-4 | 0.0503 | 0.0127 | best | 367 | 17.2 GB |
| 14 | 6 | 5.0e-4 | 0.0471 | 0.0133 | 1/6 | 366 | 17.2 GB |
| 15 | 6 | 5.0e-4 | 0.0453 | 0.0124 | best | 364 | 17.2 GB |
| 16 | 6 | 5.0e-4 | 0.0440 | **0.0114** | **BEATS persistence**, best | 363 | 17.2 GB |
| 17 | 6 | 5.0e-4 | 0.0431 | 0.0122 | 1/6 | 367 | 17.2 GB |
| 18 | 6 | 5.0e-4 | 0.0432 | 0.0117 | 2/6 | 367 | 17.2 GB |
| 19 | 6 | **2.5e-4** | 0.0421 | 0.0132 | 3/6 (LR halved) | 365 | 17.2 GB |
| 20 | 6 | 2.5e-4 | 0.0389 | **0.0109** | **BEATS**, best | 367 | 17.2 GB |
| 21 | 6 | 2.5e-4 | 0.0376 | 0.0112 | BEATS, 1/6 | 364 | 17.2 GB |
| 22 | 6 | 2.5e-4 | 0.0381 | 0.0115 | tie, 2/6 | 363 | 17.2 GB |

Stopped manually at 22; **best = epoch 20, val 0.0109 vs persistence 0.0115 =
+5.2% margin**, versus the pilot's 0.0106 vs 0.0111 = +4.5% at epoch 18 on
object-only. The LR halving at 19 produced the best epoch immediately at 20 —
the plateau scheduler earned its place.

**Reproducibility note:** two stage-A runs were launched (patience 3, then
patience 6). Their val curves are IDENTICAL through epoch 10 — same seed, same
data, deterministic. The patience-3 run would NOT have early-stopped: epoch 11
improved and reset the counter.

**Caveat that matters for Claim 2:** this margin is on the MIXED val split.
Bench (§4) measures −7.3% on libero_object alone. The world model wins on the
corpus and loses on that suite; the claim must be stated at the level it was
measured.

## 3. Stage B — two arms off the same frozen world model

Both `--load-stage-a checkpoints/full_stageA.pt --stage-b-epochs 40
--stage-b-patience 4 --dream-frac 0.25 --batch-size 64 --lr 5e-4`, differing
only in `--tqsa`.

| ep | no-TQSA loss / grip | no-TQSA val / grip | TQSA loss / grip | TQSA val / grip |
|---|---|---|---|---|
| 1 | — | — | 1.1456 / 0.550 | 0.9480 / 0.567 best |
| 2 | — | — | 0.9082 / 0.596 | 0.8590 / 0.717 best |
| 3 | 0.7515 / 0.659 | 0.8461 / 0.587 best | 0.7482 / 0.663 | 0.9173 / 0.547 (1/4) |
| 4 | 0.7149 / 0.673 | 0.6402 / 0.716 best | 0.7266 / 0.671 | 0.6330 / 0.730 best |
| 5 | 0.6744 / 0.685 | 0.6223 / 0.726 best | 0.6824 / 0.691 | 0.6303 / 0.740 best |
| 6 | 0.6663 / 0.698 | 0.6195 / 0.746 best | 0.6626 / 0.693 | 0.6154 / 0.761 best |
| 7 | 0.6549 / 0.709 | 0.6355 / 0.731 (1/4) | 0.6619 / 0.699 | 0.6239 / 0.730 (1/4) |
| 8 | 0.6676 / 0.697 | 0.6035 / 0.746 best | 0.6518 / 0.712 | 0.6123 / 0.726 best |
| 9 | 0.6515 / 0.711 | 0.5799 / 0.755 best | 0.6420 / 0.719 | 0.5942 / 0.743 best |
| 10 | 0.6431 / 0.709 | 0.5920 / 0.750 (1/4) | 0.6330 / 0.718 | 0.5719 / 0.757 best |
| 11 | 0.6161 / 0.725 | **0.5645** / 0.762 best | 0.6150 / 0.729 | **0.5613** / 0.773 best |
| 12 | 0.6056 / 0.738 | 0.5760 / 0.748 (1/4) | 0.6041 / 0.740 | 0.6231 / 0.742 (1/4) |
| 13 | 0.6161 / 0.730 | 0.5801 / 0.752 (2/4) | 0.6110 / 0.734 | 0.5762 / 0.761 (2/4) |
| 14 | 0.6016 / 0.730 | 0.5713 / 0.762 (3/4) | — | — |
| 15 | 0.5984 / 0.735 | 0.5800 / 0.731 (4/4) | — | — |

no-TQSA early-stopped at 15, **best val 0.5645 (ep 11), val grip 0.762**.
TQSA best so far **0.5613 (ep 11), val grip 0.773** — a **0.6% val improvement
at the same epoch**. Since bridge is ~75% of episodes and carries no frames,
TQSA can only affect a quarter of that val, so 0.6% overall is ~2.4% on the
part it touches. Trainer val is also not the metric of record; see §4.

## 4. Bench — v7.2 no-TQSA arm vs v7 pilot

`eval.bench --data-dir data/libero_object_v7 --sensitivity`, 30 episodes,
0.81 s/eval. Medians:

| metric | v7 pilot (object-only) | v7.2 full (no TQSA) |
|---|---|---|
| std_ratio | 0.369 | **0.175** |
| pose_mae | 0.20 | 0.190 |
| corr | 0.49 | **0.28** |
| grip_acc | 0.93 | 0.93 |
| wm_margin | +1.7% | **−7.3%** |

Per-episode spread on the v7.2 arm: std_ratio 0.131–0.238, grip 0.83–1.00,
wm_margin −20.0% to +1.6%. Both pilot columns carry the §0 caveat.

Three readings:

1. **Magnitude collapse regressed.** std_ratio 0.369 → 0.175, with corr
   0.49 → 0.28. Three suites is a harder conditional-mean problem than one, and
   MSE-BC answers diversity by shrinking. This is the direct motivation for the
   waypoint-absolute head (DESIGN.md v7.2) — it removes magnitude from the
   regression's job entirely.
2. **The world model's channel into control has gone silent.** `wm_msg`
   0.031 → 0.0007 (44x down). With `pred_box_emb` 0.0029 and `next_emb->cur`
   0.0017, nothing the TRM produces measurably changes the plan.
3. **grip_acc held at 0.93** despite the trainer reporting 0.73 — the trainer
   averages over bridge (different robot, different gripper convention); bench
   scores LIBERO only.

### Planner input sensitivity (mean |Δplan| when withheld, on-distribution)

| input | v7 pilot | v7.2 full | change |
|---|---|---|---|
| proprio | 0.291 | 0.2243 | −23% |
| geometry | 0.004 | **0.0914** | **+2185%** |
| fused | 0.023 | 0.0218 | ~flat |
| state_delta | 0.075 | 0.0134 | −82% |
| current_emb | 0.025 | 0.0133 | −47% |
| next_emb->stale | — | 0.0059 | new probe |
| pred_box_emb | 0.013 | 0.0029 | −78% |
| next_emb->cur | 0.001 | 0.0017 | ~flat |
| wm_msg | 0.031 | 0.0007 | **−98%** |

**The pruning candidate from the pilot is refuted — but conditionally, and the
condition is the point (see §4b).** `geometry` was the deadest input at 0.004
and is 0.0914 here, the second-strongest. Pruning it on pilot evidence would
have removed the input this configuration relies on most after proprioception.
§4b shows why the two readings disagree: geometry and TQSA are SUBSTITUTES, and
the pilot had TQSA while this arm does not. `pred_box_emb` and the `next_emb` path remain
dead, and `next_emb->stale` (0.0059) confirms it at full magnitude: substituting
a *wrong* prediction of the same size barely moves the plan, so the low
`next_emb->cur` reading is not merely an amplitude artifact.

Methodological note: `next_emb->cur` zeroes only the TRM's residual, which is
small next to `‖current_emb‖`, so it reads low by construction. `next_emb->stale`
(the previous tick's prediction — full magnitude, in-distribution, wrong) was
added to separate "the path is dead" from "the perturbation was tiny".

## 4b. Bench — TQSA arm, scored WITH spatial (the first honest TQSA reading)

`eval.bench --data-dir data/libero_object_v7 --sensitivity --tqsa`, 30 episodes,
1.30 s/eval (vs 0.81 without the backbone). Checkpoint = `full_stageB_full.pt`,
epoch 11 (val 0.5613); the run was interrupted at epoch 13, so this is the
best-val checkpoint, not an early-stopped one.

| metric | v7 pilot (blind) | v7.2 no-TQSA | v7.2 TQSA (with spatial) |
|---|---|---|---|
| std_ratio | 0.369 | 0.175 | **0.120** |
| pose_mae | 0.20 | 0.190 | 0.197 |
| corr | 0.49 | 0.28 | **0.35** |
| grip_acc | 0.93 | 0.93 | **0.52** |
| wm_margin | +1.7% | −7.3% | −7.3% |

`wm_margin` is identical across both v7.2 arms, as it must be: it measures the
stage-A world model on a latent rollout and never touches the planner. It is a
useful internal consistency check on the harness.

### Sensitivity — the substitution result

| input | v7 pilot (had TQSA, scored blind) | v7.2 no-TQSA | v7.2 TQSA |
|---|---|---|---|
| proprio | 0.291 | 0.2243 | **0.3492** |
| state_delta | 0.075 | 0.0134 | **0.0994** |
| **spatial** | not measurable | n/a | **0.0688** |
| fused | 0.023 | 0.0218 | 0.0248 |
| pred_box_emb | 0.013 | 0.0029 | 0.0204 |
| current_emb | 0.025 | 0.0133 | 0.0092 |
| next_emb->stale | — | 0.0059 | 0.0059 |
| **geometry** | 0.004 | **0.0914** | **0.0041** |
| next_emb->cur | 0.001 | 0.0017 | 0.0017 |
| wm_msg | 0.031 | 0.0007 | 0.0016 |

**Result 1 — TQSA carries real signal.** `spatial` at 0.0688 is the third
strongest input, an order of magnitude above the dead paths. This is the first
measurement of TQSA's contribution that has ever existed in this repo; before
2026-07-25 bench could not pass it at all (§0).

**Result 2 — geometry and TQSA are SUBSTITUTES, not complements.** Turn TQSA on
and `geometry` collapses 0.0914 -> 0.0041 (22x down) while `spatial` takes up
0.0688. Both encode "where in the wrist frame is the object": raw box centers
versus text-queried attention maps. The planner routes through whichever is
richer and abandons the other. This retro-explains the pilot's `geometry`
0.004 — the pilot had TQSA. The two readings were never in conflict; they are
the same input measured in two different configurations, and neither alone
licenses a pruning decision. Concretely: `--planner-drop geometry` is safe with
TQSA on and destructive with it off.

**Result 3 — proprio and state_delta strengthen with spatial present** (0.2243
-> 0.3492 and 0.0134 -> 0.0994). With a spatial channel to locate the target,
the planner leans harder on knowing where the arm IS and how far the task has
progressed. The world model's channel stays dead either way (`wm_msg` 0.0016).

### The gripper regression

**grip_acc 0.93 -> 0.52 is at chance.** Per-episode: most sit at 0.44/0.47/0.50/
0.53 with a handful at 0.93-1.00; the no-TQSA arm was uniformly 0.83-1.00.

The trainer disagreed — TQSA val grip 0.773 vs no-TQSA 0.762 — and the reason is
the same dilution that hides everything else in this corpus: bridge is ~75% of
episodes, carries no frames, and therefore trains the gripper head with
`spatial=None`. The trainer's val averages a gripper that works (bridge, no
spatial) with one that does not (LIBERO, with spatial). **Trainer val grip is
not a usable proxy for LIBERO gripper behaviour in a mixed-corpus run.**

Verification still owed: bench the SAME checkpoint without `--tqsa`. If grip
returns to ~0.93, feeding spatial is what breaks the gripper (a real property
of the adapter). If it stays ~0.52 the head is broken independent of the input,
and the newly-written bench `--tqsa` path is exonerated.

**Net: as configured, TQSA is not worth it.** It buys corr (+0.07) and costs
std_ratio (-0.055) and the gripper (-0.41). The gripper is the binary that
decides whether a pick-and-place task can succeed at all, so a chance-level
gripper is disqualifying regardless of the other metrics.

## 4c. Bench — waypoint arm. The auxiliary loss is the result, not the head.

`full_stageB_wp.pt`, epoch 18 (val 0.5606 = BC + waypoint, grip 0.773), trained
off the same frozen stage A with `--waypoint-weight 1.0`, no TQSA.

**This bench measured the plain BC action head with the waypoint head ABSENT**
(bench built the planner from DEFAULT_CONFIG and dropped `wp_disp_head`; fixed
in `fb5f5df`). That accident is informative: `wp_disp_head` feeds only the `wp`
output and never the plan, so these numbers isolate what the AUXILIARY LOSS did
to the action head, with zero contribution from the waypoint actuation path.

| metric | pilot (blind) | no-TQSA | TQSA | **waypoint** |
|---|---|---|---|---|
| std_ratio | 0.369 | 0.175 | 0.120 | **0.237** |
| pose_mae | 0.20 | 0.190 | 0.197 | **0.180** |
| corr | 0.49 | 0.28 | 0.35 | **0.38** |
| grip_acc | 0.93 | 0.93 | 0.52 | **0.93** |
| wm_margin | +1.7% | -7.3% | -7.3% | -7.3% |

Per-episode std_ratio 0.176-0.355; grip 0.81-1.00; 1.47 s/eval.

**Result 1 — the auxiliary alone recovers 35% of the magnitude collapse.**
std_ratio 0.175 -> 0.237 at identical architecture, identical data, identical
frozen world model; the ONLY difference is that the planner was trained beside
a loss asking it to predict metric end-effector displacement. corr and mae
improve together (0.28 -> 0.38, 0.190 -> 0.180) and the gripper is untouched at
0.93, so this is not a magnitude/accuracy trade. Cost: 771 parameters at train
time, ZERO at inference.

This is a different claim from the one the head was built for. The design
argument was about ACTUATION — command a proportional move toward a predicted
position so magnitude stops depending on the regression's amplitude. What is
measured here is REPRESENTATION: asking the network to also predict where the
arm will be reduces conditional-mean shrinkage in the action head itself.
Predicting positions is a better-conditioned target than predicting noisy teleop
commands, and the shared trunk inherits that. The actuation claim remains
untested (needs `wp_std_ratio`/`wp_mae_mm` from a bench built after fb5f5df).

**Result 2 — the world model's channel into control comes back from the dead.**
`wm_msg` 0.0007 -> **0.2394**, a 340x jump to the STRONGEST input, ahead of
proprio (0.1747).

| input | no-TQSA | TQSA | waypoint |
|---|---|---|---|
| **wm_msg** | 0.0007 | 0.0016 | **0.2394** |
| proprio | 0.2243 | 0.3492 | 0.1747 |
| state_delta | 0.0134 | 0.0994 | 0.0561 |
| current_emb | 0.0133 | 0.0092 | 0.0193 |
| fused | 0.0218 | 0.0248 | 0.0137 |
| next_emb->stale | 0.0059 | 0.0059 | 0.0097 |
| pred_box_emb | 0.0029 | 0.0204 | 0.0053 |
| geometry | 0.0914 | 0.0041 | 0.0048 |
| next_emb->cur | 0.0017 | 0.0017 | 0.0029 |
| spatial | n/a | 0.0688 | n/a |

`msg_head` is the one TRM component left trainable in stage B (the freeze policy
lets the planner's gradient shape it). Under a pure action-BC loss it received
nothing useful and went silent; under the waypoint loss it became the planner's
primary input. Predicting where the arm WILL BE requires scene dynamics in a way
predicting the next action command does not, so the world model finally has a
job the policy needs done. `geometry` returns to 0.0048 — whatever spatial
content the planner requires now arrives through `msg`.

This is the first configuration in which the world model demonstrably
contributes to CONTROL rather than only to frame prediction, which is the gap
between Claim 2 (perception-rate decoupling) and a world model that earns its
place in the loop.

**Caveat:** `wm_margin` is -7.3% in all three v7.2 arms — the stage-A model is
unchanged and still loses to persistence on libero_object. So "the world model
helps control" and "the world model predicts frames better than persistence" are
currently BOTH true and independent, on this suite in opposite directions. Do
not merge them into one claim.

**Ranking after three arms:** waypoint > no-TQSA > TQSA. The waypoint arm is
best on every metric measured, and TQSA is disqualified by its chance-level
gripper.

## 4d. THE RESULT — positions regress 3.3x less shrunk than actions

Same checkpoint (`full_stageB_wp.pt` ep18), same 30 episodes, same forward pass,
now with `wp_disp_head` actually loaded (bench rebuilt post-`fb5f5df`). The BC
metrics are bit-identical to §4c, confirming the head never touches the plan.

| quantity regressed | std_ratio | error |
|---|---|---|
| normalized ACTION (BC head) | 0.237 | pose_mae 0.180 (normalized) |
| metric DISPLACEMENT (waypoint head) | **0.787** | **wp_mae 3.0 mm** |

Healthy is ~1.0. The action head is **76% shrunk**; the waypoint head is **21%
shrunk**. 3.0 mm against the ~10-20 mm the end-effector covers per 5-step chunk
means the head is tracking the trajectory, not predicting a well-conditioned
nothing.

**The magnitude collapse was never an observability problem.** The v4-v7
diagnosis chain (paper.md "Action-interface diagnosis") attacked the planner's
INPUTS every time — direct geometry (v5), miss-hold (v5), proprioception (v6),
text-queried spatial attention (v7), dream-consistent training (v5) — and moved
std_ratio 0.12 -> 0.175 -> 0.237. Changing WHAT IS REGRESSED, at fixed
architecture, data, world model and inputs, moves it to 0.787 in one step.

The mechanism is target noise, not model capacity. MSE converges to the
conditional mean, whose magnitude is suppressed in proportion to the
irreducible noise in the target. Human teleop ACTION commands at 20 Hz are
noisy; the POSITIONS they produce are smooth and near-deterministic. Same
network, same features, same loss family — a target with less unpredictable
variance shrinks less.

This is the paper's cleanest claim because the ablation is exact: one loss term
(771 params, zero at inference) separates 0.237 from 0.787 on identical
everything else, and §4c shows the auxiliary ALSO drags the action head up
(0.175 -> 0.237) through the shared trunk.

**Caveats, stated plainly.** (i) Open-loop and teacher-forced, like every bench
number — necessary, not sufficient, and it cannot see compounding closed-loop
error. (ii) Actuating it requires inverting a fitted per-axis gain
(`preprocess/fit_waypoint_gain.py`); the commanded vigor is 0.787 only to the
extent that fit holds, so its per-axis R2 is now a load-bearing number. (iii)
`wm_margin` remains -7.3%: the stage-A world model still loses to persistence on
this suite regardless of any of the above.

## 4e. First closed-loop run with waypoint actuation — INVALID, and why

**No `mean_success` is reported here.** The runs below executed correctly as
software and are void as a measurement of the policy: the actuator was
commanding ~5x too hard and clipping. Recorded because the harness validation
and the defect are both results.

### The fitted gain (the actuation path's load-bearing number)

`preprocess/fit_waypoint_gain.py` over all three suites, 82,844 (action, Δeef)
pairs from 1500 episodes, 0 skipped for missing proprio:

| axis | gain (m per unit action per step) | R² |
|---|---|---|
| x | 0.01056 | 0.870 |
| y | 0.01200 | 0.938 |
| z | 0.01085 | 0.866 |

All far above the 0.5 usability threshold, so LIBERO's OSC translation response
really is per-axis linear and the actuator's inversion is sound. A full-scale
command moves ~1.1 cm/step, so a 5-step chunk spans ~5.5 cm — which is the
scale that makes the head's 3.0 mm prediction error small.

### Harness validation (§0 closed out)

The serial canary ran clean on the box: every heartbeat printed
(`building policy` -> `policy ready (6s)` -> `10 task(s)` -> per-trial
`START`/`DONE`), the env built, 100 steps executed, no stall. The 10-worker hang
that opened this session is not reproducible under the rebuilt harness, and the
`--workers 5 --stagger 10` run sharded and reported normally.

`success=False` at `--max-steps 100` is **by construction**, not a signal: 100
steps is 5 s of robot time at 20 Hz and a LIBERO pick-and-place needs 150-300.

### `--heads-device` is worth 16x

| configuration | s/step | 300-step episode |
|---|---|---|
| heads on CPU (the previous default, always) | 3.75 | 375 s |
| `--heads-device cuda:0`, 5 workers | 0.23 | ~70 s |

The d=1024 TRM, fusion and planner run on EVERY tick while the detector runs 1
in 15, and `--device` only ever moved the detector. At `--n-trials 20
--max-steps 300` this is the difference between ~60 hours and under 4.

### The defect: telemetry caught it in one command

Per-step telemetry over a completed 300-step episode:

    steps 300 | waypoint cmds 300
    |cmd| mean 0.5301  max 1.0000
    plan_norm mean 2.418

300 of 300 steps commanded — the actuation path fires — but `max 1.0` is the
clip and a mean of 0.53 is several times the demo's per-step magnitude under
the unified normalization. Two stacked bugs in the control law:

1. **Units (5x over-command).** `gain` is metres per unit action per ONE control
   step; a horizon-*h* waypoint spans *h* steps. The command divided by `gain`
   alone, over-commanding by exactly *h* (5 at the default horizon). The correct
   command is the per-step RATE closing the remaining error over the remaining
   steps, `error / (gain * steps_left)`, with `steps_left` counting down — which
   is where the closed-loop property actually lives: an arm that falls behind
   has the same error to cover in fewer steps, so the command GROWS.
2. **Duty cycle (2/3 idle).** The target was held across a whole perception
   period. That is only valid when the period is no longer than the horizon; at
   the deployment 15:5 ratio the arm reaches the held target in ~5 ticks and
   idles for 10. Now re-anchored every tick — the planner replans every tick
   with fresh proprio regardless, and the countdown preserves the correction.

Together these predict exactly the observed profile: saturate for ~5 ticks,
coast for 10. Fixed in `e362d2c`.

**Measured after the fix: `|cmd|` mean 0.4533** (was 0.5301), verified as the
corrected law by a direct probe — a 3 cm / 5-step waypoint yields `cmd_x =
0.568` under `error/(gain*steps)` where the old law asked for 2.84 and clipped
to 1.0. Working backwards, 0.4533 implies the head predicts ~2.4 cm over 5
steps (4.8 mm/step), i.e. an equivalent per-step action of ~0.45 against a demo
magnitude of ~0.58 — the 0.787 ratio of §4d, arriving intact at the actuator.

A prediction of ~0.10-0.11 was recorded here before that measurement and was
WRONG: it came from a guessed demo action magnitude, not a measured one. The
same guess briefly produced a false "still saturating" diagnosis of the
corrected run, on the strength of `max 1.0` alone. `max` is the largest command
in the episode and clips on the largest moves by design; the diagnostic that
actually distinguishes tracking from bang-bang is the FRACTION of steps at the
clip, which the first version of this analysis never computed.

**Method note worth keeping.** The bench numbers (§4d) could not have caught
this: bench scores the PREDICTION, and the actuator is a separate map from
prediction to command that only closed-loop exercises. The per-step
`waypoint_cmd` telemetry — one field, one command — localized it in seconds.
Every closed-loop failure in this project so far has been an interface defect
between correct components, not a model failure, and the instrument that finds
them each time is per-step telemetry rather than the aggregate score.

### Still open

`mean_success` remains unobtained. It is now blocked on nothing but a re-run:
the harness works, the policy is the best-benching arm (std_ratio 0.237,
grip 0.93, wp_std_ratio 0.787), the gain fits at R² 0.87-0.94, and the control
law is corrected. Note the ceiling that will remain even then: the actuator
replaces only the 3 TRANSLATION dims. Orientation and the gripper still come
from the BC head at std_ratio 0.237, so 4 of 7 dims are untreated. A failure to
align the wrist or close the fingers is the expected NEXT bottleneck, not
evidence against the translation result.

## 4f. ROOT CAUSE — the corpus was baked from the wrong camera

Every v7.2 number above was produced by a policy trained on a viewpoint it never
sees at deployment.

* `preprocess/libero.py --camera` defaulted to **`agentview_rgb`** — the fixed
  third-person view, rotated 180° (robosuite renders it flipped).
* The v7.2 bake command passed no `--camera`, so all three suites were baked
  from agentview.
* `eval/libero_eval.py` reads **`robot0_eye_in_hand_image`** — the WRIST camera.
* The npz key is named `wrist_frames` whatever `--camera` says, which is what
  made the mismatch invisible for an entire session.
* `eval/RUNBOX_EVAL.md` recorded the correct setting all along ("training used
  LIBERO `eye_in_hand_rgb` un-rotated"), so the v7 PILOT was baked on the wrist
  view and the v7.2 re-bake silently reverted to the default.

### This explains every anomaly, including the ones that looked like results

| observation | explanation under the camera mismatch |
|---|---|
| `geometry` 0.0048, `fused` 0.0137 — vision near-dead | box centers from a viewpoint that does not correspond to the deployed one are noise; the planner correctly learned to ignore them |
| `proprio` 0.1747 + `wm_msg` 0.2394 dominate | proprioception is the ONLY input consistent across the mismatch, so the policy became a proprio-conditioned trajectory prior |
| basket reached perfectly, object never approached | the basket is at a fixed location and needs no grounding; the object needs exactly the grounding that was destroyed |
| gripper closes at step 0, 71/300 steps closed | the visual phase signal is from the wrong view, so grasp timing has nothing to lock onto |
| `wm_margin` +1.7% (pilot, wrist) -> **-7.3%** (v7.2, agentview) | a FIXED third-person camera is nearly static frame-to-frame, so persistence is a very strong baseline; a WRIST camera moves with the arm, so persistence is weak and prediction is worth something. The sign flip is a property of the camera, not of the model. |
| bench `std_ratio`/`corr`/`grip_acc`/`wp_std_ratio` all self-consistent | bench replays the SAME baked agentview episodes, so it is internally valid and externally meaningless for a wrist-camera deployment |

**What survives.** The measurements are not wrong, they are answers to a
different question: agentview-in, agentview-out. So the §4d result — regressing
metric displacement shrinks 3.3x less than regressing normalized actions
(0.787 vs 0.237), and the auxiliary alone lifts the action head 0.175 -> 0.237 —
is a claim about TARGET PARAMETERIZATION and is independent of viewpoint. The
§4b substitution result (TQSA displaces geometry) likewise concerns two channels
carrying the same content, whichever view supplies it. The stage-A curve, the
cache measurement, the harness validation, the gain fit (R² 0.87-0.94, fitted
from actions and proprio, no pixels) all stand.

**What does not.** `wm_margin`, every grounding-dependent sensitivity reading,
and of course `mean_success`. Those must be re-measured after a wrist-camera
re-bake before any of them means anything.

### Fixed so it cannot recur

`--camera` is now REQUIRED with an explicit choice, `rotate_180` derives from it
(agentview flipped, wrist upright), the chosen camera is logged at bake time,
and anything other than `eye_in_hand_rgb` warns that it is not the view the eval
reads. The silent default cost one full three-suite bake, one stage A, three
stage-B trainings, four bench runs and two closed-loop evals.

**Method lesson, and it is the same one as §4e.** Not one aggregate metric
flagged this. Stage A converged, three stage-B arms trained cleanly and ranked
sensibly, bench produced coherent internally-consistent numbers, and the
best-benching arm was correctly identified. The mismatch surfaced only from
WATCHING THE ROBOT — "it goes to the basket perfectly and never picks anything
up" — and then reading the sensitivity table as a statement about which inputs
the policy could physically be using. Aggregate scores cannot see a train/deploy
interface defect, because both sides are individually self-consistent.

## 4g. Wrist-camera rerun — the world model result, and grounding still absent

Same corpus as the v7 pilot (`data/bridge` + `data/libero_v7`, libero_object,
wrist camera, 500 episodes), `--waypoint-weight 1.0`, no TQSA. So this is the
camera-correct replication of §4c/§4d, and it is directly comparable to the
pilot on data but not on architecture (the pilot had TQSA, this does not).

### Stage A — the best world model this project has trained

Persistence 0.0111 (identical to the pilot's, confirming the same corpus).
Best val **0.0098 at epoch 34** under `--patience 6 --lr-patience 2`, LR decayed
to 1.3e-4, still at 0.0097 when patience expired.

| run | val | persistence | margin |
|---|---|---|---|
| v7 pilot (wrist, patience 3, stopped ep 18) | 0.0106 | 0.0111 | +4.5% |
| v7.2 agentview (3 suites + bridge) | 0.0109 | 0.0115 | +5.2% |
| **v7.2 wrist (patience 6, ep 34)** | **0.0098** | 0.0111 | **+12.6%** |

The pilot was ALREADY wrist-baked, so this ~8-point gain over it is NOT the
camera — it is the training schedule. `--patience 3` stops stage A on the same
epoch the LR halving fires, so the schedule never acts (see §6.7).

### Bench

| metric | pilot (wrist, TQSA, scored blind) | agentview wp | **wrist wp** |
|---|---|---|---|
| `std_ratio` | 0.369 | 0.237 | **0.126** |
| `wp_std_ratio` | — | 0.787 | **0.604** |
| `wp_mae_mm` | — | 3.0 | **4.8** |
| `pose_mae` | 0.20 | 0.180 | 0.243 |
| `corr` | 0.49 | 0.38 | 0.31 |
| `grip_acc` | 0.93 | 0.93 | 0.93 |
| **`wm_margin`** | +1.7% | −7.3% | **+19.8%** |

**Result 1 — the camera explanation is confirmed quantitatively.** `wm_margin`
+1.7% (pilot, wrist) → −7.3% (agentview) → **+19.8%** (wrist, better-trained).
A FIXED third-person camera is nearly static frame-to-frame, making persistence
a very strong baseline; a WRIST camera moves with the arm, so persistence is weak
and prediction earns its keep. The sign flip and its magnitude are properties of
the viewpoint, and this is the cleanest support for Claim 2 to date — larger than
the +11-13% previously recorded on the older mix.

**Result 2 — the target-parameterization claim replicates and strengthens.**
Regressing metric displacement vs regressing normalized actions: **0.604 vs
0.126, a 4.8x ratio**, against 3.3x (0.787 vs 0.237) on agentview. The absolute
`wp_std_ratio` is lower and `wp_mae` slightly worse (4.8 mm vs 3.0), but the
relative claim — the collapse is a property of the TARGET, not the inputs — holds
on a second corpus with a different world model and a different camera. Note the
action head collapsed HARDER here (0.126), so at deployment the waypoint
actuation is carrying translation at ~5x the vigor the BC head would.

**Result 3 — grounding is still essentially absent, and this is now the
bottleneck.**

| input | agentview wp | wrist wp | change |
|---|---|---|---|
| `state_delta` | 0.0561 | **0.2740** | 4.9x up |
| `proprio` | 0.1747 | 0.1904 | ~flat |
| `geometry` | 0.0048 | 0.0218 | 4.5x up |
| `fused` | 0.0137 | 0.0178 | ~flat |
| `current_emb` | 0.0193 | 0.0132 | down |
| `pred_box_emb` | 0.0053 | 0.0125 | up |
| `next_emb->stale` | 0.0097 | 0.0031 | down |
| `wm_msg` | 0.2394 | **0.0006** | 400x DOWN |
| `next_emb->cur` | 0.0029 | 0.0006 | down |

PHASE signals (`state_delta` + `proprio`) sum to **0.464**; VISION (`geometry` +
`fused`) sums to **0.040**, a 12:1 ratio. Grounding did improve 4.5x with the
correct camera, but the policy remains a phase-conditioned trajectory prior:
it knows how far through the task it is and where its arm is, and almost nothing
about where the object is. That is the same mechanism §4f diagnosed from the
video (basket reached, object never approached), and the camera fix alone did not
resolve it.

**Result 4 — `wm_msg` died again (0.2394 → 0.0006), while the world model got
much better.** The two are independent: `wm_margin` +19.8% says the TRM predicts
frames well; `wm_msg` 0.0006 says the planner ignores its 32-d readout entirely.
A better world model did NOT buy a better control signal. §4c's revival of
`wm_msg` under the waypoint loss therefore does not reproduce here, and the
distinguishing variable is unknown — candidates are the corpus (3 suites +
bridge vs object + bridge) and the far stronger stage-A model this planner was
frozen against. Recorded as unexplained rather than smoothed into a trend.

### What this predicts for closed-loop

Translation is actuated at ~0.6 vigor with 4.8 mm accuracy and a gain fitted at
R² 0.88/0.99/0.94, so the arm should MOVE properly. But with vision at 4% of the
planner's sensitivity it has little basis for going to the right place, so
approach failure remains the expected mode — and the next lever is whatever
raises grounding, with TQSA the obvious candidate (`spatial` measured 0.0688,
third strongest, in §4b).

## 4h. Weight-level analysis of the stage-A checkpoint — where vision dies

Probes run directly on `checkpoints/full_stageA_wrist_v72.pt` (the +19.8%
world model), measuring the TRAINED modules' functional sensitivity to each
input rather than inspecting weights in the abstract. Fixes the location of the
grounding failure to ONE module.

**Method caveat, recorded because it inverted a conclusion.** A first pass fed
the TRM randomly-generated `fused` with mean |.| ~= 1.0. Real fusion output has
mean |.| = **9.0**. At 1/9 scale the TRM looked almost insensitive to `fused`
(0.7% of next_emb), supporting a "the world model ignores the observation"
reading. With REAL fused that reading is WRONG — see below. Synthetic probe
inputs must be scale-matched to the module that produces them, or the measured
sensitivity is an out-of-distribution artifact.

### Fusion is vision-rich

Change in `fused` (mean |dfused| as % of mean |fused|), trained weights:

| perturbation | change |
|---|---|
| box embeddings (new draw) | **47.0%** |
| `box_weight` -> 0 (full evidence fade) | **43.0%** |
| frame embedding | 38.3% |
| last action | 31.5% |
| text tokens | 13.8% |
| box centers | 12.0% |

Nearly half of `fused` is determined by box evidence. The evidence-fade path is
alive and strong (43%), so the v5 design claim holds at the weight level.

### The TRM uses vision too

Sensitivity measured against the RESIDUAL the TRM actually predicts
(`||next_emb - cur|| / ||cur||` = 0.0366), which is the honest denominator —
next_emb is dominated by the residual convention `cur + delta`:

| perturbation | change in residual | change in msg |
|---|---|---|
| `fused` -> 0 | 89.3% | 9.5% |
| `state_delta` -> 0 | 65.5% | 6.6% |
| box_weight -> 0 (faded) | **40.6%** | 4.4% |
| box embeddings (new draw) | **38.5%** | 4.7% |
| box centers (new draw) | 8.0% | 0.9% |

And by direction: `cos(residual, residual | fused=0)` = **0.634** vs
`cos(residual, residual | state_delta=0)` = **0.706**. Removing the grounded
observation destroys MORE of the residual direction than removing the drift
code. The world model is not drift-dominated, and its +19.8% margin (4g) is
grounded prediction — which strengthens Claim 2 rather than qualifying it.
Contrast the v6 probe that found 0.63-0.88 of the residual drift-explained:
`--drift-dropout` appears to have done its job.

### So the grounding failure is ENTIRELY in the planner

Fusion hands over a matrix that is ~47% box-driven. The TRM consumes it and
uses it. The planner receives the same `fused` DIRECTLY, plus `geometry`, and
weights them at 0.0178 and 0.0218 against 0.464 for phase.

**Vision is available and discarded, not absent.** No amount of upstream work
fixes this; it is a stage-B incentive problem, which is what `--phase-dropout`
targets (withhold the shortcut, since the regularizer previously withheld only
the vision paths).

### `wm_msg` is a broken channel at the SOURCE, not an ignored one

| quantity | value |
|---|---|
| mean \|msg\| | 0.479 |
| ||constant part|| (batch mean) | **3.315** |
| ||varying part|| | **0.268** |
| across-batch std / mean \|msg\| | 0.100 |
| dead dims (std < 1e-3) | 0 / 32 |
| effective rank of the varying part | **6.08 / 32** (top-4 = 71.5% energy) |

msg is a nearly FIXED 32-d vector — the informative component is 8% of its
magnitude, at ~6 effective dimensions. A constant input is absorbable into the
consumer's bias, so a planner sensitivity of 0.0006 is the CORRECT response to
it, not negligence. Fixing `wm_msg` means making msg_head's output vary, which
is a training-signal question (the planner's gradient is the only thing shaping
it in stage B) — and it DID reach 0.2394 in the agentview waypoint arm, so the
channel is capable of carrying information under some conditions.

### `next_emb->cur` is amplitude-limited by construction

The residual is 3.66% of ||cur||, so substituting `cur` for `next_emb` perturbs
that input by only ~3.7%. A near-zero `next_emb->cur` reading therefore cannot
distinguish "dead path" from "small perturbation" — which is exactly why
`next_emb->stale` (a full-magnitude, in-distribution wrong prediction) was added
to bench. Quote the stale probe, not the cur probe.

## 4i. Phase-dropout arm, and a CORRECTION to the sensitivity metric

### The metric used in 4b/4c/4d/4g mixes a discrete bit with continuous pose

`eval.bench --sensitivity` reported `mean |dplan|` over the whole
`[plan_steps, num_servos]` plan. The last column is the gripper, a HARD +/-1
(`torch.where(grip_logit > 0, ...)`), so ONE flipped gripper decision at every
step contributes exactly

    plan_steps * 2 / (plan_steps * num_servos) = 5*2/35 = 0.2857

to that mean — verified numerically. The largest sensitivity ever recorded here
is `state_delta` **0.2740**. The two are indistinguishable: a reading that size
is equally consistent with "this input strongly shapes the pose trajectory" and
"withholding this input flips the gripper decision and leaves pose untouched".

**Every combined sensitivity number in 4b, 4c, 4d and 4g is therefore
ambiguous, including the 12:1 phase:vision ratio that motivated this whole line
of work.** Bench now reports POSE-ONLY |dplan| and the GRIPPER-FLIP RATE in
separate columns; readings from the two builds are not comparable, and the
affected arms need re-measuring before the ratio is quoted again.

Related instrument fix: bench scored the waypoint head at row 0 while
`WaypointActuator` servoes toward the row derived from `cfg.waypoint_horizon`
(clamped to the last supervised row = row 3). `wp_std_ratio` / `wp_mae_mm` in
4d/4g describe a prediction the controller never executes. Now aligned.

### The arm itself: --phase-dropout 0.3 (both phase inputs, combined metric)

Same frozen stage-A world model, `--waypoint-weight 1.0`, no TQSA, no
`wm_latent`. Trained to early stop at epoch 23, best `val bc` **0.6234** against
**0.6580** without phase-dropout — a 5.3% better BC loss and val grip 0.722 vs
0.663, so withholding the shortcut did not cost fitting capacity.

| metric | no phase-drop | phase-drop 0.3 |
|---|---|---|
| `state_delta` | 0.2740 | **0.0263** (-10x) |
| `fused` | 0.0178 | **0.0466** (+2.6x) |
| `proprio` | 0.1904 | 0.2255 (UP) |
| `geometry` | 0.0218 | 0.0039 (-5.6x) |
| `wm_msg` | 0.0006 | 0.0061 |
| `next_emb->stale` | 0.0031 | 0.0160 |
| phase : vision | 11.7 : 1 | **5.0 : 1** |
| `std_ratio` | 0.126 | **0.071** |
| `wp_std_ratio` | 0.604 | 0.654 |
| `wp_mae_mm` | 4.8 | 5.3 |
| `corr` | 0.31 | 0.27 |
| `grip_acc` | 0.93 | **0.50** |
| `wm_margin` | +19.8% | +19.8% (same stage A, as it must be) |

**Result 1 — the shortcut is real but substitutable.** Withholding
`state_delta` collapsed its use 10x and more than doubled `fused`; the
phase:vision ratio improved 2.3x. But `proprio` sensitivity ROSE (0.190 ->
0.226): the planner did not turn to vision so much as migrate its phase reliance
to the other phase input. Dropping both independently at 0.3 leaves each
available 70% of the time, so a policy can always lean on whichever survives.

**Result 2 — the gripper collapsed to chance, and the mechanism is specific.**
`grip_acc` 0.93 -> 0.50. `proprio` is
`[eef_pos(3) | quat(4) | gripper(2) | valid(1)]`, and the demo gripper COMMAND
at t is strongly autocorrelated with the measured gripper STATE at t — so
proprio carries the single best predictor of the BCE target, and withholding it
30% of the time destroys that head. This is the "trade one fixed failure for
another" risk realised, and it is disqualifying on its own: a chance-level
gripper cannot complete a pick-and-place regardless of any grounding gain.

**Result 3 — magnitude regressed** (`std_ratio` 0.126 -> 0.071) while the
waypoint head held (0.604 -> 0.654). Consistent with 4d: the action head is the
fragile path and the displacement head is not, so the actuated translation is
less affected than the table suggests.

**Consequence for the design.** Drop the shortcut that is not load-bearing and
keep the one that is: `--planner-drop-rate 'state_delta=0.4'`, leaving proprio
intact. Per-input rates replace the coarse `--phase-dropout`, which could only
move both together. And with the gripper now known to ride on proprio, any
future proprio ablation should mask its POSE dims and preserve the gripper
slots.

## 4j. Stage-B redesign (v7.3/v7.4) — root cause and the four changes

A five-direction design pass (with adversarial review of each) located the root
cause one level below "the regularizer was backwards".

### Root cause: the supervision HORIZON, not the regularizer

`preprocess/common.py::chunk_actions` builds plan rows at the NATIVE rate. With
LIBERO at 20 Hz and `plan_steps=5`, the 5 rows span **0.25 s**, and
`waypoint_targets` supervises 0.05-0.20 s of displacement. **Over 0.2 s, "keep
doing what you are doing" is a near-sufficient statistic for the demo action**:
the target is first-order predictable from arm pose and task progress, and
object position is only a second-order correction. MSE-BC is a conditional-mean
estimator and consumes variance in descending order, so it takes the phase term
and leaves the vision residual on the table. The 12:1 ratio is what that
ordering looks like, not a bug in any one flag.

Two things converted that structural bias into the measured extreme, and 4h
rules out every upstream cause (fusion is 47% box-driven; zeroing `fused`
destroys 89% of the TRM's residual — **vision is available and discarded**):

1. The regularizer's sign was backwards: `--planner-input-dropout 0.15` withheld
   the VISION paths while phase was never withheld, so training actively taught
   the planner to work without the grounded observation.
2. 4h also measured box CENTERS at only 12% of `fused` and 8% of the residual —
   so the discarded signal lives in the box EMBEDDINGS, not in the center
   coordinates `geometry` carries. Widening `geometry` was never going to be it.

### The four changes

**1. Graded FADE instead of deletion** (`chrono_planner.py::forward(fade=...)`,
0 params). Withholding by passing `None` DELETES a group's tokens — for `fused`
that is 32 of ~68, an attention-softmax regime deployment never occupies, since
the loop always passes real tensors. `fade` multiplies a group's PROJECTED
content before its type embedding, byte-for-byte the idiom
`SlotResonanceFusion` already uses for box evidence
(`box_weight * (1 - drop*(1 - fade))`). CLAUDE.md's hard rule — one shared
evidence path, no binary zeroing — applies to the planner too and did not
previously hold there. Verified: `fade=None` is bit-identical to before.

**2. PER-SAMPLE fade weights** (`train_batched.py::_fade_weights`, 0 params).
The old `rng.random() < p` drew ONE scalar per (batch, timestep), so at batch 64
all 64 episodes were withheld together and a withheld step had no full-input
sample anywhere in its gradient. Now `[B, 1, 1]` draws, same continuum as fusion.

**3. Per-input withhold rates** (`--planner-drop-rate 'state_delta=0.4'`).
4i measured `--phase-dropout 0.3` (both phase inputs) improving phase:vision 2.3x
AND collapsing `grip_acc` 0.93 -> 0.50, because `proprio` carries the arm's
GRIPPER STATE — the best predictor of the gripper command. Coarse flags could
only move both together; per-input rates drop the shortcut that is not
load-bearing and keep the one that is.

**4. PRE-GRASP step weighting** (`microvla/utils/phase.py`, `step_weight` in
`split_planner_loss`, 0 params). Object position only matters BEFORE the grasp;
afterwards the trajectory is transport to a target in the same place every
episode. The phase signal needs no new data — the demo's own gripper transition
gives it. Mean-1 normalized per episode so it is not a disguised LR change, and
`step_weight` (episode TIMESTEP) composes with `row0_weight` (plan ROW) as
orthogonal axes. Episodes whose gripper never closes — all of bridge — and
episodes that first close on their LAST sampled step are marked UNUSABLE and
left at weight 1: in both, "pre-grasp" would be the whole episode, which is a
per-episode learning-rate change rather than a phase signal.

**5. LONG-HORIZON supervision (v7.4) — the change that addresses the root
cause** (`waypoint.py::long_horizon_targets`, `--waypoint-long`, 0 params, NO
re-bake). `eef_pos_chunk[..., t, 0, :]` is the absolute EEF position at SAMPLED
frame t, so the leading column of every baked npz is already a 2 Hz EEF
trajectory. Row k becomes `traj[t+k+1] - traj[t]` — **0.5 to 2.5 s** of
displacement instead of 0.05-0.20 s. Over 2.5 s the arm must actually ARRIVE, so
where the object is becomes a FIRST-order determinant of the target. This is the
only change that alters the conditional-mean ordering rather than taxing the
shortcut after the fact.

Two unit companions are mandatory and both are silent train/deploy mismatches if
missed — each has a test:

* `cfg.waypoint_range` must grow (0.15 m saturates the `[-1, 1]` clamp on any
  real reach and destroys the signal); `--waypoint-long` defaults it to 0.5.
* `cfg.waypoint_row_stride` must be `source_hz / real_frame_hz` (10 for LIBERO),
  because a row is now `(k+1)*stride` CONTROL steps out and `WaypointActuator`
  divides the positional error by the steps remaining to get a per-step rate.
  Wrong stride under-delivers the command by exactly the stride.

Cost of the tail: row k is unsupervised for the last k+1 timesteps of each
episode (~8% of rows at T~30), masked per (timestep, row) rather than per row.

### Also landed: the belief-state channel (v7.3, 4h consequence)

`RecursiveTRM.forward_full` exports `latent [B, d]`, the pooled belief state all
its readouts come from, and the planner consumes it as 8 tokens for 33K params.
`msg` was 92% a fixed vector at effective rank 6/32 — a bottleneck, not a
channel. Deliberately NOT a second TRM as the planner: the reason the TRM uses
vision and the planner does not is the OBJECTIVE, not the architecture, so
swapping architectures would not change the incentive (and d=1024 is 9.97M
against a 2.5M cap).

### Planner ledger after all of it

1,803,527 params (1,804,298 with the waypoint head) against the 2.5M cap;
trainable total 6,988,685 of 9,000,000. Every change above except the
`wm_latent` projection costs ZERO parameters. 219 tests green.

## 4k. Long-horizon arm A — vision won, the action head starved, and a bench bug

`--waypoint-long --waypoint-weight 1.0`, otherwise identical to 4g. Two defects
in the run, both mine, and one genuine result.

### DEFECT 1 (bench): the waypoint numbers are void

Bench scored the head with `waypoint_targets` (NATIVE spacing) regardless of
`cfg.waypoint_long`. A long-horizon head predicts 0.5-2.5 s of displacement and
was compared against 0.05-0.20 s targets, so a ~10x scale difference was
reported as prediction error: **wp_std_ratio 3.946, wp_mae 116.1 mm** describe
the mismatch, not the head. Fixed; bench now selects the spacing from the
checkpoint's cfg, with a test asserting the two spacings do not report the same
error.

### DEFECT 2 (loss balance): the action head starved

| metric | 4g (native) | arm A (long) |
|---|---|---|
| `std_ratio` | 0.126 | **0.022** |
| `corr` | 0.31 | **0.02** |
| `grip_acc` | 0.93 | 0.50 |
| `pose_mae` | 0.243 | 0.257 |
| `wm_margin` | +19.8% | +19.8% (same stage A) |

**RETRACTED explanation.** This was first recorded as loss imbalance: long
horizon targets have RMS ~0.163 against ~0.047 native, so ~12x the MSE at equal
`--waypoint-weight`, therefore the waypoint term swamps BC. A later
long-horizon run at weight 1.0 reported `val bc 0.6924` against `val wp 0.1107`
— **BC is 6x LARGER**, so the waypoint term was never dominating. The 12x figure
is the ratio of target MAGNITUDES and does not transfer to the loss, because a
head that fits a large target still has a small residual. The reasoning was
wrong even though the arithmetic behind it was right.

**The collapse is therefore UNEXPLAINED.** The leading candidate is
representational interference on the shared trunk rather than loss weighting: a
2.5 s displacement target is smooth and low-frequency while the action target is
high-frequency, and `feats` feeds both heads, so capacity allocated to the
former may cost the latter exactly the detail `std_ratio` measures. That is a
hypothesis, not a measurement. It is testable — sweep `--waypoint-weight` and
see whether `std_ratio` tracks it (loss weighting) or does not (interference).

### THE RESULT: vision finally dominates the planner

> **RETRACTED (§4m).** This heading rests on `phase:vision = 2.0:1`. Three
> seeds of the identical config later gave 0.3:1, 1.0:1 and 5.5:1 — the
> ratio is not measurable from one run. `fused` itself is stable; its size
> relative to proprioception is not.

First readings on the pose/grip-split instrument (4i), so not comparable to the
combined numbers above — but internally consistent:

| input | POSE \|dplan\| | grip flip % |
|---|---|---|
| **fused** | **0.0967** | 72.8% |
| state_delta | 0.0305 | 19.6% |
| current_emb | 0.0080 | 3.0% |
| wm_msg | 0.0071 | 1.3% |
| pred_box_emb | 0.0053 | 0.8% |
| wm_latent | 0.0041 | 0.1% |
| proprio | 0.0020 | 0.7% |
| geometry | 0.0008 | 0.1% |
| next_emb->stale | 0.0006 | 0.0% |

**`fused` is now the strongest input by 3.2x**, and PHASE has collapsed:
`proprio` 0.0020 and `state_delta` 0.0305 against `fused` 0.0967. Every earlier
arm had phase dominant by 5-12x. The phase:vision ordering has INVERTED.

That is the horizon hypothesis confirmed in the direction it predicted: at a
0.2 s supervision horizon "keep doing what you are doing" is a near-sufficient
statistic and vision is a second-order correction; at 0.5-2.5 s the arm must
ARRIVE somewhere, so where the object is becomes first-order and the conditional
mean has to use it. No regularizer achieved this — `--phase-dropout` (4i) only
moved the ratio 2.3x and did it by taxing the shortcut, whereas this removes the
shortcut's sufficiency.

**Two things stop this being a clean win.** (i) The action head is collapsed
(`std_ratio` 0.022), so the sensitivity is measured on a nearly-constant output:
withholding `fused` moves the plan ~8x more than the plan varies across
timesteps on-distribution. Vision reaches the planner; the output is broken by
loss balance, not by the horizon. (ii) `geometry` fell to 0.0008 — so the
planner is reading box EMBEDDINGS through `fused`, not center coordinates, which
matches 4h (box centers are only 12% of `fused` and 8% of the TRM residual).

**Next arm:** the same configuration at `--waypoint-weight 0.08`, which should
keep the inverted ordering while leaving the BC head enough gradient to stay
non-degenerate. If `fused` holds above `state_delta` at a recovered `std_ratio`
and `grip_acc`, that is the first grounded policy this project has produced.

## 4l. Long-horizon arm A, RERUN — best arm yet, and large run-to-run variance

> **SUPERSEDED (§4m).** "Best arm yet" is withdrawn: `std_ratio` 0.245 is the
> top of a 5-sample distribution of this same config (mean 0.084, sd 0.097,
> 11.1x fold), and the arm's training length — not its configuration —
> predicts its bench numbers.

Identical command to 4k (`--waypoint-long --waypoint-weight 1.0`, same frozen
stage A, same seed), re-run after the bench waypoint-scoring fix. So the
waypoint numbers here are valid where 4k's were void.

| metric | v7 pilot | wrist native (4g) | 4k arm A | **4l rerun** |
|---|---|---|---|---|
| `std_ratio` | 0.369 | 0.126 | 0.022 | **0.245** |
| `wp_std_ratio` | — | 0.604 | void | **0.799** |
| `wp_mae_mm` | — | 4.8 (0.2 s) | void | **58.2 (2.5 s)** |
| `corr` | 0.49 | 0.31 | 0.02 | **0.45** |
| `grip_acc` | 0.93 | 0.93 | 0.50 | 0.88 |
| `pose_mae` | 0.20 | 0.243 | 0.257 | **0.212** |
| `wm_margin` | +1.7% | +19.8% | +19.8% | +19.8% |

Pose-only sensitivity (the split instrument, so comparable ONLY to 4k):

| input | 4k arm A (collapsed) | **4l rerun** |
|---|---|---|
| proprio | 0.0020 | **0.1220** |
| state_delta | 0.0305 | 0.0953 |
| fused | **0.0967** | 0.0939 |
| wm_msg | 0.0071 | 0.0471 |
| current_emb | 0.0080 | 0.0230 |
| geometry | 0.0008 | 0.0137 |
| wm_latent | 0.0041 | 0.0092 |
| pred_box_emb | 0.0053 | 0.0083 |
| PHASE : VISION | 0.33 : 1 | **2.0 : 1** |

> **RETRACTED (§4m):** the phase:vision row. Seed range 0.3–5.5:1.

### THE FINDING THAT MATTERS MOST: run-to-run variance is enormous

4k and 4l are the SAME COMMAND at the same seed. One produced `std_ratio` 0.022
with `corr` 0.02 and a chance-level gripper; the other produced 0.245 / 0.45 /
0.88. **A single stage-B run is not a measurement of a configuration.**

Consequences that must be applied retroactively:

* **4k's headline — "the phase:vision ordering INVERTED" — is not supported.**
  That reading came from the collapsed run, where the output was nearly constant
  and I flagged the sensitivity as measured on a degenerate plan. With a healthy
  output from the identical config, phase is back on top (proprio 0.1220 >
  fused 0.0939). The inversion was a property of a collapsed run, not of the
  horizon.
* **The unexplained collapse in 4k is now attributable to variance**, not to
  the long horizon and not to loss weighting (which 4k already retracted). No
  `--waypoint-weight` sweep is needed to explain it.
* **Every single-run A/B in 4b, 4c, 4d, 4g, 4i is one sample.** Differences
  smaller than the 4k-vs-4l gap — which spans std_ratio 0.022 to 0.245 — carry
  no information. That gap is larger than every effect this project has claimed
  from an architecture or regularizer change.

### What survives, and it is real

**Long horizon is the best configuration measured**, on `std_ratio`, `corr` and
`pose_mae` simultaneously, with the gripper nearly intact. And vision is
materially stronger than in any native-spaced arm: `fused` 0.0939 with
`geometry` 0.0137 gives phase:vision 2.0:1, against 5-12:1 for every native arm
(on the older combined instrument, so treat the ratio as indicative, not exact).

**The target-parameterization result replicates at the long horizon**:
`wp_std_ratio` 0.799 against action `std_ratio` 0.245 is **3.3x**, matching the
3.3x measured at native spacing in 4d (0.787 vs 0.237) and the 4.8x in 4g. Three
corpora, two horizons, one conclusion — regressing metric displacement shrinks
far less than regressing normalized actions. This is the most robust claim the
project has, and notably it is the one measured WITHIN a run rather than across
runs, which is why variance does not threaten it.

`wp_mae` 58.2 mm is over a 2.5 s horizon (against 4.8 mm over 0.2 s). Judge it
against the distance the arm covers in 2.5 s, not against the short-horizon
number.

### Required next step: seeds, not new arms

Before any further architecture work, run the SAME configuration at 3 seeds and
report the spread. Any claim resting on a single stage-B run — including several
already in this document — needs that error bar before it means anything.

## 5. Infrastructure results (method-section material)

**Frozen-backbone map caching.** Stage B with `--tqsa` re-ran YOLO-World over
every framed timestep every epoch, at a 128→512 px upscale. Measured cost:
**~6.1 s/batch, 105 batches in 644 s, ~18 min/epoch, ~12 h for 40 epochs.**
The backbone never trains, so those maps are identical across all 40 epochs.
Precomputing once:

* 19,680 train frames + 1,031 val frames
* map shape **(512, 20, 20)** = **400 KB/frame** at fp16
* **7.5 GB train + 0.4 GB val = 7.9 GB** resident
* one pass: **137 s train (154–165 frames/s) + 14 s val**
* epochs afterward: **~130 s**, i.e. the same cost as the TQSA-free arm (~146 s)

**≈8x per-epoch, 12 h → ~90 min.** Not bit-identical: fp16 storage is ~2e-4
relative and chunked batching ~1e-6, both far below the signal.

Rejected after measurement: lowering `min_side` saves **0.6%, not 4x** (512,
256 and 128 all letterbox to 640×640 and yield the same 20×20 map); larger
`--batch-size` does not amortize the per-image overheads. Available but not
taken: truncating the forward at SPPF (layer 9), skipping the text-conditioned
head and NMS — measured **maxdiff 0.0, 1.8–2.0x**.

**GPU contention (why wall-clock numbers vary).** Seven processes on GPU 1
totalling ~154 GB of 192 GB, including two duplicate ~20 GB MicroVLA runs
(PIDs 374717, 400975) and two large TinyVLA jobs (63.7 GB, 36.2 GB). Identical
stage-A epochs ran **96 s uncontended vs 496 s contended (5.2x)**. Every
per-epoch second in §2/§3 is contended and is not a hardware claim.

**Parameter ledger** (`microvla.utils.param_audit`, unchanged by v7.2):
fusion 4,460,165 · drift 724,993 · planner 1,770,247 · **total 6,955,405** of
9,000,000. The waypoint head costs **771** params `((d_plan+1)*3)`. Planner
ablations: `geometry` −1,792 · `pred_box_emb` −16,640 · both −18,432 ·
plus `next_emb` −35,072 · `spatial` → 1,720,583.

Test suite over this session: **149 → 231** passing (CPU-only, mock-only, no
network, no cv2). Parameter ledger after v7.4: fusion 4,460,165 · drift 724,993 ·
planner 1,803,527 · **total 6,988,685** of 9,000,000.

## 6. Defects found and fixed (2026-07-25)

Ordered by what they would have cost if undetected.

0. **Stage-B early stopping was not comparable across arms** (§4m). The stop
   metric folded `waypoint_weight * val_wp` into the decision while `--min-delta`
   stayed absolute, so arms whose waypoint targets are ~10x larger early-stopped
   sooner. Runs ranged 8–28 epochs and every bench metric tracked epochs-survived
   at Spearman >= 0.84 — the arm rankings in §3/§4j/§4k/§4l measured stop timing,
   not architecture. This is the costliest defect in the project so far: it
   silently invalidated a 12-arm batch and several earlier single-run A/Bs.
1. **Bench scored TQSA checkpoints blind** (§0) — invalidates the comparability
   of every recorded TQSA-checkpoint number.
2. **Silent cache corruption path.** The SPPF hook retains only the LAST
   forward's output; had the detector split a frame list across several internal
   forwards, the precompute would have BROADCAST one frame's features across a
   64-episode chunk with no exception. The live path fails loudly on this; only
   the precompute could swallow it. Now raises.
3. **Random-init TQSA at deployment** for checkpoints trained without it (§0).
4. **`mp.Pool` masks worker death** — a segfaulting/OOM-killed worker is reaped
   and replaced while its chunk is never re-dispatched, so the parent blocks
   forever with no error. Reproduced: 10 live workers indefinitely, versus
   `ProcessPoolExecutor` raising `BrokenProcessPool` in **0.4 s**. This is why
   the 10-worker eval hang was a 20-minute black box.
5. **Thread env set too late.** `eval/__init__.py` imports torch at package
   import, so `os.environ` writes inside a spawned worker land after libgomp
   init and are no-ops. Moved to the parent, which children inherit.
6. **`--device` never moved the heads.** `heads_device` was hardcoded to CPU, so
   `--device cuda:0` moved only the detector — which runs 1 tick in 15 — while
   the 9.97M d=1024 TRM ran on CPU every tick. Now `--heads-device`. Verified on
   MPS, which caught two latent device bugs (the corrector's accumulator and the
   TQSA feature map).
7. **`t0` shadow** in stage B: `t0 = rng.randrange(...)` overwrote the epoch
   timer under `--unfreeze-trm`, printing ~1.7e9 s epoch times.
8. **Seed groups matched by prefix** (§4m): `startswith("longh_s")` pulled the
   unrelated `longh_sdfade` arm into the longh error bar, reported as "4 seeds".
9. **Ragged summary rows.** Checkpoints benched on the pre-split sensitivity
   instrument were rendered in the same table as pose-split ones, with a variable
   cell count. Now padded to fixed width and marked `~` as not comparable.
10. **The overnight wrapper had no SIGTERM shield** while all 21 Python CLIs did,
    so one signal to the wrapper killed the batch. `trap '' TERM HUP`, and output
    via `exec` rather than `tee` (a `tee` to a dead tty exits and takes the
    pipeline with it). Failure lines also reported `rc=$?` captured after a
    `[ -f ]` test, so every failure logged `rc=0`.

## 7. Open

* **Closed-loop `mean_success` is 0.000 over 50 trials** (§4m). Obtained, and
  negative. Every claim the paper can currently make is open-loop.
* ~~Waypoint-absolute arm untrained~~ — trained and measured; the
  displacement-vs-action ratio is 3.0x–29.1x across 19 arms (§4m).
* ~~TQSA `spatial` sensitivity unread~~ — read: 0.0487 pose, but the with/without
  head-to-head is within noise at 1.6x inference cost (§4m).
* ~~LIBERO-only stage B untested~~ — tested once, `std_ratio` 0.253 at +2.96
  prediction-sd over the training-length trend, but only 0.87 seed-sd over the
  best mixed-corpus arm. Needs 3 seeds (§4m).
* **Re-run the whole batch under the fixed protocol**
  (`SUFFIX=_fx bash scripts/overnight.sh`, `--stage-b-select bc`,
  `--stage-b-min-epochs 20`). Until then no arm ranking in this document means
  anything, and the closed-loop zero cannot be attributed to any one arm.
* **The sensitivity instrument needs a variance budget.** `proprio` and
  `geometry` vary 46–134x across seeds; any future claim on them needs >= 3 runs,
  or a different instrument.

## 4m. The 12-arm overnight batch (2026-07-26) — the arm comparison was measuring training length

`scripts/overnight.sh`, 08:34–11:10, 12 stage-B arms trained and 19 checkpoints
benched on the pose/grip-split instrument, zero failures. All arms share the one
frozen stage A (`full_stageA_wrist_v72.pt`), so `wm_margin` is **one** number
replicated across 19 rows (+19.8%), not 19 measurements. Full table:
`results/PAPER_TABLE.md`; raw records in `results/metrics.jsonl`.

### The batch's primary result is a defect in its own protocol

Arms early-stopped anywhere from **8 to 28 epochs**, and every bench metric is a
monotone function of how long the run survived (n = 9 mixed-corpus arms):

| metric | Pearson vs epochs | Spearman |
|---|---|---|
| `wp_std_ratio` | 0.891 | **0.924** |
| `grip_acc` | 0.901 | **0.907** |
| `std_ratio` | 0.770 | **0.866** |
| `pose_mae` | −0.793 | **−0.865** |
| `corr` | 0.915 | **0.840** |

Sorted by epochs, every column moves together:

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

Grouped: **≤9 epochs → `std_ratio` 0.022, `corr` 0.06, `grip` 0.53**; **≥22
epochs → 0.120, 0.40, 0.90.** An under-trained planner emits near-constant
actions, which is exactly what a low `std_ratio` measures.

### Mechanism (a real bug, not just variance)

`train/train_batched.py` gated early stopping on
`val_loss = val_bc + waypoint_weight * val_wp`, compared against an **absolute**
`--min-delta` of 1e-4. `val_wp` is ~10x larger under `--waypoint-long`
(0.5–2.5 s displacement targets vs 0.05–0.20 s), and a larger term carries
larger noise, so a fixed absolute threshold is cleared less often and `stale`
accrues faster. Long-horizon arms therefore ran a harsher effective patience
than native ones. The function's own comment asserts `val bc` is "the SAME
quantity across every arm" — true of the printed line, false of the decision it
gated. Observed best-val totals confirm the scale split: longh 0.75–0.82 against
native 0.61–0.65, consistent with the reported decomposition `val bc 0.6924` +
`wp 0.1107` = 0.803.

**Consequence.** No single-seed arm comparison in this batch is interpretable,
including the four that "lost": `longh_pregrasp` (14 ep), `longh_sdfade` (8 ep),
`longh_novis` (15 ep), `longh_all` (skipped, pre-existing). Nor is
native-beats-longh (mean `std_ratio` 0.112 vs 0.051) — native trained 22/24
epochs and longh 8/9/22.

**Fixed.** `--stage-b-select {bc,total}`, now defaulting to `bc` (the only term
on a scale shared by every arm); `--stage-b-min-epochs` floors the run length so
a noisy plateau cannot end a 40-epoch budget at epoch 8. `total` is retained
solely to reproduce this batch. `tests/test_stage_b_selection.py` pins both.
`scripts/overnight.sh` defaults to `MIN_EPOCHS=20` and takes a `SUFFIX` so a
protocol change re-runs instead of being skipped onto the old checkpoints.

### Seed spread, corrected

The generated table reported "longh over 4 seeds" because `startswith("longh_s")`
also matched `longh_sdfade` — an unrelated arm averaged into the error bar. Fixed
to an exact `_s\d+$` match. Corrected:

| config | n | mean | sd | range | fold |
|---|---|---|---|---|---|
| `native` | 3 | 0.112 | 0.076 | 0.035–0.187 | 5.4x |
| `longh` | 3 | 0.051 | 0.048 | 0.022–0.106 | 4.7x |

Pooling every sample of the identical `longh` config this project has run —
{0.022, 0.245, 0.022, 0.106, 0.024} — gives mean 0.084, sd 0.097, **11.1x
fold**. §4l's 0.245 is the top of that distribution, not a reproducible level.

### The sensitivity instrument is unmeasurable at n=1 for most inputs

Same config, seed the only difference:

| input | native (3 seeds) | fold | longh (3 seeds) | fold |
|---|---|---|---|---|
| `fused` | 0.0605, 0.0479, 0.0596 | **1x** | 0.0967, 0.0327, 0.0626 | 3x |
| `state_delta` | 0.0072, 0.0234, 0.0419 | 6x | 0.0305, 0.1068, 0.0119 | 9x |
| `proprio` | 0.0020, 0.1216, 0.1511 | **76x** | 0.0020, 0.0919, 0.0543 | 46x |
| `geometry` | 0.0, 0.0134, 0.0064 | **134x** | 0.0, 0.0035, 0.0046 | 46x |

`fused` is stable enough to read off a single run. `proprio` and `geometry` are
not, which makes the **phase:vision ratio unusable at n=1**: it spans 0.2–2.9
across native seeds and 0.3–5.5 across longh seeds.

**RETRACTION (§4k, §4l).** The claim "vision finally dominates the planner",
resting on longh `phase:vision = 2.0:1`, is withdrawn. Three seeds of that exact
config give 0.3:1, 1.0:1 and 5.5:1; 2.0 sits inside the range. What survives is
the weaker, instrument-supported statement: `fused` pose-sensitivity is
0.03–0.10 and is the largest single visual contribution, but its size relative
to proprioception cannot be established from one run.

### `longh_liberoonly` — the one arm worth 3 seeds

Trained on LIBERO alone (bridge is ~75% of mixed-corpus stage-B steps and
supplies neither proprio nor frames). 26 epochs, best val 0.5724 — not
comparable to the others, since its val set is a different corpus. It is in the
well-trained group, so the confound above does not explain it away by itself.
Fitting each metric on the 9 mixed-corpus arms and holding libero-only out:

| metric | observed | predicted @26 ep | residual | prediction-sd |
|---|---|---|---|---|
| `std_ratio` | 0.253 | 0.126 | +0.127 | **+2.96** |
| `corr` | 0.480 | 0.432 | +0.048 | +0.57 |
| `grip` | 0.940 | 0.954 | −0.014 | −0.14 |

So its best-in-batch `corr` and `grip` are **entirely explained by training
length**; only `std_ratio` exceeds the trend. Against the strongest single
comparator (`native_s2`, 0.187 at 24 ep) the margin is 1.35x but only **0.87
seed-sd** — one run cannot separate them (n = 9, 7 df; directional).

Its distinctive-looking sensitivity profile (`state_delta` 0.1933, `fused`
0.1558, `proprio` 0.1337, `wm_latent` 0.0871, `current_emb` 0.0772,
`pred_box_emb` 0.0507 — the only arm where every input registers) is **not**
claimable: it rests on `proprio`/`geometry`, the two least stable inputs.

**Status: hypothesis.** Corpus dilution is a plausible mechanism for the
grounding failure and this is the first evidence for it, but it needs 3 seeds
under the fixed protocol.

### TQSA spatial input is near-worthless

Same checkpoint, benched with and without the spatial pathway:

| | `std_ratio` | `wp_std_ratio` | `corr` | `grip` | s/eval |
|---|---|---|---|---|---|
| `longh_tqsa` (spatial on) | 0.072 | 0.913 | 0.39 | 0.91 | 0.74 |
| same ckpt, spatial off | 0.075 | 0.739 | 0.38 | 0.87 | 0.45 |

Withholding `spatial` moves pose by 0.0487 (third-ranked) but the head-to-head
is within noise on every action metric, at **1.6x the inference cost**. This is
the 2-minute measurement §4b owed.

### Closed loop: a well-sampled zero

`libero_object`, the best arm by `std_ratio` with `grip > 0.7`, 5 trials × 10
tasks = **50/50 failures, `mean_success` 0.000**, all 10 tasks completed, 0
scavenged, no failed workers. Previous zeros were single small runs; this one is
sampled well enough to state as a result.

### Closed-loop telemetry: the collapse is DIRECTIONAL, and not saturation

`eval.telemetry_probe` over the five workers of the 0/50 run (3000 steps each,
200 real ticks each, 10 trials each):

| worker | \|cmd\| mean | max | clipped | x | y | z | plan_norm | trust mean / min |
|---|---|---|---|---|---|---|---|---|
| w0 | 0.4005 | 0.9575 | 0.0% | 0.1204 | 0.8461 | 0.2349 | 2.865 | 0.519 / 0.119 |
| w1 | 0.4037 | 0.9575 | 0.0% | 0.1192 | 0.8570 | 0.2349 | 2.872 | 0.522 / 0.135 |
| w2 | 0.4007 | 0.9575 | 0.0% | 0.1140 | 0.8511 | 0.2372 | 2.866 | 0.517 / 0.131 |
| w3 | 0.4052 | 0.9575 | 0.0% | 0.1196 | 0.8575 | 0.2386 | 2.887 | 0.518 / 0.147 |
| w4 | 0.4052 | 0.9575 | 0.0% | 0.1186 | 0.8550 | 0.2420 | 2.879 | 0.521 / 0.138 |

**One axis carries 7.2x the weakest** (y 0.855 against x 0.119), sustained across
3000 steps, and the pattern is identical across all five workers to three decimal
places. The dominant axis is a per-CHECKPOINT artifact, not a property of the
architecture: an earlier arm on the same suite ran x 0.5682, y 0.2339, z 0.4174
(x-dominant), and another x 0.0953, y 0.1049, z 0.1053 (uniform and tiny, mean
0.1018).

**`clipped` is 0.0%, which excludes actuator saturation as the explanation.**
That matters: a policy pinned at the command limit and a policy emitting a
near-constant interior command look alike in `mean_success` and are
distinguished only by this number. Earlier arms did saturate (2.3-3.7%, and
24.8-32.5% before the actuator fix), so the instrument does register it. Here it
does not, so the arm is freely commanding a nearly constant direction.

Set against `wp_std_ratio` 0.75-0.94 measured open-loop on demo observations,
this is exposure bias stated precisely: **healthy output variance
on-distribution, collapse to a near-constant direction off-distribution.**

`trust` mean is 0.517-0.522 against `cfg.brake_trust` 0.5, so roughly half of
all steps are being attenuated by the delta-mode brake — against a waypoint
command, which is a positional error rather than a held motion. `policy.py`'s
own comment records that symmetry as "an assumption, not a result". Untested.

### What the batch leaves standing

1. ~~**Displacement regresses less shrunk than action** — 3.0x–29.1x across 19
   arms~~ **RETRACTED (4o).** On a sighted corpus the ratio is 1.7–2.0x. The gap
   was mostly an action head starved of object evidence, not a fact about target
   parameterization. The within-forward-pass measurement stands; its magnitude
   does not.
2. **Closed-loop success is 0/50.**
3. **`fused` pose-sensitivity is 0.03–0.10** and stable across seeds.
4. The confound, its mechanism, and its fix.
5. `longh_liberoonly` as a hypothesis on `std_ratio` alone.

Everything else in §3, §4j, §4k and §4l that rests on a single stage-B run is
suspended pending the fixed-protocol re-run.

## 4n. ROOT CAUSE: the corpus contained no object evidence at all

Measured 2026-07-26 while baking LIBERO locally. This supersedes the grounding
diagnosis in 4h and reframes every open-loop result in this document.

### The measurement

`preprocess/common.py` set the detector's classes to the PARSED TASK PHRASES —
`set_classes([source, target])`, e.g. `["alphabet soup", "basket"]`. Freshly
baked 3-episode samples of `libero_object`, per role, fraction of steps with a
detection:

| view | source (object to pick) | target (basket) | source-center std |
|---|---|---|---|
| **wrist** (`eye_in_hand_rgb`) | **0.0%**, weight 0.0000 | **0.0%**, weight 0.0000 | 0.000, 0.000 |
| agentview | **0.0%**, weight 0.0000 | 95.7%, weight 0.5048 | 0.000, 0.000 |

**The source object was never detected — in either view, on any frame.** Every
`source_center` was the fallback 0.5, 0.5 with standard deviation exactly zero.
Because `SlotResonanceFusion` weights box and geometry tokens by `box_weight`, a
weight of 0 fades them to nothing: the policy was trained with **no object
information whatsoever**, only `frame_emb`, the text tokens, proprioception and
the last action.

### Why it happened, and why no aggregate metric caught it

YOLO-World-S returns exactly 0.000 for LIBERO's product names. The objects are
plainly visible and DO detect under concrete visual categories, on the same
frames:

| prompt | agentview | wrist |
|---|---|---|
| `alphabet soup` (what was baked) | **0.000** | **0.000** |
| `bottle` | 0.604 | 0.000 |
| `box` | 0.195 | **0.499** |
| `cardboard box` | 0.217 | 0.424 |
| `can` | 0.246 | 0.000 |
| `soup can` | 0.232 | 0.000 |
| `carton` | 0.136 | 0.136 |
| `product` / `package` / `item` / `object` / `thing` | 0.000 | 0.000 |

Abstract nouns recover nothing; only concrete categories do. The failure is
silent by construction: a missed detection is indistinguishable from a
legitimately faded one, which is exactly what the graded-evidence design
intends, so `box_weight = 0` flows through fusion as a valid input rather than
an error. `det_conf` is 0.10 and resolution was not the cause — 0.000 at
`min_side` 128, 512 and 1024 alike.

### What this invalidates

* **4h's central conclusion — "the grounding failure is ENTIRELY in the planner"
  — is WRONG.** The planner could not use object evidence because none existed.
  The weight-level analysis showing fusion to be "vision-rich" was reading
  weights on inputs that were identically zero at training time.
* **Every `geometry` sensitivity reading is explained.** Centers were constant,
  so withholding them changed nothing; the 46x-134x seed folds in 4m were
  variation in how each run fit a constant.
* **The closed-loop behaviour is explained exactly.** On the agentview corpus
  the basket was detected on 95.7% of frames and the object on 0%, which is
  precisely the reported failure: the arm reaches the basket reliably and never
  approaches an object. On the wrist corpus — `full_stageA_wrist_v72.pt` and
  every long-horizon arm — there was no evidence at all.
* **The v8 relational head would have been provably inert.** It would have
  reasoned over two object slots whose weights were zero on every frame.

### The fix

`microvla/perception/yolo_world.py::set_role_prompts` already implements the
right mechanism — an ordered prompt chain per role, taking the best box of the
FIRST prompt that detected anything — and the bake was not using it. Now wired,
with a concrete-category tail after the exact phrase and its head noun:

`"alphabet soup"` -> `["alphabet soup", "soup", "box", "cardboard box", "can",
"bottle", "carton", "container"]`

Re-baked, same 3 episodes, wrist view:

| | source detected | target detected | source-center std |
|---|---|---|---|
| before | 0.0% | 0.0% | 0.000, 0.000 |
| **after** | **55.3%** | **34.0%** | **0.192, 0.177** |

The exact phrase still wins wherever it grounds, so this only adds recall. This
is the first time in the project that the corpus has carried object evidence,
and it means every open-loop number in 3, 4 and 4m was measured on a policy that
had no object input. Those results are not wrong about what they measured; they
were measuring a different system than intended.

## 4o. v8 on a sighted corpus — the action head recovers, and one headline dies

First results from the v8 stack (`DESIGN.md` "v8 plan") trained on a corpus
baked AFTER the 4n fix. Box run 2026-07-27, `scripts/box_v8.sh`; artifacts in
`eval_results/bench_v8_*.json` and `logs/box_v8/`.

### Corpus

Three suites baked at 500 episodes each, wrist view, with the prompt fallback
chain and class-agnostic proposals. Detection rates, per suite:

| suite | source detected | target detected | verdict |
|---|---|---|---|
| `libero_object` | **48.0%** | 20.3% | accepted |
| `libero_goal` | accepted | — | accepted |
| `libero_spatial` | **13.8%** | 44.6% | **gate FAILED** (floor 20%) |

Corpus used: **1000 episodes** (950 train / 50 val) from object + goal.

`libero_spatial` is not blind — it is under-detected. Its tasks are tableware
("pick up the black bowl between the plate and the ramekin and place it on the
plate") and the fallback tail was grocery-shaped (`box`, `can`, `bottle`,
`carton`), so nothing in it could fire on a bowl. The plate already grounds at
44.6%. Fixed by routing the tail on the phrase's head noun; unverified until the
next bake.

### What ran

Of six planned arms, three died to an OOM that was not ours:

```
GPU 0 has a total capacity of 191.69 GiB of which 0 bytes is free.
50.00 GiB allowed; Of the allocated memory 9.58 GiB is allocated by PyTorch
```

The card was fully occupied by other tenants while our process held 9.6 GB. The
three that died — `v8_s0` (main), `v8_blind` (the attribution control) and
`v7_arch` (the architecture ablation) — are exactly the three the design needed.
The three that survived are stage-B-only arms that loaded `v8_s0`'s stage A.

### Results

Parameter ledger: evidence 116,288 · HRM 2,110,470 · relational 2,355,400 ·
planner 1,883,146.

| arm | `std_ratio` | `wp_std_ratio` | `corr` | `grip` | `pose_mae` | `wp_mae_mm` | best `val bc` |
|---|---|---|---|---|---|---|---|
| `v8_s1` | **0.441** | 0.737 | 0.54 | 0.94 | 0.151 | 68.2 | 0.2217 |
| `v8_s2` | **0.563** | 0.953 | 0.53 | 0.94 | 0.140 | 95.0 | 0.1801 |
| `v8_norel` | 0.449 | 0.907 | 0.44 | 0.94 | 0.166 | 57.8 | 0.1935 |

**Every v8 arm beats every one of the 19 v7 arms on `std_ratio`.** The best v7
arm ever measured was 0.253 (`longh_liberoonly`, 4m); the typical arm was
0.02–0.19. The worst v8 arm is 0.441.

### The relational head is the planner's principal input

Pose-only sensitivity when an input is withheld:

| arm | `relational` | rank | next-largest |
|---|---|---|---|
| `v8_s1` | **0.1355** | **1 of 11** | `state_delta` 0.0730 |
| `v8_s2` | **0.1222** | **1 of 11** | `proprio` 0.1167 |
| `v8_norel` | 0.0705 | 3 of 11 | `proprio` 0.1395 |

Across all of v7, `fused` was the only stable visual reading at 0.03–0.10 and
proprioception dominated. Object evidence is now the largest single
contribution. This is the one v8 design claim the run supports.

### RETRACTION: the displacement-vs-action gap was largely an artifact

4m's headline — "displacement regresses 3.0x–29.1x less shrunk than action,
across 19 arms, measured within one forward pass" — does not survive a sighted
corpus:

| | v7 (19 arms) | `v8_s1` | `v8_s2` | `v8_norel` |
|---|---|---|---|---|
| `wp_std_ratio` / `std_ratio` | 3.0–29.1x | **1.7x** | **1.7x** | **2.0x** |

The ratio collapsed because the ACTION head improved (0.25 -> 0.44–0.56) while
the waypoint head held (0.74–0.95). The gap was therefore mostly a symptom of an
action head starved of object evidence, not a general fact about target
parameterization. What survives is the weaker claim: displacement still
regresses somewhat less shrunk than action (1.7–2.0x, three arms), and the
within-forward-pass measurement remains sound — the magnitude of the effect was
inflated by a blind corpus.

This is the second time a v7 "most robust claim" has failed once its measurement
conditions changed, and both times the cause was the same corpus defect.

### What is still broken

* **`wm_margin −46.8%`, identical across all three arms.** They share one stage
  A that OOM'd at epoch 3 during the horizon ramp, before it saved anything at
  max horizon. The world model is 47% WORSE than persistence. Notable
  consequence: the planner reached `std_ratio` 0.5 anyway, so **these numbers
  are a floor, not a ceiling** — no arm here has ever had a working world model.
  For contrast, a Mac run on 500 `libero_object` episodes reached val 0.0388 vs
  persistence 0.0402 (`wm_margin +3.5%`) from a cold start.
* **Closed-loop success is 0.000 across all 9 runs** (3 arms x 3 suites, 5
  trials x 10 tasks each). The policy now emits correctly-scaled, correlated
  actions and completes nothing. Perception can no longer be the explanation,
  which leaves exposure bias — consistent with the 4m telemetry showing a
  near-constant emitted direction off-distribution.
* **Seed spread is n = 2** (0.441 vs 0.563, spread 0.123). `v8_norel` at 0.449
  sits inside it, so the ablation shows nothing — and it was void anyway (below).

### Defects found in this run

1. **The relational ablation measured nothing.** `--planner-drop` is applied
   before the v8 block re-adds `relational` unconditionally, so
   `--planner-drop relational --v8` silently kept it; the arm's own header
   prints `inputs (... 'relational')`. Its bench does show `relational` demoted
   to rank 3 behind proprio, but the input was present, so the arm is a seed
   replicate rather than an ablation.
2. **Skipping the bake skipped the gate.** An already-baked suite was added to
   the corpus with `continue`, bypassing the sighted check — so `libero_spatial`,
   which had already FAILED with 500 blind episodes, would be silently
   re-included on the next run. Being cached says nothing about being sighted.
3. **Partial bakes passed unnoticed.** An earlier run trained on 145 episodes
   (85 + 60 against ~500/suite) because the script gated on detection RATE but
   never on COUNT. Stage A never beat persistence, which reads as an
   architecture failure and was a data failure.
4. **A fixed batch size on a shared box.** The OOM above is not a model-size
   problem; it is contention. Arms now retry down a batch ladder.

## 4p. THE ACCURACY BAR: LIBERO tolerates ~5% magnitude error, and we are 2-4x outside it

Measured 2026-07-28 on RunPod. This supersedes every prior explanation of the
closed-loop zero and makes further architecture work on that failure pointless
until it is addressed.

### Three controls, in order

**1. The environment is sound.** `eval/replay_check.py` replays a demo's own
actions from the demo's own recorded initial state:

```
replay success 5/5 = 1.000
```

So the eval harness, controller (OSC_POSE), init-state selection and 7-dim
action convention are all correct. Every closed-loop 0.000 this project has
recorded was measuring a policy, not a broken pipeline. That question had been
open since 4e and is now closed.

**2. The actuation path is faithful.** `eval/actuation_check.py` pushes
ground-truth actions through each deployed stage:

| stage | success | `|a|` vs raw |
|---|---|---|
| raw | 5/5 | 1.000x |
| normalizer round-trip | 5/5 | 1.000x |
| trust brake @ deployed mean | 5/5 | 1.000x |
| both | 5/5 | 1.000x |

The normalizer round-trips EXACTLY (max `|a - inverse(norm(a))|` = 0.0000). The
15% of dims reported "clipped" are the +/-1 gripper column, which is lossless.

**3. The task tolerates almost no magnitude error.** Scaling the POSE columns of
ground-truth actions (gripper untouched), 4 demos each:

| scale | 1.30 | 1.20 | 1.10 | **1.05** | **1.00** | 0.95 | 0.90 | 0.85 | 0.80 | 0.60 | 0.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| success | 0.00 | 0.00 | 0.25 | **1.00** | **1.00** | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**The passing band is approximately [0.95, 1.05].** A 5% magnitude error halves
success; 10% eliminates it.

### Why this ends the search

`std_ratio` is precisely the ratio this sweep varies — emitted action magnitude
over demonstrated. Measured across every arm this project has trained:

| stack | `std_ratio` | inside [0.95, 1.05]? |
|---|---|---|
| v7, 19 arms | 0.02-0.25 | no, 4-50x low |
| v8, sighted corpus | 0.44-0.56 | no, ~2x low |
| v8_s0 (best world model, `wm_margin` +43.3%) | 0.264 | no, ~4x low |

No arm has ever been within a factor of two of the passing band. **The 0.000 is
fully explained by magnitude, and no change to perception, fusion, the world
model or the relational head could have fixed it** — those improve *direction*,
and `corr` reached 0.54 while success stayed at zero. The grounding work in
4n/4o was worth doing and was solving a different problem than the one that
gates the benchmark.

### The caveat that must travel with this number

This is an OPEN-LOOP replay: the full action sequence is executed from the
initial state with no feedback, so a constant magnitude error integrates over
~150 steps into a large terminal position error. A closed-loop policy observing
state each tick can in principle correct such drift, and the band for a
feedback controller would be wider.

It applies to our failure anyway, and this is the substantive point: MSE
behaviour cloning shrinks toward the conditional mean, which is a **systematic,
signed** magnitude deficit, not zero-mean noise. A systematic deficit does not
cancel across steps — it integrates exactly like the open-loop bias this sweep
imposes. Random noise of the same variance would be far less damaging. So the
measurement is the right one for the failure we have, and the honest statement
is: *for a policy with systematic magnitude shrinkage, LIBERO's tolerance is
about +/-5%.*

Establishing the band for a genuinely feedback-corrected controller is a
separate experiment and is not claimed here.

### What follows

1. **The delta brake can only hurt on this benchmark.** It scales commands DOWN
   by `min(1, tau/brake_trust)`, and down is fatal: at trust 0.40 (scale 0.80)
   ground truth scores 0/4. Deployment ran at trust mean 0.521, min 0.119, so
   roughly half of all steps were attenuated. It must default off for `delta`
   action spaces until a feedback-corrected band is measured.
2. **The stage-B objective needs a magnitude term.** Pure MSE provably converges
   to the conditional mean and therefore shrinks; the architecture is not
   choosing to shrink, the loss is. Candidates: an explicit `|a_pred| / |a_demo|`
   penalty, or a variance-matching term.
3. **`std_ratio` should be reported against the band, not as a bare number.** A
   run at 0.56 is not "better than" 0.26 in any way that matters if both are
   outside [0.95, 1.05].

## 4q. The v8 architecture, and the first world model that clearly works

Locked 2026-07-26, first trained end to end 2026-07-28. Contract in `DESIGN.md`
("v8 plan"); weights and their paired stats in `checkpoints/v8_pod/`.

### What changed and why

v8 replaces three of the five trainable modules. Each change answers a specific
measured failure, not a design preference.

```
perception (frozen YOLO-World-S)  — DATA RICH: K=8 class-agnostic proposals at
   |                                full 512-d, from the SAME detector forward
   |  frame_emb [512] + obj_emb [8,512] + obj_center [8,2] + obj_weight [8]
   v
HRMBackbone            replaces AnchoredDriftEncoder
   |  slow module steps on REAL ticks (2 Hz); fast module every tick (30 Hz)
   |  -> state [256] + learned per-axis control gains
   v
RecursiveTRM           unchanged contract, residual convention preserved
   |  -> next_emb [512]
   v
RelationalHead         replaces SlotResonanceFusion, runs AFTER the TRM
   |  cross-attn(object tokens x predicted latent x text)
   v
ChronoQueryPlanner  -> plan [5,7] + waypoint [5,3]
```

**Ordering.** v7 ran fusion -> TRM. v8 runs **TRM -> relational**, so
object-object reasoning conditions on the latent the planner actually consumes
rather than on a separate pre-TRM summary. The TRM is the one component with a
consistently positive result, so it goes first and its contract is untouched;
`EvidenceEncoder` (116,288 params) feeds its unchanged `[B,32,5]` port.

**Why an HRM.** Its two timescales are not an imported abstraction — the
hierarchy already exists in the deployment loop. It absorbs three jobs v7 did
separately: drift encoding, the hand-fitted proportional gains of
`fit_waypoint_gain.py` (now learned outputs), and long-horizon reasoning.

**Why class-agnostic proposals.** 4n established that role-conditioned detection
returns 0.000 on LIBERO product names. The same detector forward already
produces every other box; v8 keeps them instead of discarding them, so the
relational head reasons over a scene rather than two argmax slots.

### Parameter ledger

| module | params | cap |
|---|---|---|
| evidence (`EvidenceEncoder`) | 116,288 | — |
| hrm (`HRMBackbone`) | 2,110,470 | 3,000,000 (raised from 1.5M) |
| relational (`RelationalHead`) | 2,355,400 | 5,000,000 (inherited from fusion) |
| planner (`ChronoQueryPlanner`) | 1,883,146 | 2,500,000 |
| **trainable total** | **6,465,304** | 9,000,000 |

Against v7's 6,988,685 — v8 is SMALLER while adding relational reasoning, a
learned control law, and an 8-object scene representation.

### The world model result

Stage A on 1000 sighted episodes (`libero_object` + `libero_goal`), RTX A4500:

```
[stage A] epoch 44 | H=6 | val 0.0291 vs persistence 0.0504 (BEATS persistence)
[stage A] early stop at H=6, best val 0.0287
```

**`wm_margin +43.3%`** — the best this project has produced, by more than 2x:

| stage A | corpus | `wm_margin` |
|---|---|---|
| **v8_s0 (this)** | 1000 ep, sighted | **+43.3%** |
| v7 `full_stageA_wrist_v72` | wrist, blind | +19.8% |
| v8 Mac cold start | 500 ep, sighted | +3.5% |
| v8 ROCm (OOM'd mid horizon-ramp) | 1000 ep | -46.8% |

Two things made it possible. The corpus can see (4n): source objects detected on
44-48% of frames against 0.0% for every earlier corpus. And gradient
checkpointing held peak allocation at **1.5 GB** where the unchecked rollout
graph had grown to 9.3 GB and lost a race for memory on a shared card — the
world model had never before been given a full horizon ramp to train through.

### Bench, and the honest reading

`full_stageB_v8_s0.pt`, 20 held-out episodes, stage B still mid-training:

| metric | value |
|---|---|
| `wm_margin` | **+43.3%** |
| `std_ratio` | 0.264 |
| `corr` | 0.48 |
| `grip_acc` | 0.94 |
| `pose_mae` | 0.202 |
| `relational` sensitivity | 0.0940 (2nd of 11, behind `proprio` 0.1433) |

The world model and the gripper are genuinely good. `std_ratio` 0.264 is **~4x
below the [0.95, 1.05] band 4p measures as necessary**, so this checkpoint
cannot succeed closed-loop regardless of the rest, and reporting its `corr` or
`grip_acc` as progress toward Claim 1 would be misleading. v8's contribution is
to the world model and to grounding; the magnitude problem is orthogonal to it
and is addressed in 4p.

### 4p-CORRECTION: magnitude is necessary but NOT sufficient

The eval-time gain sweep tests 4p's implied conclusion directly — if direction is
right and only scale is wrong, a constant multiplier should rescue the task with
no retraining. `full_stageB_v8_s0.pt`, `libero_object`, 3 tasks x 2 trials:

| `--action-gain` | 1.0 | 2.0 | 3.0 | 3.8 | 5.0 | 8.0 |
|---|---|---|---|---|---|---|
| `mean_success` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**No gain rescues it**, including 3.8x, which is exactly the multiplier that
would carry `std_ratio` 0.264 into the measured passing band.

**What survives.** The band itself is a ground-truth measurement and stands:
scaling DEMO actions to 0.90 gives 0/4, so any policy outside ~[0.95, 1.05]
fails. Magnitude remains a *necessary* condition, and every arm this project has
trained violates it.

**What does not.** The inference "therefore fixing magnitude fixes the task" is
wrong, and 4p should not be read that way. Rescaling multiplies the error
component as much as the signal: ground truth has `corr` 1.0 by construction, so
the sweep preserved perfect direction while varying only scale, whereas this
policy sits at `corr` 0.48 — roughly half its commanded direction is wrong, and
amplifying it amplifies the error too. A scalar gain cannot separate the two.

**The honest joint statement:** LIBERO requires magnitude within ~5% AND
direction far better than `corr` 0.48, and neither alone is enough. The
`--action-gain` knob is retained as a diagnostic, and its negative result is the
reason the actuation loss (which trains magnitude and direction jointly, through
the command that is actually emitted) is the right fix rather than an
inference-time correction.

This is the second time in this project that a clean measurement supported a
conclusion that a cheap follow-up then refuted — the first being the
displacement-vs-action ratio in 4m/4o. Both times the measurement was sound and
the inference from it was not.

## 4r. The wrist view is object-poor: K=8 slots, 0.68 objects per frame

Weight analysis of `full_stageB_v8_s0.pt` found the evidence port's object slots
`obj2`-`obj7` sitting at ~2.00 block norm, essentially identical to each other,
while `obj0` (2.565) and `obj1` (2.387) had moved. Only two slots trained. The
cause is the corpus, not the encoder.

### Occupancy, measured over 637 frames of `libero_object_v8`

| slot | % of frames used | mean weight |
|---|---|---|
| 0 | 52.7% | 0.1255 |
| 1 | 13.2% | 0.0231 |
| 2 | 1.9% | 0.0024 |
| 3 | 0.2% | 0.0002 |
| 4-7 | **0.0%** | 0.0000 |

```
proposals per frame: mean 0.68 | median 1 | max 4
frames with exactly k objects: {0: 301, 1: 252, 2: 72, 3: 11, 4: 1}
```

**47% of frames contain no detected object at all**, and slots 4-7 are never
occupied in the entire sample. A slot at weight 0 receives no gradient, which is
exactly why those blocks are still at initialization — the graded-evidence
design working as specified, on evidence that is not there.

### What this means for v8

The "data rich, K=8 class-agnostic proposals" premise (`DESIGN.md` v8, 4o) is
not realized by this corpus. The relational head — 2,355,400 parameters built to
reason about object-object structure — is being handed an empty scene on half
its frames and a single object on most of the rest. Its measured planner
sensitivity (0.0940, second behind `proprio` 0.1433) is therefore an upper bound
on what it could show, not a verdict on the design.

Earlier probes make the cause specific: on the SAME scenes the third-person view
yields **3.40 proposals/frame** against the wrist view's **1.40**, and per-prompt
firing rates are roughly double. The wrist camera at 128x128 is close to the
objects, frequently occluded by the gripper, and often pointed at table surface;
it is a hard detection target regardless of prompt engineering, and 4n's prompt
work raised the source rate from 0.0% to ~48% without changing that ceiling.

### The tension this exposes, and a way out

4f established that the bake camera MUST match the eval camera, and eval reads
`robot0_eye_in_hand_image`. That coupling is what forces detection onto the
object-poor view.

It is avoidable. LIBERO's observation dict carries `agentview_image` alongside
the wrist image at every step, so a policy may legitimately consume BOTH: the
third-person view for OBJECT DETECTION (3.4 proposals/frame) and the wrist view
for the ego frame embedding the world model predicts. Nothing about the
deployment story forbids it either — a real rig with a fixed scene camera plus a
wrist camera is the ordinary configuration, and the Pi budget is unchanged
because detection already runs at 2 Hz on one image.

This is proposed, not measured. What is measured is that the current
single-wrist-view corpus cannot support the relational head's premise, and that
any conclusion about relational reasoning drawn from these checkpoints is
bounded by 0.68 objects per frame.

## 4s. The actuation loss works — and every prior v8 closed-loop number was void

### The fix

4r/weight analysis found the HRM's learned control law had never received a
gradient: `gain_head.weight` and `.bias` were EXACTLY zero (bit-for-bit their
init) and `log_gain_base` was still the hand-fitted least-squares prior. Three
independent causes, all in the v8 wiring:

1. `DriftAdapter.forward` returned only `.state`, discarding `.gains`, so
   nothing downstream referenced the head.
2. Stage B froze the entire HRM. The control law converts a predicted
   displacement into an emitted command, which makes it POLICY, not world model.
3. **No loss could see it.** `waypoint_loss` supervises displacement in METRES,
   upstream of the gain that turns displacement into a command — so emitted
   magnitude appeared in no objective in this project, ever.

`--actuation-weight` supervises the actuator's own law,
`cmd = gain_scale * disp * range / (gain * steps)`, against the demo's pose
action, so it trains exactly what runs.

### Result: `v8_act` vs `v8_s0`, same stage A, same corpus

| metric | `v8_s0` | `v8_act` | |
|---|---|---|---|
| **`wp_std_ratio`** | 0.121 | **1.097** | **9.1x**, and inside the band 4p measures |
| `std_ratio` | 0.264 | 0.421 | +59% |
| `corr` | 0.48 | 0.55 | |
| `pose_mae` | 0.202 | 0.144 | |
| `grip_acc` | 0.94 | 0.94 | |
| `relational` sensitivity | 0.0940 (2nd) | 0.0923 (**1st**) | first time above `proprio` |

The learned gains moved off the prior: `[0.01085, 0.01306, 0.01180]` ->
`[0.01133, 0.01325, 0.01456]`, with `gain_head` non-zero at 4.1e-02. The control
law is being learned rather than tuned, as designed.

`wp_std_ratio` is the quantity that drives the deployed actuator, and 1.097 is
the first time any head in this project has landed near the [0.95, 1.05] band.
`relational` overtaking `proprio` is the first evidence that object evidence is
the planner's primary input rather than a decoration.

### VOID: every v8 closed-loop number reported before 2026-07-28

`eval/policy.py` moves fusion, drift, trm and planner to the heads device
explicitly. `relational` was added later and missed, so every v8 closed-loop run
died at the FIRST TICK with

```
RuntimeError: Expected all tensors to be on the same device, but got mat1 is on
cuda:0, different from other tensors on cpu
```

and the harness summarised it as `tasks_completed 0`, `mean_success 0.000` —
which in the results JSON is indistinguishable from a policy that merely never
succeeds. **Every v8 closed-loop 0.000 in 4o and in `results/V8_TABLE.md` was
measuring a crash, not a policy**, and is withdrawn. The v7 numbers are
unaffected; v7 has no relational head.

After the fix, episodes actually execute (`tasks_completed 2`, 200 steps each).
The re-measured numbers are reported in 4t.

This is the third instance in this project of a null result that was an
instrumentation failure rather than a finding — after bench scoring TQSA
checkpoints blind (§0) and the blind corpus (4n). All three shared a signature:
a failure that produced a PLAUSIBLE number instead of an error.

## 4t. The first honest v8 closed-loop numbers — and the fourth instrumentation failure

With the device bug of 4s fixed, both v8 arms ran to completion for the first
time: 10 tasks x 3 trials x 300 steps, `tasks_completed 10` on each.

| arm | mean_success | tasks_completed | trials |
|---|---|---|---|
| `v8_act` (actuation loss) | **0.000** | 10/10 | 3 |
| `v8_s0` (no actuation loss) | **0.000** | 10/10 | 3 |

These are real measurements, not crashes: 9000 executed ticks with full
telemetry. And the telemetry says the policy was not merely inaccurate — it was
**saturated**, in a way that makes success impossible rather than unlikely.

### The emitted actions, against the corpus the policy was trained on

| quantity | emitted (9000 ticks) | corpus (row 0) |
|---|---|---|
| gripper, unique values | `{-1.0}` — one value, std **0.000000** | closes (>0) on **52.3%** of frames |
| z, mean | **-0.949** (max -0.432) | -0.037, std 0.490 |
| net EEF displacement | 0.267 m mean | — |
| lowest z minus start | -0.251 m | — |

The gripper never closed, on any tick, of any trial, of any task. A pick-and-place
policy whose gripper never closes cannot score above zero, so nothing else about
these runs needed explaining — the accuracy question (4p) never arose. The z
column tells the same story from the other side: pinned near the tanh bound, the
arm drove 0.25 m straight down and stayed there.

### Why saturation, when the same checkpoint scores 0.94 open-loop

Stage B's own validation for this checkpoint:

```
[stage B] epoch 20/60 | grip_acc 0.940 | val bc 0.2055 wp 0.3480 grip 0.934 *best*
```

**0.94 gripper accuracy on held-out data**, against a 0.477 always-open baseline.
The head is not degenerate and the column is not untrained. The difference
between 0.94 and a single saturated value is therefore entirely in the INPUTS.

The telemetry names the input difference directly. On the 600 real ticks of these
runs:

| grounding | closed-loop eval | the corpus it trained on |
|---|---|---|
| source detected | **0.0%** of real ticks | **48%** of frames |
| target detected | 20% of ticks, mean conf **0.007** | — |

**The policy was trained sighted and deployed blind.**

### Root cause: the blind-corpus fix reached the corpus and never the robot

4n fixed grounding by replacing the bare task phrase with a chain of concrete
visual categories, taking source detection from 0% to 48%. That fix was written
into `preprocess/common.py` — the bake path. `microvla/jepa/loop.py`, the
DEPLOYMENT path, kept building its own prompts:

```python
def _role_prompts(phrase):        # microvla/jepa/loop.py, before the fix
    return [phrase, strip_article(phrase)]      # -> ["alphabet soup"]
```

which is exactly the prompt 4n measured at **0.000**. Every closed-loop eval this
project has ever run — v7 included — grounded with the prompts that do not
ground.

Writing the parity test surfaced a second, independent divergence on the same
call: the bake applied `strip_article` before building chains and the loop passed
the parser's raw phrase, so the two sides sent `"alphabet soup"` and `"the
alphabet soup"` to the same detector. Both are now fixed by moving the chains to
`microvla/perception/prompts.py`, which both sides import, and normalizing inside
`role_chains` so no caller can re-introduce the skew.

### What made this invisible

Both sides were tested, and both test suites passed throughout.
`tests/test_prompt_fallbacks.py` checked that the bake built good chains;
`tests/test_grounding.py` checked that the loop built the chains the loop was
written to build. **Neither compared the two.** A test that pins each side to its
own behaviour cannot see a divergence between them, and the more thorough each
suite is, the more confidence it lends to the skew.

The general form, and the reason this keeps happening here: *a fix applied at one
end of a train/deploy pair is not a fix, and the test that would have caught it
is a comparison, not an assertion.* The same shape produced 4n itself (bake and
detector disagreed about what a class name was) and 4s (the eval harness and the
policy disagreed about what device meant).

This is the **fourth** instrumentation failure in this project that produced a
plausible number instead of an error — after bench scoring TQSA blind (§0), the
blind corpus (4n), and the device bug (4s). Four for four, the null result was
the instrument. The practical consequence for how these numbers should be read:
**a 0.000 from this stack is not evidence about a policy until something
independent confirms the policy ran under the conditions it was trained for.**
The three-experiment chain of 4p (replay 5/5, stage isolation, magnitude sweep)
established that the ENVIRONMENT can be solved; it did not establish that the
policy was being fed what it was trained on, and that gap is where all four of
these bugs lived.

### 4t-CORRECTION: grounding was not what saturated the planner

4t attributed the saturated policy to blind deployment: the policy trained with
48% source detection and deployed with 0.0%, so its inputs were off-distribution
and its outputs collapsed. The fix (commit `fbc7f8d`) worked as a grounding fix
and **refuted the causal claim.**

Re-running `v8_act` unchanged, with the shared prompt chains live:

| | before the fix | after the fix |
|---|---|---|
| source detected | 0.0% of real ticks | **75.8%** (mean conf 0.058) |
| target detected | 20.0%, conf 0.007 | 3.0%, conf 0.001 |
| gripper, unique emitted values | `{-1.0}` | **`{-1.0}`** |
| gripper closes | 0.0% of ticks | **0.0%** of ticks |
| z, mean | -0.949 | **-0.948** |
| mean_success | 0.000 | **0.000** |

Deployment is now genuinely sighted — better than the corpus it trained on — and
**not one digit of the policy's behaviour moved.** The gripper is still pinned at
exactly -1.0 with std 0.000000 across 9000 ticks. Blind grounding was a real
defect and a real train/deploy divergence, but it was not the cause of the
saturation, and 4t's causal claim is withdrawn. What survives from 4t is the
measurement (the first honest v8 closed-loop numbers) and the root-cause analysis
of the grounding defect itself; what dies is the inference from it.

Note also that the target role got WORSE (20% -> 3.0%). Disjoint role chains push
the target onto "basket"/"bin", which the wrist camera rarely sees — consistent
with 4r's finding that the wrist view is object-poor. Source and target grounding
are not one problem.

### The next candidate: proprio orientation is a different QUANTITY at deployment

Since the inputs still had to explain a 0.94-open-loop head emitting one constant
value, the input comparison was extended past grounding to proprio:

| proprio dims 3:7 (orientation) | value |
|---|---|
| corpus, per-frame mean | `[3.108, -0.104, -0.088, 0.0]`, 4th slot zero on **100%** of frames |
| live env, at reset | `[1.0, 0.0, -0.028, -0.0]` |

`_ORI_KEYS = ("ee_ori", "robot0_eef_quat")` tried LIBERO's baked key and fell
back to robosuite's live key — but `ee_ori` is an axis-angle rotation VECTOR (3)
and `robot0_eef_quat` is a QUATERNION (4, xyzw). Two different quantities, packed
into the same four slots, chosen by which key happened to exist. The bake path
gets axis-angle; the deployed policy got a quaternion, every tick of every
episode, with the first component moving from ~pi to ~1.0.

Converting that same env pose through `quat2axisangle` yields `[pi, 0, -0.088]`,
which lands on the corpus mean `[3.108, -0.104, -0.088]` to three decimals — so
the conversion is confirmed by construction rather than by plausibility. Fixed in
`87c133a`.

**This is a candidate, not a confirmed cause.** It is a verified divergence with a
verified conversion; whether it unsaturates the policy is an open measurement,
and 4t's correction is precisely what happens when that distinction is skipped.
The result is reported in 4u.

### The pattern, now four for four (and the reason to distrust the next fix too)

Every defect found in this stack so far is the same shape: **two sides of a
train/deploy pair disagreeing about what a value means, with each side
individually correct and individually tested.**

| # | the two sides | what they disagreed about |
|---|---|---|
| §0 | bench vs checkpoint | whether TQSA weights were loaded |
| 4n | bake vs detector | what a class name is |
| 4s | eval harness vs policy | what device the heads live on |
| 4t | bake vs deploy | what a detection prompt is |
| — | bake vs env | what an orientation is (axis-angle vs quaternion) |

None of these is a modelling error and none would be caught by a better loss, a
bigger model, or more data. All five produced a plausible number instead of an
error. The methodological consequence for this paper: **any single-sided test is
evidence about one side only**, and the tests that actually caught these are
comparisons — bake output against deploy output, on the same input. Where such a
comparison does not exist, the corresponding number should be read as unverified.

## 4u. It is not the inputs, and it is not covariate shift: the deployment stack does not reproduce its own training metric

4t and its correction chased train/deploy input divergences: grounding prompts
(real, fixed, 0% -> 75.8% source detection) and proprio orientation (real, fixed,
quaternion -> axis-angle). Neither moved the policy. Frame orientation was
checked and ruled out — a live wrist frame matches the stored one as-is (mean
abs error 48.98) far better than any flip (65.77-71.73), so the bake and the env
agree on orientation.

Guessing one input at a time does not terminate, so the planner was instrumented
instead (`eval/planner_probe.py`): a forward hook captures every tensor the
planner is ACTUALLY handed at deployment, compared against the corpus it trained
on.

| planner input | deploy mean / std | corpus mean / std |
|---|---|---|
| `current_emb` | 0.0000 / 1.0000 | 0.0000 / 1.0000 |
| `proprio` | 0.3773 / 1.0626 | 0.3983 / 1.0355 |
| `relational` | -0.0001 / 0.9990 | — |
| `fused` | -0.0891 / 0.9900 | — |

Per-dimension proprio agrees at the reset state; the two dimensions that drift
(EEF `z` 0.048 vs corpus 0.205, gripper always open) are CONSEQUENCES of the
policy's own descent, not causes. **The planner's deployment inputs are in
distribution.**

Two explanations survived that: a defect somewhere in the stack, or ordinary
covariate shift (the policy is fine on demonstrated states and its own early
errors take it off-distribution). They produce identical closed-loop telemetry,
so closed-loop measurement cannot separate them. `eval/openloop_check.py` does,
by teacher-forcing the real `MicroVLAPolicy`/`JEPALoop` with a demonstration's
own frames and proprio — the policy never leaves the demonstrated distribution.

| demo | steps | gripper agreement | demo closes | we close | pose corr |
|---|---|---|---|---|---|
| demo_0 | 148 | 0.459 | 58.1% | 4.1% | 0.456 |
| demo_1 | 179 | 0.464 | 54.7% | 7.8% | 0.270 |
| demo_2 | 136 | 0.485 | 62.5% | 11.0% | 0.362 |
| **mean** | | **0.469** | **58.5%** | **7.6%** | 0.363 |

**On the demonstrations' own frames the gripper closes on 7.6% of steps where the
demonstration closes on 58.5%, and agreement is 0.469 — chance.** The same
checkpoint's stage-B validation reports `grip_acc 0.934` and its training epochs
0.94. The deployment stack therefore does not reproduce stage B's own metric on
the data stage B measured it on, and **the closed-loop failure is a defect, not
covariate shift.** This also retires the "compounding error" explanation without
having to argue about it.

### What it is not

The checkpoint loads exactly. Comparing every saved tensor against the loaded
module:

```
planner    saved 75 live 75 | not-in-live 0 | not-in-ckpt 0 | SHAPE-MISMATCH 0 | VALUE-DIFF 0
fusion     saved 10 live 10 | ... VALUE-DIFF 0
drift      saved 52 live 52 | ... VALUE-DIFF 0
trm        saved 28 live 28 | ... VALUE-DIFF 0
relational saved 38 live 38 | ... VALUE-DIFF 0
```

So it is not a silent `strict=False` partial load — the hypothesis that
`load_state_dict` had left the grip head at init, which the established defect
pattern made the obvious first guess.

### Mechanism correction

4t described the gripper as "saturated", implying a tanh driven to its bound.
That is wrong about the mechanism. The planner emits the gripper as a HARD
decision:

```python
grip_logit = self.grip_head(h).squeeze(-1)
grip = torch.where(grip_logit > 0, ones, -ones)     # {-1, +1}
```

so a constant -1.0 means `grip_logit <= 0` on every step, not an activation
pushed to its limit. The observation (one unique emitted value, std 0.000000)
stands; the explanation does not. The distinction matters because a hard
threshold means an arbitrarily small logit bias flips the entire behaviour — the
head does not need to be badly wrong to be uniformly wrong.

### Where this leaves the search

Inputs match, weights match, and the failure reproduces on in-distribution data.
What remains is the ASSEMBLY: the trainer and the loop each build the planner
call themselves, from the same modules, and only their agreement was never
tested. The trainer computes `wm = trm.forward_full(fused_t, delta_t, cur)` and
passes `planner(next_emb, current_emb=cur, ...)`; the loop computes its own
`fused`, `state_delta`, `geometry` and `relational` before an equivalent call.
Every prior defect in this project sat in exactly such a gap, so the next step is
the A/B this project keeps proving is the only thing that finds them: one corpus
episode, driven through both paths with perception held identical, comparing
planner inputs tensor by tensor.

## 4v. Root cause, fully attributed: two contract violations and one exposure bias

4u left one suspect — the planner CALL ASSEMBLY, since inputs and weights were
both verified clean and the failure reproduced on in-distribution data.
`eval/train_vs_deploy.py` drives one baked episode through the trainer path and
the deployment path with perception held identical (a replay perception returns
the corpus's own embeddings and boxes, so the detector is not a variable) and
diffs the planner's inputs tensor by tensor.

### The first diff

| planner input | rel. diff (train vs deploy) |
|---|---|
| `current_emb` | 0.0000 |
| `proprio` | 0.0000 |
| `state_delta` | 0.0000 |
| `pred_box_emb` | 0.0328 |
| `wm_latent` | 0.0477 |
| `geometry` | **0.2111** |
| `wm_msg` | **0.2915** |
| `fused` | **0.3444** |
| `relational` | **0.7455** |

and, on the same episode, the trainer path puts the grip logit above zero on
**47%** of steps (corpus closes on 53%) while the deployed path closes on **13%**.
Everything downstream of box evidence diverged; everything else was bit-exact.

### Defect 6: a real-tick detection MISS held stale evidence

Printing `geometry` per tick localized it immediately — the paths agree at
t0-t2 and split at t3, exactly where the corpus records a miss:

```
  t3 train  [0.5,   0.5,   0.5,   0.5,   0.0,    0.0   ]   <- zero evidence
     deploy [0.441, 0.795, 0.441, 0.795, 0.1561, 0.1561]   <- t2's box, decayed
  t4 deploy weight 0.1093     t5 deploy weight 0.0765
```

The loop's v5 "miss-hold" keeps a role's last-known box at `miss_decay ** age`
when the detector misses on a REAL tick. The bake does not: `preprocess` writes
weight 0 at the (0.5, 0.5) fallback, and `train_batched._boxes` feeds exactly
that. So the policy learned "weight 0 == no evidence" while deployment handed it
a confident stale box on precisely those ticks — turning "I see nothing" into
"the object is there, where it used to be". CLAUDE.md's evidence-fade rule
already states the contract ("missed detections pass 0"); the loop was the side
violating it. Now opt-in via `cfg.miss_hold`, default off, because re-enabling it
is a corpus decision — the bake would have to hold too. `geometry` went to
**0.0000**.

### Defect 7: staleness recovered by division collapsed on a double miss

`_rel_tokens` recomputed the staleness factor as
`box_weight.max() / max(source.conf, target.conf)` — the right ratio only while
some role was detected. On a tick where BOTH roles miss it is `0 / 1e-6 = 0`, so
every class-agnostic proposal was zeroed, while the trainer feeds the baked
proposal weights, which are non-zero on exactly those ticks *because proposals
are not role-conditioned*. The loop already knows the factor exactly (1.0 real,
`staleness_decay ** k` dream); it now passes it instead of recovering it.

A related conflation in the same function: `getattr(percept, "proposals", ())`
followed by `if props` treated an EMPTY proposal list as "this perception has no
proposals" and fell back to the two role slots. With 0.68 proposals per frame
(4r) that fallback fired on the majority of frames.

### The dominant cause: exposure bias in fusion's action token

Neither defect moved `fused` or `relational`. The attribution test settles it —
force the loop's fusion action token to the DEMO's previous action, i.e.
teacher-force exactly as stage B does:

| planner input | self-fed | teacher-forced |
|---|---|---|
| `fused` | 0.3384 | **0.0000** |
| `relational` | 0.7461 | **0.0161** |
| `wm_msg` | 0.2915 | **0.0114** |
| `wm_latent` | 0.0477 | 0.0069 |
| `pred_box_emb` | 0.0328 | 0.0026 |
| **gripper closes** | **13%** | **47%** (trainer: 47%) |

**With the action token teacher-forced, the deployment path reproduces the
trainer bit-for-bit and the gripper statistic matches exactly.** The entire
residual failure is that fusion's 8th token — the previously executed action —
is the DEMONSTRATION's action during training and the POLICY's own action at
deployment.

That token closes a feedback loop the training protocol never exercises: a wrong
action corrupts the token, the corrupted token worsens the next action, and the
system converges to a fixed point. A constant emitted gripper is what that fixed
point looks like from outside, which is why the collapse looked like saturation
in 4t. It also explains the whole measurement chain at once:

| condition | action token | gripper closes |
|---|---|---|
| stage-B validation | demo's (teacher-forced) | 93.4% accuracy |
| A/B, forced | demo's | 47% |
| A/B, self-fed | own | 13% |
| open-loop on demo frames (4u) | own | 7.6% |
| closed loop (4t) | own | 0.0% |

The four rows below the first differ only in how far the self-feeding compounds.
No input was ever out of distribution; the policy put itself there.

### Status

Defects 6 and 7 are fixed and verified by the A/B returning `geometry` to
0.0000. The exposure bias is a TRAINING-PROTOCOL defect, not a deployment bug:
it cannot be fixed in the loop, because the loop has nothing else to feed. The
fix is to train the action token the way it is deployed — scheduled sampling on
fusion's 8th token — which requires a retrain and is the next experiment.

Note what the pattern predicted correctly and what it did not. Six of the seven
defects were two sides disagreeing about a value; this one is the two sides
agreeing about the value's *meaning* while differing in its *provenance*, which
no parity test on a single tick can catch. The A/B found it only because it ran
a SEQUENCE and let the divergence compound.

### 4v-b. The regression barrier, and what it cost to not have one

Defects 6 and 7 are now pinned by `tests/test_train_deploy_parity.py`, which runs
the trainer's per-step assembly and the loop's over one synthetic episode —
including a tick where BOTH roles miss, the exact condition both defects needed —
and asserts `geometry`, `fused` and `state_delta` agree. CPU, mocks, in the
normal suite (425 tests, 15 s).

It includes an inverted test: re-introducing defect 6 (`miss_hold=True`) must
BREAK parity. Without that, a refactor could quietly stop exercising the loop and
the parity assertions would keep passing while measuring nothing — which is the
exact failure mode of the four instrumentation nulls in 4t, restated as a test.

The accounting is worth stating plainly, because it is the paper's methodological
claim in concrete form. Before this file existed the project had 419 passing
tests, ~2200 lines of paper, and **two** test suites that separately certified
the two sides of the divergence. What that bought: a blind corpus (4n), a void
30-trial batch (4s), three false root causes in 4t/4u, and roughly two days of
GPU time spent measuring crashes and instrument artefacts. The test that would
have caught all of it is 180 lines and runs in half a second, and it is a
COMPARISON — neither side asserted anything new about itself.

Generalization, stated as a rule this project now follows: *wherever a value is
produced on one side of a train/deploy pair and consumed on the other, the test
must run both sides and diff. An assertion about either side alone is evidence
about that side alone.* Single-sided tests are not worthless — they caught real
bugs here — but they are structurally incapable of seeing the class of defect
that has caused every headline failure in this project.

### 4v-c. A limitation of the scheduled-sampling fix, stated before its result

`--action-token-sampling` is applied in stage B only, and that is a partial fix
by construction. Stage A trains the world model with `real_paths`, which feeds
fusion `batch["pwm_targets"][:, t-1, 0]` — the demonstration's action — at every
timestep, and it has no alternative: there is no policy yet, so there is no "own
action" to substitute. The TRM's notion of what a fused matrix looks like is
therefore built entirely from teacher-forced action tokens, and at deployment it
receives fused matrices built from the policy's own.

So stage B can teach the PLANNER to read a self-fed fused matrix, but the world
model underneath it is still trained on a distribution it will never see. If the
retrain moves the closed-loop number only partway, that asymmetry is the first
place to look, and the natural remedy is noise or dropout on stage A's action
token (which needs no policy) rather than sampling.

Recording this now, before the result, because the honest reading of a partial
improvement and the honest reading of a failure are different, and deciding
which after seeing the number is how 4t's causal claim got made.

## 4w. The world model's dt, and the end of the mechanical explanations

### Defect 9: one TRM step is 10 env steps, and the loop applied it every step

The TRM is trained on the baked corpus, whose stride is the source control rate
divided by `real_frame_hz` — LIBERO: 20 Hz / 2 Hz = **10 env steps, 0.5 s**. So
one TRM step means "half a second later". Measured, 40 episodes, one step:

| one TRM step, cosine to | |
|---|---|
| the next 2 Hz sample (t+1) — what it was trained to predict | **0.9922** |
| persistence (t vs t+1) | 0.9910 |
| the sample after (t+2) | 0.9874 |

The deployment loop steps it **once per env step** and dreams 14 times between
real frames, so it extrapolates ~7 s of predicted time per 0.7 s elapsed — a 10x
temporal overshoot, compounded 14 times. `perception_period` is also 15 ticks
where the corpus stride is 10. Same defect family as the other eight: two sides
disagreeing about what a value means, here Δt.

`--chunk-exec` fixes it without retraining: advance the loop once per sample
interval and execute the plan's rows in between, which is exactly what the bake
defines a chunk to be ("the next `plan_steps` NATIVE-rate actions"). The waypoint
command is still recomputed every step against fresh proprio, so position stays
closed-loop while the latent advances at its trained rate.

### It changed nothing

| arm | per-tick (default) | chunk-exec, replan=10 | chunk-exec, replan=5 |
|---|---|---|---|
| `v8_ss05` | 0.000 | **0.000** | **0.000** |
| `v8_act` | 0.000 | **0.000** | — |

10 tasks x 3 trials each, `tasks_completed 10` throughout. The temporal
mismatch was real and is fixed; it was not what stood between this policy and a
completed task.

### What that means, stated plainly

Five mechanical defects have now been found, fixed, and individually verified —
prompts (0% -> 75.8% detection), proprio representation, miss-hold, staleness,
and Δt — plus scheduled sampling for the exposure bias of 4v, which improved
every open-loop metric it touched (val bc 0.2055 -> 0.1512, val grip 0.934 ->
0.949). **Closed-loop success is 0.000 after every one of them.**

The honest conclusion is that the mechanical explanations are exhausted. This is
no longer a stack with a bug in it; it is a policy that is not good enough, and
the remaining gap is capability rather than plumbing. Two measurements say so
directly:

* On the demonstrations' OWN frames, teacher-forced, the gripper closes on 7.6%
  of steps where the demo closes on 58.5% (4u). Nothing about deployment is
  implicated in that number.
* Pose correlation with the demonstrator is 0.27-0.46 open-loop, against a task
  whose measured tolerance on magnitude alone is ~[0.95, 1.05] (4p).

Two structural facts bound what BC can do here. The corpus is **500 episodes x
15 sampled frames = 7,680 decision points**, roughly an order of magnitude below
what LIBERO BC baselines train on, because the 2 Hz subsample discards 14 of
every 15 frames — a rate chosen for the PERCEPTION budget and then inherited, unexamined,
by the action supervision. And the objective is action MSE while the metric is
task completion, which 4p shows are not merely loosely coupled but opposed:
MSE-optimal regression shrinks magnitude, and shrinkage is precisely what the
task does not tolerate.

The next experiments therefore target capability, not correctness: task-aligned
objectives (a progress critic the planner descends through the world model, an
imagined rollout, and explicit variance matching — all differentiable, no
environment in the loop) and a re-bake that supervises actions at the control
rate rather than the perception rate.

## 4x. Exhaustive defect sweep — ten confirmed, and a retraction of 4s

Nine defects had been found one at a time, each by chasing a specific symptom.
That does not terminate and gives no coverage guarantee, so the codebase was
swept systematically: six independent lenses (train/deploy parity; timescales and
units; shapes and indexing; gradient flow and freezing; checkpoint and config
plumbing; eval-harness correctness), each reading the source with no knowledge of
the others' findings, followed by an adversarial verifier per candidate
instructed to REFUTE by default and to confirm only on a traced path from real
inputs to wrong behaviour.

**27 candidates; the 10 highest-severity were verified and all 10 survived; 17
lower-severity were not verified and are recorded as unverified.**

Four of the ten were introduced BY THE FIXES EARLIER IN THIS SAME DOCUMENT.

### The four that invalidate earlier results

**1. `Perception.proposals` was dropped on every real tick.** The loop rebuilds
`Perception` twice per tick — once for the device move (`_percept_to`), once for
the miss-hold, whose result becomes `self._last_percept` — and neither carried
the optional `proposals` field, which defaults to `()`. So `_rel_tokens` always
saw an empty tuple.

This made 4v's own fix actively harmful. That commit changed `if props:` to
`if props is not None:` on the premise that a perception supplying `proposals` is
a v8 detector; the premise is defeated one function above, where they had already
been stripped. Instead of falling back to the source/target role slots at their
real confidences, the deployed `RelationalHead` received **all-zero object
evidence on 100% of ticks**, while the trainer feeds the baked scene on the 52.7%
of frames that carry a proposal. An optional field with an empty default is
silent when dropped: no error, no shape change, evidence quietly replaced by
zeros.

`tests/test_train_deploy_parity.py` could not catch it — it constructs `JEPALoop`
with seven positional arguments, so `relational=None` and the path is never
exercised. The parity harness written specifically to catch this defect family
had a hole in exactly the place the next one appeared.

**2-4. The actuation loss was mis-specified in three independent ways.**

| | defect |
|---|---|
| units | `cmd` is in RAW action units; `Y` holds NORMALIZED targets. The bake is `fit_symmetric`, so normalized = raw / `q_high` — the objective asked for **+6.7% on x/z and +9.5% on y** over the demonstrator, on a task whose measured tolerance is [0.95, 1.05]. |
| row | `row` indexes the WAYPOINT grid (rows `waypoint_row_stride` = 10 control steps apart) while `Y` is the native-rate action chunk. The command to execute NOW was regressed onto the demo action 30 control steps in the future. |
| gain | the loss divides by the HRM's LEARNED gain; `eval/policy.py` divides by the FITTED gain from `waypoint_stats.json` and never reads the learned one. |

The gain defect is the consequential one. `g` is three numbers shared across the
batch, so the cheapest descent direction for a global magnitude error is to move
`g` rather than the displacement head — and `g` is not in the deployed path.

**Therefore 4s's headline is retracted.** `wp_std_ratio` 0.121 -> 1.097 was
reported as evidence that the actuation loss taught the head to emit
correctly-scaled commands. It is at least as consistent with the loss moving
three gain numbers to absorb a magnitude error the head never learned to fix,
against a target that was itself biased by 6.7-9.5% and taken at the wrong
timestep. That reading also explains what was otherwise puzzling: the metric
moved into the passing band and closed-loop success did not move at all. All
three are fixed; the quantity must be re-measured before it means anything.

### Two more that change how existing numbers should be read

**`--chunk-exec` over-commanded.** `replan_every` defaulted to
`waypoint_row_stride` (10) while a chunk holds `plan_steps` = 5 rows, so chunk
positions 0..9 mapped to rows 0,1,2,3,4,**4,4,4,4,4** — row 4 executed on 60% of
env steps. In a delta action space that is not "hold position", it re-commands
the same motion six more times. **The chunk-exec results in 4w were measured
under this defect** and are withdrawn.

**Every `--mock-env` success number is 1.000.** `MockLiberoEnv`'s success
threshold is crossed by an UNTRAINED policy in 3-5 steps, so `ours`, `persistence`
and `linear` all score 1.000 at every `perception_period`. CLAUDE.md mandates the
mock path and `eval/sweep.py` defaults to it, so any E4 "kill bar" comparison
computed there was degenerate — a fifth instance of a number that looked like a
result and measured nothing.

### What the sweep says about the method

The nine earlier defects were found by symptom-chasing; this sweep found ten more
in a single pass, four of them created by the previous round of fixes. Two
lessons, both uncomfortable:

* **Fixes are defect sources at the same rate as features.** Four of ten came
  from this session's own repairs, including one that made behaviour strictly
  worse than before the fix. A change made under time pressure to a system
  nobody fully holds in their head is not obviously net-positive, and nothing in
  the workflow was checking.
* **A regression barrier only covers what it exercises.** 4v-b claimed the
  parity test made this defect class impossible to reintroduce. It did not: the
  very next defect landed in the relational path the harness passes `None` for.
  The correct claim is narrower — it covers `geometry`, `fused` and `state_delta`
  on a v7 stack, and nothing else.

## 4y. First movement: the gripper starts closing

`v8_var` (scheduled sampling on the action token + variance matching, on the 2 Hz
corpus) is the first arm to move the metric that made success impossible.
Open-loop, teacher-forced FRAMES but the policy's OWN action token — i.e. the
condition 4u showed collapsing to 7.6%:

| | `v8_act` | `v8_ss05` | **`v8_var`** | demo |
|---|---|---|---|---|
| gripper closes | 7.6% | — | **25.1%** | 58.5% |
| agreement with demo | 0.469 | — | **0.644** | 1.000 |
| best val bc | 0.2055 | 0.1512 | 0.1580 | — |

A policy whose gripper never closes cannot score above zero on a pick task, so
7.6% was a hard ceiling of exactly zero regardless of everything else. 25.1% is
not yet 58.5%, and closed-loop is still 0.000, but it is the first time the
binding constraint has moved.

Note which intervention did it. Both terms attack the SAME defect from opposite
ends: scheduled sampling removes the shortcut (the demonstrator's previous action
predicts the current one, so the head never had to look at the scene), and
variance matching penalizes the shrinkage that MSE rewards. Neither is a new
input or a bigger model — both are corrections to a mis-specified objective.

That is consistent with 4w's conclusion and against the "needs more capacity"
reading: the stack could already predict the gripper at 0.94 accuracy when handed
the demonstrator's action, and what it lacked was an objective that did not let
it lean on that.

### 4x-b. Five more, from the unverified tail

The sweep's 17 unverified lower-severity candidates were worked through by hand.
Five were real, and two of them disabled a module outright.

**The HRM never saw the end-effector.** `HRMBackbone.forward` accepts `eef` and
builds `[eef, eef - anchor, validity]` through `eef_proj`, but `DriftAdapter`
called `self.hrm(frame_emb, is_real=True)` with no `eef`, and
`_eef_features(None, ...)` returns ZEROS. The entire metric branch contributed a
constant, `eef_proj` received gradient in no code path, and the module whose
stated job is "learned PID + drift + reasoning" was doing it **on vision alone**,
in training and at deployment. Fixed on both sides, gated on the proprio validity
flag.

This one is worth dwelling on because it is not a train/deploy mismatch — both
sides were wrong *identically*, so every parity test in the suite would pass. The
defect is invisible to comparison and visible only to asking "is this parameter
reachable at all?". A dead branch produces no error and no shape change; it just
quietly makes the model smaller than its parameter count claims.

**Three context-window conventions for one argument.** `rollout` (stage A) seeds
`ctx = [latent]`, so the window ENDS with the latent being predicted from; the
deployment loop appended AFTER the call, so its window held only PREVIOUS ticks;
stage B passed no context at all. The planner's `next_emb` in stage B therefore
came from a call shaped unlike anything deployment makes, and deployment fed the
TRM a window one step staler than any it was trained on. Unified.

**Scheduled sampling patched the wrong module.** `--action-token-sampling` fed
the model's own action to FUSION only. The v8 `RelationalHead` carries its own
action token and is the planner's dominant input, so the exposure bias of 4v
survived intact in the module that REPLACED fusion — the fix and the defect were
one function apart.

**`has_objects` was reduced across the batch.** `.max()` meant a single v8
episode in a mixed bucket sent EVERY episode down the baked path, and a v7
episode's `obj_*` are zero-filled, so those samples fed the relational head
nothing. Now per-sample.

**Validation ran the relational head in `train()` mode**, so `modality_dropout`
was active during the "clean" pass: best-checkpoint selection was scored with
random evidence withheld, differently every epoch.

**`--grip-weight` did not exist.** `split_planner_loss` takes it; the trainer
never passed it, so the gripper BCE sat at 1.0 and was never swept — on a task
where a policy that does not close scores exactly zero.

Running total: **nineteen** defects. Six of them were introduced by fixes made
earlier in this same document.

## 4z. Running log — corrected losses, dense supervision, and a spatial channel

Recorded as results land, including the ones that did not work.

### `fixed` — every loss correction, 2 Hz corpus, existing stage A

First arm trained against a correct objective (units, chunk row, fitted gain,
plus variance matching and scheduled sampling). Best val bc **0.1593**
(`v8_act` 0.2055).

| eval config | mean_success |
|---|---|
| plan-only, brake ON | **0.000** |
| plan-only, no-brake | **0.000** |
| plan-only, no-brake, chunk | pending |

Both waypoint-free configs are new: every closed-loop run before this one went
through the `WaypointActuator`, which overrides x/y/z, so the planner's own
regressed translation had never once been executed. It is not the difference.

Correcting the objective moved the OPEN-loop numbers (4y: gripper 7.6% -> 25.1%)
and has so far moved closed-loop success not at all. That is the same pattern as
every fix before it, and at some point the honest reading is that the objective
was never the binding constraint on THIS corpus — 7,680 decision points is an
order of magnitude below what LIBERO BC baselines use.

### Dense corpus — 4.9x the supervision

`--frame-hz 10` (stride 2 on 20 Hz demos): **500 episodes, T=74, 37,380
supervised decision points** against 7,680 at 2 Hz. The 2 Hz rate was chosen for
the PERCEPTION budget and then inherited, unexamined, by the ACTION supervision,
which has no reason to be subsampled at all — the demonstrator's actions exist at
every control step.

Re-launched once, deliberately: the first run predated the HRM-eef and
TRM-context fixes, and the HRM is FROZEN in stage B, so its control branch is
learned in stage A or never.

### The spatial channel — the architectural change

The clearest remaining capability gap, and not one an objective or more data can
close. `frame_emb` is a GAP of the SPPF map, so it discards WHERE — and on a
wrist camera WHERE *is* the servo error. The only spatial signal reaching the
planner was the two role-box centres, and 4r measured **0.68 proposals per
frame** on this view: on roughly half of all frames the policy had no spatial
information whatsoever. It was being asked to servo from a global average.

`TextQueriedSpatialAdapter` was built for exactly this and has never run in a
single trained checkpoint — every eval to date logs "checkpoint carries no TQSA
weights". The reason is cost: it needed raw frames stored and the frozen backbone
re-run every epoch.

The fix removes that cost rather than paying it. The frozen backbone already runs
once, at bake time, so its map is pooled to 4x4 and stored (`--spatial-grid 4`,
16 x 512 per frame, ~1.2 GB for the dense corpus). TQSA then trains directly on
the baked grid — no frames, no backbone, and it works on a `--no-frames` corpus.

Two safeguards, both aimed at this project's characteristic failure:

* the grid joins the BUCKET KEY, so one episode lacking it cannot strip it from
  the whole bucket (the exact mechanism that once left TQSA with 0/37 buckets);
* `--tqsa` now REFUSES to start when neither a grid nor frames is present,
  instead of training on nothing while reporting healthy losses.

The deployment loop prefers the baked grid too, so the two sides match in
RESOLUTION and not merely in content — feeding TQSA a 4x4 grid in training and a
full-resolution map at inference would be the resolution-flavoured version of
every other defect in this document.

---

# PAPER SKELETON (assembled 2026-07-29)

Everything above is a lab notebook. This is the argument it supports, in the
order a reviewer should meet it. Numbers marked **[pending]** are the ones the
current runs must supply; everything else is measured and in this document.

## Title (working)

*Twenty Ways a Vision-Language-Action Stack Can Silently Report Nothing: a
forensic study, and the parity discipline that ends it.*

## The claim

A VLA is not one model. It is a **producer/consumer pair** — an offline
preprocessing + training path that PRODUCES values, and a deployment path that
CONSUMES them — and essentially every failure we found lives in the seam, not in
either side. Each side was individually correct, individually tested, and
individually confident.

## Why anyone should care

The field reports closed-loop success rates. This paper reports what those
numbers are made of. We found **nineteen** defects; **every single one** left
open-loop metrics healthy — gripper accuracy 0.94, pose correlation 0.55 — while
closed-loop success sat at exactly 0.000. A number that low reads as "the model
is bad" and invites the standard responses: more data, more parameters, a better
loss. In this system it meant, in order:

| what 0.000 actually meant | § |
|---|---|
| the bench never loaded the checkpoint's weights | §0 |
| the corpus contained no object evidence at all | 4n |
| every episode crashed on tick 1 (device mismatch) | 4s |
| the policy was trained sighted and deployed blind | 4t |
| a stale box was fed where training taught zero | 4v |
| the relational head received all-zero evidence | 4x |
| the HRM never saw the end-effector | 4x-b |

Six of the nineteen were introduced BY THE FIXES for the previous ones,
including one that made behaviour strictly worse than before it was "fixed".

## The three contributions

1. **A defect taxonomy for producer/consumer ML systems**, with the failure mode
   named: an optional field with a benign default, a unit, a rate, an index
   convention, or a representation, disagreeing across the seam — silent because
   the consumer's default is well-formed. `Perception.proposals` defaults to
   `()`; dropping it produced no error, no shape change, and zero evidence.
2. **Parity testing as the discipline that catches them.** A test asserting
   either side's behaviour is evidence about that side alone. The test that finds
   these runs BOTH paths on ONE input and diffs, tensor by tensor. Ours is 180
   lines, runs in half a second, and would have caught the majority. We also show
   its limit honestly: it covers only what it exercises, and the very next defect
   landed in a code path it passed `None` for.
3. **[pending] The working system**: what the stack achieves once the seams are
   sealed — dense action supervision at the control rate rather than the
   perception rate, and a detection-independent spatial channel.

## The experiments that carry it

* **Instrumentation-first diagnosis.** Replay (5/5) proves the environment;
  stage isolation proves the actuation path; a magnitude sweep gives the task's
  tolerance band, [0.95, 1.05]. Only then is a policy number interpretable.
* **The A/B.** One episode through the trainer and through the deployment loop
  with perception held identical, diffing planner inputs. This is what localized
  the exposure bias to a single token: teacher-forced, `fused` diff 0.3384 ->
  **0.0000** and the gripper matched the trainer exactly.
* **Adversarial verification.** 27 candidate defects, verifiers instructed to
  refute by default; 10 of 10 high-severity survived. Reported with the refuted
  count and the unverified tail, because a sweep that only reports hits is the
  same instrument failure one level up.

## What we will NOT claim

* Not a new architecture. The parts are standard; the contribution is what
  connects them and how it fails.
* Not state of the art on LIBERO. **[pending]** — whatever we report, we report
  against baselines run in the same harness, since our own mock harness scored
  1.000 for an untrained policy (4x) and that is precisely the trap.
* No result computed on the mock env, and no number from a run whose parity we
  have not checked.

## Reproducibility

Every defect above is a commit with its measurement in the message. The paper is
the git history.

### `fixed`, completed configs — and a process correction

| eval config | mean_success |
|---|---|
| plan-only, brake ON | 0.000 |
| plan-only, no-brake | 0.000 |
| plan-only, no-brake, chunk | 0.000 |
| waypoint, no-brake | pending |

Three configurations, none of which had ever been evaluated before (every prior
closed-loop run passed `--waypoint-stats`, so the `WaypointActuator` overrode
x/y/z and the planner's own regressed translation was never executed once), and
all three are 0.000 on the 2 Hz corpus with a fully corrected objective.

**Process correction, recorded because it changes what a later table means.**
The dense-corpus run was reported in this document as "re-launched so stage A
learns the eef branch". It was not. The relaunch command died with its SSH
connection and the original process kept running; the checkpoint mtimes gave it
away (05:14/05:21, both predating the claimed 05:26 restart) and `ps` confirmed
the start time as 05:08:55. **`dense10` therefore never contained the HRM-eef or
TRM-context fixes**, and since the HRM is frozen in stage B its control branch
would have stayed dead for that entire arm. The run has been killed rather than
reported, because a dense-supervision result confounded with two missing fixes
answers no question anyone asked.

Worth naming the shape: a command was issued, produced no output, and was
recorded as done. That is the same class of error as everything else in this
document — an absent signal read as a successful one — committed this time by
the experimenter rather than the code. The fix is the same as it is everywhere
else here: verify the state, do not trust the issuing.

The `grid10` arm (dense corpus + spatial grid + every fix) is the one that
carries the question.

## 5a. Two changes that came from reading the numbers, not the code

### The world model was predicting almost nothing, and hiding it

`RecursiveTRM.forward_full` returned `next_emb = current_emb + head(pooled)` —
a residual, per the design contract — and then handed only the SUM downstream.
4w measured consecutive frame embeddings at cosine **0.9922**, so `next_emb` is
~99% a copy of its own input. Every consumer of it, including the planner's
largest memory group, had to recover the prediction by subtracting two nearly
equal vectors, and the 1-step margin over persistence (+1.7% MSE) says how little
survived that subtraction.

The residual is now returned as `delta` and reaches the planner as its own group
(`wm_delta`), **standardized at the projection**. That detail is the whole point:
the delta's magnitude is ~0.12 of the latent it rides on, so an unnormalized
delta arrives as near-zero tokens and the group is inert — the same
"wired but does nothing" failure that made the relational head's first result
meaningless. Its DIRECTION is the signal. Pinned by a test that scales a
realistically-sized delta and asserts the plan does not move.

This costs 16k parameters (7,005,837 total, cap 9M) and no retraining of anything
upstream: the quantity already existed and was being thrown away.

### Stage A was training on NaN and early-stopping on it

The dense run reported `train nan | val nan` from epoch 8 (the first at H=4)
through the end, then "early stop at H=4, best val 0.0182" — a checkpoint
selected on a number that had stopped meaning anything five epochs earlier.

Cause: `clip_grad_norm_` RETURNS the pre-clip norm and does not sanitize.
Clipping a NaN norm leaves NaN, so one bad batch writes NaN into the weights and
every forward afterwards is NaN. Stage A had clipping and it did not help; stage
B had no clipping at all. Both now skip the optimizer step when the gradient norm
is non-finite, and count the skips.

Worth noting how this presented: not as a crash, but as a training run that
completed, early-stopped, saved a checkpoint, and reported a plausible best
validation loss. The corpus itself is clean — 200 episodes scanned, zero
non-finite values, zero degenerate embeddings — so nothing upstream would have
flagged it either. Nineteen defects in, the pattern holds: **the failure mode of
this system is not an error, it is a number.**

### `fixed` — complete. Four actuation configs, all zero.

| eval config | mean_success |
|---|---|
| plan-only, brake ON | 0.000 |
| plan-only, no-brake | 0.000 |
| plan-only, no-brake, chunk-exec | 0.000 |
| waypoint, no-brake | 0.000 |
| waypoint, no-brake, chunk-exec | 0.000 |

Open loop (demo frames, policy's own action token): gripper agreement **0.643**,
closes on **28.1%** of steps against the demo's 58.5%.

So the gripper trend across the objective fixes is **7.6% -> 25.1% -> 28.1%**,
and closed-loop success across five actuation configurations is 0.000 throughout.
Two readings, and the evidence now separates them:

* The gripper is no longer the binding constraint. It fires on more than a
  quarter of steps, so episodes are no longer arithmetically incapable of
  succeeding.
* What binds now is **accuracy**. Pose correlation with the demonstrator is
  0.27-0.46, and 4p measured the task's tolerance on magnitude ALONE at
  [0.95, 1.05]. A policy that reaches in roughly the right direction a third of
  the time does not complete a pick-and-place, and no actuation setting can
  repair direction that poorly correlated — which is exactly what five
  configurations returning the same 0.000 demonstrates.

That relocates the problem for the third and last time in this document: not
plumbing (nineteen defects, all fixed), not the objective (corrected, and it
moved the open-loop numbers), but **the information the policy is given and the
amount of it**. Both are now being addressed directly — a spatial channel that
exists on every frame rather than the 53% that carry a detection, and action
supervision at the control rate rather than the perception rate (2 Hz -> 10 Hz ->
20 Hz, 7,680 -> 37,380 -> ~75,000 decision points).

## 5b. THE MEASUREMENT THAT REFRAMES EVERYTHING: the policy does not beat a linear probe

Nineteen defects, a corrected objective, five actuation configurations — all
0.000. Before spending more on the stack, we asked the prior question nobody had:
**given the representations the frozen backbone produces, how much of the
demonstrator's action is linearly recoverable?**

Ridge regression from baked features to the executed action, split **by
episode** (a frame-level split leaks almost the whole test set: consecutive
frames are cosine 0.99 apart), 200 train / 50 test episodes, features randomly
projected to 1024 dims:

| features | pose R² | pose corr | gripper corr |
|---|---|---|---|
| proprio only (10 numbers) | **0.170** | **0.415** | **0.787** |
| frame_emb (GAP, 512) | 0.075 | 0.425 | 0.800 |
| frame_emb + proprio | 0.079 | 0.450 | 0.843 |
| spatial_grid (16x512) | 0.076 | **0.465** | 0.828 |
| spatial_grid + proprio | 0.085 | 0.445 | 0.808 |
| **trained 7M policy** | — | **0.27–0.46** | **0.64** (agreement) |

**A linear map on ten proprioception numbers matches or beats the trained
seven-million-parameter policy on every axis measured.** Gripper: 0.787 linear
against 0.64 for the policy. Pose correlation: 0.415 linear against 0.27–0.46.

That is not a data problem and not a features problem. A model that cannot beat
ridge regression on its own inputs is **underfitting**, and it reframes every
result above: the whole stack has been operating at roughly linear-probe
capability the entire time, which is exactly consistent with 0.000 across every
actuation configuration, corpus size, and objective correction we tried.

Two things it also establishes, both useful:

* **The spatial grid carries real information.** 0.465 pose correlation from the
  grid alone against 0.425 from the GAP embedding it replaces — the architecture
  change of 4z is justified by measurement rather than by argument.
* **Pose R² is low everywhere (0.08–0.17).** Linear models recover little of the
  action's variance from these features. That does not bound a nonlinear model,
  but it does say the P5 (stride-32) feature level may be too coarse for fine
  manipulation, which is the next thing to test if capacity turns out not to be
  the answer.

### The immediate consequence

Stage B trains with 15% planner-input dropout, 30% modality dropout, 25% dream
steps, and **six** auxiliary losses stacked on the BC term (waypoint, actuation,
variance, smoothness, world-model rollout, gripper BCE). Each was added to fix a
measured problem and each was individually justified. Together they may simply be
preventing the model from fitting.

The `cleanbc` arm removes all of it — no dreams, no dropout, no auxiliaries, pure
behaviour cloning — so "over-regularized" becomes a testable claim rather than a
story. If pure BC clears the linear bar, the fix is the training recipe. If it
does not, the P5 features are the wall and the next move is a finer level of the
frozen backbone.

### 5b-i. The NaN was the accumulator, not the model

`cleanbc` reported `loss nan` from its first epoch. Before restarting anything,
the exact stage-B forward was reproduced offline on the dense corpus with the
trained stage-A weights, in `train()` mode: `P`, `G`, `Y` all finite,
`step_weight` finite (0.638-1.914), loss **1.96**. Every intermediate was
checked at t = 0, 5, 40 and 73 — fused, state_delta, TRM context, all four world
model heads, TQSA's three outputs, relational tokens, plan and grip logits — and
all were finite with sane magnitudes.

The model was never diverging. `run += float(loss)` folded one non-finite batch
into the running mean, and a running mean containing NaN stays NaN forever, so
every subsequent epoch line read `nan` while training proceeded normally. The
same epoch line proves it: `grip_acc 0.889 | val bc 0.2362 grip 0.952`, all
finite, all computed on separate accumulators.

**`val grip 0.952` is the best gripper number this project has produced**, and it
arrived in epoch 1 of the first arm that trains with the spatial channel live.

Non-finite losses are now counted rather than summed, and the epoch line reports
`NONFINITE Nb/Mskip`, so "a few bad batches" and "the run is gone" stop looking
identical. Note what this cost: the earlier dense-corpus run was killed on the
strength of its `nan` display, and that decision now looks premature — its VAL
was also nan, which is a genuinely different signal, but the train-loss reading
that first raised the alarm was an artifact.

Twenty defects. The instrument, again.

## 5c. THE FIX: the model was never allowed to fit

`cleanbc` — pure behaviour cloning, nothing else: no dream steps, no planner-input
dropout, no phase dropout, and every auxiliary loss set to zero (waypoint,
actuation, variance, smoothness, world-model rollout). Same corpus, same frozen
stage A, same architecture as the arms above.

| epoch | val bc | val grip | train grip_acc |
|---|---|---|---|
| 1 | 0.2362 | 0.952 | 0.889 |
| 3 | 0.1345 | 0.961 | 0.944 |
| 6 | **0.1028** | **0.969** | 0.951 |

Best val bc across every previously reported arm was **0.1512** (`v8_ss05`), and
`v8_act` — the arm whose actuation loss produced this document's most-cited
headline — sat at **0.2055**. Six epochs of plain BC beat both by 32%, and
`val grip 0.969` is the best gripper number the project has produced.

**Every one of the removed terms was individually justified by a measurement.**
The actuation loss fixed `wp_std_ratio` (0.121 -> 1.097). Scheduled sampling fixed
the exposure bias. Variance matching attacked MSE shrinkage. Modality dropout
implements the graded-evidence contract. Dream steps train the regime the loop
actually runs in. The world-model auxiliary protects frame prediction under BC
fine-tuning. Each was added to solve a real problem, each demonstrably improved
the metric it targeted, and **nobody ever measured what they did together.**

Together they prevented the model from fitting its primary objective. That is why
5b's linear probe beat the trained policy: the policy was not underpowered, it
was constrained by six simultaneous regularizers plus a 25% dream rate plus 45%
combined input/modality dropout — on a corpus of 7,680 decision points.

This is a different failure from the nineteen before it. Those were seams: two
sides disagreeing about a value. This one is a **composition** failure — every
component correct, every component justified, the sum pathological — and it is
invisible to any test of an individual component, including the parity discipline
this paper advocates. Ablations are usually run to show a component HELPS. The
missing experiment was the one that removes everything at once and asks whether
the base model can still fit.

### 5c-CORRECTION: five of the six were dead weight; one was load-bearing

5c reported that pure BC beat every regularized arm by 32% on `val bc` and
produced the best `val grip` (0.969) in the project. Both numbers are real. The
conclusion drawn from them was wrong, and the check that caught it took four
minutes: evaluating that same snapshot **self-fed**.

| `cleanbc` snapshot | teacher-forced (validation) | self-fed (deployable) |
|---|---|---|
| gripper | val grip **0.969** | closes on **0.0%** of steps |
| pose | val bc **0.1028** (best ever) | corr **-0.085 / 0.043 / -0.018** |
| agreement with demo | — | **0.415** (vs 0.643 for `fixed`) |

Pose correlation of approximately zero, and a gripper that never closes — from
the arm with the best validation numbers this project has ever recorded.

The cause is immediate once stated. Validation is TEACHER-FORCED: it feeds
fusion and the relational head the demonstrator's previous action. `cleanbc`
removed everything, and "everything" included `--action-token-sampling`. With the
shortcut fully available and no term penalizing reliance on it, the model took
it — reaching a record validation loss by predicting the demonstrator's next
action from the demonstrator's last one, which is unavailable at deployment by
construction.

So the composition result stands with one correction: **five of the six
auxiliaries were dead weight, and the sixth was doing essential work.** The
regularizers were not collectively harmful; they were collectively hiding which
one mattered. Removing them one at a time would never have shown this, because
each removal leaves the others to mask it — and removing them all at once is what
made the load-bearing one visible, by breaking loudly.

`synth10`/`synth05` are the synthesis: no dreams, no dropout, no waypoint,
actuation, variance, smoothness or world-model auxiliary — and scheduled sampling
at 1.0 and 0.5.

**The methodological point, which is the one worth publishing.** This project has
now been misled twice by the same structure: a metric that improves while the
quantity it stands in for gets worse. 4s (`wp_std_ratio` 0.121 -> 1.097, achieved
by moving a gain deployment never reads) and 5c (`val grip` 0.969, achieved by
reading an input deployment never has). In both cases the metric was correctly
computed, the improvement was real, and the inference was invalid.

The rule that would have caught both: **every training metric must be reported
alongside its deployment-condition twin.** A teacher-forced validation number is
not a policy number, and this document contains roughly two thousand lines
written before anyone insisted on the distinction.

### 5c-ii. `cleanbc` closed loop, and the corpus is not the NaN source

The `cleanbc` snapshot's closed-loop result completes the picture: **0.000**,
`tasks_completed 10`. A policy that closes its gripper on 0.0% of steps cannot
score otherwise, so this is the expected consequence of the shortcut, not an
independent failure.

The full 500-episode grid corpus was then scanned end to end: **zero non-finite
values across every float key, zero zero-variance frame embeddings (0/37,380),
zero zero-variance grid cells.** The data is clean, so the `NONFINITE 11b/11skip`
that `synth10` now reports on its epoch line comes from the model, not the
corpus — roughly 17% of batches producing a non-finite loss, skipped by the
gradient guard rather than poisoning the weights.

That counter is itself new (5b-i). Before it, this run would have printed
`loss nan` and looked identical to a diverged one; now it reads "11 batches
skipped, 52 trained", which is a survivable inefficiency rather than an
emergency. Root-causing the remaining 17% is deferred: it costs data, not
correctness, and there is a result closer to hand.
