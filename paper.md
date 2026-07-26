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

Test suite over this session: **149 → 198** passing (CPU-only, mock-only, no
network, no cv2).

## 6. Defects found and fixed (2026-07-25)

Ordered by what they would have cost if undetected.

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

## 7. Open

* **Closed-loop `mean_success` remains unobtained** — the number the paper's
  Claim 1 needs. Harness is now self-localizing (handoff §0).
* Waypoint-absolute arm untrained; `wp_std_ratio` vs `std_ratio` is the test.
* TQSA arm mid-flight; `spatial` sensitivity will be its first honest reading.
* LIBERO-only stage B untested — bridge is 75% of stage-B steps and supplies
  neither proprio nor frames, so it may be diluting the policy rather than
  regularizing it.
