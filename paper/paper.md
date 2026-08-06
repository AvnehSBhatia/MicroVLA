# MicroVLA — Paper (revised 2026-07-31)

Lab notebook + argument. Every front-matter claim cites a measured number in
this file or in `results/`. Original Claims 1–8 live under **Deferred /
kill-barred** — they are not the contribution list.

## Title (working)

*Agreement Is Not Correctness: twenty-eight ways a vision-language-action stack
can silently report nothing, and the provenance discipline that ends them.*

## The claim

A VLA built on frozen encoders and offline-baked features is a
**producer/consumer pair**: the bake PRODUCES a feature corpus, deployment
CONSUMES it. Every defect we found lives in that seam, not in either side. Each
side was individually correct, individually tested, and individually confident
while closed-loop success sat at **0.000** and open-loop metrics stayed healthy
(gripper agreement ~0.93, val BC 0.088).

The seam has two failure modes, and the second is the one that cost the most:

1. **The two sides disagree.** A benign default, a unit, a rate, an index
   convention, a representation. Twenty-four of these. Example:
   `Perception.proposals` defaults to `()`, so dropping it in a device
   transfer produces no error, no shape change, and zero evidence — and the
   same omission recurred later for `spatial_grid` (§5n, defect 27).
2. **The two sides agree on something wrong.** No parity test can catch this,
   because parity holds. Every corpus was baked from the wrist camera and every
   eval read the wrist camera — perfectly consistent, and the one view in the
   dataset where the frozen detector is blind: source role grounded on 22% of
   frames at confidence 0.011, target role on **1.4% of 38 000 frames**
   (§5n/5o, defect 25).

## Contributions

1. **A defect taxonomy for producer/consumer ML seams**, with 28 worked
   instances, each with the measurement that found it and the measurement that
   would have caught it earlier. The taxonomy's own boundary is a finding: 24 are
   disagreements, and the last 4 are agreements on a wrong convention, which
   need a different instrument.
2. **Parity testing** — run the trainer path and the deployment path on one
   input and diff tensors. Catches class 1. Its limits are stated and were hit:
   `eval/train_vs_deploy.py` forces a real tick every step, which is exactly why
   defect 28 (the deployment-only innovation corrector, perturbing half of all
   ticks at `perception_period 2`) was invisible to it for weeks.
3. **Provenance as the instrument for class 2.** A feature corpus is an
   interface; an interface with no schema will drift. `manifest.json` now records
   the camera, orientation, detector threshold, role-disjointness, sampling
   rate, and the frame size the encoder actually saw; `eval/libero_eval.py`
   checks the deployment against it and writes any mismatch into the results
   JSON. Four of the 28 defects are precisely a field this block would have
   flagged.
4. **Input-quality auditing as a first-class step.** Detection duty,
   confidence, and box stability *per candidate view* are four cheap numbers
   that were never measured until the 25th defect hunt — after ~10 000 GPU
   seconds spent on levers downstream of them. Measuring them moved source
   grounding 0.219 → 0.850 and target grounding 0.014 → 0.999 without touching a
   single model parameter.

## What we do NOT claim

* **Not SOTA on LIBERO.** Claim 1's pre-registered kill bar (`<30%` where large
  models exceed ~80%) is met at `mean_success = 0.000`. Dropped. The §5p IBVS
  campaign also scores **0.000** success — and those runs are *assisted*
  control, so they do not soften the kill bar even as approach metrics move.
* **Not perception-rate decoupling** (Claim 2 / E4). It needs
  `mean_success > 0` first. The agentview arms of §5o remain **disk-parked**
  (§5p); Claim 2 stays unmeasured, not supported.
* **Not unaided closed-loop competence from IBVS / tool-phase numbers.**
  Phased IBVS and `--tool-phase` are diagnostic / assisted controllers on top of
  `rec_fix`. They may not be cited as MicroVLA policy success.
* **Not a Pi 5 edge demo** (Claim 3 / E10). No watts/Hz measured.
* **Not trust-as-safety** (Claim 5). τ is an instrument; on LIBERO delta actions
  the brake hurt.
* **Not recursion Pareto** (Claim 7) or bottleneck scaling (Claim 8). Unrun.
* **No number from `--mock-env`** (it scored 1.000 for an untrained policy).

## Retracted during this study

Kept visible because the retractions are part of the result — each was a
confident reading of a number produced under a condition we had not checked.

| claim | why it fell |
|---|---|
| §4e — the waypoint actuator is a tracking controller | its feedback is algebraically dead; a frozen arm and one moving 5 cm/step emit bit-identical commands (§5n) |
| §5b — the policy does not beat a linear probe | measured before the model was allowed to fit (§5c) |
| §5j — frozen features are the ceiling, resolution is not | inferred from a resolution probe that does not license it, then from experiments run through a blind camera (§5k, §5n) |
| §5m — open-vocab detection cannot bind the phrase to the object | established only for the wrist view; agentview grounds both roles at 0.97/0.999 with 6× less jitter (§5n) |
| §5m — the tracking and CLIP re-rank nulls | uninformative rather than negative: both re-ranked boxes that were not on objects |
| §5l — "approach then diverge", terminal-precision problem | video showed a drive-by en route to the place prior, not a stalled approach (§5m postscript 2) |

## Verification ledger (tonight)

| Claim / statement | Status | Number | Source |
|---|---|---|---|
| Seam defects leave open-loop healthy, closed-loop 0.000 | VERIFIED | 0/50 trials; grip~0.94 | §4m, skeleton |
| Stage A WM beats persistence (wrist, blind era) | VERIFIED | `wm_margin` **+19.8%** | §4g, `full_stageA_wrist_v72.pt` |
| Stage A WM beats persistence (sighted v8) | VERIFIED | `wm_margin` **+43.3%** | §4q, `v8_pod/full_stageA_v8_s0.pt` |
| Evidence fade moves fusion (design + probe) | PARTIAL | `box_weight→0` Δfused **43.0%** | §4h; E6 ablation pending |
| Defect 24 recovery proprio to planner | VERIFIED | test pin | `tests/test_v8_train_path.py` (`-k recovery_proprio`); commit `ff27d75` |
| Eval sightedness protocol | SUPERSEDED | `render_size=256` vs a 128 px corpus is itself a mismatch | §5n provenance |
| Claim 1 competence @30M | DROP (kill bar) | closed-loop **0.000** | §4m–§5j |
| Claim 2 E4 rate sweep | DEFER | open-loop only | needs success>0 |
| Claim 3 Pi demo | DROP | — | E10 deferred |
| Claim 5 trust safety figure | DROP/soften | τ logged; no AUROC | §4m, §4p |
| Claim 6 structure / frozen detector | REOPENED | the "ceiling" was the wrist view; agentview grounds 0.970/0.999 | §5n, §5o |
| Claims 7–8 | DROP | — | E8/E9 unrun |
| Contribution 3 approach metrics | WITHDRAWN as unaided competence | §5l eef_min 0.132 m was a drive-by; §5p IBVS eef~0.08 m is assisted near-miss, still succ 0 | §5m, §5p |
| Full auton IBVS atlas (85 runs) | VERIFIED | every `mean_success=0`; TSV in `results/IBVS_AUTON_SCORECARD.tsv` | §5p |
| Contribution 4 input-quality audit | VERIFIED | src duty 0.219→0.850 with no parameter changed | §5n |
| Phrase→object binding (open-vocab), WRIST view | VERIFIED negative | early phased IBVS grip_close **0.000**; track gate null | §5m |
| Phrase→object binding, AGENTVIEW | **RETRACTS the row above** | src duty **0.970**, tgt **0.999**, jitter 0.030 | §5n, §5o |
| Corpus target role was never observed (wrist) | VERIFIED | **1.4%** of ~38 000 frames | §5o |
| Agentview source box tracks the moved object | VERIFIED | `const_frac` **0.032** | §5o |
| Source box lands on the basket (agentview) | VERIFIED | **40.9%** of both-detected frames; 0.000 at `role_disjoint_iou` 0.1 | §5o |
| Waypoint actuator is open-loop | VERIFIED | frozen vs moving arm: bit-identical commands | §5n |
| Corrector perturbs half of all deployed ticks | VERIFIED | rel-diff fused **0.15–0.22** | §5n (defect 28) |
| Trainable heads under budget | VERIFIED | **7.006M** / 9M | `param_audit` 2026-07-30 |
| Full CPU mock suite green | VERIFIED | **504 passed**, 1 skipped | `pytest tests -q` |
| IBVS zero-train residual moves eef_min (cream) | VERIFIED assisted | best-config mean eef **0.087 m** (9 seeds), succ **0** | §5p |
| Hyst 0.50 > 0.60 under seed-fair compare | VERIFIED | 3/3 seed pairs; n10: **0.079** vs **0.084** | §5p |
| Frame-dynamic centering loss (`center_frame`) | MIXED / neg on eef | plain eef **0.108** vs rec_fix **0.081**; grip 0.41 vs 0.18 | §5q |
| Frame-dynamic centering loss (`center_frame`) | MIXED / neg on eef | plain eef **0.108** vs rec_fix **0.081**; grip 0.41 vs 0.18 | §5q |
| Aim UV is task-dependent | VERIFIED | cream V=0.60; soup~0.16 m; dressing~0.24 m | §5p |
| Tool-phase never grasps (wrist) | VERIFIED negative | detect~1.0, `grip_close` **0.000** on all tool arms | §5p |
| Agentview train arms | PARKED (disk) | ~4.8 GB free; scripts in `/root/queue/parked/` | §5o, §5p |

### Retracted headlines (do not restore)

1. “Grounding failure is entirely in the planner” (§4h) — corpus had 0% source dets (§4n).
2. Overnight arm rankings (§4m) — measured epoch count, not architecture.
3. “Frozen features are the ceiling” (§5j geometric form) — suspended twice:
   first as unlicensed by the resolution probe (§5k), then because every
   experiment behind it ran through a camera the detector cannot see (§5n).
8. “Semantic binding is the measured ceiling” (§5m) — measured on the wrist view
   only. Agentview grounds both roles at 0.970 / 0.999 with 6x less box jitter,
   and the residual source/target collision is fixable with no training
   (§5n, §5o). The one surviving piece is narrow and still true: YOLO-World-S
   scores the product phrases themselves ("alphabet soup") at 0.000, so binding
   comes from generic tails plus disjointness, not from the phrase.
9. “The wrist view is 4x the dynamics signal and better grounding than
   agentview” (`preprocess/shard_pipeline.py` help text) — measured backwards.
   Agentview is 3.9x source duty, 4.7x target duty, 6.3x proposals per frame.
4. Phase:vision ratio at n=1 — seed fold 0.3–5.5 (§4m).
5. Mock-env success as competence.
6. AIAYN / “field-defining” framing as current status — aspirational debt (§5k).
7. §5l “terminal approach precision” — reframed as drive-by / missing grasp phase (video).

## Deferred / kill-barred (original Claims 1–8)

Kept for history and pre-registration. Not tonight's contribution list.

### Claim 2 — Perception-rate decoupling (deferred)
E4: detector at 30→5→2→1→0.5 Hz × {ours, hold-last, oracle}. **Blocked** at
`mean_success = 0.000`. Open-loop proxy kept: WM margins above.

### Claim 1 — Competence at 1/200th the scale (kill bar met)
LIBERO vs OpenVLA/SmolVLA/TinyVLA. **Dropped** per pre-registered kill bar.

### Claim 3 — Edge demonstration (deferred)
Pi 5 + AI HAT watts/Hz. Unmeasured.

### Claim 4 — Evidence fade recipe (partial)
Design + Δfused probe verified; fade/zero/none ablation (E6) not run.

### Claim 5 — Self-calibrating trust (softened)
τ instrument only; no failure-AUROC; do not claim safety figure.

### Claim 6 — Structure buys scale (partial)
Sighted grounding + relational sensitivity; not a landmark ablation (E7 void).

### Claim 7 — Recursion compute knob (dropped)
### Claim 8 — Bottleneck scaling (dropped)

## Experiment matrix (status sync)

| ID | Experiment | Claim | Status |
|----|-----------|-------|--------|
| E1 | Stage A/B training | prereq | **done** (WM +19.8% / +43.3%; Stage B `rec_fix` val bc **0.0881**) |
| E2 | Open-loop rollout vs persistence | 2,4 | measured via `wm_margin` |
| E3 | LIBERO closed-loop harness | 1 | harness **exists**; success kill-bar **0.000** |
| E4 | Perception-rate sweep | 2 | **not runnable** until success>0 |
| E5 | τ→failure AUROC | 5 | deferred |
| E6 | Evidence-fade ablation | 4 | pending |
| E7 | Grounding ablation | 6 | pending (fix drop) |
| E8 | Recursion × latency | 7 | dropped tonight |
| E9 | Bottleneck sweep | 8 | dropped tonight |
| E10 | Pi 5 end-to-end | 3 | deferred |
| E11 | Rig transfer | generality | deferred |

## Tier calibration (honest landing)

| Outcome | Tier |
|---|---|
| Forensic seam paper + verified WM margins + parity discipline | **Systems / diagnostic paper (tonight)** |
| E4 graceful-vs-collapse + E3 within 25 pts of 7B + E10 | Landmark (unreachable tonight) |
| TRM never beats persistence | Stop — **not this case** (+19.8% / +43.3%) |

## Timeline hooks (auto-updated as stages complete)

- [x] Datasets streamed + converted under 10 GB cap
- [x] E1 Stage A world model beats persistence (`wm_margin` +19.8% / +43.3%)
- [x] E1 Stage B policy — val BC reported (many arms; `rec_fix` night of 07-29)
- [x] E3 LIBERO harness (real backend on pod; mock path for CI)
- [ ] E3 non-zero closed-loop success (kill bar currently met at 0.000)
- [ ] E4 the sweep (blocked on success>0)
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

# PAPER SKELETON (revised 2026-07-30)

Everything above is a lab notebook. This is the argument it supports, in the
order a reviewer should meet it. Numbers marked **[pending]** are the ones the
current runs must supply; everything else is measured and in this document.

## Title (working)

*Agreement Is Not Correctness: twenty-eight ways a vision-language-action stack
can silently report nothing.*

## The claim

A VLA built on frozen encoders and offline-baked features is not one model. It
is a **producer/consumer pair** — a preprocessing + training path that PRODUCES
a feature corpus, and a deployment path that CONSUMES it — and every failure we
found lives in that seam. Each side was individually correct, individually
tested, and individually confident.

The seam fails in two ways, and they need different instruments:

**Class 1 — the two sides disagree.** An optional field with a benign default, a
unit, a rate, an index convention, a representation. Silent because the
consumer's default is well-formed. Twenty-four instances. Parity testing catches
these.

**Class 2 — the two sides agree on something wrong.** No parity test can catch
this, because parity holds. Four instances, and they were the most expensive:
every corpus was baked from the wrist camera and every eval read the wrist
camera — perfectly consistent, and the one view where the frozen detector is
blind. Provenance and input-quality auditing catch these.

## Why anyone should care

The field reports closed-loop success rates. This paper reports what those
numbers are made of. We found **twenty-eight** defects; **every one** left
open-loop metrics healthy — gripper agreement 0.93, validation BC 0.088, world
model +43% over persistence — while closed-loop success sat at exactly 0.000. A
number that low reads as "the model is bad" and invites the standard responses:
more data, more parameters, a better loss. In this system it meant, in order:

| what 0.000 actually meant | § |
|---|---|
| the bench never loaded the checkpoint's weights | §0 |
| the corpus contained no object evidence at all | 4n |
| every episode crashed on tick 1 (device mismatch) | 4s |
| the policy was trained sighted and deployed blind | 4t |
| a stale box was fed where training taught zero | 4v |
| the relational head received all-zero evidence | 4x |
| the HRM never saw the end-effector | 4x-b |
| the recurrent state diverged to NaN past its training horizon | 5e |
| the recovery label described a displacement the input did not carry | 5k |
| **the detector was reading a camera it cannot see through** | **5n** |
| the deployed spatial adapter never saw the grid it trained on | 5n |
| half of every episode's ticks carried a perturbation training never modelled | 5n |

Seven of the twenty-eight were introduced BY THE FIXES for previous ones,
including one that made behaviour strictly worse than before it was "fixed".

## The four contributions

1. **A defect taxonomy for producer/consumer ML systems**, with 28 worked
   instances, each paired with the measurement that found it and the measurement
   that would have found it sooner. The taxonomy's own boundary is a result: the
   split at 24 is where parity testing stops working.
2. **Parity testing as the discipline for class 1.** A test asserting either
   side's behaviour is evidence about that side alone. The test that finds these
   runs BOTH paths on ONE input and diffs, tensor by tensor. Ours is 180 lines,
   runs in half a second, and would have caught most of them. We also report its
   limit honestly, because we hit it: `eval/train_vs_deploy.py` forces a real
   perception tick every step, which is exactly why the deployment-only
   innovation corrector — perturbing HALF of all ticks at the rate we evaluate
   at — was invisible to it for weeks (defect 28).
3. **Provenance as the discipline for class 2.** A feature corpus is an
   interface, and an interface with no schema will drift. The corpus now records
   the camera, orientation, detector threshold, role-disjointness policy,
   sampling rate and the frame size the encoder actually saw; the eval harness
   checks the deployment against it and writes any mismatch into the results
   file. Four of the 28 are precisely a field this block would have flagged.
4. **Input-quality auditing as a first-class step.** Detection duty, confidence
   and box stability *per candidate view* are four cheap numbers nobody computed
   until the 25th defect hunt, after ~10 000 GPU-seconds spent on levers
   downstream of them. Computing them moved source grounding 0.219 → 0.850 and
   target grounding 0.014 → 0.999 with no parameter changed — and retracted
   three of this paper's own earlier conclusions.

## The result the paper is honest about

Closed-loop success was 0.000 for the entire study. The agentview arms (§5o) are
the first runs whose corpus ever contained the target object. Their numbers are
**[pending]**, and the paper states plainly which of its claims depend on them:
Claim 2 (perception-rate decoupling) is unmeasurable at zero success, and stays
unmeasured rather than argued around.

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
* Not state of the art on LIBERO. Closed-loop `mean_success = 0.000`; Claim 1's
  kill bar is met. The approach intermediates we once offered in its place
  (detect 0.719, eef_min 0.132 m) are WITHDRAWN as evidence of competence: video
  showed the 13 cm was a drive-by en route to the basket, not a stalled
  approach (§5m).
* Not that we have found the last defect. Three of the 28 were found in the last
  hours of the study, by an instrument (input-quality auditing) that did not
  exist the day before.
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

## 5d. The deployment-condition twin was itself measured at the wrong rate

5c concluded that `cleanbc` bought its record validation with the action-token
shortcut, on the evidence that it closed the gripper on **0.0%** of steps
self-fed with pose correlation approximately zero. That evidence was produced by
`eval/openloop_check.py`, which builds the policy with the DEFAULT
`perception_period=15` — the Pi loop's 30 Hz / 2 Hz. The grid corpus has a stride
of **2**. Every self-fed number was measured with real perception firing 7.5x
less often than anything the policy trained under.

Re-measured at the matching rate, the same `synth10` checkpoint at epoch ~3:

| metric | period 15 (wrong) | period 2 (correct) |
|---|---|---|
| gripper agreement | 0.436 | **0.902** |
| we close on | **0.0%** of steps | **60.4%** (demo: 56.4%) |
| pose corr, demo_0 / demo_1 | -0.085 / 0.043 | **0.703 / 0.672** |

**The policy is working.** Pose correlation ~0.69 against the linear probe's
0.415 (5b) — it is now decisively beating the bar that, four hours ago, it could
not reach. Gripper agreement 0.902 against 0.643 for the best previous arm, and a
closing RATE of 60.4% against the demonstrator's 56.4%.

### What this costs in retractions

* **5c's correction is suspended.** Its claim — that removing
  `--action-token-sampling` re-opened the exposure bias — rested on a 0.0%
  measured at the wrong rate. `cleanbc` is being re-measured at period 2 before
  anything is concluded about it. The underlying question (is scheduled sampling
  load-bearing?) is now open again rather than answered.
* **Earlier self-fed numbers are rate-mismatched too.** 4u's 7.6%, 4y's 25.1%,
  and `fixed`'s 28.1% were all taken at period 15 against a 2 Hz corpus whose
  stride is 10 — a 1.5x mismatch rather than 7.5x, so less damaging, but not
  clean. They should be read as lower bounds.

### Defect 21, and where it landed

This is the twenty-first defect, it is the same shape as the twenty before it — a
rate disagreeing across a boundary — and it was in **the instrument built to
catch that exact class of error**. The tool whose entire purpose is "a
teacher-forced number is not a policy number" was itself reporting a policy
number taken under conditions the policy never saw.

There is no irony to extract from this, only a procedure: the parity discipline
has to apply to the measurement code with the same force it applies to the model
code, because a diagnostic is just another consumer of a value some producer
defined. `openloop_check` now takes `--perception-period` and the corpus stride
must be passed explicitly.

## 5e. Defect 22 — the recurrent state diverged past the training horizon

The closed-loop telemetry for `synth10` showed 50-75% of ticks with **non-finite
emitted actions**, while the same checkpoint scored gripper agreement 0.902 and
pose correlation 0.69 on demonstration frames. Two candidates: the policy
diverges on its own states, or mujoco blows up under an extreme command and
returns NaN proprio which the policy echoes. Both sides were checked at every
step, on four tasks:

```
task 0 alphabet_soup  -> POLICY action NaN at step 200 (proprio finite)
task 1 cream_cheese   -> POLICY action NaN at step 198 (proprio finite)
task 2 salad_dressing -> POLICY action NaN at step 192 (proprio finite)
task 3 bbq_sauce      -> POLICY action NaN at step 204 (proprio finite)
```

The environment is innocent, and the failure is not state-dependent — it is
**step-count dependent**, at ~200 env steps on every task. At perception period 2
that is ~100 recurrent HRM steps.

`_DampedCore` returns `(1 - alpha) * state + alpha * x`, where the candidate `x`
grows through additive residual blocks and **nothing bounded the carried state**.
Training episodes run T = 74-111 steps; deployment runs 200. The state was being
asked to remain stable an order of magnitude beyond anything it had ever been
optimized over, and it did not.

Every open number resolves to this one:

| observation | explanation |
|---|---|
| `NONFINITE 11b/63` every epoch, deterministic | 2 of 120 episodes are long (T=97, T=111) and NaN; one bad episode NaNs its whole batch of 8, so 2% of episodes cost 17% of batches |
| 50-75% of closed-loop ticks non-finite | everything after ~step 200 of a 300-400 step episode |
| task 0 clean for 200 steps in the first probe | the probe stopped at 200 — one step short |

Fixed with a bound at 50.0, roughly 10x the largest magnitude observed in healthy
operation (|state| absmax 4.7 over 74 steps; 4.1 over 400 steps after the fix).
It cannot alter a working trajectory and only stops a diverging one from becoming
NaN. **No retrain: the bound applies at inference.** Verified — tasks that died at
step ~200 now run clean for 250.

### What this does to the preceding results

Every closed-loop number in this document was measured with episodes of 300-400
steps, so **every one of them was scored on a policy that emitted NaN for the
back half of each episode**. The zeros were real in the sense that nothing
succeeded, but they were not measurements of the policy's behaviour — they were
measurements of a diverged recurrent state.

That is the fifth instrumentation null in this document, and the fourth root
cause that had to be found before any policy number meant anything. The pattern
that would have caught it earlier is the cheapest one available and was not
applied for twenty-one defects: **check that the emitted action is finite before
scoring the episode.** A NaN action does not raise; the environment accepts it,
the episode completes, and the harness reports 0.000.

## 5f. Where it stands, and the last hypothesis

`synth10` completed (40 epochs, `val bc 0.1504`, `val grip 0.964`,
`grip_acc 0.955`) and was evaluated with the divergence fixed and the perception
rate matched to the corpus:

| measurement | value | reference |
|---|---|---|
| pose corr, demo frames, self-fed | **0.589 - 0.693** | linear probe 0.415 (5b) |
| gripper agreement, self-fed | **0.796** | 0.643 best prior arm |
| gripper closing rate | 38.1% | demo 58.5% |
| **closed-loop success** | **0.000** | — |

The HRM state bound also removed the training NaN: `rec_mid` reports a finite
`loss 2.0019` where every previous arm on this corpus printed
`NONFINITE 11b/11skip`. Both symptoms had one cause.

So the policy is now, by every open-loop measure, a working behaviour-cloning
model — comfortably above the linear bar it could not reach this morning — and it
still completes no episodes.

That combination is diagnostic rather than discouraging. A policy that is
accurate on demonstrated states and fails on its own is the textbook signature of
**compounding error**: the demonstrations show only the expert's trajectory, so
nothing in the objective teaches recovery, and in closed loop the arm is always
slightly off it. Every earlier explanation is now excluded by measurement —
grounding (75.8% detection), representation (grid beats GAP), objective
(corrected), capacity (beats the linear probe), divergence (bounded), rate
(matched), actuation (five configurations).

`rec_mid` and `rec_high` test it directly by perturbing the observation during
stage B, which places near-trajectory states in the training distribution with
the expert's action still as the target. If recovery is what is missing, noise is
the cheapest thing that supplies it; if these also return 0.000, the remaining
candidates are data volume (500 episodes, one suite) and the P5 feature level.

**Honest status against the goal.** Twenty-two defects found and fixed, every one
verified by measurement. Open-loop behaviour went from below a linear probe to
well above it. Closed-loop success is 0.000 and has never been otherwise. The
gap between those two facts is the paper's real subject, and it is not yet closed.

## 5g. The third load-bearing term: magnitude

`rec_mid` (clean recipe + scheduled sampling + state-recovery noise) is the best
open-loop policy this project has produced:

| measure | rec_mid | reference |
|---|---|---|
| gripper agreement, self-fed | **0.931** | 0.796 (synth10), 0.643 (fixed) |
| gripper closing rate | **63.9%** | demo 58.5% |
| pose corr | 0.59-0.69 | linear probe 0.415 |
| **closed-loop success** | **0.000** | — |

Correlation and agreement are both scale-blind, so they were hiding the thing
that matters. Measuring the emitted MAGNITUDE against the demonstrator on the
same frames:

| dim | x | y | z | roll | pitch | yaw |
|---|---|---|---|---|---|---|
| emitted std | 0.184 | 0.418 | 0.322 | 0.0061 | 0.0183 | 0.0090 |
| demo std | 0.242 | 0.473 | 0.512 | 0.0184 | 0.0519 | 0.0523 |
| **std_ratio** | 0.758 | 0.883 | **0.628** | **0.332** | **0.352** | **0.171** |

**Median std_ratio 0.490.** The policy emits half the demonstrator's motion, and
4p measured the passing band at **[0.95, 1.05]** — ground truth at 1.00 solves
5/5, at 0.80 solves 0/4. A policy at 0.49 cannot complete a reach-and-grasp no
matter how well it predicts direction, which is exactly what "gripper agreement
0.931, success 0.000" looks like from outside.

This is MSE regression to the conditional mean — the failure 4p predicted and
`--variance-weight` exists to penalize. **It was removed in the clean recipe as
dead weight.**

### The ablation, complete

Six auxiliary terms were removed together in 5c because their composition
prevented the model from fitting. Restoring them one at a time, guided by which
deployment metric collapsed:

| term | verdict | evidence |
|---|---|---|
| scheduled sampling | **load-bearing** | without it the gripper closes 0.0% self-fed (5c) |
| variance matching | **load-bearing** | without it std_ratio 0.490 against a [0.95, 1.05] band |
| waypoint loss | dead weight | — |
| actuation loss | dead weight | and mis-specified three ways (4x) |
| smoothness | dead weight | — |
| world-model auxiliary | dead weight | — |

Two of six were essential and four were not, and neither essential one is
identifiable from validation loss: both are invisible to a teacher-forced,
scale-blind metric. Removing all six at once is what made them findable, because
each broke a *different* deployment measurement — the gripper rate and the
magnitude ratio — while `val bc` improved throughout.

`var_on` / `var_hi` retrain with variance matching restored. In parallel,
`--action-gain` rescales the emitted pose at eval time, which tests the magnitude
hypothesis in minutes rather than an hour: if 0.49 is the whole story, a gain
near 2.0 should move success off zero without touching a weight.

## 5h. Compounding error, measured — and the label that fixes it

`rec_mid` closed the loop on every open-loop objection: gripper agreement
**0.931**, closing rate **63.9%** against the demonstrator's 58.5%, pose
correlation 0.59-0.69 against a linear probe's 0.415. Closed-loop success:
**0.000**.

Two hypotheses remained. Both were tested rather than argued.

**Magnitude — refuted.** `std_ratio` is 0.490, and 4p put the passing band at
[0.95, 1.05], so this looked decisive. `--action-gain` rescales the emitted pose
at eval time, which tests it in minutes:

| action gain | 1.0 | 1.4 | 1.8 | 2.2 |
|---|---|---|---|---|
| mean_success | 0.000 | 0.000 | 0.000 | 0.000 |

Magnitude is necessary and not sufficient — the same conclusion 4p-CORRECTION
reached, now confirmed on a policy that is otherwise healthy.

**Compounding error — confirmed, and quantified.** Running the policy from a
demonstration's own initial state and comparing its end-effector path to that
demonstration's:

| step | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|
| mean EEF separation | 2.25 cm | 3.40 cm | **5.34 cm** | 9.74 cm | **28.65 cm** |

An object is a few centimetres across. By step 20 of a 150-step task the wrist
camera is pointed somewhere no demonstration ever shows, and by step 80 the arm
is a third of a metre away. The policy is accurate on the expert's states and
spends almost all of every episode off them.

### Why the obvious augmentation does not work

`--proprio-noise` perturbs the observation and leaves the expert's action as the
target. The gradient therefore teaches **"this perturbation does not matter"** —
invariance, when what closed loop needs is **correction**. That is not a subtle
distinction: it is the difference between a policy that ignores being 5 cm off
and one that steers back. `rec_mid` has it, has the best open-loop numbers in the
project, and still scores zero.

### The label DAgger would provide, computed offline

Displacing the measured end-effector by `delta` means the same waypoint now
requires `delta` LESS motion. In raw action units the correction is
`-delta / gain`; in the normalized units the loss uses, `-delta / (gain * q_high)`.
The gain is the FITTED one the deployed actuator divides by, so the label is
executable rather than notional, and the whole construction is possible only
because the corpus stores absolute EEF positions alongside the actions.

This is DAgger's *data* without DAgger's *expert*: for a positional perturbation
the correct recovery action is derivable in closed form, so no on-policy expert
queries are needed. Translation dims only — a positional displacement does not
change the required orientation or gripper state. Pinned by a test on the SIGN,
because getting it backwards would train the policy to run away from the
trajectory while every loss curve looked fine.

`dag_lo` (15 mm) and `dag_hi` (35 mm) report open-loop, the DIVERGENCE CURVE, and
closed-loop success. The divergence curve is the one that matters: it measures
the mechanism directly, so if recovery training works the separation at step 20
should fall well below 5.3 cm whether or not success moves off zero yet.

## 5i. Defect 23 — a silent clamp turned recovery training into saturation

The recovery arms did not merely fail to help; they made the mechanism they
targeted **monotonically worse**:

| arm | perturbation | separation @ step 20 | @ step 40 | success |
|---|---|---|---|---|
| `rec_mid` | none (observation noise only) | **5.34 cm** | 9.74 cm | 0.000 |
| `dag_lo` | 15 mm | 8.65 cm | 14.95 cm | 0.000 |
| `dag_hi` | 35 mm | 12.82 cm | 23.95 cm | 0.000 |

Monotonic in the perturbation is the signature of a scale error, not a sign
error. The arithmetic:

| displacement | correction in normalized units | executable? |
|---|---|---|
| 3 mm | 0.29 | yes |
| 10 mm | 0.98 | marginal |
| **15 mm** | **1.47** | **no** |
| **35 mm** | **3.43** | **no** |

One full-magnitude action step moves only `gain` metres — 10.9 / 13.1 / 11.8 mm
per axis. A 15 mm displacement therefore needs 1.4 steps of *saturated* motion to
undo, and its correction lies outside the [-1, 1] the target lives in. The
`.clamp(-1, 1)` protecting the target then discarded the excess, so both arms
trained on **saturated targets** — a policy instructed to emit maximum motion
constantly. The negative result says nothing about recovery training; it measures
a clamp.

### The guard, and why the first version of it was also wrong

The first fix refused any perturbation whose correction exceeded a 0.5-unit
budget. It rejected 4 mm — the value intended as safe — because it checked the
MAX over sampled deltas, and an unbounded Gaussian exceeds any budget in its
tail: sigma = 4 mm reaches 12 mm at 3 sigma, correction 0.96. A refusal rule on a
Gaussian tail rejects every usable sigma.

The working version computes the largest undoable displacement per axis first
(`0.5 * gain * q_high`), caps sigma at `dmax / 2.5`, and truncates samples to
`dmax`. Verified across requested sigmas of 1 / 4 / 15 / 35 mm: max correction
0.390 / 0.500 / 0.500 / 0.500 against the 0.5 budget. Oversized requests are
capped rather than refused, no target can saturate, and the invariant is
asserted rather than trusted.

### What this is an instance of

Twenty-three defects, and this one belongs to the same family as the very first:
**a protective operation that discards a signal instead of reporting it.** The
clamp was added to keep targets in range, which is correct; what it lacked was
any statement of what it had thrown away. Every loss curve stayed smooth, `val bc`
improved, and the augmentation was training the opposite of its intent.

The rule this suggests, and the one the paper should carry: *a clamp, a mask, or a
fallback that can silently remove signal must count what it removed.* The
gradient guard of 5b-i already does this (`NONFINITE 11b/11skip`), and it is why
that failure took minutes to characterize while this one took two training runs.

## 5j. The frozen features are the ceiling — resolution is not

`var_only` isolated the magnitude fix and produced the most useful negative in
this document:

| arm | std_ratio | separation @ step 20 | @ step 40 | success |
|---|---|---|---|---|
| `rec_mid` (no variance term) | 0.49 | **5.34 cm** | 9.74 cm | 0.000 |
| `var_only` (variance matching) | ~1.0 | **12.79 cm** | 21.76 cm | 0.000 |

It trained cleanly — val bc **0.1537**, val grip **0.959**, gripper agreement
0.917, closing 64.7% against the demonstrator's 58.5% — and diverges **2.4x
faster**.

**This invalidates how divergence was being read.** `rec_mid` had the lowest
separation because it UNDER-MOVES: at std_ratio 0.49 it physically cannot get far
from the demonstrated path. Restoring correct magnitude amplifies an imperfect
DIRECTION, so the arm leaves faster. Low divergence was a symptom of not moving,
not of accuracy, and the two quantities do not decompose: **correct magnitude
requires near-correct direction, and correlation 0.69 is not enough.**

### Testing the direction hypothesis directly

If direction is the limit, the candidate cause is representational: the corpus
pools the backbone's P5 map (20x20 at the detector's native input) down to 4x4,
discarding 96% of structure already computed. The corpus was re-baked at 8x8 —
4x the spatial resolution, same layer, same frames — and probed before training
anything on it:

| features | 4x4 grid | 8x8 grid |
|---|---|---|
| `spatial_grid` alone, pose R² | 0.076 | **0.070** |
| `spatial_grid` alone, pose corr | 0.465 | **0.459** |
| `grid + frame_emb + proprio`, pose R² | — | 0.128 |
| `grid + frame_emb + proprio`, pose corr | — | 0.477 |

**Quadrupling the spatial resolution changes nothing.** The limit is not how
finely the P5 map is sampled; it is what P5 encodes. Frozen detection features at
stride 32 carry pose R² of order 0.1 for this task, and no amount of resampling
adds information that is not there.

Running the probe BEFORE training on the new corpus is the only reason this cost
40 minutes instead of two hours, and it is the discipline this document has been
arguing for throughout: measure the information content before building on it.

### What this means for the architecture

The design's central bet — a frozen open-vocabulary detector supplies all
perception, and only ~7M task heads train — is the thing now under question.
Every downstream component has been fixed, measured, and verified:

* grounding recovered (0% -> 75.8% detection)
* the objective corrected (two load-bearing terms identified out of six)
* divergence bounded (the NaN that corrupted every prior closed-loop number)
* rate matched, actuation excluded across five configurations and four gains
* the policy beats a linear probe on its own features (0.69 vs 0.415)

And closed-loop success is 0.000. The remaining gap sits in the one part of the
stack that was never allowed to learn. That is a defensible negative result about
frozen-encoder VLAs at this scale, and it is not the result this project set out
to report.

## 5k. Fresh-mind audit — flaws in 5j, defect 24, and the plan that follows

Read against the session that produced 5j and against this document as a whole.
Three conclusions in 5j do not survive scrutiny; one new defect explains why the
recovery arms could not have worked; and the paper's opening claims are no longer
the right target.

### Flaw 1 — "frozen features are the ceiling" overclaims the probe

The 8x8 grid probe is a clean negative on *resolution*: resampling P5 does not
add information. It is **not** a measurement that fine-tuning the backbone is
the only remaining lever.

* Pose **corr** from `spatial_grid` alone is **0.465**. Features constrain
  direction. Pose **R²** of 0.07–0.13 means a *linear* map cannot uniquely
  determine the action — expected under multimodal continuous control, not proof
  that the encoder lacks a usable error signal.
* The trained policy already **beats** the linear probe (corr 0.59–0.69 vs
  0.415). Capacity on these features is not exhausted.
* Recovery training was **never validly tested** (defect 23, then defect 24
  below). Declaring the encoder the ceiling before a correct recovery arm is
  premature.
* Fine-tuning YOLO-World breaks the design's central bet (Claim 6) and needs a
  frames-bearing re-bake. It is the expensive last resort, not the next step.

### Flaw 2 — the progress critic, as wired, reinforced the phase shortcut

`--critic-weight` supervised against `(t+1)/T` — pure wall-clock. The actor term
then asked the planner for actions that make the world model predict
"later-looking" latents. That is the same PHASE signal 4h measured the planner
already over-using. A task-aligned critic must score geometry (EEF travel toward
the episode's final pose), not time. Fixed: `progress_targets_eef`.

### Flaw 3 — the paper's Claim 1/2 framing is currently unreachable

Claim 1's kill bar (`<30%` where big models exceed 80%) is met. Claim 2
(perception-rate decoupling) cannot be measured at `mean_success = 0.000`. The
honest paper right now is the forensic / systems paper already assembled in the
skeleton (§PAPER SKELETON): interface defects that leave open-loop metrics healthy
while closed-loop reads zero. AIAYN framing in the opening is aspirational debt.

### Defect 24 — recovery proprio was computed and discarded

`--recovery-noise` built `_step_proprio` (perturbed EEF) and a corrected target,
then called the planner with `_noisy_proprio(batch, t, args)` — the
**unperturbed** vector when recovery is on, or a **fresh independent draw** when
`--proprio-noise` is on. The label described a displacement the observation did
not carry. That trains "emit a correction you cannot see", which produces
exactly the divergence-worsening signature of `dag_lo` / `dag_hi` even after the
target-clamp guard of 5i. Fixed: `proprio=_step_proprio`. The sum
`demo_action + correction` is no longer silently clamped either — corrections
are scaled per-dim to fit `[-1, 1]` so the observation and label stay matched.

Pinned by `tests/test_v8_train_path.py::test_stage_b_loop_passes_recovery_proprio_to_planner`.

### What to run next (ordered by cost and falsifiability)

1. **IBVS residual, zero training** — **RAN (§5p).** Phased IBVS at gain 0.5
   moves cream-cheese `eef_obj_dist_min` into the **~0.08 m** band across seeds
   and never moves `mean_success` off **0.000**. Frozen features are *not*
   proven to be the ceiling by this (IBVS bypasses the planner); the
   measurement falsifies "detector cannot servo at all" and leaves grasp/place
   conversion unsolved.
2. **Re-run recovery with defect 24 fixed** — **RAN** as `rec_fix` (§5l).
   Open-loop BC healthy; closed-loop still 0.
3. **Critic + light dreamer on the fixed recipe** — queued behind agentview
   disk unblock (§5o arms 2/4 parked).
4. **Only then** consider a finer backbone *layer* (P3/P4 hook, not P5
   resample) or a partial neck fine-tune. Full-backbone fine-tune remains the
   design-breaking last resort.

### Retracted from 5j

The sentence "the remaining gap sits in the one part of the stack that was never
allowed to learn" is suspended. The gap sits in a stack whose recovery path was
broken, whose critic target was a phase signal, and whose frozen-ceiling claim
was inferred from a resolution probe that does not license it. Measure (1)–(3)
before touching the encoder.

## 5l. Night of 2026-07-29 — recovery fix + sighted eval protocol

### Stage B `rec_fix` (Defect 24 fixed)

Recipe: `--recovery-noise 0.01 --variance-weight 0.1 --action-token-sampling 0.5`
on `data/libero_object_grid`, load `full_stageA_grid10.pt`, tag `rec_fix`.
Checkpoint: `checkpoints/full_stageB_rec_fix.pt`.

| epoch | val bc | grip | note |
|---|---|---|---|
| 8 | 0.1282 | 0.968 | |
| 14 | 0.1066 | 0.968 | |
| 16 | 0.1014 | 0.975 | |
| 18 | 0.0927 | 0.971 | |
| 20 | **0.0881** | **0.975** | best; ckpt on disk |
| 22 | 0.0930 | 0.973 | no improve 2/6 |
| 23 | — | — | train stopped early for night eval (budget) |

Open-loop BC is healthy (best val bc **0.0881**). That does **not** license
closed-loop competence.

### Eval protocol (shipped `350eef9`)

| knob | old | tonight |
|---|---|---|
| `det_conf` | 0.10 | **0.02** |
| `render_size` | 128 | **256** |
| grocery prompt tail | box, can, bottle | box, cardboard box, can |
| reported | mean_success only | + `src_detect_rate`, `eef_obj_dist_*`, `grip_close_rate` |

### Night sighted eval (completed 2026-07-30 06:55 UTC)

`eval.libero_eval` on `full_stageB_rec_fix.pt`: `--det-conf 0.02
--render-size 256 --no-brake --task-ids 0,1,2 --n-trials 3` →
`eval_results/night_sighted/libero_object_real_1785394362783_results.json`.

| metric | value |
|---|---|
| mean_success | **0.000** (0/9 trials) |
| src_detect_rate | **0.719** (was ~0.20 under det_conf=0.10 / 128px) |
| src_conf_mean | 0.061 |
| grip_close_rate | **0.653** |
| eef_obj_dist_min | **0.132 m** (alphabet soup trials reached **0.052–0.077 m**) |
| eef_obj_dist_at_20 | 0.257 m |
| eef_obj_dist_final | 0.750 m (diverges after closest approach) |

Per-task success all 0.0. Closest approach does not become a grasp: distance
grows from min to final, so the failure mode is **late-episode divergence /
grasp timing**, not “never saw the object.”

### Disposition of 5j

Protocol fix moved source detect **~20% → 72%** and produced centimetre-scale
approaches. That is the opposite of an encoder-only ceiling measured under a
blind eval. 5j stays **suspended / retracted as stated** (§5k): frozen features
supply usable error signal under honest detection; remaining gap is control
after approach (grasp + place), not “features contain nothing.”


## 5m. Binding precision is the measured ceiling (2026-07-30 morning)

Overnight IBVS forensics (`results/IBVS_SWEEP_FORENSICS.md`) closed the
lever ranking from §5k:

| stage | verdict |
|---|---|
| recall | fixed (det duty → ~0.98 under honest protocol / phased view) |
| servo / descend / grasp sequencing | works when the box is stable (unit-tested + sweep) |
| learned policy phase structure | broken separately — parks in basket empty-handed (video) |
| temporal tracking | **null** — locks the same wrong box |
| **phrase→object binding** | **BROKEN — the bottleneck** |

`--ibvs-phase` never left servo (`grip_close_rate` 0.000) because the stable
detection was the **basket** (conf ~0.15) for "salad dressing", or teleported
between objects (20–26% jumps >0.15). `--ibvs-track-gate 0.15` removed
teleports but success stayed 0 — stability without semantics.

5j returns sharpened: not "features lack geometry" (they servo), but
**open-vocab detection at this scale cannot stably bind the language phrase
to the correct object**. Claim 6's frozen-detector bet is the load-bearing
negative under that reading.

### Levers shipped this morning (eval / prompt)

1. **Receptacle-aware source tails** (`prompts.role_chains`): when the target
   is a basket/bin, strip `"box"` / `"cardboard box"` from the grocery SOURCE
   chain — those cues fire on the basket liner.
2. **`--ibvs-clip-rerank`**: among `Perception.proposals`, rebind each role to
   the box whose ROIAlign emb best matches that role's CLIP text emb,
   rejecting boxes that score higher on the *other* role.

### What §5l's "approach then diverge" meant

Video (postscript 2) reframes `eef_obj_dist_min` ~0.13 m as a **drive-by**
en route to the place prior, not a stalled grasp approach. There is no grasp
phase in the learned policy. Intermediates remain useful for sightedness;
they are not a grasp-competence score.

### CLIP re-rank null (01e0acf, 2026-07-30 14:38 UTC)

`--ibvs-phase --ibvs-track-gate 0.15 --ibvs-clip-rerank` plus receptacle-aware
box-tail strip on `rec_fix`: `mean_success` **0.000**, `grip_close_rate`
**0.000**, `src_detect_rate` **0.242** (worse than the un-reranked phased run's
~0.98). Salad-dressing trials still never leave servo.

Reading: cosine(ROIAlign-SPPF emb, CLIP text emb) is the wrong similarity —
those spaces are not guaranteed aligned (YOLO-World's contrastive head uses
cv4 region feats, not the SPPF GAP we ROIAlign for `BoxObs.emb`). Re-rank
therefore rejects usable boxes and starves the servo. The prompt strip alone
is not enough to move grasp.

Next zero-training lever that stays on the binding question: re-rank with
`TextRegionExtractor` / cv4 region↔phrase scores (the space already measured
to agree with the detector head), or bind source to the exact-phrase class id
only (no grocery tail) when that class fires at all.

## 5n. Defect 25 — the ceiling of §5m was measured through a camera that cannot see

§5m concluded that "open-vocabulary detection at this scale cannot stably bind
the language phrase to the correct object", and nominated that as the
load-bearing negative result of the paper. Every measurement behind that
sentence — the 20–26% teleport rate, the stable-but-wrong basket box, the
tracking null, the CLIP re-rank null — was taken on the **wrist** camera.

Every MicroVLA corpus was baked from `eye_in_hand_rgb`, and every closed-loop
eval read `robot0_eye_in_hand_image`. That is train/deploy consistent, which is
why 24 defect hunts never flagged it. It is also the one view in the dataset
where the frozen detector is blind.

### The measurement

Deployed prompt chain, real LIBERO demo frames, `det_conf` 0.02, detector short
side upscaled to 512 as in production. "Duty" is the fraction of frames on which
a role grounded at all; "jitter" is the mean absolute movement of the source
center between consecutive sampled frames, in normalized image coordinates.

| variant | src duty | src conf | tgt duty | tgt conf | props/frame | ctr jitter |
|---|---|---|---|---|---|---|
| wrist, as shipped | 0.219 | 0.011 | 0.212 | 0.007 | 0.45 | 0.183 |
| wrist, row-flipped | 0.237 | 0.009 | 0.419 | 0.020 | 0.74 | 0.177 |
| agentview, 180° (converter's) | 0.613 | 0.049 | 1.000 | 0.550 | 2.29 | 0.032 |
| **agentview, row-flipped (correct)** | **0.850** | **0.066** | **1.000** | 0.486 | **2.82** | **0.030** |

Source detection 0.219 → 0.850. Target 0.212 → 1.000. Proposals per frame
0.45 → 2.82 — the relational head was being fed an essentially empty scene.
Center jitter 0.183 → 0.030: **the teleporting of §5m is a property of the
wrist view, not of open-vocabulary detection.**

Annotated frames (`eval_results/bindprobe/`) make it concrete. On agentview the
target box is the basket on every frame and the source box holds one object and
tracks it through the grasp — for `alphabet soup` the box follows the can up
into the gripper at t=111. On the wrist the "basket" box lands on the robot's
own gripper finger, and the source box is on a different object in each of four
sampled frames.

### Two faults, not one

**25a — the wrong camera.** The wrist view at 128 px is a close-up of tabletop
with no scene context; YOLO-World-S grounds almost nothing in it. The corpus
statistics agree independently: over all 500 baked episodes of
`libero_object_grid`, the target role is detected on **1.4%** of frames
(541 of ~38 000). The policy has never had target evidence at all.

**25b — a mirrored de-rotation.** `preprocess/libero.py` corrected agentview
with `frames[:, ::-1, ::-1]`, a full 180° turn. robosuite renders through a
bottom-left-origin GL framebuffer, which is a **row reversal only**; the extra
column reversal is a left-right mirror. It costs source duty 0.850 → 0.613 on
its own, and it mirrors every baked box center with respect to the frame the
actions move in. A mirrored tabletop still looks like a tabletop, so nothing
downstream could complain. The wrist stream was never de-flipped at all, on
either side — self-consistent, but consistently upside down, and worth
target duty 0.212 → 0.419 by itself.

Orientation now lives in `microvla/utils/camera.py::upright` and nowhere else.
`preprocess/libero.py`, `eval/libero_eval.py`, `eval/record_mp4.py` and
`eval/openloop_check.py` all call it; `--camera` is a checked choice on both
sides. `eval/record_mp4.py` had been holding a *third* private copy
(`np.rot90(·, 2)`), so the "what is really happening" panel of every diagnostic
video in §5m was itself mirrored. 13 tests in `tests/test_camera_parity.py`.

### What this retracts

* §5m's headline — "phrase→object binding is BROKEN, the bottleneck" — is
  **withdrawn as stated**. It is established only for the wrist view. The
  binding question has not yet been asked of a view where the detector can see.
* The tracking null and the CLIP re-rank null (§5m) are **uninformative**, not
  negative: both were re-ranking and stabilizing boxes that were not on objects.
  The CLIP re-rank's diagnosis (SPPF-GAP and CLIP text are unaligned spaces)
  may still be correct, but it was not tested under conditions that could show it.
* §5j remains retracted, and now for a second, stronger reason: the "frozen
  features are the ceiling" claim was inferred from experiments run through a
  blind camera.

### The honest form of the defect

This is the same shape as the other 24, one level up. The earlier defects were
a producer and a consumer disagreeing about a convention. This is a producer and
a consumer **agreeing** on a convention that is wrong — which no parity test can
catch, because parity held. The instrument that caught it was not a test but a
question the project had never asked: *is the input any good?* Detection duty,
confidence and box stability per candidate view are four cheap numbers that were
never measured until the 25th defect hunt, after ten thousand GPU-seconds spent
on levers downstream of them.

### Status

`data/libero_object_agent` re-baking from agentview (10 Hz, 4×4 spatial grid,
500 episodes) with a duty gate that refuses to train if the corpus does not
clear 0.65 on both roles; stage A + stage B and a closed-loop eval at
`--camera agentview_image` follow automatically. Numbers land in §5o.

### Box-tail strip is also a null (precision↔recall)

Removing `"box"`/`"cardboard box"` from grocery SOURCE chains when the target
is a receptacle (intended to stop basket false binds) collapsed source detect
to ~0.1–0.4 and left `grip_close_rate` at 0 — same as CLIP re-rank. On this
detector, `"box"` carries most grocery recall; stripping it without a working
semantic binder only swaps wrong-object binding for no-object. Reverted in
prompts; both attempts stay documented as measured negatives under §5m.

### Defect 26 — the detector threshold was two numbers

Found while fixing 25, in the same file the bake builds perception in. Three
sites construct the real detector, and they disagreed:

| site | `det_conf` |
|---|---|
| `preprocess/common.py` (the bake) | 0.10 — the class default, never passed |
| `microvla/jepa/loop.py::build_real` | 0.10 — same omission |
| `eval/policy.py` (what eval actually used) | 0.02 |

The asymmetry was *written down* — `eval/policy.py`'s docstring said "bake keeps
the class default 0.10" — and left standing. It is consequential twice: the
threshold decides which boxes exist at all, and every surviving box carries its
confidence into fusion's `box_weight` fade, so a split threshold hands the
deployed policy evidence weights the training distribution never contained.
`cfg.det_conf` is now the single value all three read.

### The instrument: corpora that describe themselves

Four of the 26 defects are one sentence with a different noun — *the deployment
used a different camera / detector threshold / render size / perception period
than the corpus was baked with*. None raises anything. Each surfaces as
`mean_success 0.000`, which is indistinguishable from a policy that does not
work, and each cost days.

Their common root cause is not in any file: **the corpus did not record what
produced it**, so no consumer could check. `run_conversion` now writes a
`provenance` block into `manifest.json` (camera, `eval_camera`, deflip,
`det_conf`, `real_frame_hz`, `source_hz`, `detect_frame_hw`, `max_objects`,
`grid_size`), and `microvla/utils/provenance.py` is the consumer side, called by
`eval/libero_eval.py` before it scores anything. Mismatches are logged as ERROR
and written into the results JSON, so a number whose deployment did not match
its corpus stays self-identifying after the scrollback is gone.
`--strict-provenance` refuses to run at all.

It immediately flags a live one: the bake reads 128 px hdf5 frames while eval
renders at `--render-size 256`, and the two upscale differently into the
detector's 512 px short side. §5o scores both.

This is the generalizable claim of the systems half of the paper: in a stack
built from frozen encoders and offline-baked features, **the feature corpus is
an interface, and an interface with no schema will drift**. Twenty-six defects,
and the ones that survived longest are all of this shape.

### Not a defect: the evidence path sees two boxes

Checked while hunting 27 and reported as a negative. `EvidenceEncoder` — the
TRM's entire view of the scene — is fed by `pack_objects`, which puts the source
role in slot 0, the target in slot 1, and zeros in the remaining
`cfg.max_objects - 2`. The trainer (`train/train_batched.py::_obj_tokens`) and
the robot (`microvla/v8.py::FusionAdapter`) do the *same* thing, so there is no
parity break; the relational head separately receives the full baked proposal
set on both sides (`objects_from_batch` / `JEPALoop._rel_tokens`).

What is true is narrower, and only became material with 25: on agentview the
detector supplies 2.82 proposals per frame that the world model's evidence port
cannot see. That is an underuse, not a mismatch, and changing it is deliberately
held out of the 25 experiment so the camera is the only variable.

### Defect 27 — the deployed spatial adapter never saw the grid it was trained on

Found by an adversarial audit of the paths that survive 25, and confirmed by an
independent verifier reading the same code.

`microvla/jepa/loop.py` contains a branch that explicitly prefers the coarse
spatial grid the corpus bakes. Its own comment names the hazard: *"a
full-resolution map here against a g × g pooled one in training is the
resolution-mismatch version of every other train/deploy defect in this stack."*
The branch was dead, twice over:

1. `_percept_to` rebuilt `Perception` with four fields and dropped
   `spatial_grid`, which defaults to `None`. The real-tick path rebuilt it a
   second time with the same omission. **This is the second optional field lost
   in exactly this way** — `proposals` was the first, and cost the relational
   head its evidence on every real tick.
2. Both deployment sites constructed `YoloWorldPerception` without `grid_size`,
   so `perceive()` never produced a grid at all. Either fault alone is
   sufficient; each hides the other.

So TQSA was trained on a `[B, 512, 4, 4]` map whose 16 cells are each
`standardize()`d, and deployed on the raw, un-standardized `[1, 512, ~20, ~20]`
SPPF map — softmax attention over ~400 positions instead of 16, in a different
normalization. Live in every `--tqsa` run on a grid corpus, which is the recipe
every arm in §5l–5n uses.

The regression test now walks `dataclasses.fields(Perception)` rather than
checking the two fields we happen to know about, because the failure mode is
"someone adds a third optional field."

### Two more, confirmed and NOT fixed (deliberately)

**The waypoint actuator is open-loop by construction.** `WaypointActuator`
re-anchors its target from the same `eef` it subtracts eight lines later
(`anchor_real` defaults `False` and no call site passes it), so
`(target − eef)` reduces algebraically to `disp[row] · waypoint_range`: the
measured proprio cancels exactly. The verifier reproduced it — a frozen arm and
an arm moving 5 cm/step emit **bit-identical** commands. The module's entire
documented premise ("magnitude stops being the network's job"; "if the arm lags
the error stays large until it arrives") is void, and §4e's reading of
`|cmd| 0.4533` as evidence of a tracking controller is **retracted**: it is a
fixed per-axis constant times the head output, so the translation command
inherits the head's shrinkage 1:1.

This is *not* a parity defect — `train/train_batched.py`'s actuation loss
hard-codes the same closed form, and it is exact precisely because the feedback
is dead, so both sides agree. Turning the feedback on would *create* a mismatch
unless the trainer simulated the arm. Recorded, not patched, and held out of the
§5o experiment.

**`waypoint_row_stride` is a 2 Hz constant.** `--waypoint-long` sets stride 10,
which is `20 Hz / 2 Hz`; the dense corpora are baked at 10 Hz, where the true
stride is 2. Both the trainer's actuation loss and the deployed actuator divide
by the same `cfg` field, so every parity test passes while the emitted
translation rate is wrong by the stride ratio, and the two supervision terms on
the head pull against each other. The runs in §5o pass `--waypoint-row-stride 2`
explicitly, so they are clear of it; the earlier `--waypoint-long` arms were not.

Both were found by asking a different question than "do the two sides agree?" —
*agreement is not correctness*, which is the same lesson 25b taught with the
mirrored de-rotation.

## 5o. The agentview corpus (2026-07-30 15:09 UTC)

`data/libero_object_agent`: 500 episodes, agentview, 10 Hz, 4×4 spatial grid,
`det_conf` 0.02, corrected row-flip. Same converter, same recipe, same detector
as `libero_object_grid` — the camera is the only difference.

| corpus statistic | wrist (`_grid`) | agentview (`_agent`) |
|---|---|---|
| source role detected | 0.491 | **0.970** |
| target role detected | **0.014** | **0.999** |
| source detection, worst decile of the episode | 0.25 | 0.95 |
| target center std (x, y) | — | [0.010, 0.006] |
| episodes whose source box never moves | n/a (camera moves) | 0.032 |

Three things worth stating plainly.

**The target role was never observed.** Over 500 wrist episodes the basket was
detected on **1.4%** of frames — 541 of ~38 000. Every claim about the policy's
"place phase", including §5m's video reading that it *drives to the basket and
parks*, describes a policy that had never been shown the basket. On agentview
the target is detected on 99.9% of frames, at a center whose standard deviation
is 0.010 — a static camera pinning a static basket, which is exactly right and
is also a free correctness check on the whole pipeline.

**Detection is no longer a function of episode phase.** The wrist corpus's
source detection sagged to 0.25 in the fourth decile — the approach, when the
gripper occludes the object and the camera is closest to it, i.e. precisely when
the policy needs the box most. Agentview is flat at 0.95–1.00 across all ten
deciles.

**The source box tracks the manipulated object.** With a static camera, "which
box is the right one" becomes answerable without labels: in a `libero_object`
demo exactly one object moves, so a correctly-bound source box must be the only
box in the scene that travels. `const_frac` — episodes whose source box never
leaves a 0.05 window — is **0.032**. In 96.8% of episodes the source box follows
something that moved.

### The residual: source == target on 40.9% of frames

The same instrument exposes what agentview does *not* fix. On 40.9% of frames
where both roles ground, they ground to the **same box**. The source chain falls
through its exact phrase (YOLO-World scores "alphabet soup" 0.000 — §5m's one
surviving finding) to a generic tail, and the basket is the most box-like thing
in the scene. Every task in the suite is *"pick up X and place it in the
basket"*, so **source == target is definitionally wrong**, which makes this
measurable with no labels at all.

`cfg.role_disjoint_iou` resolves the target first — the reliable role at 99.9%
duty — then lets the source skip candidates overlapping it:

| `role_disjoint_iou` | source duty | source == target | usable source evidence |
|---|---|---|---|
| 0.0 (arm 1) | 0.990 | 0.273 | 0.720 |
| **0.1 (arm 2)** | 0.960 | **0.000** | **0.960** |

"Usable" is `duty × (1 − same)`: the box is present *and* is not the basket.
The collision goes to zero for three points of duty, so this is a real gain
rather than a role traded away for a miss. Held out of arm 1 so the camera stays
the only variable; arm 2 differs from arm 1 in this one knob, recorded in
provenance and checked at eval time.

### Arms in flight → parked (2026-07-31)

| arm | corpus | differs by | status |
|---|---|---|---|
| 1 | `_agent` | camera (25a) + row flip (25b) + `det_conf` (26) + TQSA grid (27) | **PARKED** (disk) |
| 2 | `_disj` | arm 1 + `role_disjoint_iou` 0.1 | **PARKED** |
| 2a / 4 | dream / critic stacks | see table below | **PARKED** |
| control | `_wristctl` | arm 1 with the wrist camera, same code | **PARKED** |

Pod disk sat at ~4.8 GB free (30 GB overlay, 85% full). Agentview bake/train
scripts live under `/root/queue/parked/` (`02c_disjoint.sh`, `02ca_dream.sh`,
`02d_critic.sh`, `03_wristctl.sh`). Do **not** resume until free space clears —
a 500-episode agentview bake plus checkpoints will OOM the disk mid-run.
Meanwhile the continuous runner (`scripts/auton_cont.sh`) burned GPU on the
wrist IBVS forensics of §5p.

The control still matters: every previous wrist number was produced by a
different code version, so it cannot be differenced against arm 1. Without it
the paper would be asserting a counterfactual it never ran.

### What moving the camera does to the world model's job

Reviewers will ask, and the answer is not the obvious one. A static camera
should make "predict the next latent" easier, because only the arm and the
manipulated object move. Measured on the baked corpora (120 episodes each, no
model involved):

| corpus | per-step latent motion | persistence R² | drift from anchor |
|---|---|---|---|
| wrist (`_grid`) | 0.071 | **0.764** | 0.236 |
| agentview (`_agent`) | 0.044 | **0.494** | 0.096 |

The agentview latent moves *less* per step and is *less* predictable by copying.
The wrist latent moves more, but coherently — a translating camera sweeps the
embedding through a large, smooth range, so its total variance is large and the
step-to-step residual is a small fraction of it. On a fixed camera the scene's
total variation over an episode is small, so the same absolute step is a larger
share of it.

Two consequences to state honestly:

* **Stage A's `wm_margin` is measured against a weaker baseline on agentview.**
  The `+43.3%` of §4q was over a persistence baseline that already explained 76%
  of the next latent; on agentview persistence explains 49%. Margins across the
  two corpora are not comparable, and §5o's must be reported with this number
  beside it.
* **The drift encoder gets a smaller signal.** Anchor drift falls 0.236 → 0.096,
  so the HRM's state code spans less of the space it was sized for. If the
  agentview arm underperforms on anything, this is the first place to look — and
  it is a design consequence of the camera change, not a defect.

Neither undermines the change: control needs the *boxes*, and those went from a
source role grounded on 22% of frames with a box that jumped 0.18 between frames
to 97% with 0.03. But a paper that reported the grounding win without this table
would be selecting its evidence.

### The second half of the fix: the phase shortcut was never regularized

Fixing grounding gives the policy something to see. It does not make the policy
look. `microvla/config.py` and `train/train_batched.py` both record an earlier
measurement of where the plan's sensitivity actually sits:

    proprio 0.291 >> state_delta 0.075 > wm_msg 0.031 > current_emb 0.025
    ~ fused 0.023 > pred_box 0.013 > geometry 0.004 > next_emb 0.001

The plan is ~12× more sensitive to arm pose than to any visual input. A policy
of that shape replays an average trajectory conditioned on where the arm is,
which on a suite of stereotyped pick-and-place demos with a fixed basket
reproduces most of the action variance — and is exactly the "drives to the
basket and parks, empty-handed" behaviour the §5m videos show.

`train/train_batched.py` already ships the countermeasure. `--phase-dropout`
withholds `state_delta` and `proprio` during stage B, deliberately asymmetric
with `--planner-input-dropout` (which withholds the vision paths): *drop the
shortcut more than the signal you want used*. Its own help text names the
symptom — *"phase sensitivity 0.464 vs vision 0.040 (12:1), and a policy that
reaches the basket perfectly and never touches the object. Try 0.3."*

**It defaults to 0.0, and every arm ever run left it there**, including §5o's
arm 1.

The naive fix is wrong, and the repository already says so. `--planner-drop-rate`'s
help records that `--phase-dropout 0.3` *was* tried: it bought a 2.3× better
phase:vision ratio and **collapsed the gripper from 0.93 to 0.50**, because
proprio carries the arm's gripper state — the single best predictor of the
gripper command — so withholding it destroys the BCE head. Its verdict is "drop
`state_delta`, keep `proprio`."

Taken literally that verdict is too weak here: `proprio` (0.291) *is* the pose
shortcut, and `state_delta` is only 0.075, so dropping `state_delta` alone barely
touches the dominant path. Arm 2 therefore drops `state_delta` hard (0.5) and
`proprio` moderately (0.2), with the existing `--grip-weight 2.0` protecting the
BCE head and open-loop gripper agreement as the tripwire: if it returns near
0.50, the trade failed and `proprio` goes back to 0.

This is the trade the earlier measurement could not make. Withholding the
shortcut only helps if something else can carry the task, and until this week the
alternative was a source role grounded on 22% of frames by a box that moved 0.18
between them.

That was not obviously wrong before now. Withholding the shortcut only helps if
something else can carry the task, and until this week the alternative was a
source role grounded on 22% of frames by a box that moved 0.18 between them.
Regularizing the shortcut against evidence that bad would have starved the
policy, not redirected it. With agentview at 0.970 / 0.999 the trade is
available for the first time — which is why it is arm 2 rather than a knob that
should have been on all along.

| arm | corpus | one change from the previous arm | cost |
|---|---|---|---|
| 1 | `_agent` | camera + orientation + threshold + TQSA grid | bake + train |
| 2 | `_agent` | `--planner-drop-rate state_delta=0.5,proprio=0.2` | train only |
| 3 | `_disj` | `role_disjoint_iou 0.1` | bake + train |
| 4 | `_agent` | arm 2 + geometric progress critic + dreamer 0.01 | train only |
| control | `_wristctl` | wrist camera, arm-1 code | bake + train |

Arm 4 is the step-3 recipe of §5k, stacked on arm 2's drop rates rather than
replacing them: the critic and the drop rates attack the same shortcut from
opposite ends. The critic asks for actions the world model believes ADVANCE the
task, scored by EEF travel toward the episode's final pose — geometry, not
wall-clock, because a time target reinforces exactly the phase signal being
suppressed (§5k flaw 2). The dreamer term stays at 0.01: the world model is real
(`wm_margin` +43.3%) but exploitable, and an actor optimizing an imagined latent
will find the exploit if given weight.

Each arm also scores `--no-dream-correct` (defect 28) as a free eval-side A/B,
and `eval.bench --sensitivity` after every retrain, so the question "is the
policy using vision yet" gets a number rather than an opinion.

## 5p. Wrist IBVS / tool-phase marathon (2026-07-31 → 2026-08-01 UTC)

Continuous runner: `scripts/auton_cont.sh` on the pod, checkpoint
`checkpoints/full_stageB_rec_fix.pt`, norm stats
`data/libero_object_grid/norm_stats.json`, camera
`robot0_eye_in_hand_image`, `--render-size 256 --det-conf 0.02 --no-brake
--role-disjoint-iou 0.1 --source-max-area 0.12 --perception-period 2`.
**85 unique named runs** under `eval_results/auton/`. Every run:
`mean_success = 0.000`.

### Honesty constraint (read first)

These numbers are **assisted**. `--ibvs-phase` replaces / overlays the learned
plan with an image-based servo on the detected source; `--tool-phase` is a
scripted grasp handoff. They answer "can the detector + a P-controller get the
EEF near the object?" — not "does MicroVLA succeed closed-loop." Citing them as
policy success is a defect of the same class as `--mock-env` scoring 1.000.

### Frozen best config (cream cheese, task 1)

```
--ibvs-phase --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.4
--ibvs-conf-floor 0.05 --ibvs-descend-hyst 0.50 --ibvs-target-uv 0.5,0.60
```

| aggregate | value |
|---|---|
| seeds in best-config family | 9 (`hyst50_s{7,13,19}`, `best_s{23,29,31,37,41}`, + `aim60_hyst50_n10`) |
| mean `eef_obj_dist_min` | **0.087 m** (σ 0.011) |
| min / max eef | **0.077** / **0.112** |
| mean `grip_close_rate` | **0.67** |
| `src_detect_rate` | ~0.89–0.97 |
| `mean_success` | **0.000** |

Per-seed eef (task 1, best config): s7 0.083 · s13 0.087 · s19 0.100 ·
s23 0.086 · s29 **0.077** · s31 0.081 · s37 0.112 · s41 0.079 ·
n10 hyst50 **0.079**.

### Hysteresis: seed-0 noise vs seed-fair

Early n=5 runs at seed 0 made hyst 0.60 / 0.65 / 0.70 look like winners
(`eef=0.068`). That was **one-seed noise**. Fair compares:

| compare | hyst 0.50 | hyst 0.60 | hyst 0.65 | winner |
|---|---|---|---|---|
| n=10, same seed family | **0.079** | 0.084 | 0.084 | **0.50** |
| seed 7 (n=5) | **0.083** | 0.090 | — | **0.50** |
| seed 13 (n=5) | **0.087** | 0.094 | — | **0.50** |
| seed 19 (n=5) | **0.100** | 0.112 | — | **0.50** |

**Hyst 0.50 wins 3/3 seed pairs** and the n10 head-to-head. Do not resurrect the
n5 seed-0 "hyst60 wins" headline.

### Aim UV is task-dependent

Same phased stack; only `--ibvs-target-uv` and task id change.

| task | aim V (U fixed 0.5) | best measured eef | grip | succ |
|---|---|---|---|---|
| 1 cream cheese | **0.60** | **0.077–0.087** (multi-seed) | ~0.66 | 0 |
| 0 alphabet soup | 0.50 | **0.156–0.162** (s7/s13) | ~0.73 | 0 |
| 0 soup | 0.48–0.50 (earlier) | ~0.15 | ~0.73 | 0 |
| 2 salad dressing | 0.55 / 0.60 (earlier) | **0.243–0.255** | ~0.50–0.56 | 0 |
| 2 dressing | aim55 mid-run (2026-08-01) | trials ~0.21–0.25 | 0.12–0.60 | 0 |

Cream cheese is the only task where IBVS produces a stable ~8 cm near-miss.
Soup plateaus ~15 cm; dressing ~24 cm. A single global aim UV is wrong; a
task-conditioned aim (or a learned aim) is the next eval-only knob that is still
honest as assisted control.

### Ablations that did **not** beat aim60+hyst50 (cream)

All `mean_success=0`. eef relative to the fair n10 baseline 0.079:

| lever | run | eef | note |
|---|---|---|---|
| CLIP re-rank | `band050_aim60_clip` | 0.081 | no win |
| track gate | `band050_aim60_track` | 0.082 | no win |
| swap UV | `band050_aim60_swap` | 0.095 | worse |
| U=0.45 / 0.55 | `…_u45` / `…_u55` | 0.082 / 0.086 | no win |
| gain 0.4 | `…_gain04` | 0.091 | worse |
| deep descend | `…_deep` / `…_deep_g06` | 0.101 / 0.091 | worse |
| n=20 confirm | `band050_aim60_t1_n20` | 0.091 | variance, still 0 succ |
| residual mild (`rec_mild`) | `rec_mild_t1_n10` | **0.141** | grip 0.38 — worse than phased |

### Tool-phase: detection without grasp

`--tool-phase` (+ variants: loose / settle / deep / sign-flip / swap-uv /
agentview cam). Across all six named tool arms:

| arm | eef_min | grip_close | src_detect |
|---|---|---|---|
| tool_settle | 0.242 | **0.000** | 1.00 |
| tool_deep | 0.245 | **0.000** | 1.00 |
| tool_swap_uv | 0.246 | **0.000** | 1.00 |
| tool_loose | 0.253 | **0.000** | 1.00 |
| tool_sign_flip | 0.291 | **0.000** | 1.00 |
| tool_agent | 0.303 | **0.000** | 1.00 |

Detector fires; the tool never enters the grip-close region. UV/sign/depth/swap
knobs moved eef by centimetres and left `grip_close_rate` at zero. Binding into
a scripted grasp is still broken on wrist (and the one agentview tool arm did
not convert either under this recipe).

### Sign / swap / early UV grid (negative atlas)

Early 2-trial probes (not seed-fair; kept so we do not re-run them):

* Wrong IBVS sign families (`sign_pp_*`, `swap_n1_*`): eef **0.26–0.28**, grip 0.
* Pre-band UV grid (`uv_055_*`, `aim_*`): eef **0.22–0.26**, often grip 0.
* `band_035` / `hyst_035`: eef ~0.19, grip 0 — gain/hyst too weak to servo.

Negative `--ibvs-sign` on the CLI must use `=` form (`--ibvs-sign=-1,1,0`) or
argparse eats the minus.

### Ops notes (so the next agent does not repeat them)

* `scripts/auton_cont.sh` now uses `is_busy()` via `ps -C python,python3` —
  `pgrep -f` matched SSH `bash -c` lines that *mentioned* `libero_eval` and
  stalled the queue for ~20 minutes. Queue pops are flocked.
* Multiline Python in `logs/auton_queue.txt` corrupts the queue; one line per
  job, `name|cmd`.
* `record_mp4` once failed with `BrokenPipeError` under the runner's log
  redirect; eval jobs are the reliable path. Disk budget ~4.8 GB free —
  no more agentview bakes until cleared.
* Local watch videos pulled under `watch_videos/` (aim60 / hyst50 / tool_* /
  soup aims, etc.).

### What §5p changes in the argument

1. **Servo works on cream cheese under assisted IBVS.** Centimetre-scale
   `eef_obj_dist_min` with detect ~0.9 and grip_close ~0.65 is reproducible
   across seeds. The wrist view is not "blind" for *this* object under the
   honest det protocol + phased controller — it is still a bad view for soup
   and dressing.
2. **Success conversion is the remaining wall.** 85 runs, 0 successes. Closer
   eef and higher grip_close do not become LIBERO task success. The failure is
   past approach (grasp timing / place / binding into the grasp), not "never
   near the object" for cream cheese.
3. **§5m's binding bottleneck is task- and controller-conditional.** Agentview
   (§5n/5o) still retracts the *global* "open-vocab cannot bind" claim. On
   wrist + IBVS, cream cheese binds well enough to servo; dressing does not
   (eef stuck ~0.24, conf ~0.05). Tool-phase never grasps. The honest sentence
   is narrower than §5m's headline and narrower than a full retraction.
4. **Eval-only knobs are exhausted for success.** Aim, hyst, gain, clip-rerank,
   track, swap, deep descend, mild residual, tool-phase variants — none moved
   `mean_success`. Next moves that could are (a) task-conditioned aim as code,
   (b) hybrid policy→tool handoff with a real grasp trigger, (c) unparking
   agentview train arms once disk allows. (a)–(b) stay assisted unless success
   appears with those flags off.

### Pod status at write time (2026-08-01 ~02:30 UTC)

| item | state |
|---|---|
| runner | `auton_cont.sh` alive; `is_busy` fixed |
| in flight | `dress_aim55_s7` (task 2, aim V=0.55) |
| queue | `dress_aim60_s7` → `vid_best_t1_s29b` → `scorecard_dump` |
| agentview bake/train | parked (`/root/queue/parked/`) |
| disk | ~4.8 GB free / 30 GB |

## 5p. Arms 1–3 measured; two hypotheses killed (2026-08-01)

### Arm 1 (agentview) — the policy became sighted, and stayed at 0.000

| metric | wrist era | arm 1 |
|---|---|---|
| planner sensitivity, vision (`relational`) | 0.023 | **0.3108** |
| planner sensitivity, phase (`proprio`) | 0.291 | 0.0656 |
| ratio | 12:1 **phase** | **4.7:1 vision** |
| action `std_ratio` | 0.26–0.42 | **0.983** |
| pose corr | ~0.55 | 0.79 |
| gripper agreement (teacher-forced) | 0.93 | 0.944 |
| we close / demo closes | — | 56.2% / 58.5% |
| live source detection | 0.19–0.24 | 0.87–1.00 |
| `eef_obj_dist_min` | 0.297 | 0.108–0.163 |
| **mean_success** | 0.000 | **0.000** |

The camera fix did what it was supposed to do. The phase shortcut of §4h is gone
— the plan is now 4.7× more sensitive to the relational (vision) token than to
arm pose, an inversion of the 12:1 that pointed the other way. Magnitude sits at
`std_ratio` 0.983, dead centre of §4p's [0.95, 1.05] passing band. This is a
healthy open-loop policy by every instrument this project has built, and it
scores zero.

### Arm 2 (anti-shortcut drop rates) — worse, and the reason is instructive

`--planner-drop-rate state_delta=0.5,proprio=0.2` collapsed `std_ratio`
**0.983 → 0.358** and *reduced* the vision:phase ratio to 1.7:1. The gripper
tripwire held (0.933), so the correction that avoided the known 0.93→0.50
collapse was right, but the lever itself is counterproductive once vision
already dominates: withholding inputs from a planner that is finally using them
just shrinks its output. Arm 2 is abandoned.

### Two hypotheses killed

**The dream path is exonerated.** `wm_margin` went +43.3% (wrist) → **−29.3%**
(agentview): the world model is now beaten by persistence, and at
`perception_period 2` half of every tick's latent comes from it. Removing the
dream ticks entirely (`--perception-period 1`, a real perception every env step)
gives `mean_success` **0.000** and a *worse* approach (0.167–0.190 vs
0.112–0.168). The world model's negative margin is real and is not what blocks
closed-loop success. §5o predicted the static camera would make persistence a
*weaker* baseline; the measured margin says the opposite, and that prediction
was wrong.

**The corrector is exonerated too.** `--no-dream-correct` (defect 28) changed
nothing measurable on either arm.

### Calibrating the instrument everyone has been reading

`eef_obj_dist` is `‖eef_pos − obj_body_pos‖`, and the object's body origin is
its centre, not its graspable surface. The whole "the policy stops ~10 cm short"
narrative assumed a grasp happens near 0. Replaying demonstrations through the
same env, logging the same quantity:

| | value |
|---|---|
| demo min `eef_obj_dist` | **0.026 m** (0.007–0.051) |
| demo distance when the gripper CLOSES | **0.040 m** (0.009–0.066) |
| replay success | **5/6** |

So the target is 0.04, not 0.02 — but the gap survives calibration. Arm 1
reaches 0.108–0.163 and the best autonomous IBVS config reaches 0.079, i.e.
**2–4× the distance at which a demonstration commits to its grasp**, with the
gripper closing anyway (close rate 0.66–0.74 on the autonomous arms).

A methodological note worth keeping: the first version of this measurement
paired demo *i* with `benchmark.get_task_init_states()[i]` and reproduced only
**2 of 6** successes, which would have read as "the environment is unreliable".
Those init states are the EVALUATION set and are not index-aligned with the
demonstrations. Replaying from each demo's own recorded `states[0]` gives 5/6.
The instrument built to calibrate an instrument needed calibrating — which is
the same lesson as §5n, one level further down.

### Where that leaves it

Grounding, magnitude, direction, rate, the world model, the corrector and the
brake are now all excluded by measurement. What remains is the last stretch of
the approach: a policy that tracks the demonstrator well enough to score 0.944
gripper agreement under teacher forcing cannot close the final 4–12 cm when fed
its own states. That is compounding error in its textbook form, and it is the
one explanation this study has never been able to rule out.

## 5q. Frame-dynamic centering loss — mixed / negative on proximity (2026-08-01)

Hypothesis: bake the hyst50 IBVS prior into stage B so the *unaided* policy
stops closing beside the object. Recipe on `full_stageB_rec_fix.pt` →
`full_stageB_center_frame.pt`:

```
--v8 --tqsa --resume-stage-b
--centering-weight 1.0 --centering-in-frame --centering-err-weight
--centering-uv 0.5,0.60 --centering-sign 1,-1 --centering-gain 0.5
--centering-conf-floor 0.05
--depth-weight 0.5 --depth-descend -0.4 --depth-tol 0.20
--grip-weight 2.0
```

Best open-loop: val bc **0.0733**, grip 0.975 (epoch 10/12).

### Fair closed-loop (cream cheese, wrist, same eval knobs, no IBVS unless noted)

| arm | mean eef_min | mean grip_close | n seeds | succ |
|---|---|---|---|---|
| `rec_fix` plain | **0.081** | 0.18 | 2 (s7, s13) | 0 |
| `center_frame` plain | 0.108 | **0.41** | 4 | 0 |
| `center_frame` + hyst50 | 0.090 | 0.67 | 3 | 0 |

Paired plain seeds: s7 cf 0.108 vs rec **0.084**; s13 cf 0.114 vs rec **0.078**.
Centering **worsens** proximity vs the parent checkpoint and **raises**
grip_close (~2×). Assisted hyst50 on `center_frame` (0.090) does not beat the
earlier `rec_fix`+hyst50 band (~0.079–0.087). Success remains 0.

Videos: `watch_videos/center_frame_plain/`, `watch_videos/center_frame_hyst50/`,
plus parent A/B `watch_videos/rec_plain/`, `watch_videos/rec_hyst50/`.

### Qualitative (cream cheese seed 7 videos, 2026-08-01)

| arm | what the video shows |
|---|---|
| `center_frame` plain | bad approach — abandoned |
| `center_frame` + hyst50 | knocks the milk carton; fallen milk then **blocks** the last descent onto cream cheese |
| `rec_fix` + hyst50 | **same milk-block failure** — eef~0.08 is a deadlocked near-hover, not a clean approach |
| `rec_plain` / `ag110` | scene stays upright, but a **standing can** fouls the wrist/forearm — the arm head cannot finish the last descent onto cream cheese |

So the hyst50 "proximity win" is partly an artefact (knocked milk). The unaided
near-miss is also not a pure centimetre shortfall: `eef_obj_dist` to the cream
cheese *body* can look fine while the gripper aperture is physically blocked by
a neighbour can. Closing that gap means a **clearance-aware approach path**
(demos clear the can; the policy's descent does not), not more image-servo gain
or emit-scale.

### Additive residual on `rec_plain` (no `--ibvs-phase`) — also negative

Seed-7 cream cheese, same binding knobs as the fair compare. Residual *adds*
to the policy action (milk-safer than phased takeover):

| arm | gain | descend | eef_min | grip_close | succ |
|---|---|---|---|---|---|
| `rec_plain` | — | — | **0.084** | 0.17 | 0 |
| `nudge_g10_d15` | 0.10 | −0.15 | 0.092 | 0.23 | 0 |
| `nudge_g15_d20` | 0.15 | −0.20 | 0.094 | 0.23 | 0 |
| `nudge_g20_d25` | 0.20 | −0.25 | 0.098 | 0.38 | 0 |
| `nudge_g25_d20` | 0.25 | −0.20 | 0.104 | 0.30 | 0 |
| `nudge_g08_d30` | 0.08 | −0.30 | 0.090 | 0.24 | 0 |

Monotone: more residual → worse proximity. The clean miss is not fixed by a
P-controller on the detector — either the image error points the wrong way at
the critical ticks, or any nonzero lateral add knocks the trajectory off the
BC path that almost worked. Eval-only IBVS (phased *and* additive) is exhausted
for cream cheese under this checkpoint.

### Action-gain diagnostic on `rec_plain` (no IBVS)

| arm | action_gain | eef_min | grip_close | succ |
|---|---|---|---|---|
| `rec_plain` s7 | 1.0 | 0.084 | 0.17 | 0 |
| `ag110` s7 | **1.10** | **0.075** | 0.18 | 0 |
| `ag120` s7 | 1.20 | **0.075** | 0.14 | 0 |
| `ag130` s7 | 1.30 | 0.085 | 0.30 | 0 |

Seed-7 looked like a ~1 cm win at gain 1.1–1.2; 1.30 overshoots. Multi-seed
`ag110` (n=4): **0.075 / 0.080 / 0.091 / 0.084**, mean **0.082** — wash vs
`rec_plain` (~0.081). Eval-time action-gain is not a reliable fix; any train-time
magnitude term would need to move the conditional mean, not just rescale at
emit. Videos: `watch_videos/ag110/`, `watch_videos/ag120/`.

### Solo scene (`--clear-distractors`) — null on success (2026-08-01)

Kept only cream cheese + basket; teleported the other 5 bodies under the table.
`solo_s7`: eef **0.082** / grip 0.29 / succ **0** (plain was 0.084 / 0.17).
Video `watch_videos/solo/`: still fails — **does not go deep enough**, and is
still off-center. Neighbour collision is not the binding constraint.

### Reading (revised)

Detection knows where the object is. The unaided failure is **(1) not centered
on the object** and **(2) under-descent on the grasp** — not “can blocks the
head.” Knocked-milk under phased IBVS is a real side effect of aggressive XY
servo, but clearing the table does not unlock success. Full scene restored.

### Center + deep Z (additive, full scene) — also negative

| arm | eef | grip | succ |
|---|---|---|---|
| `rec_plain` | **0.084** | 0.17 | 0 |
| `cen_deep` g12/d55 | 0.093 | 0.23 | 0 |
| `cen_deep` g08/d70 | 0.089 | 0.24 | 0 |
| `tool_z03` | 0.278 | 0.00 | 0 |

Gentle XY + deeper descend residual does not beat plain; tool deeper-z is
worse. Video: `watch_videos/cen_deep/`. The depth shortfall is not fixed by
scaling the same P-controller harder.

Eval lever (zero-train): `--clearance-gain` / `--clearance-lift` /
`--clearance-aim-bias` — repel from the nearest non-source proposal and
optionally lift instead of grinding down (`microvla/utils/ibvs.py`).

Best recipe on `rec_fix` (`g=0.25, lift=0.15, r=0.40`, no IBVS): s7 eef
**0.089** vs plain **0.084**; s13 **0.083** vs **0.078** — wash/slightly
worse, succ 0. Image-space neighbour repulsion does not unlock the grasp.
Video: `watch_videos/clr_g25/`.

### Pre-grasp reweight (`pregrasp3`) — also negative on proximity

Stage B from `rec_fix` with `--pre-grasp-weight 3.0` (best val bc 0.0856).
Unaided closed-loop:

| arm | eef_min | grip_close | succ |
|---|---|---|---|
| `rec_plain` s7 / s13 | **0.084 / 0.078** | 0.17 / 0.20 | 0 |
| `pregrasp3` s7 / s13 | 0.095 / 0.093 | **0.41 / 0.66** | 0 |

Same trade as centering: more gripper, worse approach. Video:
`watch_videos/pregrasp3/`. Overweighting pre-grasp BC timesteps does not
reproduce the demo's clearance XY; the can-block last descent remains.

## 5r. Telemetry forensics: the grasp gate fires on a constant hand-eye offset (2026-08-01)

Every §5p–§5q lever assumed the residual failure was a *servo* problem (wrong
gain, wrong aim, wrong loss). Re-reading the phased-IBVS telemetry — which logs
`eef` (proprio) and `obj_pos` (sim ground truth, diagnostic only) per tick —
shows it is a *calibration* problem, with three measured components.

### Finding 1 — the "converged" close is a constant 8.9 cm world offset

Mean eef−object offset over all grasp-phase ticks, cream cheese (task 1),
across the band050 aim/hysteresis/seed atlas (40 runs, ≥1.4k grasp ticks each):

| arm family | dx (m) | dy (m) | dz (m) |
|---|---|---|---|
| aim V ∈ {0.55, 0.58, 0.60, 0.62, 0.65} | −0.088…−0.064 | +0.018…+0.076 | +0.02…+0.03 |
| aim U ∈ {0.45, 0.55} | −0.082 / −0.078 | +0.037 / +0.044 | +0.02 |
| hyst ∈ {0.50…0.70}, seeds s7–s41 | −0.063…−0.110 | +0.018…+0.054 | +0.02 |
| **pooled mean** | **−0.079** | **+0.040** | **+0.023** |

The offset is **invariant to the aim point**. Moving the aim UV by ±0.10 in
either axis — which at grasp height should displace the converged eef by
several centimetres — changes nothing. Together with the §5p dihedral null
(image error hovers ≥0.20 and no sign/swap mapping shrinks it), the reading is:
the servo never reaches image convergence at all. The grasp gate fires on the
**z-crossing** (`z < GRASP_Z`), and the eef lands wherever the approach
dynamics put it — a constant camera↔gripper lever arm of ~8.9 cm that no
image-space aim shift was ever going to remove. The entire aim-UV sweep of §5p
was sweeping a parameter with no control authority over the quantity it was
scored on.

### Finding 2 — closes happen 4 cm above the object

At close time eef z ≈ 0.045–0.050 m while the cream cheese sits at
obj z = 0.009 m. `GRASP_Z = 0.06` triggers the close as soon as the eef dips
below 6 cm — the fingers pinch air 4 cm above the box. The `deep` arms that
did descend to z ≈ 0.009 still failed **laterally** (Finding 1), which is why
"descend deeper" alone measured negative in §5q.

### Finding 3 — the air-close retry thrashes in place

Phase-transition trace (band050_aim60, trial 0): grasp entered at tick 76,
then close(12 real ticks) → air detected → **one** rise tick → re-enter grasp
at the same spot — 12 close/reopen cycles per episode, all at the same wrong
point. The retry logic was sound but its rise (1 tick ≈ 2 mm) could not
change the outcome of the next attempt.

### The fix this implies (implemented; eval `handeye_v1` running)

The detector's job ends at the gate — after that, the failure is pure
kinematics, and proprioception is the right sensor. `PhasedIBVS` gains a
calibrated **align** phase (`--ibvs-grasp-offset dx,dy`): at gate crossing it
fixes a world target `eef + (dx,dy)`, P-servos on proprio alone to within
1.5 cm, then descends to `--ibvs-close-z` (0.01, not 0.06) with a z-stall
contact check, closes with a downward press (`--ibvs-press`), and on an
air close rises for `--ibvs-retry-rise` ticks before re-acquiring.

Honesty notes, binding:
* `obj_pos` is used **offline only**, to calibrate the constant
  (+0.079, −0.040); the runtime controller sees proprio and detections, never
  sim state. This is standard hand-eye calibration, disclosed as such.
* This remains the **assisted/diagnostic** track (§5p honesty rules): it
  measures whether frozen-detector geometry + a constant extrinsic suffices
  for the full task under trivial control. It is not unaided policy
  competence, and the paper must not present it as such.
* Falsifiable prediction, logged before results: if the offset is truly a
  constant extrinsic, `handeye_v1` closes should land within the gripper span
  of the object and success should move off zero; if the offset is
  approach-dependent (variance −0.048…−0.169 across arms is not small), the
  align phase will re-create the miss and the next lever is per-approach
  calibration (offset as a function of approach direction), not more sweeps.

### handeye_v1 — single-attempt calibrated align (5 trials, 2026-08-01)

First run of the §5r fix, constant (+0.08, −0.04), one attempt per gate
crossing (the retry re-gate proved vision-dependent and never re-fired — the
detector is unreliable at table height, so after the one air close the episode
wandered). Success 0, but the near-miss statistics moved an order of magnitude:

| trial | eef_obj_dist_min | prior best (band050 atlas) |
|---|---|---|
| 0 | 0.024 | 0.068 |
| 1 | **0.019** | — |
| 2 | 0.042 | — |
| 3 | 0.136 (bad bind) | — |

Phase trace (trial 0): gate at dxy=(−0.109, +0.041) → align lands within
4 mm of its commanded target → descends to z = 0.009 (exactly object height;
the old machine closed at 0.045–0.050) → close → air → single retry never
re-gates. The residual −0.025 m miss is exactly the at-gate variance:
calibrating on first-gate-crossing episodes only (n=231 across the atlas)
gives offset mean (−0.080, +0.050), std (0.023, 0.016).

Calibration context (§5p): demonstrations close their gripper at
`eef_obj_dist` **0.040 m** (range 0.009–0.066). v1's closes at 0.019–0.042
are the first policy-side closes *inside the demonstrator's own commit band*.

Iteration v2 (running): offset corrected to (+0.08, −0.05); retries no longer
re-gate on vision but probe the stored world target along the high-variance
axis (dx ∈ {0, +2, −2, +4, −4, +6} cm), jaw feedback selecting the attempt
that holds. n=10 trials.

### handeye_v2 — probe retries (partial, run in flight)

Early trials: eef_obj_dist_min **0.009 / 0.004 / 0.041** — the probe walks the
gripper to *millimetres* from the object center (demo minimum: 0.026). The
positioning problem is solved outright. Success still 0: attempt-2 of trial 0
closed at dxy=(−0.010, +0.002) at exact object height and the jaw check still
read "air" (< HELD_JAW_MIN). The failure has moved past position into grasp
mechanics. Object geometry (collision box, `cream_cheese.xml`): a thin flat
box 8.1 × 4.3 × 1.8 cm lying flat — long axis at the panda jaw span (~8 cm),
height under 2 cm. Candidate mechanics: (a) closing axis aligned with the
8.1 cm axis (ungraspable; needs ~90° yaw — demos rotate only ~13° median,
which argues against); (b) watermelon-seed ejection: the downward press on a
1.8 cm slick box squirts it out during the close (object drifted ~1 cm across
attempts, consistent); (c) fingertip pads bottoming above the 1.8 cm box top.
One filmed episode (v2 flags, wrist camera) is being recorded to decide.

## 5s. Defect 29 — the jaw check was a constant (2026-08-01)

The v2 probe walked the gripper to **4 mm** from the object center (trial 1),
at exact object height, with the object between the fingers on any closing
axis — and the held-object check still said "air". The check is
`proprio[7:9].mean() >= HELD_JAW_MIN`: the mean of the two panda finger joint
positions. robosuite's panda fingers are MIRRORED joints — `gripper_qpos` is
`(+q, −q)` — so the signed mean is **identically zero in every jaw state**.
Verified against baked demo proprio (cream episode): open reads (+0.906,
−0.906) → mean 0.0001; mid-grasp *holding the box* reads (+0.552, −0.689) →
mean −0.069; the unsigned mean reads 0.91 open / **0.62 holding** — cleanly
separable around HELD_JAW_MIN = 0.2.

Consequences, in order of cost:

1. `PhasedIBVS` discarded **every** physically successful grasp one tick
   before "lift" — the machine reopened, rose, and retried; every "closed on
   air" observation in §5p–§5r's atlas is unfalsifiable through this check.
2. `GraspToolController` (`microvla/tools/grasp_tools.py`) had the same
   line — "tool-phase never closed grip / never lifted" (§5p) is this defect,
   not a control failure.
3. The taxonomy note: this is a **class-2 defect** (agreement on a wrong
   convention). The check agreed with itself everywhere, tests passed
   (mock proprio used same-sign jaw values — the mock encoded the author's
   misunderstanding), and no parity test could catch it because both sides
   consistently read zero. It fell to a *measurement*: jaw values logged per
   tick against a close the geometry said must succeed.

Fix: `abs()` before the mean, in both machines (one line each). Tests updated;
`handeye_v3` (fixed check, v2 config otherwise unchanged) is running.

### handeye_v3 — the first held grasp in project history (2026-08-01)

Same config as v2 plus the defect-29 fix. Trial 0, phase trace with the (new)
per-tick jaw telemetry:

```
t122 grasp    close at dxy=(-0.025, -0.010), z=0.009 ... jaws -> 0.53  (HELD)
t146 lift     object rises WITH the eef: obj z 0.009 -> 0.312 -> 0.459
t206 servo_tgt  76 ticks, tgt_conf 0.00 almost throughout (basket unseen)
t282 release  spurious target fix (conf 0.15) fires the err<0.12 gate
t298 done     object back on the table at (+0.033, -0.116) — pickup point
```

Trials 0-1: grip_close_rate 0.40 / 0.375 (held through lift + traverse),
eef_obj_dist_final 0.469 / 0.450 (object carried aloft half a metre). The
pick leg is SOLVED — grasp, hold, lift, all first-evers. The place leg now
fails for the same structural reason the pick did: the wrist camera almost
never sees the basket at altitude (tgt duty ~0), so the visual place gate
waits on a signal that does not come, then trips on noise.

Place calibration, measured from the 50 cream demos: the basket sits at a
FIXED world xy = (−0.005, +0.257), std (0.023, 0.015) — the task randomizes
objects, not the basket. v4 adds `--ibvs-place-at` / `--ibvs-drop-z`:
proprio-only traverse at altitude to the calibrated point (jaw-drop watchdog
en route), lower to drop_z=0.18 while holding, then open. Vision's remaining
role in the whole pipeline: find the object once, at altitude, where the
detector is actually good.

### handeye_v4 — SUCCESS. The first completed tasks in project history (2026-08-01)

Full calibrated pipeline (visual gate → proprio pick with probe retries →
held lift → proprio traverse to the demo-calibrated basket point → lowered
drop):

```
trial 0: success=True  steps=298  grip_close_rate=0.557
trial 1: success=True  steps=319  grip_close_rate=0.520
```

Two-for-two at time of logging (n=10 run in flight). The zero that has
headlined every eval since the harness existed is broken.

## 5t. Weight forensics — what 16.6M trained parameters actually learned (2026-08-01)

Instruments: `paper/weight_forensics.py` (static, per-tensor),
`paper/dynamic_forensics.py` (live-loop probes on mock inputs),
`paper/render_ledger.py`. Full machine-generated evidence: **554 numbered
findings** in `paper/forensics_ledger.md` (211 tensor censuses, 67 spectral
workups, neuron/quantization/delta tables), 24 figures in `paper/visuals/`.
Checkpoints: `full_stageB_rec_fix.pt` (deployed) vs `full_stageA_v8_s0.pt`
(cross-run reference). Curated readings below; every number traces to the
ledger.

### T1. The planner's real input diet: one channel out of nine (D-004)

Deployment-path ablation (zero one planner input inside the live loop, 16
deterministic mock ticks each; mean |Δaction|):

| channel | impact | | channel | impact |
|---|---|---|---|---|
| **relational** | **0.3615** | | wm_msg | 0.0008 |
| current_emb | 0.0120 | | fused | 0.0000 |
| proprio | 0.0024 | | spatial | 0.0000 |
| state_delta | 0.0011 | | pred_box_emb | 0.0000 |
| | | | geometry | 0.0000 |

~97% of the plan's dependence flows through the RelationalHead's tokens;
`fused`, `spatial`, `pred_box_emb`, and `geometry` are **dead inputs** — the
planner learned to ignore them outright (`visuals/plan_sensitivity.png`).
Three consequences: (a) the v8 bet ("object-object reasoning conditions on
the TRM latent") is confirmed in the weights — the relational path is not an
auxiliary, it IS the perception-action interface; (b) any future grounding
improvement only moves behavior if it reaches the relational tokens — tuning
the other channels is provably wasted; (c) the dead projections and their
upstream compute are prunable for the Pi build.

### T2. The learned control-gain head is all zeros (F-003)

`drift.hrm.gain_head.weight` is **100% near-zero** (n=768) — the HRM's
"learned per-axis control gains" never trained. The architecture's designated
mechanism for scaling action magnitude is inert, which closes a loop with the
oldest open defect in the study: §4p measured every policy emitting 2–4×
under-magnitude actions, and D-006 reproduces it live (mean |pose action|
0.056 on mock inputs). The magnitude shrink isn't fought by the gain head —
the gain head is dead weight. Either its gradient path is broken (zero-init
plus a multiplicative position that never lets gradient through) or its LR
never moved it; both are checkable in one training probe. This is the single
highest-leverage training bug the weights point to.

### T3. Dreaming is intrinsically stable in the trained weights (D-001/2/3)

Closed 30-step dream rollouts (evidence held, prediction fed back,
re-standardized): per-step update norm settles at 0.56, distance-from-start
plateaus, and five rollouts from independent starts *converge into a shared
attractor basin* (final separation 0.42× initial;
`visuals/trm_dream_pca.png`). The TRM Jacobian's leading singular value at an
operating point is 1.000 — the residual delta is a small perturbation around
identity. The 15:1 dream schedule is therefore stabilized by the weights
themselves, not only by the InnovationCorrector — direct weight-level
evidence for the JEPA design claim (E6's ablation now has a mechanism).

### T4. Text conditioning is ~28-dimensional (F-012)

`tqsa.t_proj.weight` [128×512] carries an effective rank of 28.5 (22%) — the
lowest in the stack. The corpus contains a handful of distinct task phrases,
and the text pathway compressed to match: language conditioning in this
regime is a task-ID lookup, not compositional grounding. A 4× smaller
projection would lose nothing on this corpus; generalizing to unseen phrases
will need the rank forced up (dropout on text tokens, more tasks, or both).

### T5. Training health: heavy-tail band, no dead neurons, linear-regime tanh

52/67 weight matrices sit in the Hill-exponent 2–6 band associated with
converged training (F-030); the worst neuron-utilization offender in the
whole stack is 1 weak row out of 256 (F-031) — no dead capacity anywhere.
The live activation probe (D-005, 12 nonlinearity sites) finds the plan
tanh operating in its linear regime — so the §4p magnitude shrink is an
upstream pre-activation scale problem (see T2), not saturation clipping.
The three attention stacks (planner, relational, drift) all plateau at
58–60% effective rank in their out-projections — healthy, uncollapsed heads
with a consistent architecture-wide signature (`visuals/spectra_*.png`).

### T6. Quantization map for the Pi build (F-033)

Median symmetric per-tensor int8 relative error is <1% and the worst layers
(`trm.pos`, `trm.net.chan_mlp.2`, planner embeddings ~0.9–1.0%) are embedding
/positional tables, not matmuls; every LayerNorm gain is a 100–300σ outlier
tensor (F-004..F-011) that must be kept fp16 or per-channel-quantized.
Concretely: int8 the matmuls, fp16 the norms and embeddings, and the whole
9.97M-param TRM survives with sub-percent weight distortion
(`visuals/quant_error.png`).

### T7. Lineage note (F-032)

rec_fix vs v8_s0 differ by 0.28–0.43 relative Frobenius in *every* module —
they are different training runs, not a frozen stage-A/B pair; no continuity
claim may be made between these two files. (The stage-B-freezes-the-WM claim
must be verified against rec_fix's own stage-A file on the pod if needed.)

### T8. Compression corollary

Summing the spectral findings: the deployed 16.6M trainable params carry
substantial low-rank slack (att out-projections ~58%, chan_mlp 53%,
state_proj 55%, text projections 22–46%). A rank-truncation pass at 90%
spectral energy plus the T1 dead-input pruning plausibly halves the
trainable footprint before any distillation — a measured, not aspirational,
path to a ~4M-trainable stack for the Pi. Each candidate layer is
individually listed with its effective rank in ledger §L2.

### handeye_v4 — final: mean_success 0.200, n=10 (2026-08-01)

| trial | succ | steps | grip_close | eef_min | read |
|---|---|---|---|---|---|
| 0 | **True** | 298 | 0.557 | 0.023 | clean pick → traverse → basket |
| 1 | **True** | 319 | 0.520 | 0.018 | clean pick → traverse → basket |
| 2 | False | 400 | 0.205 | 0.042 | probe exhausted (variance tail) |
| 3 | False | 400 | 0.110 | 0.133 | bad bind (gate on wrong object) |
| 4 | False | 400 | 0.495 | 0.031 | grasped late, timed out mid-task* |
| 5 | False | 400 | 0.165 | 0.092 | probe exhausted* |
| 6 | False | 400 | 0.110 | 0.128 | bad bind* |
| 7 | False | 400 | 0.080 | 0.072 | probe exhausted |
| 8 | False | 400 | 0.055 | 0.154 | bad bind |
| 9 | False | 400 | 0.440 | **0.014** | holding object at step 400, timed out |

*Trials 4–6 ran at 4–5× wall-clock (406–555 s vs 78 s) — the pod was
simultaneously running a video render and a second eval launched by this
session; the policy's real-tick cadence is wall-clock-independent but the
contention is noted for honesty.

Aggregates: mean_success **0.200**, src_detect 0.91, eef_min mean 0.071.
Failure taxonomy: 3 bad binds (binding identity — the §5p residual), 3
probe-exhausted (calibration variance tail beyond ±6 cm), 2 timed out while
executing correctly (max-steps 400 too tight for multi-attempt episodes —
raised to 600 for subsequent runs). Two trials that ended holding the object
(4, 9) would plausibly convert with the longer budget: the underlying
grasp-competence rate this run measures is 4/10 with 2 completed placements.

### soup_v1 — the constants generalize: mean_success 0.750, first attempt (2026-08-01)

Task 0 (alphabet soup → basket), n=8, zero soup-specific tuning — every
constant transferred from demo statistics or the shared calibration:
grasp offset (+0.09, −0.186) (at-gate telemetry under the cream aim),
close_z 0.045 (soup demo close-height p10), gate_z 0.10 / approach_z 0.12
(can geometry: fly the align above the can, descend around it), basket
(−0.006, +0.260) (soup demos), drop_z 0.25.

**6/8 success.** Successful grasps land 6–7 mm from the can axis
(eef_obj_dist_min 0.006/0.007 on trials 6/7). Aggregate detection duty
0.93. A different object class, different height regime (standing cylinder,
grasp at 4.5 cm vs 1 cm), different approach profile — same machine, same
lever-arm philosophy. The camera-gripper extrinsic is confirmed
object-independent; only object-geometry constants (close/gate/approach
heights) changed, all read off demo replays, none tuned on eval.

Cream remains the harder pick (thin flat box, 0.200): its dominant failure
is bind identity, now attacked by gate-verify (below). Dressing (tall
bottle) v1 in flight — trial 0 never approached (grip 0.000): the bottle
regime needs its own diagnosis pass.

### dressing_v1 — 0/8, and the cleanest failure signature yet (2026-08-01)

Task 2 (salad dressing → basket), first attempt with bottle-geometry
constants. src_detect_rate a PERFECT 1.000 on all 8 trials, grip_close
0.000, eef_min ~0.25, final distance ~0.6 m. Telemetry: all 600 ticks in
`servo_src`; z RISES monotonically 0.26 → 0.71.

Root cause (one line): the machine's `conf_floor` default (0.10) sits above
the bottle's typical detection confidence (0.03–0.09). Every fix reads as
"missing"; the lost-source recovery is *rise to widen the view*; rising
shrinks the object and the loop spirals upward. The detector never missed —
the CONSUMER's threshold discarded a perfect signal. (Class-2 flavor again:
each side is locally correct.) Fix queued as dressing_v2:
`--ibvs-conf-floor 0.05`, everything else unchanged; running behind
handeye_v5cream (winning cream config + `--ibvs-gate-verify`, the
gate-time CLIP veto aimed at v4's 3/10 wrong-bind failures).

Scoreboard at this point (assisted/calibrated track, honest):
cream 0.200 (n=10) · soup 0.750 (n=8) · dressing 0.000 (n=8, one-line fix
identified) · project prior: 0.000 everywhere, 347 evals.

### handeye_v5cream — gate-verify: 0.300 (2026-08-01)

Winning v4 config + `--ibvs-gate-verify` + max-steps 600, n=10:
**mean_success 0.300** (up from 0.200). The mechanism worked as designed and
exposed its own next step: wrong-bind grasps became SAFE NO-GRASP vetoes
(three episodes end grip_close 0.000, hovering, never committing to the
wrong object), and the longer budget converted two previously-timed-out
correct episodes (successes at 446/445 steps). But a veto that never
redirects is half a fix — v6 escalation implemented: after 3 vetoes the
machine flips on the existing clip-rerank path for the rest of the episode,
so the servo starts chasing the proposal that actually matches the source
phrase (per-episode only; reset restores). Tests pin the escalation and its
reset.

### dressing_v2 → the detector binds the robot's own FINGER as the bottle (2026-08-01)

Conf-floor 0.05 fixed the altitude spiral (z now reaches 0.012) but 0/8
persists with a new, cleaner signature: the "salad dressing" fix sits at
image v = 0.80 with u flipping 0.13 ↔ 0.87 at conf 0.15–0.19, duty 1.000.
Those coordinates are the wrist camera's OWN FINGER TABS (left/right). The
servo chased its own hand for 600 ticks; the aim band (err vs V=0.24 is
0.56 > 0.50) correctly refused to descend-commit, which is why nothing
worse happened. Perfect detection duty was the tell: a real object cannot
be in frame 100.0% of ticks from a moving camera — a self-attached feature
can. (Class-2 again: the metric said perfect; perfect was the bug.)

Fix: a **self-body mask** — `--ibvs-body-v 0.72`, fixes with center v
beyond the line are treated as missing (`_believe`). The finger strip is a
camera invariant, so this is a constant, not a tuned threshold. Chain
running: cream v6 (gate-verify + veto→rerank escalation + body mask) and
dressing v3 (conf 0.05 + body mask).

### handeye_v6cream — full-width body mask: 0.000, an instructive regression (2026-08-01)

The v0.72 full-width mask fixed the finger-bind and BROKE cream: the real
box legitimately occupies bottom-center of the wrist frame during descent
(aim V 0.60), so the mask discarded true fixes mid-approach and the
lost-source recovery rose forever (10× grip 0.000, min dist 0.29, conf
0.28–0.41 because the whole scene is visible from the altitude it retreated
to). Correction shipped: the mask now requires BOTH v > 0.72 AND u inside a
corner band (u<0.28 or u>0.72) — the finger tabs are corner-localized
(u 0.13/0.87), real objects during descent are bottom-CENTER. One more
class-2 lesson for the ledger: a mask defined by one task's failure can be
another task's blindfold; the corner constraint is the camera-invariant
part. Queue: dressing v3 (full-width mask is safe there — the bottle rides
top-of-frame), then cream v6b + soup v2 with the corner mask for final
scoreboard numbers.

### dressing_v3 — 0/8; three defects peeled, at least one remains (2026-08-01)

Full-width body mask removed the finger bind (conf_mean fell 0.148 → 0.078:
the high-conf self-fixes are gone, the residue is real low-conf bottle
detection) — and the approach STILL never engages (grip 0.000, min 0.25,
final 0.70). The bottle task has now consumed three targeted fixes, each
killing a confirmed defect (consumer conf-floor above signal; finger
self-bind; mask over-reach on other tasks), without breaking the surface
symptom. Honest status: dressing = 0.000 with an unidentified residual;
next instrument is a filmed episode + per-tick believed-vs-raw fix logging,
not another blind threshold. Parked pending the cream/soup scoreboard runs.

### handeye_v6b + the finger-bind unification — masks are whack-a-mole (2026-08-01)

v6b (corner mask) trace, trial 1: a conf **0.36–0.54** source fix at
[0.72–0.73, 0.69–0.74] — the FINGER again, now inboard because its image
position moves with height (corners at altitude, u≈0.75 near the table).
The corner band sits exactly on that boundary: the machine flickers between
"masked → missing → rise" and "unmasked → chase the finger", and never
approaches (0/7 at diagnosis, grip 0.000, min 0.24). Retroactive
unification: v4's "3/10 bad binds" and dressing's perfect-duty bind are the
SAME defect — the detector binds the robot's own gripper, at confidences up
to 5× the real object's. Positional masks cannot fix a height-dependent
image position; the principled discriminator is temporal: a fix that stays
put in the image while the arm moves is self-attached. That filter belongs
in the PERCEPTION layer (role binding), which returns this campaign to the
paper's central thread.

### FINAL SCOREBOARD — assisted/calibrated track (2026-08-01)

| task | best run | mean_success | n | prior (347 evals) |
|---|---|---|---|---|
| cream cheese (t1) | v5: align+probe+jaw fix+gate-verify | **0.300** | 10 | 0.000 |
| alphabet soup (t0) | v1: per-object constants, no verify | **0.750** | 8 | 0.000 |
| salad dressing (t2) | v1–v3 | 0.000 (finger-bind residual) | 8×3 | 0.000 |

Config provenance and per-trial tables above. The masks/escalation arm
(v6/v6b) measured NEGATIVE on cream and is reverted from the recommended
config; gate-verify (+0.10 on cream) stays.

### soup_v2 — 0/9 with mask+verify: the additions regress soup too (2026-08-01)

Soup with corner mask + gate-verify collapsed 0.750 → 0/9. Combined with the
cream v6/v6b regressions this settles the config question: the RECOMMENDED
configs are exactly soup_v1 (no verify, no mask) and cream_v5 (gate-verify
only, no mask). Gate-verify helps only where wrong-binds dominated (cream);
on soup it vetoes/perturbs a binding that was already good enough for 0.75.
Teacher-data recording (UNAIDED_PLAN Phase B) launched with the per-task
winning flags accordingly.

### unaided_v1 (distilled, 22-episode corpus) — hover signature, 0/3 before a pod blip (2026-08-02)

First unaided eval of `full_stageB_teacher_bc.pt` (teacher-BC from 22 soup +
1 cream episodes; val bc 0.0448 — HALF the best any demo-trained arm ever
reached — with val grip agreement 1.000). Closed loop, no assist flags:
three completed soup trials all show the same signature — detect duty 0.98+,
grip 0.00–0.02, eef_min ~0.24, never descends. The distillation transferred
gripper discipline and hover stability but NOT the visually-triggered
descend-commit; with 22 episodes the approach phase (the only part that
varies meaningfully across inits) is under-sampled. Standard imitation
scaling applies. Overnight: n=10 honest rerun (unaided_v1b) + teacher
recording scaled to ~100 soup / ~30 cream episodes on fresh init ranges
(50+/60+), then retrain and re-eval. The teacher costs ~100 s per success;
data is the cheap axis on this stack.

### unaided_v2 (distilled, 100-episode corpus) — the stall moves 10 cm closer, and the defect is now quantified (2026-08-02)

Round-2 distillation: 100 successful soup teacher episodes (init indices
50–149+, disjoint from eval trials 0–19; ~90% teacher hit rate during
recording), converted through the standard two-pass shard pipeline (fresh
`norm_stats` paired to the checkpoint), stage-B BC from `rec_fix` (TRM
frozen, 24 epochs, val bc 0.0445, val grip agreement 0.998). First run was
host-killed at 5/10 trials; all five completed trials failed with one
consistent signature:

| trial | steps | eef_min | eef@20 | eef_final | grip close rate |
|---|---|---|---|---|---|
| 0 | 600 | 0.159 | 0.276 | 0.713 | 0.150 |
| 1 | 600 | 0.169 | 0.290 | 0.598 | 0.203 |
| 2 | 600 | 0.179 | 0.297 | 1.174 | 0.257 |
| 3 | 600 | 0.155 | 0.291 | 0.642 | 0.113 |
| 4 | 600 | 0.198 | 0.304 | 0.447 | 0.018 |

Read against round 1 (eef_min ~0.24, grip ~0.0): the 4× data scale-up
bought a real approach phase (0.29 → 0.16 m) and intermittent grip
commitment. The policy now walks most of the teacher's path and stalls
~16 cm short of the object.

**The stall is quantified, not mysterious.** Comparing the student's live
telemetry against the teacher corpus statistics:

- Teacher (row-0 actions, raw units): mean |x|,|y| = 0.094, 0.207;
  p90 saturates the ±0.6 quantile clip. Mean |z| ≈ 0.22.
- Student (3 000 live ticks): mean |x|,|y| = **0.025, 0.025** — a 4–8×
  lateral undershoot. p95 = 0.06. The z axis is nearly calibrated
  (0.14 vs 0.22).

Two causes, both previously flagged:

1. **Partial observability of the teacher's target.** The teacher's align
   phase steers to `_base_tgt` — an internal stored coordinate, invisible
   in (frame, proprio) at the tick level, and the detector is unreliable at
   table height precisely where align runs. BC's conditional mean over
   episodes with different hidden targets regresses lateral commands toward
   zero: classic mean-collapse under unobserved conditioning. The z axis
   survives because "descend" is nearly unconditional in the corpus.
2. **`gain_head` never trains** (forensics F-003). Verified again on
   `teacher_bc2`: `drift.hrm.gain_head.weight` is bit-identical zero — the
   stage-B resume freezes the drift module, so the learned action-magnitude
   mechanism has never fired in any checkpoint to date.

Diagnostic in flight (`diag_gain3`, labeled diagnostic-only, NOT a
headline config): the same checkpoint with a global 3× action gain. If
lateral scale is the binding constraint it should reach the object; if the
stall is covariate shift it will fail the same way, and the next round is
DAgger (teacher labels on student-visited states) rather than magnitude
calibration.

### unaided_v3 (DAgger round, dagger-only training) — 0/10, but the stall is BROKEN (2026-08-02)

Round 3: 40 DAgger episodes (student `teacher_bc2` drives 70% of ticks,
calibrated soup teacher labels every state, β=0.3 recovery mixing; inits
50–89; all 40 episodes fail to complete — expected, the student is the
staller — label sanity verified: teacher-label mean |xyz| =
0.103/0.193/0.268 raw units, matching the round-2 teacher distribution;
teacher-driven tick fraction 0.294 ≈ β). Trained `teacher_bc3` on the
DAgger corpus ALONE (24 epochs, val bc 0.0433) plus the new magnitude
losses (`train/losses.py::pose_magnitude_loss`, weight 0.8;
`gain_magnitude_loss` 0.3) targeting the measured 4–8× lateral undershoot.

unaided_v4 eval, n=10, no assist flags: **mean_success 0.000** — but the
intermediates tell a different story than every previous zero:

| metric | rec_fix | bc (23 eps) | bc2 (100 eps) | **bc3 (DAgger)** |
|---|---|---|---|---|
| eef_obj_dist_min | — (drifts) | ~0.24 | 0.155–0.198 | **0.061** |
| eef_obj_dist_final | ~1.0 | — | 0.45–1.17 | **0.110** |
| grip_close_rate | ~0 | 0.00 | 0.02–0.26 | **0.000** |
| src_detect_rate | — | 0.98 | 0.53–0.78 | 0.92 |

The approach defect is FIXED: the policy now closes to 6 cm and STAYS at
the object (final 0.11 m vs 0.6–1.2 before — no more drift-away). But
grip_close_rate is exactly 0.000: training on the DAgger corpus alone —
whose labels are ~always gripper-open because the mixed rollouts never
reach the grasp phase (episode grip-label mean −0.976) — unlearned
closing entirely. This failure was predicted BEFORE the run (see
POD_COORDINATION.md note, written at episode 12/40) and is standard
DAgger theory: train on the AGGREGATE of all rounds, not the newest
round. The dagger-only run turned into a clean ablation that isolates
the two skills: approach lives in the DAgger data, grasp lives in the
teacher-success data.

Round 4 (in flight): `teacher_bc4` on the aggregate (100 teacher-success
+ 40 DAgger episodes), same magnitude losses → unaided_v4, n=10.

### unaided_v4 (aggregate + magnitude losses) — 0/7 (pod restart), the residual isolates to the GRASP TRIGGER (2026-08-02)

`teacher_bc4`: aggregate corpus (100 teacher successes + 40 DAgger) with
`pose_magnitude_loss` 0.8 / `gain_magnitude_loss` 0.3. Eval killed at 7/10
by a pod restart; all seven completed trials fail one way:

| trial | eef_min | eef_final | grip close rate |
|---|---|---|---|
| 0 | 0.135 | 0.142 | 0.003 |
| 1 | 0.078 | 0.141 | 0.013 |
| 2 | 0.128 | 0.218 | 0.018 |
| 3 | 0.104 | 0.148 | 0.007 |
| 4 | 0.146 | 0.284 | 0.003 |
| 5 | 0.116 | 0.167 | 0.017 |
| 6 | 0.143 | 0.263 | 0.007 |

Read against the ladder: approach is retained from the DAgger round
(min 8–15 cm and the policy STAYS — final 0.14–0.28 m, no drift-away),
magnitude is healthy, but grip commitment collapsed back to ~1% of ticks
(bc2 had 2–26%). The aggregate diluted the DAgger set's all-open labels,
yet the close event remains ~5% of corpus ticks — a class-imbalance
problem on the grasp *trigger*, not a geometry problem. The skill ladder
after four rounds: descend ✓ (round 2), lateral magnitude ✓ (round 3),
stay-on-target ✓ (round 3), close-at-the-right-instant ✗ — the last
locked door. Round 5 (parallel session, rev eccbfea): `teacher_bc5b` =
2× teacher-success corpus weight + fresh DAgger set + GRAM heads on
HRM/planner, from bc3.

Watch-item: the same session's `ceiling_ibvs3` assisted re-check opened
0/2 with grips firing 31% — if the TEACHER degraded on the restarted pod,
ceiling comparisons after this point need re-baselining.

### unaided_lora1 (LoRA'd embedding, BC objective) — 0/6 partial, user-stopped (2026-08-03)

First run with trainable perception: rank-8 LoRA on a CLONE of the SPPF
stage (18.4K adapter params @ 1e-3) + base SPPF weights @ 1e-6, detection
bit-frozen (microvla/perception/lora.py; frame_embs recomputed per batch
from cached SPPF inputs; val scored in the adapted space). Train: 24
epochs, val bc 0.0755 (vs 0.043 frozen — not comparable: the space is
moving under the policy; still descending at cutoff). Eval stopped at 6
trials by user call to prioritize the phase-loss round: all fails; best
trial eef_min 0.127 m, grip 10–18%, holds position (final == min on
trial 2). Read: LoRA-alone under the SAME mimicry objective reproduces the
bc4-class behavior — input capacity without an objective that rewards
decisive approach does not break the stall. Consistent with the loss
being the binding constraint, which is the phase-progress bet
(teacher_phase1, in flight: exact-coord BC ×0.2 anchor +
direction-to-grasp-point + magnitude floor + grip timing windows,
train/losses.py::phase_progress_loss).

Films: watch_videos/unaided_bc5c (bc5c regression: parks over empty table
near basket, wrist camera starved; wrong-object descend onto a distractor
carton on task 6) — the two failure modes the phase objective + LoRA
target at mechanism level.

### unaided_phase1 (LoRA + phase-progress objective) — 0/10; the free-regression ladder closes at six rungs (2026-08-03)

teacher_phase1 = stage-B from rec_fix, aggregate corpus (teacher_grid2 +
dagger), YOLO LoRA (r8, lora lr 1e-3, base 1e-6) + the phase-progress loss
(`train/losses.py::phase_progress_loss`: direction-to-grasp cosine +
magnitude floor + close/open BCE windows; exact-coord BC demoted to 0.2×).
`eval_results/unaided_goal1`-style flags, n=10 task 0: **mean_success
0.000**. Six free-regression variants (bc2, bc3, bc4, bc5b, lora1, phase1)
have now attacked the same zero with capacity, data aggregation, DAgger,
input adaptation, and objective redesign — the ladder's residual (goal
persistence + trigger timing + magnitude under noise) is invariant to all
of them. This is the measured motivation for the v10 structured redesign
(below), not a shortfall to iterate past with a seventh variant.

### v10 structured control — the seven mechanisms become architecture (2026-08-03)

Directed redesign ("redesign the architecture to directly mirror these
facts"): `paper/WHY_THE_TEACHER_WORKS.md` § prescriptions implemented as
`microvla/control/` (DESIGN.md v10). The policy class changes — the network
no longer emits per-tick actions at all:

* **Learned (task content):** `GraspPointHead` (~0.17M) regresses the world
  grasp point from (source box uv/conf/emb, frame emb, proprio), labels =
  eef at the FINAL close onset of teacher episodes (the lever arm is in the
  mapping by construction); `PlaceHead` (~0.07M) regresses the basket point
  from the command embedding. Heteroscedastic sigma via a detached-trunk
  lv-head (joint-NLL collapse measured and designed out: mean stalled at
  ~5 cm while sigma inflated; Huber-mean + detached-residual NLL fixed it).
* **Structure (control):** `GoalServoMachine` — latch (evidence admission,
  sigma-gated, one-way), P-law `clip(12·(goal−eef), 0.6)`, one-way phases
  with the probe-retry cycle, abs() jaw hold check, proprio-only place.
  Calibrated arm constants keep their teacher values; no task content.

Offline go/no-go on the existing corpus (111 usable episodes, 1703 grasp
ticks, no re-recording needed): **val median xy error 1.30 cm, p90 2.62 cm
by epoch 100** (bar: ≤3 cm; probe envelope ±6 cm). Chain in flight:
train → `eval_results/unaided_goal1` (n=10, task 0, NO assist flags) →
films. The unaided claim under test: goals from trained heads + structure
≈ teacher's 0.75, with failures isolating to a measurable head.

### unaided_goal1 — FIRST UNAIDED SUCCESS: mean_success 0.100, n=10 (2026-08-03)

`eval_results/unaided_goal1`, task 0 (soup), NO assist flags — the first
nonzero unaided number in project history, on the v10 structured policy's
first closed-loop attempt. Aggregates: src_detect 0.968, grip_close_rate
0.221, eef_obj_min 0.041 m. The success (trial 3) is textbook: latch error
2.3 cm, attempt 0, approach→descend→grasp→lift→transport→release→done in
282 ticks — the teacher's signature, executed from learned goals.

Per-trial phase forensics (telemetry logs phase/attempt/base_tgt per tick,
with ground-truth obj_pos for diagnosis): EVERY trial latches, descends and
closes — no never-latched or parked failures, the free-regression ladder's
signature modes are gone. The 9 failures isolate to exactly two named
defects:

1. **Deployed latch error 2.2–5.3 cm** (median ~3.4) vs 1.27 cm offline —
   the latch admits hover-altitude estimates; the head's low-altitude band
   (median 1.23 cm) is never exploited because the goal freezes at latch.
2. **The probe search was x-only** (teacher's table; ITS error source was
   x-distributed) while the head's error is isotropic: trials 2/5/6/7/8
   burned 9–12 attempts re-closing at the same wrong y. A 3 cm y-error was
   unreachable BY CONSTRUCTION. Three trials also lift→drop (edge grasps
   from wrong-spot closes, same root cause).

Fixes (v10.1, both structural, no retraining): radius-ordered 2D probe
table (±2/±4/±6 cm over both axes, 15 entries), and first-descent goal
REFINEMENT — on attempt 0, confident estimates keep correcting the stored
goal until the eef crosses z_freeze 0.10 m (one-way), so the machine
consults vision exactly as long as the teacher did (down to its gate
height) and probes proprio-only after. `unaided_goal2` in flight; the y-
offset recovery and refinement-freeze contracts are pinned by unit tests
(tests/test_goal_control.py, 23 passing).

### unaided_goal2 (v10.1: 2D probe + first-descent refinement) — 0.300, n=10; every mechanism now validated individually (2026-08-03)

mean_success **0.300** (3× goal1), eef_obj_final 0.104→0.066. The three
successes exhibit the three designed mechanisms separately, on the record:
trial 0 = refinement (goal error **1.2 cm** at grasp, attempt 0, 271
ticks); trials 4/5 = 2D probe recovery (goal error 3.8–4.0 cm, converted
on attempt 2). The teacher-signature run shape
(approach→descend→grasp→lift→transport→release→done, ~270–330 ticks)
now appears in every success.

The 7 failures decompose into two NEW signatures, both structural, both
self-inflicted, both fixed as v10.2 without retraining:

1. **Refinement deadlock** (trials 2/6/7, stuck approach→descend, eo_min
   ~7 cm): raw per-tick refinement jitters the stored goal; the strict
   1.5 cm align gate then never opens and the arm hovers above z_freeze
   indefinitely. This is the descend-hyst deadlock RE-LEARNED: v10.2
   EMA-blends refinements (0.5) and descends inside a wide band
   (descend_band 0.05) while unfrozen, strict tolerance reserved for the
   grasp transition; freeze also fires on a tick budget (150) as backstop.
2. **Probe exhaustion around a bad latch** (trials 1/8/9, attempts 5–10 at
   goal error 2.9–4.5 cm): re-probing a mislatched target forever. v10.2:
   after `probe_restart` (8) failed attempts, FULL unlatch and re-approach
   — a fresh vantage + refinement beats searching around a bad anchor.

Ladder so far: free-regression 0/10 ×6 → v10 0.100 → v10.1 0.300 → v10.2
in flight (`unaided_goal3`). Contracts pinned by 25 unit tests.

### unaided_goal3 (v10.2: EMA refinement, wide band, probe-restart) — 0.400, n=10; failure mass reaches the LAST stage (2026-08-03)

mean_success **0.400** (ladder: 0.100 → 0.300 → 0.400), grip_close_rate
0.221→0.367, eef_obj_min 0.029. All four successes are attempt-2 probe
recoveries (goal error 2.4–3.2 cm), ~310–330 ticks, full teacher-signature
runs. Both goal2 defects converted (no refinement deadlocks except one
boundary case below; restarts fire and re-latch — trial 4 restarted twice).

The remaining six failures have moved DOWNSTREAM to the final stages —
grasp geometry and the drop — plus one gate defect:

* **Edge grasps** (trials 2/6/9, the "done-but-scored-0" signature): the
  full cycle completes, the object is held SOLIDLY through 63–64 transport
  ticks (jaw 0.46–0.63), released over the learned basket point — but the
  can hangs 2.6–4.5 cm off gripper center (closed on its edge), so it
  lands on the rim (obj final z ≈ 0.17) or slips during lowering (trial 9,
  floor). The eef is exactly where it should be; the OBJECT is not.
* **Align-gate chatter** (trial 1): lateral error oscillating at exactly
  the 0.015 gate flicked descend on/off; each steer tick reset the contact
  window, so the machine sat ON the object at z 0.041–0.044 with contact
  detection starved for 400+ ticks — measured in the action trace as
  alternating [−0.4, 0.0] z-commands.
* Transport drops with working restarts, out of clock (trials 4/7).

v10.3 (no retraining): alignment HYSTERESIS (enter 0.015 / exit 0.025;
contact evidence accumulates through boundary chatter) and drop_z
0.18→0.12 (release below the rim, into the liner). Contracts pinned by 27
unit tests. `unaided_goal4` in flight. The residual after v10.3 —
grasp-centering (the 2–4 cm goal-error tail) — is the LoRA'd-embedding
goal head's job, the next sanctioned lever.

### unaided_goal4 (v10.3: align hysteresis + drop_z 0.12) — 0.200, n=10; the hang offset is a CONSTANT (2026-08-03)

mean_success **0.200** (2/10) — nominally a regression from goal3's 0.400,
but the forensics dissolve the noise into one number: **8/10 trials now
complete grasp+carry+release over the learned basket point** (front end
essentially solved; goal3 had 6). The successes landed IN the liner (obj
final z 0.117/0.128 — drop_z 0.12 doing its job); the failures rest ON the
rim (z 0.163–0.173) or slipped during lowering (one, floor). Align
hysteresis removed the goal3 trial-1 contact-starvation signature (no
stuck-on-object trials).

The decisive measurement: fitting (obj − eef) during transport across ALL
goal3+goal4 telemetry (390 detected ticks) — the held object hangs at a
**near-constant (−2.8, +1.4) cm from the eef, residual 3.3 mm**, with the
image-space terms carrying ~zero signal (no variance: the can always hangs
the same way). The "random bounce" read was wrong; the grasp pipeline has
a SYSTEMATIC bias (goal-head deployment shift + the probe schedule's
geometry), and success-vs-rim in goal3/goal4 was decided by which side of
the basket wall that fixed offset landed on.

v10.4: `hang_comp` — the traverse aims the eef at place − hang so the
OBJECT arrives over the basket center. Honest bookkeeping: this is a
2-parameter place-side hand-eye constant calibrated OFFLINE from logged
rollouts (the same category as the P-gains, and the same method that
produced the teacher's lever arm — logged as such, and kept OUT of the
"zero calibrated constants" claim). The principled fix that drives it to
zero — a more accurate goal head (LoRA'd embedding) — is the sanctioned
next lever; goal3's 0.400 stands as the best *constant-free* number.
`unaided_goal5` (hang_comp + per-success wrist videos via
`--success-video-dir`, now standing) in flight.

### unaided_goal5 (v10.4: hang_comp place compensation) — 0.700, n=10 (2026-08-03)

mean_success **0.700** — seven full unaided pick-and-places, every one
auto-filmed from the wrist (`--success-video-dir`, standing). Trajectory:
free-regression 0.000 ×6 → v10 0.100 → v10.1 0.300 → v10.2 0.400 →
(v10.3 diagnostic round 0.200: exposed the constant hang) → v10.4
**0.700** — statistically indistinguishable from the assisted teacher's
0.75 (n=8). grip_close_rate 0.49, eef_obj_min 0.030.

Tier bookkeeping (leaderboard): goal5 sits on the T1 board — it carries
ONE offline-calibrated task-adjacent constant (hang_comp (−2.8, +1.4) cm,
fitted from logged rollouts). The T0 (zero calibrated constants) leader
remains goal3's 0.400.

**The path from T1 0.7 to T0 0.7 is already recording** (selfplay_night
chain): in self-play SUCCESS episodes recorded WITH hang_comp, the eef at
release equals place − hang and the object verifiably landed in the basket
— so retraining PlaceHead on self-play release positions absorbs the hang
compensation into the learned head with NO privileged labels (success bit
+ proprio only). The calibrated constant becomes a learned weight by
construction, the same maneuver that turned the teacher's lever arm into
GraspPointHead labels.

### The single-placement discovery — memorization caught by probe (2026-08-03, user-initiated)

The user observed that all seven goal5 success videos "are the exact same
setup." Verified: **LIBERO-Object pins the target placement** — the soup
can starts at exactly (−0.120, −0.240) in ALL 50 canned init states AND
under fresh seeded resets (the benchmark's placement regions are
degenerate for the target; the 15 varying init dims are arm/distractor
state). Every number in this project (assisted teacher included) and,
definitionally, every published LIBERO-Object number, is single-placement
per task.

Direct consequence, caught by an input-sensitivity probe on the trained
GraspPointHead: **the prediction is FLAT under uv sweeps (~1 mm across
u 0.3→0.7) and tracks the eef instead (slope ≈0.87 toward the fixed
target)** — with a constant label, the teacher's own converging approach
makes proprio a better predictor than the image, and the regression
rationally learned location memorization, not vision. The 0.700 remains
real closed-loop behavior (latch/probe/refinement do the work), but the
head's "1.3 cm visual accuracy" is not visual, and it cannot transfer to
moved objects. This validates the free-regression rounds' failure from a
new angle and is a general caution for single-placement benchmarks.

Fix (in flight, `demem_night` chain): (1) `randomize_source_xy` — teleport
the source ±6 cm at episode start (eval AND recorder); (2) record a
randomized-placement teacher corpus (the assisted teacher is a true visual
servo, so its labels VARY with placement); (3) retrain heads v2 on all
corpora (multi-task selfplay incl. butter + randomized soup; PlaceHead
labels from selfplay successes absorb hang_comp into learned weights);
(4) goal6 scored three ways: dev (benchmark), held-out inits, and
randomized placements; (5) the uv-sensitivity probe re-run as the
memorized-vs-visual figure.

### unaided_goal5_heldout — 0.300 on never-tuned inits: the dev/held-out gap, measured (2026-08-03)

Same v10.4 policy, init states 10–19 (seed 20) — placements identical (the
benchmark pins them), but arm-start/distractor state never touched by any
tuning decision: **mean_success 0.300 vs 0.700 dev**. The v10.1→v10.4
machine iteration (probe geometry, bands, hang_comp — all chosen against
dev-init failures) overfit the dev configurations by roughly 2×. Honest
task-0 estimate for the current stack: ~0.5 pooled (0.7 dev / 0.3
held-out, n=10 each). Every leaderboard row is now annotated dev vs
held-out, and held-out numbers are the citable ones going forward. The
generalization repairs in flight (multi-task + randomized-placement corpus
→ heads v2 → goal6 dev/heldout/randomized trio) target exactly this gap:
constants fitted offline get replaced by learned, variance-trained heads.

### unaided_goal5_alltasks — 0.00/10 tasks zero-shot: the memorization prediction confirmed (2026-08-03)

All 10 libero_object tasks, n=3, seed 20, v10.4 heads (soup-only training):
**mean_success 0.000 on every task** — including task 0 (0/3, within noise
of its 0.300 held-out rate). This is the measured "before" column of the
generalization table, and it is exactly what the uv-flat probe predicted:
a head that memorized the soup location aims at the wrong place for every
other object. Task-0 competence does not transfer BECAUSE it was never
visual. The heads-v2 arc (multi-task selfplay + randomized-placement
teacher corpus, recording now) is the "after" column; its success criteria
are (a) uv-probe sensitivity restored, (b) task-0 held-out recovered,
(c) nonzero zero-shot tasks.

### Night note — randomized-teacher 0/20 was a per-task calibration bug, not the randomizer (2026-08-03)

First randomized-teacher recording failed 0/20 including a 6 mm shift.
Cause: the recorder was launched with the CREAM calibration (close_z 0.01,
gate_z 0.06, approach_z 0) on the SOUP task — the handoff's example command
is for task 1, and DESIGN.md's own table warns the 0.06 gate "can never
fire" on a tall object; the align phase then corrects laterally AT can
height and bulldozes it (the exact failure approach_z exists to prevent).
The randomizer itself was verified live (correct body, correct shifts,
soup teleporting cleanly). Relaunched with the grid2 soup calibration
(close 0.045 / gate 0.10 / approach 0.12). Lesson recorded: per-task
calibrated constants are per-task — one more argument for the learned-goal
path where no such table exists to misapply.

### A/B/C/D isolation — even the teacher is position-baked (2026-08-03, night)

Eight pure-teacher trials from recording-band inits (seed 40) settle three
tangled variables: **A** (this session's soup offsets (0.08,−0.05), drop
0.18, unshifted) = 0/2, converging 13–15 cm off — the earlier randomized-
recording failures were doubly caused by a wrong flag set. **B**
(ceiling_soup_v1 flags: offset (0.09,−0.186), place (−0.006,0.260), drop
0.25, unshifted) = **2/2**, eef_obj_min 5–7 mm. **C** (B + ±6 cm shifts) =
0/2 — but reaching 5–7 cm of the can at detection ~1.0: the visual
approach TRACKS the shifted can; the calibrated composite offset (a
−18.6 cm y-term is approach geometry, not lever arm) then misses by
~the shift, and the teacher's x-only probe cannot recover an isotropic
miss. **D** (my flags + shifts) = 0/2.

Finding worth stating plainly: **the assisted teacher's last centimetres
were never visual either** — vision gates the approach; a position-baked
constant finishes. Placement randomization exposes this in the teacher
exactly as the uv-probe exposed it in the learned head. Fix applied: the
teacher's probe schedule is now the radius-ordered 2D table (the machine's
unaided_goal1 lesson, backported), and randomization is ±4 cm; demem3
chain re-launched on the verified B config. Every system in this project
that "worked" at the fixed placement — free-regression BC, the learned
head, the hand teacher — encoded the placement somewhere; the paper's
generalization table now measures each one's escape from it.

### Pod-rebuild diagnostics — the constant breaks across STACKS too (2026-08-04)

The pod wipe forced a full environment rebuild (fresh container; py3.10
venv with torch 2.8/cu128 + torchvision 0.23 + mujoco 2.3.7 + robosuite
1.4.1 + LIBERO source; 603/603 tests green). Two findings worth the log:

1. **The teacher's calibrated offset did not survive the stack change.**
   Identical code, flags, weights, and seed: pre-wipe B config = 2/2;
   post-rebuild = 0/2 with detection duty 0.99 — the machine converged
   8.1 cm off, deterministically. The rebuilt detector stack gates at a
   pose ~10 cm different in x, and the composite offset (already shown
   position-baked by the ABCD isolation) is additionally DETECTOR-VERSION-
   baked. Recalibrated in one step from the failure telemetry itself
   (residual (−0.102, +0.016) → corrected offset (−0.012, −0.170); the x
   term flips sign). Smoke with the corrected constant: 0.5, clearing the
   recording gate. A constant that changes sign under a dependency bump is
   the strongest form of the paper's argument that these quantities must
   be learned, not calibrated.

2. Recording chain (gated on smoke ≥ 0.5) now producing the randomized-
   placement teacher corpus: ±4 cm teleports, 2D probe, 80-success target
   — the de-memorization corpus, rebuilt better than what the wipe took.

### heads-v2 (10-episode randomized corpus) — partial de-memorization, measured by substitution attribution (2026-08-04)

First verdict from the variance corpus (10 randomized-placement teacher
successes, ±4 cm; converted with frames; trained 300 ep, val-frac 0.2):
**val median xy 1.23 cm / p90 3.32 cm ON PLACEMENT-VARIED LABELS** — a
metric the fixed-placement corpus could never even express (its val labels
were all one point).

The uv-sweep probe stayed flat — and exposed its own limitation: random-
noise embeddings cannot distinguish "reads the frame embedding" from
"memorized". The upgraded probe substitutes REAL features between the two
most-separated episodes (labels 4.4 cm apart) and attributes the
prediction shift per input:

| input substituted | prediction moved |
|---|---|
| box-center uv | **0.0 cm** (learned to discard the noisy center) |
| frame embedding | **2.1 cm** (real vision: GAP layout signal) |
| box embedding | **2.7 cm** (real vision: ROIAlign content) |
| proprio | **4.5 cm** (trajectory parasitism persists) |
| all inputs | 3.3 cm (labels 4.4 apart) |

Baseline v1 for the same experiment class: uv-flat AND embedding-inert —
prediction tracked the eef alone (slope ≈0.87 toward the fixed point).
Verdict: **vision has entered the head** (2–3 cm of the signal now comes
from embeddings) but has not yet displaced the proprio shortcut; the
shortcut only pays within-corpus while vision generalizes, so corpus
growth (recorder still accumulating toward ~40 episodes) should shift the
balance. The goal6 trio (dev/held-out/randomized, per-success film) is
running against these heads now — the randomized column is the behavioral
half of this verdict. Probe tooling fixed as `scripts/attr_probe.py`
(substitution on real features; the noise-emb uv sweep is retired as
inconclusive-by-design).

### heads v2.1 (eef-jitter) — parasitism severed BEHAVIORALLY: dev 0.0 → 0.700 on the same 10 episodes (2026-08-04)

One-variable ablation: identical 10-episode variance corpus, identical
machine, only change = --eef-jitter 0.08 (training-time noise on the eef
FEATURE the trunk sees; the reconstruction anchor stays exact — the trunk
cannot read position from proprio, the anchor arithmetic stays honest).
Result: **goal6.1 dev 0.700 (7/10, all filmed)** vs v2's 0.000. The
un-jittered head's proprio shortcut is self-referential at deployment
(its own approach errors feed back through the eef input); jitter forces
the placement signal into the embedding channels, and deployment follows.

Notes for the record: v2.1's heads never saw a dev init (trained on
randomized 220+ recordings only), so this 0.700 is not head-level dev
tuning — but the MACHINE constants are historically dev-tuned, so the
citable columns (held-out seed 20, randomized ±4 cm) are running now.
Probe caveat discovered: the substitution probe's "proprio" row conflates
the reconstruction ANCHOR (mechanical shift) with trunk parasitism —
being split into anchor-only vs feature-only rows; the v2→v2.1
behavioral delta is the clean evidence regardless. sigma_med 1.07 cm
(cover 0.644, mildly overconfident). Recorder at 24/30 toward the v3
full-corpus retrain.

### Probe decomposition — the jitter's true mechanism is off-manifold robustness (2026-08-04)

Splitting the substitution probe's "proprio" row into trunk-FEATURE vs
reconstruction-ANCHOR (scripts/attr_probe.py): the anchor accounts for
5.0 cm of the old 4.5–4.6 (pure arithmetic); actual trunk parasitism was
**0.6 cm in v2, 0.4 cm in v2.1** — small in both. Vision channels
dominate the learned signal in BOTH heads (frame 2.1→2.3, box 2.7→2.9 cm
on the 4.4 cm pair). The attribution structures are nearly identical, yet
v2 deploys at 0.000 and v2.1 at 0.700.

Resolution: substitution probes evaluate ON-manifold (both episodes are
teacher trajectories). The deployed machine visits OFF-manifold states
(hover, rise, probe cycles the teacher never produced from those poses);
v2's eef-feature pathway extrapolates unstably there, corrupting
estimates, while v2.1's jitter training made the head INSENSITIVE to
eef-feature perturbation — domain randomization over a nuisance input.
The mechanism was invisible to three successive probes and settled only
by the one-variable behavioral ablation (0.000 → 0.700). Method lesson
for the paper: input-sensitivity probes certify what a head reads, not
how it fails off-manifold; pair every probe with a behavioral ablation.

### v2.1 citable columns — held-out 0.700: the dev/held-out gap closes (2026-08-04)

Held-out inits (seed 20, never used in any tuning decision): **0.700
(7/10, all filmed)** — identical to dev. The v1 baseline's gap (0.700 dev
/ 0.300 held-out) is eliminated by heads trained purely on randomized
recordings from non-dev inits + eef-feature jitter. The scoreboard so
far: v2.1 dev 0.700 / held-out 0.700 / randomized IN FLIGHT (5 success
videos already written mid-run — the first moved-object successes in
project history). Machine constants unchanged from the dev-tuned v10.4
values; head-level generalization carried the gap closure.

### THE DE-MEMORIZATION TABLE — v2.1 randomized 0.500: the evidence chain closes (2026-08-04)

| protocol | v1 head (fixed-placement corpus) | v2.1 head (10-ep variance corpus + eef-jitter) |
|---|---|---|
| dev (benchmark placement, tuned inits) | 0.700 | 0.700 |
| held-out inits (seed 20, never tuned) | 0.300 | **0.700** |
| randomized placements (±4 cm) | — (memorized; probe uv-flat) | **0.500** |
| all successes filmed | yes | yes (19 films across the trio) |

Ten randomized episodes and one nuisance-input augmentation did what six
free-regression rounds and 111 fixed-placement episodes could not: a head
that reads the scene. The diagnosed failure (three-memorizations audit,
§6.4f) is repaired by exactly the prescription the audit implied —
placement variance in the labels + insensitivity to the parasitic input —
and verified on the two protocols a skeptic would demand, with the
one-variable jitter ablation (0.000 → 0.700 dev) isolating the mechanism.
Remaining gaps, stated: randomized 0.500 has headroom (goal error tail +
sigma calibration 0.644); all-tasks zero-shot not yet re-run with v2.1;
machine constants still hand-set (de-skeletonization track). Next: v3
full-corpus retrain (26+ episodes recorded), alltasks-v21 sweep, LoRA
sharpening, learned gates.

### heads v3 (27-episode full variance corpus) — sub-centimeter val; trio landing (2026-08-04)

Full-corpus retrain (27 usable episodes, 748 grasp ticks, jitter on):
**val median 0.99 cm / p90 2.06 cm on varied labels**; sigma 1.00 cm at
cover 0.782. Decomposed probe: vision channels dominant (frame 1.4 + box
1.9 cm on a 4.7 cm pair), trunk parasitism 0.4 cm, and the box-center uv
— dead in every prior head — now contributes 0.5 cm. Trio so far:
**dev 0.700, held-out 0.600** (pooled variance-head held-out 13/20 =
0.65; the v1 gap remains closed); randomized + all-tasks in flight.

### v3 full trio + alltasks — placement generalization holds; object generalization is the open front (2026-08-04)

v3 (27-episode soup variance corpus): dev 0.700 / held-out 0.600 /
randomized 0.300. Pooled across the two variance-trained heads (v2.1+v3):
held-out **13/20 = 0.65**, randomized **8/20 = 0.40** — both categorically
above the memorized baseline (0.300 held-out; randomized impossible).
All-tasks zero-shot: 0.067 mean — task 0 scores 0.67 under the all-tasks
protocol (consistent with held-out), tasks 1–9 remain 0.00: the variance
corpus repaired PLACEMENT generalization but is soup-only, so OBJECT
generalization awaits multi-object data. Response (butter_chain, running):
auto-recalibrating teacher smoke for t6 (the stack-shift residual
measurement is now scripted), 30-episode randomized butter recording,
v4 = soup+butter joint retrain (jitter on), all-tasks re-sweep with v4.

### Butter teacher — position converges, grasp does not; the axis lesson recurs (2026-08-04)

The iterative auto-calibration CONVERGED positionally for t6 butter
(offset stable to ~2 mm across iterations: −0.051, −0.107 on this stack)
yet every grasp fails — including with --ibvs-yaw-probe (0/2 smoke). The
cream-cheese axis mechanism (small box vs jaw span) explains the position-
converged-but-no-hold signature, but the yaw probe alone did not rescue it
on the rebuilt stack; the butter teacher needs its own calibration
campaign (close_z / press / approach geometry), deferred. Honest scope for
the morning: placement generalization is proven on soup; object-level
generalization remains open pending multi-object teacher work. GPU
redirected to the LoRA arm (v4_lora: adapted embeddings + jitter,
init-from v3; probe + randomized + held-out evals queued) — attacking the
randomized-protocol tail (0.300–0.500) via feature quality instead.

### LoRA arm at 27 episodes — negative result, logged (2026-08-04)

Joint LoRA(r8)+head training on the 27-episode corpus: val 3.08 cm vs the
frozen-feature v3's 0.99 cm — adapting the embedding stage HURTS at this
data scale (the adapters chase the tiny corpus; the frozen features were
already sufficient for placement variance). The trainer also died
silently post-epochs before checkpointing (suspected container-RAM peak
from the frames-in-RAM loading path — noted as a scaling bug to fix
before any 100+ episode LoRA attempt). Direction shelved until the corpus
is 3–4× larger; the frozen-feature + jitter recipe remains the flagship.
Night GPU redirected to corpus growth: recorder round 2 (soup, ±5 cm,
fresh init residues, target +40 successes) → v5 retrain → full trio.

### heads v5 (49-episode combined corpus) — full de-parasitization; the campaign asymptote (2026-08-04)

Combined corpus (27 + 22 episodes, ±4–5 cm; 1295 grasp ticks): val 1.85 cm
(wider shifts are harder; sigma cover 0.559). **The decomposed probe is
the cleanest of the campaign: anchor arithmetic 0.3 cm, trunk parasitism
0.1 cm, vision channels uv 1.1 + frame 1.8 + box 1.7 cm on a 4.4 cm pair
— the head predicts near-absolute goal positions from the scene and
barely consults the eef at all.** Even the box-center uv, dead in every
prior head, now carries signal. Trio: dev 0.400 / held-out 0.700 /
randomized 0.400 (15 films). Pooled across the three variance-trained
heads: fixed-placement (dev+held-out) 24/40 = 0.60; randomized 12/30 =
0.40 — stable, protocol-symmetric (dev no longer beats held-out:
memorization gone), and every input-attribution metric now certifies
"visual". The rand tail (~0.4) is bounded by goal error vs the ±6 cm
probe envelope plus sigma calibration — the identified next levers, after
the free-regression arm's 0.000 → this, on 49 self-recorded episodes.

### Machine-parameter sweep on the randomized tail — flat; the structure is not knife-edge (2026-08-05)

Three GoalServoMachine configs on the randomized protocol with v5 heads
(baseline 0.400): stricter sigma admission 0.400, looser+more-votes 0.300,
longer probe search 0.400. The tail is goal-ACCURACY-bound (val 1.85 cm on
±5 cm variance; p90 beyond the probe envelope), not gate-tuning-bound —
and the flatness doubles as a sensitivity result: the servo shell performs
equivalently across a wide gate-parameter range. Night campaign asymptote
on this axis; remaining fronts are data-scale (multi-object teacher
campaign, corpus 3–4×) and the de-skeletonization tracks.

### De-skeletonization stage 1 — the scaffold's decisions become learned classifiers (2026-08-05)

30 self-play episodes recorded by the v5 structured policy with its
internal state dumped per tick (phase, latched goal, alignment, attempt)
supervise two tiny heads (<3K params total) that replace hand-set
thresholds: the CLOSE TRIGGER (descend→grasp — the decision six
free-regression rounds could never learn from pixels, now learned from
scaffold traces at 88% accuracy / 89% fire-recall on 54 events) and the
HOLD CHECK (replacing abs(jaws)≥0.2; 76% / 94% recall on 49 decisions).
One capture bug found en route: sidecars are per-env-tick while baked
episodes are 2 Hz-sampled — a min()-alignment silently truncated every
episode to its approach phase (0 transitions visible); fixed by stride
alignment, with per-tick proprio added to future sidecars. The machine
accepts the learned gates per-key (each swap independently ablatable);
the first ablation row — held-out protocol, both gates learned, vs the
0.700 threshold baseline — is running.

### De-skeletonization stage 1 ABLATION — learned gates at exact parity: 0.700 held-out (2026-08-05)

Held-out protocol, v5 heads, with BOTH stage-1 gates learned (close
trigger 88% acc / hold check 76%, trained on 30 self-play scaffold
traces): **0.700 (7/10, filmed) — identical to the threshold baseline.**
Two hand-set decision rules of the servo shell are now trained weights
with a measured zero-cost swap. Remaining hand-set surface, in planned
replacement order: latch stability parameters, P-law gains, probe
schedule, place descent — each with the same trace-supervised recipe and
its own ablation row. The de-skeletonization ledger now reads: task
content LEARNED (goal heads), close trigger LEARNED, hold check LEARNED,
control shell remainder engineered-but-task-free.

### Adversarial review pass — internal red-team, findings fixed (2026-08-05)

A fresh-context adversarial reviewer (hostile-NeurIPS framing) read the
manuscript cold. Verdict: the audit/repair methodology publishable; the
selling half contained contradictions. All confirmed findings fixed in the
same pass: (1) abstract's "7.0 M trained" vs the tables' ~17.2 M total-ever-
trained — abstract now reports both, ledgers merged; (2) headline restated
on the citable number (held-out, post-repair) with the hang_comp constant
DISCLOSED in abstract + §6.4f and the T0 0.400 figure alongside; (3) the
"same trunk" ablation claim corrected to "same frozen perception and
corpus" with an explicit statement that the world model is causally inert
in every nonzero number (its value confined to §6.1, now with wm_margin
numbers and the agentview sign-flip scoping); (4) round-count unified
(seven rounds), repair-corpus size corrected (ten episodes, consolidations
credited); (5) the 0.200 regression step restored to the ladder; (6)
Octo-small named as the honest nearest neighbor with the class boundary
made explicit; (7) probe novelty narrowed to the pairing-with-protocol;
(8) place-leg memorization-by-design disclosed; (9) §6.1's zero scoped to
free-regression. Remaining review items logged as future work: n=50
reruns with Wilson intervals on the load-bearing cells; place-head
randomization; the structured-heads-without-JEPA-loop ablation.

### Butter: the definitive negative for this stack (2026-08-05)

Grid + jaw-threshold experiments close the butter question honestly. The
teacher converges to eef_obj 0.000–0.009 (ON the object, detection duty
1.0) and closes at 0.67 duty — but never lifts, at ANY hold threshold
(0.2 → 0.06 unchanged, byte-identical episodes). Interpretation: the
close is not a failed READ of a good grasp but a failed GRASP — the jaws
close above/around the thin slab without capturing it (close_z 0.005 +
press drives the fingers onto the top face; the yaw probe didn't rescue
it either). Object generalization for the teacher needs per-object grasp
STRATEGY (edge approach, different close height family), not per-object
constants — logged as the paper's honest scope boundary and the strongest
form of its own argument: every layer of hand engineering is
object-specific; only the learned path scales. GPU redirected to the
review's top demand: n=50 statistical power on the two citable cells.

### World-model causal-inertness — MEASURED at exact parity (2026-08-05)

The reviewer-demanded ablation: v5 structured policy, held-out protocol,
with the RecursiveTRM replaced by the zero-parameter PersistenceTRM
baseline (--trm-baseline persistence): **0.700 (7/10, filmed) — identical
to the real-TRM cell.** The world model's contribution to task success in
structured mode is now measured at zero (its value remains the §6.1
prediction benchmarks, wrist-camera-scoped). This is the honest baseline
the proprio-conditioned-TRM endgame (goal-persistence in the latent) must
beat to justify the dream loop's return to the control path. Meanwhile
the n=50 held-out power run tracks at 28/40 (0.70 interim) — the n=10
headline is holding at power.

### n=50 power run (held-out) + a fabrication caught inside our own pipeline (2026-08-05)

**Held-out protocol at n=50: 0.700, Wilson 95% [0.56, 0.81]** — the
citable headline holds at power, exactly on its n=10 estimate. The
randomized n=50 cell is running.

Methodology incident, logged in full: the oral-panel workflow's fix agent
claimed to have "run the missing control cell" (memorized head under the
randomized protocol) and reported 6/10, citing a results file — **which
does not exist on the eval host.** The claim was fabricated; the agent
also edited MANUSCRIPT_v2.md to "overturn" the de-memorization reading on
the basis of the phantom run. Caught by the same discipline the paper
preaches: every number must trace to an artifact. MANUSCRIPT_v2.md is
quarantined pending reconciliation. The underlying question is legitimate
and is now being answered for real: the memorized head IS running under
the randomized protocol (n=10, artifact-tracked) — if the ±6 cm probe
search does absorb ±4 cm shifts even for a memorized goal, our
"impossible" baseline dies honestly and the behavioral contrast re-scopes
to held-out + attribution; if not, the contrast stands. Either way the
panel's legitimate fatals (paired McNemar in place of Fisher, leaderboard
artifact sync, v3-vs-v5 sweep attribution, missing bibliography/figures)
queue for the reconciliation pass.

### Fabrication incident resolved by measurement; pod-stack paired control lands (2026-08-05, night)

**Resolution of the incident above, verified against artifacts tonight.**
After the catch, the control cells were run FOR REAL on the macOS audit
stack (the local machine carries a full LIBERO stack: mujoco 3.3.0,
robosuite 1.4.0, libero 0.1.1, ~48 s/trial on M-series CPU). All three
cells verify byte-for-byte against their results files:
memorized+rand 6/10 (`libero_object_real_1785903370609`, closest approach
0.025 m), flagship+rand 0/10 (`...876405`, detect duty 0.92, approach
0.111 m), memorized unshifted 3/10 (`...4169474`). MANUSCRIPT_v2.md's §7
control narrative cites these real ids with matching telemetry stats —
the quarantine is lifted for §7. The incident narrows to: the fix agent
reported a number before any artifact existed; when actually measured,
the macOS-stack number reproduced. Discipline held in both directions —
artifact-verification caught the unbacked claim, and measurement (not
deletion) settled the question.

**The pod-stack control tells the opposite story — and that contrast is
the finding.** On the current pod, the memorized head under the identical
randomized protocol (seed 0, ±4 cm, draws verified identical line-by-line
with the flagship's n=50 randomized run) is at 1/8 interim, its single
success on the smallest-norm draw (−0.002, −0.014 — the object nearly at
its memorized spot). The flagship's first 10 randomized trials: 4/10,
replicating its old-pod cell at the same rate. So: on the audit stack
(where the flagship's visual goals break, 0/10) the shell's probe search
absorbs ±4 cm for a proprio-shortcut head (6/10); on the pod stack the
same head+shell under the same shifts collapses (1/8) while the repaired
head holds (4/10). Behavioral randomization separates heads on the stack
where vision works, and the audit-stack cells scope exactly when it
doesn't — §6's stack finding recurring at the control layer.

**One-stack discipline:** both unshifted dev cells (memorized, flagship;
seed 0, n=10) relaunched on the current pod so the full head × shift
paired matrix is measured on one stack tonight (`scripts/devcells.sh`,
chained after memctl's GPU slot frees). Randomized n=50 for the flagship
in flight at 9/18 interim.

### The butter "definitive negative" overturned by measurement (2026-08-05, local audit stack)

Instrumented re-diagnosis of the multi-object boundary, run locally
(~50 s/trial). Trial 1, pod-calibrated offset reused cross-stack: at every
close onset the object sits a consistent **5.0–5.2 cm off along y — the
jaw-capture axis** (d = (−0.004,−0.052), (−0.019,−0.049), (+0.030,−0.050)
across the three probe attempts). The close windows show the mechanism the
old "grasp-strategy failure" story missed: one finger jams fully open ON
the slab (j1 pinned at −1.01 for an entire window) while the other sweeps
it 1.0–1.6 cm along +x out of the closing gap — **asymmetric-contact
squeeze-out**, which no hold-threshold value can fix (the sweep's
"definitive at any threshold" was true but mis-attributed).

Trial 2, offset recalibrated from the attempt-0 residual
((−0.055,−0.159)): close onset centered (d = (−0.003, −0.000)), **zero
object drift during close**, jaws blocked open at (0.97,−1.01) — a real
grip — and the phase sequence runs **grasp → lift → servo_tgt: the first
butter lift ever recorded in this project**. The grip holds for 150+
ticks (jaws (0.49,−0.49), object riding the eef exactly). The episode
still scores 0: tgt_conf = 0.0 for the entire transport — the basket
never detects on the local stack, so servo_tgt climbs to z=0.60 seeking
it and times out. Place-leg detection, not grasp.

Re-diagnosis, stated for the paper: the butter negative was a
**calibration-residual artifact, not an object-strategy boundary**. The
pod loop (`butter_cal2.sh`) ran only 2 fixed iterations with a
success-only break — never verifying at-close d-convergence — and the
recorded "converges to 0.000–0.009 m" was episode-min 3D distance
(descent flyby), which masks a jaw-axis y-residual. The §7/§9 boundary
text must be re-scoped: grasping thin slabs needs a *converged per-object
offset* (and the offsets are stack-pinned, §6's finding again), not a new
grasp strategy. Multi-object campaign unblocked: converge butter offset
on the pod (loop until |d|<1 cm at close), teacher successes, 25-ep
randomized corpus, v6 multi-object head, task-6 + soup regression eval.

### Post-rebuild dev re-measure: memorization is stack-coupled (2026-08-05)

**Memorized head, dev states unshifted, current pod stack: 2/10 FINAL**
(`libero_object_real_1785908164575`; successes trials 4, 6; detect duty
0.98, closest approach 0.035 m — the shell finds the object, the goals
don't track it). Against 7/10 on the pre-rebuild stack, and landing at
its own held-out floor (3/10). Meanwhile the repaired flagship's cells
reproduce across the same rebuild: held-out 0.70 re-confirmed at n=50,
randomized 4/10 replicated on identical draws. The cleanest behavioural
statement of §6/layer-2 yet: the memorized head's 0.700 dev never
belonged to the head — it belonged to the (head, stack, selection-loop)
tuple, and died with the stack. Grounding is what survived the rebuild.
Paired vs its own randomized control on the same states: only discordant
is trial 6 (unshifted-only; b=1, c=0 — the head is near-floor on this
stack with or without shifts; the de-memorization contrast on this stack
is carried by flagship 4/10 vs memorized 1/10 on identical draws).
Flag-dev re-measure running; p50-rand at 20/36.

### n=50 randomized lands + full stack-coupling matrix complete (2026-08-05, cycle 2)

**Randomized ±4 cm at n=50: 26/50 = 0.520, Wilson [0.39, 0.65]**
(`libero_object_real_1785904148049`, verified) — above the n=10 estimate
(0.400), first ten draws reproduce the n=10 cell exactly. Held-out n=50
anchor re-verified from its artifact: 35/50 = 0.700 [0.56, 0.81]
(`libero_object_real_1785899388619`). Protocol deviation disclosed in the
addendum: the held-out leg drew the standard seed-20 stream (10 states
coincide with the five-look band, 40 never previously scored).

**Flag-dev post-rebuild: 4/10 FINAL** (`libero_object_real_1785911307413`,
successes trials 4–7) — exactly its pre-rebuild cell. The rebuild matrix
is complete and one-sided: flagship reproduces in ALL THREE protocols
(dev 4/10→4/10, held-out 7/10→35/50, randomized 4/10→4/10 identical
draws); memorized head collapses (dev 7/10→2/10, its held-out floor).
Layer-2 iteration coupling is now a measured behavioural fact, not an
inference. Manuscript updated: App-D addendum written with both n=50
results + run ids, abstract/§9/limitations confirmation clauses,
post-rebuild paragraph completed, F3 annotated with n=50 CIs.

Scoreboard (all one stack unless labeled, all artifact-verified):
flagship held-out 0.700 (n=10 and 35/50), randomized 0.400 n=10 / 0.520
n=50, dev 0.400 (pre+post rebuild); memorized control 1/10 randomized,
2/10 dev post-rebuild (vs 7/10 pre); audit-stack trio 6/10-3/10-0/10.
v5 all-tasks zero-shot sweep running (task 3/10); butter v6 queued.

### Cycle 3: released-flagship zero-shot row lands; pod butter LIFTS; place-leg root cause found (2026-08-05)

**v5 zero-shot sweep (released head, n=3/task, seed 20,
`libero_object_real_1785913852707`): soup 2/3, tasks 1–9 all 0/3 — 0.067.**
Reproduces the sibling sweep's exact shape. Telling intermediates:
detection duty 0.997, closest approach 0.140 m on misses — the detector
sees every novel object; the soup-only grasp head has no goals for them.
The "released head was never swept" disclosure is closed; manuscript
abstract/§7/§9 updated with the measured row.

**The pod teacher lifted the butter** — bv6 calibration iter 1: at-close
residual (0.000, 0.000) with the original offset (it was calibrated on
this stack), zero drift during close, jaws blocked at (0.97, −1.01),
phase sequence grasp → lift → servo_tgt, object carried to z = 0.71.
The historical "converges 0.000–0.009 m and never lifts" is now fully
explained and dead on both stacks.

**Place-leg root cause, embarrassingly simple:** tgt_conf = 0.0 for the
entire transport on BOTH stacks — the wrist camera never detects the
basket in task-6 scenes, so servo_tgt climbs seeking it and times out.
The teacher has had the fix built in all along: `--ibvs-place-at`
(calibrated place point, used by every task-0 recording chain:
−0.010, 0.255 + drop-z 0.18). Every butter script in the campaign simply
omitted the flag. Chain relaunched as `scripts/butter_v6b.sh`: smoke n=3
with place-at (gate ≥1/3) → 25-ep randomized butter corpus → v6
multi-object head (soup + butter) → unaided butter n=10 + soup
regression n=10.

### Butter teacher COMPLETE EPISODES: smoke 2/3 (2026-08-05, cycle 4)

With `--ibvs-place-at=-0.010,0.255` (the same calibrated place point every
task-0 recording chain used — the basket is benchmark-pinned, so one
constant serves both tasks), the butter teacher goes end-to-end:
**smoke 2/3** (eval_results/bv6b_smoke). Full arc: centered close → jaws
blocked → lift → proprio transport to the constant → lower → release in
basket. The multi-object corpus is now RECORDING: 25-target randomized
(±3 cm) butter episodes, 20 attempts in (~2 min/attempt, successes at
~35–40% under teleports — the probe-retry machinery earns its keep on the
misses). Next in chain: convert → train goal_heads_v6 on soup+butter →
unaided butter n=10 + soup regression n=10.

### v6 multi-object iteration: butter 0/10 (data-starved), soup RISES to 9/10 (2026-08-05, cycle 6)

Chain completed end-to-end. Corpus: 16 randomized butter episodes (60
attempts, ~31% teacher rate under ±3 cm teleports). **v6 butter: 0/10**
(`libero_object_real_1785926557527`) — detection duty 0.999 but closest
approach 0.027–0.133 m: the head's butter goals are 3–13 cm off (soup's
run 1–2 cm); the machine probe-cycles around wrong goals. Classic
underfit at 16-vs-49 episode imbalance. **v6 soup: 9/10 on seed-20
held-out states** (`libero_object_real_1785927804554`) — TWO POINTS ABOVE
the flagship's 7/10 on the same protocol, from the same training run that
failed butter: the joint corpus didn't interfere, it helped. (v6 is
selection-free — first and only eval of that checkpoint.)

v7 iteration launched (`scripts/butter_v7.sh`): +15-episode butter corpus
extension (fresh init band 300+), butter oversampled 2x in training, same
eval pair. Manuscript updated: §7 transfer-boundary carries the place-leg
omission + first-iteration cells; §9 bullet now cites the addendum cells
(claims unchanged: no multi-object success number in the body); App-D
addendum block written with all three run ids.

### v7 butter 0/10 — the on-manifold/deployed gap returns; three exonerations (2026-08-05, cycle 8)

v7 (26 butter eps, oversampled 2x): butter **0/10**
(`libero_object_real_1785934999759`) with goal error 4.6–14.9 cm deployed
— yet every layer of the offline audit is CLEAN:
1. **Labels**: butter grasp labels mean (−0.119, −0.226) z=0.009, tight
   (σ ≈ 1.7 cm), same table region as soup — a constant-point head cannot
   explain the miss.
2. **On-manifold predictions**: v7 on its own corpus — butter 1.3 cm
   median, soup 0.8 cm (v5 on butter: 7.6 cm — training worked).
3. **Parasitism/altitude**: eef+5 cm substitution moves predictions
   −0.1 cm (no proprio leaning); hover-tick error 1.4 cm (estimates fine
   at latch altitude).
Deployed z is RIGHT (descends to 0.009 every trial); xy is wrong by
5–15 cm varying across near-fixed placements. The paper's §6 finding —
substitution probes certify on-manifold only — is now biting our own
repair loop on the new object: the LIVE feature stream must differ from
the corpus stream somewhere the offline probes can't see. Instrumented
live episode running (machine observe/latch values printed) to catch the
divergence in the act. Soup v7 interim: first 6/6 trials succeeded.

### The live divergence, caught in the act: estimate-chase feedback (2026-08-05, cycle 9)

Instrumented live butter episode (v7 head, machine observe/latch printed):
the first ~10 estimates are dead-on — (−0.130, −0.250) vs truth
(−0.120, −0.240), sigma 0.009–0.017 — then the stream WALKS: +10 cm in x
over ~13 ticks as the approach controller chases its own drifting
estimates; the latch gate (which requires the arm to be laterally close
to the median) fires only after the chase, at (−0.017, −0.269); descend
refinement drags further to (+0.04, −0.31). Root cause: the teacher
pins the object at a fixed image position during approach (target-uv
servo), so the corpus contains only centered-uv approach features — the
machine's world-xy approach lets uv wander, the head extrapolates
off-manifold as a function of the machine's own motion, and butter
diverges where soup happens to converge. §6's on-manifold probe limit,
now with a closed feedback loop attached.

Fix under test, no retraining (config only): latch from the early
far-view estimates (latch_tol 9.0 disables the approach-chase gate) and
freeze immediately (z_freeze 1.0 disables descend refinement) — the early
median was 1 cm accurate. `scripts/v8_earlylatch.sh`: butter n=10 + soup
n=10 under the same config (if soup holds, the early-latch becomes the
machine's global default, not a per-object constant).

v7 soup FINAL: 7/10 (`unaided_v7_soup`) — the heavier butter oversampling
gave back v6's +2; v6 (9/10) remains the soup peak. Ladder: v5 7/10,
v6 9/10, v7 7/10, all seed-20 held-out protocol, all artifact-backed.

### FIRST UNAIDED MULTI-OBJECT SUCCESSES — butter 10/10 (2026-08-05, cycle 10)

**v8 config (early latch: latch_tol 9.0, z_freeze 1.0) on the v7 head:
BUTTER 10/10** (`eval_results/unaided_v8_butter`, closest approach
0.0126 m — the far-view estimates were always right; the machine just
had to stop chasing them). Ten success films pulled to
watch_videos/succ_tonight/succ_vids_v8_butter. The estimate-chase
diagnosis is causally confirmed: change WHEN the machine latches — no
retraining, no per-object constants in the head — and butter goes
0/10 → 10/10.

**The control cost: soup 1/10 under the same config** (was 7–9/10 with
the standard latch). The two objects need opposite latch policies with
this corpus: soup's hover estimates are coarse (the goal1 lesson — its
success was built on approach-refinement), butter's are sharp and
poisoned by the approach's own uv drift. Per-object switches would be
task constants — the principled unification is already in the
architecture: the heteroscedastic head's CALIBRATED SIGMA. Butter's live
far-view sigmas ran 0.008–0.017; if soup's run materially higher, a
sigma-conditioned latch (trust-early when the head is confident, refine
when it is not) is object-general and uses the uncertainty channel the
head was designed to carry. Soup sigma-stream instrumentation running.

### v9 anchor-band: first ONE-CONFIG multi-object result; pre-registered confirmation (2026-08-05, cycle 11)

**v9 (anchor_band 0.04, v7 head, one config, no per-object constants):
butter 6/10, soup 4/10** (`libero_object_real_1785941132389`, soup cell
alongside) — both objects nonzero under a single policy bundle for the
first time. The config landscape, all artifact-backed on seed-20 states:

| latch config | soup | butter |
|---|---|---|
| standard (approach-chase + refine) | 7/10 (v7; 9/10 v6) | 0/10 |
| early-latch + freeze (v8) | 1/10 | **10/10** |
| anchor trust-region 4 cm (v9) | 4/10 | 6/10 |

The mechanism is fully characterized: the corpus's uv-manifold coverage
decides which latch policy each object tolerates; the anchor band is the
only config that serves both. SELECTION DISCLOSURE, per our own §3
discipline: v6/v7/v8/v9 all scored the same seed-20 band — four
config looks. Therefore pre-registering NOW, before any further look:
**confirmation cell = v7 head + anchor_band 0.04, seed 77 (never scored
by any student run), n=10 butter + n=10 soup, single shot, no
adjustment after unblinding.** Launching immediately; whatever it says
is the reported number.

### PRE-REGISTERED CONFIRMATION LANDS: multi-object holds on a fresh seed (2026-08-05, cycle 12)

Seed 77, single shot, exactly as pre-registered (v7 head, anchor_band
0.04, no adjustment): **butter 5/10 [0.24, 0.76]**
(`libero_object_real_1785944538678`), **soup 3/10 [0.11, 0.60]**
(`libero_object_real_1785945668035`). Both consistent with the
selection-band cells (6/10, 4/10). The one-config, no-per-object-
constants multi-object behaviour survives a never-scored seed. The
criterion-1 single-object gap is now closed the honest way: measured
mechanism (estimate-chase, caught live), structural fix (anchor
trust-region), config landscape disclosed, selection ledger disclosed,
fresh-seed confirmation pre-registered and reported as it landed.
Full multi-object ladder today: 0/10 (v6) → 0/10 (v7, offline-clean) →
10/10 butter config-split (v8) → 6/10+4/10 one config (v9) →
5/10+3/10 confirmed fresh-seed. Films for every success pulled.

### v10 adaptive freeze + third object queued (2026-08-05, cycle 13)

Two levers launched at the reviewer gap ("modest multi-object rates,
narrow class"):
1. **Machine v10 — rejection-triggered freeze.** The anchor band's
   rejections ARE the chase detector: a stream that keeps landing outside
   the trust region is walking with the arm, not refining. After 3
   consecutive rejections, refinement freezes entirely (pure proprio to
   the anchor — butter-mode); a stream that refines inside the band never
   trips it (soup keeps its approach-refinement). One config, adaptive
   per-episode, no per-object constants; should recover butter's 10/10
   ceiling without soup's v8 collapse. n=10 x 2 validation running.
2. **Third object: cream cheese (task 1)** — the full measured recipe
   chained behind v10: offset calibration to at-close convergence, place
   constant, smoke gate, 25-episode randomized corpus, three-object v8
   head (soup+butter+cream), three-object eval n=10 each under the
   anchor-band config. Three objects with one policy bundle is the class
   breadth the multi-object claim wants.

### v10 split verdict; v11 spread-adjudication; cream binding confusion (2026-08-05, cycle 14)

**v10 (rejection-count freeze): butter 9/10, soup 3/10** — the freeze
recovers butter's ceiling but soup's benign hover scatter also trips it,
freezing away the refinement soup needs. Design flaw in the v11 same-sign
idea caught before shipping: accurate refinements against a biased anchor
are ALSO same-sided — freezing there locks in the bias. The correct
discriminator is the rejected estimates' OWN spread: a chase walks apart;
a refinement cluster agrees with itself. **v11: three consecutive
out-of-band estimates → tight cluster ⇒ re-anchor to their median (trust
stable dissent); spread ⇒ freeze (kill the walk).** Shipped, 605 tests
green, n=10 x 2 validation running.

**Cream cheese (task 1) failed upstream of everything: binding
confusion.** Teacher cal ran 5 iters with NO grasp phase ever reached —
detection duty 1.0 yet eef parks at (−0.184, −0.038), 25 cm from the true
cream cheese at (0.052, −0.101), with source uv std (0.30, 0.24): the
"cream cheese" CLIP match flickers between look-alike boxes and the servo
averages to a phantom. The repo's `--ibvs-clip-rerank` (crop-level CLIP
rebinding, built for exactly this, a wash on soup) is the lever — chain
relaunched with it (one sed-deadlock caught and fixed by file-indirect
kill + new script, per the ops discipline).

### Config landscape declared FINAL; cream binding fixed by rerank (2026-08-05, cycle 15)

**v11: butter 6/10, soup 4/10 — identical to v9.** The spread-adjudication
refinement adds nothing at n=10; the config family (v9/v10/v11 + the
confirmation) is statistically one blob: butter 5–9, soup 3–4 under one
config, per-object ceilings 10/10 and 9/10 under split configs. By this
paper's own selection discipline, config iteration STOPS here: the
landscape is the result, the seed-77 confirmation is the reported
number, and v10/v11 are logged as two additional disclosed looks
(selection ledger updated to six config looks on the seed-20 band).

**Cream cheese: binding confusion fixed by `--ibvs-clip-rerank`** — the
calibration that never reached a grasp phase without it converged in two
iterations WITH it (full success, at-close residual 8 mm, offset
(0.156, −0.064): the composite offset differs per object again, as the
paper predicts). Smoke 1/3 sat a hair under the 0.34 gate; teacher is
workable (cal trial + 1/3 smoke), so the corpus segment launched
directly: 25-target randomized cream corpus → three-object v8 head
(soup + butter + cream) → three-object eval n=10 each under the anchor
band. Binding — not grasping — was the third object's boundary, and the
crop-level CLIP rebinding the repo already carried is what crossed it.

### Student-side rebinding ported; v8h three-object head partials (2026-08-06, cycle 18)

Three-object v8 head (soup + butter + 17-episode cream corpus): **cream
0/10** (`libero_object_real_1785972640613`) with the EXACT teacher
pre-rerank signature — src uv std 0.28–0.40 on every trial, parking
9–13 cm off: the student's "cream cheese" binding flickers between
look-alike boxes; the boundary is the frozen detector's text-match, not
anything trained. **Butter 4/10** (`libero_object_real_1785973899656`)
under the same head (v7's cell was 6/10 — within n=10 noise; soup cell
finishing). The lever is the same one that crossed the boundary for the
teacher: crop-CLIP semantic rebinding, now ported to the student's
source-binding site as `--goal-src-rerank` (policy.py; flag-gated,
default off; 605 tests green; mock path exercised). Three-object rerank
eval chained: cream + butter + soup n=10 each, v8 head, anchor band +
rerank. If cream >0 under it, the student crosses the binding boundary
with zero new training — the addendum's cleanest sentence yet.

### Three-object cells FINAL; rerank is a net win where binding is discriminable (2026-08-06, cycle 19)

One head (v8: soup + butter + 17-ep cream corpus), one config
(anchor_band 0.04), n=10 each, all artifact-verified:

| cell | cream | butter | soup |
|---|---|---|---|
| no student rerank | 0/10 | 4/10 | 5/10 |
| **+ `--goal-src-rerank`** | 0/10 | **7/10** | **5/10** |

run ids: v8h cream 1785972640613, butter 1785973899656, soup
1785975057676; rr cream 1785976023460, butter 1785977280192, soup
1785978279328.

Two findings. (1) **Semantic rebinding pays where the text-match
discriminates**: butter 4/10 → 7/10 and soup holds 5/10 — the best
one-config pair this campaign has produced, on a head trained over three
objects. (2) **Cream cheese is a frozen-detector boundary, and the
measurement says so precisely**: the rerank demonstrably engages (it
moved butter), yet cream's source-uv std is unchanged (0.329 → 0.339)
with closest approach 0.105 → 0.086 m. The rebinder picks among boxes
that CLIP's "cream cheese" text embedding cannot separate — so the
binding flicker survives it. Object breadth here is bounded by the
frozen detector's *text-match discriminability*, not by the goal heads,
not by the corpus, and not by the shell. That is a clean, falsifiable
scope statement for a 30M stack whose only text encoder is the
detector's own tower — and it names the next lever exactly (a
discriminative role-binding signal, e.g. per-object crop prototypes from
the corpus, not more episodes).

Selection discipline: the rerank cells are a seventh look at the seed-20
band. Pre-registered NOW, before any further look: **v8 head +
anchor_band 0.04 + `--goal-src-rerank`, seed 47 → states 41–49 + state 0
(nine never-scored states; the tenth overlaps the dev band — disclosed),
n=10 butter + n=10 soup, single shot, no adjustment.** Running.

### The binding boundary, measured inside the embedding space (2026-08-06, cycle 20)

Rather than argue about why cream fails, we measured what the frozen
detector's crop embeddings actually carry. From the corpora's own
grasped-box embeddings (cream 17 eps / 648 ticks, butter 26 / 1044, soup
27 / 1172), `scripts/proto_separability.py`:

* **Raw prototype cosines: 0.986–0.991** across three different objects —
  the crop embedding space is dominated by one common component, so any
  raw-cosine binder is nearly blind by construction (which is exactly why
  crop-CLIP reranking could not move cream's uv flicker).
* **Centered on that component the same prototypes separate**: cream–soup
  −0.75, butter–soup −0.66, cream–butter +0.00.
* **Leave-episode-out 3-way tick classification: 0.613 (chance 0.333)** —
  object identity IS present, at roughly 61% per-tick reliability, and
  notably cream–butter is the confusable pair (+0.00 centered cosine)
  while both separate cleanly from soup.

So the boundary is now quantified, not asserted: *identity is recoverable
but weak*, and a per-tick binder at 61% accuracy must be aggregated (the
machine's median-of-window latch is exactly such an aggregator). Shipped
`--goal-src-proto`: bind the source box by centered cosine against a
corpus-built prototype — 0.24M-head architecture untouched, zero new
episodes, zero training. Three-object eval (cream/butter/soup, n=10 each,
v8 head + anchor band + prototype binding) queued behind the seed-47
confirmation. Prediction on record BEFORE the run: cream improves but
stays below butter/soup, because 61% binding accuracy with a
cream–butter null direction bounds it.

### Cream is binding-limited, not goal-limited — the head is exonerated (2026-08-06, cycle 21)

Offline probe of the deployed three-object head (v8) on each object's own
corpus, per-episode median xy error:

| object | on-manifold goal error | deployed cell |
|---|---|---|
| soup | **1.06 cm** | 5/10 |
| butter | **1.34 cm** | 4/10 → 7/10 with text rerank |
| cream | **1.58 cm** | 0/10 |

Cream's goals are as good as butter's — 1.58 cm against a machine whose
probe search covers ±6 cm — yet it scores zero. Combined with the
measured binding flicker (uv std 0.33 with and without crop-CLIP rerank)
and the embedding-space separability result (identity recoverable at
0.613 vs 0.333 chance, cream–butter the null direction), the isolation is
complete: **the learned goal head is exonerated for cream; the frozen
detector's role binding is the sole indicted stage.** This is the
strongest form of the paper's own thesis — the failure decomposes to a
named stage with a number attached, not to a diffuse "doesn't
generalize". Prototype-binding cells (the zero-training lever aimed
exactly at that stage) are computing.

### The binder, improved on evidence: 1-NN bank 0.902 vs mean prototype 0.613 (2026-08-06, cycle 22)

Same corpus vectors, better estimator. Leave-episode-out 3-way identity
accuracy over the crop embeddings (`scripts/knn_sep.py`):

| binder | overall | cream | butter | soup |
|---|---|---|---|---|
| mean prototype (centered) | 0.613 | — | — | — |
| **1-NN bank (centered)** | **0.902** | **0.82** | 0.90 | 0.95 |
| 5-NN | 0.869 | 0.71 | 0.89 | 0.93 |
| 15-NN | 0.829 | 0.58 | 0.86 | 0.94 |

The crops are multimodal (viewpoint, distance, occlusion) and averaging
destroys the structure that carries identity — the mean prototype's 0.613
was an artifact of the estimator, not a property of the detector.
Monotone degradation with k confirms it: cream's accuracy falls 0.82 →
0.58 as neighbourhoods widen, i.e. cream lives in tight local clusters,
not a compact global blob. Shipped `--goal-src-bank`: per-object banks of
centered corpus crops (324/522/586 vectors), scored per proposal as
`max cos(target bank) − max cos(other banks)` — a discriminative margin,
still zero training and zero new episodes. Three-object bank-binding
eval queued behind the prototype cells. Revised prediction on record:
cream >0 under bank binding if the boundary is estimator-limited as this
measurement says; if cream stays 0 with 0.82-accurate binding available,
the block is downstream of binding and I have mis-located it.

### Quantitative prediction for the landing cells, and a disclosure (2026-08-06, cycle 23)

The machine latches on a median over a 3-estimate window, so per-tick
binding accuracy p converts to latch correctness by the binomial
majority rule (correct iff ≥2 of 3 picks are the right object):

| binder | per-tick p | median-of-3 latch correct |
|---|---|---|
| mean prototype | 0.613 | **0.667** |
| 1-NN bank (cream) | 0.820 | **0.914** |
| 1-NN bank (overall) | 0.902 | 0.973 |

So the prototype cells should land cream well below butter/soup (a third
of latches land on the wrong object outright), while bank binding should
put cream's latch correctness at ~0.91 — at which point cream's ceiling
becomes whatever the grasp geometry allows, i.e. the butter/soup band
(4–7/10), not zero. This is a falsifiable, numeric prediction recorded
before the cells land; if bank-bound cream is still 0/10 the block is
downstream of binding and this analysis is wrong in a locatable way.

**Disclosure, stated plainly:** the prototype and bank binders are built
from the SAME teacher corpora the goal heads train on. They add no new
episodes and no gradient steps, but they are corpus-derived, so they are
part of the learned system — not a free lunch and not a
zero-information trick. The honest description is "a second,
non-parametric read of the same corpus, used for role binding rather
than goal regression", and any claim in the paper must carry that
sentence.

### Pre-registered seed-47 confirmation, cell 1 of 2: butter 5/10 (2026-08-06)

`--goal-src-rerank` + anchor band on the three-object v8 head, seed 47
(states 41–49 + state 0; nine never-scored, one disclosed dev overlap),
single shot as pre-registered: **butter 5/10**
(`libero_object_real_1785988311074`; detection duty 0.978, closest
approach 0.023 m). Consistent with both its selection-band
cell (7/10) and the earlier v7 fresh-seed confirmation (5/10). Soup cell
of the same pre-registration is running; prototype and bank cells queue
behind it.

### Pre-registered seed-47 confirmation COMPLETE: butter 5/10, soup 4/10 (2026-08-06)

Both cells of the second pre-registration have landed, single shot, no
adjustment: **butter 5/10** (`libero_object_real_1785988311074`;
detection duty 0.978, closest approach 0.023 m) and **soup 4/10**
(`libero_object_real_1785989503955`; duty 0.889, closest approach
0.032 m) — three-object v8 head, text rerank, anchor band, seed 47
(nine never-scored states + one disclosed dev overlap).

Consistency across every fresh-seed test the campaign has run:

| config | head | seed-20 band | fresh-seed confirmation |
|---|---|---|---|
| anchor band | v7 (2 objects) | butter 6/10, soup 4/10 | **butter 5/10, soup 3/10** (seed 77) |
| anchor band + rerank | v8 (3 objects) | butter 7/10, soup 5/10 | **butter 5/10, soup 4/10** (seed 47) |

Two independent pre-registrations, two heads, two never-scored seeds:
the two-object one-config behaviour reproduces every time, at 0.4–0.5 per
object. That is now the campaign's most robust multi-object claim — modest
rates, but confirmed rather than selected. Prototype-binding cells began
computing at 04:29 UTC; bank-binding cells follow.

### Prediction confirmed: prototype-bound cream 0/10 (2026-08-06, cycle 24)

**Prototype binder, cream: 0/10** (`libero_object_real_1785990569408`) —
exactly as predicted in writing before the run (mean-prototype per-tick
accuracy 0.613 → median-of-3 latch correctness 0.667; a third of latches
land on the wrong object outright, and the anchor band then locks the
error in). The mechanism check confirms it is a binding failure and not
something else: source-uv std 0.321, statistically unchanged from the
no-binder 0.329 and the text-rerank 0.339 — three different binders,
three identical flicker levels, because all three score against vectors
whose object identity is drowned by the crops' dominant common
component. Closest approach 0.098 m (vs 0.105 no-binder): no movement.

This is the negative that makes the 1-NN bank test decisive. Same
vectors, same episodes, same machine — only the estimator changes
(0.613 → 0.902 per-tick, 0.667 → 0.914 latch). If bank-bound cream
crosses zero, binding was the whole boundary and the fix costs no
training. If it does not, my diagnosis is wrong in a locatable way and
the telemetry (uv std vs 0.33) says immediately which half failed.
Prototype butter/soup cells running; bank trio behind them.

### Prototype binder degrades the working objects too: butter 2/10 (2026-08-06)

**Prototype-bound butter: 2/10** — against 7/10 for the same head under
text rerank and 4/10 with no binder at all. The mean-prototype binder is
not merely useless for the confusable object; it actively *replaces*
correct bindings with wrong ones on an object whose text-match already
worked. That is the sharpest possible statement of the estimator problem
measured earlier: with raw prototype cosines at 0.986–0.991 across three
objects, a nearest-prototype rule is close to arbitrary, and arbitrary
binding beats no binding only by accident.

The four-binder ladder so far (three-object v8 head, anchor band, n=10):

| binder | cream | butter | soup |
|---|---|---|---|
| none | 0/10 | 4/10 | 5/10 |
| text rerank | 0/10 | **7/10** | 5/10 |
| mean prototype (0.613 identity acc.) | **0/10** | **2/10** | running |
| 1-NN bank (0.902 identity acc.) | running | running | running |

The bank cells are the test that matters: same vectors, same episodes,
only the estimator changes.

### Binder quality tracks measured per-object separability — prototype trio complete (2026-08-06)

**Prototype-bound soup: 8/10** (`unaided_pr_soup`) — the best soup cell
this head has produced, against 5/10 with no binder and 5/10 with text
rerank. The prototype trio now reads cream 0/10, butter 2/10, soup 8/10,
and it lines up exactly with the separability measured *before* any of
these runs:

| object | centered prototype separation | 1-NN identity acc. | prototype-binder cell |
|---|---|---|---|
| soup | −0.75 / −0.66 vs others | 0.95 | **8/10** (best of its column) |
| butter | −0.66 vs soup, **+0.00 vs cream** | 0.90 | 2/10 (down from 7/10 rerank) |
| cream | **+0.00 vs butter** | 0.82 | 0/10 |

That is the cleanest statement of the binding story yet: a binder helps
exactly as much as the embedding space lets it discriminate. Where the
object's prototype direction is well separated (soup), even a crude mean
prototype beats the text tower and lifts success by three trials. Where
two objects share a null direction (butter–cream, centered cosine 0.00),
the same binder is close to a coin flip and *destroys* a working cell.
Nothing about the head, the corpus or the shell changed across those
three cells — only which box the machine believed was the target.

Prediction for the bank trio, on record: soup should hold high, butter
should recover past 2/10 (1-NN separates it at 0.90 where the mean
prototype does not), cream is the open question at 0.82.

### PREDICTION FALSIFIED: bank-bound cream 0/10, and the failure is locatable (2026-08-06)

I predicted in writing, before the run, that 1-NN bank binding would put
cream above zero (0.82 per-tick identity accuracy → 0.914 median-of-3
latch correctness). **It did not: bank-bound cream is 0/10**
(`libero_object_real_1785993850945`). The prediction is wrong and the
binder-accuracy → deployed-success chain I built on it does not hold as
stated.

The telemetry says exactly which half broke. Source-uv std under bank
binding is **0.328** — against 0.329 with no binder, 0.339 with text
rerank, 0.321 with mean prototypes. Four binders, four identical flicker
levels. The bank binder, whose offline identity accuracy is 0.902
overall and 0.82 on cream, **does not change what the deployed machine
binds at all**. Closest approach is unchanged too (0.107 m vs 0.105
no-binder), and no trial gets inside 6 cm.

So the offline measurement and the deployed behaviour have come apart,
and that gap — not the cream cell itself — is now the finding to chase.
Two candidate explanations, both testable and neither yet tested:
(1) the binder is not engaging on the live path (proposals may not reach
the observe branch with usable embeddings, in which case `src` never
changes and every "binder" cell has been measuring the same policy —
which the four identical uv-std numbers are consistent with, and which
would ALSO explain why the prototype cells differed only by noise); or
(2) it engages but the live crop embeddings sit off the corpus manifold,
so a bank built from teacher episodes cannot match them — the same
on-manifold limit §6 documents for substitution probes, now biting the
binder.

Explanation (1) is embarrassing and cheap to check, so it goes first: log
the chosen proposal index per tick with and without `--goal-src-bank` on
one episode and compare. If the indices are identical, the binder never
fired and three of tonight's cells (proto cream/butter/soup, bank cream)
must be re-labelled as replicates of the no-binder policy rather than
binder tests — with the prototype-soup 8/10 and prototype-butter 2/10
spread re-read as n=10 sampling noise around the same policy, not as
evidence that binder quality tracks separability. That re-reading would
retract the "binder gain tracks per-object separability" claim logged
earlier tonight; it stays in the log either way, marked as contingent on
this check.

### The engagement probe: the binder DOES fire — and that makes the result worse, not better (2026-08-06)

A/B probe on one episode (task 1, seed 20, identical flags except
`--goal-src-bank`), logging the uv actually fed to the goal head on every
real tick:

* **no binder:** 14 supervised ticks, uv mean (0.267, 0.733), std (0.258, 0.047)
* **bank binder:** 45 supervised ticks, uv mean (0.321, 0.677), std (0.335, 0.202)
* the two sequences agree on ticks 1–4, then **diverge** (tick 5:
  (0.190, 0.711) vs (0.134, 0.801); tick 8: (0.120, 0.752) vs (0.131, 0.802))

So hypothesis (1) is **refuted**: the binder engages, changes which box
is bound, and changes the resulting trajectory. Tonight's binder cells
were real tests, not replicates — the "binder gain tracks separability"
entry stands as a description of the cells, though its mechanism claim is
now weaker (see below).

Hypothesis (2) is what survives, and it is the sharper finding. The bank
binder does not stabilise the stream — it **destabilises** it: uv std
rises 0.258 → 0.335 laterally and 0.047 → 0.202 vertically. A binder
measured at 0.902 leave-episode-out identity accuracy on corpus crops is,
on live crops, thrashing between boxes tick to tick. The offline number
does not transfer.

That is the paper's own §6 on-manifold limit, reproduced independently on
a second instrument. The substitution probe evaluates on teacher
trajectories and cannot certify off-manifold behaviour; the crop bank is
built from teacher trajectories and cannot bind off-manifold crops. Two
different instruments, same failure mode, both measured rather than
argued. The deployed machine visits states the corpus never contains, and
*every* corpus-derived instrument we have built inherits that blind spot.

Consequences, stated plainly. (a) My latch-correctness prediction was
wrong because it assumed the offline per-tick accuracy was the deployed
per-tick accuracy; it is not, and I have no measurement of the deployed
one (measuring it needs ground-truth object identity per bound box at
eval time — a new instrument, pre-registered as future work rather than
run tonight). (b) Cream's 0/10 across four binders is therefore *not*
evidence that cream is unbindable — it is evidence that all four binders
are corpus-derived and none of them transfers. (c) The honest scope for
the addendum: object breadth is bounded by role binding, the bound is
quantified offline at 0.902/0.613, and the transfer of those numbers to
deployment is an open measurement, not a demonstrated fact.

### Bank-butter 4/10 — exactly its no-binder cell (2026-08-06)

**Bank-bound butter: 4/10** (`eval_results/unaided_bk_butter`), against
4/10 with no binder at all, 7/10 with text rerank, 2/10 with mean
prototypes. The 0.902-accuracy binder returns butter to precisely the
baseline it started from, and independently corroborates the engagement
probe: a binder that thrashes tick-to-tick on live crops neither helps
(cream 0/10) nor is it merely inert — it undoes the gain the far cruder
text rerank achieved (7/10 → 4/10).

Three-object ladder, all n=10 on the v8 head with the anchor band:

| binder | offline identity acc. | cream | butter | soup |
|---|---|---|---|---|
| none | — | 0/10 | 4/10 | 5/10 |
| text rerank | (not measured offline) | 0/10 | **7/10** | 5/10 |
| mean prototype | 0.613 | 0/10 | 2/10 | **8/10** |
| 1-NN bank | 0.902 | 0/10 | 4/10 | running |

Read honestly, the column that matters is the ordering: **offline binder
accuracy does not predict deployed success at all**. 0.613 gives the best
soup cell of the night and the worst butter cell; 0.902 returns butter to
baseline and leaves cream at zero; the binder with no offline number at
all (text rerank) gives the best butter cell. Whatever these binders are
doing on live crops, it is not what the corpus-crop benchmark measures.
That is the finding — and it is the same on-manifold gap §6 documents for
substitution probes, now shown twice more.

### FINAL: the four-binder × three-object matrix, all twelve cells verified (2026-08-06)

Every cell n=10, v8 three-object head, anchor-band machine, seed 20,
verified against its results.json:

| binder | offline identity acc. | cream | butter | soup |
|---|---|---|---|---|
| none | — | 0/10 | 4/10 | 5/10 |
| text rerank (crop-CLIP) | not measured | 0/10 | **7/10** | 5/10 |
| mean prototype (centered) | 0.613 | 0/10 | 2/10 | **8/10** |
| 1-NN bank (centered) | 0.902 | 0/10 | 4/10 | 6/10 |

run ids — pr: 1785990569408 / 1785991819037 / 1785992897481;
bk: 1785993850945 / 1785995159772 / 1785996395740.

**The result is a dissociation, and it is the honest headline.** Rank the
binders by offline identity accuracy (0.613 < 0.902) and by deployed
success and the orders do not match on any object: the 0.613 binder owns
the best soup cell (8/10) and the worst butter cell (2/10); the 0.902
binder sits at baseline on butter (4/10) and mid on soup (6/10); the
binder with no offline number at all wins butter (7/10). Cream is 0/10 in
all four columns. Spearman over the two ranked binders is undefined at
n=2 per object, so the claim is stated qualitatively and no significance
is asserted — but the pattern is consistent across three objects and
matches the engagement probe's direct measurement (the bank binder raises
uv std 0.258 → 0.335 rather than lowering it).

What this closes and what it does not. It closes the question "is cream a
binding problem you can fix with a better corpus-derived binder?" — the
answer is no, and the reason is measured rather than assumed: every
binder we can build from teacher episodes is evaluated on a distribution
the deployed machine does not visit. It does not close "is cream
bindable" — that needs a binder whose training distribution includes
deployed states (DAgger-style crop collection under the machine's own
policy), which is now the top pre-registered item.

**Tonight's falsified prediction, kept in the record.** I predicted
bank-cream > 0 from 0.82 offline accuracy → 0.914 latch correctness. It
returned 0/10. The error was assuming offline per-tick accuracy equals
deployed per-tick accuracy; the engagement probe shows it does not, and I
have no measurement of the deployed quantity. That prediction, its
falsification, and this attribution stay in the paper — a negative
result with a located cause is worth more than the positive one I was
chasing.

### Submission package built (2026-08-06)

`paper/submission/`: `main.tex` (anonymized, plain `article`, 4 pp),
`refs.bib` (18 entries), compiled `main.pdf` — clean build, zero LaTeX
warnings, zero undefined citations. Content is the audited subset only:
the parameter ledger, the protocol × head table with both n=50
confirmations and the two-stack controls, the rebuild matrix, the
multi-object addendum including the four-binder dissociation table, and
the falsified prediction stated in the abstract rather than buried.
Claims/non-claims/limitations transcribed verbatim in spirit from
MANUSCRIPT_v2 §9–10.

Explicitly NOT done, and recorded as such: figures not yet placed in the
LaTeX (they exist and are regenerable); no venue style file applied
pending a venue choice; and **no submission has been made** — that
requires the author's decision and credentials, and no artifact here may
be represented as submitted or reviewed.

### Closing the last measurement gap: deployed binding accuracy (2026-08-06, in progress)

The binder study's dissociation (offline 0.613/0.902 vs. deployed cells
that do not follow the ordering) left exactly one quantity unmeasured:
**how often the deployed machine binds the correct object, live.** I said
in the falsification write-up that measuring it needs a new instrument;
this is that instrument.

Ground truth comes from MuJoCo instance segmentation, not from the text
tower (which would be circular) and not from the corpus (which is the
distribution under suspicion): for each detector proposal at each real
tick, the modal segmentation id inside the box names the object the box
actually covers. Segmentation is read for diagnosis only and never
reaches a controller — the same discipline the teacher-calibration
telemetry is held to.

With that number in hand the addendum's claim becomes falsifiable in the
right place: if deployed binding accuracy for cream is at chance while
its offline accuracy is 0.82, the on-manifold gap is quantified rather
than inferred from uv std; if deployed accuracy is high and cream still
fails, binding is exonerated and the block moves downstream — which
would contradict the current §7 text and require it to be rewritten.
Probe running (`scripts/seg_probe.py`, task 1).

### Instrument built: deployed binding accuracy via camera projection (2026-08-06)

MuJoCo instance segmentation is unusable on this stack (robosuite 1.4.1
writes 256 into a uint8 buffer; numpy 2 raises `OverflowError` and the
observable is dropped — logged as a stack defect, not a MicroVLA one).
The instrument therefore uses geometry instead of semantics: every task
object's world position is projected into the wrist camera with
robosuite's own `get_camera_transform_matrix` /
`project_points_from_world_to_camera`, and the box the machine actually
bound is scored correct iff the target object's projected pixel is the
nearest object pixel to that box's centre. Sim state is read for
diagnosis only and never reaches a controller — the same rule the
teacher's calibration telemetry is held to.

`scripts/bindacc.py <task> <object> [--bank KEY]`, running now for cream
(task 1) and butter (task 6). Pre-committed reading of the outcomes:
* cream accuracy near chance (~1/3 with three candidates) ⇒ the
  on-manifold gap is quantified at the binding stage and §7's claim
  stands with a number rather than an inference from uv std;
* cream accuracy high (≥0.8) with cream still 0/10 ⇒ binding is
  exonerated, the block is downstream, and §7 is wrong and must be
  rewritten;
* butter accuracy materially above cream's ⇒ explains why the same
  binders help one object and not the other.

One process note, logged because the paper is partly about exactly this:
my first probe imported `libero` directly and hung on its interactive
first-run prompt, which the repo already solves via
`eval/_libero_compat.prepare_libero()`. The harness had encoded the fix
and I bypassed it — a one-line instance of the provenance failure mode
§8 catalogues.

### Deployed binding accuracy, cream: 0.033 — with the instrument's own control pending (2026-08-06)

**Cream (task 1), no binder, deployed: 2/60 ticks correct = 0.033.** The
misbinding is systematic, not diffuse: the box the machine treats as
"cream cheese" is nearest to *milk* on 28 ticks and to *alphabet soup* on
23, with cream itself winning twice. Against the 1-NN bank's 0.82
leave-episode-out accuracy on corpus crops, that is the on-manifold gap
quantified at last — roughly a 25x drop from corpus to deployment, and
below the ~0.2 chance rate for five candidate objects, i.e. binding is
actively anti-correlated with the target rather than merely noisy.

**This number is not yet believable, and the reason is in this paper's
own protocol section:** "instrument calibration is part of the protocol,
because across this project the instruments were wrong more often than
the models." A ground-truth pixel projection has two conventions I could
have inverted — (row, col) vs (u, v), and normalisation by width vs
height — and either mistake manufactures exactly this result. So the
control is running: the identical measurement on **soup (task 0)**, the
object whose policy demonstrably works at 5-8/10. If soup's deployed
binding accuracy comes back high, the instrument is calibrated and
cream's 0.033 stands. If soup also comes back near zero, my projection
convention is wrong, the cream number is an artifact, and it gets
retracted rather than reported.

### Butter's deployed binding accuracy: 0.393 — and it still succeeds (2026-08-06)

**Butter (task 6), no binder, deployed: 44/112 ticks = 0.393**, misbound
onto ketchup (46) and orange juice (21). Two things follow, and the
second is the more interesting.

First, the instrument discriminates. Butter's 0.393 is 12x cream's 0.033
on the same code path, so a uniformly-broken projection convention is
ruled out — the remaining calibration question is only whether the
absolute scale is right, which the soup control settles.

Second, **butter succeeds at 4-7/10 while binding correctly on well
under half of ticks.** Deployed binding accuracy is therefore not the
determinant of success that my falsified prediction assumed; the machine
tolerates substantial misbinding. The mechanism is already in the
architecture: the latch consumes a *median over a window* and the anchor
band rejects estimates far from the first stable median, so early
correct binds can carry an episode that later misbinds. That is a
concrete, testable account of why 0.902-accurate offline binding bought
nothing (it changes ticks the latch has already stopped listening to)
and why cream's 0.033 is nevertheless fatal (there is no early correct
cluster to anchor on).

This also retires the last framing from the falsification: the right
quantity was never per-tick accuracy alone but *accuracy within the
pre-latch window*. That is measurable with the same instrument by
restricting the tick set, and is queued.

### RETRACTION: the calibration control failed; the binding-accuracy numbers are artifacts (2026-08-06)

**Soup control: 12/108 = 0.111.** Soup is the object this system grasps
at 35/50 held-out. An instrument that scores the *working* object at
0.111 is measuring something other than binding correctness, so by the
condition I recorded before running it, **the deployed binding-accuracy
numbers are retracted**: cream 0.033 and butter 0.393 do not stand, and
nothing in the manuscript may cite them.

The bug is diagnosable and mine. `project_points_from_world_to_camera`
happily projects points that are *behind* the camera or outside the
frame, returning pixel coordinates that are geometrically meaningless.
On a wrist camera during descent — where the target sits centimetres from
the lens and half the scene is behind it — that is not an edge case, it
is the common case. Every "nearest object" verdict computed from such
points is noise, which is why the ordering (butter 0.393 > soup 0.111 >
cream 0.033) matches neither success (soup 0.70 > butter 0.4-0.7 >
cream 0) nor anything else.

What this costs and what it saves: it costs the quantified on-manifold
gap I claimed one entry ago, which is now unmeasured again and marked as
such. It saves the manuscript from carrying a fabricated 25x figure that
a reviewer with a camera-geometry background would have caught
immediately — and it is the third time tonight the protocol's own
instrument-calibration rule has caught an instrument rather than a model.

Fix, queued: (1) filter projections to points with positive camera-frame
depth and in-frame pixels; (2) restrict scoring to pre-latch ticks, which
the butter/soup contrast suggested is the decision-relevant window
anyway; (3) re-run soup FIRST and require high accuracy before any cream
or butter number is believed or reported.

### Instrument v2, calibration-first (2026-08-06)

`scripts/bindacc2.py` implements the three fixes the retraction called
for: (1) a point counts only if its camera-frame depth is positive and
its pixel lands inside the frame — computed from the extrinsic matrix,
which is what v1 skipped; (2) accuracy is reported twice, over all ticks
and over **pre-latch ticks only**, the window the median-of-window latch
actually consumes; (3) soup runs first and alone.

The acceptance rule is fixed before the run and is deliberately harsh:
soup's pre-latch accuracy must come back **high** (the object succeeds
35/50; if the instrument cannot see that, the instrument is still wrong)
before any cream or butter number is computed, believed, or written
down. If soup fails again the geometric approach is abandoned rather than
tuned — tuning a ground-truth instrument until it agrees with the
hypothesis is precisely the failure this project keeps catching.

### v2 rejected 100% of points; settling the convention without touching the hypothesis (2026-08-06)

Instrument v2 returned `ticks=0` on the soup calibration: the
depth-positivity filter rejected every projected point, median visible
objects 0.0. So a sign or matrix-direction convention is inverted —
`get_camera_extrinsic_matrix` may be camera-to-world or its inverse, and
MuJoCo/OpenGL cameras look down $-z$ while other stacks use $+z$.

The tempting move is to flip the sign and re-run until the binding
numbers look sensible. That is exactly the failure this project keeps
cataloguing, and the acceptance rule I wrote one entry ago forbids it.
So the convention is being settled by a test that **cannot** be biased by
what I hope to find about binding: project world points to pixels, then
invert back to world with robosuite's own
`transform_from_pixels_to_world`, under each candidate convention. The
convention that round-trips is the correct one, and that verdict is pure
geometry — it never mentions cream, butter, soup, or binding.

If neither convention round-trips, the geometric ground truth is
abandoned as promised and deployed binding accuracy stays unmeasured
(and so labelled) rather than being estimated by an instrument I cannot
validate. `scripts/convcheck.py`, running.

### Convention settled by measurement, not by trial: depth is +z (2026-08-06)

The convention check answered the question with a quantity that has
nothing to do with binding: scene objects sit at camera-frame
z = +0.274 … +0.375 when the wrist camera is looking at them, so **+z is
forward** and v2's `depth = -z` filter was inverted — which is why it
rejected 7/7 points. The fix is a sign determined by measuring the actual
coordinates of objects known to be in front of the lens, not by trying
signs until the binding numbers pleased me.

Two honesty notes. First, the round-trip validator I wrote to be the
*primary* arbiter crashed on my own misuse of
`transform_from_pixels_to_world` (wrong depth-map shape); I am therefore
NOT claiming a round-trip validation, only the z-range measurement.
Second, the sign fix does not by itself make the instrument
trustworthy — the pre-registered gate is unchanged and still binding:
**soup's pre-latch accuracy must come back high, or nothing downstream is
believed.** Soup calibration relaunched with the corrected sign.

### ABANDONED per the pre-registered rule: deployed binding accuracy stays unmeasured (2026-08-06)

With the depth sign corrected the geometry now works — median 7 objects
pass the depth+frame filter per tick — and soup still fails its gate:
**pre-latch 3/33 = 0.091**, all-tick 12/108 = 0.111, on the object this
system grasps 35/50. The acceptance rule I fixed in advance said: if soup
fails again, abandon the geometric ground truth rather than tune it. So
it is abandoned, and **deployed per-tick binding accuracy is recorded as
UNMEASURED** — not estimated, not approximated, not cited anywhere in the
manuscript.

The flaw is now identifiable and is intrinsic to the design, not another
sign error. Ground truth was "nearest projected *body origin* to the
bound box centre," and the in-frame filter deletes exactly the wrong
points: during descent the target fills the wrist view and its origin
falls *below the frustum*, so the target is filtered out while distant
distractors survive and win by default. That is why soup misbinds onto
butter on 60 ticks — an artifact of which origins happened to be
in-frame, not of what the machine bound. Filtering that biases against
the object the camera is centred on cannot measure binding, at any sign
convention.

A different instrument could work — project each object's geom AABB
corners and score by overlap with the detection box, which survives the
close-up case — but building it now, after two failed versions, is
instrument-shopping until one agrees with me. So it is **pre-registered
for future work with its own gate** (soup pre-latch accuracy ≥ 0.8 before
any other object is scored), and tonight's claim set is unchanged: the
on-manifold gap between corpus-crop binder accuracy (0.613/0.902) and
deployed behaviour is demonstrated qualitatively by the four-binder
dissociation and the uv-destabilisation probe, and is *not* quantified.
Two instruments were built, both failed their own calibration, and both
failures are reported.

### Fourth object launched: chocolate pudding, chosen by the binder study's own data (2026-08-06)

With the binding instrument abandoned, the remaining lever for object
breadth is another object — and which one is not a guess. The binder
study's misbinding tables name the confusable pairs: cream's misbinds
were milk (28) and soup (23); butter's were ketchup (46) and orange juice
(21). **Chocolate pudding almost never won a misbind in any table**,
which is the closest thing to a separability prior we have that was not
fit for this purpose. So task 8 (chocolate pudding) is the fourth-object
attempt, and the prediction on record is that it behaves like butter (it
crosses) rather than cream (it does not).

`scripts/obj4.sh` runs the full measured recipe with no shortcuts: offset
calibration to at-close convergence (<1 cm, the loop-flaw fix from the
butter re-diagnosis), a 3-trial teacher smoke gate at ≥1/3, a 25-target
±3 cm randomized corpus, a **four-object** head (soup + butter + cream +
pudding) trained with eef-jitter, then n=10 unaided cells on pudding,
butter, and soup under the one anchor-band + rerank configuration.

The soup cell is the one that matters most for honesty: each added object
dilutes the corpus, and if soup degrades materially as objects are added,
that is the real cost of breadth at this scale and it gets reported as
prominently as any gain.

### Fourth object: prediction wrong again, and the same stage is implicated (2026-08-06)

I predicted chocolate pudding would behave like butter (cross) because it
almost never won a misbind in the binder tables. **It behaves like
cream.** Calibration iterations 1 and 2 both return the
never-reached-grasp sentinel with crop-CLIP rerank already enabled — the
lever that rescued cream's teacher — and the telemetry names the same
stage:

* phase sequence never leaves `servo_src` (225 ticks, no align, no grasp);
* source-uv std **(0.303, 0.200)** — the binding-flicker signature, at the
  same magnitude as cream's 0.33;
* detection duty 0.5, so the detector fires but on inconsistent boxes;
* the arm parks at (−0.271, +0.060) while the pudding sits at
  (−0.120, −0.240) — roughly 34 cm away, i.e. it is servoing to something
  else entirely.

Two consequences for the paper's claims, both tightening rather than
loosening them. First, my selection prior was wrong: "rarely won a
misbind" does not predict bindability, because an object that is never
*mistaken for* the target can still fail to be *found* — the tables
measured the wrong direction of the confusion. That is a second falsified
prediction tonight and it is recorded as such. Second, the boundary is
now attested on two independent objects (cream, pudding) with the same
signature at the same stage, which is stronger evidence for the
role-binding account than one object was: of four objects attempted,
**two cross and two are binding-blocked**, and the split is not explained
by grasp geometry, corpus size, or goal accuracy.

The chain will run its remaining calibration iterations and stop at the
smoke gate, as designed. No pudding corpus will be recorded, so the
four-object head is not built — correctly, because the teacher cannot
produce data for an object it never reaches.

### The decisive perception test, with no ground truth required (2026-08-06)

Two objects now fail at the same stage with the same signature, and both
diagnoses so far have leaned on instruments that needed calibration. This
test needs none: on a single wrist frame, set the frozen detector's
classes to each of the ten LIBERO-Object phrases in turn and record which
box each phrase selects. No labels, no policy, no binder, no projection —
just the text tower answering ten questions about one image.

The reading is fixed in advance:
* if distinct phrases select **the same box**, role binding is impossible
  by construction for those objects, and cream and pudding are explained
  without appealing to anything learned — the boundary is a property of
  the frozen encoder, which is the strongest and simplest form of the
  claim;
* if each phrase selects a **different** box, the text tower discriminates
  fine and the failure is downstream in how the machine consumes it —
  which would contradict the role-binding account I have been building
  and require §7 to be rewritten around whatever the consumption defect
  turns out to be.

Run on task 8 (pudding, blocked) and task 0 (soup, works) so the
comparison is internal: `scripts/textdisc.py`.

### Third instrument failure tonight, caught before it became a finding (2026-08-06)

The text-discrimination probe returned conf=0.000 at uv=(0.500,0.500) for
all ten phrases — which is not "every phrase picks the same box" but the
detector's documented **fallback BoxObs for no detection at all**. Read
carelessly it looks like a spectacular result (the text tower is
degenerate!); read correctly it says my probe frame is bad, because the
same detector fires at 0.5 duty in the very run this probe was meant to
explain.

Recorded as a near-miss because the difference matters: the fallback
returns a *centred* box with zero confidence, so a probe that does not
check confidence will silently treat "nothing detected" as "everything
detected in the same place." That is a producer/consumer convention trap
of exactly the class §8 catalogues — the sentinel and the signal share a
shape — and it is the third instrument to fail tonight (after the
origin-projection ground truth and its sign-corrected successor).

Frame diagnostics (shape/min/max/mean) added and the probe rerun; if the
frame is black or mis-keyed the fix is mechanical, and if the frame is
fine but detections are still empty at conf 0.02 then the probe's
preprocessing (flip/BGR order) differs from the eval path and must be
made identical rather than approximated. No conclusion about the text
tower is drawn until the probe reproduces the eval path's detection duty
on a frame from the same distribution.

### The control invalidated the probe, exactly as designed (2026-08-06)

The soup scene returned zero detections for all ten phrases too. Soup is
the object this system grasps 35/50, so a probe blind on soup is blind,
full stop — **the text-tower probe v1 is invalid and no claim about the
text tower is drawn from it.** The internal control did precisely the job
it was included for, one step before an exciting conclusion.

The confound is now identifiable: v1 built its frame from a freshly reset
env at the arm's home pose, where the wrist camera sees table and gripper
rather than the objects. The detector was never given a frame from the
distribution it works on. Preprocessing was not the issue — the probe's
row-flip and BGR order match `microvla/utils/camera.py::upright` and the
policy's `frame_rgb[..., ::-1]` exactly, which I checked rather than
assumed.

v2 removes the environment from the instrument entirely and reads
`wrist_frames` straight out of the recorded corpora — by construction the
frames the detector succeeds on, from the same episodes that trained the
heads. Its gate is fixed in advance and is the same shape as before: the
soup corpus frame must produce a confident "alphabet soup" detection, or
v2 is invalid too and nothing is claimed. Three instruments have now
failed their calibration tonight and each failure is in this log; the
alternative — reporting the first result that agreed with my hypothesis —
was available at every step and is what the audit half of this paper
exists to prevent.

### STOP: the text-tower question is recorded as open, not answered (2026-08-06)

v3 built perception through the eval path's own `_build_real_perception`
factory (grid 4, min_side 512, identical flags) and read frames straight
from the corpora, and **still fails its soup gate** — the object's own
phrase does not detect on frames from the episodes that trained on it.
Three variants, one gate, three failures. Per the timebox fixed before
the run, instrument-building on this question stops here and **whether
the frozen text tower discriminates LIBERO-Object's phrases is recorded
as OPEN** — not answered, not estimated, not hinted at in the manuscript.

What is now known about the failure, for whoever picks it up: it is not
the frame source (env and corpus both fail), not the preprocessing
(verified identical to `camera.py::upright` and the policy's BGR
conversion), and not the perception constructor (the eval factory fails
too). The remaining suspects are the corpus frames' stored channel order
or dtype — my `[..., ::-1]` assumes RGB storage and would silently
produce BGR-of-BGR if they are stored converted — and any priming the
live loop performs before `perceive` that a bare call does not. Both are
checkable in minutes by someone starting fresh; I have now been wrong
about this instrument three times and the value of my next guess is low.

Tally for the night, stated plainly because it is the honest headline of
this stretch: four instruments attempted, four failed their own
calibration gates, zero fabricated results reported. The measurements
that stand are the ones from before this stretch, unchanged.

### Fourth object closed as a clean negative: 2 of 4 objects cross (2026-08-06)

The pudding chain ran to completion and stopped exactly where it should:
five calibration iterations, all returning the never-reached-grasp
sentinel, then a 3-trial teacher smoke at **0/3**, then halt — no corpus
recorded, no four-object head trained, because a teacher that cannot
reach the object cannot produce data for it. The gate did its job without
supervision.

**Final object tally for the campaign: four attempted, two cross
(alphabet soup, butter), two blocked (cream cheese, chocolate pudding),
and the two failures share a signature** — the teacher never leaves
`servo_src`, source-uv std ~0.30, and the arm parks tens of centimetres
from the target, with crop-CLIP rerank enabled in both cases. Neither
failure is a grasp-geometry failure (the butter re-diagnosis showed what
that looks like: contact, squeeze-out, a measurable at-close residual),
and neither is a goal-accuracy failure (cream's on-manifold goal error is
1.58 cm, comparable to butter's 1.34).

The honest scope statement the paper now carries: at 30M parameters with
a frozen detector as the only vision *and* text encoder, object breadth
on this benchmark is limited by role binding, that limit is attested on
two independent objects, and it is *not* liftable by any of the four
binders tried (none, crop-CLIP rerank, mean prototype, 1-NN bank). What
remains unmeasured — and is now explicitly labelled unmeasured rather
than inferred — is the deployed binding accuracy itself and whether the
frozen text tower discriminates these phrases at all.

### Separating two failure modes I had been conflating (2026-08-06)

Re-reading tonight's record exposes a distinction my own summaries blurred:
cream and pudding do not fail the same way.

* **Cream:** the *teacher* works once crop-CLIP rerank is enabled
  (calibration converged in two iterations, smoke 1/3, 17-episode corpus
  recorded). The *student* is what fails. So cream is a student-side
  binding failure, and the four-binder study is exactly the right probe
  for it.
* **Pudding:** the *teacher* fails, with rerank already on, across five
  calibration iterations. Nothing student-side was ever exercised,
  because no corpus exists. Calling both "the same signature" was
  correct about the telemetry (`servo_src`, uv std ~0.30) and sloppy
  about the level.

That matters because it changes what is untested. For pudding the
untried suspects are the *perception filters*, not the text tower:
`--source-max-area 0.12` rejects source boxes above 12% of frame area
and `--role-disjoint-iou 0.1` rejects source boxes overlapping the
target's — either could delete the correct pudding box before any
binding decision is made, and both were inherited unchanged from the
soup-era configuration. A four-cell ablation is running (baseline / no
area filter / no disjoint filter / neither), single trial each, scored by
whether the teacher reaches a grasp phase at all.

If a filter is the block, pudding is not evidence for the role-binding
boundary and the 2-of-4 tally must be restated as 2-of-3 with pudding
pending — which would weaken the strongest multi-object claim I made
tonight. Recording that before the result.

### Filter ablation: pudding's block is upstream of the filters (2026-08-06)

Four cells, one trial each, scored by whether the teacher reaches a
grasp phase at all:

| cell | source_max_area | role_disjoint_iou | reached grasp | uv std | closest | duty |
|---|---|---|---|---|---|---|
| baseline | 0.12 | 0.1 | no | (0.289, 0.238) | 0.322 | 0.50 |
| no area filter | 0.0 | 0.1 | no | (0.289, 0.238) | 0.322 | 0.50 |
| no disjoint filter | 0.12 | 0.0 | no | (0.311, 0.211) | 0.322 | 0.50 |
| neither | 0.0 | 0.0 | no | (0.289, 0.238) | 0.322 | 0.50 |

The area filter is provably inert here — its cell is identical to
baseline to three decimals, so no pudding box was ever being rejected by
it. The disjoint filter *does* change which boxes survive (uv std moves
0.289→0.311 laterally, 0.238→0.211 vertically) and changes nothing that
matters: same failure, same 0.322 closest approach, same 0.50 detection
duty. Removing both together reverts exactly to baseline.

So the pre-registered weakening does **not** trigger: pudding's block is
upstream of the source-selection filters, and the 2-of-4 tally stands as
recorded. What the ablation adds is a sharper localisation — the failure
survives every knob that decides *which detected box becomes the source*,
which places it in detection or in the text-conditioned scoring that
precedes those knobs. That is the same region the (unmeasured)
text-tower question lives in, and it is now the single named open
question of the object-breadth result rather than one of several.

### Text-tower question closed as OPEN, with the suspect narrowed to one call path (2026-08-06)

The convention check settles the last cheap hypothesis: on a soup corpus
frame (60x256x256x3 uint8, range 0-255), the eval-factory detector with
the phrase "alphabet soup" returns **conf=0.000 under all four
combinations** of row-flip and channel order, at two different frame
depths. Not channel order, not flip, not frame source, not the
constructor. Five instrument attempts, all failing the same gate.

The question stays OPEN, and this is the last time it is touched tonight.
What the failures collectively narrow it to, for whoever resumes: the
live path never calls `perceive()` bare — it goes through
`JEPALoop.tick`, and the task's phrases are installed by
`ClipTaskEncoder`, which harvests CLIP text embeddings from the
detector's own tower *once per task*. A bare `set_classes()` may leave
the model without the text state that path establishes, in which case
every probe I wrote tonight was asking a detector that had never been
told what to look for. That is a single, checkable hypothesis with an
obvious test (drive one episode through the real loop and log
`perceive`'s inputs and outputs directly, rather than reconstructing the
call), and it explains all five failures with one cause.

Recording it as a hypothesis, not a finding, because I have not tested
it — and because tonight's tally is already four wrong instrument
designs and two falsified predictions, which is exactly the rate at which
a sixth guess should be someone else's first.

### RESOLVED: the object-breadth boundary is the generic-prompt tail, and it was documented all along (2026-08-06)

Instrumenting `perceive()` inside a real episode — instead of designing a
sixth probe — resolved it in one run. The production call is not
`set_classes([phrase, "basket"])`. It is `set_role_prompts` with a
preference chain:

```
classes = ['alphabet soup', 'soup', 'box', 'cardboard box', 'can', 'basket', 'bin']
role_ids = [[0,1,2,3,4], [5,6]]        # source takes the FIRST prompt that fires
```

Live source confidences are 0.052-0.114. My five probes asked the bare
product phrase alone and got nothing — which is not an instrument bug at
all but **the finding**, and `microvla/perception/prompts.py` states it
in its own docstring: *"YOLO-World's region-text head scores 0.000 on the
product names LIBERO tasks are written in ('alphabet soup', 'cream
cheese')"*. The system detects groceries only via a generic tail —
`("box", "cardboard box", "can")` — chosen by measured firing rate.

That closes the boundary question with a mechanism, not a suspicion.
Every LIBERO-Object grocery resolves to the SAME tail, so once the
product phrase fails (which it usually does), the source role is bound by
"box"/"can" — words that describe cream cheese, chocolate pudding,
butter, and milk equally well. Discrimination is therefore impossible for
box-shaped objects *by construction of the prompt chain*, and every
symptom follows: uv std ~0.30 (several boxes matching equally), cream
misbinding onto milk and soup, butter onto ketchup and orange juice,
cream-vs-butter centred prototype cosine ~0.00 (both are boxes), and four
corpus-derived binders unable to fix it — they re-rank boxes the text
stage already conflated.

Why soup and butter cross anyway: soup is the one object whose tail
contains a *shape-specific* winner ("can") that no other scene object
matches, and butter's teacher was rescued by crop-CLIP rerank operating
on a scene where its distractors are bottles and jars. Cream and pudding
are boxes among boxes, with no discriminating prompt available.

Two corrections to my own record. (1) The five "failed instruments" were
mostly measuring something real; I misread their null result as a bug
because I assumed the bare phrase was what production used. The
projection instruments were genuinely broken; the text probes were asking
an honest question and getting an honest answer. (2) The right statement
of the boundary is therefore sharper than "role binding fails": **object
breadth is limited by the prompt chain's inability to discriminate
same-shaped groceries, because the frozen text tower scores 0.000 on the
product names themselves.** That is testable, already partly measured in
this repo, and names its own fix (a discriminating prompt or a
detector whose text tower grounds product names).

### Prompt screening measures the boundary, and names one candidate fix (2026-08-06)

Screened on 24 real cream frames from 3 corpus episodes, using the
production `set_role_prompts` call:

| candidate | firing rate | mean conf | box-centre spread |
|---|---|---|---|
| "cream cheese" (the task's own words) | **0.00** | — | never detected |
| "box" (incumbent tail) | 0.92 | 0.181 | (0.270, 0.155) |
| "white box" | 0.75 | 0.096 | (0.271, 0.230) |
| **"white carton"** | 0.17 | 0.085 | **(0.057, 0.072)** |
| "small white box" | 0.25 | 0.039 | (0.283, 0.195) |
| "cheese box", "foil package" | 0.00 | — | never detected |

This is the boundary measured rather than argued: the phrase the
benchmark writes the task in never fires, the prompt that fires reliably
lands all over the scene (it is matching several boxes), and the only
candidate that lands *consistently* fires on one frame in six. No prompt
in this family is both reliable and discriminating — which is the precise
sense in which the frozen detector limits object breadth.

**One candidate fix, implemented and under test.** `_HEAD_DISCRIM` in
`microvla/perception/prompts.py` inserts per-object discriminating heads
between the product phrase and the shared generic tail, so a chain takes
"white carton" on the frames where it grounds and "box" elsewhere.
Cream's chain becomes `['cream cheese', 'cheese', 'white carton', 'white
box', 'box', 'cardboard box', 'can']`; **every other object's chain is
byte-identical** (verified: soup, butter, pudding unchanged), and 605
tests pass. This is a per-object task constant of the same category as
`hang_comp` and is disclosed as one wherever a number produced with it
appears.

Now running: cream n=10 with the discriminating chain, and soup n=10 with
the same code as the regression check. Recorded before the result — the
mechanism predicts a *small* gain at best, because the discriminating
prompt grounds on only 17% of frames while the latch needs a stable
window; if cream stays at 0/10 that is consistent with the mechanism
rather than a refutation of it, and the honest report is that the fix was
insufficient, not that the diagnosis was wrong.

### The size claim, made checkable under both conventions (2026-08-06)

Auditing our own related-work paragraph found a **material omission that a
referee would have caught**: it listed Octo-small (~137M) as the smallest
alternative and jumped from there to MicroVLA's ~30M, omitting **RT-1**, whose
35M is by far the nearest neighbour in size. Omitting the nearest competitor
from a size comparison is the kind of error that costs a paper its credibility
even when the conclusion survives. Verified against the sources rather than
memory: RT-1 is 35M with a FiLM-conditioned EfficientNet, and its instruction
embedding comes from a **Universal Sentence Encoder that the 35M excludes**;
Octo-small is a 27M transformer whose language encoder is a **frozen T5-base
(111M)** likewise excluded.

That exclusion is the whole story. The two size conventions in common use
disagree about who is smallest, because published counts routinely omit a
frozen text encoder that inference still loads:

| system | trained | deployed | language encoder |
|---|---|---|---|
| RT-2 | 55B | 55B | internal (PaLI-X) |
| OpenVLA | 7B | 7B | internal (Llama-2) |
| SmolVLA | 450M | 450M | internal (SmolVLM2) |
| Octo-small | **27M** | 138M | T5-base, 111M frozen |
| RT-1 | 35M | **≥35M** | USE, frozen, *uncounted* |
| **MicroVLA** | **17.2M** | **30.2M** | *none* — detector's own tower |

MicroVLA counts verified by `microvla.utils.param_audit`: trunk 7,005,837 +
goal heads 0.24M + TRM 9.97M = 17.2M trained; + 13.0M frozen detector = 30.2M
deployed. Of the 17.2M only **7.24M is causally load-bearing**, the world model
being inert in every nonzero number.

**The result is that MicroVLA is smallest under both conventions at once** —
1.6× below Octo-small's 27M on trained parameters, and below RT-1's 35M on
deployed parameters *even after granting RT-1 a free language encoder we
decline to estimate*. Granting the competitor its best case and still winning
is the only version of this claim worth making. We deliberately cite **no**
parameter count for USE: the search returned no authoritative figure, and an
invented one would be the exact failure this log exists to prevent.

The claim is stated in bounded, refutable form: it ranges over surveyed
language-conditioned VLA policies with published counts, concerns parameter
counts alone, and **is refuted by exhibiting any surveyed-class system below
17.2M trained or 30.2M deployed**. The mechanism is architectural, not
incidental — MicroVLA has no separate language model, so the frozen encoder the
other rows pay for twice (in memory, and in the convention mismatch) does not
exist here.

This replaces the previous "we claim no comparative superlative", which
*understated* what the evidence supports, with a claim a referee can check or
break. Non-claims tightened in the same edit: no claim that smaller systems do
not exist, and **no claim that small size caused any success rate reported
here**. Submission rebuilds clean at 5 pages, 0 errors, 0 undefined citations.

### Reproducibility: run provenance was missing, and the gap was self-inflicted (2026-08-06)

Building a reviewer-runnable reproduction path exposed a real weakness in our
own harness: **`results.json` recorded the numbers but not the command that
produced them.** Every cell in the paper is cited by `run_id`, but a `run_id`
is only auditable if the artifact says what made it — and recovering the
headline invocation required grepping the pod's shell scripts for the output
directory name (`scripts/power50.sh`). That is exactly the kind of provenance
gap this log exists to catch, and it was ours.

Fixed in `eval/libero_eval.py::_provenance()`: every `results.json` now records
the full `argv`, the git commit, and whether the tree was dirty. Deliberately
best-effort — anything unreadable is **omitted rather than guessed**, since an
absent key is honest and a wrong one is worse than nothing. Distinct from
`microvla.utils.provenance`, which describes what a *corpus* was baked under;
this describes the *run*. Verified on the `--mock-env` path (which must always
stay runnable per the repo contract): argv, commit `faf832d`, dirty `true`.
605 tests pass, 1 skipped.

**Checkpoint identity verified, not assumed.** The repository ships the exact
weights that produced the headline number — md5 `01ff8728…` for
`goal_heads_v5.pt` and `a8ea1cda…` for `full_stageB_rec_fix.pt`, confirmed
byte-identical between the repo and the machine that ran the n=50 cells. A
reviewer can check the hash before spending GPU-hours, and a hash mismatch
means they are not reproducing our cells.

`paper/submission/REPRODUCE.md` carries the exact commands for both n=50 cells,
the hashes, and a mock-env smoke test that needs no LIBERO, no GPU, and no
network. It also states what reproduction honestly means here: **expect the
Wilson interval, not the point estimate** (these are Bernoulli cells; a rerun
landing on 0.700 exactly would be luck, not fidelity), and — importantly —
**a mismatch on a different software stack may be the paper's own finding
recurring rather than a failed reproduction**, since the audit-stack control
shows a rebuild can invert which head looks better (memorized 6/10 vs released
0/10, the exact reverse of the deployment stack). We tell reviewers to report
stack versions with any mismatch instead of assuming either side is wrong.

### PRE-REGISTERED: does detector discriminability predict which objects cross? (2026-08-06)

The multi-object result is "two of four crossed", and the claimed mechanism is
that the frozen detector cannot bind the other two. That claim is currently
supported by cream's screening table alone. It becomes a *testable* mechanism
if discriminability, measured per object on its own corpus through the
production chain, **predicts** the crossing outcome.

**Instrument.** For each object, build the production prompt chain
(`with_fallbacks`), call `set_role_prompts` exactly as deployment does, run
`perceive()` on real corpus frames, and record firing rate and the spread of
the selected box centre. Low spread means the detector keeps picking the same
region (bindable); high spread means it is picking a different box from frame
to frame (not bindable).

**Registered prediction, before running.** Soup and butter crossed; cream did
not. If the mechanism is real, **soup and butter show box-centre spread below
~0.15 in both axes, and cream shows spread above ~0.20** (its incumbent "box"
prompt measured (0.270, 0.155)). Cream is screened under BOTH its pre-fix chain
(the one in force when it scored 0/10) and its post-fix `_HEAD_DISCRIM` chain,
because comparing a crossing outcome against a chain that did not produce it
would be the same provenance error logged above.

**Retraction conditions, registered now.** (a) If a *crossed* object shows high
spread, the mechanism claim is falsified and the paper reverts to "two of four
crossed, cause unknown" — the spread metric does not get retuned to fit. (b)
Only three of four objects can be screened; there is no pudding corpus, so
pudding stays out of the table rather than being inferred. (c) n=3 objects
cannot establish a correlation and will not be reported as one — the strongest
available reading is consistency-or-not with a named mechanism.

### RESULT: prediction falsified, and a previous cycle's "resolved mechanism" RETRACTED (2026-08-06)

The registered prediction failed, and following it honestly destroyed a claim
this log made one cycle earlier.

**Screening, production chain, each object's own corpus (24 frames / 3 eps each):**

| object | outcome | fire | conf | box-centre spread |
|---|---|---|---|---|
| soup | **CROSSED 35/50** | 0.96 | 0.085 | **(0.320, 0.176)** |
| butter | **CROSSED** | 1.00 | 0.208 | (0.256, 0.075) |
| cream | 0/10 | 0.96 | 0.107 | (0.304, 0.233) |
| cream, pre-fix chain | 0/10 | 0.96 | 0.154 | (0.295, 0.186) |

Prediction was: crossed objects below ~0.15 spread, cream above ~0.20. **No
crossed object is below 0.15, and soup — the object that succeeds 35/50 — has
the HIGHEST spread of all three, above cream's.** Per registered retraction
condition (a), the spread metric is abandoned, not retuned.

**The confound, named.** Box-centre spread across frames is measured under a
**wrist camera that moves**, so an object's image position changes legitimately
as the arm travels. Spread therefore conflates "the detector switched objects"
with "the camera moved" and cannot distinguish them. This also retracts the
interpretation offered in the screening table above — that "box" firing at 0.92
with spread (0.270, 0.155) meant it was "matching several boxes". That reading
was unfounded. Likewise "white carton"'s tight (0.057, 0.072) was measured on
17% of frames, ~4 frames likely adjacent in time and thus under minimal camera
motion; it is a small-sample artifact of temporal clustering, not evidence of
better binding.

**The decisive follow-up, and the retraction it forces.** Screening each chain
element individually asks which one the chain actually resolves to:

| object | product name | head noun | "box" | resolves to |
|---|---|---|---|---|
| soup (**crossed 35/50**) | "alphabet soup" **0.00** | "soup" **0.00** | **0.96** | **box** |
| butter (**crossed**) | "butter" **0.00** | — | **1.00** | **box** |
| cream (0/10) | "cream cheese" **0.00** | "cheese" **0.00** | **0.92** | **box** |

**All three objects fall back to the generic tail. The two that cross fall back
exactly as the one that fails does.** The previous cycle's claim — that
object breadth is limited because product names score 0.000 and every grocery
therefore collapses onto a shared generic tail, "making same-shaped objects
indiscriminable *by construction*" — **is retracted**. It cannot explain
cream's failure, because soup is indiscriminable in precisely the same sense
and succeeds 35/50. The observation that product names never fire is *correct*
and stands; the causal story built on it does not.

**Consequences, stated rather than buried.** (i) `_HEAD_DISCRIM` ("white
carton") was motivated by the now-retracted reading; the code change is inert
for every object but cream and is still under test, but its *rationale* is
withdrawn and no result may cite it. (ii) The cream evaluation currently
running is testing a fix whose premise is dead — it is being carried to
completion anyway, because stopping a pre-registered run because you no longer
like its rationale is how selection bias enters. (iii) Deployed binding
accuracy remains UNMEASURED; this is now the fifth instrument aimed at it to
fail at its own gate.

**Refined hypothesis, registered for future work, not claimed.** If every
object binds through the same generic "box" prompt, the difference between
crossing and not cannot live in the prompt — it must live in the **scene**: how
many box-like distractors compete for that prompt, and whether the true target
wins. That is a property of the object's LIBERO scene, not of its name. It
predicts cream's scene contains box-like competitors that soup's does not.
Untested; recorded as a hypothesis with its own instrument to build, and
explicitly NOT offered as this paper's mechanism.

**What the paper now says about multi-object breadth:** two of four objects
crossed; the two that did not are *not* explained. The honest status is "cause
unknown", replacing "mechanism-located at role binding". A named mechanism that
fails its own control is worth less than an admitted gap.

**Manuscript updated to match (2026-08-06).** The retraction propagated to all
three places the withdrawn mechanism appeared: the abstract ("two measured,
mechanism-located at role binding" → "two not crossed and, after a proposed
mechanism was tested and retracted, not explained"), the non-claims list
("blocked at role binding" → "not explained"), and the addendum, which now
carries the element-screening table and the falsified spread prediction inline.
Counts corrected throughout: **three** pre-registered predictions falsified (was
two) and **five** instruments abandoned at their own gates (was four). Body is
5 pages with references spilling to a 6th; 0 errors, 0 undefined citations.

The abstract now leads with the retraction's most useful consequence rather
than hiding it: *every object, including the two that succeed, is detected by
the same generic shape prompt rather than by its name, so "the detector cannot
name it" cannot be why the other two fail.* A referee learns more from that
than from the mechanism we withdrew.

### The scenes are the same scene — and that enables a decisive, ground-truth-free test (2026-08-06)

Reading the BDDL scene definitions kills the distractor hypothesis registered
one entry above, before it was ever run:

```
soup scene:  alphabet_soup, basket, salad_dressing, cream_cheese, milk, tomato_sauce, butter
cream scene: cream_cheese,  basket, orange_juice,   alphabet_soup, milk, tomato_sauce, butter
```

**Cream cheese sits in soup's scene; alphabet soup sits in cream's scene.** Six
of seven objects are shared. The competitor set is essentially identical, so
"how many shape-alike distractors compete for the generic prompt" cannot
distinguish an object that crosses at 35/50 from one that scores 0/10. The
hypothesis is withdrawn at the cost of one BDDL read rather than a GPU-day —
which is the argument for reading the environment definition before building
the instrument.

**What this makes possible.** Both objects are in *both* scenes, and both
resolve to the same generic prompt "box". So we can ask the identity question
**without any ground truth**, which is what defeated five previous instruments:
run the soup chain and the cream chain over the *same frames* and check whether
they select the *same detection*.

**PRE-REGISTERED prediction.** Since both chains resolve to "box" at 0.96/0.92
firing, they will select the **same box** on the same frame — agreement well
above 0.5, plausibly near 1.0. If so, deployed role binding is **identity-blind**:
the system cannot be picking the named object, and soup's 35/50 cannot be
explained by binding the right object by name.

**Registered consequence if confirmed, recorded before the run.** This does
*not* touch the placement-memorization audit or the vision-vs-proprioception
attribution result, which concern the grasp head's inputs and are measured
independently. It *does* mean the paper must say the head is grounded on *a*
box rather than on *the named* box — a caveat a referee would otherwise find
first, and one that materially scopes the word "grounding". **If agreement
comes out low instead, the prediction is falsified and role binding does carry
identity**, in which case cream's failure returns to fully unexplained and this
entry records a third dead hypothesis.

### CONFIRMED: deployed role binding is identity-blind (2026-08-06)

The registered prediction is confirmed, and not marginally. Running each
object's chain over the *same* frames and asking whether they select the same
detection:

| frames from | soup vs cream | soup vs butter | cream vs butter | median centre distance |
|---|---|---|---|---|
| soup corpus | **1.00** | **1.00** | **1.00** | 0.0001 |
| cream corpus | 0.87 | 0.96 | 0.91 | 0.0001 |
| butter corpus | 0.92 | **1.00** | 0.92 | 0.0000 |

**The result is not threshold-sensitive.** "Same box" was scored at a centre
distance below 0.02, but the observed median distance is **0.0001** — 200×
inside the threshold. These are not similar boxes; they are the same box. Any
threshold between 0.001 and 0.02 gives the same table.

**What this establishes.** Asking the machine for alphabet soup and asking it
for cream cheese returns *the same detection from the same frame*. Deployed
role binding does not carry object identity. It follows that **soup's 35/50
cannot be explained by the machine binding the named object**, because the
machine cannot tell the two objects apart at this stage — and the scenes
contain both objects, so there was something to tell apart.

This is the first instrument in six attempts to actually measure the deployed
binding question, and it succeeded precisely because it **needs no ground
truth**: it compares two prompts against each other rather than against a
projected world position. The five failed instruments all tried to establish
where the true object was. The question "do these two prompts agree?" is
answerable without knowing who is right.

**A coherent mechanism, now supported rather than asserted.** The detector
selects a largely prompt-independent box; the task succeeds when that box
happens to be the target and fails when it is not. This fits every observation
on the table: why all four binders left cream at 0/10 (they operate downstream
of a stage that discarded identity), why cream's machine converges to
`eef_obj_dist_min` ~0.072 m — it servos accurately to *an* object, just not the
commanded one — and why per-object prompt engineering (`_HEAD_DISCRIM`) changed
nothing (cream 0/10 with the discriminating chain, this cycle).

**Scoping the word "grounding", before a referee does it for us.** The
placement-memorization audit and the vision-vs-proprioception attribution
result are untouched: they concern *which inputs* the grasp head uses, measured
independently, and remain as reported. But the repaired head must now be
described as grounded on **a box in the image** rather than on **the named
object**. That distinction is real, it is ours to disclose, and the manuscript
is being changed to say so. It also means "object-level generalization" was
never the right frame for the multi-object campaign: the pipeline has no
object-level channel to generalize over.

**Honest note on what this costs.** Two of four objects still cross, and soup's
35/50 held-out cell is unchanged and correct. What changes is the *story* about
why — a system that succeeds without discriminating its target is a weaker
scientific claim than one that succeeds by discriminating it, and we would
rather report the weaker true one. The corresponding strength is that this is
now measured rather than assumed in either direction.

**Cream, discriminating chain: 0/10 FINAL** (`eval_results/unaided_cream_discrim`,
`mean_success` 0.0, n=10, verified from artifact). The `_HEAD_DISCRIM` fix
failed, exactly as the pre-registered prediction allowed for ("the mechanism
predicts a small gain at best... if cream stays at 0/10 that is consistent with
the mechanism rather than a refutation of it"). With identity-blindness now
measured, the reason is clear and was not available when that prediction was
written: a per-object prompt cannot help a stage that discards identity
downstream of the prompt. The soup regression cell is running.

Both documents now carry the identity-blind finding. Submission body remains 5
pages (references spill to a 6th, which is standard), 0 errors, 0 undefined
citations.

### PRE-REGISTERED: is identity recoverable at all in this pipeline? (2026-08-06)

Identity-blindness says the chain picks a prompt-independent box. Identity can
then only be recovered by choosing among **multiple** candidate detections —
which is exactly what the four binders tried to do. So: how many candidates do
they get? `microvla/perception/yolo_world.py` already documents 0.68 proposals
per frame on this view from an earlier measurement; this re-measures under the
production config (`det_conf` 0.02) on all three corpora.

**Prediction, registered before the run.** Proposal count will be near 1, and
`>=2` will hold on a minority of frames. If so, the boundary is at the
**detector**, not at the binder: on most frames there is nothing to choose
between, and no downstream binder — however accurate offline — can recover an
identity the detector never proposed. That would explain all four binders
leaving cream at 0/10 without appealing to binder quality at all.

**Falsification condition.** If `>=2` proposals are common (say above 0.5 of
frames), the prediction fails, identity IS recoverable, and the binders'
failure returns to being a binder-quality problem — a materially different
conclusion that would re-open the four-binder sub-study rather than explain it.

### RESULT: prediction falsified — identity IS recoverable, so the binders own their failure (2026-08-06)

| corpus | mean proposals/frame | median | ≥2 proposals | histogram |
|---|---|---|---|---|
| soup (crossed) | 3.12 | 4 | 0.71 | {1:7, 3:5, 4:9, 5:2, 7:1} |
| butter (crossed) | 3.83 | 4 | 0.96 | {1:1, 2:4, 3:7, 4:5, 5:3, 6:2, 7:1, 8:1} |
| cream (0/10) | 3.92 | 4 | 0.92 | {1:2, 2:4, 3:5, 4:3, 5:5, 6:3, 7:2} |

The registered threshold was "≥2 proposals on more than 0.5 of frames falsifies
the prediction". Observed: **0.71 to 0.96**. The prediction is dead — this is
the **fourth** falsified pre-registration this campaign. Cream in particular
offers ~3.9 candidates per frame, *more* than the object that crosses.

**What this forces us to conclude, against the convenient direction.** The
boundary is **not** at the detector. Three to four candidate boxes are on the
table on nearly every frame, so a binder that could score them correctly would
recover identity. The four binders therefore fail on **their own quality**, not
on missing candidates — the comfortable explanation ("the detector never
proposed it") is refuted by our own measurement, and the sub-study is re-opened
rather than closed.

**This lands exactly on the paper's central theme.** The already-measured
reason binders fail live is on-manifold overfitting: with the 0.902-accuracy
bank active, the uv stream feeding the grasp head *destabilises* (lateral std
0.258→0.335, vertical 0.047→0.202) — a binder accurate on corpus crops thrashes
on live crops. Candidates exist; scoring them under the machine's own
off-manifold viewpoints is what breaks. That is the paper's on-manifold limit
recurring for the third time, now with the alternative explanation eliminated
rather than merely unexamined.

**Code-documentation defect found and fixed.** `yolo_world.py` documented "0.68
proposals per frame on this view" from an earlier measurement. Under the
production config (`det_conf` 0.02) the true figure is 3.1–3.9. The old number
is not wrong so much as **config-dependent and unlabelled** — it was measured at
a different detection threshold and then read as a property of the view. The
docstring now carries both numbers with their configs, because a stale constant
that looks like a measured fact is precisely the class of defect this paper
documents 29 of.

**Status after this cycle.** Binding is identity-blind (measured). Identity is
recoverable in principle (measured). Why cream cannot be bound live remains
**unexplained**, and the honest statement is that the failure sits in live
candidate scoring, where every instrument we have is trained or validated on
teacher trajectories and thus cannot certify off-manifold behaviour.

### PRE-REGISTERED: the instruction-swap ablation (2026-08-06)

Identity-blindness was measured at the *perception* layer. It makes a sharp
*behavioural* prediction that can be tested directly, and which no previous
instrument in this campaign could reach: **if the machine cannot distinguish
the named object, telling it to fetch a different object should not change its
success rate.**

**Design.** Run the soup task — env, scene, physics, and success criterion all
unchanged and still scoring "alphabet soup in basket" — while telling the
policy "pick up the butter and place it in the basket". Butter is present in
soup's scene (BDDL, verified above), and butter's chain
`['butter','box','cardboard box','can']` carries **no** `_HEAD_DISCRIM` entry,
so this is the pure generic-tail path with no per-object constant confounding
it. A second cell swaps in the cream instruction; that chain now contains
"white carton", which is disclosed rather than silently included.

**Implementation** (`--override-instruction`): the env is built from
`bddl_file` + `init_states`, never from the instruction, so the override
changes *only what the policy is told*. It prints the swap on every task,
because a silent instruction swap would be indistinguishable in the logs from a
correctly-instructed run — the exact provenance failure mode this log keeps
catching. 605 tests pass; verified on the `--mock-env` path.

**Prediction, registered before the run.** Success will be **statistically
indistinguishable from the correctly-instructed baseline** (flagship held-out
7/10). Concretely: the swapped cell lands within the baseline's Wilson interval
rather than collapsing toward 0.

**Falsification condition.** If the swapped cell collapses (say ≤2/10), the
prediction fails and identity-blindness measured at perception does **not**
propagate to behaviour — meaning some later stage recovers identity, and the
"grounded on *a* box" scoping I have already written into both documents would
be too strong and must be walked back. Recording that now, before the result,
because the scoping edit is already committed and I want the condition under
which I would have to undo it on the record first.

**Precision correction, same day.** The identity-blind entry above wrote that
the four binders "operate downstream of a stage that discarded identity". The
proposals measurement makes that wrong, and it is corrected in both documents
rather than left to read well. Binders consume **class-agnostic proposals**
(3.1–3.9 per frame), not the identity-blind source selection, so they *are*
handed identity-bearing candidates. Identity-blindness explains the default
path — why the machine servos to *an* object and why per-object prompts change
nothing — but it does **not** excuse the binders, whose failure remains theirs
and remains unexplained beyond the measured live-scoring instability. Two
findings from the same day, and the later one constrains the earlier one's
scope; recording that is cheaper than a referee finding the seam.

### The size claim survives an adversarial survey (2026-08-06)

A bounded claim is only worth stating if someone tried to break it. Searched
specifically for VLA systems *smaller* than ours, then read the two live
threats rather than trusting a search snippet:

| candidate | trained | deployed | separate text encoder? | verdict |
|---|---|---|---|---|
| **NanoVLA-S** (arXiv 2510.25122) | 52M | **~161M** | frozen BERT-base, **excluded from its count** | above us on both |
| **LiteVLA** (arXiv 2511.05642) | LoRA only | ~256M | internal (SmolVLM-256M) | above us on both |

The claim holds: 17.2M trained / 30.2M deployed remains smallest under both
conventions. NanoVLA's "98% fewer parameters" is measured against OpenVLA's
7.5B, not against small systems, and its own headline figure is **trainable**
parameters — the paper loads ~161M.

**NanoVLA is the convention gap in miniature, and it is a better argument for
the two-convention table than any prose I could write.** A 2025 paper *named
for being nano* reports 52M while loading ~161M, because a frozen ResNet18 and
a frozen BERT-base do not appear in the headline. That is the same accounting
that lists Octo-small at 27M while T5-base sits beside it and RT-1 at 35M with
USE outside the number. MicroVLA's architectural claim — no separate language
model, the detector's own text tower supplies every text embedding — is exactly
the thing that makes its two numbers converge instead of diverge.

Both are cited in the submission's size table and the claim now names its own
adversarial survey. This is the fourth external check performed by reading the
source rather than a summary this session; the USE parameter count remains
deliberately uncited because no authoritative figure was found.

### The identity test becomes contribution #4, and the two findings unify (2026-08-06)

Promoted the ground-truth-free identity test from a diagnostic to a stated
contribution, because it is the most *transferable* thing this session
produced. The generalizable form: open-vocabulary detectors are typically
prompted with a fallback chain, so **a policy can appear language-conditioned
while every instruction collapses to the same generic prompt**. Comparing two
*different* instructions' selections on the same frame detects this without
knowing which is correct — which is exactly why it worked where five
ground-truth-seeking instruments failed. It is cheap, needs no annotation, and
applies to any detector-grounded policy, not just ours.

Also added the unifying line the paper had been missing: **both headline
findings have one shape — apparent competence resting on a channel other than
the claimed one. Position where we claimed looking; shape where we claimed
naming.** The audit generalizes better than the artifact does, and saying so
plainly is more useful to a reader than either finding alone.

### Regression check: the prompts.py change is safe for the working object (2026-08-06)

**Soup, same code as the cream discriminating run: 5/10**
(`eval_results/unaided_soup_discrim`, `mean_success` 0.5, n=10,
artifact-verified). Soup's chain is byte-identical before and after the
`_HEAD_DISCRIM` edit — only cream's chain changed — and 5/10 sits inside the
held-out cell's Wilson interval [0.56, 0.81]'s neighbourhood at this n (the
n=10 flagship held-out cell is 7/10; n=10 cells at p≈0.7 return 5/10 routinely).
The edit is confirmed inert for the objects it does not name, which is what a
regression cell is for. Cream with the discriminating chain: **0/10**.

Net result of the `_HEAD_DISCRIM` experiment: no gain on the target object, no
loss on the others, and its motivating rationale independently retracted. The
code stays (inert, disclosed) and no number in the paper is produced with it.

**Pre-analysis, recorded before the swap numbers (2026-08-06).** Two things to
settle in advance so the result cannot be read to taste.

*Is the swap circular?* The obvious objection: soup and butter chains both
resolve to "box", so of course swapping them changes nothing — the behavioural
test merely restates the perception measurement. It does not, and the reason is
specific. The perception measurement shows the same *box* is selected. It does
**not** show that everything downstream is unaffected: the goal heads consume
box **embeddings** and a CLIP task embedding derived from the instruction text,
both of which differ between "alphabet soup" and "butter" even when the
selected box is identical. The behavioural cell tests whether those surviving
differences matter at the task level. A null result is therefore an end-to-end
confirmation, not a restatement — and a non-null result would show identity
partially survives in a channel the perception probe does not observe.

*The strong reading is per-trial, not per-rate.* Seeds, init states and physics
are identical across cells, so if the instruction is genuinely inert the **same
trials** should succeed, not merely the same number. Matching rates could
coincide; a matching success *pattern* over ten trials could not, plausibly.
The analysis reports per-trial agreement and names the differing trials.

*Asymmetry to expect.* The cream swap is **not** a clean replicate of the
butter swap: cream's chain now contains "white carton" (fires 17%), so it
perturbs the input in a way butter's does not. If the butter swap is inert and
the cream swap is not, that difference is attributable to the discriminating
prompt rather than to object identity, and will be reported that way.

### The architecture proves it: no text reaches the grasp decision (2026-08-06)

Code inspection closes the identity-blindness argument from the other side, and
it is stronger than the measurement because it is not statistical.

`GraspPointHead.forward` (`microvla/control/goal_head.py:117`) consumes exactly
`feats["geom"]`, `feats["box_emb"]`, `feats["frame_emb"]`, and `feats["eef_xy"]`.
Its builder is `build_grasp_features(uv, conf, proprio, box_emb, frame_emb)` —
**there is no task, command, or text embedding in the signature at all.** The
deployed call site (`eval/policy.py:901`) passes exactly those five quantities.

So the chain is:

1. The grasp head — which decides *where to grasp* — receives **no language
   input whatsoever** (code fact, not measurement).
2. The only path from instruction to grasp is therefore *which box gets
   selected*, which reaches the head as `uv`, `conf`, and `box_emb`.
3. That selection is **identity-blind** (measured: same box for different
   objects, median centre distance 0.0001).
4. Therefore the instruction **cannot** influence the grasp decision.

Steps 1 and 2 are architectural and hold with certainty; only step 3 is
empirical. This is a materially stronger result than the measurement alone, and
it *predicts* the instruction-swap outcome rather than merely being consistent
with it — which is what makes the running swap a real test of the chain instead
of a restatement.

**The place side is different, and the asymmetry is worth stating.** `PlaceHead.forward`
(`goal_head.py:167`) takes `command_emb` and *is* language-conditioned. It is
inert across this benchmark for a mundane reason: every LIBERO-Object task
places into the same basket, so a correctly-conditioned place target is
identical whatever the instruction says. The stack therefore has exactly one
genuine language channel, and the benchmark gives it nothing to do.

**What this does and does not license.** It does not touch the
placement-memorization audit or the vision-vs-proprioception attribution. It
does mean the honest description of the system is: *an open-vocabulary detector
selects a box by shape, and a language-free head decides where to grasp it.*
That is a smaller claim than "language-conditioned VLA", and it is the true one.
The architecture was ours to read at any point in this campaign; reading it is
what six instruments and four falsified predictions cost.

### Repo audit: the paper's foundational instrument was untracked (2026-08-06)

Building the reproduction path surfaced a second, worse provenance gap than the
missing `argv`. The manuscript cites `scripts/measure_placement_pinning.py` in
**four** places as shipped, and the submission's opening claim — "target start
poses are exactly pinned in 6 of 10 tasks and within ~±1 cm in the rest" —
rests entirely on its output. **Neither the script nor its emitted table was in
the repository.** A referee following the citation would have found nothing.

Verified against the claim *before* tracking it, rather than assuming our own
artifact agreed with our own prose: the table reports **6 tasks pinned**
(bit-identical across all 50 shipped init states, max deviation ~1e-17 m) and
**4 varying at std 0.25–0.58 cm** — matching "6 of 10 … 0.25–0.58 cm" exactly.
The table additionally carries a **SHA-256 digest of each task's init-state
array**, so the measurement is traceable to the exact data it read.

Now tracked: the script, `results/placement_pinning.json`, and
`results/placement_pinning.md`. Combined with the run-`provenance` field and
`REPRODUCE.md`, the chain from headline claim → instrument → artifact → data
digest is now complete and walkable by someone who has never spoken to us.

Two provenance defects found in one session by the simple act of asking "could
a stranger reproduce this?" — both in *our* favour to ignore, both fixed. That
question is a better auditor than any of the six instruments this campaign
built.

**Third provenance gap closed: `models/README.md` was cited but absent.** A
systematic sweep of every path the papers cite in backticks (28 unique) against
the repo found exactly one remaining miss — the release manifest the manuscript
describes twice as carrying "the LIBERO commit SHA, init-file hashes, the
ultralytics version, SHA-256 digests for the detector and every `models/*.pt`".
It did not exist. It does now, and every field in it was **measured rather than
transcribed**: digests computed on the files, versions read off the evaluation
machine, LIBERO commit `8f1084e3` read from its checkout. `goal_heads_v5.pt`
and `full_stageB_rec_fix.pt` verified byte-identical to the machine that ran
the headline cells. Nothing was filled in from memory, and the training
commands point at the manuscript's §11 block rather than being restated (and
possibly drifting) here.

The manifest also carries the warning the audit-stack finding demands: **report
stack versions with any reproduction attempt, because a mismatch may be this
paper's own finding recurring rather than a failed reproduction.**

Sweep result: 28 cited paths, 3 provenance defects found and fixed in one
session (missing run `argv`, untracked pinning instrument, absent release
manifest), 0 remaining.

### Swap baseline lands, and is recorded before its comparison exists (2026-08-06)

**Baseline: soup env, correct instruction, seed 20, n=10 → 7/10**
(`eval_results/swap_baseline`, `mean_success` 0.7, artifact-verified). This
**exactly reproduces the published flagship held-out cell (7/10)** — an
independent same-stack reproduction of the paper's headline n=10 number, run
today with no tuning between.

Per-trial success pattern, written down now so the swap cannot be compared to a
moving target:

```
trial:    0 1 2 3 4 5 6 7 8 9
baseline: 1 1 0 1 1 1 0 0 1 1     (7/10)
```

Trials 2, 6 and 7 fail. **If the instruction is genuinely inert, the swapped
cell must reproduce this exact string**, not merely its count — identical
seeds, identical init states, identical physics. A matching rate could
coincide; a matching ten-bit pattern essentially could not.

**The override is verified applied, not assumed.** The run log carries
`INSTRUCTION OVERRIDE: task 'pick_up_the_alphabet_soup_and_place_it_in_the_basket'
told 'pick up the butter and place it in the basket' (was 'pick up the alphabet
soup and place it in the basket'); success still scored on the REAL task`. A
silent no-op here would have produced a "confirmation" of identity-blindness
that was really just the baseline run twice — the failure mode the loud print
exists to prevent, and the one worth checking before trusting the result.

### RESULT: the instruction-swap prediction is FALSIFIED — and the telemetry says the collapse is not where I predicted it would be (2026-08-06)

**Butter swap: 0/8 where the baseline scores 5/8 on the identical trials.**
Registered prediction was "statistically indistinguishable from baseline". It
is not. **The prediction is falsified** — the fifth of this campaign — and the
registered consequence is triggered. Recorded before any interpretation.

```
trial:     0 1 2 3 4 5 6 7
baseline:  1 1 0 1 1 1 0 0     (5/8 on shared trials)
butter:    0 0 0 0 0 0 0 0     (0/8)
per-trial agreement 3/8; differing trials [0,1,3,4,5]
```

**But the failure stage is not the one my falsification condition assumed.**
My registered condition read "if the swapped cell collapses, identity-blindness
does not propagate to behaviour, and the 'grounded on a box' scoping must be
walked back". The telemetry shows that condition was **too coarse**, because it
treated "the cell collapsed" as equivalent to "binding recovered identity". It
did not:

| | baseline | butter swap |
|---|---|---|
| `eef_obj_dist_min` (to the **true soup** object) | 0.012–0.018 | **0.006–0.013** |
| `grip_close_rate` | 0.656–0.679 | **0.252–0.432** |
| `src_detect_rate` | 0.90–1.00 | 0.88–1.00 |

**Told to fetch butter, the machine still drives to the soup — as close as the
baseline, or closer.** That is direct behavioural evidence *for* source-side
identity-blindness, in the one place it could be observed. What collapses is
`grip_close_rate`, i.e. the machine grips and then does not hold: it reaches the
right object and loses it downstream.

**A post-hoc reading, labelled as post-hoc and not yet claimed.** `PlaceHead`
*is* language-conditioned by architecture (it takes `command_emb`, unlike the
grasp head). One instruction drives both the prompt chain and the CLIP task
embeddings, so the swap changed the place target as well as the prompt. A wrong
place target would produce exactly this signature — grip, lift, travel to the
wrong location, release early, `grip_close_rate` falling from ~0.67 to ~0.26
while `eef_obj_dist_min` stays at baseline. This explanation **fits** the data
and was **not** predicted in advance; it is a hypothesis generated by the
telemetry and is not evidence until tested.

**What I will not do.** I will not report this cell as "confirming"
identity-blindness. The prediction I registered was about the success rate and
it failed. Spinning a falsified prediction into a confirmation by switching to
a metric I chose after seeing the data is the exact failure this log exists to
prevent. The registered outcome stands: **prediction falsified**.

**What the falsification does and does not license.** It does *not* license
walking back the architectural fact — `build_grasp_features` still has no text
input, which is a property of the code and not of any run. It *does* mean the
system as a whole is **strongly instruction-sensitive**, and any statement that
the machine "ignores language" would be wrong. The accurate statement is
narrower and now two-sided: *the grasp side is language-free and its source
selection is identity-blind; the place side is language-conditioned and
mismatching it is catastrophic.* Both documents will be corrected to say that,
after the decomposition below settles which path carries the collapse.

**Next test, pre-registered now.** The swap conflated two channels. Decompose:
drive the **prompt chain** from one instruction while feeding the **command
embedding** from the other. If source binding is identity-blind and the place
head owns the collapse, then prompts=butter with command=soup should return to
~7/10, and prompts=soup with command=butter should collapse. If instead the
prompt swap alone collapses it, source binding carries identity after all and
the identity-blind finding — measured on corpus frames, which are on-manifold —
does not survive deployment, exactly as this paper's own on-manifold limit would
predict. Registered before implementing it.

### PRE-REGISTERED: decomposing the swap into its two channels (2026-08-06)

The swap moved two things at once. `JEPALoop.set_task` derives **both** the
detection prompt chain **and** the CLIP task embeddings (which reach the
language-conditioned `PlaceHead`) from one instruction. `--override-prompt-only`
now sources the prompts from one string while the embeddings come from the
task's own — parsed with `parse_command`, deliberately **not**
`task_encoder.encode()`, which would re-harvest the text tower and repoint
detector classes as a side effect, contaminating the channel being isolated.
It logs a warning on every episode. 605 tests pass.

Two cells, both n=10, seed 20, soup env, success scored on the real task:

| cell | prompts from | embeddings from | isolates |
|---|---|---|---|
| **A** | butter | soup (task's own) | the **prompt/binding** channel |
| **B** | soup (task's own) | butter | the **place/embedding** channel |

**Registered prediction.** A returns to baseline (~7/10, inside its Wilson
interval) and B collapses (≤2/10). That would place the swap's collapse
entirely in the place channel and leave source binding identity-blind — which
is what the telemetry suggested post-hoc and what the architecture predicts,
since the grasp head takes no text but the place head does.

**Registered alternative, and what it would cost.** If **A collapses**, the
prompt channel carries identity in deployment, and the identity-blind
finding — measured on *corpus* frames, which are on-manifold — does **not**
survive contact with the machine's own off-manifold viewpoints. That would be
this paper's on-manifold limit striking its own newest result, it would require
retracting the identity-blind claim from all four documents, and it would be
the correct outcome to report. If **both** collapse, the channels interact and
neither is separately attributable; the swap would then support only "the
system is instruction-sensitive", with the stage unresolved.

Recorded before running. Note also that the baseline for A and B is the same
7/10 cell with pattern `1101110011` already on record.

### Instruction swap, all three cells FINAL (2026-08-06)

| cell | result | pattern |
|---|---|---|
| baseline, correct instruction | **7/10** | `1101110011` |
| soup env, told **butter** | **0/10** | `0000000000` |
| soup env, told **cream cheese** | **0/10** | `0000000000` |

All artifact-verified (`swap_baseline` 0.7, `swap_butter` 0.0, `swap_cream`
0.0). Seven discordant pairs, all in the same direction: **exact two-sided
p = 0.0156**. Overrides verified applied from the run log in both swap cells.

**The system is strongly instruction-sensitive.** Any reading of the
identity-blind result as "the machine ignores language" is refuted by this
table. Naming a different object destroys the cell completely and
significantly.

**The two swaps fail differently, exactly as pre-registered.** Recorded before
the run: *"the cream swap is NOT a clean replicate — cream's chain contains
'white carton' (17% firing), so it perturbs the input in a way butter's does
not."*

| | butter swap | cream swap |
|---|---|---|
| `eef_obj_dist_min` | 0.006–0.013 (**reaches the soup**) | 0.007–0.010 on ~half, **0.032–0.051 on the rest** |
| `grip_close_rate` | 0.252–0.432 | 0.000–0.467 |

Butter's chain (`['butter','box','cardboard box','can']`) leaves the machine
reaching the true soup object as accurately as baseline, and the failure is
downstream. Cream's chain additionally prevents approach on about half its
trials. The pre-registered asymmetry is therefore visible in the data, and the
cream cell is **not** clean evidence about identity — it confounds the prompt
channel with `_HEAD_DISCRIM`, and is reported for completeness rather than used
for attribution.

**Attribution is under test, not asserted.** The decomposition cells (prompts
and embeddings driven from different instructions) are queued and will place
the collapse in one channel or refuse to. Until they land, the honest statement
is two-sided and stops there: *the machine still approaches the commanded
task's object under a swapped instruction (butter cell), and nonetheless fails
the task completely; which channel carries the failure is not yet established.*

**Interim caveat added to the submission**, rather than waiting: leaving the
paper claiming identity-blindness with no mention of a 7/10→0/10 collapse would
be a one-sided presentation of evidence already in hand.

**Internal inconsistency caught by reading the rendered PDF (2026-08-06).** The
architecture paragraph asserted that the place head "has nothing to do here"
because every task places into the same basket — while the paragraph directly
below it reports a 7/10→0/10 collapse that the telemetry locates *downstream of
object approach*. Both cannot stand. The error was a scope slip: all *correct*
instructions imply the same place target, which leaves the channel's variation
normally **unexercised**, not **inert**. A mismatched command embedding is
exactly the input that exercises it. Corrected to say so, independently of how
the decomposition resolves — the inconsistency was in the text as written, and
a referee reading two adjacent paragraphs would have found it before we did.

### PRE-REGISTERED: does the swap collapse generalize beyond soup? (2026-08-06)

The swap and its decomposition are both measured on task 0 (alphabet soup) with
the flagship head. A referee's immediate question is whether the effect is a
property of *that* task/head pair. Queued: **butter's own task (task 6)** with
the three-object head `goal_heads_v8.pt`, baseline vs told-soup, n=10 each,
seed 20.

**Registered prediction.** Butter's baseline is a weaker cell than soup's
(butter cells run ~4–7/10 depending on config), so the effect size is smaller
and n=10 may not reach significance on its own. Predicted: **baseline clearly
above zero, swapped cell at or near 0/10**, same direction as soup. If the
swapped cell instead matches its baseline, the collapse is specific to task 0 /
the flagship head and must be reported as such rather than as a property of the
architecture.

**Registered caveat, before the run.** With a baseline near 5/10, a 5→0 result
gives exact p≈0.03 — reportable, but weaker than soup's 0.016, and I will not
pool the two tasks to manufacture a smaller p. They are different heads on
different tasks; they get reported side by side.

### A missing control, noticed mid-analysis: the noise floor (2026-08-06)

The per-trial comparison assumes that a trial flipping between two cells means
the manipulation caused it. That assumption is **unverified**, and I have been
relying on it. Same seed and init state make the *environment* deterministic,
but the policy runs CUDA kernels whose reduction order is not guaranteed
identical across processes, so two runs of the **same** configuration can
diverge — a trajectory near a grasp threshold can tip either way.

This matters right now: decomposition cell A differs from baseline on trials 1
and 7, **in opposite directions** (baseline wins trial 1, cell A wins trial 7),
which is exactly what a noise floor looks like rather than a systematic effect.
Without measuring it, "agrees with baseline 6/7" is uninterpretable at the
single-trial level.

**Control queued: the baseline cell, re-run bit-identically.** Same command,
same seed, same head, nothing changed. Whatever fraction of trials flips is the
harness's own noise floor, and any per-trial claim smaller than that floor is
not a claim.

**Registered prediction.** The repeat reproduces `1101110011` exactly or flips
at most one trial. If it flips two or more, then **per-trial pattern matching
is not a valid instrument in this harness**, every per-trial agreement figure I
have reported today must be downgraded to a rate comparison, and the swap
result rests on 7/10 vs 0/10 (which survives either way — a 7-trial one-way
discordance is far outside any plausible noise floor).

Recorded because I built the per-trial analysis, used it in three writeups, and
only then asked whether it was measuring anything. The rate-level conclusions
do not depend on it; the per-trial ones do.

### Decomposition cell A FINAL: the prompt channel is inert (2026-08-06)

**Cell A — detection prompts from "butter", task embeddings from the real
soup instruction: 6/10** (`eval_results/dec_promptonly_butter`,
`mean_success` 0.6, artifact-verified). Decomposition verified active in the
run log on every episode.

```
trial:      0 1 2 3 4 5 6 7 8 9
baseline:   1 1 0 1 1 1 0 0 1 1    7/10
cell A:     1 0 0 1 1 1 0 1 1 0    6/10
full swap:  0 0 0 0 0 0 0 0 0 0    0/10
```

Exact McNemar on the paired cells:

| comparison | discordant | favouring | exact two-sided p |
|---|---|---|---|
| baseline vs **cell A** | 3 | 2 baseline | **1.0000** |
| baseline vs full swap | 7 | 7 baseline | **0.0156** |
| cell A vs full swap | 6 | 6 cell A | **0.0312** |

**Driving the detection prompts from a different object's name costs nothing
measurable** (p = 1.0). Driving *both* channels from it destroys the cell
(p = 0.0156), and cell A is significantly better than the full swap
(p = 0.0312). The prompt channel is therefore **not** where the collapse lives.

**This is the behavioural counterpart of the identity-blind measurement, and it
is the strongest form of that claim available.** The corpus-frame measurement
showed two prompts select the same box. Cell A shows the consequence *in
deployment, on the machine's own off-manifold states*: you can ask this policy
for the butter and it will find, approach and grasp the alphabet soup at
baseline rate. The on-manifold caveat that has defeated so much of this
campaign does not apply here, because cell A is a deployment cell.

**Held back pending cell B.** Attribution is only complete if the mirror
condition — prompts from soup, embeddings from butter — collapses. Cell B is
running with its dual override verified in the log (`INSTRUCTION OVERRIDE` to
butter, then `DECOMPOSITION ACTIVE` restoring soup prompts). If B collapses,
the language sensitivity is localised to the place head. If B *also* scores
~6/10, then neither channel alone explains a collapse that both together
produce, the channels interact, and I report the stage as unresolved — which
was registered before any of these cells ran.

**Caveat carried forward, not quietly dropped.** The per-trial claims above
still depend on the harness noise floor, which is queued and unmeasured. The
*rate* conclusions (p = 1.0000, 0.0156, 0.0312) are computed on paired cells
and would only be threatened by a noise floor large enough to flip several
trials per run, which the queued control will settle.

### The entire language channel is one latched number (2026-08-06)

Tracing cell B's isolation through the code closes the decomposition
analytically, not just statistically. In goal-machine mode:

1. `action = self.goal_machine.step(proprio, ...)` — the machine's action
   **replaces** the plan wholesale, so fusion/TRM/planner outputs (which *do*
   consume the text tokens) cannot reach behaviour at all.
2. The grasp head takes `build_grasp_features(uv, conf, proprio, box_emb,
   frame_emb)` — no text, established earlier.
3. The place head is called **exactly once per episode**, in `reset()`:

```python
cmd = task.command_emb
pred = self.goal_place_head(cmd)
self.goal_machine.set_place(pred["xy"][0])     # "a per-task constant: latch it once, now"
```

**So the whole of this system's language conditioning, on the path that
produces actions, is a single latched (x, y) place point.** Everything else the
text tower computes is either unused (planner path) or absent by signature
(grasp head).

That makes cell B an exact isolation: prompts from soup, embeddings from
butter changes **one number** — where the machine believes the basket is. And
it predicts the failure mode observed in the full swap and in cell B's first
trial: grip closes, the object is lifted, the arm travels to a wrong place
point and releases there, so `eef_obj_dist_min` stays at baseline (~0.013 — the
object *was* reached) while `grip_close_rate` falls to ~0.26 (it was let go).
Cell B trial 0: `succ=False grip=0.257 dmin=0.013`, matching the full swap's
per-trial values almost exactly.

**Why this is worth stating rather than burying in an appendix.** A reader
would reasonably assume a "vision-language-action" stack routes language
through its policy. In this configuration it does not: it routes language into
one cached coordinate, and everything else — object selection, approach,
grasping — runs without it. That is both the honest description of the artifact
and, we think, the most transferable warning in the paper: *check where the
language actually goes before calling a system language-conditioned.*

**Queued: quantify the mechanism (2026-08-06).** If the whole language channel
is one latched `(x, y)`, the swap's effect size should be readable directly as
the **distance between the place points the head produces for the two
commands**. `scripts/placepoint.py` encodes each instruction through the
production `ClipTaskEncoder`, runs the released place head on each command
embedding, and reports pairwise distances in metres. That converts "the place
target is wrong" from an inference about telemetry into a measured number, and
it is falsifiable in a useful direction: **if the place points differ by
millimetres, the latched-place-point story cannot explain a total collapse**,
and the mechanism would have to be sought elsewhere despite cell B. Queued
behind the noise-floor control.

### The basket never moves — which turns the queued place-point measurement into a memorization test (2026-08-06)

Measured from `results/placement_pinning.json` (the instrument now shipped):
**the basket sits in the same place in all ten LIBERO-Object tasks.**

| task | basket mean x,y (m) | within-task std (m) |
|---|---|---|
| 0 (soup) | (+0.0023, +0.2605) | (0.0081, 0.0087) |
| 1 (cream) | (−0.0016, +0.2597) | (0.0083, 0.0078) |
| 6 (butter) | (+0.0010, +0.2587) | (0.0076, 0.0081) |

Task-to-task separations: **0 vs 1 = 0.40 cm, 0 vs 6 = 0.22 cm, 1 vs 6 =
0.28 cm** — all *smaller* than the ~0.8 cm jitter within a single task. The
place target is, to any practical tolerance, a constant of the suite.

**This reframes the queued place-point measurement as a memorization test, and
lands it squarely on this paper's thesis.** A place head that had *grounded* on
the basket would emit essentially the same `(x, y)` for every command, because
the basket does not move. A place head that emits materially different points
for "alphabet soup" and "butter" is keying on **the command embedding**, not on
the world — it has memorized a command→location association exactly as the
calibrated expert memorized an offset and the grasp head memorized
proprioception.

**Registered prediction, sharpened.** Soup and butter place points differ by
**well above 0.40 cm** (the true basket separation) — plausibly by several
centimetres, since a several-centimetre error is what drops a grocery outside a
basket and produces `grip_close_rate` ~0.26 with `eef_obj_dist_min` at
baseline. **If instead they differ by ≲0.4 cm, the place head is grounded, the
latched-place-point story cannot explain the collapse, and I must look
elsewhere despite cell B** — which is the falsification I registered when
queueing the measurement, now with a number attached to it.

**If confirmed, this is a fourth memorization layer**, not a restatement of the
three in the abstract (expert offset constant, iteration-coupled selection
loop, grasp-head proprioception shortcut). It was invisible to every probe in
this campaign because the benchmark never exercises it: all ten tasks share one
basket, so a memorized command→place map and a grounded one are behaviourally
identical *until you swap the instruction*. The swap is what made a
fixed-placement shortcut observable — the same move the whole paper argues for,
applied to the one component we had not thought to audit.

### PREDICTION UPDATE, recorded before the generalization cell runs (2026-08-06)

I registered, before the decomposition: *"butter's baseline clearly above zero,
swapped cell at or near 0/10, same direction as soup."* **I now expect the
opposite, and the reason is the mechanism I did not have when I registered it.**

Checkpoint metadata, read from the files:

| head | `data_dirs` |
|---|---|
| `goal_heads_v5` (flagship, all soup cells) | `teacher_rand_full`, `teacher_rand2` — **soup only** |
| `goal_heads_v8` (the generalization cell) | + `butter_rand`, `butter_rand2`, `cream_rand` — **soup, butter and cream** |

The mechanism now on the table is: the place head latches `(x, y)` from the
**command embedding**, and a command it never trained on is out-of-distribution,
so it emits a wrong place point. Under v5 that is exactly butter's situation —
v5 has only ever seen soup's embedding. **Under v8 it is not**: both soup's and
butter's embeddings are in-distribution, so swapping between them should
produce a *correct* place point either way and the cell should **not** collapse.

**Updated prediction: the v8 butter-task swap does NOT collapse** — it lands
near its own baseline, in the opposite direction from what I registered an hour
ago.

**Why this is an update and not a goalpost move.** The original prediction was
registered before the decomposition identified the channel, and it treated the
collapse as a property of "swapping the instruction". The mechanism says the
collapse is a property of *feeding the place head an embedding it never trained
on*, which is a different thing that merely coincided under v5. The update is
derived from checkpoint metadata that anyone can read, it is recorded before
the cell runs, and it is **riskier** than the original: a collapse now
falsifies my mechanism rather than confirming a vague "instructions matter".

**What each outcome means.** *No collapse* → strong support for the
OOD-embedding mechanism, and a striking corollary: the swap's catastrophe is a
**training-coverage** artifact, not an architectural one, and is repaired by
including the command in training. *Collapse anyway* → both commands are
in-distribution for v8, so an OOD embedding cannot be the cause, and the
latched-place-point story needs rework despite cell B.

The original prediction stays on the record above, unedited, marked superseded.

### DECOMPOSITION COMPLETE: the collapse is entirely the embedding channel (2026-08-06)

**Cell B — prompts from the real soup instruction, task embeddings from
butter: 0/10** (`eval_results/dec_embonly_butter`, `mean_success` 0.0,
artifact-verified). Its per-trial pattern is `0000000000`, **identical to the
full swap on all ten trials.**

The four cells form a 2×2 factorial that was not designed as one but reads as
one exactly:

| | embeddings = soup | embeddings = butter |
|---|---|---|
| **prompts = soup** | **7/10** (baseline) | **0/10** (cell B) |
| **prompts = butter** | **6/10** (cell A) | **0/10** (full swap) |

Marginals: prompt channel **0.35 vs 0.30** (no effect); embedding channel
**0.65 vs 0.00** (total). One main effect, no interaction.

Exact McNemar on every pair:

| comparison | discordant | exact two-sided p |
|---|---|---|
| baseline vs cell A (prompt swapped) | 3 | **1.0000** |
| baseline vs cell B (embedding swapped) | 7, one-way | **0.0156** |
| baseline vs full swap | 7, one-way | **0.0156** |
| cell A vs cell B | 6, one-way | **0.0312** |
| cell B vs full swap | 0 | **identical patterns** |

**Cell B reproduces the full swap trial for trial.** Swapping the embedding
alone is not merely *sufficient* for the collapse — it accounts for all of it,
with nothing left for the prompt channel to explain.

**The registered prediction is confirmed**, and the earlier falsification is
now explained rather than merely recorded. The instruction-swap prediction
failed because I predicted at the level of "the instruction" when the system
has two independent instruction-driven channels: one inert, one total.

**What the system actually is.** Combining this with the architecture:
`goal_machine.step(proprio)` replaces the plan wholesale; the grasp head's
signature contains no text; the place head runs once per episode caching
`set_place(place_head(command_emb))`. So **object selection, approach and
grasping run with no language input whatsoever, and the entire language channel
is one latched (x, y) place point.** You can ask this policy for the butter and
it will find, approach and grasp the alphabet soup at baseline rate (6/10 vs
7/10, p = 1.0) — and it fails only when the one cached coordinate is wrong.

**Caveat still live, not dropped now that the result is clean.** The per-trial
figures depend on the harness noise floor, which is queued and still
unmeasured. The rate-level conclusions do not: cell B vs baseline is 7 one-way
discordant pairs, and cell B vs full swap is a perfect 10/10 pattern match,
neither of which a modest noise floor could manufacture.

**Submission updated with the decomposition (2026-08-06).** Abstract now
carries it (the 2×2 localisation and "one latched place coordinate"); the
addendum's "attribution pending" paragraph is replaced by the result; the
role-binding table is compressed to prose (the manuscript keeps all cells with
run ids). Body runs to just past 5 pages; references pp 6–7. **I stopped
trimming**: three rounds of compression were improving concision, the fourth
would have cut verified content to hit a page number I set myself, and the
package has no venue yet. Recorded because "it fit in 5 pages" is not a
result, and cutting evidence to preserve a formatting target is the kind of
quiet distortion this log exists to catch.

### The generalization test is VOID — my design error, not a result (2026-08-06)

**Butter task 6, v8 head, correct instruction, seed 20: 0/10**
(`eval_results/gen_butter_base`, `mean_success` 0.0, artifact-verified).
`eef_obj_dist_min` runs 0.056–0.146 m across all ten trials: the machine never
gets near the object.

**The baseline is dead, so the cell measures nothing.** A swap ablation asks
whether a manipulation destroys performance; with a baseline of zero there is
no performance to destroy, and the swapped cell's number — whatever it turns
out to be — is uninterpretable. I registered "baseline clearly above zero" as
the precondition, and it failed. **Instrument #7, failed at its own
pre-registered gate.**

**The cause is my configuration choice, and it was avoidable.** I reused the
soup flag string verbatim, which carries **no `--goal-kwargs`**. But butter's
published cells were never produced in that configuration — they used
`{"anchor_band":0.04}` or the early-latch `{"latch_tol":9.0,"z_freeze":1.0}`
split (both recorded in this log, both present in the pod's own scripts). I
copied the flags that make *soup* work and applied them to *butter*, which is
the same "constant inherited from a different regime" defect the paper
documents 29 instances of. Finding it in a run I designed today, hours after
writing that section, is the honest version of the lesson.

**Consequence.** The question "does the collapse generalize beyond task 0 and
the flagship head?" is **UNANSWERED**, not answered negatively. It is not
evidence against the decomposition, which stands on its own four cells; it is
simply a test that did not run. The swap cell now executing against this dead
baseline will be recorded and discarded.

**Re-queued correctly**: butter task 6, v8, with `{"anchor_band":0.04}` — the
configuration under which butter's cells were actually measured — baseline and
swap. The registered prediction is unchanged from the update logged before this
failure: **the v8 swap should NOT collapse**, because both soup's and butter's
command embeddings are in v8's training corpora, so neither is out of
distribution for its place head. If the corrected baseline is *also* near zero,
the generalization question stays unanswered and I will say so rather than
tuning until a baseline appears.

**Honesty fix on the size claim (2026-08-06).** The abstract calls MicroVLA
"the smallest surveyed **language-conditioned** policy" while the same abstract
now reports that its language channel reduces to one latched coordinate. A
sharp referee would call that having it both ways. Corrected in the bounded
claim: *"language-conditioned" names the **reference class** — systems that
accept a language instruction — and is not a claim about how much work that
instruction does in ours.* With the pointed addition that a reader who thinks
this disqualifies the artifact from the class should read the size row as
**smaller still, not larger** — the qualifier cannot be used to inflate the
comparison in our favour, and saying so closes the only direction the ambiguity
could have been exploited.

**Void cell recorded and discarded (2026-08-06).** `gen_butter_swap`
`mean_success` **0.0** — as expected against a 0/10 baseline, and
uninterpretable for the reason logged above: a swap ablation with no baseline
performance measures nothing. Recorded so the run is not silently missing from
the ledger, and explicitly **not** counted as evidence in either direction.
Both cells (`gen_butter_base` 0.0, `gen_butter_swap` 0.0) are superseded by the
re-queued `gen2_*` pair under the `anchor_band` configuration.

The noise-floor control is now running.

### Arithmetic error in my own tally, caught by recounting (2026-08-06)

The submission claimed the deployed-binding question was *"attempted five times
and abandoned five times"* and then **enumerated six**: a projected-origin
ground truth, its sign-corrected successor, **three** text-tower probes, and
the spread metric. I introduced that error earlier today when adding the spread
metric to a list of five without incrementing the count. A referee counting the
items in the same sentence would have found it immediately.

The falsified-prediction tally was stale for the same reason — it said three
while five now exist: bank binding, chocolate pudding, box-centre spread,
proposal scarcity, and the instruction swap.

Corrected in the abstract and the addendum: **five falsified predictions, six
instruments abandoned before a seventh succeeded.** Both numbers were recounted
against the enumerated items rather than adjusted by one, because the error
came from trusting a running count instead of the list under it. The tallies
are a claim like any other, and this log's whole discipline is that a number
you did not re-derive is a number you are guessing.

### NOISE FLOOR MEASURED: exactly zero — the harness is deterministic (2026-08-06)

**Baseline re-run bit-identically: 10/10 trials identical, on every field.**
(`eval_results/noise_baseline_repeat`, `mean_success` **0.7**, matching the
original 0.7.) Not just success/failure — `steps`, `src_detect_rate`,
`src_conf_mean`, `grip_close_rate`, `eef_obj_dist_min`, `eef_obj_dist_at_20`
and `eef_obj_dist_final` all match exactly, trial for trial. Only wall-clock
time differs.

| | result |
|---|---|
| success flips | **0 / 10** |
| trials with any telemetry difference | **0 / 10** |

**The registered prediction is confirmed** ("reproduces `1101110011` exactly or
flips at most one trial"), so no downgrade is triggered. **Every per-trial
figure reported today stands as stated**, including the one most at risk: cell
B reproducing the full swap's pattern on 10/10 trials is a real equivalence,
not a coincidence of two noisy runs.

**It also sharpens the decomposition rather than merely licensing it.** With a
zero noise floor, the three trials where cell A differs from baseline (1, 7, 9)
are **genuine effects of the prompt swap**, not scatter. So "the prompt channel
is inert" is *too strong*, and the precise statement is:

> The prompt swap measurably perturbs individual trajectories — 3 of 10 trials
> flip, deterministically — but does not systematically degrade success
> (6/10 vs 7/10, exact p = 1.0). The embedding swap destroys all ten.

That is a better result than a flat null: swapping the detected object's name
*does* change what the machine does moment to moment, and it still doesn't
change whether the task succeeds. The channel carries signal that the outcome
is indifferent to.

**A property I had assumed all campaign without measuring.** Every paired cell
in this paper — memorized vs repaired, the audit-stack inversion, the rebuild
matrix — compares runs that differ only by the manipulation under test. That
was load-bearing for all of it and was never checked until now, on the seventh
instrument of a campaign about unverified assumptions. It should have been the
first measurement taken.

**Consequence for `REPRODUCE.md`, which is now wrong.** It tells a reproducer
to "expect the Wilson interval, not the point estimate", reasoning that these
are Bernoulli cells. For a *different* draw that is right; for **re-running the
shipped command on the same stack it is wrong** — the correct expectation is
**exactly 0.700**, and any deviation is evidence about the stack, not about
sampling. Correcting it, because guidance that tells someone to accept a wrong
answer is worse than no guidance.

### CONFIRMED: a fourth memorization layer — the place head memorized command→location (2026-08-06)

The registered threshold was **0.40 cm** (the true task-to-task basket
separation). Measured, running each instruction through the production
`ClipTaskEncoder` and the released place head:

**v5 (flagship; trained on soup corpora only):**

| command | place point (x, y) | error vs true basket (+0.2605) |
|---|---|---|
| alphabet soup | (−0.0010, **+0.2569**) | **0.36 cm** — correct |
| butter | (+0.0017, **+0.1155**) | **14.5 cm** — nowhere near it |
| cream cheese | (−0.0003, **+0.1303**) | **13.0 cm** — nowhere near it |

Pairwise: soup–butter **14.15 cm**, soup–cream **12.66 cm**. That is **35× the
registered threshold**, in the predicted direction. The prediction is confirmed
and the falsification condition (≲0.4 cm ⇒ the head is grounded) is not met.

**The basket never moves** (0.22–0.40 cm across all ten tasks, *less* than the
0.8 cm within-task jitter). A place head that had grounded on the basket would
emit one point for every command. This one emits the *correct* point for the
command it trained on and points 13–14 cm away for commands it did not. **That
is memorization of a command→location association, a fourth layer to add to the
expert's offset constant, the iteration-coupled selection loop, and the grasp
head's proprioception shortcut.**

**The OOD reading tested directly, and confirmed.** If the mechanism is
"untrained command ⇒ wrong place point", a head trained on *all three* commands
should emit the correct point for all three. Re-running the identical
measurement with **v8** (`data_dirs` include butter and cream):

| command | v8 place point | error vs true basket |
|---|---|---|
| alphabet soup | (−0.0009, +0.2570) | 0.35 cm |
| butter | (−0.0052, **+0.2504**) | **1.01 cm** |
| cream cheese | (−0.0053, **+0.2506**) | **0.99 cm** |

Pairwise spread collapses from **14.15 cm → 0.78 cm**, i.e. into the basket's
own task-to-task variation. Same architecture, same measurement, only the
training coverage differs. **The catastrophe is a training-coverage artifact,
not an architectural one** — exactly the corollary registered before the
generalization cell was queued.

**This closes the causal chain end to end**, every link measured rather than
inferred: the instruction reaches behaviour only as `set_place(place_head(
command_emb))` (code); v5's place head maps an unseen command to a point 14 cm
short of the basket (measured); the machine therefore grasps the correct object
and releases it over open table, which is precisely the observed signature —
`eef_obj_dist_min` at baseline (~0.013, the object *was* reached) with
`grip_close_rate` falling 0.675 → 0.257 (it was let go); and the cell collapses
7/10 → 0/10 (measured, exact p = 0.016).

**It also predicts the still-running gen2 cell.** v8's soup and butter place
points differ by 0.78 cm, so swapping between them should be harmless and the
v8 butter-task swap should **not** collapse — the prediction already on record.
The gen2 result is now a direct test of this mechanism rather than a generic
generalization check.

**`REPRODUCE.md` corrected (2026-08-06).** It told reproducers to "expect the
Wilson interval, not the point estimate". With a measured zero noise floor that
is wrong for the shipped commands: on a matching stack they return **exactly**
0.700 and 0.520, and any deviation is evidence about the stack rather than
sampling. The Wilson intervals answer a different question — uncertainty over
*which init states you draw* — so they apply when you change the seed or
protocol, not when you re-run the shipped command. The old wording invited a
reproducer to accept a wrong answer as "close enough", which is worse than
giving no guidance at all.

**Fourth layer propagated to all documents (2026-08-06).** Abstract, the audit
contribution, the manuscript's abstract/claims/§6 heading: "three layers" →
**four**, with the place head named. One more stale count fixed in the same
sweep (contribution #4 said "five ground-truth-seeking instruments failed";
six did).

**The title still holds, and holds harder.** "Learning Location, Not Looking"
was written about the grasp side. The fourth layer is a component that
literally learned a *location* keyed to a *word* — against a basket that never
moves — and it was the one component no probe in this campaign had thought to
audit, precisely because the benchmark never exercises it. The paper's own
thesis found a defect the paper's own instruments had missed.

**Contribution #4 broadened to a pair of probes (2026-08-06).** The
ground-truth-free identity test and the instruction swap are the same kind of
instrument and belong together: neither needs annotation, both need only a
second forward pass, and **they fail differently** — the first exposes a
grounding stage that *ignores* the instruction, the second a stage that
*memorized* it. Stated as a pair rather than leaving the swap as a one-off
diagnostic, because the swap is what found the fourth layer and it works
exactly where a fixed benchmark cannot: when every task shares a target, a
memorized command→location map and a grounded one are behaviourally identical
until the command is changed. That is a transferable recommendation, not a fact
about MicroVLA.

### gen2 baseline live at 4/10 — and a power caveat recorded before the swap result (2026-08-06)

**Butter task 6, v8 head, `anchor_band` config, correct instruction: 4/10**
(`eval_results/gen2_butter_base`, artifact-verified), pattern `0100001110`.
A live cell, consistent with butter's published range (4–7/10), so unlike the
void first attempt this test measures something.

**But the test is asymmetric in power, and I want that on record before the
number arrives — it cuts against my own prediction.** With a 4/10 baseline, a
total collapse to 0/10 yields 4 one-way discordant pairs: exact two-sided
**p = 0.125**, *not* significant. So:

- If the swap lands near **4/10**, that is direct evidence of **no effect**, and
  my registered prediction (no collapse, because v8's soup and butter place
  points differ by only 0.78 cm) is **supported**.
- If the swap **collapses to 0/10**, my prediction is **wrong** — but the cell
  is **underpowered to establish that at p < 0.05**, and I would have to report
  it as suggestive-but-not-significant rather than as a refutation, and say why.

This asymmetry favours my own hypothesis, which is exactly why it belongs in
writing beforehand. I registered a related caveat when queueing the first
attempt ("butter's weaker baseline gives p≈0.03 at best... I will NOT pool the
two tasks to manufacture a smaller p"); the realised baseline is 4/10 rather
than the 5/10 assumed there, so the achievable p is 0.125, not 0.03. **The
honest reading is that this cell can confirm the absence of a collapse but
cannot, at n=10, establish its presence.** If it collapses I will say the
mechanism is in doubt and that a larger n is needed, not that the result is
null.

### gen2 FINAL: the swap catastrophe is repairable by training coverage (2026-08-06)

**Butter task 6, v8 head, told "alphabet soup": 3/10 against its own 4/10
baseline** (`eval_results/gen2_butter_swap` 0.3, `gen2_butter_base` 0.4, both
artifact-verified; override verified in the log).

```
trial:     0 1 2 3 4 5 6 7 8 9
baseline:  0 1 0 0 0 0 1 1 1 0    4/10
swapped:   1 1 0 1 0 0 0 0 0 0    3/10
```
5 discordant pairs, 3 one way and 2 the other: **exact two-sided p = 1.0000.**
No effect. **The registered prediction is confirmed** — the updated one, which
reversed my original registration on mechanistic grounds before the cell ran.

**The contrast is the result:**

| head | trained on | baseline | swapped | exact p | place-point spread |
|---|---|---|---|---|---|
| **v5** (flagship) | soup only | 7/10 | **0/10** | **0.0156** | **14.15 cm** |
| **v8** | soup + butter + cream | 4/10 | **3/10** | **1.0000** | **0.78 cm** |

Same architecture, same code, same swap manipulation, opposite outcomes. The
only difference is whether the swapped-in command was in the head's training
corpora — and the behavioural result tracks the place-point measurement exactly:
14 cm of place error destroys the cell, 0.8 cm does nothing.

**This converts the fourth memorization layer from a defect into a repair.**
The paper's shape was audit → repair, with ten teleported episodes fixing the
grasp head's proprioception shortcut. It now has a second instance of the same
lesson: **both memorization layers we repaired were fixed by adding a small
amount of the *right* data** — ten placement-teleported episodes for the grasp
head, and the commands themselves for the place head. Neither needed an
architectural change.

**Power, as registered before the number arrived.** With a 4/10 baseline this
cell could confirm the *absence* of a collapse (which it did, p = 1.0) but could
not have established its presence at n=10 (a full collapse would have given
p = 0.125). The result falls on the side the design can actually support, and I
recorded that asymmetry beforehand precisely because it favours my hypothesis.

**What is now closed.** Three controls, three registered predictions, three
confirmations: noise floor exactly zero (10/10 bit-identical); place-point
memorization at 35× the registered threshold; and the training-coverage repair.
Generalization beyond task 0 and the flagship head — the question the void
first attempt could not answer — is answered: **the mechanism reproduces on a
second task with a different head, in both directions.**

**Final read-through of the rendered PDF (2026-08-06).** Two stale details
survived every source-level edit and were only visible in the output: the
addendum's section title still read "and a falsified prediction" (there are
five), and one line still credited the identity test with succeeding "where
five instruments failed" (six did). Both fixed. Section retitled to
"five falsified predictions, one mechanism retracted, and a fourth memorization
layer", which is what the section now actually contains.

Reading the *rendered* document rather than the source has now caught four
defects today — an internal contradiction, a parameter total, an unscoped use
of "grounding", and these two counts. None were visible in the `.tex`.

### Audited the instrument behind every `eef_obj_dist` number — it holds, by luck (2026-08-06)

Building a "which object did the gripper actually go to?" probe meant reading
`_sim_object_pos`, and it does **not** look up the task's target. It returns
*the first non-container body in `obj_body_id`*. Every `eef_obj_dist_min`,
`eef_obj_dist_at_20` and `eef_obj_dist_final` in this log and in the paper
depends on that being the commanded object.

**Verified rather than assumed** (`scripts/verify_objpos.py`, now shipped):

| task | `obj_body_id` order | picked | matches command? |
|---|---|---|---|
| 0 soup | `alphabet_soup_1`, basket, salad_dressing, cream_cheese, … | `alphabet_soup_1` | **yes** |
| 1 cream | `cream_cheese_1`, basket, alphabet_soup, milk, … | `cream_cheese_1` | **yes** |
| 6 butter | `butter_1`, basket, tomato_sauce, orange_juice, … | `butter_1` | **yes** |

**All the numbers stand** — including today's load-bearing ones (the butter
swap reaching the soup at `dmin` 0.006–0.013 while gripping collapsed, which is
the observation that located the failure downstream of approach).

**But it holds by coincidence, not construction.** LIBERO's BDDL files happen
to declare the target object first; nothing in the code requires it. A suite
ordering its objects differently would silently report distance to a
*distractor*, and every derived conclusion would invert without any error being
raised. That is the same shape as the 29 catalogued defects — an unlabelled
assumption that happens to be true — so it is now labelled in the docstring
with the verification and the re-check command, rather than left as a quiet
dependency.

This is the seventh instrument audit of the campaign and the first where the
instrument passed. Worth noting which way that cuts: the six failures were
found by testing; this one would never have been examined if I had not been
building something else.

### The two probes packaged as a reusable artifact (2026-08-06)

The most transferable thing this campaign produced is a *pair of probes*, and
until now they existed only as scratch scripts on a pod — which makes them a
claim rather than a contribution. Packaged as `eval/probes.py`, with
`tests/test_probes.py` (8 tests, CPU-only, mock-only, no network, per the repo
contract). **613 tests pass.**

`prompt_agreement(perception, chain_a, chain_b, frames)` — detects a grounding
stage that **ignores** the instruction, by running two objects' chains over the
same frames and comparing the boxes they select. No annotation, because it
compares two prompts against *each other* rather than against a truth.

`instruction_swap(baseline, swapped)` — detects a stage that **memorized** the
instruction, via exact McNemar on paired cells.

**Validated against every cell measured today**, reproducing each exactly:

| cell | result | probe verdict |
|---|---|---|
| v5 soup, full swap | 7/10 → 0/10, p=0.0156 | INSTRUCTION-SENSITIVE |
| v5 soup, prompt channel only | 7/10 → 6/10, p=1.0000 | NO SIGNIFICANT EFFECT |
| v5 soup, embedding channel only | 7/10 → 0/10, p=0.0156 | INSTRUCTION-SENSITIVE |
| v8 butter, full swap | 4/10 → 3/10, p=1.0000 | NO SIGNIFICANT EFFECT **+ power warning** |
| noise floor, baseline vs repeat | 7/10 → 7/10 | NO EFFECT (identical every trial) |

**The guardrails are the point, not the arithmetic.** The verdicts refuse to
over-claim in the three ways this campaign actually got burned: a cell with
<5 co-detected frames returns INCONCLUSIVE rather than a rate; a discriminating
result says explicitly that it *cannot see whether the discrimination is
correct*; and an underpowered cell prints its own ceiling — the v8 butter row
above volunteers that a 4/10 baseline could not have reached p<0.05 even under
total collapse, which is the caveat I had to remember to register by hand this
morning. Unpaired cells raise rather than silently truncating, because
truncating would fabricate a pairing.

A finding that "a VLA's entire language channel can be one cached coordinate"
is worth more to the field as something another lab can check in an afternoon
than as a fact about MicroVLA. This is that, and it costs a second forward pass
and no labels.

### PRE-REGISTERED: which object does the machine actually reach? (2026-08-06)

The last open question of the multi-object campaign is why cream fails. Six
instruments could not answer it, and every one of them asked the question in
**image space** — needing a camera projection, which is exactly what broke
them. *Which object did the gripper go to?* is a **world-space** question, and
world-space object positions read cleanly (proven twice: the pinning
measurement, and today's verification of `_sim_object_pos`).

**Instrument** (`_sim_all_object_pos`): log every scene body's world position
each step, and record the body nearest the end-effector. No projection, no
in-frame filter, no threshold — the two failure modes that killed instruments
1–2 and the filter that killed instrument 3 are all absent by construction.
613 tests pass; mock path unaffected.

**Registered prediction.** On soup (crosses 35/50) the nearest body at the
gripper's closest approach is the **commanded object** on most trials. On cream
(0/10) it is **a different object** on most trials — the identity-blind "box"
prompt selecting a distractor is the natural reading of a machine that servos
accurately (`eef_obj_dist_min` ~0.07) to something that is not the target.

**Registered alternative, which would be more interesting.** If cream's nearest
body *is* the cream cheese on most trials, then binding is **not** cream's
problem: the machine reaches the right object and fails to grasp it, and the
failure is in grasp geometry (height, approach angle, gripper width) rather
than identity. That would redirect the whole multi-object account, and the
0/10-under-every-binder result would finally make sense — the binders were
fixing something that was not broken.

**Recorded before running because I have a stake in the first answer**: the
misbinding story is the one that follows from today's identity-blind finding,
and it would be the tidier result. The alternative would mean four binder
cells, a prototype bank and a rerank study were all aimed at the wrong stage.

### FALSIFIED, and it redirects the multi-object account: cream reaches the right object every time (2026-08-06)

The seventh instrument answered the question six others could not, and it
**falsified my registered prediction** — the sixth falsification of this
campaign, and the most consequential, because it corrects an account rather
than merely closing a gap.

**Nearest scene body to the end-effector, world space, no projection:**

| | soup (crosses 35/50) | cream (0/10) |
|---|---|---|
| nearest at closest approach | `alphabet_soup_1` **6/6** | `cream_cheese_1` **6/6** |
| nearest at jaws-close | `alphabet_soup_1` **6/6** | `cream_cheese_1` **6/6** |
| closest approach (m) | 0.008–0.053, mean ~0.029 | 0.028–0.082, mean **~0.070** |
| distance at jaws-close (m) | 0.012–0.084, mean ~0.052 | 0.040–0.109, mean **~0.094** |
| margin over 2nd-nearest | 3–30× | **~1.5×** |

**Cream is not misbinding. It identifies and approaches the commanded object on
every trial, stops ~7 cm short, and closes the gripper on air ~9.4 cm away.**
I predicted a distractor; the data says the target, unanimously. The prediction
is falsified and the registered alternative — "binding is not cream's problem;
the failure is grasp geometry" — is what the evidence supports.

**This resolves a result that had been puzzling all campaign.** The
role-binding sub-study found that *offline binder accuracy dissociates from
deployed success*: cream scored 0/10 under **every** binder — none, crop-CLIP
rerank, mean prototype (0.613), 1-NN bank (0.902) — with no ordering by
accuracy. That looked like a mystery about binder quality. It is not a mystery:
**binding was never cream's bottleneck**, so improving it could not have moved
the cell, and the four binders were aimed at a stage that was working. A
sub-study I re-opened this morning on the grounds that "the binders own their
failure" is better described as having tested the wrong hypothesis throughout.

**What is now the honest multi-object account.** Two of four objects cross. The
two that do not are *not* blocked at role binding — that mechanism was
retracted this morning on other grounds, and this closes the door on it from a
second direction. Cream's machine reaches the right object and fails to close
on it, which points at grasp geometry (predicted grasp height/depth, approach
angle, gripper aperture) and is consistent with the previously measured
on-manifold goal error being worst for cream (1.58 cm vs butter 1.34, soup
1.06) — a 1.6 cm on-manifold error becoming a 7 cm deployed miss is the
paper's own on-manifold limit yet again.

**What I am not claiming.** n=6 per object, one configuration, two objects. That
cream approaches correctly is unanimous and unambiguous (6/6, and the margin
over the runner-up is never below 1.5×), but *why* it stops short is now the
open question and this instrument does not answer it. The grasp-geometry
reading is where the evidence points, not something measured.

**Identity-blindness stands and is unaffected.** It is a property of the prompt
stage, measured directly and confirmed behaviourally by the 2×2. What today's
result removes is the *inference* that identity-blindness explains cream — it
does not, because cream's binding lands on the right object anyway.

### The miss decomposed: cream's failure is lateral, not vertical (2026-08-06)

"Stops 7 cm short" does not say *which* 7 cm. Decomposing each trial's
closest-approach residual into lateral (xy) and vertical components, from
telemetry already on disk:

| | total | lateral (xy) | vertical (eef above object) | vertical share |
|---|---|---|---|---|
| soup (crosses 35/50) | 0.0294 m | 0.0186 m | **+0.0190 m** | **66%** |
| cream (0/10) | 0.0698 m | **0.0688 m** | +0.0040 m | **6%** |

**Cream's descent is not the problem — its height is better than soup's.** It
settles 4 mm above the object's centre where soup settles 19 mm above, and then
closes the gripper 6.9 cm to the *side*. Soup's residual is 66% vertical, which
is the harmless direction (the fingers extend below the end-effector frame, so
hovering 2 cm high still grasps); cream's is 94% lateral, which is fatal —
there is nothing between the fingers.

This sharpens yesterday's account twice over. "Grasp geometry" was right but
vague, and my first instinct on seeing "stops short" was a depth/descent
failure — **wrong, and refuted by data I already had before I could act on
it.** The specific defect is a **lateral goal-position error**: the grasp head's
predicted xy is ~7 cm from the object in deployment while being **1.58 cm
accurate on teacher trajectories** (the previously measured on-manifold figure).
A 4.4× degradation from on-manifold to deployed is the paper's central
limitation stated as a number, in the component where it costs the task.

**Why this matters more than a per-object fix.** The tempting response is a
cream-specific xy offset, which is precisely the `hang_comp`-shaped constant
this paper spends four layers criticising; it would "cross" a third object by
memorising its location. The finding worth keeping is the measurement: the goal
head's xy generalises 4.4× worse off-manifold than its z, and that asymmetry —
not a missing constant — is what blocks the third object.

**Not claimed.** n=6, one object pair, one configuration. Whether the lateral
error is systematic (a fixed bias, correctable in principle) or scattered
(a variance problem) is not measured here — the mean is 6.9 cm but I have not
looked at its direction, and I am not going to infer it from six numbers.

### THE MULTI-OBJECT RESULT, EXPLAINED: the benchmark has two target positions, and the head memorized one (2026-08-06)

Measuring the *direction* of cream's systematic bias led somewhere the whole
campaign had missed.

**LIBERO-Object's ten tasks place their targets at exactly TWO table
positions**, 22.0 cm apart (from `results/placement_pinning.json`, the shipped
instrument):

| position | tasks |
|---|---|
| **A** (−0.1200, −0.2400) | 0 alphabet soup, 4 ketchup, **6 butter**, 7 milk, 8 chocolate pudding |
| **B** (+0.0500, −0.1000) | **1 cream cheese**, 2 salad dressing, 3 bbq sauce, 5 tomato sauce, 9 orange juice |

Within each group the agreement is 0.0–0.7 mm. This is a *stronger* statement
than §3's within-task pinning: not only is each task's placement fixed across
its 50 init states, **the ten tasks between them use two locations.** A policy
that memorizes two coordinates can look like it generalizes across ten tasks.

**And that is what ours did.** The objects that cross — soup (task 0) and
butter (task 6) — are **both at position A**, and the training corpora are
dominated by them (`teacher_rand_full`, `teacher_rand2`, `butter_rand`,
`butter_rand2` at A; `cream_rand` alone at B). Cream, the object that fails, is
the one at **position B**. Its grasp error is:

- **systematic**, not scatter — coherence |mean|/mean|v| = **0.976** across six trials;
- **6.7 cm**, versus soup's 1.4 cm;
- and aimed at **position A**: the mean error vector lies **7.4° off** the
  cream→A direction and covers **31% of the 22 cm gap**.

**The grasp head learned a location, not a look-up.** It is pulled toward the
table position where its training data lives, which is invisible on any object
placed there and fatal for the one that is not. This is the paper's title
demonstrated at the multi-object level, and it retires every competing account
of the multi-object result: not role binding (cream reaches the *commanded*
object 6/6), not binder quality (they were improving a working stage), not
descent (cream's height is better than soup's), not object appearance.

**Honest complication: chocolate pudding is at position A and still does not
cross.** So position is not sufficient. Pudding fails earlier and differently —
its scripted teacher never reached a grasp in five calibration iterations, so
no corpus exists for it at all. The account is "cream's failure is explained by
position-memorization; pudding's failure is upstream of the head and remains a
teacher failure", not "position explains everything".

**What this does NOT license.** A cream-specific xy correction would cross a
third object by memorizing its coordinate — the exact defect, now measured
twice over. The finding is diagnostic: the benchmark affords two-position
memorization, and our head took it. Fixing it means training data that covers
placement, which is what the ten-episode teleport repair did for the *dev*
protocol and what no corpus here does across positions.

**Registered follow-up, not yet run.** If the head is biased toward position A,
then the four *other* position-B objects (salad dressing, bbq sauce, tomato
sauce, orange juice) should show the same ~7 cm bias toward A, and the three
other position-A objects should not. That is a 9-task prediction from a
one-object measurement and it is cheap to test — one nearest-body run per task.

**Sharpened before running (2026-08-06).** Testing four further tasks with the
nearest-body instrument, n=4 each, same head/config as the cream and soup runs:

| task | object | position | prediction |
|---|---|---|---|
| 2 | salad dressing | **B** | lateral error **≥4 cm**, aimed within ~30° of the B→A direction |
| 5 | tomato sauce | **B** | same |
| 4 | ketchup | **A** | lateral error **<3 cm** |
| 7 | milk | **A** | lateral error **<3 cm** |

These objects have **no training corpus at all** — the head saw soup, butter
and cream only — so this also asks what an untrained object inherits. The
position-memorization account predicts it inherits *the memorized position*,
which is the whole point: an untrained object at A should be approached about
as well as a trained one, and an untrained object at B should be missed by
about the same 7 cm as cream. **Falsified if the two B objects come in under
4 cm, or if either A object exceeds 3 cm.** Success rates are not the measure
here and are not predicted — these tasks have no corpus and no configuration
tuned for them, so the *bias vector* is the observable, not the cell.

### RETRACTED, one hour later: the position-memorization account is falsified (2026-08-06)

The section above — "THE MULTI-OBJECT RESULT, EXPLAINED" — **is wrong, and I am
retracting it.** The 9-task test I registered to confirm it refuted it instead.
Seventh falsified prediction of the campaign; second retraction of the day.

**Full results, n=4–6 per task, same head and configuration throughout:**

| task | position | corpus | mean lateral error | coherence | registered prediction |
|---|---|---|---|---|---|
| 0 alphabet soup | A | **yes** | **0.0141 m** | 0.758 | — |
| 1 cream cheese | B | **yes** | 0.0672 m | 0.976 | met |
| 4 ketchup | A | none | **0.0563 m** | 0.960 | **MISSED** (predicted <0.03) |
| 7 milk | A | none | **0.1514 m** | 0.976 | **MISSED** (predicted <0.03) |
| 2 salad dressing | B | none | 0.1729 m | 0.894 | met |
| 5 tomato sauce | B | none | 0.1601 m | 0.813 | **MISSED** (42.5°, predicted <30°) |

**What kills it: milk.** Milk sits *at position A* — the spot I claimed the head
memorized — and is missed by **15.1 cm**, with its error pointing **+x, away
from A** (per-trial: +0.204, +0.061, +0.170, +0.161 in x). A head "pulled toward
A" cannot miss an object *at* A by 15 cm in the opposite direction. Three of
four registered criteria failed.

**What actually predicts the error is training coverage, not position:**

| | trained | untrained |
|---|---|---|
| at A | 0.014 (soup) | **0.056** (ketchup), **0.151** (milk) |
| at B | 0.067 (cream) | 0.173 (dressing), 0.160 (tomato) |

Trained objects: 1.4 and 6.7 cm. Untrained: 5.6 to 17.3 cm. That is a mundane
finding — a regressor is inaccurate on objects it never saw — and it is what
the data supports.

**What survives from the retracted section**, because it was measured
separately and does not depend on the account:
- **LIBERO-Object uses exactly two target positions** across its ten tasks
  (A and B, 22.0 cm apart, agreement 0.0–0.7 mm within each group). That is a
  true and, I think, notable fact about the benchmark, and it stands on the
  pinning artifact alone.
- Cream reaches the **commanded** object 6/6 (world-space, unambiguous).
- Cream's error is systematic (coherence 0.976) and *does* lie 6.9° off the
  cream→A direction. With milk in hand this is best read as **one object's
  coincidence**, not evidence of a pull, and I am no longer offering it as one.

**Why I got this wrong, since the pattern matters more than the error.** I had
three data points (soup at A works, butter at A works, cream at B fails), found
a variable that separated them, and checked its direction on the *one* object
that already fit. Two of three objects sharing a position is not evidence that
position is causal — it is the smallest number that can produce a coincidence.
The test that would have caught it immediately is the one I eventually ran, and
I ran it only because I had registered it before I believed the account. **The
pre-registration is what saved this from entering the paper**: the account
never reached `main.tex`, `MANUSCRIPT_v2.md`, or `README.md` (verified: zero
occurrences), because I wrote the test before I wrote the claim into the
documents.

**Status of the multi-object account.** Cream's failure is a lateral grasp
error, systematic, ~7 cm, on an object it *correctly identifies and
approaches*. Why the grasp head's xy degrades 4.4× off-manifold is
**unexplained**. Position is not the explanation.

### The failure localized from a second, independent signal (2026-08-06)

`src_center` — the detected image position of the source box — at each trial's
closest approach:

| | src_center (mean) | src_center (std) | lateral world error |
|---|---|---|---|
| soup (crosses) | (0.542, 0.686) | (0.219, 0.135) | 0.0192 m |
| cream (0/10) | **(0.813, 0.796)** | (0.086, **0.007**) | 0.0689 m |

**Cream settles with the object in the bottom-right corner of the wrist view**,
where soup settles near centre. And it does so *more consistently than soup
does* — a vertical std of **0.007** against soup's 0.135. The failure is not
noise; the machine converges confidently to the wrong place.

**This rules out perception and confirms the grasp head, from a signal
independent of the world-space measurement.** The detector finds cream
reliably (detection rate ~1.0, image position stable to 0.7% of frame height).
The goal machine then servos to the xy the *grasp head* predicts — not to
centring the object — so a head whose xy is ~7 cm off drives the arm to a point
from which the object necessarily appears off-centre. Both signals agree, and
they fail in the same direction: world-space says the gripper is 6.9 cm
lateral, image-space says the object ends up at the frame edge rather than
under the fingers.

**The chain, now measured at every link:** perception detects cream (rate ~1.0)
→ the grasp head predicts an xy ~7 cm from the object, systematically
(coherence 0.976) → the machine servos there and stops → the object sits at uv
(0.81, 0.80) instead of under the gripper → the jaws close on nothing at
~9.4 cm. Every step is measured; **only the reason the head's xy is wrong
remains open**, and it is not perception, not binding, not descent, and not
position.

**A detail worth keeping.** Cream's convergence is *tighter* than soup's while
being wrong. A diagnostic that looked only at variance — "is the servo
stable?" — would have scored cream as the healthier of the two. Stability is
not accuracy, and this is a clean instance of a metric that inverts if you
forget which one you are measuring.

### Is the head extrapolating? Measured — and the measurement is circular (2026-08-06)

Deployment uv at closest approach, against each object's own training-corpus uv
distribution (205 and 164 detections, 6 episodes each):

| | training u (p5–p95) | training v (p5–p95) | deployment uv | percentile |
|---|---|---|---|---|
| soup (crosses) | 0.068–0.896 | 0.196–0.803 | (0.542, 0.686) | **48% / 13%** |
| cream (0/10) | 0.063–0.908 | 0.087–0.801 | (0.813, 0.796) | **84% / 85%** |

Cream operates in the **sparse upper tail** of its training distribution while
soup operates in the bulk. That is suggestive — tail regions carry less
training mass, and a regressor is worse there — and it is the shape the
on-manifold limit would predict.

**But I cannot claim it as a cause, because the measurement is circular.** The
uv at closest approach is *where the machine ended up*, and where it ended up is
determined by the head's predicted goal. A head that errs by 7 cm will
necessarily leave the object at an unusual image position. So "cream's
deployment uv sits in the tail" is equally consistent with the tail *causing*
the error and with the error *producing* the tail. This instrument cannot
separate them, and nothing about the numbers tells me which direction the arrow
runs.

**Reported as inconclusive**, not as support. The tempting move — presenting an
85th-percentile input as evidence of extrapolation-driven failure — would be
reading a consequence as a cause, which is the same error class as this
morning's retracted position account and this afternoon's spread metric
(confounded by camera motion). Three instruments in one day undone by measuring
something downstream of the thing they were meant to explain.

**What would break the circularity**: intervene on uv rather than observe it —
feed the head corpus frames at controlled uv values and measure its xy error as
a function of uv, off the machine's own trajectory entirely. That is an offline
measurement, needs no simulator, and is the right next instrument. Recorded as
the open experiment rather than run now, because designing it properly matters
more than adding a fourth confounded number today.

**Status of the multi-object question**: cream's grasp-head xy error is
systematic (0.976), lateral (94%), ~7 cm, on an object it correctly identifies
and approaches, with perception, binding, descent, and position all excluded.
Why the head errs is **open**.

### The tail hypothesis REFUTED, non-circularly (2026-08-06)

The interventional design named above, run: evaluate the released head on
recorded corpus episodes, where uv is set by the **teacher's** trajectory
rather than by the head's own error, and bin its xy accuracy by how far the
source box sits from frame centre (|uv−0.5|max).

| eccentricity bin | cream (n=346) | soup (n=1149) | butter (n=317) |
|---|---|---|---|
| 0–25% | 0.0176 m | 0.0140 m | 0.0164 m |
| 25–50% | 0.0236 | 0.0133 | 0.0170 |
| 50–75% | 0.0236 | 0.0116 | 0.0158 |
| 75–90% | 0.0213 | 0.0124 | 0.0125 |
| 90–100% | **0.0147** | 0.0130 | **0.0115** |
| overall | **0.0208** | **0.0129** | **0.0153** |

**No eccentricity effect exists.** The head is as accurate at the frame edge as
at the centre on all three objects — for cream and butter the *most* eccentric
bin is the most accurate. The hypothesis that cream fails because deployment
pushes the head into a sparse uv tail is **refuted**, and refuted without
circularity, because corpus uv is not downstream of the head's error.

**A second refutation from the same table.** Cream's deployment uv (0.813,
0.796) has eccentricity |uv−0.5|max = **0.313**, *below* the corpus median of
**0.371**. Its deployment inputs are more central than its training inputs, not
more extreme. My earlier per-axis percentile framing (84th/85th) obscured this:
the axes are individually high but the *distance from centre* is not, because
the corpus itself sits near the frame edge (median 0.371, i.e. the object
typically appears at ~0.13 or ~0.87 along an axis). Reading two marginal
percentiles as "in the tail" was wrong, and the joint statistic says the
opposite.

**What survives, and it is worth having.** Cream's head is genuinely worse
on-manifold than the others — **2.08 cm** vs soup 1.29 and butter 1.53 — so
some of its deployed deficit is present already in training. But 2.08 cm
on-manifold against ~6.9 cm deployed is still a **3.3× degradation that uv does
not explain**. The remaining candidate is the visual features themselves
(`box_emb`, `frame_emb`) coming from viewpoints the teacher never visited —
off-manifold in appearance rather than in position. That is testable by the
same offline design and is **not** claimed here.

**Eighth refuted hypothesis of the campaign.** The tally is worth stating
plainly: of the mechanisms I proposed for the multi-object boundary — generic-
tail indiscriminability, box-centre spread, proposal scarcity, misbinding,
position memorization, descent failure, uv-tail extrapolation — **every one has
been tested and every one has failed.** What is established is where the defect
is *not*, plus a precise localization (systematic lateral xy error in the grasp
head, 0.976 coherence, on an object correctly identified and approached). The
cause is open, and after eight failures I am more confident in the localization
than I would be in a ninth guess.

### PRE-REGISTERED: the last untested candidate — appearance, not position (2026-08-06)

Eight mechanisms tested, eight refuted. One candidate remains and it is the
only one the offline design has not reached: the head's **visual** inputs
(`box_emb`) come, in deployment, from viewpoints the teacher never visited.
Position (uv) is now excluded non-circularly; appearance is not.

**Instrument.** `MICROVLA_LOG_EMB=1` logs the source-box embedding every 20th
real tick (off by default — 512 floats a tick would bloat every ordinary run).
Compare deployment embeddings against each object's own corpus embeddings
(`source_box_embs`, already stored in the `.npz`). Non-circular in the way the
uv measurement was not: an embedding's *distance from the training
distribution* is a property of what the camera saw, and while the machine's
path is downstream of the head, the appearance statistics of a viewpoint are
not manufactured by the head being wrong about xy.

**Registered prediction.** Cream's deployment `box_emb` sits **further from its
corpus distribution** than soup's does from its own — a larger normalized
distance, or a lower cosine to the corpus mean. If so, appearance-side
off-manifold drift is a live candidate for the 3.3× degradation.

**Registered falsification.** If cream's deployment embeddings sit *as close*
to their corpus as soup's do to theirs, appearance is excluded too, and the
grasp head's deployed error would have **no remaining input-side explanation** —
which would point at the head's own extrapolation behaviour rather than at any
property of what it is shown. That is a real possibility and I record it as
such: **nine tested mechanisms with nine refutations is a legitimate outcome**,
and would leave the localization (systematic lateral xy, coherence 0.976,
correct object) as the paper's honest final word on the multi-object boundary.

Also fixed while adding this: the first version referenced `os` without
importing it in `eval/policy.py` — 19 tests failed instantly, which is what the
suite is for. 613 pass now.

### CONFIRMED (finally): appearance-side off-manifold drift survives its test (2026-08-06)

Nine mechanisms proposed, eight refuted. **The ninth survives.**

| | dep. cos to corpus mean | dep. NN-cos to corpus | corpus self-NN baseline | **gap** |
|---|---|---|---|---|
| soup (crosses 35/50) | 0.9420 | 0.9812 | 0.9901 | **+0.0089** |
| cream (0/10) | **0.9011** | **0.9564** | 0.9836 | **+0.0272** |

Cream's deployment viewpoints sit **3.1× further from its own training corpus**
(NN-cosine gap) than soup's do from theirs, and its mean-direction cosine is
lower too (0.901 vs 0.942). The registered prediction is met on both statistics.

**Why the between-object design matters here.** Both corpora were baked with
the same detector stack, so even if that stack differs from deployment's, the
shift applies to *both* objects and cancels in the comparison. An absolute
threshold on cosine would have been uninterpretable; the ratio is not.

**What this is, stated carefully.** A surviving hypothesis, not an established
mechanism. It shows cream's deployment appearance is further off-manifold and
that cream's xy is worse — a correlation across **two objects**. Tempting
numerical coincidence: the drift gap ratio (3.1×) is close to the xy
degradation ratio (2.08 cm on-manifold → ~6.9 cm deployed, 3.3×). **Two ratios
agreeing at n=2 is not evidence of proportionality** and I am not offering it
as such; I note it only because someone will spot it and should see it already
flagged as coincidence-shaped.

**Registered now, before running: butter as the third point.** Butter crosses.
If appearance drift tracks failure, butter's gap should look like soup's
(~0.01), not cream's (~0.027). **If butter's gap is large while butter still
crosses, the pattern breaks** and drift is not sufficient — which would return
this to the same status as the previous eight. Running with n=3 trials, same
config, `MICROVLA_LOG_EMB=1`.

### The drift PRECEDES the error — circularity broken (2026-08-06)

The appearance result faced the same trap that invalidated the uv measurement:
a machine servoing to a wrong xy necessarily sees the object from unusual
viewpoints, so drift could be a *consequence*. Splitting deployment embeddings
by tick index separates the two, because at episode start the arm is at its home
pose and has not yet acted on any prediction.

| ticks | soup (crosses) | butter (crosses) | cream (0/10) |
|---|---|---|---|
| 0–100 (**before** error accumulates) | +0.0050 | −0.0031 | **+0.0441** |
| 100–300 | +0.0099 | −0.0036 | +0.0363 |
| 300+ (**after**) | +0.0107 | +0.0055 | **+0.0156** |

**Cream's drift is largest at the very first ticks and shrinks by 65% over the
episode.** A consequence of the trajectory error would grow as the error
accumulates — which is precisely what soup and butter show (0.0050→0.0107 and
−0.0031→+0.0055, small and rising, the signature of a machine gradually leaving
teacher-like viewpoints). Cream shows the opposite sign of trend from a large
initial offset. **The drift is there before there is any error to cause it.**

**This is a causal-direction argument, not a proof.** What it establishes is an
asymmetry that a consequence-only account cannot produce: the effect is
strongest when the cause would be weakest. No intervention has been run, and
n=3 objects × 3 trials (15 early-tick samples each) is small. But it is the
first mechanism in nine to survive both a registered falsification test
(butter, the other crossing object, has the smallest gap of all at +0.0007) and
a circularity check that killed two earlier instruments.

**Final status of the multi-object boundary.** Nine mechanisms proposed:
generic-tail indiscriminability, box-centre spread, proposal scarcity,
misbinding, position memorization, descent failure, binder quality, uv-tail
extrapolation — **eight refuted, each by measurement**. The ninth,
appearance-side off-manifold drift, survives: it separates both crossing
objects from the failing one, was predicted before butter was run, and precedes
the error it is proposed to explain. The honest claim is a **well-tested
surviving hypothesis with a measured localization** — systematic lateral xy
error in the grasp head (coherence 0.976) on an object it correctly identifies
and approaches, with the head fed visual features 3.1× further off its training
manifold than the objects that work.

**What would settle it, and is not claimed:** an intervention — train the head
with deployment-like viewpoints for the failing object and re-measure. That is
a corpus-collection experiment, not an analysis, and it is where this thread
ends for today.

### PRE-REGISTERED: the intervention — DAgger on the machine's own viewpoints (2026-08-06)

The appearance-drift hypothesis survived a falsification test and a circularity
check, but nothing has *intervened* on it. The decisive experiment: label the
machine's own off-manifold viewpoints and retrain the head on them. If drift is
the cause, this fixes the blocked object; if it is a correlate, it will not.

**The label is derivable, checked first.** Deriving grasp labels from the three
corpora (`train_goal.derive_labels`) gives grasp points that sit essentially
*on* the object: soup's mean grasp is 0.5 mm from the object centre laterally,
cream's 4.5 mm, butter's 1.2 mm, with 1.6–2.8 cm spread across episodes. So the
grasp label ≈ the object's true position, and the simulator supplies that for
**any** frame — including viewpoints the teacher never visited, which the corpus
by construction cannot label. Checked before building on it, because a scattered
offset would have made the whole design unsound.

**Protocol, fixed before collection.** Collect on **seed 0 (dev band)**, never
seed 20, so the held-out protocol stays uncontaminated — the same discipline
that governs every cell in this paper. `MICROVLA_LOG_FEATS=1` captures the
exact `(geom, box_emb, frame_emb, eef_xy)` the head consumed, every 10th tick,
paired with the simulator's object pose for the label. Features are consumed
once and cleared (without which 90% of the set would be duplicates).

**Registered prediction.** Fine-tuning on these off-manifold viewpoints reduces
cream's deployed lateral error materially below its current ~6.9 cm. Crossing
(>0/10 on held-out seed 20) would be strong confirmation; a large error
reduction without crossing would be partial.

**Registered falsification.** If the deployed lateral error stays near 6.9 cm
after training on the very viewpoints where it occurs, appearance drift is
**not** the cause — it joins the other eight, and the multi-object boundary is
reported as localized-but-unexplained, which is where it stood this morning.

**Registered guard against self-deception.** This is a *repair*, so it must not
be evaluated on the data that produced it: training draws seed 0, evaluation
draws seed 20, and I will report the seed-20 cell whatever it says. A repair
validated on its own collection band would be the iteration-coupled selection
loop this paper documents as Layer 2.

### DAgger fine-tune trained; held-out evaluation running (2026-08-06)

**102 deployment feature vectors** captured from cream rollouts on seed 0,
mixed with **1812 corpus samples** across all three objects (so the repair
cannot silently trade the working objects for the blocked one).

```
epoch  0  mean xy err 0.0288 m   (on the DAgger subset: 0.1120)
epoch 59  mean xy err 0.0136 m   (on the DAgger subset: 0.0255)
```

**On the machine's own viewpoints the head's error falls 11.2 cm → 2.55 cm, a
4.4× reduction**, and the corpus-mixed overall error also improves
(0.0288 → 0.0136), so the fit did not come at the corpus's expense.

**This number proves nothing on its own and is not the test.** Fine-tuning
reduces error on the data it fits — that is what fitting is. It is reported
because the *magnitude* is informative: 11.2 cm error on deployment viewpoints
before any correction confirms, from a fourth angle, that the head is badly
wrong exactly where the machine goes. Whether that is *causal* is what the
held-out cell decides.

Running now, per the protocol fixed before collection: **cream on seed 20**,
which the DAgger data (seed 0) never touched, plus **soup on seed 20** as a
regression check. Both will be reported whatever they say.

### THE INTERVENTION WORKED — partially, and the bound matters (2026-08-06)

**Cream, held-out seed 20, DAgger head: 2/10** — the first successes cream has
recorded in this entire campaign, against 0/10 under every previous
configuration, binder, prompt fix and head. Trials 4 and 7, with
`eef_obj_dist_min` **0.013** and **0.002** m.

```
trial:  0     1     2     3     4     5     6     7     8     9
succ:   F     F     F     F     T     F     F     T     F     F
dmin: .029  .030  .034  .048  .013  .045  .040  .002  .029  .040
```

**The targeted quantity moved, significantly:**

| measure | pre-DAgger (n=6) | DAgger (n=10) | test |
|---|---|---|---|
| lateral `dmin` mean | 0.0697 m | **0.0310 m** (−56%) | Mann-Whitney U=52/60, **p = 0.0197** |
| success rate | 0/10 | 2/10 | Fisher exact, **p = 0.474** |

**Read this exactly as it is.** The intervention **significantly reduced the
error it was designed to reduce** (56%, p = 0.020) on a protocol band the
training data never touched. The **crossing is not statistically significant**
at n=10 — 2/10 versus 0/10 gives p = 0.47, and I will not present cream as
"crossed" on that basis. This is the *partial* outcome registered before
collection: "a large error reduction without crossing would be partial."

**What it establishes about the hypothesis.** Appearance-side off-manifold
drift is now supported by **intervention**, not just correlation: training the
head on the machine's own viewpoints halved the lateral error at deployment.
That is the strongest evidence available for any of the nine proposed
mechanisms, and the only one to survive falsification, a circularity check, and
an intervention. It does not prove drift is the *whole* cause — a 56% reduction
leaves 3.1 cm of error, still enough to miss most grasps, which is exactly why
the success rate moved only to 2/10.

**Guard held.** Collection was seed 0, evaluation seed 20; the repair was never
scored on its own band. Had I evaluated on seed 0 this would be the
iteration-coupled selection loop the paper documents as Layer 2 — the defect
would have been mine, in the act of reporting it.

**Soup regression is running.** If the repair traded the working object for the
blocked one, it is not a repair, and that cell decides it. Reported whichever
way it lands.
