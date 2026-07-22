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