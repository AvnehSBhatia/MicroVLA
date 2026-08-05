# MicroVLA: A 30-Million-Parameter Vision-Language-Action Stack, Measured to the Centimetre

**Manuscript draft (2026-08-01).** This is the submission-shaped distillation of
the full experimental record in `paper.md` (cited throughout as §*n*); every
number here is measured and traceable to that log, `results/`, or
`eval_results/`. Claims this project explicitly does *not* make are in §9 —
read them before quoting the abstract.

---

## Abstract

We present MicroVLA, a language-conditioned vision-language-action (VLA) stack
whose deployed footprint is ~30 M parameters — 13 M of it a frozen
open-vocabulary detector reused as the *only* vision **and** text encoder, and
just **7.0 M trained parameters** — targeting CPU-class robot hardware
(Raspberry Pi 5, 7 servos). Three design decisions carry the size: (1) frozen
YOLO-World-S supplies frame embeddings, per-role object embeddings, and its own
CLIP text tower, so no separate language model exists anywhere in the stack;
(2) a JEPA-style predictive loop runs perception at 2 Hz and control at 30 Hz,
with 14 of every 15 ticks driven by a learned world model "dreaming" forward
through the same fusion pathway used for real evidence — dream evidence is a
continuous confidence fade, never a binary flag, making training-time modality
dropout and inference-time dreaming the *same code path*; (3) every module is
capped by an enforced parameter budget that the test suite fails on violation.

We report the world model beating persistence and linear-extrapolation
baselines under deployment-matched rollouts — and unaided closed-loop task
success of **zero across six controlled variants** of free-form action
regression (capacity, data aggregation, DAgger, input adaptation, objective
redesign), which we treat as the paper's central measurement problem. A
forensic chain of 28+ documented defects reduces the failure to missing
*structure*: a policy that must re-infer a persistent goal, a phase
commitment, and a control law from single noisy frames regresses to
hovering. Replacing the free [5×7] action head with **structured decoding**
— two small learned heads (0.24 M) that predict *where* to grasp and place,
driving a task-content-free servo shell (latch, P-law, one-way phases,
2D probe search) — takes the same trunk from 0.000 to **0.700 unaided**
on the benchmark protocol. We then audit our own number the way a reviewer
would, and find *three layers of placement memorization*: the benchmark
pins object poses (identical across all 50 init states), our grasp head
scored 0.700 while provably ignoring the image (an input-sensitivity probe
we introduce), and even the hand-calibrated expert encodes pose in its
"lever-arm" constant — which flips sign under a software-stack rebuild.
The repair is small and reproducible: ~50 teacher episodes recorded under
source-pose teleportation plus one nuisance-input augmentation. The
resulting head is certifiably visual (attribution: vision channels
1.1–1.8 cm, proprio parasitism 0.1 cm), closes the dev/held-out gap
(0.300 → 0.700), and produces the first successes on displaced objects
(0.400–0.500 at ±4 cm), all video-documented. We argue that at this scale
the binding constraints are decoding structure and *measurement
discipline* — the audit-repair loop, the substitution probe, and the
randomized-placement protocol are the transferable contributions, and we
propose the latter two as standard practice for fixed-placement
manipulation benchmarks.

---

## 1. Introduction

The prevailing answer to "how do we get a robot to follow language" is scale:
RT-2 (55 B), OpenVLA (7 B), and their descendants fine-tune internet-pretrained
VLMs into action decoders. A second wave — Octo (27–93 M), TinyVLA,
SmolVLA (450 M) — asks how far the recipe compresses. MicroVLA asks a
different question: **what is the minimum *new* machinery a language-conditioned
manipulation policy actually needs, if a frozen open-vocabulary detector is
taken seriously as a complete perception system?**

Our answer is architectural parsimony taken to its limit:

* **No text model.** The command is parsed into source/target phrases; CLIP
  embeddings for (command, source, target) are harvested *once per task* from
  YOLO-World's own text tower. The marginal cost of language conditioning is
  zero parameters and zero per-frame compute.
* **No visual encoder of our own.** The detector's SPPF feature map (via a
  forward hook + ROIAlign) supplies a 512-d frame embedding and per-role box
  embeddings. Perception standardizes every embedding at the boundary
  (canonical space, `microvla/utils/embedding.py`); nothing downstream ever
  sees a raw feature.
* **7.0 M trained parameters, enforced.** Fusion ≤ 5 M, drift encoder ≤ 1.5 M,
  planner ≤ 2.5 M, jointly < 9 M — asserted by `microvla.utils.param_audit`
  and by the test suite, so the budget is a property of the build, not a table
  in a paper.

The honest cost of that parsimony is the second half of this paper. A stack
this small has no capacity to paper over systems defects, so every seam
mattered: we document 28, classify them, and show that the last centimetres of
a pick — the part scale normally buys — reduce at this scale to a hand-eye
calibration constant that vision alone could not observe.

**Contributions.**

1. The smallest deployed language-conditioned VLA stack we are aware of
   (~30 M total, auditable ledger in §2), with a reuse pattern
   (detector-as-text-encoder) that removes the language model entirely —
   and unaided task success carried by **0.24 M task-content parameters**
   under a structured decoding head (§6.4e).
2. **Structured decoding as the capacity-floor result**: on a fixed trunk
   and corpus, free per-tick action regression scores 0.000 across six
   controlled variants; predicting *goals* for a task-content-free servo
   shell scores 0.700 — with a staged de-skeletonization path back toward
   fully-learned control whose first stage (learned close-trigger and hold
   gates) already swaps in at exact parity (§6.4e–g).
3. **The memorization audit and its repair** (§6.4f): fixed-placement
   benchmarks admit non-visual policies — shown at three layers (the
   benchmark's pinned poses, our head's image-insensitivity, the expert's
   position-baked constant) — and repaired with ten-episode placement
   variance plus one nuisance-input augmentation, verified on held-out and
   randomized protocols with an input-substitution probe we propose as
   standard practice.
4. A training-inference alignment principle — *dream evidence is faded
   evidence on the shared path* (§3) — and a world model beating
   persistence/linear-extrapolation baselines under deployment-exact
   rollouts (§6.1).
5. A defect taxonomy for producer/consumer ML seams (28+ worked
   instances): disagreements catchable by parity testing vs *agreements on
   a wrong convention* catchable only by provenance manifests (§7) — plus
   the forensic reduction of "success = 0" to a measured physical constant
   (§6.3–6.4), every intermediate hypothesis logged with its killing
   measurement.

---

## 2. Architecture

**The size claim, made auditable.** "Smallest in its class" is scoped
(§8, §9) to language-conditioned manipulation stacks with open-vocabulary
perception, and is enforced, not asserted: the test suite fails any build
whose trainable heads exceed the budget (`utils/param_audit.py`,
`tests/test_param_budget.py`). The deployed ledger:

| component | params | trained? |
|---|---|---|
| YOLO-World-S (vision + the ONLY text encoder) | 13.0 M | frozen |
| RecursiveTRM world model | 9.97 M | frozen at deployment |
| fusion + drift + relational + planner heads | 7.0 M | trained (≤ 9 M cap) |
| structured-decoding goal heads + gates | 0.24 M | trained |
| **deployed total** | **≈ 30 M** | |

For comparison, the smallest published VLA baselines we know of are
SmolVLA (~450 M) and TinyVLA's smallest backbone (~70 M *before* its
action head) — our entire stack is ~2× smaller than that backbone alone,
and the *task-content* parameters of the structured policy (the goal
heads that turned success from 0.000 to 0.700) are 0.24 M — under 1% of
the deployed total. No separate language model exists anywhere in the
stack: text rides the detector's own CLIP tower, harvested once per task.

```
text ──parse──> (source, target) phrases ──CLIP tower (frozen, once/task)──> task embs
frame (2 Hz) ──YOLO-World-S (frozen, 13M)──> frame_emb [512] + role boxes/embs/centers
                                │
                                v
        SlotResonanceFusion  [B,32,5]        (≤5M; evidence-weighted slots)
        AnchoredDriftEncoder [B,256]         (≤1.5M; anchored on first real frame)
                                │
                                v
        RecursiveTRM         [B,512]         (~9.5M; weight-tied recursion, FiLM
                                              drift conditioning, residual: out =
                                              current_emb + Δ; stateless)
                                │
                                v
        ChronoQueryPlanner   plan [5,7]      (≤2.5M; 5 sequential updates,
                                              tanh-bounded, 7 servos)
```

**The 30 Hz loop.** `JEPALoop` ticks at 30 Hz. Every 15th tick is *real*:
the frame is perceived, the drift encoder steps, and the TRM predicts the next
frame embedding. The other 14 are *dream* ticks: the corrected TRM prediction
is fed back through fusion's evidence path with held last-real boxes whose
confidence decays as `conf · staleness_decay^k`. A parameter-free
`InnovationCorrector` compares each real perception against the standing
prediction, corrects drift, and scales trust in the emitted plan.

**Trust is action-space aware.** For delta action spaces (LIBERO/Bridge),
where zero means "no motion", low trust *brakes* the plan toward zero
(holding a stale delta is momentum — measured as the drift-into-wall failure,
§4-series). For absolute PWM targets (the Pi rig), where zero is servo
mid-range, low trust *hold-blends* toward the last emitted plan and must never
scale toward zero. One corrector, two semantics, chosen by config — this
distinction was itself the product of an eval failure, not foresight.

**Parameter ledger** (audited, `microvla.utils.param_audit`):

| component | params | trained |
|---|---|---|
| YOLO-World-S + CLIP text tower | 13.0 M | frozen |
| RecursiveTRM (world model) | ~9.5 M | yes (stage A) |
| Fusion + drift + planner (heads) | 7.0 M (< 9 M cap) | yes |
| **deployed total** | **~30 M** | |

For calibration: OpenVLA is ~235× this deployed size; Octo-small (27 M) is the
only comparably sized generalist we know of, and it carries no open-vocabulary
detection grounding. "Smallest in its class" is claimed *as of this writing,
for language-conditioned manipulation stacks with open-vocabulary perception*,
and we would welcome a counterexample.

---

## 3. The alignment principle: dream evidence is faded evidence

The single design rule the project treats as load-bearing: **there is no dream
flag.** Fusion weights box and geometry tokens by `box_weight` — confidence ×
freshness in [0,1]. Real detections arrive at their confidence; held boxes on
dream ticks arrive decayed; missed detections arrive at 0; and train-time
modality dropout fades the *same* weights by a random factor. The network
cannot tell training from deployment because, on this axis, they are the same
distribution. The world model is likewise trained (stage A) on
deployment-exact schedules: 15-tick open-loop rollouts between real frames,
prediction fed back through the dream path, discounted losses over the
horizon. The §4-series documents what happened before this alignment existed —
open-loop metrics healthy, closed-loop dead — and §5d/5e the two rate/horizon
mismatches that survived even it.

---

## 4. Training within the budget

Data: LIBERO and BridgeData V2 episodes converted to compressed `.npz` shards
by a download→convert→delete pipeline hard-capped at 10 GB of disk *including
transient state* (`BudgetGuard`); training runs on a single 24 GB M-series
laptop (MPS), with pod bursts for closed-loop evaluation only. Stage A trains
fusion + drift + TRM on the spec loss (cosine + raw MSE on standardized
embeddings — no normalization inside the loss, which would forgive the scale
errors the feedback loop cannot survive). Stage B behavior-clones the planner
through the frozen world model: pose MSE + gripper BCE (the gripper is bimodal;
MSE averages it into "never quite close"), row-0 and pre-grasp step weighting,
recovery noise, and a variance term against magnitude collapse (§4p measured
LIBERO's tolerance band for action magnitude at roughly ±5%; MSE-BC lands 2–4×
below it).

---

## 5. Evaluation protocol

Three tiers, each with a stated purpose:

1. **Wind tunnel** (`eval/bench.py`): no simulator, mock everything,
   < 0.1 s/eval. Exists so that architecture changes get a same-day signal.
   Baselines live here permanently: `PersistenceTRM` (predict no change) and
   `LinearExtrapolationTRM`.
2. **Mock closed loop** (`MockLiberoEnv`): the full policy loop with
   deterministic mocks — every test in CI runs without sim, network, or cv2.
3. **Real closed loop**: LIBERO (MuJoCo), wrist camera, deployment-matched
   perception period, telemetry per tick (`eef`, detections, phase, trust —
   and `obj_pos`, sim ground truth, *logged for diagnosis and barred from the
   controller*).

Instrument calibration is part of the protocol, because the instruments were
wrong more often than the models: the score everyone read (`eef_obj_dist`)
measures distance to the object's *body origin*, and demonstration replays
put its grasp-commit value at 0.040 m (0.009–0.066), not 0 (§5p). Every
proximity claim below is stated against that reference.

---

## 6. Results

### 6.1 The world model works; the policy's last centimetres do not

Under deployment-exact rollout conditions the trained TRM beats persistence
and linear extrapolation on held-out episodes (§4q, `eval_results/bench*`),
and the v8 architecture recovered the action head from magnitude collapse
(§4o–4s). Open-loop agreement is high (gripper agreement 0.944 under teacher
forcing). Unaided closed-loop success on LIBERO object tasks is **0.000**
across every configuration ever run (n > 300 real evaluations, scorecard in
`results/`). The remainder of this section is the anatomy of that zero.

### 6.2 What the zero is not

Each of these was a live hypothesis; each has a killing measurement (log § in
parentheses):

* Not grounding: auditing detection duty per candidate view moved source
  grounding 0.219 → 0.850 and target 0.014 → 0.999 with zero training (§4n,
  §5-series); detection duty during eval is 0.96.
* Not the dream path or the corrector: exonerated by ablation (§5p).
* Not scene clutter: removing every distractor leaves success at 0 (§5q).
* Not action magnitude, rate, or sign conventions: measured, fixed, and
  re-measured (§4p, §4v, §4w).
* Not the last-centimetre *servo*: a 40-run sweep atlas over aim point,
  hysteresis, gains, and all 8 dihedral image-to-world mappings never moved
  image error below ~0.20 (§5p) — which was the clue.

### 6.3 The zero, located: a constant the controller cannot see

Telemetry forensics across the sweep atlas (§5r): at every grasp attempt, the
end-effector sits a **constant** world-frame offset from the object —
pooled mean (−0.079, +0.040, +0.023) m; at first gate crossing, over 231
episodes, mean (−0.080, +0.050) m with std (0.023, 0.016) — **invariant to
the aim point**. Moving the servo's aim by ±0.10 in either image axis, which
should displace the converged gripper by centimetres, changes nothing, because
the grasp gate fires on a height crossing while image error still hovers at
0.20+. Meanwhile closes trigger at z ≈ 0.045–0.050 m over an object whose
centre is at 0.009 m: the fingers pinch air 4 cm up and 8 cm over, every
time, in every arm of every sweep. The entire aim-UV parameter family — weeks
of sweeps — had no control authority over the scored quantity.

This is a camera-to-gripper lever arm: classical hand-eye calibration, absent.
Nothing in a 7 M-parameter stack (or, we suspect, a 7 B one) can learn it from
behavior cloning against a wrist camera that never observes its own extrinsics.

### 6.4 The calibrated handoff: vision gates, proprioception finishes

The fix (`eval/ibvs_phase.py`, flags `--ibvs-grasp-offset/--ibvs-close-z/
--ibvs-press/--ibvs-retry-rise`): at the visual gate, fix a world target =
eef + (+0.080, −0.050); P-servo on proprioception alone to within 1.5 cm;
descend to z = 0.01 with a z-stall contact check; close with a downward
press; on an air close (jaw feedback), rise and re-approach the *stored*
target shifted along the high-variance axis (dx ∈ {0, ±2, ±4, +6} cm) — the
detector is unreliable at table height, so retries never re-consult vision.
The calibration constant is derived offline from logged episodes; the runtime
controller reads detections and proprio only.

Single-attempt runs (`handeye_v1`, n=5) moved closest-approach from the prior
all-time best of 0.068 m to 0.019–0.042 m; probe-retry runs (`handeye_v2`)
reached **0.004 m** — and still no grasp registered. That impossibility was
the finding.

### 6.4b Defect 29: the one bit of feedback was a constant

A close at 4 mm from the object's centre, at its exact height, cannot read
"air" on any closing axis of an 8.1 × 4.3 cm box between ±4 cm fingers. The
held-object check was `mean(gripper_qpos) ≥ 0.2` — and the panda's finger
joints are *mirrored*, `(+q, −q)`: the signed mean is identically zero in
every jaw state. Verified against demo proprio: open (+0.906, −0.906) →
mean 0.0001; *holding the box* (+0.552, −0.689) → mean −0.069; unsigned mean
0.91 / 0.62 — cleanly separable. One `abs()` (in both grasp state machines)
was the entire fix.

Every grasp the project ever physically completed had been discarded by this
line, one tick before "lift". It is the taxonomy's 29th defect and its purest
class-2 instance: self-consistent, mock-consistent (the test mocks encoded the
same same-sign misunderstanding), invisible to parity — and felled by one
logged measurement at a close the geometry proved must succeed.

### 6.4c First held grasp; the place leg; end-to-end runs

With the check fixed (`handeye_v3`): close at dxy = (−0.025, −0.010) →
jaws 0.53 (**held**) → lift carries the object from z 0.009 to 0.459 — the
first grasp-hold-lift in the project's history, twice in two trials. The
place leg then failed exactly the way the pick had: the wrist camera almost
never sees the basket at altitude (target duty ~0 over the traverse), so the
visual release gate waited on a missing signal and tripped on noise. The
symmetric fix: the basket sits at a *fixed* world position across all 50
demonstrations ((−0.005, +0.257), std < 2.5 cm), so the place leg is a
proprio-only traverse to a calibrated point, a lowered drop, and a jaw-drop
watchdog. Vision's remaining role in the pipeline is the one thing it is
measurably good at: finding the object once, from altitude.

End-to-end results, multi-task (all constants from demo statistics or logged
telemetry; nothing tuned on evaluation rollouts):

| task | object regime | mean success | n | project prior |
|---|---|---|---|---|
| cream cheese → basket | thin flat box, close at 1 cm | **0.300** | 10 | 0.000 |
| alphabet soup → basket | standing can, fly-over + close at 4.5 cm | **0.750** | 8 | 0.000 |
| salad dressing → basket | tall bottle | 0.000 | 8 | 0.000 |

The first cream run (n=10, max 400 steps) scored 0.200 with a clean failure
decomposition: three wrong-object bindings at the visual gate, three
probe-exhausted misses (calibration variance beyond the ±6 cm span), two
timeouts while executing correctly — one ended *holding the object* 1.4 cm
from its centre at the step cap. Adding a gate-time CLIP verification (veto
the gate when the bound box matches the target phrase better than the
source, or when a better source proposal exists elsewhere) raised cream to
0.300 by converting wrong-bind grasps into safe no-commits. Soup succeeded
at 0.750 on its *first attempt* with constants transferred purely from demo
statistics — grasps land 6–7 mm from the can axis — establishing that the
hand-eye lever arm is object-independent and only object-geometry heights
change. The dressing failure was traced through three layered defects
(a consumer confidence-floor above the detector's signal; the detector
binding the robot's OWN FINGER as the bottle at 100.0% duty — a fix that
cannot leave a moving camera's frame is self-attached; and the
height-dependence of that finger's image position defeating positional
masks) and remains open: its residual is a perception-layer role-binding
problem, which returns the investigation to the paper's central seam.
Per-trial tables and every intermediate run: log §5r–§5t.

Framing, binding: §6.4 is an *assisted, diagnostic* result family. It
measures whether frozen-detector geometry plus offline-calibrated constants
plus trivial control suffices for the full task. It is not unaided policy
competence and is never aggregated with policy numbers. Its value to the
architecture claim is indirect but real: it certifies the perception stack
and exposes the exact constants a learned policy would have to encode.

### 6.4d Distilling the machine back into the policy: the unaided track

The assisted stack is also a *teacher* on the true eval distribution. We
recorded its successful rollouts (init states disjoint from all eval
trials), converted them through the standard shard pipeline, and
behavior-cloned stage B from them. Four rounds, each diagnosing the next
(soup task, wrist camera, NO assist flags at eval — vision → JEPA loop →
planner → actions):

| round | corpus | val BC | min eef→obj (m) | final (m) | grip fires | success (n) |
|---|---|---|---|---|---|---|
| BC-23 | 23 teacher successes | 0.045 | ~0.24 (hover) | — | no | 0/3 |
| BC-100 | 100 teacher successes | 0.045 | 0.155–0.198 | 0.45–1.17 | 2–26% | 0/10 |
| DAgger-only | 40 student-driven eps, teacher labels (β=0.3) | 0.043 | **0.061** | **0.110** | 0.0% | 0/10 |
| Aggregate | 100 + 40 + magnitude losses | — | 0.078–0.146 | 0.14–0.28 | ~1% | 0/7* |
| Grasp-weighted | aggregate + close-window upweighting | — | ~0.15 | — | 12–42% | 0/10 |
| LoRA input | + trainable SPPF subspace (r=8) | — | 0.127 | holds | 10–18% | 0/6† |
| Phase objective | + phase-progress loss, BC demoted 0.2× | — | — | — | — | 0/10 |

*Interrupted by a host restart at 7 of 10 trials; all completed trials
fail identically. †User-stopped after 6 trials; approach best-on-film
(object at the jaws), no close commitment.

Each zero is *located*, not mysterious. BC-100's stall was quantified
against live telemetry: the student's lateral commands average |xy| =
0.025 raw units vs the teacher's 0.094–0.207 — a 4–8× undershoot whose
mechanism is mean-collapse under partial observability (the teacher's
final approach steers to a stored internal target invisible in the
per-tick observation, precisely where the detector is unreliable). The
DAgger round fixed the approach — the policy closes to 6.1 cm and *stays*
(final 0.11 m vs drifting to a metre) — while cleanly ablating the two
skills: trained on the DAgger corpus alone (whose labels never reach the
grasp phase and are therefore ~always gripper-open), the policy unlearned
closing entirely, a failure predicted in writing at episode 12 of 40.
Approach lives in the on-student-distribution corrections; grasp lives in
the teacher's completed episodes; the aggregate round trains on both.

The aggregate round then sharpened the residual to a single mechanism:
approach and station-keeping are retained (the policy reaches 8–15 cm and
stays), but the grasp *trigger* — closing at the right instant — fires on
~1% of ticks. The close event occupies ~5% of corpus ticks, so uniform BC
underweights precisely the decision that completes the task: a class-
imbalance problem on an event, not a geometry problem. The unaided ladder
after seven rounds reads: descend ✓, lateral magnitude ✓, stay-on-target ✓,
close-at-the-right-instant ✗ — invariant under capacity, data aggregation,
DAgger, input adaptation (LoRA), and objective redesign. We take the
invariance itself as the finding: the residual is not a training deficiency
but a *policy-class* deficiency, and it motivates §6.4e.

### 6.4e Structured decoding: the mechanisms become architecture, and the zero breaks

The teacher's measured mechanisms (§6.4–6.4c; the seven-item analysis in
the supplement) prescribe a different decoding head, not a better
regression. In the v10 structured policy the network stops emitting
per-tick actions entirely. Two small heads learn the *task content*: a
grasp-point head (0.17 M) regresses the world grasp point from (source box
uv/confidence/embedding, frame embedding, proprio) — supervised by the eef
position at each teacher episode's final close onset, a label that contains
the hand-eye lever arm by construction — and a place head (0.07 M) reads
the basket point off the command embedding. A parameter-free servo shell
(`GoalServoMachine`) supplies the *control*: sigma-gated goal latching,
a P-law `clip(12·(goal − eef), 0.6)`, one-way phases with a debounced
retry cycle, the abs() jaw hold check, a radius-ordered 2D probe search,
and a proprio-only place leg. Trust gates evidence admission, never action
magnitude — parking is impossible in this policy class.

Trained offline in minutes on the existing corpus (111 episodes, 1 703
supervised ticks, no re-recording): val median grasp error **1.27 cm**
(p90 2.70) uniform across altitude bands; place error **0.85 cm**, the
head's mean prediction recovering the hand-calibrated basket constant to
~6 mm. First closed-loop eval (task 0, n=10, no assist flags):
**mean_success 0.10 — the first unaided success of the project** — with
every failure isolating via phase telemetry to two named, structural
defects (hover-altitude latching; an x-only probe against isotropic error),
both fixed without retraining. The free-regression arm (§6.4d) becomes the
ablation: same trunk, same corpus, same eval — the decoding structure is
the difference in kind.

### 6.4f The generalization audit: catching every layer's memorization

Structured decoding broke the unaided zero and climbed to 0.700 on the
benchmark protocol (§6.4e ladder: 0.100 → 0.300 → 0.400 → 0.700, each
step a named, structural fix diagnosed from phase telemetry). We then
audited that number the way a skeptical reviewer would, and report the
audit as a result in its own right — because every layer of the system,
including the hand-built teacher, turned out to encode the benchmark's
hidden constant somewhere.

**The benchmark pins placements.** In LIBERO-Object, the target object's
start pose is identical across all 50 canned init states AND under fresh
seeded resets (measured: the task-0 soup can sits at exactly
(−0.120, −0.240) in every episode ever scored, here and definitionally in
prior work on this suite). The 15 init dims that vary are arm and
distractor state.

**Three memorizations, three probes.** (1) The grasp head: an input-
sensitivity probe shows its prediction flat (~1 mm) under image-position
sweeps while tracking the end-effector (slope ≈0.87 toward the fixed
target) — with a constant label, the teacher's own converging approach
makes proprioception a better predictor than vision, so regression
rationally learns location, not looking. (2) Model selection: the same
policy scores 0.700 on the ten init states its versions were iterated
against and 0.300 on ten never-tuned ones — the dev/held-out gap of the
machine's own knobs. (3) The teacher: under ±6 cm placement teleports its
visual approach still reaches 5–7 cm of the object (detection ≈1.0), but
its calibrated composite offset — a −18.6 cm y-term that encodes approach
geometry, not hand-eye physics — misses by the shift, and its
x-distributed probe cannot recover an isotropic error. Vision, in every
layer of this system's history, gated the approach; a memorized constant
finished the job.

Zero-shot transfer quantifies the consequence: 0.000 across all ten
suite tasks for the soup-trained heads.

**The repair, and its verification.** Ten teacher episodes recorded under
±4 cm source teleportation (the assisted teacher, given a 2D probe
schedule, completes shifted picks at ~30%) replace the fixed-placement
corpus; one augmentation — training-time jitter on the end-effector
*feature* while the reconstruction anchor stays exact — removes the
head's dependence on a nuisance input whose off-manifold behavior had
silently controlled deployment (isolated by a one-variable ablation:
0.000 → 0.700 dev with jitter as the only change). The retrained head,
which never saw a development init state:

| protocol | memorized head | variance-trained head |
|---|---|---|
| development (benchmark placement) | 0.700 | 0.700 |
| held-out init states | 0.300 | **0.700** |
| randomized placements (±4 cm) | — | **0.500** |

Every success across the table is video-recorded. Ten episodes of label
variance and one nuisance-input augmentation achieved what six
free-regression rounds and an 111-episode fixed-placement corpus could
not. We propose the substitution probe + the randomized protocol as
standard practice for fixed-placement benchmarks: a policy's score and
its placement-sensitivity are separate claims, and only the pair supports
"visual."

### 6.4g Consolidation, negative results, and the de-skeletonization ledger

Scaling the variance corpus (10 → 27 → 49 episodes) holds the result and
sharpens the attribution: the 49-episode head's substitution profile is
anchor 0.3 cm / proprio-feature 0.1 cm / vision channels 1.1–1.8 cm — the
prediction is near-absolute and visual, and even the detector's box
center (inert in every earlier head) becomes informative. Pooled across
the three variance-trained heads: fixed-placement 24/40, randomized ±4 cm
12/30, protocol-symmetric. Negative results, reported: joint LoRA
adaptation of the embedding stage *hurts* at this corpus scale (3.08 vs
0.99 cm val), and a machine-gate parameter sweep is flat (the shell is
not knife-edge tuned — its performance is head-accuracy-bound).

**De-skeletonization, stage 1.** Self-play episodes recorded with the
machine's internal state dumped per tick supervise two sub-3K-parameter
classifiers that replace the close trigger (88% accuracy, 89% fire
recall) and the hold check (76%, 94% recall); swapped into the machine,
the held-out protocol scores **0.700 — exact parity with the
thresholds**. The scaffold's decisions are learnable from its own traces
at zero measured cost; the remaining hand-set surface (latch stability,
P-law gains, probe schedule, place descent) follows the same recipe, each
with its own ablation row. Object-level generalization remains open —
zero-shot transfer to the suite's other nine objects is 0.00, awaiting
per-object variance corpora (the expert's per-object calibration is the
current bottleneck, itself §6.4f evidence that calibrated constants do
not transfer).

### 6.5 Efficiency

Full test suite (537 tests, CPU, mocks): ~15 s. Wind-tunnel eval: < 0.1 s.
Training fits a laptop; the 10 GB disk cap held through three dataset
conversions. Per-tick deployment compute is dominated by the 2 Hz detector;
the 30 Hz heads are CPU-viable — the design target for the Pi 5.

---

## 7. The defect taxonomy (why the methodology is a contribution)

28 defects, each logged with discovery measurement and the cheaper measurement
that would have caught it earlier (§4-series, §5n, §6 of the log). The
taxonomy's boundary is itself the finding:

* **24 disagreements** between the producer (training/bake) and consumer
  (deployment) of a feature corpus — units, rates, defaults, index and camera
  conventions. Catchable by **parity testing**: run both paths on one input,
  diff tensors (`eval/train_vs_deploy.py`). Limits stated: parity forced real
  ticks every step, which is exactly why the deployment-only corrector defect
  was invisible to it.
* **4 agreements on a wrong convention** — both sides consistent, both wrong
  (the corpus baked from a camera in which the detector is blind: target role
  grounded on 1.4% of 38,000 frames). No parity test can catch consistency;
  only **provenance** can: `manifest.json` records camera, orientation,
  thresholds, rates, and frame geometry, and eval refuses silently mismatched
  corpora.

We believe this taxonomy, not the architecture, is the transferable half of
the paper: every frozen-encoder stack has this seam, most have no instrument
pointed at it, and "agreement is not correctness" is the failure mode that
scale actively hides.

---

## 8. Related work

RT-2 and OpenVLA established VLM-to-action fine-tuning at 55 B / 7 B; Octo
(27–93 M) is the closest size point and, like us, trains a policy head over a
compact backbone, but carries its own vision encoder and no open-vocabulary
grounding. TinyVLA spans ~0.07–1.4 B (compact VLM backbone + diffusion
action head; TinyVLA-H at ~1.3 B beats OpenVLA with 5.5× fewer parameters);
SmolVLA sits at ~450 M. MicroVLA's deployed footprint is ~45× smaller than
TinyVLA-H, and even TinyVLA's smallest backbone alone is ~2× our entire
stack (~10× our trainable parameters).

For calibration on absolute performance: sub-1B successors of TinyVLA
report ~87% on LIBERO-Object (CoTinyVLA) and ~90% suite averages (XS-VLA,
<0.5 B); SmolVLA-0.25B is reported around ~83% suite average — all
fine-tuning end-to-end on large demonstration corpora with pretrained
backbones. We do not compete on that axis and do
not claim to: our regime is 50 demos per task, frozen perception, no
pretraining, and a deployment budget an order of magnitude below the
smallest entry above. The comparison this paper supports is
capability-per-parameter-per-demo under a hard edge-hardware budget, plus
the transferable diagnosis methodology; the absolute-success race belongs
to a different resource class. Visual servoing and hand-eye calibration are classical (decades of
literature we re-derived one telemetry table at a time — §6.3 is, knowingly,
a rediscovery with modern instrumentation). JEPA-style predictive world models
motivate the dream loop; our contribution there is the *shared-path fade*
alignment, not the predictive objective.

The structured decoding of §6.4e belongs to the learned-goal →
engineered-primitive lineage — MOKA's keypoint affordances executed by
motion primitives, VoxPoser's LLM-composed value maps executed by a
planner, point-prediction interfaces (RoboPoint, PIVOT) parameterizing
fixed skills, and classical grasp-synthesis pipelines. Our contribution to
that lineage is placing it *inside* a 30 M-parameter VLA and measuring,
on a fixed trunk and corpus, that at this scale the decoding structure is
the difference between zero and task success. π0.5's hierarchical
inference validates the same decompose-what-from-how thesis with both
levels learned at frontier scale; we read the two results as ends of one
curve — as capacity shrinks, the motor level must become structure,
because six controlled attempts at learning it (capacity, aggregation,
DAgger, input adaptation, objective redesign) each reproduced the same
zero. The de-skeletonization program (scaffold-state supervision, staged
replacement of gates then latch then control law) is our path back toward
the learned end of that curve.

## 9. What we claim, and what we do not

Claimed: unaided task success for the structured policy (§6.4e) — no
teacher, no assist flags, no simulator state, no hand-given task constants
at eval — at 0.700 on the development protocol and 0.300 on never-tuned
init states (task 0, n=10 each), with the purest zero-calibrated-constant
configuration at 0.400 (dev). Every claimed success is video-recorded.

Not claimed, stated plainly:

* End-to-end learned motor control. The free-regression policy's success
  is zero across six controlled variants and is reported as such; the
  structured policy's control shell is engineered (task-content-free), and
  §6.4f documents that its learned goal heads memorized the benchmark's
  fixed placement — the randomized-corpus repair is reported with its own
  before/after protocol.
* Dev-protocol numbers as generalization. The dev/held-out gap (0.700 vs
  0.300) is itself a reported result; held-out and randomized-placement
  protocols are the citable columns.
* The assisted-teacher (§6.4) numbers as policy competence — they are the
  measured skill decomposition and the data engine, nothing more.
* "Smallest in its class" is scoped to language-conditioned manipulation
  stacks with open-vocabulary perception, as of this writing.
* Sim-only: no physical-robot result is claimed; the Pi 5 rig is the design
  target, not a reported deployment.
* Single-task-family evidence for §6.3–6.4 (cream cheese; the offset constant
  is shown for one object geometry and one approach family).

## 10. Reproducibility

Single repo; `DESIGN.md` is a binding interface contract; every dimension
flows from one config object. `pytest tests -q` (CPU, no network) covers the
full deployment path against mocks; `python -m eval.bench --synthetic 30`
reproduces the wind tunnel; parameter and disk budgets are asserted, not
promised. The full experimental log — including every negative result and
every retraction — ships as `paper.md`.

---

*Corresponding log: `paper.md` §4a–§5r. Scorecards: `results/`. Telemetry:
`eval_results/`.*
