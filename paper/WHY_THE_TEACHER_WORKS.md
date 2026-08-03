# Why the successful model works — mechanism analysis

**The successful model** (reference config, re-verified 2026-08-03 at
mean_success **1.0, n=4**; campaign numbers soup **0.75** n=8, cream
**0.30** n=10): `rec_fix` checkpoint for perception/binding + calibrated
`PhasedIBVS` (`eval/ibvs_phase.py`) owning the actions. Flags in
`handoff.md` §3. Everything below is measured, with sources.

## The seven mechanisms, ranked by how much they explain

### 1. Vision is consulted exactly once — from where vision actually works
The wrist detector is reliable from altitude (detect duty 0.9+, conf well
above floor) and unreliable at table height (duty collapse measured in
§5r; the dressing arc showed it binds the robot's own finger down there).
The machine exploits this asymmetry: ONE visual fix from altitude stores
`_base_tgt`; every later step — descent, probe retries, grasp — is
proprio-only. **The learned policy has no such privilege: it re-reads
vision every tick, including at heights where vision is noise.** The
lora1 film shows the consequence: the can is IN FRAME at the jaws and the
policy still cannot use the signal the way the stored target does.

### 2. The hand-eye lever arm is treated as what it is — a rigid-body constant
The camera sits ~8.9 cm from the TCP. That offset is a property of the
robot, not the scene: constant across 231 at-gate episodes, all objects,
all inits (band050 atlas, §5r). The machine adds (+0.080, −0.050) m once.
No amount of aim-sweeping could remove it (measured: invariant to every
`target-uv` setting) and no BC round has internalized it (rounds 2–5).
One number, calibrated offline, beats 16.6M trained parameters at this
specific job because the quantity is *literally constant*.

### 3. Closed-loop P-control in ONE coordinate frame
`action = gain × (target − eef)` — error and command live in the same
metric space, so magnitude scales with distance automatically: far → fast,
near → slow, at target → stop. This kills three failure modes the learned
policy exhibits at once: mean-collapse (no ambiguity to average over),
magnitude miscalibration (§4p shrink; the 4–8× xy undershoot), and
parking (zero command only AT the target, never beside it).

### 4. Commitment is state, not per-tick inference
Phases are one-way: servo → align → grasp → lift → transport → release.
Once the gate fires, descent happens NO MATTER what the next 300 frames
look like. The BC policy re-decides every tick from a noisy frame — and a
policy that re-decides under noise regresses to "hover" (the fixed-point
signature of rounds 1–2, and the reason grip fires at 10–40% duty without
ever COMMITTING at depth in rounds 5b/lora1). Hysteresis (`descend-hyst
0.50` as a band) is the same idea at the gate: cross once, stay crossed.

### 5. The one bit of feedback that matters is read correctly
"Am I holding the object?" gates retry-vs-lift. Defect 29 (signed mean of
mirrored finger joints ≡ 0) had every good grasp discarded one tick
before lift — with it fixed (`abs()`), the machine converts grasp
competence into task success at ~100%. The learned policy has no explicit
hold check at all; it must infer it, and nothing in BC supervises that
inference.

### 6. Failure is handled by SEARCH, not hope
Probe retries shift the stored target ±6 cm in a deterministic pattern
(vision-free — retries never re-consult the unreliable detector). This
converts the calibration variance tail into extra attempts instead of
misses: cream's probe-exhausted failures are exactly the >±6 cm tail. BC
has no retry concept; DAgger approximates it weakly by covering
off-trajectory states.

### 7. The place leg uses the ONLY invariant vision could never see
The basket never moves: (−0.005…−0.010, +0.255…0.260) across all 50
demos/task, std < 2.5 cm — and it is almost never wrist-visible at
altitude (target duty ~0 in transit, §6.4c). Proprio-only transport to a
constant beats a vision-based place that would be blind precisely when it
matters.

## The unifying principle

Every mechanism is the same move: **replace a per-tick inference under
partial observability with either (a) a constant that is actually
constant, or (b) a feedback loop in a fully observed variable (proprio).**
Vision is used only for the one quantity that is neither constant nor
proprioceptively observable — which object, and roughly where — and only
from the regime where vision is trustworthy.

## What this prescribes for the learned policy (why the current bets are
the right ones)

| teacher mechanism | learned-policy analogue | status |
|---|---|---|
| one-shot target + proprio servo | internal goal representation stable across ticks (TRM/JEPA latent should carry `_base_tgt`) | missing — the drift/park failures ARE this gap |
| phase commitment | phase-progress objective (grip timing windows, direction-to-goal) | teacher_phase1 in flight |
| P-control magnitude | magnitude floor (moving decisively is free) | in phase loss |
| lever-arm constant | must be internalized as a learned offset (UNAIDED_PLAN §D: `f(uv, proprio) → Δxy` aux) | next after phase1 |
| hold check | grasp-window jaw supervision (`abs(qpos)`) | partially in grip losses |
| vision only from altitude | LoRA'd embedding could learn height-conditional trust; or gate perception_period by z | open idea |

The assisted 0.75/1.0 is not a workaround to be embarrassed about — it is
the measured proof of WHICH capabilities the end-to-end policy must
acquire, one mechanism at a time.
