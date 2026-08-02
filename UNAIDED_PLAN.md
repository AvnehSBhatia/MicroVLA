# UNAIDED_PLAN — from assisted 0.30/0.75 to learned-policy success

Companion to `handoff.md`. Success criterion: `mean_success > 0` with **no**
`--ibvs-phase`, `--tool-phase`, grasp-offset, place-at, press/retry, or
clear-distractors. Unaided = `rec_fix`-class stack + binding knobs only
(`det-conf`, role-disjoint, etc.). Vision → JEPA loop → planner → actions.

## Map: aid → unaided fix

1. **Phased IBVS owns the action** → distill the winning handeye stack as a
   *teacher*: stage-B BC (or DAgger) on `(obs, proprio) → teacher action`
   over pick + place segments. Freeze perception; train planner (+ waypoint
   head). Metric: same seeds as handeye_v5, flags off; phase completion
   (grip hold, lift Δz, place release).
2. **`--ibvs-grasp-offset` (lever arm)** → learn "close here in
   proprio/world": waypoint/eef-target head supervised on teacher/demo grasp
   XY at close (NOT UV-only centering — `center_frame` rewrote BC and hurt).
   Optional `f(source_uv, proprio) → Δxy` MLP matching the calibrated
   offset, baked into the planner. Metric: eef−object XY at close vs demos.
3. **`--ibvs-close-z` / press (depth)** → grasp-window-only z-at-close loss
   + magnitude/variance guard; match demo z and post-close jaw width (with
   the `abs(qpos)` check — that is a bugfix, keep in deploy). Metric:
   proprio z at close vs demo (0.009–0.015); held-object rate.
4. **`--ibvs-place-at` + drop-z** → short path: distill the place leg from
   the teacher. Real path: agentview/dual-cam at place time (basket rarely
   wrist-visible at altitude). NOT unaided while any hardcoded world
   constant remains.
5. **Probe retries** → better first-shot offset+depth so retries aren't
   needed; optionally distill teacher retry segments. Metric: first-attempt
   close success.
6. **CLIP gate-verify / binding** → keep role-disjoint/area priors as deploy
   config (protocol, not a controller). The finger-self-bind class (cream
   "bad binds", dressing 0) needs a PERCEPTION fix — leading candidate: a
   temporal self-attachment filter (a fix whose image position is invariant
   while the arm moves is the robot's own hand; positional masks measured
   NEGATIVE, §5t).
7. **Jaw-hold `abs(qpos)` (defect 29)** → bugfix, stays everywhere.

## Sequence

| Phase | Work | Exit criterion | status 2026-08-01 |
|---|---|---|---|
| A. Instrument | UV err, eef XY, z, jaw at close: rec_plain vs demo vs teacher | gap table | **DONE** (§5r–§5t): demo closes at 0.040 m / z 0.009–0.015; teacher closes at ±4–7 mm, obj z exactly; rec_plain closed 8 cm off at z 0.045–0.050 |
| B. Teacher dataset | Roll winning handeye on train inits; store frames+proprio+boxes+actions via shard pipeline (BudgetGuard) | N episodes cream (+ soup) | **DONE** (`preprocess/teacher_rollouts.py`): 23 eps round 1, 100 soup eps round 2 (inits 50+, ~90% teacher hit rate), raw purged per budget rule |
| C. Distill stage B | From `rec_fix`, BC/DAgger on teacher actions; grasp+place weighted; freeze fusion/TRM first | val mimics teacher; **unaided cream succ > 0, n≥10** | round 1 (23 eps): val bc 0.0448, unaided hover 0/3. round 2 (100 eps): val bc 0.0445 grip 0.998; unaided approaches 0.29→0.16 m, grips fire, stalls 16 cm short (0/5, n=10 rerun in flight). **Defect quantified: student xy 4–8× under teacher scale** (mean-collapse under the teacher's hidden `_base_tgt` + frozen-zero `gain_head`). diag_gain3 (3× gain, diagnostic-only) decides magnitude-calibration vs DAgger |
| D. Internalize offset+z | eef-target / z-at-close aux losses so distillation isn't the only geometry carrier | ablate teacher, keep aux; succ ≠ 0 | next, informed by diag_gain3 |
| E. Place without `place_at` | distill place; then agentview target binding | unaided place, no constants | — |
| F. Multi-task | repeat C–E for soup (teacher at 0.75), dressing (perception fix first) | honest per-task unaided table | — |

Also first, cheapest training win (forensics F-003): the HRM `gain_head` is
still exactly its zero init — its gradient path never fired. Fix before C so
the distilled policy can express demo-scale magnitudes (§4p shrink).

## What not to do

- More eval P-controller gain/descend sweeps on the naked policy.
- Centering-in-frame loss rewriting whole-episode BC.
- Calling distilled/waypoint-aided runs "unaided" if any handeye flag remains.
- Big agentview bakes before A–C without clearing disk.

## Definition of done (paper-honest)

Unaided `mean_success` table (cream/soup/dressing, flags listed off) +
same-ckpt with-handeye ceiling ablation + the sentence: "the assisted stack
shows frozen geometry + calibration suffice; the unaided numbers are the
learned policy after distilling / supervising that geometry."
