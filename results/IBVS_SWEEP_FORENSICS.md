# IBVS sweep forensics — why every config scored 0.0, and why that is NOT the ceiling result

Source: per-tick telemetry of the 2026-07-29/30 pod IBVS sweep (10 completed
configs × 3 tasks × 5 trials, `full_stageB_rec_mid.pt`, gains 0.05–0.5, sign
variants, `--ibvs-descend`, `--no-brake`). All 10 runs: `mean_success 0.0`,
3/3 tasks completed, **0 failed workers** (the torch-2.6 `weights_only` fix
held throughout). Analyzed runs: `1785390631119/34279/37292` (mid-sweep) and
`1785391192645/95651/98705`, `1785391423445/26485/29571` (final rounds) —
45 trials, fields: `src_conf`, `src_center`, `eef`, `ibvs_cmd`, `action`.

## The structural fact first

`libero_object` success = object **in the basket**. The IBVS falsifier
(7f3da5a) does source-centering + descend + grasp — it has **no lift-and-place
phase**. `mean_success` was therefore structurally unreachable for this
controller, and the sweep's zeros carry **no evidence for the frozen-feature
ceiling hypothesis (5j)**. The falsifier's real readout is the approach
metrics below — and they point the other way.

## Per-stage forensics (45 trials)

| stage | measured | verdict |
|---|---|---|
| (a) detection | `src_conf > 0.005` on 6–46% of real ticks (run means 12–31%); **median conf 0.000** everywhere | SPARSE — intermittent fixes, not tracking |
| (b) IBVS engagement | fires on essentially every detected tick (13–93 cmds/trial) | works, gated on (a) |
| (c) approach | image error `src_center`→frame-center falls 0.55–0.64 → **0.03–0.07** (best-tick) in most trials; EEF path 45–210 cm | **WORKS** — the P-controller centers the target |
| (d) descend | EEF z: +0.24..0.28 → **+0.01..0.05** (table/grasp height) in ~80% of trials | works |
| (e) grasp | sustained (≥10-tick) gripper closes at z ≈ +0.01–0.05 in 34/45 trials; **20–50 cm rise after close** in most | closes and lifts; telemetry has no object pose, so grasped-vs-empty is UNMEASURED |
| (f) place | no basket-directed phase exists in the controller | **cannot happen by construction** |

## What this licenses

1. **5j's ceiling claim loses ground.** Frozen YOLO-World features, read by a
   zero-training P-controller, localize the source well enough to center it,
   descend to it, and close on it. The information is there; binary success
   was the wrong instrument to detect it (exactly the "residual approach
   metrics" argument in the paper front matter).
2. **Detection sparsity is the live bottleneck** for any detector-guided
   residual: median per-tick confidence is zero and the controller runs on
   6–46% duty cycle. Note: **every analyzed run predates 350eef9** (the
   sightedness fix — `det_conf=0.02`, `render_size=256` landed after the
   sweep finished). The night eval is the first run under the honest
   protocol; its `src_detect_rate` supersedes row (a).
3. **The missing measurement** is grasped-vs-empty at stage (e): either log
   object pose in telemetry, add a lift-success criterion (object z rise), or
   record one MP4 per config (`eval/record_mp4.py`). If lifts carry the
   object, the residual is a grasp engine and the gap is pure place-phase
   sequencing.

## Postscript — night eval (post-350eef9) numbers landed 2026-07-30 06:55 UTC

`full_stageB_rec_fix.pt`, honest protocol (`det_conf=0.02`, `render_size=256`),
3 tasks × 3 trials (`eval_results/night_sighted/`): `mean_success` **0.000**,
`src_detect_rate` **0.719** (supersedes row (a): sightedness fix confirmed —
detection went from 6–46% duty / median conf 0 to 72%), `src_conf_mean` 0.061,
`grip_close_rate` 0.653, `eef_obj_dist_min` **0.132 m**, `eef_obj_dist_at_20`
0.257 m, `eef_obj_dist_final` **0.750 m**. The trained policy approaches to
~13 cm — never grasp range — then diverges to 75 cm by episode end. Note the
contrast with stage (c)–(e) above: the zero-training IBVS residual got closer
(grasp height, sustained closes) than the trained policy does. Detection is no
longer the bottleneck; terminal approach precision and late-episode divergence
are.

## Postscript 2 — video ground truth (2026-07-30 08:05 UTC): it's the phase shortcut, not an approach floor

Two MP4s of the exact scored config (`rec_fix` + ibvs 0.5, honest protocol;
`eval_results/videos_rec_fix_ibvs/` on the pod, recorded via the new
`record_mp4` deployment-knob passthrough, 6342a67) show the same behavior:
**the policy drives to the basket and parks there, empty-handed, never
attempting a grasp.** Task 2 ends with the wrist camera buried in the basket
liner; task 0 sweeps past the objects early, then overshoots left of the
basket onto bare table.

This reframes both earlier postscripts:

* `eef_obj_dist_min` ≈ 0.10–0.13 is a **drive-by** en route to the place
  phase, not an approach that stalls — there is no "terminal approach
  precision" problem, there is **no grasp phase at all** (the 4h/5k phase
  shortcut, executed in the sim).
* The IBVS residual cannot rescue this and gain is irrelevant: once the arm
  is over the basket the source is not in the wrist view, so there are no
  detections and the residual is **silent by construction**. A residual
  defined on the wrist image only has authority where the policy's prior
  already points the camera.
* Follow-on eval runs (`rec_fix_ibvs`, gain 0.2 ratio-to-policy 0.22;
  `rec_fix_ibvs05`, gain 0.5) confirmed: intermediates statistically
  identical to bare `rec_fix`. High `src_detect_rate` on some trials
  (0.88–0.98 with the wrist in the basket) is low-conf false positives at
  `det_conf=0.02` (conf mean ~0.06), so detect-rate intermediates need a
  confidence-weighted variant before being trusted.

Implication for the lever ranking: the cheapest path to a nonzero
`mean_success` is **phase sequencing**, not more controller authority — e.g.
IBVS-first scheduling (servo+grasp until holding, only then hand over to the
policy's place prior), or gating place-phase motion on a grasped condition.
The falsifier's approach evidence (sweep stages c–e above) says the features
support the servo half of that plan.

## Method (reproducible)

Group telemetry rows by (run, task, trial); per trial compute: detection rate
over real ticks at conf > 0.005, per-trial `ibvs_cmd` count, net/path EEF
displacement, z-min, best image-space distance of `src_center` to frame
center, sustained-close segments (≥10 consecutive ticks with `action[6] > 0`)
and max z-rise within 60 ticks after each. One-file script; ask any session to
regenerate from `eval_results/libero_object_real_*_telemetry.jsonl`.
