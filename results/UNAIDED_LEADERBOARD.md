# Unaided leaderboard (task 0 soup, n=10, no assist flags at eval)

Two boards, per user directive: best by SCORE, and best by PURITY (how
little calibrated/hand-given information the run used). Every row's weights
are backed up under `results_backup/weights/`; per-round results/telemetry
under `results_backup/rounds/`; success films under `watch_videos/`.

## Board 1 — best mean_success

| rank | run | score | machine version | calibrated extras beyond arm physics | weights |
|---|---|---|---|---|---|
| 1 | unaided_goal5 | **0.700** | v10.4 (hang_comp) | hang_comp (calibrated, T1) | goal_heads_v10_goal3_goal4.pt + full_stageB_rec_fix.pt |
| 2 | unaided_goal3 | 0.400 | v10.2 (2D probe, EMA refine, restart) | none | same |
| 2 | unaided_goal2 | 0.300 | v10.1 (2D probe, raw refine) | none | same |
| 3 | unaided_goal4 | 0.200 | v10.3 (hysteresis, drop_z 0.12) | none | same |
| 4 | unaided_goal1 | 0.100 | v10.0 (x-only probe, hover latch) | none | same |
| — | free-regression arm (bc2…phase1, 6 rounds) | 0.000 | n/a (per-tick action regression) | none | stageB ckpts on pod |

## Board 2 — most unaided (purity), best score within each tier

Tier definitions:
* **T0 — fully learned task content, no calibrated task-adjacent constants:**
  goals from trained heads only; machine constants are arm physics
  (gains/heights/timing shared with any pick-and-place).
* **T1 — T0 plus offline-calibrated task-adjacent constant(s):** e.g.
  `hang_comp` (place-side hand-eye offset fitted from logged rollouts —
  same method that produced the teacher's lever arm).
* **T2 — assisted:** teacher machine and/or task constants passed at eval
  (`--ibvs-*`). Reference ceiling, never claimed as unaided.

| tier | best run | score | note |
|---|---|---|---|
| T0 | unaided_goal3 | **0.400** | the headline claim-safe number |
| T1 | unaided_goal5 | **0.700 dev / 0.300 held-out** | hang_comp; dev-tuned — held-out is the citable number |
| T2 | ceiling_soup_v1 (teacher) | 1.000 (n=4) / 0.750 (n=8) | calibrated PhasedIBVS |

Updated: 2026-08-04. Pod wiped+rebuilt (py3.10 venv, 603 tests green); randomized-placement corpus recording; heads-v2 + goal6 trio pending. De-skeletonization track (learned
gates, LoRA'd goal head, goal-persistence aux) aims to raise T0 directly.

## v2.1 update (2026-08-04) — de-memorization verified

| protocol | v1 head | **v2.1 head (variance + jitter)** |
|---|---|---|
| dev | 0.700 | 0.700 |
| held-out inits | 0.300 | **0.700** (citable) |
| randomized ±4 cm | impossible | **0.500** (citable; first moved-object successes) |

Weights: `goal_heads_v21.pt`. 19 success films across the trio in
`watch_videos/succ_g61_*`. Corpus: 10 randomized teacher episodes
(rebuilt stack). Machine constants unchanged (still hand-set; the
de-skeletonization track owns them).
