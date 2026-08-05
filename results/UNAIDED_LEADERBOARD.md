# Unaided leaderboard (LIBERO-Object task 0 soup, no assist flags at eval)

Two boards, per user directive: best by SCORE, and best by PURITY (how
little calibrated/hand-given information the run used). Every row's weights
are backed up under `results_backup/weights/`; per-round results/telemetry
under `results_backup/rounds/` or the run dirs named below; success films
under `watch_videos/`. Every number traces to a `*_results.json` artifact.
Cells are n=10 unless marked; Wilson 95% CIs as in MANUSCRIPT_v2.md §7.

## Board 1 — best mean_success (goal-structured era, variance-trained heads)

| rank | head / cell | dev | held-out | randomized ±4 cm | provenance |
|---|---|---|---|---|---|
| 1 | **flagship v5 (released, 49 ep)** | 4/10 | **7/10 [0.40, 0.89]** — confirmed at n=50: **35/50 = 0.700 [0.56, 0.81]** | 4/10 [0.17, 0.69] — confirmed at n=50: **26/50 = 0.520 [0.39, 0.65]** | `models/goal_heads_v5.pt`; p50 runs on pod |
| 2 | sibling v2.1 (10 ep) | 7/10 | 7/10 | 5/10 | `goal_heads_v21.pt` |
| 3 | sibling v3 (27 ep) | 7/10 | 6/10 | 3/10 | `goal_heads_v3.pt` |
| — | memorized v1 (111 ep fixed corpus) | 7/10 | 3/10 | **1/10** (pod control, identical draws) | audited baseline |
| — | free per-tick regression (7 variants, 6 rounds) | 0/56 pooled | — | — | the policy-class zero |

Controls and parity cells (all artifact-verified):

* **Pod-stack control (2026-08-05):** memorized head under the flagship's
  exact randomized protocol and identical shift draws → **1/10**, lone
  success on the smallest draw (norm 1.4 cm). De-memorization is
  behaviourally certified on the deployment stack.
* **Audit-stack controls (2026-08-04, macOS stack, never pooled with pod
  cells):** memorized randomized 6/10 vs its unshifted 3/10; flagship
  randomized **0/10** (visual goals are detector-stack-pinned). The
  behavioural certificate is joint in (head, stack).
* **World-model inertness:** persistence-TRM ablation at exact parity
  (0.700 held-out) — no task-success contribution from the TRM in
  structured mode.
* **Learned gates:** close/hold gate swap at exact parity 0.700 (stage-1
  de-skeletonization holds score).

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
| T0 | unaided_goal3 (v10.2 machine, no hang_comp) | 0.400 | claim-safe fully-learned-goals number |
| T1 | **flagship v5** | **0.700 held-out (n=10 and n=50)** | hang_comp disclosed; the citable headline |
| T1 | flagship v5, randomized ±4 cm | **0.520 (26/50, n=50)**; 0.400 at n=10 | moved-object protocol; memorized control 1/10 on identical draws |
| T2 | ceiling_soup_v1 (teacher) | 1.000 (n=4) / 0.750 (n=8) | calibrated PhasedIBVS |

## History (v10 ladder, structured-control era, single-object)

| run | score | machine version |
|---|---|---|
| unaided_goal5 | 0.700 dev | v10.4 (hang_comp) |
| unaided_goal3 | 0.400 dev | v10.2 (2D probe, EMA refine, restart) |
| unaided_goal2 | 0.300 dev | v10.1 |
| unaided_goal4 | 0.200 dev | v10.3 |
| unaided_goal1 | 0.100 dev | v10.0 |

Boundary rows (kept honest): all-tasks zero-shot 0.067 (soup 2/3, tasks
1–9 = 0.00 at n=3 — measured with sibling v3, the memorized head, AND
the released v5 itself, `libero_object_real_1785913852707`: soup-only
corpus bounds the head, not the detector); butter re-diagnosed 2026-08-05 — the "grasp-strategy
negative" was a jaw-axis calibration residual (squeeze-out mechanism
measured; centered offset → first butter lift, audit stack) and the v6
multi-object campaign is unblocked; LoRA joint training negative at 27 ep
(val 3.08 vs 0.99 cm).

Post-rebuild re-measures (2026-08-05): memorized dev 7/10 -> 2/10 across
the stack rebuild; flagship dev 4/10 -> 4/10 (reproduces in all three
protocols). Memorization is stack-coupled; grounding survived.

## Multi-object addendum (2026-08-05, dated cells, v7 head = soup+butter corpus)

| cell | soup | butter | note |
|---|---|---|---|
| standard latch | 7/10 (v7) / 9/10 (v6) | 0/10 | offline-clean head, live estimate-chase |
| early latch + freeze (v8) | 1/10 | **10/10** | chase causally confirmed |
| anchor trust-region 4 cm (v9) | 4/10 | 6/10 | one config, no per-object constants |
| **seed-77 confirmation (pre-registered, single shot)** | **3/10** | **5/10** | fresh seed; consistent with band |

Selection ledger: v6-v9 scored the seed-20 band (4 config looks) before
the pre-registration; the confirmation seed was never scored by any
student run.

Updated: 2026-08-05 cycle 12 — n=50 cells landed (35/50 held-out,
26/50 randomized), rebuild matrix complete, v5 zero-shot row measured,
multi-object addendum confirmed on fresh seed (5/10 butter + 3/10 soup).
