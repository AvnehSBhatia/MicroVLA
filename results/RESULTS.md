# MicroVLA — Results (auto-generated from results/metrics.jsonl)

Generated 2026-07-22T04:32:00+00:00 · 13 records · **do not hand-edit** (regenerate: `python -m experiments.tracker report`)

## Provenance

- `8cc7c66` 2026-07-22T04:28:55+00:00 — Dataset: Bridge V2 RLDS 4839 eps (43% grounded, ext cam) + LIBERO 1500 eps (wrist cam eye_in_hand); 6339 total, 90k perception-baked frames. Trainable heads 6.79M/9M. TRM RecursiveTRM d=1024 9.5M.

## Stage-A world model (rollout loss vs persistence)

| run | recipe | ep | H | train | val | persistence | margin | s |
|---|---|---|---|---|---|---|---|---|
| pilot | prefix-8tick-single-target | 1/3 | — | 0.1125 | 0.0558 | 0.0216 | -158% | 606 |
| run1-bridge+libero | scheduled-horizon-datarate | 1/4 | 1 | 0.0104 | 0.0084 | 0.0082 | -2% | 641 |
| run1-bridge+libero | scheduled-horizon-datarate | 2/4 | 3 | 0.0135 | 0.0117 | 0.0132 | +11% | 1110 |
| run1-bridge+libero | scheduled-horizon-datarate | 3/4 | 4 | 0.0135 | 0.0119 | 0.0147 | +19% | 1367 |
| run2-restart | scheduled-horizon-datarate | 1/4 | 1 | 0.0104 | 0.0084 | 0.0082 | -2% | 641 |

## Horizon curve (Claim 2 early evidence — margin vs rollout depth)

| checkpoint | H | val | persistence | margin |
|---|---|---|---|---|
| full_stageA_ep3_backup.pt | 1 | 0.00712 | 0.00753 | +6% |
| full_stageA_ep3_backup.pt | 2 | 0.00906 | 0.01013 | +11% |
| full_stageA_ep3_backup.pt | 3 | 0.01092 | 0.0132 | +17% |
| full_stageA_ep3_backup.pt | 4 | 0.0119 | 0.01466 | +19% |
| full_stageA_ep3_backup.pt | 5 | 0.01255 | 0.0158 | +20% |
| full_stageA_ep3_backup.pt | 6 | 0.0138 | 0.01715 | +20% |
| full_stageA_ep3_backup.pt | 8 | 0.01497 | 0.01821 | +18% |
