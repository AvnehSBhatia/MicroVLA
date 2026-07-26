# MicroVLA — Results (auto-generated from results/metrics.jsonl)

Generated 2026-07-26T01:56:20+00:00 · 66 records · **do not hand-edit** (regenerate: `python -m experiments.tracker report`)

## Provenance

- `8cc7c66` 2026-07-22T04:28:55+00:00 — Dataset: Bridge V2 RLDS 4839 eps (43% grounded, ext cam) + LIBERO 1500 eps (wrist cam eye_in_hand); 6339 total, 90k perception-baked frames. Trainable heads 6.79M/9M. TRM RecursiveTRM d=1024 9.5M.
- `bb578ee` 2026-07-26T01:55:50+00:00 — v7.2 full-data run. Suites baked one at a time (disk budget), then preprocess.unify_norm_stats onto one shared symmetric scale.

## Stage-A world model (rollout loss vs persistence)

| run | recipe | ep | H | train | val | persistence | margin | s |
|---|---|---|---|---|---|---|---|---|
| pilot | prefix-8tick-single-target | 1/3 | — | 0.1125 | 0.0558 | 0.0216 | -158% | 606 |
| run1-bridge+libero | scheduled-horizon-datarate | 1/4 | 1 | 0.0104 | 0.0084 | 0.0082 | -2% | 641 |
| run1-bridge+libero | scheduled-horizon-datarate | 2/4 | 3 | 0.0135 | 0.0117 | 0.0132 | +11% | 1110 |
| run1-bridge+libero | scheduled-horizon-datarate | 3/4 | 4 | 0.0135 | 0.0119 | 0.0147 | +19% | 1367 |
| run2-restart | scheduled-horizon-datarate | 1/4 | 1 | 0.0104 | 0.0084 | 0.0082 | -2% | 641 |
| v72_stageA_full |  | 1/40 | 1 | 0.0964 | 0.0072 | 0.0064 | -12% | 77 |
| v72_stageA_full |  | 2/40 | 3 | 0.0667 | 0.012 | 0.0091 | -32% | 177 |
| v72_stageA_full |  | 3/40 | 4 | 0.0654 | 0.0154 | 0.0102 | -51% | 144 |
| v72_stageA_full |  | 4/40 | 6 | 0.0674 | 0.0163 | 0.0115 | -42% | 218 |
| v72_stageA_full |  | 5/40 | 6 | 0.0638 | 0.0166 | 0.0115 | -44% | 222 |
| v72_stageA_full |  | 6/40 | 6 | 0.0628 | 0.0157 | 0.0115 | -37% | 165 |
| v72_stageA_full |  | 7/40 | 6 | 0.0596 | 0.0172 | 0.0115 | -50% | 171 |
| v72_stageA_full |  | 8/40 | 6 | 0.057 | 0.0132 | 0.0115 | -15% | 168 |
| v72_stageA_full |  | 9/40 | 6 | 0.0528 | 0.0154 | 0.0115 | -34% | 262 |
| v72_stageA_full |  | 10/40 | 6 | 0.0541 | 0.0133 | 0.0115 | -16% | 382 |
| v72_stageA_full |  | 11/40 | 6 | 0.0539 | 0.0129 | 0.0115 | -12% | 487 |
| v72_stageA_full |  | 12/40 | 6 | 0.0498 | 0.0167 | 0.0115 | -45% | 496 |
| v72_stageA_full |  | 13/40 | 6 | 0.0503 | 0.0127 | 0.0115 | -10% | 367 |
| v72_stageA_full |  | 14/40 | 6 | 0.0471 | 0.0133 | 0.0115 | -16% | 366 |
| v72_stageA_full |  | 15/40 | 6 | 0.0453 | 0.0124 | 0.0115 | -8% | 364 |
| v72_stageA_full |  | 16/40 | 6 | 0.044 | 0.0114 | 0.0115 | +1% | 363 |
| v72_stageA_full |  | 17/40 | 6 | 0.0431 | 0.0122 | 0.0115 | -6% | 367 |
| v72_stageA_full |  | 18/40 | 6 | 0.0432 | 0.0117 | 0.0115 | -2% | 367 |
| v72_stageA_full |  | 19/40 | 6 | 0.0421 | 0.0132 | 0.0115 | -15% | 365 |
| v72_stageA_full |  | 20/40 | 6 | 0.0389 | 0.0109 | 0.0115 | +5% | 367 |
| v72_stageA_full |  | 21/40 | 6 | 0.0376 | 0.0112 | 0.0115 | +3% | 364 |
| v72_stageA_full |  | 22/40 | 6 | 0.0381 | 0.0115 | 0.0115 | +0% | 363 |

## Stage-B policy (behavior cloning)

| run | tqsa | ep | train | grip | val | val grip | s |
|---|---|---|---|---|---|---|---|
| v72_stageB_nospatial | False | 3/40 | 0.7515 | 0.659 | 0.8461 | 0.587 | — |
| v72_stageB_nospatial | False | 4/40 | 0.7149 | 0.673 | 0.6402 | 0.716 | — |
| v72_stageB_nospatial | False | 5/40 | 0.6744 | 0.685 | 0.6223 | 0.726 | — |
| v72_stageB_nospatial | False | 6/40 | 0.6663 | 0.698 | 0.6195 | 0.746 | — |
| v72_stageB_nospatial | False | 7/40 | 0.6549 | 0.709 | 0.6355 | 0.731 | — |
| v72_stageB_nospatial | False | 8/40 | 0.6676 | 0.697 | 0.6035 | 0.746 | — |
| v72_stageB_nospatial | False | 9/40 | 0.6515 | 0.711 | 0.5799 | 0.755 | — |
| v72_stageB_nospatial | False | 10/40 | 0.6431 | 0.709 | 0.592 | 0.75 | — |
| v72_stageB_nospatial | False | 11/40 | 0.6161 | 0.725 | 0.5645 | 0.762 | — |
| v72_stageB_nospatial | False | 12/40 | 0.6056 | 0.738 | 0.576 | 0.748 | — |
| v72_stageB_nospatial | False | 13/40 | 0.6161 | 0.73 | 0.5801 | 0.752 | — |
| v72_stageB_nospatial | False | 14/40 | 0.6016 | 0.73 | 0.5713 | 0.762 | — |
| v72_stageB_nospatial | False | 15/40 | 0.5984 | 0.735 | 0.58 | 0.731 | — |
| v72_stageB_tqsa | True | 1/40 | 1.1456 | 0.55 | 0.948 | 0.567 | — |
| v72_stageB_tqsa | True | 2/40 | 0.9082 | 0.596 | 0.859 | 0.717 | — |
| v72_stageB_tqsa | True | 3/40 | 0.7482 | 0.663 | 0.9173 | 0.547 | — |
| v72_stageB_tqsa | True | 4/40 | 0.7266 | 0.671 | 0.633 | 0.73 | — |
| v72_stageB_tqsa | True | 5/40 | 0.6824 | 0.691 | 0.6303 | 0.74 | — |
| v72_stageB_tqsa | True | 6/40 | 0.6626 | 0.693 | 0.6154 | 0.761 | — |
| v72_stageB_tqsa | True | 7/40 | 0.6619 | 0.699 | 0.6239 | 0.73 | — |
| v72_stageB_tqsa | True | 8/40 | 0.6518 | 0.712 | 0.6123 | 0.726 | — |
| v72_stageB_tqsa | True | 9/40 | 0.642 | 0.719 | 0.5942 | 0.743 | — |
| v72_stageB_tqsa | True | 10/40 | 0.633 | 0.718 | 0.5719 | 0.757 | — |
| v72_stageB_tqsa | True | 11/40 | 0.615 | 0.729 | 0.5613 | 0.773 | — |
| v72_stageB_tqsa | True | 12/40 | 0.6041 | 0.74 | 0.6231 | 0.742 | — |
| v72_stageB_tqsa | True | 13/40 | 0.611 | 0.734 | 0.5762 | 0.761 | — |

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

## Bench (open-loop fidelity — the gate before sim)

| run | data | std_ratio | mae | corr | grip | wm_margin | spatial |
|---|---|---|---|---|---|---|---|
| v7_pilot | data/libero_v7 (libero_object only) | 0.369 | 0.2 | 0.49 | 0.93 | 1.7% | False |
| v72_stageB_nospatial | data/libero_object_v7 | 0.175 | 0.19 | 0.28 | 0.93 | -7.3% | False |

`spatial=False` on a TQSA-TRAINED checkpoint means the planner was scored without ~27% of its memory tokens — see paper.md §0.

### Planner input sensitivity (mean |dPlan| when withheld)

| input | v7_pilot | v72_stageB_nospatial |
|---|---|---|
| current_emb | 0.0250 | 0.0133 |
| fused | 0.0230 | 0.0218 |
| geometry | 0.0040 | 0.0914 |
| next_emb->cur | 0.0010 | 0.0017 |
| next_emb->stale | — | 0.0059 |
| pred_box_emb | 0.0130 | 0.0029 |
| proprio | 0.2910 | 0.2243 |
| state_delta | 0.0750 | 0.0134 |
| wm_msg | 0.0310 | 0.0007 |

## Infrastructure measurements

- **spatial_map_cache** — before_s_per_epoch 1080, after_s_per_epoch 130, one_time_pass_s 151, speedup 8.3, frames_train 19680, frames_val 1031, map_shape [512, 20, 20], kb_per_frame_fp16 400, gb_resident 7.9, frames_per_s 160
  - Frozen backbone => maps identical across epochs. Rejected after measurement: min_side 512->256 saves 0.6% not 4x (all letterbox to 640, same 20x20 map). Untaken: SPPF-truncated forward, maxdiff 0.0, 1.8-2.0x.
- **gpu_contention** — processes_on_gpu1 7, vram_used_gb 154, vram_total_gb 192, stage_a_epoch_s_uncontended 96, stage_a_epoch_s_contended 496, slowdown 5.2
  - Every per-epoch second recorded this session is contended and is not a hardware claim.
