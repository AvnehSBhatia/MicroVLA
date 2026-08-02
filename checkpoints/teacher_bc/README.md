# teacher_bc — unaided distillation track

Stage-B BC / DAgger students warm-started from `full_stageB_rec_fix.pt`.

| ckpt | data | note |
|---|---|---|
| `full_stageB_teacher_bc.pt` | teacher_grid (~23) | round 1 |
| `full_stageB_teacher_bc2.pt` | teacher_grid2 (100 soup) | approach improved; still stalls |
| `full_stageB_teacher_bc3.pt` | dagger-only (40) + magnitude losses | **approach fixed** (eef_min ~0.061 m) but `grip_close_rate` 0.000 |

Eval protocol (unaided): no IBVS/tool-phase; wrist cam; see `unaided_v3_results.json`.

`teacher_bc4` (aggregate 100+40) trains separately — not in this bundle until eval finishes.
