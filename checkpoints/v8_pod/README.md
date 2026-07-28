# v8_s0 — first model with a genuinely working world model

Trained 2026-07-28 on RunPod (RTX A4500) over 1000 episodes
(`libero_object` + `libero_goal`, 500 each; `libero_spatial` was excluded by the
sighted-corpus gate at 13.8% source detection).

| file | contents |
|---|---|
| `full_stageA_v8_s0.pt` | world model: `EvidenceEncoder` + `HRMBackbone` + `RecursiveTRM` (12,195,734 params) |
| `full_stageB_v8_s0.pt` | the above + `RelationalHead` + `ChronoQueryPlanner` (16,435,560) |
| `norm_stats.json` | action normalizer **fitted on this corpus** |
| `waypoint_stats.json` | per-axis waypoint gains for this corpus |

The two JSONs are mandatory. A normalizer from another corpus silently rescales
every command, and a waypoint gain fitted under a different normalization is
meaningless.

## Measured

**`wm_margin +43.3%`** — the best world model this project has produced, by a
wide margin:

| run | wm_margin |
|---|---|
| **v8_s0 (this)** | **+43.3%** |
| v7 best (`full_stageA_wrist_v72`) | +19.8% |
| Mac cold start, 500 ep | +3.5% |
| ROCm box (OOM'd mid horizon-ramp) | −46.8% |

Bench on 20 held-out episodes: `std_ratio` 0.264 · `corr` 0.48 · `grip` 0.94 ·
`pose_mae` 0.202. Planner sensitivity puts `relational` second (0.0940) behind
`proprio` (0.1433).

Stage B was **still training** when this snapshot was taken — it is a usable
checkpoint, not a converged one.

## The number that matters more than any of these

Ground-truth demo actions, replayed through the real simulator, solve the task
**5/5**. Scaled to **0.8x magnitude they solve 0/4.** The task tolerates almost
no magnitude loss, and `std_ratio` measures exactly that ratio — so at 0.264
this policy cannot succeed regardless of how well it predicts direction.

That is not a property of this checkpoint; it is the bar the benchmark imposes.
See `eval/replay_check.py` and `eval/actuation_check.py`.

## Run it

```bash
python -m eval.bench --checkpoint checkpoints/v8_pod/full_stageB_v8_s0.pt \
  --data-dir data/libero_object_v8 --episodes 30 --sensitivity --device cpu

python -m eval.libero_eval --suite libero_object --n-trials 5 --max-steps 300 \
  --checkpoint checkpoints/v8_pod/full_stageB_v8_s0.pt \
  --norm-stats checkpoints/v8_pod/norm_stats.json \
  --waypoint-stats checkpoints/v8_pod/waypoint_stats.json --device cpu
```
