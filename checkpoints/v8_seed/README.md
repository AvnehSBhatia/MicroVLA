# v8 seed checkpoints — the first MicroVLA weights trained on a corpus that could see

Trained 2026-07-26 on `libero_object` (500 episodes, wrist view), freshly baked
AFTER the root-cause fix in `paper.md` §4n. Every earlier checkpoint in this
project — including `full_stageA_wrist_v72.pt` — was trained on a corpus where
the source object was detected on **0.0%** of frames, so the policy had no
object information at all. This corpus has it on **45.1%**.

| file | what it is |
|---|---|
| `full_stageA_v8_fixed.pt` | world model: `EvidenceEncoder` + `HRMBackbone` + `RecursiveTRM` |
| `full_stageB_v8_fixed.pt` | the above plus `RelationalHead` + `ChronoQueryPlanner` |
| `norm_stats.json` | action normalizer **fitted on this corpus** |
| `waypoint_stats.json` | per-axis waypoint gains fitted on this corpus |

The two JSONs are not optional. An action normalizer fitted on a different
corpus silently rescales every command, and a waypoint gain fitted under a
different normalization is meaningless — always carry the pair that shipped with
the weights.

## Measured

* **Stage A beats persistence**: val `0.0388` vs persistence `0.0402`
  (`wm_margin +3.5%`), from a cold start. Early-stopped at epoch 16, best at 13.
* **Stage B, epoch 11/40** when it was stopped by hand to move to a bigger
  corpus: `val bc` 0.2417, `grip` 0.942. It had NOT converged and the
  `--stage-b-min-epochs 20` floor had not yet been reached.

`val bc` here is **not comparable** to numbers from other corpora. It is an
MSE-family loss and this corpus fitted its own symmetric normalizer, so the loss
scales with the square of target magnitude. Compare `std_ratio` (a ratio, hence
normalization-invariant) or closed-loop success instead.

## Using it as a warm start

```bash
python train/train_batched.py --data-dir <bigger corpus> --v8 \
  --load-stage-a checkpoints/v8_seed/full_stageA_v8_fixed.pt ...
```

Caveat worth weighing: this stage A saw ONE suite. On a three-suite corpus a
cold stage A may beat warm-starting from a narrower one — the world model is
cheap relative to stage B, so it is worth running both once and keeping the
better `wm_margin`.
