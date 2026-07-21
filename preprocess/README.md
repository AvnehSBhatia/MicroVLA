# Dataset preprocessing

Converts raw robot datasets into MicroVLA's native `.npz` episode format
(`train/dataset.py::EPISODE_KEYS`). The frozen perception stack — YOLO-World-S
SPPF embeddings, per-role boxes, CLIP text tokens — runs exactly **once, here**;
training never touches an image, episodes are ~1000× smaller than raw video, and
the training distribution is bit-identical to what deployed perception produces.

**Nothing is downloaded by these scripts.** Obtain the datasets yourself, then
point the converters at your local copies. Converted data lands under `data/`
(git-ignored).

| dataset | role | rate | converter |
|---|---|---|---|
| [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) (raw layout) | main pretraining set | 5 Hz | `python -m preprocess.bridge` |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (hdf5 suites) | fine-tune + eval | 20 Hz | `python -m preprocess.libero` |

Both are 7-DoF (Δxyz, Δrpy, gripper) — exactly `cfg.num_servos = 7`.

## Quick reference

```bash
pip install -e ".[perception,data]"   # adds ultralytics/cv2 (+ h5py for LIBERO)

# Smoke everything with mock perception first (no weights, no GPU):
python -m preprocess.libero /data/libero_object data/libero_object --dry-run --limit 3
python -m preprocess.bridge /data/bridgedata_raw data/bridge --dry-run --limit 3

# Real conversion:
python -m preprocess.libero /data/libero_object data/libero_object --device cpu
python -m preprocess.bridge /data/bridgedata_raw data/bridge --device cpu

# Then train:
python train/train_planner.py --data-dir data/bridge
```

## What conversion does

1. **Pass 1 — action stats.** Streams every episode, fits per-dimension
   q01/q99 quantile normalization (robust to teleop spikes), saves
   `norm_stats.json`. Keep this file with any checkpoint — planner outputs
   only mean something through `ActionNormalizer.load(...).inverse(...)`.
2. **Pass 2 — perception + write.** Subsamples frames to
   `cfg.real_frame_hz = 2` (same integer-counter cadence as the online
   `VideoStreamSampler`), parses each instruction (`parse_command` →
   article-stripped source/target detector classes), harvests CLIP text
   tokens, runs YOLO-World per sampled frame (standardized frame/box
   embeddings, per-role confidences → `box_weights`), builds
   `pwm_targets[t]` = the next `plan_steps=5` native-rate actions from
   frame `t` (end padded by holding the last action), and writes one `.npz`
   per episode plus `manifest.json`.

Notes:
- **LIBERO frames are rotated 180°** by default (robosuite's agentview camera
  renders upside down — same handling as OpenVLA/Octo). `--no-rotate-180`
  disables.
- **Bridge trajectories without `lang.txt` are skipped** by default (MicroVLA
  is language-conditioned); `--fallback-task-lang` uses the task directory
  name instead.
- Plan rows are spaced at the **dataset's native control rate** (5 Hz Bridge /
  20 Hz LIBERO). At deployment the loop replans every tick, so only row 0 is
  ever executed per tick — but be aware of the timescale difference when
  mixing corpora.
- TRM rollout training needs no extra keys: the next-real-frame target is
  `frame_embs[t+1]` of the same episode.

## Teacher distillation (TinyVLA)

Any converter can relabel actions with a larger pretrained VLA instead of the
human teleop — knowledge distillation ([TinyVLA](https://tiny-vla.github.io/)
supported; its diffusion head natively emits action chunks that map onto our
5-row plans):

```bash
git clone <TinyVLA repo> /opt/TinyVLA           # you obtain repo + checkpoint
python -m preprocess.bridge /data/bridgedata_raw data/bridge_distilled \
    --teacher tinyvla \
    --teacher-checkpoint /opt/tinyvla.ckpt \
    --teacher-repo /opt/TinyVLA \
    --teacher-cache data/teacher_cache
```

- `--teacher-cache` stores relabeled actions per episode so the teacher pays
  once across the two conversion passes (and across reruns).
- `--teacher mock` exercises the full distillation path with a deterministic
  fake teacher (used by the tests).
- One integration point: `TinyVLATeacher._load_policy` in
  `preprocess/teacher.py` — adapt it to your TinyVLA checkout's eval API
  (their entrypoints move between releases; everything else — caching,
  chunk tiling, normalization — is already wired).
- **Embodiment caveat:** distill in the dataset's own action frame (Bridge's
  WidowX convention matches TinyVLA's Bridge configs). For your physical
  rig, finetune the teacher first or retarget its actions — see the README's
  distillation section.
- `manifest.json` records `label_source` so distilled and human-labeled runs
  can't be silently mixed up.

## Output layout

```
data/<name>/
  norm_stats.json      # q01/q99 per action dim (ActionNormalizer.save)
  manifest.json        # label_source + per-episode {file, id, T, instruction}
  <episode_id>.npz     # EPISODE_KEYS arrays, one per demo/trajectory
```
