"""Batched MicroVLA training — for a real GPU (CUDA / AMD ROCm, e.g. MI300X).

The Mac trainer (train_full.py) runs batch=1 because the drift encoder is
stateful. But every module already accepts a batch dimension, so on a real GPU
we batch by **bucketing episodes by exact length T** (Bridge/LIBERO T ranges
~10-17, a handful of buckets) — within a bucket all episodes are the same
length, so batching needs no padding or masking. Everything else (objective,
scheduled-horizon rollout, early stopping, best-checkpoint) matches train_full
exactly, so results are comparable.

VRAM cap: ``--max-vram-gb`` hard-limits the process via
``torch.cuda.set_per_process_memory_fraction`` (ROCm honors the cuda API), so
on a 192 GB MI300X asked for 50 GB, the process physically cannot exceed 50 GB
— it OOMs inside the cap rather than eating the whole card. Peak usage is
printed after epoch 1 so you can tune ``--batch-size``.

Data is preloaded into RAM once (the whole corpus is <1 GB), so epochs are
GPU-bound, not disk-bound.

Example (MI300X, ROCm):
    python train/train_batched.py --data-dir data/bridge --data-dir data/libero \\
        --device cuda --batch-size 64 --max-vram-gb 50 \\
        --stage-a-epochs 30 --warmup-epochs 4 --max-horizon 6 --patience 3 \\
        --stage-b-epochs 3
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import random
import sys
import time
from collections import defaultdict
import json
from pathlib import Path

# ROCm's hipBLASLt throws INTERNAL_ERROR on some GEMM shapes (e.g. the v4 TRM
# box head's 256x512 backward at batch 64), corrupting the BLAS state -> a fatal
# `hipblasCreate ALLOC_FAILED`. Force the stable rocBLAS path. HIP-only var
# (no-op on CUDA); must be set BEFORE importing torch; override by exporting it.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.relational import RelationalHead
from microvla.v8 import (DriftAdapter, FusionAdapter, objects_from_batch,
                         pack_objects)
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.critic import ProgressCritic, frozen_value, progress_targets
from microvla.utils.embedding import standardize
from microvla.utils.phase import pre_grasp_weights
from microvla.utils.signals import ignore_sigterm
from microvla.utils.waypoint import long_horizon_targets, waypoint_targets
from train.dataset import EPISODE_KEYS, OPTIONAL_KEYS, EpisodeDataset
from train.losses import (planner_bc_loss, smoothness_loss, split_planner_loss,
                          total_planner_loss, waypoint_loss)
from train.train_full import _scheduled_horizon, _tagged_name, save
from microvla.utils.param_audit import count_trainable_params
from train.train_planner import resolve_device
from TRM import RecursiveTRM, spec_loss


#: Minimum seconds between intra-epoch progress lines. Long enough to stay out
#: of the way on fast epochs, short enough that "silent" always means "stuck".
_HEARTBEAT_SEC: float = 60.0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", action="append", required=True)
    p.add_argument("--batch-size", type=int, default=64,
                   help="episodes per batch (within a length bucket).")
    p.add_argument("--max-vram-gb", type=float, default=50.0,
                   help="hard cap on GPU memory for this process (cuda/ROCm).")
    p.add_argument("--stage-a-epochs", type=int, default=30, help="hard cap (early stop usually halts first)")
    p.add_argument("--warmup-epochs", type=int, default=4)
    p.add_argument("--max-horizon", type=int, default=6)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--box-loss-weight", type=float, default=0.5,
                   help="weight on the v4 TRM box-prediction spec_loss in stage A "
                        "(0 disables it). Target = recorded source_box_embs[t+k].")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr-patience", type=int, default=2,
                   help="halve LR after this many at-max-horizon epochs without val "
                        "improvement (< --patience, so LR drops before early stop).")
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--stage-b-epochs", type=int, default=3)
    p.add_argument("--dream-frac", type=float, default=0.0,
                   help="stage B: fraction of steps trained in the DREAM regime "
                        "(current latent = TRM prediction, held/faded boxes) the "
                        "planner actually runs in 14/15 ticks at deployment. "
                        "0 = old real-only behavior; 0.25 recommended.")
    p.add_argument("--variance-weight", type=float, default=0.0,
                   help="stage B: weight on matching the per-dim STD of the "
                        "predicted pose to the demo's. MSE regression to the "
                        "conditional mean is why every arm emits std_ratio "
                        "0.26-0.42 against a task whose passing band is ~[0.95, "
                        "1.05] (paper.md 4p); MSE alone rewards shrinking exactly "
                        "this quantity. 0.1 is a reasonable start.")
    p.add_argument("--critic-weight", type=float, default=0.0,
                   help="stage B: weight on fitting the ProgressCritic (a "
                        "TRAINING-ONLY head, outside the deployed 9M budget) to "
                        "position-within-episode. Must be > 0 for "
                        "--progress-weight or --dream-weight to do anything.")
    p.add_argument("--progress-weight", type=float, default=0.0,
                   help="stage B: weight on maximizing the critic's value of the "
                        "IMAGINED next latent reached by the planner's own "
                        "action. Task-aligned and fully differentiable -- fusion's "
                        "action token carries the plan into the world model, so no "
                        "environment is needed. This is the term that asks for "
                        "actions which ADVANCE the task rather than ones that "
                        "merely resemble the demonstrator's.")
    p.add_argument("--dream-weight", type=float, default=0.0,
                   help="stage B: weight on the imagined rollout BEYOND one step "
                        "(Dreamer-style). Keep it small: it compounds world-model "
                        "error, and 4w measures the 1-step margin over persistence "
                        "at only +1.7%% MSE, so a policy can exploit the model "
                        "faster than it can learn from it. 0.01-0.05.")
    p.add_argument("--dream-horizon", type=int, default=1,
                   help="imagined steps per timestep; 1 = critic term only.")
    p.add_argument("--dream-gamma", type=float, default=0.9,
                   help="discount applied down the imagined rollout.")
    p.add_argument("--action-token-sampling", type=float, default=0.0,
                   help="stage B: probability of feeding FUSION'S ACTION TOKEN the "
                        "model's OWN previous plan row 0 instead of the "
                        "demonstration's (scheduled sampling). Fusion's 8th token is "
                        "'the previously executed action', and training fed it the "
                        "demo's while deployment can only feed the policy's own; "
                        "paper.md 4v attributes essentially the whole closed-loop "
                        "failure to that one asymmetry (teacher-forcing the token at "
                        "eval takes the gripper from 13% to 47% of steps closed and "
                        "makes the deployed stack reproduce the trainer bit-for-bit). "
                        "0 = old teacher-forced behavior; 0.5 recommended.")
    p.add_argument("--planner-input-dropout", type=float, default=0.15,
                   help="stage B: per-step probability of WITHHOLDING the planner's "
                        "dominant inputs (fused; independently current_emb) so the "
                        "predictive/geometric paths get gradient. Interpretability "
                        "probe showed fused 7x dominant, geometry/next_emb dead — "
                        "redundant-path death; same cure as fusion's modality dropout.")
    p.add_argument("--waypoint-long", action="store_true",
                   help="v7.4: supervise the waypoint head at the SAMPLED (2 Hz) spacing "
                        "instead of the native one — 0.5-2.5 s of displacement instead of "
                        "0.05-0.20 s. Over 0.2 s 'keep doing what you are doing' is a "
                        "near-sufficient statistic and object position is only a "
                        "second-order correction, which is the conditional-mean ordering "
                        "that produces the measured 12:1 phase:vision ratio; over 2.5 s the "
                        "arm must ARRIVE, so where the object is becomes first-order. Zero "
                        "new params, no re-bake. Sets --waypoint-range and "
                        "--waypoint-row-stride defaults, both of which are unit-critical.")
    p.add_argument("--waypoint-range", type=float, default=None,
                   help="metres spanned by the waypoint head's [-1, 1] output. Default "
                        "0.15 (native spacing) or 0.5 with --waypoint-long, where 0.15 "
                        "would clamp any real reach and destroy the signal.")
    p.add_argument("--waypoint-row-stride", type=int, default=None,
                   help="CONTROL steps between waypoint rows = source_hz/real_frame_hz "
                        "(10 for LIBERO's 20 Hz against 2 Hz sampling). 1 for native "
                        "spacing. The actuator divides the positional error by the number "
                        "of control steps remaining, so a wrong stride under-delivers the "
                        "command by exactly that factor.")
    p.add_argument("--pre-grasp-weight", type=float, default=1.0,
                   help="multiplier on PRE-GRASP timesteps in the stage-B losses (1.0 = "
                        "off). Object position only matters before the grasp; afterwards "
                        "the trajectory is transport to a target in the same place every "
                        "episode, which needs no grounding — so without this most of the "
                        "gradient is spent where vision is useless. Derived from the demo's "
                        "own gripper transition; episodes whose gripper never closes (all "
                        "of bridge) are left at weight 1. Mean-1 normalized per episode, so "
                        "it is not a disguised LR change. Try 3.0.")
    p.add_argument("--planner-drop-rate", type=str, default="",
                   help="stage B: PER-INPUT withhold probabilities, e.g. "
                        "'state_delta=0.4,fused=0.15'. Names from "
                        "ChronoQueryPlanner.INPUT_NAMES. Overrides the coarse "
                        "--planner-input-dropout / --phase-dropout for any name given. "
                        "Applied as a per-sample graded FADE (fusion's own "
                        "1-drop*(1-u) continuum), not deletion, so the attention token "
                        "count matches deployment. Exists because --phase-dropout 0.3 "
                        "(both phase inputs at once) "
                        "measured a 2.3x better phase:vision ratio AND a gripper collapse "
                        "0.93 -> 0.50: proprio carries the arm's GRIPPER STATE, the best "
                        "predictor of the gripper command, so withholding it destroys the "
                        "BCE head. Drop state_delta, keep proprio.")
    p.add_argument("--phase-dropout", type=float, default=0.0,
                   help="stage B: per-step probability of WITHHOLDING each PHASE input "
                        "(state_delta, proprio) from the planner, independently. These are "
                        "the shortcut: a policy that predicts the action from task progress "
                        "and arm pose alone never needs to locate the object, and in "
                        "stereotyped pick-and-place demos with a fixed target that covers "
                        "most of the action variance. Measured consequence of leaving it at "
                        "0: phase sensitivity 0.464 vs vision 0.040 (12:1), and a policy "
                        "that reaches the basket perfectly and never touches the object. "
                        "Try 0.3. Deliberately asymmetric with --planner-input-dropout, "
                        "which withholds the vision paths — drop the shortcut MORE than the "
                        "signal you want used.")
    p.add_argument("--drift-dropout", type=float, default=0.1,
                   help="stage A: per-segment probability of zeroing state_delta into "
                        "the TRM rollout, forcing scene-content dynamics instead of "
                        "the drift-dominated delta (probe: 0.63-0.88 of the residual "
                        "was a function of the drift code alone).")
    p.add_argument("--smooth-weight", type=float, default=0.05,
                   help="pose smoothness (jerk) penalty weight in stage B.")
    p.add_argument("--row0-weight", type=float, default=2.0,
                   help="extra pose-MSE weight on plan row 0 — the only row "
                        "executed at deployment (mean-normalized; 1.0 = uniform).")
    p.add_argument("--segments-per-episode", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--trm-d", type=int, default=1024)
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--ablate-grounding", action="store_true")
    p.add_argument("--load-stage-a", type=str, default=None,
                   help="path to a trained full_stageA.pt: load the world model and skip "
                        "stage A, retraining ONLY the planner (e.g. after a planner change).")
    p.add_argument("--stage-b-patience", type=int, default=0,
                   help="stage B early stopping: >0 enables per-epoch VAL loss, keeps the "
                        "best checkpoint, and halts after this many epochs without a "
                        "--min-delta improvement. 0 = old fixed-epoch behavior.")
    p.add_argument("--actuation-weight", type=float, default=1.0,
                   help="weight on the ACTUATION loss: supervise the command the "
                        "robot receives, not just the displacement the head "
                        "predicts. This is the only gradient path to the HRM's "
                        "learned control law (gain_head sat at exactly its zero "
                        "init in the first v8 checkpoint) and the only term that "
                        "sees emitted MAGNITUDE, which paper.md 4p measures as the "
                        "barrier to task success. 0 disables.")
    p.add_argument("--ckpt-rollout", action="store_true",
                   help="gradient-checkpoint the TRM inside the rollout. Stage A "
                        "sums --segments-per-episode rollouts of depth H into ONE "
                        "graph before backward, so activations scale as segments x "
                        "H and peak grew 3.2 -> 5.2 -> 9.3 GB as H ramped 1 -> 2 -> "
                        "4. Checkpointing stores inputs and recomputes in backward: "
                        "roughly H x less activation memory for one extra forward "
                        "per step. Use when the card is shared.")
    p.add_argument("--reserve-vram-gb", type=float, default=0.0,
                   help="claim this much VRAM at startup and hold it for the whole "
                        "run (see reserve_vram). On a SHARED card the free pool "
                        "shrinks under you — a co-tenant was measured growing "
                        "60 -> 83 GB mid-run — so asking first is the only reliable "
                        "way to keep it. 0 disables.")
    p.add_argument("--v8", action="store_true",
                   help="build the v8 stack (DESIGN.md 'v8 plan'): HRMBackbone in "
                        "place of AnchoredDriftEncoder, RelationalHead in place of "
                        "SlotResonanceFusion and running AFTER the TRM, and "
                        "EvidenceEncoder feeding the TRM's unchanged [B,32,5] port. "
                        "Drops the 'fused'/'geometry'/'pred_box_emb' planner inputs "
                        "(fusion is gone; the relational tokens carry that evidence) "
                        "and adds 'relational'. NOTE: v7 checkpoints are NOT "
                        "loadable — --load-stage-a will fail, so stage A must be "
                        "retrained. Effective K is 2 until the corpus is re-baked "
                        "with microvla/perception/text_region.py; see _obj_tokens.")
    p.add_argument("--stage-b-select", choices=("bc", "total"), default="bc",
                   help="which VAL quantity gates early stopping and best-checkpoint "
                        "selection. 'bc' (default) uses the behavior-cloning term ALONE "
                        "— it is the only term on a common scale across arms, so it is "
                        "the one that makes arms comparable. 'total' adds "
                        "waypoint_weight*val_wp, which is what the 2026-07-25 overnight "
                        "batch used and is NOT comparable: --min-delta is absolute, so "
                        "an arm whose waypoint targets are ~10x larger (--waypoint-long) "
                        "registers 'no improvement' sooner and early-stops sooner. Every "
                        "bench metric in that batch tracked epochs-survived at Spearman "
                        ">=0.84, so the arm rankings measured stop timing, not "
                        "architecture. Kept only to reproduce those runs.")
    p.add_argument("--stage-b-min-epochs", type=int, default=0,
                   help="floor on stage-B epochs: track val and keep the best checkpoint "
                        "as usual, but do not let --stage-b-patience halt before this "
                        "epoch. Guards against a noisy early plateau ending a run at "
                        "epoch 8 of a 40-epoch budget (observed: 8-28 epochs across "
                        "otherwise-identical arms).")
    p.add_argument("--resume-stage-a", action="store_true",
                   help="continue stage A from its last completed epoch (weights, "
                        "optimizer, LR schedule, patience counter) instead of starting "
                        "over. Written every epoch to full_stageA[_tag].resume.pt, "
                        "separate from the best-checkpoint file. For flaky machines: "
                        "the training box SIGTERMs processes intermittently from outside "
                        "the container, and without this a kill at epoch 19 costs the "
                        "whole stage.")
    p.add_argument("--resume-stage-b", action="store_true",
                   help="with --load-stage-a pointing at a full_stageB.pt: ALSO load the "
                        "planner (+tqsa) from it and continue stage B, instead of "
                        "retraining the policy from scratch.")
    p.add_argument("--unfreeze-trm", action="store_true",
                   help="stage B: fine-tune the WHOLE TRM at 0.1x LR with a world-model "
                        "auxiliary rollout loss (frame prediction cannot collapse; verify "
                        "with bench wm_margin). Default trains only the msg head.")
    p.add_argument("--wm-aux-weight", type=float, default=0.5,
                   help="weight of the stage-A rollout auxiliary during --unfreeze-trm.")
    p.add_argument("--tqsa", action="store_true",
                   help="v7: train the Text-Queried Spatial Adapter alongside the planner "
                        "in stage B. Requires wrist_frames in the baked npz (re-bake with "
                        "the v7 converter) and ultralytics (frozen backbone map extractor). "
                        "Buckets without frames train planner-only (spatial=None).")
    p.add_argument("--no-cache-spatial", dest="cache_spatial", action="store_false",
                   help="recompute the frozen backbone's feature maps every epoch instead of "
                        "caching them after the first pass. The cache is ~40x less compute for "
                        "a few GB of RAM and the SAME maps (the backbone never trains) — only "
                        "disable it if the box is RAM-starved.")
    p.add_argument("--waypoint-weight", type=float, default=0.0,
                   help="v7.2: > 0 enables the WAYPOINT-ABSOLUTE head (predict the metric "
                        "EEF displacement to each plan step) and weights its masked MSE. "
                        "The std_ratio lever: at eval the translation command becomes a "
                        "proportional move toward the predicted position measured against "
                        "live proprio, so magnitude stops depending on the regression's "
                        "amplitude. Needs eef_pos_chunk in the npz; fit the actuation gain "
                        "with `python -m preprocess.fit_waypoint_gain <data-dir>`. Try 1.0.")
    p.add_argument("--planner-drop", type=str, default="",
                   help="comma-separated planner memory groups to ABLATE, e.g. "
                        "'geometry,pred_box_emb'. Valid names: "
                        + ",".join(ChronoQueryPlanner.INPUT_NAMES) + ". The choice is "
                        "baked into the checkpoint's cfg, so eval/bench rebuild the same "
                        "planner automatically. Use `eval.bench --sensitivity` as the "
                        "evidence for what is safe to drop.")
    return p.parse_args(argv)


def reserve_vram(device: torch.device, gb: float) -> None:
    """Claim ``gb`` of VRAM up front and keep it for this process.

    On a shared card the free pool is whatever the neighbours leave, and it
    shrinks: a co-tenant was measured growing 60 -> 83 GB while a run was in
    flight, which is why stage A failed a 32 MB allocation with a 10 GB reserve
    and 119 GB free moments earlier. No batch size or allocator tuning survives
    that, because the memory is gone by the time we ask.

    The fix is to ask FIRST. Allocating a large block and immediately dropping
    the reference frees it to PyTorch's CACHING allocator, which does not return
    it to the driver — so it stays reserved to this process and every later
    allocation is served from it without a driver call. Nothing else on the card
    can take it back.

    The one thing that WOULD give it back is ``empty_cache()``, which is why
    stage A no longer calls it.
    """
    if device.type != "cuda" or gb <= 0:
        return
    free, total = torch.cuda.mem_get_info(device)
    want = int(gb * 1024**3)
    if want > free:
        print(f"VRAM reserve: asked {gb:.0f} GB but only {free/1024**3:.0f} GB is "
              f"free — reserving what is available minus 1 GB.", flush=True)
        want = max(0, free - 1024**3)
    if want <= 0:
        print("VRAM reserve: nothing to reserve", flush=True)
        return
    try:
        block = torch.empty(want, dtype=torch.uint8, device=device)
    except RuntimeError as exc:
        print(f"VRAM reserve: failed ({exc.__class__.__name__}); continuing "
              f"without a reservation", flush=True)
        return
    del block  # -> caching allocator, NOT the driver
    print(f"VRAM reserve: holding {torch.cuda.memory_reserved(device)/1024**3:.1f} GB "
          f"for this process ({free/1024**3:.0f} GB was free)", flush=True)


def cap_vram(device: torch.device, max_gb: float) -> None:
    """Hard-caps this process's GPU memory (cuda/ROCm)."""
    if device.type != "cuda":
        return
    if max_gb <= 0:
        # 0/negative means NO CAP. Without this, frac = 0/total = 0.0 and
        # set_per_process_memory_fraction(0) forbids every allocation — the
        # "disable it" value silently became the most restrictive one.
        print("VRAM cap: disabled (--max-vram-gb <= 0)", flush=True)
        return
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    frac = min(1.0, max_gb / total)
    # set_per_process_memory_fraction requires an explicit device index;
    # torch.device("cuda") has none, so resolve to the current device.
    idx = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(frac, idx)
    print(f"VRAM cap: {max_gb:.0f} GB of {total:.0f} GB total (fraction {frac:.3f})", flush=True)


def preload_buckets(data_dirs, val_frac, seed, device, load_frames: bool = False):
    """Loads every episode into RAM and buckets each split by length T.

    Returns (train_buckets, val_buckets), each a dict[T] -> dict of stacked
    tensors on ``device`` with a leading batch dim: frame_embs [N, T, 512],
    pwm_targets [N, T, 5, 7], text_tokens [N, 3, 512], etc. With
    ``load_frames`` (v7 TQSA), buckets whose EVERY episode has baked
    ``wrist_frames`` also carry them — as uint8 on CPU (a bucket of frames is
    ~50x the embedding payload; the trainer moves one batch at a time).
    """
    sets = [EpisodeDataset(d, load_frames=load_frames) for d in data_dirs]
    index = [(i, j) for i, ds in enumerate(sets) for j in range(len(ds))]
    rng = random.Random(seed)
    rng.shuffle(index)
    n_val = max(1, int(len(index) * val_frac)) if val_frac > 0 else 0
    splits = {"val": index[:n_val], "train": index[n_val:]}

    out = {}
    for name, idx in splits.items():
        # Bucket key = (T, has_frames): frameless (Bridge) and frame-carrying
        # (v7 LIBERO) episodes of the SAME length must never share a bucket —
        # otherwise one Bridge episode strips frames from the whole bucket and
        # the TQSA silently trains on nothing (observed: 0/37 buckets).
        by_key = defaultdict(list)
        for i, j in idx:
            ep = sets[i][j]
            key = (ep["frame_embs"].shape[0], load_frames and "wrist_frames" in ep)
            by_key[key].append(ep)
        buckets = {}
        all_keys = EPISODE_KEYS + OPTIONAL_KEYS  # optional keys are zero-filled by the dataset
        for (T, has_frames), eps in by_key.items():
            b = {k: torch.stack([e[k] for e in eps]).to(device) for k in all_keys}
            if has_frames:
                b["wrist_frames"] = torch.stack([e["wrist_frames"] for e in eps])  # uint8, CPU
            buckets[(T, has_frames)] = b
        out[name] = buckets
    return out["train"], out["val"]


def _boxes(batch, idx, fade, cfg, ablate):
    """Held box evidence at timestep idx for the whole batch (or zeros if ablated)."""
    B = batch["frame_embs"].shape[0]
    if ablate:
        z = batch["frame_embs"].new_zeros(B, cfg.vis_dim)
        z2 = batch["frame_embs"].new_zeros(B, 2)
        return z, z, z2, z2, batch["frame_embs"].new_zeros(B, 2)
    return (batch["source_box_embs"][:, idx], batch["target_box_embs"][:, idx],
            batch["source_centers"][:, idx], batch["target_centers"][:, idx],
            batch["box_weights"][:, idx] * fade)


def _obj_tokens(batch, idx, fade, cfg, ablate):
    """v8 object tokens from the baked evidence, padded to ``cfg.max_objects``.

    The baked ``.npz`` episodes carry TWO roles (source, target), not K
    class-agnostic proposals — the v7 bake ran YOLO-World with
    ``set_classes([source, target])`` and kept one box each. So v8 trains today
    at an effective K of 2, with the remaining ``cfg.max_objects - 2`` slots
    padded at weight 0.0. That is not a degenerate case: weight-0 objects are
    bit-identically inert through both ``EvidenceEncoder`` and
    ``RelationalHead``, so the padding costs nothing but also buys nothing.

    The full data-rich path needs a re-bake through ``preprocess/`` with
    ``microvla/perception/text_region.py``, which is what supplies K
    class-agnostic proposals with text-space embeddings. Until then the
    relational head reasons over two objects, which is exactly enough to
    express the source->target relation every task in the corpus is built on,
    and not enough to test distractor rejection.

    Returns ``(obj_emb [B,K,vis_dim], obj_center [B,K,2], obj_weight [B,K])``.
    """
    sbe, tbe, sc, tc, bw = _boxes(batch, idx, fade, cfg, ablate)
    B, K = sbe.shape[0], cfg.max_objects
    obj = sbe.new_zeros(B, K, cfg.vis_dim)
    ctr = sbe.new_zeros(B, K, 2)
    w = sbe.new_zeros(B, K)
    obj[:, 0], obj[:, 1] = sbe, tbe
    ctr[:, 0], ctr[:, 1] = sc, tc
    w[:, :2] = bw
    return obj, ctr, w


def _relational(relational, next_emb, batch, t, box_idx, box_fade, cfg,
                last_action=None):
    """Relational tokens for the planner, or None on the v7 stack.

    The v8 ordering change lives here: this runs on the TRM's PREDICTED latent,
    so object-object reasoning is conditioned on the same state the planner is
    about to decode, rather than on a separate pre-TRM summary.

    ``box_idx``/``box_fade`` mirror the caller's own evidence choice exactly —
    on a dream step that is t-1 at the mid-dream fade, on a real step t at 1.0.
    They are passed rather than assumed because handing the relational head
    fresher evidence than the rest of the step sees would leak the future into
    the one module whose job is to reason about the present.
    """
    if relational is None:
        return None
    obj, ctr, w = objects_from_batch(batch, box_idx, box_fade, cfg)
    # The v8 relational head carries its OWN action token, and it is the
    # planner's dominant input. --action-token-sampling originally fed the
    # model's own action to FUSION only, leaving this one teacher-forced --
    # so the exposure bias of paper.md 4v survived in the module that replaced
    # fusion. Same fix-one-side-of-the-pair shape as every other defect here.
    if last_action is None:
        last_action = (batch["pwm_targets"][:, t - 1, 0] if t > 0
                       else batch["pwm_targets"].new_zeros(
                           obj.shape[0], batch["pwm_targets"].shape[-1]))
    return relational(next_emb, obj, ctr, w, batch["text_tokens"],
                      last_action=last_action)


def real_paths_v8(batch, evidence, hrm, cfg, ablate):
    """v8 counterpart of :func:`real_paths`: evidence port + HRM state per step.

    Returns ``ev_all[t] -> [B, fused_rows, fused_cols]`` (the TRM's unchanged
    evidence port) and ``state_all[t] -> [B, hrm_dim]``, lists over t, with grad.

    Every timestep here is a REAL tick (``is_real=True``): this is the
    data-rate path, one entry per baked 2 Hz frame. The fast/slow split only
    becomes observable in the 30 Hz deployment loop, where dream ticks step the
    fast module alone.
    """
    B, T = batch["frame_embs"].shape[:2]
    text = batch["text_tokens"]
    hrm.reset()
    ev_all, state_all = [], []
    zeros_act = batch["pwm_targets"].new_zeros(B, batch["pwm_targets"].shape[-1])
    for t in range(T):
        last_action = batch["pwm_targets"][:, t - 1, 0] if t > 0 else zeros_act
        obj, ctr, w = _obj_tokens(batch, t, 1.0, cfg, ablate)
        ev_all.append(evidence(obj, ctr, w, batch["frame_embs"][:, t], text,
                               last_action=last_action))
        state_all.append(hrm(batch["frame_embs"][:, t], is_real=True).state)
    return ev_all, state_all


def real_paths(batch, fusion, drift, cfg, ablate):
    """Batched grounded fused matrices + drift codes for every timestep.

    Returns fused_all[t] -> [B, 32, 5] and delta_all[t] -> [B, 256], lists over
    t, with grad. Drift runs sequentially over T (batched over episodes).
    """
    B, T = batch["frame_embs"].shape[:2]
    text = batch["text_tokens"]
    drift.reset()
    fused_all, delta_all = [], []
    zeros_act = batch["pwm_targets"].new_zeros(B, batch["pwm_targets"].shape[-1])
    for t in range(T):
        last_action = batch["pwm_targets"][:, t - 1, 0] if t > 0 else zeros_act
        sbe, tbe, sc, tc, bw = _boxes(batch, t, 1.0, cfg, ablate)
        fused_all.append(fusion(text, batch["frame_embs"][:, t], sbe, tbe, sc, tc,
                                box_weight=bw, last_action=last_action))
        delta_all.append(drift(batch["frame_embs"][:, t]))
    return fused_all, delta_all


def rollout(batch, t, fused_t, delta_t, fusion, trm, cfg, H, gamma, ablate, box_w=0.0,
            ckpt_rollout=False):
    """Batched H-step data-rate rollout loss (mean over batch + discounted steps).

    ``box_w > 0`` adds the v4 TRM box-prediction term: at every rollout step the
    TRM also predicts the next-tick SOURCE box embedding, supervised (same
    discounted spec_loss) against the recorded ``source_box_embs[t+k]``. Kept
    OUT of the val objective (``box_w=0`` there) so val stays frame-only and
    directly comparable to the frame-only persistence baseline.
    """
    text = batch["text_tokens"]
    frames = batch["frame_embs"]
    T = frames.shape[1]
    latent = frames[:, t]
    ctx = [latent]
    fused_k, delta_k = fused_t, delta_t
    loss = torch.zeros((), device=frames.device)
    box_loss = torch.zeros((), device=frames.device)
    wsum = 0.0
    want_box = box_w > 0.0
    for k in range(1, H + 1):
        context = torch.stack(ctx[-cfg.context_window:], dim=1)  # [B, K, 512]
        if ckpt_rollout and torch.is_grad_enabled():
            # Activation memory here is the whole problem: the graph is H TRM
            # forwards deep, each a d=1024 recursive refinement, so peak grows
            # linearly in H. On a SHARED MI300X VF whose free pool is whatever
            # the neighbours leave (observed: one co-tenant grew 60 -> 83 GB
            # mid-run), a peak that grows with H eventually loses the race no
            # matter how small the batch is. Checkpointing stores only the
            # inputs and recomputes the forward during backward: ~H x less
            # activation memory for ~1 extra forward per step.
            out = torch.utils.checkpoint.checkpoint(
                lambda f, d, l, c: trm(f, d, l, context=c, return_box=want_box),
                fused_k, delta_k, latent, context, use_reentrant=False)
        else:
            out = trm(fused_k, delta_k, latent, context=context, return_box=want_box)
        pred, box = out if want_box else (out, None)
        w = gamma ** (k - 1)
        loss = loss + w * spec_loss(pred, frames[:, t + k])
        if want_box:
            box_loss = box_loss + w * spec_loss(box, batch["source_box_embs"][:, t + k])
        wsum += w
        if k == H:
            break
        latent = standardize(pred)
        ctx.append(latent.detach())
        sbe, tbe, sc, tc, bw = _boxes(batch, t, cfg.staleness_decay ** k, cfg, ablate)
        act_idx = min(t + k, T - 1)
        last_action = batch["pwm_targets"][:, act_idx, 0]
        fused_k = fusion(text, latent, sbe, tbe, sc, tc, box_weight=bw, last_action=last_action)
    total = loss / wsum
    if want_box:
        total = total + box_w * (box_loss / wsum)
    return total


def persistence(batch, t, H, gamma) -> float:
    frames = batch["frame_embs"]
    cur = frames[:, t]
    loss, wsum = 0.0, 0.0
    for k in range(1, H + 1):
        w = gamma ** (k - 1)
        loss += w * float(spec_loss(cur, frames[:, t + k]))
        wsum += w
    return loss / wsum


def iter_batches(buckets, H, batch_size, rng, need=1):
    """Yields (T, batch) over (T, has_frames)-buckets with T >= H+need."""
    order = list(buckets.keys())
    rng.shuffle(order)
    for key in order:
        T = key[0] if isinstance(key, tuple) else key
        if T < H + need:
            continue
        b = buckets[key]
        N = b["frame_embs"].shape[0]
        perm = list(range(N)); rng.shuffle(perm)
        for s in range(0, N, batch_size):
            sel = perm[s:s + batch_size]
            yield T, {k: v[sel] for k, v in b.items()}


@torch.no_grad()
def evaluate(val_buckets, fusion, drift, trm, cfg, H, gamma, ablate, batch_size, rng):
    fusion.eval(); drift.eval(); trm.eval()
    vs, ps, n = 0.0, 0.0, 0
    for T, batch in iter_batches(val_buckets, H, batch_size, rng, need=1):
        fused_all, delta_all = real_paths(batch, fusion, drift, cfg, ablate)
        for t in range(0, T - H, max(1, (T - H) // 4)):
            vs += float(rollout(batch, t, fused_all[t], delta_all[t], fusion, trm, cfg, H, gamma, ablate))
            ps += persistence(batch, t, H, gamma)
            n += 1
    return vs / max(n, 1), ps / max(n, 1)


def stage_a(args, cfg, train_b, val_b, fusion, drift, trm, device):
    params = [*fusion.parameters(), *drift.parameters(), *trm.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    # Halve the LR when val plateaus (at max horizon), so a high initial LR
    # gets fast early progress then settles to a finer minimum instead of
    # oscillating. Pairs with early stopping: LR reduces before patience trips.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-5)
    rng = random.Random(args.seed)
    ckpt = _tagged_name("full_stageA.pt", args.tag)
    best, stale = float("inf"), 0
    start_epoch = 1

    # --resume-stage-a: pick the run back up from its last saved epoch instead of
    # starting over. This exists because the training box kills processes
    # intermittently from outside the container — SIGTERM, no dmesg entry, no
    # visible reaper, at 500 MB or 60 GB alike. Without resume a kill at epoch 19
    # costs the whole ~1 h stage; with it, one epoch.
    # Written EVERY epoch, separate from `ckpt` so the best-checkpoint semantics
    # are untouched: `ckpt` stays the best weights, this carries the schedule.
    resume_name = ckpt.replace(".pt", ".resume.pt")
    if args.resume_stage_a:
        path = Path(args.checkpoint_dir, resume_name)
        if not path.exists():
            print(f"[stage A] --resume-stage-a: no {path}; starting fresh", flush=True)
        else:
            st = torch.load(path, map_location=device, weights_only=True)
            fusion.load_state_dict(st["fusion"])
            drift.load_state_dict(st["drift"])
            trm.load_state_dict(st["trm"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            best = float(st["best_val"])
            stale = int(st["stale"])
            start_epoch = int(st["epoch"]) + 1
            rng = random.Random(args.seed + start_epoch)  # don't replay one epoch's order
            print(f"[stage A] resumed at epoch {start_epoch} (best val {best:.4f}, "
                  f"stale {stale}/{args.patience}, lr {opt.param_groups[0]['lr']:.1e})",
                  flush=True)

    prev_H = None
    for epoch in range(start_epoch, args.stage_a_epochs + 1):
        H = _scheduled_horizon(epoch, args.warmup_epochs, args.max_horizon)
        at_max = H >= args.max_horizon
        # NOTE: deliberately NO empty_cache() here. An earlier version dropped
        # the cache at each horizon bump on a fragmentation theory that the
        # reserved-memory instrumentation later disproved (reserve was 10.2 GB
        # while the device had 119 GB free). On a SHARED card, empty_cache()
        # hands our pool back to the driver — i.e. straight to the co-tenant
        # that grew 60 -> 83 GB mid-run — and we cannot get it back. Holding the
        # reservation is the entire strategy; see reserve_vram.
        prev_H = H
        fusion.train(); drift.train(); trm.train()
        run, nb, t0 = 0.0, 0, time.time()
        last_beat = t0
        for T, batch in iter_batches(train_b, H, args.batch_size, rng, need=1):
            fused_all, delta_all = real_paths(batch, fusion, drift, cfg, args.ablate_grounding)
            ts = list(range(T - H)); rng.shuffle(ts); ts = ts[: args.segments_per_episode]
            if not ts:
                continue
            opt.zero_grad()
            loss = torch.zeros((), device=device)
            for t in ts:
                # Drift-dropout (interpretability fix): occasionally hide the
                # drift code so the TRM must model scene content, not just
                # "keep moving the way we've been moving".
                delta_in = (torch.zeros_like(delta_all[t])
                            if rng.random() < args.drift_dropout else delta_all[t])
                loss = loss + rollout(batch, t, fused_all[t], delta_in, fusion, trm,
                                      cfg, H, args.gamma, args.ablate_grounding,
                                      box_w=args.box_loss_weight,
                                      ckpt_rollout=args.ckpt_rollout)
            loss = loss / len(ts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run += float(loss.detach()); nb += 1
            if time.time() - last_beat >= _HEARTBEAT_SEC:
                last_beat = time.time()
                print(f"[stage A] epoch {epoch} .. {nb} batches | "
                      f"loss {run/max(nb,1):.4f} | {last_beat-t0:.0f}s", flush=True)

        val, pers = evaluate(val_b, fusion, drift, trm, cfg, H, args.gamma,
                             args.ablate_grounding, args.batch_size, rng)
        verdict = "BEATS persistence" if val < pers else "not yet below persistence"
        tag = ""
        if not at_max:
            save(args, cfg, ckpt, fusion=fusion, drift=drift, trm=trm)
        elif val < best - args.min_delta:
            best, stale = val, 0
            save(args, cfg, ckpt, fusion=fusion, drift=drift, trm=trm); tag = " *best*"
        else:
            stale += 1; tag = f" (no improve {stale}/{args.patience})"
        if at_max:
            sched.step(val)  # only at fixed horizon (val rises during warmup by design)
        lr_now = opt.param_groups[0]["lr"]
        # RESERVED, not just allocated. peakVRAM reported only live tensors, so a
        # run showing "9.3GB" could be holding 100+ GB from the driver in cached
        # blocks — which is what "0 bytes is free" means while our tensors are
        # tiny. The gap between the two IS the fragmentation.
        peak = (f" | VRAM alloc {torch.cuda.max_memory_allocated(device)/1024**3:.1f}"
                f"/reserved {torch.cuda.max_memory_reserved(device)/1024**3:.1f}GB"
                f" free {torch.cuda.mem_get_info(device)[0]/1024**3:.0f}GB"
                if device.type == "cuda" else "")
        print(f"[stage A] epoch {epoch} | H={H} | lr {lr_now:.1e} | train {run/max(nb,1):.4f} "
              f"| val {val:.4f} vs persistence {pers:.4f} ({verdict}){tag} "
              f"| {time.time()-t0:.0f}s{peak}", flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        save(args, cfg, resume_name, fusion=fusion, drift=drift, trm=trm,
             opt=opt, sched=sched, epoch=epoch, best_val=best, stale=stale)
        if at_max and args.patience > 0 and stale >= args.patience:
            print(f"[stage A] early stop at H={args.max_horizon}, best val {best:.4f}", flush=True)
            break

    if Path(args.checkpoint_dir, ckpt).exists():
        st = torch.load(Path(args.checkpoint_dir, ckpt), map_location=device, weights_only=True)
        fusion.load_state_dict(st["fusion"]); drift.load_state_dict(st["drift"]); trm.load_state_dict(st["trm"])


def _pre_grasp(batch, weight: float):
    """Mean-1 per-timestep pre-grasp weights, or None when the feature is off."""
    if weight == 1.0:
        return None
    w, _t_close, _usable = pre_grasp_weights(batch["pwm_targets"], weight=weight)
    return w


def _wp_targets(batch, cfg):
    """Waypoint supervision for a bucket batch, native- or sampled-spaced."""
    if cfg.waypoint_long:
        return long_horizon_targets(batch["eef_pos_chunk"], cfg.plan_steps,
                                    cfg.waypoint_range)
    return waypoint_targets(batch["eef_pos_chunk"], cfg.plan_steps, cfg.waypoint_range)


def _fade_weights(rates: dict, batch: int, device) -> dict:
    """Per-sample evidence-fade weights for the planner's memory groups.

    One ``[batch, 1, 1]`` weight per group with a nonzero rate, drawn as
    ``1 - drop * (1 - u)`` with ``drop ~ Bernoulli(rate)`` and ``u ~ U[0, 1)`` —
    the same continuum ``SlotResonanceFusion`` uses for box evidence, so a
    withheld input degrades the way stale evidence does at deployment instead of
    vanishing. Groups at rate 0 are omitted, and an empty dict is bit-identical
    to not fading at all.

    Args:
        rates: ``{group_name: probability}``; zero rates are skipped.
        batch: Batch size.
        device: Device for the drawn tensors.

    Returns:
        ``{group_name: [batch, 1, 1]}``, empty when every rate is 0.
    """
    out = {}
    for name, rate in rates.items():
        if rate <= 0:
            continue
        drop = torch.bernoulli(torch.full((batch, 1, 1), float(rate), device=device))
        out[name] = 1.0 - drop * (1.0 - torch.rand(batch, 1, 1, device=device))
    return out


def _batch_spatial(batch, t, tqsa, backbone, device):
    """TQSA outputs for timestep t of a bucket batch, or None when unavailable.

    Prefers PRECOMPUTED backbone maps (see :func:`precompute_spatial_maps`);
    falls back to running the frozen backbone live on the baked uint8 RGB
    frames (converted to the detector's BGR). Only the map hits the GPU; the
    TRAINABLE adapter is what consumes it, conditioned on the episode's text.
    """
    if tqsa is None:
        return None
    if "spatial_maps" in batch:
        maps = batch["spatial_maps"][:, t].to(device=device, dtype=torch.float32)
        return tqsa(maps, batch["text_tokens"])
    if backbone is None or "wrist_frames" not in batch:
        return None
    frames = batch["wrist_frames"][:, t].numpy()          # [B, H, W, 3] RGB uint8
    maps = backbone.feature_maps([f[..., ::-1] for f in frames]).to(device)
    return tqsa(maps, batch["text_tokens"])


def precompute_spatial_maps(buckets, backbone, batch_size, label="", dtype=torch.float16):
    """Runs the FROZEN backbone once over every framed timestep and caches it.

    The backbone never trains, so for a given (episode, timestep) its SPPF map
    is the same on every epoch — yet stage B was recomputing all of them every
    epoch, at a 128->512 px upscale. That is the whole cost of ``--tqsa``
    (measured: ~18 min/epoch vs seconds without it). One pass up front makes
    every subsequent epoch nearly free.

    Not bit-identical, and the two sources of difference are both far below the
    signal: the cache is stored in ``dtype`` (fp16 = ~2e-4 relative, straight
    into the adapter's 1x1 conv + GroupNorm) and the maps are computed in
    fixed-size chunks rather than in the epoch's shuffled batches (~1e-6 from
    batch composition). fp32 buys bit-identity at 2x the RAM.

    Each timestep is moved to the training device on demand — the fp16 slice is
    ~26 MB per (batch, timestep) and lands as ~52 MB of fp32, so a T-step batch
    holds ~0.7 GB of maps in the graph at batch 64.

    Args:
        buckets: ``dict[key] -> bucket dict`` from :func:`preload_buckets`;
            each bucket carrying ``wrist_frames`` gains ``spatial_maps``
            ``[N, T, C, Hf, Wf]``.
        backbone: A ``YoloWorldPerception`` (frozen map extractor).
        batch_size: Frames per backbone forward.
        label: Prefix for the progress lines ("train"/"val").
        dtype: Cache dtype.

    Returns:
        Total cache size in bytes (0 when nothing was cached).
    """
    framed = [b for b in buckets.values() if "wrist_frames" in b]
    if not framed:
        return 0
    total = sum(int(b["wrist_frames"].shape[0]) * int(b["wrist_frames"].shape[1])
                for b in framed)
    print(f"[spatial cache] {label}: {total} frames through the frozen backbone "
          f"(once, not once per epoch)", flush=True)

    done, t0, last_beat = 0, time.time(), time.time()
    nbytes = 0
    for b in framed:
        frames = b["wrist_frames"]                    # [N, T, H, W, 3] uint8 CPU
        N, T = int(frames.shape[0]), int(frames.shape[1])
        maps = None
        for t in range(T):
            for s in range(0, N, batch_size):
                chunk = frames[s:s + batch_size, t].numpy()
                m = backbone.feature_maps([f[..., ::-1] for f in chunk])
                # The SPPF hook keeps only the LAST forward's output. If the
                # detector ever splits a list across several internal forwards,
                # m would come back with batch 1 and the assignment below would
                # BROADCAST one frame's features across the whole chunk — no
                # exception, just silently wrong training data. The live path
                # fails loudly on this (the adapter rejects the batch mismatch);
                # only the precompute could swallow it.
                if m.shape[0] != chunk.shape[0]:
                    raise RuntimeError(
                        f"backbone returned {m.shape[0]} maps for {chunk.shape[0]} "
                        f"frames — the SPPF hook saw more than one forward per "
                        f"call, so the cache cannot be trusted. Lower "
                        f"--batch-size or use --no-cache-spatial."
                    )
                if maps is None:
                    if nbytes == 0:   # first map of the run: project the footprint
                        per = m[0].numel() * torch.empty((), dtype=dtype).element_size()
                        print(f"[spatial cache] {label}: map {tuple(m.shape[1:])}, "
                              f"{per/1024:.0f} KB/frame -> ~{per*total/1024**3:.1f} GB RAM",
                              flush=True)
                    maps = torch.empty((N, T, *m.shape[1:]), dtype=dtype)
                maps[s:s + batch_size, t] = m.to("cpu", dtype)
                done += chunk.shape[0]
                if time.time() - last_beat >= _HEARTBEAT_SEC:
                    last_beat = time.time()
                    rate = done / max(last_beat - t0, 1e-6)
                    print(f"[spatial cache] {label}: {done}/{total} frames "
                          f"({rate:.0f}/s, ~{(total-done)/max(rate,1e-6):.0f}s left)",
                          flush=True)
        b["spatial_maps"] = maps
        # The frames have served their only purpose. Dropping them frees ~1 GB
        # and, more usefully, stops iter_batches gathering 40 MB of pixels into
        # every batch that nothing reads any more. Safe: the (T, has_frames)
        # bucket key was fixed at load time and _batch_spatial checks
        # spatial_maps first.
        del b["wrist_frames"]
        nbytes += maps.numel() * maps.element_size()
    print(f"[spatial cache] {label}: done in {time.time()-t0:.0f}s, "
          f"{nbytes/1024**3:.1f} GB resident ({str(dtype).split('.')[-1]})", flush=True)
    return nbytes


@torch.no_grad()
def _stage_b_val(args, cfg, val_b, fusion, drift, trm, planner, device,
                 tqsa, backbone, rng, relational=None):
    """Clean stage-B val loss: real path, NO input-dropout, NO dream steps.

    Matches the deployment forward (planner sees every input), so the number
    tracks generalization of the actual policy, not the regularized training
    objective.

    Returns:
        ``(bc, grip_acc, wp)`` — the BC term and the waypoint term SEPARATELY,
        because folding them into one number makes a ``--waypoint-weight`` run
        incomparable to every other arm and gives no way to decompose it after
        the fact. Callers sum them for early stopping and print both.
    """
    planner.eval()
    if tqsa is not None:
        tqsa.eval()
    # The relational head was left in TRAIN mode, so its modality_dropout stayed
    # ACTIVE during the "clean" validation pass -- the number that selects the
    # best checkpoint was scored with random evidence withheld, and differently
    # on every epoch. Restored to train() by the caller after each validation.
    if relational is not None:
        relational.eval()
    tot = ga = wp_tot = 0.0
    nb = 0
    for T, batch in iter_batches(val_b, 1, args.batch_size, rng, need=1):
        fused_all, delta_all = real_paths(batch, fusion, drift, cfg, args.ablate_grounding)
        preds, grips, wps = [], [], []
        for t in range(T):
            cur = batch["frame_embs"][:, t]
            wm = trm.forward_full(fused_all[t], delta_all[t], cur)
            sbe, tbe, sc, tc, bw = _boxes(batch, t, 1.0, cfg, args.ablate_grounding)
            geom = torch.cat([sc, tc, bw], dim=-1)
            spatial = _batch_spatial(batch, t, tqsa, backbone, device)
            rel = _relational(relational, wm["next_emb"], batch, t, t, 1.0, cfg)
            plan, grip, wp = planner(wm["next_emb"], current_emb=cur, state_delta=delta_all[t],
                                     fused=fused_all[t], pred_box_emb=wm["next_box"],
                                     geometry=geom, proprio=batch["proprio"][:, t],
                                     spatial=spatial, wm_msg=wm["msg"],
                                         wm_latent=wm.get("latent"), relational=rel,
                                         return_wp=True)
            preds.append(plan); grips.append(grip)
            if wp is not None:
                wps.append(wp)
        P = torch.stack(preds, 1).reshape(-1, cfg.plan_steps, cfg.num_servos)
        G = torch.stack(grips, 1).reshape(-1, cfg.plan_steps)
        Y = batch["pwm_targets"].reshape(-1, cfg.plan_steps, cfg.num_servos)
        tot += float(split_planner_loss(P, G, Y, smooth_weight=args.smooth_weight,
                                        row0_weight=args.row0_weight))
        if wps:
            wp_t, row_mask = _wp_targets(batch, cfg)
            wp_tot += float(waypoint_loss(
                torch.stack(wps, dim=1), wp_t, row_mask, valid=batch["proprio"][..., -1]))
        ga += float(((G > 0) == (Y[..., -1] > 0)).float().mean())
        nb += 1
    planner.train()
    if tqsa is not None:
        tqsa.train()
    return tot / max(nb, 1), ga / max(nb, 1), wp_tot / max(nb, 1)


def stage_b(args, cfg, train_b, val_b, fusion, drift, trm, planner, device,
            tqsa=None, backbone=None, relational=None):
    if relational is not None:
        relational.train()
    for m in (fusion, drift):
        for p in m.parameters():
            p.requires_grad_(False)
    # ...except the HRM's learned control law. It converts a predicted
    # displacement into an emitted command, so it belongs to the POLICY, not the
    # world model — and freezing it with the rest left gain_head at exactly its
    # zero init in the first v8 checkpoint, with log_gain_base still the
    # hand-fitted prior. Nothing had ever given it a gradient. Magnitude is what
    # paper.md 4p measures as the barrier to task success, and this is the one
    # module that owns it.
    gain_params = [p for n, p in drift.named_parameters()
                   if "gain_head" in n or "log_gain_base" in n]
    for p in gain_params:
        p.requires_grad_(True)
    if gain_params:
        print(f"[stage B] HRM control law TRAINABLE: "
              f"{sum(p.numel() for p in gain_params):,} params", flush=True)
    # TRM freeze policy (v7.1): core frozen, msg_head TRAINABLE — the planner's
    # gradient shapes the 32-d belief message while the world model stays
    # provably intact. --unfreeze-trm trains the whole TRM at 0.1x LR with a
    # world-model auxiliary rollout loss so frame prediction cannot collapse.
    unfreeze = bool(getattr(args, "unfreeze_trm", False))
    for name, p in trm.named_parameters():
        p.requires_grad_(unfreeze or name.startswith("msg_head"))
    params = list(planner.parameters()) + (list(tqsa.parameters()) if tqsa is not None else [])
    if relational is not None:
        # v8: the relational head is POLICY, not world model — it consumes the
        # frozen TRM's prediction and feeds the planner. Omitting it here would
        # leave it at init while everything else trained, which looks exactly
        # like "the relational head does not help".
        params += list(relational.parameters())
    params += gain_params
    # TRAINING-ONLY head: shapes the planner during stage B and is then
    # discarded. Deliberately NOT added to the deployed stack and not counted
    # against cfg.trainable_param_budget, which governs what ships.
    # Symmetric half-span per action dim, so the actuation loss can compare
    # like with like. None when no norm_stats.json is found next to the data,
    # in which case the actuation term is skipped rather than run in mixed units.
    act_scale = None
    for _d in (args.data_dir if isinstance(args.data_dir, (list, tuple)) else [args.data_dir]):
        _ns = Path(_d) / "norm_stats.json"
        if _ns.exists():
            act_scale = torch.as_tensor(
                json.loads(_ns.read_text())["q_high"], dtype=torch.float32,
                device=device).clamp_min(1e-6)
            print(f"[stage B] action scale for the actuation loss: "
                  f"{[round(float(v), 4) for v in act_scale[:3]]} (from {_ns})",
                  flush=True)
            break
    if act_scale is None and args.actuation_weight > 0:
        print("[stage B] WARNING: no norm_stats.json found; the actuation loss "
              "would compare raw-unit commands against normalized targets, so it "
              "is DISABLED for this run.", flush=True)
        args.actuation_weight = 0.0
    # The gain the DEPLOYED actuator uses, so the actuation loss optimizes the
    # quantity that actually runs.
    fitted_gain = None
    for _d in (args.data_dir if isinstance(args.data_dir, (list, tuple)) else [args.data_dir]):
        _ws = Path(_d) / "waypoint_stats.json"
        if _ws.exists():
            fitted_gain = torch.as_tensor(
                json.loads(_ws.read_text())["gain"], dtype=torch.float32,
                device=device).clamp_min(1e-6)
            print(f"[stage B] actuation loss uses the FITTED gain "
                  f"{[round(float(v), 5) for v in fitted_gain[:3]]} (from {_ws}) "
                  f"-- the same one eval/policy.py deploys", flush=True)
            break
    critic = None
    if args.critic_weight > 0:
        critic = ProgressCritic(cfg).to(device)
        critic.train()
        params += list(critic.parameters())
        print(f"[stage B] progress critic ON: critic_w {args.critic_weight} "
              f"progress_w {args.progress_weight} dream_w {args.dream_weight} "
              f"H {args.dream_horizon} | variance_w {args.variance_weight}",
              flush=True)
    elif args.progress_weight > 0 or args.dream_weight > 0:
        # Silently doing nothing is the failure mode this project keeps paying
        # for; refuse instead.
        raise SystemExit(
            "--progress-weight/--dream-weight require --critic-weight > 0, "
            "otherwise the critic is never fit and the actor term maximizes an "
            "untrained network.")
    groups = [{"params": params, "lr": args.lr}]
    trm_trainable = [p for p in trm.parameters() if p.requires_grad]
    if trm_trainable:
        groups.append({"params": trm_trainable,
                       "lr": args.lr * (0.1 if unfreeze else 1.0)})
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    rng = random.Random(args.seed + 1)
    ckpt = _tagged_name("full_stageB.pt", args.tag)
    best_val, stale = float("inf"), 0

    # Mid-dream evidence fade for --dream-frac steps: at deployment the planner
    # runs on dream features for 14 of every 15 ticks; use the fade of a
    # mid-rollout dream tick as the representative training regime.
    period_fade = cfg.staleness_decay ** max(1, cfg.dream_ticks_per_real // 2)

    if args.stage_b_patience > 0:
        # Logged because the selection metric decides how long the run lives, and
        # run length turned out to predict every bench metric.
        print(f"[stage B] select on val {args.stage_b_select} | patience "
              f"{args.stage_b_patience} | min epochs {args.stage_b_min_epochs} "
              f"| budget {args.stage_b_epochs}", flush=True)

    for epoch in range(1, args.stage_b_epochs + 1):
        planner.train(); fusion.eval(); drift.eval(); trm.eval()
        if tqsa is not None:
            tqsa.train()
        run = 0.0; grip_acc = 0.0; nb = 0; t0 = last_beat = time.time()
        for T, batch in iter_batches(train_b, 1, args.batch_size, rng, need=1):
            # NOT under no_grad: every parameter in fusion/drift is frozen
            # except the HRM's gain head, and that head needs a graph or it gets
            # no gradient at all — which is exactly what happened in the first
            # v8 checkpoint, where gain_head stayed bit-for-bit at its zero init
            # through a full training run. Frozen params accumulate nothing
            # regardless, so the only cost is the graph itself.
            fused_all, delta_all = real_paths(batch, fusion, drift, cfg, args.ablate_grounding)
            fused_all = [f.detach() for f in fused_all]
            delta_all = [d.detach() for d in delta_all]
            preds, grips, wps = [], [], []
            value_terms = []          # critic + dreamer actor terms, per step
            prev_action = None      # the policy's own last plan row 0, for
            # --action-token-sampling; None on t=0, where there is no previous
            # action and the trainer's zeros_act convention already matches the
            # loop's reset state.
            for t in range(T):
                # Dream-consistent stage B (v5): with prob --dream-frac, train
                # this step in the DREAM regime the planner actually runs in at
                # 30 Hz — current latent = the (standardized) TRM prediction
                # from t-1, boxes held from t-1 at mid-dream fade, drift held.
                dream = t > 0 and rng.random() < args.dream_frac
                # Scheduled sampling on FUSION'S ACTION TOKEN (v8.1). Fusion's
                # 8th token is "the previously executed action". Training fed it
                # the DEMONSTRATION's previous action at every step while
                # deployment can only feed the POLICY's own, and paper.md 4v
                # attributes essentially the whole closed-loop failure to that
                # one asymmetry: with the token teacher-forced, the deployed
                # stack reproduces the trainer bit-for-bit (fused rel-diff
                # 0.3384 -> 0.0000) and the gripper closes on 47% of steps
                # instead of 13%. Self-feeding closes a loop training never
                # exercised, so a wrong action corrupts the token, which worsens
                # the next action, until the policy sits at a fixed point.
                #
                # With probability p, substitute the model's OWN previous plan
                # row 0 so the token is trained the way it is deployed. Free
                # here: stage B runs fusion under no_grad (it is frozen), so
                # this costs one extra frozen forward on sampled steps.
                self_feed = (prev_action is not None
                             and rng.random() < args.action_token_sampling)
                with torch.no_grad():
                    if dream:
                        prev = batch["frame_embs"][:, t - 1]
                        cur = standardize(trm(fused_all[t - 1], delta_all[t - 1], prev))
                        box_idx, box_fade = t - 1, period_fade
                        sbe, tbe, sc, tc, bw = _boxes(batch, box_idx, box_fade, cfg,
                                                      args.ablate_grounding)
                        act = (prev_action if self_feed
                               else batch["pwm_targets"][:, t - 1, 0])
                        fused_t = fusion(batch["text_tokens"], cur, sbe, tbe, sc, tc,
                                         box_weight=bw, last_action=act)
                        delta_t = delta_all[t - 1]
                    else:
                        cur = batch["frame_embs"][:, t]
                        box_idx, box_fade = t, 1.0
                        sbe, tbe, sc, tc, bw = _boxes(batch, t, 1.0, cfg, args.ablate_grounding)
                        fused_t, delta_t = fused_all[t], delta_all[t]
                        if self_feed:
                            # fused_all[t] was built with the demo's action; redo
                            # this one step with the policy's.
                            fused_t = fusion(batch["text_tokens"], cur, sbe, tbe,
                                             sc, tc, box_weight=bw,
                                             last_action=prev_action)
                # forward_full OUTSIDE no_grad: msg_head (and, under
                # --unfreeze-trm, the core) needs planner-loss gradient; frozen
                # params accumulate none regardless.
                wm = trm.forward_full(fused_t, delta_t, cur)
                next_emb, next_box = wm["next_emb"], wm["next_box"]
                geom = torch.cat([sc, tc, bw], dim=-1)              # [B, 6]
                # v7 TQSA: dream steps use t-1 frames (held-real evidence, same
                # as boxes); real steps use t. Backbone frozen; adapter trains.
                spatial = _batch_spatial(batch, t - 1 if dream else t, tqsa,
                                         backbone, device)
                # v6: arm state at t — fresh even on dream steps (encoders are
                # fast at deployment; only the camera is slow).
                # Input dropout exists to withhold whatever the planner has come
                # to over-rely on, so the paths it is ignoring receive gradient.
                # WHICH inputs dominate has changed: the v7 probe found `fused`
                # 7x dominant, which is what --planner-input-dropout targets. The
                # v7.2 wrist measurement found PHASE dominant instead —
                # state_delta 0.2740 + proprio 0.1904 = 0.464 against vision
                # (geometry + fused) 0.040, a 12:1 ratio — and neither phase
                # input was ever dropped. So the regularizer was withholding the
                # signals we WANT used while never touching the shortcut.
                #
                # --phase-dropout withholds the shortcut. Asymmetric on purpose:
                # a policy that can predict the action from task progress and arm
                # pose alone never needs to find the object, and in a corpus of
                # stereotyped pick-and-place demos with a FIXED basket that
                # shortcut covers most of the action variance.
                # PER-SAMPLE graded fade, not a per-batch coin flip. The previous
                # `rng.random() < p` drew ONE scalar per (batch, timestep), so at
                # batch 64 all 64 episodes were withheld together and a withheld
                # step had no full-input sample anywhere in its gradient. Fading
                # also keeps the token count, which deletion did not.
                fade = _fade_weights(args._drop_rates, cur.shape[0], cur.device)
                rel = _relational(relational, next_emb, batch, t,
                                  box_idx, box_fade, cfg,
                                  last_action=prev_action if self_feed else None)
                plan, grip, wp = planner(next_emb, current_emb=cur, state_delta=delta_t,
                                         fused=fused_t, pred_box_emb=next_box,
                                         geometry=geom, proprio=batch["proprio"][:, t],
                                         spatial=spatial, wm_msg=wm["msg"],
                                         wm_latent=wm.get("latent"), relational=rel,
                                         fade=fade, return_wp=True)
                preds.append(plan); grips.append(grip)
                # What the loop will have in hand next tick.
                prev_action = plan[:, 0].detach()
                if wp is not None:
                    wps.append(wp)
                # ---- task-aligned terms (critic / dreamer) ------------------
                # Fusion's 8th token is the previously executed action, so
                #   plan -> fusion(last_action=plan[:,0]) -> TRM -> latent -> V
                # is differentiable w.r.t. the EMITTED action with no env in the
                # loop. Maximizing V of the imagined next latent asks for actions
                # that ADVANCE THE TASK rather than actions that merely resemble
                # the demonstrator's -- which is the gap paper.md 4p measures
                # (MSE shrinks magnitude; the task tolerates ~5%).
                if critic is not None and (args.progress_weight > 0
                                           or args.dream_weight > 0):
                    lat_i, act_i = cur, plan[:, 0]
                    disc = 1.0
                    for h in range(max(1, args.dream_horizon)):
                        # Evidence is HELD across imagined steps and faded, the
                        # same shared path a dream tick uses -- no new semantics.
                        f_h = fusion(batch["text_tokens"], lat_i, sbe, tbe, sc, tc,
                                     box_weight=bw * (cfg.staleness_decay ** h),
                                     last_action=act_i)
                        lat_i = standardize(trm(f_h, delta_t, lat_i))
                        v = frozen_value(critic, lat_i)
                        # h == 0 is the one-step CRITIC term; h > 0 is the
                        # imagined DREAMER rollout, weighted separately because
                        # it compounds world-model error (4w measures the 1-step
                        # margin over persistence at only +1.7% MSE).
                        w_h = args.progress_weight if h == 0 else args.dream_weight * disc
                        value_terms.append(-w_h * v.mean())
                        if h + 1 < max(1, args.dream_horizon):
                            act_i = planner(lat_i, current_emb=lat_i,
                                            state_delta=delta_t, fused=f_h,
                                            geometry=geom,
                                            proprio=batch["proprio"][:, t])[:, 0]
                            disc *= args.dream_gamma
            preds = torch.stack(preds, dim=1)          # [B, T, 5, 7]
            grips = torch.stack(grips, dim=1)          # [B, T, 5]
            # ---- (1) anti-shrinkage: match the DISPERSION, not just the mean.
            # MSE regression to the conditional mean is why every arm emits
            # std_ratio 0.26-0.42 against a task that passes only near 1.0
            # (paper.md 4p). Penalizing the per-dim std gap attacks that term of
            # the error directly, where MSE alone rewards shrinking it.
            var_term = None
            if args.variance_weight > 0:
                pe = preds[..., : cfg.num_servos - 1].reshape(-1, cfg.num_servos - 1)
                pd = batch["pwm_targets"][..., : cfg.num_servos - 1].reshape(
                    -1, cfg.num_servos - 1)
                var_term = args.variance_weight * (
                    pe.std(dim=0) - pd.std(dim=0)).pow(2).mean()
            target = batch["pwm_targets"]               # [B, T, 5, 7]
            P = preds.reshape(-1, *preds.shape[2:]); G = grips.reshape(-1, grips.shape[-1])
            Y = target.reshape(-1, *target.shape[2:])
            # Pre-grasp emphasis: [B, T] mean-1 per episode, all-ones for
            # episodes whose gripper never closes (bridge) so they are neither
            # up- nor down-weighted.
            step_w = _pre_grasp(batch, args.pre_grasp_weight)
            loss = split_planner_loss(P, G, Y, smooth_weight=args.smooth_weight,
                                      row0_weight=args.row0_weight,
                                      step_weight=None if step_w is None
                                      else step_w.reshape(-1))
            if wps:
                wp_t, row_mask = _wp_targets(batch, cfg)
                # Clamped: this is a 0/1 validity FLAG, and it multiplies
                # squared terms. An out-of-range value silently flips their sign
                # and hands the optimizer an unbounded-below objective — it will
                # take that free lunch and the BC signal goes with it. Observed
                # while debugging with a synthetic corpus whose flag was
                # gaussian: stage-B loss reached -3.06.
                wp_valid = batch["proprio"][..., -1].clamp(0.0, 1.0)
                if step_w is not None:
                    wp_valid = wp_valid * step_w
                loss = loss + args.waypoint_weight * waypoint_loss(
                    torch.stack(wps, dim=1), wp_t, row_mask, valid=wp_valid)

                # ACTUATION loss: supervise the command the robot actually
                # receives, not just the displacement the head predicts.
                #
                # Two things this fixes. The HRM's learned control law had no
                # gradient path at all — gain_head sat at exactly its zero init
                # in the first v8 checkpoint while log_gain_base kept the
                # hand-fitted prior — because nothing downstream of it appeared
                # in any loss. And the deployed magnitude was never trained:
                # paper.md 4p measures LIBERO's passing band at ~[0.95, 1.05] of
                # demo magnitude while every arm emits 0.02-0.56, and the
                # waypoint loss cannot see that because it supervises
                # displacement in metres, upstream of the gain that converts it
                # to a command.
                #
                # cmd = gain_scale * disp * waypoint_range / (gain * steps),
                # which is the actuator's law with (target - eef) substituted by
                # its definition, so this trains exactly what runs.
                if args.actuation_weight > 0 and getattr(drift, "last_gains", None) is not None:
                    # Shapes: wps stacks to [B, T, plan_steps, waypoint_dim];
                    # Y is flattened to [B*T, plan_steps, num_servos]. The
                    # actuator services ROW `waypoint_horizon - 1` (the last
                    # SUPERVISED row — waypoint_targets masks the final one), so
                    # both sides are taken at that row and flattened the same way.
                    # Use the SAME gain deployment divides by. eval/policy.py
                    # builds WaypointActuator from waypoint_stats.json's FITTED
                    # gain and never reads the HRM's learned one, so optimizing
                    # against the learned gain optimizes a number that is not in
                    # the deployed path.
                    #
                    # It also let the term cheat: `g` is 3 numbers shared across
                    # the batch, so the cheapest descent direction for a global
                    # magnitude error is to move `g` rather than the displacement
                    # head -- which is what the head is supposed to learn. That
                    # makes the earlier wp_std_ratio 0.121 -> 1.097 result a
                    # measurement of the gain moving, not the head improving,
                    # and explains why it did not transfer to closed loop.
                    if fitted_gain is not None:
                        g = fitted_gain.view(1, -1)                  # [1, 3], constant
                    else:
                        g = drift.last_gains.clamp_min(1e-6).detach()  # [B, 3], no grad
                    steps = max(1, cfg.waypoint_horizon * cfg.waypoint_row_stride)
                    disp = torch.stack(wps, dim=1)                   # [B,T,rows,3]
                    row = max(0, min(cfg.waypoint_horizon, disp.shape[2]) - 1)
                    d_row = disp[:, :, row, :cfg.waypoint_dim]       # [B, T, 3]
                    cmd = (cfg.waypoint_gain_scale * d_row * cfg.waypoint_range) \
                        / (g.unsqueeze(1) * steps)                   # [B, T, 3]
                    # UNITS: `cmd` is in RAW action units (metres / gain), while Y
                    # holds NORMALIZED targets in [-1, 1]. Comparing them directly
                    # asked for a command 1/s times the demonstrator's, i.e.
                    # +6.7% on x/z and +9.5% on y for the shipped norm_stats
                    # (q_high [0.9375, 0.91339, 0.9375]) -- a magnitude bias baked
                    # into the objective, on a task whose measured tolerance is
                    # ~[0.95, 1.05] (paper.md 4p). The bake is fit_symmetric, so
                    # normalized = raw / q_high exactly.
                    if act_scale is not None:
                        cmd = cmd / act_scale[: cfg.waypoint_dim].view(1, 1, -1)
                    # ROW: `row` indexes the WAYPOINT grid, whose rows are
                    # waypoint_row_stride control steps apart, but Y is the
                    # NATIVE-rate action chunk. The actuator emits a per-step rate
                    # to be executed NOW, so it must be regressed onto the action
                    # executed NOW -- chunk row 0 -- not onto the demo action
                    # row*stride steps in the future.
                    y_row = Y[:, 0, :cfg.waypoint_dim].reshape(
                        cmd.shape[0], cmd.shape[1], cfg.waypoint_dim)
                    m = wp_valid.reshape(cmd.shape[0], cmd.shape[1], 1)
                    loss = loss + args.actuation_weight * (
                        ((cmd - y_row) ** 2 * m).sum()
                        / m.sum().clamp_min(1.0) / cfg.waypoint_dim)
            if unfreeze and T > 4:
                # World-model auxiliary: one random 3-step rollout per batch so
                # BC fine-tuning cannot erode frame prediction (bench verifies).
                # NB: not `t0` — that names the epoch timer this heartbeat and
                # the epoch line both read.
                t_aux = rng.randrange(0, T - 4)
                loss = loss + args.wm_aux_weight * rollout(
                    batch, t_aux, fused_all[t_aux], delta_all[t_aux], fusion, trm, cfg,
                    3, args.gamma, args.ablate_grounding,
                    ckpt_rollout=args.ckpt_rollout)
            if var_term is not None:
                loss = loss + var_term
            # ---- (2)+(3) task-aligned terms ---------------------------------
            # The critic is fit on REAL latents against position-in-episode; the
            # actor terms above already hold its weights fixed (frozen_value),
            # so these two objectives do not fight each other.
            if critic is not None:
                tgt = progress_targets(T, batch["frame_embs"].shape[0], cur.device)
                lat = batch["frame_embs"].reshape(-1, cfg.vis_dim).detach()
                loss = loss + args.critic_weight * (
                    critic(lat) - tgt.reshape(-1)).pow(2).mean()
                if value_terms:
                    loss = loss + torch.stack(value_terms).sum() / len(value_terms)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                run += float(loss)
                # gripper decision accuracy vs the demo (are we learning to close?)
                grip_acc += float(((G > 0) == (Y[..., -1] > 0)).float().mean())
            nb += 1
            # Intra-epoch heartbeat: with --tqsa an epoch runs the frozen
            # backbone over every framed timestep (~23k 512px forwards), so a
            # single epoch can take many minutes with nothing on stdout —
            # indistinguishable from a hang. Time-throttled, so it costs
            # nothing on fast epochs.
            if time.time() - last_beat >= _HEARTBEAT_SEC:
                last_beat = time.time()
                print(f"[stage B] epoch {epoch} .. {nb} batches | "
                      f"loss {run/max(nb,1):.4f} | grip {grip_acc/max(nb,1):.3f} "
                      f"| {last_beat-t0:.0f}s", flush=True)
        extra = {"tqsa": tqsa} if tqsa is not None else {}
        if relational is not None:
            extra["relational"] = relational
        if args.stage_b_patience > 0:
            val_bc, val_ga, val_wp = _stage_b_val(args, cfg, val_b, fusion, drift, trm,
                                                  planner, device, tqsa, backbone, rng,
                                                  relational=relational)
            if relational is not None:
                relational.train()     # _stage_b_val put it in eval()
            # Selection metric. `bc` alone is the only term on a scale shared by
            # every arm; folding in the waypoint term against an ABSOLUTE
            # --min-delta gives long-horizon arms a harsher effective patience
            # (see --stage-b-select).
            val_loss = (val_bc if args.stage_b_select == "bc"
                        else val_bc + args.waypoint_weight * val_wp)
            tag = ""
            if val_loss < best_val - args.min_delta:
                best_val, stale = val_loss, 0
                save(args, cfg, ckpt, fusion=fusion, drift=drift, trm=trm,
                     planner=planner, **extra)
                tag = " *best*"
            else:
                stale += 1
                tag = f" (no improve {stale}/{args.stage_b_patience})"
            # `val bc` is the SAME quantity across every arm; the waypoint term
            # is reported beside it, never folded in, so runs stay comparable.
            wp_txt = f" wp {val_wp:.4f}" if val_wp else ""
            print(f"[stage B] epoch {epoch}/{args.stage_b_epochs} | loss {run/max(nb,1):.4f} "
                  f"| grip_acc {grip_acc/max(nb,1):.3f} | val bc {val_bc:.4f}{wp_txt} "
                  f"grip {val_ga:.3f}{tag} | {time.time()-t0:.0f}s", flush=True)
            if stale >= args.stage_b_patience and epoch >= args.stage_b_min_epochs:
                print(f"[stage B] early stop, best val {best_val:.4f} (checkpoint kept)", flush=True)
                break
        else:
            print(f"[stage B] epoch {epoch}/{args.stage_b_epochs} | loss {run/max(nb,1):.4f} "
                  f"| grip_acc {grip_acc/max(nb,1):.3f} | {time.time()-t0:.0f}s", flush=True)
            save(args, cfg, ckpt, fusion=fusion, drift=drift, trm=trm, planner=planner, **extra)


def main(argv=None) -> None:
    args = parse_args(argv)
    ignore_sigterm()
    cfg = DEFAULT_CONFIG
    if args.planner_drop:
        drop = {s.strip() for s in args.planner_drop.split(",") if s.strip()}
        unknown = sorted(drop - set(ChronoQueryPlanner.INPUT_NAMES))
        if unknown:
            raise SystemExit(f"--planner-drop: unknown input(s) {unknown}; "
                             f"valid: {list(ChronoQueryPlanner.INPUT_NAMES)}")
        kept = tuple(n for n in cfg.planner_inputs if n not in drop)
        if not kept:
            raise SystemExit("--planner-drop would ablate every planner input.")
        cfg = dataclasses.replace(cfg, planner_inputs=kept)
        print(f"planner inputs: {kept} (dropped {sorted(drop)})", flush=True)
    # Resolve per-input withhold rates once: the coarse flags set the defaults,
    # --planner-drop-rate overrides any individual name.
    rates = {n: 0.0 for n in ChronoQueryPlanner.INPUT_NAMES}
    rates["current_emb"] = args.planner_input_dropout
    # The withhold regularizer exists to stop ONE dominant visual group starving
    # the redundant paths. On v8 that group is `relational`, not `fused` —
    # pointing it at `fused` there would silently regularize nothing, since
    # fusion is gone and the name is not in cfg.planner_inputs.
    rates["relational" if args.v8 else "fused"] = args.planner_input_dropout
    rates["state_delta"] = rates["proprio"] = args.phase_dropout
    for item in (s for s in args.planner_drop_rate.split(",") if s.strip()):
        name, _, val = item.partition("=")
        name = name.strip()
        if name not in rates:
            raise SystemExit(f"--planner-drop-rate: unknown input {name!r}; "
                             f"valid: {list(ChronoQueryPlanner.INPUT_NAMES)}")
        rates[name] = float(val)
    args._drop_rates = rates
    active = {k: v for k, v in rates.items() if v > 0}
    if active:
        print(f"stage-B input withhold rates: {active}", flush=True)

    if args.waypoint_weight > 0:
        rng_m = args.waypoint_range if args.waypoint_range is not None else (
            0.5 if args.waypoint_long else cfg.waypoint_range)
        stride = args.waypoint_row_stride if args.waypoint_row_stride is not None else (
            10 if args.waypoint_long else 1)
        cfg = dataclasses.replace(cfg, waypoint_action=True, waypoint_long=args.waypoint_long,
                                  waypoint_range=rng_m, waypoint_row_stride=stride)
        print(f"waypoint head ON (weight {args.waypoint_weight}, range {rng_m} m, "
              f"{'SAMPLED (2 Hz)' if args.waypoint_long else 'native'} spacing, "
              f"row_stride {stride})", flush=True)
        if args.waypoint_long:
            # Long-horizon targets are ~3.5x larger in magnitude. That does NOT
            # imply the loss term is larger — measured on a weight-1.0 run,
            # val bc 0.6924 vs val wp 0.1107, so BC dominated 6:1. The first
            # long-horizon arm still collapsed the action head (std_ratio
            # 0.126 -> 0.022, corr 0.31 -> 0.02) for a reason not yet identified,
            # so watch the split rather than assuming a weight fixes it.
            print("note: --waypoint-long targets are ~3.5x larger in magnitude, but "
                  "the observed val split was BC-dominated 6:1 — read the per-epoch "
                  "`val bc X wp Y` line rather than assuming the wp term dominates. "
                  "The first long-horizon arm collapsed std_ratio 0.126 -> 0.022 for "
                  "reasons not yet attributed (paper.md 4k).", flush=True)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    cap_vram(device, args.max_vram_gb)
    # Claim our pool BEFORE the corpus load and the first forward, while memory
    # is still free. Later is too late on a contended card.
    reserve_vram(device, args.reserve_vram_gb)
    print(f"batched training on {device} | batch {args.batch_size} | data {args.data_dir}", flush=True)

    train_b, val_b = preload_buckets(args.data_dir, args.val_frac, args.seed, device,
                                     load_frames=args.tqsa)
    n_train = sum(v["frame_embs"].shape[0] for v in train_b.values())
    n_val = sum(v["frame_embs"].shape[0] for v in val_b.values())
    print(f"episodes: train {n_train} ({len(train_b)} length-buckets), val {n_val}", flush=True)

    if args.v8:
        # v8 stack: HRM replaces the drift encoder, RelationalHead replaces
        # fusion and runs AFTER the TRM (on the predicted latent, the same state
        # the planner is conditioned on), and EvidenceEncoder feeds the TRM's
        # UNCHANGED [B,32,5] port. See DESIGN.md "v8 plan".
        if "relational" not in cfg.planner_inputs:
            # --planner-drop is applied BEFORE this block, so re-adding
            # "relational" unconditionally silently undid `--planner-drop
            # relational --v8` and the ablation measured nothing. Respect an
            # explicit drop.
            _dropped = {s.strip() for s in args.planner_drop.split(",") if s.strip()}
            _add = () if "relational" in _dropped else ("relational",)
            cfg = dataclasses.replace(
                cfg,
                planner_inputs=tuple(n for n in cfg.planner_inputs
                                     if n not in ("fused", "geometry", "pred_box_emb"))
                + _add,
            )
            if not cfg.planner_inputs:
                raise SystemExit("v8 planner would have no inputs left.")
        fusion = FusionAdapter(cfg).to(device)     # wraps EvidenceEncoder
        drift = DriftAdapter(cfg).to(device)       # wraps HRMBackbone
        relational = RelationalHead(cfg).to(device)
        trm = RecursiveTRM(cfg, d=args.trm_d).to(device)
        planner = ChronoQueryPlanner(cfg).to(device)
        print(f"v8 stack: evidence {count_trainable_params(fusion):,} | "
              f"hrm {count_trainable_params(drift):,} | "
              f"relational {count_trainable_params(relational):,} | "
              f"planner {count_trainable_params(planner):,} | "
              f"inputs {cfg.planner_inputs}", flush=True)
    else:
        relational = None
        fusion = SlotResonanceFusion(cfg).to(device)
        drift = AnchoredDriftEncoder(cfg).to(device)
        trm = RecursiveTRM(cfg, d=args.trm_d).to(device)
        planner = ChronoQueryPlanner(cfg).to(device)

    resume_state: dict = {}
    if args.load_stage_a:
        # Retrain ONLY the policy: load the trained world model, skip stage A.
        st = torch.load(args.load_stage_a, map_location=device, weights_only=True)
        fusion.load_state_dict(st["fusion"]); drift.load_state_dict(st["drift"]); trm.load_state_dict(st["trm"])
        print(f"loaded world model from {args.load_stage_a}; skipping stage A", flush=True)
        args.stage_a_epochs = 0
        if args.resume_stage_b:
            # Continue the POLICY too (planner + tqsa) instead of fresh init —
            # point --load-stage-a at a full_stageB.pt (it carries every key,
            # including the stage-B-trained TRM msg_head inside 'trm').
            if "planner" not in st:
                raise SystemExit("--resume-stage-b needs a stage-B checkpoint "
                                 f"(no 'planner' key in {args.load_stage_a})")
            planner.load_state_dict(st["planner"])
            print("resumed planner from stage-B checkpoint", flush=True)
            resume_state = st

    if args.stage_a_epochs > 0:
        stage_a(args, cfg, train_b, val_b, fusion, drift, trm, device)
    if args.stage_b_epochs > 0:
        tqsa = backbone = None
        if args.tqsa:
            from microvla.perception.spatial_adapter import TextQueriedSpatialAdapter
            from microvla.perception.yolo_world import YoloWorldPerception
            tqsa = TextQueriedSpatialAdapter(cfg).to(device)
            if args.resume_stage_b and "tqsa" in resume_state:
                tqsa.load_state_dict(resume_state["tqsa"])
                print("resumed tqsa from stage-B checkpoint", flush=True)
            backbone = YoloWorldPerception(device=str(device))  # frozen map extractor
            n_frames = sum(1 for b in train_b.values() if "wrist_frames" in b)
            print(f"TQSA stage B: {n_frames}/{len(train_b)} train buckets carry frames", flush=True)
            if args.cache_spatial:
                nb = precompute_spatial_maps(train_b, backbone, args.batch_size, "train")
                if args.stage_b_patience > 0:   # val is only scored when it early-stops
                    nb += precompute_spatial_maps(val_b, backbone, args.batch_size, "val")
                if nb:
                    print(f"[spatial cache] {nb/1024**3:.1f} GB total — every epoch after "
                          f"this one skips the backbone entirely (~{args.stage_b_epochs}x "
                          f"less backbone compute over this run). --no-cache-spatial "
                          f"trades the RAM back.", flush=True)
        stage_b(args, cfg, train_b, val_b, fusion, drift, trm, planner, device,
                tqsa=tqsa, backbone=backbone, relational=relational)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
