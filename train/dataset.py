"""Episode dataset and synthetic-episode generator for planner training (v2).

Episodes are ``.npz`` files with the v2 keys defined in DESIGN.md:

    frame_embs        [T, vis_dim]                  — YOLO-World frame embeddings
    source_box_embs    [T, vis_dim]                  — source-box embeddings
    target_box_embs    [T, vis_dim]                  — target-box embeddings
    source_centers     [T, 2]                        — source (cx, cy) in [0, 1]
    target_centers     [T, 2]                        — target (cx, cy) in [0, 1]
    box_weights        [T, 2]                        — per-role evidence weight in [0, 1]
                                                       (detection confidence; 0 = missed)
    text_tokens        [3, text_dim]                 — (command, source, target) CLIP tokens
    pwm_targets        [T, plan_steps, num_servos]   — normalized PWM plans in [-1, 1]

All embeddings are stored in the canonical standardized space (zero mean /
unit std per vector — see microvla/utils/embedding.py), matching what
perception emits at inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from microvla.config import MicroVLAConfig

EPISODE_KEYS: tuple[str, ...] = (
    "frame_embs",
    "source_box_embs",
    "target_box_embs",
    "source_centers",
    "target_centers",
    "box_weights",
    "text_tokens",
    "pwm_targets",
)

#: v6 optional keys — present after `preprocess/patch_proprio.py`; ZERO-FILLED
#: when absent (Bridge, un-patched files, synthetic pre-v6 fixtures) so every
#: consumer can rely on them unconditionally. proprio's last dim is a validity
#: flag, so a zero-fill is self-describing ("no proprio available").
#:   proprio        [T, PROPRIO_DIM=10]  — arm state per sampled frame
#:   eef_pos_chunk  [T, plan_steps, 3]   — absolute EEF xyz at the chunk steps
#:   obj_embs       [T, K, vis_dim]       — v8 class-agnostic proposal embeddings
#:   obj_centers    [T, K, 2]             — their normalized centers
#:   obj_weights    [T, K]                — their confidences, 0.0 on pad slots
#: The obj_* trio is OPTIONAL so a v7 corpus still loads; train_batched falls
#: back to packing the two role slots when they are absent.
#:   has_objects    [1]                   — 1.0 iff obj_* came from disk
#: obj_* is zero-filled when absent so bucket stacking stays uniform across a
#: mixed corpus. Zero-fill is INDISTINGUISHABLE from "detector found nothing on
#: every frame", which is exactly the v7 corpus's real state (paper.md 4n), so
#: `has_objects` records provenance explicitly rather than letting a consumer
#: infer it from all-zero weights.
OPTIONAL_KEYS: tuple[str, ...] = ("proprio", "eef_pos_chunk",
                                  "obj_embs", "obj_centers", "obj_weights",
                                  "has_objects")


class EpisodeDataset(Dataset):
    """Dataset over a directory of ``.npz`` episode files.

    Each item is one full episode (variable ``T``), returned as a dict of
    float32 torch tensors keyed by ``EPISODE_KEYS``.
    """

    def __init__(self, root: str | Path, load_frames: bool = False) -> None:
        """Indexes the episode files.

        Args:
            root: Directory containing ``*.npz`` episode files.
            load_frames: Also load ``wrist_frames`` (v7, uint8 — for TQSA
                training) when a file has them. Off by default: frames are
                ~50x the size of the embedding keys and only the v7 trainer
                needs them. NO zero-fill when absent (unlike OPTIONAL_KEYS) —
                consumers must check for the key.

        Raises:
            FileNotFoundError: If the directory has no ``.npz`` files.
        """
        self.root = Path(root)
        self.load_frames = load_frames
        self.files: list[Path] = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz episode files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Loads one episode.

        Args:
            idx: Episode index.

        Returns:
            Dict with the ``EPISODE_KEYS`` tensors (float32).

        Raises:
            KeyError: If the file is missing a required key.
        """
        with np.load(self.files[idx]) as data:
            episode: dict[str, torch.Tensor] = {}
            for key in EPISODE_KEYS:
                if key not in data:
                    raise KeyError(
                        f"Episode {self.files[idx]} is missing required key {key!r}"
                    )
                episode[key] = torch.as_tensor(data[key], dtype=torch.float32)
            # v6 optional keys: zero-fill when absent (proprio's validity flag
            # is part of the vector, so zeros self-describe as "unavailable").
            T = episode["frame_embs"].shape[0]
            plan_steps = episode["pwm_targets"].shape[1]
            from microvla.config import DEFAULT_CONFIG as _cfg
            k = _cfg.max_objects
            fills = {"proprio": (T, 10), "eef_pos_chunk": (T, plan_steps, 3),
                     "obj_embs": (T, k, _cfg.vis_dim), "obj_centers": (T, k, 2),
                     "obj_weights": (T, k), "has_objects": (1,)}
            for key in OPTIONAL_KEYS:
                if key in data:
                    episode[key] = torch.as_tensor(data[key], dtype=torch.float32)
                else:
                    episode[key] = torch.zeros(fills[key], dtype=torch.float32)
            episode["has_objects"] = torch.tensor(
                [1.0 if "obj_embs" in data else 0.0], dtype=torch.float32)
            if self.load_frames and "wrist_frames" in data:
                # uint8 on purpose: keep RAM/VRAM small; the trainer converts
                # per-batch right before the frozen backbone forward.
                episode["wrist_frames"] = torch.as_tensor(data["wrist_frames"])
        return episode


def make_synthetic_episode(
    T: int,
    cfg: MicroVLAConfig,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Generates a coherent, smooth synthetic episode for smoke training.

    The episode mimics the statistics the real v2 stack would produce: the
    frame embedding drifts slowly from an anchor (a smoothed random walk),
    the source and target box embeddings each track the frame embedding with
    small stable offsets, the source and target box centers follow smooth
    sinusoidal paths inside ``[0, 1]^2`` that converge toward each other over
    the episode (mirroring a "move source to target" task), the 3 text
    tokens are unit-norm random vectors (mimicking CLIP text-tower output),
    and the PWM targets are consecutive windows of one smooth servo
    trajectory — so the plan at time t is the "future" of the plan at time
    t-1, exactly the temporal coherence the smoothness loss expects.

    Args:
        T: Number of timesteps in the episode.
        cfg: Config providing ``vis_dim``, ``text_dim``, ``n_text_tokens``,
            ``plan_steps``, ``num_servos``.
        seed: Local RNG seed (no global seeding).

    Returns:
        Dict of float32 numpy arrays with the ``EPISODE_KEYS`` shapes.
    """
    rng = np.random.default_rng(seed)

    def _standardize(x: np.ndarray) -> np.ndarray:
        """Per-vector zero-mean/unit-std, matching perception's canonical space."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return ((x - mean) / (std + 1e-6)).astype(np.float32)

    # Frame embeddings: anchor + slow random walk (cumulative small steps),
    # standardized like real perception output.
    anchor = rng.normal(0.0, 1.0, size=(cfg.vis_dim,))
    steps = rng.normal(0.0, 0.05, size=(T, cfg.vis_dim))
    frame_embs = _standardize(anchor[None, :] + np.cumsum(steps, axis=0))

    # Source/target box embeddings: track the frame embedding with small,
    # distinct, stable offsets (as if two different objects were detected).
    source_offset = rng.normal(0.0, 0.1, size=(cfg.vis_dim,))
    target_offset = rng.normal(0.0, 0.1, size=(cfg.vis_dim,))
    source_noise = rng.normal(0.0, 0.02, size=(T, cfg.vis_dim))
    target_noise = rng.normal(0.0, 0.02, size=(T, cfg.vis_dim))
    source_box_embs = _standardize(frame_embs + source_offset[None, :] + source_noise)
    target_box_embs = _standardize(frame_embs + target_offset[None, :] + target_noise)

    # Box centers: source and target each wander smoothly but drift toward a
    # shared meeting point over the episode ("move can to ball").
    t = np.arange(T, dtype=np.float64)
    progress = np.clip(t / max(T - 1, 1), 0.0, 1.0)  # 0 -> 1 over the episode
    meet_point = rng.uniform(0.35, 0.65, size=2)

    source_start = rng.uniform(0.1, 0.4, size=2)
    target_start = rng.uniform(0.6, 0.9, size=2)
    wander_phase = rng.uniform(0.0, 2.0 * np.pi, size=2)
    wander_freq = rng.uniform(0.03, 0.08, size=2)
    wander_amp = 0.03

    source_centers = (
        source_start[None, :]
        + (meet_point - source_start)[None, :] * progress[:, None]
        + wander_amp * np.sin(2.0 * np.pi * wander_freq[None, :] * t[:, None] + wander_phase[None, :])
    )
    target_centers = (
        target_start[None, :]
        + (meet_point - target_start)[None, :] * progress[:, None]
        + wander_amp * np.cos(2.0 * np.pi * wander_freq[None, :] * t[:, None] + wander_phase[None, :])
    )
    source_centers = np.clip(source_centers, 0.0, 1.0).astype(np.float32)
    target_centers = np.clip(target_centers, 0.0, 1.0).astype(np.float32)

    # Text tokens: 3 unit-norm random vectors (mimics command/source/target
    # CLIP text-tower embeddings), stacked [3, text_dim].
    raw_tokens = rng.normal(0.0, 1.0, size=(cfg.n_text_tokens, cfg.text_dim))
    norms = np.linalg.norm(raw_tokens, axis=1, keepdims=True)
    text_tokens = (raw_tokens / norms).astype(np.float32)

    # Servo trajectory over T + plan_steps ticks: smooth random walk squashed
    # into [-1, 1]. pwm_targets[t] is the next plan_steps-tick window, so
    # consecutive plans overlap and evolve smoothly.
    traj_len = T + cfg.plan_steps
    servo_steps = rng.normal(0.0, 0.15, size=(traj_len, cfg.num_servos))
    servo_traj = np.tanh(np.cumsum(servo_steps, axis=0))
    pwm_targets = np.stack(
        [servo_traj[i + 1 : i + 1 + cfg.plan_steps] for i in range(T)],
        axis=0,
    ).astype(np.float32)

    # Evidence weights: mostly-confident detections with occasional dips
    # (mimicking flaky small-object confidence on a real detector).
    box_weights = rng.uniform(0.75, 0.95, size=(T, 2)).astype(np.float32)

    # v6 proprio: a smooth synthetic EEF trajectory whose per-step deltas are
    # loosely the servo trajectory's translation dims (mimics a wrist tracking
    # the commanded motion); quat ~ identity + noise; gripper follows servo 7.
    eef_traj = 0.4 + 0.05 * np.cumsum(servo_traj[:, :3], axis=0) / max(T, 1)
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (T, 1))
    quat += rng.normal(0.0, 0.02, size=(T, 4))
    grip_q = 0.5 + 0.5 * servo_traj[:T, -1:]  # [T, 1] in [0, 1] (already scaled O(1))
    proprio = np.concatenate(
        [eef_traj[:T], quat, np.repeat(grip_q, 2, axis=1),
         np.ones((T, 1))],  # valid = 1
        axis=1,
    ).astype(np.float32)
    eef_pos_chunk = np.stack(
        [np.concatenate([eef_traj[i + 1 : i + 1 + cfg.plan_steps],
                         np.repeat(eef_traj[-1:], max(0, (i + 1 + cfg.plan_steps) - traj_len), axis=0)])
         [: cfg.plan_steps]
         for i in range(T)],
        axis=0,
    ).astype(np.float32)

    return {
        "frame_embs": frame_embs,
        "source_box_embs": source_box_embs,
        "target_box_embs": target_box_embs,
        "source_centers": source_centers,
        "target_centers": target_centers,
        "box_weights": box_weights,
        "text_tokens": text_tokens,
        "pwm_targets": pwm_targets,
        "proprio": proprio,
        "eef_pos_chunk": eef_pos_chunk,
    }


def save_episode(path: str | Path, episode: dict[str, np.ndarray]) -> Path:
    """Saves an episode dict to a ``.npz`` file readable by ``EpisodeDataset``.

    Args:
        path: Destination path (``.npz`` appended by numpy if missing).
        episode: Episode dict with the ``EPISODE_KEYS`` arrays.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compressed: the disk budget is hard-capped (see memory/CLAUDE.md — 10 GB
    # total ever); zlib buys ~15-30% on float32 embeddings for negligible
    # load-time cost.
    np.savez_compressed(path, **episode)
    return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")
