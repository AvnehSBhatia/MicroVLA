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


class EpisodeDataset(Dataset):
    """Dataset over a directory of ``.npz`` episode files.

    Each item is one full episode (variable ``T``), returned as a dict of
    float32 torch tensors keyed by ``EPISODE_KEYS``.
    """

    def __init__(self, root: str | Path) -> None:
        """Indexes the episode files.

        Args:
            root: Directory containing ``*.npz`` episode files.

        Raises:
            FileNotFoundError: If the directory has no ``.npz`` files.
        """
        self.root = Path(root)
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

    return {
        "frame_embs": frame_embs,
        "source_box_embs": source_box_embs,
        "target_box_embs": target_box_embs,
        "source_centers": source_centers,
        "target_centers": target_centers,
        "box_weights": box_weights,
        "text_tokens": text_tokens,
        "pwm_targets": pwm_targets,
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
    np.savez(path, **episode)
    return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")
