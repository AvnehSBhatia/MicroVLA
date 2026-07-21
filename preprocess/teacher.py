"""Teacher-policy distillation hooks: relabel episode actions with a large VLA.

Motivation: MicroVLA's planner is trained by behavior cloning on
``pwm_targets``. Those targets can come from the dataset's human
teleoperation (default) or from a LARGER pretrained VLA acting as a teacher —
knowledge distillation. A teacher gives (a) denoised, consistent action labels,
(b) labels for frames the human data covers badly, and (c) a path to DAgger-style
relabeling later (let the student drive, ask the teacher what it would have done).

The supported teacher here is **TinyVLA** (https://tiny-vla.github.io/) — a
compact VLA family with a diffusion-policy action head that natively emits
*action chunks*, which map 1:1 onto MicroVLA's ``plan_steps``-row plan windows.

Nothing is downloaded by this module. ``TinyVLATeacher`` lazily imports the
TinyVLA repository (which you clone yourself) and loads a checkpoint you
provide; ``MockTeacher`` is a deterministic stand-in for tests and dry-runs.

Usage (from either dataset converter):

    python -m preprocess.libero <root> <out> --teacher tinyvla \\
        --teacher-checkpoint /path/to/tinyvla.ckpt --teacher-repo /path/to/TinyVLA \\
        --teacher-cache ./teacher_cache

The cache directory stores relabeled action arrays per episode id, so the
(expensive) teacher forward pass runs once even though conversion streams the
dataset twice (stats pass + write pass).
"""

from __future__ import annotations

import abc
import hashlib
import logging
from pathlib import Path

import numpy as np

from preprocess.common import SourceEpisode

logger = logging.getLogger(__name__)


class TeacherPolicy(abc.ABC):
    """Relabels a raw episode's actions with a pretrained teacher policy."""

    #: How many native frames each teacher query's predicted chunk covers.
    chunk_len: int = 5

    @abc.abstractmethod
    def predict_chunk(self, frame_rgb: np.ndarray, instruction: str) -> np.ndarray:
        """Predicts one action chunk from a single observation.

        Args:
            frame_rgb: ``[H, W, 3]`` uint8 RGB frame.
            instruction: Natural-language task string.

        Returns:
            ``[chunk_len, 7]`` float32 action chunk in the DATASET's raw
            action convention (Δpos xyz, Δrot rpy, gripper) — normalization
            happens downstream with the same quantile stats as everything
            else.
        """

    def relabel(self, episode: SourceEpisode) -> np.ndarray:
        """Produces a full ``[T_raw, 7]`` relabeled action array.

        Queries the teacher every ``chunk_len`` native frames and tiles each
        predicted chunk over the frames it covers (receding-horizon
        execution, exactly how chunked policies are deployed).

        Args:
            episode: The raw demonstration (frames + instruction used;
                original actions ignored).

        Returns:
            ``[T_raw, 7]`` float32 teacher actions.
        """
        T = len(episode.frames)
        out = np.zeros((T, 7), dtype=np.float32)
        for start in range(0, T, self.chunk_len):
            chunk = np.asarray(
                self.predict_chunk(np.ascontiguousarray(episode.frames[start]), episode.instruction),
                dtype=np.float32,
            )
            end = min(start + self.chunk_len, T)
            out[start:end] = chunk[: end - start]
        return out


class CachedTeacher(TeacherPolicy):
    """Wraps any teacher with an on-disk per-episode cache.

    Conversion streams the dataset twice; this makes the teacher pay for each
    episode once. Cache files are ``<cache_dir>/<episode_id>.npy``.
    """

    def __init__(self, inner: TeacherPolicy, cache_dir: str | Path) -> None:
        self.inner = inner
        self.chunk_len = inner.chunk_len
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def predict_chunk(self, frame_rgb: np.ndarray, instruction: str) -> np.ndarray:
        return self.inner.predict_chunk(frame_rgb, instruction)

    def relabel(self, episode: SourceEpisode) -> np.ndarray:
        path = self.cache_dir / f"{episode.episode_id}.npy"
        if path.exists():
            return np.load(path)
        actions = self.inner.relabel(episode)
        np.save(path, actions)
        return actions


class MockTeacher(TeacherPolicy):
    """Deterministic pseudo-teacher for tests and ``--dry-run`` (no model).

    Each (frame bytes, instruction) pair hashes to a smooth, bounded action
    chunk. No global RNG state is touched.
    """

    def __init__(self, chunk_len: int = 5) -> None:
        self.chunk_len = chunk_len

    def predict_chunk(self, frame_rgb: np.ndarray, instruction: str) -> np.ndarray:
        digest = hashlib.sha256(
            np.ascontiguousarray(frame_rgb).tobytes() + instruction.encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        base = rng.uniform(-0.5, 0.5, size=(1, 7))
        drift = rng.uniform(-0.05, 0.05, size=(self.chunk_len, 7)).cumsum(axis=0)
        return np.clip(base + drift, -1.0, 1.0).astype(np.float32)


class TinyVLATeacher(TeacherPolicy):
    """TinyVLA (https://tiny-vla.github.io/) as a distillation teacher.

    TinyVLA = compact multimodal backbone + diffusion-policy head emitting
    action chunks. This adapter queries it per observation and returns the
    predicted chunk in the dataset's raw action convention.

    Setup (nothing is downloaded automatically):
        1. Clone the TinyVLA repository and install its requirements into a
           Python environment that also has this repo on the path.
        2. Download/obtain a TinyVLA checkpoint finetuned (or pretrained) on
           an action space compatible with 7-DoF (Δpos, Δrot, gripper) —
           their Bridge/franka configurations both qualify.
        3. Pass ``repo_path`` (the clone) and ``checkpoint`` here.

    IMPORTANT — embodiment gap: a teacher trained on a different robot than
    your target rig produces actions in ITS convention. For BridgeData V2 the
    convention matches (WidowX, 7-DoF delta EEF); for your own rig you must
    either finetune the teacher first or retarget its actions. Distill in the
    DATASET's frame, then let MicroVLA's normalizer handle scaling.

    Args:
        checkpoint: Path to the TinyVLA checkpoint.
        repo_path: Path to the cloned TinyVLA repository (added to sys.path).
        device: Torch device string.
        chunk_len: Actions per query; keep equal to the diffusion head's
            trained chunk size (their default is >= 8; ours consumes 5, extra
            rows are simply unused by ``relabel``'s tiling).
    """

    def __init__(self, checkpoint: str | Path, repo_path: str | Path,
                 device: str = "cpu", chunk_len: int = 8) -> None:
        import sys

        self.chunk_len = chunk_len
        self.device = device
        repo = Path(repo_path)
        if not repo.exists():
            raise FileNotFoundError(
                f"TinyVLA repo not found at {repo}. Clone it from "
                "https://tiny-vla.github.io/ (linked GitHub) first — nothing is "
                "downloaded automatically."
            )
        sys.path.insert(0, str(repo))
        try:
            # TinyVLA's public repo exposes policy construction + a
            # per-observation inference call; the exact entrypoints have moved
            # between releases, so resolve them at runtime and fail with a
            # actionable message rather than pinning a fragile import.
            import torch  # noqa: F401

            self._policy = self._load_policy(Path(checkpoint))
        except ImportError as err:  # pragma: no cover - depends on user env
            raise ImportError(
                "TinyVLA imports failed. Install the TinyVLA repo's "
                f"requirements in this environment. Original error: {err}"
            ) from err

    def _load_policy(self, checkpoint: Path):  # pragma: no cover - needs weights
        """Loads the TinyVLA policy; adjust here if their API differs.

        This is the single integration point with the TinyVLA codebase. It
        expects the repo to provide a policy object with an
        ``inference/predict``-style method taking (image, instruction) and
        returning an action chunk; consult their ``eval_*`` scripts for the
        exact call in your checkout and adapt THIS method only.
        """
        raise NotImplementedError(
            "Wire your TinyVLA checkout here: load the policy from "
            f"{checkpoint} following the repo's eval script, and implement "
            "predict_chunk() against it. This adapter deliberately ships as "
            "a documented integration point because TinyVLA's API surface "
            "varies between releases — everything downstream (caching, "
            "relabeling, chunking, normalization) is already wired."
        )

    def predict_chunk(self, frame_rgb: np.ndarray, instruction: str) -> np.ndarray:  # pragma: no cover
        chunk = self._policy.predict(frame_rgb, instruction)
        return np.asarray(chunk, dtype=np.float32)[: self.chunk_len]


def build_teacher(name: str | None, checkpoint: str | None, repo: str | None,
                  cache: str | None, device: str = "cpu") -> TeacherPolicy | None:
    """CLI helper: builds (and optionally caches) the requested teacher.

    Args:
        name: ``None`` (no distillation), ``"mock"``, or ``"tinyvla"``.
        checkpoint: TinyVLA checkpoint path (tinyvla only).
        repo: TinyVLA repo clone path (tinyvla only).
        cache: Optional cache directory for relabeled actions.
        device: Torch device string.

    Returns:
        A ready teacher, or ``None``.
    """
    if name is None:
        return None
    if name == "mock":
        teacher: TeacherPolicy = MockTeacher()
    elif name == "tinyvla":
        if not checkpoint or not repo:
            raise ValueError("--teacher tinyvla requires --teacher-checkpoint and --teacher-repo")
        teacher = TinyVLATeacher(checkpoint, repo, device=device)
    else:
        raise ValueError(f"unknown teacher {name!r} (expected 'mock' or 'tinyvla')")
    return CachedTeacher(teacher, cache) if cache else teacher
