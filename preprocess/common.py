"""Dataset-agnostic preprocessing core: raw robot episodes -> MicroVLA .npz.

Both supported datasets (LIBERO, BridgeData V2) are converted OFFLINE into the
episode format `train/dataset.py::EpisodeDataset` consumes:

    frame_embs        [T, 512]   standardized YOLO-World SPPF GAP embeddings
    source_box_embs   [T, 512]   per-role best-box embeddings
    target_box_embs   [T, 512]
    source_centers    [T, 2]     normalized (cx, cy)
    target_centers    [T, 2]
    box_weights       [T, 2]     detection confidence per role (0 = missed)
    text_tokens       [3, 512]   (command, source, target) CLIP text embs
    pwm_targets       [T, 5, 7]  action chunks, normalized to [-1, 1]

Key property: the frozen encoders (YOLO-World-S + its CLIP text tower) run
exactly ONCE, here — training never touches images, episodes are ~1000x
smaller than the raw video, and the training distribution is bit-identical
to what the deployed perception front-end produces.

Action convention (both datasets are 7-DoF, matching ``cfg.num_servos=7``):
    dims 0-2  Δ end-effector position (x, y, z)
    dims 3-5  Δ end-effector orientation (roll, pitch, yaw)
    dim  6    gripper command
``pwm_targets[t]`` is the chunk of the next ``plan_steps`` NATIVE-rate actions
starting at sampled frame ``t`` (padded by repeating the final action), i.e.
the plan rows are spaced at the dataset's control rate. Normalization is
quantile-based (q01/q99 -> [-1, 1], clipped), computed over the whole run and
saved to ``norm_stats.json`` next to the episodes — keep that file with any
trained checkpoint, since the planner's outputs only mean something through
its inverse.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.perception.command_parser import strip_article

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SourceEpisode:
    """One raw demonstration, normalized across datasets.

    Attributes:
        frames: ``[T_raw]`` list/array of HxWx3 uint8 **RGB** frames (readers
            yield RGB; conversion to the BGR the detector expects happens in
            :class:`EpisodeBuilder`).
        actions: ``[T_raw, 7]`` float raw actions at the native control rate.
        instruction: Natural-language task string.
        source_hz: Native control/frame rate of the demonstration.
        episode_id: Stable identifier used for the output filename.
    """

    frames: list
    actions: np.ndarray
    instruction: str
    source_hz: float
    episode_id: str


def subsample_indices(n_frames: int, source_hz: float, target_hz: float) -> list[int]:
    """Frame indices sampling a native-rate episode down to ``target_hz``.

    Uses the same integer-counter emit rule as ``VideoStreamSampler`` (emit
    when ``t >= k / target_hz``), so offline preprocessing and the online
    2 Hz sampler pick identical frames for identical streams.

    Args:
        n_frames: Number of native frames.
        source_hz: Native frame rate.
        target_hz: Desired sampled rate (``cfg.real_frame_hz``).

    Returns:
        Sorted native-frame indices (always includes index 0).
    """
    if target_hz >= source_hz:
        return list(range(n_frames))
    indices, emitted = [], 0
    for i in range(n_frames):
        if i / source_hz >= emitted / target_hz:
            indices.append(i)
            emitted += 1
    return indices


def chunk_actions(
    actions: np.ndarray, frame_indices: list[int], plan_steps: int
) -> np.ndarray:
    """Builds per-sampled-frame action chunks at the native rate.

    ``chunk[t] = actions[i_t : i_t + plan_steps]`` where ``i_t`` is sampled
    frame ``t``'s native index; chunks running off the episode end are padded
    by repeating the last action (hold pose).

    Args:
        actions: ``[T_raw, A]`` native-rate actions.
        frame_indices: Output of :func:`subsample_indices`.
        plan_steps: Rows per chunk (``cfg.plan_steps``).

    Returns:
        ``[len(frame_indices), plan_steps, A]`` float32 array.
    """
    T_raw = actions.shape[0]
    padded = np.concatenate(
        [actions, np.repeat(actions[-1:], plan_steps, axis=0)], axis=0
    )
    return np.stack(
        [padded[min(i, T_raw - 1) : min(i, T_raw - 1) + plan_steps] for i in frame_indices],
        axis=0,
    ).astype(np.float32)


class ActionNormalizer:
    """Quantile action normalization: per-dim q01/q99 -> [-1, 1], clipped.

    The q01/q99 window (rather than min/max) is robust to the outlier action
    spikes teleoperated datasets always contain. ``inverse`` maps planner
    output back to raw action units for execution.
    """

    def __init__(self, q_low: np.ndarray, q_high: np.ndarray) -> None:
        self.q_low = np.asarray(q_low, dtype=np.float64)
        self.q_high = np.asarray(q_high, dtype=np.float64)
        span = self.q_high - self.q_low
        # A constant dim (e.g. an unused axis) gets span 1 to avoid div-by-0;
        # it normalizes to a constant, which is correct.
        self._span = np.where(span > 1e-8, span, 1.0)

    @classmethod
    def fit(cls, action_arrays: Iterable[np.ndarray]) -> "ActionNormalizer":
        """Computes stats over every action of every episode."""
        stacked = np.concatenate([np.asarray(a, dtype=np.float64) for a in action_arrays], axis=0)
        return cls(np.quantile(stacked, 0.01, axis=0), np.quantile(stacked, 0.99, axis=0))

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        x = (np.asarray(actions, dtype=np.float64) - self.q_low) / self._span
        return np.clip(2.0 * x - 1.0, -1.0, 1.0).astype(np.float32)

    def inverse(self, normalized: np.ndarray, zero_center: bool = False) -> np.ndarray:
        """Maps a normalized action back to raw units.

        Default: the exact inverse of ``__call__`` (``x=-1 -> q_low``,
        ``x=+1 -> q_high``), so a neutral output ``x=0`` maps to the RANGE
        MIDPOINT ``(q_low+q_high)/2`` — which is NOT zero motion when the
        quantiles are asymmetric. For a delta-action policy that regresses
        toward neutral, that midpoint is a constant per-step drift.

        ``zero_center=True`` instead scales by the half-span and drops the
        offset (``x=0 -> 0`` motion, ``x=±1 -> ±span/2``), so a collapsed /
        neutral policy output means STAY STILL rather than drift. Diagnostic /
        mitigation for the drift-into-wall failure; the principled fix is to
        train against zero-centered (symmetric) targets.
        """
        x = np.asarray(normalized, dtype=np.float64)
        if zero_center:
            return (x * (self._span / 2.0)).astype(np.float32)
        x = (x + 1.0) / 2.0
        return (x * self._span + self.q_low).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"q_low": self.q_low.tolist(), "q_high": self.q_high.tolist()}, indent=2)
        )

    @classmethod
    def load(cls, path: str | Path) -> "ActionNormalizer":
        d = json.loads(Path(path).read_text())
        return cls(np.asarray(d["q_low"]), np.asarray(d["q_high"]))


class EpisodeBuilder:
    """Runs the frozen MicroVLA perception front-end over one raw episode.

    Args:
        cfg: Canonical config.
        mock: When True, uses the deterministic mock perception/text encoders
            (dry-runs, tests, CI — no weights, no downloads). When False,
            lazily builds ``YoloWorldPerception`` + ``ClipTaskEncoder``
            (requires the ``perception`` extra).
        device: Torch device for the real detector.
    """

    def __init__(self, cfg: MicroVLAConfig = DEFAULT_CONFIG, mock: bool = False,
                 device: str = "cpu") -> None:
        self.cfg = cfg
        if mock:
            from microvla.perception.text_encoder import MockTaskEncoder
            from microvla.perception.yolo_world import MockYoloWorldPerception

            self.perception = MockYoloWorldPerception(vis_dim=cfg.vis_dim)
            self.task_encoder = MockTaskEncoder(cfg.text_dim)
        else:
            from microvla.perception.text_encoder import ClipTaskEncoder
            from microvla.perception.yolo_world import YoloWorldPerception

            self.perception = YoloWorldPerception(device=device)
            self.task_encoder = ClipTaskEncoder(self.perception)
        # CLIP text encoding costs ~1-2 s per call; datasets repeat the same
        # instruction across many demos (LIBERO: 50 demos/instruction), so
        # cache TaskEncodings and skip redundant set_classes calls.
        self._task_cache: dict[str, object] = {}
        self._active_classes: list[str] | None = None

    def build(self, episode: SourceEpisode, normalizer: ActionNormalizer) -> dict[str, np.ndarray]:
        """Converts one raw episode into the MicroVLA .npz key dict.

        Mirrors ``MicroVLAPipeline.set_task`` exactly: encode the instruction
        once, point the detector at the article-stripped ordered
        ``[source, target]`` classes, then perceive each sampled frame.

        Args:
            episode: The raw demonstration.
            normalizer: Fitted action normalizer.

        Returns:
            Dict with the ``train.dataset.EPISODE_KEYS`` arrays.

        Raises:
            ValueError: If the action dim does not match ``cfg.num_servos``.
        """
        if episode.actions.shape[-1] != self.cfg.num_servos:
            raise ValueError(
                f"{episode.episode_id}: action dim {episode.actions.shape[-1]} != "
                f"cfg.num_servos ({self.cfg.num_servos}); remap in the dataset reader."
            )

        task = self._task_cache.get(episode.instruction)
        if task is None:
            task = self.task_encoder.encode(episode.instruction)
            self._task_cache[episode.instruction] = task
        parsed = task.parsed
        src, tgt = strip_article(parsed.source), strip_article(parsed.target)
        classes = [src] if src == tgt else [src, tgt]
        if classes != self._active_classes:
            self.perception.set_classes(classes)
            self._active_classes = classes

        indices = subsample_indices(
            len(episode.frames), episode.source_hz, self.cfg.real_frame_hz
        )
        frame_embs, s_embs, t_embs, s_ctrs, t_ctrs, weights = [], [], [], [], [], []
        for i in indices:
            frame_rgb = np.ascontiguousarray(episode.frames[i])
            frame_bgr = np.ascontiguousarray(frame_rgb[..., ::-1])  # detector convention
            p = self.perception.perceive(frame_bgr)
            frame_embs.append(p.frame_emb.numpy())
            s_embs.append(p.source.emb.numpy())
            t_embs.append(p.target.emb.numpy())
            s_ctrs.append(p.source.center.numpy())
            t_ctrs.append(p.target.center.numpy())
            weights.append([p.source.confidence, p.target.confidence])

        pwm = chunk_actions(normalizer(episode.actions), indices, self.cfg.plan_steps)
        return {
            "frame_embs": np.stack(frame_embs).astype(np.float32),
            "source_box_embs": np.stack(s_embs).astype(np.float32),
            "target_box_embs": np.stack(t_embs).astype(np.float32),
            "source_centers": np.stack(s_ctrs).astype(np.float32),
            "target_centers": np.stack(t_ctrs).astype(np.float32),
            "box_weights": np.asarray(weights, dtype=np.float32),
            "text_tokens": task.tokens().numpy().astype(np.float32),
            "pwm_targets": pwm,
        }


def run_conversion(
    episodes: Callable[[], Iterator[SourceEpisode]],
    out_dir: str | Path,
    cfg: MicroVLAConfig = DEFAULT_CONFIG,
    mock: bool = False,
    device: str = "cpu",
    limit: int | None = None,
    teacher=None,
) -> Path:
    """Two-pass conversion driver: fit action stats, then write episodes.

    Pass 1 streams every episode's actions to fit the :class:`ActionNormalizer`
    (saved as ``norm_stats.json``); pass 2 runs perception and writes one
    ``.npz`` per episode plus a ``manifest.json`` (id, length, instruction).
    ``episodes`` is a zero-arg callable returning a FRESH iterator so both
    passes can stream without holding the dataset in memory.

    Args:
        episodes: Factory of :class:`SourceEpisode` iterators.
        out_dir: Output directory for ``.npz`` + stats + manifest.
        cfg: Canonical config.
        mock: Use mock perception (dry-run; no weights needed).
        device: Torch device for the real detector.
        limit: Optional cap on episodes converted (applies to both passes).
        teacher: Optional ``preprocess.teacher.TeacherPolicy`` — when given,
            every episode's actions are RELABELED by the teacher (knowledge
            distillation; wrap with ``CachedTeacher`` so the teacher runs
            once across the two passes). Stats are fitted on the teacher
            actions, so the planner distills the teacher's distribution.

    Returns:
        The output directory path.
    """
    from train.dataset import save_episode  # local import: keeps torch optional at import time

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _take(it: Iterator[SourceEpisode]) -> Iterator[SourceEpisode]:
        for n, ep in enumerate(it):
            if limit is not None and n >= limit:
                return
            if teacher is not None:
                ep = dataclasses.replace(ep, actions=teacher.relabel(ep))
            yield ep

    logger.info("pass 1/2: fitting action normalization stats")
    normalizer = ActionNormalizer.fit(ep.actions for ep in _take(episodes()))
    normalizer.save(out / "norm_stats.json")

    logger.info("pass 2/2: running frozen perception and writing episodes")
    builder = EpisodeBuilder(cfg, mock=mock, device=device)
    manifest = []
    for n, ep in enumerate(_take(episodes())):
        arrays = builder.build(ep, normalizer)
        path = out / f"{ep.episode_id}.npz"
        save_episode(path, arrays)
        manifest.append(
            {"file": path.name, "id": ep.episode_id, "T": int(arrays["frame_embs"].shape[0]),
             "instruction": ep.instruction}
        )
        if (n + 1) % 50 == 0:
            logger.info("  %d episodes written", n + 1)

    (out / "manifest.json").write_text(
        json.dumps(
            {"label_source": type(teacher).__name__ if teacher else "dataset",
             "episodes": manifest},
            indent=2,
        )
    )
    logger.info("done: %d episodes -> %s", len(manifest), out)
    return out
