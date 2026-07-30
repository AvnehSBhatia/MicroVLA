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
from microvla.perception.prompts import role_chains, with_fallbacks

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
    # v7 optional raw robot state (LIBERO fills these; Bridge leaves None):
    #   proprio_raw [T_raw, 10]  — utils.proprio.build_proprio per native step
    #   eef_pos_raw [T_raw, 3]   — absolute EEF xyz per native step
    proprio_raw: np.ndarray | None = None
    eef_pos_raw: np.ndarray | None = None
    #: v8 optional SECOND view, used for OBJECT DETECTION only. The wrist camera
    #: supplies 0.68 proposals/frame with 47% of frames empty (paper.md 4r),
    #: which caps what any object-reasoning module can learn; the third-person
    #: view of the same scenes yields 3.40. `frames` still drives the frame
    #: embedding the world model predicts, so the ego-view coupling 4f requires
    #: is unchanged — only the detector moves.
    detect_frames: list | None = None


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

    @classmethod
    def fit_symmetric(cls, action_arrays: Iterable[np.ndarray]) -> "ActionNormalizer":
        """Quantile-robust SYMMETRIC stats: normalized 0 <=> raw 0 (no motion).

        ``s = max(|q01|, |q99|)`` per dim, ``q_low = -s, q_high = +s``. The v5
        hard rule: an asymmetric mapping turns a neutral/collapsed policy
        output into a constant drift command (measured drift-into-wall). All
        NEW bakes use this; ``preprocess/renorm_symmetric.py`` retrofits old ones.
        """
        stacked = np.concatenate([np.asarray(a, dtype=np.float64) for a in action_arrays], axis=0)
        s = np.maximum(np.abs(np.quantile(stacked, 0.01, axis=0)),
                       np.abs(np.quantile(stacked, 0.99, axis=0)))
        s = np.where(s > 1e-8, s, 1.0)
        return cls(-s, s)

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


# The prompt chains live in microvla/perception/prompts.py so the BAKE path and
# the DEPLOYMENT path (microvla/jepa/loop.py) ground roles identically. They used
# to live here, which meant the 0%-detection fix reached the corpus but not the
# robot: every closed-loop eval ran a sighted-trained policy blind. See that
# module's docstring for the measurements.
_with_fallbacks = with_fallbacks
_role_chains = role_chains


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
                 device: str = "cpu", store_frames: bool = False,
                 grid_size: int = 0) -> None:
        self.cfg = cfg
        # v7: also bake the sampled raw frames (uint8) so perception (TQSA) is
        # trainable. Off by default (Bridge world-model-only bakes stay lean);
        # the LIBERO converter turns it on.
        self.store_frames = store_frames
        self.grid_size = int(grid_size)
        if mock:
            from microvla.perception.text_encoder import MockTaskEncoder
            from microvla.perception.yolo_world import MockYoloWorldPerception

            self.perception = MockYoloWorldPerception(vis_dim=cfg.vis_dim,
                                                      grid_size=grid_size)
            self.task_encoder = MockTaskEncoder(cfg.text_dim)
        else:
            from microvla.perception.text_encoder import ClipTaskEncoder
            from microvla.perception.yolo_world import YoloWorldPerception

            # det_conf from cfg, NOT the detector class default: the robot
            # reads the same field, so the two sides cannot drift (defect 26).
            self.perception = YoloWorldPerception(
                device=device, grid_size=grid_size, det_conf=cfg.det_conf,
                role_disjoint_iou=cfg.role_disjoint_iou)
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
        # MEASURED 2026-07-26: YOLO-World-S returns EXACTLY 0.000 for every
        # LIBERO product name — "alphabet soup", "bbq sauce", "cream cheese".
        # Baking with set_classes([src, tgt]) therefore produced a corpus in
        # which the source object was detected on 0% of frames, in BOTH camera
        # views, so box_weight was 0 everywhere and fusion faded all box and
        # geometry evidence to nothing. The same objects DO detect under
        # concrete visual categories on the same frames: agentview "bottle"
        # 0.604 / "can" 0.246 / "box" 0.195; wrist "box" 0.499 / "cardboard
        # box" 0.424. Abstract fallbacks are useless ("product", "package",
        # "item", "object", "thing" all 0.000) — a tail must name a CONCRETE
        # visual category to recover anything.
        #
        # set_role_prompts takes the best box of the FIRST prompt that detected
        # anything, so the exact phrase still wins wherever it grounds and the
        # tail only supplies recall where it does not.
        role_src, role_tgt = _role_chains(src, tgt)
        signature = [role_src, role_tgt]
        if signature != self._active_classes:
            self.perception.set_role_prompts(role_src, role_tgt)
            self._active_classes = signature

        indices = subsample_indices(
            len(episode.frames), episode.source_hz, self.cfg.real_frame_hz
        )
        frame_embs, s_embs, t_embs, s_ctrs, t_ctrs, weights = [], [], [], [], [], []
        grids = []      # [T, g*g, vis_dim] coarse spatial map, when enabled
        o_embs, o_ctrs, o_wts = [], [], []
        k = self.cfg.max_objects
        det = episode.detect_frames if episode.detect_frames is not None else episode.frames
        for i in indices:
            frame_rgb = np.ascontiguousarray(episode.frames[i])
            frame_bgr = np.ascontiguousarray(frame_rgb[..., ::-1])  # detector convention
            # Detection may run on a DIFFERENT view than the frame embedding.
            # Both are the same scene at the same instant, so boxes and centers
            # describe the world the ego frame is looking at; only the pixels
            # the detector sees change.
            det_bgr = (frame_bgr if episode.detect_frames is None
                       else np.ascontiguousarray(np.asarray(det[i])[..., ::-1]))
            p = self.perception.perceive(det_bgr)
            frame_embs.append(p.frame_emb.numpy())
            if p.spatial_grid is not None:
                grids.append(p.spatial_grid.numpy())
            s_embs.append(p.source.emb.numpy())
            t_embs.append(p.target.emb.numpy())
            s_ctrs.append(p.source.center.numpy())
            t_ctrs.append(p.target.center.numpy())
            weights.append([p.source.confidence, p.target.confidence])
            # v8: the CLASS-AGNOSTIC scene, from the same detector forward. The
            # two role slots above collapse a frame to a hard argmax that
            # returns nothing when the phrase does not ground (paper.md 4n);
            # these keep every box the forward already produced. Padded to
            # cfg.max_objects at weight 0.0, which is bit-identically inert
            # downstream, so K is a fixed shape rather than a real cap.
            oe = np.zeros((k, self.cfg.vis_dim), dtype=np.float32)
            oc = np.zeros((k, 2), dtype=np.float32)
            ow = np.zeros((k,), dtype=np.float32)
            for j, b in enumerate(p.proposals[:k]):
                oe[j] = b.emb.numpy()
                oc[j] = b.center.numpy()
                ow[j] = b.confidence
            o_embs.append(oe); o_ctrs.append(oc); o_wts.append(ow)

        pwm = chunk_actions(normalizer(episode.actions), indices, self.cfg.plan_steps)
        out = {
            "frame_embs": np.stack(frame_embs).astype(np.float32),
            "source_box_embs": np.stack(s_embs).astype(np.float32),
            "target_box_embs": np.stack(t_embs).astype(np.float32),
            "source_centers": np.stack(s_ctrs).astype(np.float32),
            "target_centers": np.stack(t_ctrs).astype(np.float32),
            "box_weights": np.asarray(weights, dtype=np.float32),
            "obj_embs": np.stack(o_embs).astype(np.float32),
            "obj_centers": np.stack(o_ctrs).astype(np.float32),
            "obj_weights": np.stack(o_wts).astype(np.float32),
            "text_tokens": task.tokens().numpy().astype(np.float32),
            "pwm_targets": pwm,
        }
        # The coarse spatial map. GAP (frame_embs) throws away WHERE, and on a
        # wrist camera WHERE is the servo error; the two role boxes were the
        # only spatial channel and 4r measured 0.68 proposals per frame, so on
        # roughly half the frames the policy had none. Baking the grid makes the
        # signal detection-independent AND lets TQSA train without storing raw
        # frames or re-running the frozen backbone every epoch.
        if grids:
            out["spatial_grid"] = np.stack(grids).astype(np.float32)
        # v7: raw sampled frames (uint8, compressed by savez) — makes perception
        # TRAINABLE (TQSA) without ever re-downloading. ~50 KB/frame at 128 px.
        if self.store_frames:
            out["wrist_frames"] = np.stack(
                [np.ascontiguousarray(episode.frames[i]) for i in indices]
            ).astype(np.uint8)
        # v7: proprio + absolute EEF chunk, sampled at the SAME indices.
        if episode.proprio_raw is not None:
            out["proprio"] = np.asarray(
                [episode.proprio_raw[i] for i in indices], dtype=np.float32
            )
        if episode.eef_pos_raw is not None:
            out["eef_pos_chunk"] = chunk_actions(
                np.asarray(episode.eef_pos_raw, dtype=np.float32),
                indices, self.cfg.plan_steps,
            )
        return out


def run_conversion(
    episodes: Callable[[], Iterator[SourceEpisode]],
    out_dir: str | Path,
    cfg: MicroVLAConfig = DEFAULT_CONFIG,
    mock: bool = False,
    device: str = "cpu",
    limit: int | None = None,
    teacher=None,
    store_frames: bool = False,
    grid_size: int = 0,
    provenance: dict | None = None,
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
        provenance: Extra dataset-specific facts to record in
            ``manifest.json``'s ``provenance`` block — the LIBERO converter
            passes the camera and the de-flip. Anything a DEPLOYMENT must
            reproduce belongs here; the shared knobs read off ``cfg`` are
            added automatically.

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

    provenance = dict(provenance or {})
    logger.info("pass 1/2: fitting SYMMETRIC action normalization stats")
    normalizer = ActionNormalizer.fit_symmetric(ep.actions for ep in _take(episodes()))
    normalizer.save(out / "norm_stats.json")

    logger.info("pass 2/2: running frozen perception and writing episodes")
    builder = EpisodeBuilder(cfg, mock=mock, device=device,
                             store_frames=store_frames, grid_size=grid_size)
    manifest = []
    frame_hw = None
    for n, ep in enumerate(_take(episodes())):
        if frame_hw is None:
            src = ep.detect_frames if ep.detect_frames is not None else ep.frames
            if len(src):
                frame_hw = [int(np.asarray(src[0]).shape[0]),
                            int(np.asarray(src[0]).shape[1])]
        arrays = builder.build(ep, normalizer)
        path = out / f"{ep.episode_id}.npz"
        save_episode(path, arrays)
        manifest.append(
            {"file": path.name, "id": ep.episode_id, "T": int(arrays["frame_embs"].shape[0]),
             "instruction": ep.instruction}
        )
        if (n + 1) % 50 == 0:
            logger.info("  %d episodes written", n + 1)

    # The corpus SELF-DESCRIBES the conditions the robot must reproduce. Three
    # of the 26 defects so far were a deployment knob silently differing from
    # the value the corpus was built with (camera, detector threshold,
    # perception period); none was catchable from the .npz files, because the
    # .npz files did not say. `provenance` is what makes a mismatch checkable.
    (out / "manifest.json").write_text(
        json.dumps(
            {"label_source": type(teacher).__name__ if teacher else "dataset",
             "provenance": {
                 "det_conf": float(cfg.det_conf),
                 "role_disjoint_iou": float(cfg.role_disjoint_iou),
                 "real_frame_hz": float(cfg.real_frame_hz),
                 "max_objects": int(cfg.max_objects),
                 "grid_size": int(grid_size),
                 "vis_dim": int(cfg.vis_dim),
                 "perception": type(builder.perception).__name__,
                 # The pixels the frozen detector actually saw. A live env
                 # rendered at a different size upscales differently into the
                 # detector's 512-px short side, so the deployed features come
                 # from images the corpus never contained -- same shape of
                 # defect as the camera and the threshold.
                 "detect_frame_hw": frame_hw,
                 **provenance,
             },
             "episodes": manifest},
            indent=2,
        )
    )
    logger.info("done: %d episodes -> %s", len(manifest), out)
    return out
