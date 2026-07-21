"""Simple 2 Hz real-only MicroVLA inference pipeline.

Wires the frozen perception front-end (task encoder + YOLO-World, dual-box
grounded) into the trainable heads (SlotResonanceFusion -> AnchoredDriftEncoder
-> TRM -> ChronoQueryPlanner) and drives them over a video stream sampled at
``cfg.real_frame_hz``.

:class:`MicroVLAPipeline` is the debug/harness path: every ``step()`` is
exactly a real (non-dream) JEPA tick, run without the
:class:`~microvla.jepa.corrector.InnovationCorrector` — no innovation
tracking, no trust scaling on the plan. The deployment path is
:class:`~microvla.jepa.loop.JEPALoop`, which additionally dreams at 30 Hz
between real frames; this pipeline exists for straightforward 2 Hz debugging
and as the TRM builder's minimal harness.

Shapes follow DESIGN.md exactly. Runtime tensors are unbatched; ``step()``
adds a batch dim before calling modules and strips it again in the returned
``StepResult``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.perception.video_stream import VideoStreamSampler
from microvla.perception.yolo_world import Perception
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.trm.interface import TRMBase
from microvla.trm.mock_trm import MockTRM

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cost
    from microvla.perception.text_encoder import TaskEncoding

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Everything produced for one sampled (real) frame.

    Attributes:
        perception: Raw dual-box perception output for the frame (frozen
            encoders): frame embedding plus ordered source/target boxes.
        fused: Slot-fusion output, ``[fused_rows, fused_cols]`` = ``[32, 5]``.
        state_delta: Anchored drift code, ``[state_dim]`` = ``[256]``.
        next_emb: TRM-predicted next-frame embedding, ``[vis_dim]`` = ``[512]``.
        plan: Planner output in ``[-1, 1]``, ``[plan_steps, num_servos]`` =
            ``[5, 7]`` — rows are sequential timesteps, columns are servos.
            Unscaled: the pipeline runs without the JEPA corrector, so there
            is no trust factor here (unlike ``JEPALoop.tick().plan``).
    """

    perception: Perception
    fused: torch.Tensor
    state_delta: torch.Tensor
    next_emb: torch.Tensor
    plan: torch.Tensor


class MicroVLAPipeline:
    """Composable text+video -> servo-plan pipeline, real frames only.

    The pipeline is inference-only: ``step()`` puts every trainable module in
    eval mode and runs under ``torch.no_grad()``. Training goes through the
    modules directly (see ``train/train_planner.py``).

    Args:
        cfg: Shared configuration (dims, fps, budgets).
        task_encoder: Object with ``encode(text) -> TaskEncoding``, e.g.
            :class:`~microvla.perception.text_encoder.ClipTaskEncoder` or
            :class:`~microvla.perception.text_encoder.MockTaskEncoder`.
        perception: Object with ``set_classes(list[str])`` and
            ``perceive(frame_bgr) -> Perception``.
        fusion: Slot Resonance Fusion module.
        drift: Anchored Drift Encoder (holds per-episode runtime state).
        trm: Any ``TRMBase`` implementation (``MockTRM`` until the real
            ~10M-param TRM slots in).
        planner: Chrono-Query Planner module.
    """

    def __init__(
        self,
        cfg: MicroVLAConfig,
        task_encoder,
        perception,
        fusion: SlotResonanceFusion,
        drift: AnchoredDriftEncoder,
        trm: TRMBase,
        planner: ChronoQueryPlanner,
    ) -> None:
        self.cfg = cfg
        self.task_encoder = task_encoder
        self.perception = perception
        self.fusion = fusion
        self.drift = drift
        self.trm = trm
        self.planner = planner
        self._task: Optional["TaskEncoding"] = None

    def set_task(self, text: str) -> None:
        """Sets the language task for the episode.

        Encodes the text once (frozen encoder — no need to re-encode per
        frame), points the open-vocabulary detector at the ordered
        ``[source, target]`` classes, and resets the drift encoder so the
        next frame becomes the new anchor.

        Args:
            text: Natural-language task description, e.g. "move can to ball".
        """
        self._task = self.task_encoder.encode(text)
        parsed = self._task.parsed
        # One active class when the command has no distinct destination —
        # duplicate class strings would otherwise occupy two class ids.
        classes = [parsed.source] if parsed.source == parsed.target else [parsed.source, parsed.target]
        self.perception.set_classes(classes)
        self.drift.reset()

    def step(self, frame_bgr) -> StepResult:
        """Runs a single real frame through the whole stack.

        Exactly a JEPA real tick (grounded fusion, ``dream=False``) without
        the innovation corrector: no ``on_measurement`` bookkeeping and no
        trust scaling on the returned plan.

        Args:
            frame_bgr: ``np.ndarray`` HxWx3 uint8 BGR frame.

        Returns:
            A ``StepResult`` with unbatched tensors.

        Raises:
            RuntimeError: If ``set_task()`` has not been called.
        """
        if self._task is None:
            raise RuntimeError("MicroVLAPipeline.step() called before set_task().")

        self.fusion.eval()
        self.drift.eval()
        self.trm.eval()
        self.planner.eval()

        with torch.no_grad():
            percept = self.perception.perceive(frame_bgr)

            # Unbatched runtime tensors -> add batch dim for the modules.
            text_tokens = self._task.tokens().unsqueeze(0)              # [1, 3, text_dim]
            frame_emb = percept.frame_emb.unsqueeze(0)                  # [1, vis_dim]
            source_emb = percept.source.emb.unsqueeze(0)                # [1, vis_dim]
            target_emb = percept.target.emb.unsqueeze(0)                # [1, vis_dim]
            source_center = percept.source.center.unsqueeze(0)          # [1, 2]
            target_center = percept.target.center.unsqueeze(0)          # [1, 2]

            fused = self.fusion(
                text_tokens,
                frame_emb,
                source_emb,
                target_emb,
                source_center,
                target_center,
                dream=False,
            )                                                            # [1, 32, 5]
            state_delta = self.drift(frame_emb)                          # [1, 256]
            next_emb = self.trm(fused, state_delta)                      # [1, 512]
            plan = self.planner(next_emb)                                # [1, plan_steps, num_servos]

        return StepResult(
            perception=percept,
            fused=fused.squeeze(0),
            state_delta=state_delta.squeeze(0),
            next_emb=next_emb.squeeze(0),
            plan=plan.squeeze(0),
        )

    def run(self, source, text: str, max_steps: Optional[int] = None) -> list[StepResult]:
        """Runs the pipeline over a video source sampled at ``real_frame_hz``.

        Args:
            source: Anything ``VideoStreamSampler`` accepts — a path/int for
                ``cv2.VideoCapture`` or an iterable of frames /
                ``(frame, timestamp)`` pairs.
            text: Task text; passed to ``set_task()`` before the first frame.
            max_steps: Optional cap on the number of sampled frames processed.

        Returns:
            One ``StepResult`` per sampled (emitted) frame, in order.
        """
        self.set_task(text)
        sampler = VideoStreamSampler(source, target_fps=self.cfg.real_frame_hz)
        results: list[StepResult] = []
        for frame_bgr, _t in sampler:
            results.append(self.step(frame_bgr))
            if max_steps is not None and len(results) >= max_steps:
                break
        return results

    @classmethod
    def build_mock(cls, cfg: Optional[MicroVLAConfig] = None) -> "MicroVLAPipeline":
        """Builds an all-mock pipeline (no downloads, CPU-only) for tests.

        Args:
            cfg: Optional config; defaults to ``DEFAULT_CONFIG``.

        Returns:
            A pipeline with ``MockTaskEncoder``, ``MockYoloWorldPerception``,
            freshly initialized fusion/drift/planner heads, and ``MockTRM``.
        """
        from microvla.perception.text_encoder import MockTaskEncoder
        from microvla.perception.yolo_world import MockYoloWorldPerception

        cfg = cfg or DEFAULT_CONFIG
        return cls(
            cfg=cfg,
            task_encoder=MockTaskEncoder(cfg.text_dim),
            perception=MockYoloWorldPerception(vis_dim=cfg.vis_dim),
            fusion=SlotResonanceFusion(cfg),
            drift=AnchoredDriftEncoder(cfg),
            trm=MockTRM(cfg),
            planner=ChronoQueryPlanner(cfg),
        )

    @classmethod
    def build_real(
        cls,
        cfg: Optional[MicroVLAConfig] = None,
        trm: Optional[TRMBase] = None,
        device: str = "cpu",
    ) -> "MicroVLAPipeline":
        """Builds the real pipeline with frozen YOLO-World-S + its CLIP text tower.

        Requires the ``perception`` extra (``pip install microvla[perception]``);
        the heavy dependencies are imported lazily inside the encoder classes.

        Args:
            cfg: Optional config; defaults to ``DEFAULT_CONFIG``.
            trm: Optional real TRM implementing ``TRMBase``. If ``None``, a
                ``MockTRM`` placeholder is used and a warning is logged.
            device: Torch device string for the frozen encoders.

        Returns:
            A fully wired pipeline.
        """
        from microvla.perception.text_encoder import ClipTaskEncoder
        from microvla.perception.yolo_world import YoloWorldPerception

        cfg = cfg or DEFAULT_CONFIG
        perception = YoloWorldPerception(device=device)
        if trm is None:
            logger.warning(
                "No TRM provided to build_real(); falling back to the MockTRM "
                "placeholder (~0.21M linear params, see microvla/trm/mock_trm.py). "
                "Pass a real TRMBase implementation (~10M params, see "
                "microvla/trm/TRM_SPEC.md) for meaningful next-frame prediction."
            )
            trm = MockTRM(cfg)
        return cls(
            cfg=cfg,
            task_encoder=ClipTaskEncoder(perception),
            perception=perception,
            fusion=SlotResonanceFusion(cfg),
            drift=AnchoredDriftEncoder(cfg),
            trm=trm,
            planner=ChronoQueryPlanner(cfg),
        )
