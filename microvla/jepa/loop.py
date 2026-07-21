"""JEPA latent rollout loop: the v2 30 Hz deployment path.

Real YOLO-World perception runs at ``cfg.real_frame_hz`` (2 Hz); every other
tick of the ``cfg.tick_hz`` (30 Hz) control loop is a "dream" tick that feeds
the TRM's own (corrector-adjusted) prediction back through fusion instead of
a fresh camera frame — the same code path fusion uses for train-time
modality dropout, so dream mode is a trained mode, not an untested fallback.

::

    camera 30 Hz ─┬─ every 15th tick (2 Hz) ─ REAL TICK ─── perceive + corrector.on_measurement
                  └─ other 14 ticks ──────── DREAM TICK ─── corrector.correct(pending_pred)
                                                                      │
                              fusion -> drift -> TRM -> next_emb ────┘
                                                                      │
                                            planner(next_emb) * corrector.trust -> plan
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Optional

import torch

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.jepa.corrector import InnovationCorrector
from microvla.perception.yolo_world import Perception
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.trm.interface import TRMBase
from microvla.trm.mock_trm import MockTRM

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cost
    from microvla.perception.text_encoder import TaskEncoding

logger = logging.getLogger(__name__)


@dataclass
class TickResult:
    """Everything produced for one 30 Hz control tick.

    Attributes:
        is_real: ``True`` for a real (YOLO-perceived) tick, ``False`` for a
            dream tick driven by the corrected TRM prediction.
        latent: ``[vis_dim]`` = ``[512]`` frame embedding actually used to
            drive fusion/drift this tick — the real ``frame_emb`` on real
            ticks, or ``corrector.correct(pending_pred)`` on dream ticks.
        fused: ``[fused_rows, fused_cols]`` = ``[32, 5]`` slot-fusion output.
        state_delta: ``[state_dim]`` = ``[256]`` drift code.
        next_emb: ``[vis_dim]`` = ``[512]`` raw TRM prediction for the tick
            one step ahead; carried forward as the next tick's
            ``pending_pred``.
        plan: ``[plan_steps, num_servos]`` = ``[5, 7]`` in ``[-1, 1]``,
            already scaled by ``trust``.
        trust: Current corrector trust ``tau`` (the scale factor applied to
            ``plan``).
        perception: The raw dual-box :class:`Perception` on real ticks;
            ``None`` on dream ticks (no camera frame was consumed).
    """

    is_real: bool
    latent: torch.Tensor
    fused: torch.Tensor
    state_delta: torch.Tensor
    next_emb: torch.Tensor
    plan: torch.Tensor
    trust: float
    perception: Optional[Perception]


class JEPALoop:
    """30 Hz JEPA latent rollout: real perception at 2 Hz, dream in between.

    Inference-only: :meth:`tick` puts every trainable module in eval mode
    and runs under ``torch.no_grad()``, adding/removing the batch dimension
    internally so callers deal in unbatched tensors exactly like
    :class:`~microvla.pipeline.MicroVLAPipeline`.

    Args:
        cfg: Shared configuration (dims, rates, corrector hyperparameters).
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
        self.corrector = InnovationCorrector(cfg)

        self._task: Optional["TaskEncoding"] = None
        self._pending_pred: Optional[torch.Tensor] = None
        self._seen_real: bool = False

    def set_task(self, text: str) -> None:
        """Sets the language task for the episode.

        Encodes the task once (``task_encoder.encode`` — for
        :class:`~microvla.perception.text_encoder.ClipTaskEncoder` this
        already harvests the CLIP text tower and points the detector's
        active classes at ``[source, target]`` as a side effect), then
        explicitly (re-)applies ``perception.set_classes([source, target])``
        so encoders that don't carry a reference to ``perception`` (e.g.
        :class:`~microvla.perception.text_encoder.MockTaskEncoder`) still
        leave it correctly configured — idempotent, and cheap since it runs
        once per task rather than per tick. Resets drift, the corrector, and
        the internal latent-rollout state (pending prediction, whether a
        real frame has been seen yet) so the next tick starts a fresh
        episode.

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
        self.corrector.reset()
        self._pending_pred = None
        self._seen_real = False

    def tick(self, frame_bgr=None) -> TickResult:
        """Advances the loop by one 30 Hz tick.

        Args:
            frame_bgr: ``np.ndarray`` HxWx3 uint8 BGR frame for a REAL tick,
                or ``None`` for a DREAM tick.

        Returns:
            A ``TickResult`` with unbatched tensors.

        Raises:
            RuntimeError: If called before :meth:`set_task`, or if this is a
                dream tick (``frame_bgr is None``) and no real frame has ever
                been observed in the current episode (there is nothing yet
                to dream from).
        """
        if self._task is None:
            raise RuntimeError("JEPALoop.tick() called before set_task().")

        self.fusion.eval()
        self.drift.eval()
        self.trm.eval()
        self.planner.eval()

        with torch.no_grad():
            text_tokens = self._task.tokens().unsqueeze(0)  # [1, 3, text_dim]
            is_real = frame_bgr is not None

            if is_real:
                percept = self.perception.perceive(frame_bgr)
                frame_emb = percept.frame_emb  # [vis_dim]

                if self._pending_pred is not None:
                    self.corrector.on_measurement(self._pending_pred, frame_emb)

                fused = self.fusion(
                    text_tokens,
                    frame_emb.unsqueeze(0),
                    percept.source.emb.unsqueeze(0),
                    percept.target.emb.unsqueeze(0),
                    percept.source.center.unsqueeze(0),
                    percept.target.center.unsqueeze(0),
                    dream=False,
                )  # [1, 32, 5]
                state_delta = self.drift(frame_emb.unsqueeze(0))  # [1, 256]

                latent = frame_emb
                out_perception: Optional[Perception] = percept
                self._seen_real = True
            else:
                if not self._seen_real:
                    raise RuntimeError(
                        "JEPALoop.tick(): dream tick requested before any real "
                        "frame has been observed in this episode. Call "
                        "tick(frame_bgr=...) at least once after set_task()."
                    )
                assert self._pending_pred is not None  # implied by _seen_real
                latent = self.corrector.correct(self._pending_pred)  # [vis_dim]

                zero_box = torch.zeros(1, self.cfg.vis_dim, dtype=latent.dtype)
                zero_center = torch.zeros(1, 2, dtype=latent.dtype)

                fused = self.fusion(
                    text_tokens,
                    latent.unsqueeze(0),
                    zero_box,
                    zero_box,
                    zero_center,
                    zero_center,
                    dream=True,
                )  # [1, 32, 5]
                state_delta = self.drift(latent.unsqueeze(0))  # [1, 256]

                out_perception = None

            next_emb = self.trm(fused, state_delta)  # [1, 512]
            next_emb_unbatched = next_emb.squeeze(0)  # [512]
            self._pending_pred = next_emb_unbatched

            plan = self.planner(next_emb) * self.corrector.trust  # [1, plan_steps, num_servos]

        return TickResult(
            is_real=is_real,
            latent=latent,
            fused=fused.squeeze(0),
            state_delta=state_delta.squeeze(0),
            next_emb=next_emb_unbatched,
            plan=plan.squeeze(0),
            trust=self.corrector.trust,
            perception=out_perception,
        )

    def run(self, frames: Iterable, text: str) -> List[TickResult]:
        """Runs the loop over a sequence of frames sampled at ``tick_hz``.

        Sets the task, then ticks once per element of ``frames``. Every
        ``int(round(tick_hz / real_frame_hz))``-th tick (index 0, 15, 30,
        ... at the default 30/2 Hz) is REAL and consumes that frame; every
        other tick is a DREAM tick and the corresponding frame is ignored
        (``tick(None)`` is called instead).

        Args:
            frames: Iterable of ``np.ndarray`` HxWx3 uint8 BGR frames,
                sampled at the full ``tick_hz`` (30 fps) rate — i.e. every
                frame of the control loop, not pre-downsampled.
            text: Task text; passed to :meth:`set_task` before the first
                tick.

        Returns:
            One ``TickResult`` per element of ``frames``, in order.
        """
        self.set_task(text)
        period = int(round(self.cfg.tick_hz / self.cfg.real_frame_hz))
        results: List[TickResult] = []
        for i, frame in enumerate(frames):
            is_real_tick = i % period == 0
            results.append(self.tick(frame if is_real_tick else None))
        return results

    @classmethod
    def build_mock(cls, cfg: Optional[MicroVLAConfig] = None) -> "JEPALoop":
        """Builds an all-mock loop (no downloads, CPU-only) for tests.

        Args:
            cfg: Optional config; defaults to ``DEFAULT_CONFIG``.

        Returns:
            A loop with ``MockTaskEncoder``, ``MockYoloWorldPerception``,
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
    ) -> "JEPALoop":
        """Builds the real loop with frozen YOLO-World-S perception + CLIP text.

        Requires the ``perception`` extra (``pip install microvla[perception]``);
        the heavy dependencies are imported lazily inside the perception
        classes.

        Args:
            cfg: Optional config; defaults to ``DEFAULT_CONFIG``.
            trm: Optional real TRM implementing ``TRMBase``. If ``None``, a
                ``MockTRM`` placeholder is used and a warning is logged.
            device: Torch device string for the frozen encoders.

        Returns:
            A fully wired ``JEPALoop``.
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
