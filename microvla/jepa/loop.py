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
              fusion(…, held boxes x decayed weight, last action) ───┘
                    -> TRM(fused, held drift code, current latent) -> next_emb
                                                                      │
        plan = tau * planner(next_emb) + (1 - tau) * previous plan  (hold-blend)
        executed action = plan[0], fed back to fusion next tick

v3 semantics (per the architecture review):
    * Box evidence is HELD from the last real tick during dreams, with its
      weight decayed by ``staleness_decay ** k`` — objects don't teleport in
      33 ms, so v2's zeroing threw away near-perfect information.
    * The drift encoder steps on REAL ticks only; its code is held constant
      across dreams (state change is a summary of *measured* evidence).
    * Low trust is action-space aware (v5, ``cfg.action_space``): for DELTA
      actions (LIBERO/Bridge) it BRAKES — attenuates the plan toward zero
      motion, since holding a delta is momentum that perpetuates drift; for
      ABSOLUTE PWM targets it HOLD-blends toward the previously emitted plan
      and never scales toward zero (zero commands servo mid-range).
    * The dream latent is re-standardized after correction so fusion always
      sees the canonical embedding space it was trained on.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Optional

import torch

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.jepa.corrector import InnovationCorrector
from microvla.perception.yolo_world import BoxObs, Perception
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.perception.command_parser import strip_article
from microvla.trm.interface import TRMBase
from microvla.trm.mock_trm import MockTRM
from microvla.utils.embedding import standardize

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cost
    from microvla.perception.text_encoder import TaskEncoding

logger = logging.getLogger(__name__)


def _role_prompts(phrase: str) -> list[str]:
    """Ordered detection prompts for one role: full phrase, then bare noun.

    The full phrase carries any spatial disambiguator ("black bowl between the
    plate and the ramekin"); the article/clause-stripped noun ("black bowl") is
    the recall fallback. Deduplicated so a phrase with no clause (``"the can"``
    -> ``"can"``) still yields both distinct prompts, while an already-bare
    phrase collapses to one.
    """
    prompts: list[str] = []
    for p in (phrase, strip_article(phrase)):
        p = p.strip()
        if p and p not in prompts:
            prompts.append(p)
    return prompts


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
            already trust-blended: ``tau * new_plan + (1 - tau) * previous
            emitted plan`` (low trust holds the current commands rather than
            moving toward servo-neutral). Row 0 is the action executed this
            tick; rows 1+ are the receding horizon at 1-tick spacing.
        trust: Current corrector trust ``tau`` (the blend factor).
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
        # Held evidence between real ticks (v3):
        self._last_percept: Optional[Perception] = None   # last REAL boxes
        self._last_state_delta: Optional[torch.Tensor] = None  # [1, state_dim]
        self._dream_k: int = 0                            # dream ticks since real
        self._last_plan: Optional[torch.Tensor] = None    # [plan_steps, num_servos]
        self._last_action: Optional[torch.Tensor] = None  # [num_servos], executed row 0
        # v5: last-known box per role across REAL ticks + how many consecutive
        # real frames that role has MISSED. A real-tick detection miss holds
        # the last-known box at cfg.miss_decay**age weight instead of resetting
        # to the (0.5, 0.5)/weight-0 fallback — the wrist camera loses the
        # object exactly at approach/grasp, when geometry matters most.
        self._held_boxes: list[Optional[BoxObs]] = [None, None]
        self._miss_age: list[int] = [0, 0]
        # Rolling window of the latents that drove recent ticks (oldest ->
        # newest), handed to the TRM as its context window each call.
        self._latent_ctx: deque[torch.Tensor] = deque(maxlen=cfg.context_window)

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
        # Per-role detection prompts in preference order: the FULL phrase first
        # (so the frozen region-text head can disambiguate "black bowl between
        # the plate and the ramekin" from the other black bowls), then the
        # article-stripped bare noun ("black bowl") as a recall fallback. The
        # text-token EMBEDDINGS still use the full phrases (harvested by the
        # task encoder); only the detection prompts are role-ordered here.
        # source == target -> a single role; the target shares its box.
        same = parsed.source == parsed.target
        source_prompts = _role_prompts(parsed.source)
        target_prompts = None if same else _role_prompts(parsed.target)
        self.perception.set_role_prompts(source_prompts, target_prompts)
        self.drift.reset()
        self.corrector.reset()
        self._pending_pred = None
        self._seen_real = False
        self._last_percept = None
        self._last_state_delta = None
        self._dream_k = 0
        self._last_plan = None
        self._last_action = None
        self._held_boxes = [None, None]
        self._miss_age = [0, 0]
        self._latent_ctx.clear()

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

            last_action = (
                self._last_action.unsqueeze(0)
                if self._last_action is not None
                else torch.zeros(1, self.cfg.num_servos)
            )

            if is_real:
                raw_percept = self.perception.perceive(frame_bgr)
                frame_emb = raw_percept.frame_emb  # [vis_dim], standardized

                if self._pending_pred is not None:
                    self.corrector.on_measurement(self._pending_pred, frame_emb)

                # v5 miss-hold: a role whose detection MISSED this real tick
                # keeps its last-known box at miss_decay**age weight instead of
                # resetting to the (0.5, 0.5)/weight-0 fallback. A hit refreshes
                # the held box and resets the age.
                eff_boxes: list[BoxObs] = []
                for i, obs in enumerate((raw_percept.source, raw_percept.target)):
                    if obs.confidence > 0.0:
                        self._held_boxes[i] = obs
                        self._miss_age[i] = 0
                        eff_boxes.append(obs)
                    elif self._held_boxes[i] is not None:
                        self._miss_age[i] += 1
                        held_box = self._held_boxes[i]
                        eff_boxes.append(BoxObs(
                            emb=held_box.emb,
                            center=held_box.center,
                            xyxy=held_box.xyxy,
                            confidence=held_box.confidence
                            * self.cfg.miss_decay ** self._miss_age[i],
                        ))
                    else:
                        eff_boxes.append(obs)  # genuine cold miss: fallback stands
                percept = Perception(
                    frame_emb=frame_emb, source=eff_boxes[0], target=eff_boxes[1]
                )

                # Evidence weight per role = detection confidence (fresh or
                # miss-held-and-decayed).
                box_weight = torch.tensor(
                    [[percept.source.confidence, percept.target.confidence]],
                    dtype=torch.float32,
                )
                # Raw geometry for the planner (v5): src/tgt centers + weights —
                # for a wrist camera the target's frame position IS the visual-
                # servo error vector; hand it to the planner directly instead of
                # only through fusion's bottleneck.
                geom = torch.cat(
                    [percept.source.center.unsqueeze(0),
                     percept.target.center.unsqueeze(0), box_weight], dim=-1
                )  # [1, 6]
                fused = self.fusion(
                    text_tokens,
                    frame_emb.unsqueeze(0),
                    percept.source.emb.unsqueeze(0),
                    percept.target.emb.unsqueeze(0),
                    percept.source.center.unsqueeze(0),
                    percept.target.center.unsqueeze(0),
                    box_weight=box_weight,
                    last_action=last_action,
                )  # [1, 32, 5]

                # Drift steps on measured evidence only.
                state_delta = self.drift(frame_emb.unsqueeze(0))  # [1, 256]
                self._last_state_delta = state_delta

                latent = frame_emb
                out_perception: Optional[Perception] = percept
                self._last_percept = percept
                self._dream_k = 0
                self._seen_real = True
            else:
                if not self._seen_real:
                    raise RuntimeError(
                        "JEPALoop.tick(): dream tick requested before any real "
                        "frame has been observed in this episode. Call "
                        "tick(frame_bgr=...) at least once after set_task()."
                    )
                assert self._pending_pred is not None  # implied by _seen_real
                assert self._last_percept is not None and self._last_state_delta is not None

                # Corrected latent, re-standardized into the canonical space
                # fusion/TRM were trained on.
                latent = standardize(self.corrector.correct(self._pending_pred))

                # Hold the last REAL boxes; fade their evidence weight with
                # staleness (objects don't teleport between measurements).
                self._dream_k += 1
                held = self._last_percept
                fade = self.cfg.staleness_decay ** self._dream_k
                box_weight = torch.tensor(
                    [[held.source.confidence * fade, held.target.confidence * fade]],
                    dtype=torch.float32,
                )
                geom = torch.cat(
                    [held.source.center.unsqueeze(0),
                     held.target.center.unsqueeze(0), box_weight], dim=-1
                )  # [1, 6] — held centers, staleness-faded weights
                fused = self.fusion(
                    text_tokens,
                    latent.unsqueeze(0),
                    held.source.emb.unsqueeze(0),
                    held.target.emb.unsqueeze(0),
                    held.source.center.unsqueeze(0),
                    held.target.center.unsqueeze(0),
                    box_weight=box_weight,
                    last_action=last_action,
                )  # [1, 32, 5]

                # Drift code held: no measurement, no state update.
                state_delta = self._last_state_delta

                out_perception = None

            # TRM context window: the latents that drove the previous ticks.
            context = (
                torch.stack(list(self._latent_ctx), dim=0).unsqueeze(0)  # [1, K, 512]
                if self._latent_ctx
                else None
            )
            next_emb, next_box = self.trm(
                fused, state_delta, latent.unsqueeze(0), context=context, return_box=True
            )
            next_emb_unbatched = next_emb.squeeze(0)  # [512]
            self._pending_pred = next_emb_unbatched
            self._latent_ctx.append(latent)

            raw_plan = self.planner(next_emb, current_emb=latent.unsqueeze(0),
                                    state_delta=state_delta, fused=fused,
                                    pred_box_emb=next_box,
                                    geometry=geom).squeeze(0)  # [plan_steps, num_servos]

            # Trust semantics depend on the ACTION SPACE (v5):
            #   * "delta" (LIBERO/Bridge EEF deltas): zero IS "no motion", so
            #     low trust BRAKES — attenuates the commanded motion toward a
            #     stop. Holding the previous plan here would be momentum: a
            #     delta held is a motion continued, which perpetuates drift.
            #   * "absolute" (the Pi's PWM rig): zero is a real commanded pose
            #     (servo mid-range), so low trust HOLD-blends toward the
            #     previously emitted plan and NEVER scales toward zero.
            # The gripper is a hard +/-1 decision either way; blending it would
            # produce a meaningless fractional command.
            tau = self.corrector.trust
            if self._last_plan is None:
                plan = raw_plan
            elif self.cfg.action_space == "delta":
                plan = tau * raw_plan
                plan[:, -1] = torch.sign(raw_plan[:, -1])  # gripper stays hard +/-1
            else:
                plan = tau * raw_plan + (1.0 - tau) * self._last_plan
                plan[:, -1] = torch.sign(raw_plan[:, -1])  # gripper stays hard +/-1
            self._last_plan = plan
            self._last_action = plan[0]  # row 0 is executed this tick

        return TickResult(
            is_real=is_real,
            latent=latent,
            fused=fused.squeeze(0),
            state_delta=state_delta.squeeze(0),
            next_emb=next_emb_unbatched,
            plan=plan,
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
