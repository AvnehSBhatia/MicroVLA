"""MicroVLAPolicy: closed-loop inference wrapper around JEPALoop (paper E3).

Bridges the deployment-path :class:`~microvla.jepa.loop.JEPALoop` (which
speaks unbatched tensors and BGR frames at a fixed 30 Hz internal cadence) to
the duck-typed policy interface a LIBERO (or LIBERO-like mock) evaluation
harness expects: ``reset(instruction)`` once per episode, then
``act(frame_rgb) -> np.ndarray[7]`` once per environment step.

Checkpoint loading follows the SHARED CONTRACT ledger
(``train/train_full.py::save``): ``full_stageB.pt`` carries
``{cfg, trm_d, fusion, drift, trm, planner}``; ``full_stageA.pt`` carries
everything but ``planner``. ``checkpoint`` may be:

* ``None`` (or the literal string ``"none"``) -- every module is freshly
  initialized (untrained); this is the smoke-test path, and the world model
  defaults to :class:`~microvla.trm.mock_trm.MockTRM` (loudly warned) unless
  ``trm=`` is supplied.
* A path to a specific ``.pt`` file, or a directory containing
  ``full_stageB.pt`` / ``full_stageA.pt``. Stage-B is preferred; if only
  stage-A is found (by content -- the ``"planner"`` key, not the filename),
  the planner is left freshly initialized and a warning is logged.

``trm=`` always overrides whatever the checkpoint carries for the TRM slot --
this is how the E4 sweep plugs in the zero-parameter foils in
``eval/baselines.py`` (``PersistenceTRM``, ``LinearExtrapolationTRM``) while
still using the checkpoint's trained fusion/drift/planner weights.

Perception defaults to the REAL ``YoloWorldPerception`` + ``ClipTaskEncoder``
(lazily imported, exactly like ``JEPALoop.build_real``) so a policy built
with no extra arguments is deploy-ready. Tests and the ``--mock-env`` CLI
path inject ``perception=`` / ``task_encoder=`` (typically
``MockYoloWorldPerception`` / ``MockTaskEncoder``) so the heavy
``ultralytics``/``torchvision`` imports never happen -- CPU-only, no
network, no downloads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.jepa.loop import JEPALoop
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.trm.interface import TRMBase
from microvla.trm.mock_trm import MockTRM
from preprocess.common import ActionNormalizer

logger = logging.getLogger(__name__)

#: Stage-B / stage-A checkpoint filenames, per train/train_full.py::save.
_STAGE_B_NAME = "full_stageB.pt"
_STAGE_A_NAME = "full_stageA.pt"


def _is_fresh_sentinel(checkpoint: Optional[str]) -> bool:
    """True when ``checkpoint`` means "no checkpoint" (``None`` or "none")."""
    return checkpoint is None or str(checkpoint).strip().lower() in ("", "none")


def _load_checkpoint_state(
    checkpoint: Optional[str], device: str
) -> tuple[Optional[dict], Optional[Path], bool]:
    """Resolves and loads a checkpoint per the stage-B-preferred fallback rule.

    Args:
        checkpoint: ``None``/``"none"``, a specific ``.pt`` file, or a
            directory containing ``full_stageB.pt`` / ``full_stageA.pt``.
        device: Torch device string to map the loaded tensors onto.

    Returns:
        ``(state_dict_or_None, path_used_or_None, is_stage_b)``. ``is_stage_b``
        is decided by CONTENT (presence of the ``"planner"`` key), never by
        filename, since a caller may point ``checkpoint`` at an arbitrarily
        named file.

    Raises:
        FileNotFoundError: If ``checkpoint`` is given but no checkpoint file
            can be found at the resolved location(s).
    """
    if _is_fresh_sentinel(checkpoint):
        return None, None, False

    path = Path(checkpoint)  # type: ignore[arg-type]
    if path.is_dir():
        candidates = [path / _STAGE_B_NAME, path / _STAGE_A_NAME]
    else:
        candidates = [path]
        if path.name == _STAGE_B_NAME:
            candidates.append(path.with_name(_STAGE_A_NAME))

    tried = []
    for i, cand in enumerate(candidates):
        tried.append(str(cand))
        if not cand.exists():
            continue
        state = torch.load(cand, map_location=device, weights_only=True)
        is_stage_b = "planner" in state
        if i > 0:
            logger.warning(
                "MicroVLAPolicy: preferred checkpoint %s not found; falling "
                "back to %s with a freshly initialized (UNTRAINED) planner.",
                candidates[0], cand,
            )
        elif not is_stage_b:
            logger.warning(
                "MicroVLAPolicy: checkpoint %s has no 'planner' state (a "
                "stage-A-only checkpoint); using a freshly initialized "
                "(UNTRAINED) planner.", cand,
            )
        return state, cand, is_stage_b

    raise FileNotFoundError(
        f"MicroVLAPolicy: no checkpoint found (tried: {tried}). Pass "
        f"checkpoint=None (or 'none') for freshly initialized modules."
    )


def _build_real_perception(device: str):
    """Lazily builds the real ``YoloWorldPerception`` + ``ClipTaskEncoder``.

    Mirrors ``JEPALoop.build_real``'s construction exactly. Only called when
    the caller did not inject ``perception=``/``task_encoder=`` -- so tests
    that inject mocks never trigger the ``ultralytics``/``torchvision``
    imports this needs.
    """
    from microvla.perception.text_encoder import ClipTaskEncoder
    from microvla.perception.yolo_world import YoloWorldPerception

    perception = YoloWorldPerception(device=device)
    task_encoder = ClipTaskEncoder(perception)
    return perception, task_encoder


class MicroVLAPolicy:
    """Closed-loop MicroVLA policy: raw RGB frames in, raw env actions out.

    Owns a :class:`~microvla.jepa.loop.JEPALoop` and maintains the
    real/dream tick schedule ITSELF from ``perception_period`` (a call
    counter, reset every :meth:`reset`) -- deliberately independent of
    ``cfg.tick_hz``/``cfg.real_frame_hz``, since the whole point of the E4
    perception-rate sweep is to vary this knob per run without touching the
    trained model's config.

    Attributes:
        telemetry: List of per-``act()`` call dicts for the CURRENT episode
            (cleared by :meth:`reset`), each ``{tick_index, is_real, trust,
            plan_norm}``. ``plan_norm`` is the L2 norm of the emitted
            (already trust-blended) ``[plan_steps, num_servos]`` plan tensor
            -- a compact per-tick magnitude diagnostic for the E5 trust/
            failure-prediction analysis, paired with ``trust`` in the same
            record.
        trust_trace: ``list[float]`` of ``corrector.trust`` values for the
            current episode, in call order (same length as ``telemetry``).
    """

    def __init__(
        self,
        checkpoint: Optional[str],
        norm_stats: str,
        cfg: Optional[MicroVLAConfig] = None,
        perception_period: int = 15,
        trm: Optional[TRMBase] = None,
        device: str = "cpu",
        perception=None,
        task_encoder=None,
    ) -> None:
        """Builds the policy.

        Args:
            checkpoint: ``None``/``"none"`` for fresh (untrained) modules, or
                a checkpoint file/directory -- see the module docstring.
            norm_stats: Path to the ``norm_stats.json`` paired with the
                checkpoint (``preprocess.common.ActionNormalizer.load``).
                Always required, even in fresh mode -- pass the identity
                normalizer shipped at ``eval/identity_norm_stats.json`` for
                smoke runs with no trained action distribution yet.
            cfg: Config override. Defaults to the checkpoint's saved config
                (``MicroVLAConfig(**state["cfg"])``) when a checkpoint is
                given, else ``DEFAULT_CONFIG``.
            perception_period: Ticks between REAL perceptions -- the sweep
                knob for E4. ``act()`` call ``i`` (0-indexed from the last
                :meth:`reset`) is real iff ``i % perception_period == 0``.
            trm: Optional ``TRMBase`` override (baselines: see
                ``eval/baselines.py``). When given, it is used verbatim and
                the checkpoint's own ``"trm"`` state is never loaded into
                it -- the override may not even share the real TRM's
                architecture.
            device: Torch device for every module (perception, fusion,
                drift, TRM, planner). Policy execution itself is CPU-cheap;
                heavier devices only matter for the real YOLO-World
                perception front-end.
            perception: Optional injected perception object (e.g.
                ``MockYoloWorldPerception``) -- skips the lazy real-model
                import entirely. Paired with ``task_encoder``.
            task_encoder: Optional injected task encoder (e.g.
                ``MockTaskEncoder``). If only one of ``perception``/
                ``task_encoder`` is given, the other is still built for real.
        """
        self.perception_period = max(1, int(perception_period))
        self.device = device
        self.normalizer = ActionNormalizer.load(norm_stats)

        state, ckpt_used, is_stage_b = _load_checkpoint_state(checkpoint, device)
        self.checkpoint_path = str(ckpt_used) if ckpt_used is not None else None
        self.is_stage_b = is_stage_b

        if cfg is None:
            cfg = MicroVLAConfig(**state["cfg"]) if state is not None else DEFAULT_CONFIG
        self.cfg = cfg

        fusion = SlotResonanceFusion(cfg)
        drift = AnchoredDriftEncoder(cfg)
        planner = ChronoQueryPlanner(cfg)

        trm_overridden = trm is not None
        if trm is None:
            if state is not None:
                from TRM import RecursiveTRM  # root-level file; torch-only import

                trm = RecursiveTRM(cfg, d=state.get("trm_d", 1024))
            else:
                logger.warning(
                    "MicroVLAPolicy: no checkpoint and no trm= override; "
                    "falling back to the MockTRM placeholder (no predictive "
                    "power -- see microvla/trm/mock_trm.py). Fine for a "
                    "harness smoke test, not for a meaningful eval."
                )
                trm = MockTRM(cfg)

        if state is not None:
            fusion.load_state_dict(state["fusion"])
            drift.load_state_dict(state["drift"])
            if not trm_overridden:
                trm.load_state_dict(state["trm"])
            if is_stage_b:
                planner.load_state_dict(state["planner"])

        fusion.to(device).eval()
        drift.to(device).eval()
        trm.to(device).eval()
        planner.to(device).eval()

        if perception is None or task_encoder is None:
            real_perception, real_task_encoder = _build_real_perception(device)
            perception = perception if perception is not None else real_perception
            task_encoder = task_encoder if task_encoder is not None else real_task_encoder

        self.loop = JEPALoop(
            cfg=cfg,
            task_encoder=task_encoder,
            perception=perception,
            fusion=fusion,
            drift=drift,
            trm=trm,
            planner=planner,
        )

        self.telemetry: list[dict] = []
        self.trust_trace: list[float] = []
        self._tick_index = 0

    def reset(self, instruction: str) -> None:
        """Starts a fresh episode: sets the task, clears per-episode state.

        Args:
            instruction: Natural-language task description.
        """
        self.loop.set_task(instruction)
        self._tick_index = 0
        self.telemetry = []
        self.trust_trace = []

    def act(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Advances one env step; returns a denormalized raw action.

        Every ``perception_period``-th call (0-indexed since the last
        :meth:`reset`) is a REAL tick: ``frame_rgb`` is converted RGB->BGR
        (the detector's native convention) and fed to
        ``JEPALoop.tick(frame_bgr)``. Every other call is a DREAM tick:
        ``frame_rgb`` is accepted (env-loop symmetry) but ignored, and
        ``JEPALoop.tick(None)`` drives the loop from the corrected TRM
        prediction instead.

        Args:
            frame_rgb: ``HxWx3`` uint8 RGB frame from the environment.

        Returns:
            ``[cfg.num_servos]`` float32 raw action
            (``ActionNormalizer.inverse`` of the planner's row-0 output).
        """
        is_real = self._tick_index % self.perception_period == 0
        frame_bgr = np.ascontiguousarray(frame_rgb[..., ::-1]) if is_real else None
        result = self.loop.tick(frame_bgr)

        plan = result.plan  # [plan_steps, num_servos], already trust-blended
        action = self.normalizer.inverse(plan[0].detach().cpu().numpy())

        self.telemetry.append({
            "tick_index": self._tick_index,
            "is_real": bool(result.is_real),
            "trust": float(result.trust),
            "plan_norm": float(plan.norm().item()),
        })
        self.trust_trace.append(float(result.trust))
        self._tick_index += 1

        return action.astype(np.float32)
