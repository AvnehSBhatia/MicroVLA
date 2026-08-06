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
        # Our checkpoints are repo-owned. torch>=2.6 defaults weights_only=True,
        # which rejects MicroVLA state dicts that embed numpy / cfg objects
        # (UnpicklingError "Unsupported operand 118" — observed on the pod IBVS
        # sweep, 3/3 workers dead, mean_success covering 0 tasks). Try the
        # safe path first; fall back for our own files.
        try:
            state = torch.load(cand, map_location=device, weights_only=True)
        except Exception:
            state = torch.load(cand, map_location=device, weights_only=False)
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


def _load_relaxed(module, sd, name: str) -> None:
    """Loads ``sd`` into ``module`` tolerantly, warning on architecture drift.

    A checkpoint trained before the v4 box head is BOTH missing keys (TRM
    ``box_head``, planner ``box_proj``) AND shape-mismatched (planner
    ``type_emb`` grew 4->5 rows). ``strict=False`` tolerates the former but
    still raises on the latter, so this:

      * loads every key whose shape matches,
      * PREFIX-COPIES a checkpoint tensor into a param that only grew along its
        leading dim (``type_emb[4]`` -> the first 4 rows of ``type_emb[5]``),
        preserving the trained rows and leaving the new row (the box type) at
        init,
      * drops anything else and leaves genuinely-new params (``box_head``,
        ``box_proj``) at their random init.

    So an OLD checkpoint still runs (video, eval, probe) before a retrain.
    Everything skipped/partial is logged loudly — the box conditioning is
    meaningless until the box head is trained.
    """
    model_sd = module.state_dict()
    to_load: dict = {}
    partial: list[str] = []
    dropped: list[str] = []
    for k, v in sd.items():
        if k not in model_sd:
            dropped.append(k)
            continue
        tgt = model_sd[k]
        if tgt.shape == v.shape:
            to_load[k] = v
        elif (tgt.dim() == v.dim() and tgt.shape[1:] == v.shape[1:]
              and v.shape[0] <= tgt.shape[0]):
            grown = tgt.clone()
            grown[: v.shape[0]] = v            # keep trained rows, new rows stay at init
            to_load[k] = grown
            partial.append(f"{k} {tuple(v.shape)}->{tuple(tgt.shape)}")
        else:
            dropped.append(k)
    result = module.load_state_dict(to_load, strict=False)
    new_at_init = list(result.missing_keys)
    if new_at_init or partial or dropped or result.unexpected_keys:
        logger.warning(
            "%s: checkpoint predates current architecture — RETRAIN to populate. "
            "at-init(new)=%s partial(prefix-copied)=%s dropped=%s unexpected=%s",
            name, new_at_init, partial, dropped, list(result.unexpected_keys),
        )


def _build_real_perception(device: str, det_conf: float = DEFAULT_CONFIG.det_conf,
                           role_disjoint_iou: float = DEFAULT_CONFIG.role_disjoint_iou,
                           source_max_area: float = DEFAULT_CONFIG.source_max_area,
                           source_min_aspect: float = DEFAULT_CONFIG.source_min_aspect):
    """Lazily builds the real ``YoloWorldPerception`` + ``ClipTaskEncoder``.

    Mirrors ``JEPALoop.build_real``'s construction exactly. Only called when
    the caller did not inject ``perception=``/``task_encoder=`` -- so tests
    that inject mocks never trigger the ``ultralytics``/``torchvision``
    imports this needs.

    ``det_conf`` comes from ``cfg.det_conf``, which the BAKE also reads
    (``preprocess/common.py``). It used to be 0.02 here against the detector
    class default 0.10 there — defect 26. The threshold decides which boxes
    exist, and each surviving box carries its confidence into fusion's
    ``box_weight`` fade, so a split threshold hands the deployed policy
    evidence weights that training never produced.
    """
    from microvla.perception.text_encoder import ClipTaskEncoder
    from microvla.perception.yolo_world import YoloWorldPerception

    # grid_size is what makes perceive() emit the coarse spatial grid. Without
    # it the deployed TQSA falls through to the raw full-resolution SPPF map
    # while the trainer feeds a g x g pooled, per-cell standardized one.
    perception = YoloWorldPerception(device=device, det_conf=float(det_conf),
                                     grid_size=DEFAULT_CONFIG.tqsa_grid,
                                     role_disjoint_iou=float(role_disjoint_iou),
                                     source_max_area=float(source_max_area),
                                     source_min_aspect=float(source_min_aspect))
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
        chunk_exec: bool = False,
        replan_every: int = 0,
        no_brake: bool = False,
        no_dream_correct: bool = False,
        trm: Optional[TRMBase] = None,
        device: str = "cpu",
        perception=None,
        task_encoder=None,
        zero_center_actions: bool = False,
        action_gain: float = 1.0,
        waypoint_stats: Optional[str] = None,
        heads_device: Optional[str] = None,
        waypoint_brake: bool = True,
        ibvs_gain: float = 0.0,
        ibvs_target_uv: tuple[float, float] = (0.5, 0.55),
        ibvs_conf_floor: float = 0.1,
        ibvs_sign: tuple[float, float, float] = (1.0, -1.0, 0.0),
        ibvs_descend: float = 0.0,
        clearance_gain: float = 0.0,
        clearance_radius: float = 0.35,
        clearance_lift: float = 0.0,
        clearance_aim_bias: float = 0.0,
        ibvs_phase: bool = False,
        ibvs_track_gate: float = 0.0,
        ibvs_clip_rerank: bool = False,
        ibvs_descend_hyst: float = 0.0,
        ibvs_swap_uv: bool = False,
        ibvs_center_first: bool = False,
        ibvs_center_tol: float = 0.06,
        ibvs_half_fill: float = 0.50,
        ibvs_grasp_offset: tuple[float, float] = (0.0, 0.0),
        ibvs_close_z: float = 0.06,
        ibvs_press: float = 0.0,
        ibvs_retry_rise: int = 1,
        ibvs_yaw_probe: bool = False,
        ibvs_yaw_sign: float = 1.0,
        ibvs_place_at: tuple[float, float] | None = None,
        ibvs_drop_z: float = 0.18,
        ibvs_gate_z: float = 0.06,
        ibvs_approach_z: float = 0.0,
        ibvs_gate_verify: bool = False,
        ibvs_body_v: float = 1.0,
        ibvs_held_jaw_min: float = 0.2,
        tool_phase: bool = False,
        tool_center_tol: float = 0.05,
        tool_gain: float = 1.0,
        tool_grasp_z: float = 0.06,
        tool_i_gain: float = 0.35,
        det_conf: float = DEFAULT_CONFIG.det_conf,
        role_disjoint_iou: float = DEFAULT_CONFIG.role_disjoint_iou,
        source_max_area: float = DEFAULT_CONFIG.source_max_area,
        source_min_aspect: float = DEFAULT_CONFIG.source_min_aspect,
        goal_ckpt: Optional[str] = None,
        goal_kwargs: Optional[dict] = None,
        prompt_instruction: Optional[str] = None,
        goal_src_rerank: bool = False,
        goal_src_proto: Optional[str] = None,
        goal_src_proto_key: Optional[str] = None,
        goal_src_bank: Optional[str] = None,
        goal_src_bank_key: Optional[str] = None,
        gates_ckpt: Optional[str] = None,
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
            heads_device: Device for fusion/drift/TRM/planner/TQSA. Defaults to
                CPU (unchanged behavior). Point it at the accelerator to move
                the per-tick cost off the CPU — the heads run at ``tick_hz``
                while the detector runs at ``real_frame_hz``, so on the 15:1
                schedule they, not the detector, dominate wall-clock.
            waypoint_stats: Path to the ``waypoint_stats.json`` fitted by
                ``preprocess/fit_waypoint_gain.py`` (v7.2). Given AND the
                checkpoint carries a waypoint head, the TRANSLATION dims of
                every emitted action come from a proportional move toward the
                predicted end-effector position measured against live proprio,
                instead of from the regressed plan; orientation and gripper
                still come from the plan. Without it (or without proprio at a
                given step) the plan drives all seven dims as before.
        """
        self.perception_period = max(1, int(perception_period))
        # Chunk execution advances the world model once per SAMPLE interval and
        # executes the plan's rows in between (paper.md 4w). replan_every
        # defaults to cfg.waypoint_row_stride, which is defined as the source
        # env's control rate / real_frame_hz -- exactly the number of env steps
        # one TRM step was trained to span.
        self.chunk_exec = bool(chunk_exec)
        self._replan_every_arg = int(replan_every)
        self._chunk = None
        self._chunk_pos = 0
        self.zero_center_actions = bool(zero_center_actions)
        # Multiplies the emitted POSE columns (gripper exempt — it is a hard
        # +/-1 decision, and scaling it would only move it toward the threshold).
        #
        # Why this exists: measured on real demos, LIBERO's passing band for
        # action magnitude is about [0.95, 1.05] — ground truth at 1.00 solves
        # 4/4, at 0.90 solves 0/4 (paper.md 4p). Every policy this project has
        # trained emits std_ratio 0.02-0.56, i.e. 2-50x below that band, because
        # MSE behaviour cloning converges to the conditional mean and therefore
        # shrinks. This is the eval-time test of whether the DIRECTION is right
        # and only the SCALE is wrong; it is a diagnostic, not a fix. The fix is
        # a magnitude term in the stage-B objective.
        self.action_gain = float(action_gain)
        # Zero-training visual-servo residual (paper.md 5k). ``0`` disables.
        # Falsifies the "frozen encoder is the ceiling" claim: if a P-controller
        # on the detected source center moves success, the features carry the
        # signal and the learned objective is the bottleneck.
        self.ibvs_gain = float(ibvs_gain)
        self.ibvs_target_uv = (float(ibvs_target_uv[0]), float(ibvs_target_uv[1]))
        self.ibvs_conf_floor = float(ibvs_conf_floor)
        self.ibvs_sign = tuple(float(v) for v in ibvs_sign)
        self.ibvs_descend = float(ibvs_descend)
        self.clearance_gain = float(clearance_gain)
        self.clearance_radius = float(clearance_radius)
        self.clearance_lift = float(clearance_lift)
        self.clearance_aim_bias = float(clearance_aim_bias)
        # Phased mode OWNS the action (servo/grasp/lift/place state machine)
        # instead of adding a residual — see eval/ibvs_phase.py for why
        # blending can never work when the prior skips the grasp phase.
        self.ibvs_machine = None
        self.tool_machine = None
        self._phase_action = None
        # Structured control (microvla/control): learned world goals + the
        # teacher's servo shell. UNAIDED — no calibrated constants beyond arm
        # physics; the task content (where to grasp / place) comes from the
        # trained goal heads. Takes the same replace-wholesale action slot as
        # the assisted machines and is mutually exclusive with them.
        self.goal_machine = None
        # Diagnostic only (see JEPALoop.set_task): when set, detection prompts
        # come from THIS string while the task embeddings come from the
        # episode's real instruction, isolating the two channels one
        # instruction normally drives together.
        self.prompt_instruction = prompt_instruction or None
        self.goal_src_rerank = bool(goal_src_rerank)
        # Visual role prototype (the cream lesson): a unit vector built from
        # the corpus's own grasped-box embeddings, used to bind the source box
        # by cosine when the detector's TEXT tower cannot separate look-alikes.
        self.goal_src_proto = None
        self.goal_src_proto_gmean = None
        if goal_src_proto:
            protos = torch.load(goal_src_proto, map_location="cpu",
                                weights_only=False)
            gm = protos.pop("_global_mean", None)
            if gm is not None:
                gm = torch.as_tensor(gm, dtype=torch.float32).reshape(-1)
                self.goal_src_proto_gmean = gm / (gm.norm() + 1e-8)
            if goal_src_proto_key:
                vec = protos[goal_src_proto_key]
            elif len(protos) == 1:
                vec = next(iter(protos.values()))
            else:
                raise ValueError(
                    "goal_src_proto holds several prototypes "
                    f"({sorted(protos)}) — pass goal_src_proto_key")
            vec = torch.as_tensor(vec, dtype=torch.float32).reshape(-1)
            self.goal_src_proto = vec / (vec.norm() + 1e-8)
        # Bank binder: 1-NN over the corpus's own centered crop embeddings.
        # Measured 0.902 leave-episode-out identity accuracy vs 0.613 for the
        # mean prototype — the crops are multimodal and averaging destroys it.
        self.goal_src_bank = None
        self.goal_src_bank_others = None
        if goal_src_bank:
            banks = torch.load(goal_src_bank, map_location="cpu",
                               weights_only=False)
            gm = banks.pop("_global_mean", None)
            if gm is not None:
                gm = torch.as_tensor(gm, dtype=torch.float32).reshape(-1)
                self.goal_src_proto_gmean = gm / (gm.norm() + 1e-8)
            key = goal_src_bank_key or (next(iter(banks)) if len(banks) == 1
                                        else None)
            if key is None:
                raise ValueError(f"goal_src_bank holds {sorted(banks)} — "
                                 "pass goal_src_bank_key")
            self.goal_src_bank = torch.as_tensor(banks[key],
                                                 dtype=torch.float32)
            others = [torch.as_tensor(v, dtype=torch.float32)
                      for k, v in banks.items() if k != key]
            self.goal_src_bank_others = (torch.cat(others, dim=0)
                                         if others else None)
        self.goal_grasp_head = None
        self.goal_place_head = None
        if goal_ckpt:
            if tool_phase or ibvs_phase:
                raise ValueError(
                    "goal_ckpt (structured control) is mutually exclusive "
                    "with --ibvs-phase/--tool-phase assist modes")
            from microvla.control import GoalServoMachine, load_goal_heads
            self.goal_grasp_head, self.goal_place_head, _meta = \
                load_goal_heads(goal_ckpt)
            gkw = dict(goal_kwargs or {})
            if gates_ckpt:
                # De-skeletonization stage 1: learned close-trigger + hold
                # gates replace their thresholds (per-gate ablatable).
                from microvla.control.gate_heads import load_gates
                close, hold, gmeta = load_gates(gates_ckpt)
                gkw["gates"] = {"close": close, "hold": hold}
                logger.info("MicroVLAPolicy: learned gates ON (%s)", gmeta)
            self.goal_machine = GoalServoMachine(**gkw)
            logger.info("MicroVLAPolicy: structured goal-servo control ON "
                        "(heads: %s)", goal_ckpt)
        if tool_phase:
            # Accuracy tools OWN the action (reach_center → descend → grasp…).
            # Prefer this over residual IBVS when the miss is "almost centred".
            from microvla.tools import GraspToolController
            self.tool_machine = GraspToolController(
                gain=float(tool_gain),
                sign=(float(ibvs_sign[0]), float(ibvs_sign[1])),
                descend_rate=float(ibvs_descend) if ibvs_descend != 0.0 else -0.4,
                grasp_uv=self.ibvs_target_uv,
                conf_floor=self.ibvs_conf_floor,
                center_tol=float(tool_center_tol),
                grasp_z=float(tool_grasp_z),
                i_gain=float(tool_i_gain),
            )
        elif ibvs_phase:
            from eval.ibvs_phase import PhasedIBVS
            self.ibvs_machine = PhasedIBVS(
                gain=self.ibvs_gain, sign=self.ibvs_sign,
                descend=self.ibvs_descend, target_uv=self.ibvs_target_uv,
                conf_floor=self.ibvs_conf_floor,
                track_gate=float(ibvs_track_gate),
                clip_rerank=bool(ibvs_clip_rerank),
                descend_hyst=float(ibvs_descend_hyst),
                swap_uv=bool(ibvs_swap_uv),
                center_first=bool(ibvs_center_first),
                center_tol=float(ibvs_center_tol),
                half_fill=float(ibvs_half_fill),
                grasp_offset=(float(ibvs_grasp_offset[0]),
                              float(ibvs_grasp_offset[1])),
                close_z=float(ibvs_close_z),
                press=float(ibvs_press),
                retry_rise=int(ibvs_retry_rise),
                yaw_probe=bool(ibvs_yaw_probe),
                yaw_sign=float(ibvs_yaw_sign),
                place_at=(None if ibvs_place_at is None
                          else (float(ibvs_place_at[0]),
                                float(ibvs_place_at[1]))),
                drop_z=float(ibvs_drop_z),
                gate_z=float(ibvs_gate_z),
                approach_z=float(ibvs_approach_z),
                gate_verify=bool(ibvs_gate_verify),
                body_v=float(ibvs_body_v),
                held_jaw_min=float(ibvs_held_jaw_min))
        self.device = device
        # Perception runs on `device` and detaches its outputs to CPU. The heads
        # used to be pinned to CPU with them, which was right when they were
        # tiny — but the d=1024 TRM alone is 9.97M params and runs on EVERY
        # 30 Hz tick, while the detector runs on 1 tick in 15. So `--device
        # cuda:0` was leaving the dominant cost on the CPU. `heads_device`
        # makes that a choice; the loop moves perception's tensors at the
        # boundary (JEPALoop.device), so the two may differ freely.
        heads_device = heads_device or "cpu"
        self.normalizer = ActionNormalizer.load(norm_stats)

        state, ckpt_used, is_stage_b = _load_checkpoint_state(checkpoint, heads_device)
        self.checkpoint_path = str(ckpt_used) if ckpt_used is not None else None
        self.is_stage_b = is_stage_b

        if cfg is None:
            cfg = MicroVLAConfig(**state["cfg"]) if state is not None else DEFAULT_CONFIG
        if no_brake:
            # brake_trust = 0 disables the delta-mode brake everywhere (the loop
            # and the waypoint path both guard on `brake_trust > 0`).
            #
            # Why this switch exists: the brake scales the plan by
            # min(1, tau / brake_trust), and deployment telemetry measured trust
            # mean 0.568 with a minimum of 0.216 against brake_trust 0.5 -- a
            # scale as low as 0.43. paper.md 4p measured the task's tolerance on
            # action magnitude at ~[0.95, 1.05]. So every step with trust < 0.475
            # is braked OUTSIDE the band the task can pass, by construction. The
            # brake was designed to stop drift compounding and is measured to
            # cost more than it saves here.
            import dataclasses as _dc
            cfg = _dc.replace(cfg, brake_trust=0.0)
        self.cfg = cfg
        # Resolved here because it reads cfg: one TRM step was trained to span
        # cfg.waypoint_row_stride env steps (the source control rate divided by
        # real_frame_hz), which is the correct re-planning interval.
        # The chunk holds plan_steps NATIVE-rate actions, so it can only cover
        # plan_steps env steps. Defaulting purely to waypoint_row_stride (10 on
        # the 2 Hz corpus) mapped chunk positions 0..9 onto rows 0,1,2,3,4,4,4,4,4,4
        # -- row 4 executed for 60% of steps. In a DELTA action space that is not
        # "hold position", it is that delta commanded six more times, i.e. a
        # large over-command exactly when the arm should be settling.
        _stride = max(1, int(getattr(cfg, "waypoint_row_stride", 1)))
        self.replan_every = (self._replan_every_arg if self._replan_every_arg > 0
                             else max(1, min(_stride, int(cfg.plan_steps))))

        from microvla.perception.spatial_adapter import TextQueriedSpatialAdapter

        # v8 is detected from the checkpoint's own keys, never a flag: loading
        # v8 weights into v7 modules silently yields a random stack.
        self.is_v8 = state is not None and "relational" in state
        if self.is_v8:
            from microvla.relational import RelationalHead
            from microvla.v8 import DriftAdapter, FusionAdapter

            fusion, drift = FusionAdapter(cfg), DriftAdapter(cfg)
            relational = RelationalHead(cfg)
        else:
            fusion = SlotResonanceFusion(cfg)
            drift = AnchoredDriftEncoder(cfg)
            relational = None
        planner = ChronoQueryPlanner(cfg)
        tqsa = TextQueriedSpatialAdapter(cfg)

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
            _load_relaxed(fusion, state["fusion"], "fusion")
            _load_relaxed(drift, state["drift"], "drift")
            if not trm_overridden:
                _load_relaxed(trm, state["trm"], "trm")
            if is_stage_b:
                _load_relaxed(planner, state["planner"], "planner")
            if relational is not None:
                _load_relaxed(relational, state["relational"], "relational")
                logger.info("MicroVLAPolicy: v8 stack (relational head active)")
            if "tqsa" in state:
                _load_relaxed(tqsa, state["tqsa"], "tqsa")
            else:
                # A random-init adapter is WORSE than no adapter: the planner
                # would take 22 of its ~82 memory tokens from untrained
                # perception, and a planner trained without spatial has never
                # seen that input at all (its spat/heat projections are also at
                # init). Disable it and let the planner run the way it trained.
                tqsa = None
                logger.warning(
                    "MicroVLAPolicy: checkpoint carries no TQSA weights (pre-v7, "
                    "or stage B trained without --tqsa) — running WITHOUT the "
                    "spatial adapter rather than feeding the planner a "
                    "random-init one. Retrain stage B with --tqsa to use it."
                )

        fusion.to(heads_device).eval()
        drift.to(heads_device).eval()
        trm.to(heads_device).eval()
        planner.to(heads_device).eval()
        if relational is not None:
            # Missing this made EVERY v8 closed-loop run fail at the first tick
            # with "mat1 is on cuda:0, different from other tensors on cpu", and
            # the harness reported it as tasks_completed 0 / mean_success 0.000
            # — indistinguishable in the summary from a policy that simply never
            # succeeds.
            relational.to(heads_device).eval()
        if tqsa is not None:
            tqsa.to(heads_device).eval()

        if perception is None or task_encoder is None:
            real_perception, real_task_encoder = _build_real_perception(
                device, det_conf=float(det_conf),
                role_disjoint_iou=float(role_disjoint_iou),
                source_max_area=float(source_max_area),
                source_min_aspect=float(source_min_aspect))
            perception = perception if perception is not None else real_perception
            task_encoder = task_encoder if task_encoder is not None else real_task_encoder

        # LoRA-adapted frame-embedding stage (paper §6.4d, capacity probe):
        # rebuild the adapted SPPF clone from the checkpoint and let
        # perceive() route frame_emb through it. Only applies to the real
        # backbone — mocks have no SPPF and ignore the state.
        if state is not None and "yolo_lora" in state and hasattr(perception, "enable_lora"):
            meta = state.get("yolo_lora_meta", {"r": 8, "alpha": 8.0})
            adapted = perception.enable_lora(int(meta["r"]), float(meta["alpha"]))
            adapted.load_state_dict(state["yolo_lora"])
            adapted.eval()
            logger.info("MicroVLAPolicy: YOLO LoRA frame-embedding stage loaded "
                        "(r=%s alpha=%s)", meta["r"], meta["alpha"])
        # A goal-head checkpoint may carry its OWN adapted embedding stage
        # (train_goal --yolo-lora-r): apply it the same way. Mutually
        # exclusive with a main-checkpoint LoRA by precedence: the goal
        # ckpt's adapter wins in structured mode (its head trained on it).
        if (goal_ckpt and self.goal_grasp_head is not None
                and hasattr(perception, "enable_lora")):
            g_extra = torch.load(goal_ckpt, map_location="cpu",
                                 weights_only=False)
            if "yolo_lora" in g_extra:
                gm = g_extra.get("yolo_lora_meta", {"r": 8, "alpha": 8.0})
                adapted = perception.enable_lora(int(gm["r"]),
                                                 float(gm["alpha"]))
                adapted.load_state_dict(g_extra["yolo_lora"])
                adapted.eval()
                logger.info("MicroVLAPolicy: goal-ckpt LoRA embedding stage "
                            "loaded (r=%s alpha=%s)", gm["r"], gm["alpha"])

        self.loop = JEPALoop(
            cfg=cfg,
            task_encoder=task_encoder,
            perception=perception,
            fusion=fusion,
            drift=drift,
            trm=trm,
            planner=planner,
            tqsa=tqsa,
            relational=relational,
        )
        # Deployment-only recurrent state, made optional. The trainer's dream
        # regime feeds standardize(trm(...)) with NO innovation term -- nothing
        # in train_batched.py instantiates a corrector -- so with this on, every
        # dream tick hands the planner inputs it never trained against
        # (measured rel-diff: fused 0.15-0.22, wm_delta 0.12-0.13). At
        # perception_period 2 that is half of all ticks.
        self.loop.dream_correction = not bool(no_dream_correct)

        # v7.2 waypoint actuation: only ever active when BOTH the checkpoint
        # has the head and a fitted gain was supplied. A gain fitted under a
        # different action normalization is meaningless, so it is paired with
        # the checkpoint exactly like norm_stats.json.
        self.actuator = None
        self.waypoint_brake = bool(waypoint_brake)
        if waypoint_stats:
            from microvla.utils.waypoint import WaypointActuator, WaypointGain

            if not cfg.waypoint_action:
                logger.warning(
                    "MicroVLAPolicy: --waypoint-stats given but this checkpoint "
                    "has no waypoint head (cfg.waypoint_action=False) — actions "
                    "come from the plan as usual. Retrain with "
                    "`train_batched.py --waypoint-weight 1.0` to use it."
                )
            else:
                self.actuator = WaypointActuator(
                    WaypointGain.load(waypoint_stats),
                    waypoint_range=cfg.waypoint_range,
                    horizon=cfg.waypoint_horizon,
                    gain_scale=cfg.waypoint_gain_scale,
                    row_stride=cfg.waypoint_row_stride,
                )
                logger.info("MicroVLAPolicy: waypoint actuation ON (gain %s)",
                            self.actuator.gain)

        self.telemetry: list[dict] = []
        self.trust_trace: list[float] = []
        self._tick_index = 0

    def reset(self, instruction: str) -> None:
        """Starts a fresh episode: sets the task, clears per-episode state.

        Args:
            instruction: Natural-language task description.
        """
        self.loop.set_task(instruction, prompt_text=self.prompt_instruction)
        self._tick_index = 0
        self._chunk = None            # cached plan/waypoints for chunk execution
        self._chunk_pos = 0
        self.telemetry = []
        self.trust_trace = []
        if self.actuator is not None:
            self.actuator.reset()
        self._phase_action = None
        if self.ibvs_machine is not None:
            self.ibvs_machine.reset()
        if self.tool_machine is not None:
            self.tool_machine.reset()
        if self.goal_machine is not None:
            self.goal_machine.reset()
            # The place point is a per-task constant: latch it once, now.
            task = getattr(self.loop, "_task", None)
            cmd = getattr(task, "command_emb", None) if task is not None else None
            if cmd is not None:
                with torch.no_grad():
                    pred = self.goal_place_head(
                        torch.as_tensor(cmd, dtype=torch.float32).reshape(1, -1))
                self.goal_machine.set_place(pred["xy"][0].cpu().numpy())

    def _emit(self, result, row: int, proprio, fresh: bool) -> np.ndarray:
        """Turns plan ``row`` of ``result`` into a raw action for this step.

        Factored out so chunk execution and the per-tick schedule share ONE
        actuation path. Two sides of a pair computing the same quantity
        separately is the defect shape that produced paper.md 4t-4v; there is no
        reason to reproduce it here.
        """
        action = self.normalizer.inverse(
            result.plan[row].detach().cpu().numpy(),
            zero_center=self.zero_center_actions)
        if self.action_gain != 1.0:
            action = action.copy()
            action[:-1] = action[:-1] * self.action_gain
        wp_cmd = None
        if (self.actuator is not None and result.waypoints is not None
                and proprio is not None
                and float(np.asarray(proprio).reshape(-1)[-1]) > 0.5):
            wp_cmd = self.actuator.command(
                result.waypoints.detach().cpu().numpy(),
                np.asarray(proprio, dtype=np.float64).reshape(-1)[:3],
                is_real=bool(fresh))
            if (self.waypoint_brake and self.cfg.action_space == "delta"
                    and self.cfg.brake_trust > 0.0):
                wp_cmd = wp_cmd * min(1.0, float(result.trust) / self.cfg.brake_trust)
            action[: wp_cmd.shape[0]] = wp_cmd
        ibvs_cmd = None
        if (result.perception is not None
                and self.ibvs_machine is None and self.tool_machine is None
                and float(result.perception.source.confidence) >= self.ibvs_conf_floor):
            from microvla.utils.ibvs import (
                clearance_aim_shift, clearance_residual, ibvs_residual,
            )
            src_c = result.perception.source.center
            if torch.is_tensor(src_c):
                src_c = src_c.detach().cpu().numpy()
            # Neighbours = proposals that are not the source box (IoU / center).
            neigh = []
            sc = np.asarray(src_c, dtype=np.float64).reshape(-1)[:2]
            for p in getattr(result.perception, "proposals", ()) or ():
                if float(getattr(p, "confidence", 0.0)) < self.ibvs_conf_floor:
                    continue
                pcen = p.center
                if torch.is_tensor(pcen):
                    pcen = pcen.detach().cpu().numpy()
                pc = np.asarray(pcen, dtype=np.float64).reshape(-1)[:2]
                if float(np.linalg.norm(pc - sc)) < 0.04:
                    continue  # same box as source
                neigh.append(pc)
            target_uv = self.ibvs_target_uv
            if self.clearance_aim_bias > 0.0 and neigh:
                target_uv = clearance_aim_shift(
                    sc, neigh, target_uv,
                    bias=self.clearance_aim_bias,
                    radius=self.clearance_radius,
                )
            if self.ibvs_gain > 0.0:
                ibvs_cmd = ibvs_residual(
                    src_c,
                    float(result.perception.source.confidence),
                    gain=self.ibvs_gain,
                    sign=self.ibvs_sign,
                    descend=self.ibvs_descend,
                    target_uv=target_uv,
                    conf_floor=self.ibvs_conf_floor,
                    action_dim=int(action.shape[0]),
                )
                if ibvs_cmd is not None:
                    action = action.copy()
                    action[:3] = action[:3] + ibvs_cmd[:3]
            if self.clearance_gain > 0.0 and neigh:
                clr = clearance_residual(
                    sc, neigh,
                    gain=self.clearance_gain,
                    radius=self.clearance_radius,
                    sign=self.ibvs_sign,
                    lift=self.clearance_lift,
                    action_dim=int(action.shape[0]),
                )
                if clr is not None:
                    action = action.copy()
                    action[:3] = action[:3] + clr[:3]
                    ibvs_cmd = clr if ibvs_cmd is None else (ibvs_cmd + clr)
        if self.goal_machine is not None:
            # Structured control: on REAL ticks feed the grasp head's world-
            # goal estimate to the machine (evidence admission); EVERY tick
            # the machine servos on proprio alone — vision never steers
            # directly, exactly the teacher's information diet.
            if result.perception is not None and proprio is not None:
                src = result.perception.source
                # Student-side semantic rebinding (the cream lesson, addendum
                # 2026-08-05): for look-alike objects the detector-text match
                # flickers between boxes (uv std ~0.3 every trial) exactly as
                # the teacher's did pre-rerank; re-pick the source by crop-emb
                # vs the SOURCE phrase, rejecting better-target matches — the
                # same zero-training lever the teacher used to cross it.
                if self.goal_src_bank is not None:
                    props = getattr(result.perception, "proposals", ()) or ()
                    best, best_s = None, float("-inf")
                    for p in props:
                        if float(getattr(p, "confidence", 0.0)) < self.ibvs_conf_floor:
                            continue
                        e = torch.as_tensor(
                            p.emb, dtype=torch.float32).detach().cpu().reshape(-1)
                        if e.numel() != self.goal_src_bank.shape[1]:
                            continue
                        e = e / (e.norm() + 1e-8)
                        g = self.goal_src_proto_gmean
                        if g is not None:
                            e = e - float((e * g).sum()) * g
                            e = e / (e.norm() + 1e-8)
                        s = float((self.goal_src_bank @ e).max())
                        if self.goal_src_bank_others is not None:
                            s -= float((self.goal_src_bank_others @ e).max())
                        if s > best_s:
                            best, best_s = p, s
                    if best is not None:
                        src = best
                elif self.goal_src_proto is not None:
                    props = getattr(result.perception, "proposals", ()) or ()
                    best, best_s = None, float("-inf")
                    for p in props:
                        if float(getattr(p, "confidence", 0.0)) < self.ibvs_conf_floor:
                            continue
                        e = torch.as_tensor(
                            p.emb, dtype=torch.float32).detach().cpu().reshape(-1)
                        if e.numel() != self.goal_src_proto.numel():
                            continue
                        e = e / (e.norm() + 1e-8)
                        if self.goal_src_proto_gmean is not None:
                            g = self.goal_src_proto_gmean
                            e = e - float((e * g).sum()) * g
                            e = e / (e.norm() + 1e-8)
                        s = float((e * self.goal_src_proto).sum())
                        if s > best_s:
                            best, best_s = p, s
                    if best is not None:
                        src = best
                elif self.goal_src_rerank:
                    task = getattr(self.loop, "_task", None)
                    props = getattr(result.perception, "proposals", ()) or ()
                    if task is not None and getattr(task, "source_emb", None) is not None and props:
                        from eval.ibvs_phase import clip_rerank_box
                        reb = clip_rerank_box(
                            props, task.source_emb,
                            reject_emb=getattr(task, "target_emb", None))
                        if reb is not None:
                            src = reb
                conf = float(src.confidence)
                pvec = np.asarray(proprio, dtype=np.float64).reshape(-1)
                if (conf >= self.ibvs_conf_floor and pvec.shape[0] >= 10
                        and pvec[-1] > 0.5):
                    from microvla.control import (GraspPointHead,
                                                  build_grasp_features)

                    def _npv(x):
                        return (x.detach().cpu().numpy() if torch.is_tensor(x)
                                else np.asarray(x))

                    feats = build_grasp_features(
                        uv=_npv(src.center), conf=conf, proprio=pvec,
                        box_emb=_npv(src.emb),
                        frame_emb=_npv(result.perception.frame_emb))
                    with torch.no_grad():
                        pred = self.goal_grasp_head(feats)
                    self.goal_machine.observe(
                        pred["xy"][0].cpu().numpy(), float(pred["z"][0, 0]),
                        float(GraspPointHead.sigma(pred)[0, :2].max()))
            action = self.goal_machine.step(
                proprio, action_dim=int(action.shape[0]))
            ibvs_cmd = [float(v) for v in action[:3]]
        elif self.tool_machine is not None:
            if result.perception is not None:
                action = self.tool_machine.step(
                    result.perception.source, result.perception.target,
                    proprio, action_dim=int(action.shape[0]))
                self._phase_action = action
            elif self._phase_action is not None:
                action = self._phase_action
            ibvs_cmd = [float(v) for v in action[:3]]
        elif self.ibvs_machine is not None:
            # Phased mode REPLACES the action. On held-perception ticks the
            # machine still steps (confidence is already staleness-faded, so
            # a stale box decays below the floor on its own).
            if result.perception is not None:
                task = getattr(self.loop, "_task", None)
                action = self.ibvs_machine.step(
                    result.perception.source, result.perception.target,
                    proprio, action_dim=int(action.shape[0]),
                    proposals=getattr(result.perception, "proposals", ()) or (),
                    source_emb=None if task is None else task.source_emb,
                    target_emb=None if task is None else task.target_emb)
                self._phase_action = action
            elif self._phase_action is not None:
                action = self._phase_action
            ibvs_cmd = [float(v) for v in action[:3]]
        self.telemetry.append({
            "tick_index": self._tick_index,
            "is_real": bool(fresh),
            "chunk_row": int(row),
            "trust": float(result.trust),
            "plan_norm": float(result.plan.norm().item()),
            "waypoint_cmd": None if wp_cmd is None else [float(v) for v in wp_cmd],
            "ibvs_cmd": None if ibvs_cmd is None else [float(v) for v in ibvs_cmd[:3]],
            "action": [float(v) for v in action],
            "eef": (None if proprio is None
                    else [float(v) for v in np.asarray(proprio).reshape(-1)[:3]]),
            "jaws": (None if proprio is None
                     else [float(v) for v in np.asarray(proprio).reshape(-1)[7:9]]),
            **({} if result.perception is None else {
                "src_conf": float(result.perception.source.confidence),
                "tgt_conf": float(result.perception.target.confidence),
                "src_center": [float(v) for v in result.perception.source.center],
            }),
            **({} if self.ibvs_machine is None
               else {"phase": self.ibvs_machine.phase}),
            **({} if self.tool_machine is None
               else {"phase": self.tool_machine.phase,
                     "tool": self.tool_machine.last_tool}),
            **({} if self.goal_machine is None
               else {"phase": self.goal_machine.phase,
                     "attempt": self.goal_machine.attempt,
                     "base_tgt": (None if self.goal_machine.base_tgt is None
                                  else [float(v)
                                        for v in self.goal_machine.base_tgt])}),
        })
        self.trust_trace.append(float(result.trust))
        self._tick_index += 1
        return action.astype(np.float32)

    def act(self, frame_rgb: np.ndarray, proprio: np.ndarray | None = None) -> np.ndarray:
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
            proprio: Optional ``[10]`` arm-state vector (v6, see
                ``microvla/utils/proprio.py``) — pass every step when the env
                exposes it (``proprio_from_obs``); encoders are fast, so it is
                fresh even on dream ticks. ``None`` -> the loop holds/zeros.

        Returns:
            ``[cfg.num_servos]`` float32 raw action
            (``ActionNormalizer.inverse`` of the planner's row-0 output).
        """
        # ---- chunk execution: advance the world model at its TRAINED dt -----
        # One TRM step is trained to predict the next SAMPLED frame, which is
        # cfg.waypoint_row_stride env steps ahead (LIBERO: 20 Hz / real_frame_hz
        # 2 = 10 steps = 0.5 s). The default schedule steps it once per env step
        # and dreams 14 times between real frames, so it extrapolates ~7 s of
        # predicted time per 0.7 s elapsed -- a 10x temporal overshoot,
        # compounded 14 times (paper.md 4w).
        #
        # With chunk execution the loop advances ONCE per sample interval and
        # the plan's rows -- which the bake defines as "the next plan_steps
        # NATIVE-rate actions" -- are executed in between, which is what an
        # action chunk is for. The waypoint command is still recomputed every
        # step against fresh proprio, so this stays closed-loop in position even
        # though the latent advances at the sample rate.
        if self.chunk_exec:
            if self._chunk is None or self._chunk_pos >= self.replan_every:
                frame_bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
                self._chunk = self.loop.tick(frame_bgr, proprio=proprio)
                self._chunk_pos = 0
            result = self._chunk
            # Rows past the chunk repeat the last one, matching the bake, which
            # pads a short chunk by repeating its final action.
            row = min(self._chunk_pos, result.plan.shape[0] - 1)
            self._chunk_pos += 1
            return self._emit(result, row, proprio, fresh=(row == 0))

        is_real = self._tick_index % self.perception_period == 0
        frame_bgr = np.ascontiguousarray(frame_rgb[..., ::-1]) if is_real else None
        result = self.loop.tick(frame_bgr, proprio=proprio)
        # One actuation path. The previous copy here drifted from `_emit`
        # (IBVS, telemetry keys) — same defect shape as paper.md 4t-4v.
        return self._emit(result, 0, proprio, fresh=bool(result.is_real))
