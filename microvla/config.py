"""Canonical configuration for the MicroVLA v2 stack.

Every dimension that crosses a module boundary lives here. Modules must read
these fields instead of hardcoding numbers so the TRM slot, fusion, drift
encoder, planner, and JEPA loop always agree on shapes.

v2 changes vs v1:
    - MiniLM removed. Text comes from YOLO-World's internal CLIP text tower
      (once per task), so ``text_dim`` is 512 and there are 3 ordered text
      tokens: command, source phrase, target phrase.
    - Fused matrix widened to 32x5 (five columns of 32) per user spec.
    - Budget reinvested: TRM reserved at 10M; trainable heads scaled up to a
      9M cap (fusion ~4.5M, drift ~0.9M, planner ~1.6M).
    - JEPA latent rollout: 30 Hz ticks, real YOLO perception at 2 Hz, the
      other ticks feed the TRM's corrected prediction back into fusion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MicroVLAConfig:
    """Shared hyperparameters and interface dimensions.

    Attributes:
        text_dim: CLIP text-tower embedding size (YOLO-World's own text
            branch; harvested once per task at ``set_classes`` time).
        n_text_tokens: Ordered text tokens fed to fusion: command, source
            noun phrase, target noun phrase.
        vis_dim: Channel count of the hooked YOLO-World-S SPPF (P5) feature
            map; frame embedding, both box embeddings, and the TRM's
            predicted next-frame embedding all use it.
        state_dim: Width of the drift encoder's state-delta code fed to the
            TRM alongside the fused matrix.
        fused_rows: Rows of the fusion output matrix (32 slots).
        fused_cols: Columns of the fusion output matrix.
        plan_steps: Sequential PWM updates per plan (rows of the plan).
        num_servos: Servo channels per update (columns of the plan).
        d_model: Token width inside the fusion module.
        d_plan: Token width inside the planner.
        n_heads: Attention heads used by fusion and planner blocks.
        n_fusion_blocks: Cross-attention rounds the fusion slots run.
        n_planner_blocks: Cross-attention rounds the planner queries run.
        n_fourier: Frequency pairs for the Fourier encoding of each box
            center (source, target, and their relative displacement).
        modality_dropout: Train-time probability of FADING the box evidence
            (box tokens + geometry) by a random factor in [0, 1) — the same
            evidence-weighting path JEPA dream ticks use with stale,
            confidence-decayed boxes, so keep it > 0 when training.
        tick_hz: Control-loop rate. Every tick produces a plan.
        real_frame_hz: Rate of real YOLO perception; ticks in between are
            dream ticks driven by the corrected TRM prediction.
        correction_beta: EMA factor for the innovation (drift-correction)
            vector accumulated at each real frame.
        correction_decay: Per-dream-tick decay of the applied correction.
        staleness_decay: Per-dream-tick decay of the held (last-real) box
            evidence weights fed to fusion during dreams.
        trust_temperature: Sharpness of the corrector's self-calibrating
            error-ratio -> trust mapping (tau = exp(-0.5 * ratio^2 *
            temperature / 4); default 4 gives tau ~= 0.61 at a typical-sized
            innovation, -> 1 when tracking well, -> 0 when diverged).
        context_window: Length K of the rolling context windows: the drift
            encoder's memory of recent REAL-frame embeddings (K frames at
            real_frame_hz = 4 s of state-change context) and the JEPA loop's
            window of recent tick latents passed to the TRM.
        drift_horizons: Lag offsets (in real frames) the drift encoder
            compares the current embedding against, in addition to the
            episode anchor — multi-timescale state change.
        trainable_param_budget: Hard cap on fusion + drift + planner params.
            Ledger: 32M total - ~13M frozen YOLO-World-S - 10M reserved TRM
            leaves ~9M for the trainable heads (see utils/param_audit.py).
    """

    action_space: str = "delta"  # "delta": zero action = no motion (LIBERO/Bridge
    # EEF deltas) -> low corrector trust BRAKES the plan toward zero. "absolute":
    # commands are absolute targets (the Pi's PWM rig; zero = servo mid-range)
    # -> low trust HOLD-blends toward the previous plan, never toward zero.
    miss_hold: bool = False  # hold a role's last-known box when the detector
    # MISSES on a real tick, instead of passing the weight-0 fallback. OFF by
    # default because the corpus bakes a miss as ZERO evidence and the trainer
    # feeds exactly that, so holding is a deployment-only behaviour the policy
    # was never trained to read: it turns "I see nothing" into a confident stale
    # box (measured weights 0.156/0.109/0.077 on ticks the corpus zeroed).
    # Measured cost of leaving it on: paper.md 4v. Enabling it again requires
    # baking held evidence into the corpus too, so both sides agree.
    miss_decay: float = 0.7  # per missed REAL frame, when miss_hold is on:
    # weight decay on the held last-known box (objects don't teleport because
    # the detector blinked at the grasp moment).
    brake_trust: float = 0.5  # delta-mode brake threshold: trust >= this ->
    # FULL-magnitude actions (scale 1); below it, linear attenuation to a stop
    # at trust 0. A flat tau*raw would shrink every action by typical tau~0.6
    # even when tracking well — braking is for divergence, not a resting tax.
    text_dim: int = 512
    n_text_tokens: int = 3
    vis_dim: int = 512
    state_dim: int = 256
    fused_rows: int = 32
    fused_cols: int = 5
    plan_steps: int = 5
    num_servos: int = 7
    waypoint_dim: int = 3  # planner stage 1: the 3D (xyz) end-effector coords
    d_model: int = 384
    d_plan: int = 256
    n_heads: int = 8
    n_fusion_blocks: int = 3
    n_planner_blocks: int = 3
    n_fourier: int = 16
    modality_dropout: float = 0.3
    tick_hz: float = 30.0
    real_frame_hz: float = 2.0
    correction_beta: float = 0.7
    correction_decay: float = 0.9
    staleness_decay: float = 0.9
    trust_temperature: float = 4.0
    context_window: int = 8
    drift_horizons: tuple[int, ...] = (1, 2, 4, 8)
    # --- TQSA (v7): Text-Queried Spatial Adapter on the frozen backbone ---
    tqsa_dim: int = 128        # projected channel width of the spatial map
    tqsa_grid: int = 4         # spatial-token grid (4x4 -> 16 planner tokens)
    tqsa_heat: int = 8         # downsampled heatmap side (8x8 -> 64 per role)
    # Which memory groups the planner actually builds a projection for. Every
    # caller keeps passing everything it has; the planner IGNORES whatever is
    # not listed here, so ablating an input is a config change, not a call-site
    # change. Measured on-distribution sensitivity (`eval.bench --sensitivity`)
    # is the evidence for dropping one — v7 read proprio 0.291 >> state_delta
    # 0.075 > wm_msg 0.031 > current_emb 0.025 ~ fused 0.023 > pred_box 0.013 >
    # geometry 0.004 > next_emb 0.001. Names must be a subset of
    # ChronoQueryPlanner.INPUT_NAMES; at least one must remain.
    # Width of the TRM's pooled belief state, exported by
    # RecursiveTRM.forward_full as "latent" and consumed by the planner's
    # wm_latent group. Must match the TRM's `d` (train_batched --trm-d, default
    # 1024) and be divisible by the planner's memory-token count (8). It is here
    # rather than read off the TRM because cfg is the single source of truth for
    # every dimension crossing a module boundary.
    wm_latent_dim: int = 1024
    planner_inputs: tuple[str, ...] = (
        "next_emb", "current_emb", "fused", "state_delta",
        "pred_box_emb", "geometry", "proprio", "spatial", "wm_msg",
        "wm_latent",
        # The TRM's RESIDUAL as its own group. next_emb is current_emb + delta
        # and consecutive embeddings are cosine 0.9922 apart, so next_emb is
        # ~99% a copy of its own input and any module reading it must recover
        # the prediction by subtracting two nearly equal vectors. Standardized
        # at the projection, because the delta's direction is the signal and
        # its magnitude is tiny next to the latent it rides on.
        "wm_delta",
    )
    # --- v7.2 WAYPOINT-ABSOLUTE actuation (opt-in; the std_ratio lever) ---
    # Regressing normalized ACTIONS with MSE collapses to the timid conditional
    # mean (measured std_ratio ~0.37 = a third of demo vigor) because raw teleop
    # action commands are noisy. EEF POSITIONS are not: predicting the metric
    # displacement to a future waypoint and closing the loop on MEASURED EEF
    # each replan makes the commanded magnitude a function of the remaining
    # positional error, not of the regression's amplitude. Enable with
    # `train_batched.py --waypoint-weight W` (bakes into the checkpoint cfg);
    # needs `eef_pos_chunk` in the npz and a fitted gain
    # (`preprocess/fit_waypoint_gain.py` -> waypoint_stats.json).
    waypoint_action: bool = False   # build the metric-displacement head
    waypoint_range: float = 0.15    # metres spanned by the head's [-1, 1] output
    waypoint_horizon: int = 4       # servo toward the EEF pose `horizon` steps out.
    # MUST index a SUPERVISED row: waypoint_targets builds row k from
    # chunk[k+1]-chunk[0] and the bake has only plan_steps rows, so rows
    # 0..plan_steps-2 have targets and the last one does not. horizon 5 aimed
    # at the unsupervised row (measured: |cmd| 0.11 vs the ~0.34 the head's
    # 0.604 vigor implies). The actuator clamps this regardless.
    waypoint_gain_scale: float = 1.0  # multiplies the fitted proportional term
    # LONG-HORIZON supervision (v7.4). waypoint_long=True supervises the head
    # against displacement at the SAMPLED (2 Hz) spacing instead of the native
    # one: 0.5-2.5 s of motion instead of 0.05-0.20 s. Over 0.2 s "keep going"
    # is a near-sufficient statistic and object position is a second-order
    # correction, which is the conditional-mean ordering that produces the
    # measured 12:1 phase:vision ratio; over 2.5 s the arm must ARRIVE, so
    # where the object is becomes first-order. Zero new parameters.
    # Both companions are MANDATORY when it is on, or the units silently break:
    #   waypoint_range must cover a whole reach (0.15 m clamps and destroys it)
    #   waypoint_row_stride must be the SOURCE env's control rate / real_frame_hz
    #   (LIBERO: 20 Hz / 2 Hz = 10), or the actuator's per-step rate is wrong by
    #   exactly that factor. NOT tick_hz/real_frame_hz — tick_hz (30) is the Pi
    #   deployment loop, while the actuator counts steps of whatever control loop
    #   is actually calling it, which under LIBERO eval is 20 Hz. An earlier
    #   version of this comment said tick_hz and implied 15.
    waypoint_long: bool = False
    waypoint_row_stride: int = 1
    # ---- v8: relational head (replaces SlotResonanceFusion) ------------------
    # Runs AFTER the TRM, on the predicted latent, so relational reasoning
    # operates on the same state the planner is conditioned on.
    max_objects: int = 8            # K proposals kept per frame (data-rich: full
                                    # 512-d each, no [32,5] bottleneck)
    # Detector score threshold, shared by the BAKE and the ROBOT. It has to be
    # one number: it decides which boxes exist at all, and every surviving box
    # carries its confidence into fusion's `box_weight` fade, so a threshold
    # that differs between the two sides puts evidence mass at deployment
    # exactly where training had none. The bake used the detector class default
    # (0.10) while eval/policy.py passed 0.02 — defect 26, and the asymmetry was
    # even documented in eval/policy.py's docstring rather than reconciled.
    # 0.02 is the measured operating point: on agentview it grounds the source
    # role on 85% of frames with 2.82 proposals/frame, against 0.10's sparser
    # scene (paper.md 5n).
    det_conf: float = 0.02
    # Reject a SOURCE box that overlaps the TARGET box by more than this IoU,
    # and take the source chain's next candidate instead. 0.0 disables.
    #
    # Measured need: on the agentview libero_object corpus the two roles resolve
    # to the SAME box on 40.9% of frames where both are detected. The source
    # chain falls through its exact phrase ("alphabet soup", which YOLO-World
    # scores 0.000) to a generic tail ("box", "can"), and the basket is the most
    # box-like thing in the scene -- so on 4 frames in 10 the policy is told the
    # object to pick up is exactly where it must be put. Every task in the
    # corpus is "pick up X and place it in the basket", so source == target is
    # definitionally wrong, which makes this checkable without any label.
    #
    # Shared by the bake and the robot like det_conf, and recorded in
    # manifest.json provenance, because a corpus baked with a different value
    # is a different corpus.
    role_disjoint_iou: float = 0.0
    # Reject SOURCE boxes larger than this fraction of the frame. The basket
    # (target) is routinely the biggest "box"-like detection when the grocery
    # phrase falls through; a size prior is label-free and orthogonal to
    # role_disjoint_iou. 0 disables. 0.12 is a starting point for LIBERO
    # object (bottles/cans << basket liner).
    source_max_area: float = 0.0
    # Prefer tall SOURCE boxes (bottles/cans) over wide ones (basket liner).
    # height/width must be >= this; 0 disables. 1.15 is a soft bottle prior.
    source_min_aspect: float = 0.0
    rel_dim: int = 384              # relational token width
    rel_tokens: int = 12            # tokens emitted to the planner
    rel_layers: int = 2
    rel_heads: int = 8
    # ---- v8: HRM long-horizon backbone (replaces AnchoredDriftEncoder) -------
    # Two timescales matched to the loop itself: the SLOW module steps only on
    # real perception ticks (2 Hz), the FAST module every tick (30 Hz). Absorbs
    # the hand-fitted proportional gains of fit_waypoint_gain.py into learned
    # parameters ("learned PID"), plus drift and long-horizon reasoning.
    hrm_dim: int = 256              # state code width (matches drift's [B,256])
    hrm_slow_layers: int = 2
    hrm_fast_layers: int = 2
    hrm_gain_dim: int = 3           # learned per-axis control gains (x, y, z)
    # ---- GRAM (Baek et al. 2026): stochastic guidance on HRM + planner heads ----
    # NOT on the TRM. Slow HRM update = high-level recursive state; planner
    # features before waypoint/orient/grip/wp_disp = multi-hypothesis decode.
    # Off by default so existing checkpoints stay bit-identical.
    gram_hrm: bool = False          # ε after slow_core (high-level only)
    gram_planner: bool = False      # ε on plan feats before output heads
    gram_noise_dim: int = 32        # bottleneck width of μ/σ
    gram_n_samples: int = 1         # inference width (parallel trajectories)
    gram_kl_weight: float = 0.1     # ELBO KL weight when posterior is used
    trainable_param_budget: int = 9_000_000

    @property
    def fps(self) -> float:
        """Back-compat alias: the real-perception sampling rate."""
        return self.real_frame_hz

    @property
    def dream_ticks_per_real(self) -> int:
        """Dream ticks between consecutive real frames (14 at 30/2 Hz)."""
        return int(round(self.tick_hz / self.real_frame_hz)) - 1


DEFAULT_CONFIG = MicroVLAConfig()
