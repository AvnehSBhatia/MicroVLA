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
