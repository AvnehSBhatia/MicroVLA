"""Innovation corrector for the JEPA dream loop (Kalman-lite, no learned params).

Between real (2 Hz) YOLO-World measurements, the JEPA loop free-runs on TRM
predictions for up to 14 "dream" ticks. :class:`InnovationCorrector` is the
mechanism that keeps those dream-tick latents anchored to reality without any
learned weights: it is a **complementary filter**, the classic "Kalman-lite"
trick of blending a fast, noisy signal (the raw TRM prediction) with a slow,
smoothed correction derived from the most recent measurement error, rather
than tracking a full state covariance the way a real Kalman filter would.

Mechanics:
    * At every REAL frame, the corrector observes the *innovation* (residual)
      between what the TRM predicted for this tick and what YOLO-World
      actually saw: ``e = real - pred``. This is exponentially averaged into
      an accumulator ``c`` (EMA factor ``correction_beta``), so ``c`` tracks a
      slowly-drifting estimate of "how wrong the TRM tends to be right now"
      rather than reacting to single-frame noise.
    * The same measurement also updates a scalar **trust** ``tau``, computed
      from the *self-calibrating error ratio*: the innovation norm divided by
      an EMA of past innovation norms, mapped through
      ``tau = exp(-0.5 * ratio^2 * trust_temperature / 4)``. A typical-sized
      error gives tau ~= 0.61; errors far below the running norm -> tau -> 1;
      errors far above it -> tau -> 0. This deliberately avoids any fixed
      cosine threshold: real (standardized) frame embeddings of a mostly
      static scene are always highly correlated, so an absolute-cosine trust
      would saturate high and never discriminate — the ratio compares the
      TRM against its OWN recent accuracy instead. Downstream, ``tau``
      blends the emitted plan toward hold (see ``JEPALoop.tick``) so the
      robot commits less NEW motion when confidence is low.
    * On every DREAM tick, :meth:`correct` adds the accumulated correction to
      the raw TRM prediction, but geometrically decayed by
      ``correction_decay ** k`` where ``k`` is the number of dream ticks
      since the last real measurement. This is the "complementary" half of
      the filter: correction dominates right after a measurement (small
      ``k``) and fades out (large ``k``) as we drift further from ground
      truth, letting the TRM's own open-loop prediction take over rather
      than blindly extrapolating a stale correction forever.
    * ``k`` resets to 0 every time a new measurement arrives, so the
      correction "recharges" at every real frame.

All tensors here are **unbatched** ``[vis_dim]`` (512) float32 — the JEPA
loop deals in a single running latent per episode, not a batch.
"""

from __future__ import annotations

import math

import torch

from microvla.config import MicroVLAConfig

_EPS = 1e-8

#: EMA factor for the running innovation-norm baseline (slower than the
#: correction EMA: the baseline should reflect "typical" error, not react to
#: single spikes — spikes are exactly what trust must detect).
_ERR_BAR_BETA = 0.9


class InnovationCorrector:
    """Kalman-lite complementary filter bridging JEPA real and dream ticks.

    No learned parameters — pure running-state bookkeeping over three plain
    attributes (``c``, ``tau``, ``k``), all reset per episode via
    :meth:`reset`.

    Args:
        cfg: Shared MicroVLA configuration; supplies ``vis_dim`` (the latent
            width), ``correction_beta``, ``correction_decay``, and
            ``trust_temperature``.
    """

    def __init__(self, cfg: MicroVLAConfig) -> None:
        self.cfg = cfg
        self.c: torch.Tensor = torch.zeros(cfg.vis_dim, dtype=torch.float32)
        self.tau: float = 1.0
        self.k: int = 0
        self.err_bar: float | None = None  # EMA of innovation norms

    def reset(self) -> None:
        """Resets the accumulator, trust, error baseline, and tick counter.

        Called at the start of every episode (``c=0``, so the very first
        dream tick before any measurement applies zero correction; ``tau=1``
        so the plan is fully trusted until evidence says otherwise;
        ``err_bar=None`` so the baseline re-calibrates to the new episode).
        """
        self.c = torch.zeros(self.cfg.vis_dim, dtype=torch.float32)
        self.tau = 1.0
        self.k = 0
        self.err_bar = None

    def on_measurement(self, pred_emb: torch.Tensor, real_emb: torch.Tensor) -> None:
        """Updates the correction and trust from a real-frame measurement.

        Args:
            pred_emb: ``[vis_dim]`` TRM prediction made for this tick (the
                ``next_emb`` the loop carried forward from the previous
                tick). Skipped by the caller if no such prediction exists
                yet (i.e. this is the very first real frame of the episode).
            real_emb: ``[vis_dim]`` actual (standardized) YOLO-World frame
                embedding observed this tick.
        """
        e = real_emb - pred_emb
        beta = self.cfg.correction_beta
        self.c = beta * self.c + (1.0 - beta) * e

        # Self-calibrating trust: compare this innovation against the running
        # norm of recent innovations rather than any fixed threshold.
        err = float(e.norm().item())
        if self.err_bar is None:
            self.err_bar = err
        else:
            self.err_bar = _ERR_BAR_BETA * self.err_bar + (1.0 - _ERR_BAR_BETA) * err
        ratio = err / (self.err_bar + _EPS)
        self.tau = math.exp(-0.5 * ratio * ratio * (self.cfg.trust_temperature / 4.0))
        self.k = 0

    def correct(self, pred_emb: torch.Tensor) -> torch.Tensor:
        """Applies the decayed correction to a raw TRM prediction.

        Args:
            pred_emb: ``[vis_dim]`` raw TRM prediction for this dream tick.

        Returns:
            ``[vis_dim]`` corrected latent: ``pred_emb + correction_decay**k
            * c``. Advances ``k`` by one afterward, so successive dream ticks
            (without an intervening measurement) apply an ever-smaller
            correction.
        """
        corrected = pred_emb + (self.cfg.correction_decay**self.k) * self.c
        self.k += 1
        return corrected

    @property
    def trust(self) -> float:
        """Current trust ``tau`` in ``[0, 1]``, from the last measurement."""
        return self.tau
