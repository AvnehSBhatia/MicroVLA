"""HRM Backbone (v8) — two-timescale state model, replacing AnchoredDriftEncoder.

**The design claim: the hierarchy is already in the deployment loop.** A
Hierarchical Reasoning Model couples a slow module that reasons over long
horizons with a fast module that relaxes toward a local equilibrium between
slow updates. That is not an architecture we have to invent a schedule for —
this stack already runs exactly two clocks:

* ``cfg.real_frame_hz`` (2 Hz): a REAL YOLO-World measurement arrives. New
  measured evidence, new long-horizon context. → the **SLOW** module steps.
* ``cfg.tick_hz`` (30 Hz): every control tick, real or dream. During the
  ``cfg.dream_ticks_per_real`` (14) dream ticks in between, the only new input
  is the corrected TRM latent — imagination, not evidence. → the **FAST**
  module steps, settling toward the equilibrium implied by the slow state that
  the last measurement left behind.

So the two-timescale factorization is a description of the existing loop rather
than an import from a paper: the slow module never sees an imagined latent as
context, and the fast module supplies a fresh state code on every tick instead
of the v7 arrangement where the drift code was computed at 2 Hz and *held
constant* for 14 ticks (see ``JEPALoop.tick``).

It subsumes three jobs the v7 stack did separately or by hand:

(a) **Drift** — everything ``AnchoredDriftEncoder`` did: displacement from the
    episode anchor (the first REAL frame) and from a ``cfg.context_window``-deep
    memory of real frames at the lags in ``cfg.drift_horizons`` (1, 2, 4, 8
    frames ≈ 0.5–4 s at 2 Hz). Those multi-horizon drift tokens are the slow
    module's input; the fast module additionally reads ``frame_emb − anchor``
    directly, so anchor-relative drift is measured at 30 Hz, not 2 Hz.
(b) **Learned PID** — ``preprocess/fit_waypoint_gain.py`` fits, by
    least squares over baked demos, the metres of EEF travel per unit raw
    action per control step that ``WaypointActuator`` divides by: x 0.01085,
    y 0.01306, z 0.01180 (LIBERO OSC). Those three hand-fitted constants
    become ``HRMState.gains`` — a trained, state-conditioned output. The
    control law stops being tuned and starts being learned, and because the
    head is zero-initialized on top of a log-space prior seeded with the
    fitted numbers (:data:`FITTED_GAIN_PRIOR`), an untrained HRM emits
    *exactly* the hand fit. Training can only move off a known-good operating
    point, never start away from one.
(c) **Long-horizon reasoning** — the slow module reads the context window
    through cross-attention *queried by its own state*, so which timescale
    matters (has the scene changed since the anchor? since 0.5 s ago?) is a
    function of where the episode currently is, not a fixed pooling weight.

Both modules use the SAME cell: a pre-LN MLP block stack plus a damped,
input-conditioned update ``(1−α)·state + α·candidate`` with ``α = sigmoid(·)``
(:class:`_DampedCore`). Keeping the cells identical is deliberate — the claim
of this module is about *when* each module steps and *what* it reads, so the
two paths must not differ in any other way that could explain a measured
difference. The damping is what makes "relaxes toward equilibrium" more than a
figure of speech: with an unchanging drive, each step moves a fraction ``α`` of
the way to the candidate rather than overshooting it. Nothing *enforces*
convergence, and the loop never runs more than 14 fast steps before the next
slow update, so this is a shaping choice, not a guarantee.

Within a real tick the SLOW module steps FIRST, then the fast module reads the
freshly updated slow state. HRM's published unroll does the opposite (N fast
steps, then one slow), but that ordering is for offline unrolled training; in a
streaming controller the causal order is measurement → update the slow belief →
let the fast state relax toward it → emit. Emitting a state code that ignores
the measurement that just arrived, until the next tick 33 ms later, would be a
gratuitous one-tick delay on the only ticks that carry new information.

Runtime-state semantics are copied verbatim from ``AnchoredDriftEncoder`` (they
are hard rules in CLAUDE.md, and callers depend on them):

* The first forward after :meth:`reset` stores the anchor, seeds the context
  window, zero-inits both module states, and returns an **exactly-zero** state
  code *without stepping either recurrence*. Zero means "no drift yet" — the
  anchor frame cannot have drifted from itself. It is produced by an explicit
  ``new_zeros``, never by ``LayerNorm(0)``, whose learned bias is not zero
  after training.
* Both module states are DETACHED between steps (local BPTT), so a tick's
  backward pass cannot reach a previous tick's graph.
* All runtime state (anchor, window deque, slow/fast hidden, EEF anchor) lives
  in plain Python attributes — never a parameter, never a buffer. A
  ``state_dict`` therefore contains only learned weights, and a checkpoint can
  never smuggle one episode's state into another.
* A batch-size change silently resets (debug-logged), same as v7.

``frame_emb`` is consumed exactly as given: it is already in the canonical
standardized space (``microvla/utils/embedding.py::standardize``) and this
module must not re-normalize it. The only ``LayerNorm`` on an output is on the
internal state code, which is the output contract v7 already had
(``LayerNorm(hidden) -> [B, 256]``).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from torch import nn

from microvla.config import MicroVLAConfig

logger = logging.getLogger(__name__)

#: Per-axis (x, y, z) EEF response gain in metres of travel per unit raw action
#: per control step, as fitted by ``preprocess/fit_waypoint_gain.py`` over baked
#: LIBERO episodes (through-the-origin least squares; R² well above 0.5 on all
#: three translation axes). This is the prior the learned gains start from, not
#: a constant the module uses — the whole point of job (b) is that these become
#: trainable. A config with ``hrm_gain_dim > 3`` pads with their mean; one with
#: ``hrm_gain_dim < 3`` truncates.
FITTED_GAIN_PRIOR: tuple[float, ...] = (0.01085, 0.01306, 0.01180)

#: Multiplicative band the STATE-CONDITIONED part of the gain may explore
#: around the episode's learned baseline: ``exp(±log 4)`` = a quarter to four
#: times it. The bound exists because the actuator DIVIDES by the gain
#: (``gain_scale·(target − eef)/gain``), so a head that wandered toward zero on
#: one unusual frame would emit a saturated command from an arbitrarily small
#: positional error. A factor of 4 is far wider than the spread between the
#: three fitted axes (0.01085 … 0.01306, a factor of 1.2), so it cannot be what
#: limits learning; the baseline itself is unbounded within the limits below.
GAIN_LOG_RANGE: float = math.log(4.0)

#: Numerical guard on the learned per-axis log baseline: gains stay inside
#: ``[1e-4, 10]`` m per unit action per step before the band above is applied.
#: This is a floor against divide-by-zero, not a learning constraint — the
#: fitted LIBERO OSC prior sits at ~0.012, five decades above the low limit and
#: three below the high one, and a different embodiment (the Pi's servo rig)
#: would still land far inside. A ``clamp`` rather than a ``tanh`` squash for
#: exactly that reason: within the physical range the baseline must be free and
#: unwarped. Without it, ``exp`` underflows to a hard 0.0 in float32 for a
#: baseline below about −103, and the actuator's division would then hand the
#: robot ``inf`` (or ``NaN`` for a zero positional error), which no downstream
#: clip can rescue.
LOG_GAIN_LIMITS: tuple[float, float] = (math.log(1e-4), math.log(1e1))

#: Submodules/parameters that ONLY a real tick can touch (the slow module).
#: Named explicitly rather than derived, so :meth:`HRMBackbone.slow_parameters`
#: is an assertion about the design ("this is the 2 Hz path") that a test can
#: check, instead of a description of whatever the code happens to do.
_SLOW_MODULE_NAMES = (
    "drift_proj",
    "ctx_q_norm",
    "ctx_kv_norm",
    "ctx_attn",
    "fast_to_slow",
    "slow_core",
)
_SLOW_PARAM_NAMES = ("horizon_emb",)

#: The 30 Hz path. The readout (``out_norm``, ``gain_head``, ``log_gain_base``)
#: is grouped here because it runs on every tick, dream ticks included.
_FAST_MODULE_NAMES = (
    "fast_in",
    "eef_proj",
    "slow_to_fast",
    "fast_core",
    "out_norm",
    "gain_head",
)
_FAST_PARAM_NAMES = ("log_gain_base",)


@dataclass
class HRMState:
    """One tick's output of :class:`HRMBackbone`.

    Attributes:
        state: ``[B, cfg.hrm_dim]`` state/drift code — the drop-in replacement
            for ``AnchoredDriftEncoder``'s ``state_delta``, fed to the TRM.
            Exactly zero on the first tick of an episode.
        gains: ``[B, cfg.hrm_gain_dim]`` learned per-axis control gains,
            strictly positive, in the units of
            ``preprocess/fit_waypoint_gain.py`` (metres of EEF travel per unit
            raw action per control step). This module only *emits* them; the
            actuator applies them.
    """

    state: torch.Tensor
    gains: torch.Tensor


#: Absolute bound on a recurrent HRM state. Chosen ~10x the largest magnitude
#: observed in healthy operation (|state| absmax 4.7 over a 74-step episode), so
#: it is a divergence rail rather than a regularizer.
_STATE_LIMIT: float = 50.0


class _DampedCore(nn.Module):
    """Pre-LN MLP stack + damped gated update — the cell both timescales share.

    ``forward(state, drive)`` runs ``n_layers`` residual pre-LN GELU blocks over
    ``state + drive`` to get a candidate, then returns
    ``(1 − α)·state + α·candidate`` with ``α = sigmoid(rate(drive))`` per
    channel. Damping the step by a *drive-conditioned* rate lets the module
    learn to move less when its input carries less information — which is
    precisely the fast module's situation on a dream tick, where the input is a
    TRM prediction with staleness-decayed evidence behind it.

    Attributes:
        blocks: ``n_layers`` pre-LN residual MLP blocks (hidden ``2·dim``).
        rate: Produces the per-channel update rate from the drive.
    """

    def __init__(self, dim: int, n_layers: int) -> None:
        """Builds the cell.

        Args:
            dim: State width (``cfg.hrm_dim``).
            n_layers: Number of residual blocks (``cfg.hrm_slow_layers`` /
                ``cfg.hrm_fast_layers``).
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 2 * dim),
                nn.GELU(),
                nn.Linear(2 * dim, dim),
            )
            for _ in range(n_layers)
        )
        self.rate = nn.Linear(dim, dim)

    def forward(self, state: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        """Advances ``state`` one step under ``drive``.

        Args:
            state: ``[B, dim]`` current module state.
            drive: ``[B, dim]`` this step's input (already projected to
                ``dim`` and summed by the caller — HRM-style input injection).

        Returns:
            ``[B, dim]`` updated state.
        """
        x = state + drive
        for block in self.blocks:
            x = x + block(x)
        alpha = torch.sigmoid(self.rate(drive))
        out = (1.0 - alpha) * state + alpha * x
        # SAFETY RAIL on the recurrent state. The candidate `x` grows through
        # additive residual blocks and nothing bounds the carried state, so over
        # a long episode it can run away to inf and then NaN. Measured: episodes
        # of T=97 and T=111 produce a non-finite loss while the T=74 majority do
        # not, and one bad episode NaNs its whole batch -- 2% of episodes cost
        # 17% of batches. Deployment is worse: 400 env steps at perception
        # period 2 is 200 recurrent steps, longer than almost any training
        # episode, and closed-loop telemetry showed 50-75% of ticks non-finite.
        #
        # The bound is far outside normal operation (measured |state| absmax
        # ~4.7 over 74 steps), so it never binds on healthy dynamics and cannot
        # change a working trajectory -- it only stops a diverging one from
        # becoming NaN and destroying the episode.
        return out.clamp(-_STATE_LIMIT, _STATE_LIMIT)


class HRMBackbone(nn.Module):
    """Two-timescale (2 Hz slow / 30 Hz fast) drift, gain, and context model.

    See the module docstring for the design argument. Per tick:

    1. First forward after :meth:`reset`: store the anchor and (if given) the
       EEF anchor, seed the window, zero the two states, return a zero code
       with the prior gains — no recurrence step.
    2. If ``is_real``: build one drift token per reference (the anchor plus
       each lag in ``cfg.drift_horizons``, clamped to the filled window) as
       ``GELU(Linear(cat([emb − ref, emb ⊙ ref])))`` + a learned horizon
       embedding; read them with cross-attention queried by the slow state;
       step the slow core on that read plus ``fast_to_slow(fast state)``;
       append ``frame_emb`` to the window.
    3. Step the fast core on ``fast_in(cat([emb, emb − anchor]))`` +
       ``slow_to_fast(slow state)`` + ``eef_proj(EEF features)``.
    4. Emit ``LayerNorm(fast state)`` and the gains read off it.

    Attributes:
        horizons: ``cfg.drift_horizons`` as a tuple (lags in REAL frames).
        drift_proj: Shared projection of per-horizon drift features (slow).
        horizon_emb: Learned per-reference embeddings, ``[n_horizons + 1,
            hrm_dim]`` (index 0 = anchor).
        ctx_attn: Slow-state-queried cross-attention over the drift tokens.
        slow_core: 2 Hz cell.
        fast_in: Projection of ``[frame_emb, frame_emb − anchor]`` (fast).
        eef_proj: Projection of the measured-EEF features (fast).
        fast_core: 30 Hz cell.
        out_norm: LayerNorm on the emitted state code.
        gain_head: Zero-initialized head producing the log-space gain
            modulation; with the log prior below it, an untrained module emits
            exactly :data:`FITTED_GAIN_PRIOR`.
        log_gain_base: Learned per-axis log gain, initialized to the fitted
            prior.
    """

    def __init__(self, cfg: MicroVLAConfig) -> None:
        """Builds the backbone from the canonical config.

        Args:
            cfg: Shared MicroVLA configuration; uses ``vis_dim``,
                ``waypoint_dim``, ``n_heads``, ``context_window``,
                ``drift_horizons``, ``hrm_dim``, ``hrm_slow_layers``,
                ``hrm_fast_layers``, and ``hrm_gain_dim``.
        """
        super().__init__()
        self.cfg = cfg
        self.horizons = tuple(cfg.drift_horizons)
        d = cfg.hrm_dim
        drift_feat_dim = 2 * cfg.vis_dim  # [emb - ref, emb * ref]

        # --- SLOW module (steps on REAL ticks only, cfg.real_frame_hz) -------
        self.drift_proj = nn.Linear(drift_feat_dim, d)
        self.horizon_emb = nn.Parameter(
            torch.randn(len(self.horizons) + 1, d) * d**-0.5
        )
        # cfg has no HRM-specific head count, so use the canonical one; fall
        # back to a single head when an ablation config picks a width that is
        # not divisible by it, rather than crashing deep inside attention.
        heads = cfg.n_heads if d % cfg.n_heads == 0 else 1
        if heads != cfg.n_heads:
            logger.debug(
                "HRMBackbone: hrm_dim=%d is not divisible by cfg.n_heads=%d; "
                "using 1 attention head for the context read.",
                d,
                cfg.n_heads,
            )
        self.ctx_q_norm = nn.LayerNorm(d)
        self.ctx_kv_norm = nn.LayerNorm(d)
        self.ctx_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.fast_to_slow = nn.Linear(d, d)
        self.slow_core = _DampedCore(d, cfg.hrm_slow_layers)
        self.act = nn.GELU()

        # --- FAST module (steps every tick, cfg.tick_hz) ---------------------
        # Anchor-relative drift at 30 Hz: the fast module gets the raw latent
        # AND its displacement from the episode anchor, so it does not have to
        # rediscover the anchor from the slow state's summary of it.
        self.fast_in = nn.Linear(2 * cfg.vis_dim, d)
        self.eef_proj = nn.Linear(self._eef_feat_dim, d)
        self.slow_to_fast = nn.Linear(d, d)
        self.fast_core = _DampedCore(d, cfg.hrm_fast_layers)

        # --- Readout (every tick) --------------------------------------------
        self.out_norm = nn.LayerNorm(d)
        self.gain_head = nn.Linear(d, cfg.hrm_gain_dim)
        nn.init.zeros_(self.gain_head.weight)
        nn.init.zeros_(self.gain_head.bias)
        self.log_gain_base = nn.Parameter(torch.log(self._gain_prior(cfg)))

        # Per-episode runtime state — plain attributes, deliberately NOT
        # parameters or buffers, so a state_dict is weights and nothing else.
        self._anchor: torch.Tensor | None = None
        self._window: deque[torch.Tensor] | None = None
        self._slow: torch.Tensor | None = None
        self._fast: torch.Tensor | None = None
        self._eef_anchor: torch.Tensor | None = None

    @property
    def _eef_feat_dim(self) -> int:
        """Width of the EEF feature vector: ``[eef, eef − anchor, validity]``.

        The validity scalar is a feature, not just a multiplier — the same
        reason fusion appends its raw ``box_weight`` to the geometry features.
        Without it, a missing EEF (zero-filled) is indistinguishable from an
        arm sitting exactly at its anchor pose.
        """
        return 2 * self.cfg.waypoint_dim + 1

    @staticmethod
    def _gain_prior(cfg: MicroVLAConfig) -> torch.Tensor:
        """The fitted per-axis prior, stretched to ``cfg.hrm_gain_dim``."""
        prior = list(FITTED_GAIN_PRIOR[: cfg.hrm_gain_dim])
        if len(prior) < cfg.hrm_gain_dim:
            mean = sum(FITTED_GAIN_PRIOR) / len(FITTED_GAIN_PRIOR)
            prior += [mean] * (cfg.hrm_gain_dim - len(prior))
        return torch.tensor(prior, dtype=torch.float32)

    def reset(self) -> None:
        """Clears every per-episode attribute (anchors, window, both states)."""
        self._anchor = None
        self._window = None
        self._slow = None
        self._fast = None
        self._eef_anchor = None

    def slow_parameters(self) -> Iterator[nn.Parameter]:
        """Yields the parameters only a REAL tick can touch (2 Hz path).

        A dream-only rollout must leave every one of these with a ``None``
        gradient; that is the mechanical statement of "the slow module does not
        step on imagination", and ``tests/test_hrm.py`` asserts it.
        """
        for name in _SLOW_MODULE_NAMES:
            yield from getattr(self, name).parameters()
        for name in _SLOW_PARAM_NAMES:
            yield getattr(self, name)

    def fast_parameters(self) -> Iterator[nn.Parameter]:
        """Yields the parameters that step on EVERY tick (30 Hz path + readout)."""
        for name in _FAST_MODULE_NAMES:
            yield from getattr(self, name).parameters()
        for name in _FAST_PARAM_NAMES:
            yield getattr(self, name)

    def _drift_token(self, emb: torch.Tensor, ref: torch.Tensor, idx: int) -> torch.Tensor:
        """One reference's drift token: shared projection + horizon embedding."""
        features = torch.cat([emb - ref, emb * ref], dim=-1)
        return self.act(self.drift_proj(features)) + self.horizon_emb[idx]

    def _context_read(self, frame_emb: torch.Tensor) -> torch.Tensor:
        """Cross-attends the multi-horizon drift tokens, queried by the slow state.

        Args:
            frame_emb: ``[B, vis_dim]`` current REAL frame embedding.

        Returns:
            ``[B, hrm_dim]`` context read. Lags longer than the filled window
            reuse its oldest entry, so early ticks degrade to shorter horizons
            instead of inventing history.
        """
        assert self._window is not None and self._anchor is not None
        history = list(self._window)  # oldest .. newest
        tokens = [self._drift_token(frame_emb, self._anchor, 0)]
        for i, h in enumerate(self.horizons, start=1):
            ref = history[-min(h, len(history))]
            tokens.append(self._drift_token(frame_emb, ref, i))
        stacked = self.ctx_kv_norm(torch.stack(tokens, dim=1))  # [B, n_ref, d]

        query = self.ctx_q_norm(self._slow).unsqueeze(1)  # [B, 1, d]
        pooled, _ = self.ctx_attn(query, stacked, stacked, need_weights=False)
        return pooled.squeeze(1)

    def _validate_eef(self, eef: torch.Tensor, batch: int) -> None:
        """Rejects a mis-shaped EEF at the boundary, on EVERY tick.

        Checked before the anchor branch as well as on ordinary ticks, because
        the anchor tick is the one that *stores* ``_eef_anchor``: a full
        7-D proprio vector accepted there poisons every later subtraction, and
        the caller would then see a broadcast ``RuntimeError`` on tick 2 naming
        neither the argument nor the episode boundary that caused it. Batch and
        rank are checked alongside width for the same reason — a ``[B, 3]``
        anchor minus a bare ``[3]`` reading is a rank error thrown from inside
        ``torch.cat``, which reads as a bug in this module rather than in the
        call.

        Raises:
            ValueError: If ``eef`` is not exactly ``[batch, cfg.waypoint_dim]``.
        """
        if eef.dim() != 2 or tuple(eef.shape) != (batch, self.cfg.waypoint_dim):
            raise ValueError(
                f"eef must be [B, {self.cfg.waypoint_dim}] (cfg.waypoint_dim) "
                f"with B={batch} to match frame_emb, got {tuple(eef.shape)}"
            )

    def _eef_features(self, eef: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        """Builds ``[eef, eef − eef_anchor, validity]``, zeros when unavailable.

        Args:
            eef: ``[B, cfg.waypoint_dim]`` measured end-effector position, or
                ``None`` (Bridge episodes, the mock rig, and any tick whose
                proprio flag is unset carry no EEF). Already validated by
                :meth:`_validate_eef`.
            like: Tensor supplying batch size, device, and dtype.

        Returns:
            ``[B, 2·waypoint_dim + 1]`` feature vector.

        Note:
            Validity is per CALL, matching the per-episode nature of the bake's
            proprio flag; a training batch that mixes episodes with and without
            proprio must be bucketed by availability, exactly as the loader
            already buckets by ``(T, has_frames)``.
        """
        batch = like.shape[0]
        if eef is None:
            return like.new_zeros(batch, self._eef_feat_dim)
        if self._eef_anchor is None:
            # First EEF ever seen this episode is the pose everything else is
            # measured against — the metric twin of the visual anchor.
            self._eef_anchor = eef.detach()
        return torch.cat([eef, eef - self._eef_anchor, like.new_ones(batch, 1)], dim=-1)

    def _emit(self, code: torch.Tensor) -> HRMState:
        """Packs a state code with the gains read off it.

        ``gain = exp(clamp(log_gain_base) + GAIN_LOG_RANGE · tanh(head(code)))``
        is strictly positive by construction — a negative control gain is a
        sign flip in the control law, i.e. an arm that drives away from its
        waypoint — and with the zero-initialized head it equals
        :data:`FITTED_GAIN_PRIOR` before any training. See
        :data:`LOG_GAIN_LIMITS` for why the exponent is floored rather than
        left free.
        """
        base = self.log_gain_base.clamp(*LOG_GAIN_LIMITS)
        modulation = GAIN_LOG_RANGE * torch.tanh(self.gain_head(code))
        return HRMState(state=code, gains=torch.exp(base + modulation))

    def forward(
        self,
        frame_emb: torch.Tensor,
        is_real: bool = True,
        eef: Optional[torch.Tensor] = None,
    ) -> HRMState:
        """Advances the backbone by one control tick.

        Args:
            frame_emb: ``[B, cfg.vis_dim]`` standardized latent for this tick —
                the real YOLO-World frame embedding when ``is_real``, the
                corrected TRM prediction on a dream tick. Consumed as given;
                it is already canonical.
            is_real: Whether this tick carries a real measurement. ``True``
                steps the slow module as well and extends the context window;
                ``False`` steps only the fast module, so no imagined latent
                ever enters the 2 Hz history.
            eef: ``[B, cfg.waypoint_dim]`` measured end-effector position when
                available, else ``None``.

        Returns:
            :class:`HRMState` for this tick. ``state`` is exactly zero on the
            first tick after :meth:`reset`; ``gains`` are always positive.

        Raises:
            ValueError: If ``eef`` is given with a shape other than
                ``[B, cfg.waypoint_dim]`` — including on the anchor tick, which
                is where a bad shape would otherwise be silently stored.
        """
        batch = frame_emb.shape[0]
        if eef is not None:
            self._validate_eef(eef, batch)

        if self._anchor is not None and self._anchor.shape[0] != batch:
            logger.debug(
                "HRMBackbone: batch size changed %d -> %d; silently resetting "
                "(new anchor). If this fires mid-episode, a caller forgot an "
                "explicit reset().",
                self._anchor.shape[0],
                batch,
            )
            self.reset()

        if self._anchor is None:
            if not is_real:
                logger.debug(
                    "HRMBackbone: first forward after reset() has is_real=False; "
                    "anchoring the episode on an imagined latent. The loop is "
                    "expected to open every episode with a real frame."
                )
            self._anchor = frame_emb.detach()
            self._window = deque([frame_emb.detach()], maxlen=self.cfg.context_window)
            self._slow = frame_emb.new_zeros(batch, self.cfg.hrm_dim)
            self._fast = frame_emb.new_zeros(batch, self.cfg.hrm_dim)
            if eef is not None:
                self._eef_anchor = eef.detach()
            # Zero code, not LayerNorm(0): the anchor frame has not drifted
            # from itself, and a trained LayerNorm bias would say otherwise.
            return self._emit(frame_emb.new_zeros(batch, self.cfg.hrm_dim))

        assert self._slow is not None and self._fast is not None

        if is_real:
            slow_drive = self._context_read(frame_emb) + self.fast_to_slow(self._fast)
            slow = self.slow_core(self._slow, slow_drive)
            self._slow = slow.detach()  # local BPTT
            self._window.append(frame_emb.detach())
        else:
            # Dream tick: the slow module is untouched, and because its stored
            # state is already detached the fast path below cannot backprop
            # into any 2 Hz parameter.
            slow = self._slow

        fast_drive = (
            self.fast_in(torch.cat([frame_emb, frame_emb - self._anchor], dim=-1))
            + self.slow_to_fast(slow)
            + self.eef_proj(self._eef_features(eef, frame_emb))
        )
        fast = self.fast_core(self._fast, fast_drive)
        self._fast = fast.detach()  # local BPTT

        return self._emit(self.out_norm(fast))


if __name__ == "__main__":
    from microvla.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    hrm = HRMBackbone(cfg)
    hrm.eval()
    hrm.reset()

    torch.manual_seed(0)
    # One deployment-shaped episode: a real frame every dream_ticks_per_real+1.
    period = cfg.dream_ticks_per_real + 1
    for tick in range(2 * period + 3):
        real = tick % period == 0
        out = hrm(
            torch.randn(2, cfg.vis_dim),
            is_real=real,
            eef=torch.randn(2, cfg.waypoint_dim) * 0.05 if real else None,
        )
        assert out.state.shape == (2, cfg.hrm_dim)
        assert out.gains.shape == (2, cfg.hrm_gain_dim)
        assert torch.all(out.gains > 0), "control gains must stay positive"
        if tick == 0:
            assert torch.all(out.state == 0), "first-tick state code must be zero"
            assert torch.allclose(
                out.gains, HRMBackbone._gain_prior(cfg).expand(2, -1)
            ), "an untrained HRM must emit exactly the fitted gain prior"
        if tick in (0, 1, period, period + 1):
            kind = "REAL" if real else "dream"
            print(
                f"tick {tick:3d} {kind}: |state|={out.state.norm().item():.4f} "
                f"gains={out.gains[0].tolist()} window={len(hrm._window)}"
            )

    assert not [k for k in hrm.state_dict() if "anchor" in k or "window" in k], (
        "runtime state leaked into the state_dict"
    )

    n_params = sum(p.numel() for p in hrm.parameters() if p.requires_grad)
    n_slow = sum(p.numel() for p in hrm.slow_parameters())
    n_fast = sum(p.numel() for p in hrm.fast_parameters())
    print(f"params: {n_params:,} ({n_params / 1e6:.3f}M) = slow {n_slow:,} + fast {n_fast:,}")
    assert n_slow + n_fast == n_params, "slow/fast groups must partition the parameters"
    assert n_params <= 3_000_000, f"HRMBackbone over budget: {n_params:,}"
