"""HRMBackbone tests (CPU-only, mock-only, no network, no cv2).

Two families of assertion:

* **Runtime-state contract**, inherited verbatim from ``AnchoredDriftEncoder``
  and binding per CLAUDE.md: zero code on the first tick after ``reset()``,
  detached hidden state, and episode scratch that never reaches a
  ``state_dict``.
* **The two-timescale claim itself**, which is only a claim if it can fail: a
  dream tick must move the fast state and leave every slow parameter without a
  gradient, and a real tick must move both.
"""

from __future__ import annotations

import dataclasses
from collections import deque

import pytest
import torch

from microvla.config import DEFAULT_CONFIG
from microvla.hrm import FITTED_GAIN_PRIOR, HRMBackbone, HRMState
from microvla.hrm.hrm_backbone import GAIN_LOG_RANGE, LOG_GAIN_LIMITS

CFG = DEFAULT_CONFIG

#: Cap for this module (raised from the v7 drift encoder's 1.5M for the three
#: jobs it absorbs); the target is 2.5M.
HRM_PARAM_CAP = 3_000_000


def _emb(seed: int, batch: int = 2, cfg=CFG) -> torch.Tensor:
    """Deterministic stand-in for a standardized frame embedding."""
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn(batch, cfg.vis_dim, generator=gen)
    return (raw - raw.mean(dim=-1, keepdim=True)) / raw.std(dim=-1, keepdim=True)


def _eef(seed: int, batch: int = 2, cfg=CFG) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1000 + seed)
    return torch.randn(batch, cfg.waypoint_dim, generator=gen) * 0.05


def _build(cfg=CFG, seed: int = 0) -> HRMBackbone:
    torch.manual_seed(seed)
    hrm = HRMBackbone(cfg)
    hrm.eval()
    hrm.reset()
    return hrm


def _snapshot(hrm: HRMBackbone) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in hrm.state_dict().items()}


def _scalar_loss(out: HRMState, seed: int = 7) -> torch.Tensor:
    """A loss whose Jacobian w.r.t. the state code is not identically zero.

    ``out.state.sum()`` is the obvious choice and is WRONG for any gradient
    assertion here: the code is a ``LayerNorm`` output, so at initialization
    (unit weight, zero bias) ``sum(LayerNorm(x)) == sum(x̂) == 0`` for every
    input, and its derivative is analytically zero everywhere. The gain path is
    no better — ``gain_head`` is zero-initialized by design, so ``∂gains/∂code``
    is exactly zero too. Backpropagating that pair leaves every parameter
    upstream of the readout with a gradient of float32 cancellation dust
    (measured: 3e-4, against 4e+4 for the projected loss below), which makes an
    ``any(grad != 0)`` assertion pass on noise rather than on signal. A fixed
    random projection restores a generic loss without perturbing any weight.
    """
    proj = torch.randn(out.state.shape, generator=torch.Generator().manual_seed(seed))
    return (out.state * proj).sum() + out.gains.sum()


def _run_episode(hrm: HRMBackbone, n_real: int = 3, dreams: int = 4, batch: int = 2,
                 with_eef: bool = True, cfg=CFG) -> list[HRMState]:
    """A deployment-shaped episode: each real tick followed by dream ticks."""
    out: list[HRMState] = []
    for r in range(n_real):
        out.append(hrm(_emb(r, batch, cfg), is_real=True,
                       eef=_eef(r, batch, cfg) if with_eef else None))
        for k in range(dreams):
            out.append(hrm(_emb(100 * (r + 1) + k, batch, cfg), is_real=False))
    return out


class TestInterface:
    def test_hrm_state_fields(self):
        assert [f.name for f in dataclasses.fields(HRMState)] == ["state", "gains"]

    def test_shapes(self):
        hrm = _build()
        for batch in (1, 4):
            hrm.reset()
            for tick in range(3):
                out = hrm(_emb(tick, batch), is_real=tick % 2 == 0, eef=_eef(tick, batch))
                assert isinstance(out, HRMState)
                assert out.state.shape == (batch, CFG.hrm_dim)
                assert out.gains.shape == (batch, CFG.hrm_gain_dim)

    def test_defaults_allow_a_bare_call(self):
        """``is_real`` defaults to True and ``eef`` to None (v7 call shape)."""
        hrm = _build()
        hrm(_emb(0))
        out = hrm(_emb(1))
        assert out.state.shape == (2, CFG.hrm_dim)
        assert len(hrm._window) == 2, "a bare call must count as a REAL tick"

    def test_uses_config_dims_not_hardcoded_ones(self):
        cfg = dataclasses.replace(
            CFG,
            hrm_dim=64,
            hrm_gain_dim=5,
            hrm_slow_layers=1,
            hrm_fast_layers=3,
            drift_horizons=(1, 3),
            context_window=3,
        )
        hrm = _build(cfg)
        outs = _run_episode(hrm, n_real=3, dreams=2, cfg=cfg)
        assert all(o.state.shape == (2, 64) for o in outs)
        assert all(o.gains.shape == (2, 5) for o in outs)
        assert torch.all(outs[-1].gains > 0)
        assert len(hrm._window) <= cfg.context_window

    def test_width_not_divisible_by_head_count_still_builds(self):
        """An ablation width falls back to one attention head, never a crash."""
        cfg = dataclasses.replace(CFG, hrm_dim=100)
        assert cfg.hrm_dim % cfg.n_heads != 0
        hrm = _build(cfg)
        outs = _run_episode(hrm, n_real=2, dreams=2, cfg=cfg)
        assert outs[-1].state.shape == (2, 100)


class TestFirstTickIsZero:
    def test_first_forward_after_reset_is_exactly_zero(self):
        hrm = _build()
        out = hrm(_emb(0))
        assert torch.all(out.state == 0)

    def test_zero_survives_a_trained_output_norm(self):
        """The zero code must be a literal zero, not ``LayerNorm(0)`` = bias."""
        hrm = _build()
        with torch.no_grad():
            hrm.out_norm.bias.fill_(0.3)
            hrm.out_norm.weight.fill_(2.0)
        hrm.reset()
        assert torch.all(hrm(_emb(0)).state == 0)

    def test_reset_restores_the_zero_tick(self):
        hrm = _build()
        _run_episode(hrm)
        hrm.reset()
        assert torch.all(hrm(_emb(7)).state == 0)

    def test_recurrence_does_not_step_on_the_anchor_tick(self):
        hrm = _build()
        hrm(_emb(0))
        assert torch.all(hrm._slow == 0) and torch.all(hrm._fast == 0)
        assert len(hrm._window) == 1

    def test_second_tick_is_nonzero(self):
        hrm = _build()
        hrm(_emb(0))
        assert hrm(_emb(1)).state.abs().sum() > 0


class TestGains:
    def test_untrained_gains_are_exactly_the_fitted_prior(self):
        hrm = _build()
        out = hrm(_emb(0))
        prior = torch.tensor(FITTED_GAIN_PRIOR[: CFG.hrm_gain_dim])
        assert torch.allclose(out.gains, prior.expand(2, -1), atol=1e-6)

    def test_gains_strictly_positive_across_an_episode(self):
        hrm = _build()
        for out in _run_episode(hrm, n_real=4, dreams=3):
            assert torch.all(out.gains > 0)

    def test_gains_positive_and_banded_for_any_finite_state_code(self):
        """The guarantee is per state code, so state it that way: a negative
        gain is a sign flip in the control law and a zero gain is a division by
        zero in the actuator, and neither may be reachable from ANY code an
        arbitrarily diverged head could be handed."""
        hrm = _build()
        with torch.no_grad():
            hrm.gain_head.weight.normal_(0.0, 1e3)
            hrm.gain_head.bias.normal_(0.0, 1e3)
        base = hrm.log_gain_base.detach().clamp(*LOG_GAIN_LIMITS)
        codes = [
            torch.zeros(2, CFG.hrm_dim),
            torch.randn(2, CFG.hrm_dim),
            torch.randn(2, CFG.hrm_dim) * 1e6,
            torch.full((2, CFG.hrm_dim), -1e6),
        ]
        for code in codes:
            gains = hrm._emit(code).gains
            assert torch.all(gains > 0) and torch.isfinite(gains).all()
            assert torch.all(gains <= torch.exp(base + GAIN_LOG_RANGE) + 1e-6)
            assert torch.all(gains >= torch.exp(base - GAIN_LOG_RANGE) - 1e-9)

    def test_gains_stay_positive_on_a_badly_scaled_episode(self):
        hrm = _build()
        with torch.no_grad():
            hrm.gain_head.weight.normal_(0.0, 10.0)
        hrm.reset()
        for r in range(3):
            out = hrm(_emb(r) * 100.0, is_real=True, eef=_eef(r) * 100.0)
            assert torch.isfinite(out.state).all()
            assert torch.all(out.gains > 0)

    def test_a_diverged_baseline_cannot_underflow_the_gain_to_zero(self):
        """``exp`` underflows to a hard 0.0 in float32 well before the baseline
        stops being a number; the actuator divides by this."""
        hrm = _build()
        with torch.no_grad():
            hrm.log_gain_base.fill_(-300.0)
        hrm.reset()
        out = hrm(_emb(0))
        assert torch.all(out.gains > 0)
        assert torch.allclose(out.gains, out.gains.new_full(out.gains.shape, 1e-4))

    def test_the_gain_band_is_wide_enough_to_leave_the_fitted_prior(self):
        """The band is a guard rail, not the thing that limits learning."""
        prior = torch.tensor(FITTED_GAIN_PRIOR)
        assert prior.max() / prior.min() < 1.5, "axis spread, for scale"
        low, high = LOG_GAIN_LIMITS
        assert low < float(torch.log(prior).min()) - GAIN_LOG_RANGE
        assert high > float(torch.log(prior).max()) + GAIN_LOG_RANGE

    def test_gains_are_state_conditioned_once_the_head_is_trained(self):
        """Zero-init makes them constant at step 0; they must not STAY constant."""
        hrm = _build()
        with torch.no_grad():
            hrm.gain_head.weight.normal_(0.0, 0.5)
        hrm.reset()
        outs = _run_episode(hrm, n_real=2, dreams=2)
        assert not torch.allclose(outs[1].gains, outs[-1].gains)
        assert not torch.allclose(outs[1].gains[0], outs[1].gains[1]), (
            "gains must depend on the per-sample state, not only on the axis prior"
        )

    def test_gain_prior_stretches_to_a_wider_gain_dim(self):
        cfg = dataclasses.replace(CFG, hrm_gain_dim=6)
        prior = HRMBackbone._gain_prior(cfg)
        assert prior.shape == (6,)
        assert torch.allclose(prior[:3], torch.tensor(FITTED_GAIN_PRIOR))
        assert torch.all(prior > 0)


class TestRuntimeStateIsNotWeights:
    def test_state_dict_has_no_runtime_keys(self):
        hrm = _build()
        _run_episode(hrm)
        keys = list(hrm.state_dict())
        assert keys, "sanity: the module does have weights"
        for forbidden in ("anchor", "hidden", "window"):
            assert not [k for k in keys if forbidden in k], f"'{forbidden}' in {keys}"

    def test_reset_does_not_change_the_state_dict(self):
        hrm = _build()
        _run_episode(hrm)
        before = _snapshot(hrm)
        hrm.reset()
        after = _snapshot(hrm)
        assert before.keys() == after.keys()
        for k in before:
            assert torch.equal(before[k], after[k]), k

    def test_running_an_episode_does_not_change_the_state_dict(self):
        hrm = _build()
        before = _snapshot(hrm)
        _run_episode(hrm, n_real=3, dreams=5)
        after = _snapshot(hrm)
        assert before.keys() == after.keys()
        for k in before:
            assert torch.equal(before[k], after[k]), k

    def test_runtime_state_lives_in_plain_attributes(self):
        hrm = _build()
        _run_episode(hrm)
        param_ids = {id(p) for p in hrm.parameters()}
        buffer_ids = {id(b) for b in hrm.buffers()}
        for name in ("_anchor", "_window", "_slow", "_fast", "_eef_anchor"):
            value = getattr(hrm, name)
            assert value is not None, f"{name} should be populated mid-episode"
            if isinstance(value, torch.Tensor):
                assert not isinstance(value, torch.nn.Parameter), name
                assert id(value) not in param_ids and id(value) not in buffer_ids, name

    def test_module_registers_no_buffers_at_all(self):
        assert list(_build().named_buffers()) == []

    def test_load_state_dict_does_not_carry_episode_state(self):
        donor = _build(seed=0)
        _run_episode(donor)
        fresh = _build(seed=1)
        fresh.load_state_dict(donor.state_dict())
        assert fresh._anchor is None and fresh._slow is None and fresh._fast is None
        assert torch.all(fresh(_emb(0)).state == 0)


class TestDetachment:
    def test_backward_does_not_reach_the_previous_tick(self):
        hrm = _build()
        hrm(_emb(0))
        e1 = _emb(1).requires_grad_(True)
        hrm(e1)
        e2 = _emb(2).requires_grad_(True)
        out = hrm(e2)
        (out.state.sum() + out.gains.sum()).backward()
        assert e1.grad is None, "hidden state was not detached between steps"
        assert e2.grad is not None

    def test_emitted_state_is_differentiable(self):
        hrm = _build()
        hrm(_emb(0))
        e = _emb(1).requires_grad_(True)
        hrm(e).state.sum().backward()
        assert e.grad is not None and torch.isfinite(e.grad).all()

    def test_stored_states_never_require_grad(self):
        hrm = _build()
        _run_episode(hrm)
        assert not hrm._slow.requires_grad and not hrm._fast.requires_grad
        assert not hrm._anchor.requires_grad and not hrm._eef_anchor.requires_grad


class TestTwoTimescales:
    def test_dream_tick_moves_the_fast_state_but_not_the_slow_one(self):
        hrm = _build()
        hrm(_emb(0))
        hrm(_emb(1), is_real=True)
        slow_before = hrm._slow.clone()
        fast_before = hrm._fast.clone()
        out_before = hrm(_emb(2), is_real=False)

        assert torch.equal(hrm._slow, slow_before), "slow module stepped on a dream tick"
        assert not torch.allclose(hrm._fast, fast_before), "fast module did not step"

        out_after = hrm(_emb(3), is_real=False)
        assert not torch.allclose(out_before.state, out_after.state), (
            "the emitted state must be fresh every tick — v7 held it for 14 ticks"
        )

    def test_real_tick_moves_both_states(self):
        hrm = _build()
        hrm(_emb(0))
        hrm(_emb(1), is_real=False)
        slow_before = hrm._slow.clone()
        fast_before = hrm._fast.clone()
        hrm(_emb(2), is_real=True)
        assert not torch.allclose(hrm._slow, slow_before)
        assert not torch.allclose(hrm._fast, fast_before)

    def test_dream_only_sequence_gives_slow_parameters_no_gradient(self):
        hrm = _build()
        hrm(_emb(0))
        for k in range(5):
            out = hrm(_emb(10 + k), is_real=False)
        _scalar_loss(out).backward()

        slow_grads = [p.grad for p in hrm.slow_parameters()]
        assert slow_grads, "slow group is empty"
        assert all(g is None for g in slow_grads), (
            "a dream-only rollout backpropagated into the 2 Hz path"
        )
        assert all(p.grad is not None for p in hrm.fast_parameters()), "fast group is empty"
        # Not just "a gradient exists": the 30 Hz path must carry real signal,
        # or the assertion above would hold for a fast module wired to nothing.
        for module in (hrm.fast_in, hrm.fast_core, hrm.out_norm):
            for name, p in module.named_parameters():
                assert p.grad.abs().sum() > 0, f"dead 30 Hz gradient: {name}"

    def test_real_tick_gives_slow_parameters_gradient(self):
        hrm = _build()
        hrm(_emb(0))
        hrm(_emb(1), is_real=True)
        out = hrm(_emb(2), is_real=True, eef=_eef(2))
        _scalar_loss(out).backward()
        # EVERY slow parameter, not `any` — with a degenerate loss `any` passes
        # on float32 dust (see _scalar_loss), so the strong form is what makes
        # this a statement about the 2 Hz path being trainable.
        for name, p in hrm.named_parameters():
            if any(p is q for q in hrm.slow_parameters()):
                assert p.grad is not None and p.grad.abs().sum() > 0, name

    def test_slow_and_fast_groups_partition_the_parameters(self):
        """Guards the gradient tests above: a parameter in neither group would
        make them vacuous instead of failing."""
        hrm = _build()
        slow_ids = [id(p) for p in hrm.slow_parameters()]
        fast_ids = [id(p) for p in hrm.fast_parameters()]
        all_ids = [id(p) for p in hrm.parameters()]
        assert len(set(slow_ids)) == len(slow_ids)
        assert len(set(fast_ids)) == len(fast_ids)
        assert not set(slow_ids) & set(fast_ids)
        assert set(slow_ids) | set(fast_ids) == set(all_ids)

    def test_fast_module_reads_anchor_relative_drift_every_tick(self):
        """The 30 Hz half of job (a): the fast module reads ``emb − anchor``
        itself rather than inheriting anchor drift through the slow summary.

        Isolating that needs two backbones differing ONLY in their anchor, so
        the recurrent states are copied across after both have run a real tick;
        the compared tick is then a dream tick, on which the slow state is
        frozen by construction and the anchor term is the sole remaining
        difference. Without it both emit the same code.
        """
        a, b = _build(), _build()
        a(_emb(0))
        b(_emb(5))
        assert not torch.allclose(a._anchor, b._anchor), "sanity: anchors differ"
        a(_emb(1), is_real=True)
        b(_emb(1), is_real=True)
        b._slow, b._fast = a._slow.clone(), a._fast.clone()

        dream = _emb(9)
        assert not torch.allclose(a(dream, is_real=False).state, b(dream, is_real=False).state), (
            "the fast module ignores frame_emb − anchor; anchor drift is only "
            "reaching it at 2 Hz through the slow state"
        )

    def test_only_real_ticks_extend_the_context_window(self):
        hrm = _build()
        hrm(_emb(0))
        for k in range(6):
            hrm(_emb(20 + k), is_real=False)
        assert len(hrm._window) == 1, "an imagined latent entered the 2 Hz history"
        hrm(_emb(30), is_real=True)
        assert len(hrm._window) == 2

    def test_window_is_capped_by_context_window(self):
        hrm = _build()
        for r in range(CFG.context_window + 6):
            hrm(_emb(r), is_real=True)
        assert len(hrm._window) == CFG.context_window

    def test_each_horizon_reads_a_DISTINCT_history_entry(self):
        """Job (a) is *multi*-horizon: ``cfg.drift_horizons`` must index
        different frames, not degenerate to "the previous real frame" repeated.

        End-to-end perturbation cannot show this — an old frame also reaches the
        current tick through the accumulated slow state, so an output change
        proves nothing about the lag indexing. So this drives ``_context_read``
        directly with a hand-edited window: it is a pure function of the window,
        the anchor and the slow state, and only a lag ≥ 2 touches the entry
        edited here.
        """
        assert max(CFG.drift_horizons) <= CFG.context_window, "sanity: lags fit"
        hrm = _build()
        for r in range(CFG.context_window):
            hrm(_emb(r), is_real=True)
        assert len(hrm._window) == CFG.context_window

        probe = _emb(50)
        before = hrm._context_read(probe).clone()
        history = list(hrm._window)
        history[0] = history[0] + 5.0  # oldest entry == the longest lag only
        hrm._window = deque(history, maxlen=CFG.context_window)
        after = hrm._context_read(probe)
        assert not torch.allclose(before, after, atol=1e-6), (
            "editing the oldest window entry changed nothing: every horizon is "
            "reading the same (most recent) frame"
        )

    def test_horizons_longer_than_the_history_are_clamped(self):
        """Lag 8 on tick 2 must reuse the oldest entry, not crash or invent one."""
        assert max(CFG.drift_horizons) > 2
        hrm = _build()
        hrm(_emb(0))
        out = hrm(_emb(1), is_real=True)
        assert torch.isfinite(out.state).all()


class TestEpisodeSemantics:
    def test_reset_makes_episodes_independent(self):
        hrm = _build()
        first = [o.state.clone() for o in _run_episode(hrm, n_real=3, dreams=2)]
        hrm.reset()
        second = [o.state.clone() for o in _run_episode(hrm, n_real=3, dreams=2)]
        for a, b in zip(first, second):
            assert torch.allclose(a, b, atol=1e-6)

    def test_batch_size_change_silently_resets(self):
        hrm = _build()
        _run_episode(hrm, batch=2)
        out = hrm(_emb(0, batch=5))
        assert out.state.shape == (5, CFG.hrm_dim)
        assert torch.all(out.state == 0), "a batch change must re-anchor"
        assert hrm._anchor.shape[0] == 5 and len(hrm._window) == 1

    def test_batch_independence(self):
        """Row i of a batched rollout must equal that row run on its own."""
        batched = _build()
        singles = [_build(), _build(), _build()]
        seq = [(0, True), (1, False), (2, False), (3, True), (4, False)]
        for seed, real in seq:
            emb = _emb(seed, batch=3)
            eef = _eef(seed, batch=3) if real else None
            out = batched(emb, is_real=real, eef=eef)
            for i, single in enumerate(singles):
                one = single(
                    emb[i: i + 1], is_real=real, eef=None if eef is None else eef[i: i + 1]
                )
                assert torch.allclose(out.state[i], one.state[0], atol=1e-5)
                assert torch.allclose(out.gains[i], one.gains[0], atol=1e-6)

    def test_anchor_is_the_first_real_frame(self):
        hrm = _build()
        first = _emb(0)
        hrm(first)
        _run_episode(hrm, n_real=2, dreams=2)
        assert torch.equal(hrm._anchor, first)


class TestEndEffector:
    def test_eef_is_optional(self):
        hrm = _build()
        outs = _run_episode(hrm, n_real=3, dreams=2, with_eef=False)
        assert torch.isfinite(outs[-1].state).all()
        assert hrm._eef_anchor is None

    def test_eef_presence_changes_the_state(self):
        without, with_eef = _build(), _build()
        without(_emb(0))
        with_eef(_emb(0), eef=_eef(0))
        a = without(_emb(1))
        b = with_eef(_emb(1), eef=_eef(1))
        assert not torch.allclose(a.state, b.state), "eef input is ignored"

    def test_eef_VALUES_change_the_state_not_just_the_validity_flag(self):
        """The test above passes on the validity scalar alone — an eef_proj that
        multiplied the measurement by zero would still flip 0 → 1 and shift the
        state. Feed two episodes the same everything INCLUDING eef validity and
        eef anchor, differing only in the measured pose."""
        h1, h2 = _build(), _build()
        anchor_pose = _eef(0)
        h1(_emb(0), eef=anchor_pose)
        h2(_emb(0), eef=anchor_pose)
        one = h1(_emb(1), eef=_eef(1))
        two = h2(_emb(1), eef=_eef(2))
        assert not torch.allclose(one.state, two.state), (
            "the measured EEF pose is discarded; only its presence is read"
        )

    def test_eef_anchor_is_stored_once(self):
        hrm = _build()
        anchor = _eef(0)
        hrm(_emb(0), eef=anchor)
        assert torch.equal(hrm._eef_anchor, anchor)
        hrm(_emb(1), eef=_eef(1))
        assert torch.equal(hrm._eef_anchor, anchor), "eef anchor moved mid-episode"

    def test_eef_appearing_mid_episode_anchors_there(self):
        hrm = _build()
        hrm(_emb(0))
        assert hrm._eef_anchor is None
        late = _eef(5)
        hrm(_emb(1), eef=late)
        assert torch.equal(hrm._eef_anchor, late)

    def test_wrong_eef_width_raises(self):
        hrm = _build()
        hrm(_emb(0))
        with pytest.raises(ValueError, match="waypoint_dim"):
            hrm(_emb(1), eef=torch.zeros(2, CFG.waypoint_dim + 4))

    def test_wrong_eef_width_raises_on_the_ANCHOR_tick_too(self):
        """The anchor tick is the one that stores ``_eef_anchor``, so an
        unvalidated pose there is the expensive case: a 7-D proprio vector
        accepted here poisons every later subtraction and surfaces one tick
        late as a broadcast error naming neither the argument nor the episode."""
        hrm = _build()
        with pytest.raises(ValueError, match="waypoint_dim"):
            hrm(_emb(0), eef=torch.zeros(2, CFG.waypoint_dim + 4))

    def test_eef_rank_and_batch_are_validated(self):
        hrm = _build()
        hrm(_emb(0))
        with pytest.raises(ValueError, match="waypoint_dim"):
            hrm(_emb(1), eef=torch.zeros(CFG.waypoint_dim))  # unbatched
        with pytest.raises(ValueError, match="waypoint_dim"):
            hrm(_emb(1), eef=torch.zeros(3, CFG.waypoint_dim))  # batch mismatch


class TestBudget:
    def test_param_count_under_cap(self, capsys):
        hrm = HRMBackbone(CFG)
        total = sum(p.numel() for p in hrm.parameters() if p.requires_grad)
        slow = sum(p.numel() for p in hrm.slow_parameters())
        fast = sum(p.numel() for p in hrm.fast_parameters())
        with capsys.disabled():
            print(
                f"\nHRMBackbone: {total:,d} params ({total / 1e6:.3f}M) = "
                f"slow {slow:,d} + fast {fast:,d}"
            )
        assert total == slow + fast
        assert total <= HRM_PARAM_CAP, (
            f"HRMBackbone has {total:,d} params, exceeding its {HRM_PARAM_CAP:,d} cap"
        )

    def test_stays_on_the_input_device_and_dtype(self):
        hrm = _build()
        out = hrm(_emb(0))
        assert out.state.dtype == torch.float32 and out.gains.dtype == torch.float32
        assert out.state.device == _emb(0).device
