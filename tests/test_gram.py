"""GRAM stochastic guidance on HRM + planner heads (not TRM)."""

from __future__ import annotations

import dataclasses

import torch

from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.hrm import HRMBackbone
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.utils.gram import StochasticGuidance


def _gram_cfg(**kwargs) -> MicroVLAConfig:
    base = dict(
        gram_hrm=True,
        gram_planner=True,
        gram_noise_dim=16,
        gram_n_samples=4,
    )
    base.update(kwargs)
    return dataclasses.replace(DEFAULT_CONFIG, **base)


class TestStochasticGuidance:
    def test_shapes_and_zero_init_near_identity(self):
        g = StochasticGuidance(dim=32, noise_dim=8)
        u = torch.randn(2, 32)
        z, _ = g.sample_prior(u, deterministic=True)
        # Zero-init μ/up → z ≈ u before any training.
        assert torch.allclose(z, u, atol=1e-5)

    def test_posterior_kl_finite(self):
        g = StochasticGuidance(dim=32, noise_dim=8, target_dim=10)
        u = torch.randn(4, 32)
        tgt = torch.randn(4, 10)
        z, kl = g.sample_posterior(u, tgt)
        assert z.shape == u.shape
        assert torch.isfinite(kl)
        assert kl.ndim == 0

    def test_n_samples_leading_dim(self):
        g = StochasticGuidance(dim=16, noise_dim=4)
        u = torch.randn(3, 16)
        z, _ = g.sample_prior(u, n_samples=5, deterministic=False)
        assert z.shape == (5, 3, 16)


class TestGRAMHRM:
    def test_off_by_default_no_module(self):
        hrm = HRMBackbone(DEFAULT_CONFIG)
        assert hrm.gram_slow is None

    def test_slow_path_still_zero_on_anchor(self):
        cfg = _gram_cfg()
        hrm = HRMBackbone(cfg)
        hrm.reset()
        out = hrm(torch.randn(2, cfg.vis_dim), is_real=True)
        assert torch.equal(out.state, torch.zeros_like(out.state))

    def test_train_real_tick_uses_gram(self):
        cfg = _gram_cfg()
        hrm = HRMBackbone(cfg)
        hrm.train()
        hrm.reset()
        # Anchor tick
        hrm(torch.randn(2, cfg.vis_dim), is_real=True)
        # Second real tick steps slow + gram
        out = hrm(torch.randn(2, cfg.vis_dim), is_real=True)
        assert torch.isfinite(out.state).all()
        assert (out.gains > 0).all()
        # Slow params include gram_slow
        names = {id(p) for p in hrm.slow_parameters()}
        assert any(id(p) in names for p in hrm.gram_slow.parameters())


class TestGRAMPlanner:
    def test_off_by_default(self):
        p = ChronoQueryPlanner(DEFAULT_CONFIG)
        assert p.gram_feat is None

    def test_single_sample_shape(self):
        cfg = _gram_cfg(planner_inputs=("next_emb", "current_emb", "state_delta"))
        p = ChronoQueryPlanner(cfg)
        p.eval()
        plan = p(torch.randn(2, cfg.vis_dim),
                 current_emb=torch.randn(2, cfg.vis_dim),
                 state_delta=torch.randn(2, cfg.state_dim))
        assert plan.shape == (2, cfg.plan_steps, cfg.num_servos)
        assert plan.min() >= -1.0 and plan.max() <= 1.0

    def test_width_scaling_shape(self):
        cfg = _gram_cfg(planner_inputs=("next_emb",), gram_n_samples=5)
        p = ChronoQueryPlanner(cfg)
        p.eval()
        plan, grip = p(torch.randn(2, cfg.vis_dim), return_aux=True, n_samples=5)
        assert plan.shape == (2, cfg.plan_steps, cfg.num_servos)
        assert grip.shape == (2, cfg.plan_steps)

    def test_posterior_returns_kl(self):
        cfg = _gram_cfg(planner_inputs=("next_emb",))
        p = ChronoQueryPlanner(cfg)
        p.train()
        tgt = torch.randn(2, cfg.plan_steps, cfg.num_servos).tanh()
        out = p(torch.randn(2, cfg.vis_dim), return_aux=True, gram_target=tgt)
        plan, grip, kl = out
        assert plan.shape[0] == 2
        assert torch.isfinite(kl)

    def test_param_budget_with_gram(self):
        cfg = _gram_cfg()
        p = ChronoQueryPlanner(cfg)
        n = sum(x.numel() for x in p.parameters() if x.requires_grad)
        assert n <= 2_500_000, f"planner over budget with GRAM: {n:,}"
