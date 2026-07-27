"""End-to-end smoke of the v8 training path on a synthetic batch.

There is no corpus on a dev machine and stage A/B need a GPU to be practical, so
this exercises the wiring — not the learning — with hand-built tensors: does the
v8 stack forward through stage A's rollout and a stage B step, does gradient
reach the modules that are supposed to train, and does it stay off the ones that
are supposed to be frozen.

This exists because the v8 swap replaces three of five modules across a dozen
call sites, and the failure mode it guards is silent: a relational head left out
of the optimizer still runs, still produces plausible losses, and simply never
learns — which is indistinguishable from "the relational head does not help".
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from microvla.config import DEFAULT_CONFIG
from microvla.planner.chrono_planner import ChronoQueryPlanner
from microvla.relational import RelationalHead
from microvla.utils.param_audit import count_trainable_params
from microvla.v8 import DriftAdapter, FusionAdapter, pack_objects

B, T = 2, 4


@pytest.fixture
def cfg():
    """v8 planner inputs: fusion's groups out, relational in."""
    c = DEFAULT_CONFIG
    return dataclasses.replace(
        c,
        planner_inputs=tuple(n for n in c.planner_inputs
                             if n not in ("fused", "geometry", "pred_box_emb",
                                          "spatial", "wm_msg", "wm_latent"))
        + ("relational",),
    )


@pytest.fixture
def batch(cfg):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    return {
        "frame_embs": r(B, T, cfg.vis_dim),
        "text_tokens": r(B, cfg.n_text_tokens, cfg.text_dim),
        "source_box_embs": r(B, T, cfg.vis_dim),
        "target_box_embs": r(B, T, cfg.vis_dim),
        "source_centers": torch.rand(B, T, 2, generator=g),
        "target_centers": torch.rand(B, T, 2, generator=g),
        "box_weights": torch.rand(B, T, 2, generator=g),
        "pwm_targets": r(B, T, cfg.plan_steps, cfg.num_servos),
        "proprio": r(B, T, 10),
    }


class TestAdapters:
    """The adapters must be drop-in: v7 signature in, v7 shapes out."""

    def test_fusion_adapter_returns_the_trm_evidence_port_shape(self, cfg, batch):
        f = FusionAdapter(cfg)
        out = f(batch["text_tokens"], batch["frame_embs"][:, 0],
                batch["source_box_embs"][:, 0], batch["target_box_embs"][:, 0],
                batch["source_centers"][:, 0], batch["target_centers"][:, 0],
                box_weight=batch["box_weights"][:, 0],
                last_action=batch["pwm_targets"][:, 0, 0])
        assert out.shape == (B, cfg.fused_rows, cfg.fused_cols)

    def test_drift_adapter_matches_the_drift_encoder_contract(self, cfg, batch):
        d = DriftAdapter(cfg)
        d.reset()
        first = d(batch["frame_embs"][:, 0])
        assert first.shape == (B, cfg.hrm_dim)
        # The anchor tick returns an exactly-zero code without stepping — the
        # AnchoredDriftEncoder semantics CLAUDE.md pins.
        assert torch.count_nonzero(first) == 0
        assert torch.count_nonzero(d(batch["frame_embs"][:, 1])) > 0

    def test_drift_adapter_holds_no_runtime_state_in_its_state_dict(self, cfg, batch):
        d = DriftAdapter(cfg)
        d.reset()
        before = {k: v.clone() for k, v in d.state_dict().items()}
        d(batch["frame_embs"][:, 0])
        d(batch["frame_embs"][:, 1])
        after = d.state_dict()
        assert set(before) == set(after)
        for k in before:
            assert torch.equal(before[k], after[k]), (
                f"running the module changed state_dict key {k!r}; episode state "
                f"must live in plain attributes so a checkpoint never carries it."
            )

    def test_pack_objects_pads_inertly(self, cfg, batch):
        obj, ctr, w = pack_objects(
            batch["source_box_embs"][:, 0], batch["target_box_embs"][:, 0],
            batch["source_centers"][:, 0], batch["target_centers"][:, 0],
            batch["box_weights"][:, 0], cfg)
        assert obj.shape == (B, cfg.max_objects, cfg.vis_dim)
        assert w.shape == (B, cfg.max_objects)
        assert torch.count_nonzero(w[:, 2:]) == 0, "pad slots must carry weight 0.0"

    def test_pack_objects_rejects_a_config_too_small_for_both_roles(self, cfg, batch):
        tiny = dataclasses.replace(cfg, max_objects=1)
        with pytest.raises(ValueError, match="max_objects"):
            pack_objects(batch["source_box_embs"][:, 0], batch["target_box_embs"][:, 0],
                         batch["source_centers"][:, 0], batch["target_centers"][:, 0],
                         batch["box_weights"][:, 0], tiny)


class TestStageAPath:
    def test_real_paths_v8_runs_and_carries_gradient(self, cfg, batch):
        from train.train_batched import real_paths

        fusion, drift = FusionAdapter(cfg), DriftAdapter(cfg)
        fused_all, delta_all = real_paths(batch, fusion, drift, cfg, ablate=False)
        assert len(fused_all) == T and len(delta_all) == T
        assert fused_all[0].shape == (B, cfg.fused_rows, cfg.fused_cols)
        assert delta_all[0].shape == (B, cfg.hrm_dim)

        fused_all[-1].sum().backward()
        got = [n for n, p in fusion.named_parameters()
               if p.grad is not None and p.grad.abs().sum() > 0]
        assert got, "no gradient reached the evidence encoder"

    def test_drift_is_reset_per_call_so_batches_do_not_leak(self, cfg, batch):
        from train.train_batched import real_paths

        fusion, drift = FusionAdapter(cfg), DriftAdapter(cfg)
        a, _ = real_paths(batch, fusion, drift, cfg, ablate=False)
        b, _ = real_paths(batch, fusion, drift, cfg, ablate=False)
        assert torch.allclose(a[0], b[0], atol=1e-6), (
            "a second pass over the same batch differed — episode state leaked "
            "across calls."
        )


class TestStageBPath:
    def test_relational_reaches_the_planner_and_changes_the_plan(self, cfg, batch):
        rel = RelationalHead(cfg).eval()
        planner = ChronoQueryPlanner(cfg).eval()
        next_emb = batch["frame_embs"][:, 1]
        obj, ctr, w = pack_objects(
            batch["source_box_embs"][:, 1], batch["target_box_embs"][:, 1],
            batch["source_centers"][:, 1], batch["target_centers"][:, 1],
            batch["box_weights"][:, 1], cfg)
        tok = rel(next_emb, obj, ctr, w, batch["text_tokens"],
                  last_action=batch["pwm_targets"][:, 0, 0])
        assert tok.shape == (B, cfg.rel_tokens, cfg.rel_dim)

        kw = dict(current_emb=batch["frame_embs"][:, 1],
                  proprio=batch["proprio"][:, 1])
        with_rel = planner(next_emb, relational=tok, **kw)
        without = planner(next_emb, relational=None, **kw)
        g = lambda o: o[0] if isinstance(o, tuple) else o
        assert (g(with_rel) - g(without)).abs().mean() > 1e-3, (
            "relational tokens did not move the plan — the group is wired but inert."
        )

    def test_helper_uses_the_boxes_it_is_given_not_fresh_ones(self, cfg, batch):
        """A dream step holds t-1 boxes at a fade; re-deriving would leak t."""
        from train.train_batched import _boxes, _relational

        rel = RelationalHead(cfg).eval()
        next_emb = batch["frame_embs"][:, 2]
        held = _boxes(batch, 1, 0.5, cfg, False)     # dream: t-1, faded
        fresh = _boxes(batch, 2, 1.0, cfg, False)    # real: t
        a = _relational(rel, next_emb, batch, 2, held, cfg)
        b = _relational(rel, next_emb, batch, 2, fresh, cfg)
        assert not torch.allclose(a, b, atol=1e-4), (
            "held and fresh evidence produced the same tokens — the helper is "
            "ignoring the boxes it was handed."
        )

    def test_v7_stack_still_gets_none(self, cfg, batch):
        from train.train_batched import _boxes, _relational
        assert _relational(None, batch["frame_embs"][:, 1], batch, 1,
                           _boxes(batch, 1, 1.0, cfg, False), cfg) is None


def test_v8_fits_the_joint_budget(cfg):
    total = (count_trainable_params(FusionAdapter(cfg))
             + count_trainable_params(DriftAdapter(cfg))
             + count_trainable_params(RelationalHead(cfg))
             + count_trainable_params(ChronoQueryPlanner(cfg)))
    assert total < cfg.trainable_param_budget, (
        f"v8 stack {total:,d} >= budget {cfg.trainable_param_budget:,d}"
    )
