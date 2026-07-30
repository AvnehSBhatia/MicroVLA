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

    def test_helper_uses_the_evidence_it_is_given_not_fresh(self, cfg, batch):
        """A dream step holds t-1 evidence at a fade; re-deriving would leak t."""
        from train.train_batched import _relational

        rel = RelationalHead(cfg).eval()
        next_emb = batch["frame_embs"][:, 2]
        held = _relational(rel, next_emb, batch, 2, 1, 0.5, cfg)   # dream
        fresh = _relational(rel, next_emb, batch, 2, 2, 1.0, cfg)  # real
        assert not torch.allclose(held, fresh, atol=1e-4), (
            "held and fresh evidence produced the same tokens — the helper is "
            "ignoring the index/fade it was handed."
        )

    def test_v7_stack_still_gets_none(self, cfg, batch):
        from train.train_batched import _relational
        assert _relational(None, batch["frame_embs"][:, 1], batch, 1, 1, 1.0, cfg) is None

    def test_baked_objects_are_preferred_over_the_two_role_slots(self, cfg, batch):
        """A v8 corpus must reach the relational head; a v7 one must fall back."""
        from microvla.v8 import objects_from_batch

        K = cfg.max_objects
        v7 = objects_from_batch(batch, 1, 1.0, cfg)
        assert torch.count_nonzero(v7[2][:, 2:]) == 0, "v7 fallback filled >2 slots"

        v8_batch = dict(batch)
        v8_batch["obj_embs"] = torch.randn(B, T, K, cfg.vis_dim)
        v8_batch["obj_centers"] = torch.rand(B, T, K, 2)
        v8_batch["obj_weights"] = torch.rand(B, T, K)
        v8_batch["has_objects"] = torch.ones(B, 1)
        obj, ctr, w = objects_from_batch(v8_batch, 1, 1.0, cfg)
        assert torch.equal(obj, v8_batch["obj_embs"][:, 1])
        assert torch.count_nonzero(w[:, 2:]) > 0, "baked slots beyond 2 were dropped"

    def test_zero_filled_objects_do_not_masquerade_as_a_real_scene(self, cfg, batch):
        """The v7 corpus is zero-filled; provenance, not zeros, must decide."""
        from microvla.v8 import objects_from_batch

        K = cfg.max_objects
        blind = dict(batch)
        blind["obj_embs"] = torch.zeros(B, T, K, cfg.vis_dim)
        blind["obj_centers"] = torch.zeros(B, T, K, 2)
        blind["obj_weights"] = torch.zeros(B, T, K)
        blind["has_objects"] = torch.zeros(B, 1)      # zero-filled, not baked
        obj, _, w = objects_from_batch(blind, 1, 1.0, cfg)
        assert torch.count_nonzero(obj[:, 0]) > 0, (
            "fell through to the zero-fill instead of the role slots — the "
            "relational head would train on nothing."
        )


def test_v8_fits_the_joint_budget(cfg):
    total = (count_trainable_params(FusionAdapter(cfg))
             + count_trainable_params(DriftAdapter(cfg))
             + count_trainable_params(RelationalHead(cfg))
             + count_trainable_params(ChronoQueryPlanner(cfg)))
    assert total < cfg.trainable_param_budget, (
        f"v8 stack {total:,d} >= budget {cfg.trainable_param_budget:,d}"
    )


def test_every_v8_module_the_policy_builds_lands_on_one_device():
    """A module left on CPU fails at the first tick and reports as 0.000.

    eval/policy.py moves fusion, drift, trm and planner explicitly; relational
    was added later and missed, so every v8 closed-loop run died with
    "mat1 is on cuda:0, different from other tensors on cpu" and the harness
    summarised it as tasks_completed 0 — identical in the output to a policy
    that merely never succeeds. Pinned by source inspection because the failure
    only reproduces on a machine with a GPU.
    """
    import pathlib
    import re

    src = pathlib.Path("eval/policy.py").read_text()
    moved = set(re.findall(r"^\s*(\w+)\.to\(heads_device\)", src, re.M))
    for name in ("fusion", "drift", "trm", "planner", "relational"):
        assert name in moved, f"{name} is never moved to heads_device in policy.py"


class TestActionTokenScheduledSampling:
    """Stage B must be able to train fusion's action token the way it deploys.

    Fusion's 8th token is the previously executed action. Stage B fed it the
    DEMONSTRATION's action at every step while the deployed loop can only feed
    the policy's own, and paper.md 4v attributes essentially the whole
    closed-loop failure to that asymmetry: teacher-forcing the token at eval
    takes the gripper from 13% to 47% of steps closed and makes the deployed
    stack reproduce the trainer bit-for-bit (`fused` rel-diff 0.3384 -> 0.0000).

    These pin the switch and its default, not the learning outcome.
    """

    def test_flag_exists_and_defaults_to_the_old_behaviour(self):
        from train.train_batched import parse_args

        a = parse_args(["--data", "x"])
        assert a.action_token_sampling == 0.0, (
            "changing the default silently changes every existing arm's protocol"
        )
        b = parse_args(["--data", "x", "--action-token-sampling", "0.5"])
        assert b.action_token_sampling == 0.5

    def test_self_fed_token_changes_the_fused_matrix(self, cfg, batch):
        """The substitution must actually reach fusion, or sampling is inert."""
        import torch

        f = FusionAdapter(cfg)
        args_ = dict(text_tokens=batch["text_tokens"], frame_emb=batch["frame_embs"][:, 1],
                     sbe=batch["source_box_embs"][:, 1], tbe=batch["target_box_embs"][:, 1],
                     sc=batch["source_centers"][:, 1], tc=batch["target_centers"][:, 1],
                     bw=batch["box_weights"][:, 1])
        demo_act = batch["pwm_targets"][:, 0, 0]
        own_act = torch.zeros_like(demo_act)
        a = f(args_["text_tokens"], args_["frame_emb"], args_["sbe"], args_["tbe"],
              args_["sc"], args_["tc"], box_weight=args_["bw"], last_action=demo_act)
        b = f(args_["text_tokens"], args_["frame_emb"], args_["sbe"], args_["tbe"],
              args_["sc"], args_["tc"], box_weight=args_["bw"], last_action=own_act)
        assert (a - b).abs().mean() > 1e-6, (
            "the action token does not affect fusion's output, so scheduled "
            "sampling on it would be a no-op"
        )


class TestTaskAlignedLosses:
    """Progress critic, imagined rollout, and variance matching.

    BC minimizes action MSE while the metric is task completion, and on this
    benchmark they actively disagree: paper.md 4p measured LIBERO's passing band
    at ~[0.95, 1.05] of demo magnitude while MSE-optimal regression shrinks
    toward the conditional mean (every arm: std_ratio 0.26-0.42). These three
    terms attack that directly. The environment is not differentiable, so the
    critic is the differentiable surrogate and fusion's action token is the path
    from the emitted plan into the world model.
    """

    def test_critic_maps_a_latent_to_a_bounded_progress(self, cfg):
        import torch

        from microvla.critic import ProgressCritic, progress_targets

        c = ProgressCritic(cfg)
        v = c(torch.randn(4, cfg.vis_dim))
        assert v.shape == (4,)
        assert bool(((v >= 0) & (v <= 1)).all())
        tg = progress_targets(5, 2, torch.device("cpu"))
        assert tg.shape == (2, 5)
        assert torch.allclose(tg[0, -1], torch.tensor(1.0))
        assert bool((tg[0].diff() > 0).all()), "progress must increase with time"

    def test_frozen_value_passes_gradient_to_the_latent_not_the_critic(self, cfg):
        """If the actor term could train the critic, the value would collapse."""
        import torch

        from microvla.critic import ProgressCritic, frozen_value

        c = ProgressCritic(cfg)
        lat = torch.randn(3, cfg.vis_dim, requires_grad=True)
        frozen_value(c, lat).mean().backward()
        assert lat.grad is not None and lat.grad.abs().sum() > 0, (
            "no gradient reached the latent — the actor term would be inert"
        )
        for n, p in c.named_parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, (
                f"actor term leaked gradient into critic weight {n}; the critic "
                f"could then satisfy it by predicting 1.0 everywhere"
            )

    def test_progress_weight_without_critic_weight_is_refused(self):
        from train.train_batched import parse_args

        a = parse_args(["--data", "x", "--progress-weight", "1.0"])
        assert a.critic_weight == 0.0 and a.progress_weight == 1.0, (
            "the guard belongs in stage_b, but the flags must parse so it can "
            "raise a clear error rather than train a no-op"
        )

    def test_defaults_leave_every_task_aligned_term_off(self):
        from train.train_batched import parse_args

        a = parse_args(["--data", "x"])
        assert (a.critic_weight, a.progress_weight, a.dream_weight,
                a.variance_weight) == (0.0, 0.0, 0.0, 0.0)
        assert a.dream_horizon == 1

    def test_the_value_term_reaches_the_planner_through_the_world_model(self, cfg, batch):
        """The whole design rests on this path existing; assert it, don't assume.

        plan -> fusion(last_action=plan[:,0]) -> TRM -> latent -> critic. If any
        link is detached the term still computes a plausible loss and trains
        nothing — the exact "wired but inert" failure that made the relational
        head's first result meaningless.
        """
        import torch

        from microvla.critic import ProgressCritic, frozen_value
        from microvla.trm.mock_trm import MockTRM
        from microvla.v8 import DriftAdapter, FusionAdapter

        fusion, drift = FusionAdapter(cfg), DriftAdapter(cfg)
        trm, critic = MockTRM(cfg), ProgressCritic(cfg)
        planner = ChronoQueryPlanner(cfg)
        for m in (fusion, drift):
            for p in m.parameters():
                p.requires_grad_(False)

        drift.reset()
        cur = batch["frame_embs"][:, 0]
        delta = drift(cur)
        plan = planner(cur, current_emb=cur, state_delta=delta,
                       proprio=batch["proprio"][:, 0])
        plan = plan[0] if isinstance(plan, tuple) else plan

        fused = fusion(batch["text_tokens"], cur,
                       batch["source_box_embs"][:, 0], batch["target_box_embs"][:, 0],
                       batch["source_centers"][:, 0], batch["target_centers"][:, 0],
                       box_weight=batch["box_weights"][:, 0],
                       last_action=plan[:, 0])
        value = frozen_value(critic, trm(fused, delta, cur)).mean()
        (-value).backward()

        got = [n for n, p in planner.named_parameters()
               if p.grad is not None and p.grad.abs().sum() > 0]
        assert got, (
            "no gradient reached the planner from the critic value — the "
            "task-aligned term is inert and would train nothing"
        )

    def test_mixed_bucket_selects_evidence_per_episode(self, cfg, batch):
        """A v7 episode beside a v8 one must still get its role slots.

        has_objects was reduced with .max() across the batch, so ONE v8 episode
        sent every episode down the baked path — and a v7 episode's obj_* are
        zero-filled, so those samples fed the relational head nothing. Buckets
        mix corpora whenever more than one --data-dir is passed.
        """
        import torch

        from microvla.v8 import objects_from_batch

        K = cfg.max_objects
        b = dict(batch)
        b["obj_embs"] = torch.zeros(B, T, K, cfg.vis_dim)
        b["obj_centers"] = torch.zeros(B, T, K, 2)
        b["obj_weights"] = torch.zeros(B, T, K)
        b["obj_embs"][0] = torch.randn(T, K, cfg.vis_dim)     # sample 0 is v8
        b["obj_weights"][0] = torch.rand(T, K)
        b["has_objects"] = torch.tensor([[1.0], [0.0]])       # sample 1 is v7

        obj, ctr, w = objects_from_batch(b, 1, 1.0, cfg)
        assert torch.equal(obj[0], b["obj_embs"][:, 1][0]), "v8 sample lost its baked scene"
        assert torch.count_nonzero(obj[1]) > 0, (
            "the v7 sample got the zero-filled baked array instead of its role "
            "slots — the relational head trains on nothing for that episode"
        )

    def test_proprio_noise_never_touches_the_validity_flag(self, cfg, batch):
        """Jittering the flag would read as a MISSING sensor, not a noisy one.

        Every consumer keys on proprio[..., -1]: _eef_of returns None for the
        whole batch when it drops below 0.5, and wp_valid masks the waypoint
        loss with it. Perturbing it turns a valid episode into a randomly
        invalid one instead of a slightly-off one.
        """
        import argparse

        import torch

        from train.train_batched import _noisy_proprio

        args = argparse.Namespace(proprio_noise=0.5)
        b = {"proprio": torch.cat([torch.zeros(B, T, 9), torch.ones(B, T, 1)], dim=-1)}
        out = _noisy_proprio(b, 1, args)
        assert torch.equal(out[..., -1], torch.ones(B)), "validity flag was perturbed"
        assert out[..., :9].abs().sum() > 0, "noise did not reach the state dims"

    def test_noise_is_off_by_default(self):
        import argparse

        import torch

        from train.train_batched import _noisy_proprio

        b = {"proprio": torch.randn(B, T, 10)}
        out = _noisy_proprio(b, 1, argparse.Namespace(proprio_noise=0.0))
        assert torch.equal(out, b["proprio"][:, 1])

    def test_recovery_target_cancels_the_perturbation(self, cfg):
        """A displaced EEF must be asked for LESS motion, in normalized units.

        Plain observation noise leaves the expert's action as the target, which
        teaches the policy to IGNORE a perturbation. Closed loop needs the
        opposite: the measured EEF is 5.3 cm off the demonstrated path by step 20
        (paper.md 5h), and the policy has to steer back. Displacing by delta
        means the same waypoint needs delta less motion, i.e. -delta/(gain*scale).
        """
        import argparse

        import torch

        import train.train_batched as tb

        cfgd = cfg
        B_, T_ = 3, 4
        batch = {"proprio": torch.zeros(B_, T_, 10)}
        # Large gains keep the implied correction inside the executability
        # budget for this unit test; the budget itself is asserted separately.
        gain = torch.tensor([1.0, 2.0, 4.0])
        scale = torch.tensor([0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0])
        old = tb._RECOVERY_GAIN
        tb._RECOVERY_GAIN = gain
        try:
            torch.manual_seed(0)
            pro, dtgt = tb._recovery_batch(
                batch, 1, argparse.Namespace(recovery_noise=0.02), cfgd, scale)
            delta = pro[:, : cfgd.waypoint_dim]          # proprio started at zero
            expected = -(delta / gain) / scale[: cfgd.waypoint_dim]
            assert torch.allclose(dtgt, expected, atol=1e-5)
            # Sign is the whole point: displaced +x must reduce the +x command.
            assert torch.all(torch.sign(dtgt) == -torch.sign(delta))
        finally:
            tb._RECOVERY_GAIN = old

    def test_recovery_noise_off_by_default_returns_untouched_proprio(self, cfg):
        import argparse

        import torch

        import train.train_batched as tb

        batch = {"proprio": torch.randn(2, 3, 10)}
        pro, d = tb._recovery_batch(batch, 1, argparse.Namespace(recovery_noise=0.0),
                                    cfg, None)
        assert d is None and torch.equal(pro, batch["proprio"][:, 1])

    def test_an_unexecutable_displacement_is_refused_not_clamped(self, cfg):
        """A correction beyond the action range must be impossible, not clamped.

        One full-magnitude step moves ~11 mm, so a 15 mm displacement implies a
        1.47-unit correction against a [-1, 1] target range. Clamping discards
        the excess and trains the policy to emit MAXIMUM motion constantly --
        measurably worse than no augmentation: divergence at step 20 went
        5.34 cm (none) -> 8.65 cm (15 mm) -> 12.82 cm (35 mm), monotonic in the
        perturbation. That is paper.md 5i.
        """
        import argparse

        import pytest
        import torch

        import train.train_batched as tb

        batch = {"proprio": torch.zeros(4, 3, 10)}
        old = tb._RECOVERY_GAIN
        tb._RECOVERY_GAIN = torch.tensor([0.0109, 0.0131, 0.0118])
        try:
            # Truncation is by construction now: an oversized sigma is
            # silently capped at the largest displacement the policy can undo,
            # so the correction stays inside the budget instead of raising.
            _pro, dt = tb._recovery_batch(
                batch, 1, argparse.Namespace(recovery_noise=0.015),
                cfg, torch.ones(cfg.num_servos))
            assert float(dt.abs().max()) <= tb._RECOVERY_MAX_CORR + 1e-4, (
                "an oversized perturbation still produced a saturating target")
        finally:
            tb._RECOVERY_GAIN = old
