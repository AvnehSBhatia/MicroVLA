"""Tests for role-ordered spatial grounding prompts (Feature 1).

The LIBERO ``libero_spatial`` suite disambiguates *which* black bowl by a
spatial clause ("between the plate and the ramekin"). The command parser
correctly extracts the full source phrase, but detection used to ground the
bare noun ("black bowl") and keep the best box by raw confidence — blind to the
clause, so among several identical bowls it grounded an arbitrary one. The fix
feeds per-role detection prompts in preference order (full phrase, then bare
noun) so the frozen region-text head can pick the box matching the clause.

These tests exercise the prompt construction + mock wiring (CPU/mock-only, no
model); the region-text preference itself is validated in the real detector.
"""

from __future__ import annotations

from microvla.config import DEFAULT_CONFIG
from microvla.jepa.loop import JEPALoop
from microvla.perception.prompts import role_chains
from microvla.perception.yolo_world import MockYoloWorldPerception


class TestRolePrompts:
    """Deployment prompts come from the shared chain builder, not a local one.

    This class used to pin ``JEPALoop._role_prompts``, which built
    ``[full phrase, bare noun]`` — and those assertions passing is exactly what
    hid the bug, because the BAKE path had meanwhile moved to concrete-category
    chains and nothing compared the two. See ``tests/test_prompt_fallbacks.py``
    for the parity tests and paper.md 4t for the measurement.

    Note what changed for libero_spatial: the clause-bearing phrase is no longer
    a detection prompt, because ``strip_article`` reduces it to the noun phrase
    on BOTH sides. That is a deliberate loss. The clause-first prompt was
    motivated by disambiguation, but the measured grounding says it buys nothing
    on the suites actually run (the full LIBERO phrases score 0.000) while chain
    length actively suppresses confidence (8 prompts -> 0.253 vs 2 -> 0.505), so
    re-adding it would cost recall on every task to help none. Which black bowl
    gets grounded on libero_spatial is therefore still open.
    """

    def test_spatial_clause_reduces_to_the_detectable_noun_phrase(self):
        src, _ = role_chains("the black bowl between the plate and the ramekin",
                             "the plate")
        assert src[0] == "black bowl"

    def test_chain_carries_concrete_category_fallbacks(self):
        src, _ = role_chains("the alphabet soup", "the basket")
        assert src[0] == "alphabet soup"
        assert len(src) > 1, "no fallback tail — this is the blind-corpus bug"
        assert "can" in src or "box" in src or "bottle" in src

    def test_article_is_stripped_for_both_roles(self):
        src, tgt = role_chains("the can", "the basket")
        assert src[0] == "can"
        assert tgt[0] == "basket"

    def test_one_object_for_both_roles_yields_no_target_chain(self):
        src, tgt = role_chains("bowl", "bowl")
        assert tgt is None and src[0] == "bowl"


class TestMockRolePrompts:
    """The mock perception records active roles from the primary prompts."""

    def test_two_roles(self):
        p = MockYoloWorldPerception(vis_dim=DEFAULT_CONFIG.vis_dim)
        p.set_role_prompts(["black bowl between the plate", "black bowl"], ["plate"])
        assert p.active_classes == ["black bowl between the plate", "plate"]

    def test_single_role(self):
        p = MockYoloWorldPerception(vis_dim=DEFAULT_CONFIG.vis_dim)
        p.set_role_prompts(["the red block", "red block"], None)
        assert p.active_classes == ["the red block"]


class TestLoopSetTaskGrounding:
    """The JEPA loop wires role prompts through set_task for real relational text."""

    def test_relational_instruction_configures_two_roles(self):
        loop = JEPALoop.build_mock(DEFAULT_CONFIG)
        loop.set_task(
            "pick up the black bowl between the plate and the ramekin "
            "and place it on the plate"
        )
        # The mock records the primary (first) prompt per role, and the primary
        # is now the detectable noun phrase rather than the clause-bearing one —
        # same reduction the bake applies, which is the whole point.
        assert loop.perception.active_classes[0] == "black bowl"
        assert loop.perception.active_classes[-1] == "plate"

    def test_no_destination_single_role(self):
        loop = JEPALoop.build_mock(DEFAULT_CONFIG)
        loop.set_task("pick up the black bowl")
        # source == target -> one active role.
        assert len(loop.perception.active_classes) == 1


class TestProprio:
    """v6: proprio vector layout, dataset zero-fill, planner + loop plumbing."""

    def test_build_proprio_layout_and_validity(self):
        import numpy as np

        from microvla.utils.proprio import GRIPPER_SCALE, PROPRIO_DIM, build_proprio

        v = build_proprio([0.1, 0.2, 0.3], [0, 0, 0, 1], [0.02, 0.04])
        assert v.shape == (PROPRIO_DIM,)
        assert np.allclose(v[:3], [0.1, 0.2, 0.3])
        assert v[6] == 1.0                       # quat w
        assert np.allclose(v[7:9], [0.02 * GRIPPER_SCALE, 0.04 * GRIPPER_SCALE])
        assert v[9] == 1.0                       # valid
        # axis-angle (3-dim) orientation zero-pads the 4th slot
        v3 = build_proprio([0, 0, 0], [0.1, 0.2, 0.3], None)
        assert v3[6] == 0.0 and v3[9] == 1.0

    def test_proprio_from_obs_key_spellings_and_missing(self):
        import numpy as np

        from microvla.utils.proprio import proprio_from_obs

        libero_style = {"ee_pos": np.ones(3), "ee_ori": np.zeros(3),
                        "gripper_states": np.zeros(2)}
        robosuite_style = {"robot0_eef_pos": np.ones(3),
                           "robot0_eef_quat": np.array([0, 0, 0, 1.0])}
        assert proprio_from_obs(libero_style) is not None
        assert proprio_from_obs(robosuite_style) is not None
        assert proprio_from_obs({"agentview_image": np.zeros((4, 4, 3))}) is None

    def test_dataset_zero_fills_missing_optional_keys(self, tmp_path):
        import numpy as np

        from microvla.config import DEFAULT_CONFIG as cfg
        from train.dataset import EpisodeDataset, make_synthetic_episode

        ep = make_synthetic_episode(6, cfg, seed=0)
        # Simulate a PRE-v6 file: drop the optional keys before saving.
        legacy = {k: v for k, v in ep.items() if k not in ("proprio", "eef_pos_chunk")}
        np.savez_compressed(tmp_path / "legacy.npz", **legacy)
        item = EpisodeDataset(tmp_path)[0]
        assert item["proprio"].shape == (6, 10)
        assert float(item["proprio"].abs().sum()) == 0.0    # zeros => valid flag 0
        assert item["eef_pos_chunk"].shape == (6, cfg.plan_steps, 3)

    def test_loop_tick_accepts_and_holds_proprio(self):
        import numpy as np
        import torch

        from microvla.config import DEFAULT_CONFIG as cfg
        from microvla.jepa.loop import JEPALoop

        loop = JEPALoop.build_mock(cfg)
        loop.set_task("move can to ball")
        frame = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        p = np.arange(10, dtype=np.float32) / 10.0
        loop.tick(frame, proprio=p)
        assert torch.allclose(loop._last_proprio, torch.as_tensor(p).reshape(1, -1))
        loop.tick(None)  # dream tick without proprio -> held, not cleared
        assert loop._last_proprio is not None
        loop.set_task("move can to ball")
        assert loop._last_proprio is None       # reset per episode


class TestWorldModelMsg:
    """v7.1: the TRM's 32-d latent message channel reaches the planner."""

    def test_mock_forward_full_contract(self):
        import torch

        from microvla.config import DEFAULT_CONFIG as cfg
        from microvla.trm.mock_trm import MockTRM

        trm = MockTRM(cfg)
        out = trm.forward_full(torch.randn(3, cfg.fused_rows, cfg.fused_cols),
                               torch.randn(3, cfg.state_dim),
                               torch.randn(3, cfg.vis_dim))
        assert set(out) == {"next_emb", "next_box", "msg", "latent"}
        assert out["next_emb"].shape == (3, cfg.vis_dim)
        assert out["next_box"].shape == (3, cfg.vis_dim)
        assert out["msg"].shape == (3, 32)
        # v7.3: the pooled belief state the planner's wm_latent group reads.
        # The mock has none, so it emits zeros at the configured width.
        assert out["latent"].shape == (3, cfg.wm_latent_dim)
        assert float(out["latent"].abs().sum()) == 0.0

    def test_planner_consumes_wm_msg(self):
        import torch

        from microvla.config import DEFAULT_CONFIG as cfg
        from microvla.planner.chrono_planner import ChronoQueryPlanner

        pl = ChronoQueryPlanner(cfg)
        with torch.no_grad():
            base = pl(torch.randn(2, cfg.vis_dim))
            with_msg = pl(torch.randn(2, cfg.vis_dim), wm_msg=torch.randn(2, 32))
        assert base.shape == with_msg.shape == (2, cfg.plan_steps, cfg.num_servos)

    def test_real_trm_msg_head_grad_flows_when_core_frozen(self):
        import torch

        from microvla.config import DEFAULT_CONFIG as cfg
        from TRM import RecursiveTRM

        trm = RecursiveTRM(cfg)
        # Stage-B freeze policy: core frozen, msg_head trainable.
        for name, p in trm.named_parameters():
            p.requires_grad_(name.startswith("msg_head"))
        out = trm.forward_full(torch.randn(2, cfg.fused_rows, cfg.fused_cols),
                               torch.randn(2, cfg.state_dim),
                               torch.randn(2, cfg.vis_dim))
        out["msg"].sum().backward()
        assert trm.msg_head.weight.grad is not None
        assert float(trm.msg_head.weight.grad.norm()) > 0
        # Core stayed frozen: no grads anywhere else.
        assert trm.head.weight.grad is None
        # And the world-model outputs are unaffected by msg_head existing:
        assert out["next_emb"].shape == (2, cfg.vis_dim)


class TestWorldModelLatent:
    """v7.3: the planner reads the TRM's pooled belief state, not its 32-d readout.

    `msg` measured 92% a FIXED vector (constant norm 3.32 vs varying 0.268,
    effective rank 6/32) on the +19.8% stage-A checkpoint, while the state it is
    read from is vision-rich — zeroing `fused` destroys 89% of the TRM's
    predicted residual. See paper.md 4h.
    """

    def test_real_trm_exports_the_pooled_state_it_reads_msg_from(self):
        import torch

        from microvla.config import DEFAULT_CONFIG as cfg
        from TRM import RecursiveTRM

        trm = RecursiveTRM(cfg, d=cfg.wm_latent_dim, T=1, n_inner=1).eval()
        with torch.no_grad():
            out = trm.forward_full(torch.randn(2, cfg.fused_rows, cfg.fused_cols),
                                   torch.randn(2, cfg.state_dim),
                                   torch.randn(2, cfg.vis_dim))
        assert out["latent"].shape == (2, cfg.wm_latent_dim)
        # msg is a linear readout OF latent, so it must be reproducible from it.
        with torch.no_grad():
            assert torch.allclose(trm.msg_head(out["latent"]), out["msg"], atol=1e-5)

    def test_planner_consumes_it_as_multiple_tokens(self):
        import torch

        from microvla.config import DEFAULT_CONFIG as cfg
        from microvla.planner.chrono_planner import ChronoQueryPlanner

        planner = ChronoQueryPlanner(cfg).eval()
        assert planner.wm_latent_proj is not None
        # 8 tokens, not 1 — it has to compete with fused's 32 for attention.
        assert planner.wm_latent_chunk == cfg.wm_latent_dim // planner.n_mem_tokens
        next_emb = torch.randn(2, cfg.vis_dim)
        with torch.no_grad():
            a = planner(next_emb)
            b = planner(next_emb, wm_latent=torch.randn(2, cfg.wm_latent_dim))
        assert not torch.allclose(a, b), "wm_latent had no effect on the plan"

    def test_costs_far_less_than_a_full_projection(self):
        import dataclasses

        from microvla.config import DEFAULT_CONFIG as cfg
        from microvla.planner.chrono_planner import ChronoQueryPlanner

        without = dataclasses.replace(
            cfg, planner_inputs=tuple(n for n in cfg.planner_inputs if n != "wm_latent"))
        cost = (sum(p.numel() for p in ChronoQueryPlanner(cfg).parameters())
                - sum(p.numel() for p in ChronoQueryPlanner(without).parameters()))
        chunked = (cfg.wm_latent_dim // 8 + 1) * cfg.d_plan
        assert cost == chunked
        # A single Linear(wm_latent_dim, d_plan) would be ~8x this.
        assert cost < (cfg.wm_latent_dim + 1) * cfg.d_plan / 4

    def test_optional_so_zero_param_baselines_still_work(self):
        """eval/baselines.py foils have no belief state; that must stay legal."""
        import torch

        from microvla.config import DEFAULT_CONFIG as cfg
        from eval.baselines import PersistenceTRM
        from microvla.planner.chrono_planner import ChronoQueryPlanner

        out = PersistenceTRM(cfg).forward_full(
            torch.randn(2, cfg.fused_rows, cfg.fused_cols),
            torch.randn(2, cfg.state_dim), torch.randn(2, cfg.vis_dim))
        planner = ChronoQueryPlanner(cfg).eval()
        with torch.no_grad():   # .get() yields None; the planner ignores it
            plan = planner(out["next_emb"], wm_latent=out.get("latent"))
        assert plan.shape == (2, cfg.plan_steps, cfg.num_servos)
