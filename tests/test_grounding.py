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
from microvla.jepa.loop import JEPALoop, _role_prompts
from microvla.perception.yolo_world import MockYoloWorldPerception


class TestRolePrompts:
    """``_role_prompts`` yields [full phrase, bare noun], deduped."""

    def test_spatial_clause_keeps_full_then_noun(self):
        prompts = _role_prompts("the black bowl between the plate and the ramekin")
        assert prompts == [
            "the black bowl between the plate and the ramekin",
            "black bowl",
        ]

    def test_article_only_still_two_distinct_prompts(self):
        # "the can" -> full "the can", noun "can": distinct, both kept.
        assert _role_prompts("the can") == ["the can", "can"]

    def test_already_bare_noun_collapses_to_one(self):
        assert _role_prompts("can") == ["can"]

    def test_whitespace_is_stripped(self):
        assert _role_prompts("  the red cup  ") == ["the red cup", "red cup"]


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
        # source role -> "black bowl ..." full phrase primary; target -> "the plate"
        # (mock records the primary/full prompt per role).
        assert loop.perception.active_classes[0].startswith("the black bowl between")
        assert loop.perception.active_classes[-1] == "the plate"

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
