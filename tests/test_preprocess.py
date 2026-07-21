"""Tests for the offline dataset preprocessing (mocks only, no data, no deps)."""

from __future__ import annotations

import json

import numpy as np
import torch

from microvla import DEFAULT_CONFIG
from preprocess.bridge import task_name_to_instruction
from preprocess.common import (
    ActionNormalizer,
    EpisodeBuilder,
    SourceEpisode,
    chunk_actions,
    run_conversion,
    subsample_indices,
)
from preprocess.libero import instruction_from_filename
from preprocess.teacher import CachedTeacher, MockTeacher, build_teacher
from train.dataset import EPISODE_KEYS, EpisodeDataset

CFG = DEFAULT_CONFIG


def _episode(T: int = 40, hz: float = 20.0, seed: int = 0, eid: str = "ep0") -> SourceEpisode:
    rng = np.random.default_rng(seed)
    frames = [rng.integers(0, 256, (48, 64, 3), dtype=np.uint8) for _ in range(T)]
    actions = rng.uniform(-0.8, 0.8, size=(T, CFG.num_servos)).astype(np.float32)
    return SourceEpisode(frames=frames, actions=actions,
                         instruction="move the can to the ball", source_hz=hz,
                         episode_id=eid)


class TestSubsampleAndChunk:
    def test_subsample_matches_video_sampler_cadence(self):
        # 20 Hz -> 2 Hz over 40 frames: every 10th frame, starting at 0.
        idx = subsample_indices(40, 20.0, CFG.real_frame_hz)
        assert idx == [0, 10, 20, 30]

    def test_subsample_source_slower_than_target_keeps_all(self):
        assert subsample_indices(7, 1.0, 2.0) == list(range(7))

    def test_chunk_shapes_and_end_padding(self):
        actions = np.arange(12 * 7, dtype=np.float32).reshape(12, 7)
        chunks = chunk_actions(actions, [0, 5, 10], CFG.plan_steps)
        assert chunks.shape == (3, CFG.plan_steps, 7)
        assert np.array_equal(chunks[0], actions[0:5])
        # Chunk at index 10 runs past the end: rows 10, 11, then last repeated.
        assert np.array_equal(chunks[2][0], actions[10])
        assert np.array_equal(chunks[2][2], actions[11])
        assert np.array_equal(chunks[2][4], actions[11])


class TestActionNormalizer:
    def test_range_and_roundtrip(self):
        rng = np.random.default_rng(1)
        arrays = [rng.normal(0, 2, size=(100, 7)) for _ in range(5)]
        norm = ActionNormalizer.fit(arrays)
        z = norm(arrays[0])
        assert z.min() >= -1.0 and z.max() <= 1.0
        # Roundtrip is exact for values inside the quantile window.
        inner = np.clip(arrays[0], norm.q_low + 1e-6, norm.q_high - 1e-6)
        back = norm.inverse(norm(inner))
        assert np.allclose(back, inner, atol=1e-4)

    def test_constant_dim_does_not_divide_by_zero(self):
        a = np.zeros((50, 7))
        a[:, 0] = 3.0  # constant dim
        a[:, 1:] = np.random.default_rng(2).normal(size=(50, 6))
        z = ActionNormalizer.fit([a])(a)
        assert np.isfinite(z).all()

    def test_save_load(self, tmp_path):
        norm = ActionNormalizer.fit([np.random.default_rng(3).normal(size=(60, 7))])
        norm.save(tmp_path / "stats.json")
        loaded = ActionNormalizer.load(tmp_path / "stats.json")
        x = np.random.default_rng(4).normal(size=(5, 7))
        assert np.allclose(norm(x), loaded(x))


class TestEpisodeBuilder:
    def test_mock_build_produces_valid_episode(self):
        ep = _episode()
        norm = ActionNormalizer.fit([ep.actions])
        arrays = EpisodeBuilder(CFG, mock=True).build(ep, norm)
        assert set(arrays) == set(EPISODE_KEYS)
        T = arrays["frame_embs"].shape[0]
        assert T == len(subsample_indices(40, 20.0, CFG.real_frame_hz))
        assert arrays["pwm_targets"].shape == (T, CFG.plan_steps, CFG.num_servos)
        assert arrays["text_tokens"].shape == (CFG.n_text_tokens, CFG.text_dim)
        assert np.abs(arrays["pwm_targets"]).max() <= 1.0
        # Standardized embeddings: per-vector ~zero mean.
        assert abs(float(arrays["frame_embs"].mean(axis=-1).max())) < 1e-3

    def test_action_dim_mismatch_raises(self):
        ep = _episode()
        ep = SourceEpisode(ep.frames, ep.actions[:, :5], ep.instruction, ep.source_hz, "bad")
        norm = ActionNormalizer.fit([ep.actions])
        try:
            EpisodeBuilder(CFG, mock=True).build(ep, norm)
        except ValueError as err:
            assert "num_servos" in str(err)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


class TestRunConversion:
    def test_end_to_end_and_trainable(self, tmp_path):
        out = run_conversion(
            lambda: iter(_episode(T=30, seed=s, eid=f"ep{s}") for s in range(3)),
            tmp_path / "out", mock=True,
        )
        assert (out / "norm_stats.json").exists()
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["label_source"] == "dataset"
        assert len(manifest["episodes"]) == 3
        # The output must be loadable by the training dataset as-is.
        ds = EpisodeDataset(out)
        item = ds[0]
        assert set(item) == set(EPISODE_KEYS)
        assert isinstance(item["pwm_targets"], torch.Tensor)

    def test_limit(self, tmp_path):
        out = run_conversion(
            lambda: iter(_episode(seed=s, eid=f"ep{s}") for s in range(5)),
            tmp_path / "out", mock=True, limit=2,
        )
        assert len(json.loads((out / "manifest.json").read_text())["episodes"]) == 2


class TestTeacher:
    def test_mock_teacher_relabels_full_length_deterministically(self):
        ep = _episode(T=23)
        teacher = MockTeacher()
        a1, a2 = teacher.relabel(ep), teacher.relabel(ep)
        assert a1.shape == (23, 7)
        assert np.array_equal(a1, a2)
        assert not np.array_equal(a1, ep.actions)

    def test_cached_teacher_hits_disk_once(self, tmp_path):
        ep = _episode(T=12, eid="cache_me")

        class CountingTeacher(MockTeacher):
            calls = 0

            def relabel(self, episode):
                CountingTeacher.calls += 1
                return super().relabel(episode)

        cached = CachedTeacher(CountingTeacher(), tmp_path)
        a1 = cached.relabel(ep)
        a2 = cached.relabel(ep)
        assert CountingTeacher.calls == 1
        assert np.array_equal(a1, a2)

    def test_conversion_with_teacher_marks_manifest(self, tmp_path):
        out = run_conversion(
            lambda: iter(_episode(T=25, seed=s, eid=f"ep{s}") for s in range(2)),
            tmp_path / "out", mock=True,
            teacher=CachedTeacher(MockTeacher(), tmp_path / "cache"),
        )
        assert json.loads((out / "manifest.json").read_text())["label_source"] == "CachedTeacher"

    def test_build_teacher_validation(self):
        assert build_teacher(None, None, None, None) is None
        assert isinstance(build_teacher("mock", None, None, None), MockTeacher)
        try:
            build_teacher("tinyvla", None, None, None)
        except ValueError as err:
            assert "tinyvla" in str(err)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


class TestInstructionHelpers:
    def test_libero_filename_instruction(self):
        got = instruction_from_filename("KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo")
        assert got == "put the black bowl on the plate"

    def test_bridge_task_instruction(self):
        assert task_name_to_instruction("put_carrot_in_pot") == "put carrot in pot"
