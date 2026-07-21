"""Eval-stack tests: policy wrapper, baselines, closed-loop harness, sweep,
scorecard.

CPU-only, mocks only, no network, no cv2 — matches the rest of the suite.
``eval.policy.MicroVLAPolicy`` defaults to REAL YOLO-World perception (it is
deploy-ready out of the box); every test here explicitly injects
``perception=MockYoloWorldPerception()`` / ``task_encoder=MockTaskEncoder()``
(the same injection the ``eval/*`` CLIs use for their own ``--mock-env``
paths) so nothing here ever touches ``ultralytics``/``torchvision``/cv2 or
downloads ``yolov8s-worldv2.pt``.

``eval.sweep.run_sweep`` does not (yet) expose that injection point on its
own ``policy_factory`` — every condition it builds goes through
``MicroVLAPolicy``'s real-perception default. The one test here that
exercises it (:class:`TestRunSweepSmoke`) monkeypatches
``eval.policy._build_real_perception`` for the duration of the test so the
grid still runs CPU-only / offline without editing ``eval/sweep.py`` (out of
this file's scope) — see the docstring note there and the run summary this
task returns.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from microvla.aux_state.drift_encoder import AnchoredDriftEncoder
from microvla.config import DEFAULT_CONFIG, MicroVLAConfig
from microvla.fusion.slot_fusion import SlotResonanceFusion
from microvla.perception.text_encoder import MockTaskEncoder
from microvla.perception.yolo_world import MockYoloWorldPerception
from microvla.planner.chrono_planner import ChronoQueryPlanner
from preprocess.common import ActionNormalizer
from train.dataset import make_synthetic_episode, save_episode
from TRM import RecursiveTRM

from eval.baselines import LinearExtrapolationTRM, PersistenceTRM
from eval.libero_eval import run_eval
from eval.policy import MicroVLAPolicy
from eval.sweep import run_sweep
from eval.scorecard import main as scorecard_main

CFG = DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _frame(seed: int, h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _write_norm_stats(path: Path, num_servos: int = CFG.num_servos) -> Path:
    rng = np.random.default_rng(0)
    actions = rng.normal(size=(256, num_servos)).astype(np.float32)
    ActionNormalizer.fit([actions]).save(path)
    return path


def _mock_policy_kwargs() -> dict:
    """Injection kwargs that keep MicroVLAPolicy CPU-only / offline."""
    return {"perception": MockYoloWorldPerception(), "task_encoder": MockTaskEncoder()}


def _make_fresh_checkpoint(path: Path, cfg: MicroVLAConfig, trm_d: int = 32) -> Path:
    """A real SHARED CONTRACT checkpoint dict, from freshly init'd modules."""
    fusion = SlotResonanceFusion(cfg)
    drift = AnchoredDriftEncoder(cfg)
    trm = RecursiveTRM(cfg, d=trm_d, T=1, n_inner=1)  # tiny recursion depth: fast, same shapes
    planner = ChronoQueryPlanner(cfg)
    torch.save(
        {
            "cfg": dataclasses.asdict(cfg),
            "trm_d": trm_d,
            "fusion": fusion.state_dict(),
            "drift": drift.state_dict(),
            "trm": trm.state_dict(),
            "planner": planner.state_dict(),
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# MicroVLAPolicy
# ---------------------------------------------------------------------------


class TestMicroVLAPolicySmoke:
    def _policy(self, tmp_path: Path, perception_period: int = 15) -> MicroVLAPolicy:
        norm_stats = _write_norm_stats(tmp_path / "norm_stats.json")
        return MicroVLAPolicy(
            checkpoint=None,
            norm_stats=str(norm_stats),
            perception_period=perception_period,
            **_mock_policy_kwargs(),
        )

    def test_act_returns_finite_raw_action(self, tmp_path):
        policy = self._policy(tmp_path)
        policy.reset("move can to ball")
        action = policy.act(_frame(0))
        assert isinstance(action, np.ndarray)
        assert action.shape == (CFG.num_servos,)
        assert np.all(np.isfinite(action))

    def test_multiple_acts_stay_finite(self, tmp_path):
        policy = self._policy(tmp_path, perception_period=15)
        policy.reset("move can to ball")
        for i in range(20):
            action = policy.act(_frame(i))
            assert action.shape == (CFG.num_servos,)
            assert np.all(np.isfinite(action))

    def test_perception_period_schedule_honored(self, tmp_path):
        period = 5
        policy = self._policy(tmp_path, perception_period=period)
        policy.reset("move can to ball")
        n_calls = 23
        for i in range(n_calls):
            policy.act(_frame(i))

        assert len(policy.telemetry) == n_calls
        for entry in policy.telemetry:
            assert {"tick_index", "is_real", "trust", "plan_norm"} <= entry.keys()

        real_count = sum(1 for entry in policy.telemetry if entry["is_real"])
        expected_real = len([i for i in range(n_calls) if i % period == 0])
        assert real_count == expected_real
        # First call of an episode must be real: nothing to dream from yet.
        assert policy.telemetry[0]["is_real"] is True

    def test_perception_period_one_is_all_real(self, tmp_path):
        policy = self._policy(tmp_path, perception_period=1)
        policy.reset("move can to ball")
        for i in range(10):
            policy.act(_frame(i))
        assert all(entry["is_real"] for entry in policy.telemetry[-10:])

    def test_reset_clears_state_between_episodes(self, tmp_path):
        period = 4
        policy = self._policy(tmp_path, perception_period=period)
        policy.reset("move can to ball")
        for i in range(6):  # ends mid-cycle: last call (index 5) is a dream tick
            policy.act(_frame(i))
        assert policy.telemetry[-1]["is_real"] is False

        policy.reset("move cup to plate")
        assert policy.telemetry == []
        action = policy.act(_frame(99))
        assert np.all(np.isfinite(action))
        # A fresh episode restarts the real-tick schedule at its own call 0.
        assert policy.telemetry[-1]["is_real"] is True


# ---------------------------------------------------------------------------
# Baseline TRMs (eval/baselines.py, never microvla/trm/)
# ---------------------------------------------------------------------------


class TestBaselineTRMs:
    def _inputs(self, batch: int = 2):
        fused = torch.randn(batch, CFG.fused_rows, CFG.fused_cols)
        state_delta = torch.randn(batch, CFG.state_dim)
        current_emb = torch.randn(batch, CFG.vis_dim)
        return fused, state_delta, current_emb

    def test_persistence_returns_current_emb_exactly(self):
        trm = PersistenceTRM(CFG)
        fused, state_delta, current_emb = self._inputs()
        out = trm(fused, state_delta, current_emb)
        assert torch.equal(out, current_emb)

    def test_persistence_ignores_context(self):
        trm = PersistenceTRM(CFG)
        fused, state_delta, current_emb = self._inputs(batch=1)
        context = torch.randn(1, CFG.context_window, CFG.vis_dim)
        out = trm(fused, state_delta, current_emb, context=context)
        assert torch.equal(out, current_emb)

    def test_linear_extrapolation_falls_back_without_context(self):
        trm = LinearExtrapolationTRM(CFG)
        fused, state_delta, current_emb = self._inputs(batch=1)
        out_none = trm(fused, state_delta, current_emb, context=None)
        assert torch.equal(out_none, current_emb)

        empty_context = torch.zeros(1, 0, CFG.vis_dim)
        out_empty = trm(fused, state_delta, current_emb, context=empty_context)
        assert torch.equal(out_empty, current_emb)

    def test_linear_extrapolation_extrapolates_with_context(self):
        trm = LinearExtrapolationTRM(CFG)
        fused, state_delta, current_emb = self._inputs(batch=1)
        previous = torch.randn(1, CFG.vis_dim)
        context = previous.unsqueeze(1)  # [1, 1, 512]: one prior tick's latent
        out = trm(fused, state_delta, current_emb, context=context)
        expected = current_emb + (current_emb - previous)
        assert torch.allclose(out, expected)
        assert not torch.equal(out, current_emb)


# ---------------------------------------------------------------------------
# run_eval (mock LIBERO harness)
# ---------------------------------------------------------------------------


class TestRunEvalMock:
    def _factory(self, tmp_path: Path):
        norm_stats = _write_norm_stats(tmp_path / "norm_stats.json")

        def factory():
            return MicroVLAPolicy(
                checkpoint=None,
                norm_stats=str(norm_stats),
                perception_period=15,
                **_mock_policy_kwargs(),
            )

        return factory

    def test_contract_shape(self, tmp_path):
        factory = self._factory(tmp_path)
        result = run_eval(
            factory,
            suite="libero_object",
            n_trials=2,
            max_steps=8,
            mock_env=True,
            seed=0,
            out_dir=str(tmp_path / "eval_results"),
        )
        assert set(result.keys()) >= {"suite", "per_task", "mean_success", "n_trials", "telemetry_path"}
        assert result["suite"] == "libero_object"
        assert result["n_trials"] == 2
        assert isinstance(result["per_task"], dict) and len(result["per_task"]) > 0
        assert all(0.0 <= v <= 1.0 for v in result["per_task"].values())
        assert 0.0 <= result["mean_success"] <= 1.0
        assert Path(result["telemetry_path"]).exists()

    def test_deterministic_under_fixed_seed(self, tmp_path):
        factory = self._factory(tmp_path)
        kwargs = dict(suite="libero_object", n_trials=2, max_steps=8, mock_env=True, seed=0)
        r1 = run_eval(factory, out_dir=str(tmp_path / "run1"), **kwargs)
        r2 = run_eval(factory, out_dir=str(tmp_path / "run2"), **kwargs)
        assert r1["mean_success"] == r2["mean_success"]
        assert r1["per_task"] == r2["per_task"]


# ---------------------------------------------------------------------------
# run_sweep (perception-rate x baseline smoke grid)
# ---------------------------------------------------------------------------


class TestRunSweepSmoke:
    def test_reduced_grid_smoke(self, tmp_path, monkeypatch):
        # eval.sweep's own policy_factory never injects a mock
        # perception/task_encoder (unlike eval.libero_eval's --mock-env CLI
        # path), so left alone it would hit real YOLO-World / a network
        # download even under mock_env=True. Patch the same seam
        # MicroVLAPolicy itself uses so this smoke test stays CPU-only and
        # offline without touching eval/sweep.py.
        monkeypatch.setattr(
            "eval.policy._build_real_perception",
            lambda device: (MockYoloWorldPerception(), MockTaskEncoder()),
        )
        norm_stats = _write_norm_stats(tmp_path / "norm_stats.json")

        result = run_sweep(
            perception_periods=[1, 15],
            conditions=["ours", "persistence"],
            checkpoint=None,
            norm_stats=str(norm_stats),
            suite="libero_object",
            n_trials=1,
            max_steps=8,
            mock_env=True,
            seed=0,
            device="cpu",
            out_path=str(tmp_path / "eval_results" / "sweep.json"),
        )

        assert set(result.keys()) >= {"rows", "auroc", "meta"}
        assert len(result["rows"]) == 4  # 2 perception_periods x 2 conditions
        seen = {(row["condition"], row["perception_period"]) for row in result["rows"]}
        assert seen == {("ours", 1), ("ours", 15), ("persistence", 1), ("persistence", 15)}
        for row in result["rows"]:
            assert 0.0 <= row["mean_success"] <= 1.0

        assert set(result["auroc"]) == {"ours", "persistence"}
        for auc in result["auroc"].values():
            assert math.isnan(auc) or 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# scorecard (synthetic data-dir smoke, via its CLI entry point)
# ---------------------------------------------------------------------------


class TestScorecardSmoke:
    def _synthetic_data_dir(self, tmp_path: Path, n_episodes: int = 6) -> Path:
        data_dir = tmp_path / "synthetic_data"
        data_dir.mkdir()
        for i in range(n_episodes):
            episode = make_synthetic_episode(T=20, cfg=CFG, seed=i)
            save_episode(data_dir / f"ep{i}.npz", episode)
        return data_dir

    def test_scorecard_on_synthetic_data_dir(self, tmp_path):
        # scorecard.py's --checkpoint is required (no checkpoint=None smoke
        # path), so this builds a real SHARED CONTRACT checkpoint from fresh
        # modules per the assignment's documented fallback.
        data_dir = self._synthetic_data_dir(tmp_path)
        ckpt_path = _make_fresh_checkpoint(tmp_path / "scorecard_ckpt.pt", CFG)
        out_path = tmp_path / "eval_results" / "scorecard.json"

        result = scorecard_main([
            "--checkpoint", str(ckpt_path),
            "--data-dir", str(data_dir),
            "--val-frac", "0.5",
            "--max-val-episodes", "4",
            "--device", "cpu",
            "--out", str(out_path),
        ])

        assert isinstance(result, dict)
        assert result["checkpoint"] == str(ckpt_path)
        assert set(result["rollout"]) == {"1", "5", "10", "15"}
        for row in result["rollout"].values():
            assert row["n_samples"] >= 0
        assert result["innovation_norm"]["k"] == 15
        assert result["planner"] is not None
        assert result["planner"]["n_episodes"] >= 0
        assert out_path.exists()
