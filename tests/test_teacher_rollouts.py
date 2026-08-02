"""Teacher-rollout converter smoke (UNAIDED_PLAN Phase B).

CPU-only, mock perception, no sim: fabricates raw teacher npz episodes and
runs the convert path end to end through preprocess.common.run_conversion.
The record path needs a live LIBERO env and is exercised on the pod.
"""
import json

import numpy as np
import pytest


def _fake_raw(dir_, n=2, T=24):
    rng = np.random.default_rng(0)
    for i in range(n):
        np.savez_compressed(
            dir_ / f"teacher_libero_object_t1_i{20+i:04d}.npz",
            frames=(rng.random((T, 64, 64, 3)) * 255).astype(np.uint8),
            actions=rng.standard_normal((T, 7)).astype(np.float32) * 0.3,
            proprio=rng.standard_normal((T, 10)).astype(np.float32),
            instruction=np.array("pick up the cream cheese and place it in the basket"),
            init_index=np.array(20 + i),
            camera=np.array("robot0_eye_in_hand_image"),
        )


def test_convert_dry_run_produces_training_schema(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _fake_raw(raw)
    out = tmp_path / "shards"

    from preprocess.teacher_rollouts import convert
    convert(["--raw-dir", str(raw), "--out", str(out), "--dry-run",
             "--spatial-grid", "4"])

    eps = sorted(out.glob("*.npz"))
    assert len(eps) == 2
    d = np.load(eps[0])
    for key in ("frame_embs", "source_centers", "target_centers", "box_weights",
                "pwm_targets", "proprio", "eef_pos_chunk", "text_tokens"):
        assert key in d, key
    assert d["pwm_targets"].shape[1:] == (5, 7)
    assert (out / "norm_stats.json").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    prov = manifest.get("provenance", {})
    assert prov.get("teacher") == "PhasedIBVS-handeye"


def test_convert_purge_raw_deletes_source(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _fake_raw(raw, n=1)
    out = tmp_path / "shards"

    from preprocess.teacher_rollouts import convert
    convert(["--raw-dir", str(raw), "--out", str(out), "--dry-run",
             "--purge-raw"])
    assert not raw.exists()


def test_convert_tolerates_dagger_keys(tmp_path):
    """DAgger episodes carry executed_actions/success/dagger_beta; the
    converter must still consume them (labels live in `actions`)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(1)
    T = 20
    np.savez_compressed(
        raw / "teacher_libero_object_t0_i0150.npz",
        frames=(rng.random((T, 64, 64, 3)) * 255).astype(np.uint8),
        actions=rng.standard_normal((T, 7)).astype(np.float32) * 0.3,
        proprio=rng.standard_normal((T, 10)).astype(np.float32),
        executed_actions=rng.standard_normal((T, 7)).astype(np.float32) * 0.3,
        success=np.array(False),
        dagger_beta=np.array(0.3),
        instruction=np.array("pick up the alphabet soup and place it in the basket"),
        init_index=np.array(150),
        camera=np.array("robot0_eye_in_hand_image"),
    )
    out = tmp_path / "shards"

    from preprocess.teacher_rollouts import convert
    convert(["--raw-dir", str(raw), "--out", str(out), "--dry-run",
             "--spatial-grid", "4"])
    eps = sorted(out.glob("*.npz"))
    assert len(eps) == 1
    d = np.load(eps[0])
    assert d["pwm_targets"].shape[1:] == (5, 7)


def test_record_requires_subcommand():
    from preprocess.teacher_rollouts import main
    import sys
    argv = sys.argv
    sys.argv = ["teacher_rollouts.py"]
    try:
        with pytest.raises(SystemExit):
            main()
    finally:
        sys.argv = argv
