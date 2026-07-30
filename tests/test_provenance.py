"""A corpus/deployment mismatch must be reported, not silently scored.

Four of the 26 defects in paper.md are the same sentence with a different noun:
the deployment used a different camera / detector threshold / render size /
perception period than the corpus was baked with. None of them raises anything;
they surface as `mean_success 0.000`, which is indistinguishable from a policy
that simply does not work.

CPU-only, no sim, no network.
"""
from __future__ import annotations

import json

from microvla.utils import provenance as prov

BAKED = {
    "camera": "agentview_rgb",
    "eval_camera": "agentview_image",
    "det_conf": 0.02,
    "real_frame_hz": 10.0,
    "source_hz": 20.0,
    "detect_frame_hw": [128, 128],
    "deflip": True,
}

MATCHED = dict(camera="agentview_image", det_conf=0.02,
               render_size=128, perception_period=2)


def test_a_matched_deployment_reports_nothing():
    assert prov.mismatches(BAKED, **MATCHED) == []


def test_absent_provenance_is_unknown_not_all_clear():
    """Corpora baked before 2026-07-30 have no provenance block.

    Returning [] there is correct — nothing is knowable — but the caller must
    not be able to confuse it with a verified match, which is why describe()
    says so out loud.
    """
    assert prov.mismatches({}, camera="anything", det_conf=0.9) == []
    assert "absent" in prov.describe({})


def test_wrong_camera_is_caught():
    out = prov.mismatches(BAKED, **{**MATCHED, "camera": "robot0_eye_in_hand_image"})
    assert len(out) == 1 and "camera" in out[0]
    # Both values must appear; "mismatch" without the numbers is not actionable.
    assert "robot0_eye_in_hand_image" in out[0] and "agentview_image" in out[0]


def test_wrong_threshold_is_caught():
    out = prov.mismatches(BAKED, **{**MATCHED, "det_conf": 0.10})
    assert len(out) == 1 and "det_conf" in out[0]


def test_wrong_render_size_is_caught():
    out = prov.mismatches(BAKED, **{**MATCHED, "render_size": 256})
    assert len(out) == 1 and "256" in out[0] and "128" in out[0]


def test_perception_period_is_checked_against_the_corpus_stride():
    """The Pi loop's 30/2 = 15 default is wrong for a 10 Hz corpus (paper.md 5d)."""
    out = prov.mismatches(BAKED, **{**MATCHED, "perception_period": 15})
    assert len(out) == 1 and "stride 2" in out[0]


def test_several_mismatches_are_all_reported():
    out = prov.mismatches(BAKED, camera="robot0_eye_in_hand_image", det_conf=0.10,
                          render_size=256, perception_period=15)
    assert len(out) == 4


def test_unknown_deployment_knobs_are_skipped_not_guessed():
    assert prov.mismatches(BAKED, camera=None, det_conf=None,
                           render_size=None, perception_period=None) == []


def test_load_accepts_a_corpus_dir_a_manifest_or_a_sibling_norm_stats(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"provenance": BAKED}))
    (tmp_path / "norm_stats.json").write_text("{}")
    for arg in (tmp_path, tmp_path / "manifest.json", tmp_path / "norm_stats.json"):
        assert prov.load(arg)["camera"] == "agentview_rgb"


def test_load_of_a_missing_or_broken_corpus_is_empty_not_an_exception(tmp_path):
    assert prov.load(tmp_path) == {}
    (tmp_path / "manifest.json").write_text("{not json")
    assert prov.load(tmp_path) == {}


def test_describe_names_the_knobs_a_deployment_has_to_match():
    d = prov.describe(BAKED)
    for k in ("eval_camera", "det_conf", "real_frame_hz", "detect_frame_hw"):
        assert k in d
