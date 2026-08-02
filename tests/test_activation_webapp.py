"""Smoke test for paper/activation_webapp/generate_trace.py.

CPU-only, mock perception, short rollout — asserts the JSON schema the
webapp depends on. Skips the full 900-tick path (that is a manual/CLI job).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "activation_webapp" / "generate_trace.py"


@pytest.mark.skipif(not GEN.exists(), reason="activation_webapp not present")
def test_generate_trace_schema(tmp_path):
    out = tmp_path / "trace.json"
    cmd = [
        sys.executable,
        str(GEN),
        "--ticks", "30",
        "--out", str(out),
    ]
    # Prefer real checkpoint when present; generator falls back to mock.
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert out.exists()
    data = json.loads(out.read_text())
    assert "meta" in data and "modules" in data and "ticks" in data
    meta = data["meta"]
    assert meta["ticks"] == 30
    assert meta["weights"] in ("rec_fix", "mock")
    assert isinstance(meta.get("flow"), list) and "planner" in meta["flow"]
    assert len(data["modules"]) >= 20
    assert len(data["ticks"]) == 30
    row = data["ticks"][0]
    for key in ("t", "is_real", "trust", "plan", "acts", "group_e", "scene"):
        assert key in row
    assert row["is_real"] is True
    assert len(row["plan"]) == 5 and len(row["plan"][0]) == 7
    assert "corrector" in row["acts"]
    assert "eef" in row["scene"]
    # At least one real tick after 0 should carry sensitivity
    sens_rows = [r for r in data["ticks"] if r.get("sens")]
    assert sens_rows, "expected planner sensitivity on a real tick"
