"""Tests for the two annotation-free grounding probes.

CPU-only, mock-only, no network, no cv2 — per the repo contract. The mock
perception is deterministic (hash-seeded), which is what lets the
identity-blindness case be asserted exactly rather than statistically.
"""

import numpy as np
import pytest

from eval.probes import instruction_swap, prompt_agreement


class _FixedPerception:
    """Perception whose selected box depends on the chain, or does not.

    ``discriminating=False`` reproduces the failure the probe exists to
    catch: every chain collapses onto the same generic term, so the same box
    comes back whatever was asked for.
    """

    def __init__(self, discriminating: bool):
        self.discriminating = discriminating
        self._chain = None

    def set_role_prompts(self, source, target):
        self._chain = list(source)

    def perceive(self, frame):
        base = float(np.mean(frame)) / 255.0
        shift = 0.3 if (self.discriminating and self._chain[0].startswith("b")) else 0.0
        obj = type("O", (), {})()
        obj.center = (0.4 + shift, 0.5 + base * 0.01)
        obj.confidence = 0.5
        out = type("P", (), {})()
        out.source = obj
        return out


def _frames(n=12):
    return [np.full((8, 8, 3), 10 + 7 * i, dtype=np.uint8) for i in range(n)]


def test_prompt_agreement_detects_identity_blindness():
    r = prompt_agreement(_FixedPerception(discriminating=False),
                         ["alphabet soup", "box"], ["butter", "box"], _frames())
    assert r.n_compared == 12
    assert r.same_rate == 1.0
    assert r.median_distance == pytest.approx(0.0, abs=1e-9)
    assert "IDENTITY-BLIND" in r.verdict()


def test_prompt_agreement_detects_discrimination():
    r = prompt_agreement(_FixedPerception(discriminating=True),
                         ["alphabet soup", "box"], ["butter", "box"], _frames())
    assert r.same_rate == 0.0
    assert r.median_distance == pytest.approx(0.3, abs=1e-6)
    assert "DISCRIMINATING" in r.verdict()


def test_prompt_agreement_handles_no_frames():
    r = prompt_agreement(_FixedPerception(False), ["a"], ["b"], [])
    assert r.n_compared == 0
    assert "INCONCLUSIVE" in r.verdict()


def test_prompt_agreement_excludes_frames_missing_either_detection():
    class Missing(_FixedPerception):
        def perceive(self, frame):
            p = super().perceive(frame)
            if int(np.mean(frame)) % 2 == 0 and self._chain[0] == "butter":
                p.source.confidence = 0.0
            return p

    r = prompt_agreement(Missing(False), ["soup"], ["butter"], _frames())
    # frames where the second chain missed cannot agree or disagree
    assert r.n_compared < 12
    assert r.detect_rate_b < 1.0


def test_instruction_swap_total_collapse_is_significant():
    base = [True] * 7 + [False] * 3
    swap = [False] * 10
    r = instruction_swap(base, swap)
    assert r.n_discordant == 7 and r.favour_baseline == 7
    assert r.exact_p == pytest.approx(0.015625, abs=1e-6)
    assert "INSTRUCTION-SENSITIVE" in r.verdict()


def test_instruction_swap_no_effect():
    base = [True, False, True, True, False, False, True, False, True, False]
    r = instruction_swap(base, list(base))
    assert r.n_discordant == 0
    assert r.exact_p == 1.0
    assert "NO EFFECT" in r.verdict()


def test_instruction_swap_flags_underpowered_cell():
    """A weak baseline cannot establish a collapse; the verdict must say so."""
    base = [True] * 4 + [False] * 6
    swap = [False] * 10
    r = instruction_swap(base, swap)
    assert r.exact_p == pytest.approx(0.125, abs=1e-6)
    v = r.verdict()
    assert "NO SIGNIFICANT EFFECT" in v
    assert "ABSENCE" in v          # must warn it cannot show presence


def test_instruction_swap_rejects_unpaired_cells():
    with pytest.raises(ValueError, match="paired"):
        instruction_swap([True, False], [True, False, True])
