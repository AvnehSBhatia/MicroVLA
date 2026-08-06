"""Two annotation-free probes for detector-grounded policies.

Both answer questions about *language grounding* without ground truth, which is
what makes them cheap enough to run on someone else's system. They were built
for MicroVLA after six ground-truth-seeking instruments failed at their own
calibration gates; these two worked on the first attempt, and they found a
memorization layer no other probe in that campaign had reached.

They fail differently, and that is the point:

``prompt_agreement``
    Detects a grounding stage that **ignores** the instruction. Open-vocabulary
    detectors are usually prompted with a fallback chain ("alphabet soup" →
    "soup" → "box" → …). When the specific terms score zero, every instruction
    collapses onto the same generic term and the detector returns the same
    region whatever you asked for --- while the policy still looks
    language-conditioned from the outside. Running two *different* objects'
    chains over the *same* frames exposes this: if they select the same box,
    the stage carries no identity. No annotation is needed because the probe
    compares two prompts against **each other**, never against a truth.

``instruction_swap``
    Detects a stage that **memorized** the instruction. Run a task while
    telling the policy to fetch a *different* object, leaving the environment
    and the success criterion bound to the real task. A policy whose success is
    unchanged is not using the instruction; a policy that collapses is using it
    somewhere, and per-stage telemetry says where. This is the only way to see
    a memorized command→X map on a benchmark where every task shares the same
    X --- there, memorized and grounded maps are behaviourally identical until
    the command changes. In this repo the swap arm is
    ``eval/libero_eval.py --override-instruction`` (with
    ``--override-prompt-only`` to drive the prompt and embedding channels
    apart); this module provides the analysis half.

Neither probe needs labels, a simulator, or a second model --- only a
perception object with ``set_role_prompts`` / ``perceive``, or paired
success vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class AgreementResult:
    """Outcome of :func:`prompt_agreement`.

    Attributes:
        n_compared: Frames where BOTH chains produced a detection. Frames where
            either missed are excluded, since "no box" cannot agree or disagree
            with a box.
        same_rate: Fraction of compared frames whose selected box centres fall
            within ``tol``. Near 1.0 means the stage is identity-blind.
        median_distance: Median centre distance over compared frames, in
            normalized frame units. Report this alongside ``same_rate``: it is
            what shows the result is not an artifact of the threshold. If the
            median is orders of magnitude below ``tol``, any nearby threshold
            gives the same answer.
        detect_rate_a: Detection rate of chain A over all frames.
        detect_rate_b: Detection rate of chain B over all frames.
    """

    n_compared: int
    same_rate: float
    median_distance: float
    detect_rate_a: float
    detect_rate_b: float

    def verdict(self, tol: float = 0.02) -> str:
        """One-line reading, deliberately hedged where the data is thin."""
        if self.n_compared < 5:
            return (f"INCONCLUSIVE: only {self.n_compared} frames had both "
                    f"detections; the probe needs co-detected frames to compare.")
        if self.same_rate >= 0.8:
            return (f"IDENTITY-BLIND: the two instructions select the same box on "
                    f"{self.same_rate:.0%} of frames (median centre distance "
                    f"{self.median_distance:.4f}, {tol/max(self.median_distance,1e-9):.0f}x "
                    f"inside the threshold). This stage carries no object identity.")
        if self.same_rate <= 0.2:
            return (f"DISCRIMINATING: the two instructions select different boxes on "
                    f"{1 - self.same_rate:.0%} of frames. This stage does distinguish "
                    f"them --- though it does not follow that it is distinguishing "
                    f"them CORRECTLY, which this probe cannot see.")
        return (f"MIXED: same box on {self.same_rate:.0%} of frames. Partial "
                f"discrimination; report the rate rather than a label.")


def prompt_agreement(perception, chain_a: Sequence[str], chain_b: Sequence[str],
                     frames: Iterable[np.ndarray],
                     target_prompts: Sequence[str] = ("basket", "bin"),
                     tol: float = 0.02) -> AgreementResult:
    """Do two objects' prompt chains select the same detection?

    Runs each chain over the same frames through the caller's own perception
    object --- the deployed path, not a re-implementation --- and compares the
    selected source-box centres frame by frame.

    Args:
        perception: Anything with ``set_role_prompts(source, target)`` and
            ``perceive(frame) -> Perception`` (``.source.center``,
            ``.source.confidence``). Use the real detector; a mock will only
            tell you the probe runs.
        chain_a: Source prompt chain for the first object, in preference order.
            Pass the chain your policy actually deploys, not the bare noun ---
            the fallback tail is the thing under test.
        chain_b: Source prompt chain for the second object.
        frames: Frames in the policy's expected colour order. Real frames from
            the deployed viewpoint; a probe run on training-corpus frames tells
            you about the corpus, not the deployment.
        target_prompts: Destination chain, held fixed across both runs.
        tol: Centre distance below which two boxes count as "the same", in
            normalized frame units. Report ``median_distance`` too, so readers
            can see whether the verdict depends on this number.

    Returns:
        :class:`AgreementResult`.

    Note:
        Frames are materialised once and reused for both chains --- comparing
        two prompts against different frames would measure nothing.
    """
    frames = [np.ascontiguousarray(f) for f in frames]
    if not frames:
        return AgreementResult(0, 0.0, 0.0, 0.0, 0.0)

    def run(chain):
        perception.set_role_prompts(list(chain), list(target_prompts))
        out = []
        for f in frames:
            p = perception.perceive(f)
            c = float(p.source.confidence)
            out.append((float(p.source.center[0]), float(p.source.center[1]))
                       if c > 0 else None)
        return out

    a, b = run(chain_a), run(chain_b)
    both = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not both:
        return AgreementResult(0, 0.0, 0.0,
                               sum(v is not None for v in a) / len(frames),
                               sum(v is not None for v in b) / len(frames))
    d = np.array([float(np.hypot(x[0] - y[0], x[1] - y[1])) for x, y in both])
    return AgreementResult(
        n_compared=len(both),
        same_rate=float((d < tol).mean()),
        median_distance=float(np.median(d)),
        detect_rate_a=sum(v is not None for v in a) / len(frames),
        detect_rate_b=sum(v is not None for v in b) / len(frames),
    )


@dataclass
class SwapResult:
    """Outcome of :func:`instruction_swap`.

    ``exact_p`` is an exact McNemar (binomial) test on the discordant pairs.
    Cells must be *paired* --- same seeds, same initial states --- or the test
    does not apply and the field is meaningless.
    """

    n: int
    baseline_successes: int
    swapped_successes: int
    n_discordant: int
    favour_baseline: int
    exact_p: float

    def verdict(self) -> str:
        """One-line reading that refuses to over-claim on an underpowered cell."""
        if self.n_discordant == 0:
            return ("NO EFFECT: identical outcomes on every paired trial. The "
                    "instruction does not reach behaviour through any stage this "
                    "cell exercises.")
        if self.exact_p > 0.05:
            worst = 2.0 / (2 ** self.baseline_successes) if self.baseline_successes else 1.0
            note = ("" if self.baseline_successes >= 5 else
                    f" Note the ceiling: with a {self.baseline_successes}/{self.n} "
                    f"baseline even a total collapse could only reach p={min(1.0, worst):.3f}, "
                    f"so this cell can confirm the ABSENCE of an effect but not its presence.")
            return (f"NO SIGNIFICANT EFFECT: {self.baseline_successes}/{self.n} -> "
                    f"{self.swapped_successes}/{self.n}, exact p={self.exact_p:.4f}.{note}")
        direction = "collapses" if self.favour_baseline > self.n_discordant / 2 else "improves"
        return (f"INSTRUCTION-SENSITIVE: the cell {direction} "
                f"{self.baseline_successes}/{self.n} -> {self.swapped_successes}/{self.n}, "
                f"exact p={self.exact_p:.4f}. Some stage uses the instruction; "
                f"per-stage telemetry is needed to say which.")


def instruction_swap(baseline: Sequence[bool], swapped: Sequence[bool]) -> SwapResult:
    """Compare a task's baseline cell against the same cell run under a swapped instruction.

    The environment, physics and success criterion must stay bound to the REAL
    task; only what the policy is told changes. Success is therefore still
    scored on the original task, and a policy that keeps succeeding is one that
    was not using the instruction.

    Args:
        baseline: Per-trial successes with the correct instruction.
        swapped: Per-trial successes under the swapped instruction, from the
            SAME seeds and initial states, in the same order.

    Returns:
        :class:`SwapResult`.

    Raises:
        ValueError: If the two sequences differ in length --- unpaired cells
            cannot be compared trial by trial, and silently truncating them
            would fabricate a pairing.
    """
    from math import comb

    if len(baseline) != len(swapped):
        raise ValueError(
            f"cells must be paired trial-for-trial: got {len(baseline)} baseline "
            f"and {len(swapped)} swapped trials. Re-run with identical seeds.")
    n = len(baseline)
    disc = [(a, b) for a, b in zip(baseline, swapped) if bool(a) != bool(b)]
    k = sum(1 for a, _ in disc if a)
    m = len(disc)
    p = 1.0 if m == 0 else min(
        1.0, 2 * sum(comb(m, j) for j in range(0, min(k, m - k) + 1)) / 2 ** m)
    return SwapResult(
        n=n,
        baseline_successes=sum(bool(x) for x in baseline),
        swapped_successes=sum(bool(x) for x in swapped),
        n_discordant=m,
        favour_baseline=k,
        exact_p=float(p),
    )
