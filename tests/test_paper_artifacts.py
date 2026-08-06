"""Every artifact the papers cite must exist and be tracked.

Four gaps of this exact class were found by hand on 2026-08-06: run `argv` not
recorded, the pinning instrument untracked, the release manifest absent, and
the DAgger checkpoints missing from ``models/`` while two documents cited them
for a headline cell. Each was found by asking "could a stranger reproduce
this?" rather than by any check, so the question is automated here.

The failure mode is specific and recurring: a result gets written up the moment
it lands, and the artifact it depends on stays wherever it was made. A citation
to a file that is not in the repository is a broken claim, not a typo.

CPU-only, no network, no imports beyond the stdlib.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Documents that make claims a reader is invited to check.
DOCS = [
    "README.md",
    "paper/MANUSCRIPT_v2.md",
    "paper/submission/README.md",
    "paper/submission/REPRODUCE.md",
    "models/README.md",
    "results/UNAIDED_LEADERBOARD.md",
]

#: Repo directories whose contents are shipped artifacts. A backticked path
#: starting with one of these is a promise that the file is present.
SHIPPED = ("models/", "scripts/", "eval/", "train/", "microvla/", "preprocess/",
           "results/", "paper/", "tests/", "demo/")

#: Cited paths that are deliberately absent, each with the reason. Anything
#: added here needs a justification in the comment, not just an entry.
ALLOWED_ABSENT = {
    # Upstream weights, not redistributed (~26 MB); digest is in models/README.
    "yolov8s-worldv2.pt",
}

_CODE = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|pt|json|md|sh|jsonl))`")


def _cited_paths() -> dict[str, list[str]]:
    """Backticked file paths per document, restricted to shipped directories."""
    out: dict[str, list[str]] = {}
    for doc in DOCS:
        p = ROOT / doc
        if not p.exists():
            continue
        hits = []
        for m in _CODE.finditer(p.read_text()):
            cand = m.group(1)
            if cand.startswith(SHIPPED) and Path(cand).name not in ALLOWED_ABSENT:
                hits.append(cand)
        if hits:
            out[doc] = sorted(set(hits))
    return out


def test_cited_artifacts_exist():
    """A path a document tells the reader to look at must be there."""
    missing = []
    for doc, paths in _cited_paths().items():
        for rel in paths:
            if not (ROOT / rel).exists():
                missing.append(f"{doc} cites {rel}, which does not exist")
    assert not missing, (
        "documents cite artifacts that are not in the repository:\n  "
        + "\n  ".join(missing)
        + "\n(this is the defect class that hid the DAgger checkpoints backing a "
          "headline cell; ship the file or stop citing it)")


def test_cited_artifacts_are_tracked():
    """Existing-but-untracked is the subtler half: it works for us, not for a clone."""
    try:
        tracked = set(subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
            timeout=30, check=True).stdout.split())
    except Exception:  # no git, shallow export, etc. -- not a paper defect
        pytest.skip("git unavailable; tracking cannot be checked here")

    untracked = []
    for doc, paths in _cited_paths().items():
        for rel in paths:
            if (ROOT / rel).exists() and rel not in tracked:
                untracked.append(f"{doc} cites {rel}, which exists but is untracked")
    assert not untracked, (
        "documents cite artifacts present locally but absent from a clone:\n  "
        + "\n  ".join(untracked)
        + "\n(this is how the placement-pinning instrument -- which the paper's "
          "opening claim rests on -- went four months uncommitted)")


def test_the_audit_covers_the_documents_that_make_claims():
    """Guard the guard: if a claims-bearing document is renamed, notice."""
    present = [d for d in DOCS if (ROOT / d).exists()]
    assert len(present) >= 4, (
        f"only {len(present)} of {len(DOCS)} claim-bearing documents found: "
        f"{present}. If one was renamed, update DOCS -- an audit that silently "
        f"stops reading a document is worse than no audit.")
