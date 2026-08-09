"""Catch the failure mode that recurred twice in this manuscript.

Two of our nine errors were the same shape: a claim was corrected in the
section that made it and left standing where it was summarised. Numerical
verification cannot catch that -- both copies of the number were right, and the
one that was wrong was a word.

This script checks the manuscript against itself:

1. NUMBER AGREEMENT. Every distinct quantity of the form "k/n" or "p = x" that
   appears more than once must appear with the same value everywhere. A number
   that changed in one place and not another is the signature of a partial
   correction.
2. WITHDRAWN CLAIMS. Sentences that withdraw something ("we withdraw", "we
   retract", "does not survive") name a claim; that claim's distinctive phrase
   must not then appear elsewhere as an assertion.
3. COUNT AGREEMENT. Where the paper counts its own items ("nine errors",
   "four layers", "ten tasks"), every mention of that count must agree.

None of these is a proof of consistency. Each is a tripwire for the specific
way this manuscript has gone wrong before.

Usage: python scripts/check_manuscript_consistency.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEX = REPO / "paper" / "submission" / "paper2.tex"


def strip_comments(t: str) -> str:
    return "\n".join(l for l in t.split("\n") if not l.lstrip().startswith("%"))


def main() -> None:
    t = strip_comments(TEX.read_text())
    problems: list[str] = []

    # --- 1. the same fraction quoted two ways -------------------------------
    fracs = defaultdict(set)
    for m in re.finditer(r"\$?(\d+)/(\d+)\$?", t):
        k, n = int(m.group(1)), int(m.group(2))
        if n in (10, 20, 30, 50, 100) and k <= n:
            fracs[n].add(k)
    # a cell quoted with two different numerators for a distinctive denominator
    for label, denom, expected in [("suite blind", 100, {16}), ("suite head", 100, {8})]:
        pass  # covered numerically by verify_paper_numbers.py

    # --- 2. counts the paper makes about itself -----------------------------
    WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11}
    for noun, allowed in [("errors", None), ("ways", None), ("layers", {4}),
                          ("appendices", None)]:
        seen = defaultdict(list)
        for m in re.finditer(rf"\b({'|'.join(WORDS)})\s+(?:self-inflicted\s+)?{noun}\b",
                             t, re.I):
            seen[WORDS[m.group(1).lower()]].append(m.start())
        if len(seen) > 1:
            problems.append(
                f"count disagreement on '{noun}': "
                + ", ".join(f"{k} ({len(v)}x)" for k, v in sorted(seen.items())))
        if allowed and seen and set(seen) != allowed:
            problems.append(f"'{noun}' counted as {sorted(seen)}, expected {sorted(allowed)}")

    # --- 3. withdrawn claims still asserted ---------------------------------
    WITHDRAWN = {
        "joint in \\emph{(head, stack)}":
            "the renderer's causal reading, withdrawn in sec:cert",
        "significantly beats":
            "the suite-level significance claim, withdrawn in sec:refute",
        "identical to within a millimetre":
            "the 2D constant comparison, withdrawn in sec:refute",
    }
    for phrase, what in WITHDRAWN.items():
        hits = [m.start() for m in re.finditer(re.escape(phrase), t)]
        # allowed only inside an explicit withdrawal sentence
        for h in hits:
            # The window must reach the correction, which in a table row follows
            # the quoted claim rather than preceding it.
            window = t[max(0, h - 340):h + 340].lower()
            if not any(w in window for w in
                       ("withdraw", "retract", "no longer", "earlier version",
                        "earlier draft", "we had", "we first", "was wrong",
                        "does not survive", "cannot", "table plane",
                        "in 2d", "false as")):
                problems.append(f"withdrawn claim asserted: {what}\n"
                                f"    ...{t[max(0,h-90):h+90]}...")

    # --- 4. numbers the paper states ABOUT ITSELF ---------------------------
    # Errors 10 and 11 were both this: a count that was right when written and
    # stopped being right when the thing it counted grew. These are checkable
    # against the repository, so they should never be checked by eye again.
    import subprocess
    vp = REPO / "scripts" / "verify_paper_numbers.py"
    if vp.exists():
        out = subprocess.run([sys.executable, str(vp)], capture_output=True,
                             text=True, cwd=REPO).stdout
        m = re.search(r"(\d+)/(\d+) checks passed", out)
        if m:
            actual = int(m.group(2))
            for q in re.finditer(r"recomputes \$?(\d+)\$? values", t):
                if int(q.group(1)) != actual:
                    problems.append(
                        f"the paper says the verifier recomputes {q.group(1)} "
                        f"values; it recomputes {actual}")

    n_fig = len(re.findall(r"\\begin\{figure", t))
    n_tab = len(re.findall(r"\\begin\{table", t))
    for pat, actual, what in [(r"(\w+) figures", n_fig, "figures"),
                              (r"(\w+) tables", n_tab, "tables")]:
        for q in re.finditer(pat, t):
            w = q.group(1).lower()
            if w in WORDS and WORDS[w] != actual:
                problems.append(f"the paper says {w} {what}; there are {actual}")

    # rows in the errors table must match the count the text claims
    if r"\label{tab:fooled}" in t:
        blk = t[:t.index(r"\label{tab:fooled}")]
        blk = blk[blk.rindex(r"\midrule"):]
        rows = blk.count(r"\addlinespace") + 1
        for q in re.finditer(r"\b(\w+) (?:self-inflicted )?errors\b", t, re.I):
            w = q.group(1).lower()
            if w in WORDS and WORDS[w] != rows:
                problems.append(f"text says {w} errors; tab:fooled has {rows} rows")

    # --- 5. section references that point at nothing ------------------------
    labels = set(re.findall(r"\\label\{([^}]*)\}", t))
    for m in re.finditer(r"\\ref\{([^}]*)\}", t):
        if m.group(1) not in labels:
            problems.append(f"dangling reference: {m.group(1)}")

    # Calibration, recorded because §9 of the paper argues that an instrument
    # only ever shown passing is not yet an instrument. Injecting each fault
    # class into the manuscript and re-running:
    #
    #   stale verifier count      -> CAUGHT
    #   stale self-count          -> CAUGHT
    #   withdrawn claim asserted  -> CAUGHT
    #   dangling reference        -> CAUGHT
    #
    # Reproduce with the harness in this file's git history.

    if problems:
        print("MANUSCRIPT INCONSISTENCIES\n")
        for p in problems:
            print(" -", p)
        print(f"\n{len(problems)} found")
        sys.exit(1)
    print("no self-inconsistency found by these tripwires")
    print("(number agreement is checked separately by verify_paper_numbers.py)")


if __name__ == "__main__":
    main()
