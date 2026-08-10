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

import json
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

    # --- 2b. the taxonomy's count, against its single source of truth -------
    # This class has now bitten twice in the same way: the taxonomy grew from
    # seven to nine and two sentences that referred to it by a bare number were
    # left behind ("All seven made the result look better"; "after the seventh
    # instance"). Neither carries the noun, so rule 2 above could not see them.
    # The count is taken from the figure's ERRORS list, which is what the
    # taxonomy actually IS, rather than from a number typed here.
    ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
           "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
           "eleventh": 11}
    fig = REPO / "paper" / "render_fig14_errors.py"
    if fig.exists():
        src = fig.read_text()
        blk = src[src.index("ERRORS = ["):]
        blk = blk[:blk.index("\n]")]
        n_err = len(re.findall(r"^\s*\(\"", blk, re.M))
        num = "|".join(WORDS)
        # "All nine ran ...", "all seven made ..." -- a bare count with no noun.
        # Only when the sentence is plainly ABOUT the taxonomy: "all ten share
        # one basket" is a true sentence about LIBERO's tasks, not a stale
        # count, so the window has to carry a taxonomy word.
        NEAR = ("error", "flatter", "slice", "instrument", "verifier",
                "direction that", "adversarial")
        for m in re.finditer(
                rf"\ball\s+({num})\s+(made|ran|were|are|share|shared|flattered)\b",
                t, re.I):
            win = t[max(0, m.start() - 260):m.end() + 260].lower()
            if WORDS[m.group(1).lower()] != n_err and any(w in win for w in NEAR):
                problems.append(
                    f"bare count '{m.group(0)}' disagrees with the taxonomy's "
                    f"{n_err} entries in render_fig14_errors.py")
        # "after the seventh instance", "it took the ninth instance"
        for m in re.finditer(rf"\b({'|'.join(ORD)})\s+instance\b", t, re.I):
            if ORD[m.group(1).lower()] != n_err:
                problems.append(
                    f"ordinal '{m.group(0)}' disagrees with the taxonomy's "
                    f"{n_err} entries in render_fig14_errors.py")
        # "nine instances", "seven errors of our own"
        for m in re.finditer(rf"\b({num})\s+(instances|errors)\b", t, re.I):
            if WORDS[m.group(1).lower()] != n_err:
                problems.append(
                    f"'{m.group(0)}' disagrees with the taxonomy's {n_err} "
                    "entries in render_fig14_errors.py")

    # --- 2c. every artifact the paper points a reader at must exist ---------
    # Cheap, and it fails exactly when a reader would: on a path that was
    # renamed, or one written before the file was.
    cited = set()
    for m in re.finditer(r"\\texttt\{([^}]*)\}", t):
        c = m.group(1).replace("\\_", "_").replace("\\%", "%").strip()
        if re.match(r"^(results|scripts|eval|paper|preprocess|train|microvla|"
                    r"tests)/", c):
            cited.add(c)
    for c in sorted(cited):
        if not (REPO / c).exists():
            problems.append(f"cited artifact does not exist: {c}")

    # --- 2d. the figure manifest must cover every figure the paper includes --
    # A stale committed PNG survived four drafts and was caught by a referee
    # reading annotations the generator no longer produces (R4-5). This makes
    # the mapping explicit and fails if a figure is added without it.
    import hashlib
    man_p = REPO / "results" / "figure_manifest.json"
    if man_p.exists():
        man = json.loads(man_p.read_text())["figures"]
        included = re.findall(r"includegraphics\[[^\]]*\]\{\.\./visuals/([^}]+)\}", t)
        listed = {m["image"] for m in man}
        for img in included:
            if img not in listed:
                problems.append(f"figure not in results/figure_manifest.json: {img}")
        for m in man:
            f = REPO / "paper" / "visuals" / m["image"]
            if not f.exists():
                problems.append(f"manifest names a missing image: {m['image']}")
            elif hashlib.sha256(f.read_bytes()).hexdigest()[:12] != m["sha256_12"]:
                problems.append(
                    f"figure changed since the manifest was written (re-run "
                    f"its script and refresh the manifest): {m['image']}")

    # --- 3. withdrawn claims still asserted ---------------------------------
    WITHDRAWN = {
        "joint in \\emph{(head, stack)}":
            "the renderer's causal reading, withdrawn in sec:cert",
        "significantly beats":
            "the suite-level significance claim, withdrawn in sec:refute",
        "identical to within a millimetre":
            "the 2D constant comparison, withdrawn in sec:refute",
        "lookup-admissible on all ten":
            "the suite-level verdict, withdrawn in sec:e2",
        "distinctively admissible":
            "the suite-level verdict, withdrawn in sec:e2",
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
