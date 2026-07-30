"""Per-role detection prompt chains, shared by the bake path and deployment.

YOLO-World's region-text head scores 0.000 on the product names LIBERO tasks are
written in ("alphabet soup", "cream cheese"), so a corpus baked with
``set_classes([source, target])`` had the source object detected on **0.0% of
frames** — the blind corpus of paper.md 4n. The same objects DO ground under
concrete visual categories on the same frames (agentview "bottle" 0.604 / "can"
0.246 / "box" 0.195; wrist "box" 0.499), which is what a chain supplies:
``set_role_prompts`` takes the best box of the FIRST prompt that detected
anything, so the exact phrase still wins wherever it grounds and the tail only
supplies recall where it does not. Adding chains took source detection from 0%
to 48% of frames.

**This module exists because the two sides must agree.** The chains originally
lived in ``preprocess/common.py``, so the fix reached the corpus but not the
robot: ``microvla/jepa/loop.py`` kept building prompts as ``[phrase, bare noun]``
— exactly the prompts that score 0.000 — and every closed-loop eval therefore ran
a policy that had TRAINED sighted BLIND. Measured on the v8_act run: source
detected on 0.0% of real ticks, target on 20% at mean confidence 0.007, against a
corpus with 48% source detection. The resulting off-distribution inputs saturated
the planner (emitted gripper pinned at -1.0 on 9000/9000 ticks against a corpus
that closes on 52.3% of frames), which is a guaranteed 0.000 on any pick task.

Anything that grounds a role — bake, deploy, or eval — must call
:func:`role_chains` from here. ``tests/test_prompt_fallbacks.py`` pins that the
deployment path and the bake path produce identical chains for the same task.

Pure stdlib by design: ``microvla`` must import with torch+numpy alone, and this
module is on the deployment path.
"""
from __future__ import annotations

from microvla.perception.command_parser import strip_article

#: Concrete visual categories appended to a role's prompt chain, keyed by what
#: the phrase is ABOUT. A single tail cannot work: libero_object is groceries
#: ("alphabet soup" -> box/can/bottle) while libero_spatial is tableware
#: ("black bowl" -> ceramic bowl). Kept SHORT: an 8-prompt chain grounded the
#: same objects at conf 0.253 where a 2-prompt chain scored 0.505 on the same
#: frames, and detection rate went 75% -> 88%. More fallbacks is not more recall.
#:
#: Entries are ordered by MEASURED firing rate on real wrist frames, not by how
#: natural they read. "ceramic bowl" fires on 100% of libero_spatial frames at
#: conf 0.950; "black bowl" — the phrase the task actually uses — fires on 25%
#: at 0.235. Since a chain takes the first prompt that detects ANYTHING, a weak
#: early entry blocks a strong later one.
_TAIL_GROCERY: tuple[str, ...] = ("box", "cardboard box", "can")
_TAIL_TABLEWARE: tuple[str, ...] = ("ceramic bowl", "bowl")
_TAIL_RECEPTACLE: tuple[str, ...] = ("basket", "bin")
#: Target-role tails, kept DISJOINT from the source tails above — see
#: :func:`role_chains` for why that matters.
_TAIL_TARGET_TABLEWARE: tuple[str, ...] = ("white plate", "plate")

#: Head nouns that select a tail. Matched against the LAST word of the phrase
#: first, then anywhere in it, so "black bowl" and "wine bottle" both resolve.
_TAIL_BY_NOUN: dict[str, tuple[str, ...]] = {}
for _n in ("bowl", "plate", "cup", "mug", "dish", "tray", "pan", "pot",
           "ramekin", "saucer"):
    _TAIL_BY_NOUN[_n] = _TAIL_TABLEWARE
for _n in ("basket", "bin", "caddy", "crate"):
    _TAIL_BY_NOUN[_n] = _TAIL_RECEPTACLE
for _n in ("soup", "sauce", "cheese", "butter", "juice", "milk", "pudding",
           "ketchup", "dressing", "can", "bottle", "box", "carton", "cream"):
    _TAIL_BY_NOUN[_n] = _TAIL_GROCERY

#: Used when the phrase matches nothing above. Both families, tableware first —
#: a wrongly-fired grocery box on a tableware scene is worse than a miss,
#: because set_role_prompts takes the FIRST prompt that detected anything.
_TAIL_DEFAULT: tuple[str, ...] = _TAIL_TABLEWARE + _TAIL_GROCERY


def with_fallbacks(phrase: str) -> list[str]:
    """Prompt chain for one role: the exact phrase, then concrete categories.

    The head noun goes in before the generic tail — "alphabet soup" -> "soup" —
    because a multi-word product name sometimes grounds on its noun alone when
    Takes the phrase as given: :func:`role_chains` is the entry point that
    normalizes, and this stays a pure chain builder so a caller that has already
    picked its exact prompt keeps it verbatim.
    """
    chain = [phrase]
    parts = phrase.lower().split()
    if len(parts) > 1:
        chain.append(parts[-1])
    tail = None
    if parts:
        tail = _TAIL_BY_NOUN.get(parts[-1])      # head noun wins
    if tail is None:
        for w in parts:                          # then any word in the phrase
            if w in _TAIL_BY_NOUN:
                tail = _TAIL_BY_NOUN[w]
                break
    chain += [c for c in (tail or _TAIL_DEFAULT) if c not in chain]
    return chain


def role_chains(src: str, tgt: str) -> tuple[list[str], list[str] | None]:
    """Prompt chains for the two roles, guaranteed DISJOINT.

    Overlapping chains silently collapse the two roles onto one object. Measured
    on libero_spatial, where source "black bowl" and target "plate" both expand
    to tails containing bowl/plate/cup: 70% of frames returned the SAME BOX for
    both roles. The corpus then reports high target detection that is really an
    echo of the source, and the planner sees the thing it must move sitting
    exactly where it must move it to.

    Disjointness is enforced by dropping shared prompts from the TARGET chain:
    the source object is what grounding actually needs, and a target that
    resolves to a generic backup is less harmful than a source that resolves to
    the target's box.

    Both phrases go through ``strip_article`` HERE — the detector-friendly noun
    phrase is what grounds, and doing it at this one entry point is what keeps
    bake and deploy identical. The bake used to call ``strip_article`` itself
    before building chains while the deployment path passed the parser's raw
    phrase, so the two sides sent "alphabet soup" and "the alphabet soup" to the
    same detector: a second train/deploy divergence, independent of the missing
    tails, and invisible to any test that exercised one side alone.

    Returns:
        ``(source_chain, target_chain)``; ``target_chain`` is None when the task
        names one object for both roles, which the caller passes straight to
        ``set_role_prompts`` to make the target share the source's box.
    """
    src = strip_article(src).strip() or src
    tgt = strip_article(tgt).strip() or tgt
    if src == tgt:
        return with_fallbacks(src), None
    s = with_fallbacks(src)
    t_chain = with_fallbacks(tgt)
    tgt_noun = tgt.split()[-1] if tgt.split() else ""
    # Tableware targets get their own tail so "plate" does not fall back onto
    # the same bowl prompts the source is using.
    if _TAIL_BY_NOUN.get(tgt_noun) is _TAIL_TABLEWARE:
        t_chain = [tgt] + [c for c in _TAIL_TARGET_TABLEWARE if c != tgt]
    # Receptacle targets (basket/bin): grocery source tails include "box" /
    # "cardboard box", which fire on the basket liner (IBVS forensics:
    # "salad dressing" bound the BASKET at conf 0.15). Strip those from the
    # SOURCE chain when the place target is a receptacle — keep exact phrase,
    # noun, and non-box grocery cues ("can").
    if _TAIL_BY_NOUN.get(tgt_noun) is _TAIL_RECEPTACLE:
        _boxish = {"box", "cardboard box"}
        s = [c for c in s if c not in _boxish] or [src]
    t_chain = [c for c in dict.fromkeys(t_chain) if c not in s]
    if not t_chain:
        # Every target prompt collided with the source chain. Keep the exact
        # target phrase and surrender it from the source rather than return an
        # empty chain, which would silently disable the target role entirely.
        t_chain = [tgt]
        s = [c for c in s if c != tgt] or [src]
    return s, t_chain
