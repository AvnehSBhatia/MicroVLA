"""Rule-based natural-language command parser (v2, MiniLM-free).

Extracts a normalized verb phrase plus an ordered SOURCE / TARGET noun-phrase
pair from a free-text instruction, e.g. "move the can to the ball" ->
verb="move", source="the can", target="the ball". Pure Python, zero
dependencies, so it can be imported (and reused by ``MockTaskEncoder``)
without pulling in torch.

Normalization: the input is lowercased, trailing sentence punctuation is
stripped, and internal whitespace is collapsed to single spaces before any
pattern is matched. Articles and modifiers inside a noun phrase ("the red
cup") are preserved verbatim -- only the surrounding verb/preposition tokens
are consumed by the patterns below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Verbs that take an explicit "X <preposition> Y" destination pattern.
_MOVE_VERBS = (
    "move",
    "put",
    "place",
    "push",
    "bring",
    "carry",
    "slide",
    "drag",
    "take",
)

#: Prepositions linking source and destination in the "verb X prep Y" pattern.
#: Sorted longest-first so multi-word phrases ("in front of") are tried before
#: any shorter phrase that happens to share a prefix.
_PREPOSITIONS = sorted(
    (
        "to",
        "onto",
        "on",
        "into",
        "in",
        "near",
        "next to",
        "toward",
        "towards",
        "at",
        "by",
        "behind",
        "in front of",
    ),
    key=len,
    reverse=True,
)

#: Directional suffixes for the bare "push X <direction>" pattern.
_DIRECTION = r"left|right|up|down|forward|back(?:ward)?"

_VERB_PREP_RE = re.compile(
    r"^(" + "|".join(_MOVE_VERBS) + r")\s+(.+?)\s+("
    + "|".join(re.escape(p) for p in _PREPOSITIONS)
    + r")\s+(.+)$"
)
_PUSH_DIRECTION_RE = re.compile(r"^push\s+(.+?)\s+(" + _DIRECTION + r")$")
_PICK_UP_RE = re.compile(r"^pick up\s+(.+)$")
_GRAB_GRASP_LIFT_RE = re.compile(r"^(grab|grasp|lift)\s+(.+)$")
_POINT_RE = re.compile(r"^point\s+(?:at|to)\s+(.+)$")
_GO_TO_RE = re.compile(r"^go to\s+(.+)$")
_LOOK_AT_RE = re.compile(r"^look at\s+(.+)$")

_TRAILING_PUNCT_RE = re.compile(r"[.!?]+$")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ParsedCommand:
    """Structured decomposition of a natural-language robot command.

    Attributes:
        raw: The exact string passed to ``parse_command`` (unmodified).
        verb: Normalized verb phrase, e.g. "move", "pick up", "push left".
        source: Noun phrase for the object acted on, e.g. "the red cup".
        target: Destination noun phrase; equals ``source`` when the command
            has no explicit destination (grab/point/go-to/... style verbs).
    """

    raw: str
    verb: str
    source: str
    target: str


def _normalize(text: str) -> str:
    """Lowercases, strips trailing punctuation, and collapses whitespace."""
    cleaned = text.strip().lower()
    cleaned = _TRAILING_PUNCT_RE.sub("", cleaned).strip()
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def strip_article(phrase: str) -> str:
    """Drops a leading article for use as a detector class prompt.

    "the red cup" -> "red cup". CLIP text/box alignment in open-vocab
    detectors is slightly stronger on bare noun phrases, so detection class
    prompts are article-stripped while the full phrases (articles intact)
    are still what gets embedded for fusion's text tokens.

    Args:
        phrase: A noun phrase, already lowercased/normalized.

    Returns:
        The phrase without a leading "the"/"a"/"an" (unchanged if the
        article is the whole phrase).
    """
    stripped = re.sub(r"^(?:the|a|an)\s+", "", phrase)
    return stripped if stripped else phrase


def parse_command(text: str) -> ParsedCommand:
    """Parses a free-text instruction into a verb + ordered source/target.

    Args:
        text: Natural-language robot command, any casing/whitespace/
            punctuation.

    Returns:
        A ``ParsedCommand``. Order is significant: "move can to ball" yields
        ``source="can", target="ball"``; "move ball to can" swaps them.
        Commands with no destination (e.g. "grab the cup") yield
        ``source == target``. Unrecognized commands fall back to
        ``verb="do", source=target=<normalized text>``.
    """
    cleaned = _normalize(text)

    match = _VERB_PREP_RE.match(cleaned)
    if match:
        verb, source, _prep, target = match.groups()
        return ParsedCommand(raw=text, verb=verb, source=source.strip(), target=target.strip())

    match = _PUSH_DIRECTION_RE.match(cleaned)
    if match:
        obj, direction = match.groups()
        return ParsedCommand(raw=text, verb=f"push {direction}", source=obj.strip(), target=obj.strip())

    match = _PICK_UP_RE.match(cleaned)
    if match:
        obj = match.group(1).strip()
        return ParsedCommand(raw=text, verb="pick up", source=obj, target=obj)

    match = _GRAB_GRASP_LIFT_RE.match(cleaned)
    if match:
        verb, obj = match.groups()
        obj = obj.strip()
        return ParsedCommand(raw=text, verb=verb, source=obj, target=obj)

    match = _POINT_RE.match(cleaned)
    if match:
        obj = match.group(1).strip()
        return ParsedCommand(raw=text, verb="point", source=obj, target=obj)

    match = _GO_TO_RE.match(cleaned)
    if match:
        obj = match.group(1).strip()
        return ParsedCommand(raw=text, verb="go to", source=obj, target=obj)

    match = _LOOK_AT_RE.match(cleaned)
    if match:
        obj = match.group(1).strip()
        return ParsedCommand(raw=text, verb="look at", source=obj, target=obj)

    return ParsedCommand(raw=text, verb="do", source=cleaned, target=cleaned)
