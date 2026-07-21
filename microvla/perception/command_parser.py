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
    "pour",
    "stack",
    "set",
    "drop",
    "insert",
    "transfer",
)

#: Prepositions linking source and destination in the "verb X prep Y" pattern.
#: Sorted longest-first so multi-word phrases ("on top of") are tried before
#: any shorter phrase that happens to share a prefix ("on").
_PREPOSITIONS = sorted(
    (
        "to",
        "onto",
        "on",
        "on top of",
        "into",
        "in",
        "inside",
        "inside of",
        "near",
        "next to",
        "beside",
        "toward",
        "towards",
        "at",
        "by",
        "behind",
        "in front of",
        "out of",
        "off of",
        "off",
        "under",
        "underneath",
        "over",
        "above",
        "against",
        "between",
    ),
    key=len,
    reverse=True,
)

#: Leading spatial wrappers stripped when deriving a DETECTABLE noun phrase
#: ("right side of the table" -> "the table"): open-vocab detectors ground
#: objects, not regions.
_LOCATION_WRAPPER_RE = re.compile(
    r"^(?:(?:the\s+)?(?:left|right|top|bottom|front|back|middle|center|edge|side|corner)"
    r"(?:\s+(?:side|edge|corner|part|half))?\s+of\s+)+"
)

#: Directional suffixes for the bare "push X <direction>" pattern.
_DIRECTION = r"left|right|up|down|forward|back(?:ward)?"

_VERB_PREP_RE = re.compile(
    r"^(" + "|".join(_MOVE_VERBS) + r")\s+(.+?)\s+("
    + "|".join(re.escape(p) for p in _PREPOSITIONS)
    + r")\s+(.+)$"
)
_PUSH_DIRECTION_RE = re.compile(r"^push\s+(.+?)\s+(" + _DIRECTION + r")$")
#: Compound pick-and-place ("pick up the X and place it in the Y") — the
#: dominant LIBERO instruction template; must match before the bare pick-up
#: pattern or the whole clause becomes the object.
_PICK_PLACE_RE = re.compile(
    r"^(pick up|grab|grasp|take|lift)\s+(.+?)\s+and\s+(?:place|put|drop|set)\s+"
    r"(?:it|them)\s+(?:" + "|".join(re.escape(p) for p in _PREPOSITIONS) + r")\s+(.+)$"
)
_PICK_UP_RE = re.compile(r"^pick up\s+(.+)$")
_GRAB_GRASP_LIFT_RE = re.compile(r"^(grab|grasp|lift)\s+(.+)$")
_POINT_RE = re.compile(r"^point\s+(?:at|to)\s+(.+)$")
_GO_TO_RE = re.compile(r"^go to\s+(.+)$")
_LOOK_AT_RE = re.compile(r"^look at\s+(.+)$")
#: Articulated-object commands ("open the top drawer [and ...]") — the object
#: is the grounding target; any trailing "and ..." clause is secondary.
_ARTICULATE_RE = re.compile(
    r"^(open|close|shut|turn on|turn off|press|toggle)\s+(.+?)(?:\s+and\s+.+)?$"
)

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
    """Derives a DETECTABLE noun phrase for use as a detector class prompt.

    Open-vocab detectors ground *objects*, not regions or clauses, and CLIP
    text/box alignment is strongest on short bare noun phrases. Real
    teleop annotations ("right side of the table on top of the block",
    "small spoon from basket") need three cleanups, applied in order:

    1. Strip leading spatial wrappers: "right side of the table" -> "the table".
    2. Cut trailing prepositional clauses: "the table on top of the block" ->
       "the table"; "spoon from basket" -> "spoon".
    3. Strip a leading article: "the table" -> "table".

    The FULL phrase (untouched) is still what gets CLIP-embedded for fusion's
    text tokens — only the detection prompt is simplified.

    Args:
        phrase: A noun phrase, already lowercased/normalized.

    Returns:
        A short detector-friendly noun phrase (never empty; falls back to
        the input).
    """
    p = _LOCATION_WRAPPER_RE.sub("", phrase).strip()
    clause = re.split(
        r"\s+(?:" + "|".join(re.escape(x) for x in _PREPOSITIONS) + r"|from|of|that|which)\s+",
        p,
        maxsplit=1,
    )[0].strip()
    if clause:
        p = clause
    stripped = re.sub(r"^(?:the|a|an)\s+", "", p).strip()
    return stripped or phrase


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

    match = _PICK_PLACE_RE.match(cleaned)
    if match:
        verb, source, target = match.groups()
        return ParsedCommand(raw=text, verb=verb, source=source.strip(), target=target.strip())

    match = _VERB_PREP_RE.match(cleaned)
    if match:
        verb, source, _prep, target = match.groups()
        # "put the spoon from the basket to the tray": the origin clause
        # belongs to neither role — the object is the head, the destination
        # is the final target.
        source = re.split(r"\s+from\s+", source, maxsplit=1)[0]
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

    match = _ARTICULATE_RE.match(cleaned)
    if match:
        verb, obj = match.groups()
        obj = obj.strip()
        return ParsedCommand(raw=text, verb=verb, source=obj, target=obj)

    return ParsedCommand(raw=text, verb="do", source=cleaned, target=cleaned)
