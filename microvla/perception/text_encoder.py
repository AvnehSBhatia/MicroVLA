"""Task text encoders producing L2-normalized CLIP-space embeddings (v2).

MiniLM is gone in v2: the only "real" text encoder is ``ClipTaskEncoder``,
which harvests the CLIP text-tower embeddings YOLO-World's own open-vocabulary
head already computes from ``set_classes`` -- no separate sentence encoder is
loaded. ``MockTaskEncoder`` is a deterministic, model-free stand-in for tests.

Both produce a ``TaskEncoding`` holding three ordered ``[text_dim]``
float32, L2-normalized embeddings: the full command, the source noun phrase,
and the target noun phrase (equal to the source when the command has no
distinct destination).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from microvla.perception.command_parser import ParsedCommand, parse_command

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from microvla.perception.yolo_world import YoloWorldPerception


@dataclass
class TaskEncoding:
    """The three ordered text tokens for one parsed task instruction.

    Attributes:
        command_emb: ``[text_dim]`` float32, L2-normalized embedding of the
            full command string.
        source_emb: ``[text_dim]`` float32, L2-normalized embedding of the
            source noun phrase.
        target_emb: ``[text_dim]`` float32, L2-normalized embedding of the
            target noun phrase (identical tensor to ``source_emb`` when the
            parsed command has no distinct destination).
        parsed: The ``ParsedCommand`` this encoding was built from.
    """

    command_emb: torch.Tensor
    source_emb: torch.Tensor
    target_emb: torch.Tensor
    parsed: ParsedCommand

    def tokens(self) -> torch.Tensor:
        """Stacks the three embeddings into fusion's text-token order.

        Returns:
            ``[3, text_dim]`` float32 tensor: rows are (command, source,
            target), matching ``MicroVLAConfig.n_text_tokens``.
        """
        return torch.stack([self.command_emb, self.source_emb, self.target_emb], dim=0)


class ClipTaskEncoder:
    """Harvests CLIP text embeddings from a ``YoloWorldPerception``'s model.

    YOLO-World's open-vocabulary head already runs every class string through
    a frozen CLIP text tower whenever ``set_classes`` is called; this class
    reuses that tower instead of loading a second text encoder. It touches
    ``ultralytics`` only lazily, through the already-constructed ``perception``
    object -- importing this module never requires ``ultralytics`` itself.

    Args:
        perception: A ``YoloWorldPerception`` wrapping a loaded YOLO-World
            model (``perception.model`` is the ultralytics ``YOLOWorld``
            instance).
    """

    def __init__(self, perception: "YoloWorldPerception") -> None:
        self.perception = perception

    def encode(self, text: str) -> TaskEncoding:
        """Parses ``text`` and harvests (command, source, target) CLIP embeddings.

        Sets the underlying model's class list to ``[command, source, target]``
        (or ``[command, source]`` when the parse has no distinct destination,
        avoiding running the text tower twice on the same phrase) to read the
        internal ``txt_feats`` the CLIP text tower just computed, then leaves
        the model's ACTIVE detection classes as ``[source, target]`` (or
        ``[source]`` when ``source == target``) via ``perception.set_classes``.

        Args:
            text: Natural-language task instruction.

        Returns:
            A ``TaskEncoding`` with detached, CPU, float32 embeddings.
        """
        parsed = parse_command(text)
        same = parsed.source == parsed.target
        harvest_classes = [parsed.raw, parsed.source] if same else [parsed.raw, parsed.source, parsed.target]

        with torch.no_grad():
            self.perception.model.set_classes(list(harvest_classes))
            txt_feats = self._read_txt_feats()

        txt_feats = txt_feats.detach().to(device="cpu", dtype=torch.float32)
        command_emb = txt_feats[0, 0].clone()
        source_emb = txt_feats[0, 1].clone()
        target_emb = source_emb if same else txt_feats[0, 2].clone()

        active_classes = [parsed.source] if same else [parsed.source, parsed.target]
        self.perception.set_classes(active_classes)

        return TaskEncoding(
            command_emb=command_emb,
            source_emb=source_emb,
            target_emb=target_emb,
            parsed=parsed,
        )

    def _read_txt_feats(self) -> torch.Tensor:
        """Locates the ``[1, N, text_dim]`` CLIP text-tower output.

        Ultralytics stores ``txt_feats`` on the underlying detection
        ``nn.Module`` (``YOLOWorld.model``) after ``set_classes``; this falls
        back to the wrapper itself defensively in case that layout differs
        across ultralytics versions.

        Returns:
            The raw ``txt_feats`` tensor, un-normalized/un-detached (the
            caller handles that).

        Raises:
            RuntimeError: If no ``txt_feats`` attribute can be found.
        """
        underlying = getattr(self.perception.model, "model", None)
        txt_feats = getattr(underlying, "txt_feats", None) if underlying is not None else None
        if txt_feats is None:
            txt_feats = getattr(self.perception.model, "txt_feats", None)
        if txt_feats is None:
            raise RuntimeError(
                "Could not locate 'txt_feats' on the YOLO-World model after "
                "set_classes(); the ultralytics CLIP text-tower API may have "
                "changed."
            )
        return txt_feats


class MockTaskEncoder:
    """Deterministic (sha256-seeded per phrase) ``TaskEncoding`` -- no model.

    Each distinct phrase maps to a fixed pseudo-random unit vector: the
    SHA-256 digest of the UTF-8 phrase seeds a local ``torch.Generator``, a
    ``[text_dim]`` sample is drawn from N(0, 1), and the result is
    L2-normalized. Uses the same rule-based ``parse_command`` as the real
    encoder, so source/target ordering and dedup behave identically. No
    global RNG state is touched.

    Args:
        text_dim: Output embedding dimensionality.
    """

    def __init__(self, text_dim: int = 512) -> None:
        self.text_dim = text_dim

    def _phrase_emb(self, phrase: str) -> torch.Tensor:
        """Deterministic unit vector for one phrase, seeded by its SHA-256."""
        digest = hashlib.sha256(phrase.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little")
        generator = torch.Generator()
        generator.manual_seed(seed)
        emb = torch.randn(self.text_dim, generator=generator, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=0)

    def encode(self, text: str) -> TaskEncoding:
        """Encodes text into a deterministic pseudo-random ``TaskEncoding``.

        Args:
            text: Natural-language task instruction.

        Returns:
            ``TaskEncoding`` with hash-seeded, L2-normalized embeddings.
            Identical text always yields an identical encoding; identical
            source/target phrases yield the same embedding tensor values.
        """
        parsed = parse_command(text)
        command_emb = self._phrase_emb(parsed.raw)
        source_emb = self._phrase_emb(parsed.source)
        target_emb = source_emb if parsed.source == parsed.target else self._phrase_emb(parsed.target)
        return TaskEncoding(
            command_emb=command_emb,
            source_emb=source_emb,
            target_emb=target_emb,
            parsed=parsed,
        )
