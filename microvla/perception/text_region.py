"""Text-space region proposals from YOLO-World's contrastive head (v8).

WHY — the defect this replaces. ``perception/yolo_world.py`` calls
``set_classes([source, target])``, keeps the single highest-confidence box per
class, and ROIAligns the hooked **SPPF** map inside it. SPPF is a *backbone*
(pre-head) tensor, so that embedding is NOT in CLIP text space: the text's
entire causal influence on perception is *which box got picked* — a hard,
non-differentiable argmax over two classes. With ``det_conf = 0.10`` and a
fallback box at confidence 0.0, a missed detection makes the visual evidence
vanish silently, and everything else the frame contains is thrown away before
the world model ever sees it.

WHAT this does instead. YOLO-World's detection head is ultralytics
``WorldDetect`` (verified children on the deployed weights:
``['cv2', 'cv3', 'dfl', 'cv4']``). ``cv4`` is a contrastive head: it maps the
per-anchor feature into CLIP text space, dots it against the L2-normalized text
features, and scales the result by a learned ``logit_scale`` to get class
logits. The embedding crossing that boundary therefore already lives in text
space, while the head's *output* has collapsed those 512 channels to one number
per phrase. Tapping cv4 keeps the vector instead of the number, so
region↔phrase similarity becomes a plain dot product: defined for EVERY
proposal, soft, monotone in how well the language matches, and differentiable
with respect to the text features — instead of a winner-take-all pick.

WHICH tensor is "the embedding" depends on the head variant, and getting it
wrong is silent. ultralytics ships two:

* ``ContrastiveHead`` (world**v1**) computes ``F.normalize(x) @ F.normalize(w)``,
  so cv4's *input* ``x`` is already the scored vector and the right tap is a
  forward **pre**-hook. Its ``logit_scale`` multiplies a genuine cosine, so
  ``1 / exp(logit_scale)`` is a meaningful softmax temperature to adopt.
* ``BNContrastiveHead`` (world**v2** — including this repo's
  ``yolov8s-worldv2.pt``) does **no** L2 normalization: it computes
  ``BatchNorm2d(x) @ F.normalize(w)``. The vector the head actually scores is
  ``norm(x)``, and the BN's learned per-channel affine is not a no-op. Measured
  on ``yolov8s-worldv2.pt``: cosine-argmax over three task phrases agrees
  between the pre-BN and post-BN vectors on only 7% / 94% / 0% of anchors
  (P3/P4/P5). Tapping cv4's input on a v2 checkpoint would therefore report a
  *different* phrase than the detector's own head on two of three FPN levels.
  So on this variant the tap is a forward hook on ``cv4[i].norm``, and its
  ``logit_scale`` (0.5801 on these weights) is NOT adopted: it scales an
  unnormalized dot product, not a cosine. Reading it as a cosine temperature
  gives tau = 0.560, which flattens the three-phrase softmax to a measured mean
  peak of 0.348–0.355 against a uniform floor of 0.333. Measured over the eight
  proposals of one real detection, ``match.max()`` then spans just 0.352–0.361
  — a range of 0.009, i.e. the language term in ``weight`` is a constant to
  within 1%, which is exactly the defect this module exists to remove. At the
  default 0.07 the same eight proposals span 0.472–0.574 (range 0.102, 11x).

Every tap accepts a tensor only when its channel count is ``cfg.vis_dim``, so a
version-shifted head fails loudly in :meth:`TextRegionExtractor.extract` rather
than feeding logits downstream.

:class:`TextRegionExtractor` returns the top ``cfg.max_objects`` proposals by
objectness — class-agnostic, i.e. ranked over ALL detections rather than one
winner per prompt — each with a text-space ``emb``, a normalized ``center``, a
graded ``weight``, and its soft ``match`` against every task phrase, padded to
a static ``cfg.max_objects`` rows. That is the data-rich replacement for two
boxes: several objects per frame, each carrying its own language affinity, so
downstream relational reasoning can decide *which* region matters instead of
inheriting a decision perception already made irreversibly.

Evidence semantics follow CLAUDE.md's one-shared-path rule: ``weight`` is a
graded scalar in ``[0, 1]`` exactly like fusion's ``box_weight`` (objectness ×
soft text match), callers fade it by ``staleness_decay ** k`` on dream ticks and
by ``modality_dropout`` at train time, and a padded row is simply the limit
``weight -> 0``. There is no separate "missing" flag and no binary zeroing.

Zero trainable parameters: the backbone and its head stay frozen, and these are
plain classes rather than ``nn.Module``s, so nothing here can reach an optimizer
or the parameter audit. ``ultralytics`` and ``cv2`` are touched only inside
:meth:`TextRegionExtractor.attach` / :meth:`TextRegionExtractor.extract`; region
pooling is pure ``torch`` (``grid_sample`` on the box's sample grid, which is
``roi_align(sampling_ratio=1, aligned=True)`` followed by a mean over bins)
rather than ``torchvision.ops.roi_align``, so the entire assembly path — the
part that carries the semantics — is importable and testable with torch alone.
``MockTextRegionExtractor`` shares that assembly path verbatim and invents only
the proposals, so the mock and the real detector cannot drift apart in the parts
tests actually check.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from microvla.config import MicroVLAConfig
from microvla.utils.embedding import standardize

#: Softmax temperature for region↔phrase similarity. ultralytics'
#: ``ContrastiveHead`` initializes ``logit_scale = log(1 / 0.07)``, so 0.07 is
#: CLIP's (and that head's) own sharpness; :meth:`TextRegionExtractor.attach`
#: adopts the *learned* value only from that cosine-scoring variant. Left
#: unscaled, region/phrase cosines sit in a narrow band and a softmax over three
#: phrases is nearly uniform — the weight would then carry no language signal at
#: all. Measured on ``yolov8s-worldv2.pt``'s post-BN embeddings, 0.07 yields a
#: mean peak of 0.45–0.51 over three phrases (floor 0.333); the head's own
#: ``logit_scale``, misread as a cosine temperature, yields 0.348–0.355.
_DEFAULT_MATCH_TEMPERATURE = 0.07

#: Center reported for a padded row: the same (0.5, 0.5) ``BoxObs`` uses for a
#: missed detection. It is never ambiguous with a real object at frame center,
#: because a padded row also carries weight 0 (CLAUDE.md's disambiguation rule).
_PAD_CENTER = 0.5

MapsLike = Union[torch.Tensor, Sequence[torch.Tensor]]


@dataclass
class RegionObs:
    """Top-K class-agnostic proposals for one frame (detached, CPU, float32).

    Rows are ordered by descending objectness with the valid ones contiguous at
    the front; every row beyond the number of proposals found is a pad carrying
    ``weight = 0.0``. Consumers are expected to read all ``cfg.max_objects``
    rows and let ``weight`` fade the pads out — the shapes are static precisely
    so nothing downstream needs to branch on how many objects were seen.

    Attributes:
        emb: ``[max_objects, vis_dim]`` text-space region embeddings,
            standardized per vector (the canonical space; pads are exact zeros,
            the ``weight -> 0`` limit of a faded row).
        center: ``[max_objects, 2]`` normalized ``(cx, cy)`` in ``[0, 1]``
            (``(0.5, 0.5)`` for pads).
        weight: ``[max_objects]`` graded evidence in ``[0, 1]``: objectness ×
            soft text match. Exactly ``0.0`` for pads.
        match: ``[max_objects, n_phrases]`` soft similarity to each task phrase
            (a per-proposal softmax over phrases, so rows of real proposals sum
            to 1). Pads are all-zero — deliberately outside the simplex, so a
            pad can never be mistaken for a genuine uniform match.
    """

    emb: torch.Tensor
    center: torch.Tensor
    weight: torch.Tensor
    match: torch.Tensor

    @property
    def n_valid(self) -> int:
        """Number of real proposals (rows with positive weight)."""
        return int((self.weight > 0).sum().item())


def _as_phrase_matrix(text_feats: torch.Tensor) -> torch.Tensor:
    """Normalizes a text-feature tensor to ``[n_phrases, dim]``.

    Accepts what the two producers actually hand over: ultralytics'
    ``txt_feats`` (``[1, P, dim]``), ``TaskEncoding.tokens()`` (``[P, dim]``),
    or a single phrase (``[dim]``).

    Args:
        text_feats: Text-tower features in any of those layouts.

    Returns:
        ``[n_phrases, dim]`` float32 CPU view of the same values.

    Raises:
        ValueError: If the tensor is empty or has more than 3 dims.
    """
    t = torch.as_tensor(text_feats).detach().to(device="cpu", dtype=torch.float32)
    if t.dim() == 3:
        if t.shape[0] != 1:
            raise ValueError(
                f"text_feats with a leading batch dim must have batch 1, got "
                f"{tuple(t.shape)}"
            )
        t = t[0]
    elif t.dim() == 1:
        t = t.unsqueeze(0)
    elif t.dim() != 2:
        raise ValueError(f"text_feats must be 1-, 2- or 3-D, got {tuple(t.shape)}")
    if t.shape[0] == 0:
        raise ValueError("text_feats has no phrases; region↔text matching is undefined")
    return t


def assemble_regions(
    emb: torch.Tensor,
    objectness: torch.Tensor,
    center: torch.Tensor,
    text_feats: torch.Tensor,
    vis_dim: int,
    max_objects: int,
    temperature: float = _DEFAULT_MATCH_TEMPERATURE,
) -> RegionObs:
    """Scores, ranks, standardizes and pads raw proposals into a ``RegionObs``.

    This is the whole semantic core, shared verbatim by the real extractor and
    the mock: everything about *which* proposals survive and *how much* they
    count happens here, so the two paths cannot diverge.

    The soft match is computed on the RAW embeddings, L2-normalized against
    L2-normalized text features — that dot product is precisely the logit
    ``ContrastiveHead`` computes, minus its learned scale and bias, so the
    similarity means what the frozen head trained it to mean. Standardization to
    the canonical zero-mean/unit-std space happens only afterwards, on the
    embeddings that leave: it is an affine per-vector map and would otherwise
    silently change the cosine.

    Ranking is by objectness alone and never per phrase. A "source" and a
    "target" prompt firing on the same region no longer buy that region two
    slots, and a region no prompt names still gets a slot if the detector is
    confident something is there — which is the point of class-agnostic
    proposals.

    Args:
        emb: ``[N, vis_dim]`` raw (un-standardized) text-space embeddings.
        objectness: ``[N]`` detector confidence in ``[0, 1]`` (clamped).
        center: ``[N, 2]`` normalized box centers (clamped to ``[0, 1]``).
        text_feats: Task phrase features, ``[P, vis_dim]`` or ``[1, P, vis_dim]``.
        vis_dim: ``MicroVLAConfig.vis_dim`` — the width every embedding must have.
        max_objects: ``MicroVLAConfig.max_objects`` — the static row count K.
        temperature: Softmax temperature over phrases (see
            ``_DEFAULT_MATCH_TEMPERATURE``).

    Returns:
        A ``RegionObs`` with exactly ``max_objects`` rows.

    Raises:
        ValueError: If ``emb``'s width is not ``vis_dim``, or the phrase
            features are a different width (the dot product would be
            meaningless), or ``temperature`` is not positive.
    """
    if temperature <= 0.0:
        raise ValueError(f"match temperature must be > 0, got {temperature}")
    phrases = _as_phrase_matrix(text_feats)
    n_phrases = int(phrases.shape[0])
    if int(phrases.shape[1]) != int(vis_dim):
        raise ValueError(
            f"text features are {phrases.shape[1]}-d but the region embeddings "
            f"are cfg.vis_dim={vis_dim}-d; cv4's embedding width and the CLIP "
            f"text width must agree for a dot product to mean anything."
        )

    emb = torch.as_tensor(emb).detach().to(device="cpu", dtype=torch.float32)
    if emb.dim() != 2:
        raise ValueError(f"emb must be [N, vis_dim], got {tuple(emb.shape)}")
    n_found = int(emb.shape[0])
    if n_found and int(emb.shape[1]) != int(vis_dim):
        raise ValueError(
            f"region embeddings are {emb.shape[1]}-d but cfg.vis_dim={vis_dim}; "
            f"the hooked cv4 tensor is probably not the embedding branch."
        )

    out_emb = torch.zeros(max_objects, vis_dim, dtype=torch.float32)
    out_center = torch.full((max_objects, 2), _PAD_CENTER, dtype=torch.float32)
    out_weight = torch.zeros(max_objects, dtype=torch.float32)
    out_match = torch.zeros(max_objects, n_phrases, dtype=torch.float32)

    if n_found == 0:
        # No proposals at all: an all-pad observation. Nothing raises and no
        # shape changes — a blind frame is the weight-0 end of the same
        # evidence continuum, exactly like a fully stale dream tick.
        return RegionObs(emb=out_emb, center=out_center, weight=out_weight, match=out_match)

    obj = (
        torch.as_tensor(objectness)
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(-1)
        .clamp(0.0, 1.0)
    )
    ctr = torch.as_tensor(center).detach().to(device="cpu", dtype=torch.float32).reshape(-1, 2)
    if obj.shape[0] != n_found or ctr.shape[0] != n_found:
        raise ValueError(
            f"emb/objectness/center disagree on the proposal count: "
            f"{n_found}/{obj.shape[0]}/{ctr.shape[0]}"
        )

    cos = F.normalize(emb, dim=-1) @ F.normalize(phrases, dim=-1).t()  # [N, P]
    match = torch.softmax(cos / temperature, dim=-1)
    # Graded evidence: objectness says something is there, the match peak says
    # the language cares about it. Both in [0, 1], so the product is too.
    weight = obj * match.max(dim=-1).values

    # Stable sort so equal objectness keeps detector order — a tie must not
    # make perception non-deterministic frame to frame.
    order = torch.argsort(obj, descending=True, stable=True)[:max_objects]
    keep = int(order.numel())
    out_emb[:keep] = standardize(emb.index_select(0, order))
    out_center[:keep] = ctr.index_select(0, order).clamp(0.0, 1.0)
    out_weight[:keep] = weight.index_select(0, order).clamp(0.0, 1.0)
    out_match[:keep] = match.index_select(0, order)
    return RegionObs(emb=out_emb, center=out_center, weight=out_weight, match=out_match)


class _RegionAssembly:
    """Dependency-free half of the extractor API (real + mock inherit it).

    Holds the task phrase features and the map→proposal geometry. Inheriting
    both classes from this is what makes "the mock has the identical API" a
    property of the code rather than a promise in a docstring; it also keeps the
    pooling and scoring reachable in the torch-only test venv.

    Args:
        cfg: Canonical config; supplies ``vis_dim`` and ``max_objects``.
        match_temperature: Softmax temperature over phrases. ``None`` means
            "use the head's default 0.07, and adopt its learned ``logit_scale``
            if :meth:`TextRegionExtractor.attach` can read one".
        pool_samples: Side ``S`` of the ``S x S`` bilinear sample grid placed
            inside each box. 3 keeps a whole small object inside the samples
            while staying cheap; the pooled vector is their mean.
    """

    def __init__(
        self,
        cfg: MicroVLAConfig,
        match_temperature: Optional[float] = None,
        pool_samples: int = 3,
    ) -> None:
        if pool_samples < 1:
            raise ValueError(f"pool_samples must be >= 1, got {pool_samples}")
        self.cfg = cfg
        self.pool_samples = int(pool_samples)
        self._temperature_is_explicit = match_temperature is not None
        self.match_temperature = (
            float(match_temperature)
            if match_temperature is not None
            else _DEFAULT_MATCH_TEMPERATURE
        )
        self._text_feats: Optional[torch.Tensor] = None

    def set_text_features(self, text_feats: torch.Tensor) -> None:
        """Sets the task phrase features every proposal is matched against.

        These must be the CLIP text-tower features the frozen head itself
        scores against — ``TaskEncoding.tokens()`` from
        ``ClipTaskEncoder`` (the ordered command/source/target rows harvested
        out of ``txt_feats``) is exactly that tensor. Anything else lives in a
        different space and the dot product becomes noise.

        Args:
            text_feats: ``[n_phrases, vis_dim]`` (or ``[1, n_phrases, vis_dim]``)
                text features. Call once per task, like ``set_classes``.
        """
        self._text_feats = _as_phrase_matrix(text_feats)

    def _resolve_text(self, text_feats: Optional[torch.Tensor]) -> torch.Tensor:
        """Returns the per-call features, else the ones set for the task.

        Raises:
            RuntimeError: If neither is available — matching is the whole point
                of this path, so silently degrading to "no language" would
                reproduce the very defect the module exists to fix.
        """
        if text_feats is not None:
            return _as_phrase_matrix(text_feats)
        if self._text_feats is None:
            raise RuntimeError(
                "No text features: call set_text_features(TaskEncoding.tokens()) "
                "once per task, or pass text_feats= to this call."
            )
        return self._text_feats

    def regions_from(
        self,
        region_maps: MapsLike,
        boxes_xyxy: torch.Tensor,
        objectness: torch.Tensor,
        frame_hw: tuple[int, int],
        input_hw: Optional[tuple[int, int]] = None,
        text_feats: Optional[torch.Tensor] = None,
    ) -> RegionObs:
        """Builds a ``RegionObs`` from tapped cv4 maps and decoded boxes.

        Boxes are normalized into *letterboxed network input* coordinates
        rather than into any one map's cell grid: ultralytics letterboxes the
        frame (uniform scale ``r = min(H_in/H0, W_in/W0)`` plus centered
        padding) and each cv4 level is a uniform downsample of that same input,
        so ONE normalized box indexes all levels. Every level is pooled and the
        results averaged, which is well-defined precisely because all levels are
        scored against a SINGLE shared text-feature matrix — their embeddings
        must already live in one common space, or the head could not share text
        features across them.

        Args:
            region_maps: One ``[1, vis_dim, H, W]`` cv4 embedding map, or a
                sequence of them (one per FPN level).
            boxes_xyxy: ``[N, 4]`` decoded boxes in original-frame pixels.
            objectness: ``[N]`` detector confidences.
            frame_hw: ``(H, W)`` of the frame the boxes are in.
            input_hw: ``(H_in, W_in)`` network input size. ``None`` skips the
                letterbox correction and scales the frame directly onto the map
                (right when the map covers the frame, e.g. in tests).
            text_feats: Per-call phrase features; ``None`` uses the task's.

        Returns:
            ``RegionObs`` with ``cfg.max_objects`` rows.
        """
        phrases = self._resolve_text(text_feats)
        maps = [region_maps] if torch.is_tensor(region_maps) else list(region_maps)
        maps = [self._as_map(m) for m in maps]
        if not maps:
            raise ValueError("region_maps is empty; nothing to pool from")

        boxes = (
            torch.as_tensor(boxes_xyxy)
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(-1, 4)
        )
        n = int(boxes.shape[0])
        if n == 0:
            return assemble_regions(
                emb=torch.zeros(0, self.cfg.vis_dim),
                objectness=torch.zeros(0),
                center=torch.zeros(0, 2),
                text_feats=phrases,
                vis_dim=self.cfg.vis_dim,
                max_objects=self.cfg.max_objects,
                temperature=self.match_temperature,
            )

        frame_h, frame_w = int(frame_hw[0]), int(frame_hw[1])
        center = torch.stack(
            [
                (boxes[:, 0] + boxes[:, 2]) * 0.5 / max(frame_w, 1),
                (boxes[:, 1] + boxes[:, 3]) * 0.5 / max(frame_h, 1),
            ],
            dim=-1,
        )
        unit = self._boxes_to_unit(boxes, (frame_h, frame_w), input_hw)
        emb = self._pool(maps, unit)
        return assemble_regions(
            emb=emb,
            objectness=objectness,
            center=center,
            text_feats=phrases,
            vis_dim=self.cfg.vis_dim,
            max_objects=self.cfg.max_objects,
            temperature=self.match_temperature,
        )

    def _as_map(self, region_map: torch.Tensor) -> torch.Tensor:
        """Coerces one tapped map to ``[1, C, H, W]`` float32 CPU.

        Raises:
            ValueError: If it is not 3- or 4-D, or is a real batch (> 1).
        """
        m = torch.as_tensor(region_map).detach().to(device="cpu", dtype=torch.float32)
        if m.dim() == 3:
            m = m.unsqueeze(0)
        if m.dim() != 4:
            raise ValueError(
                f"a cv4 map must be [C, H, W] or [1, C, H, W], got {tuple(m.shape)}"
            )
        if m.shape[0] != 1:
            raise ValueError(
                f"perception is per-frame; got a batch of {m.shape[0]} cv4 maps"
            )
        return m

    def _boxes_to_unit(
        self,
        boxes: torch.Tensor,
        frame_hw: tuple[int, int],
        input_hw: Optional[tuple[int, int]],
    ) -> torch.Tensor:
        """Maps boxes from frame pixels to ``[0, 1]`` network-input coordinates.

        Args:
            boxes: ``[N, 4]`` xyxy in original-frame pixels.
            frame_hw: ``(H, W)`` of that frame.
            input_hw: ``(H_in, W_in)`` letterboxed network input, or ``None``.

        Returns:
            ``[N, 4]`` xyxy in ``[0, 1]``, level-independent.
        """
        frame_h, frame_w = frame_hw
        if input_hw is None:
            scale = boxes.new_tensor(
                [1.0 / max(frame_w, 1), 1.0 / max(frame_h, 1)] * 2
            )
            unit = boxes * scale
        else:
            in_h, in_w = int(input_hw[0]), int(input_hw[1])
            r = min(in_h / max(frame_h, 1), in_w / max(frame_w, 1))
            pad_w = (in_w - frame_w * r) * 0.5
            pad_h = (in_h - frame_h * r) * 0.5
            unit = torch.stack(
                [
                    (boxes[:, 0] * r + pad_w) / in_w,
                    (boxes[:, 1] * r + pad_h) / in_h,
                    (boxes[:, 2] * r + pad_w) / in_w,
                    (boxes[:, 3] * r + pad_h) / in_h,
                ],
                dim=-1,
            )
        return unit.clamp(0.0, 1.0)

    def _pool(self, maps: list[torch.Tensor], unit_boxes: torch.Tensor) -> torch.Tensor:
        """Mean of bilinear samples on an ``S x S`` grid inside each box.

        Identical to ``roi_align(output_size=(S, S), sampling_ratio=1,
        aligned=True)`` followed by a mean over bins, but written with
        ``F.grid_sample`` so this path needs no ``torchvision``: the module then
        imports — and its semantics stay testable — in the torch-only venv.

        Args:
            maps: Per-level ``[1, C, H, W]`` cv4 maps.
            unit_boxes: ``[N, 4]`` xyxy in ``[0, 1]`` input coordinates.

        Returns:
            ``[N, C]`` raw text-space region embeddings, averaged over levels.
        """
        s = self.pool_samples
        n = int(unit_boxes.shape[0])
        # Bin centers: the (i + 0.5)/S convention aligned with grid_sample's
        # align_corners=False pixel-center coordinates.
        offs = (torch.arange(s, dtype=torch.float32) + 0.5) / s
        x1, y1, x2, y2 = unit_boxes.unbind(-1)
        xs = x1[:, None] + offs[None, :] * (x2 - x1)[:, None]  # [N, S]
        ys = y1[:, None] + offs[None, :] * (y2 - y1)[:, None]  # [N, S]
        gx = (2.0 * xs - 1.0)[:, None, :].expand(n, s, s)
        gy = (2.0 * ys - 1.0)[:, :, None].expand(n, s, s)
        grid = torch.stack([gx, gy], dim=-1).reshape(1, n, s * s, 2)

        pooled = None
        for m in maps:
            sampled = F.grid_sample(
                m, grid, mode="bilinear", padding_mode="border", align_corners=False
            )  # [1, C, N, S*S]
            level = sampled.mean(dim=-1).squeeze(0).t()  # [N, C]
            pooled = level if pooled is None else pooled + level
        return pooled / float(len(maps))


class TextRegionExtractor(_RegionAssembly):
    """Taps ``WorldDetect.cv4`` on a frozen YOLO-World for text-space proposals.

    Rides an already-loaded detector instead of owning one: on a Pi the model is
    the expensive object, and the v7 box path and this path must in any case see
    the same frame. Adds no parameters and mutates nothing but hooks.

    Args:
        cfg: Canonical config; supplies ``vis_dim`` and ``max_objects``.
        perception: A ``YoloWorldPerception`` (or an ultralytics ``YOLOWorld``)
            to :meth:`attach` to immediately; ``None`` to attach later.
        det_conf: Detection threshold. Lower than the single-box path's 0.10
            because nothing is winner-take-all any more: a weak proposal costs
            one of ``cfg.max_objects`` rows and fades itself out through its own
            weight, while a proposal never emitted is unrecoverable — the same
            argument yolo_world.py makes for 0.10, taken one step further.
        min_side: Frames whose short side is below this are bicubically upscaled
            before detection, matching ``YoloWorldPerception.min_side``; dataset
            frames are tiny and the detector starves at native size (measured on
            LIBERO: basket 0.00 -> 0.57 confidence at 4x).
        match_temperature: See :class:`_RegionAssembly`. ``None`` adopts the
            attached head's learned ``logit_scale``.
        pool_samples: See :class:`_RegionAssembly`.
        device: Torch device string for inference; ``None`` takes the attached
            perception's device, else ``"cpu"``.
    """

    def __init__(
        self,
        cfg: MicroVLAConfig,
        perception: object = None,
        det_conf: float = 0.02,
        min_side: int = 512,
        match_temperature: Optional[float] = None,
        pool_samples: int = 3,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(cfg, match_temperature=match_temperature, pool_samples=pool_samples)
        self.det_conf = float(det_conf)
        self.min_side = int(min_side)
        self.device = device or "cpu"
        self.model = None  # ultralytics wrapper exposing .predict
        #: Which cv4 variant :meth:`attach` found: "bn" (world v2, scores
        #: ``norm(x)``) or "cosine" (world v1, scores ``F.normalize(x)``).
        #: Exposed because it decides both the tap point and whether the head's
        #: ``logit_scale`` is a usable temperature.
        self.head_kind: Optional[str] = None
        self._maps: list[Optional[torch.Tensor]] = []
        self._handles: list = []
        self._input_hw: Optional[tuple[int, int]] = None
        if perception is not None:
            self.attach(perception)

    def attach(self, perception: object) -> None:
        """Registers the cv4 taps (and an input-size probe) on a loaded model.

        The head is located by walking modules and matching the class name
        ``"WorldDetect"`` — never by index, for the same reason the SPPF hook
        does: ultralytics reorders layers between releases.

        The tap point is chosen per variant, because the two contrastive heads
        score *different* tensors (see the module docstring for the measurement
        that makes this consequential rather than pedantic):

        * ``cv4[i]`` has a ``norm`` child (``BNContrastiveHead``, world v2) →
          hook ``norm``'s OUTPUT, which is literally the vector the head dots
          against the text features.
        * otherwise (``ContrastiveHead``, world v1, or a ``fuse()``-d BN head
          whose ``norm`` has been deleted and whose forward is the identity) →
          a forward PRE-hook on ``cv4[i]``, whose *input* is the scored vector,
          plus a forward hook that fires only if the pre-hook caught nothing.

        Every tap accepts a tensor only if its channel count is
        ``cfg.vis_dim``, so a head whose contract moved yields a loud failure in
        :meth:`extract` instead of logits masquerading as embeddings.

        Args:
            perception: ``YoloWorldPerception``-like object exposing ``.model``
                (the ultralytics wrapper), or the wrapper itself.

        Raises:
            TypeError: If no object with ``.predict`` can be found.
            RuntimeError: If the model has no ``WorldDetect`` with a ``cv4``
                branch (the whole premise of this module).
        """
        # Order matters and is not cosmetic: ultralytics' BaseModel ALSO defines
        # a `predict(x, profile=..., visualize=..., augment=..., embed=...)`, so
        # probing `.model` first binds YOLOWorld.model (a WorldModel) instead of
        # the YOLOWorld wrapper whenever the wrapper itself is passed in — and
        # the mismatch only surfaces later, as `predict() got an unexpected
        # keyword argument 'device'` from extract(). The outer object wins:
        # YoloWorldPerception has no `predict` at all and correctly falls
        # through to its `.model`, while a YOLOWorld matches immediately.
        wrapper = perception if hasattr(perception, "predict") else None
        if wrapper is None:
            inner_wrapper = getattr(perception, "model", None)
            if inner_wrapper is not None and hasattr(inner_wrapper, "predict"):
                wrapper = inner_wrapper
        if wrapper is None:
            raise TypeError(
                "attach() needs a YoloWorldPerception (or an ultralytics "
                "YOLOWorld) — the object must expose .predict()."
            )
        self.model = wrapper
        if self.device == "cpu":
            self.device = getattr(perception, "device", None) or self.device

        inner = getattr(wrapper, "model", None)
        detection_model = inner if isinstance(inner, torch.nn.Module) else wrapper
        if not isinstance(detection_model, torch.nn.Module):
            raise TypeError("could not find the underlying nn.Module detection model")

        head = None
        for module in detection_model.modules():
            if type(module).__name__ == "WorldDetect":
                head = module  # keep the last match, as the SPPF hook does
        if head is None:
            raise RuntimeError(
                "No WorldDetect head found; this model has no open-vocabulary "
                "contrastive head to tap for text-space region embeddings."
            )
        cv4 = getattr(head, "cv4", None)
        if cv4 is None:
            raise RuntimeError(
                "WorldDetect has no 'cv4' branch (expected children "
                "['cv2', 'cv3', 'dfl', 'cv4']); the ultralytics head layout "
                "has changed and the text-space tap must be re-derived."
            )

        # attach() is idempotent: the detector is a SHARED frozen object (the v7
        # box path rides the same one), so re-attaching — a second extractor, a
        # per-episode rebuild — must not leave the previous run's hooks bolted
        # onto it. Handles are kept solely so they can be removed here.
        for handle in self._handles:
            handle.remove()
        self._handles = []

        levels = list(cv4)
        self._maps = [None] * len(levels)
        norms = [getattr(level, "norm", None) for level in levels]
        is_bn = any(isinstance(n, torch.nn.Module) for n in norms)
        self.head_kind = "bn" if is_bn else "cosine"
        for index, level in enumerate(levels):
            norm = norms[index]
            if isinstance(norm, torch.nn.Module):
                self._handles.append(norm.register_forward_hook(self._make_post_hook(index)))
            else:
                self._handles.append(
                    level.register_forward_pre_hook(self._make_pre_hook(index))
                )
                self._handles.append(level.register_forward_hook(self._make_post_hook(index)))
        self._handles.append(detection_model.register_forward_pre_hook(self._capture_input))

        # Adopt the head's own sharpness when the caller did not pin one: a
        # cosine-scoring head was trained with a learned logit_scale, and a
        # softmax at some other temperature reads its cosines more (or less)
        # sharply than the detector itself does. NOT for the BN variant: there
        # logit_scale multiplies an unnormalized BN'd dot product, so it is not
        # a cosine temperature at all — on yolov8s-worldv2.pt reading it as one
        # gives tau = 0.560 and a three-phrase softmax peak of ~0.35 against a
        # 0.333 floor, i.e. it would erase the language term from `weight`.
        if not self._temperature_is_explicit and levels and not is_bn:
            scale = getattr(levels[0], "logit_scale", None)
            if torch.is_tensor(scale):
                learned = float(scale.detach().float().reshape(-1)[0].exp().item())
                if math.isfinite(learned) and learned > 0.0:
                    self.match_temperature = 1.0 / learned

    def _make_pre_hook(self, index: int):
        """Forward pre-hook capturing cv4's input (the text-space embedding)."""

        def hook(_module, inputs) -> None:
            if inputs and torch.is_tensor(inputs[0]) and self._is_embedding(inputs[0]):
                self._maps[index] = inputs[0].detach()

        return hook

    def _make_post_hook(self, index: int):
        """Forward hook on the module that EMITS the embedding.

        On the BN variant that module is ``cv4[i].norm`` and this is the primary
        tap. On the cosine variant it is ``cv4[i]`` itself and this is a
        fallback that only fires when the pre-hook caught nothing — hence the
        ``is None`` guard, which is what keeps it from overwriting a good
        capture with the head's logits.
        """

        def hook(_module, _inputs, output) -> None:
            if self._maps[index] is None and torch.is_tensor(output) and self._is_embedding(output):
                self._maps[index] = output.detach()

        return hook

    def _is_embedding(self, tensor: torch.Tensor) -> bool:
        """True for a ``[B, vis_dim, H, W]`` per-anchor embedding map."""
        return tensor.dim() == 4 and int(tensor.shape[1]) == int(self.cfg.vis_dim)

    def _capture_input(self, _module, inputs) -> None:
        """Records the letterboxed network input size for the box mapping."""
        if inputs and torch.is_tensor(inputs[0]) and inputs[0].dim() == 4:
            self._input_hw = (int(inputs[0].shape[-2]), int(inputs[0].shape[-1]))

    def extract(
        self, frame_bgr: "np.ndarray", text_feats: Optional[torch.Tensor] = None
    ) -> RegionObs:
        """Detects on one frame and returns its top-K text-space proposals.

        Args:
            frame_bgr: ``HxWx3`` uint8 BGR frame (ultralytics' native layout).
            text_feats: Per-call phrase features; ``None`` uses the task's.

        Returns:
            ``RegionObs`` with ``cfg.max_objects`` rows; an empty detection set
            yields an all-pad observation rather than an error.

        Raises:
            RuntimeError: If :meth:`attach` has not run, or the cv4 taps
                captured no embedding map (the head's contract moved).
        """
        if self.model is None:
            raise RuntimeError("attach() a loaded YOLO-World model before extract()")
        frame = frame_bgr
        short = min(frame.shape[0], frame.shape[1])
        if short < self.min_side:
            import cv2  # lazy: present wherever the real detector runs

            scale = self.min_side / short
            frame = cv2.resize(
                frame,
                (round(frame.shape[1] * scale), round(frame.shape[0] * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        with torch.no_grad():
            self._maps = [None] * len(self._maps)
            # agnostic_nms: two prompts firing on one object must not spend two
            # of the K proposal slots on it. half=False forces fp32 for the same
            # reason yolo_world.py does (ROCm half kernels segfault mid-detect).
            results = self.model.predict(
                frame,
                device=self.device,
                conf=self.det_conf,
                half=False,
                verbose=False,
                agnostic_nms=True,
            )

        # Clone AFTER predict returned: ultralytics runs it under
        # torch.inference_mode(), and an inference tensor cannot be saved for
        # backward — that is the crash the TQSA hit on the SPPF map.
        maps = [m.float().cpu().clone() for m in self._maps if m is not None]
        if not maps:
            raise RuntimeError(
                "The cv4 taps captured no [1, vis_dim, H, W] embedding map. "
                "Either the forward did not run, or this ultralytics head no "
                "longer exposes per-anchor embeddings at cv4."
            )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            xyxy = torch.zeros(0, 4, dtype=torch.float32)
            conf = torch.zeros(0, dtype=torch.float32)
        else:
            xyxy = boxes.xyxy.detach().float().cpu()
            conf = boxes.conf.detach().float().cpu()

        return self.regions_from(
            region_maps=maps,
            boxes_xyxy=xyxy,
            objectness=conf,
            frame_hw=(int(frame.shape[0]), int(frame.shape[1])),
            input_hw=self._input_hw,
            text_feats=text_feats,
        )


class MockTextRegionExtractor(_RegionAssembly):
    """Deterministic pseudo-proposals — same API, no model, no downloads.

    Every output is a pure function of the frame bytes: their SHA-256 digest
    seeds a local ``torch.Generator`` (the ``MockYoloWorldPerception``
    construction) that draws the embeddings and each proposal's orbit
    parameters, so identical frames reproduce identical proposals bit for bit.
    Proposals are spread around a common circle by index and given their own
    radii, so no two of them land on top of each other and a downstream
    relational head sees a genuine spatial arrangement. No global RNG state is
    touched.

    The proposal COUNT also varies with the frame (1 .. ``max_objects + 2``) so
    the padding and the top-K truncation branches are both exercised by ordinary
    mock use rather than only by a test that reaches for them. Scoring, ranking,
    standardization and padding are inherited verbatim from
    :class:`_RegionAssembly`, so this mock cannot disagree with the real
    extractor about anything except which regions exist.

    Args:
        cfg: Canonical config; supplies ``vis_dim`` and ``max_objects``.
        n_proposals: Force a fixed proposal count (``0`` for a blind frame)
            instead of deriving it from the frame; a test lever.
        match_temperature: See :class:`_RegionAssembly`.
        pool_samples: See :class:`_RegionAssembly` (only used by the inherited
            :meth:`regions_from`).
    """

    #: Frame size assumed when a degenerate (non-2D) frame is supplied (W, H).
    _DEFAULT_WH: tuple[int, int] = (640, 480)

    def __init__(
        self,
        cfg: MicroVLAConfig,
        n_proposals: Optional[int] = None,
        match_temperature: Optional[float] = None,
        pool_samples: int = 3,
    ) -> None:
        super().__init__(cfg, match_temperature=match_temperature, pool_samples=pool_samples)
        self.n_proposals = n_proposals
        self.attached = False

    def attach(self, perception: object) -> None:
        """Records that attachment happened (mock analogue of the real hooks).

        Args:
            perception: Ignored; kept for API parity so the loop can attach
                unconditionally.
        """
        self.attached = True

    def extract(
        self, frame_bgr: "np.ndarray", text_feats: Optional[torch.Tensor] = None
    ) -> RegionObs:
        """Produces deterministic pseudo-proposals for one frame.

        Args:
            frame_bgr: ``HxWx3`` uint8 BGR frame (any array-like; its bytes
                determine every output).
            text_feats: Per-call phrase features; ``None`` uses the task's.

        Returns:
            ``RegionObs`` with ``cfg.max_objects`` rows, matched and weighted by
            the same code the real extractor runs.
        """
        phrases = self._resolve_text(text_feats)
        frame = np.ascontiguousarray(frame_bgr)
        digest = hashlib.sha256(frame.tobytes()).digest()
        seed = int.from_bytes(digest[:8], byteorder="little")

        if self.n_proposals is None:
            n = 1 + digest[8] % (self.cfg.max_objects + 2)
        else:
            n = max(0, int(self.n_proposals))

        generator = torch.Generator()
        generator.manual_seed(seed)
        draws = max(n, 1)  # a generator must always be stepped the same way
        emb = torch.randn(
            draws, self.cfg.vis_dim, generator=generator, dtype=torch.float32
        )[:n]
        # Per proposal: orbit radius, phase jitter, objectness.
        aux = torch.rand(draws, 3, generator=generator, dtype=torch.float32)[:n]

        index = torch.arange(n, dtype=torch.float32)
        # Radius capped at 0.29 so centers stay inside [0.2, 0.8]; the index
        # term fans proposals around the circle so they never coincide.
        radius = 0.15 + 0.14 * aux[:, 0]
        theta = (2.0 * math.pi) * (index / max(n, 1) + 0.15 * aux[:, 1])
        centers = torch.stack(
            [
                (0.5 + radius * torch.cos(theta)).clamp(0.2, 0.8),
                (0.5 + radius * torch.sin(theta)).clamp(0.2, 0.8),
            ],
            dim=-1,
        )
        # Unsorted on purpose: the shared top-K ranking must do real work.
        objectness = 0.2 + 0.75 * aux[:, 2]

        return assemble_regions(
            emb=emb,
            objectness=objectness,
            center=centers,
            text_feats=phrases,
            vis_dim=self.cfg.vis_dim,
            max_objects=self.cfg.max_objects,
            temperature=self.match_temperature,
        )


if __name__ == "__main__":
    from microvla.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    text = F.normalize(torch.randn(cfg.n_text_tokens, cfg.vis_dim), dim=-1)

    mock = MockTextRegionExtractor(cfg)
    mock.set_text_features(text)
    frame = (np.arange(64 * 64 * 3, dtype=np.uint8) % 251).reshape(64, 64, 3)
    obs = mock.extract(frame)
    assert obs.emb.shape == (cfg.max_objects, cfg.vis_dim)
    assert obs.center.shape == (cfg.max_objects, 2)
    assert obs.weight.shape == (cfg.max_objects,)
    assert obs.match.shape == (cfg.max_objects, cfg.n_text_tokens)
    assert torch.equal(obs.weight, mock.extract(frame).weight), "mock is not deterministic"
    assert float(obs.weight.min()) >= 0.0 and float(obs.weight.max()) <= 1.0

    # Real extractor's assembly path, driven by a synthetic cv4 map (no model).
    real = TextRegionExtractor(cfg)
    real.set_text_features(text)
    maps = [torch.randn(1, cfg.vis_dim, h, h) for h in (16, 8, 4)]
    boxes = torch.tensor([[10.0, 10.0, 60.0, 70.0], [80.0, 40.0, 120.0, 90.0]])
    from_map = real.regions_from(maps, boxes, torch.tensor([0.4, 0.9]), frame_hw=(128, 128))
    assert from_map.n_valid == 2
    assert from_map.weight[0] > 0 and from_map.weight[2] == 0.0
    valid = from_map.emb[: from_map.n_valid]
    assert torch.allclose(valid.mean(-1), torch.zeros(2), atol=1e-5)
    assert torch.allclose(valid.std(-1, unbiased=False), torch.ones(2), atol=1e-3)

    blind = real.regions_from(maps, torch.zeros(0, 4), torch.zeros(0), frame_hw=(128, 128))
    assert blind.n_valid == 0 and float(blind.weight.abs().sum()) == 0.0

    params = [
        obj
        for holder in (mock, real)
        for obj in vars(holder).values()
        if isinstance(obj, torch.nn.Parameter)
    ]
    assert not params, "this is a frozen extraction path: it must own no parameters"
    print(
        f"mock: {obs.n_valid}/{cfg.max_objects} valid proposals, "
        f"weights [{float(obs.weight.min()):.3f}, {float(obs.weight.max()):.3f}], "
        f"match rows sum {float(obs.match[0].sum()):.3f} | "
        f"map path: {from_map.n_valid} valid, tau={real.match_temperature:.3f} | "
        f"trainable params: 0"
    )
