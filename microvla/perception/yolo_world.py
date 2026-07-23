"""Frozen YOLO-World-S open-vocabulary perception with dual-box grounding (v2).

``YoloWorldPerception`` runs a frozen YOLO-World-S detector whose active
class list is set from the parsed task (``[source]`` or ``[source, target]``),
and extracts, per frame, from a hooked SPPF (P5) feature map:

* ``frame_emb`` -- global average pool (GAP) of the hooked map, ``[vis_dim]``.
* ``source`` / ``target`` -- per-class best-box ``BoxObs`` (ROIAlign of the
  hooked map inside the highest-confidence detection of that class, 7x7 then
  GAP, plus its normalized center and pixel ``xyxy``).

``ultralytics`` and ``torchvision`` are imported lazily in ``__init__`` so the
package imports with only ``torch`` + ``numpy``. ``MockYoloWorldPerception``
provides the same API deterministically for tests, with no downloads.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from microvla.utils.embedding import standardize


@dataclass
class BoxObs:
    """A single grounded box observation (detached, CPU, float32).

    Attributes:
        emb: ``[vis_dim]`` float32 ROIAlign-GAP embedding inside the box (a
            clone of the frame embedding when nothing was detected).
        center: ``[2]`` float32 ``(cx, cy)`` normalized to ``[0, 1]``
            (``(0.5, 0.5)`` when nothing was detected).
        xyxy: ``[4]`` float32 pixel coordinates in the original frame (zeros
            when nothing was detected).
        confidence: Detection confidence; ``0.0`` if this is a fallback.
    """

    emb: torch.Tensor
    center: torch.Tensor
    xyxy: torch.Tensor
    confidence: float


@dataclass
class Perception:
    """Per-frame perception outputs: a frame embedding plus two grounded boxes.

    Attributes:
        frame_emb: ``[vis_dim]`` float32 GAP of the hooked SPPF map.
        source: Best-box observation for the SOURCE class.
        target: Best-box observation for the TARGET class (shares ``source``'s
            ``BoxObs`` when only one active class was set, i.e. source == target).
    """

    frame_emb: torch.Tensor
    source: BoxObs
    target: BoxObs


class YoloWorldPerception:
    """Frozen YOLO-World-S wrapper exposing SPPF-map embeddings per class.

    A forward hook is registered on the SPPF module (found by walking the
    model's modules and matching the class name ``"SPPF"`` -- never by index)
    to capture the 512-channel P5 map each forward pass. A forward pre-hook
    on the detection model records the actual network input size so ROIAlign
    boxes can be mapped from original-frame pixels into feature-map
    coordinates with the correct spatial scale.

    Args:
        weights: Ultralytics weights name or path for ``YOLOWorld``.
        device: Torch device string for inference.
        det_conf: Detection confidence threshold. Deliberately low (0.10):
            open-vocab phrase prompts score weaker than closed-set classes,
            and downstream consumers weight every box by its confidence
            (``box_weights``), so admitting weak evidence is safe while
            discarding it is not recoverable.
        min_side: Frames whose short side is below this are bicubically
            upscaled before detection. Dataset frames are often tiny
            (LIBERO 128px, Bridge 256px) and the detector is starved at
            native size — measured on LIBERO: basket 0.00 -> 0.57 confidence
            at 4x. Returned ``xyxy`` are in the (possibly upscaled) frame's
            pixels; normalized centers are unaffected.

    Raises:
        RuntimeError: If no SPPF module exists in the loaded model.
    """

    def __init__(self, weights: str = "yolov8s-worldv2.pt", device: str = "cpu",
                 det_conf: float = 0.10, min_side: int = 512) -> None:
        self.det_conf = det_conf
        self.min_side = min_side
        # Lazy imports: ultralytics + torchvision are heavy optional deps.
        from torchvision.ops import roi_align
        from ultralytics import YOLOWorld

        # Ultralytics logs a WARNING via its own logger on every predict()
        # because we pass half=False (deprecated flag, but the only way to force
        # fp32 on ROCm where half kernels segfault). Silence that logger so a
        # long eval isn't drowned in identical warnings — it hides the summary.
        import logging
        logging.getLogger("ultralytics").setLevel(logging.ERROR)

        self._roi_align = roi_align
        self.device = device
        # Public: ClipTaskEncoder reaches through this to call
        # `model.set_classes(...)` directly and read the CLIP text tower's
        # `txt_feats` (see perception/text_encoder.py).
        self.model = YOLOWorld(weights)
        self.model.to(device)

        # Runtime capture slots filled by the hooks on every forward pass.
        self._feat: Optional[torch.Tensor] = None  # [1, C, Hf, Wf] SPPF map
        self._input_hw: Optional[Tuple[int, int]] = None  # network (H_in, W_in)
        # Ordered ACTIVE detection classes; role i == class id i.
        self._active_classes: list[str] = []

        detection_model = self.model.model  # underlying nn.Module
        for p in detection_model.parameters():
            p.requires_grad_(False)
        detection_model.eval()

        sppf = None
        for module in detection_model.modules():
            if type(module).__name__ == "SPPF":
                sppf = module  # keep last match = last backbone SPPF (P5)
        if sppf is None:
            raise RuntimeError(
                "No SPPF module found in the YOLO-World model; cannot hook the "
                "P5 feature map."
            )

        def _capture_feat(_module, _inputs, output) -> None:
            self._feat = output.detach()

        def _capture_input(_module, inputs) -> None:
            if inputs and torch.is_tensor(inputs[0]) and inputs[0].dim() == 4:
                self._input_hw = (int(inputs[0].shape[-2]), int(inputs[0].shape[-1]))

        sppf.register_forward_hook(_capture_feat)
        detection_model.register_forward_pre_hook(_capture_input)

    def set_classes(self, classes: list[str]) -> None:
        """Sets the ordered ACTIVE open-vocabulary detection classes.

        Args:
            classes: ``[source]`` (source == target) or ``[source, target]``,
                ordered so detection class id ``i`` corresponds to role ``i``.
        """
        with torch.no_grad():
            self.model.set_classes(list(classes))
        self._active_classes = list(classes)

    def perceive(self, frame_bgr: "np.ndarray") -> Perception:
        """Runs detection on one frame and extracts per-class SPPF-map embeddings.

        Args:
            frame_bgr: ``HxWx3`` uint8 BGR frame (OpenCV convention, as
                consumed natively by ultralytics).

        Returns:
            ``Perception`` with detached CPU float32 tensors. For each active
            class, the highest-confidence detection of that class id is used;
            a missing class falls back to ``BoxObs(emb=frame_emb.clone(),
            center=(0.5, 0.5), xyxy=zeros, confidence=0.0)``. When only one
            class is active (source == target), both roles share the same
            ``BoxObs``.
        """
        short = min(frame_bgr.shape[0], frame_bgr.shape[1])
        if short < self.min_side:
            import cv2  # lazy: present wherever the real detector runs

            scale = self.min_side / short
            frame_bgr = cv2.resize(
                frame_bgr,
                (round(frame_bgr.shape[1] * scale), round(frame_bgr.shape[0] * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        with torch.no_grad():
            self._feat = None
            # half=False forces fp32 inference. Ultralytics defaults to
            # half-precision on GPU, and ROCm/HIP half kernels are a common
            # cause of a hard segfault mid-detection on AMD GPUs (MI300X);
            # fp32 is safe and the detector is not the compute bottleneck.
            results = self.model.predict(
                frame_bgr, device=self.device, conf=self.det_conf, half=False, verbose=False
            )

            if self._feat is None:
                raise RuntimeError(
                    "SPPF hook captured no feature map; the model forward did "
                    "not run as expected."
                )
            # Pull the map to CPU float32 immediately: ROIAlign/pooling run on
            # CPU regardless of detector device (torchvision MPS kernel
            # coverage is incomplete, and the map is tiny).
            feat = self._feat.float().cpu()  # [1, C, Hf, Wf]
            # GAP -> [C], then standardized: ALL embeddings leaving perception
            # live in the canonical zero-mean/unit-std space (see
            # microvla/utils/embedding.py) so TRM feedback stays in-distribution
            # and losses/innovations are scale-honest.
            frame_emb = standardize(
                feat.mean(dim=(2, 3)).squeeze(0).detach().cpu().float()
            )

            frame_h, frame_w = frame_bgr.shape[:2]
            best_by_class: dict[int, tuple[float, torch.Tensor]] = {}
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                cls_ids = boxes.cls
                confs = boxes.conf
                xyxys = boxes.xyxy
                for i in range(len(boxes)):
                    cid = int(cls_ids[i].item())
                    conf = float(confs[i].item())
                    prev = best_by_class.get(cid)
                    if prev is None or conf > prev[0]:
                        best_by_class[cid] = (conf, xyxys[i].detach().float())

            def _fallback() -> BoxObs:
                return BoxObs(
                    emb=frame_emb.clone(),
                    center=torch.tensor([0.5, 0.5], dtype=torch.float32),
                    xyxy=torch.zeros(4, dtype=torch.float32),
                    confidence=0.0,
                )

            def _box_for_class(cid: int) -> BoxObs:
                entry = best_by_class.get(cid)
                if entry is None:
                    return _fallback()
                conf, box_xyxy = entry
                feat_box = self._map_box_to_feature(box_xyxy, frame_h, frame_w, feat)
                rois = torch.cat(
                    [torch.zeros(1, device=feat.device, dtype=feat.dtype), feat_box]
                ).unsqueeze(0)  # [1, 5] = (batch_idx, x1, y1, x2, y2) in feature coords
                pooled = self._roi_align(
                    feat, rois, output_size=(7, 7), spatial_scale=1.0, aligned=True
                )  # [1, C, 7, 7]
                box_emb = pooled.mean(dim=(2, 3)).squeeze(0)  # GAP -> [C]

                cx = float((box_xyxy[0] + box_xyxy[2]) * 0.5 / frame_w)
                cy = float((box_xyxy[1] + box_xyxy[3]) * 0.5 / frame_h)
                center = torch.tensor(
                    [min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)],
                    dtype=torch.float32,
                )
                return BoxObs(
                    emb=standardize(box_emb.detach().cpu().float()),
                    center=center,
                    xyxy=box_xyxy.detach().cpu().float(),
                    confidence=conf,
                )

            n_active = len(self._active_classes)
            if n_active >= 2:
                source = _box_for_class(0)
                target = _box_for_class(1)
            elif n_active == 1:
                box = _box_for_class(0)
                source = box
                target = box
            else:
                fallback = _fallback()
                source = fallback
                target = fallback

            return Perception(frame_emb=frame_emb, source=source, target=target)

    def _map_box_to_feature(
        self,
        box_xyxy: torch.Tensor,
        frame_h: int,
        frame_w: int,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        """Maps a box from original-frame pixels to hooked-map coordinates.

        Ultralytics letterboxes the frame to the network input (uniform scale
        ``r = min(H_in/H0, W_in/W0)`` plus centered padding), and the hooked
        SPPF map downsamples that input by a uniform stride. Both transforms
        are reconstructed from the *actual* recorded input size and the
        *actual* hooked map size, giving the correct spatial scale without
        hardcoding a stride. If the input size was somehow not captured, a
        direct per-axis frame->map scaling is used as a fallback.

        Args:
            box_xyxy: ``[4]`` box in original-frame pixel coordinates.
            frame_h: Original frame height in pixels.
            frame_w: Original frame width in pixels.
            feat: ``[1, C, Hf, Wf]`` hooked feature map.

        Returns:
            ``[4]`` box in feature-map coordinates (same device/dtype as
            ``feat``).
        """
        feat_h, feat_w = int(feat.shape[-2]), int(feat.shape[-1])
        box = box_xyxy.to(device=feat.device, dtype=feat.dtype)

        if self._input_hw is not None:
            in_h, in_w = self._input_hw
            r = min(in_h / frame_h, in_w / frame_w)
            pad_w = (in_w - frame_w * r) * 0.5
            pad_h = (in_h - frame_h * r) * 0.5
            # frame pixels -> letterboxed input pixels -> feature-map cells.
            sx = feat_w / in_w
            sy = feat_h / in_h
            x1 = (box[0] * r + pad_w) * sx
            y1 = (box[1] * r + pad_h) * sy
            x2 = (box[2] * r + pad_w) * sx
            y2 = (box[3] * r + pad_h) * sy
        else:  # pragma: no cover - defensive fallback
            x1 = box[0] * (feat_w / frame_w)
            y1 = box[1] * (feat_h / frame_h)
            x2 = box[2] * (feat_w / frame_w)
            y2 = box[3] * (feat_h / frame_h)

        out = torch.stack([x1, y1, x2, y2])
        out[0::2] = out[0::2].clamp(0.0, float(feat_w))
        out[1::2] = out[1::2].clamp(0.0, float(feat_h))
        return out


class MockYoloWorldPerception:
    """Deterministic pseudo-perception for tests -- same API, no model download.

    Every output is a pure function of the frame contents: the SHA-256 digest
    of the frame bytes seeds a local ``torch.Generator`` for the embeddings and
    parameterizes two distinct circular "orbits" -- one for the source box,
    one for the target box, with different radius/phase/angular-speed so they
    move differently -- both centered in ``[0.2, 0.8]`` normalized frame
    coordinates. Identical frames always reproduce identical outputs; a
    sequence of distinct frames yields boxes that appear to move smoothly
    along their respective orbits. No global RNG state is touched.

    Args:
        vis_dim: Embedding width, matching ``MicroVLAConfig.vis_dim``.
    """

    #: Frame size assumed when a non-ndarray frame is supplied (W, H).
    _DEFAULT_WH: Tuple[int, int] = (640, 480)

    def __init__(self, vis_dim: int = 512) -> None:
        self.vis_dim = vis_dim
        self.active_classes: list[str] = []

    def set_classes(self, classes: list[str]) -> None:
        """Records the active class list (mock analogue of the real API).

        Args:
            classes: ``[source]`` or ``[source, target]``.
        """
        self.active_classes = list(classes)

    def perceive(self, frame_bgr: "np.ndarray") -> Perception:
        """Produces two deterministic, distinctly-orbiting pseudo-detections.

        Args:
            frame_bgr: ``HxWx3`` uint8 BGR frame (any array-like works; its
                bytes determine every output).

        Returns:
            ``Perception`` with hash-seeded N(0, 1) embeddings, source/target
            box centers each in ``[0.2, 0.8]`` normalized coordinates,
            ``xyxy`` consistent with those centers on the actual frame size
            (640x480 by default), and ``confidence = 0.9`` for both.
        """
        frame = np.ascontiguousarray(frame_bgr)
        if frame.ndim >= 2:
            frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
        else:  # pragma: no cover - degenerate input
            frame_w, frame_h = self._DEFAULT_WH

        digest = hashlib.sha256(frame.tobytes()).digest()
        seed = int.from_bytes(digest[:8], byteorder="little")
        generator = torch.Generator()
        generator.manual_seed(seed)
        embs = torch.randn(3 * self.vis_dim, generator=generator, dtype=torch.float32)
        # Standardized like the real perception's outputs (canonical space).
        frame_emb = standardize(embs[: self.vis_dim].contiguous())
        source_emb = standardize(embs[self.vis_dim : 2 * self.vis_dim].contiguous())
        target_emb = standardize(embs[2 * self.vis_dim :].contiguous())

        def _unit(lo: int) -> float:
            """Uniform [0, 1) value from two digest bytes at offset ``lo``."""
            return int.from_bytes(digest[lo : lo + 2], "little") / 65536.0

        # Two circular orbits around the frame center (0.5, 0.5); radius is
        # capped at 0.29 so cx, cy always land inside [0.2, 0.8] (0.5 +/- 0.29
        # ~= [0.21, 0.79]). Different phase, radius, and angular speed for
        # the target orbit make it move visibly differently from the source.
        phase = _unit(16) * 2.0 * math.pi
        source_radius = 0.15 + 0.14 * _unit(18)  # [0.15, 0.29)
        target_radius = 0.15 + 0.14 * _unit(20)
        target_phase_offset = math.pi + _unit(22) * math.pi  # roughly opposite side
        target_freq = 1.3 + _unit(0)  # different angular speed => distinct orbit

        source_theta = phase
        target_theta = phase * target_freq + target_phase_offset

        def _box(theta: float, radius: float, emb: torch.Tensor, half_lo: int) -> BoxObs:
            cx = min(max(0.5 + radius * math.cos(theta), 0.2), 0.8)
            cy = min(max(0.5 + radius * math.sin(theta), 0.2), 0.8)
            half_w = 0.05 + 0.05 * _unit(half_lo)
            half_h = 0.05 + 0.05 * _unit(half_lo + 2)
            xyxy = torch.tensor(
                [
                    (cx - half_w) * frame_w,
                    (cy - half_h) * frame_h,
                    (cx + half_w) * frame_w,
                    (cy + half_h) * frame_h,
                ],
                dtype=torch.float32,
            )
            center = torch.tensor([cx, cy], dtype=torch.float32)
            return BoxObs(emb=emb, center=center, xyxy=xyxy, confidence=0.9)

        source = _box(source_theta, source_radius, source_emb, half_lo=24)
        target = _box(target_theta, target_radius, target_emb, half_lo=28)

        return Perception(frame_emb=frame_emb, source=source, target=target)
