"""Fixed-rate frame sampler over video files, camera devices, or frame iterables.

Implements the ``VideoStreamSampler`` contract from ``DESIGN.md``: frames are
drained from the source in order and one is emitted whenever its timestamp
crosses the next scheduled emit time (``t >= next_emit_time``), after which the
schedule advances by ``1 / target_fps``. This keeps the output cadence at
``target_fps`` regardless of the source frame rate.

``cv2`` is imported lazily inside the iteration path so the package imports
with only ``torch`` + ``numpy`` installed.
"""

from __future__ import annotations

import math
import numbers
import os
from typing import Iterable, Iterator, Tuple

import numpy as np

from microvla.config import DEFAULT_CONFIG


class VideoStreamSampler:
    """Samples a video source down to a fixed target frame rate.

    The source may be anything ``cv2.VideoCapture`` accepts (a file path,
    URL string, or integer camera index) or any Python iterable yielding
    either plain frames or ``(frame, timestamp)`` pairs. For iterables that
    yield plain frames, timestamps are synthesized as
    ``t = index / assumed_source_fps``.

    Sampling rule (from DESIGN.md): a frame is emitted when its timestamp
    ``t`` satisfies ``t >= next_emit_time``; the schedule then advances by
    ``1 / target_fps``. The first frame (``t == 0.0``) is always emitted.

    Args:
        source: A path/URL string, ``os.PathLike``, or camera index for
            ``cv2.VideoCapture``; otherwise any iterable of frames or of
            ``(frame, timestamp)`` pairs.
        target_fps: Output sampling rate in frames per second.
        assumed_source_fps: Frame rate assumed for sources that provide no
            timestamps (plain-frame iterables, and cv2 sources that report
            an invalid FPS).
    """

    def __init__(
        self,
        source,
        target_fps: float = DEFAULT_CONFIG.real_frame_hz,
        assumed_source_fps: float = 30.0,
    ) -> None:
        if target_fps <= 0.0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if assumed_source_fps <= 0.0:
            raise ValueError(
                f"assumed_source_fps must be positive, got {assumed_source_fps}"
            )
        self.source = source
        self.target_fps = float(target_fps)
        self.assumed_source_fps = float(assumed_source_fps)
        # str / PathLike / int (camera index) go through cv2.VideoCapture;
        # anything else must be iterable. Note: bool is an int subclass but is
        # still a valid (if odd) camera index, so no special-casing is needed.
        self._is_cv2_source = isinstance(source, (str, os.PathLike, int))
        if not self._is_cv2_source and not hasattr(source, "__iter__"):
            raise TypeError(
                "source must be a path/int for cv2.VideoCapture or an iterable "
                f"of frames, got {type(source).__name__}"
            )

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
        """Yields ``(frame_bgr, t)`` pairs sampled at ``target_fps``.

        Yields:
            Tuples of (``np.ndarray`` HxWx3 uint8 BGR frame, timestamp in
            seconds as ``float``).
        """
        if self._is_cv2_source:
            timestamped = self._iter_cv2()
        else:
            timestamped = self._iter_iterable(self.source)

        period = 1.0 / self.target_fps
        next_emit_time = 0.0
        for frame, t in timestamped:
            if t >= next_emit_time:
                yield frame, t
                next_emit_time += period

    # ------------------------------------------------------------------ #
    # Source adapters: normalize every source to a (frame, t) stream.    #
    # ------------------------------------------------------------------ #

    def _iter_cv2(self) -> Iterator[Tuple[np.ndarray, float]]:
        """Decodes frames via cv2.VideoCapture, timestamping by frame index.

        Timestamps are ``index / source_fps`` using the FPS reported by the
        container; ``assumed_source_fps`` is used when the report is missing
        or invalid (common for live cameras). Index-based timing is preferred
        over ``CAP_PROP_POS_MSEC``, which is unreliable across backends.
        """
        import cv2  # Lazy: heavy dep, only needed for cv2-openable sources.

        source = self.source
        if isinstance(source, os.PathLike):
            source = os.fspath(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"cv2.VideoCapture could not open source: {self.source!r}")
        try:
            reported_fps = cap.get(cv2.CAP_PROP_FPS)
            if (
                not isinstance(reported_fps, numbers.Real)
                or not math.isfinite(reported_fps)
                or reported_fps <= 0.0
            ):
                reported_fps = self.assumed_source_fps
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame, index / reported_fps
                index += 1
        finally:
            cap.release()

    def _iter_iterable(
        self, source: Iterable
    ) -> Iterator[Tuple[np.ndarray, float]]:
        """Adapts an iterable of frames / (frame, timestamp) pairs.

        An item counts as a pair when it is a 2-element tuple or list whose
        second element is a real number; everything else (notably bare
        ``np.ndarray`` frames) is treated as a plain frame and given the
        synthesized timestamp ``index / assumed_source_fps``.
        """
        for index, item in enumerate(source):
            if (
                isinstance(item, (tuple, list))
                and len(item) == 2
                and isinstance(item[1], numbers.Real)
            ):
                frame, t = item
                yield frame, float(t)
            else:
                yield item, index / self.assumed_source_fps
