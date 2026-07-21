"""Perception subpackage: video sampling, command parsing, text encoding,
and YOLO-World taps (v2).

Heavy dependencies (cv2, ultralytics, torchvision) are imported lazily inside
the classes that need them, so importing this package requires only
``torch`` + ``numpy``. MiniLM is gone in v2 -- task text is encoded via
YOLO-World's own CLIP text tower (``ClipTaskEncoder``).
"""

from microvla.perception.command_parser import ParsedCommand, parse_command, strip_article
from microvla.perception.text_encoder import (
    ClipTaskEncoder,
    MockTaskEncoder,
    TaskEncoding,
)
from microvla.perception.video_stream import VideoStreamSampler
from microvla.perception.yolo_world import (
    BoxObs,
    MockYoloWorldPerception,
    Perception,
    YoloWorldPerception,
)

__all__ = [
    "VideoStreamSampler",
    "parse_command",
    "strip_article",
    "ParsedCommand",
    "TaskEncoding",
    "ClipTaskEncoder",
    "MockTaskEncoder",
    "BoxObs",
    "Perception",
    "YoloWorldPerception",
    "MockYoloWorldPerception",
]
