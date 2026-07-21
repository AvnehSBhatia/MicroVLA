"""Offline dataset preprocessing: raw robot datasets -> MicroVLA .npz episodes.

Converters (nothing is downloaded — point them at local copies):
    python -m preprocess.libero <libero_root> <out_dir> [flags]
    python -m preprocess.bridge <bridge_root> <out_dir> [flags]

Both support ``--dry-run`` (mock perception, no weights), ``--limit``, and
teacher distillation via ``--teacher {mock,tinyvla}`` (see preprocess/teacher.py).
"""

from preprocess.common import (
    ActionNormalizer,
    EpisodeBuilder,
    SourceEpisode,
    chunk_actions,
    run_conversion,
    subsample_indices,
)
from preprocess.teacher import (
    CachedTeacher,
    MockTeacher,
    TeacherPolicy,
    TinyVLATeacher,
    build_teacher,
)

__all__ = [
    "ActionNormalizer",
    "EpisodeBuilder",
    "SourceEpisode",
    "chunk_actions",
    "run_conversion",
    "subsample_indices",
    "CachedTeacher",
    "MockTeacher",
    "TeacherPolicy",
    "TinyVLATeacher",
    "build_teacher",
]
