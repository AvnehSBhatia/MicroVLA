"""BridgeData V2 RLDS/TFDS shard reader -> MicroVLA episodes.

The raw Bridge release only ships as monolithic zips (441 GB / 32 GB) that can
never fit the 10 GB disk budget. The TFDS (RLDS) release, however, is sharded:

    .../data/tfds/bridge_dataset/1.0.0/
        bridge_dataset-train.tfrecord-00000-of-01024   (~110 MB each)
        ...
        features.json  dataset_info.json               (tiny, downloaded once)

Each shard holds ~60 full episodes, so the shard pipeline can stream them one
at a time (download -> convert -> delete) far under the budget. RLDS schema
(per episode): ``steps/observation/image_0`` (256x256x3 JPEG-encoded frames,
primary camera), ``steps/action`` ([T, 7]: Δxyz, Δrpy, gripper),
``steps/language_instruction`` (bytes; often empty — such episodes are
skipped by default).

Deserialization uses ``tensorflow`` + ``tensorflow_datasets`` (the shard
pipeline environment installs them; the core package never imports TF).
``features.json`` provides the exact feature spec, so parsing tracks the
upstream schema instead of hardcoding proto keys.

Standalone usage mirrors the other converters:

    python -m preprocess.shard_pipeline bridge_shards.txt data/bridge \\
        --dataset bridge_rlds --budget-gb 6 --device mps
    # reader_kwargs features_dir defaults to <out>/_rlds_meta; fetch
    # features.json + dataset_info.json there first (tiny files).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from preprocess.common import SourceEpisode

logger = logging.getLogger(__name__)

#: BridgeData V2 control / camera rate (RLDS release, same as raw).
BRIDGE_HZ = 5.0

_FEATURES = None  # cached tfds FeatureConnector (one per process)


def _load_features(features_dir: Path):
    """Loads (and caches) the TFDS feature spec from ``features.json``."""
    global _FEATURES
    if _FEATURES is None:
        from tensorflow_datasets.core.features import FeatureConnector

        path = Path(features_dir) / "features.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download features.json (and "
                "dataset_info.json) from the bridge_dataset/1.0.0/ directory "
                "into this folder once — they are a few KB."
            )
        _FEATURES = FeatureConnector.from_config(str(features_dir))
    return _FEATURES


def iter_bridge_rlds_episodes(
    root: str | Path,
    features_dir: str | Path,
    camera: str = "image_0",
    require_lang: bool = True,
) -> Iterator[SourceEpisode]:
    """Streams episodes from RLDS tfrecord shard file(s).

    Args:
        root: A single ``*.tfrecord-*`` shard file, or a directory of them.
        features_dir: Directory containing the dataset's ``features.json``.
        camera: Observation image key (``image_0`` = primary).
        require_lang: Skip episodes with an empty language instruction
            (MicroVLA is language-conditioned end to end).

    Yields:
        One :class:`SourceEpisode` per usable episode (frames RGB uint8,
        actions ``[T, 7]`` float32).

    Raises:
        FileNotFoundError: If no shard files are found.
    """
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # decode on CPU, quietly

    root = Path(root)
    shards = [root] if root.is_file() else sorted(root.glob("*.tfrecord-*"))
    if not shards:
        raise FileNotFoundError(f"no tfrecord shards under {root}")
    features = _load_features(Path(features_dir))

    skipped_lang = 0
    for shard in shards:
        shard_tag = shard.name.split(".")[-1]  # e.g. tfrecord-00007-of-01024
        for i, record in enumerate(tf.data.TFRecordDataset(str(shard))):
            ep = features.deserialize_example(record)
            steps = ep["steps"]

            # ``steps`` is a tf.data dataset of step dicts. Peek the first
            # step's language annotation before decoding a whole episode's
            # JPEGs for nothing.
            first_step = next(iter(steps.take(1)), None)
            if first_step is None:
                continue
            lang = (
                first_step["language_instruction"].numpy()
                .decode("utf-8", errors="ignore").strip()
            )
            if lang == "" and require_lang:
                skipped_lang += 1
                continue

            frames: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            for step in steps:
                obs = step["observation"]
                frames.append(np.asarray(obs[camera].numpy(), dtype=np.uint8))  # RGB
                actions.append(np.asarray(step["action"].numpy(), dtype=np.float32))
                if lang == "":
                    lang = (
                        step["language_instruction"].numpy()
                        .decode("utf-8", errors="ignore").strip()
                    )

            if len(frames) < 2 or not actions:
                continue
            action_arr = np.stack(actions, axis=0)
            if action_arr.shape[-1] != 7:
                continue
            if lang == "":
                skipped_lang += 1
                continue

            yield SourceEpisode(
                frames=frames,
                actions=action_arr,
                instruction=lang.lower(),
                source_hz=BRIDGE_HZ,
                episode_id=f"{shard_tag}__ep{i}",
            )

    if skipped_lang:
        logger.info("skipped %d episodes without language annotation", skipped_lang)
