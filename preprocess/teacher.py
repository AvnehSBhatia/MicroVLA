"""Teacher-policy distillation hooks: relabel episode actions with a large VLA.

Motivation: MicroVLA's planner is trained by behavior cloning on
``pwm_targets``. Those targets can come from the dataset's human
teleoperation (default) or from a LARGER pretrained VLA acting as a teacher —
knowledge distillation. A teacher gives (a) denoised, consistent action labels,
(b) labels for frames the human data covers badly, and (c) a path to DAgger-style
relabeling later (let the student drive, ask the teacher what it would have done).

The supported teacher here is **TinyVLA** (https://tiny-vla.github.io/) — a
compact VLA family with a diffusion-policy action head that natively emits
*action chunks*, which map 1:1 onto MicroVLA's ``plan_steps``-row plan windows.

Nothing is downloaded by this module. ``TinyVLATeacher`` lazily imports the
TinyVLA repository (which you clone yourself) and loads a checkpoint you
provide; ``MockTeacher`` is a deterministic stand-in for tests and dry-runs.

Usage (from either dataset converter):

    python -m preprocess.libero <root> <out> --teacher tinyvla \\
        --teacher-checkpoint /path/to/tinyvla.ckpt --teacher-repo /path/to/TinyVLA \\
        --teacher-cache ./teacher_cache

The cache directory stores relabeled action arrays per episode id, so the
(expensive) teacher forward pass runs once even though conversion streams the
dataset twice (stats pass + write pass).
"""

from __future__ import annotations

import abc
import hashlib
import logging
from pathlib import Path

import numpy as np

from preprocess.common import SourceEpisode

logger = logging.getLogger(__name__)


class TeacherPolicy(abc.ABC):
    """Relabels a raw episode's actions with a pretrained teacher policy."""

    #: How many native frames each teacher query's predicted chunk covers.
    chunk_len: int = 5

    @abc.abstractmethod
    def predict_chunk(self, frame_rgb: np.ndarray, instruction: str,
                      state: np.ndarray | None = None) -> np.ndarray:
        """Predicts one action chunk from a single observation.

        Args:
            frame_rgb: ``[H, W, 3]`` uint8 RGB frame.
            instruction: Natural-language task string.
            state: Optional robot state vector for state-conditioned teachers
                (v7: the converter passes per-frame ``proprio_raw`` when the
                dataset carries it; mock ignores it).

        Returns:
            ``[chunk_len, 7]`` float32 action chunk in the DATASET's raw
            action convention (Δpos xyz, Δrot rpy, gripper) — normalization
            happens downstream with the same quantile stats as everything
            else.
        """

    def relabel(self, episode: SourceEpisode) -> np.ndarray:
        """Produces a full ``[T_raw, 7]`` relabeled action array.

        Queries the teacher every ``chunk_len`` native frames and tiles each
        predicted chunk over the frames it covers (receding-horizon
        execution, exactly how chunked policies are deployed).

        Args:
            episode: The raw demonstration (frames + instruction used;
                original actions ignored).

        Returns:
            ``[T_raw, 7]`` float32 teacher actions.
        """
        T = len(episode.frames)
        out = np.zeros((T, 7), dtype=np.float32)
        proprio = getattr(episode, "proprio_raw", None)
        for start in range(0, T, self.chunk_len):
            state = proprio[start] if proprio is not None and start < len(proprio) else None
            chunk = np.asarray(
                self.predict_chunk(np.ascontiguousarray(episode.frames[start]),
                                   episode.instruction, state=state),
                dtype=np.float32,
            )
            end = min(start + self.chunk_len, T)
            out[start:end] = chunk[: end - start]
        return out


class CachedTeacher(TeacherPolicy):
    """Wraps any teacher with an on-disk per-episode cache.

    Conversion streams the dataset twice; this makes the teacher pay for each
    episode once. Cache files are ``<cache_dir>/<episode_id>.npy``.
    """

    def __init__(self, inner: TeacherPolicy, cache_dir: str | Path) -> None:
        self.inner = inner
        self.chunk_len = inner.chunk_len
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def predict_chunk(self, frame_rgb: np.ndarray, instruction: str,
                      state: np.ndarray | None = None) -> np.ndarray:
        return self.inner.predict_chunk(frame_rgb, instruction, state=state)

    def relabel(self, episode: SourceEpisode) -> np.ndarray:
        path = self.cache_dir / f"{episode.episode_id}.npy"
        if path.exists():
            return np.load(path)
        actions = self.inner.relabel(episode)
        np.save(path, actions)
        return actions


class MockTeacher(TeacherPolicy):
    """Deterministic pseudo-teacher for tests and ``--dry-run`` (no model).

    Each (frame bytes, instruction) pair hashes to a smooth, bounded action
    chunk. No global RNG state is touched.
    """

    def __init__(self, chunk_len: int = 5) -> None:
        self.chunk_len = chunk_len

    def predict_chunk(self, frame_rgb: np.ndarray, instruction: str,
                      state: np.ndarray | None = None) -> np.ndarray:
        digest = hashlib.sha256(
            np.ascontiguousarray(frame_rgb).tobytes() + instruction.encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        base = rng.uniform(-0.5, 0.5, size=(1, 7))
        drift = rng.uniform(-0.05, 0.05, size=(self.chunk_len, 7)).cumsum(axis=0)
        return np.clip(base + drift, -1.0, 1.0).astype(np.float32)


class TinyVLATeacher(TeacherPolicy):
    """TinyVLA (https://tiny-vla.github.io/) as a distillation teacher.

    TinyVLA = compact multimodal backbone + diffusion-policy head emitting
    action chunks. This adapter queries it per observation and returns the
    predicted chunk in the dataset's raw action convention.

    Setup (nothing is downloaded automatically):
        1. Clone the TinyVLA repository and install its requirements into a
           Python environment that also has this repo on the path.
        2. Download/obtain a TinyVLA checkpoint finetuned (or pretrained) on
           an action space compatible with 7-DoF (Δpos, Δrot, gripper) —
           their Bridge/franka configurations both qualify.
        3. Pass ``repo_path`` (the clone) and ``checkpoint`` here.

    IMPORTANT — embodiment gap: a teacher trained on a different robot than
    your target rig produces actions in ITS convention. For BridgeData V2 the
    convention matches (WidowX, 7-DoF delta EEF); for your own rig you must
    either finetune the teacher first or retarget its actions. Distill in the
    DATASET's frame, then let MicroVLA's normalizer handle scaling.

    Args:
        checkpoint: Path to the TinyVLA checkpoint.
        repo_path: Path to the cloned TinyVLA repository (added to sys.path).
        device: Torch device string.
        chunk_len: Actions per query; keep equal to the diffusion head's
            trained chunk size (their default is >= 8; ours consumes 5, extra
            rows are simply unused by ``relabel``'s tiling).
    """

    def __init__(self, checkpoint: str | Path, repo_path: str | Path,
                 device: str = "cpu", chunk_len: int = 8,
                 model_base: str | None = None, stats_path: str | None = None) -> None:
        import sys

        self.chunk_len = chunk_len
        self.device = device
        self.model_base = model_base
        self.stats_path = stats_path
        repo = Path(repo_path)
        if not repo.exists():
            raise FileNotFoundError(
                f"TinyVLA repo not found at {repo}. Clone "
                "https://github.com/liyaxuanliyaxuan/TinyVLA first — nothing is "
                "downloaded automatically."
            )
        sys.path.insert(0, str(repo))
        try:
            import torch  # noqa: F401

            self._policy = self._load_policy(Path(checkpoint))
        except ImportError as err:  # pragma: no cover - depends on user env
            raise ImportError(
                "TinyVLA imports failed. Install the repo's requirements plus "
                "`pip install -e policy_heads` and `pip install -e llava-pythia` "
                f"from the TinyVLA checkout. Original error: {err}"
            ) from err

    def _load_policy(self, checkpoint: Path):  # pragma: no cover - needs weights
        """Loads the TinyVLA policy; adjust here if their API differs.

        Wired against ``eval_real_franka.py`` of
        github.com/liyaxuanliyaxuan/TinyVLA (checked 2026-07): loads via
        ``llava_pythia.model.builder.load_pretrained_model(model_path,
        model_base, model_name)``, prompts through the ``pythia`` conv
        template with an image token, and calls ``policy(**batch, eval=True)``
        -> ``[B, chunk_size, action_dim]``. NOTE: ``checkpoint`` must be a
        TRAINED VLA directory (their training output after
        ``scripts/process_ckpts.sh``); the HuggingFace Llava-Pythia models are
        BASE VLMs (pass one as ``--teacher-base``) and have no action head.
        """
        import pickle

        from llava_pythia.mm_utils import get_model_name_from_path
        from llava_pythia.model.builder import load_pretrained_model

        model_path = str(checkpoint)
        model_name = get_model_name_from_path(model_path)
        self._tokenizer, self._model, self._image_processor, _ctx = load_pretrained_model(
            model_path, self.model_base, model_name, False, False
        )
        self._model.to(self.device).eval()
        self._stats = None
        stats = self.stats_path or str(Path(model_path) / "dataset_stats.pkl")
        if Path(stats).exists():
            with open(stats, "rb") as f:
                self._stats = pickle.load(f)
        else:  # pragma: no cover - depends on checkpoint layout
            logger.warning("TinyVLA stats pickle not found (%s): emitting the "
                           "policy's normalized actions UN-denormalized — pass "
                           "--teacher-stats.", stats)
        return self._model

    def predict_chunk(self, frame_rgb: np.ndarray, instruction: str,
                      state: np.ndarray | None = None) -> np.ndarray:  # pragma: no cover
        """One TinyVLA query following their eval script's batch construction."""
        import torch
        from llava_pythia.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava_pythia.conversation import conv_templates
        from llava_pythia.mm_utils import tokenizer_image_token
        from PIL import Image

        conv = conv_templates["pythia"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        img = Image.fromarray(np.ascontiguousarray(frame_rgb))
        px = self._image_processor.preprocess(img, return_tensors="pt")["pixel_values"]
        px = px.to(self.device, dtype=self._model.dtype)
        states = torch.zeros(1, getattr(self._model.config, "state_dim", 7),
                             device=self.device, dtype=self._model.dtype)
        if state is not None:
            s = torch.as_tensor(np.asarray(state, dtype=np.float32),
                                device=self.device).reshape(1, -1)
            n = min(s.shape[1], states.shape[1])
            states[:, :n] = s[:, :n].to(self._model.dtype)

        with torch.inference_mode():
            actions = self._model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                images=px, images_r=px,  # single wrist cam: duplicate views
                states=states, eval=True,
            )
        a = actions[0].float().cpu().numpy()          # [chunk_size, action_dim]

        if self._stats is not None:
            st = self._stats
            if "action_max" in st:      # diffusion head convention
                a = ((a + 1.0) / 2.0) * (np.asarray(st["action_max"]) - np.asarray(st["action_min"])) \
                    + np.asarray(st["action_min"])
            elif "action_std" in st:    # act head convention
                a = a * np.asarray(st["action_std"]) + np.asarray(st["action_mean"])
        if a.shape[-1] > 7:
            # Franka convention (xyz + 6D rot + grip) -> (xyz, rpy, grip).
            import torch_utils as TorchUtils

            rot = TorchUtils.rot_6d_to_euler_angles(
                torch.as_tensor(a[:, 3:9], dtype=torch.float32)
            ).numpy()
            a = np.concatenate([a[:, :3], rot, a[:, -1:]], axis=-1)
        return np.asarray(a, dtype=np.float32)[: self.chunk_len]


def build_teacher(name: str | None, checkpoint: str | None, repo: str | None,
                  cache: str | None, device: str = "cpu",
                  model_base: str | None = None,
                  stats_path: str | None = None) -> TeacherPolicy | None:
    """CLI helper: builds (and optionally caches) the requested teacher.

    Args:
        name: ``None`` (no distillation), ``"mock"``, or ``"tinyvla"``.
        checkpoint: TinyVLA checkpoint path (tinyvla only).
        repo: TinyVLA repo clone path (tinyvla only).
        cache: Optional cache directory for relabeled actions.
        device: Torch device string.

    Returns:
        A ready teacher, or ``None``.
    """
    if name is None:
        return None
    if name == "mock":
        teacher: TeacherPolicy = MockTeacher()
    elif name == "tinyvla":
        if not checkpoint or not repo:
            raise ValueError("--teacher tinyvla requires --teacher-checkpoint and --teacher-repo")
        teacher = TinyVLATeacher(checkpoint, repo, device=device,
                                 model_base=model_base, stats_path=stats_path)
    else:
        raise ValueError(f"unknown teacher {name!r} (expected 'mock' or 'tinyvla')")
    return CachedTeacher(teacher, cache) if cache else teacher
