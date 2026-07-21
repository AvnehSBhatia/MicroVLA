"""LIBERO closed-loop eval harness (paper E3): success rates over 3 suites.

``run_eval`` drives a ``policy_factory``-built policy (duck-typed:
``reset(instruction)`` / ``act(frame_rgb) -> action``, e.g.
:class:`eval.policy.MicroVLAPolicy`) through every task of a LIBERO suite,
``n_trials`` seeded episodes each, and reports per-task + mean success.

Two backends behind the same ``run_eval`` signature:

* ``mock_env=True`` -- :class:`MockLiberoEnv`, a dependency-free,
  deterministic stand-in (hash-seeded 128x128 frames, success gated on
  cumulative action norm crossing a seeded threshold). No ``libero``
  install, no sim, no network -- this is what makes the harness testable
  everywhere and lets the CLI run end-to-end today
  (``--mock-env --checkpoint none``).
* ``mock_env=False`` -- real LIBERO, lazily imported (never touched by the
  mock path): ``libero.libero.benchmark`` enumerates the suite's tasks +
  language + seeded initial states, ``libero.libero.envs.OffScreenRenderEnv``
  runs them. Requires ``pip install libero`` (the sim stack) separately --
  NOT part of this repo's core/perception/dev extras.

Telemetry (one JSON object per env step, merging the step/task/trial context
with the policy's own per-tick record -- see ``MicroVLAPolicy.telemetry``) is
appended to a JSONL file; final per-task/mean results are also written as
JSON. Both land under ``eval_results/`` (gitignored -- see ``.gitignore``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Synthetic (task_name_suffix, instruction) pairs used by the mock backend
#: so `--mock-env` never needs a real LIBERO suite definition.
_MOCK_TASK_INSTRUCTIONS = [
    "pick up the red block and place it on the plate",
    "move the mug to the basket",
    "open the drawer and place the bowl inside",
]


@dataclass
class _TaskSpec:
    """One evaluable task: an instruction plus (real-backend-only) env args.

    Attributes:
        name: Stable identifier used as the ``per_task`` results key.
        instruction: Natural-language instruction passed to
            ``policy.reset``.
        bddl_file: Real backend only -- LIBERO's per-task BDDL scene file.
        init_states: Real backend only -- ``[n_init, state_dim]`` seeded
            initial states (``benchmark.get_task_init_states``); trial ``i``
            uses ``init_states[i % len(init_states)]``.
    """

    name: str
    instruction: str
    bddl_file: Optional[str] = None
    init_states: Optional[np.ndarray] = None


class MockLiberoEnv:
    """Deterministic, dependency-free stand-in for a LIBERO env (tests/CI).

    No sim, no rendering, no network: every observation is derived from a
    SHA-256 digest, so a given ``(task, trial_seed)`` always reproduces the
    identical frame sequence and success outcome. 7-dim continuous action
    space (matches ``cfg.num_servos``); "success" fires once the episode's
    cumulative action L2-norm crosses a seeded per-(task, trial) threshold --
    different policies (different action magnitudes/directions) cross it at
    different times or not at all, so success rate is a real (if synthetic)
    discriminator between policies, not a coin flip.

    Args:
        task: Task name (mixed into the seed).
        camera: Accepted for interface parity with the real env; unused (the
            mock always yields one synthetic RGB stream).
        max_steps: Episode step cap; ``step`` sets ``done=True`` at or past
            this many calls even absent success.
    """

    #: Frame side length (uint8 RGB), matching a typical LIBERO camera obs.
    FRAME_SIZE = 128

    def __init__(self, task: str, camera: str = "robot0_eye_in_hand_image",
                 max_steps: int = 300) -> None:
        self.task = task
        self.camera = camera
        self.max_steps = max_steps
        self._episode_seed = 0
        self._t = 0
        self._cum_norm = 0.0
        self._threshold = 1.0
        self._success = False

    def reset(self, trial_seed: int) -> np.ndarray:
        """Starts a new deterministic episode; returns the first frame.

        Args:
            trial_seed: Caller-supplied seed identifying this trial
                (combined with ``task`` -- distinct tasks never collide).

        Returns:
            ``[FRAME_SIZE, FRAME_SIZE, 3]`` uint8 RGB frame.
        """
        digest = hashlib.sha256(f"{self.task}|{trial_seed}".encode()).digest()
        self._episode_seed = int.from_bytes(digest[:8], "little")
        # Threshold in [3.0, 7.0): large enough that a near-zero-action
        # (untrained / fresh-module) policy rarely succeeds by accident,
        # small enough that a policy actually moving succeeds sometimes.
        self._threshold = 3.0 + 4.0 * (int.from_bytes(digest[8:10], "little") / 65536.0)
        self._t = 0
        self._cum_norm = 0.0
        self._success = False
        return self._frame()

    def _frame(self) -> np.ndarray:
        digest = hashlib.sha256(f"{self._episode_seed}|{self._t}".encode()).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=(self.FRAME_SIZE, self.FRAME_SIZE, 3), dtype=np.uint8)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Advances one step: accumulates action norm, checks the threshold.

        Args:
            action: ``[7]`` raw action (any real values; only its L2 norm
                matters to this mock).

        Returns:
            ``(frame, reward, done, info)`` -- ``reward`` is always ``0.0``
            (unused by this harness), ``info = {"success": bool}``.
        """
        self._t += 1
        self._cum_norm += float(np.linalg.norm(np.asarray(action, dtype=np.float64)))
        if not self._success and self._cum_norm >= self._threshold:
            self._success = True
        done = self._success or self._t >= self.max_steps
        return self._frame(), 0.0, done, {"success": self._success}


def _mock_tasks(suite: str) -> list[_TaskSpec]:
    """A small fixed set of synthetic tasks for the mock backend."""
    return [
        _TaskSpec(name=f"{suite}__mock_task_{i}", instruction=instr)
        for i, instr in enumerate(_MOCK_TASK_INSTRUCTIONS)
    ]


def _real_tasks(suite: str) -> list[_TaskSpec]:
    """Enumerates a real LIBERO suite's tasks via its public benchmark API.

    Lazily imports ``libero`` (the sim stack -- not a core/dev dependency of
    this repo; install separately). Best-effort against LIBERO's documented
    ``benchmark`` module; if a given LIBERO release renamed a method, this
    is the one place to patch.

    Args:
        suite: A key of ``libero.libero.benchmark.get_benchmark_dict()``,
            e.g. ``"libero_spatial"``, ``"libero_object"``, ``"libero_goal"``.

    Returns:
        One ``_TaskSpec`` per task in the suite, with ``bddl_file`` and
        ``init_states`` populated.

    Raises:
        ImportError: If ``libero`` is not installed.
        ValueError: If ``suite`` is not a known benchmark key.
    """
    try:
        from libero.libero import benchmark
    except ImportError as e:  # pragma: no cover - exercised only without libero
        raise ImportError(
            "run_eval(mock_env=False) requires the LIBERO sim stack "
            "('pip install libero' / robosuite + its deps); not part of "
            "this repo's core/dev/perception extras. Use mock_env=True for "
            "a dependency-free dry run."
        ) from e

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite not in benchmark_dict:
        raise ValueError(f"unknown LIBERO suite {suite!r}; available: {sorted(benchmark_dict)}")
    bench = benchmark_dict[suite]()
    n_tasks = bench.get_num_tasks() if hasattr(bench, "get_num_tasks") else bench.n_tasks

    specs = []
    for i in range(n_tasks):
        task = bench.get_task(i)
        bddl_file = bench.get_task_bddl_file_path(i)
        init_states = np.asarray(bench.get_task_init_states(i))
        specs.append(_TaskSpec(
            name=getattr(task, "name", f"{suite}__task_{i}"),
            instruction=task.language,
            bddl_file=str(bddl_file),
            init_states=init_states,
        ))
    return specs


def _run_mock_trial(
    policy, task: _TaskSpec, trial_seed: int, max_steps: int, camera: str,
) -> tuple[bool, list[dict]]:
    """Runs one episode against :class:`MockLiberoEnv`."""
    env = MockLiberoEnv(task=task.name, camera=camera, max_steps=max_steps)
    frame = env.reset(trial_seed)
    policy.reset(task.instruction)

    telemetry: list[dict] = []
    success = False
    for step in range(max_steps):
        action = policy.act(frame)
        frame, _reward, done, info = env.step(action)
        step_telemetry = policy.telemetry[-1] if policy.telemetry else {}
        telemetry.append({"step": step, **step_telemetry})
        success = bool(info.get("success", False))
        if done:
            break
    return success, telemetry


def _run_real_trial(
    policy, task: _TaskSpec, trial_seed: int, max_steps: int, camera: str,
) -> tuple[bool, list[dict]]:
    """Runs one episode against a real LIBERO ``OffScreenRenderEnv``."""
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=task.bddl_file,
        camera_heights=MockLiberoEnv.FRAME_SIZE,
        camera_widths=MockLiberoEnv.FRAME_SIZE,
    )
    try:
        if hasattr(env, "seed"):
            env.seed(trial_seed)
        obs = env.reset()
        if task.init_states is not None and len(task.init_states) > 0:
            init_state = task.init_states[trial_seed % len(task.init_states)]
            obs = env.set_init_state(init_state)
        policy.reset(task.instruction)

        telemetry: list[dict] = []
        success = False
        for step in range(max_steps):
            frame = obs[camera]
            action = policy.act(frame)
            obs, _reward, done, info = env.step(action)
            step_telemetry = policy.telemetry[-1] if policy.telemetry else {}
            telemetry.append({"step": step, **step_telemetry})
            success = bool(info.get("success", False)) if isinstance(info, dict) else False
            if not success and hasattr(env, "check_success"):
                success = bool(env.check_success())
            if done:
                break
        return success, telemetry
    finally:
        if hasattr(env, "close"):
            env.close()


def run_eval(
    policy_factory: Callable[[], object],
    suite: str,
    n_trials: int,
    max_steps: int,
    camera: str = "robot0_eye_in_hand_image",
    mock_env: bool = False,
    seed: int = 0,
    out_dir: str | Path = "eval_results",
) -> dict:
    """Runs ``n_trials`` seeded episodes of every task in ``suite``.

    ``policy_factory`` is called ONCE to build a single policy instance,
    reused across every task/trial via ``policy.reset(instruction)`` at the
    start of each episode -- exactly the deploy pattern (build once, reset
    per task) and the cheapest for a real checkpoint (one load).

    Args:
        policy_factory: Zero-arg callable returning a policy exposing
            ``reset(instruction)`` / ``act(frame_rgb) -> action`` (and,
            optionally, ``.telemetry`` -- used if present to enrich the
            saved per-step telemetry with the policy's own tick record).
        suite: LIBERO suite name (real backend) or an arbitrary label used
            to namespace the synthetic tasks (mock backend).
        n_trials: Episodes per task.
        max_steps: Max env steps per episode.
        camera: Observation key for the RGB frame handed to ``policy.act``.
        mock_env: Use :class:`MockLiberoEnv` (no sim deps, deterministic)
            instead of real LIBERO.
        seed: Base seed; trial ``t`` of any task uses
            ``seed * 1_000_003 + t`` (deterministic, collision-free across a
            plausible range of trial counts).
        out_dir: Directory for the telemetry JSONL + results JSON.

    Returns:
        ``{"suite", "per_task": {task_name: success_rate}, "mean_success",
        "n_trials", "telemetry_path"}``.
    """
    policy = policy_factory()
    tasks = _mock_tasks(suite) if mock_env else _real_tasks(suite)
    run_trial = _run_mock_trial if mock_env else _run_real_trial

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = f"{suite}_{'mock' if mock_env else 'real'}_{int(time.time() * 1000)}"
    telemetry_path = out / f"{run_id}_telemetry.jsonl"

    per_task: dict[str, float] = {}
    with telemetry_path.open("w") as tf:
        for task in tasks:
            successes = 0
            for trial in range(n_trials):
                trial_seed = seed * 1_000_003 + trial
                success, telemetry = run_trial(policy, task, trial_seed, max_steps, camera)
                successes += int(success)
                for rec in telemetry:
                    tf.write(json.dumps({
                        "suite": suite, "task": task.name, "trial": trial,
                        "success": success, **rec,
                    }) + "\n")
            per_task[task.name] = successes / n_trials if n_trials else 0.0

    mean_success = sum(per_task.values()) / len(per_task) if per_task else 0.0
    results = {
        "suite": suite,
        "per_task": per_task,
        "mean_success": mean_success,
        "n_trials": n_trials,
        "telemetry_path": str(telemetry_path),
    }
    (out / f"{run_id}_results.json").write_text(json.dumps(results, indent=2))
    return results


def _make_policy_factory(args: argparse.Namespace) -> Callable[[], object]:
    """Builds the zero-arg policy factory the CLI hands to ``run_eval``."""
    checkpoint = None if str(args.checkpoint).strip().lower() == "none" else args.checkpoint
    norm_stats = args.norm_stats or str(Path(__file__).resolve().parent / "identity_norm_stats.json")

    def factory():
        from eval.policy import MicroVLAPolicy

        perception = task_encoder = None
        if args.mock_env:
            from microvla.perception.text_encoder import MockTaskEncoder
            from microvla.perception.yolo_world import MockYoloWorldPerception

            perception = MockYoloWorldPerception()
            task_encoder = MockTaskEncoder()

        return MicroVLAPolicy(
            checkpoint=checkpoint,
            norm_stats=norm_stats,
            perception_period=args.perception_period,
            device=args.device,
            perception=perception,
            task_encoder=task_encoder,
        )

    return factory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_spatial",
                    help="LIBERO benchmark key (real backend) or a label for the synthetic tasks (mock)")
    p.add_argument("--n-trials", type=int, default=50, help="episodes per task")
    p.add_argument("--max-steps", type=int, default=300, help="max env steps per episode")
    p.add_argument("--camera", default="robot0_eye_in_hand_image")
    p.add_argument("--mock-env", action="store_true", help="use MockLiberoEnv (no sim deps)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default="none",
                    help="full_stageB.pt/full_stageA.pt path or directory; 'none' for fresh modules")
    p.add_argument("--norm-stats", default=None,
                    help="norm_stats.json path; defaults to eval/identity_norm_stats.json")
    p.add_argument("--perception-period", type=int, default=15)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="eval_results")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    results = run_eval(
        _make_policy_factory(args),
        suite=args.suite,
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        camera=args.camera,
        mock_env=args.mock_env,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
