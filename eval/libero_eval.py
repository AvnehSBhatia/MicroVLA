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
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from microvla.config import DEFAULT_CONFIG
from microvla.utils.camera import ENV_KEY, upright
from microvla.utils import provenance as _prov
from microvla.utils.signals import ignore_sigterm

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
        from eval._libero_compat import prepare_libero
        prepare_libero()
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
    render_size: int = 256,
) -> tuple[bool, list[dict]]:
    """Runs one episode against a real LIBERO ``OffScreenRenderEnv``.

    ``render_size`` defaults to 256 (was hardcoded 128). YOLO-World upscales
    short sides to 512, but cubic upscaling from 128 invents texture that the
    region-text head cannot ground — measured source detection ~20% on wrist
    frames. 256 is the cheapest resolution that still carries usable edges.
    """
    from eval._libero_compat import prepare_libero
    prepare_libero()
    from libero.libero.envs import OffScreenRenderEnv

    # Render ONLY the policy's camera: osmesa is CPU software rendering and the
    # default env renders agentview + wrist every step — half of it discarded.
    cam_base = camera[:-6] if camera.endswith("_image") else camera
    size = max(64, int(render_size))
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=task.bddl_file,
            camera_heights=size,
            camera_widths=size,
            camera_names=[cam_base],
        )
    except TypeError:  # older robosuite/LIBERO without camera_names passthrough
        env = OffScreenRenderEnv(
            bddl_file_name=task.bddl_file,
            camera_heights=size,
            camera_widths=size,
        )
    try:
        if hasattr(env, "seed"):
            env.seed(trial_seed)
        obs = env.reset()
        if task.init_states is not None and len(task.init_states) > 0:
            init_state = task.init_states[trial_seed % len(task.init_states)]
            obs = env.set_init_state(init_state)
        policy.reset(task.instruction)

        from microvla.utils.proprio import proprio_from_obs

        telemetry: list[dict] = []
        success = False
        nonfinite_steps = 0
        for step in range(max_steps):
            # Same row flip the bake applies (microvla/utils/camera.py). The
            # detector is not rotation invariant and neither side was allowed
            # to hold a private copy of this convention again.
            frame = upright(obs[camera], camera)
            # v6: arm state every step (None on envs that don't expose it).
            action = policy.act(frame, proprio=proprio_from_obs(obs))
            # A NON-FINITE ACTION DOES NOT RAISE. The env accepts it, the
            # episode runs to max_steps, and the harness reports 0.000 -- which
            # is indistinguishable from a policy that simply never succeeds.
            # paper.md 5e: the HRM's recurrent state diverged past its training
            # horizon and every closed-loop number in this project was scored on
            # a policy emitting NaN for the back half of each episode. Counting
            # it is nearly free and turns a silent null into a stated one.
            if not np.isfinite(action).all():
                nonfinite_steps += 1
                action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            obs, _reward, done, info = env.step(action)
            step_telemetry = policy.telemetry[-1] if policy.telemetry else {}
            # Intermediate diagnostics: binary success alone collapses approach
            # / grasp / place into one bit. Gripper close + src detection are
            # free from existing telemetry; EEF-object distance when the sim
            # exposes object bodies.
            extra = {}
            try:
                eef = step_telemetry.get("eef")
                obj = _sim_object_pos(env)
                if eef is not None and obj is not None:
                    extra["eef_obj_dist"] = float(np.linalg.norm(
                        np.asarray(eef, dtype=np.float64)[:3]
                        - np.asarray(obj, dtype=np.float64)[:3]))
                    extra["obj_pos"] = [float(v) for v in obj[:3]]
            except Exception:
                pass
            telemetry.append({"step": step, **step_telemetry, **extra})
            success = bool(info.get("success", False)) if isinstance(info, dict) else False
            if not success and hasattr(env, "check_success"):
                success = bool(env.check_success())
            if done:
                break
        if nonfinite_steps:
            # Loud on purpose. A run that emits NaN for a third of its steps has
            # not been evaluated, whatever number the aggregate reports.
            logger.warning(
                "NON-FINITE ACTIONS on %d/%d steps of this episode (zeroed before "
                "env.step). The reported success rate does NOT measure this "
                "policy -- see paper.md 5e.", nonfinite_steps, max_steps)
        return success, telemetry
    finally:
        if hasattr(env, "close"):
            env.close()


def _sim_object_pos(env) -> list[float] | None:
    """Best-effort source-object position from the live robosuite/LIBERO sim.

    Returns ``[x,y,z]`` or ``None`` when the env does not expose bodies. Used
    only for intermediate metrics — never for control.
    """
    try:
        inner = getattr(env, "env", env)
        sim = getattr(inner, "sim", None)
        if sim is None:
            return None
        # LIBERO typically keeps obj_body_id: {name: body_id}. Prefer the first
        # non-target-looking body; fall back to the first entry.
        body_ids = getattr(inner, "obj_body_id", None) or {}
        if not body_ids:
            return None
        names = list(body_ids.keys())
        pick = next((n for n in names
                     if not any(t in n.lower() for t in ("basket", "bin", "plate", "tray"))),
                    names[0])
        pos = sim.data.body_xpos[body_ids[pick]]
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    except Exception:
        return None


def _episode_intermediates(telemetry: list[dict]) -> dict:
    """Approach / grasp / detection rates from one episode's telemetry."""
    out: dict = {}
    real = [r for r in telemetry if r.get("is_real")]
    if real:
        src = [float(r["src_conf"]) for r in real if "src_conf" in r]
        if src:
            out["src_detect_rate"] = float(sum(c > 0.0 for c in src) / len(src))
            out["src_conf_mean"] = float(sum(src) / len(src))
    acts = [r["action"] for r in telemetry if r.get("action")]
    if acts:
        # Last dim is gripper; convention: >0 closed / closing.
        grips = [float(a[-1]) for a in acts if len(a) >= 7]
        if grips:
            out["grip_close_rate"] = float(sum(g > 0.0 for g in grips) / len(grips))
    dists = [float(r["eef_obj_dist"]) for r in telemetry if r.get("eef_obj_dist") is not None]
    if dists:
        out["eef_obj_dist_min"] = float(min(dists))
        out["eef_obj_dist_at_20"] = float(dists[min(19, len(dists) - 1)])
        out["eef_obj_dist_final"] = float(dists[-1])
    return out


def run_eval(
    policy_factory: Callable[[], object],
    suite: str,
    n_trials: int,
    max_steps: int,
    camera: str = "robot0_eye_in_hand_image",
    mock_env: bool = False,
    seed: int = 0,
    out_dir: str | Path = "eval_results",
    task_filter: list[int] | None = None,
    run_tag: str = "",
    tasks: list[_TaskSpec] | None = None,
    render_size: int = 256,
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
        task_filter: Optional task indices to run (parallel sharding); ``None``
            runs every task in the suite.
        run_tag: Suffix appended to the run id (keeps per-worker telemetry
            files collision-free).
        tasks: Pre-enumerated task specs, bypassing ``_real_tasks``. The
            parallel driver enumerates ONCE in the parent and ships the specs
            to workers, so N workers do not import ``libero`` and re-read every
            task's init states simultaneously.
        render_size: Camera resolution for the real backend (default 256).

    Returns:
        ``{"suite", "per_task": {task_name: success_rate}, "mean_success",
        "n_trials", "telemetry_path", "intermediates"}``.
    """
    if mock_env:
        run_trial = _run_mock_trial
    else:
        def run_trial(policy, task, trial_seed, max_steps, camera):
            return _run_real_trial(policy, task, trial_seed, max_steps, camera,
                                   render_size=render_size)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = f"{suite}_{'mock' if mock_env else 'real'}_{int(time.time() * 1000)}{run_tag}"
    telemetry_path = out / f"{run_id}_telemetry.jsonl"

    per_task: dict[str, float] = {}
    inter_acc: dict[str, list[float]] = {}
    wtag = (run_tag or "main").lstrip("_")
    # The telemetry file is opened BEFORE the expensive policy build on purpose:
    # its existence is the cheapest possible proof that a worker got this far.
    # (A 10-worker run once stalled for 20 minutes with no file and no output,
    # which left the whole build+enumeration window indistinguishable from a
    # worker that never started — see the heartbeats below.)
    with telemetry_path.open("w") as tf:
        t_build = time.time()
        print(f"[{wtag}] building policy...", flush=True)
        policy = policy_factory()
        print(f"[{wtag}] policy ready ({time.time() - t_build:.0f}s); "
              f"enumerating {suite}", flush=True)
        t_enum = time.time()
        if tasks is None:
            tasks = _mock_tasks(suite) if mock_env else _real_tasks(suite)
        if task_filter is not None:
            tasks = [t for i, t in enumerate(tasks) if i in set(task_filter)]
        print(f"[{wtag}] {len(tasks)} task(s) ({time.time() - t_enum:.0f}s); starting trials",
              flush=True)
        for task in tasks:
            successes = 0
            for trial in range(n_trials):
                trial_seed = seed * 1_000_003 + trial
                print(f"[{wtag}] START {task.name} trial {trial}", flush=True)
                t_start = time.time()
                success, telemetry = run_trial(policy, task, trial_seed, max_steps, camera)
                successes += int(success)
                inter = _episode_intermediates(telemetry)
                for k, v in inter.items():
                    inter_acc.setdefault(k, []).append(v)
                inter_s = " ".join(f"{k}={v:.3f}" for k, v in inter.items()) or "-"
                print(f"[{wtag}] DONE  {task.name} trial {trial}: "
                      f"success={success} steps={len(telemetry)} "
                      f"({time.time() - t_start:.0f}s) {inter_s}", flush=True)
                for rec in telemetry:
                    tf.write(json.dumps({
                        "suite": suite, "task": task.name, "trial": trial,
                        "success": success, **rec,
                    }) + "\n")
                tf.flush()  # live progress for tail/watch
            per_task[task.name] = successes / n_trials if n_trials else 0.0

    mean_success = sum(per_task.values()) / len(per_task) if per_task else 0.0
    intermediates = {k: float(sum(vs) / len(vs)) for k, vs in inter_acc.items() if vs}
    results = {
        "suite": suite,
        "per_task": per_task,
        "mean_success": mean_success,
        "n_trials": n_trials,
        "telemetry_path": str(telemetry_path),
        "intermediates": intermediates,
        "render_size": None if mock_env else int(render_size),
    }
    results_path = out / f"{run_id}_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    # Printed because this file is a worker's OWN durable output: if the parent
    # later aborts the pool (a sibling worker died, the watchdog fired), the
    # shards that did finish are still on disk under these names.
    print(f"[{wtag}] mean_success {mean_success:.3f} intermediates={intermediates} "
          f"-> {results_path}", flush=True)
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
            zero_center_actions=args.zero_center_actions,
            action_gain=getattr(args, "action_gain", 1.0),
            waypoint_stats=getattr(args, "waypoint_stats", None),
            waypoint_brake=not getattr(args, "waypoint_no_brake", False),
            heads_device=getattr(args, "heads_device", None),
            chunk_exec=getattr(args, "chunk_exec", False),
            replan_every=getattr(args, "replan_every", 0),
            no_brake=getattr(args, "no_brake", False),
            no_dream_correct=getattr(args, "no_dream_correct", False),
            ibvs_gain=getattr(args, "ibvs_gain", 0.0),
            ibvs_conf_floor=getattr(args, "ibvs_conf_floor", 0.1),
            ibvs_sign=tuple(float(v) for v in
                            getattr(args, "ibvs_sign", "1,-1,0").split(",")),
            ibvs_descend=getattr(args, "ibvs_descend", 0.0),
            ibvs_phase=getattr(args, "ibvs_phase", False),
            ibvs_track_gate=getattr(args, "ibvs_track_gate", 0.0),
            ibvs_clip_rerank=getattr(args, "ibvs_clip_rerank", False),
            det_conf=getattr(args, "det_conf", 0.02),
        )

    return factory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_spatial",
                    help="LIBERO benchmark key (real backend) or a label for the synthetic tasks (mock)")
    p.add_argument("--n-trials", type=int, default=50, help="episodes per task")
    p.add_argument("--max-steps", type=int, default=300, help="max env steps per episode")
    p.add_argument("--camera", default="robot0_eye_in_hand_image",
                   choices=sorted(ENV_KEY.values()),
                   help="live observation the policy sees. MUST match the "
                        "corpus the checkpoint was trained on: agentview_image "
                        "for an agentview bake, robot0_eye_in_hand_image for a "
                        "wrist bake. On libero_object the wrist view grounds "
                        "the source phrase on 22%% of frames at conf 0.011 and "
                        "the box jumps 0.18 between frames; agentview grounds "
                        "85%% at 0.066 with 0.03 jitter (paper.md 5m).")
    p.add_argument("--mock-env", action="store_true", help="use MockLiberoEnv (no sim deps)")
    p.add_argument("--no-dream-correct", action="store_true",
                   help="skip InnovationCorrector on dream ticks. The corrector "
                        "is deployment-only state the trainer never modelled, so "
                        "this makes a dream tick's planner inputs match what "
                        "stage B actually optimised (paper.md 5n, defect 28).")
    p.add_argument("--strict-provenance", action="store_true",
                   help="refuse to run when the deployment knobs disagree with "
                        "the corpus's manifest.json provenance (camera, "
                        "det_conf, render size, perception period). Off by "
                        "default so a deliberate ablation still runs — the "
                        "mismatches are logged and written into the results "
                        "JSON either way.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default="none",
                    help="full_stageB.pt/full_stageA.pt path or directory; 'none' for fresh modules")
    p.add_argument("--norm-stats", default=None,
                    help="norm_stats.json path; defaults to eval/identity_norm_stats.json")
    p.add_argument("--no-brake", action="store_true",
                   help="disable the delta-mode trust brake (sets cfg.brake_trust=0). "
                        "The brake scales the plan by min(1, tau/brake_trust) and "
                        "measured trust runs as low as 0.216 against brake_trust 0.5, "
                        "i.e. a 0.43x scale -- while paper.md 4p measured the task's "
                        "magnitude tolerance at ~[0.95, 1.05]. Any braked step is "
                        "outside the passing band by construction.")
    p.add_argument("--chunk-exec", action="store_true",
                   help="advance the world model once per SAMPLE interval and "
                        "execute the plan's rows in between, instead of stepping "
                        "the TRM every env step. One TRM step is trained to "
                        "predict the next SAMPLED frame (LIBERO: 10 env steps / "
                        "0.5 s), so the default schedule extrapolates ~7 s of "
                        "predicted time per 0.7 s elapsed. See paper.md 4w.")
    p.add_argument("--replan-every", type=int, default=0,
                   help="env steps between re-plans under --chunk-exec; 0 uses "
                        "cfg.waypoint_row_stride, the trained sample stride.")
    p.add_argument("--action-gain", type=float, default=1.0,
                   help="scale the emitted POSE action by this factor (gripper "
                        "exempt). LIBERO's measured passing band for magnitude is "
                        "~[0.95, 1.05] while our policies emit 0.02-0.56 of demo "
                        "magnitude (paper.md 4p), so this tests whether the "
                        "DIRECTION is right and only the SCALE is wrong. A "
                        "diagnostic, not a fix.")
    p.add_argument("--ibvs-gain", type=float, default=0.0,
                   help="add a proportional residual that drives the detected "
                        "source center toward the wrist-frame grasp point. "
                        "Zero training. If this moves closed-loop success, "
                        "frozen features are NOT the ceiling — the BC "
                        "objective is (paper.md 5k). Try 0.05-0.2.")
    p.add_argument("--ibvs-sign", default="1,-1,0",
                   help="per-axis multipliers for the IBVS residual, 'sx,sy,sz'. "
                        "The image-down vs robot-up convention is not "
                        "self-evident and a wrong sign pushes AWAY from the "
                        "object, which would be a false negative from the one "
                        "experiment meant to falsify the encoder claim. Sweep it.")
    p.add_argument("--ibvs-descend", type=float, default=0.0,
                   help="raw downward action applied once the object is centred, "
                        "scaled by how centred it is. A residual that only "
                        "centres can never grasp, so a null without this says "
                        "nothing about the features.")
    p.add_argument("--ibvs-track-gate", type=float, default=0.0,
                   help="temporal box association for --ibvs-phase: reject a fix "
                        "that jumps more than this (normalized image inf-norm) "
                        "from the tracked box unless it persists ~5 fixes. 0 "
                        "disables. Attacks the measured 20-26%% teleport rate "
                        "of per-frame argmax binding (postscript 3).")
    p.add_argument("--ibvs-clip-rerank", action="store_true",
                   help="with --ibvs-phase: rebind source/target each tick to "
                        "the proposal whose ROIAlign emb best matches the role "
                        "CLIP text emb (reject if the other role scores higher). "
                        "Attacks the tracking-null finding that a stable box "
                        "can still be the wrong object (basket for salad dressing).")
    p.add_argument("--ibvs-phase", action="store_true",
                   help="phased falsifier: the state machine in eval/ibvs_phase.py "
                        "OWNS the action (servo/grasp/lift/place from detections + "
                        "proprio; the checkpoint contributes perception plumbing "
                        "only). Measures whether frozen features support the FULL "
                        "task under trivial control — video showed the learned "
                        "prior skips the grasp phase, so residual blending "
                        "(--ibvs-gain alone) can never complete it.")
    p.add_argument("--ibvs-conf-floor", type=float, default=0.1,
                   help="ignore source detections below this confidence when "
                        "applying --ibvs-gain.")
    p.add_argument("--det-conf", type=float, default=DEFAULT_CONFIG.det_conf,
                   help="YOLO-World confidence floor at EVAL (bake stays at "
                        "0.10). Closed-loop wrist detection was ~20%% at 0.10; "
                        "0.02 matches the text-region bake path.")
    p.add_argument("--render-size", type=int, default=256,
                   help="LIBERO camera resolution (was 128). Cubic upscale from "
                        "128 to YOLO's 512 invents texture the detector cannot "
                        "ground; 256 is the cheapest usable size.")
    p.add_argument("--waypoint-no-brake", action="store_true",
                    help="do NOT scale the waypoint translation command by the corrector's "
                         "trust. The delta-mode brake exists because a held DELTA is a "
                         "continued motion; a waypoint command is a positional error and is "
                         "self-limiting, so braking it only slows convergence. Ablation.")
    p.add_argument("--waypoint-stats", default=None,
                    help="v7.2 waypoint_stats.json (preprocess/fit_waypoint_gain.py). With a "
                         "waypoint-trained checkpoint, translation commands become a "
                         "proportional move toward the predicted EEF position measured "
                         "against live proprio instead of the regressed plan dims.")
    p.add_argument("--perception-period", type=int, default=15)
    p.add_argument("--device", default="cpu",
                   help="device for the YOLO-World detector (1 tick in --perception-period).")
    p.add_argument("--heads-device", default=None,
                   help="device for fusion/drift/TRM/planner/TQSA (default: cpu, unchanged). "
                        "These run on EVERY tick while the detector runs on one in 15, so on "
                        "the 15:1 schedule the d=1024 TRM — not the detector — dominates "
                        "wall-clock. Try matching --device on a box with GPU headroom.")
    p.add_argument("--out-dir", default="eval_results")
    p.add_argument("--zero-center-actions", action="store_true",
                   help="denormalize actions zero-centered (x=0 -> no motion) so a "
                        "collapsed/neutral policy stays still instead of drifting into a "
                        "wall. Diagnostic for the asymmetric-quantile drift; a proper fix "
                        "trains against symmetric targets.")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel eval processes. Episodes are embarrassingly parallel "
                        "and osmesa rendering is CPU work, so N workers ~= N x "
                        "throughput until the cores run out. Each worker owns a task "
                        "shard + its own policy/env, and its thread pools are capped to "
                        "cpu_count/workers. Works with --mock-env too (that is how the "
                        "parallel path stays testable without a simulator).")
    p.add_argument("--task-ids", default="",
                   help="comma-separated task indices to run, e.g. '0,3,7'. Manual "
                        "sharding: N single-task processes in a shell loop is the "
                        "fallback when in-process parallelism misbehaves.")
    p.add_argument("--stagger", type=float, default=5.0,
                   help="seconds x worker index to delay each worker's start, spreading "
                        "GPU-context / renderer / import storms. 0 disables.")
    p.add_argument("--worker-timeout", type=float, default=0.0,
                   help="seconds before unfinished workers are declared stalled, killed, "
                        "and reported (0 = wait forever). Partial results from the "
                        "workers that DID finish are kept and clearly marked partial.")
    return p.parse_args(argv)


def _limit_worker_threads(n_workers: int) -> int:
    """Caps per-process thread pools for a run of ``n_workers`` workers.

    Nothing on the eval path ever called ``torch.set_num_threads``, so each of
    N worker processes would build an OpenMP team sized to the HOST core count
    (PyTorch does not read cgroup quotas), plus an OpenBLAS pool, plus OSMesa's
    llvmpipe rasterizer threads — hundreds of threads per worker, thousands per
    run. Measured on the d=1024 TRM: an oversized team is 2.5x SLOWER than
    single-threaded at batch 1, so the oversubscription is not merely wasteful,
    it is a direct slowdown of the thing being evaluated.

    MUST BE CALLED IN THE PARENT, before workers are spawned. ``eval/__init__``
    imports ``eval.policy``, which imports torch — so by the time any code in
    this module runs, libgomp is already initialized in THIS process and these
    variables would be no-ops here. Spawned children inherit ``os.environ`` and
    read them at their own startup, which is where they bite.

    Args:
        n_workers: Total worker processes sharing this machine.

    Returns:
        The per-worker thread budget applied.
    """
    budget = max(1, (os.cpu_count() or 1) // max(1, n_workers))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(budget))
    # OSMesa's software rasterizer spawns its own pool per GL context.
    os.environ.setdefault("LP_NUM_THREADS", str(budget))
    # Train-parity hygiene, not a fix for anything seen here: the trainers set
    # this for a known hipBLASLt segfault on some ROCm GEMMs and eval never did.
    os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
    return budget


def _parallel_worker(payload: dict) -> dict:
    """Spawn-safe worker: runs one task shard of the suite in this process.

    Rebuilds the CLI args namespace from a plain dict (picklable across the
    ``spawn`` boundary — required for CUDA/ROCm safety), builds its OWN policy
    and envs, and runs ``run_eval`` restricted to its shard.

    Every step prints, because the failure this replaced was a silent one: the
    worker announces itself BEFORE the stagger sleep (so "started but waiting"
    is distinguishable from "never scheduled"), and ``run_eval`` brackets the
    policy build and task enumeration with their own heartbeats.
    """
    w = payload["worker"]
    ignore_sigterm(verbose=False)   # spawned process: its own signal handlers
    print(f"[w{w}] spawned (pid {os.getpid()}, {len(payload['tasks'])} tasks)", flush=True)
    # A worker can stall INSIDE a C call the heartbeats cannot reach (the prime
    # suspect for the 20-minute hang is concurrent HIP context creation inside
    # `.to('cuda:0')`). faulthandler dumps the native+Python stack from a
    # watchdog thread, so a stuck worker self-reports without py-spy — which
    # needs SYS_PTRACE that containers often deny. `kill -USR1 <pid>` forces a
    # dump on demand; the pid is on the line above.
    import faulthandler
    import signal

    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1)
    # NO periodic dump_traceback_later: it fires on a timer regardless of whether
    # the worker is stuck, so a pool worker idling on the task queue between
    # trials printed a full "Timeout (0:10:00)!" traceback into the middle of the
    # run's output. On-demand only — `kill -USR1 <pid>`, and the pid is on the
    # startup line.

    budget = payload["thread_budget"]
    delay = payload["stagger"] * w
    if delay:
        print(f"[w{w}] staggering {delay:.0f}s (threads<={budget})", flush=True)
        time.sleep(delay)  # stagger GPU-context / renderer / import storms
    args = argparse.Namespace(**payload["args"])
    try:
        import torch

        torch.set_num_threads(budget)
    except Exception as e:  # pragma: no cover - torch is always present in practice
        print(f"[w{w}] could not cap torch threads: {e}", flush=True)
    return run_eval(
        _make_policy_factory(args),
        suite=args.suite,
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        camera=args.camera,
        mock_env=args.mock_env,
        seed=args.seed,
        out_dir=args.out_dir,
        run_tag=f"_w{w}",
        tasks=payload["tasks"],
        render_size=getattr(args, "render_size", 256),
    )


def _scavenge_worker_results(
    out: Path, suite: str, workers: list[int], since: float, per_task: dict[str, float],
) -> int:
    """Recovers finished shards of workers whose RESULT never came back.

    One dead worker breaks the whole executor, failing its healthy siblings'
    futures too — but every worker writes its own ``*_results.json`` the moment
    it finishes, so that work is on disk even when the return trip was lost.

    Only failed workers are scavenged and merges are keyed by task name, so a
    worker that wrote its file and THEN died during return-pickling cannot be
    counted twice.

    Args:
        out: Results directory.
        suite: Suite name (the results filename prefix).
        workers: Indices of workers that failed.
        since: Ignore files older than this timestamp (a previous run's).
        per_task: Already-reported results; updated in place with new entries.

    Returns:
        How many tasks were recovered.
    """
    scavenged = 0
    for w in workers:
        found = [p for p in out.glob(f"{suite}_*_w{w}_results.json")
                 if p.stat().st_mtime >= since]
        if not found:
            continue
        try:
            recovered = json.loads(max(found, key=lambda p: p.stat().st_mtime).read_text())
        except (OSError, ValueError):
            continue
        new = {k: v for k, v in recovered.get("per_task", {}).items() if k not in per_task}
        if new:
            per_task.update(new)
            scavenged += len(new)
            print(f"[w{w}] recovered {len(new)} task(s) from its on-disk results",
                  flush=True)
    return scavenged


def _run_parallel(args: argparse.Namespace, tasks: list[_TaskSpec]) -> dict:
    """Shards ``tasks`` across worker processes and merges their results.

    Uses :class:`concurrent.futures.ProcessPoolExecutor`, NOT ``mp.Pool``:
    ``Pool.map`` has no worker-death detection — a segfaulting or OOM-killed
    worker is quietly reaped and replaced while its chunk is never
    re-dispatched, so the parent blocks forever with no error. That is exactly
    the shape of an unexplainable 20-minute hang. ``ProcessPoolExecutor``
    raises ``BrokenProcessPool`` instead, in under a second.

    Each worker writes its OWN results/telemetry as it finishes, so a shard
    that completed is never lost to a sibling's failure — ``_scavenge_worker_
    results`` picks those up when the executor breaks.
    """
    import concurrent.futures as cf
    import multiprocessing as mp

    workers = min(max(1, int(args.workers)), len(tasks))
    # Set the thread env HERE, in the parent: children inherit os.environ at
    # spawn and read it before their own libgomp/OpenBLAS come up. Setting it
    # inside the worker would be too late (see _limit_worker_threads).
    budget = _limit_worker_threads(workers)
    shards = [tasks[w::workers] for w in range(workers)]
    payloads = [
        {"args": vars(args), "tasks": s, "worker": w, "thread_budget": budget,
         "stagger": args.stagger}
        for w, s in enumerate(shards) if s
    ]
    print(f"parallel eval: {len(tasks)} tasks across {len(payloads)} workers "
          f"({args.stagger:.0f}s stagger, {args.worker_timeout or 0:.0f}s timeout)",
          flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    per_task: dict[str, float] = {}
    telemetry_paths: list[str] = []
    failed: dict[int, str] = {}
    ctx = mp.get_context("spawn")  # fork + CUDA/ROCm is unsafe
    deadline = time.time() + args.worker_timeout if args.worker_timeout else None
    pool = cf.ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx)
    futures = {pool.submit(_parallel_worker, p): p["worker"] for p in payloads}
    try:
        for fut in cf.as_completed(futures, timeout=None if deadline is None
                                   else max(1.0, deadline - time.time())):
            w = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                failed[w] = f"{type(e).__name__}: {e}"
                print(f"[w{w}] FAILED — {failed[w]}", flush=True)
                continue
            per_task.update(r["per_task"])
            telemetry_paths.append(r["telemetry_path"])
            print(f"[w{w}] done: {len(r['per_task'])} tasks, "
                  f"mean {r['mean_success']:.3f}", flush=True)
    except cf.TimeoutError:
        stalled = sorted(w for f, w in futures.items() if not f.done())
        failed.update({w: "timeout" for w in stalled})
        print(f"\nTIMED OUT after {args.worker_timeout:.0f}s — workers {stalled} "
              f"never finished; killing them. To see WHERE one is stuck, before "
              f"killing:\n  pip install py-spy && py-spy dump --pid <PID>\n"
              f"(every worker prints its pid on the line it starts with).", flush=True)
    finally:
        for f in futures:
            f.cancel()
        # A running future cannot be cancelled, and the executor's shutdown
        # WAITS for it — so a stalled worker would turn the watchdog itself
        # into a hang. Kill the processes first, then shut down.
        if failed:
            for proc in (getattr(pool, "_processes", None) or {}).values():
                if proc.is_alive():
                    proc.kill()
        pool.shutdown(wait=True)

    scavenged = _scavenge_worker_results(out, args.suite, sorted(failed),
                                         t_start, per_task)
    results = {
        "suite": args.suite,
        "per_task": per_task,
        "mean_success": sum(per_task.values()) / len(per_task) if per_task else 0.0,
        "n_trials": args.n_trials,
        "workers": len(payloads),
        "tasks_expected": len(tasks),
        "tasks_completed": len(per_task),
        "tasks_scavenged": scavenged,
        "failed_workers": failed,
        "telemetry_paths": telemetry_paths,
    }
    if failed:
        print(f"WARNING: {len(failed)}/{len(payloads)} workers did not finish "
              f"({failed}); mean_success covers {len(per_task)}/{len(tasks)} tasks "
              f"and is NOT the suite number.", flush=True)
    merged = out / f"{args.suite}_real_{int(time.time() * 1000)}_results.json"
    merged.write_text(json.dumps(results, indent=2))
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ignore_sigterm()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Enumerate ONCE, in the parent: N workers importing libero and re-reading
    # every task's init states simultaneously is a concurrency hazard bought for
    # nothing, and the specs pickle fine across the spawn boundary.
    # Does this deployment match the corpus the checkpoint was trained on?
    # Four of the 26 defects in paper.md are exactly this mismatch, and none of
    # them raises anything at run time -- the eval simply reports 0.000. The
    # corpus now records what produced it (manifest.json provenance); this is
    # the one place that reads it back.
    prov = _prov.load(args.norm_stats) if args.norm_stats else {}
    logging.info("%s", _prov.describe(prov))
    bad = _prov.mismatches(
        prov,
        camera=args.camera,
        det_conf=getattr(args, "det_conf", None),
        render_size=getattr(args, "render_size", None),
        perception_period=args.perception_period,
    )
    for m in bad:
        logging.error("PROVENANCE MISMATCH: %s", m)
    if bad and getattr(args, "strict_provenance", False):
        raise SystemExit(
            "refusing to score a deployment that does not match its corpus; "
            "pass --no-strict-provenance to measure it anyway")

    tasks = _mock_tasks(args.suite) if args.mock_env else _real_tasks(args.suite)
    if args.task_ids:
        keep = {int(i) for i in args.task_ids.split(",") if i.strip() != ""}
        unknown = sorted(i for i in keep if not 0 <= i < len(tasks))
        if unknown:
            raise SystemExit(f"--task-ids {unknown} out of range "
                             f"(suite {args.suite} has {len(tasks)} tasks)")
        tasks = [t for i, t in enumerate(tasks) if i in keep]
        print(f"task filter: {sorted(keep)} -> {len(tasks)} task(s)", flush=True)

    if max(1, int(args.workers)) > 1:
        results = _run_parallel(args, tasks)
    else:
        _limit_worker_threads(1)
        results = run_eval(
            _make_policy_factory(args),
            suite=args.suite,
            n_trials=args.n_trials,
            max_steps=args.max_steps,
            camera=args.camera,
            mock_env=args.mock_env,
            seed=args.seed,
            out_dir=args.out_dir,
            tasks=tasks,
            render_size=getattr(args, "render_size", 256),
        )
    # Carried in the results file, not only the log: a scored number whose
    # deployment did not match its corpus must stay self-identifying after the
    # terminal scrollback is gone.
    if isinstance(results, dict):
        results["provenance_mismatches"] = bad
        results["corpus_provenance"] = prov
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
