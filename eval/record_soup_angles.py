"""Record a LIBERO alphabet-soup SUCCESS at 1920×1080 from four camera angles.

Uses the soup_v1 assisted config (the run that scored 0.75). Policy still sees
the wrist cam at ``--policy-res`` (default 256); each env step also renders
four cinematic MuJoCo cameras at full HD via ``sim.render`` and writes four
separate MP4s plus a small JSON sidecar the activation webapp can play.

    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa PYTHONPATH=/root/LIBERO \\
      python -m eval.record_soup_angles \\
        --checkpoint checkpoints/full_stageB_rec_fix.pt \\
        --norm-stats data/libero_object_grid/norm_stats.json \\
        --device cuda:0 --out-dir eval_results/soup_1080

Angles (MuJoCo names): agentview, robot0_eye_in_hand (wrist), birdview,
galleryview.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from microvla.config import DEFAULT_CONFIG
from microvla.utils.camera import ENV_KEY, WRIST, upright, view_of
from microvla.utils.signals import ignore_sigterm

#: Four angles — wrist replaces sideview so the pack includes the policy eye.
FILM_CAMS = ("agentview", "robot0_eye_in_hand", "birdview", "galleryview")
#: Friendly tab labels for the webapp (MuJoCo name → UI).
FILM_LABELS = {
    "agentview": "agentview",
    "robot0_eye_in_hand": "wrist",
    "birdview": "birdview",
    "galleryview": "galleryview",
    "sideview": "sideview",
}


def _open_writers(out_dir: Path, cams: tuple[str, ...], init_i: int,
                  film_w: int, film_h: int, fps: int):
    """Open one VideoWriter per camera — stream frames, do not buffer HD in RAM."""
    import cv2

    writers = {}
    paths = {}
    # mp4v (MPEG-4 Part 2) is what OpenCV reliably writes; browsers need H.264 —
    # we re-encode with ffmpeg after the episode (_to_h264).
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for cam in cams:
        path = out_dir / f"soup_success_init{init_i}_{cam}_{film_w}x{film_h}.mp4"
        w = cv2.VideoWriter(str(path), fourcc, float(fps), (film_w, film_h))
        if not w.isOpened():
            raise RuntimeError(f"could not open VideoWriter for {path}")
        writers[cam] = w
        paths[cam] = path
    return writers, paths


def _to_h264(path: Path) -> None:
    """In-place re-encode to H.264/yuv420p so Chromium/Safari can play it."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        print(f"[soup1080] ffmpeg missing — left {path.name} as mp4v "
              "(may not play in browsers)", flush=True)
        return
    tmp = path.with_suffix(".h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", str(tmp),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not tmp.exists():
        print(f"[soup1080] ffmpeg failed on {path.name}; keeping mp4v", flush=True)
        if tmp.exists():
            tmp.unlink()
        return
    tmp.replace(path)
    print(f"[soup1080] h264 {path.name}", flush=True)


def _get_sim(env):
    e = env
    for _ in range(6):
        if hasattr(e, "sim") and e.sim is not None:
            return e.sim
        e = getattr(e, "env", None)
        if e is None:
            break
    raise RuntimeError("could not find MjSim on OffScreenRenderEnv")


def _render_cam(sim, name: str, height: int, width: int) -> np.ndarray:
    """MuJoCo framebuffer is bottom-left; flip to ordinary image coords.

    Wrist also goes through :func:`upright` so the film matches the policy eye.
    """
    img = sim.render(camera_name=name, height=height, width=width)
    img = np.ascontiguousarray(img[::-1])
    try:
        if view_of(name) == WRIST or view_of(name + "_image") == WRIST:
            img = upright(img, WRIST)
    except KeyError:
        if "eye_in_hand" in name:
            img = upright(img, WRIST)
    return img


def _film_label(cam: str) -> str:
    return FILM_LABELS.get(cam, cam)


def _label(img: np.ndarray, text: str, cv2) -> np.ndarray:
    out = np.ascontiguousarray(img)
    scale = max(0.7, out.shape[0] / 900.0)
    org = (16, int(36 * scale))
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 2, cv2.LINE_AA)
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/full_stageB_rec_fix.pt")
    ap.add_argument("--norm-stats", default="data/libero_object_grid/norm_stats.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--heads-device", default="cpu")
    ap.add_argument("--out-dir", default="eval_results/soup_1080")
    ap.add_argument("--webapp-dir", default="paper/activation_webapp/data/soup_success",
                    help="also copy/symlink the pack here for the local webapp")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--policy-res", type=int, default=256,
                    help="wrist/agentview obs size fed to the policy")
    ap.add_argument("--film-w", type=int, default=1920)
    ap.add_argument("--film-h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--init-index", type=int, default=0,
                    help="first LIBERO init-state index to try (soup_v1 trial 0 worked)")
    ap.add_argument("--max-inits", type=int, default=8,
                    help="try this many init indices until one succeeds")
    ap.add_argument("--skip-scout", action="store_true",
                    help="film --init-index directly (no scout loop)")
    ap.add_argument("--cameras", default=",".join(FILM_CAMS),
                    help="comma-separated MuJoCo camera names to film")
    # soup_v1 knobs (no gate-verify — that arm is soup_v2 and was failing)
    ap.add_argument("--perception-period", type=int, default=2)
    ap.add_argument("--det-conf", type=float, default=0.02)
    ap.add_argument("--role-disjoint-iou", type=float, default=0.1)
    ap.add_argument("--source-max-area", type=float, default=0.12)
    ap.add_argument("--source-min-aspect", type=float,
                    default=DEFAULT_CONFIG.source_min_aspect)
    return ap.parse_args(argv)


def build_policy(args):
    """Build via libero_eval's factory — same path as teacher_rollouts (soup_v1)."""
    from eval import libero_eval as LE

    # Exact soup_v1 argv (no gate-verify). Trailing "--" style not needed here.
    argv = [
        "--suite", "libero_object",
        "--checkpoint", str(args.checkpoint),
        "--norm-stats", str(args.norm_stats),
        "--device", str(args.device),
        "--heads-device", str(args.heads_device or "cpu"),
        "--camera", "robot0_eye_in_hand_image",
        "--render-size", str(args.policy_res),
        "--perception-period", str(args.perception_period),
        "--det-conf", str(args.det_conf),
        "--no-brake",
        "--role-disjoint-iou", str(args.role_disjoint_iou),
        "--source-max-area", str(args.source_max_area),
        "--ibvs-phase",
        "--ibvs-gain", "0.5",
        "--ibvs-sign", "1,-1,0",
        "--ibvs-descend", "-0.4",
        "--ibvs-descend-hyst", "0.50",
        "--ibvs-press", "0.2",
        "--ibvs-retry-rise", "8",
        # no --ibvs-body-v: soup_v1 / teacher used default 1.0 (mask off)
        "--ibvs-target-uv", "0.5,0.60",
        "--ibvs-grasp-offset", "0.09,-0.186",
        "--ibvs-close-z", "0.045",
        "--ibvs-gate-z", "0.10",
        "--ibvs-approach-z", "0.12",
        "--ibvs-place-at=-0.006,0.260",
        "--ibvs-drop-z", "0.25",
        "--max-steps", str(args.max_steps),
        "--workers", "1",
    ]
    targs = LE.parse_args(argv)
    return LE._make_policy_factory(targs)()


def main(argv=None) -> int:
    args = parse_args(argv)
    ignore_sigterm()
    film_cams = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    if len(film_cams) != 4:
        raise SystemExit(f"need exactly 4 cameras, got {film_cams}")

    from eval._libero_compat import prepare_libero
    prepare_libero()
    from eval import libero_eval as LE
    from eval.policy import MicroVLAPolicy  # noqa: F401 — import before env (GL)

    print("[soup1080] building policy (libero_eval factory / soup_v1)…", flush=True)
    t0 = time.time()
    policy = build_policy(args)
    print(f"[soup1080] policy ready ({time.time()-t0:.0f}s)", flush=True)

    # Activation hooks — same schema as paper/activation_webapp (live model dump).
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paper.activation_webapp.hook_bank import (  # noqa: E402
        attach_policy_hooks, corrector_act, _r,
    )
    hook_bank = attach_policy_hooks(policy)

    from libero.libero.envs import OffScreenRenderEnv
    from microvla.utils.proprio import proprio_from_obs
    import cv2

    tasks = LE._real_tasks("libero_object")
    task = tasks[0]
    tid = 0
    bddl = task.bddl_file
    inits = np.asarray(task.init_states) if task.init_states is not None else np.zeros((0,))
    instruction = task.instruction
    print(f"[soup1080] task {tid}: {instruction!r}  n_inits={len(inits)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wrist_key = ENV_KEY[WRIST]
    tick_hz = float(getattr(policy.cfg, "tick_hz", 30.0) or 30.0)

    def _make_env():
        # Match teacher_rollouts: wrist-only obs stream (film cams via sim.render).
        cam_base = "robot0_eye_in_hand"
        try:
            return OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=args.policy_res,
                camera_widths=args.policy_res,
                camera_names=[cam_base],
            )
        except TypeError:
            return OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=args.policy_res,
                camera_widths=args.policy_res,
            )

    def _rollout(env, init_i: int, film: bool):
        """One episode. If film=True, stream 1080p cams + full activation trace."""
        if hasattr(env, "seed"):
            env.seed(init_i)
        obs = env.reset()
        if len(inits) > 0:
            obs = env.set_init_state(inits[init_i % len(inits)])
        policy.reset(instruction)
        sim = _get_sim(env) if film else None
        writers = paths = None
        if film:
            for cam in film_cams:
                p = out_dir / f"soup_success_init{init_i}_{cam}_{args.film_w}x{args.film_h}.mp4"
                if p.exists():
                    p.unlink()
            writers, paths = _open_writers(
                out_dir, film_cams, init_i, args.film_w, args.film_h, args.fps
            )
        steps = []
        tick_rows = []
        ok = False
        try:
            for step in range(args.max_steps):
                if film:
                    hook_bank.clear_latest()
                policy_view = upright(obs[wrist_key], wrist_key)
                proprio = proprio_from_obs(obs)
                action = np.asarray(
                    policy.act(policy_view, proprio=proprio), dtype=np.float32
                )
                action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
                if film:
                    for cam in film_cams:
                        frame = _label(
                            _render_cam(sim, cam, args.film_h, args.film_w),
                            _film_label(cam),
                            cv2,
                        )
                        writers[cam].write(np.ascontiguousarray(frame[..., ::-1]))
                tele = {}
                if getattr(policy, "telemetry", None) and policy.telemetry:
                    tele = dict(policy.telemetry[-1])
                steps.append({
                    "step": step,
                    "action": [float(x) for x in action.tolist()],
                    "eef": tele.get("eef"),
                    "trust": tele.get("trust"),
                    "phase": tele.get("phase"),
                    "src_conf": tele.get("src_conf"),
                    "src_center": tele.get("src_center"),
                    "plan_norm": tele.get("plan_norm"),
                    "is_real": tele.get("is_real"),
                    "ibvs_cmd": tele.get("ibvs_cmd"),
                })
                if film:
                    acts = dict(hook_bank.latest)
                    acts["corrector"] = corrector_act(policy.loop)
                    phase = tele.get("phase")
                    acts["ibvs_phase"] = {
                        "n": 1,
                        "l2": 1.0 if phase else 0.0,
                        "mean": 1.0,
                        "absmax": 1.0,
                        "sat": 0.0,
                        "top": [],
                        "phase": phase,
                    }
                    group_e: dict[str, float] = {}
                    for mid, s in acts.items():
                        g = mid.split(".", 1)[0]
                        group_e[g] = group_e.get(g, 0.0) + float(s.get("l2", 0.0))
                    tick_rows.append({
                        "t": step,
                        "t_s": _r(step / max(1.0, float(args.fps))),
                        "is_real": bool(tele.get("is_real")),
                        "trust": _r(float(tele.get("trust") or 0.0)),
                        "phase": phase,
                        "action": [_r(float(v)) for v in action.tolist()],
                        "ibvs_cmd": tele.get("ibvs_cmd"),
                        "plan_norm": _r(float(tele.get("plan_norm") or 0.0)),
                        "eef": tele.get("eef"),
                        "jaws": tele.get("jaws"),
                        "src_conf": (None if tele.get("src_conf") is None
                                     else _r(float(tele["src_conf"]))),
                        "tgt_conf": (None if tele.get("tgt_conf") is None
                                     else _r(float(tele["tgt_conf"]))),
                        "src_center": tele.get("src_center"),
                        "acts": acts,
                        "group_e": {k: _r(v) for k, v in group_e.items()},
                        "control": "assisted_phased_ibvs",
                    })
                obs, _reward, done, info = env.step(action)
                ok = bool(info.get("success", False)) if isinstance(info, dict) else False
                if not ok and hasattr(env, "check_success"):
                    ok = bool(env.check_success())
                if step > 0 and step % 100 == 0:
                    phase = tele.get("phase")
                    print(f"[soup1080]   init {init_i} step {step} "
                          f"phase={phase} film={film} hooks={len(hook_bank.latest)}",
                          flush=True)
                if done or ok:
                    break
        finally:
            if writers:
                for w in writers.values():
                    w.release()
        if film and not ok and paths:
            for p in paths.values():
                if p.exists():
                    p.unlink()
            paths = None
            tick_rows = []
        return ok, steps, paths, tick_rows

    # ---- Phase 1: fast scout (no HD) until a success ----
    used_init = None
    if args.skip_scout:
        used_init = args.init_index
        print(f"[soup1080] skip-scout → film init {used_init}", flush=True)
    else:
        print(f"[soup1080] SCOUT (no HD) over up to {args.max_inits} inits "
              f"from {args.init_index}…", flush=True)
        for k in range(args.max_inits):
            init_i = (args.init_index + k) % max(1, len(inits))
            env = _make_env()
            try:
                ok, steps, *_ = _rollout(env, init_i, film=False)
                print(f"[soup1080] scout init {init_i}: steps={len(steps)} "
                      f"success={ok}", flush=True)
                if ok:
                    used_init = init_i
                    break
            finally:
                if hasattr(env, "close"):
                    env.close()

    if used_init is None:
        hook_bank.close()
        raise SystemExit("no soup success in scout — aborting (no MP4s written)")

    # ---- Phase 2: re-run the winning init with 1080p film + activation dump ----
    print(f"[soup1080] FILM init {used_init} at "
          f"{args.film_w}x{args.film_h} (+activation hooks)…", flush=True)
    env = _make_env()
    try:
        success, diag_steps, video_paths, tick_rows = _rollout(
            env, used_init, film=True
        )
    finally:
        if hasattr(env, "close"):
            env.close()
        hook_bank.close()
    n_steps = len(diag_steps)
    print(f"[soup1080] film init {used_init}: steps={n_steps} "
          f"success={success} trace_ticks={len(tick_rows)}", flush=True)
    if not success or not video_paths:
        raise SystemExit(
            f"scout succeeded on init {used_init} but film re-roll failed — aborting"
        )

    for cam in film_cams:
        _to_h264(video_paths[cam])

    stamp = int(time.time() * 1000)
    trace = {
        "meta": {
            "source": "record_soup_angles",
            "task": instruction,
            "init_index": used_init,
            "fps": args.fps,
            "tick_hz": tick_hz,
            "checkpoint": args.checkpoint,
            "config": "soup_v1",
            "control": "assisted_phased_ibvs",
            "unaided": False,
            "n_ticks": len(tick_rows),
            "note": (
                "PhasedIBVS owns the emitted action; planner/TRM still forward "
                "every tick — activations are real, control is assisted."
            ),
        },
        "modules": hook_bank.modules,
        "ticks": tick_rows,
    }
    trace_path = out_dir / "trace.json"
    trace_path.write_text(json.dumps(trace, separators=(",", ":")))
    print(f"[soup1080] activation trace -> {trace_path} "
          f"({trace_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    pack = {
        "task": instruction,
        "task_id": tid,
        "init_index": used_init,
        "success": True,
        "n_steps": n_steps,
        "film_w": args.film_w,
        "film_h": args.film_h,
        "fps": args.fps,
        "cameras": list(film_cams),
        "camera_labels": {_film_label(c): c for c in film_cams},
        "labels": {c: _film_label(c) for c in film_cams},
        "checkpoint": args.checkpoint,
        "config": "soup_v1",
        "control": "assisted_phased_ibvs",
        "unaided": False,
        "trace": "trace.json",
        "videos": {c: video_paths[c].name for c in film_cams},
        "steps": diag_steps,
    }
    for cam in film_cams:
        mb = video_paths[cam].stat().st_size / (1024 * 1024)
        print(f"[soup1080] {cam}: {mb:.1f} MB -> {video_paths[cam]}", flush=True)

    meta_path = out_dir / f"soup_success_init{used_init}_{stamp}.json"
    # Slim meta for webapp (full steps stay in out_dir meta; web gets short + trace)
    slim = {k: v for k, v in pack.items() if k != "steps"}
    slim["steps"] = [
        {kk: s.get(kk) for kk in (
            "step", "action", "eef", "trust", "phase", "src_conf",
            "src_center", "plan_norm", "is_real",
        )}
        for s in diag_steps
    ]
    meta_path.write_text(json.dumps(pack, indent=2))
    (out_dir / "latest.json").write_text(json.dumps(slim, indent=2))
    print(f"[soup1080] meta -> {meta_path}", flush=True)

    web = Path(args.webapp_dir)
    web.mkdir(parents=True, exist_ok=True)
    import shutil
    for cam in film_cams:
        shutil.copy2(video_paths[cam], web / pack["videos"][cam])
    # drop stale sideview if present
    for stale in web.glob("*sideview*"):
        stale.unlink()
    shutil.copy2(trace_path, web / "trace.json")
    (web / "latest.json").write_text(json.dumps({
        **slim,
        "source": str(out_dir),
    }, indent=2))
    print(f"[soup1080] webapp pack -> {web}/", flush=True)
    print("[soup1080] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
