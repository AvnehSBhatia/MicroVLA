"""Open-loop mechinterp: feed a demo MP4 through MicroVLA, dump a scrub pack.

Crops the wrist pane from dual-view demo films (agent | wrist), runs the
policy with real YOLO-World (no IBVS), and writes a JSON pack the
``demo.html`` UI scrubs against the video clock.

Usage::

    .venv/bin/python paper/activation_webapp/generate_demo_trace.py
    .venv/bin/python paper/activation_webapp/generate_demo_trace.py \\
        --video watch_videos/demo_cream/demo_0_....mp4 \\
        --checkpoint checkpoints/teacher_bc/full_stageB_teacher_bc3.pt \\
        --compare checkpoints/rec_fix/full_stageB_rec_fix.pt
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.policy import MicroVLAPolicy  # noqa: E402
from paper.activation_webapp.hook_bank import attach_policy_hooks, corrector_act  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / "demo_cream"
DEFAULT_VIDEO = (
    ROOT / "watch_videos/demo_cream"
    / "demo_0_success_pick_up_the_cream_cheese_and_place_it_in.mp4"
)
DEFAULT_CKPT = ROOT / "checkpoints/teacher_bc/full_stageB_teacher_bc3.pt"
DEFAULT_NORM = ROOT / "checkpoints/teacher_bc/norm_stats_teacher_grid2.json"
DEFAULT_COMPARE = ROOT / "checkpoints/rec_fix/full_stageB_rec_fix.pt"
DEFAULT_COMPARE_NORM = ROOT / "checkpoints/rec_fix/norm_stats.json"
INSTRUCTION = "pick up the cream cheese and place it in the basket"
GROUP_ORDER = ["fusion", "drift", "trm", "tqsa", "relational", "planner", "corrector"]
ROUND = 4


def _r(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return round(float(x), ROUND)


def _wrist_crop(bgr: np.ndarray) -> np.ndarray:
    """Dual-pane demos are agentview | wrist; model eval uses wrist."""
    _h, w = bgr.shape[:2]
    wrist = bgr[:, w // 2 :]
    if wrist.shape[0] != 256 or wrist.shape[1] != 256:
        wrist = cv2.resize(wrist, (256, 256), interpolation=cv2.INTER_AREA)
    return wrist


def _group_energy(acts: dict[str, dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for path, s in acts.items():
        g = path.split(".", 1)[0]
        out[g] = out.get(g, 0.0) + float(s.get("l2", 0.0))
    return {k: _r(v) for k, v in out.items()}


def _build_policy(ckpt: Path, norm: Path, device: str, period: int) -> MicroVLAPolicy:
    return MicroVLAPolicy(
        checkpoint=str(ckpt),
        norm_stats=str(norm),
        perception_period=period,
        device=device,
        heads_device=device,
        no_brake=True,
        det_conf=0.02,
        role_disjoint_iou=0.1,
        source_max_area=0.12,
        ibvs_phase=False,
        tool_phase=False,
    )


def _tick_pack(policy: MicroVLAPolicy, bank, action: np.ndarray) -> dict[str, Any]:
    loop = policy.loop
    acts = dict(bank.latest)
    acts["corrector"] = corrector_act(loop)
    tel = policy.telemetry[-1] if policy.telemetry else {}
    row = [_r(float(v)) for v in np.asarray(action).reshape(-1)[:7].tolist()]
    src_uv = tel.get("src_center")
    return {
        "is_real": bool(tel.get("is_real", False)),
        "trust": _r(float(tel.get("trust", getattr(loop.corrector, "trust", 0.0)))),
        "plan_norm": _r(float(tel.get("plan_norm", 0.0))),
        "action": row,
        "grip": _r(float(row[-1])) if row else 0.0,
        "src_conf": _r(float(tel["src_conf"])) if "src_conf" in tel else None,
        "tgt_conf": _r(float(tel["tgt_conf"])) if "tgt_conf" in tel else None,
        "src_center": [_r(float(u)) for u in src_uv] if src_uv is not None else None,
        "acts": acts,
        "group_e": _group_energy(acts),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--norm-stats", type=Path, default=DEFAULT_NORM)
    p.add_argument("--compare", type=Path, default=DEFAULT_COMPARE)
    p.add_argument("--compare-norm", type=Path, default=DEFAULT_COMPARE_NORM)
    p.add_argument("--no-compare", action="store_true")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--perception-period", type=int, default=2)
    p.add_argument("--stride", type=int, default=1, help="Keep every Nth video frame")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--instruction", default=INSTRUCTION)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"missing video: {args.video}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    if not args.norm_stats.is_file():
        raise SystemExit(f"missing norm stats: {args.norm_stats}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_name = "demo.mp4"
    dest_video = args.out_dir / video_name
    # Browser-playable H.264 (OpenCV mp4v often won't play in Chrome/Safari).
    import subprocess
    rc = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(args.video),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", str(dest_video),
        ],
        capture_output=True,
        text=True,
    )
    if rc.returncode != 0:
        print("ffmpeg h264 encode failed; copying source as-is", flush=True)
        if dest_video.resolve() != args.video.resolve():
            shutil.copy2(args.video, dest_video)

    print(f"loading primary {args.checkpoint.name} on {args.device}", flush=True)
    primary = _build_policy(args.checkpoint, args.norm_stats, args.device, args.perception_period)
    bank = attach_policy_hooks(primary)
    modules = list(bank.modules)
    primary.reset(args.instruction)

    compare = None
    if not args.no_compare and args.compare.is_file() and args.compare_norm.is_file():
        print(f"loading compare {args.compare.name}", flush=True)
        compare = _build_policy(args.compare, args.compare_norm, args.device, args.perception_period)
        compare.reset(args.instruction)

    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ticks: list[dict] = []
    fi = 0
    kept = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if fi % args.stride != 0:
            fi += 1
            continue
        wrist_bgr = _wrist_crop(bgr)
        wrist_rgb = np.ascontiguousarray(wrist_bgr[..., ::-1])

        bank.clear_latest()
        act_p = primary.act(wrist_rgb, proprio=None)
        pack = _tick_pack(primary, bank, act_p)
        pack["t"] = kept
        pack["frame"] = fi
        pack["time"] = _r(fi / fps)

        if compare is not None:
            act_c = compare.act(wrist_rgb, proprio=None)
            tel_c = compare.telemetry[-1] if compare.telemetry else {}
            row_c = [_r(float(v)) for v in np.asarray(act_c).reshape(-1)[:7].tolist()]
            pack["compare"] = {
                "action": row_c,
                "grip": _r(float(row_c[-1])) if row_c else 0.0,
                "trust": _r(float(tel_c.get("trust", 0.0))),
                "plan_norm": _r(float(tel_c.get("plan_norm", 0.0))),
                "is_real": bool(tel_c.get("is_real", False)),
            }
            pack["action_l1"] = _r(float(np.abs(
                np.asarray(pack["action"]) - np.asarray(row_c)).sum()))

        ticks.append(pack)
        kept += 1
        if kept % 10 == 0 or kept == 1:
            print(
                f"  frame {fi}/{n_total} tick {kept} "
                f"grip={pack['grip']:.2f} trust={pack['trust']:.2f} "
                f"src={pack.get('src_conf')}",
                flush=True,
            )
        fi += 1
        if args.max_frames and kept >= args.max_frames:
            break
    cap.release()
    bank.close()

    grips = [t["grip"] for t in ticks]
    close_frac = float(np.mean([1.0 if g < 0.0 else 0.0 for g in grips])) if grips else 0.0

    def _rel(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    pack_out = {
        "task": args.instruction,
        "video": video_name,
        "fps": fps,
        "n_frames_video": n_total,
        "n_ticks": len(ticks),
        "stride": args.stride,
        "pane": "wrist=right_half",
        "control": "open_loop_demo_obs",
        "unaided": True,
        "primary": {
            "checkpoint": _rel(args.checkpoint),
            "norm_stats": args.norm_stats.name,
            "grip_close_frac": _r(close_frac),
            "mean_plan_norm": _r(float(np.mean([t["plan_norm"] for t in ticks] or [0]))),
        },
        "compare": None if compare is None else {
            "checkpoint": _rel(args.compare),
            "norm_stats": args.compare_norm.name,
        },
        "modules": modules,
        "group_order": GROUP_ORDER,
        "servo_names": ["dx", "dy", "dz", "dax", "day", "daz", "grip"],
        "ticks": ticks,
        "note": (
            "Open-loop on demo wrist frames — the MP4 has no logged actions, "
            "so choice difference is primary vs compare checkpoint on the "
            "same observations (and vs what the demo visually does)."
        ),
    }
    latest = args.out_dir / "latest.json"
    trace = args.out_dir / "trace.json"
    latest.write_text(json.dumps({
        "task": pack_out["task"],
        "video": video_name,
        "fps": fps,
        "n_ticks": len(ticks),
        "trace": "trace.json",
        "primary": pack_out["primary"],
        "compare": pack_out["compare"],
        "control": pack_out["control"],
        "unaided": True,
        "note": pack_out["note"],
    }, indent=2))
    trace.write_text(json.dumps(pack_out))
    print(f"wrote {latest} and {trace} ({len(ticks)} ticks)", flush=True)


if __name__ == "__main__":
    main()
