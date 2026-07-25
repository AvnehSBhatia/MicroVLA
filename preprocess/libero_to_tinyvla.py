"""LIBERO hdf5 -> TinyVLA/ACT-style episode files (teacher-training data).

TinyVLA's trainer (github.com/liyaxuanliyaxuan/TinyVLA) consumes ALOHA-style
per-episode hdf5s:

    episode_N.hdf5
      |- action            [T, A]
      |- language_raw      (1,)  utf-8 task instruction
      |- observations/
          |- images/<cam>  [T, H, W, 3] uint8   (per camera_names in their
          |                                      aloha_scripts/constants.py)
          |- joint_positions [T, 7]
          |- qpos            [T, 7]
          |- qvel            [T, 7]

This converter maps LIBERO demos onto that layout:
  * cameras: ``front`` = agentview (rotated 180, robosuite quirk),
             ``wrist`` = eye_in_hand — matching their two-view eval input.
  * qpos/joint_positions = LIBERO ``joint_states`` (7-DoF Panda);
    qvel = zeros (LIBERO stores no joint velocities).
  * action = LIBERO's native 7-dim (dx dy dz droll dpitch dyaw grip).
    ``--action-10d`` converts to the Franka 10-dim convention
    (xyz + 6D rotation + grip) if their head insists on action_dim=10.
  * language_raw from problem_info / filename (same logic as the main bake).

Disk math before you run it: eps x T x cams x H x W x 3 bytes. All three
LIBERO suites (~1500 demos, T~150-250, 128px, 2 cams) ~ 20-30 GB raw, ~2-3x
less with gzip (used here). Use --demos-per-task and --cameras to stay inside
budget; the tool prints its running size and the estimate up front.

    python -m preprocess.libero_to_tinyvla /root/libero_raw /root/tinyvla_data/libero_all \
        --demos-per-task 25 --cameras front wrist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess.libero import _demo_sort_key, _instruction_for_file  # noqa: E402


def _euler_to_6d(rpy: np.ndarray) -> np.ndarray:
    """Delta-euler [T, 3] -> 6D rotation representation [T, 6] (first two
    columns of the rotation matrix, row-major) — the Franka convention TinyVLA's
    droid_diffusion head was built around."""
    cr, sr = np.cos(rpy[:, 0]), np.sin(rpy[:, 0])
    cp, sp = np.cos(rpy[:, 1]), np.sin(rpy[:, 1])
    cy, sy = np.cos(rpy[:, 2]), np.sin(rpy[:, 2])
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll); take columns 0 and 1.
    r00 = cy * cp
    r10 = sy * cp
    r20 = -sp
    r01 = cy * sp * sr - sy * cr
    r11 = sy * sp * sr + cy * cr
    r21 = cp * sr
    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1).astype(np.float32)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", help="dir with LIBERO *.hdf5 (already downloaded)")
    ap.add_argument("out", help="output dir for episode_N.hdf5 files")
    ap.add_argument("--cameras", nargs="+", default=["front", "wrist"],
                    choices=["front", "wrist"],
                    help="camera streams to store (front=agentview, wrist=eye_in_hand)")
    ap.add_argument("--demos-per-task", type=int, default=None,
                    help="cap demos taken per hdf5 task file (disk budget lever)")
    ap.add_argument("--action-10d", action="store_true",
                    help="emit Franka 10-dim actions (xyz + 6D rot + grip) instead of 7")
    args = ap.parse_args(argv)

    import h5py  # lazy heavy dep

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [root] if root.is_file() else sorted(root.rglob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no .hdf5 under {root}")

    cam_src = {"front": "agentview_rgb", "wrist": "eye_in_hand_rgb"}
    n_out = 0
    bytes_out = 0
    for h5path in files:
        with h5py.File(h5path, "r") as f:
            instruction = _instruction_for_file(f, h5path.stem)
            demos = sorted(f["data"].keys(), key=_demo_sort_key)
            if args.demos_per_task is not None:
                demos = demos[: args.demos_per_task]
            for demo in demos:
                grp = f["data"][demo]
                obs = grp["obs"]
                actions = np.asarray(grp["actions"], dtype=np.float32)  # [T, 7]
                T = actions.shape[0]
                if args.action_10d:
                    act = np.concatenate(
                        [actions[:, :3], _euler_to_6d(actions[:, 3:6]), actions[:, 6:7]],
                        axis=-1,
                    )
                else:
                    act = actions
                joints = (np.asarray(obs["joint_states"], dtype=np.float32)[:T]
                          if "joint_states" in obs else np.zeros((T, 7), np.float32))

                dst = out / f"episode_{n_out}.hdf5"
                with h5py.File(dst, "w") as g:
                    g.create_dataset("action", data=act)
                    g.create_dataset(
                        "language_raw", data=np.array([instruction], dtype=object),
                        dtype=h5py.string_dtype("utf-8"),
                    )
                    o = g.create_group("observations")
                    for name in ("joint_positions", "qpos"):
                        o.create_dataset(name, data=joints)
                    o.create_dataset("qvel", data=np.zeros_like(joints))
                    imgs = o.create_group("images")
                    for cam in args.cameras:
                        frames = np.asarray(obs[cam_src[cam]])[:T]
                        if cam == "front":
                            frames = frames[:, ::-1, ::-1]  # robosuite agentview flip
                        imgs.create_dataset(
                            cam, data=np.ascontiguousarray(frames),
                            compression="gzip", compression_opts=4,
                            chunks=(1, *frames.shape[1:]),
                        )
                bytes_out += dst.stat().st_size
                n_out += 1
        print(f"[{h5path.stem}] total episodes {n_out}, {bytes_out / 1e9:.2f} GB", flush=True)

    print(f"\ndone: {n_out} episodes -> {out} ({bytes_out / 1e9:.2f} GB)")
    print("add to TinyVLA aloha_scripts/constants.py:\n"
          f"  'libero_all': {{'dataset_dir': '{out}', 'episode_len': 400,\n"
          f"                 'camera_names': {args.cameras!r}}}")


if __name__ == "__main__":
    main()
