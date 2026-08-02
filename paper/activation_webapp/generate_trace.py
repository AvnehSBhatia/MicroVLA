"""Dump a 30s (900-tick) per-micro-module activation trace for the webapp.

Loads ``full_stageB_rec_fix.pt`` with mock perception (CPU, no LIBERO/cv2),
registers forward hooks on EVERY nn.Module under fusion/drift/trm/planner/
tqsa/relational, runs a synthetic tabletop episode, and writes a compressed
JSON bundle the static webapp can scrub through.

Usage::

    .venv/bin/python paper/activation_webapp/generate_trace.py
    .venv/bin/python paper/activation_webapp/generate_trace.py --ticks 30
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.policy import MicroVLAPolicy  # noqa: E402
from microvla.perception.text_encoder import MockTaskEncoder  # noqa: E402
from microvla.perception.yolo_world import MockYoloWorldPerception  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "data"
TOPK = 8
ROUND = 4

# Skip pure containers — they have no useful activation of their own.
_SKIP_TYPES = (nn.ModuleList, nn.ModuleDict)


def _r(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return round(float(x), ROUND)


def _summarize(t: torch.Tensor) -> dict[str, Any]:
    """Compact activation summary — never store full tensors."""
    x = t.detach().float().reshape(-1)
    if x.numel() == 0:
        return {"n": 0, "l2": 0.0, "mean": 0.0, "absmax": 0.0, "sat": 0.0, "top": []}
    n = int(x.numel())
    absx = x.abs()
    # tanh/gelu saturation proxies
    sat = float(((absx > 0.97) | (absx < 1e-6)).float().mean())
    # top-|activation| values (magnitude-sorted, keep sign)
    k = min(TOPK, n)
    _, idx = absx.topk(k)
    top = [_r(float(x[i])) for i in idx.tolist()]
    return {
        "n": n,
        "l2": _r(float(x.norm())),
        "mean": _r(float(x.mean())),
        "absmax": _r(float(absx.max())),
        "sat": _r(sat),
        "top": top,
    }


def _mat(t: torch.Tensor) -> list:
    return [[_r(float(v)) for v in row] for row in t.detach().float().cpu().tolist()]


def _vec(t: torch.Tensor) -> list:
    return [_r(float(v)) for v in t.detach().float().reshape(-1).cpu().tolist()]


class HookBank:
    """Collects one-forward summaries keyed by ``group.name`` paths."""

    def __init__(self) -> None:
        self.latest: dict[str, dict] = {}
        self._handles: list[Any] = []
        self.modules: list[dict] = []  # static graph entries

    def attach(self, group: str, root: nn.Module) -> None:
        for name, mod in root.named_modules():
            if isinstance(mod, _SKIP_TYPES):
                continue
            path = f"{group}.{name}" if name else group
            n_params = sum(p.numel() for p in mod.parameters(recurse=False))
            self.modules.append({
                "id": path,
                "group": group,
                "name": name or group,
                "type": type(mod).__name__,
                "params": int(n_params),
                "leaf": len(list(mod.children())) == 0,
                "parent": (f"{group}.{name.rsplit('.', 1)[0]}" if name and "." in name
                           else (group if name else None)),
            })

            def _make(key: str):
                def _hook(_m, _inp, out):
                    tens = None
                    if torch.is_tensor(out):
                        tens = out
                    elif isinstance(out, (tuple, list)):
                        for o in out:
                            if torch.is_tensor(o):
                                tens = o
                                break
                    elif isinstance(out, dict):
                        for o in out.values():
                            if torch.is_tensor(o):
                                tens = o
                                break
                    if tens is not None:
                        self.latest[key] = _summarize(tens)
                return _hook

            self._handles.append(mod.register_forward_hook(_make(path)))

    def clear_latest(self) -> None:
        self.latest = {}

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def build_policy(checkpoint: Path, norm_stats: Path) -> tuple[MicroVLAPolicy, str]:
    """Load rec_fix when present; else fresh mock weights + identity norm.

    Returns:
        ``(policy, weights_tag)`` where ``weights_tag`` is ``"rec_fix"`` or
        ``"mock"`` (written into ``meta.weights`` for the webapp).
    """
    identity = ROOT / "eval" / "identity_norm_stats.json"
    if checkpoint.exists() and norm_stats.exists():
        pol = MicroVLAPolicy(
            checkpoint=str(checkpoint),
            norm_stats=str(norm_stats),
            device="cpu",
            perception=MockYoloWorldPerception(),
            task_encoder=MockTaskEncoder(),
            perception_period=15,
            chunk_exec=False,
        )
        return pol, "rec_fix"
    print(f"checkpoint/norm missing ({checkpoint.exists()=}, {norm_stats.exists()=}); "
          f"falling back to untrained mock weights + identity norm")
    pol = MicroVLAPolicy(
        checkpoint="none",
        norm_stats=str(identity),
        device="cpu",
        perception=MockYoloWorldPerception(),
        task_encoder=MockTaskEncoder(),
        perception_period=15,
        chunk_exec=False,
    )
    return pol, "mock"


class TableSim:
    """Tiny kinematic tabletop used both for proprio and the webapp scene."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        # LIBERO-ish object layout (meters, table at z=0)
        self.objects = {
            "cream_cheese": {
                "pos": [0.05 + 0.02 * rng.randn(), -0.08 + 0.02 * rng.randn(), 0.035],
                "size": [0.06, 0.04, 0.07],
                "color": "#f2f0e6",
                "role": "source",
            },
            "milk": {
                "pos": [-0.10 + 0.02 * rng.randn(), -0.06 + 0.02 * rng.randn(), 0.08],
                "size": [0.05, 0.05, 0.16],
                "color": "#f4f4f4",
                "role": "distractor",
            },
            "tomato_sauce": {
                "pos": [0.12 + 0.02 * rng.randn(), 0.02 + 0.02 * rng.randn(), 0.05],
                "size": [0.04, 0.04, 0.10],
                "color": "#c23b22",
                "role": "distractor",
            },
            "basket": {
                "pos": [-0.05 + 0.02 * rng.randn(), 0.14 + 0.02 * rng.randn(), 0.04],
                "size": [0.14, 0.10, 0.08],
                "color": "#c4a574",
                "role": "target",
            },
        }
        self.eef = np.array([0.0, -0.22, 0.22], dtype=np.float64)
        self.eef_rpy = np.zeros(3, dtype=np.float64)
        self.grip = 1.0  # +1 open, -1 closed
        self.held: str | None = None
        self.table_z = 0.0
        self.trail: list[list[float]] = []

    def proprio(self) -> np.ndarray:
        # [x,y,z, qx,qy,qz,qw-ish, jaw_l, jaw_r, valid]
        j = 0.04 * max(0.0, self.grip)  # open width proxy
        return np.array([
            self.eef[0], self.eef[1], self.eef[2],
            self.eef_rpy[0], self.eef_rpy[1], self.eef_rpy[2], 1.0,
            j, j, 1.0,
        ], dtype=np.float32)

    def step(self, action: np.ndarray, dt: float = 1.0 / 30.0) -> dict:
        # Delta-style action on a LIBERO-sized table (~0.7×0.55 m).
        # Gain chosen so typical |a|~0.1 moves ~2–4 cm/s, not meters/s.
        a = np.asarray(action[:3], dtype=np.float64)
        self.eef = self.eef + 0.04 * a
        self.eef[0] = float(np.clip(self.eef[0], -0.30, 0.30))
        self.eef[1] = float(np.clip(self.eef[1], -0.28, 0.22))
        self.eef[2] = float(np.clip(self.eef[2], 0.03, 0.42))
        self.eef_rpy = self.eef_rpy + 0.15 * np.asarray(action[3:6], dtype=np.float64) * dt
        self.grip = float(np.clip(action[6], -1.0, 1.0))

        # Crude grasp: close near cream cheese
        src = self.objects["cream_cheese"]
        d = np.linalg.norm(self.eef - np.asarray(src["pos"]))
        if self.held is None and self.grip < -0.5 and d < 0.06:
            self.held = "cream_cheese"
        if self.held is not None:
            self.objects[self.held]["pos"] = [
                float(self.eef[0]), float(self.eef[1]), float(max(0.03, self.eef[2] - 0.03))
            ]
            # Place into basket
            bask = np.asarray(self.objects["basket"]["pos"])
            if self.grip > 0.3 and np.linalg.norm(self.eef[:2] - bask[:2]) < 0.08:
                self.objects[self.held]["pos"] = [
                    float(bask[0]), float(bask[1]), float(bask[2] + 0.03)
                ]
                self.held = None

        self.trail.append([_r(float(v)) for v in self.eef])
        if len(self.trail) > 900:
            self.trail = self.trail[-900:]

        src_c = self._uv(self.objects["cream_cheese"]["pos"])
        tgt_c = self._uv(self.objects["basket"]["pos"])
        return {
            "eef": [_r(float(v)) for v in self.eef],
            "eef_rpy": [_r(float(v)) for v in self.eef_rpy],
            "grip": _r(self.grip),
            "held": self.held,
            "objects": {k: {"pos": [_r(float(v)) for v in v["pos"]],
                            "size": v["size"], "color": v["color"], "role": v["role"]}
                        for k, v in self.objects.items()},
            "src_uv": src_c,
            "tgt_uv": tgt_c,
            "dist_src": _r(float(d)),
            "success": bool(
                self.held is None
                and np.linalg.norm(
                    np.asarray(self.objects["cream_cheese"]["pos"][:2])
                    - np.asarray(self.objects["basket"]["pos"][:2])
                ) < 0.09
            ),
        }

    def _uv(self, pos) -> list[float]:
        """Crude wrist-camera projection of a world point → [0,1]^2."""
        rel = np.asarray(pos, dtype=np.float64) - self.eef
        # look mostly -Y (toward table center from start)
        u = 0.5 + 1.6 * rel[0] / max(0.05, -rel[1] + 0.15)
        v = 0.5 - 1.6 * rel[2] / max(0.05, -rel[1] + 0.15)
        return [_r(float(np.clip(u, 0.0, 1.0))), _r(float(np.clip(v, 0.0, 1.0)))]

    def render_frame(self, t: int) -> np.ndarray:
        """Synthetic RGB wrist view (hash-stable content via sim state)."""
        h = w = 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # table gradient
        for row in range(h):
            img[row, :, :] = (40 + row // 4, 55 + row // 5, 48)
        # draw objects as rectangles from UV
        for name, obj in self.objects.items():
            u, v = self._uv(obj["pos"])
            cx, cy = int(u * (w - 1)), int(v * (h - 1))
            half = 8 if name != "basket" else 14
            color = {
                "cream_cheese": (230, 225, 200),
                "milk": (240, 240, 240),
                "tomato_sauce": (200, 60, 40),
                "basket": (180, 140, 80),
            }[name]
            img[max(0, cy - half):min(h, cy + half),
                max(0, cx - half):min(w, cx + half)] = color
        # crosshair
        img[h // 2 - 1:h // 2 + 1, :] = np.minimum(img[h // 2 - 1:h // 2 + 1, :] + 30, 255)
        img[:, w // 2 - 1:w // 2 + 1] = np.minimum(img[:, w // 2 - 1:w // 2 + 1] + 30, 255)
        # tick salt so mock hashes evolve
        img[0, t % w, 0] = (t * 17) % 255
        return img


def planner_sensitivity(loop, result, last_kw: dict) -> dict[str, float]:
    """Withhold each planner channel; report pose |Δplan| share."""
    planner = loop.planner
    if not last_kw:
        return {}
    def _plan(out):
        return out[0] if isinstance(out, (tuple, list)) else out

    next_emb = result.next_emb.unsqueeze(0)
    # Don't request aux/wp — we only need the plan tensor.
    last_kw = {k: v for k, v in last_kw.items()
               if k not in ("return_aux", "return_wp", "fade")}
    base = _plan(planner(next_emb, **last_kw))
    base_pose = base[..., :-1]
    scores: dict[str, float] = {}
    probes = ["fused", "current_emb", "state_delta", "geometry", "proprio",
              "pred_box_emb", "wm_msg", "wm_latent", "spatial", "relational"]
    with torch.no_grad():
        for name in probes:
            if last_kw.get(name) is None:
                continue
            alt = _plan(planner(next_emb, **{**last_kw, name: None}))
            scores[name] = _r(float((alt[..., :-1] - base_pose).abs().mean()))
        # persistence substitute
        cur = last_kw.get("current_emb")
        if cur is not None:
            alt = _plan(planner(cur, **last_kw))
            scores["next_emb"] = _r(float((alt[..., :-1] - base_pose).abs().mean()))
    total = sum(scores.values()) + 1e-12
    return {k: _r(v / total) for k, v in scores.items()}


def capture_planner_kwargs(loop) -> dict:
    """Re-read the last-tick planner inputs the loop still holds."""
    # JEPALoop stores several private held tensors we can reuse for sensitivity.
    kw = {}
    for attr, key in (
        ("_last_state_delta", "state_delta"),
        ("_last_spatial", "spatial"),
        ("_last_proprio", "proprio"),
    ):
        v = getattr(loop, attr, None)
        if v is not None:
            kw[key] = v if key != "spatial" else v
    return kw


def run(ticks: int, checkpoint: Path, norm_stats: Path, out: Path) -> None:
    logging.disable(logging.WARNING)
    torch.manual_seed(0)
    np.random.seed(0)

    pol, weights_tag = build_policy(checkpoint, norm_stats)
    loop = pol.loop
    cfg = pol.cfg
    bank = HookBank()
    for group, mod in (
        ("fusion", loop.fusion),
        ("drift", loop.drift),
        ("trm", loop.trm),
        ("planner", loop.planner),
    ):
        bank.attach(group, mod)
    if loop.tqsa is not None:
        bank.attach("tqsa", loop.tqsa)
    if loop.relational is not None:
        bank.attach("relational", loop.relational)

    # Also record corrector as a synthetic "module"
    bank.modules.append({
        "id": "corrector", "group": "corrector", "name": "corrector",
        "type": "InnovationCorrector", "params": 0, "leaf": True, "parent": None,
    })

    sim = TableSim(seed=0)
    text = "pick up the cream cheese and place it in the basket"
    loop.set_task(text)

    tick_rows: list[dict] = []
    # wrap planner to steal kwargs each forward
    planner_kw_box: dict[str, Any] = {"kw": {}}
    _orig_planner = loop.planner.forward

    def _planner_wrap(*args, **kwargs):
        planner_kw_box["kw"] = {k: v for k, v in kwargs.items()}
        return _orig_planner(*args, **kwargs)

    loop.planner.forward = _planner_wrap  # type: ignore[method-assign]

    period = int(getattr(pol, "perception_period", None) or cfg.tick_hz // cfg.real_frame_hz)
    for t in range(ticks):
        bank.clear_latest()
        prop = sim.proprio()
        frame = sim.render_frame(t)
        is_real = (t % period == 0)
        frame_bgr = frame[..., ::-1].copy() if is_real else None
        result = loop.tick(frame_bgr, proprio=prop)

        # Execute plan row 0 in the toy sim (denormalized via policy normalizer)
        plan0 = result.plan[0].detach().cpu().numpy()
        raw = pol.normalizer.inverse(plan0.astype(np.float32))
        scene = sim.step(raw)

        acts = dict(bank.latest)
        # corrector synthetic activation
        corr = loop.corrector
        c_norm = float(corr.c.norm()) if getattr(corr, "c", None) is not None else 0.0
        acts["corrector"] = {
            "n": 1,
            "l2": _r(c_norm),
            "mean": _r(float(corr.trust)),
            "absmax": _r(c_norm),
            "sat": 0.0,
            "top": [_r(float(corr.trust)), _r(c_norm),
                    _r(float(getattr(corr, "k", 0))),
                    _r(float(getattr(corr, "err_bar", 0.0) or 0.0))],
        }

        sens = {}
        if is_real and planner_kw_box["kw"]:
            # Ensure required tensors from TickResult are present
            kw = dict(planner_kw_box["kw"])
            kw.setdefault("current_emb", result.latent.unsqueeze(0))
            kw.setdefault("state_delta", result.state_delta.unsqueeze(0))
            kw.setdefault("fused", result.fused.unsqueeze(0))
            sens = planner_sensitivity(loop, result, kw)

        # group energy for DAG node fill
        group_e: dict[str, float] = {}
        for mid, s in acts.items():
            g = mid.split(".", 1)[0]
            group_e[g] = group_e.get(g, 0.0) + float(s.get("l2", 0.0))

        row = {
            "t": t,
            "t_s": _r(t / float(cfg.tick_hz)),
            "is_real": bool(result.is_real),
            "trust": _r(float(result.trust)),
            "plan": _mat(result.plan),
            "fused_row_l2": [_r(float(r.norm())) for r in result.fused],
            "fused": _mat(result.fused) if is_real else None,  # full matrix on real only
            "latent_l2": _r(float(result.latent.norm())),
            "next_l2": _r(float(result.next_emb.norm())),
            "state_l2": _r(float(result.state_delta.norm())),
            "action": [_r(float(v)) for v in raw.tolist()],
            "plan0": [_r(float(v)) for v in plan0.tolist()],
            "acts": acts,
            "group_e": {k: _r(v) for k, v in group_e.items()},
            "sens": sens,
            "scene": scene,
            "src_conf": (None if result.perception is None
                         else _r(float(result.perception.source.confidence))),
            "tgt_conf": (None if result.perception is None
                         else _r(float(result.perception.target.confidence))),
            "src_center": (None if result.perception is None
                           else _vec(result.perception.source.center)),
            "tgt_center": (None if result.perception is None
                           else _vec(result.perception.target.center)),
        }
        tick_rows.append(row)
        if (t + 1) % 100 == 0 or t == 0:
            print(f"  tick {t + 1}/{ticks}  trust={row['trust']:.3f}  "
                  f"modules_fired={len(acts)}  dist_src={scene['dist_src']:.3f}")

    loop.planner.forward = _orig_planner  # type: ignore[method-assign]
    bank.close()

    # Static weight snapshot (param L2 per leaf)
    weight_stats = []
    for group, mod in (
        ("fusion", loop.fusion), ("drift", loop.drift), ("trm", loop.trm),
        ("planner", loop.planner), ("tqsa", loop.tqsa), ("relational", loop.relational),
    ):
        if mod is None:
            continue
        for name, p in mod.named_parameters():
            flat = p.detach().float().reshape(-1)
            std = float(flat.std(unbiased=False)) if flat.numel() else 0.0
            weight_stats.append({
                "id": f"{group}.{name}",
                "group": group,
                "numel": int(flat.numel()),
                "l2": _r(float(flat.norm())),
                "mean": _r(float(flat.mean()) if flat.numel() else 0.0),
                "std": _r(std),
                "absmax": _r(float(flat.abs().max()) if flat.numel() else 0.0),
                "sparsity": _r(float((flat.abs() < 1e-3).float().mean()) if flat.numel() else 0.0),
            })

    # Mean sensitivity across real ticks
    sens_keys = sorted({k for r in tick_rows for k in r["sens"]})
    sens_mean = {
        k: _r(float(np.mean([r["sens"][k] for r in tick_rows if k in r["sens"]] or [0.0])))
        for k in sens_keys
    }

    ckpt_label = (
        str(checkpoint.relative_to(ROOT)) if checkpoint.exists() and checkpoint.is_relative_to(ROOT)
        else ("none" if weights_tag == "mock" else str(checkpoint))
    )
    bundle = {
        "meta": {
            "checkpoint": ckpt_label,
            "weights": weights_tag,
            "ticks": ticks,
            "tick_hz": int(cfg.tick_hz),
            "perception_period": period,
            "duration_s": _r(ticks / float(cfg.tick_hz)),
            "text": text,
            "device": "cpu",
            "n_modules": len(bank.modules),
            "n_weight_tensors": len(weight_stats),
            "note": (
                "Contribution scores are planner input-withhold shares "
                "(plan attribution), not LIBERO task success."
            ),
            "shapes": {
                "vis_dim": cfg.vis_dim,
                "fused": [cfg.fused_rows, cfg.fused_cols],
                "plan": [cfg.plan_steps, cfg.num_servos],
                "state_dim": cfg.state_dim,
            },
            "flow": [
                "perception", "fusion", "drift", "trm", "corrector",
                "tqsa", "relational", "planner", "plan",
            ],
        },
        "modules": bank.modules,
        "weights": weight_stats,
        "sens_mean": sens_mean,
        "ticks": tick_rows,
        "trail": sim.trail,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    # Write compact JSON
    with out.open("w") as f:
        json.dump(bundle, f, separators=(",", ":"))
    mb = out.stat().st_size / (1024 * 1024)
    print(f"wrote {out} ({mb:.2f} MB, {len(bank.modules)} modules, {ticks} ticks)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticks", type=int, default=900)
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "checkpoints" / "full_stageB_rec_fix.pt")
    p.add_argument("--norm-stats", type=Path,
                   default=ROOT / "data" / "libero_object_grid" / "norm_stats.json")
    p.add_argument("--out", type=Path, default=OUT_DIR / "trace.json")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.ticks, args.checkpoint, args.norm_stats, args.out)
