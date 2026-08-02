"""Shared forward-hook bank for activation traces (mock + real film)."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

TOPK = 8
ROUND = 4
_SKIP_TYPES = (nn.ModuleList, nn.ModuleDict)


def _r(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return round(float(x), ROUND)


def _summarize(t: torch.Tensor) -> dict[str, Any]:
    x = t.detach().float().reshape(-1)
    if x.numel() == 0:
        return {"n": 0, "l2": 0.0, "mean": 0.0, "absmax": 0.0, "sat": 0.0, "top": []}
    n = int(x.numel())
    absx = x.abs()
    sat = float(((absx > 0.97) | (absx < 1e-6)).float().mean())
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
        self.modules: list[dict] = []

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


def attach_policy_hooks(policy) -> HookBank:
    """Hook fusion/drift/TRM/planner/(tqsa)/(relational) on a MicroVLAPolicy."""
    loop = policy.loop
    bank = HookBank()
    for group, mod in (
        ("fusion", loop.fusion),
        ("drift", loop.drift),
        ("trm", loop.trm),
        ("planner", loop.planner),
    ):
        bank.attach(group, mod)
    if getattr(loop, "tqsa", None) is not None:
        bank.attach("tqsa", loop.tqsa)
    if getattr(loop, "relational", None) is not None:
        bank.attach("relational", loop.relational)
    bank.modules.append({
        "id": "corrector", "group": "corrector", "name": "corrector",
        "type": "InnovationCorrector", "params": 0, "leaf": True, "parent": None,
    })
    bank.modules.append({
        "id": "ibvs_phase", "group": "ibvs", "name": "PhasedIBVS",
        "type": "PhasedIBVS", "params": 0, "leaf": True, "parent": None,
    })
    return bank


def corrector_act(loop) -> dict:
    corr = loop.corrector
    c_norm = float(corr.c.norm()) if getattr(corr, "c", None) is not None else 0.0
    return {
        "n": 1,
        "l2": _r(c_norm),
        "mean": _r(float(corr.trust)),
        "absmax": _r(c_norm),
        "sat": 0.0,
        "top": [
            _r(float(corr.trust)),
            _r(c_norm),
            _r(float(getattr(corr, "k", 0))),
            _r(float(getattr(corr, "err_bar", 0.0) or 0.0)),
        ],
    }
