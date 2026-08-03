"""LoRA adapters for conv stages of a frozen perception backbone.

Motivation (unaided distillation rounds 2–5, paper §6.4d): the policy fits
the teacher on-distribution (val BC ~0.044, val grip agreement ~1.0) yet
never closes at the right instant at eval. The suspect is not parameter
capacity downstream but the INPUT: the frame embedding is a global average
pool of the frozen SPPF map, and "the object is between the jaws NOW" is a
spatial configuration the frozen features were never trained to make
GAP-visible. LoRA gives the embedding path a small trainable subspace so
BC gradients can reshape what the embedding encodes.

Design constraints, in order:

1. **Detection must not move.** Boxes, confidences and role binding are the
   calibrated, measured part of the stack (assisted 0.75 ceiling). Adapters
   therefore never touch the live detector: the consumer deep-copies the
   stage it wants (SPPF) and adapts the COPY, fed from the original's
   captured input. The frozen path stays bit-identical.
2. **Identity at init.** ``up`` is zero-initialized, so an adapted stage
   starts exactly equal to its frozen source and drifts only under
   gradient — training begins from the distribution every downstream
   module (fusion, frozen TRM) was trained on.
3. **CPU/mock testable.** Pure torch; no ultralytics import. Works on any
   ``nn.Conv2d`` (1x1 SPPF convs are the intended targets, where LoRA is
   exactly a low-rank linear).
"""
from __future__ import annotations

import copy
from typing import Iterator, Tuple

import torch
from torch import nn


class LoRAConv2d(nn.Module):
    """A frozen ``nn.Conv2d`` plus a parallel low-rank trainable branch.

    ``y = conv(x) + (alpha / r) * up(down(x))`` with ``down`` a 1x1 conv to
    ``r`` channels (Kaiming init) and ``up`` a 1x1 conv from ``r`` channels
    zero-initialized — the module is exactly ``conv`` at init.

    The low-rank branch is intentionally 1x1 regardless of the wrapped
    kernel: it adapts channel mixing (what GAP can see), not spatial
    receptive fields, and 1x1 keeps the adapter parameter count
    ``r * (c_in + c_out)``.
    """

    def __init__(self, conv: nn.Conv2d, r: int, alpha: float = 1.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad_(False)
        self.scale = float(alpha) / float(r)
        self.down = nn.Conv2d(conv.in_channels, r, kernel_size=1, bias=False)
        self.up = nn.Conv2d(r, conv.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.conv(x)
        add = self.up(self.down(x))
        if add.shape[-2:] != base.shape[-2:]:
            # Wrapped conv has stride/padding that changed resolution; match
            # it. SPPF convs are stride-1 so this is a safety net, not a path.
            add = nn.functional.adaptive_avg_pool2d(add, base.shape[-2:])
        return base + self.scale * add


def _iter_conv2d(module: nn.Module) -> Iterator[Tuple[nn.Module, str, nn.Conv2d]]:
    for parent in module.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Conv2d):
                yield parent, name, child


def clone_with_lora(stage: nn.Module, r: int, alpha: float = 1.0) -> nn.Module:
    """Deep-copies ``stage`` and wraps every ``nn.Conv2d`` in :class:`LoRAConv2d`.

    The original stays untouched (constraint 1). The clone's non-LoRA
    parameters are frozen; only ``down``/``up`` weights train. BatchNorm
    modules in the clone are put in eval mode and their stats frozen — the
    clone must not drift from the frozen source through running statistics.
    """
    adapted = copy.deepcopy(stage)
    n = 0
    for parent, name, conv in list(_iter_conv2d(adapted)):
        if isinstance(parent, LoRAConv2d):
            continue
        setattr(parent, name, LoRAConv2d(conv, r=r, alpha=alpha))
        n += 1
    if n == 0:
        raise ValueError("clone_with_lora found no nn.Conv2d to adapt")
    for p in adapted.parameters():
        p.requires_grad_(False)
    for m in adapted.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            m.track_running_stats = False
        elif isinstance(m, LoRAConv2d):
            for p in m.down.parameters():
                p.requires_grad_(True)
            for p in m.up.parameters():
                p.requires_grad_(True)
    return adapted


def base_parameters(adapted: nn.Module) -> Iterator[nn.Parameter]:
    """The clone's ORIGINAL (main-YOLO) weights: wrapped convs + BN affine.

    Frozen by :func:`clone_with_lora`; a caller opting into slow base drift
    (e.g. ``--yolo-base-lr 1e-6``) re-enables grad on exactly these. BN
    running stats stay frozen regardless (the clone must not drift from the
    frozen source through statistics).
    """
    for m in adapted.modules():
        if isinstance(m, LoRAConv2d):
            yield from m.conv.parameters()
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            yield from m.parameters(recurse=False)


def lora_parameters(adapted: nn.Module) -> Iterator[nn.Parameter]:
    for m in adapted.modules():
        if isinstance(m, LoRAConv2d):
            yield from m.down.parameters()
            yield from m.up.parameters()


def lora_state_dict(adapted: nn.Module) -> dict:
    """Only the trainable adapter weights, for compact checkpointing."""
    out = {}
    for k, v in adapted.state_dict().items():
        if ".down." in k or ".up." in k:
            out[k] = v
    return out


def load_lora_state_dict(adapted: nn.Module, state: dict) -> None:
    missing, unexpected = adapted.load_state_dict(state, strict=False)
    bad = [k for k in missing if ".down." in k or ".up." in k]
    if bad or unexpected:
        raise RuntimeError(
            f"LoRA state mismatch: missing adapters {bad}, unexpected {list(unexpected)}")
