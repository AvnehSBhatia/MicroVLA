#!/usr/bin/env python3
"""Render MicroVLA Architecture PDF (pipeline, TRM pattern, losses, tasks).

    .venv/bin/python paper/render_architecture_pdf.py
    # -> paper/MicroVLA_Architecture.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.patches import Circle
import matplotlib.patheffects as pe

OUT = Path(__file__).resolve().parent / "MicroVLA_Architecture.pdf"

# TRM-paper palette (matches the recursive diagram the user referenced)
NAVY = "#0b1a33"
NAVY2 = "#122744"
INK = "#e8eef7"
MUTED = "#9eb0c8"
X_GRAY = "#6b7280"
Y_ORANGE = "#e89a3c"
Z_BLUE = "#3b82c4"
ATTN = "#c9a227"
NORM = "#5a9e6f"
LOSS = "#c44b4b"
ACCENT = "#7dd3fc"
WHITE = "#ffffff"
EDGE = "#d4e0f0"


def _page(fig_w=11.0, fig_h=8.5):
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=NAVY)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(NAVY)
    return fig, ax


def _rounded(ax, xy, w, h, fc, ec=EDGE, lw=1.2, r=0.4, alpha=1.0, z=2):
    p = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=z,
    )
    ax.add_patch(p)
    return p


def _txt(ax, x, y, s, size=10, color=INK, weight="regular", ha="center", va="center",
         family="DejaVu Sans", z=5, **kw):
    return ax.text(
        x, y, s, fontsize=size, color=color, fontweight=weight,
        ha=ha, va=va, family=family, zorder=z, **kw,
    )


def _arrow(ax, x1, y1, x2, y2, color=EDGE, lw=1.6, style="-|>", rad=0.0, z=3):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=z,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)
    return a


def _footer(ax, page: int, n: int):
    _txt(ax, 50, 2.2, f"MicroVLA Architecture  ·  as-built v9  ·  {page}/{n}",
         size=8, color=MUTED)
    _txt(ax, 96, 2.2, "DESIGN.md binding", size=7.5, color=MUTED, ha="right")


def page_cover(ax, n):
    _txt(ax, 50, 82, "MicroVLA", size=36, weight="bold", color=WHITE)
    _txt(ax, 50, 74, "Architecture Reference", size=18, color=ACCENT)
    _txt(ax, 50, 66,
         "~30M deployed  ·  frozen YOLO-World-S + recursive world model + chrono planner",
         size=10, color=MUTED)

    cards = [
        (12, 38, "Pipeline", "2 Hz vision → 30 Hz\nJEPA dream loop →\nplan [5×7]"),
        (38, 38, "TRM pattern", "x / y / z recursion\nweight-tied TinyNet\nresidual next_emb"),
        (64, 38, "Losses & tasks", "spec · BC · grip BCE\nLIBERO object suite\naided vs unaided"),
    ]
    for x, y, title, body in cards:
        _rounded(ax, (x, y), 24, 22, NAVY2, ec=ATTN, lw=1.4, r=0.6)
        _txt(ax, x + 12, y + 16, title, size=12, weight="bold", color=ATTN)
        _txt(ax, x + 12, y + 8, body, size=9, color=INK)

    _txt(ax, 50, 28, "Checkpoint of record: full_stageB_rec_fix.pt", size=9, color=MUTED)
    _txt(ax, 50, 22, "Trainable ≈ 16.6M  ·  Frozen detector ≈ 13M  ·  Target: Pi 5 / 7 servos",
         size=9, color=MUTED)
    _footer(ax, 1, n)


def page_pipeline(ax, n):
    _txt(ax, 50, 95, "End-to-end dataflow (as-built v9)", size=16, weight="bold", color=WHITE)
    _txt(ax, 50, 90.5, "Dims are the live checkpoint contract — see DESIGN.md §v9",
         size=8.5, color=MUTED)

    # Perception band
    _rounded(ax, (3, 72), 94, 15, NAVY2, ec=Z_BLUE, r=0.5)
    _txt(ax, 6, 84, "PERCEPTION (frozen, ~13M)", size=9, weight="bold", color=Z_BLUE, ha="left")
    boxes = [
        (6, 74, 18, 8, "command text\nparse → src/tgt"),
        (28, 74, 20, 8, "CLIP text tower\n[3×512] once/task"),
        (52, 74, 22, 8, "YOLO-World-S\nframe + boxes + grid"),
        (78, 74, 16, 8, "standardize\nμ=0, σ=1"),
    ]
    for x, y, w, h, t in boxes:
        _rounded(ax, (x, y), w, h, "#1a3358", ec=EDGE, r=0.35)
        _txt(ax, x + w / 2, y + h / 2, t, size=8, color=INK)
    for x in (24, 48, 74):
        _arrow(ax, x, 78, x + 3.5, 78, color=ACCENT, lw=1.3)

    # Trainable stack
    _rounded(ax, (3, 28), 94, 40, NAVY2, ec=Y_ORANGE, r=0.5)
    _txt(ax, 6, 64.5, "TRAINABLE STACK (30 Hz)", size=9, weight="bold", color=Y_ORANGE, ha="left")

    stack = [
        (6, 52, 28, 10, "EvidenceEncoder\n0.12M", "→ fused [32,5]"),
        (36, 52, 28, 10, "HRMBackbone\n2.11M", "→ state_delta [256]"),
        (66, 52, 28, 10, "RecursiveTRM\n9.97M  d=1024", "→ next_emb [512]"),
        (6, 36, 28, 10, "TQSA 0.13M", "text × spatial → [128]"),
        (36, 36, 28, 10, "RelationalHead\n2.36M", "12 tokens (97% plan)"),
        (66, 36, 28, 10, "ChronoQueryPlanner\n1.90M", "plan [5,7] tanh"),
    ]
    for x, y, w, h, title, sub in stack:
        _rounded(ax, (x, y), w, h, "#1e3a5f", ec=EDGE, r=0.35)
        _txt(ax, x + w / 2, y + h * 0.62, title, size=8.5, weight="bold", color=WHITE)
        _txt(ax, x + w / 2, y + h * 0.28, sub, size=7.5, color=MUTED)

    _arrow(ax, 50, 72, 50, 66, color=Y_ORANGE)
    _arrow(ax, 34, 57, 36, 57, color=ACCENT)
    _arrow(ax, 64, 57, 66, 57, color=ACCENT)
    _arrow(ax, 80, 52, 80, 46.5, color=ACCENT)
    _arrow(ax, 64, 41, 66, 41, color=ACCENT)
    _arrow(ax, 34, 41, 36, 41, color=ACCENT)

    # Runtime / control
    _rounded(ax, (3, 8), 45, 16, NAVY2, ec=NORM, r=0.5)
    _txt(ax, 25.5, 20.5, "JEPALoop + InnovationCorrector", size=9, weight="bold", color=NORM)
    _txt(ax, 25.5, 14.5,
         "real every N ticks · dream = faded boxes\n"
         "trust τ brakes delta actions  min(1, τ/0.5)",
         size=8, color=INK)

    _rounded(ax, (52, 8), 45, 16, NAVY2, ec=LOSS, r=0.5)
    _txt(ax, 74.5, 20.5, "Control layer (eval)", size=9, weight="bold", color=LOSS)
    _txt(ax, 74.5, 14.5,
         "unaided: emit plan row 0\n"
         "assisted: PhasedIBVS hand-eye (0 params)",
         size=8, color=INK)

    _arrow(ax, 50, 28, 25, 24.5, color=NORM, rad=0.1)
    _arrow(ax, 50, 28, 74, 24.5, color=LOSS, rad=-0.1)
    _footer(ax, 2, n)


def page_trm(ax, n):
    """TRM recursive diagram in the style of the Tiny Recursive Model figure."""
    _txt(ax, 50, 96, "RecursiveTRM — model pattern", size=16, weight="bold", color=WHITE)
    _txt(ax, 50, 91.5,
         "Weight-tied TinyNet · two latents (y prediction, z reasoning) · FiLM observation x",
         size=8.5, color=MUTED)

    # Left column: inputs
    _rounded(ax, (4, 70), 16, 10, X_GRAY, ec=WHITE, lw=1.5, r=0.45)
    _txt(ax, 12, 77, "Input (x)", size=10, weight="bold", color=WHITE)
    _txt(ax, 12, 73.5, "Observation", size=8, color="#f3f4f6")
    _txt(ax, 12, 66.5, "fused · current · history\nFiLM(state_delta)", size=7.5, color=MUTED)

    _rounded(ax, (4, 48), 16, 10, Y_ORANGE, ec=WHITE, lw=1.5, r=0.45)
    _txt(ax, 12, 55, "Prediction (y)", size=10, weight="bold", color=WHITE)
    _txt(ax, 12, 51.5, "Answer / emb", size=8, color="#fff7ed")

    _rounded(ax, (4, 26), 16, 10, Z_BLUE, ec=WHITE, lw=1.5, r=0.45)
    _txt(ax, 12, 33, "Latent (z)", size=10, weight="bold", color=WHITE)
    _txt(ax, 12, 29.5, "Reasoning", size=8, color="#eff6ff")

    # Sum node
    c = Circle((30, 53), 2.4, facecolor=NAVY2, edgecolor=WHITE, lw=1.5, zorder=4)
    ax.add_patch(c)
    _txt(ax, 30, 53, "+", size=16, weight="bold", color=WHITE, z=5)
    _arrow(ax, 20, 75, 28.2, 54.5, color=X_GRAY, lw=1.8)
    _arrow(ax, 20, 53, 27.4, 53, color=Y_ORANGE, lw=1.8)
    _arrow(ax, 20, 31, 28.2, 51.5, color=Z_BLUE, lw=1.8)

    # Transformer stack (4 conceptual layers — TinyNet is weight-tied; show pattern)
    _txt(ax, 58, 84, "TinyNet  (weight-tied, called T·(n_inner+1)×)", size=9,
         weight="bold", color=ATTN)
    layer_y = [72, 58, 44, 30]
    for i, y in enumerate(layer_y):
        _rounded(ax, (40, y + 5), 28, 5.5, ATTN, ec=WHITE, lw=1.0, r=0.25)
        _txt(ax, 54, y + 7.7, "Self-Attn / token-mix", size=8, weight="bold", color=NAVY)
        _rounded(ax, (40, y), 28, 4.5, NORM, ec=WHITE, lw=1.0, r=0.25)
        _txt(ax, 54, y + 2.2, "Add & Norm", size=8, weight="bold", color=WHITE)
        if i < len(layer_y) - 1:
            _arrow(ax, 54, y, 54, layer_y[i + 1] + 10.5, color=EDGE, lw=1.2)
    # MLP + norm for last block emphasis
    _rounded(ax, (40, 18), 28, 5.5, ATTN, ec=WHITE, lw=1.0, r=0.25)
    _txt(ax, 54, 20.7, "MLP / channel-mix", size=8, weight="bold", color=NAVY)
    _rounded(ax, (40, 13), 28, 4.5, NORM, ec=WHITE, lw=1.0, r=0.25)
    _txt(ax, 54, 15.2, "Add & Norm", size=8, weight="bold", color=WHITE)
    _arrow(ax, 54, 30, 54, 23.5, color=EDGE, lw=1.2)
    _arrow(ax, 32.4, 53, 40, 53, color=WHITE, lw=1.5)

    # Feedback loops
    _arrow(ax, 68, 75, 20.5, 53.5, color=Y_ORANGE, lw=2.0, rad=-0.35, style="-|>")
    _txt(ax, 78, 68, "update y", size=8, weight="bold", color=Y_ORANGE, ha="left")
    _arrow(ax, 68, 35, 20.5, 31, color=Z_BLUE, lw=2.0, rad=0.25, style="-|>")
    _txt(ax, 78, 38, "update z", size=8, weight="bold", color=Z_BLUE, ha="left")

    # Right: schedule + readout
    _rounded(ax, (74, 70), 23, 18, NAVY2, ec=Z_BLUE, r=0.4)
    _txt(ax, 85.5, 84, "Reasoning phase", size=9, weight="bold", color=Z_BLUE)
    _txt(ax, 85.5, 78.5, "n_inner = 6\nz ← net(x+y+z)\nimprove latent z", size=8, color=INK)

    _rounded(ax, (74, 48), 23, 16, NAVY2, ec=Y_ORANGE, r=0.4)
    _txt(ax, 85.5, 60, "Answer phase", size=9, weight="bold", color=Y_ORANGE)
    _txt(ax, 85.5, 54.5, "1 step\ny ← net(y+z)\nimprove prediction y", size=8, color=INK)

    _rounded(ax, (74, 26), 23, 16, NAVY2, ec=ACCENT, r=0.4)
    _txt(ax, 85.5, 38, "Outer deep refine", size=9, weight="bold", color=ACCENT)
    _txt(ax, 85.5, 32.5, "T = 3 blocks\ngrad on last only\nn_sup_infer = 1 (deploy)",
         size=8, color=INK)

    _rounded(ax, (74, 8), 23, 14, "#3a1515", ec=LOSS, r=0.4)
    _txt(ax, 85.5, 18.5, "Readout", size=9, weight="bold", color=LOSS)
    _txt(ax, 85.5, 13.2, "next_emb = current + Δ\nbox / msg / latent", size=8, color=INK)

    _arrow(ax, 54, 13, 54, 9, color=LOSS, lw=1.4)
    _rounded(ax, (40, 5.5), 28, 3.5, LOSS, ec=WHITE, r=0.2)
    _txt(ax, 54, 7.2, "spec_loss  ·  residual convention", size=8, weight="bold", color=WHITE)

    _footer(ax, 3, n)


def page_modules(ax, n):
    _txt(ax, 50, 95, "Module patterns & parameter ledger", size=16, weight="bold", color=WHITE)

    rows = [
        ("EvidenceEncoder", "0.12M", "frame|objs|text|action → fused[32,5]; conf×freshness fade"),
        ("HRMBackbone", "2.11M", "slow (real) + fast (30 Hz) two-rate state; gain_head zero-init"),
        ("RecursiveTRM", "9.97M", "FiLM observe → (y,z) recursion → residual next_emb[512]"),
        ("TQSA", "0.13M", "text queries 4×4 spatial grid → token[128] + heat"),
        ("RelationalHead", "2.36M", "12 queries × objs/text/latent/action — 97% plan ablation"),
        ("ChronoQueryPlanner", "1.90M", "5 time-queries × memory → plan[5,7] + grip + waypoints"),
        ("InnovationCorrector", "0", "EMA innovation → trust τ; delta brake / absolute hold"),
        ("YOLO-World-S", "~13M*", "frozen vision+text; *not in trainable budget"),
    ]
    y = 86
    _rounded(ax, (4, y - 2), 22, 5, "#1e3a5f", ec=ATTN, r=0.25)
    _rounded(ax, (27, y - 2), 12, 5, "#1e3a5f", ec=ATTN, r=0.25)
    _rounded(ax, (40, y - 2), 56, 5, "#1e3a5f", ec=ATTN, r=0.25)
    _txt(ax, 15, y + 0.5, "Module", size=9, weight="bold", color=ATTN)
    _txt(ax, 33, y + 0.5, "Params", size=9, weight="bold", color=ATTN)
    _txt(ax, 68, y + 0.5, "Pattern", size=9, weight="bold", color=ATTN)

    for i, (name, params, pat) in enumerate(rows):
        yy = 78 - i * 8.2
        fc = NAVY2 if i % 2 == 0 else "#152a48"
        _rounded(ax, (4, yy - 2), 22, 7, fc, ec=EDGE, r=0.25)
        _rounded(ax, (27, yy - 2), 12, 7, fc, ec=EDGE, r=0.25)
        _rounded(ax, (40, yy - 2), 56, 7, fc, ec=EDGE, r=0.25)
        _txt(ax, 15, yy + 1.5, name, size=8.5, weight="bold", color=WHITE)
        _txt(ax, 33, yy + 1.5, params, size=8.5, color=ACCENT)
        _txt(ax, 68, yy + 1.5, pat, size=7.8, color=INK)

    _txt(ax, 50, 8,
         "Hard caps: fusion≤5M · drift≤1.5M · planner≤2.5M · trainable < 9M heads + TRM slot ~10M",
         size=8, color=MUTED)
    _footer(ax, 4, n)


def page_jepa(ax, n):
    _txt(ax, 50, 95, "JEPA runtime pattern — real vs dream", size=16, weight="bold", color=WHITE)
    _txt(ax, 50, 90.5,
         "Core claim: dream evidence is faded evidence on the SAME path as train modality_dropout",
         size=8.5, color=MUTED)

    # Timeline
    _txt(ax, 8, 82, "tick", size=8, color=MUTED, ha="left")
    for i in range(15):
        x = 12 + i * 5.5
        is_real = i == 0
        fc = Y_ORANGE if is_real else Z_BLUE
        _rounded(ax, (x, 76), 4.5, 6, fc, ec=WHITE, r=0.2)
        _txt(ax, x + 2.25, 79, "R" if is_real else "D", size=8, weight="bold", color=WHITE)
        _txt(ax, x + 2.25, 73.5, str(i), size=6.5, color=MUTED)
    _txt(ax, 50, 69, "design: 1 real + 14 dream @ 30 Hz  ·  eval often uses perception_period=2",
         size=8, color=MUTED)

    # Two columns
    _rounded(ax, (4, 28), 44, 36, NAVY2, ec=Y_ORANGE, r=0.5)
    _txt(ax, 26, 59, "REAL tick", size=12, weight="bold", color=Y_ORANGE)
    steps = [
        "1. YOLO perceive → standardized embs",
        "2. EvidenceEncoder (full conf)",
        "3. HRM slow-step + fast",
        "4. TRM forward_full → next_emb",
        "5. Relational + Planner → plan",
        "6. Corrector: innovate, update τ",
        "7. Execute plan row 0; hold boxes",
    ]
    for i, s in enumerate(steps):
        _txt(ax, 8, 52 - i * 3.5, s, size=8.5, color=INK, ha="left")

    _rounded(ax, (52, 28), 44, 36, NAVY2, ec=Z_BLUE, r=0.5)
    _txt(ax, 74, 59, "DREAM tick", size=12, weight="bold", color=Z_BLUE)
    steps_d = [
        "1. frame token = corrected next_emb",
        "2. boxes HELD from last real",
        "3. weight *= staleness_decay^k (0.9^k)",
        "4. SAME EvidenceEncoder path",
        "5. HRM fast only (slow held)",
        "6. TRM → planner → trust-scaled plan",
        "7. feed corrected latent forward",
    ]
    for i, s in enumerate(steps_d):
        _txt(ax, 56, 52 - i * 3.5, s, size=8.5, color=INK, ha="left")

    _rounded(ax, (4, 8), 92, 16, "#1a2f1a", ec=NORM, r=0.45)
    _txt(ax, 50, 19.5, "Evidence fade identity", size=10, weight="bold", color=NORM)
    _txt(ax, 50, 13.5,
         "train: modality_dropout fades box/geometry weights by U[0,1)\n"
         "infer dream: confidence × 0.9^k   ·   missed detect → weight 0   ·   last-action token never faded",
         size=8.5, color=INK)
    _footer(ax, 5, n)


def page_losses(ax, n):
    _txt(ax, 50, 95, "Special loss functions", size=16, weight="bold", color=WHITE)

    losses = [
        (Y_ORANGE, "TRM spec_loss",
         "L = 1.0 · (1 − cos(ŷ, y)) + 0.5 · MSE(ŷ, y)\n"
         "on CANONICAL standardized embeddings — no LayerNorm inside the loss\n"
         "deep supervision via refine_forward · n_sup=3 train / n_sup_infer=1 deploy"),
        (Z_BLUE, "Stage A world-model",
         "fusion + HRM + TRM · deployment-exact 15-tick rollouts between real frames\n"
         "spec_loss on predicted next frame emb (and box when enabled)"),
        (ATTN, "Stage B planner BC",
         "split_planner_loss: MSE on pose dims + BCE on gripper logit\n"
         "+ smoothness (2nd-diff) · optional row0_weight · waypoints MSE masked\n"
         "centering_loss / depth_loss: IBVS-shaped aux for last-cm xy / z"),
        (NORM, "Optional auxiliaries",
         "modality_consistency_loss: MSE(fused_dropped, fused_full.detach())\n"
         "aligns train dropout with dream fade · waypoint_loss on metric EEF disp"),
        (LOSS, "What is NOT a learned loss",
         "PhasedIBVS / grasp-offset / place_at are calibrated constants (0 params)\n"
         "InnovationCorrector trust is EMA ratio — not backpropagated"),
    ]
    y = 82
    for color, title, body in losses:
        _rounded(ax, (4, y - 10), 92, 12, NAVY2, ec=color, r=0.4)
        _txt(ax, 8, y - 1.5, title, size=10, weight="bold", color=color, ha="left")
        _txt(ax, 8, y - 6.5, body, size=8, color=INK, ha="left", va="center")
        y -= 14.5

    _footer(ax, 6, n)


def page_tasks(ax, n):
    _txt(ax, 50, 95, "Tasks, training stages, evaluation tracks", size=16, weight="bold", color=WHITE)

    # Tasks
    _rounded(ax, (4, 62), 92, 28, NAVY2, ec=ACCENT, r=0.5)
    _txt(ax, 50, 85, "LIBERO-Object suite (primary)", size=11, weight="bold", color=ACCENT)
    tasks = [
        ("T0", "alphabet soup → basket", "assisted 0.75", Y_ORANGE),
        ("T1", "cream cheese → basket", "assisted 0.30", Z_BLUE),
        ("T2", "salad dressing → basket", "perception hard", LOSS),
    ]
    for i, (tid, name, note, c) in enumerate(tasks):
        x = 8 + i * 30
        _rounded(ax, (x, 66), 26, 14, "#152a48", ec=c, r=0.35)
        _txt(ax, x + 13, 76, tid, size=12, weight="bold", color=c)
        _txt(ax, x + 13, 72, name, size=8, color=INK)
        _txt(ax, x + 13, 68.5, note, size=7.5, color=MUTED)

    # Training
    _rounded(ax, (4, 34), 44, 24, NAVY2, ec=Y_ORANGE, r=0.45)
    _txt(ax, 26, 53, "Training", size=11, weight="bold", color=Y_ORANGE)
    _txt(ax, 8, 46,
         "A  train_full stage A — WM (fusion/HRM/TRM)\n"
         "B  stage B — freeze WM, BC planner\n"
         "   teacher_bc — distill PhasedIBVS successes\n"
         "data  LIBERO demos → .npz shards (BudgetGuard)\n"
         "device  prefer MPS / CUDA; tests CPU-mock only",
         size=8, color=INK, ha="left", va="top")

    _rounded(ax, (52, 34), 44, 24, NAVY2, ec=Z_BLUE, r=0.45)
    _txt(ax, 74, 53, "Evaluation tracks", size=11, weight="bold", color=Z_BLUE)
    _txt(ax, 56, 46,
         "unaided   plan only → currently 0.0 success\n"
         "assisted  plan + PhasedIBVS constants\n"
         "wind tunnel  eval.bench --synthetic (no sim)\n"
         "mock-env  always required (no LIBERO deps)\n"
         "honesty   assisted ≠ policy competence",
         size=8, color=INK, ha="left", va="top")

    _rounded(ax, (4, 8), 92, 22, NAVY2, ec=ATTN, r=0.45)
    _txt(ax, 50, 25, "Plan / action contract", size=11, weight="bold", color=ATTN)
    _txt(ax, 50, 16,
         "plan shape [5, 7] — rows = timesteps, cols = servos (EEF Δxyz + rot + grip)\n"
         "action_space=\"delta\" (LIBERO): low trust BRAKES → 0   ·   \"absolute\" (Pi PWM): HOLD-blend\n"
         "canonical emb space everywhere; TRM residual: next = current + Δ",
         size=8.5, color=INK)
    _footer(ax, 7, n)


def page_novelty(ax, n):
    _txt(ax, 50, 95, "Design claims (what is / is not novel)", size=16, weight="bold", color=WHITE)

    claims = [
        (NORM, "Distinctive",
         "• Detector-as-text-encoder (no separate LM)\n"
         "• Dream evidence ≡ faded evidence (one code path)\n"
         "• Test-enforced parameter + disk budgets\n"
         "• Forensic defect taxonomy + lever-arm handoff"),
        (MUTED, "Composition of known ideas",
         "• JEPA-style predictive world models\n"
         "• Recursive / Tiny recursive transformers (TRM)\n"
         "• Cross-attention planners / relational tokens\n"
         "• Classical IBVS / hand-eye calibration"),
        (LOSS, "Explicit non-claims",
         "• No unaided LIBERO success claimed (still 0)\n"
         "• Assisted numbers ≠ learned policy competence\n"
         "• Sim-only; Pi 5 is design target, not reported deploy\n"
         "• Not competing with OpenVLA-scale absolute SR"),
    ]
    for i, (c, title, body) in enumerate(claims):
        x = 4 + i * 32
        _rounded(ax, (x, 28), 30, 58, NAVY2, ec=c, r=0.5)
        _txt(ax, x + 15, 80, title, size=11, weight="bold", color=c)
        _txt(ax, x + 2.5, 72, body, size=8.2, color=INK, ha="left", va="top")

    _txt(ax, 50, 18, "Binding docs: DESIGN.md  ·  microvla/config.py  ·  TRM.py  ·  paper/MANUSCRIPT.md",
         size=8.5, color=MUTED)
    _txt(ax, 50, 12, "Generate:  .venv/bin/python paper/render_architecture_pdf.py",
         size=8, color=MUTED)
    _footer(ax, 8, n)


def main() -> Path:
    pages = [
        page_cover,
        page_pipeline,
        page_trm,
        page_modules,
        page_jepa,
        page_losses,
        page_tasks,
        page_novelty,
    ]
    n = len(pages)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        for fn in pages:
            fig, ax = _page()
            fn(ax, n)
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)
        d = pdf.infodict()
        d["Title"] = "MicroVLA Architecture Reference"
        d["Author"] = "MicroVLA"
        d["Subject"] = "Pipeline, TRM pattern, losses, tasks (as-built v9)"
    print(f"wrote {OUT} ({n} pages)")
    return OUT


if __name__ == "__main__":
    main()
