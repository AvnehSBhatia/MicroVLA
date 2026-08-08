"""F2 --- the dataflow, with the language path highlighted end to end.

The paper's central architectural claim is a code fact: ``GraspPointHead.forward``
takes no text argument, so the only path from the instruction to a grasp is box
selection, and box selection is identity-blind. That argument currently lives in
prose. This draws it: every text-carrying edge is red, and the reader can follow
the red edges and see that they terminate in a single cached ``(x, y)``.

Nothing here is inferred. The head signatures are read off
``microvla/control/heads.py`` and ``eval/policy.py``; the latch point is
``MicroVLAPolicy.reset()``, which calls ``set_place(place_head(command_emb))``
once per episode and never again.

Run: python paper/render_fig2_dataflow.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "visuals" / "F2_dataflow_language.png"

TEXT = "#c1121f"      # carries language
VIS = "#2a6f97"       # carries pixels
NEU = "#5c677d"       # carries neither

fig, ax = plt.subplots(figsize=(12.4, 5.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 46)
ax.axis("off")


def box(x, y, w, h, label, color=NEU, fill="#ffffff", fs=8.2, bold=False,
        mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                linewidth=1.6, edgecolor=color, facecolor=fill,
                                zorder=2))
    # matplotlib does not parse LaTeX markup here, so code is set with a
    # monospace family rather than \texttt{} -- which would render literally.
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            zorder=3, fontweight="bold" if bold else "normal", linespacing=1.35,
            family="monospace" if mono else None)
    return (x, y, w, h)


def arrow(a, b, color=NEU, lw=1.6, label=None, side="top", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((a[0], a[1]), (b[0], b[1]),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color, zorder=1,
                                 linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx, my + (1.6 if side == "top" else -2.4), label, ha="center",
                fontsize=7.0, color=color, zorder=4)


# ---- inputs ---------------------------------------------------------------
instr = box(1, 36, 17, 7, "instruction\n\"pick up the alphabet soup\"", TEXT,
            "#fdecec", bold=True)
frame = box(1, 8, 17, 7, "wrist frame\n(256$\\times$256)", VIS, "#eaf1f7", bold=True)

# ---- parser / detector ----------------------------------------------------
parser = box(24, 36, 16, 7, "command parser\n$\\rightarrow$ source/target phrases", TEXT, "#fdecec")
det = box(24, 20, 16, 11, "frozen YOLO-World-S\n(only vision AND text encoder)",
          NEU, "#f4f4f6", bold=True)

arrow((18, 39.5), (24, 39.5), TEXT, 2.2)
arrow((18, 11.5), (30, 20), VIS, 2.2, rad=-0.12)
arrow((32, 36), (32, 31), TEXT, 2.2, "prompts", side="bot")

# ---- what the detector emits ---------------------------------------------
boxes = box(46, 24, 15, 7, "selected box\n(uv, conf, emb)", VIS, "#eaf1f7")
cmdemb = box(46, 36, 15, 7, "command\nembedding", TEXT, "#fdecec")
arrow((40, 26), (46, 27), VIS, 2.2)
arrow((40, 39.5), (46, 39.5), TEXT, 2.2, "text tower", side="top")

# ---- heads ----------------------------------------------------------------
grasp = box(67, 20, 20, 11,
            "GraspPointHead.forward(\n  uv, conf, proprio,\n"
            "  box_emb, frame_emb)\n\u2192 NO TEXT ARGUMENT",
            VIS, "#eaf1f7", fs=7.4, mono=True)
place = box(67, 36, 20, 7, "PlaceHead(command_emb)\n\u2192 place (x, y)",
            TEXT, "#fdecec", fs=7.8, mono=True)
arrow((61, 27), (67, 26), VIS, 2.2)
arrow((61, 39.5), (67, 39.5), TEXT, 2.4)

prop = box(46, 8, 15, 7, "proprioception\n(eef pose)", NEU, "#f4f4f6")
arrow((61, 11.5), (72, 20), NEU, 1.8, rad=0.1)

# ---- the latch ------------------------------------------------------------
latch = box(63, 1.5, 24, 6.5,
            "set_place(...) LATCHED ONCE\nat episode start, never re-read",
            TEXT, "#fdecec", fs=7.6, bold=True, mono=True)
# Routed around the grasp head, not through it: the place path never touches it.
arrow((88, 37), (88, 8), TEXT, 2.6, rad=-0.45)

shell = box(91, 20, 8, 11, "shell\n(servo)", NEU, "#f4f4f6")
arrow((87, 26), (91, 26), VIS, 2.0)
arrow((87, 4.7), (94, 20), TEXT, 2.0, rad=-0.30)

# ---- legend / caption-in-figure ------------------------------------------
h = [plt.Line2D([], [], color=TEXT, lw=2.6), plt.Line2D([], [], color=VIS, lw=2.6),
     plt.Line2D([], [], color=NEU, lw=2.0)]
ax.legend(h, ["carries the instruction", "carries pixels", "carries neither"],
          frameon=False, fontsize=8.4, loc="lower left", bbox_to_anchor=(0.0, -0.02),
          ncol=3)

ax.text(50, 44.6,
        "Follow the red edges: every path from the instruction ends in ONE cached $(x,y)$. "
        "Selection, approach and grasping are reached by no red edge at all.",
        ha="center", fontsize=9.0, style="italic", color="#333333")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, bbox_inches="tight")
print(f"wrote {OUT}")
