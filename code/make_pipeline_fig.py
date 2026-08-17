"""Method figure: the protocol-controlled pipeline.

Replaces the old architecture diagram, which showed a plain train/test split with
no validation partition -- i.e. exactly the design this paper argues against.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#199e70"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#c9c8c3"

D = 6.0   # vertical offset of the tuning lane, leaving room between the lanes
plt.rcParams.update({"font.family": "serif", "font.size": 6.6})
fig, ax = plt.subplots(figsize=(3.5, 2.0 * (74 + D) / 74))
ax.set_xlim(-1.5, 100.5); ax.set_ylim(0, 74 + D); ax.axis("off")   # margin so box borders at x=0 are not clipped


def box(x, y, w, h, text, edge, fill="#ffffff", bold=False, fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.9,rounding_size=2.0",
                 linewidth=0.9, edgecolor=edge, facecolor=fill, zorder=2,
                 clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", zorder=3,
            fontsize=fs, color=INK, fontweight="bold" if bold else "normal",
            linespacing=1.35)


def arrow(x1, y1, x2, y2, color=MUTED, style="-|>", rad=0.0, lw=0.85):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=7, linewidth=lw, color=color, zorder=1,
                 connectionstyle=f"arc3,rad={rad}"))


# ---- lane labels
ax.text(0.5, 70.5 + D, "OFFLINE  ·  tuning", fontsize=6.2, color=BLUE,
        fontweight="bold", ha="left")
ax.text(0.5, 33.0, "FINAL  ·  evaluated once", fontsize=6.2, color=GREEN,
        fontweight="bold", ha="left")
ax.plot([0, 100], [38.0, 38.0], lw=0.7, color=RULE, ls=(0, (3, 2)), zorder=0)

# ---- offline lane
box(0, 50 + D, 25, 15, "Survey captures\n(many per\nreference point)", RULE)
box(31, 50 + D, 22, 15, "Split\n$\\it{by\\ reference}$\n$\\it{point}$", BLUE,
    fill="#eef4fc", bold=True)
box(59, 58.5 + D, 18, 9.5, "fit fold", RULE)
box(59, 45.5 + D, 18, 9.5, "inner val\n(RP-disjoint)", BLUE, fill="#eef4fc")
box(83, 50 + D, 16, 15, "CMA-ES\n$(C,\\gamma,\\varepsilon)$", ORANGE,
    fill="#fdf0ea", bold=True)

arrow(25.6, 57.5 + D, 30.4, 57.5 + D)
arrow(53.6, 57.5 + D, 58.4, 62.0 + D)
arrow(53.6, 57.5 + D, 58.4, 49.5 + D)
# tuning loop: CMA-ES proposes -> fit -> score on inner val -> back to CMA-ES
arrow(82.4, 61.0 + D, 77.6, 63.0 + D, ORANGE)          # CMA-ES -> fit fold
ax.text(80.0, 66.4 + D, "propose", fontsize=5.4, color=ORANGE, ha="center")
arrow(68.0, 58.0 + D, 68.0, 55.4 + D, RULE)            # fit fold -> inner val
arrow(77.6, 49.0 + D, 82.4, 53.0 + D, ORANGE)          # inner val -> CMA-ES
ax.text(80.6, 43.6 + D, "mean 2D\nerror (m)", fontsize=5.4, color=ORANGE,
        ha="center", va="top", linespacing=1.3)

# ---- final lane
box(0, 9, 25, 15, "Refit $f_x,f_y$\non ALL\ntraining data", RULE)
box(31, 9, 22, 15, "Held-out set\nnew users,\ndevices, months", GREEN,
    fill="#eaf5f0", bold=True)
box(59, 9, 18, 15, "invert target\nscaling", GREEN, fill="#eaf5f0")
box(83, 9, 16, 15, "error in\nMETRES", GREEN, fill="#eaf5f0", bold=True)

arrow(25.6, 16.5, 30.4, 16.5)
arrow(53.6, 16.5, 58.4, 16.5)
arrow(77.6, 16.5, 82.4, 16.5)

# tuned parameters drop from the CMA-ES box into the final refit. Routed as an
# elbow -- straight down from the CMA-ES box, across beneath the lane label, and
# down into the refit box -- so that it neither passes behind any box nor
# crosses any label. Drawn as an explicit path in data units (dpi-independent).
Y_RUN, r = 29.0, 3.0
_verts = [(91.0, 49.1 + D), (91.0, Y_RUN + r),
          (91.0, Y_RUN), (91.0 - r, Y_RUN),
          (12.5 + r, Y_RUN),
          (12.5, Y_RUN), (12.5, Y_RUN - r),
          (12.5, 24.9)]
_codes = [Path.MOVETO, Path.LINETO,
          Path.CURVE3, Path.CURVE3,
          Path.LINETO,
          Path.CURVE3, Path.CURVE3,
          Path.LINETO]
ax.add_patch(FancyArrowPatch(path=Path(_verts, _codes), arrowstyle="-|>",
             mutation_scale=7, linewidth=0.85, color=ORANGE, zorder=1))
ax.text(52, Y_RUN + 0.9, "best $(C^{*},\\gamma^{*},\\varepsilon^{*})$",
        fontsize=6.2, color=ORANGE, ha="center", va="bottom", fontweight="bold")

ax.text(50, 2.4, "the held-out set is never used to select anything",
        fontsize=5.9, color=MUTED, ha="center", style="italic")

fig.tight_layout(pad=0.15)
fig.savefig("pipeline.pdf", bbox_inches="tight")
fig.savefig("pipeline.svg", bbox_inches="tight")   # editable vector, importable into Canva/Illustrator
fig.savefig("pipeline.png", dpi=200, bbox_inches="tight")
print("wrote pipeline.pdf / .png")
