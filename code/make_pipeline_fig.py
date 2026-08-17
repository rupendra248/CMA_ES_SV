"""Method figure: the protocol-controlled pipeline.

Replaces the old architecture diagram, which showed a plain train/test split with
no validation partition -- i.e. exactly the design this paper argues against.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#199e70"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#c9c8c3"

plt.rcParams.update({"font.family": "serif", "font.size": 6.6})
fig, ax = plt.subplots(figsize=(3.5, 2.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 74); ax.axis("off")


def box(x, y, w, h, text, edge, fill="#ffffff", bold=False, fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.9,rounding_size=2.0",
                 linewidth=0.9, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", zorder=3,
            fontsize=fs, color=INK, fontweight="bold" if bold else "normal",
            linespacing=1.35)


def arrow(x1, y1, x2, y2, color=MUTED, style="-|>", rad=0.0, lw=0.85):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=7, linewidth=lw, color=color, zorder=1,
                 connectionstyle=f"arc3,rad={rad}"))


# ---- lane labels
ax.text(0.5, 70.5, "OFFLINE  ·  tuning", fontsize=6.2, color=BLUE,
        fontweight="bold", ha="left")
ax.text(0.5, 30.0, "FINAL  ·  evaluated once", fontsize=6.2, color=GREEN,
        fontweight="bold", ha="left")
ax.plot([0, 100], [34.5, 34.5], lw=0.7, color=RULE, ls=(0, (3, 2)), zorder=0)

# ---- offline lane
box(0, 50, 25, 15, "Survey captures\n(many per\nreference point)", RULE)
box(31, 50, 22, 15, "Split\n$\\it{by\\ reference}$\n$\\it{point}$", BLUE,
    fill="#eef4fc", bold=True)
box(59, 58.5, 18, 9.5, "fit fold", RULE)
box(59, 45.5, 18, 9.5, "inner val\n(RP-disjoint)", BLUE, fill="#eef4fc")
box(83, 50, 16, 15, "CMA-ES\n$(C,\\gamma,\\varepsilon)$", ORANGE,
    fill="#fdf0ea", bold=True)

arrow(25.6, 57.5, 30.4, 57.5)
arrow(53.6, 57.5, 58.4, 62.0)
arrow(53.6, 57.5, 58.4, 49.5)
# tuning loop: CMA-ES proposes -> fit -> score on inner val -> back to CMA-ES
arrow(82.4, 61.0, 77.6, 63.0, ORANGE)          # CMA-ES -> fit fold
ax.text(80.0, 66.4, "propose", fontsize=5.4, color=ORANGE, ha="center")
arrow(68.0, 58.0, 68.0, 55.4, RULE)            # fit fold -> inner val
arrow(77.6, 49.0, 82.4, 53.0, ORANGE)          # inner val -> CMA-ES
ax.text(80.6, 43.6, "mean 2D\nerror (m)", fontsize=5.4, color=ORANGE,
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
# tuned params drop from the CMA-ES box into the final refit
arrow(91.0, 49.4, 12.5, 24.6, ORANGE, rad=0.10, lw=0.85)
ax.text(52, 36.4, "best $(C^{*},\\gamma^{*},\\varepsilon^{*})$", fontsize=6.2,
        color=ORANGE, ha="center", fontweight="bold")

ax.text(50, 2.4, "the held-out set is never used to select anything",
        fontsize=5.9, color=MUTED, ha="center", style="italic")

fig.tight_layout(pad=0.15)
fig.savefig("pipeline.pdf", bbox_inches="tight")
fig.savefig("pipeline.svg", bbox_inches="tight")   # editable vector, importable into Canva/Illustrator
fig.savefig("pipeline.png", dpi=200, bbox_inches="tight")
print("wrote pipeline.pdf / .png")
