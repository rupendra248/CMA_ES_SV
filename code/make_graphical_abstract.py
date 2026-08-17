"""Graphical abstract for Elsevier Measurement.

One idea only: the same instrument and the same data, evaluated three ways,
give three different answers -- and the random division is the flattering one.
The gap between the flattering answer and the deployment answer is the
protocol-induced systematic effect the article measures against an
uncertainty budget for the evaluation procedure.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#199e70"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d8d8d4"

plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")

ax.text(50, 56, "A random train/test division does not measure localisation",
        ha="center", va="top", fontsize=13.5, fontweight="bold", color=INK)
ax.text(50, 49.5,
        "UJIIndoorLoc: 19 937 fingerprints, but only 933 reference points (21.4 captures each)",
        ha="center", va="top", fontsize=9.2, color=MUTED)
ax.text(50, 44.8,
        "$\\Rightarrow$ a random division places 100% of test positions in the training partition",
        ha="center", va="top", fontsize=9.6, color=ORANGE, fontweight="bold")

# three protocol bars -- same instrument, same data, three answers
data = [("A  random division", 6.25, BLUE, "re-identification"),
        ("C  RP-disjoint", 9.22, ORANGE, "spatial generalisation"),
        ("B  device/time holdout", 10.85, GREEN, "deployment")]
x0, w, scale = 30.0, 2.9, 2.8
ends = {}
for i, (lab, val, col, what) in enumerate(data):
    y = 34.0 - i * 8.0
    ends[lab[0]] = (x0 + val * scale, y)
    ax.text(x0 - 1.5, y + w / 2, lab, ha="right", va="center",
            fontsize=9.6, color=INK, fontweight="bold")
    ax.add_patch(FancyBboxPatch((x0, y), val * scale, w,
                 boxstyle="round,pad=0,rounding_size=0.9",
                 linewidth=0, facecolor=col, zorder=2))
    ax.text(x0 + val * scale + 1.6, y + w / 2, f"{val:.2f} m",
            ha="left", va="center", fontsize=10.2, fontweight="bold", color=col)
    ax.text(78.0, y + w / 2, what,
            ha="left", va="center", fontsize=8.6, color=MUTED, style="italic")

ax.plot([x0, x0], [9.5, 38.0], lw=0.9, color=RULE, zorder=1)

# the effect: the distance between the flattering answer and the deployment answer
xa, xb = ends["A"][0], ends["B"][0]
yb = 10.5
for x, ytop in ((xa, ends["A"][1]), (xb, ends["B"][1])):
    ax.plot([x, x], [yb, ytop], lw=0.8, ls=(0, (2, 2)), color=MUTED, zorder=1)
ax.annotate("", xy=(xa, yb), xytext=(xb, yb),
            arrowprops=dict(arrowstyle="<->", lw=1.1, color=INK))
ax.text(xb + 2.4, yb + 1.7, "$-4.60$ m protocol-induced effect",
        ha="left", va="center", fontsize=9.0, fontweight="bold", color=INK)
ax.text(xb + 2.4, yb - 1.9, "$3.7\\times$ the expanded uncertainty of $1.25$ m",
        ha="left", va="center", fontsize=8.2, color=MUTED)

ax.text(52, 0.5, "identical instrument  ·  identical data  ·  only the protocol differs",
        ha="center", va="bottom", fontsize=8.8, color=MUTED, style="italic")

fig.tight_layout(pad=0.3)
fig.savefig("graphical_abstract.pdf", bbox_inches="tight")
fig.savefig("graphical_abstract.png", dpi=300, bbox_inches="tight")
print("wrote graphical_abstract.pdf / .png")
