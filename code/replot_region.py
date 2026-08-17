"""Redraw the protocol CDF from saved per-sample errors (no refitting)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

err = {k: np.sort(np.asarray(v)) for k, v in json.load(open("out_region_errors.json")).items()}
COL = {"A": "#2a78d6", "C": "#eb6834", "B": "#199e70"}
LABEL = {"A": "A  random division", "C": "C  RP-disjoint", "B": "B  device/time holdout"}
# stagger the anchor percentile per protocol so direct labels never collide
ANCHOR = {"A": 0.42, "C": 0.62, "B": 0.80}

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.2,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
})
fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.95), sharex=True)

for ax, method in zip(axes, ("SVR", "kNN")):
    for prot in ("A", "C", "B"):
        e = err[f"{method}_{prot}"]
        cdf = np.arange(1, e.size + 1) / e.size
        ax.plot(e, cdf, color=COL[prot], lw=1.3,
                ls="-" if method == "SVR" else (0, (4, 1.6)),
                label=LABEL[prot], solid_capstyle="round")
        q = ANCHOR[prot]
        xi = float(np.percentile(e, q * 100))
        ax.annotate(prot, xy=(xi, q), xytext=(3.0, -7.0),
                    textcoords="offset points", color=COL[prot],
                    fontsize=7.4, fontweight="bold")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.02)
    ax.grid(True, lw=0.35, color="#d8d8d4", alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylabel("fraction of samples")
    ax.text(0.975, 0.07, method, transform=ax.transAxes, ha="right",
            fontsize=8.2, fontweight="bold", color="#0b0b0b")

axes[1].set_xlabel("2D positioning error (m)")
axes[0].legend(loc="center right", frameon=False, fontsize=6.4,
               handlelength=2.4, borderaxespad=0.5, labelspacing=0.32,
               bbox_to_anchor=(1.0, 0.38))
fig.align_ylabels(axes)
fig.tight_layout(pad=0.35, h_pad=0.7)
fig.savefig("protocol_cdf.pdf", bbox_inches="tight")
fig.savefig("protocol_cdf.png", dpi=200, bbox_inches="tight")
print("redrawn")
for m in ("SVR", "kNN"):
    for p in ("A", "C", "B"):
        e = err[f"{m}_{p}"]
        print(f"  {m:4s} {p}  MAE={e.mean():6.2f}  med={np.median(e):6.2f}")
