import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": 8, "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
NAVY, ORANGE, RED, TEAL = "#1B3A6B", "#D9822B", "#C0392B", "#0F8B7D"
r = np.logspace(-1, 1, 600)
fig, ax = plt.subplots(figsize=(8.4/2.54, 5.6/2.54))
for z, col in tqdm([(0.1, ORANGE), (0.3, NAVY), (0.6, RED)], desc="curves"):
    T = np.sqrt((1 + (2*z*r)**2) / ((1 - r**2)**2 + (2*z*r)**2))
    ax.loglog(r, T, color=col, lw=1.1, label=rf"passive, $\zeta$ = {z}")
Ts = 1 / np.sqrt((1 - r**2)**2 + (2*0.3*r)**2)
ax.loglog(r, Ts, color=TEAL, lw=1.4, ls="--", label=r"skyhook, $\zeta$ = 0.3")
ax.axvline(np.sqrt(2), color="0.6", lw=0.6); ax.text(1.47, 6, r"$r=\sqrt{2}$", fontsize=7)
ax.axhline(1, color="0.75", lw=0.5)
ax.set_xlim(0.1, 10); ax.set_ylim(0.008, 8)
ax.set_xlabel(r"frequency ratio $r=\omega/\omega_n$"); ax.set_ylabel(r"transmissibility $|Z/Z_r|$")
ax.legend(loc="lower left", fontsize=6.8, handlelength=2.2)
fig.savefig("figures/fig_trans.pdf")
