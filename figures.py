"""Paper-style figures for the ambulance suspension study (LaTeX fonts, minimal colour)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.0, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": False,
})
NAVY, ORANGE, RED, TEAL = "#1B3A6B", "#D9822B", "#C0392B", "#0F8B7D"
BLUE = TEAL
CM = 1 / 2.54
MS, MU, KT, G = 1000.0, 100.0, 350e3, 9.81
FST = (MS + MU) * G
R = json.load(open("results.json")); S = np.load("series.npz"); W = np.load("sweep.npz")
PASSIVE = [("soft", dict(color=ORANGE, ls="-")), ("stock", dict(color=NAVY, ls="-", lw=1.3)), ("stiff", dict(color=RED, ls="-"))]
LBL = {"soft": r"soft ($f_s$ = 1.0 Hz, $\zeta$ = 0.15)", "stock": r"stock ($f_s$ = 1.6 Hz, $\zeta$ = 0.30)", "stiff": r"stiff ($f_s$ = 2.4 Hz, $\zeta$ = 0.60)"}


def fig_freq():
    ks, cs = R["stock"]["ks"], R["stock"]["cs"]
    f = np.logspace(np.log10(0.3), np.log10(30), 800); w = 2 * np.pi * f
    A = -MS*w**2 + ks + 1j*w*cs; B = ks + 1j*w*cs; C = -MU*w**2 + ks + KT + 1j*w*cs
    Xu = KT / (C - B*B/A); Xs = B/A*Xu
    fig, ax = plt.subplots(figsize=(7.4*CM, 5.0*CM))
    ax.axvspan(4, 8, color=ORANGE, alpha=0.15, lw=0)
    ax.loglog(f, abs(Xs), color=NAVY, lw=1.4, label=r"body, $|Z_s/Z_r|$")
    ax.loglog(f, abs(Xu), color=TEAL, ls="--", lw=1.2, label=r"wheel, $|Z_u/Z_r|$")
    ax.axhline(1, color="0.6", lw=0.5)
    i = np.argmax(abs(Xs) * (f < 3)); fw = np.sqrt((ks + KT) / MU) / (2*np.pi)
    ax.annotate(rf"$f_b = {f[i]:.1f}$ Hz", (f[i], abs(Xs[i])), xytext=(0.32, 3.2), fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5))
    j = np.argmin(abs(f - fw))
    ax.annotate(rf"$f_w = {fw:.1f}$ Hz", (f[j], abs(Xu[j])), xytext=(6.2, 3.2), fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.text(5.66, 0.0125, "4-8 Hz\nmost felt", ha="center", fontsize=6.5, color="#9A5410")
    ax.set_xlim(0.3, 30); ax.set_ylim(0.008, 6)
    ax.set_xticks([0.5, 1, 2, 4, 8, 16]); ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16"])
    ax.set_xlabel("road input frequency (Hz)"); ax.set_ylabel("response / road input")
    ax.legend(loc="lower left", handlelength=1.8)
    fig.savefig("figures/fig_freq.pdf"); plt.close(fig)


def fig_passive():
    x = S["x"]
    fig, axs = plt.subplots(2, 2, figsize=(9.6*CM, 5.6*CM), sharey="row", gridspec_kw=dict(wspace=0.08, hspace=0.12))
    for col, (lo, hi, name) in enumerate([(28, 36, "pothole (7 cm)"), (68, 78, "speed breaker (10 cm)")]):
        m = (x > lo) & (x < hi)
        for n, st in PASSIVE:
            j = ["soft", "stock", "stiff"].index(n)
            axs[0, col].plot(x[m], S["acc"][m, j], label=n, **st)
            axs[1, col].plot(x[m], (FST + S["ft"][m, j]) / FST * 100, **st)
        axs[1, col].axhspan(-5, 0, color=RED, alpha=0.10, lw=0); axs[1, col].axhline(0, color="0.6", lw=0.5)
        axs[0, col].set_title(name, fontsize=7.5, pad=3)
        axs[1, col].set_xlabel("distance (m)")
    axs[0, 0].set_ylabel(r"$\ddot z_s$ (m/s$^2$)"); axs[1, 0].set_ylabel(r"tyre load (%)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=3, handlelength=2.2, columnspacing=1.6)
    for a in axs[0]: a.tick_params(labelbottom=False)
    fig.savefig("figures/fig_passive.pdf"); plt.close(fig)


def fig_sweep(smart=False, name="figures/fig_sweep.pdf"):
    f, z, c, g = W["f"], W["z"], W["comfort"], W["grip"]
    fig, ax = plt.subplots(figsize=(7.6*CM, 5.6*CM))
    styles = {1.0: dict(color=ORANGE, marker="o", mfc=ORANGE), 1.6: dict(color=NAVY, marker="o", mfc=NAVY, lw=1.3),
              2.4: dict(color=RED, marker="s", mfc=RED)}
    sag = {1.0: 248, 1.6: 97, 2.4: 43}
    for fr, st in styles.items():
        m = np.isclose(f, fr); o = np.argsort(z[m])
        ax.plot(c[m][o], g[m][o], ms=2.6, mew=0.6, label=rf"$f_s$ = {fr:.1f} Hz (sag {sag[fr]} mm)", **st)
    m = np.isclose(f, 1.6); o = np.argsort(z[m]); cc, gg, zz = c[m][o], g[m][o], z[m][o]
    for zv, (dx, dy) in {0.1: (4, -2), 0.9: (-24, 2)}.items():
        k = np.argmin(abs(zz - zv)); ax.annotate(rf"$\zeta$ = {zv:.1f}", (cc[k], gg[k]), xytext=(dx, dy), textcoords="offset points", fontsize=6.5)
    st = R["stock"]
    ax.plot(st["comfort"], st["grip"], "o", ms=7, mfc="none", mec="k", mew=0.7)
    if smart:
        sm = R["smart"]
        ax.plot(sm["comfort"], sm["grip"], "*", ms=9, color=BLUE, zorder=5)
        ax.annotate("", xy=(sm["comfort"] + 0.015, sm["grip"]), xytext=(st["comfort"] - 0.015, st["grip"]),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=BLUE))
        ax.annotate("smart damper", (sm["comfort"], sm["grip"]), xytext=(0.565, 10.45), fontsize=6.5, color=BLUE)
    ax.annotate(r"stock, $\zeta$ = 0.3", (st["comfort"], st["grip"]), xytext=(6, -12), textcoords="offset points", fontsize=6.5)
    for b in [0.315, 0.63, 1.0]:
        ax.axvline(b, color="0.85", lw=0.5, zorder=0)
    ax.set_xlim(0.4, 1.45); ax.set_ylim(9.2, 18.8)
    ax.set_xlabel(r"discomfort $a_w$ (m/s$^2$)"); ax.set_ylabel(r"grip loss $\sigma_F/F_0$ (%)")
    ax.legend(loc="upper left", handlelength=2.2, labelspacing=0.25)
    fig.savefig(name); plt.close(fig)


def fig_smart():
    x = S["x"]
    fig, axs = plt.subplots(2, 2, figsize=(9.6*CM, 5.6*CM), sharey="row", gridspec_kw=dict(wspace=0.08, hspace=0.12, height_ratios=[1.3, 1]))
    for col, (lo, hi, name) in enumerate([(28, 36, "pothole (7 cm)"), (68, 78, "speed breaker (10 cm)")]):
        m = (x > lo) & (x < hi)
        axs[0, col].plot(x[m], S["acc"][m, 1], color=NAVY, lw=0.9, alpha=0.75, label="stock damper")
        axs[0, col].plot(x[m], S["s_acc"][m], color=BLUE, lw=1.1, label="smart damper")
        axs[1, col].plot(x[m], S["s_c"][m] / 1000, color=BLUE, lw=0.9)
        axs[1, col].axhline(R["stock"]["cs"] / 1000, color=NAVY, lw=0.8, ls="--")
        axs[0, col].set_title(name, fontsize=7.5, pad=3)
        axs[1, col].set_xlabel("distance (m)")
    axs[1, 1].text(77.6, 6.6, "stock: 6.0", fontsize=6.5, ha="right")
    axs[0, 0].set_ylabel(r"$\ddot z_s$ (m/s$^2$)"); axs[1, 0].set_ylabel(r"$c$ (kN s/m)")
    axs[1, 0].set_ylim(0, 15.5)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=2, handlelength=2.2, columnspacing=1.6)
    for a in axs[0]: a.tick_params(labelbottom=False)
    fig.savefig("figures/fig_smart.pdf"); plt.close(fig)


for fn in tqdm([fig_freq, fig_passive, lambda: fig_sweep(), lambda: fig_sweep(True, "figures/fig_sweep_smart.pdf"), fig_smart], desc="figures"):
    fn()
