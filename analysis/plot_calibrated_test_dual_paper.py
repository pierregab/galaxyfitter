import numpy as np
import plot_projected_paper as pp

def error_min_threshold(x_core_low, core):
    xs = np.unique(x_core_low)
    cands = (xs[:-1] + xs[1:]) / 2
    errs = np.array([int(((x_core_low < t) != core).sum()) for t in cands])
    i = int(np.argmin(errs))
    return float(cands[i]), int(errs[i])

def draw(ax_s, ax_e, x, y, tstar, tfmt, xlabel, letters, title, rng, left, legend, pub=None):
    """x: NFW values, y: Burkert values. pub: publication_style module (optional
    house style shared with the other paper figures: letters inside the panels,
    error axis limited to its useful range, no per-panel titles)."""
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.11 * (hi - lo)
    ax_s.set_xlim(lo - pad, hi + pad)
    ax_s.set_ylim(-0.72, 1.70)
    ax_s.axvline(tstar, color=pp.C_INK, lw=1.0, zorder=2)
    for j, (h, v) in enumerate((("nfw", x), ("burkert", y))):
        pp.strip_h(ax_s, v, j, pp.COLOR[h], rng, half=0.15, s=8)
    if pub is None:
        ax_s.text(tstar, 1.62, rf"$t_\star = {tstar:{tfmt}}$", ha="center", va="top", fontsize=7.2,
                  bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))
    else:
        # label on the side of the threshold line that has room, clear of the panel letter
        right = (tstar - (lo - pad)) < 0.5 * (hi - lo + 2 * pad)
        off = 0.025 * (hi - lo + 2 * pad) * (1 if right else -1)
        ax_s.text(tstar + off, 1.62, rf"$t_\star = {tstar:{tfmt}}$", ha="left" if right else "right",
                  va="top", fontsize=7.2, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))
    ax_s.set_yticks([0, 1])
    if left:
        ax_s.set_yticklabels([pp.SHORT["nfw"], pp.SHORT["burkert"]])
    else:
        ax_s.set_yticklabels([])
    ax_s.set_xlabel(xlabel)
    if pub is None:
        pp.panel(ax_s, letters[0], title)
    else:
        pub.letter(ax_s, letters[0], x=.05, y=.93)
    pp.minor(ax_s, y=False)

    grid = np.linspace(lo - pad, hi + pad, 600)
    core_hi = y.mean() > x.mean()
    e_n = np.array([np.mean(x > g) if core_hi else np.mean(x < g) for g in grid])
    e_b = np.array([np.mean(y < g) if core_hi else np.mean(y > g) for g in grid])
    ax_e.set_ylim(-4, 150 if pub is None else 122)
    ax_e.set_yticks([0, 50, 100])
    ax_e.set_xlim(lo - pad, hi + pad)
    ax_e.plot(grid, 100 * e_n, color=pp.C_CUSP, lw=1.2, label="cusp called core")
    ax_e.plot(grid, 100 * e_b, color=pp.C_CORE, lw=1.2, label="core called cusp")
    ax_e.axvline(tstar, color=pp.C_INK, lw=0.8, ls=(0, (3, 3)))
    ax_e.set_xlabel(r"threshold $t$")
    if left:
        ax_e.set_ylabel("mislabelled [%]")
    else:
        ax_e.tick_params(labelleft=False)
    if legend and pub is None:
        ax_e.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=6.0,
                    handlelength=1.1, handletextpad=0.4, borderpad=0.2, labelspacing=0.15)
    if pub is None:
        pp.panel(ax_e, letters[1], "error rates")
    else:
        pub.letter(ax_e, letters[1], x=.05, y=.95)
    pp.minor(ax_e)
    core = np.r_[np.zeros(len(x), bool), np.ones(len(y), bool)]
    vals = np.r_[x, y]
    pred = vals > tstar if core_hi else vals < tstar
    return int((pred != core).sum())
