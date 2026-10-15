import numpy as np
import publication_style as p
COLOURS = {'nfw': p.BLUE, 'burkert': p.CORE}

def strip(ax, values, x, color):
    # Deterministic rank spacing shows all catalogues without random jitter.
    values = np.sort(values)
    offsets = (np.arange(len(values)) % 7 - 3) * .040
    ax.scatter(x + offsets, values, s=9, color=color, alpha=.75, lw=0, zorder=3)
    lo, mid, hi = np.percentile(values, [16, 50, 84])
    ax.plot([x, x], [lo, hi], color=color, lw=2.1, zorder=4)
    ax.plot([x-.19, x+.19], [mid, mid], color=p.INK, lw=1.1, zorder=5)
