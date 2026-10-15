import numpy as np
from matplotlib.ticker import AutoMinorLocator
C_CUSP, C_CORE, C_INK = '#0072B2', '#D55E00', '#333333'
COLOR = {'nfw': C_CUSP, 'burkert': C_CORE}
SHORT = {'nfw': 'NFW', 'burkert': 'Burkert'}

def threshold(x, y):
    if x.mean() < y.mean():
        return 0.5 * (x.max() + y.min()), bool(x.max() < y.min())
    return 0.5 * (y.max() + x.min()), bool(y.max() < x.min())

def minor(ax, x=True, y=True):
    """Minor ticks, skipping any axis on a log scale (where matplotlib
    supplies its own decade subticks and AutoMinorLocator is invalid)."""
    if x and ax.get_xscale() == "linear":
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if y and ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

def strip_h(ax, vals, ypos, color, rng, half=0.17, s=11, jit=0.055):
    vals = np.asarray(vals, float)
    ax.scatter(vals, ypos + rng.uniform(-jit, jit, len(vals)),
               s=s, color=color, alpha=0.80, lw=0, zorder=3)
    ax.vlines(np.median(vals), ypos - half, ypos + half,
              color=color, lw=2.0, zorder=4)

def panel(ax, letter, text, floor=7.4):
    """Left-aligned panel title that is checked against the axes width.

    A long title on a single-column figure silently ran past the right edge
    and squeezed the axes; nothing in the build caught it. Measure the drawn
    title, shrink it a little if it does not fit, and say so loudly if even
    the floor size is too wide -- the fix then is a shorter title, not a
    smaller font.
    """
    ax.set_title(f"({letter}) {text}", loc="left")
    fig = ax.figure
    fig.canvas.draw()
    t = ax.title
    while t.get_window_extent().width > ax.get_window_extent().width:
        size = t.get_fontsize() - 0.2
        if size < floor:
            print(f"  ! panel ({letter}) title overflows at {floor}pt, "
                  f"shorten it: {text!r}")
            return
        t.set_fontsize(size)
        fig.canvas.draw()
