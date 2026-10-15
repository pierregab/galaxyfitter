"""Shared NewTX/PGF styling for the publication generators; opt-in by import."""
from pathlib import Path
import os
import subprocess
import json
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/figures'
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'.mpl-cache-publication'))
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
W1, W2 = 244/72.27, 508/72.27
BLUE, CORE, GREEN, INK, GREY = '#0072B2', '#D55E00', '#009E73', '#242424', '#777777'
FIGURES = []

def style():
    plt.rcParams.update({
        'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False,
        'pgf.preamble': r'\usepackage{amsmath,amssymb}\usepackage{newtxtext,newtxmath}',
        'font.family': 'serif', 'font.size': 8,
        'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'legend.fontsize': 7.2, 'legend.frameon': False,
        'legend.handlelength': 1.8, 'legend.labelspacing': .3,
        'axes.linewidth': .6, 'lines.linewidth': 1.05,
        'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.minor.width': .45, 'ytick.minor.width': .45,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.bbox': None, 'savefig.pad_inches': 0,
    })

def frame(ax, xminor=True, yminor=True):
    ax.grid(False)
    if xminor and ax.get_xscale() == 'linear':
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if yminor and ax.get_yscale() == 'linear':
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

def letter(ax, s, x=.045, y=.95):
    ax.text(x, y, '('+s+')', transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=.9), zorder=10)


def save(fig, stem, width, caption):
    for folder in ['pdf','pgf','png','captions']:
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'pgf'/(stem+'.pgf'))
    # Some canonical analysis imports select Agg globally. Explicitly choose
    # PGF here so the PDF still goes through pdflatex and embeds NewTX fonts.
    fig.savefig(OUT/'pdf'/(stem+'.pdf'),backend='pgf')
    subprocess.run(['pdftoppm','-r','220','-png','-singlefile',str(OUT/'pdf'/(stem+'.pdf')),str(OUT/'png'/stem)], check=True, capture_output=True)
    (OUT/'captions'/(stem+'.tex')).write_text(caption+'\n')
    FIGURES.append(dict(stem=stem,width=width,caption=caption))
    plt.close(fig)
    print('Wrote',OUT/'pdf'/(stem+'.pdf'))
