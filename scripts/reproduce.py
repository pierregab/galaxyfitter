#!/usr/bin/env python3
"""Recompute the packaged 89-map statistics; optionally render five paper figures. No fits."""
from pathlib import Path
import argparse,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--figures',action='store_true',help='Requires pdflatex/newtx and pdftoppm on PATH')
a=p.parse_args()
env=os.environ.copy();env.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.mpl-cache'))
subprocess.run([sys.executable,str(ROOT/'analysis/paper_numbers_grid2d.py')],cwd=ROOT,env=env,check=True)
if a.figures:
    subprocess.run([sys.executable,str(ROOT/'analysis/paper_figures_grid2d.py'),'--compact-diagnostics'],cwd=ROOT,env=env,check=True)
