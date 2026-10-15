#!/usr/bin/env python3
"""Portable launcher for the manuscript's 7 x 5 projected protocol.
Prints the command by default. Training requires --execute. No import of the fitter.
"""
from pathlib import Path
import argparse
import os
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def command(input_dir, output_dir, seed=3000, sample='1k', photometry=None,
            init_from=None, systemic_from_data=False, resume=False):
    engine = ROOT / 'fitters' / ('projected_resume.py' if resume else 'projected.py')
    args = [sys.executable, '-u', str(engine), '--input-dir', str(Path(input_dir).resolve()),
            '--output-dir', str(Path(output_dir).resolve()), '--seed', str(seed),
            '--potential-model', 'corenfw', '--training-mode', 'projected', '--use-vsp-anchor',
            '--sigma-los-weight', '10', '--vsp-weight', '5', '--proj-n-w', '384',
            '--batch-size', '500', '--proj-n-phi', '48', '--skip-kernel-check',
            '--epochs-warmup', '60' if sample == '10k' else '250',
            '--epochs-joint', '200' if sample == '10k' else '800',
            '--scan-inner-slope', '--scan-epochs', '200', '--skip-laplace',
            '--rc-center-mult', '1.75', '--rc-hi-mult', '5.0',
            '--strong-prior-norm', 'nll_units', '--strong-prior-weight', '1.0',
            '--rc-prior-weight', '0', '--rs-prior-weight', '0', '--n-core-min', '-0.5',
            '--scan-grid=-0.5,-0.25,0.0,0.25,0.5,0.75,1.0',
            '--scan-rs-grid', '2,4,8,16,32', '--scan-rs-warmup', '100',
            '--proj-kernel', 'analytic', '--moment-rule', 'gl', '--moment-nt', '96',
            '--moment-neta', '32', '--df-hidden', '48', '--save-per-star-nll',
            '--pnll-no-checkpoint', '--gather-index-select', '--half-velocity-cells',
            '--warmup-freeze-potential', '--compile-cells']
    if photometry: args += ['--photometry-file', str(Path(photometry).resolve())]
    if init_from: args += ['--init-from', str(Path(init_from).resolve())]
    if systemic_from_data: args += ['--pm-use-systemic-from-data']
    if resume: args += ['--resume-checkpoints']
    return args

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_dir', type=Path); p.add_argument('output_dir', type=Path)
    p.add_argument('--seed', type=int, default=3000)
    p.add_argument('--sample', choices=['1k','10k'], default='1k')
    p.add_argument('--photometry', type=Path)
    p.add_argument('--init-from', type=Path)
    p.add_argument('--systemic-from-data', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--execute', action='store_true')
    a=p.parse_args()
    cmd=command(a.input_dir,a.output_dir,a.seed,a.sample,a.photometry,a.init_from,a.systemic_from_data,a.resume)
    print(shlex.join(cmd), flush=True)
    if not a.execute:
        print('Preview only. Add --execute to train.'); return
    if not (a.input_dir/'draco_nfw_observations.csv').is_file(): p.error('Missing input observations')
    if a.photometry and not a.photometry.is_file(): p.error('Missing photometry file')
    if a.init_from and not (a.init_from/'model_state.pt').is_file(): p.error('Missing initial joint state')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        if not a.resume: p.error('Output is nonempty; use a new directory or a checkpoint-enabled resume')
        if (a.output_dir/'metrics.json').exists(): p.error('Completed output already exists')
        if not (a.output_dir/'ckpt').is_dir(): p.error('No resume checkpoints in this directory')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    env=os.environ.copy()
    env.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.mpl-cache'))
    env.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ROOT/'outputs/.inductor_cache'))
    with (a.output_dir/'fit.log').open('a') as log:
        result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)

if __name__=='__main__': main()
