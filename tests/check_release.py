#!/usr/bin/env python3
"""CPU-only checks: inventory, complete maps, statistics, and launcher configuration parity.
Run scripts/reproduce.py first. No optimizer or fitter module is imported.
"""
from pathlib import Path
import argparse,ast,hashlib,json,math,os,sys
from dataclasses import dataclass,field,asdict
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from fit_projected import command


def equal(a,b,path='root'):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a: equal(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,f'{path}[{i}]')
    elif isinstance(a,(int,float)):
        assert np.isclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True),(path,a,b)
    else:assert a==b,(path,a,b)

# Only argument/config definitions are executed, never training code.
s=(ROOT/'fitters/projected.py').read_text();tree=ast.parse(s)
parts=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in {'parse_args','Config','apply_args'}]
exec(compile(ast.Module(body=parts,type_ignores=[]),'config_only','exec'))
for sample,lane,tag in [('1k','runs_1k','nfw_s3000'),('10k','runs_10k','gcusp_gs2_10k')]:
    sys.argv=command('input','output',sample=sample)[2:]
    actual=json.loads(json.dumps(asdict(apply_args(Config(),parse_args()))))
    expected=json.loads((ROOT/f'data/{lane}/{tag}/fit_config.json').read_text())
    for k in expected.keys() & actual.keys() - {'input_dir','output_dir','seed','resume_checkpoints'}:
        equal(expected[k],actual[k],sample+'.'+k)
print('Launcher matches stored 1k and 10k scientific configuration fields.')
for lane,count in [('runs_1k',89),('runs_10k',2)]:
    runs=list((ROOT/'data'/lane).glob('*/metrics.json'));assert len(runs)==count
    for m in runs:
        t=pd.read_csv(m.with_name('inner_slope_profile_scan_2d.csv'))
        assert len(t)==35 and not t.duplicated(['n','Rs_grid_Rhalf']).any(),m
        assert set(t.n)==set(np.arange(-.5,1.01,.25)),m
        assert set(t.Rs_grid_Rhalf)=={2,4,8,16,32},m
        assert np.isfinite(t[['nll_total','slope_at_Rhalf']].to_numpy()).all(),m
print('All 91 maps have complete finite 7 x 5 grids.')
equal(json.loads((ROOT/'reference/paper_numbers_grid2d.json').read_text()),json.loads((ROOT/'data/paper_numbers_grid2d.json').read_text()))
a=pd.read_csv(ROOT/'reference/paper_numbers_grid2d_runs.csv');b=pd.read_csv(ROOT/'data/paper_numbers_grid2d_runs.csv')
pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-10,atol=1e-10)
print('Recomputed paper summary and all 89 run readouts match the reference.')
manifest=json.loads((ROOT/'PROVENANCE.json').read_text())
for e in manifest['files']:
    p=ROOT/e['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'],p
print('All packaged source/data checksums match the inventory.')
