"""Recompute the paper's projected statistics from the included 89 maps.
A_asym minimizes calibration errors (first midpoint with cores oriented low).
dNLL uses the midpoint of class overlap extrema. The lower hull is a fixed
shape heuristic, not proof of convergence. Weights are not a Bayesian posterior.
Paths adapted for the release; numerical formulas copied from the paper script.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / 'data'
from plot_nll_2d import lower_hull_flags
import plot_calibrated_test_dual_paper as ctd                      # règles de seuil du papier (lecture seule)
import plot_projected_paper as pp

STY = pd.read_csv(HERE / 'families.csv').set_index('tag').family
NN_ = np.array([-.5, -.25, 0., .25, .5, .75, 1.])
rng = np.random.default_rng(0)


def trap(x, y):
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x) / 2)) if len(x) > 1 else 0.


def load(run):
    m = json.loads((run / 'metrics.json').read_text()); g = m['gravsphere_slope']
    t = pd.read_csv(run / 'inner_slope_profile_scan_2d.csv'); t['d'] = t.nll_total - t.nll_total.min(); t['bad'] = False
    for v, grp in t.groupby('Rs_grid_Rhalf'):
        t.loc[grp.index, 'bad'] = lower_hull_flags(grp.n.to_numpy(), grp.d.to_numpy())
    ok = t[~t.bad].copy(); ok['d'] = ok.nll_total - ok.nll_total.min()
    P = ok.groupby('n').nll_total.min().reindex(NN_).to_numpy(); d = P - np.nanmin(P)
    lo, hi = NN_ <= .5, NN_ > .5
    w = np.exp(-ok.d.to_numpy()); w /= w.sum(); s = ok.slope_at_Rhalf.to_numpy(); mean = float(w @ s)
    return dict(tag=run.name, truth=g['slope_at_Rhalf_true'], rh=g['R_half_2d_kpc'], gamma=mean,
                sd=float(np.sqrt(w @ (s - mean) ** 2)), dnll=float(P[-1] - P[0]),
                A=trap(NN_[lo], d[lo]) - trap(NN_[hi], d[hi]), node_d=ok.d.to_numpy(), node_s=s,
                beta1=float(w @ ok.beta_1.to_numpy()), beta05=float(w @ ok.beta_0p5.to_numpy()), mr3=m.get('M_r3_log10_ratio'),
                sig=m.get('sigma_los_chi2_per_bin'), vsp=m.get('vsp_chi2'))


runs = sorted(p.parent for p in (HERE / 'runs_1k').glob('*/metrics.json') if not p.parent.name.startswith('_'))
D = pd.DataFrame([load(r) for r in runs])
D['family'] = [STY.get(t, 'Gaia core' if t.startswith('gcore') else 'Gaia cusp') for t in D.tag]
D['core'] = D.family.isin(['Burkert', 'Gaia core', r'Burkert, $R_0$'])
D['err'] = D.gamma - D.truth
out = {}


def fit_threshold(v, core, core_high):
    """Règles du papier : A (cœur haut) minimise les erreurs parmi les milieux entre valeurs voisines, premier seuil en
    cas d'égalité avec les cœurs orientés bas (ctd.error_min_threshold) ; Delta NLL prend le milieu des extrema du
    recouvrement des classes (pp.threshold)."""
    if core_high:
        t, e = ctd.error_min_threshold(-v, core); return -float(t), int(e)
    t, _ = pp.threshold(v[~core], v[core]); return float(t), int(((v < t) != core).sum())


def classify(v, t, core_high):
    return (v > t) if core_high else (v < t)


def auc(v, core, core_high):
    a, b = (v[core], v[~core]) if core_high else (-v[core], -v[~core])
    return float(np.mean([(x > y) + .5 * (x == y) for x in a for y in b]))


cal = D[D.tag.str.match(r'(nfw|burkert)_s30')].reset_index(drop=True)
gaia = D[D.tag.str.startswith(('gcore', 'gcusp'))].reset_index(drop=True)
rules = {}
for key, core_high in (('A', True), ('dnll', False)):
    v, c = cal[key].to_numpy(), cal.family.eq('Burkert').to_numpy()
    t, e_in = fit_threshold(v, c, core_high)
    loo = sum(int(classify(v[i], fit_threshold(np.delete(v, i), np.delete(c, i), core_high)[0], core_high) != c[i]) for i in range(len(v)))
    gv, gc = gaia[key].to_numpy(), gaia.core.to_numpy()
    rules[key] = dict(threshold=t, errors_in=e_in, errors_loo=loo, auc=auc(v, c, core_high),
                      med_nfw=float(np.median(v[~c])), med_burkert=float(np.median(v[c])),
                      gaia_errors=int((classify(gv, t, core_high) != gc).sum()), gaia_auc=auc(gv, gc, core_high),
                      gaia_core_called_cusp=int((~classify(gv[gc], t, core_high)).sum()),
                      gaia_cusp_called_core=int(classify(gv[~gc], t, core_high).sum()),
                      # recouvrement des classes : cœur au-dessus (A) ou au-dessous (dnll) du seuil
                      overlap=float(max(0., (v[~c].max() - v[c].min()) if core_high else (v[c].max() - v[~c].min()))),
                      gaia_core_range=[float(gv[gc].min()), float(gv[gc].max())])
    a, b = v[c], v[~c]
    rules[key]['Z'] = float(abs(a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
out['rules'] = rules

rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
reg = np.where(D.truth < -.85, 'cuspy', np.where(D.truth > -.55, 'cored', 'intermediate'))
out['budget'] = {r: dict(n=int((reg == r).sum()), rmse=rm(D.err[reg == r]), bias=float(D.err[reg == r].mean()))
                 for r in ('cuspy', 'intermediate', 'cored')}
out['budget']['all'] = dict(n=len(D), rmse=rm(D.err), bias=float(D.err.mean()))
resp = float(np.polyfit(D.truth, D.gamma, 1)[0])
boot = [np.polyfit(D.truth.iloc[i], D.gamma.iloc[i], 1)[0] for i in (rng.integers(0, len(D), len(D)) for _ in range(2000))]
out['response'] = dict(value=resp, p16=float(np.percentile(boot, 16)), p84=float(np.percentile(boot, 84)))
out['by_set'] = {s: dict(n=int(len(g)), rmse=rm(g.err)) for s, g in (('calibration', cal), ('ladder', D[D.tag.str.startswith(('gnfw', 'burk_'))]),
                                                                    ('gaia', gaia), ('nfw', cal[~cal.family.eq('Burkert')]), ('burkert', cal[cal.family.eq('Burkert')]))}
out['spread_coverage'] = float((D.err.abs() <= D.sd).mean())
span = lambda r, th: float(np.ptp(r['node_s'][r['node_d'] <= th])) if (r['node_d'] <= th).sum() > 1 else 0.
out['node_span'] = {'1': float(np.median([span(r, 1) for _, r in D.iterrows()])), '2': float(np.median([span(r, 2) for _, r in D.iterrows()]))}

TH = np.arange(.25, 20.01, .25)
cov = np.array([[(r.node_s[r.node_d <= th].min() <= r.truth <= r.node_s[r.node_d <= th].max()) for th in TH] for _, r in D.iterrows()])
wid = np.array([[np.ptp(r.node_s[r.node_d <= th]) for th in TH] for _, r in D.iterrows()])


def th_for(target, idx):
    c = cov[idx].mean(0)
    k = int(np.argmax(c >= target)) if (c >= target).any() else len(TH) - 1   # cible non atteinte : plus grand seuil
    return float(TH[k]), k


t68, k68 = th_for(.68, np.arange(len(D))); t95, k95 = th_for(.95, np.arange(len(D)))
out['interval'] = dict(t68=t68, cov68=float(cov[:, k68].mean()), width68=float(np.median(wid[:, k68])),
                       t95=t95, cov95=float(cov[:, k95].mean()), width95=float(np.median(wid[:, k95])), lofo={})
for f in sorted(D.family.unique()):
    ins = np.flatnonzero(D.family != f); t_, k = th_for(.68, ins); o = np.flatnonzero(D.family == f)
    out['interval']['lofo'][f] = dict(n=len(o), threshold=t_, coverage=float(cov[o, k].mean()), width=float(np.median(wid[o, k])))
D['int68_lo'] = [r.node_s[r.node_d <= t68].min() for _, r in D.iterrows()]
D['int68_hi'] = [r.node_s[r.node_d <= t68].max() for _, r in D.iterrows()]

dg = cal.assign(cls=np.where(cal.family.eq('Burkert'), 'burkert', 'nfw'))
out['diagnostics'] = dict(mass_abs_max=float(np.abs(dg.mr3).max()),
                          mass_median={k: float(g.mr3.median()) for k, g in dg.groupby('cls')},
                          sigma_chi2_median={k: float(g.sig.median()) for k, g in dg.groupby('cls')},
                          vsp_median={k: float(g.vsp.median()) for k, g in dg.groupby('cls')}, vsp_max=float(dg.vsp.max()))

gaia = gaia.merge(D[['tag', 'int68_lo', 'int68_hi']], on='tag')
om = lambda r: float(r.rh ** 2 / (r.rh ** 2 + (.25 if r.tag.startswith('gcore') else .1) ** 2))
gaia['beta_err'] = [r.beta1 - om(r) for _, r in gaia.iterrows()]
out['gaia'] = dict(rmse_core=rm(gaia.err[gaia.core]), rmse_cusp=rm(gaia.err[~gaia.core]), rmse_all=rm(gaia.err),
                   beta_rmse_core=rm(gaia.beta_err[gaia.core]), beta_rmse_cusp=rm(gaia.beta_err[~gaia.core]),
                   cusp_errors=[float(x) for x in gaia.err[~gaia.core]], within_int68=int(((gaia.truth >= gaia.int68_lo) & (gaia.truth <= gaia.int68_hi)).sum()))
cmp = pd.read_csv(HERE / 'gs2_published_comparison.csv').set_index('case')
out['gs2'] = cmp.round(4).to_dict(orient='index')
D.drop(columns=['node_d', 'node_s']).to_csv(HERE / 'paper_numbers_grid2d_runs.csv', index=False)
(HERE / 'paper_numbers_grid2d.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
print(json.dumps(out, indent=1, default=lambda x: round(float(x), 3)))
