"""Figures du papier pour le protocole en grille 2D (partie projetée), écrites dans paper/figures/ par la chaîne du
papier (publication_style.save : pgf, pdf par pdflatex, png, légende dans captions/<stem>.tex). Nouveaux noms, aucun
fichier existant n'est écrasé ; aucune référence au scan 1D.

  fig_los_fit_diagnostics_grid2d   diagnostics des 60 fits joints de calibration (masse pivot, chi2 dispersion, VSP)
  fig_calibration_grid2d_paper     cartes médianes NFW/Burkert, profil sur R_s, dégénérescence beta-pente
  fig_calibrated_test_grid2d       test cusp/cœur calibré : A_asym et Delta NLL par mock, seuils, taux d'erreur
  fig_slope_grid2d_paper           pente à R1/2 lue contre vraie (intervalle calibré) et couverture contre le seuil
  fig_gs2_matched_grid2d           gamma(r) sur la paire appariée : grille 2D contre GravSphere2, 1k et 10k
Entrées : paper_numbers_grid2d.json et _runs.csv (paper_numbers_grid2d.py), runs_1k/, runs_10k/, profils GS2.
Usage : python paper_figures_grid2d.py [noms...]   (sans argument : toutes)
        --compact-diagnostics : version sur une colonne de la figure de diagnostics.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / 'data'
import publication_style as p
from plot_validation_panels_paper import strip, COLOURS
import plot_calibrated_test_dual_paper as ctd
import plot_projected_paper as pp
from plot_gaia_slope_paper import profiles
from plot_nll_2d import lower_hull_flags
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams['contour.negative_linestyle'] = 'solid'
p.OUT = REPO / 'outputs/figures'; p.style()
RED = '#C0392B'
NUM = json.loads((HERE / 'paper_numbers_grid2d.json').read_text())
RUNS = pd.read_csv(HERE / 'paper_numbers_grid2d_runs.csv')
NN_ = np.array([-.5, -.25, 0., .25, .5, .75, 1.]); RS_ = np.array([2., 4., 8., 16., 32.])
T68 = NUM['interval']['t68']
CAL = RUNS[RUNS.tag.str.match(r'(nfw|burkert)_s30')].copy(); CAL['halo'] = CAL.tag.str.split('_').str[0]


def nodes(run):
    t = pd.read_csv(run / 'inner_slope_profile_scan_2d.csv'); t['d'] = t.nll_total - t.nll_total.min(); t['bad'] = False
    for v, g in t.groupby('Rs_grid_Rhalf'):
        t.loc[g.index, 'bad'] = lower_hull_flags(g.n.to_numpy(), g.d.to_numpy())
    ok = t[~t.bad].copy(); ok['d'] = ok.nll_total - ok.nll_total.min()
    return t, ok


def fig_diagnostics():
    compact = '--compact-diagnostics' in sys.argv
    fig, axes = plt.subplots(1, 3, figsize=(p.W1, 1.85) if compact else (p.W2, 2.05))
    if compact:
        fig.subplots_adjust(left=.14, right=.985, bottom=.19, top=.95, wspace=.78)
    else:
        fig.subplots_adjust(left=.083, right=.982, bottom=.16, top=.94, wspace=.51)
    mass_label = r'$\log_{10}(M_{\rm fit}/M_{\rm true})$' if compact else r'$\log_{10}[M_{\rm fit}/M_{\rm true}]_{r_{\rm piv}}$'
    spec = [('mr3', mass_label, False),
            ('sig', r'$\chi^2_{\sigma}/N_{\rm bin}$', True), ('vsp', r'$\chi^2_{\rm VSP}$', True)]
    for ax, (key, label, log), letter in zip(axes, spec, 'abc'):
        for j, h in enumerate(COLOURS):
            strip(ax, CAL[CAL.halo == h][key].to_numpy(), j, COLOURS[h])
        ax.set_xlim(-.5, 1.5); ax.set_xticks([0, 1], ['NFW', 'Burkert']); ax.set_ylabel(label)
        if compact:
            ax.set_ylabel(label, fontsize=7.5, labelpad=2)
            ax.tick_params(axis='both', labelsize=6.8, pad=2)
            for points in ax.collections:
                points.set_sizes([5])
            for line in ax.lines:
                line.set_linewidth(min(line.get_linewidth(), 1.4))
        if log:
            ax.set_yscale('log'); ax.set_ylim(CAL[key].min() * .5, CAL[key].max() * 3)
        else:
            ax.axhline(0, color=p.GREY, ls=(0, (3, 2)), lw=.7); ax.set_ylim(-.16, .16)
            if compact:
                ax.set_yticks([-.1, 0, .1])
        # Keep the panel label clear of the NFW VSP outlier in the narrow layout.
        p.letter(ax, letter, x=.76 if compact and key == 'vsp' else .045)
        p.frame(ax, xminor=False)
    cap = (r'Diagnostics of the joint fits of the $60$ LOS calibration catalogues. (a) Pivot-mass error, (b) dispersion '
           r'residual per radial bin, and (c) the VSP residual. Every point is one catalogue; bars and ticks show the '
           r'central 68 per cent and median. The full range, including the large VSP residual, is retained. The moments '
           r'also regularize training, so their agreement is not an independent validation of the density slope or of the '
           r'cusp/core decision.')
    p.save(fig, 'fig_los_fit_diagnostics_grid2d', 'single' if compact else 'double', cap)


def fig_calibration():
    M, S, B, E, H = [], [], [], [], []
    for tag in CAL.tag:
        t, ok = nodes(HERE / 'runs_1k' / tag)
        rel = (t.nll_total - ok.nll_total.min())
        M.append(rel.groupby([t.Rs_grid_Rhalf, t.n]).first().unstack().reindex(index=RS_, columns=NN_).to_numpy())
        S.append(t.pivot(index='Rs_grid_Rhalf', columns='n', values='slope_at_Rhalf').reindex(index=RS_, columns=NN_).to_numpy())
        P = ok.groupby('n').nll_total.min().reindex(NN_).to_numpy(); E.append(P - np.nanmin(P)); H.append(tag.split('_')[0])
    M, S, E, H = np.array(M), np.array(S), np.array(E), np.array(H)
    fig = plt.figure(figsize=(p.W1, 3.95))
    gs = fig.add_gridspec(2, 2, left=.155, right=.855, top=.975, bottom=.105, wspace=.1, hspace=.5)
    maps = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]; cx, dx = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    cax = maps[1].inset_axes([1.06, 0, .07, 1])
    nf = np.linspace(-.5, 1, 160); lrf = np.linspace(np.log(2), np.log(32), 140); NNf, RRf = np.meshgrid(nf, np.exp(lrf))
    fine = lambda A: (lambda R: np.array([[np.interp(l, np.log(RS_), R[:, j]) for j in range(len(nf))] for l in lrf]))(
        np.array([np.interp(nf, NN_, r) for r in A]))
    for k, (ax, h, name) in enumerate(zip(maps, ('nfw', 'burkert'), ('NFW', 'Burkert'))):
        sel = H == h; Z = fine(np.nanmedian(M[sel], 0)); SL = fine(np.nanmedian(S[sel], 0))
        im = ax.pcolormesh(NNf, RRf, np.minimum(Z, 10), cmap='cividis_r', vmin=0, vmax=10, shading='auto', rasterized=True)
        ax.contour(NNf, RRf, Z, levels=[1, 4], colors='white', linewidths=[1.0, .6], linestyles=['-', (0, (3, 2))])
        ax.contour(NNf, RRf, SL, levels=[float(CAL[CAL.halo == h].truth.median())], colors=RED, linewidths=1.3)
        if h == 'nfw':
            ax.axhline(2.0 / float(CAL[CAL.halo == h].rh.median()), color='0.85', lw=.8, ls=(0, (5, 2, 1, 2)))
        ax.set_yscale('log'); ax.set_ylim(2, 32); ax.set_xlim(-.5, 1); ax.minorticks_off()
        ax.set_yticks(RS_); ax.set_yticklabels([f'{v:g}' for v in RS_] if k == 0 else [])
        ax.set_xticks([-.5, 0, .5, 1]); ax.set_xticklabels(['', '0', '0.5', '1']); ax.tick_params(labelsize=6.5, pad=2)
        ax.set_xlabel(r'$n$', labelpad=1, fontsize=7.5)
        ax.text(.05, .95, f'({"ab"[k]}) {name}', transform=ax.transAxes, va='top', fontsize=6.8,
                bbox=dict(boxstyle='round,pad=.18', fc='white', ec='none', alpha=.9))
    maps[0].set_ylabel(r'$R_s/R_{1/2}$', labelpad=2, fontsize=7.5)
    cb = fig.colorbar(im, cax=cax); cb.set_ticks([0, 5, 10]); cb.ax.tick_params(labelsize=6.3, pad=1.5)
    cb.set_label(r'median $\Delta$NLL', fontsize=6.8, labelpad=2)
    for h, c, mk, ls in (('nfw', p.BLUE, 'o', '-'), ('burkert', p.CORE, '^', (0, (5, 2)))):
        q = np.percentile(E[H == h], [5, 16, 50, 84, 95], axis=0)
        cx.fill_between(NN_ - 1, q[0], q[4], color=c, alpha=.10, lw=0); cx.fill_between(NN_ - 1, q[1], q[3], color=c, alpha=.23, lw=0)
        cx.plot(NN_ - 1, q[2], color=c, ls=ls, lw=1.1, marker=mk, ms=2.4, mew=0)
    cx.axvline(-.5, color=p.GREY, ls=(0, (1, 2)), lw=.7); cx.set_xlim(-1.53, .03); cx.set_ylim(-1, 60)
    cx.set_xticks([-1.5, -1, -.5, 0]); cx.set_xlabel(r'$\gamma_0=n-1$', labelpad=2, fontsize=7.5)
    cx.set_ylabel(r'$\Delta$NLL, profiled over $R_s$', labelpad=2, fontsize=7.5); cx.tick_params(labelsize=6.5, pad=2)
    cx.text(.05, .95, '(c)', transform=cx.transAxes, va='top', fontsize=7)
    dx.axhline(0, color='0.8', lw=.6, zorder=1); dx.axvline(0, color='0.8', lw=.6, zorder=1); txt = []
    for h, c, mk, name in (('nfw', p.BLUE, 'o', 'NFW'), ('burkert', p.CORE, '^', 'Burkert')):
        s = CAL[CAL.halo == h]; e = s.gamma - s.truth
        dx.scatter(s.beta05 if 'beta05' in s else s.beta1, e, s=10, marker=mk, color=c, alpha=.8, lw=0, zorder=3)
    for h, c, mk, name in (('nfw', p.BLUE, 'o', 'NFW'), ('burkert', p.CORE, '^', 'Burkert')):
        s = CAL[CAL.halo == h]; x = s.beta05 if 'beta05' in s else s.beta1; e = s.gamma - s.truth
        k_, c0 = np.polyfit(x, e, 1); xx = np.linspace(x.min(), x.max(), 10); dx.plot(xx, k_ * xx + c0, color=c, lw=.9, zorder=2)
        txt.append(f'{name}: $r={np.corrcoef(x, e)[0, 1]:.2f}$')
    dx.set_xlabel(r'fitted $\beta(0.5R_{1/2})$ (true 0)', labelpad=2, fontsize=7.5)
    dx.set_ylabel(r'error in $\gamma(R_{1/2})$', labelpad=2, fontsize=7.5)
    dx.yaxis.set_label_position('right'); dx.tick_params(axis='y', labelleft=False, labelright=True); dx.tick_params(labelsize=6.5, pad=2)
    dx.text(.05, .95, '(d)', transform=dx.transAxes, va='top', fontsize=7)
    dx.text(.97, .04, '\n'.join(txt), transform=dx.transAxes, ha='right', va='bottom', fontsize=5.9, linespacing=1.25,
            bbox=dict(boxstyle='round,pad=.2', fc='white', ec='none', alpha=.9))
    for a in (cx, dx):
        p.frame(a)
    hl = [Line2D([], [], color=RED, lw=1.3, label=r'true $\gamma(R_{1/2})$'),
          Line2D([], [], color='0.55', lw=.8, ls=(0, (5, 2, 1, 2)), label=r'true $R_s$ (NFW)'),
          Line2D([], [], color='0.55', lw=1.0, label=r'$\Delta$NLL$\,=1$'),
          Line2D([], [], color='0.55', lw=.6, ls=(0, (3, 2)), label=r'$\Delta$NLL$\,=4$'),
          Line2D([], [], color=p.BLUE, marker='o', ls='-', ms=3.2, lw=1.0, label='NFW (c, d)'),
          Line2D([], [], color=p.CORE, marker='^', ls=(0, (5, 2)), ms=3.2, lw=1.0, label='Burkert (c, d)')]
    fig.legend(handles=hl, loc='center', bbox_to_anchor=(.505, .545), ncol=3, fontsize=6.1, handlelength=1.9,
               columnspacing=.8, handletextpad=.35, frameon=False)
    cap = (r'Likelihood maps on the $(n,R_s)$ grid for the $30$ NFW and $30$ Burkert calibration catalogues. '
           r'(a, b) Node-wise median of $\Delta$NLL, each map taken relative to its own minimum; white contours at '
           r'$\Delta$NLL$\,=1$ and $4$. The red line marks nodes whose slope at $\Rhalf$ equals the true value, and the '
           r'dash-dotted line the true NFW scale radius, $2\kpc\simeq20\,\Rhalf$. (c) The profile over $R_s$, '
           r'$\min_{R_s}\mathrm{NLL}(n,R_s)$, relative to its minimum: lines and symbols are node-wise medians, dark and '
           r'light bands the 16th--84th and 5th--95th percentiles across catalogues. (d) Error of the slope readout at '
           r'$\Rhalf$ against the central anisotropy of the same readout; the mocks are isotropic, so any non-zero '
           r'$\beta$ is an error. Lines are least-squares fits within each class.')
    p.save(fig, 'fig_calibration_grid2d_paper', 'single', cap)


def fig_calibrated_test():
    rA, rD = NUM['rules']['A'], NUM['rules']['dnll']
    xa, ya = CAL[CAL.halo == 'nfw'].A.to_numpy(), CAL[CAL.halo == 'burkert'].A.to_numpy()
    xd, yd = CAL[CAL.halo == 'nfw'].dnll.to_numpy(), CAL[CAL.halo == 'burkert'].dnll.to_numpy()
    fig, axs = plt.subplots(2, 2, figsize=(p.W1, 3.85), gridspec_kw=dict(height_ratios=[.92, 1.], hspace=.62, wspace=.10,
                                                                          left=.175, right=.985, top=.94, bottom=.115))
    rng = np.random.default_rng(11)
    ea = ctd.draw(axs[0, 0], axs[1, 0], xa, ya, rA['threshold'], '.2f', r'$A_{\rm asym}$', 'ac', '', rng, left=True, legend=True, pub=p)
    ed = ctd.draw(axs[0, 1], axs[1, 1], xd, yd, rD['threshold'], '.1f', r'$\Delta\mathrm{NLL}$', 'bd', '', rng, left=False, legend=False, pub=p)
    assert ea == rA['errors_in'] and ed == rD['errors_in'], (ea, ed)
    fig.legend(handles=[Line2D([], [], color=pp.C_CUSP, lw=1.2, label='cusp called core'),
                        Line2D([], [], color=pp.C_CORE, lw=1.2, label='core called cusp')],
               loc='center', bbox_to_anchor=(.58, .505), ncol=2, columnspacing=1.2, handlelength=1.6)
    cap = (rf'Calibrated line-of-sight cusp/core test on the profile over $R_s$. \emph{{(a)}} $A_{{\rm asym}}$ and '
           rf'\emph{{(b)}} $\dnll$ per mock, with the calibrated thresholds $t_\star={rA["threshold"]:.2f}$ and '
           rf'${rD["threshold"]:.1f}$; bars mark class medians. \emph{{(c,d)}} Per-class error rates against the threshold. '
           rf'Neither statistic has an interval over which both rates vanish; the calibrated thresholds make '
           rf'{rA["errors_in"]} and {rD["errors_in"]} errors in $30+30$.')
    p.save(fig, 'fig_calibrated_test_grid2d', 'single', cap)


def fig_slope():
    STY = pd.read_csv(HERE / 'families.csv').drop_duplicates('family').set_index('family')
    ORDER = ['NFW', 'Burkert', r'gNFW, $R_s=2$', r'gNFW, $R_s=0.5$', r'Burkert, $R_0$', 'Gaia core', 'Gaia cusp']
    col = lambda f: STY.colour.get(f, '#9A6FBF' if f == 'Gaia core' else '#5B3A87')
    mk = lambda f: STY.marker.get(f, 'D')
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(p.W2, 3.3))
    fig.subplots_adjust(left=.085, right=.985, bottom=.14, top=.84, wspace=.28)
    lim = (-1.60, -.05); ax.axvspan(-.85, -.55, color='#F2F2F2', zorder=0, lw=0); ax.plot(lim, lim, color=p.INK, lw=.85, zorder=1)
    for f in ORDER:
        s = RUNS[RUNS.family == f]; dense = f in ('NFW', 'Burkert')
        ax.vlines(s.truth, s.int68_lo, s.int68_hi, color=col(f), lw=.6, alpha=.35 if dense else .6, zorder=2)
        ax.scatter(s.truth, s.gamma, s=10 if dense else 20, marker=mk(f), color=col(f), alpha=.6 if dense else .85, lw=0, zorder=4)
    b = NUM['budget']['all']; iv = NUM['interval']
    ax.text(.955, .055, rf'RMSE $={b["rmse"]:.3f}$' + '\n' + rf'truth within interval: ${100 * iv["cov68"]:.0f}$ per cent',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7.2, linespacing=1.3)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-1.5, -1, -.5]); ax.set_yticks([-1.5, -1, -.5])
    ax.set_xlabel(r'True $\gamma(R_{1/2})$'); ax.set_ylabel(r'Estimated $\gamma(R_{1/2})$'); p.letter(ax, 'a', x=.045, y=.96); p.frame(ax)
    # (b) couverture de l'intervalle {nœuds à Delta NLL <= seuil} selon le seuil, par régime de pente vraie
    TH = np.arange(.25, 12.01, .25); cov = {}
    reg = np.where(RUNS.truth < -.85, 'cuspy', np.where(RUNS.truth > -.55, 'cored', 'intermediate'))
    NODE = {tag: nodes(HERE / 'runs_1k' / tag)[1] for tag in RUNS.tag}
    C = np.array([[(lambda ok: ok.slope_at_Rhalf[ok.d <= th].min() <= tr <= ok.slope_at_Rhalf[ok.d <= th].max())(NODE[tag])
                   for th in TH] for tag, tr in zip(RUNS.tag, RUNS.truth)])
    for lab, sel, c, ls in (('all 89', np.ones(len(RUNS), bool), p.INK, '-'), (f'cored ({(reg == "cored").sum()})', reg == 'cored', p.CORE, '-'),
                            (f'intermediate ({(reg == "intermediate").sum()})', reg == 'intermediate', '0.55', (0, (3, 2))),
                            (f'cuspy ({(reg == "cuspy").sum()})', reg == 'cuspy', p.BLUE, '-')):
        bx.plot(TH, 100 * C[sel].mean(0), color=c, ls=ls, lw=1.3 if lab.startswith('all') else 1.0, label=lab)
    cusp_cov = C[reg == 'cuspy'].mean(0); t_cusp = float(TH[np.argmax(cusp_cov >= .68)])
    plateau = 100 * C[reg == 'cored'].mean(0).max()
    bx.axhline(68, color=p.GREY, lw=.7, ls=(0, (1, 2))); bx.axvline(T68, color=p.GREY, lw=.7, ls=(0, (5, 3)))
    bx.text(T68 + .15, 8, rf'$\Delta\mathrm{{NLL}}_\star={T68:.2f}$', fontsize=6.8, color='0.3')
    bx.set_xlim(0, 12); bx.set_ylim(0, 102); bx.set_xlabel(r'interval threshold $\Delta\mathrm{NLL}$')
    bx.set_ylabel('truth within interval [per cent]'); bx.legend(fontsize=6.8, loc='lower right', frameon=False)
    p.letter(bx, 'b', x=.045, y=.96); p.frame(bx)
    handles = [Line2D([], [], color=col(f), marker=mk(f), ls='', ms=4.4, label=f) for f in ORDER]
    fig.legend(handles=handles, ncol=7, loc='upper center', bbox_to_anchor=(.5, .995), fontsize=6.3, columnspacing=.7,
               handletextpad=.3)
    cap = (rf'Local-slope recovery at $\Rhalf$ for the $89$ LOS fits in seven families. \emph{{(a)}} Likelihood-weighted '
           rf'readout of the $(n,R_s)$ map against the truth; bars span the slopes of all nodes within '
           rf'$\Delta\mathrm{{NLL}}_\star={T68:.2f}$ of the minimum, the threshold at which this interval contains the '
           rf'truth for $68$ per cent of the fits. The shaded strip marks intermediate true slopes. \emph{{(b)}} Fraction of '
           rf'fits whose interval contains the truth, against the threshold, for all fits and by regime of the true slope. '
           rf'The cored and intermediate profiles reach $68$ per cent below $\Delta\mathrm{{NLL}}_\star$; the cuspy ones need '
           rf'$\Delta\mathrm{{NLL}}\simeq{t_cusp:.1f}$, because their likelihood minimum lies on the shallow side of the truth. The '
           rf'cored curve stops at ${plateau:.0f}$ per cent: those truths lie beyond the shallowest slope reached on the grid.')
    p.save(fig, 'fig_slope_grid2d_paper', 'double', cap)


def fig_gs2():
    cases = [('core', '1k', 'runs_1k/gcore_gs2_1k', 'gravsphere2_PlumCoreOm.npz', 0.),
             ('cusp', '1k', 'runs_1k/gcusp_gs2_1k', 'gravsphere2_PlumCuspOm.npz', 1.),
             ('core', '10k', 'runs_10k/gcore_gs2_10k', 'gravsphere2_PlumCoreOm_10k.npz', 0.),
             ('cusp', '10k', 'runs_10k/gcusp_gs2_10k', 'gravsphere2_PlumCuspOm_10k.npz', 1.)]
    G = NUM['gs2']
    fig, axes = plt.subplots(1, 4, figsize=(p.W2, 2.55), sharey=True)
    fig.subplots_adjust(left=.07, right=.99, bottom=.18, top=.8, wspace=.06)
    for ax, (halo, N, run, npz, gam), letter in zip(axes, cases, 'abcd'):
        run = HERE / run; rh = json.loads((run / 'metrics.json').read_text())['gravsphere_slope']['R_half_2d_kpc']
        t, ok = nodes(run); xr = np.geomspace(.1, 10, 181); r = xr * rh
        w = np.exp(-ok.d.to_numpy()); w /= w.sum(); gj = profiles(ok, r); est = w @ gj
        band = gj[ok.d.to_numpy() <= T68]
        color = p.CORE if halo == 'core' else p.BLUE
        truth = -gam + (gam - 3) * r / (1 + r)
        gs = np.load(REPO / 'data/gs2_profiles' / npz)
        gx = gs['r'] / rh; inside = (gx >= .1) & (gx <= 10)
        ax.fill_between(gx[inside], gs['slope_16'][inside], gs['slope_84'][inside], color='0.55', alpha=.22, lw=0, zorder=1)
        ax.plot(gx[inside], gs['slope_50'][inside], color='0.35', lw=1.0, zorder=3)
        ax.fill_between(xr, band.min(0), band.max(0), color=color, alpha=.22, lw=0, zorder=2)
        ax.plot(xr, est, color=color, lw=1.5, zorder=4)
        ax.plot(xr, truth, color=p.INK, lw=1.0, ls=(0, (5, 3)), zorder=5)
        ax.axvspan(.5, 2, color='#F4F4F4', zorder=0, lw=0)
        key = f'Plum{halo.capitalize()} {N}'; g = G[key]
        dfe = g['neo_grid_2d']; gse = g['gs2_local_posterior_median']
        ax.text(.04, .05, f'secant error\n2D grid ${dfe:+.2f}$\nGS2 ${gse:+.2f}$', transform=ax.transAxes, fontsize=6.2,
                va='bottom', linespacing=1.25, bbox=dict(boxstyle='round,pad=.25', fc='white', ec='0.7', alpha=.95))
        ax.set_xscale('log'); ax.set_xlim(.1, 10)
        ax.set_xticks([.1, .3, 1, 3, 10], ['0.1' if letter == 'a' else '', '0.3', '1', '3', '10'])   # pas de chevauchement
        ax.set_ylim(-2.6, .15); ax.set_xlabel(r'$r/R_{1/2}$')
        p.letter(ax, letter, x=.045, y=.95)
        ax.text(.955, .95, f'{"Core" if halo == "core" else "Cusp"}, {N.replace("k", r"$\,$000") if N == "10k" else "1000"} stars',
                transform=ax.transAxes, ha='right', va='top', fontsize=6.8)
        p.frame(ax, xminor=False)
    axes[0].set_ylabel(r'$\gamma(r)=\mathrm{d}\ln\rho/\mathrm{d}\ln r$')
    h = [Line2D([], [], color=p.INK, lw=1.0, ls=(0, (5, 3)), label='Input'),
         Line2D([], [], color=p.BLUE, lw=1.5, label=r'2D grid, readout'),
         Patch(facecolor=p.BLUE, alpha=.22, edgecolor='none', label=rf'2D grid, nodes within $\Delta\mathrm{{NLL}}_\star$'),
         Line2D([], [], color='0.35', lw=1.0, label=r'\textsc{GravSphere2}, median'),
         Patch(facecolor='0.55', alpha=.22, edgecolor='none', label='\\textsc{GravSphere2}, 16--84 per cent'),
         Patch(facecolor='#F4F4F4', edgecolor='0.8', lw=.4, label=r'secant interval')]
    fig.legend(handles=h, loc='upper center', ncol=3, fontsize=6.8, bbox_to_anchor=(.53, 1.0), columnspacing=1.2, handlelength=1.8)
    cap = (rf'Density-slope profiles on the matched \textsc{{GravSphere2}} catalogues: the core and the cusp with '
           rf'(a, b) $1000$ and (c, d) $10\,000$ tracers. Solid coloured lines are the likelihood-weighted readouts of '
           rf'the $(n,R_s)$ map, with the pointwise range of the node profiles within $\Delta\mathrm{{NLL}}_\star={T68:.2f}$ of '
           rf'the minimum (colour band); grey lines and bands are the \textsc{{GravSphere2}} posterior medians and 16--84 per '
           rf'cent ranges from the same stars; dashed lines are the input profiles. Boxes give the signed errors of the secant '
           rf'over $0.5$--$2\,\Rhalf$ (shaded), from the node readout and from the posterior median of the secant.')
    p.save(fig, 'fig_gs2_matched_grid2d', 'double', cap)


if __name__ == '__main__':
    todo = [arg for arg in sys.argv[1:] if arg != '--compact-diagnostics'] or ['diagnostics', 'calibration', 'test', 'slope', 'gs2']
    F = dict(diagnostics=fig_diagnostics, calibration=fig_calibration, test=fig_calibrated_test, slope=fig_slope, gs2=fig_gs2)
    for k in todo:
        F[k](); print('fait :', k)
