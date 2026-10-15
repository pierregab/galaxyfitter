#!/usr/bin/env python3
"""
Historical calibration helper (not the current paper launcher).

The original defaults below use the retired projected implementation.
For current projected runs use scripts/fit_projected.py. Truth/observed
controls require their archived configuration overrides; changing the mode
alone is not a reproduction recipe. Scientific defaults are preserved.
============================================================
Frozen-protocol seed calibration of the cusp/core Delta_NLL test.

For each seed s:
  1. Generate a FRESH mock realization (new star draw + new noise draw)
     for BOTH halos, halo parameters held fixed:
         NFW     : M0 = 5e8 Msun, Rs = 2.0 kpc      (gamma0 = -1, cusp)
         Burkert : rho0 = 3e8 Msun/kpc^3, r0 = 0.3  (gamma0 =  0, core)
  2. Run the FROZEN fitting protocol (next_gen.py) on each mock with
     fitter seed = s. The protocol is identical for every seed and both
     halos -- never tuned per mock.
  3. Harvest delta_nll_core_minus_cusp, gamma0_profiled (+68% CI) and
     slope(R_half) from metrics.json.

Then build the calibration figure (the note's Section-4 money plot) and
a summary JSON quantifying the empirical separation of the two Delta_NLL
distributions (= the calibrated cusp/core test + honest error rate).

Usage:
    python run_calibration.py --n-seeds 10 --seed0 101
    python run_calibration.py --seeds 101 102 103
    python run_calibration.py --plot-only            # re-plot cached results
    python run_calibration.py --fit-repeats 3 --seeds 101   # optimizer-jitter check

Resume-safe: any (halo, seed, repeat) whose metrics.json already exists
is skipped unless --force is given.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter, NullFormatter

# common radial grid in units of each mock's half-light radius
XG_RHALF = np.logspace(np.log10(0.1), np.log10(10.0), 140)   # r / R_half

# Observed-mode ensemble diagnostics
ESS_CDF_GRID = np.linspace(0.0, 1.0, 201)

# Bins used for the ensemble radial-deconvolution diagnostic.
# The plotted quantity is the per-run RMS error in log10 radius.
DECONV_EDGES_RHALF = np.logspace(
    np.log10(0.1),
    np.log10(10.0),
    10,
)
DECONV_CENTERS_RHALF = np.sqrt(
    DECONV_EDGES_RHALF[:-1] * DECONV_EDGES_RHALF[1:]
)

# ============================================================
# Publication style (self-contained copy of setup_pub_style)
# ============================================================
C_NFW = "#0072B2"      # Okabe-Ito blue   (cusp)
C_BUR = "#D55E00"      # Okabe-Ito vermilion (core)
C_GREY = "0.45"


def setup_pub_style(use_tex=False):
    """LaTeX-like serif figures; mathtext-CM needs no TeX install."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "text.usetex": use_tex,
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 4.0, "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
        "legend.frameon": False, "legend.handlelength": 1.5,
        "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 110,
    })


def save_pub(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name + ".pdf"))
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=300)


# ============================================================
# Frozen experiment definition
# ============================================================
HALOS = {
    "nfw": dict(
        gen_script=str(REPO_ROOT / "mocks/data_generation_nfw.py"),
        gen_args=["--m0-nfw", "5.0e8", "--rs-nfw", "2.0", "--a-star", "0.1",
                  "--noise-preset", "learnable"],
        gamma0_true=-1.0,
        label="NFW (cusp)",
        color=C_NFW,
    ),
    "burkert": dict(
        gen_script=str(REPO_ROOT / "mocks/data_generation_burkert.py"), 
        gen_args=["--rho0", "3.0e8", "--r0", "0.3", "--a-star", "0.10",
                  "--noise-preset", "learnable"],
        gamma0_true=0.0,
        label="Burkert (core)",
        color=C_BUR,
    ),
}

# The FROZEN protocol. Identical for every seed and both halos.

FROZEN_FIT = [
    "--potential-model", "corenfw", "--training-mode", "projected",
    "--use-vsp-anchor", "--sigma-los-weight", "10", "--vsp-weight", "5",
    "--proj-n-w", "384",
    "--batch-size", "500", "--epochs-joint", "800",
    "--scan-inner-slope", "--scan-epochs", "200", "--skip-laplace",
    "--rc-center-mult", "1.75", "--rc-hi-mult", "5.0",
    "--strong-prior-norm", "nll_units", "--strong-prior-weight", "1.0",
    "--rc-prior-weight", "5", "--rs-prior-weight", "2",
]


# ============================================================
# Run management
# ============================================================
def sh(cmd, log_path):
    print("  $ " + " ".join(shlex.quote(c) for c in cmd))
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"  !! FAILED (see {log_path}) -- continuing with other runs")
        return False
    return True


def ensure_mock(halo, seed, base, py, force):
    mock_dir = os.path.join(base, "mocks", f"{halo}_s{seed}")
    obs = os.path.join(mock_dir, "draco_nfw_observations.csv")
    if os.path.exists(obs) and not force:
        return mock_dir, True
    os.makedirs(mock_dir, exist_ok=True)
    h = HALOS[halo]
    cmd = ([py, h["gen_script"], "--output-dir", mock_dir,
            "--seed", str(seed), "--n-stars", "1000"] + h["gen_args"])
    ok = sh(cmd, os.path.join(mock_dir, "generator.log"))
    return mock_dir, ok and os.path.exists(obs)


def ensure_fit(halo, seed, rep, mock_dir, base, fitter, py, force, fit_extra):
    tag = f"{halo}_s{seed}" + (f"_r{rep}" if rep > 0 else "")
    run_dir = os.path.join(base, "runs", tag)
    metrics = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics) and not force:
        return run_dir, True
    os.makedirs(run_dir, exist_ok=True)
    fit_seed = seed + 5000 * rep          # rep>0 = same mock, new optimizer seed
    cmd = ([py, fitter, "--input-dir", mock_dir, "--output-dir", run_dir,
            "--seed", str(fit_seed)] + FROZEN_FIT + fit_extra)
    ok = sh(cmd, os.path.join(run_dir, "fit.log"))
    return run_dir, ok and os.path.exists(metrics)


def harvest(base, halos, seeds, repeats):
    rows = []
    for halo in halos:
        for s in seeds:
            for rep in range(repeats):
                tag = f"{halo}_s{s}" + (f"_r{rep}" if rep > 0 else "")
                mpath = os.path.join(base, "runs", tag, "metrics.json")
                if not os.path.exists(mpath):
                    continue
                with open(mpath) as f:
                    m = json.load(f)
                scan = m.get("inner_slope_scan")
                if scan is None:
                    continue
                rows.append({
                    "halo": halo, "seed": s, "repeat": rep,
                    "delta_nll": scan["delta_nll_core_minus_cusp"],
                    "gamma0": scan["gamma0_profiled"],
                    "gamma0_lo": scan["gamma0_68CI"][0],
                    "gamma0_hi": scan["gamma0_68CI"][1],
                    "slope_Rhalf": scan["slope_at_Rhalf_profiled"],
                    "slope_Rhalf_true": m["gravsphere_slope"]["slope_at_Rhalf_true"],
                    "verdict": scan["verdict"],
                    "M_r3_log10_ratio": m.get("M_r3_log10_ratio", np.nan),
                })
    return pd.DataFrame(rows)

# ============================================================
# Observed-mode posterior diagnostics
# ============================================================
def harvest_posterior_diagnostics(base, halos, seeds, repeats=1):
    """
    Harvest per-star posterior diagnostics written by fitter.py.

    Expected per-run file:
        posterior_star_diagnostics.csv

    The function is intentionally tolerant:
      - ESS diagnostics can be used without a truth catalogue;
      - deconvolution diagnostics require r_true_kpc;
      - old runs lacking the file are simply skipped.
    """
    rows = []

    for halo in halos:
        for seed in seeds:
            for rep in range(repeats):
                tag = (
                    f"{halo}_s{seed}"
                    + (f"_r{rep}" if rep > 0 else "")
                )

                run_dir = os.path.join(base, "runs", tag)
                diag_path = os.path.join(
                    run_dir,
                    "posterior_star_diagnostics.csv",
                )
                metrics_path = os.path.join(
                    run_dir,
                    "metrics.json",
                )

                if not os.path.exists(diag_path):
                    continue

                try:
                    tab = pd.read_csv(diag_path)
                except Exception as exc:
                    print(
                        f"[diag] could not read {diag_path}: {exc}"
                    )
                    continue

                if len(tab) == 0:
                    continue

                metrics = {}
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path) as f:
                            metrics = json.load(f)
                    except Exception:
                        metrics = {}

                # Recover K robustly.
                if "K" in tab.columns:
                    kvals = pd.to_numeric(
                        tab["K"],
                        errors="coerce",
                    ).dropna()
                    K_diag = (
                        float(kvals.iloc[0])
                        if len(kvals)
                        else np.nan
                    )
                else:
                    K_diag = float(
                        metrics.get(
                            "posterior_diagnostics",
                            {},
                        ).get("K", np.nan)
                    )
                    tab["K"] = K_diag

                # Recover normalized ESS if only the raw ESS was saved.
                if (
                    "posterior_ess_over_K" not in tab.columns
                    and "posterior_ess" in tab.columns
                    and np.isfinite(K_diag)
                    and K_diag > 0
                ):
                    tab["posterior_ess_over_K"] = (
                        tab["posterior_ess"] / K_diag
                    )

                required = {
                    "posterior_ess",
                    "posterior_ess_over_K",
                }
                if not required.issubset(tab.columns):
                    print(
                        f"[diag] missing ESS columns in {diag_path}; "
                        "skipping this run."
                    )
                    continue

                R_half = (
                    metrics
                    .get("gravsphere_slope", {})
                    .get("R_half_2d_kpc", np.nan)
                )

                tab["halo"] = halo
                tab["seed"] = int(seed)
                tab["repeat"] = int(rep)
                tab["run_tag"] = tag
                tab["R_half"] = float(R_half)

                rows.append(tab)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True, sort=False)


def posterior_run_summaries(diag):
    """
    Reduce per-star diagnostics to one row per mock realization.

    These run-level values are the correct units for quoting uncertainty:
    seeds are independent realizations; stars within one realization are not.
    """
    rows = []

    if diag.empty:
        return pd.DataFrame()

    for (halo, seed, rep), g in diag.groupby(
        ["halo", "seed", "repeat"]
    ):
        ess = pd.to_numeric(
            g["posterior_ess"],
            errors="coerce",
        ).to_numpy(float)

        ess_frac = pd.to_numeric(
            g["posterior_ess_over_K"],
            errors="coerce",
        ).to_numpy(float)

        valid = np.isfinite(ess) & np.isfinite(ess_frac)
        ess = ess[valid]
        ess_frac = ess_frac[valid]

        if len(ess) == 0:
            continue

        K_vals = pd.to_numeric(
            g.get("K", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()

        K_diag = (
            float(K_vals.iloc[0])
            if len(K_vals)
            else np.nan
        )

        rec = {
            "halo": halo,
            "seed": int(seed),
            "repeat": int(rep),
            "K": K_diag,
            "n_stars": int(len(ess)),
            "ess_mean": float(np.mean(ess)),
            "ess_median": float(np.median(ess)),
            "ess_p05": float(np.percentile(ess, 5)),
            "ess_p16": float(np.percentile(ess, 16)),
            "ess_p84": float(np.percentile(ess, 84)),
            "median_ess_over_K": float(np.median(ess_frac)),
            "fraction_ess_below_K_over_4": float(
                np.mean(ess_frac < 0.25)
            ),
            "fraction_ess_below_K_over_10": float(
                np.mean(ess_frac < 0.10)
            ),
        }

        truth_cols = {
            "r_true_kpc",
            "r_noisy_kpc",
            "r_posterior_mean_kpc",
        }

        if truth_cols.issubset(g.columns):
            r_true = pd.to_numeric(
                g["r_true_kpc"],
                errors="coerce",
            ).to_numpy(float)

            r_noisy = pd.to_numeric(
                g["r_noisy_kpc"],
                errors="coerce",
            ).to_numpy(float)

            r_post = pd.to_numeric(
                g["r_posterior_mean_kpc"],
                errors="coerce",
            ).to_numpy(float)

            good = (
                np.isfinite(r_true)
                & np.isfinite(r_noisy)
                & np.isfinite(r_post)
                & (r_true > 0.0)
                & (r_noisy > 0.0)
                & (r_post > 0.0)
            )

            if np.any(good):
                err_noisy = np.log10(
                    r_noisy[good] / r_true[good]
                )
                err_post = np.log10(
                    r_post[good] / r_true[good]
                )

                rec.update({
                    "radius_log_rmse_noisy": float(
                        np.sqrt(np.mean(err_noisy ** 2))
                    ),
                    "radius_log_rmse_posterior": float(
                        np.sqrt(np.mean(err_post ** 2))
                    ),
                    "radius_log_bias_noisy": float(
                        np.median(err_noisy)
                    ),
                    "radius_log_bias_posterior": float(
                        np.median(err_post)
                    ),
                    "radius_rmse_improvement_factor": float(
                        np.sqrt(np.mean(err_noisy ** 2))
                        / max(
                            np.sqrt(np.mean(err_post ** 2)),
                            1.0e-12,
                        )
                    ),
                })

        rows.append(rec)

    return pd.DataFrame(rows)


def summarize_posterior_runs(run_summary):
    """
    Median and 16--84% seed-to-seed intervals of the run-level diagnostics.
    """
    if run_summary.empty:
        return {}

    d0 = run_summary[run_summary["repeat"] == 0].copy()
    out = {}

    groups = [
        (halo, d0[d0["halo"] == halo])
        for halo in sorted(d0["halo"].unique())
    ]
    groups.append(("all", d0))

    cols = [
        "median_ess_over_K",
        "fraction_ess_below_K_over_4",
        "fraction_ess_below_K_over_10",
        "radius_log_rmse_noisy",
        "radius_log_rmse_posterior",
        "radius_rmse_improvement_factor",
    ]

    for name, g in groups:
        if len(g) == 0:
            continue

        rec = {
            "n_runs": int(len(g)),
        }

        kvals = g["K"].dropna().unique()
        if len(kvals):
            rec["K_values"] = [
                int(round(float(x)))
                for x in sorted(kvals)
            ]

        for col in cols:
            if col not in g.columns:
                continue

            x = pd.to_numeric(
                g[col],
                errors="coerce",
            ).dropna().to_numpy(float)

            if len(x) == 0:
                continue

            rec[col] = {
                "median": float(np.median(x)),
                "p16": float(np.percentile(x, 16)),
                "p84": float(np.percentile(x, 84)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            }

        out[name] = rec

    return out

# ============================================================
# NEW: aggregated paper plots rebuilt from the per-run scan tables.
# coreNFW is analytic -> no torch, no refit, works on cached runs.
# ============================================================
def truth_profiles_np(halo, r):
    """Analytic truth (rho, dln rho/dln r). MUST match HALOS gen_args."""
    r = np.asarray(r, float)
    if halo == "nfw":
        M0, Rs = 5.0e8, 2.0
        x = np.maximum(r / Rs, 1e-12)
        rho = (M0 / (4 * np.pi * Rs**3)) / (x * (1 + x) ** 2)
        slope = -1.0 - 2.0 * x / (1.0 + x)
    else:  # burkert
        rho0, r0 = 3.0e8, 0.3
        x = np.maximum(r / r0, 1e-12)
        rho = rho0 / ((1.0 + x) * (1.0 + x * x))
        slope = -(x / (1.0 + x) + 2.0 * x * x / (1.0 + x * x))
    return rho, slope


def corenfw_rho_np(r, M0, Rs, rc, n):
    """Analytic coreNFW density (mirrors CoreNFWPotential.density)."""
    r = np.maximum(np.asarray(r, float), 1e-12)
    x = r / Rs
    Mnfw = M0 * (np.log1p(x) - x / (1.0 + x))
    rhonfw = (M0 / (4 * np.pi * Rs**3)) / (x * (1.0 + x) ** 2)
    f = np.clip(np.tanh(r / rc), 1e-12, 1.0)
    rho = (f**n * rhonfw
           + n * f**(n - 1.0) * (1.0 - f * f) * Mnfw / (4 * np.pi * r * r * rc))
    return np.maximum(rho, 1e-300)

def corenfw_slope_np(r, M0, Rs, rc, n, dl=1e-3):
    """d ln rho / d ln r of the analytic coreNFW (central log-difference,
    mirrors CoreNFWPotential.log_slope)."""
    r = np.asarray(r, float)
    lp = np.log(corenfw_rho_np(r * np.exp(+dl), M0, Rs, rc, n))
    lm = np.log(corenfw_rho_np(r * np.exp(-dl), M0, Rs, rc, n))
    return (lp - lm) / (2.0 * dl)


def add_slope_probe(curves, x_probe):
    """Attach the fitted slope at r_probe = x_probe * R_half to every
    scan node. Purely analytic -> works on all cached runs."""
    out = curves.copy()
    out["slope_probe"] = [
        float(corenfw_slope_np(np.array([x_probe * row.R_half]),
                               row.M0, row.Rs, row.r_core, row.n_core)[0])
        for row in curves.itertuples()
    ]
    return out

def _profile_in_slope(gs):
    """One run's NLL as a function of slope_probe: sort nodes by slope and
    take the lower NLL envelope (nodes were profiled at fixed n, so if two
    nodes land on near-equal probe slopes -- e.g. a hidden-core n=1 node
    collapsing onto the cusp -- the better fit represents that slope)."""
    s = gs["slope_probe"].values
    nll = gs["nll_total"].values
    o = np.argsort(s)
    s, nll = s[o], nll[o]
    ks, kn = [s[0]], [nll[0]]
    for si, ni in zip(s[1:], nll[1:]):
        if si - ks[-1] < 1e-3:                 # near-duplicate slope
            kn[-1] = min(kn[-1], ni)
        else:
            ks.append(si); kn.append(ni)
    return np.asarray(ks), np.asarray(kn)


def compute_slope_statistic(curves, s_cusp, s_core):
    """Per-seed DNLL in the slope-at-r_probe coordinate:
       dnll_slope = NLL(slope = s_core) - NLL(slope = s_cusp),
    linearly interpolated on the node envelope. 'clamped' flags seeds whose
    node slopes do not bracket a hypothesis point (statistic then
    conservative for that seed). Piecewise-linear interpolation between
    profiled nodes is an approximation; the empirical calibration absorbs it."""
    rows = []
    for (halo, seed), gs in curves.groupby(["halo", "seed"]):
        s, nll = _profile_in_slope(gs)
        rows.append(dict(
            halo=halo, seed=seed,
            dnll_slope=float(np.interp(np.clip(s_core, s[0], s[-1]), s, nll)
                             - np.interp(np.clip(s_cusp, s[0], s[-1]), s, nll)),
            slope_min=float(s[0]), slope_max=float(s[-1]),
            clamped=bool(s_cusp < s[0] or s_core > s[-1]),
        ))
    return pd.DataFrame(rows)


def slope_statistic_summary(stat, old_df):
    """Separation report for the new statistic, with the old one alongside."""
    out = {}
    for halo in stat.halo.unique():
        x = stat.loc[stat.halo == halo, "dnll_slope"].values
        out[halo] = {"n": int(len(x)),
                     "median": float(np.median(x)),
                     "min": float(np.min(x)), "max": float(np.max(x)),
                     "n_clamped": int(stat.loc[stat.halo == halo, "clamped"].sum())}
    if {"nfw", "burkert"} <= set(out):
        nfw = stat.loc[stat.halo == "nfw", "dnll_slope"].values
        bur = stat.loc[stat.halo == "burkert", "dnll_slope"].values
        out["separation_gap_slope"] = float(np.min(nfw) - np.max(bur))
        out["threshold_midgap"] = float(0.5 * (np.min(nfw) + np.max(bur)))
        d0 = old_df[old_df.repeat == 0]
        out["separation_gap_endpoint_for_reference"] = float(
            d0.loc[d0.halo == "nfw", "delta_nll"].min()
            - d0.loc[d0.halo == "burkert", "delta_nll"].max())
    return out

def scan_slope_protocols(curves, old_df, outdir):
    """
    Grid-search diagnostic for the slope-coordinate statistic.

    Safe version:
    - if only one halo class is present, it writes an empty scan table
      and skips the two-class separation scan cleanly.
    """
    x_grid = [0.25, 0.35, 0.50, 0.70, 1.00, 1.40]
    s_cusp_grid = [-1.0]
    s_core_grid = [-0.15, -0.25, -0.35, -0.45, -0.55]

    cols_all = [
        "x_probe", "s_cusp", "s_core", "gap", "threshold_midgap",
        "nfw_median", "burkert_median",
        "nfw_min", "nfw_max", "burkert_min", "burkert_max",
        "nfw_errors_midgap", "burkert_errors_midgap",
        "n_clamped_total", "nfw_clamped", "burkert_clamped",
    ]

    present = set(curves["halo"].unique()) if len(curves) else set()
    if not {"nfw", "burkert"} <= present:
        tab = pd.DataFrame(columns=cols_all)
        path = os.path.join(outdir, "slope_protocol_scan.csv")
        tab.to_csv(path, index=False)

        print("\n===== SLOPE-PROTOCOL SCAN SKIPPED =====")
        print(
            "Need both NFW and Burkert completed scan curves, "
            f"but found only: {sorted(present)}"
        )
        print(f"[scan] wrote empty table: {path}")
        return tab

    rows = []

    for x_probe in x_grid:
        curves_p = add_slope_probe(curves, x_probe)

        for s_cusp in s_cusp_grid:
            for s_core in s_core_grid:
                stat = compute_slope_statistic(curves_p, s_cusp, s_core)

                if not {"nfw", "burkert"} <= set(stat.halo.unique()):
                    continue

                nfw = stat.loc[stat.halo == "nfw", "dnll_slope"].values
                bur = stat.loc[stat.halo == "burkert", "dnll_slope"].values

                if len(nfw) == 0 or len(bur) == 0:
                    continue

                gap = float(np.min(nfw) - np.max(bur))
                t_mid = float(0.5 * (np.min(nfw) + np.max(bur)))

                nfw_err = int(np.sum(nfw < t_mid))
                bur_err = int(np.sum(bur > t_mid))

                rows.append({
                    "x_probe": float(x_probe),
                    "s_cusp": float(s_cusp),
                    "s_core": float(s_core),
                    "gap": gap,
                    "threshold_midgap": t_mid,
                    "nfw_median": float(np.median(nfw)),
                    "burkert_median": float(np.median(bur)),
                    "nfw_min": float(np.min(nfw)),
                    "nfw_max": float(np.max(nfw)),
                    "burkert_min": float(np.min(bur)),
                    "burkert_max": float(np.max(bur)),
                    "nfw_errors_midgap": nfw_err,
                    "burkert_errors_midgap": bur_err,
                    "n_clamped_total": int(stat["clamped"].sum()),
                    "nfw_clamped": int(stat.loc[stat.halo == "nfw", "clamped"].sum()),
                    "burkert_clamped": int(stat.loc[stat.halo == "burkert", "clamped"].sum()),
                })

    tab = pd.DataFrame(rows, columns=cols_all)

    if tab.empty:
        path = os.path.join(outdir, "slope_protocol_scan.csv")
        tab.to_csv(path, index=False)
        print("\n===== SLOPE-PROTOCOL SCAN EMPTY =====")
        print("No valid two-class slope protocols could be computed.")
        print(f"[scan] wrote empty table: {path}")
        return tab

    tab = tab.sort_values(
        ["n_clamped_total", "nfw_errors_midgap", "burkert_errors_midgap", "gap"],
        ascending=[True, True, True, False],
    )

    path = os.path.join(outdir, "slope_protocol_scan.csv")
    tab.to_csv(path, index=False)

    print("\n===== SLOPE-PROTOCOL SCAN: TOP 15 =====")
    cols = [
        "x_probe", "s_cusp", "s_core", "gap", "threshold_midgap",
        "nfw_median", "burkert_median",
        "nfw_errors_midgap", "burkert_errors_midgap",
        "n_clamped_total",
    ]
    print(tab[cols].head(15).to_string(index=False))

    return tab

def _weighted_quantile(vals, w, qs):
    """vals [n_models, n_x], w [n_models] -> [len(qs), n_x]."""
    out = np.empty((len(qs), vals.shape[1]))
    for j in range(vals.shape[1]):
        o = np.argsort(vals[:, j])
        v, ww = vals[o, j], w[o]
        cw = (np.cumsum(ww) - 0.5 * ww) / ww.sum()
        out[:, j] = np.interp(qs, cw, v)
    return out


def harvest_scan_curves(base, halos, seeds):
    """Per-node profile-scan tables for every completed run (repeat 0),
    with R_half, the tracer-support radii, and the mock directory
    attached from metrics.json."""
    rows = []
    for halo in halos:
        for s in seeds:
            run = os.path.join(base, "runs", f"{halo}_s{s}")
            p = os.path.join(run, "inner_slope_profile_scan.csv")
            mp = os.path.join(run, "metrics.json")
            if not (os.path.exists(p) and os.path.exists(mp)):
                continue
            with open(mp) as f:
                m = json.load(f)
            t = pd.read_csv(p)
            t["halo"], t["seed"] = halo, s
            t["R_half"] = m.get("gravsphere_slope", {}).get(
                "R_half_2d_kpc", np.nan)
            sup = m.get("support", {})
            t["r05_kpc"] = sup.get("r05", np.nan)
            t["r95_kpc"] = sup.get("r95", np.nan)
            t["mock_dir"] = os.path.join(base, "mocks", f"{halo}_s{s}")
            rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def discordant_seeds(df, k=3.0):
    """Per-class robust outliers of the endpoint statistic, used ONLY to
    style/label curves in the aggregated plots. Returns {halo: {seeds}}."""
    out = {}
    d0 = df[df.repeat == 0]
    for halo, g in d0.groupby("halo"):
        x = g["delta_nll"].values.astype(float)
        med = np.median(x)
        mad = 1.4826 * np.median(np.abs(x - med)) + 1e-12
        out[halo] = set(g.loc[np.abs(x - med) / mad > k, "seed"].astype(int))
    return out

# ============================================================
# The money plot + summary
# ============================================================
def make_summary(df):
    out = {}
    d0 = df[df.repeat == 0]
    for halo in df.halo.unique():
        x = d0.loc[d0.halo == halo, "delta_nll"].values
        out[halo] = {
            "n": int(len(x)),
            "delta_nll_median": float(np.median(x)),
            "delta_nll_min": float(np.min(x)),
            "delta_nll_max": float(np.max(x)),
            "delta_nll_16_84": [float(np.percentile(x, 16)),
                                float(np.percentile(x, 84))],
        }
    if {"nfw", "burkert"} <= set(out):
        nfw = d0.loc[d0.halo == "nfw", "delta_nll"].values
        bur = d0.loc[d0.halo == "burkert", "delta_nll"].values
        # decision rule: delta_nll < 0 => core
        out["threshold"] = 0.0
        out["nfw_misclassified_as_core"] = int(np.sum(nfw < 0))
        out["burkert_misclassified_as_cusp"] = int(np.sum(bur > 0))
        out["separation_gap"] = float(np.min(nfw) - np.max(bur))
        out["note"] = ("gap > 0 means the two distributions do not overlap at "
                       "these N_seeds; error rate then bounded by ~1/(n+1) per class.")
    reps = df[df.repeat > 0]
    if len(reps):
        j = (df.groupby(["halo", "seed"])["delta_nll"].std(ddof=1)
               .dropna())
        if len(j):
            out["optimizer_jitter_std_delta_nll"] = float(j.median())
    return out

# Protocol anchor weights (must match FROZEN_FIT)
ANCHOR_W = dict(sig=10.0, pm=0.0, vsp=5.0)

def per_seed_features(curves):
    """Candidate statistics per (halo, seed), all from the cached scan CSV.
    Sign convention: negative => core-like, matching delta_nll."""
    rows = []
    for (halo, seed), g in curves.groupby(["halo", "seed"]):
        g = g.sort_values("n").reset_index(drop=True)
        lo, hi = g.iloc[0], g.iloc[-1]                   # n=0 / n=1 nodes
        anch = lambda r: (ANCHOR_W["sig"] * r.anchor_sig
                          + ANCHOR_W["pm"] * r.anchor_pm
                          + ANCHOR_W["vsp"] * r.anchor_vsp)
        nll = g["nll_total"].values
        gam = g["gamma_inner"].values
        d = nll - nll.min()
        core = gam > -0.5
        # curve-integrated evidence ratio: uses ALL nodes -> variance reduction
        ev = (np.log(np.sum(np.exp(-d[~core])))
              - np.log(np.sum(np.exp(-d[core]))))
        rows.append(dict(
            halo=halo, seed=seed,
            dnll=float(hi.nll_total - lo.nll_total),        # current statistic
            danchor=float(anch(hi) - anch(lo)),             # anchors only
            dL=float(hi.L_composite - lo.L_composite),      # full composite
            ev_ratio=float(ev),
            # slope of the NLL profile at the core end (cores: steep wall)
            core_end_slope=float((nll[-1] - nll[-3]) / (gam[-1] - gam[-3])),
        ))
    return pd.DataFrame(rows)


def class_sep(stat, col):
    nfw = stat.loc[stat.halo == "nfw", col].dropna().values
    bur = stat.loc[stat.halo == "burkert", col].dropna().values

    if len(nfw) == 0 or len(bur) == 0:
        return dict(
            skipped=True,
            reason="Need non-empty NFW and Burkert samples.",
            n_nfw=int(len(nfw)),
            n_burkert=int(len(bur)),
        )

    pooled = 0.5 * (
        np.std(nfw, ddof=1) if len(nfw) > 1 else 0.0
    ) + 0.5 * (
        np.std(bur, ddof=1) if len(bur) > 1 else 0.0
    )

    gap = float(np.min(nfw) - np.max(bur))

    return dict(
        gap=gap,
        gap_over_spread=float(gap / max(pooled, 1e-9)),
        dprime=float((np.mean(nfw) - np.mean(bur)) / max(pooled, 1e-9)),
        n_overlap=int(np.sum(nfw <= np.max(bur))),
        midgap_threshold=float(0.5 * (np.min(nfw) + np.max(bur))),
        n_nfw=int(len(nfw)),
        n_burkert=int(len(bur)),
    )


def loo_lda(stat, cols):
    """Leave-one-seed-out Fisher discriminant on 2+ features:
    honest error count without touching the held-out seed."""
    X = stat[cols].values
    y = (stat.halo == "burkert").values
    scores, errs = np.zeros(len(y)), 0
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        mu0, mu1 = X[m & ~y].mean(0), X[m & y].mean(0)
        Xc = np.vstack([X[m & ~y] - mu0, X[m & y] - mu1])
        S = np.cov(Xc.T) + 1e-6 * np.eye(len(cols))
        wv = np.linalg.solve(S, mu1 - mu0)
        scores[i] = X[i] @ wv - 0.5 * wv @ (mu0 + mu1)
        errs += int((scores[i] > 0) != y[i])
    return scores, errs


def plot_calibration(df, outdir, use_tex=False):
    setup_pub_style(use_tex=use_tex)
    d0 = df[df.repeat == 0].copy()

    fig, axs = plt.subplots(2, 1, figsize=(3.4, 5.0),
                            gridspec_kw=dict(height_ratios=[1.0, 1.0],
                                             hspace=0.42))

    # ---- (a) Delta_NLL strip plot: THE calibration ----
    ax = axs[0]
    rng = np.random.default_rng(0)
    ypos = {"nfw": 1.0, "burkert": 0.0}
    for halo, g in d0.groupby("halo"):
        h = HALOS[halo]
        y = ypos[halo] + rng.uniform(-0.10, 0.10, len(g))
        ax.scatter(g["delta_nll"], y, s=22, color=h["color"],
                   edgecolor="k", linewidth=0.4, zorder=5, label=h["label"])
        med = np.median(g["delta_nll"])
        ax.plot([med, med], [ypos[halo] - 0.22, ypos[halo] + 0.22],
                color=h["color"], lw=2.2, zorder=4)
    ax.axvline(0.0, color="k", ls="--", lw=1.0)
    ax.text(0.985, 0.04, "core", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=8, color=C_GREY)
    # place 'cusp' just right of zero
    ax.annotate("cusp", xy=(0, 0), xycoords=("data", "axes fraction"),
                xytext=(4, 3), textcoords="offset points",
                fontsize=8, color=C_GREY, ha="left", va="bottom")
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["Burkert\n(core)", "NFW\n(cusp)"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel(r"$\Delta\mathrm{NLL} \equiv \mathrm{NLL}(n{=}1)-\mathrm{NLL}(n{=}0)$")
    ax.set_title("(a) Calibrated cusp/core statistic", loc="left")
    ax.legend(loc="center left", fontsize=7.5)

    # ---- (b) profiled gamma0 per seed vs truth ----
    ax = axs[1]
    for halo, g in d0.groupby("halo"):
        h = HALOS[halo]
        g = g.sort_values("seed").reset_index(drop=True)
        x = np.arange(len(g))
        yerr = np.vstack([g["gamma0"] - g["gamma0_lo"],
                          g["gamma0_hi"] - g["gamma0"]])
        yerr = np.clip(yerr, 0.0, None)
        ax.errorbar(x, g["gamma0"], yerr=yerr, fmt="o", ms=4,
                    color=h["color"], ecolor=h["color"], elinewidth=0.9,
                    capsize=1.8, label=h["label"])
        ax.axhline(h["gamma0_true"], color=h["color"], ls=":", lw=1.0)
    ax.set_xlabel("seed index")
    ax.set_ylabel(r"profiled $\gamma_0$")
    ax.set_ylim(-1.15, 0.15)
    ax.set_title(r"(b) Profiled inner slope (dotted: truth)", loc="left")
    ax.legend(loc="center right", fontsize=7.5)

    save_pub(fig, outdir, "calibration_money_plot")
    plt.close(fig)
    print(f"[plot] wrote {outdir}/calibration_money_plot.pdf/.png")

def plot_inner_slope_profiles(curves, outdir, use_tex=False, outliers=None):
    """Delta NLL(gamma0) per seed, normalized to each seed's minimum.
    Discordant seeds (endpoint-statistic outliers) are drawn dashed and
    labeled instead of blending into the bundle; bands and medians are
    computed over ALL seeds. Panel (a) y-axis is capped at the 99th
    percentile of all curve values, with a note if any curve is clipped."""
    setup_pub_style(use_tex=use_tex)
    outliers = outliers or {}
    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(3.4, 4.9), sharex=True,
        gridspec_kw=dict(height_ratios=[1.35, 1.0], hspace=0.15))

    y99, ymax_true = [], []
    for halo in [h for h in ("nfw", "burkert") if (curves.halo == h).any()]:
        h, g = HALOS[halo], curves[curves.halo == halo]
        gam = np.sort(g["gamma_inner"].unique())
        piv = g.pivot_table(index="seed", columns="gamma_inner",
                            values="nll_total").reindex(columns=gam)
        mat = piv.values - np.nanmin(piv.values, axis=1, keepdims=True)
        q16, q50, q84 = np.nanpercentile(mat, [16, 50, 84], axis=0)
        y99.append(np.nanpercentile(mat, 99))
        ymax_true.append(np.nanmax(mat))
        bad = np.array([int(s) in outliers.get(halo, set())
                        for s in piv.index])

        for ax in (axa, axb):
            for row, is_bad in zip(mat, bad):
                if is_bad:
                    ax.plot(gam, row, color=h["color"], lw=1.0, ls="--",
                            alpha=0.85, zorder=4)
                else:
                    ax.plot(gam, row, color=h["color"], lw=0.55,
                            alpha=0.22, zorder=2)
            ax.fill_between(gam, q16, q84, color=h["color"], alpha=0.16,
                            lw=0, zorder=3)
        for row, is_bad, sd in zip(mat, bad, piv.index):
            if is_bad:
                axa.annotate(f"seed {int(sd)}", xy=(gam[-1], row[-1]),
                             xytext=(-3, 3), textcoords="offset points",
                             ha="right", va="bottom", fontsize=6,
                             color=h["color"])
        axa.plot(gam, q50, "-o", color=h["color"], lw=1.9, ms=3.0, zorder=5,
                 label=f"{h['label']}  ($N_{{\\rm seed}}={mat.shape[0]}$)")
        axb.plot(gam, q50, "-o", color=h["color"], lw=1.9, ms=3.0, zorder=5)

    ymax = 1.06 * max(y99)
    if max(ymax_true) > ymax:
        axa.annotate("axis clipped at 99th pct",
                     xy=(0.02, 0.985), xycoords="axes fraction",
                     ha="left", va="top", fontsize=6.2, color="0.45")
    axa.axhspan(0.0, 3.0, color="0.94", zorder=0)
    axa.annotate("zoom in (b)", xy=(0.985, 3.0),
                 xycoords=("axes fraction", "data"),
                 ha="right", va="bottom", fontsize=6.5, color="0.45")
    axa.set_ylim(-0.03 * ymax, ymax)
    axa.set_ylabel(r"$\Delta\mathrm{NLL}(\gamma_0)$ per seed")
    axa.set_title("(a) Profile likelihoods, all seeds", loc="left")
    axa.legend(loc="upper center", fontsize=7.2)

    pct = (lambda s: s + r"\%") if use_tex else (lambda s: s + "%")
    for th, lab in ((0.5, pct("68")), (1.92, pct("95"))):
        axb.axhline(th, ls=":", lw=0.9, color="0.35")
        axb.annotate(lab, xy=(0.995, th), xycoords=("axes fraction", "data"),
                     xytext=(0, 1.5), textcoords="offset points",
                     ha="right", va="bottom", fontsize=6.8, color="0.35")
    axb.set_xlim(-1.05, 0.05); axb.set_ylim(-0.12, 3.0)
    axb.set_xlabel(r"asymptotic inner slope $\gamma_0 = n-1$")
    axb.set_ylabel(r"$\Delta\mathrm{NLL}$ (zoom)")
    axb.set_title("(b) Interval region (Wilks lines approximate)", loc="left")
    axb.text(0.02, 0.90, r"$\leftarrow$ cusp", transform=axb.transAxes,
             ha="left", va="bottom", fontsize=7.5, color=C_GREY)
    axb.text(0.98, 0.90, r"core $\rightarrow$", transform=axb.transAxes,
             ha="right", va="bottom", fontsize=7.5, color=C_GREY)

    save_pub(fig, outdir, "calibration_inner_slope_profiles")
    plt.close(fig)
    print(f"[plot] wrote {outdir}/calibration_inner_slope_profiles.pdf/.png")

def plot_family_bands(curves, outdir, use_tex=False, outliers=None):
    """Likelihood-weighted coreNFW family bands (w ~ exp(-dNLL) over scan
    nodes), rebuilt analytically and aggregated over seeds in r/R_half
    units.

    Band = median over seeds of the per-seed 16-84% family band
    (typical single-run uncertainty); thin lines = per-seed weighted
    medians (seed scatter); thick = median of medians; dotted = truth.

    Presentation-only styling:
    seeds flagged in `outliers` are drawn dashed and labeled, but they
    remain in all bands, medians, and summary statistics.
    """
    setup_pub_style(use_tex=use_tex)
    outliers = outliers or {}

    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(3.4, 4.9), sharex=True,
        gridspec_kw=dict(hspace=0.15)
    )

    for halo in [h for h in ("nfw", "burkert") if (curves.halo == h).any()]:
        h, g = HALOS[halo], curves[curves.halo == halo]

        lo_s, med_s, hi_s = [], [], []
        lo_r, med_r, hi_r = [], [], []

        # Store per-seed weighted medians so discordant seeds can be styled
        # differently without removing them from the computation.
        per_seed = []  # list of (seed, slope_median, log_density_ratio_median)

        for seed, gs in g.groupby("seed"):
            Rh = float(gs["R_half"].iloc[0])
            if not np.isfinite(Rh):
                continue

            r = XG_RHALF * Rh
            rho_true, _ = truth_profiles_np(halo, r)

            # Likelihood weights over scan nodes for this seed.
            w = np.exp(-(gs["nll_total"].values - gs["nll_total"].min()))
            w /= w.sum()

            rho = np.array([
                corenfw_rho_np(r, m0, rs, rc, n)
                for m0, rs, rc, n in zip(
                    gs["M0"], gs["Rs"], gs["r_core"], gs["n_core"]
                )
            ])

            slope = np.gradient(np.log(rho), np.log(r), axis=1)
            lratio = np.log(rho) - np.log(rho_true)[None, :]

            sq = _weighted_quantile(slope, w, [0.16, 0.50, 0.84])
            rq = _weighted_quantile(lratio, w, [0.16, 0.50, 0.84])

            lo_s.append(sq[0])
            med_s.append(sq[1])
            hi_s.append(sq[2])

            lo_r.append(rq[0])
            med_r.append(rq[1])
            hi_r.append(rq[2])

            per_seed.append((int(seed), sq[1], rq[1]))

        if len(med_s) == 0:
            continue

        lo_s, med_s, hi_s = map(np.asarray, (lo_s, med_s, hi_s))
        lo_r, med_r, hi_r = map(np.asarray, (lo_r, med_r, hi_r))

        # ---- (a) slope family band ----
        axa.fill_between(
            XG_RHALF,
            np.median(lo_s, axis=0),
            np.median(hi_s, axis=0),
            color=h["color"],
            alpha=0.15,
            lw=0,
        )

        # ---- (b) density-ratio family band ----
        axb.fill_between(
            XG_RHALF,
            np.exp(np.median(lo_r, axis=0)),
            np.exp(np.median(hi_r, axis=0)),
            color=h["color"],
            alpha=0.15,
            lw=0,
        )

        # Per-seed weighted medians.
        # Discordant seeds are dashed + labeled, but still included above.
        for seed, srow, rrow in per_seed:
            is_bad = seed in outliers.get(halo, set())

            st = dict(
                color=h["color"],
                lw=1.0 if is_bad else 0.55,
                ls="--" if is_bad else "-",
                alpha=0.75 if is_bad else 0.20,
            )

            axa.plot(XG_RHALF, srow, **st)
            axb.plot(XG_RHALF, np.exp(rrow), **st)

            if is_bad:
                axa.annotate(
                    str(seed),
                    xy=(XG_RHALF[0], srow[0]),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize=6,
                    color=h["color"],
                )

        # Thick class median over seed medians.
        axa.plot(
            XG_RHALF,
            np.median(med_s, axis=0),
            color=h["color"],
            lw=1.9,
            label=h["label"],
        )

        axb.plot(
            XG_RHALF,
            np.exp(np.median(med_r, axis=0)),
            color=h["color"],
            lw=1.9,
        )

        # Truth profile, evaluated at the median R_half for this class.
        _, slope_true = truth_profiles_np(
            halo,
            XG_RHALF * float(np.median(g["R_half"]))
        )
        axa.plot(
            XG_RHALF,
            slope_true,
            ls=":",
            color=h["color"],
            lw=1.5,
        )

    axa.plot([], [], ls=":", color="0.25", lw=1.5, label="truth (per halo)")

    for ax in (axa, axb):
        ax.set_xscale("log")
        ax.set_xlim(XG_RHALF[0], XG_RHALF[-1])
        ax.axvline(1.0, color="0.35", ls="--", lw=0.9)

    axa.annotate(
        r"$R_{1/2}$",
        xy=(1.0, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, -9),
        textcoords="offset points",
        fontsize=7.5,
        color="0.35",
    )

    axa.set_ylim(-3.3, 0.3)
    axa.set_ylabel(r"$\mathrm{d}\ln\rho/\mathrm{d}\ln r$")
    axa.set_title("(a) 3D log-density slope", loc="left")
    axa.legend(loc="upper right", fontsize=7.2)

    axb.axhline(1.0, color="k", lw=1.0)
    axb.set_yscale("log")
    axb.set_ylim(0.3, 3.3)
    axb.set_yticks([0.5, 1.0, 2.0, 3.0])
    axb.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axb.yaxis.set_minor_formatter(NullFormatter())
    axb.set_ylabel(r"$\rho_{\rm fit}/\rho_{\rm true}$")
    axb.set_xlabel(r"$r/R_{1/2}$")
    axb.set_title("(b) Density relative to truth", loc="left")

    save_pub(fig, outdir, "calibration_family_bands")
    plt.close(fig)
    print(f"[plot] wrote {outdir}/calibration_family_bands.pdf/.png")

def plot_slope_statistic(curves, stat, old_df, s_cusp, s_core, x_probe,
                         outdir, use_tex=False):
    setup_pub_style(use_tex=use_tex)
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.4, 5.2),
                                   gridspec_kw=dict(hspace=0.42))

    # (a) per-seed profiles vs slope at r_probe, normalized to own minimum
    for halo in [h for h in ("nfw", "burkert") if (curves.halo == h).any()]:
        h = HALOS[halo]
        first = True
        for _, gs in curves[curves.halo == halo].groupby("seed"):
            s, nll = _profile_in_slope(gs)
            axa.plot(s, nll - nll.min(), color=h["color"], lw=0.7, alpha=0.35,
                     label=h["label"] if first else None)
            first = False
    for sv, lab, c in ((s_cusp, "cusp hyp.", C_NFW), (s_core, "core hyp.", C_BUR)):
        axa.axvline(sv, color=c, ls="--", lw=1.1)
        axa.annotate(lab, xy=(sv, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(2, -10), textcoords="offset points",
                     fontsize=7, color=c, rotation=90, va="top")
    axa.set_xlabel(rf"$\mathrm{{d}}\ln\rho/\mathrm{{d}}\ln r$ "
                   rf"at $r={x_probe:g}\,R_{{1/2}}$ (per node)")
    axa.set_ylabel(r"$\Delta\mathrm{NLL}$ per seed")
    axa.set_ylim(bottom=-1.5)
    axa.set_title("(a) Profile likelihood, slope coordinate", loc="left")
    axa.legend(loc="upper center", fontsize=7.2)

    # (b) endpoint statistic vs slope statistic, per seed
    d0 = old_df[old_df.repeat == 0]
    m = stat.merge(d0[["halo", "seed", "delta_nll"]], on=["halo", "seed"])
    for halo, g in m.groupby("halo"):
        h = HALOS[halo]
        axb.scatter(g["delta_nll"], g["dnll_slope"], s=20, color=h["color"],
                    edgecolor="k", linewidth=0.4, label=h["label"], zorder=5)
        cl = g[g["clamped"]]
        if len(cl):
            axb.scatter(cl["delta_nll"], cl["dnll_slope"], s=52, marker="o",
                        facecolor="none", edgecolor="k", linewidth=0.6, zorder=6)
    axb.axhline(0, color="0.5", lw=0.8, ls=":")
    axb.axvline(0, color="0.5", lw=0.8, ls=":")
    axb.set_xlabel(r"endpoint $\Delta\mathrm{NLL}$ ($n{=}1$ vs $n{=}0$)")
    axb.set_ylabel(r"slope-coordinate $\Delta\mathrm{NLL}$")
    axb.set_title("(b) Endpoint vs slope statistic (circled: clamped)",
                  loc="left")
    axb.legend(loc="upper left", fontsize=7.2)

    save_pub(fig, outdir, "calibration_slope_statistic")
    plt.close(fig)
    print(f"[plot] wrote {outdir}/calibration_slope_statistic.pdf/.png")

def _ecdf_on_grid(x, grid):
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.full_like(grid, np.nan, dtype=float)

    return (
        np.searchsorted(x, grid, side="right")
        / float(len(x))
    )


def _per_run_radial_rmse(g, edges):
    """
    Per-run RMS log10-radius error in bins of true r/R_half.
    """
    required = {
        "r_true_kpc",
        "r_noisy_kpc",
        "r_posterior_mean_kpc",
        "R_half",
    }

    if not required.issubset(g.columns):
        nbin = len(edges) - 1
        return (
            np.full(nbin, np.nan),
            np.full(nbin, np.nan),
        )

    Rh = float(g["R_half"].iloc[0])

    if not np.isfinite(Rh) or Rh <= 0.0:
        nbin = len(edges) - 1
        return (
            np.full(nbin, np.nan),
            np.full(nbin, np.nan),
        )

    r_true = pd.to_numeric(
        g["r_true_kpc"],
        errors="coerce",
    ).to_numpy(float)

    r_noisy = pd.to_numeric(
        g["r_noisy_kpc"],
        errors="coerce",
    ).to_numpy(float)

    r_post = pd.to_numeric(
        g["r_posterior_mean_kpc"],
        errors="coerce",
    ).to_numpy(float)

    good = (
        np.isfinite(r_true)
        & np.isfinite(r_noisy)
        & np.isfinite(r_post)
        & (r_true > 0.0)
        & (r_noisy > 0.0)
        & (r_post > 0.0)
    )

    x = r_true[good] / Rh

    err_noisy = np.log10(
        r_noisy[good] / r_true[good]
    )
    err_post = np.log10(
        r_post[good] / r_true[good]
    )

    rmse_noisy = np.full(len(edges) - 1, np.nan)
    rmse_post = np.full(len(edges) - 1, np.nan)

    for j in range(len(edges) - 1):
        if j == len(edges) - 2:
            m = (
                (x >= edges[j])
                & (x <= edges[j + 1])
            )
        else:
            m = (
                (x >= edges[j])
                & (x < edges[j + 1])
            )

        # Avoid unstable single-run bins.
        if np.sum(m) < 15:
            continue

        rmse_noisy[j] = np.sqrt(
            np.mean(err_noisy[m] ** 2)
        )
        rmse_post[j] = np.sqrt(
            np.mean(err_post[m] ** 2)
        )

    return rmse_noisy, rmse_post


def _column_quantiles(arr, qs=(16, 50, 84)):
    """
    Column-wise quantiles while tolerating bins that are empty in some runs.

    Parameters
    ----------
    arr : array-like, shape (n_runs, n_bins)
    qs : tuple
        Percentiles to calculate.

    Returns
    -------
    out : ndarray, shape (len(qs), n_bins)
    """
    arr = np.asarray(arr, dtype=float)

    if arr.ndim != 2:
        raise ValueError("_column_quantiles expects a two-dimensional array.")

    out = np.full((len(qs), arr.shape[1]), np.nan)

    for j in range(arr.shape[1]):
        x = arr[:, j]
        x = x[np.isfinite(x)]

        if len(x):
            out[:, j] = np.percentile(x, qs)

    return out


# ============================================================
# Drop-in replacement for plot_observed_posterior_diagnostics()
# in run_calibration.py.
#
# Same signature, same inputs (the harvested per-star diagnostics
# DataFrame), same output filename ("observed_posterior_diagnostics"),
# so nothing else in the harness or in the LaTeX source changes.
#
# Panel (a): per-star ESS/K ECDFs -- faint per-run curves, the
#   across-run 16--84% band, and the median ECDF per class (the same
#   thin-curves + band + median language as the profile-likelihood
#   figure). A dot marks where each median ECDF crosses 0.5; its
#   x-position is the class median ESS/K. The K/4 warning level is
#   marked, with a compact numeric summary in the empty mid-plot
#   region (layout verified collision-free).
# Panel (b): radius reconstruction with the improvement factor at
#   R_1/2 annotated.
#
# Uses the helpers already defined in run_calibration.py:
#   setup_pub_style, save_pub, HALOS, _ecdf_on_grid,
#   _per_run_radial_rmse, _column_quantiles,
#   DECONV_EDGES_RHALF, DECONV_CENTERS_RHALF
# ============================================================

def plot_observed_posterior_diagnostics(
    diag,
    outdir,
    use_tex=False,
    log_tail=True,
):
    """
    Compact ensemble diagnostic for the observed-mode study.

    Panel (a):
        Empirical CDFs of the per-star ESS/K: faint curves are
        individual mock realizations, the shaded band is the
        across-run 16--84% interval, thick lines are the class
        median ECDFs, and the dot marks the class median ESS/K.
        The K/4 reference marks the conventional importance-sampling
        warning level. With log_tail=True (default) the cumulative
        fraction is on a log scale, so the rare low-ESS tail spans
        decades and the curve height at the K/4 line directly reads
        off the fraction of stars below K/4. Set log_tail=False for
        the linear version.

    Panel (b):
        Radial reconstruction error. The naive noisy result is pooled
        over both halo classes because it is a property of the
        observational transformation, while posterior results are shown
        separately for NFW and Burkert fits. The arrow at R_1/2 gives
        the median improvement factor of the posterior deconvolution.
    """
    if diag.empty:
        print(
            "[diag] no posterior-star diagnostics found; "
            "skipping observed diagnostic plot."
        )
        return

    setup_pub_style(use_tex=use_tex)

    d0 = diag[diag["repeat"] == 0].copy()

    from matplotlib.ticker import (
        FixedLocator,
        FuncFormatter,
        LogLocator,
        MultipleLocator,
        NullFormatter,
    )

    fig = plt.figure(figsize=(3.4, 5.35))

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.18],
        hspace=0.34,
    )

    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])

    # ========================================================
    # (a) per-star ESS/K, log-scaled in the deficit 1 - ESS/K
    # ========================================================
    # A plain log axis in ESS/K would compress the informative
    # region (everything sits near 1). Log-scaling the *deficit*
    # d = 1 - ESS/K stretches exactly that region. Ticks are
    # labeled in ESS/K units, so the curve reads directly as
    # "fraction of stars with ESS/K of at least x".
    D_MIN = 2.0e-3
    d_grid = np.logspace(np.log10(D_MIN), 0.0, 601)

    med_essK, frac_K4 = {}, {}

    for halo in ("nfw", "burkert"):
        gh = d0[d0["halo"] == halo]
        if gh.empty:
            continue

        color = HALOS[halo]["color"]
        cdfs, med_list, frac_list = [], [], []

        for (_, _), g in gh.groupby(["seed", "repeat"]):
            x = pd.to_numeric(
                g["posterior_ess_over_K"],
                errors="coerce",
            ).to_numpy(float)
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue

            d = np.clip(1.0 - x, D_MIN, 1.0)
            c = _ecdf_on_grid(d, d_grid)
            cdfs.append(c)
            med_list.append(np.median(x))
            frac_list.append(np.mean(x < 0.25))

            # faint per-run curves: same visual language as the
            # profile-likelihood figure (thin curves + band + median)
            axa.plot(d_grid, c, color=color,
                     lw=0.5, alpha=0.15, zorder=2)

        if not cdfs:
            continue

        cdfs = np.asarray(cdfs)
        q16, q50, q84 = np.percentile(cdfs, [16, 50, 84], axis=0)

        # across-run 16--84% band and median curve
        axa.fill_between(d_grid, q16, q84, color=color,
                         alpha=0.16, lw=0, zorder=3)
        axa.plot(d_grid, q50, color=color, lw=1.9,
                 zorder=5, label=HALOS[halo]["label"])

        # dot where the median curve crosses 0.5:
        # its x-position is the class median deficit
        m_ess = float(np.median(med_list))
        axa.plot([max(1.0 - m_ess, D_MIN)], [0.5],
                 marker="o", ms=3.6, color=color,
                 markeredgecolor="white", markeredgewidth=0.5,
                 zorder=6)

        med_essK[halo] = m_ess
        frac_K4[halo] = float(np.median(frac_list))

    # guides: median level and the K/4 warning line (deficit 0.75)
    axa.axhline(0.5, color="0.88", lw=0.6, zorder=0)
    # short K/4 marker (full-height line would slice the legend region)
    axa.axvline(0.75, ymax=0.32, color="0.45", lw=0.85, ls=":", zorder=1)
    axa.annotate(
        r"$K/4$",
        xy=(0.75, 0.345),
        xycoords=("data", "axes fraction"),
        ha="center", va="bottom",
        fontsize=7.2, color="0.40",
    )

    axa.set_xscale("log")
    axa.set_xlim(D_MIN, 1.0)
    axa.set_ylim(-0.02, 1.02)

    # ticks placed at round deficits, labeled in ESS/K units
    tick_d = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
    axa.xaxis.set_major_locator(FixedLocator(tick_d))
    axa.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{1.0 - v:g}")
    )
    axa.xaxis.set_minor_locator(FixedLocator([]))
    axa.yaxis.set_major_locator(MultipleLocator(0.25))

    axa.set_xlabel(
        r"per-star $\mathrm{ESS}/K$"
        r" (log-spaced in $1-\mathrm{ESS}/K$)"
    )
    axa.set_ylabel(r"fraction with $\mathrm{ESS}/K \geq x$")
    axa.set_title("(a) Importance-sampling proposal quality", loc="left")
    axa.legend(loc="upper left", fontsize=7.4, borderaxespad=0.55,
               handlelength=1.3, labelspacing=0.35)

    # compact numeric summary (layout re-verified for this axis)
    if med_essK:
        pc = r"\%" if use_tex else r"%"
        if "nfw" in med_essK and "burkert" in med_essK:
            line1 = (rf"med. $\mathrm{{ESS}}/K$: "
                     rf"${med_essK['nfw']:.2f}$/"
                     rf"${med_essK['burkert']:.2f}$")
        else:
            k = next(iter(med_essK))
            line1 = rf"med. $\mathrm{{ESS}}/K = {med_essK[k]:.2f}$"
        frac_all = float(np.mean(list(frac_K4.values())))
        if frac_all >= 0.0005:
            line2 = (rf"$\mathrm{{ESS}}<K/4$: "
                     rf"${100 * frac_all:.1f}\,${pc} of stars")
        else:
            line2 = rf"no stars with $\mathrm{{ESS}}<K/4$"
        axa.annotate(
            line1 + "\n" + line2,
            xy=(0.935, 0.045),
            xycoords="axes fraction",
            fontsize=6.6, color="0.30",
            ha="right", va="bottom",
            linespacing=1.5,
        )

    # ========================================================
    # (b) Radial deconvolution error
    # ========================================================
    required = {
        "r_true_kpc",
        "r_noisy_kpc",
        "r_posterior_mean_kpc",
        "R_half",
    }

    if required.issubset(d0.columns):
        noisy_runs_all = []
        posterior_runs = {"nfw": [], "burkert": []}

        for (halo, seed, rep), g in d0.groupby(
            ["halo", "seed", "repeat"]
        ):
            noisy_rmse, posterior_rmse = _per_run_radial_rmse(
                g, DECONV_EDGES_RHALF
            )
            noisy_runs_all.append(noisy_rmse)
            if halo in posterior_runs:
                posterior_runs[halo].append(posterior_rmse)

        noisy_q = _column_quantiles(np.asarray(noisy_runs_all))
        valid_noisy = np.isfinite(noisy_q[1])

        axb.fill_between(
            DECONV_CENTERS_RHALF,
            noisy_q[0], noisy_q[2],
            where=valid_noisy,
            color="0.55", alpha=0.13, lw=0,
            interpolate=True,
        )
        axb.plot(
            DECONV_CENTERS_RHALF,
            noisy_q[1],
            color="0.40", lw=1.5,
            ls=(0, (4, 2.2)),
            label="noisy radii (no model)",
            zorder=2,
        )

        post_med_at_Rh = []
        for halo in ("nfw", "burkert"):
            if len(posterior_runs[halo]) == 0:
                continue

            q = _column_quantiles(np.asarray(posterior_runs[halo]))
            valid = np.isfinite(q[1])
            color = HALOS[halo]["color"]

            axb.fill_between(
                DECONV_CENTERS_RHALF,
                q[0], q[2],
                where=valid,
                color=color, alpha=0.14, lw=0,
                interpolate=True,
            )
            axb.plot(
                DECONV_CENTERS_RHALF,
                q[1],
                color=color, lw=1.9,
                marker="o", ms=2.6, markeredgewidth=0,
                label=f"{HALOS[halo]['label']} posterior",
                zorder=4,
            )
            post_med_at_Rh.append(
                float(np.interp(
                    1.0,
                    DECONV_CENTERS_RHALF[valid],
                    q[1][valid],
                ))
            )

        # improvement-factor annotation at R_half
        if post_med_at_Rh and np.any(valid_noisy):
            y_noisy = float(np.interp(
                1.0,
                DECONV_CENTERS_RHALF[valid_noisy],
                noisy_q[1][valid_noisy],
            ))
            y_post = float(np.mean(post_med_at_Rh))
            fac = y_noisy / y_post

            axb.annotate(
                "",
                xy=(1.0, y_post * 1.13),
                xytext=(1.0, y_noisy / 1.13),
                arrowprops=dict(
                    arrowstyle="-|>", lw=0.9,
                    color="0.25", shrinkA=0, shrinkB=0,
                ),
                zorder=6,
            )
            axb.annotate(
                rf"$\times{fac:.1f}$",
                xy=(1.0, np.sqrt(y_noisy * y_post)),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=7.6, color="0.15",
                ha="left", va="center",
            )

        axb.set_xscale("log")
        axb.set_yscale("log")
        axb.set_xlim(
            DECONV_EDGES_RHALF[0],
            DECONV_EDGES_RHALF[-1],
        )
        axb.yaxis.set_major_locator(
            FixedLocator([0.1, 0.2, 0.5, 1.0, 2.0])
        )
        axb.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:g}")
        )
        axb.yaxis.set_minor_formatter(NullFormatter())
        axb.set_ylim(0.09, 2.0)

        axb.axvline(1.0, color="0.72", lw=0.75, ls=":", zorder=0)
        axb.annotate(
            r"$R_{1/2}$",
            xy=(1.0, 0.015),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7.0, color="0.48",
            ha="left", va="bottom",
        )

        axb.set_xlabel(r"true radius, $r/R_{1/2}$")
        axb.set_ylabel(
            r"RMS $\log_{10}(r_{\rm est}/r_{\rm true})$ [dex]"
        )
        axb.set_title("(b) Per-star radius reconstruction", loc="left")
        axb.legend(
            loc="upper right",
            fontsize=6.9,
            handlelength=2.0,
            borderaxespad=0.5,
            labelspacing=0.35,
        )

    else:
        axb.text(
            0.5, 0.5,
            "Truth radii unavailable",
            ha="center", va="center",
            transform=axb.transAxes,
        )
        axb.set_axis_off()

    fig.align_ylabels([axa, axb])

    save_pub(fig, outdir, "observed_posterior_diagnostics")
    plt.close(fig)

    print(
        f"[plot] wrote "
        f"{outdir}/observed_posterior_diagnostics.pdf/.png"
    )

# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--seed0", type=int, default=101)
    p.add_argument("--base-dir", type=str, default="calibration")
    p.add_argument("--fitter", type=str, default=str(REPO_ROOT / "fitters/six_coordinate.py"))
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--halos", type=str, nargs="*",
                   default=["nfw", "burkert"], choices=list(HALOS))
    p.add_argument("--fit-repeats", type=int, default=1,
                   help=">1: refit the SAME mock with shifted fitter seeds "
                        "to measure pure optimizer jitter.")
    p.add_argument("--fit-extra", type=str, default="",
                   help="Extra CLI forwarded verbatim to the fitter "
                        "(e.g. '--beta-prior-weight 30').")
    p.add_argument("--force", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--use-tex", action="store_true")
    p.add_argument("--x-probe", type=float, default=0.5,
                   help="Probe radius in R_half units for the slope statistic.")
    p.add_argument("--s-cusp", type=float, default=-1.0,
                   help="Cusp-hypothesis slope at the probe radius.")
    p.add_argument("--s-core", type=float, default=-0.3,
                   help="Core-hypothesis slope at the probe radius.")
    a = p.parse_args()

    seeds = a.seeds if a.seeds else list(range(a.seed0, a.seed0 + a.n_seeds))
    fit_extra = shlex.split(a.fit_extra)
    os.makedirs(a.base_dir, exist_ok=True)

    if not a.plot_only:
        t0 = time.time()
        total = len(a.halos) * len(seeds) * a.fit_repeats
        k = 0
        for s in seeds:
            for halo in a.halos:
                mock_dir, ok = ensure_mock(halo, s, a.base_dir, a.python, a.force)
                if not ok:
                    print(f"[skip] mock generation failed: {halo} seed {s}")
                    continue
                for rep in range(a.fit_repeats):
                    k += 1
                    print(f"\n=== [{k}/{total}] fit {halo} seed {s} rep {rep} "
                          f"({(time.time()-t0)/60:.1f} min elapsed) ===")
                    ensure_fit(halo, s, rep, mock_dir, a.base_dir,
                               a.fitter, a.python, a.force, fit_extra)

    df = harvest(a.base_dir, a.halos, seeds, a.fit_repeats)
    if df.empty:
        print("No completed runs found. Nothing to plot.")
        return
    df.to_csv(os.path.join(a.base_dir, "calibration_results.csv"), index=False)

    summary = make_summary(df)

    # ------------------------------------------------------------
    # Observed-mode ESS and deconvolution diagnostics
    # ------------------------------------------------------------
    posterior_diag = harvest_posterior_diagnostics(
        a.base_dir,
        a.halos,
        seeds,
        repeats=a.fit_repeats,
    )

    if posterior_diag.empty:
        print(
            "\n[diag] no per-star observed-mode diagnostics found. "
            "This is expected for truth-mode calibration runs."
        )
    else:
        posterior_diag.to_csv(
            os.path.join(
                a.base_dir,
                "posterior_star_diagnostics_all_runs.csv",
            ),
            index=False,
        )

        posterior_runs = posterior_run_summaries(
            posterior_diag
        )

        posterior_runs.to_csv(
            os.path.join(
                a.base_dir,
                "posterior_diagnostic_run_summary.csv",
            ),
            index=False,
        )

        posterior_summary = summarize_posterior_runs(
            posterior_runs
        )

        summary["observed_posterior_diagnostics"] = (
            posterior_summary
        )

        with open(
            os.path.join(
                a.base_dir,
                "posterior_diagnostic_summary.json",
            ),
            "w",
        ) as f:
            json.dump(
                posterior_summary,
                f,
                indent=2,
            )

        print("\n===== OBSERVED POSTERIOR DIAGNOSTICS =====")
        print(json.dumps(posterior_summary, indent=2))

        plot_observed_posterior_diagnostics(
            posterior_diag,
            a.base_dir,
            use_tex=a.use_tex,
        )

    with open(
        os.path.join(
            a.base_dir,
            "calibration_summary.json",
        ),
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n===== CALIBRATION SUMMARY =====")
    print(json.dumps(summary, indent=2))

    #plot_calibration(df, a.base_dir, use_tex=a.use_tex)

    # NEW: aggregated paper plots from the per-run scan tables
    curves = harvest_scan_curves(a.base_dir, a.halos, seeds)
    if curves.empty:
        print("[plot] no inner_slope_profile_scan.csv found -- skipping "
              "aggregated profile/family plots.")
    else:
        curves.to_csv(os.path.join(a.base_dir, "calibration_scan_curves.csv"),
                      index=False)
        outliers = discordant_seeds(df)
        plot_inner_slope_profiles(curves, a.base_dir, use_tex=a.use_tex,
                                  outliers=None)
        plot_family_bands(curves, a.base_dir, use_tex=a.use_tex,
                          outliers=None)

        # Improvement 1: statistic in the data-supported slope coordinate
        curves_p = add_slope_probe(curves, a.x_probe)
        stat = compute_slope_statistic(curves_p, a.s_cusp, a.s_core)
        stat.to_csv(os.path.join(a.base_dir, "slope_statistic.csv"), index=False)
        ssum = slope_statistic_summary(stat, df)
        ssum["protocol"] = {"x_probe": a.x_probe,
                            "s_cusp": a.s_cusp, "s_core": a.s_core}
        with open(os.path.join(a.base_dir, "slope_statistic_summary.json"),
                  "w") as f:
            json.dump(ssum, f, indent=2)
        print("\n===== SLOPE-COORDINATE STATISTIC =====")
        print(json.dumps(ssum, indent=2))
        plot_slope_statistic(curves_p, stat, df, a.s_cusp, a.s_core,
                             a.x_probe, a.base_dir, use_tex=a.use_tex)
        scan_slope_protocols(curves, df, a.base_dir)

        feats = per_seed_features(curves)
        feats.to_csv(os.path.join(a.base_dir, "statistic_candidates.csv"),
                     index=False)
        report = {c: class_sep(feats, c)
                  for c in ("dnll", "danchor", "dL", "ev_ratio",
                            "core_end_slope")}
        _, lda_err = loo_lda(feats, ["dnll", "danchor"])
        report["lda_dnll_danchor_loo_errors"] = int(lda_err)
        with open(os.path.join(a.base_dir, "statistic_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()