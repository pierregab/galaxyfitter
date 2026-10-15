"""
fitter_new.py

use for projected mode only !
"""

import os
import json
import math
import time
import warnings
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# PUBLICATION FIGURE STYLE
# ============================================================
def setup_pub_style(use_tex=False):
    """LaTeX-like serif figures. use_tex=True requires a LaTeX install."""
    plt.rcParams.update({
        # fonts
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "text.usetex": use_tex,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # lines & axes
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "axes.grid": False,
        "axes.axisbelow": True,

        # ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,

        # legend
        "legend.frameon": False,
        "legend.handlelength": 1.6,

        # output
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
    })
setup_pub_style(use_tex=False)

# Okabe-Ito colorblind-safe palette
C_TRUTH = "#000000"    # black
C_FIT = "#D55E00"      # vermilion
C_BAND = "#D55E00"
C_DATA = "#0072B2"     # blue
C_DATA2 = "#009E73"    # green
C_DATA3 = "#E69F00"    # orange
C_MODEL2 = "#56B4E9"   # sky blue
C_GREY = "0.45"
C_LIGHTGREY = "0.90"

def save_pub(fig, name):
    """Save both PDF and PNG in CFG.output_dir."""
    fig.savefig(os.path.join(CFG.output_dir, name + ".pdf"))
    fig.savefig(os.path.join(CFG.output_dir, name + ".png"), dpi=300)

import torch
import torch.nn as nn

from scipy.integrate import cumulative_trapezoid

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 0. CONFIGURATION
# ============================================================

import argparse

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input-dir", type=str, default="realistic_mock_data_NFW_equilibrium")
    p.add_argument("--output-dir", type=str, default="results_FEL_joint_fit")

    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--training-mode", type=str, default="observed",
                   choices=["truth", "observed", "projected"])
    p.add_argument("--potential-model", type=str, default="nfw",
                   choices=["nfw", "gnfw", "zhao", "corenfw", "free_mass"])
    
    p.add_argument("--diagnostics-at", type=str, default="map",
                   choices=["map", "profile"],
                   help="Render section-11 plots at the joint MAP or at the "
                        "profiled best scan node.")
    
    p.add_argument("--gamma-init", type=float, default=None,
                   help="Initial inner density slope for gnfw/zhao. "
                        "Omit to keep the model default (1).")

    p.add_argument("--n-iwae", type=int, default=32,
                   help="Importance samples per star per step.")
    p.add_argument("--batch-size", type=int, default=256)

    p.add_argument("--epochs-warmup", type=int, default=250,
                   help="Phase A: DF + encoder, potential frozen.")
    p.add_argument("--epochs-joint", type=int, default=700,
                   help="Phase B: joint DF + encoder + potential.")

    p.add_argument("--lr-df", type=float, default=1.0e-3)
    p.add_argument("--lr-pot", type=float, default=3.0e-3)

    p.add_argument("--sigma-los-weight", type=float, default=5.0,
                   help="Weight of the noise-corrected projected dispersion anchor.")
    p.add_argument("--pot-prior-weight", type=float, default=1.0)

    p.add_argument("--isotropic-df", action="store_true",
                   help="Restrict DF to f(E) (drop L dependence).")

    # truth values for diagnostics only (never used in the fit)
    p.add_argument("--m0-nfw-true", type=float, default=5.0e8)
    p.add_argument("--rs-nfw-true", type=float, default=2.0)
    p.add_argument("--a-star-true", type=float, default=0.1)

    # SBI
    p.add_argument("--run-sbi", action="store_true")
    p.add_argument("--sbi-n-sims", type=int, default=3000)
    p.add_argument("--sbi-epochs", type=int, default=300)
    p.add_argument("--sbi-n-posterior", type=int, default=5000)

    p.add_argument("--skip-laplace", action="store_true")

    # Optional projected PM and VSP anchors
    p.add_argument("--use-pm-anchor", action="store_true",
                   help="Add noise-corrected projected total proper-motion dispersion sigma_pm,tot(R).")

    p.add_argument("--use-vsp-anchor", action="store_true",
                   help="Add projected fourth-moment VSP-like constraints from observed v_los and R.")

    p.add_argument("--sigma-pm-weight", type=float, default=5.0,
                   help="Weight of the projected proper-motion dispersion anchor.")

    p.add_argument("--vsp-weight", type=float, default=1.0,
                   help="Weight of the VSP fourth-moment anchor.")

    # --- fourth-moment anchor repairs. Every default reproduces the previous
    # behaviour exactly, so earlier runs are unaffected unless a flag is given.
    p.add_argument("--n-core-min", type=float, default=0.0,
                   help="Lower bound on the coreNFW core strength n, hence on "
                        "gamma_0 = n - 1. The default 0.0 is the previous "
                        "behaviour and caps gamma_0 at -1 exactly, because "
                        "n = sigmoid(raw_n) lives in (0,1) and the scan clips "
                        "fix_n into it. That bound is a property of the "
                        "parametrisation, not of the data, and the Gaia "
                        "Challenge cusp sits just behind it: at the Rs the fit "
                        "already reaches, the secant of ln rho over 0.5-2 "
                        "R_half is -1.119 at n=0 against a truth of -1.195, and "
                        "-1.189 at n=-0.10. Every railed cusp scan in this "
                        "repository is pressed against that wall. The density "
                        "stays positive down to about n = -0.8 at those "
                        "parameters; check it for a new configuration before "
                        "going far below zero.")
    p.add_argument("--scan-restarts", type=int, default=0,
                   help="Optimise every scan node INDEPENDENTLY from the joint "
                        "MAP state, this many times, instead of chaining each "
                        "node's warm start to its neighbour. 0 keeps the "
                        "chained behaviour and is the default. Chaining makes a "
                        "node's optimisation quality depend on how far along "
                        "the path it sits, which biases the SHAPE of the "
                        "profile rather than adding symmetric noise: on the "
                        "paired cusp, --scan-multistart improved the core half "
                        "of the curve by 18 NLL units and the cusp half by 4, "
                        "tilting the profile by 14 and moving the argmin from "
                        "n=0.12 to n=0.38. Starting every node from the same "
                        "state makes the quality uniform by construction. Each "
                        "node is then individually less well optimised than a "
                        "warm-started one, which is the price. At 1 restart it "
                        "costs the same as the default; at k it costs k times "
                        "the scan.")
    p.add_argument("--scan-multistart", action="store_true",
                   help="Sweep the scan grid from several starting points and "
                        "keep the best optimum found at each node, instead of "
                        "the two half-sweeps out from the MAP that are the "
                        "default. Each node is re-optimised warm-started from "
                        "its neighbour, so the path decides which local optimum "
                        "it lands in, and the path depends on the grid: "
                        "extending a grid upward changed the curve over an "
                        "untouched region and reached an absolute NLL 5 units "
                        "lower (controls/upper_bound_test). Adding an ascending "
                        "pass from the lowest node and a descending pass from "
                        "the highest gives each node three independent chances; "
                        "the accumulator already keeps the best. Costs about "
                        "three times the scan, so roughly twice the run.")
    p.add_argument("--n-core-max", type=float, default=1.0,
                   help="Upper bound on n, hence on gamma_0 = n - 1. The "
                        "default 1.0 is the previous behaviour and caps "
                        "gamma_0 at 0, a perfectly flat core. Unlike the lower "
                        "bound, this one is physical: gamma_0 > 0 is a central "
                        "density hole. Two Burkert scans do press against it, "
                        "but their truth lies IN FRONT of it -- at n=1 the "
                        "secant is already only +0.055 and +0.066 from the "
                        "truth, and opening the bound makes it monotonically "
                        "worse (+0.132 at n=1.1, +0.834 at n=2). Provided so "
                        "the asymmetry can be tested rather than asserted.")
    p.add_argument("--scan-rs", type=str, default="free",
                   choices=["free", "frozen", "auto"],
                   help="What R_s does while the inner-slope scan moves n. "
                        "'free' is the previous behaviour and the default. "
                        "'frozen' holds log_Rs at the joint fit's own value for "
                        "the whole sweep -- exactly, by excluding it from the "
                        "optimiser, not by a stiff prior. A free R_s slides "
                        "along its degeneracy with n and absorbs the very "
                        "change the scan is trying to measure, which flattens "
                        "the curve; on the Gaia Challenge cusp that cost 0.27 "
                        "in the recovered slope. 'auto' runs both sweeps from "
                        "the same MAP state and keeps the better resolved one "
                        "(smaller span; on a tie, the deeper basin), writing "
                        "both to metrics.json so the choice is auditable. auto "
                        "costs one extra sweep, not one extra fit.")
    p.add_argument("--rs-init", type=float, default=0.0,
                   help="Override the scale-radius initialisation, which is "
                        "otherwise the heuristic Rs_init = clip(12*R_half, "
                        "0.3, 10). That heuristic ties the HALO scale to the "
                        "TRACER scale, which is a diagnostic-only assumption: "
                        "the Gaia Challenge cusp and core mocks share r0 = 1 "
                        "kpc while their R_half differ by 2.4x, and the fitted "
                        "Rs come out in exactly that ratio. Rs_init is also "
                        "the centre of the --rs-prior-weight penalty, so this "
                        "moves both. 0 keeps the heuristic.")
    p.add_argument("--anchor-perspective", action="store_true",
                   help="Build the sigma_los and VSP targets from velocities "
                        "with the systemic removed PER STAR, the same way the "
                        "projected likelihood already does it "
                        "(_build_projected_data: 'Perspective rotation removed "
                        "EXACTLY'). Without this the anchors subtract a single "
                        "scalar v_sys, so the field's perspective gradient "
                        "survives in the anchor targets while the likelihood "
                        "they are compared against has had it removed. On a "
                        "5-degree field at 80 kpc that residual reaches 6.9 "
                        "km/s against a 9 km/s dispersion, and it is largest "
                        "at the largest R, which is where the VSP weight sits.")
    p.add_argument("--vsp-r-quantile", type=float, default=1.0,
                   help="Truncate both VSP sums at this quantile of R "
                        "(1.0 = no truncation, the previous behaviour). The "
                        "model side is truncated identically. Beyond the "
                        "sampled tracer body the model's <v_los^4>(R) is pure "
                        "extrapolation and the data side is a handful of "
                        "stars, so the comparison there is between an "
                        "expectation and a single draw.")
    p.add_argument("--vsp-star-weights", action="store_true",
                   help="Use mean(v^4) and mean(v^4 R^2) instead of "
                        "mean(v^4 R) and mean(v^4 R^3). Stars sample "
                        "dN = 2 pi R Sigma(R) dR, so the surface-weighted "
                        "virial shape integrals int Sigma <v^4> R dR and "
                        "int Sigma <v^4> R^3 dR are estimated by the star "
                        "averages of v^4 and v^4 R^2. The default carries one "
                        "extra power of R in each: self-consistent between "
                        "model and data, but a higher moment than the virial "
                        "shape parameter, with the variance pushed further "
                        "into the outskirts.")
    p.add_argument("--vsp-max-concentration", type=float, default=1.0,
                   help="Disable the VSP anchor when a single star carries "
                        "more than this fraction of sum(R^p) (1.0 = never, "
                        "the previous behaviour). Computed before any fit. A "
                        "sum dominated by one star is not a measurement of a "
                        "fourth moment: its bootstrap error is meaningless "
                        "because resampling cannot produce an outer star that "
                        "was never observed.")
    p.add_argument("--vsp-n-boot", type=int, default=300,
                   help="Bootstrap resamples for VSP uncertainty estimation.")

    p.add_argument("--pm-use-systemic-from-data", action="store_true",
                   help="Estimate systemic PM from the observed catalogue instead of CFG.PM_SYS_RA/DEC.")

    p.add_argument("--pm-anchor-distance", type=str, default="center",
                   choices=["center", "observed"],
                   help=(
                       "Distance used to convert PM to km/s for the projected PM anchor. "
                       "'center' uses CFG.DIST_CENTER, recommended for projected dwarf tests. "
                       "'observed' uses each star's noisy observed distance."
                   ))
    
    p.add_argument("--scan-inner-slope", action="store_true",
                    help="Profile-likelihood scan over n (inner slope = n-1).")
    p.add_argument("--scan-grid", type=str, default="0.0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0")
    p.add_argument("--scan-epochs", type=int, default=250,
                   help="Fixed re-optimization budget per scan node.")
    p.add_argument("--fix-n", type=float, default=None)
    p.add_argument("--n-core-init", type=float, default=0.5)
    p.add_argument("--rc-lo-mult", type=float, default=0.2)
    p.add_argument("--rc-hi-mult", type=float, default=3.0,
                   help="Upper edge of the r_c box in R_half units. "
                        "Burkert r0 = 3.0 R_half sits ON the default edge -> use 5.0.")
    p.add_argument("--rc-center-mult", type=float, default=1.25,
                   help="Center of the log-normal r_c prior in R_half units "
                        "(use 1.75 with the wider box).")
    p.add_argument("--marginalize-distance", action="store_true",
               help="Exact 1D quadrature over the latent distance instead of "
                    "MC sampling (Rao-Blackwellized likelihood).")
    p.add_argument("--n-d-quad", type=int, default=96)
    p.add_argument("--d-quad-span", type=float, default=12.0,
               help="Half-width [kpc] of the distance grid around DIST_CENTER.")
    p.add_argument("--d-quad-min", type=float, default=1e-3)
    p.add_argument("--proj-n-r", type=int, default=64,
                   help="Radial nodes of the LOS-velocity grid "
                        "(strided subset of the main r-grid).")
    p.add_argument("--proj-n-q", type=int, default=10,
                   help="Nodes in q = sin(theta_LOS) = R/r.")
    p.add_argument("--proj-n-w", type=int, default=96,
                   help="Symmetric v_los grid nodes (covers +-v_esc).")
    p.add_argument("--proj-n-phi", type=int, default=6,
                   help="Gauss-Legendre nodes for the tangential azimuth.")
    p.add_argument("--proj-n-s", type=int, default=72,
                   help="LOS depth quadrature nodes per star (half line).")
    p.add_argument("--proj-s-max", type=float, default=None,
                   help="LOS half-extent [kpc]; default = r_grid_max.")
    p.add_argument("--proj-n-gh", type=int, default=12,
                   help="Gauss-Hermite nodes for the per-star noise "
                        "convolution.")
    p.add_argument("--proj-base-sigma", type=float, default=0.15,
                   help="Baseline smoothing e0 [km/s] baked into the "
                        "LOS-velocity grid. Auto-capped at "
                        "0.9*min(e_v_los).")
    p.add_argument("--proj-r-chunk", type=int, default=4,
                   help="r-chunk size of the kernel contraction "
                        "(memory knob).")
    p.add_argument("--proj-kernel", type=str, default="window",
                   choices=["window", "dense"],
                   help="LOS-velocity kernel evaluation. 'window' is "
                        "~20x cheaper and numerically identical.")
    p.add_argument("--m-piv-init-scale", type=float, default=1.0,
                   help="Multiply the Wolf-estimator initialization of "
                        "M(<r_piv). Diagnostic for init dependence of "
                        "the mass basin.")
    p.add_argument("--strong-prior-norm", type=str, default="legacy",
                   choices=["legacy", "nll_units"],
                   help="'legacy' reproduces the published truth/observed "
                        "runs (strong prior effectively x batch_size). "
                        "'nll_units' makes --strong-prior-weight mean "
                        "literal NLL units, as the paper text describes.")
    p.add_argument("--strong-prior-weight", type=float, default=1.0,
                   help="Weight of the strong shape prior. Under "
                        "'nll_units' this is in NLL units; under 'legacy' "
                        "it is multiplied by batch_size.")
    p.add_argument("--m-piv-prior-sigma", type=float, default=0.0,
                   help="Gaussian prior width [dex] on log10 M(<r_piv) "
                        "around the noise-corrected Wolf estimate. "
                        "0 disables. Projected-mode suggestion: 0.15.")
    p.add_argument("--rc-prior-weight", type=float, default=5.0,
                   help="Weight [NLL units, under --strong-prior-norm "
                        "nll_units] of the log-normal r_c prior. Small: "
                        "let the data set the core size. 5.0 makes the "
                        "Burkert truth r_0 = 3 R_half cost ~2.9 units.")
    p.add_argument("--rs-prior-weight", type=float, default=150.0,
                   help="Weight of the R_s prior. Large: R_s is the "
                        "gauge-soft direction (Appendix A) and is not "
                        "resolved by the tracers. 150 reproduces the "
                        "effective legacy strength (0.3 x batch_size).")
    p.add_argument("--anchor-loss", type=str, default="robust",
                   choices=["robust", "huber", "quad"],
                   help="Anchor residual loss. 'robust'=ln(1+z^2) "
                        "(legacy, saturates), 'huber'=quadratic to "
                        "|z|=3 then linear, 'quad'=z^2/2.")
    p.add_argument("--beta-tether-weight", type=float, default=0.0,
                   help="Weight [NLL units] of the penalty tying the "
                        "DF-implied beta(r) to a jointly-fitted COM "
                        "curve. 0 disables. Suggested: 50.")
    p.add_argument("--beta-tether-band", type=float, nargs=2,
                   default=[0.25, 4.0],
                   help="Radial band of the tether in units of R_half.")
    p.add_argument("--stat-anchor-weights", type=float, nargs=2,
                   default=None,
                   help="(w_sig, w_vsp) used in the SCAN STATISTIC "
                        "(matched-moments objective). Defaults to the "
                        "training anchor weights.")

    return p.parse_args()

@dataclass
class Config:
    input_dir: str = "realistic_mock_data_NFW_equilibrium"
    output_dir: str = "results_FEL_joint_fit"

    training_mode: str = "observed"
    potential_model: str = "nfw"
    gamma_init: float = 1          # inner density slope init for gnfw/zhao

    seed: int = 12345
    dtype: str = "float32"

    # physics constants (match the generator)
    G: float = 4.3009e-6
    M_0_NFW_TRUE: float = 5.0e8
    R_S_NFW_TRUE: float = 2.0
    A_STAR_TRUE: float = 0.1
    M_STAR_TOT: float = 1.0e6

    RA_CENTER: float = 260.0
    DEC_CENTER: float = 57.0
    DIST_CENTER: float = 80.0
    V_SYS_LOS: float = -290.0
    PM_SYS_RA: float = 0.04
    PM_SYS_DEC: float = -0.15
    KAPPA: float = 4.74047

    # DF model
    df_hidden: int = 64
    df_isotropic: bool = False
    unbound_kappa: float = 30.0      # softness of the eps<0 suppression

    strong_prior_weight: float = 1.0

    # encoder
    enc_hidden: int = 128

    # IWAE
    n_iwae: int = 32
    batch_size: int = 256

    # schedule
    epochs_warmup: int = 250
    epochs_joint: int = 700

    lr_df: float = 1.0e-3
    lr_pot: float = 3.0e-3
    grad_clip_df: float = 10.0
    grad_clip_pot: float = 5.0

    # losses
    sigma_los_weight: float = 5.0
    pot_prior_weight: float = 1.0
    pot_prior_sigma: float = 2.0      # very weak log-normal prior around init

    # phase-space quadrature grids
    n_r_grid: int = 140
    r_grid_min: float = 1.0e-3
    r_grid_max: float = 30.0          # DF normalization domain (model definition)
    n_v_grid: int = 36
    n_eta_grid: int = 12
    n_s_proj: int = 110               # LOS projection grid

    # empirical sigma_los targets
    n_sig_bins: int = 7
    sig_var_floor: float = 0.25       # km^2/s^2 floor on corrected variance

    # free spherical mass potential
    free_mass_n_knots: int = 16
    free_mass_r_min: float = 1.0e-4
    free_mass_r_max: float = 300.0

    # Weak regularization only: discourages numerical wiggles.
    # It does not force an NFW shape.
    free_mass_smooth_weight: float = 2.0e-2

    # Couples the data-free inner knots to the innermost constrained slope.
    # 0 = fully local (inner region floats on init); large = effectively
    # parametric inside. This is the knob from "fully local" to "NFW-like reach".
    free_mass_inner_anchor_weight: float = 1.0e-0

    # plotting
    r_plot_min: float = 1.0e-3
    r_plot_max: float = 50.0
    n_r_plot: int = 400

    # SBI
    run_sbi: bool = False
    sbi_n_sims: int = 3000
    sbi_epochs: int = 300
    sbi_n_posterior: int = 5000
    sbi_n_summary_bins: int = 6

    skip_laplace: bool = False

    # optional projected PM / VSP anchors
    use_pm_anchor: bool = False
    use_vsp_anchor: bool = False
    sigma_pm_weight: float = 5.0
    vsp_weight: float = 1.0
    vsp_n_boot: int = 300
    n_core_min: float = 0.0
    n_core_max: float = 1.0
    scan_multistart: bool = False
    scan_restarts: int = 0
    scan_rs: str = "free"
    rs_init_override: float = 0.0
    anchor_perspective: bool = False
    vsp_r_quantile: float = 1.0
    vsp_star_weights: bool = False
    vsp_max_concentration: float = 1.0
    pm_use_systemic_from_data: bool = False
    pm_anchor_distance: str = "center"

    rc_lo_mult: float = 0.2
    rc_hi_mult: float = 3.0
    rc_center_mult: float = 1.25

    marginalize_distance: bool = False
    n_d_quad: int = 24
    d_quad_span: float = 6.0
    d_quad_min: float = 1e-3

    proj_n_r: int = 64
    proj_n_q: int = 10
    proj_n_w: int = 96
    proj_n_phi: int = 6
    proj_n_s: int = 72
    proj_s_max: float = 0.0          # 0.0 -> use r_grid_max
    proj_n_gh: int = 12
    proj_base_sigma: float = 0.15
    proj_r_chunk: int = 4
    proj_kernel: str = "window"

    m_piv_init_scale: float = 1.0

    strong_prior_norm: str = "legacy"
    m_piv_prior_sigma: float = 0.0
    rc_prior_weight: float = 5.0
    rs_prior_weight: float = 150.0

    anchor_loss: str = "robust"

    beta_tether_weight: float = 0.0
    beta_tether_band: tuple[float, float] = (0.25, 4.0)

    # None -> use (sigma_los_weight, vsp_weight)
    stat_anchor_weights: tuple[float, float] | None = None

def apply_args(cfg: Config, a) -> Config:
    cfg.input_dir = a.input_dir
    cfg.output_dir = a.output_dir
    cfg.seed = a.seed
    cfg.training_mode = a.training_mode
    cfg.potential_model = a.potential_model
    if a.gamma_init is not None:
        cfg.gamma_init = a.gamma_init
    cfg.n_iwae = a.n_iwae
    cfg.batch_size = a.batch_size
    cfg.epochs_warmup = a.epochs_warmup
    cfg.epochs_joint = a.epochs_joint
    cfg.lr_df = a.lr_df
    cfg.lr_pot = a.lr_pot
    cfg.sigma_los_weight = a.sigma_los_weight
    cfg.pot_prior_weight = a.pot_prior_weight
    cfg.df_isotropic = a.isotropic_df
    cfg.M_0_NFW_TRUE = a.m0_nfw_true
    cfg.R_S_NFW_TRUE = a.rs_nfw_true
    cfg.A_STAR_TRUE = a.a_star_true
    cfg.run_sbi = a.run_sbi
    cfg.sbi_n_sims = a.sbi_n_sims
    cfg.sbi_epochs = a.sbi_epochs
    cfg.sbi_n_posterior = a.sbi_n_posterior
    cfg.skip_laplace = a.skip_laplace
    cfg.use_pm_anchor = a.use_pm_anchor
    cfg.use_vsp_anchor = a.use_vsp_anchor
    cfg.sigma_pm_weight = a.sigma_pm_weight
    cfg.vsp_weight = a.vsp_weight
    cfg.vsp_n_boot = a.vsp_n_boot
    cfg.n_core_min = float(a.n_core_min)
    cfg.n_core_max = float(a.n_core_max)
    cfg.scan_multistart = bool(a.scan_multistart)
    cfg.scan_restarts = int(a.scan_restarts)
    cfg.scan_rs = str(a.scan_rs)
    cfg.rs_init_override = float(a.rs_init)
    cfg.anchor_perspective = bool(a.anchor_perspective)
    cfg.vsp_r_quantile = float(a.vsp_r_quantile)
    cfg.vsp_star_weights = bool(a.vsp_star_weights)
    cfg.vsp_max_concentration = float(a.vsp_max_concentration)
    cfg.pm_use_systemic_from_data = a.pm_use_systemic_from_data
    cfg.pm_anchor_distance = a.pm_anchor_distance
    cfg.rc_lo_mult = a.rc_lo_mult
    cfg.rc_hi_mult = a.rc_hi_mult
    cfg.rc_center_mult = a.rc_center_mult
    cfg.marginalize_distance = bool(a.marginalize_distance)
    cfg.n_d_quad = int(a.n_d_quad)
    cfg.d_quad_span = float(a.d_quad_span)
    cfg.d_quad_min = float(a.d_quad_min)
    cfg.proj_n_r = int(a.proj_n_r)
    cfg.proj_n_q = int(a.proj_n_q)
    cfg.proj_n_w = int(a.proj_n_w)
    cfg.proj_n_phi = int(a.proj_n_phi)
    cfg.proj_n_s = int(a.proj_n_s)
    cfg.proj_s_max = float(a.proj_s_max) if a.proj_s_max else 0.0
    cfg.proj_n_gh = int(a.proj_n_gh)
    cfg.proj_base_sigma = float(a.proj_base_sigma)
    cfg.proj_r_chunk = int(a.proj_r_chunk)
    cfg.m_piv_init_scale = float(a.m_piv_init_scale)
 
    if cfg.training_mode == "projected":
        # Proper motions and per-star distances are NOT data in this
        # mode; disable machinery that consumes them BEFORE it is built.
        if cfg.use_pm_anchor:
            print("[projected] disabling --use-pm-anchor "
                  "(proper motions are not used as data).")
            cfg.use_pm_anchor = False
        if cfg.marginalize_distance:
            print("[projected] disabling --marginalize-distance "
                  "(no latent 6D vector in this mode).")
            cfg.marginalize_distance = False
    cfg.proj_kernel = a.proj_kernel
    cfg.strong_prior_norm = a.strong_prior_norm
    cfg.strong_prior_weight = float(a.strong_prior_weight)
    cfg.m_piv_prior_sigma = float(a.m_piv_prior_sigma)
    cfg.rc_prior_weight = float(a.rc_prior_weight)
    cfg.rs_prior_weight = float(a.rs_prior_weight)

    cfg.anchor_loss = a.anchor_loss

    cfg.beta_tether_weight = float(a.beta_tether_weight)
    cfg.beta_tether_band = (
        float(a.beta_tether_band[0]),
        float(a.beta_tether_band[1]),
    )

    cfg.stat_anchor_weights = (
        None
        if a.stat_anchor_weights is None
        else (
            float(a.stat_anchor_weights[0]),
            float(a.stat_anchor_weights[1]),
        )
    )
    return cfg
    

ARGS = parse_args()
CFG = apply_args(Config(), ARGS)
os.makedirs(CFG.output_dir, exist_ok=True)

# ============================================================
# 1. SEEDS, DEVICE
# ============================================================
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed_all(CFG.seed)
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

TD = torch.float32 if CFG.dtype == "float32" else torch.float64
print(f"--- Device: {DEVICE} | dtype={TD} ---")

with open(os.path.join(CFG.output_dir, "fit_config.json"), "w") as f:
    json.dump(asdict(CFG), f, indent=2)

DEG2RAD = math.pi / 180.0
LOG2PI = math.log(2.0 * math.pi)

# ============================================================
# 2. TRUTH FUNCTIONS (numpy, diagnostics only)
# ============================================================

# Optional model-agnostic truth profile, for example for Burkert mocks.
_TRUTH_TBL = None
_TRUTH_LABEL = "NFW"

_tp = os.path.join(CFG.input_dir, "truth_profile.csv")

def build_d_grid(n_nodes=96, s_min=1.0e-3, s_max=12.0):
    """Mirrored log grid in LOS offset s = D - DIST_CENTER, with an explicit
    s=0 node and trapezoid weights.

    MPS fix:
    torch.logspace(..., device="mps") is not implemented, so build on CPU
    then move to DEVICE.
    """
    global D_NODES, D_LOGW

    n_half = n_nodes // 2

    # Build on CPU first, then move to DEVICE.
    s_half_cpu = torch.logspace(
        math.log10(s_min),
        math.log10(s_max),
        n_half,
        dtype=TD,
        device="cpu",
    )

    s_cpu = torch.cat([
        -s_half_cpu.flip(0),
        torch.zeros(1, dtype=TD, device="cpu"),
        s_half_cpu,
    ])

    D_NODES = (CFG.DIST_CENTER + s_cpu).to(DEVICE)

    dx = D_NODES[1:] - D_NODES[:-1]
    w = torch.zeros_like(D_NODES)
    w[:-1] = w[:-1] + 0.5 * dx
    w[1:] = w[1:] + 0.5 * dx

    D_LOGW = torch.log(w.clamp_min(1.0e-30))

if CFG.marginalize_distance:
    build_d_grid(CFG.n_d_quad, CFG.d_quad_min, CFG.d_quad_span)
    IDX5 = torch.tensor([0, 1, 3, 4, 5], device=DEVICE)

if os.path.exists(_tp):
    _t = pd.read_csv(_tp)

    required_cols = {"r", "mass", "rho"}
    missing_cols = required_cols.difference(_t.columns)
    if missing_cols:
        raise ValueError(f"{_tp} is missing required columns: {sorted(missing_cols)}")

    _t = _t.sort_values("r")
    valid = (
        np.isfinite(_t["r"].values) & np.isfinite(_t["mass"].values)
        & np.isfinite(_t["rho"].values) & (_t["r"].values > 0.0)
        & (_t["mass"].values > 0.0) & (_t["rho"].values > 0.0)
    )
    _t = _t.loc[valid]
    if len(_t) < 3:
        raise ValueError(f"{_tp} must contain at least three valid positive rows.")

    _TRUTH_TBL = {
        "lr": np.log(_t["r"].values),
        "lm": np.log(_t["mass"].values),
        "lrho": np.log(_t["rho"].values),
    }
    # NEW: analytic 3D log-density slope, if the generator wrote it.
    if "dlnrho_dlnr" in _t.columns:
        _TRUTH_TBL["dslope"] = _t["dlnrho_dlnr"].values

    _TRUTH_LABEL = "tabulated truth"
    print(f"[truth] using model-agnostic profile: {_tp}"
          + (" (+analytic slope)" if "dslope" in _TRUTH_TBL else ""))

def np_mass_nfw(r, M0=None, Rs=None):
    M0 = CFG.M_0_NFW_TRUE if M0 is None else M0
    Rs = CFG.R_S_NFW_TRUE if Rs is None else Rs
    x = np.asarray(r) / Rs
    return M0 * (np.log1p(x) - x / (1.0 + x))

def np_rho_nfw(r, M0=None, Rs=None):
    M0 = CFG.M_0_NFW_TRUE if M0 is None else M0
    Rs = CFG.R_S_NFW_TRUE if Rs is None else Rs
    x = np.maximum(np.asarray(r) / Rs, 1.0e-30)
    return (M0 / (4.0 * np.pi * Rs**3)) / (x * (1.0 + x) ** 2)

def np_mass_truth(r):
    """
    Enclosed-mass truth used only for diagnostics.

    Uses truth_profile.csv when available. Otherwise falls back to the
    analytic NFW truth configured through M_0_NFW_TRUE and R_S_NFW_TRUE.
    """
    r = np.asarray(r)

    if _TRUTH_TBL is None:
        return np_mass_nfw(r)

    lr = np.log(np.maximum(r, 1.0e-30))

    # Log-log interpolation. np.interp uses the boundary values outside
    # the tabulated interval, avoiding uncontrolled extrapolation.
    return np.exp(
        np.interp(
            lr,
            _TRUTH_TBL["lr"],
            _TRUTH_TBL["lm"],
        )
    )

def np_rho_truth(r):
    """
    Density truth used only for diagnostics.

    Uses truth_profile.csv when available. Otherwise falls back to NFW.
    """
    r = np.asarray(r)

    if _TRUTH_TBL is None:
        return np_rho_nfw(r)

    lr = np.log(np.maximum(r, 1.0e-30))

    return np.exp(
        np.interp(
            lr,
            _TRUTH_TBL["lr"],
            _TRUTH_TBL["lrho"],
        )
    )

def np_force_truth(r):
    """
    Magnitude of the true spherical acceleration: G M(<r) / r^2.
    """
    r = np.asarray(r)
    return CFG.G * np_mass_truth(r) / np.maximum(r * r, 1.0e-30)

def np_dlnrho_dlnr_truth(r):
    """True 3D log-density slope d ln rho / d ln r for diagnostics.
    Uses the analytic column from truth_profile.csv when present; else
    finite-differences the tabulated density; else falls back to NFW."""
    r = np.asarray(r, dtype=float)
    lr = np.log(np.maximum(r, 1.0e-30))
    if _TRUTH_TBL is not None and "dslope" in _TRUTH_TBL:
        return np.interp(lr, _TRUTH_TBL["lr"], _TRUTH_TBL["dslope"])
    if _TRUTH_TBL is not None:
        return np.interp(lr, _TRUTH_TBL["lr"],
                         np.gradient(_TRUTH_TBL["lrho"], _TRUTH_TBL["lr"]))
    x = np.maximum(r / CFG.R_S_NFW_TRUE, 1.0e-30)
    return -1.0 - 2.0 * x / (1.0 + x)

def np_rho_plummer(r):
    r = np.asarray(r)
    a = CFG.A_STAR_TRUE
    return (
        3.0
        * CFG.M_STAR_TOT
        / (4.0 * np.pi * a**3)
        * (1.0 + (r / a) ** 2) ** (-2.5)
    )

# ============================================================
# 3. COORDINATE TRANSFORMS
# ============================================================
def _center_consts():
    ra_c = math.radians(CFG.RA_CENTER)
    dec_c = math.radians(CFG.DEC_CENTER)

    X0 = CFG.DIST_CENTER * math.cos(dec_c) * math.cos(ra_c)
    Y0 = CFG.DIST_CENTER * math.cos(dec_c) * math.sin(ra_c)
    Z0 = CFG.DIST_CENTER * math.sin(dec_c)

    vpra = CFG.PM_SYS_RA * CFG.KAPPA * CFG.DIST_CENTER
    vpdec = CFG.PM_SYS_DEC * CFG.KAPPA * CFG.DIST_CENTER

    VX0 = (CFG.V_SYS_LOS * math.cos(dec_c) * math.cos(ra_c)
           - vpra * math.sin(ra_c)
           - vpdec * math.sin(dec_c) * math.cos(ra_c))
    VY0 = (CFG.V_SYS_LOS * math.cos(dec_c) * math.sin(ra_c)
           + vpra * math.cos(ra_c)
           - vpdec * math.sin(dec_c) * math.sin(ra_c))
    VZ0 = CFG.V_SYS_LOS * math.sin(dec_c) + vpdec * math.cos(dec_c)

    return (X0, Y0, Z0), (VX0, VY0, VZ0)

CENTER_POS_NP, CENTER_VEL_NP = _center_consts()
CENTER_POS_T = torch.tensor(CENTER_POS_NP, dtype=TD, device=DEVICE)
CENTER_VEL_T = torch.tensor(CENTER_VEL_NP, dtype=TD, device=DEVICE)

def obs_to_galactocentric_torch(obs):
    """obs columns: RA[deg], Dec[deg], Dist[kpc], pmRA[mas/yr], pmDec[mas/yr], v_los[km/s]."""
    ra = torch.deg2rad(obs[:, 0])
    dec = torch.deg2rad(obs[:, 1])
    d = obs[:, 2]
    pm_ra = obs[:, 3]
    pm_dec = obs[:, 4]
    v_los = obs[:, 5]

    cra, sra = torch.cos(ra), torch.sin(ra)
    cdec, sdec = torch.cos(dec), torch.sin(dec)

    X = d * cdec * cra
    Y = d * cdec * sra
    Z = d * sdec

    pos = torch.stack([X, Y, Z], dim=1) - CENTER_POS_T[None, :]

    r_hat = torch.stack([cdec * cra, cdec * sra, sdec], dim=1)
    ra_hat = torch.stack([-sra, cra, torch.zeros_like(ra)], dim=1)
    dec_hat = torch.stack([-sdec * cra, -sdec * sra, cdec], dim=1)

    v_ra = pm_ra * CFG.KAPPA * d
    v_dec = pm_dec * CFG.KAPPA * d

    vel_abs = (v_los[:, None] * r_hat
               + v_ra[:, None] * ra_hat
               + v_dec[:, None] * dec_hat)
    vel = vel_abs - CENTER_VEL_T[None, :]
    return pos, vel

def cartesian_to_observables_np(pos_abs, vel_abs):
    X, Y, Z = pos_abs[:, 0], pos_abs[:, 1], pos_abs[:, 2]
    D = np.sqrt(X * X + Y * Y + Z * Z)
    RA = np.arctan2(Y, X)
    Dec = np.arcsin(Z / D)

    r_hat = np.column_stack([np.cos(Dec) * np.cos(RA),
                             np.cos(Dec) * np.sin(RA),
                             np.sin(Dec)])
    ra_hat = np.column_stack([-np.sin(RA), np.cos(RA), np.zeros_like(RA)])
    dec_hat = np.column_stack([-np.sin(Dec) * np.cos(RA),
                               -np.sin(Dec) * np.sin(RA),
                               np.cos(Dec)])

    v_los = np.sum(vel_abs * r_hat, axis=1)
    pm_ra = np.sum(vel_abs * ra_hat, axis=1) / (CFG.KAPPA * D)
    pm_dec = np.sum(vel_abs * dec_hat, axis=1) / (CFG.KAPPA * D)

    return np.rad2deg(RA) % 360.0, np.rad2deg(Dec), D, pm_ra, pm_dec, v_los

def projected_radius_np(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ra_c = math.radians(CFG.RA_CENTER)
    dec_c = math.radians(CFG.DEC_CENTER)
    cosang = (np.sin(dec) * math.sin(dec_c)
              + np.cos(dec) * math.cos(dec_c) * np.cos(ra - ra_c))
    ang = np.arccos(np.clip(cosang, -1.0, 1.0))
    return CFG.DIST_CENTER * ang


# ============================================================
# 4. LOAD DATA + EMPIRICAL PROJECTED TARGETS 
# ============================================================
print("Loading data...")
obs_path = os.path.join(CFG.input_dir, "draco_nfw_observations.csv")
truth_gal_path = os.path.join(CFG.input_dir, "draco_nfw_truth_galactocentric.csv")

if not os.path.exists(obs_path):
    raise FileNotFoundError(f"Missing observed file: {obs_path}")

obs_df = pd.read_csv(obs_path)
truth_gal_df = pd.read_csv(truth_gal_path) if os.path.exists(truth_gal_path) else None

N = len(obs_df)
print(f"N stars = {N}")

OBS_COLS = ["RA", "Dec", "Dist", "pmRA", "pmDec", "v_los"]
ERR_COLS = ["e_RA", "e_Dec", "e_Dist", "e_pmRA", "e_pmDec", "e_v_los"]

obs_mes = torch.tensor(obs_df[OBS_COLS].values, dtype=TD, device=DEVICE)
err_mes = torch.tensor(obs_df[ERR_COLS].values, dtype=TD, device=DEVICE)

if truth_gal_df is not None:
    truth_pos = torch.tensor(truth_gal_df[["x", "y", "z"]].values, dtype=TD, device=DEVICE)
    truth_vel = torch.tensor(truth_gal_df[["vx", "vy", "vz"]].values, dtype=TD, device=DEVICE)
    r_truth_np = np.linalg.norm(truth_gal_df[["x", "y", "z"]].values, axis=1)
else:
    truth_pos = truth_vel = None
    r_truth_np = None

with torch.no_grad():
    noisy_pos, noisy_vel = obs_to_galactocentric_torch(obs_mes)
noisy_pos_np = noisy_pos.cpu().numpy()
noisy_vel_np = noisy_vel.cpu().numpy()
r_noisy_np = np.linalg.norm(noisy_pos_np, axis=1)

R_proj_np = projected_radius_np(obs_df["RA"].values, obs_df["Dec"].values)
vlos_np = obs_df["v_los"].values
e_vlos_np = obs_df["e_v_los"].values

# error-weighted systemic velocity (do not assume V_SYS is known perfectly)
w_v = 1.0 / np.maximum(e_vlos_np, 0.05) ** 2
v_sys_hat = float(np.sum(w_v * vlos_np) / np.sum(w_v))
dv_np = vlos_np - v_sys_hat
print(f"Systemic v_los estimate: {v_sys_hat:.3f} km/s (truth input {CFG.V_SYS_LOS})")

# ---- optional: remove the systemic PER STAR for the anchor targets ----
# A galaxy moves as one body with a systemic velocity VECTOR. Each star's line
# of sight points in a slightly different direction, so the projection of that
# vector varies across the field: the perspective rotation. Subtracting a single
# scalar leaves that gradient in the data. It is largest at the largest angular
# offset -- exactly where the VSP's R^p weight lives, and it enters the fourth
# moment as the fourth power.
#
# _build_projected_data() already removes it exactly for the likelihood. Doing
# the same here makes the sigma_los and VSP targets use the same definition of
# velocity as the model they are scored against, instead of a different one.
vlos_anchor_np = vlos_np
if CFG.anchor_perspective:
    _ra_s = np.deg2rad(obs_df["RA"].values)
    _dec_s = np.deg2rad(obs_df["Dec"].values)
    _l_hat_s = np.column_stack([np.cos(_dec_s) * np.cos(_ra_s),
                                np.cos(_dec_s) * np.sin(_ra_s),
                                np.sin(_dec_s)])
    _ra_c = math.radians(CFG.RA_CENTER)
    _dec_c = math.radians(CFG.DEC_CENTER)
    _l0 = np.array([math.cos(_dec_c) * math.cos(_ra_c),
                    math.cos(_dec_c) * math.sin(_ra_c), math.sin(_dec_c)])
    _ra_hat0 = np.array([-math.sin(_ra_c), math.cos(_ra_c), 0.0])
    _dec_hat0 = np.array([-math.sin(_dec_c) * math.cos(_ra_c),
                          -math.sin(_dec_c) * math.sin(_ra_c),
                          math.cos(_dec_c)])
    if CFG.pm_use_systemic_from_data:
        _wp = 1.0 / np.maximum(obs_df["e_pmRA"].values, 1e-12) ** 2
        _pmra0 = float(np.sum(_wp * obs_df["pmRA"].values) / np.sum(_wp))
        _wp = 1.0 / np.maximum(obs_df["e_pmDec"].values, 1e-12) ** 2
        _pmdec0 = float(np.sum(_wp * obs_df["pmDec"].values) / np.sum(_wp))
    else:
        _pmra0, _pmdec0 = CFG.PM_SYS_RA, CFG.PM_SYS_DEC
    _V0 = (v_sys_hat * _l0 + CFG.KAPPA * CFG.DIST_CENTER
           * (_pmra0 * _ra_hat0 + _pmdec0 * _dec_hat0))
    _vsys_star = _l_hat_s @ _V0
    vlos_anchor_np = vlos_np - _vsys_star
    dv_np = vlos_anchor_np - float(np.sum(w_v * vlos_anchor_np) / np.sum(w_v))
    print(f"[anchors] perspective rotation removed per star; the constant-only "
          f"subtraction left a {_vsys_star.max() - _vsys_star.min():.2f} km/s "
          f"gradient across the field")


def build_sigma_los_targets(Rp, dv, ev, n_bins, var_floor):
    """Binned LOS dispersion with measurement variance subtracted in quadrature."""
    edges = np.quantile(Rp, np.linspace(0.02, 0.98, n_bins + 1))
    edges = np.unique(edges)

    Rc, sig, sig_err, counts = [], [], [], []
    for i in range(len(edges) - 1):
        m = (Rp >= edges[i]) & (Rp < edges[i + 1])
        n = int(np.sum(m))
        if n < 25:
            continue
        v = dv[m] - np.mean(dv[m])
        var_obs = np.var(v, ddof=1)
        var_meas = np.mean(ev[m] ** 2)
        var_corr = max(var_obs - var_meas, var_floor)
        s = math.sqrt(var_corr)

        # sampling error of the dispersion + propagated noise-correction error
        err_stat = s / math.sqrt(2.0 * max(n - 1, 1))
        err_meas = var_meas / (2.0 * s * math.sqrt(max(n, 1)))
        e = math.sqrt(err_stat**2 + err_meas**2)

        Rc.append(np.median(Rp[m]))
        sig.append(s)
        sig_err.append(max(e, 0.15))
        counts.append(n)

    if len(Rc) < 4:
        raise RuntimeError("Too few sigma_los bins; lower n_sig_bins or check data.")
    return (np.asarray(Rc), np.asarray(sig), np.asarray(sig_err),
            np.asarray(counts), edges)

def weighted_mean_np(x, e, floor=1.0e-12):
    """
    Error-weighted mean using only observed values and their uncertainties.
    """
    x = np.asarray(x, dtype=float)
    e = np.asarray(e, dtype=float)
    w = 1.0 / np.maximum(e, floor) ** 2
    return float(np.sum(w * x) / np.sum(w))

def build_sigma_pm_targets_RT(Rp, ra_deg, dec_deg, pmra, pmdec, epmra, epmdec,
                              n_bins, var_floor, edges=None):
    """Noise-corrected projected-RADIAL and projected-TANGENTIAL PM
    dispersion profiles (separately). Uses tangent-plane rotation about
    the galaxy center; distance = DIST_CENTER (systemic)."""
    ra = np.deg2rad(np.asarray(ra_deg)); dec = np.deg2rad(np.asarray(dec_deg))
    ra_c = math.radians(CFG.RA_CENTER); dec_c = math.radians(CFG.DEC_CENTER)
    # gnomonic tangent-plane offsets (xi = RA-direction, eta = Dec-direction)
    cosc = np.sin(dec_c)*np.sin(dec) + np.cos(dec_c)*np.cos(dec)*np.cos(ra-ra_c)
    xi  = np.cos(dec)*np.sin(ra-ra_c)/cosc
    eta = (np.cos(dec_c)*np.sin(dec)
           - np.sin(dec_c)*np.cos(dec)*np.cos(ra-ra_c))/cosc
    phi = np.arctan2(eta, xi)              # position angle in the plane
    c, s = np.cos(phi), np.sin(phi)

    if CFG.pm_use_systemic_from_data:
        pmra0 = weighted_mean_np(pmra, epmra)
        pmdec0 = weighted_mean_np(pmdec, epmdec)
    else:
        pmra0, pmdec0 = CFG.PM_SYS_RA, CFG.PM_SYS_DEC

    D = CFG.DIST_CENTER
    v_ra  = CFG.KAPPA * D * (np.asarray(pmra)  - pmra0)
    v_dec = CFG.KAPPA * D * (np.asarray(pmdec) - pmdec0)
    e_ra  = CFG.KAPPA * D * np.asarray(epmra)
    e_dec = CFG.KAPPA * D * np.asarray(epmdec)

    vR =  v_ra * c + v_dec * s            # projected radial component
    vT = -v_ra * s + v_dec * c            # projected tangential component
    eR2 = (e_ra * c) ** 2 + (e_dec * s) ** 2
    eT2 = (e_ra * s) ** 2 + (e_dec * c) ** 2

    if edges is None:
        edges = np.unique(np.quantile(Rp, np.linspace(0.02, 0.98, n_bins + 1)))

    out = {k: [] for k in ("Rc", "sR", "eSR", "sT", "eST", "n")}
    for i in range(len(edges) - 1):
        m = (Rp >= edges[i]) & (Rp < edges[i + 1])
        nb = int(np.sum(m))
        if nb < 25:
            continue
        def corr_disp(v, e2):
            vv = v[m] - np.mean(v[m])
            var_c = max(np.var(vv, ddof=1) - np.mean(e2[m]), var_floor)
            sd = math.sqrt(var_c)
            err = math.sqrt((sd / math.sqrt(2 * max(nb - 1, 1))) ** 2
                            + (np.mean(e2[m]) / (2 * sd * math.sqrt(nb))) ** 2)
            return sd, max(err, 0.15)
        sR, eSR = corr_disp(vR, eR2)
        sT, eST = corr_disp(vT, eT2)
        out["Rc"].append(np.median(Rp[m])); out["n"].append(nb)
        out["sR"].append(sR); out["eSR"].append(eSR)
        out["sT"].append(sT); out["eST"].append(eST)

    if len(out["Rc"]) < 4:
        raise RuntimeError("Too few sigma_pm bins for the R/T split.")
    return {k: np.asarray(v) for k, v in out.items()}, edges

def build_vsp_targets(Rp, vlos, evlos, n_boot=300, seed=123,
                      powers=(1, 3), r_quantile=1.0):
    """
    Build VSP-like fourth-moment constraints from observed projected data only.

    Data used:
        Rp from observed RA/Dec
        v_los observed
        e_v_los observed

    We define simple sample VSPs as:
        VSP1 = mean( <v_los^4>_corr * R )
        VSP2 = mean( <v_los^4>_corr * R^3 )

    The model side must use the same definitions.

    Noise correction:
        For Gaussian measurement noise n:
            E[(v+n)^4] = E[v^4] + 6 E[v^2] e^2 + 3 e^4

        We use a per-star approximate correction:
            v4_corr = v_obs^4 - 6 v_obs^2 e^2 + 3 e^4

        This is an approximate but non-cheating correction. It uses only
        observed v_los and e_v_los.
    """
    rng = np.random.default_rng(seed)

    Rp = np.asarray(Rp, dtype=float)
    vlos = np.asarray(vlos, dtype=float)
    evlos = np.asarray(evlos, dtype=float)

    # Optional truncation to the sampled tracer body. Past the outermost well
    # populated radius the model side is extrapolating <v_los^4>(R) and the data
    # side is one or two stars, so the two are not measuring the same thing.
    # The model is truncated to the same stars at the call site.
    if r_quantile < 1.0:
        keep = Rp <= float(np.quantile(Rp, r_quantile))
        Rp, vlos, evlos = Rp[keep], vlos[keep], evlos[keep]

    w = 1.0 / np.maximum(evlos, 0.05) ** 2
    v0 = float(np.sum(w * vlos) / np.sum(w))
    dv = vlos - v0

    def one(idx):
        Rb = Rp[idx]
        vb = dv[idx]
        eb = evlos[idx]

        e2 = eb ** 2

        # Approximate Gaussian-noise fourth-moment correction.
        v4_corr = vb ** 4 - 6.0 * vb ** 2 * e2 + 3.0 * e2 ** 2
        v4_corr = np.maximum(v4_corr, 0.0)

        return np.array([
            np.mean(v4_corr * Rb ** powers[0]),
            np.mean(v4_corr * Rb ** powers[1]),
        ])

    full = one(np.arange(len(Rp)))

    boots = []
    n = len(Rp)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(one(idx))

    boots = np.asarray(boots)
    err = np.std(boots, axis=0, ddof=1)

    # Very small floor to avoid infinite likelihood if bootstrap is pathological.
    err = np.maximum(err, 1.0e-12)

    return full, err


SIG_R_NP, SIG_T_NP, SIG_E_NP, SIG_N_NP, SIG_EDGES = build_sigma_los_targets(
    R_proj_np, dv_np, e_vlos_np, CFG.n_sig_bins, CFG.sig_var_floor)

print("\nNoise-corrected sigma_los(R) targets :")
for rc, ss, ee, n_bin in zip(SIG_R_NP, SIG_T_NP, SIG_E_NP, SIG_N_NP):
    print(f"  R={rc:.4f} kpc | sigma_los={ss:.3f} +/- {ee:.3f} km/s | N={n_bin}")

SIG_R_T = torch.tensor(SIG_R_NP, dtype=TD, device=DEVICE)
SIG_T_T = torch.tensor(SIG_T_NP, dtype=TD, device=DEVICE)
SIG_E_T = torch.tensor(SIG_E_NP, dtype=TD, device=DEVICE)

# ---- Optional projected proper-motion dispersion targets ----
if CFG.use_pm_anchor:
    PM_RT, PM_EDGES = build_sigma_pm_targets_RT(
        R_proj_np, obs_df["RA"].values, obs_df["Dec"].values,
        obs_df["pmRA"].values, obs_df["pmDec"].values,
        obs_df["e_pmRA"].values, obs_df["e_pmDec"].values,
        CFG.n_sig_bins, CFG.sig_var_floor, edges=SIG_EDGES)
    print("\nNoise-corrected sigma_pm R/T targets:")
    for i in range(len(PM_RT["Rc"])):
        print(f"  R={PM_RT['Rc'][i]:.4f} | sR={PM_RT['sR'][i]:.3f}+/-{PM_RT['eSR'][i]:.3f}"
              f" | sT={PM_RT['sT'][i]:.3f}+/-{PM_RT['eST'][i]:.3f} | N={PM_RT['n'][i]}")
    PM_R_T  = torch.tensor(PM_RT["Rc"],  dtype=TD, device=DEVICE)
    PM_SR_T = torch.tensor(PM_RT["sR"],  dtype=TD, device=DEVICE)
    PM_ESR_T= torch.tensor(PM_RT["eSR"], dtype=TD, device=DEVICE)
    PM_ST_T = torch.tensor(PM_RT["sT"],  dtype=TD, device=DEVICE)
    PM_EST_T= torch.tensor(PM_RT["eST"], dtype=TD, device=DEVICE)

else:
    PM_R_NP = PM_T_NP = PM_E_NP = PM_N_NP = None
    PM_R_T = PM_T_T = PM_E_T = None

# ---- Optional VSP fourth-moment targets ----
# Stars sample dN = 2 pi R Sigma(R) dR, so a plain average over stars of v^4 R^k
# estimates (2 pi / N) * int Sigma <v^4> R^(k+1) dR. The virial shape integrals
# carry R and R^3, hence star weights of R^0 and R^2. The historical default
# uses R and R^3, which is self-consistent between model and data but estimates
# one moment higher, with the variance pushed further into the outskirts.
VSP_POWERS = (0, 2) if CFG.vsp_star_weights else (1, 3)

if CFG.use_vsp_anchor and CFG.vsp_max_concentration < 1.0:
    _Rk = np.asarray(R_proj_np, dtype=float) ** VSP_POWERS[1]
    if CFG.vsp_r_quantile < 1.0:
        _Rk = _Rk[R_proj_np <= np.quantile(R_proj_np, CFG.vsp_r_quantile)]
    _conc = float(_Rk.max() / _Rk.sum())
    if _conc > CFG.vsp_max_concentration:
        print(f"\n[VSP] DISABLED: one star carries {_conc*100:.1f} per cent of "
              f"sum(R^{VSP_POWERS[1]}), above the {CFG.vsp_max_concentration*100:.0f} "
              f"per cent limit. A sum dominated by one star is not a measurement "
              f"of a fourth moment, and its bootstrap error is meaningless: "
              f"resampling cannot produce an outer star that was never observed.")
        CFG.use_vsp_anchor = False
    else:
        print(f"\n[VSP] concentration check passed: largest single share of "
              f"sum(R^{VSP_POWERS[1]}) is {_conc*100:.1f} per cent")

if CFG.use_vsp_anchor:
    VSP_OBS_NP, VSP_ERR_NP = build_vsp_targets(
        R_proj_np,
        vlos_anchor_np,
        e_vlos_np,
        n_boot=CFG.vsp_n_boot,
        seed=CFG.seed + 222,
        powers=VSP_POWERS,
        r_quantile=CFG.vsp_r_quantile,
    )

    print("\nVSP-like fourth-moment targets "
          f"(star weights R^{VSP_POWERS[0]}, R^{VSP_POWERS[1]}):")
    print(f"  VSP1 = {VSP_OBS_NP[0]:.4e} +/- {VSP_ERR_NP[0]:.4e}")
    print(f"  VSP2 = {VSP_OBS_NP[1]:.4e} +/- {VSP_ERR_NP[1]:.4e}")

    VSP_OBS_T = torch.tensor(VSP_OBS_NP, dtype=TD, device=DEVICE)
    VSP_ERR_T = torch.tensor(VSP_ERR_NP, dtype=TD, device=DEVICE)

    # Use observed projected radii for model-side conditional expectation.
    # This matches the data definition mean(v_los^4 * R^p) over the observed
    # stars -- including the truncation, which must be the same set of stars on
    # both sides or the two are not the same estimator.
    _R_vsp = np.asarray(R_proj_np, dtype=float)
    if CFG.vsp_r_quantile < 1.0:
        _keep_vsp = _R_vsp <= float(np.quantile(_R_vsp, CFG.vsp_r_quantile))
        _R_vsp = _R_vsp[_keep_vsp]
        print(f"  truncated at R <= {_R_vsp.max():.3f} kpc "
              f"(quantile {CFG.vsp_r_quantile}): {len(_R_vsp)} of "
              f"{len(R_proj_np)} stars, model and data alike")
    VSP_R_T = torch.tensor(_R_vsp, dtype=TD, device=DEVICE)

else:
    VSP_OBS_NP = VSP_ERR_NP = None
    VSP_OBS_T = VSP_ERR_T = VSP_R_T = None

# ---- rough potential init from the Wolf estimator on corrected dispersions ----
sig_glob = float(np.sqrt(max(np.var(dv_np, ddof=1) - np.mean(e_vlos_np**2),
                             CFG.sig_var_floor)))
R_half = float(np.median(R_proj_np))
r3 = 4.0 / 3.0 * R_half
M_wolf = 4.0 * sig_glob**2 * R_half / CFG.G       # Wolf+10: M(r3) ~ 4 sigma^2 R_e / G

Rs_init = float(np.clip(12.0 * R_half, 0.3, 10.0))
if CFG.rs_init_override > 0.0:
    print(f"Rs_init={Rs_init:.3f} from the 12*R_half heuristic, OVERRIDDEN to "
          f"{CFG.rs_init_override:.3f} (--rs-init)")
    Rs_init = float(CFG.rs_init_override)
print(f"Rs_init={Rs_init:.3f}")
x3 = r3 / Rs_init
mfun = math.log1p(x3) - x3 / (1.0 + x3)
M0_init = float(np.clip(M_wolf / max(mfun, 1e-6), 5.0e7, 5.0e9))

print(f"\nInit (noise-corrected Wolf): sigma_los={sig_glob:.2f} km/s, "
      f"R_half={R_half:.3f} kpc -> M0={M0_init:.3e}, Rs={Rs_init:.2f}")

# ---- GravSphere2-style benchmark radii (from mock metadata if present) ----
MOCK_META = {}
_meta_path = os.path.join(CFG.input_dir, "mock_metadata.json")
if os.path.exists(_meta_path):
    with open(_meta_path) as _f:
        MOCK_META = json.load(_f)

R_HALF_2D = float(MOCK_META.get("R_half_2d_empirical", R_half))
_bench = MOCK_META.get("benchmark_radial_range_kpc",
                       [0.1 * R_HALF_2D, 10.0 * R_HALF_2D])
R_BENCH_LO, R_BENCH_HI = float(_bench[0]), float(_bench[1])
print(f"R_1/2 = {R_HALF_2D:.4f} kpc | benchmark range = "
      f"[{R_BENCH_LO:.4f}, {R_BENCH_HI:.4f}] kpc")

def _trapz_w(x):
    """Trapezoid weights for a sorted 1D grid (torch)."""
    dx = x[1:] - x[:-1]
    w = torch.zeros_like(x)
    w[:-1] = w[:-1] + 0.5 * dx
    w[1:] = w[1:] + 0.5 * dx
    return w
 
 
PROJ = None
 
if CFG.training_mode == "projected":
 
    # ---- angular quadrature constants ----
    _gl_x, _gl_w = np.polynomial.legendre.leggauss(CFG.proj_n_phi)
    PHI_NODES = torch.tensor(0.5 * (_gl_x + 1.0) * math.pi,
                             dtype=TD, device=DEVICE)
    PHI_W = torch.tensor(_gl_w * (math.pi / 2.0),
                         dtype=TD, device=DEVICE)          # sums to pi
 
    Q_GRID = torch.linspace(0.0, 1.0, CFG.proj_n_q,
                            dtype=TD, device=DEVICE)       # q = R/r
 
    _stride = max(1, int(round(CFG.n_r_grid / CFG.proj_n_r)))
    PROJ_R_IDX = torch.arange(0, CFG.n_r_grid, _stride, device=DEVICE)
 
    # ---- per-star data: (R_i, w_i, e_i) ----
    def _build_projected_data():
        ra = np.deg2rad(obs_df["RA"].values)
        dec = np.deg2rad(obs_df["Dec"].values)
        l_hat = np.column_stack([np.cos(dec) * np.cos(ra),
                                 np.cos(dec) * np.sin(ra),
                                 np.sin(dec)])
 
        ra_c = math.radians(CFG.RA_CENTER)
        dec_c = math.radians(CFG.DEC_CENTER)
        l0 = np.array([math.cos(dec_c) * math.cos(ra_c),
                       math.cos(dec_c) * math.sin(ra_c),
                       math.sin(dec_c)])
        ra_hat0 = np.array([-math.sin(ra_c), math.cos(ra_c), 0.0])
        dec_hat0 = np.array([-math.sin(dec_c) * math.cos(ra_c),
                             -math.sin(dec_c) * math.sin(ra_c),
                             math.cos(dec_c)])
 
        # Systemic transverse velocity: from PM data (weighted mean) or
        # from the configured values. LOS systemic: the data estimate
        # v_sys_hat, as before.
        if CFG.pm_use_systemic_from_data:
            pmra0 = weighted_mean_np(obs_df["pmRA"].values,
                                     obs_df["e_pmRA"].values)
            pmdec0 = weighted_mean_np(obs_df["pmDec"].values,
                                      obs_df["e_pmDec"].values)
        else:
            pmra0, pmdec0 = CFG.PM_SYS_RA, CFG.PM_SYS_DEC
 
        V0 = (v_sys_hat * l0
              + CFG.KAPPA * CFG.DIST_CENTER
              * (pmra0 * ra_hat0 + pmdec0 * dec_hat0))
 
        # Perspective rotation removed EXACTLY: subtract the projection
        # of the full systemic velocity vector onto each star's own
        # line of sight (the generator applies the exact geometry, so a
        # constant v_sys subtraction would leave a ~0.3 km/s gradient).
        w_np = vlos_np - l_hat @ V0
 
        e_np = e_vlos_np.astype(float)
        e0 = float(min(CFG.proj_base_sigma, 0.9 * float(np.min(e_np))))
        if e0 <= 0.0:
            raise RuntimeError("proj_base_sigma must be positive and "
                               "<= min per-star e_v_los.")
 
        s_max = CFG.proj_s_max if CFG.proj_s_max > 0.0 else CFG.r_grid_max
        s_np = np.concatenate([[0.0],
                               np.logspace(-3, np.log10(s_max),
                                           CFG.proj_n_s - 1)])
        s_t = torch.tensor(s_np, dtype=TD, device=DEVICE)
 
        gh_x, gh_w = np.polynomial.hermite.hermgauss(CFG.proj_n_gh)
 
        return {
            "R": torch.tensor(R_proj_np, dtype=TD, device=DEVICE),
            "w": torch.tensor(w_np, dtype=TD, device=DEVICE),
            "e": torch.tensor(e_np, dtype=TD, device=DEVICE),
            "s": s_t,
            "s_w": _trapz_w(s_t),
            "gh_x": torch.tensor(gh_x, dtype=TD, device=DEVICE),
            "gh_w": torch.tensor(gh_w, dtype=TD, device=DEVICE),
            "e0": e0,
        }
 
    PROJ = _build_projected_data()
    print(f"[projected] N={N} stars | baseline smoothing e0="
          f"{PROJ['e0']:.3f} km/s | r-nodes={len(PROJ_R_IDX)} "
          f"q-nodes={CFG.proj_n_q} w-nodes={CFG.proj_n_w} "
          f"phi-nodes={CFG.proj_n_phi} s-nodes={CFG.proj_n_s}")
    print("[projected] data = (R, v_los) only; distances and proper "
          "motions are ignored. Encoder is built but never trained.")
 

# ============================================================
# 5. POTENTIAL MODELS
# ============================================================
def cumtrapz_t(y, x):
    dx = x[1:] - x[:-1]
    area = 0.5 * (y[1:] + y[:-1]) * dx
    return torch.cat([torch.zeros(1, dtype=y.dtype, device=y.device),
                      torch.cumsum(area, dim=0)])

def interp1d_t(xq, x_grid, y_grid):
    """Linear interpolation on a sorted 1D grid (values differentiable)."""
    xq_c = xq.clamp(x_grid[0], x_grid[-1])
    idx = torch.bucketize(xq_c, x_grid).clamp(1, len(x_grid) - 1)
    x0, x1 = x_grid[idx - 1], x_grid[idx]
    y0, y1 = y_grid[idx - 1], y_grid[idx]
    w = (xq_c - x0) / (x1 - x0 + 1e-30)
    return y0 + w * (y1 - y0)

class NFWPotential(nn.Module):
    def __init__(self, M0_init, Rs_init):
        super().__init__()
        self.log_M0 = nn.Parameter(torch.tensor(math.log(M0_init), dtype=TD))
        self.log_Rs = nn.Parameter(torch.tensor(math.log(Rs_init), dtype=TD))
        self.init_logM0 = math.log(M0_init)
        self.init_logRs = math.log(Rs_init)

    def params(self):
        return torch.exp(self.log_M0), torch.exp(self.log_Rs)

    def mass(self, r):
        M0, Rs = self.params()
        x = torch.clamp(r / Rs, min=1.0e-10)
        return M0 * (torch.log1p(x) - x / (1.0 + x))

    def density(self, r):
        M0, Rs = self.params()
        x = torch.clamp(r / Rs, min=1.0e-10)
        return (M0 / (4.0 * math.pi * Rs**3)) / (x * (1.0 + x) ** 2)

    def psi(self, r):
        """Relative potential Psi = -Phi > 0, Phi(inf)=0."""
        M0, Rs = self.params()
        x = torch.clamp(r / Rs, min=1.0e-10)
        return (CFG.G * M0 / Rs) * torch.log1p(x) / x

    def psi_central(self):
        M0, Rs = self.params()
        return CFG.G * M0 / Rs

    def acceleration(self, pos):
        r = torch.linalg.norm(pos, dim=1).clamp_min(1.0e-8)
        return (-CFG.G * self.mass(r) / r**3)[:, None] * pos

    def prior_loss(self):
        s = CFG.pot_prior_sigma
        return (((self.log_M0 - self.init_logM0) / s) ** 2
                + ((self.log_Rs - self.init_logRs) / s) ** 2) * 0.5

    def describe(self):
        with torch.no_grad():
            M0, Rs = self.params()
        return {"M0": float(M0), "Rs": float(Rs)}

class GNFWPotential(nn.Module):
    """gNFW: rho = rho_s x^{-gamma} (1+x)^{gamma-3}, gamma in (0.05, 1.95).
    Mass/potential via differentiable grid quadrature."""

    def __init__(self, M0_init, Rs_init, gamma_init=CFG.gamma_init,
                 r_min=1.0e-4, r_max=300.0, n_grid=240):
        super().__init__()
        rhos_init = M0_init / (4.0 * math.pi * Rs_init**3)
        self.log_rhos = nn.Parameter(torch.tensor(math.log(rhos_init), dtype=TD))
        self.log_Rs = nn.Parameter(torch.tensor(math.log(Rs_init), dtype=TD))
        g0 = (gamma_init - 0.05) / 1.9
        self.raw_gamma = nn.Parameter(torch.tensor(math.log(g0 / (1 - g0)), dtype=TD))

        self.init_log_rhos = math.log(rhos_init)
        self.init_logRs = math.log(Rs_init)

        rg = torch.logspace(
            math.log10(r_min),
            math.log10(r_max),
            n_grid,
            dtype=TD,
            device="cpu",
        )
        self.register_buffer("r_grid", rg)

    def gamma(self):
        return 0.05 + 1.9 * torch.sigmoid(self.raw_gamma)

    def _profiles(self):
        r = self.r_grid
        rhos = torch.exp(self.log_rhos)
        Rs = torch.exp(self.log_Rs)
        g = self.gamma()
        x = r / Rs
        rho = rhos * x.pow(-g) * (1.0 + x).pow(g - 3.0)

        M = 4.0 * math.pi * cumtrapz_t(rho * r * r, r)
        # analytic inner seed so M(r_min) > 0
        M = M + 4.0 * math.pi * rho[0] * r[0] ** 3 / (3.0 - g)

        I2_total = cumtrapz_t(rho * r, r)
        I2_out = I2_total[-1] - I2_total                 # int_r^rmax rho s ds
        tail = rho[-1] * r[-1] ** 2                      # rho ~ s^-3 beyond rmax
        psi = CFG.G * (M / r + 4.0 * math.pi * (I2_out + tail))
        return r, M, psi, rho

    def mass(self, r):
        rg, M, _, _ = self._profiles()
        logM = interp1d_t(torch.log(r.clamp_min(rg[0])), torch.log(rg),
                          torch.log(M.clamp_min(1e-30)))
        return torch.exp(logM)

    def psi(self, r):
        rg, _, psi, _ = self._profiles()
        return interp1d_t(torch.log(r.clamp(rg[0], rg[-1])), torch.log(rg), psi)

    def psi_central(self):
        _, _, psi, _ = self._profiles()
        return psi[0]

    def density(self, r):
        rhos = torch.exp(self.log_rhos)
        Rs = torch.exp(self.log_Rs)
        g = self.gamma()
        x = (r / Rs).clamp_min(1e-12)
        return rhos * x.pow(-g) * (1.0 + x).pow(g - 3.0)

    def acceleration(self, pos):
        r = torch.linalg.norm(pos, dim=1).clamp_min(1.0e-8)
        return (-CFG.G * self.mass(r) / r**3)[:, None] * pos

    def prior_loss(self):
        s = CFG.pot_prior_sigma
        return (((self.log_rhos - self.init_log_rhos) / s) ** 2
                + ((self.log_Rs - self.init_logRs) / s) ** 2) * 0.5

    def describe(self):
        with torch.no_grad():
            return {"rho_s": float(torch.exp(self.log_rhos)),
                    "Rs": float(torch.exp(self.log_Rs)),
                    "gamma": float(self.gamma())}

class ZhaoPotential(nn.Module):
    """
    Generalized double-power-law spherical halo:

        rho(r) = rho_s * x^(-gamma)
                 * (1 + x^alpha)^(-(beta - gamma) / alpha)

    with:
        x = r / Rs

    Recommended middle ground between:
        - gNFW: low-dimensional but restrictive
        - free_mass: very flexible but difficult to optimize

    Defaults:
        beta = 3 fixed
        gamma in (0.0, 1.8)
        alpha in (0.3, 4.0)
    """

    def __init__(
        self,
        M0_init,
        Rs_init,
        gamma_init=CFG.gamma_init,
        alpha_init=1.0,
        beta=3.0,
        r_min=1.0e-4,
        r_max=300.0,
        n_grid=280,
    ):
        super().__init__()

        self.beta = float(beta)

        rhos_init = M0_init / (4.0 * math.pi * Rs_init**3)

        self.log_rhos = nn.Parameter(
            torch.tensor(math.log(rhos_init), dtype=TD)
        )

        self.log_Rs = nn.Parameter(
            torch.tensor(math.log(Rs_init), dtype=TD)
        )

        # gamma in (0.001, 1.8)
        g_lo, g_hi = 0.001, 1.8
        g_u = (gamma_init - g_lo) / (g_hi - g_lo)
        g_u = float(np.clip(g_u, 1e-5, 1.0 - 1e-5))

        self.raw_gamma = nn.Parameter(
            torch.tensor(math.log(g_u / (1.0 - g_u)), dtype=TD)
        )

        # alpha in (0.3, 4.0)
        a_lo, a_hi = 0.3, 4.0
        a_u = (alpha_init - a_lo) / (a_hi - a_lo)
        a_u = float(np.clip(a_u, 1e-5, 1.0 - 1e-5))

        self.raw_alpha = nn.Parameter(
            torch.tensor(math.log(a_u / (1.0 - a_u)), dtype=TD)
        )

        self.init_log_rhos = math.log(rhos_init)
        self.init_logRs = math.log(Rs_init)

        rg = torch.logspace(
            math.log10(r_min),
            math.log10(r_max),
            n_grid,
            dtype=TD,
            device="cpu",
        )

        self.register_buffer("r_grid", rg)

    def gamma(self):
        return 0.001 + 1.799 * torch.sigmoid(self.raw_gamma)

    def alpha(self):
        return 0.3 + 3.7 * torch.sigmoid(self.raw_alpha)

    def density(self, r):
        rhos = torch.exp(self.log_rhos)
        Rs = torch.exp(self.log_Rs)
        gamma = self.gamma()
        alpha = self.alpha()
        beta = self.beta

        x = (r / Rs).clamp_min(1.0e-12)

        return (
            rhos
            * x.pow(-gamma)
            * (1.0 + x.pow(alpha)).pow(-(beta - gamma) / alpha)
        )

    def _profiles(self):
        r = self.r_grid
        rho = self.density(r)

        M = 4.0 * math.pi * cumtrapz_t(rho * r * r, r)

        # Analytic inner seed:
        # rho ~ rho(r_min) * (r / r_min)^(-gamma)
        gamma = self.gamma()

        M_inner = (
            4.0
            * math.pi
            * rho[0]
            * r[0] ** 3
            / (3.0 - gamma)
        )

        M = M + M_inner

        # Psi(r) = G [M(r)/r + 4 pi int_r^inf rho(s) s ds]
        I2_total = cumtrapz_t(rho * r, r)
        I2_out = I2_total[-1] - I2_total

        # beta = 3 outer behaviour: rho ~ r^-3
        tail = rho[-1] * r[-1] ** 2

        psi = CFG.G * (
            M / r
            + 4.0 * math.pi * (I2_out + tail)
        )

        return r, M, rho, psi

    def mass(self, r):
        rg, M, _, _ = self._profiles()

        return torch.exp(
            interp1d_t(
                torch.log(r.clamp(rg[0], rg[-1])),
                torch.log(rg),
                torch.log(M.clamp_min(1.0e-30)),
            )
        )

    def psi(self, r):
        rg, _, _, psi = self._profiles()

        return interp1d_t(
            torch.log(r.clamp(rg[0], rg[-1])),
            torch.log(rg),
            psi,
        )

    def psi_central(self):
        _, _, _, psi = self._profiles()
        return psi[0]

    def acceleration(self, pos):
        r = torch.linalg.norm(pos, dim=1).clamp_min(1.0e-8)

        return (
            -CFG.G * self.mass(r) / r**3
        )[:, None] * pos

    def prior_loss(self):
        s = CFG.pot_prior_sigma

        weak_scale_prior = 0.5 * (
            ((self.log_rhos - self.init_log_rhos) / s) ** 2
            + ((self.log_Rs - self.init_logRs) / s) ** 2
        )

        # Weak regularization only: discourages extreme transition shapes.
        weak_alpha_prior = 0.02 * (torch.log(self.alpha()) ** 2)

        return weak_scale_prior + weak_alpha_prior

    def describe(self):
        with torch.no_grad():
            return {
                "rho_s": float(torch.exp(self.log_rhos)),
                "Rs": float(torch.exp(self.log_Rs)),
                "gamma": float(self.gamma()),
                "alpha": float(self.alpha()),
                "beta": float(self.beta),
            }

class FreeSphericalMassPotential(nn.Module):
    """
    Semi-parametric spherical halo with adaptive inner-region coupling.

    Parametrized in the DENSITY log-slope gamma(r) = -dln rho/dln r:

        gamma(r) = gamma_backbone(r; theta_global) + delta(r; knots)

    rho(r) is reconstructed from gamma by quadrature, then M(r) and the
    relative potential Psi(r) = -Phi(r). Because rho is built as exp(log rho),
    positivity (rho>0), mass monotonicity (dM/dr>0) and 0 < dlnM/dlnr <= 3 are
    automatic -- no slope clamps.
    """

    def __init__(
        self,
        M0_init,
        Rs_init,
        r_min=1.0e-4,
        r_max=300.0,
        n_knots=16,
        r_inner_support=0.03,
        r_outer_support=None,
        gamma_inner_init=1.0,
        n_dense=None,
    ):
        super().__init__()

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.n_knots = int(n_knots)
        self.r_inner_support = float(r_inner_support)
        self.r_outer_support = float(r_outer_support) if r_outer_support else 1.0

        # Dense integration grid (full range) -> accurate M(r), Psi(r).
        self.n_dense = int(n_dense) if n_dense else max(6 * self.n_knots, 320)
        r_dense = torch.logspace(math.log10(self.r_min), math.log10(self.r_max),
                                 self.n_dense, dtype=TD, device="cpu")
        self.register_buffer("r_dense", r_dense)
        self.register_buffer("logr_dense", torch.log(r_dense))

        # Residual knots concentrated AROUND the data band (flexibility is only
        # informative there); the backbone handles everything outside.
        r_lo_knot = max(self.r_min, self.r_inner_support / 3.0)
        r_hi_knot = min(self.r_max, self.r_outer_support * 3.0)
        r_knots = torch.logspace(math.log10(r_lo_knot), math.log10(r_hi_knot),
                                 self.n_knots, dtype=TD, device="cpu")
        self.register_buffer("r_grid", r_knots)            # name kept for compat
        self.register_buffer("logr_knots", torch.log(r_knots))

        # ---- global backbone density-slope params (data-coupled) ----
        gin0 = float(np.clip(gamma_inner_init / 1.8, 1e-3, 1 - 1e-3))
        self.raw_gin = nn.Parameter(torch.tensor(math.log(gin0 / (1 - gin0)), dtype=TD))
        gout0 = float(np.clip((3.0 - 2.2) / 1.4, 1e-3, 1 - 1e-3))
        self.raw_gout = nn.Parameter(torch.tensor(math.log(gout0 / (1 - gout0)), dtype=TD))
        self.log_rt = nn.Parameter(torch.tensor(math.log(float(Rs_init)), dtype=TD))
        self.raw_w = nn.Parameter(torch.tensor(math.log(math.exp(0.6) - 1.0), dtype=TD))

        # Density normalization, anchored inside the data band.
        r_anchor = math.sqrt(self.r_inner_support * self.r_outer_support)
        self.anchor_idx = int(np.argmin(np.abs(r_dense.numpy() - r_anchor)))
        xa = r_anchor / float(Rs_init)
        rho_anchor = (M0_init / (4.0 * math.pi * float(Rs_init) ** 3)) / (
            max(xa, 1e-12) * (1.0 + xa) ** 2)
        self.log_rho_ref = nn.Parameter(torch.tensor(math.log(max(rho_anchor, 1e-12)), dtype=TD))

        # ---- nonparametric residual on the density slope ----
        self.delta = nn.Parameter(torch.zeros(self.n_knots, dtype=TD))

        # ---- learnable GP correlation length (e-folds in ln r) ----
        self.log_ell = nn.Parameter(torch.tensor(math.log(2.5), dtype=TD))

        # fixed hyperparameters (override via CFG if present)
        self._gp_sigma = float(getattr(CFG, "free_mass_gp_sigma", 0.6))
        self._ell0 = math.log(float(getattr(CFG, "free_mass_ell0", 2.5)))
        self._ell_std = float(getattr(CFG, "free_mass_ell_std", 0.5))

    # ---------------- backbone + residual -> gamma(r) ----------------
    def _scalars(self):
        g_in = 1.8 * torch.sigmoid(self.raw_gin)
        g_out = 2.2 + 1.4 * torch.sigmoid(self.raw_gout)
        w = torch.nn.functional.softplus(self.raw_w) + 0.1
        return g_in, g_out, self.log_rt, w

    def _gamma_dense(self):
        g_in, g_out, log_rt, w = self._scalars()
        t = torch.sigmoid((self.logr_dense - log_rt) / w)
        gamma_bb = g_in + (g_out - g_in) * t
        # residual interpolated from knots; outside the knot span it holds the
        # endpoint values (~0), so the backbone governs the extrapolation.
        delta_dense = interp1d_t(self.logr_dense, self.logr_knots, self.delta)
        return gamma_bb + delta_dense, g_in, g_out

    def _compute(self):
        r = self.r_dense
        logr = self.logr_dense
        gamma, g_in, g_out = self._gamma_dense()

        # log rho via  dln rho = -gamma dln r , anchored at r_anchor
        dlr = logr[1:] - logr[:-1]
        gmid = 0.5 * (gamma[1:] + gamma[:-1])
        cum = torch.cat([torch.zeros(1, dtype=TD, device=r.device),
                         torch.cumsum(gmid * dlr, dim=0)])
        log_rho = self.log_rho_ref - (cum - cum[self.anchor_idx])
        rho = torch.exp(log_rho.clamp(-60.0, 60.0))

        # enclosed mass: analytic inner seed + 4 pi int rho s^2 ds
        integ_m = rho * r * r
        dr = r[1:] - r[:-1]
        area_m = 0.5 * (integ_m[1:] + integ_m[:-1]) * dr
        M_cum = 4.0 * math.pi * torch.cat(
            [torch.zeros(1, dtype=TD, device=r.device), torch.cumsum(area_m, dim=0)])
        M_seed = 4.0 * math.pi * rho[0] * r[0] ** 3 / (3.0 - g_in).clamp_min(0.1)
        M = M_seed + M_cum

        # relative potential Psi(r) = G int_r^inf M/s^2 ds  (+ point-mass tail)
        integ_p = M / (r * r)
        area_p = 0.5 * (integ_p[1:] + integ_p[:-1]) * dr
        rev = torch.flip(torch.cumsum(torch.flip(area_p, dims=[0]), dim=0), dims=[0])
        integral_out = torch.cat([rev, torch.zeros(1, dtype=TD, device=r.device)])
        tail = M[-1] / r[-1]
        psi = CFG.G * (integral_out + tail)

        return r, M, rho, psi, gamma, g_in, g_out

    def _profiles(self):
        r, M, rho, psi, _, _, _ = self._compute()
        return r, M, rho, psi

    # ---------------- interface (unchanged signatures) ----------------
    def mass(self, r):
        rg, M, _, _ = self._profiles()
        return torch.exp(interp1d_t(torch.log(r.clamp(rg[0], rg[-1])),
                                    torch.log(rg), torch.log(M.clamp_min(1e-30))))

    def density(self, r):
        rg, _, rho, _ = self._profiles()
        return torch.exp(interp1d_t(torch.log(r.clamp(rg[0], rg[-1])),
                                    torch.log(rg), torch.log(rho.clamp_min(1e-30))))

    def psi(self, r):
        rg, _, _, psi = self._profiles()
        return interp1d_t(torch.log(r.clamp(rg[0], rg[-1])), torch.log(rg), psi)

    def psi_central(self):
        _, _, _, psi = self._profiles()
        return psi[0]

    def acceleration(self, pos):
        r = torch.linalg.norm(pos, dim=1).clamp_min(self.r_min)
        return (-CFG.G * self.mass(r) / r ** 3)[:, None] * pos

    # ---------------- GP prior on the residual ----------------
    def _gp_term(self):
        # tiny system; compute on CPU for device-robust Cholesky.
        d = self.delta.cpu()
        lr = self.logr_knots.cpu()
        n = d.numel()
        ell = torch.exp(self.log_ell).cpu().clamp(0.3, 6.0)
        D2 = (lr[:, None] - lr[None, :]) ** 2
        K = torch.exp(-0.5 * D2 / (ell * ell)) + 1e-4 * torch.eye(n, dtype=lr.dtype)
        L = torch.linalg.cholesky(K)
        x = torch.cholesky_solve(d[:, None], L)[:, 0]
        quad = torch.dot(d, x) / (self._gp_sigma ** 2)
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        return 0.5 * (quad + logdet)

    def prior_loss(self):
        _, _, _, _, gamma, _, _ = self._compute()
        dev = self.delta.device

        # 1) GP prior on the slope residual (adaptive smoothness + coupling)
        w_gp = float(getattr(CFG, "free_mass_smooth_weight", 2.0e-2))
        loss = w_gp * self._gp_term().to(dev)

        # 2) pin extrapolation: residual -> 0 outside the data band so the
        #    inner/outer profile follows the data-coupled backbone, not noise.
        lo = math.log(self.r_inner_support)
        hi = math.log(self.r_outer_support)
        band_w = torch.relu(lo - self.logr_knots) + torch.relu(self.logr_knots - hi)
        w_band = float(getattr(CFG, "free_mass_inner_anchor_weight", 1.0e-1))
        loss = loss + w_band * torch.mean(band_w * self.delta ** 2)

        # 3) physical shape priors on gamma = -dln rho/dln r
        w_mono = float(getattr(CFG, "free_mass_mono_weight", 1.0e-2))
        w_pos = float(getattr(CFG, "free_mass_nonneg_weight", 1.0e-2))
        loss = loss + w_mono * torch.mean(torch.relu(gamma[:-1] - gamma[1:]) ** 2)
        loss = loss + w_pos * torch.mean(torch.relu(-gamma) ** 2)

        # 4) hyperprior on the correlation length
        w_ell = float(getattr(CFG, "free_mass_ell_weight", 1.0))
        loss = loss + w_ell * 0.5 * ((self.log_ell - self._ell0) / self._ell_std) ** 2
        return loss

    def describe(self):
        with torch.no_grad():
            g_in, g_out, log_rt, w = self._scalars()
            r1 = torch.tensor([1.0], dtype=TD, device=self.r_dense.device)
            return {
                "gamma_inner": float(g_in),
                "gamma_outer": float(g_out),
                "r_transition": float(torch.exp(self.log_rt)),
                "trans_width_efold": float(w),
                "corr_length_efold": float(torch.exp(self.log_ell).clamp(0.3, 6.0)),
                "M_at_1kpc": float(self.mass(r1)[0]),
            }

# ============================================================
# CoreNFWPotential — mass-anchored (gauge-fixed) coreNFW
# ============================================================
# Trainable params:
#   log_Mr3 : log of M(< r_anchor), r_anchor = Wolf radius r3 = (4/3) R_half.
#             This is the combination the kinematics actually measure.
#   log_Rs  : NFW scale radius (weak data pull, proper prior)
#   raw_rc  : bounded core radius, r_c in [0.2, 5] * R_half (Read+16 regime)
#   raw_n   : core strength, n in (0,1); inner slope = n - 1.
#             Can be FROZEN for the profile-likelihood scan.
#
# M0 is DERIVED each forward pass:
#   M0 = Mr3 / [ m_nfw(r3/Rs) * tanh(r3/rc)^n ]
# so the (amplitude, r_core) gauge direction is removed by construction:
# any move in (rc, Rs, n) leaves M(r3) invariant and must pay likelihood
# cost through the in-band mass SHAPE, which is the measured thing.

class CoreNFWPotential(nn.Module):

    def __init__(self, M0_init, Rs_init, r_half=None, n_init=0.5,
                 rc_lo_mult=0.2, rc_hi_mult=3.0, rc_center_mult=1.25,
                 rc_prior_sigma_ln=0.5,
                 r_min=1.0e-4, r_max=300.0, n_grid=260,
                 M_anchor_init=None, fix_n=None):
        super().__init__()

        if r_half is None or not np.isfinite(r_half) or r_half <= 0.0:
            raise ValueError("CoreNFW v3 requires r_half (pass R_HALF_2D).")
        self.r_half = float(r_half)
        self.r_anchor = (4.0 / 3.0) * self.r_half          # Wolf radius r3

        self.rc_lo = float(rc_lo_mult * self.r_half)
        self.rc_hi = float(rc_hi_mult * self.r_half)
        self.rc_center = float(np.clip(rc_center_mult * self.r_half,
                                       self.rc_lo * 1.01, self.rc_hi * 0.99))
        self.rc_prior_sigma_ln = float(rc_prior_sigma_ln)

        # ---- amplitude leaf: M(< r3) ----
        if M_anchor_init is None:
            # translate the caller's (M0, Rs) init into M(r3)
            x3 = self.r_anchor / float(Rs_init)
            M_anchor_init = float(M0_init) * (math.log1p(x3) - x3 / (1.0 + x3))
        self.log_Mr3 = nn.Parameter(
            torch.tensor(math.log(max(M_anchor_init, 1e4)), dtype=TD))

        self.log_Rs = nn.Parameter(torch.tensor(math.log(float(Rs_init)), dtype=TD))

        u = 0.5
        self.raw_rc = nn.Parameter(torch.tensor(math.log(u / (1 - u)), dtype=TD))

        # n: either free or frozen (profile scan)
        self.n_fixed = None
        if fix_n is not None:
            _nmin = float(getattr(CFG, "n_core_min", 0.0))
            _nmax = float(getattr(CFG, "n_core_max", 1.0))
            self.n_fixed = float(np.clip(fix_n, _nmin + 1e-4, _nmax - 1e-4))
            self.raw_n = nn.Parameter(torch.tensor(0.0, dtype=TD),
                                      requires_grad=False)
        else:
            n_init = float(np.clip(n_init, 1e-3, 1 - 1e-3))
            self.raw_n = nn.Parameter(
                torch.tensor(math.log(n_init / (1 - n_init)), dtype=TD))

        self.init_logMr3 = float(self.log_Mr3.detach())
        self.init_logRs = math.log(float(Rs_init))

        rg = torch.logspace(math.log10(r_min), math.log10(r_max), n_grid,
                            dtype=TD, device="cpu")
        self.register_buffer("r_grid", rg)

    # ---------------- constrained scalars ----------------
    def _n(self):
        if self.n_fixed is not None:
            return torch.tensor(self.n_fixed, dtype=TD,
                                device=self.raw_n.device)
        # n in (n_min, 1). n_min defaults to 0, which is the historical
        # sigmoid range; below it gamma_0 = n - 1 can go steeper than -1.
        _nmin = float(getattr(CFG, "n_core_min", 0.0))
        _nmax = float(getattr(CFG, "n_core_max", 1.0))
        return _nmin + (_nmax - _nmin) * torch.sigmoid(self.raw_n)

    def _scalars(self):
        Mr3 = torch.exp(self.log_Mr3)
        Rs = torch.exp(self.log_Rs)
        rc = self.rc_lo + (self.rc_hi - self.rc_lo) * torch.sigmoid(self.raw_rc)
        n = self._n()
        # derived M0: keeps M(r3) exactly = Mr3 whatever (Rs, rc, n) do
        x3 = self.r_anchor / Rs
        m3 = torch.log1p(x3) - x3 / (1.0 + x3)
        f3 = torch.tanh(torch.tensor(1.0, dtype=TD, device=Rs.device) *
                        (self.r_anchor / rc)).clamp_min(1e-12)
        M0 = Mr3 / (m3.clamp_min(1e-12) * f3.pow(n))
        return M0, Rs, rc, n

    def n_core(self):
        return self._n()

    def inner_slope(self):
        return self._n() - 1.0

    @staticmethod
    def _M_nfw(r, M0, Rs):
        x = torch.clamp(r / Rs, min=1.0e-12)
        return M0 * (torch.log1p(x) - x / (1.0 + x))

    @staticmethod
    def _rho_nfw(r, M0, Rs):
        x = torch.clamp(r / Rs, min=1.0e-12)
        return (M0 / (4.0 * math.pi * Rs ** 3)) / (x * (1.0 + x) ** 2)

    def mass(self, r):
        M0, Rs, rc, n = self._scalars()
        f = torch.tanh((r / rc).clamp_min(0.0)).clamp_min(1.0e-30)
        return self._M_nfw(r, M0, Rs) * f.pow(n)

    def density(self, r):
        M0, Rs, rc, n = self._scalars()
        rr = r.clamp_min(1.0e-12)
        f = torch.tanh(rr / rc).clamp(1.0e-12, 1.0)
        rho_nfw = self._rho_nfw(rr, M0, Rs)
        M_nfw = self._M_nfw(rr, M0, Rs)
        term1 = f.pow(n) * rho_nfw
        term2 = (n * f.pow(n - 1.0) * (1.0 - f * f) * M_nfw
                 / (4.0 * math.pi * rr * rr * rc))
        return (term1 + term2).clamp_min(1.0e-30)

    def _profiles(self):
        r = self.r_grid
        rho = self.density(r)
        M = self.mass(r)
        I2_total = cumtrapz_t(rho * r, r)
        I2_out = I2_total[-1] - I2_total
        tail = rho[-1] * r[-1] ** 2
        psi = CFG.G * (M / r + 4.0 * math.pi * (I2_out + tail))
        return r, M, psi, rho

    def psi(self, r):
        rg, _, psi, _ = self._profiles()
        return interp1d_t(torch.log(r.clamp(rg[0], rg[-1])), torch.log(rg), psi)

    def psi_central(self):
        _, _, psi, _ = self._profiles()
        return psi[0]

    def acceleration(self, pos):
        r = torch.linalg.norm(pos, dim=1).clamp_min(1.0e-8)
        return (-CFG.G * self.mass(r) / r ** 3)[:, None] * pos

    def log_slope(self, r):
        rr = r.clamp_min(1.0e-8)
        dl = 1.0e-3
        sp = torch.log(self.density(rr * math.exp(dl)))
        sm = torch.log(self.density(rr * math.exp(-dl)))
        return (sp - sm) / (2.0 * dl)

    # ---- weak MAP-scale prior (goes through the usual inv_N weighting) ----
    def prior_loss(self):
        s = CFG.pot_prior_sigma
        return 0.5 * (((self.log_Mr3 - self.init_logMr3) / s) ** 2
                      + ((self.log_Rs - self.init_logRs) / s) ** 2)

    def strong_prior(self):
        """Shape prior on the anchored block, applied at full strength.
 
        The two terms are weighted independently: r_c is a physical
        scale the kinematics can constrain, R_s is the residual soft
        direction of Appendix A and is held near its initialization.
        Weights are in NLL units when --strong-prior-norm nll_units.
        """
        _, Rs, rc, _ = self._scalars()
 
        rc_pen = 0.5 * ((torch.log(rc) - math.log(self.rc_center))
                        / self.rc_prior_sigma_ln) ** 2
        rs_pen = 0.5 * ((torch.log(Rs) - self.init_logRs) / 1.0) ** 2
 
        w_rc = float(getattr(CFG, "rc_prior_weight", 1.0))
        w_rs = float(getattr(CFG, "rs_prior_weight", 0.3))
        out = w_rc * rc_pen + w_rs * rs_pen
 
        s_dex = float(getattr(CFG, "m_piv_prior_sigma", 0.0))
        if s_dex > 0.0:
            s_ln = s_dex * math.log(10.0)
            out = out + 0.5 * ((self.log_Mr3 - self.init_logMr3)
                               / s_ln) ** 2
        return out

    def describe(self):
        with torch.no_grad():
            M0, Rs, rc, n = self._scalars()
            return {"M0": float(M0), "Rs": float(Rs), "r_core": float(rc),
                    "n_core": float(n), "gamma_inner": float(n - 1.0),
                    "M_r3": float(torch.exp(self.log_Mr3))}
        
def make_corenfw(fix_n=None, M_anchor_init=None, n_init=0.5):
    if M_anchor_init is None:
        M_anchor_init = M_wolf * CFG.m_piv_init_scale
    return CoreNFWPotential(
        M0_init, Rs_init, r_half=R_HALF_2D, n_init=n_init,
        rc_lo_mult=CFG.rc_lo_mult, rc_hi_mult=CFG.rc_hi_mult,
        rc_center_mult=CFG.rc_center_mult,
        M_anchor_init=M_anchor_init,
        fix_n=fix_n,
    ).to(DEVICE)

if CFG.potential_model == "nfw":
    potential = NFWPotential(
        M0_init,
        Rs_init,
    ).to(DEVICE)

elif CFG.potential_model == "gnfw":
    potential = GNFWPotential(
        M0_init,
        Rs_init,
        gamma_init=CFG.gamma_init,
    ).to(DEVICE)

elif CFG.potential_model == "zhao":
    potential = ZhaoPotential(
        M0_init=M0_init,
        Rs_init=Rs_init,
        gamma_init=CFG.gamma_init,
        alpha_init=1.0,
        beta=3.0,
    ).to(DEVICE)

elif CFG.potential_model == "free_mass":
    potential = FreeSphericalMassPotential(
        M0_init=M0_init,
        Rs_init=Rs_init,
        r_min=CFG.free_mass_r_min,
        r_max=CFG.free_mass_r_max,
        n_knots=CFG.free_mass_n_knots,
    ).to(DEVICE)

elif CFG.potential_model == "corenfw":
    potential = make_corenfw(
        fix_n=getattr(ARGS, "fix_n", None),
        n_init=getattr(ARGS, "n_core_init", 0.5),
    )

else:
    raise ValueError(
        "potential_model must be 'nfw', 'gnfw', 'zhao', 'free_mass' or 'corenfw' (Used in paper)."
    )

with torch.no_grad():
    PSI_REF = float(potential.psi_central())   # fixed feature scale for the DF
L_REF = 1.0                                    # kpc * km/s
print(f"PSI_REF = {PSI_REF:.4e} (km/s)^2")


# ============================================================
# 6. DF MODEL: f(eps, L), EQUILIBRIUM BY CONSTRUCTION
# ============================================================
class FELDF(nn.Module):
    """log f(eps, L) = logf0 + a_e * log(eps/PSI_REF) + MLP(features)
    plus a smooth exponential suppression for eps <= 0 (soft unbound tail).

    Because f depends on phase space only through (eps, L), it is an exact
    steady-state solution of the CBE in any spherical potential (Jeans
    theorem). Stationarity is therefore built in, not penalized.
    """

    def __init__(self, hidden=64, isotropic=False):
        super().__init__()
        self.isotropic = isotropic
        in_dim = 1 if isotropic else 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.logf0 = nn.Parameter(torch.tensor(0.0, dtype=TD))
        self.a_e = nn.Parameter(torch.tensor(2.0, dtype=TD))

    def log_f(self, eps, L):
        u = eps.clamp_min(PSI_REF * 1.0e-8) / PSI_REF
        fe = torch.log(u)
        if self.isotropic:
            feats = fe[:, None]
        else:
            fl = torch.log1p(L.clamp_min(0.0) / L_REF)
            feats = torch.stack([fe, fl], dim=-1)
        h = self.mlp(feats).squeeze(-1)
        base = self.logf0 + self.a_e * fe
        tail = -CFG.unbound_kappa * torch.relu(-eps) / PSI_REF
        return (base + h).clamp(-80.0, 40.0) + tail


df = FELDF(hidden=CFG.df_hidden, isotropic=CFG.df_isotropic).to(DEVICE)

class BetaCOM(torch.nn.Module):
    """Jointly-fitted Cuddeford-Osipkov-Merritt anisotropy curve.
 
        beta(r) = (beta0 + (r/ra)^2) / (1 + (r/ra)^2)
 
    beta0 is box-bounded to [-0.5, 1] via a sigmoid (the GraphNPE
    prior range; Eddington inversions are unstable below -0.5 and
    observed dwarfs sit at beta0 ~ -0.2..0). log ra is free within
    [0.1, 30] x R_half. Two parameters -- exactly what two VSP
    integrals can constrain.
    """
 
    def __init__(self, r_half):
        super().__init__()
        self.raw_b0 = torch.nn.Parameter(torch.zeros(()))
        self.raw_ra = torch.nn.Parameter(torch.zeros(()))
        self.lo, self.hi = -0.5, 1.0
        self.lra0 = math.log(r_half)          # centre: ra ~ R_half
        self.lra_span = math.log(30.0 / 0.1) / 2.0
 
    def params(self):
        b0 = self.lo + (self.hi - self.lo) * torch.sigmoid(self.raw_b0)
        ra = torch.exp(self.lra0
                       + self.lra_span * torch.tanh(self.raw_ra))
        return b0, ra
 
    def forward(self, r):
        b0, ra = self.params()
        x2 = (r / ra) ** 2
        return (b0 + x2) / (1.0 + x2)
    
beta_com = BetaCOM(R_HALF_2D).to(DEVICE)

# ============================================================
# 7. MODEL GRIDS: MOMENTS, NORMALIZATION, PROJECTED SIGMA_LOS
# ============================================================
R_GRID = torch.logspace(
    math.log10(CFG.r_grid_min),
    math.log10(CFG.r_grid_max),
    CFG.n_r_grid,
    dtype=TD,
    device="cpu",
).to(DEVICE)

T_GRID = torch.linspace(
    0.02,
    0.999,
    CFG.n_v_grid,
    dtype=TD,
    device=DEVICE,
)

ETA_GRID = torch.linspace(
    0.02,
    math.pi / 2.0,
    CFG.n_eta_grid,
    dtype=TD,
    device=DEVICE,
)

S_PROJ = torch.logspace(
    -4,
    math.log10(CFG.r_grid_max),
    CFG.n_s_proj,
    dtype=TD,
    device="cpu",
).to(DEVICE)


def trapz_last(y, x):
    """Trapezoid over the last dim; x broadcastable to y's last dim."""
    dx = x[..., 1:] - x[..., :-1]
    return torch.sum(0.5 * (y[..., 1:] + y[..., :-1]) * dx, dim=-1)

def _los_velocity_grid(fv, v):
    """Local LOS-velocity density g(w | r, q) on (r_sub, q, w).
 
    Same mathematics as before (see the docstring history); the
    'window' path evaluates the Gaussian kernel only on the
    2*halfwin+1 w-cells around each quadrature sample's mean
    mu = v * A(q, eta, phi) and scatter-adds, instead of against
    every w-node. halfwin is chosen so the discarded tails are
    >= ~10 sigma: exact to float32 precision.
    """
    ridx = PROJ_R_IDX
    fv_s = fv[ridx]                          # [Rs, V, H]
    v_s = v[ridx]                            # [Rs, V]
    e0 = PROJ["e0"]
 
    with torch.no_grad():
        wmax = float(v_s.max()) * 1.02 + 4.0 * e0
    Nw = CFG.proj_n_w
    if "w_grid" not in PROJ:
        with torch.no_grad():
            wmax0 = 1.8 * float(v_s.max()) + 4.0 * e0
        PROJ["w_grid"] = torch.linspace(-wmax0, wmax0, CFG.proj_n_w,
                                        dtype=TD, device=DEVICE)
    W = PROJ["w_grid"]
    Nw = len(W)
    dW = float(W[1] - W[0])
 
    sin_e = torch.sin(ETA_GRID)
    cos_e = torch.cos(ETA_GRID)
    wT = _trapz_w(T_GRID)                    # [V]
    wH = _trapz_w(ETA_GRID)                  # [H]
    vesc_s = v_s[:, -1:] / T_GRID[-1]        # [Rs, 1]
 
    # quadrature prefactor: f * v^2 * (vesc dT) * (sin(eta) dEta)
    pre = (fv_s
           * (v_s ** 2 * vesc_s * wT[None, :])[:, :, None]
           * (sin_e * wH)[None, None, :])    # [Rs, V, H]
 
    q = Q_GRID
    A = (torch.sqrt((1.0 - q * q).clamp_min(0.0))[:, None, None]
         * cos_e[None, :, None]
         + q[:, None, None] * sin_e[None, :, None]
         * torch.cos(PHI_NODES)[None, None, :])          # [Nq, H, P]
 
    inv_norm = 1.0 / (e0 * math.sqrt(2.0 * math.pi))
    Rs, Nq = len(ridx), len(q)
 
    if CFG.proj_kernel == "dense":
        # -------- reference path (previous implementation) --------
        wphi2 = 2.0 * PHI_W
        G0 = torch.empty(Rs, Nq, Nw, dtype=TD, device=DEVICE)
        c = max(1, int(CFG.proj_r_chunk))
        for a in range(0, Rs, c):
            b = min(a + c, Rs)
            mu = v_s[a:b][:, :, None, None, None] * A[None, None]
            z = (W[None, None, None, None, None, :] - mu[..., None]) / e0
            ker = torch.exp(-0.5 * z * z) * inv_norm
            G0[a:b] = torch.einsum("cvh,p,cvqhpk->cqk",
                                   pre[a:b], wphi2, ker)
    else:
        # -------- windowed scatter path (default) --------
        # full sample tensors: [Rs, V, Nq, H, P] ~ 1.8M elements
        mu = v_s[:, :, None, None, None] * A[None, None]
        wgt = (pre[:, :, None, :, None]
               * (2.0 * PHI_W)[None, None, None, None, :])
 
        halfwin = max(2, int(math.ceil(5.0 * e0 / dW)) + 1)
        offs = torch.arange(-halfwin, halfwin + 1,
                            device=DEVICE)               # [D]
 
        with torch.no_grad():
            i0 = torch.round((mu - W[0]) / dW).long()    # [.,.,.,.,.]
        idx = i0[..., None] + offs                       # [..., D]
        ok = (idx >= 0) & (idx < Nw)
        idxc = idx.clamp(0, Nw - 1)
 
        # kernel values: differentiable via mu and wgt; indices fixed
        z = (W[idxc] - mu[..., None]) / e0
        val = wgt[..., None] * torch.exp(-0.5 * z * z) * inv_norm
        val = torch.where(ok, val,
                          torch.zeros((), dtype=TD, device=DEVICE))
 
        # scatter into flattened (r, q, w)
        r_ix = torch.arange(Rs, device=DEVICE)
        q_ix = torch.arange(Nq, device=DEVICE)
        flat = (r_ix[:, None, None, None, None, None] * (Nq * Nw)
                + q_ix[None, None, :, None, None, None] * Nw
                + idxc)
        G0 = torch.zeros(Rs * Nq * Nw, dtype=TD, device=DEVICE)
        G0 = G0.scatter_add(0, flat.reshape(-1), val.reshape(-1))
        G0 = G0.reshape(Rs, Nq, Nw)
 
    G = G0 + G0.flip(-1)                     # eta > pi/2 mirror half
    return torch.log(G.clamp_min(1.0e-30)), W

def compute_model_grids():
    """Velocity moments of f(eps,L) on the radial grid + DF normalization.

    nu(r)       = 2 pi int v^2 dv int_0^pi sin(eta) f deta
    <v_r^2> nu  = 2 pi int v^4 dv int sin(eta) cos^2(eta) f deta
    <v_t^2> nu  = 2 pi int v^4 dv int sin^3(eta) f deta
    N           = 4 pi int r^2 nu dr      (normalization over the model domain)
    """
    r = R_GRID
    psi = potential.psi(r).clamp_min(1.0e-8)
    vesc = torch.sqrt(2.0 * psi)

    v = vesc[:, None] * T_GRID[None, :]                       # [R,V]
    eps = psi[:, None] * (1.0 - T_GRID[None, :] ** 2)         # [R,V]

    sin_e = torch.sin(ETA_GRID)
    cos_e = torch.cos(ETA_GRID)

    L = r[:, None, None] * v[:, :, None] * sin_e[None, None, :]    # [R,V,H]
    eps_b = eps[:, :, None].expand(-1, -1, len(ETA_GRID))

    logf = df.log_f(eps_b.reshape(-1), L.reshape(-1)).reshape(L.shape)
    fv = torch.exp(logf.clamp(-60.0, 40.0))

    # Angular integrals.
    #
    # eta is the angle between velocity and radial direction:
    #   v_r = v cos eta
    #   v_t = v sin eta
    #
    # The tangential plane is azimuthally symmetric.

    I0 = 2.0 * trapz_last(fv * sin_e, ETA_GRID)                    # ∫ dΩ_v f

    Ir2 = 2.0 * trapz_last(fv * sin_e * cos_e ** 2, ETA_GRID)      # radial second
    It2 = 2.0 * trapz_last(fv * sin_e ** 3, ETA_GRID)              # total tangential second

    Ir4 = 2.0 * trapz_last(fv * sin_e * cos_e ** 4, ETA_GRID)      # radial fourth
    Ir2t2 = 2.0 * trapz_last(
        fv * sin_e * cos_e ** 2 * sin_e ** 2,
        ETA_GRID,
    )                                                             # radial^2 * total tangential^2
    It4 = 2.0 * trapz_last(fv * sin_e ** 5, ETA_GRID)              # total tangential fourth

    nu = 2.0 * math.pi * trapz_last(v ** 2 * I0, v)                # [R]

    m_r2 = 2.0 * math.pi * trapz_last(v ** 4 * Ir2, v)
    m_t2 = 2.0 * math.pi * trapz_last(v ** 4 * It2, v)

    # Fourth moments:
    #   m_r4     = nu <v_r^4>
    #   m_r2t2   = nu <v_r^2 v_t^2_total>
    #   m_t4     = nu <v_t_total^4>
    m_r4 = 2.0 * math.pi * trapz_last(v ** 6 * Ir4, v)
    m_r2t2 = 2.0 * math.pi * trapz_last(v ** 6 * Ir2t2, v)
    m_t4 = 2.0 * math.pi * trapz_last(v ** 6 * It4, v)

    nu_c = nu.clamp_min(1.0e-30)
    sig_r2 = (m_r2 / nu_c).clamp_min(1.0e-8)
    vt2 = m_t2 / nu_c
    beta = 1.0 - vt2 / (2.0 * sig_r2 + 1.0e-12)

    vr4 = (m_r4 / nu_c).clamp_min(1.0e-12)
    vr2vt2 = (m_r2t2 / nu_c).clamp_min(1.0e-12)
    vt4 = (m_t4 / nu_c).clamp_min(1.0e-12)

    Nnorm = 4.0 * math.pi * trapz_last(r**2 * nu, r)
    logN = torch.log(Nnorm.clamp_min(1.0e-30))

    out = {
        "r": r,
        "nu": nu,
        "log_nu": torch.log(nu_c),
        "sig_r2": sig_r2,
        "sig_t2": 0.5 * vt2,
        "beta": beta,
        "vr4": vr4,
        "vr2vt2": vr2vt2,
        "vt4": vt4,
        "logN": logN,
    }
    if CFG.training_mode == "projected":
        logG0, w_grid = _los_velocity_grid(fv, v)
        out["logG0"] = logG0
        out["w_grid"] = w_grid
        out["log_r_sub"] = torch.log(r[PROJ_R_IDX])
    return out

def model_sigma_los(R_t, grids):
    """Exact projection of the DF's own moments (Binney-Mamon):
       Sigma sig_los^2 (R) = 2 int (1 - beta R^2/r^2) nu sig_r^2 ds,  r^2 = R^2 + s^2
    """
    s = S_PROJ
    r_eff = torch.sqrt(R_t[:, None] ** 2 + s[None, :] ** 2)        # [B,S]
    lr = torch.log(r_eff.reshape(-1))
    log_rg = torch.log(grids["r"])

    nu = torch.exp(interp1d_t(lr, log_rg, grids["log_nu"])).reshape(r_eff.shape)
    sr2 = interp1d_t(lr, log_rg, grids["sig_r2"]).reshape(r_eff.shape)
    beta = interp1d_t(lr, log_rg, grids["beta"]).reshape(r_eff.shape)

    w = 1.0 - beta * (R_t[:, None] ** 2) / (r_eff ** 2)
    num = trapz_last(nu * sr2 * w, s)
    den = trapz_last(nu, s).clamp_min(1.0e-30)
    return torch.sqrt((num / den).clamp_min(1.0e-6))

def _anchor_ell(z):
    """Residual loss for anchor z-scores. See --anchor-loss."""
    if CFG.anchor_loss == "quad":
        return torch.mean(0.5 * z * z)
    if CFG.anchor_loss == "huber":
        a = 3.0
        az = torch.abs(z)
        return torch.mean(torch.where(az <= a,
                                      0.5 * z * z,
                                      a * (az - 0.5 * a)))
    return torch.mean(torch.log1p(z * z))

def beta_tether(grids):
    """Tracer-mass-weighted mismatch between the DF-implied
    anisotropy and the COM curve, inside the tether band.
 
    Uses grids['beta'] and grids['nu'] (already computed). Weight
    w(r) ~ r^3 nu(r) on the log-spaced grid concentrates the
    penalty where the tracers actually are; outside the band the
    DF's anisotropy is untouched (agnostic where data end)."""
    if CFG.beta_tether_weight <= 0.0:
        return torch.zeros((), device=DEVICE)
    r = grids["r"]
    lo = CFG.beta_tether_band[0] * R_HALF_2D
    hi = CFG.beta_tether_band[1] * R_HALF_2D
    m = (r >= lo) & (r <= hi)
    w = (r ** 3 * grids["nu"]).detach()
    w = torch.where(m, w, torch.zeros_like(w))
    w = w / w.sum().clamp_min(1e-30)
    d = grids["beta"] - beta_com(r)
    return torch.sum(w * d * d)

def sigma_los_loss(grids):
    model = model_sigma_los(SIG_R_T, grids)
    z = (model - SIG_T_T) / SIG_E_T
    return _anchor_ell(z)

def model_sigma_pm_RT(R_t, grids):
    """DF-consistent projected radial & tangential PM dispersions."""
    s = S_PROJ
    r_eff = torch.sqrt(R_t[:, None] ** 2 + s[None, :] ** 2)
    lr = torch.log(r_eff.reshape(-1)); log_rg = torch.log(grids["r"])
    nu  = torch.exp(interp1d_t(lr, log_rg, grids["log_nu"])).reshape(r_eff.shape)
    sr2 = interp1d_t(lr, log_rg, grids["sig_r2"]).reshape(r_eff.shape)
    st2 = interp1d_t(lr, log_rg, grids["sig_t2"]).reshape(r_eff.shape)
    sin2 = (R_t[:, None] ** 2) / (r_eff ** 2)
    cos2 = (s[None, :] ** 2) / (r_eff ** 2)
    pmR2 = sr2 * sin2 + st2 * cos2         # in-plane radial on sky
    pmT2 = st2                              # in-plane tangential on sky
    den = trapz_last(nu, s).clamp_min(1e-300)
    sR = torch.sqrt((trapz_last(nu * pmR2, s) / den).clamp_min(1e-6))
    sT = torch.sqrt((trapz_last(nu * pmT2, s) / den).clamp_min(1e-6))
    return sR, sT

def sigma_pm_loss(grids):
    if not CFG.use_pm_anchor:
        return torch.zeros((), dtype=TD, device=DEVICE)
    sR, sT = model_sigma_pm_RT(PM_R_T, grids)
    zR = (sR - PM_SR_T) / PM_ESR_T
    zT = (sT - PM_ST_T) / PM_EST_T
    return _anchor_ell(zR) + _anchor_ell(zT)

def model_los_fourth_moment(R_t, grids):
    """
    Projected LOS fourth moment <v_los^4>(R) from the DF's fourth moments.

    At radius r, define:
        mu = cos(angle between radial direction and LOS) = s / r
        q  = sin(angle) = R / r

    v_los = mu v_r + q v_theta

    Because the tangential plane is azimuthally symmetric:
        <v_theta^2> = 1/2 <v_t,total^2>
        <v_theta^4> = 3/8 <v_t,total^4>
        <v_r^2 v_theta^2> = 1/2 <v_r^2 v_t,total^2>

    Therefore:
        <v_los^4>
        = mu^4 <v_r^4>
        + 3 mu^2 q^2 <v_r^2 v_t,total^2>
        + (3/8) q^4 <v_t,total^4>

    Then project along the line of sight with nu(r).
    """
    s = S_PROJ
    r_eff = torch.sqrt(R_t[:, None] ** 2 + s[None, :] ** 2)

    lr = torch.log(r_eff.reshape(-1))
    log_rg = torch.log(grids["r"])

    nu = torch.exp(
        interp1d_t(lr, log_rg, grids["log_nu"])
    ).reshape(r_eff.shape)

    vr4 = interp1d_t(
        lr, log_rg, grids["vr4"]
    ).reshape(r_eff.shape)

    vr2vt2 = interp1d_t(
        lr, log_rg, grids["vr2vt2"]
    ).reshape(r_eff.shape)

    vt4 = interp1d_t(
        lr, log_rg, grids["vt4"]
    ).reshape(r_eff.shape)

    mu2 = (s[None, :] ** 2) / (r_eff ** 2)
    q2 = (R_t[:, None] ** 2) / (r_eff ** 2)

    los4_local = (
        mu2 ** 2 * vr4
        + 3.0 * mu2 * q2 * vr2vt2
        + 0.375 * q2 ** 2 * vt4
    )

    num = trapz_last(nu * los4_local, s)
    den = trapz_last(nu, s).clamp_min(1.0e-30)

    return (num / den).clamp_min(1.0e-12)

def model_vsp(grids):
    """
    Model-side VSP-like quantities using the same estimator as the data:

        VSP1 = mean( <v_los^4>(R_i) * R_i )
        VSP2 = mean( <v_los^4>(R_i) * R_i^3 )

    where R_i are the observed projected radii.

    This avoids needing to explicitly normalize Sigma(R), and matches the
    observed-sample estimator used in build_vsp_targets().
    """
    los4 = model_los_fourth_moment(VSP_R_T, grids)

    p1, p2 = VSP_POWERS
    vsp1 = torch.mean(los4 * VSP_R_T ** p1)
    vsp2 = torch.mean(los4 * VSP_R_T ** p2)

    return torch.stack([vsp1, vsp2])

def vsp_loss(grids):
    if not CFG.use_vsp_anchor:
        return torch.zeros((), dtype=TD, device=DEVICE)

    model = model_vsp(grids)
    z = (model - VSP_OBS_T) / VSP_ERR_T

    return _anchor_ell(z)

# ============================================================
# 8. AMORTIZED ENCODER + IWAE MARGINAL LIKELIHOOD
# ============================================================
class Encoder(nn.Module):
    """Per-star proposal q(s | obs, err): diagonal Gaussian in shift units,
    z = obs + s * err. Final layer zero-init -> starts at N(0, I)."""

    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 12),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats):
        out = self.net(feats)
        mu, logsig = out[:, :6], out[:, 6:]
        return mu, logsig.clamp(-3.0, 2.0)


def build_encoder_features():
    o = obs_mes.cpu().numpy()
    e = err_mes.cpu().numpy()
    feats = np.concatenate([o, np.log10(np.maximum(e, 1e-12))], axis=1)
    med = np.median(feats, axis=0)
    mad = np.median(np.abs(feats - med), axis=0) * 1.4826 + 1e-9
    return torch.tensor((feats - med) / mad, dtype=TD, device=DEVICE)


ENC_FEATS = build_encoder_features()
encoder = Encoder(ENC_FEATS.shape[1], hidden=CFG.enc_hidden).to(DEVICE)

# Jacobian constants: |d(x,v)/d(z)| = kappa^2 D^4 cos(Dec) (pi/180)^2
LOGJ_CONST = 2.0 * math.log(DEG2RAD) + 2.0 * math.log(CFG.KAPPA)


def phase_space_logdensity(z, logN):
    """log p_z(z) for z = (RA,Dec,Dist,pmRA,pmDec,vlos):
       log f(eps, L) + log|Jacobian| - log N(theta)."""
    cols = [z[:, 0], z[:, 1], z[:, 2].clamp_min(1.0), z[:, 3], z[:, 4], z[:, 5]]
    zs = torch.stack(cols, dim=1)

    pos, vel = obs_to_galactocentric_torch(zs)
    r = torch.linalg.norm(pos, dim=1).clamp_min(1.0e-8)
    L = torch.linalg.norm(torch.cross(pos, vel, dim=1), dim=1)
    eps = potential.psi(r) - 0.5 * torch.sum(vel * vel, dim=1)

    logf = df.log_f(eps, L)
    logJ = (4.0 * torch.log(zs[:, 2])
            + torch.log(torch.cos(torch.deg2rad(zs[:, 1])).clamp_min(1.0e-6))
            + LOGJ_CONST)
    return logf + logJ - logN


def iwae_nll_mc(idx, grids, K):
    """-log p(obs_i) via importance sampling with the amortized proposal.
    Gradients flow into DF, potential (through eps), and encoder."""
    B = len(idx)
    mu, logsig = encoder(ENC_FEATS[idx])
    sig = torch.exp(logsig)

    noise = torch.randn(B, K, 6, dtype=TD, device=DEVICE)
    s = mu[:, None, :] + sig[:, None, :] * noise
    z = obs_mes[idx][:, None, :] + s * err_mes[idx][:, None, :]

    zf = z.reshape(B * K, 6)
    valid = zf[:, 2] > 1.0

    logp = phase_space_logdensity(zf, grids["logN"])
    logp = torch.where(valid, logp, torch.full_like(logp, -1.0e9))
    logp = logp.reshape(B, K)

    # noise likelihood x latent measure, all in shift units (err factors cancel)
    log_prior = -0.5 * torch.sum(s * s, dim=-1) - 3.0 * LOG2PI
    log_q = torch.sum(-0.5 * noise**2 - logsig[:, None, :] - 0.5 * LOG2PI, dim=-1)

    logw = torch.nan_to_num(logp + log_prior - log_q, nan=-1.0e9,
                            posinf=-1.0e9, neginf=-1.0e9)
    lme = torch.logsumexp(logw, dim=1) - math.log(K)
    nll = -torch.mean(lme)

    with torch.no_grad():
        w = torch.softmax(logw, dim=1)
        ess = (1.0 / torch.sum(w * w, dim=1)).mean()
    return nll, float(ess)

def iwae_nll_dquad(idx, grids, K):
    """Hybrid marginal: MC (K samples) over the 5 well-measured coords,
    exact trapezoid quadrature over the latent distance."""
    B = len(idx)
    mu, logsig = encoder(ENC_FEATS[idx])
    mu5, ls5 = mu[:, IDX5], logsig[:, IDX5]
    sig5 = torch.exp(ls5)

    noise = torch.randn(B, K, 5, dtype=TD, device=DEVICE)
    s5 = mu5[:, None, :] + sig5[:, None, :] * noise
    z5 = obs_mes[idx][:, None, :][:, :, IDX5] + s5 * err_mes[idx][:, None, :][:, :, IDX5]

    nD = len(D_NODES)
    z = torch.empty(B, K, nD, 6, dtype=TD, device=DEVICE)
    z[..., 0] = z5[..., 0][..., None];  z[..., 1] = z5[..., 1][..., None]
    z[..., 3] = z5[..., 2][..., None];  z[..., 4] = z5[..., 3][..., None]
    z[..., 5] = z5[..., 4][..., None]
    z[..., 2] = D_NODES[None, None, :]

    logp = phase_space_logdensity(z.reshape(-1, 6),
                                  grids["logN"]).reshape(B, K, nD)

    Dobs, eD = obs_mes[idx][:, 2], err_mes[idx][:, 2]
    log_nD = (-0.5 * ((D_NODES[None, :] - Dobs[:, None]) / eD[:, None]) ** 2
              - torch.log(eD)[:, None] - 0.5 * LOG2PI)          # [B, nD]

    # quadrature over D (uniform grid; endpoint half-weights are negligible
    # because the tracer density vanishes at the grid edges)
    logw_D = torch.logsumexp(logp + log_nD[:, None, :] + D_LOGW[None, None, :], dim=2)

    log_prior5 = -0.5 * torch.sum(s5 * s5, dim=-1) - 2.5 * LOG2PI
    log_q5 = torch.sum(-0.5 * noise ** 2 - ls5[:, None, :] - 0.5 * LOG2PI, dim=-1)

    logw = torch.nan_to_num(logw_D + log_prior5 - log_q5,
                            nan=-1e9, posinf=-1e9, neginf=-1e9)
    lme = torch.logsumexp(logw, dim=1) - math.log(K)
    nll = -torch.mean(lme)
    with torch.no_grad():
        w = torch.softmax(logw, dim=1)
        ess = (1.0 / torch.sum(w * w, dim=1)).mean()
    return nll, float(ess)


def iwae_nll(idx, grids, K):
    if CFG.marginalize_distance:
        return iwae_nll_dquad(idx, grids, K)
    return iwae_nll_mc(idx, grids, K)


def truth_nll(idx):
    pos = truth_pos[idx]
    vel = truth_vel[idx]
    r = torch.linalg.norm(pos, dim=1).clamp_min(1.0e-8)
    L = torch.linalg.norm(torch.cross(pos, vel, dim=1), dim=1)
    eps = potential.psi(r) - 0.5 * torch.sum(vel * vel, dim=1)
    grids = compute_model_grids()
    logp = df.log_f(eps, L) - grids["logN"]
    return -torch.mean(logp), grids

def _interp_gather_uniform(y, xq, x0, dx):
    """Linear interpolation on a UNIFORM grid along the last dim.
    y: [..., K] values; xq: [..., J] queries with the same leading
    dims. Queries are clamped to the grid range."""
    K = y.shape[-1]
    t = ((xq - x0) / dx).clamp(0.0, K - 1.000001)
    i0 = t.floor().long()
    fr = t - i0.to(t.dtype)
    y0 = torch.gather(y, -1, i0)
    y1 = torch.gather(y, -1, i0 + 1)
    return y0 + fr * (y1 - y0)
 
 
def _interp_logG0(logG0, log_rg, lr, qq):
    """Bilinear interpolation of log g(w|r,q) in (log r, q).
    logG0: [Rs, Nq, Nw]; lr, qq: [B, S] -> returns [B, S, Nw]."""
    lrc = lr.clamp(log_rg[0], log_rg[-1])
    ir = torch.bucketize(lrc, log_rg).clamp(1, len(log_rg) - 1)
    r0, r1 = log_rg[ir - 1], log_rg[ir]
    tr = ((lrc - r0) / (r1 - r0 + 1e-30)).clamp(0.0, 1.0)[..., None]
 
    qc = qq.clamp(float(Q_GRID[0]), float(Q_GRID[-1]))
    iq = torch.bucketize(qc, Q_GRID).clamp(1, len(Q_GRID) - 1)
    q0, q1 = Q_GRID[iq - 1], Q_GRID[iq]
    tq = ((qc - q0) / (q1 - q0 + 1e-30)).clamp(0.0, 1.0)[..., None]
 
    g00 = logG0[ir - 1, iq - 1]
    g01 = logG0[ir - 1, iq]
    g10 = logG0[ir, iq - 1]
    g11 = logG0[ir, iq]
    return ((1.0 - tr) * ((1.0 - tq) * g00 + tq * g01)
            + tr * ((1.0 - tq) * g10 + tq * g11))
 
 
def projected_nll(idx, grids):
    """Deterministic per-star NLL on (x, y, v_los):
 
        p_i = [ 2 int_0^inf ds  (g * N(0, e_eff_i^2))(w_i; r(s), q(s)) ]
              / N(theta)
 
    The baseline smoothing e0 is already inside g; the per-star
    Gauss-Hermite step adds e_eff = sqrt(e_i^2 - e0^2), so the total
    kernel width is exactly e_i. Position errors (mas-level) are
    negligible at 80 kpc and ignored. Fully differentiable in the DF
    and the potential; zero Monte Carlo variance."""
    R = PROJ["R"][idx]
    w = PROJ["w"][idx]
    e = PROJ["e"][idx]
    s, s_w = PROJ["s"], PROJ["s_w"]
 
    r_s = torch.sqrt(R[:, None] ** 2 + s[None, :] ** 2).clamp_min(1e-6)
    q_s = (R[:, None] / r_s).clamp(0.0, 1.0)
 
    logG = _interp_logG0(grids["logG0"], grids["log_r_sub"],
                         torch.log(r_s), q_s)            # [B, S, K]
    G = torch.exp(logG)
    H = 2.0 * torch.sum(s_w[None, :, None] * G, dim=1)   # [B, K]
 
    e_eff = torch.sqrt((e * e - PROJ["e0"] ** 2).clamp_min(0.0))
    wq = w[:, None] + math.sqrt(2.0) * e_eff[:, None] * PROJ["gh_x"][None, :]
 
    W = grids["w_grid"]
    Hq = _interp_gather_uniform(H, wq, W[0], W[1] - W[0])
    p = torch.sum(PROJ["gh_w"][None, :] * Hq, dim=1) / math.sqrt(math.pi)
 
    logp = torch.log(p.clamp_min(1.0e-30)) - grids["logN"]
    return -torch.mean(logp)
 
 
def eval_data_nll(idx, grids, K):
    """Unified per-star data NLL for observed/projected modes.
    Returns (nll, ess); ess is NaN in projected mode (deterministic)."""
    if CFG.training_mode == "projected":
        return projected_nll(idx, grids), float("nan")
    return iwae_nll(idx, grids, K)
 
 
@torch.no_grad()
def check_projected_normalization():
    """Self-test: int g(w|r,q) dw must equal nu(r) for EVERY q
    (the count normalization cannot depend on the viewing angle).
    Run once before training; ratios should be ~1 to grid accuracy."""
    grids = compute_model_grids()
    W = grids["w_grid"]
    wW = _trapz_w(W)
    G = torch.exp(grids["logG0"])
    gint = torch.sum(G * wW[None, None, :], dim=-1)          # [Rs, Nq]
    nu = torch.exp(interp1d_t(grids["log_r_sub"],
                              torch.log(grids["r"]),
                              grids["log_nu"]))              # [Rs]
    ratio = (gint / nu[:, None].clamp_min(1e-300)).cpu().numpy()
    interior = ratio[2:-2]
    print(f"[projected] normalization check int g dw / nu: "
          f"median={np.median(interior):.4f} "
          f"min={interior.min():.4f} max={interior.max():.4f} "
          f"(should be ~1; tighten proj-n-w / proj-n-phi if not)")

def check_kernel_equivalence():
    """Compare window vs dense kernels on the SAME model state.
    Max |delta log g| should be ~1e-5 (float32 accumulation order);
    anything > 1e-3 in a region with non-negligible g is a bug."""
    if CFG.proj_kernel != "window":
        return
    grids_w = compute_model_grids()
    CFG.proj_kernel = "dense"
    grids_d = compute_model_grids()
    CFG.proj_kernel = "window"
 
    lg_w, lg_d = grids_w["logG0"], grids_d["logG0"]
    # compare only where g is non-negligible (above the 1e-30 floor)
    mask = lg_d > math.log(1.0e-20)
    diff = (lg_w - lg_d).abs()
    print(f"[projected] kernel equivalence: "
          f"max|dlog g|={float(diff[mask].max()):.2e} "
          f"median={float(diff[mask].median()):.2e} "
          f"(coverage {float(mask.float().mean()):.2f}) "
          f"-> window path validated" if float(diff[mask].max()) < 1e-3
          else f"[projected] WARNING kernel mismatch "
               f"max|dlog g|={float(diff[mask].max()):.2e} -- "
               f"use --proj-kernel dense and report")


@torch.no_grad()
def check_los_moment_consistency(n_R=7):
    """Compare LOS velocity moments from the projected kernel against
    the Binney-Mamon moment projection of the same DF+potential.
 
    Second moment:
        sigma_kernel^2(R) = int w^2 H dw / int H dw  -  e0^2
        (subtracting e0^2 because H carries the baseline smoothing)
    must equal model_sigma_los(R)^2.
 
    Fourth moment:
        <w^4>_kernel(R) = int w^4 H dw / int H dw
                          - 6 e0^2 sigma^2 - 3 e0^4
    must equal model_los_fourth_moment(R). This one matters more:
    the cusp/core shape information lives in the fourth moment, and
    the VSP anchor changed sides between DF configurations, which is
    what a distorted fourth moment would do.
 
    Ratios should be 1 to a few parts in 1e-2. Anything at the 10%
    level in the tracer band is the bug.
    """
    grids = compute_model_grids()
    W = grids["w_grid"]
    wW = _trapz_w(W)
    s, s_w = PROJ["s"], PROJ["s_w"]
    e0 = PROJ["e0"]
 
    qs = np.linspace(10.0, 90.0, n_R)
    R_test = torch.tensor(np.percentile(R_proj_np, qs),
                          dtype=TD, device=DEVICE)
 
    r_s = torch.sqrt(R_test[:, None] ** 2 + s[None, :] ** 2).clamp_min(1e-6)
    q_s = (R_test[:, None] / r_s).clamp(0.0, 1.0)
    Gk = torch.exp(_interp_logG0(grids["logG0"], grids["log_r_sub"],
                                 torch.log(r_s), q_s))
    H = 2.0 * torch.sum(s_w[None, :, None] * Gk, dim=1)        # [B, K]
 
    norm = torch.sum(H * wW[None, :], dim=1).clamp_min(1e-300)
    m2 = torch.sum(H * (W ** 2)[None, :] * wW[None, :], dim=1) / norm
    m4 = torch.sum(H * (W ** 4)[None, :] * wW[None, :], dim=1) / norm
 
    sig_k = torch.sqrt((m2 - e0 ** 2).clamp_min(0.0))
    los4_k = (m4 - 6.0 * e0 ** 2 * m2 + 3.0 * e0 ** 4).clamp_min(0.0)
 
    sig_m = model_sigma_los(R_test, grids)
    los4_m = model_los_fourth_moment(R_test, grids)
 
    print("\n[check] LOS moments: projected kernel vs moment projection")
    print(f"   {'R [kpc]':>9s} {'sig_ker':>9s} {'sig_mom':>9s} {'ratio':>7s}"
          f" | {'v4_ker':>10s} {'v4_mom':>10s} {'ratio':>7s}")
    bad2 = bad4 = 0.0
    for i in range(len(R_test)):
        a2, b2 = float(sig_k[i]), float(sig_m[i])
        a4, b4 = float(los4_k[i]), float(los4_m[i])
        r2 = a2 / max(b2, 1e-12)
        r4 = a4 / max(b4, 1e-12)
        bad2 = max(bad2, abs(r2 - 1.0))
        bad4 = max(bad4, abs(r4 - 1.0))
        print(f"   {float(R_test[i]):9.4f} {a2:9.4f} {b2:9.4f} {r2:7.4f}"
              f" | {a4:10.3e} {b4:10.3e} {r4:7.4f}")
 
    print(f"   max |ratio-1|:  second moment {bad2:.4f}   "
          f"fourth moment {bad4:.4f}")
    if max(bad2, bad4) < 0.02:
        print("   -> CONSISTENT. The kernel reproduces the validated moment "
              "machinery; the misranking is not a kernel-shape bug.")
    else:
        print("   -> INCONSISTENT. The projected kernel does not reproduce "
              "the model's own velocity moments. Every projected NLL is "
              "computed with a distorted velocity distribution.")
        print("      Suspects, in order of likely severity:")
        print("       (i)  Q_GRID: 10 LINEAR nodes in q = R/r. The radial "
              "projection weight is sqrt(1-q^2), which collapses over the "
              "last interval (q=0.889 -> 1.0). Linear interpolation there "
              "errs by ~30%, and q->1 is the s->0 region that carries the "
              "MOST tracer weight in the LOS integral. Try q nodes spaced "
              "uniformly in sqrt(1-q^2), or simply raise --proj-n-q.")
        print("       (ii) PHI_NODES: 6 Gauss-Legendre nodes turn the "
              "arcsine-shaped tangential projection (singular at the edges) "
              "into 6 spikes smoothed by e0=0.15 km/s. Integrates to the "
              "right counts, wrong shape -> distorts the fourth moment "
              "most, which is exactly where the cusp/core signal lives.")
        print("       (iii) w-grid: wmax is set by vesc at r_min, so the "
              "spacing is ~1.4 km/s while sigma_los ~ 5 km/s. Linear "
              "interpolation of a concave H underestimates it, and the "
              "size of the error depends on the model -> a model-dependent "
              "NLL offset.")
        
# ============================================================
# 9. TRAINING
# ============================================================
opt_df = torch.optim.Adam(
        list(df.parameters()) + list(encoder.parameters())
        + list(beta_com.parameters()), lr=CFG.lr_df)
opt_pot = torch.optim.Adam(potential.parameters(), lr=CFG.lr_pot)

indices_all = torch.arange(N, device=DEVICE)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(indices_all),
    batch_size=CFG.batch_size, shuffle=True, drop_last=False)
L_batches = len(loader)

def strong_prior_scale():
    """Multiplier that puts strong_prior in the intended units.
 
    legacy    : S / L_batches   -> effectively S * batch_size in
                                   NLL units (reproduces published runs)
    nll_units : S / N           -> exactly S in NLL units
    """
    if CFG.strong_prior_norm == "nll_units":
        return CFG.strong_prior_weight / N
    return CFG.strong_prior_weight / L_batches

history = {
    "phaseA": {
        "nll": [],"sig": [],"pm": [],"vsp": [],"ess": [],
    },
    "phaseB": {
        "nll": [],"sig": [],"pm": [],"vsp": [],"ess": [],"prior": [],"M_at_Rs_true": [],"p1": [],"p2": [],"p3": [],
    },
}


def track_params():
    d = potential.describe()
    with torch.no_grad():
        Mtr = float(potential.mass(torch.tensor([CFG.R_S_NFW_TRUE],
                                                dtype=TD, device=DEVICE))[0])
    vals = list(d.values()) + [np.nan] * 3
    return Mtr, vals[0], vals[1], vals[2]


def run_epoch(joint: bool):
    ep_nll = 0.0
    ep_sig = 0.0
    ep_pm = 0.0
    ep_vsp = 0.0
    ep_ess = 0.0
    ep_prior = 0.0
    for (idx,) in loader:
        opt_df.zero_grad(set_to_none=True)
        if joint:
            opt_pot.zero_grad(set_to_none=True)

        if CFG.training_mode == "truth":
            nll, grids = truth_nll(idx)
            ess = float("nan")
        else:
            grids = compute_model_grids()
            nll, ess = eval_data_nll(idx, grids, CFG.n_iwae)

        sig_l = sigma_los_loss(grids)
        pm_l = sigma_pm_loss(grids)
        vsp_l = vsp_loss(grids)
        prior = potential.prior_loss()

        inv_N = 1.0 / N
        loss = (
            nll
            + inv_N * CFG.sigma_los_weight * sig_l
            + inv_N * CFG.sigma_pm_weight * pm_l
            + inv_N * CFG.vsp_weight * vsp_l
        )

        if joint:
            loss = loss + inv_N * CFG.pot_prior_weight * prior
            if hasattr(potential, "strong_prior"):
                # full-strength prior, amortized over batches so one epoch
                # applies it exactly once relative to the FULL likelihood
                loss = loss + strong_prior_scale() * potential.strong_prior()
            if CFG.beta_tether_weight > 0.0:
                loss = loss + (CFG.beta_tether_weight / L_batches) \
                            * beta_tether(grids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(df.parameters()) + list(encoder.parameters()), CFG.grad_clip_df)
        opt_df.step()
        if joint:
            torch.nn.utils.clip_grad_norm_(potential.parameters(), CFG.grad_clip_pot)
            opt_pot.step()

        ep_nll += float(nll.detach())
        ep_sig += float(sig_l.detach())
        ep_prior += float(prior.detach())
        ep_pm += float(pm_l.detach())
        ep_vsp += float(vsp_l.detach())
        ep_ess += 0.0 if math.isnan(ess) else ess
    return (
        ep_nll / L_batches, ep_sig / L_batches, ep_pm / L_batches, ep_vsp / L_batches, ep_ess / L_batches, ep_prior / L_batches,
    )

if CFG.training_mode == "projected":
    check_projected_normalization()
    check_kernel_equivalence()
    check_los_moment_consistency()

print("\n=== PHASE A: DF + encoder warmup (potential frozen) ===")
for p in potential.parameters():
    p.requires_grad_(False)

t0 = time.time()
for epoch in range(CFG.epochs_warmup):
    nll, sig_l, pm_l, vsp_l, ess, _ = run_epoch(joint=False)

    history["phaseA"]["nll"].append(nll)
    history["phaseA"]["sig"].append(sig_l)
    history["phaseA"]["pm"].append(pm_l)
    history["phaseA"]["vsp"].append(vsp_l)
    history["phaseA"]["ess"].append(ess)
    if (epoch + 1) % 25 == 0 or epoch == 0:
        print(f"Phase A | {epoch+1:04d}/{CFG.epochs_warmup} | "
              f"marginal NLL={nll:.4f} | "
              f"sigma_los={sig_l:.3e} | "
              f"sigma_pm={pm_l:.3e} | "
              f"vsp={vsp_l:.3e} | "
              f"ESS={ess:.1f}/{CFG.n_iwae}")
print(f"Phase A done in {(time.time()-t0)/60:.2f} min")

print("\n=== PHASE B: JOINT DF + encoder + potential (marginal likelihood) ===")
for p in potential.parameters():
    p.requires_grad_(True)

sched_pot = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt_pot, T_max=CFG.epochs_joint, eta_min=CFG.lr_pot * 0.03)

ema = {k: v.detach().clone() for k, v in potential.state_dict().items()}
EMA_DECAY, EMA_START = 0.995, int(0.5 * CFG.epochs_joint)

t0 = time.time()
for epoch in range(CFG.epochs_joint):
    nll, sig_l, pm_l, vsp_l, ess, prior = run_epoch(joint=True)
    sched_pot.step()
    if epoch >= EMA_START:
        with torch.no_grad():
            for k, v in potential.state_dict().items():
                ema[k].mul_(EMA_DECAY).add_(v.detach(), alpha=1 - EMA_DECAY)
    Mtr, p1, p2, p3 = track_params()
    history["phaseB"]["nll"].append(nll)
    history["phaseB"]["sig"].append(sig_l)
    history["phaseB"]["pm"].append(pm_l)
    history["phaseB"]["vsp"].append(vsp_l)
    history["phaseB"]["ess"].append(ess)
    history["phaseB"]["prior"].append(prior)
    history["phaseB"]["M_at_Rs_true"].append(Mtr)
    history["phaseB"]["p1"].append(p1)
    history["phaseB"]["p2"].append(p2)
    history["phaseB"]["p3"].append(p3)
    if (epoch + 1) % 25 == 0 or epoch == 0:
        tot = (nll + (CFG.sigma_los_weight * sig_l + CFG.vsp_weight * vsp_l) / N + strong_prior_scale() * float(potential.strong_prior()))
        print(f"... | TOTAL={tot:.6f}")
        print(f"Phase B | {epoch+1:04d}/{CFG.epochs_joint} | "
              f"NLL={nll:.4f} | "
              f"sigma_los={sig_l:.3e} | "
              f"sigma_pm={pm_l:.3e} | "
              f"vsp={vsp_l:.3e} | "
              f"ESS={ess:.1f}/{CFG.n_iwae} | "
              f"pot={potential.describe()}")
print(f"Phase B done in {(time.time()-t0)/60:.2f} min")

# Snapshot the joint-MAP estimator BEFORE Laplace/scan can perturb globals.
MAP_STATE = {
    "df": {k: v.clone() for k, v in df.state_dict().items()},
    "encoder": {k: v.clone() for k, v in encoder.state_dict().items()},
    "potential": {k: v.clone() for k, v in potential.state_dict().items()},
}


# ============================================================
# 10. LAPLACE UNCERTAINTIES ON POTENTIAL PARAMETERS
# ============================================================
def laplace_potential(h=0.02):
    """Finite-difference Hessian of the full-data objective w.r.t. the
    potential parameters, DF and encoder frozen, fixed RNG seed so the
    IWAE estimate is deterministic. Profile-likelihood approximation
    (ignores DF-parameter covariance) -- stated as such in the summary."""
    params = [p for p in potential.parameters()]
    p0 = [float(p.detach()) for p in params]
    n_p = len(params)

    def objective():
        torch.manual_seed(CFG.seed + 777)
        with torch.no_grad():
            pass
        grids = compute_model_grids()
        total, nb = 0.0, 0
        for start in range(0, N, 500):
            idx = indices_all[start:start + 500]
            if CFG.training_mode == "truth":
                pos, vel = truth_pos[idx], truth_vel[idx]
                r = torch.linalg.norm(pos, dim=1).clamp_min(1e-8)
                Lm = torch.linalg.norm(torch.cross(pos, vel, dim=1), dim=1)
                eps = potential.psi(r) - 0.5 * torch.sum(vel * vel, dim=1)
                lme = df.log_f(eps, Lm) - grids["logN"]
                nll = -torch.mean(lme)
            else:
                nll, _ = eval_data_nll(idx, grids, max(CFG.n_iwae, 64))
            total += float(nll.detach()) * len(idx)
            nb += len(idx)
        nll_mean = total / nb
        sig_l = float(sigma_los_loss(grids).detach())
        pm_l = float(sigma_pm_loss(grids).detach())
        vsp_l = float(vsp_loss(grids).detach())
        # per-star average objective -> multiply by N for the information matrix
        return (
            nll_mean * N
            + CFG.sigma_los_weight * sig_l
            + CFG.sigma_pm_weight * pm_l
            + CFG.vsp_weight * vsp_l
        )

    def set_params(vals):
        with torch.no_grad():
            for p, v in zip(params, vals):
                p.copy_(torch.tensor(v, dtype=TD, device=p.device))

    H = np.zeros((n_p, n_p))
    f0 = None
    with torch.no_grad():
        f0 = objective()
    for i in range(n_p):
        for j in range(i, n_p):
            if i == j:
                vp, vm = list(p0), list(p0)
                vp[i] += h
                vm[i] -= h
                set_params(vp); fp = objective()
                set_params(vm); fm = objective()
                H[i, i] = (fp - 2.0 * f0 + fm) / h**2
            else:
                vpp, vpm, vmp, vmm = list(p0), list(p0), list(p0), list(p0)
                vpp[i] += h; vpp[j] += h
                vpm[i] += h; vpm[j] -= h
                vmp[i] -= h; vmp[j] += h
                vmm[i] -= h; vmm[j] -= h
                set_params(vpp); fpp = objective()
                set_params(vpm); fpm = objective()
                set_params(vmp); fmp = objective()
                set_params(vmm); fmm = objective()
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * h**2)
    set_params(p0)

    try:
        cov = np.linalg.pinv(H)
        sig = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        cov = np.full((n_p, n_p), np.nan)
        sig = np.full(n_p, np.nan)
    names = [n for n, _ in potential.named_parameters()]
    return names, p0, sig, cov

laplace = None

if CFG.potential_model == "free_mass" and not CFG.skip_laplace:
    print(
        "\nLaplace skipped automatically for free_mass: "
        "too many correlated potential parameters. "
        "Use bootstrap or posterior sampling for reliable uncertainties."
    )

elif not CFG.skip_laplace:
    print("\nComputing Laplace uncertainties on potential parameters...")
    try:
        lp_names, lp_vals, lp_sig, lp_cov = laplace_potential()
        laplace = {"names": lp_names,
                   "values": [float(v) for v in lp_vals],
                   "sigma": [float(s) for s in lp_sig],
                   "cov": lp_cov.tolist(),
                   "note": ("Profile-likelihood Laplace at the joint optimum; "
                            "DF/encoder parameters frozen.")}
        for n_, v_, s_ in zip(lp_names, lp_vals, lp_sig):
            print(f"  {n_} = {v_:.4f} +/- {s_:.4f}")
    except Exception as e:
        print(f"  Laplace failed: {e}")

slope_true_R12 = float(np_dlnrho_dlnr_truth(np.array([R_HALF_2D]))[0])

# ============================================================
# 10b. PROFILE-LIKELIHOOD SCAN OVER THE INNER SLOPE  (v2)
#   n=0 <-> cusp (gamma=-1); n=1 <-> core (gamma=0); LOWER objective = better.
#   - bidirectional continuation from the joint MAP (equal DF adaptation)
#   - pure-NLL profile recorded separately (calibratable log-L units) from
#     the composite (arbitrary anchor weights, optimization only)
#   - tail-min over the last quarter of epochs (robust to Adam jitter)
#   - RESTORES the joint-MAP state afterwards so diagnostics are honest
# ============================================================
SCAN_SUMMARY = None
SCAN_RS_CHOICE = None
SCAN_TABLE = None
SCAN_NODE_STATES = None
SCAN_BEST_N = None

def node_objectives():
    """Deterministic full-data objective at current params, decomposed."""
    torch.manual_seed(CFG.seed + 4242)
    with torch.no_grad():
        grids = compute_model_grids()
        total, nb = 0.0, 0
        for start in range(0, N, 500):
            idx = indices_all[start:start + 500]
            if CFG.training_mode == "truth":
                pos, vel = truth_pos[idx], truth_vel[idx]
                rr = torch.linalg.norm(pos, dim=1).clamp_min(1e-8)
                Lm = torch.linalg.norm(torch.cross(pos, vel, dim=1), dim=1)
                eps = potential.psi(rr) - 0.5 * torch.sum(vel * vel, dim=1)
                nll = -torch.mean(df.log_f(eps, Lm) - grids["logN"])
            else:
                nll, _ = eval_data_nll(idx, grids, max(CFG.n_iwae, 64))
            total += float(nll) * len(idx); nb += len(idx)
        nll_total = (total / nb) * N
        sig = float(sigma_los_loss(grids))
        pm = float(sigma_pm_loss(grids))
        vsp = float(vsp_loss(grids))
        comp = (nll_total + CFG.sigma_los_weight * sig
                + CFG.sigma_pm_weight * pm + CFG.vsp_weight * vsp)
        if hasattr(potential, "strong_prior"):
            comp += CFG.strong_prior_weight * float(potential.strong_prior())
    return {"nll": nll_total, "sig": sig, "pm": pm, "vsp": vsp, "L": comp}


_SCAN_FREEZE_RS = False


def scan_resolution(n, nll):
    """Does the scan determine its own answer, from the curve alone?

    The reported argmin says nothing about whether the data ranked the nodes or
    the optimiser's own noise did. Estimate the per-node jitter from the curve's
    second differences -- smooth curves have small ones -- with a robust scale so
    a single failed node cannot set it. Every node within 2 x jitter of the
    minimum is indistinguishable from the winner; `span` is the width in gamma_0
    they cover, to be read against the grid step.
    """
    y = np.asarray(nll, dtype=float)
    y = y - y.min()
    n = np.asarray(n, dtype=float)
    if len(y) < 5 or not np.all(np.isfinite(y)):
        return dict(span=float("nan"), depth=float("nan"),
                    jitter=float("nan"), n_plausible=-1, grid_step=float("nan"))
    d2 = np.diff(y, 2)
    jitter = max(1.4826 * float(np.median(np.abs(d2 - np.median(d2)))) / 2.0,
                 1e-9)
    ok = y <= 2.0 * jitter
    return dict(span=float(n[ok].max() - n[ok].min()), depth=float(y.max()),
                jitter=float(jitter), n_plausible=int(ok.sum()),
                grid_step=float(np.median(np.diff(np.sort(n)))))


def _optimize_node(n_fix, df_state, enc_state, pot_state, epochs):
    """Re-optimize with n frozen, warm-started. First 20% DF-only (potential
    frozen) so the DF adapts to this node's potential before (Rs,rc,Mr3) move.
    Returns (best_objectives, best_states) using tail-min checkpointing."""
    global potential
    pot_n = make_corenfw(
            fix_n=n_fix,
            M_anchor_init=float(torch.exp(pot_state["log_Mr3"])),
        )
    with torch.no_grad():
        pot_n.log_Mr3.copy_(pot_state["log_Mr3"])
        pot_n.log_Rs.copy_(pot_state["log_Rs"])
        pot_n.raw_rc.copy_(pot_state["raw_rc"])
    df.load_state_dict(df_state)
    encoder.load_state_dict(enc_state)
    potential = pot_n

    opt_df_s = torch.optim.Adam(list(df.parameters()) + list(encoder.parameters()),
                                lr=CFG.lr_df * 0.5)
    if _SCAN_FREEZE_RS:
        # Hold the scale radius exactly, by keeping it out of the optimiser.
        # A stiff prior is not equivalent: at weight 500 R_s still drifted 5-15
        # per cent across the sweep, which is enough to absorb part of n.
        pot_params = [q for nm, q in potential.named_parameters()
                      if q.requires_grad and nm != "log_Rs"]
    else:
        pot_params = [p for p in potential.parameters() if p.requires_grad]
    opt_pot_s = torch.optim.Adam(pot_params, lr=CFG.lr_pot * 0.5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_pot_s, T_max=epochs, eta_min=CFG.lr_pot * 0.02)

    n_warm = max(1, int(0.2 * epochs))
    tail_start = epochs - max(10, epochs // 4)
    best, best_states = None, None

    for ep in range(epochs):
        joint_now = ep >= n_warm
        for (idx,) in loader:
            opt_df_s.zero_grad(set_to_none=True)
            opt_pot_s.zero_grad(set_to_none=True)
            if CFG.training_mode == "truth":
                nll, grids = truth_nll(idx)
            else:
                grids = compute_model_grids()
                nll, _ = eval_data_nll(idx, grids, CFG.n_iwae)
            loss = (nll
                    + (1.0 / N) * CFG.sigma_los_weight * sigma_los_loss(grids)
                    + (1.0 / N) * CFG.sigma_pm_weight * sigma_pm_loss(grids)
                    + (1.0 / N) * CFG.vsp_weight * vsp_loss(grids)
                    + (1.0 / N) * CFG.pot_prior_weight * potential.prior_loss()
                    + strong_prior_scale() * potential.strong_prior())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(df.parameters()) + list(encoder.parameters()), CFG.grad_clip_df)
            opt_df_s.step()
            if joint_now:
                torch.nn.utils.clip_grad_norm_(pot_params, CFG.grad_clip_pot)
                opt_pot_s.step()
        sched.step()

        if ep >= tail_start and (ep % 5 == 0 or ep == epochs - 1):
            obj = node_objectives()
            if best is None or obj["L"] < best["L"]:
                best = obj
                best_states = (
                    {k: v.clone() for k, v in df.state_dict().items()},
                    {k: v.clone() for k, v in encoder.state_dict().items()},
                    {k: v.clone() for k, v in potential.state_dict().items()},
                )
    return best, best_states


def profile_scan_inner_slope():
    global potential, SCAN_SUMMARY
    n_nodes = sorted(float(x) for x in ARGS.scan_grid.split(","))

    df0 = {k: v.clone() for k, v in df.state_dict().items()}
    enc0 = {k: v.clone() for k, v in encoder.state_dict().items()}
    pot0 = {k: v.clone() for k, v in potential.state_dict().items()}
    n_map = float(torch.sigmoid(pot0["raw_n"])) if "raw_n" in pot0 else 0.5

    i0 = int(np.argmin([abs(nn - n_map) for nn in n_nodes]))
    L = len(n_nodes)
    sweeps = [list(range(i0, L)), list(range(i0 - 1, -1, -1))]
    if bool(getattr(CFG, "scan_multistart", False)):
        # Two more passes, from each end of the grid. Every node then gets three
        # tries from three different warm starts, and the accumulator below
        # already keeps whichever reached the lowest objective.
        sweeps += [list(range(L)), list(range(L - 1, -1, -1))]
    _restarts = int(getattr(CFG, "scan_restarts", 0))
    if _restarts > 0:
        # Independent mode: order no longer matters because nothing is chained,
        # so k identical passes over the whole grid is exactly k tries per node.
        sweeps = [list(range(L)) for _ in range(_restarts)]

    def _sweep(freeze_rs):
        """One full sweep of the grid from the MAP state, R_s held or not."""
        global _SCAN_FREEZE_RS
        _SCAN_FREEZE_RS = bool(freeze_rs)
        node_states, results = {}, {}
        _indep = int(getattr(CFG, "scan_restarts", 0)) > 0
        for order in sweeps:
            df_s, enc_s, pot_s = df0, enc0, pot0
            for i in order:
                n_fix = n_nodes[i]
                best, states = _optimize_node(n_fix, df_s, enc_s, pot_s,
                                              ARGS.scan_epochs)
                if not _indep:
                    df_s, enc_s, pot_s = states
                with torch.no_grad():
                    potential.load_state_dict(pot_s)
                    slope_Rh = float(potential.log_slope(
                        torch.tensor([R_HALF_2D], dtype=TD, device=DEVICE))[0])
                rec = {"n": n_fix, "gamma_inner": n_fix - 1.0,
                       "L_composite": best["L"], "nll_total": best["nll"],
                       "anchor_sig": best["sig"], "anchor_pm": best["pm"],
                       "anchor_vsp": best["vsp"], "slope_at_Rhalf": slope_Rh,
                       **potential.describe()}
                if (n_fix not in results) or (best["L"] < results[n_fix]["L_composite"]):
                    node_states[n_fix] = {
                        "df": {k: v.clone() for k, v in df_s.items()},
                        "encoder": {k: v.clone() for k, v in enc_s.items()},
                        "potential": {k: v.clone() for k, v in pot_s.items()},
                    }
                if (n_fix not in results) or (rec["L_composite"] < results[n_fix]["L_composite"]):
                    results[n_fix] = rec
                print(f"[scan{'/Rs frozen' if freeze_rs else ''}] "
                      f"n={n_fix:.3f} gamma={n_fix - 1:+.3f} "
                      f"L={rec['L_composite']:.2f} NLL={rec['nll_total']:.2f} "
                      f"slope(Rh)={slope_Rh:+.3f} rc={rec['r_core']:.3f}")
        _SCAN_FREEZE_RS = False
        tab = pd.DataFrame([results[nn] for nn in n_nodes]).sort_values("n")
        return tab, node_states, scan_resolution(tab["n"].values,
                                                 tab["nll_total"].values)

    mode = str(getattr(CFG, "scan_rs", "free"))
    global SCAN_RS_CHOICE
    if mode in ("free", "frozen"):
        sc, node_states, res = _sweep(mode == "frozen")
        SCAN_RS_CHOICE = {"mode": mode, "chosen": mode, "resolution": res}
        print(f"\n[scan] R_s {mode} during the sweep: span {res['span']:.3f} "
              f"(grid step {res['grid_step']:.3f}), depth {res['depth']:.1f}, "
              f"{res['n_plausible']} of {len(n_nodes)} nodes indistinguishable")
    else:
        # Both sweeps start from the same MAP state, so only the sweep is paid
        # twice, not the whole fit. Keep the better resolved one: smaller span,
        # and on a tie the deeper basin -- span alone ties at 0.000 on half the
        # mocks tested and then carries no information.
        sc_f, ns_f, res_f = _sweep(False)
        sc_z, ns_z, res_z = _sweep(True)
        if res_f["span"] < res_z["span"]:
            take, why = "free", "smaller span"
        elif res_z["span"] < res_f["span"]:
            take, why = "frozen", "smaller span"
        else:
            take = "free" if res_f["depth"] >= res_z["depth"] else "frozen"
            why = "span tied, deeper basin"
        sc, node_states = (sc_f, ns_f) if take == "free" else (sc_z, ns_z)
        SCAN_RS_CHOICE = {"mode": "auto", "chosen": take, "reason": why,
                          "free": res_f, "frozen": res_z,
                          "resolution": res_f if take == "free" else res_z}
        sc_f.to_csv(os.path.join(CFG.output_dir,
                                 "inner_slope_profile_scan_rsfree.csv"),
                    index=False)
        sc_z.to_csv(os.path.join(CFG.output_dir,
                                 "inner_slope_profile_scan_rsfrozen.csv"),
                    index=False)
        print(f"\n[scan] auto: R_s free   span {res_f['span']:.3f} depth {res_f['depth']:.1f}")
        print(f"[scan] auto: R_s frozen span {res_z['span']:.3f} depth {res_z['depth']:.1f}")
        print(f"[scan] auto: keeping the {take} sweep ({why}); both tables written")

    sc["dL"] = sc["L_composite"] - sc["L_composite"].min()
    sc["dNLL"] = sc["nll_total"] - sc["nll_total"].min()
    sc.to_csv(os.path.join(CFG.output_dir, "inner_slope_profile_scan.csv"), index=False)

    # cusp-vs-core statistic in PURE log-likelihood units
    nll_cusp = float(sc.loc[sc.n.idxmin(), "nll_total"])   # n=0 endpoint
    nll_core = float(sc.loc[sc.n.idxmax(), "nll_total"])   # n=1 endpoint
    delta_nll = nll_core - nll_cusp          # NEGATIVE => core fits better
    verdict = "CORE preferred" if delta_nll < 0 else "CUSP preferred"

    g = sc["gamma_inner"].values; dN = sc["dNLL"].values
    gg = np.linspace(g.min(), g.max(), 400); dd = np.interp(gg, g, dN)
    gbest = float(gg[np.argmin(dd)])

    def interval(th):
        ok = gg[dd <= th]
        return [float(ok.min()), float(ok.max())] if len(ok) else [np.nan, np.nan]

    ci68, ci95 = interval(0.5), interval(1.92)
    slope_best = float(np.interp(gbest, g, sc["slope_at_Rhalf"].values))

    print(f"\nPROFILE (pure NLL): gamma0 = {gbest:+.3f}  68% {ci68}  95% {ci95}"
          f"   [thresholds approximate -> calibrate on mock seeds]")
    print(f"slope at R_half (profiled) = {slope_best:+.3f}")
    print(f"Delta_NLL(core - cusp) = {delta_nll:+.2f}  ->  {verdict}")
    print(f"(composite Delta, arb. units: "
          f"{float(sc.loc[sc.n.idxmax(), 'L_composite'] - sc.loc[sc.n.idxmin(), 'L_composite']):+.2f})")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(g, dN, "o-", color="tab:red", label=r"$\Delta$NLL (pure likelihood)")
    ax.plot(g, sc["dL"].values, "s--", color="0.5", alpha=0.7,
            label=r"$\Delta L$ composite (arb.)")
    ax.axhline(0.5, ls=":", color="0.4"); ax.axhline(1.92, ls=":", color="0.4")
    ax.set_xlabel(r"asymptotic inner slope $\gamma_0 = n-1$")
    ax.set_ylabel(r"$\Delta$ objective")
    ax.set_title("Profile likelihood over the inner slope")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_pub(fig, "inner_slope_profile")
    plt.close(fig)

    global SCAN_TABLE, SCAN_NODE_STATES, SCAN_BEST_N
    SCAN_TABLE = sc.copy()
    SCAN_NODE_STATES = node_states
    SCAN_BEST_N = float(sc.loc[sc["nll_total"].idxmin(), "n"])

    # RESTORE the joint MAP (fresh free-n object; load values into it)
    pot_map = make_corenfw(fix_n=None)
    pot_map.load_state_dict(pot0)
    potential = pot_map
    df.load_state_dict(df0)
    encoder.load_state_dict(enc0)
    print("[scan] restored joint-MAP state for diagnostics:", potential.describe())

    SCAN_SUMMARY = {
        "gamma0_profiled": gbest, "gamma0_68CI": ci68, "gamma0_95CI": ci95,
        "slope_at_Rhalf_profiled": slope_best,
        "delta_nll_core_minus_cusp": delta_nll, "verdict": verdict,
        "note": ("Pure-NLL profile, bidirectional continuation; chi2 thresholds "
                 "approximate (bounded param + stochastic re-opt) -> calibrate "
                 "Delta_NLL on mock seeds."),
    }
    return sc


if getattr(ARGS, "scan_inner_slope", False) and CFG.potential_model == "corenfw":
    print("\n=== PROFILE-LIKELIHOOD SCAN over inner slope (continuation) ===")
    profile_scan_inner_slope()
else:
    SCAN_SUMMARY = None


def _weighted_quantile(vals, weights, qs):
    """vals [n_models, n_r], weights [n_models] -> [len(qs), n_r]."""
    out = np.empty((len(qs), vals.shape[1]))
    for j in range(vals.shape[1]):
        order = np.argsort(vals[:, j])
        v, w = vals[order, j], weights[order]
        cw = np.cumsum(w) - 0.5 * w
        cw /= w.sum()
        out[:, j] = np.interp(qs, cw, v)
    return out


def plot_profile_family_band():
    """Likelihood-weighted band over the scan-node models. This is the
    honest n-direction uncertainty (the Laplace band massively understates
    it because it freezes the DF and linearizes a bounded parameter)."""
    if SCAN_TABLE is None or SCAN_NODE_STATES is None:
        return
    r_eval = np.logspace(np.log10(R_BENCH_LO), np.log10(R_BENCH_HI), 140)
    r_t = torch.tensor(r_eval, dtype=TD, device=DEVICE)

    w = np.exp(-SCAN_TABLE["dNLL"].values)
    w = w / w.sum()

    rho_s, slope_s, mass_s = [], [], []
    for n_fix in SCAN_TABLE["n"].values:
        pot_n = make_corenfw(fix_n=float(n_fix))
        pot_n.load_state_dict(SCAN_NODE_STATES[float(n_fix)]["potential"])
        with torch.no_grad():
            rho = np.maximum(pot_n.density(r_t).cpu().numpy(), 1e-300)
            M = pot_n.mass(r_t).cpu().numpy()
        rho_s.append(rho); mass_s.append(M)
        slope_s.append(np.gradient(np.log(rho), np.log(r_eval)))
    rho_s, slope_s, mass_s = map(np.asarray, (rho_s, slope_s, mass_s))

    qs = [0.16, 0.5, 0.84]
    rho_q = _weighted_quantile(rho_s, w, qs)
    slope_q = _weighted_quantile(slope_s, w, qs)
    mass_q = _weighted_quantile(mass_s, w, qs)

    ib = int(SCAN_TABLE["nll_total"].values.argmin())      # profiled best node
    rho_true_b = np_rho_truth(r_eval)
    slope_true_b = np_dlnrho_dlnr_truth(r_eval)
    mass_true_b = np_mass_truth(r_eval)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))

    ax = axs[0]
    ax.fill_between(r_eval, rho_q[0], rho_q[2], color="tab:red", alpha=0.25,
                    label="68% profile-family band")
    ax.plot(r_eval, rho_s[ib], "r-", lw=2.2, label="profiled best")
    ax.plot(r_eval, rho_true_b, "k-", lw=2.0, label=_TRUTH_LABEL)
    ax.axvline(R_HALF_2D, color="0.3", ls="--", lw=1.1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$\rho_{\rm DM}$ [Msun/kpc$^3$]")
    ax.set_title("(a) DM density (profile-weighted)")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    ax = axs[1]
    ax.fill_between(r_eval, slope_q[0], slope_q[2], color="tab:red", alpha=0.25,
                    label="68% profile-family band")
    ax.plot(r_eval, slope_s[ib], "r-", lw=2.2, label="profiled best")
    ax.plot(r_eval, slope_true_b, "k-", lw=2.0, label=_TRUTH_LABEL)
    ax.axvline(R_HALF_2D, color="0.3", ls="--", lw=1.1)
    ax.set_xscale("log"); ax.set_ylim(-3.2, 0.5)
    ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$d\ln\rho/d\ln r$")
    ax.set_title("(b) Log-slope (profile-weighted)")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    ax = axs[2]
    ax.fill_between(r_eval, mass_q[0], mass_q[2], color="tab:red", alpha=0.25,
                    label="68% profile-family band")
    ax.plot(r_eval, mass_s[ib], "r-", lw=2.2, label="profiled best")
    ax.plot(r_eval, mass_true_b, "k-", lw=2.0, label=_TRUTH_LABEL)
    ax.axvline(R_HALF_2D, color="0.3", ls="--", lw=1.1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$M(<r)$ [Msun]")
    ax.set_title("(c) Enclosed mass (profile-weighted)")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    save_pub(fig, "profile_family_band")
    plt.close(fig)
    print("[scan] wrote profile_family_band.pdf/png (likelihood-weighted family band)")


if getattr(ARGS, "scan_inner_slope", False) and CFG.potential_model == "corenfw":
    plot_profile_family_band()


# ============================================================
# 11. DIAGNOSTICS, PLOTS, METRICS
# ============================================================
print("\nBuilding diagnostics...")

if (getattr(ARGS, "diagnostics_at", "map") == "profile"
        and SCAN_NODE_STATES is not None):
    ns = SCAN_NODE_STATES[SCAN_BEST_N]
    potential = make_corenfw(fix_n=SCAN_BEST_N)
    potential.load_state_dict(ns["potential"])
    df.load_state_dict(ns["df"])            # the DF trained WITH this potential
    encoder.load_state_dict(ns["encoder"])
    print(f"[diag] rendering at profiled best node n={SCAN_BEST_N:.3f} "
          f"(self-consistent DF+potential pair):", potential.describe())
else:
    if CFG.potential_model == "corenfw":
        potential = make_corenfw(fix_n=None) 
    potential.load_state_dict(MAP_STATE["potential"])
    df.load_state_dict(MAP_STATE["df"])
    encoder.load_state_dict(MAP_STATE["encoder"])

for p in potential.parameters():
    p.requires_grad_(False)
for p in df.parameters():
    p.requires_grad_(False)
for p in encoder.parameters():
    p.requires_grad_(False)


POSTERIOR_DIAG_K = 128

@torch.no_grad()
def projected_radius_posterior():
    """Posterior mean 3D radius E[r | R_i, w_i] and the ESS of the
    normalized LOS quadrature weights, under the fitted model.
 
    The per-star posterior over the LOS depth s is
        p(s | R_i, w_i)  ~  (g * N(0, e_eff^2))(w_i; r(s), q(s)),
    symmetric in +-s, so E[r] uses the half-line weights directly.
    Deterministic analogue of reconstruct_posterior_means; 'ESS' here
    measures how concentrated the s-posterior is on the quadrature
    grid (K = proj_n_s), not an importance-sampling ESS."""
    grids = compute_model_grids()
    W = grids["w_grid"]
    dx = W[1] - W[0]
    s, s_w = PROJ["s"], PROJ["s_w"]
 
    r_out = np.zeros(N, dtype=float)
    ess_out = np.zeros(N, dtype=float)
 
    for start in range(0, N, 256):
        idx = indices_all[start:start + 256]
        B = len(idx)
        R = PROJ["R"][idx]
        w = PROJ["w"][idx]
        e = PROJ["e"][idx]
 
        r_s = torch.sqrt(R[:, None] ** 2 + s[None, :] ** 2).clamp_min(1e-6)
        q_s = (R[:, None] / r_s).clamp(0.0, 1.0)
        G = torch.exp(_interp_logG0(grids["logG0"], grids["log_r_sub"],
                                    torch.log(r_s), q_s))    # [B, S, K]
 
        e_eff = torch.sqrt((e * e - PROJ["e0"] ** 2).clamp_min(0.0))
        wq = (w[:, None, None]
              + math.sqrt(2.0) * e_eff[:, None, None]
              * PROJ["gh_x"][None, None, :]).expand(-1, G.shape[1], -1)
        Gq = _interp_gather_uniform(G, wq, W[0], dx)         # [B, S, J]
        Ls = torch.sum(PROJ["gh_w"][None, None, :] * Gq,
                       dim=-1) / math.sqrt(math.pi)          # [B, S]
 
        wt = s_w[None, :] * Ls
        wt = wt / wt.sum(dim=1, keepdim=True).clamp_min(1e-300)
        r_mean = torch.sum(wt * r_s, dim=1)
        ess = 1.0 / torch.sum(wt * wt, dim=1).clamp_min(1e-300)
 
        r_out[start:start + B] = r_mean.cpu().numpy()
        ess_out[start:start + B] = ess.cpu().numpy()
 
    return r_out, ess_out

def reconstruct_posterior_means(K=POSTERIOR_DIAG_K, seed=None):
    """
    Self-normalized posterior means and per-star ESS.

    When CFG.marginalize_distance is enabled, this uses exactly the same
    estimator structure as training:

        - K importance samples over the five non-distance coordinates;
        - deterministic quadrature over distance;
        - ESS computed from the K distance-marginalized weights.

    The returned posterior means integrate jointly over both k and D.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    grids = compute_model_grids()

    pos_out = np.zeros((N, 3), dtype=float)
    vel_out = np.zeros((N, 3), dtype=float)
    ess_out = np.zeros(N, dtype=float)

    with torch.no_grad():
        for start in range(0, N, 256):
            idx = indices_all[start:start + 256]
            B = len(idx)

            mu, logsig = encoder(ENC_FEATS[idx])

            # =========================================================
            # Five-dimensional importance sampling + distance quadrature
            # =========================================================
            if CFG.marginalize_distance:
                mu5 = mu[:, IDX5]
                ls5 = logsig[:, IDX5]
                sig5 = torch.exp(ls5)

                noise5 = torch.randn(
                    B,
                    K,
                    5,
                    dtype=TD,
                    device=DEVICE,
                )

                s5 = mu5[:, None, :] + sig5[:, None, :] * noise5

                obs5 = obs_mes[idx][:, IDX5]
                err5 = err_mes[idx][:, IDX5]

                z5 = (
                    obs5[:, None, :]
                    + s5 * err5[:, None, :]
                )

                nD = len(D_NODES)

                z = torch.empty(
                    B,
                    K,
                    nD,
                    6,
                    dtype=TD,
                    device=DEVICE,
                )

                z[..., 0] = z5[..., 0][..., None]
                z[..., 1] = z5[..., 1][..., None]
                z[..., 2] = D_NODES[None, None, :]
                z[..., 3] = z5[..., 2][..., None]
                z[..., 4] = z5[..., 3][..., None]
                z[..., 5] = z5[..., 4][..., None]

                logp = phase_space_logdensity(
                    z.reshape(-1, 6),
                    grids["logN"],
                ).reshape(B, K, nD)

                Dobs = obs_mes[idx][:, 2]
                eD = err_mes[idx][:, 2]

                log_nD = (
                    -0.5
                    * (
                        (D_NODES[None, :] - Dobs[:, None])
                        / eD[:, None]
                    ) ** 2
                    - torch.log(eD)[:, None]
                    - 0.5 * LOG2PI
                )

                # Model likelihood × distance likelihood × quadrature weight.
                log_integrand = (
                    logp
                    + log_nD[:, None, :]
                    + D_LOGW[None, None, :]
                )

                log_prior5 = (
                    -0.5 * torch.sum(s5 * s5, dim=-1)
                    - 2.5 * LOG2PI
                )

                log_q5 = torch.sum(
                    -0.5 * noise5 ** 2
                    - ls5[:, None, :]
                    - 0.5 * LOG2PI,
                    dim=-1,
                )

                # One distance-marginalized weight for each k.
                logw_k = (
                    torch.logsumexp(log_integrand, dim=2)
                    + log_prior5
                    - log_q5
                )

                logw_k = torch.nan_to_num(
                    logw_k,
                    nan=-1.0e9,
                    posinf=-1.0e9,
                    neginf=-1.0e9,
                )

                w_k = torch.softmax(logw_k, dim=1)

                ess = 1.0 / torch.sum(w_k * w_k, dim=1)
                ess_out[start:start + B] = ess.cpu().numpy()

                # Joint posterior weights over both proposal samples and
                # distance nodes, used for posterior means.
                logw_joint = (
                    log_integrand
                    + log_prior5[:, :, None]
                    - log_q5[:, :, None]
                )

                logw_joint = torch.nan_to_num(
                    logw_joint,
                    nan=-1.0e9,
                    posinf=-1.0e9,
                    neginf=-1.0e9,
                )

                w_joint = torch.softmax(
                    logw_joint.reshape(B, K * nD),
                    dim=1,
                ).reshape(B, K, nD)

                pos, vel = obs_to_galactocentric_torch(
                    z.reshape(-1, 6)
                )

                pos = pos.reshape(B, K, nD, 3)
                vel = vel.reshape(B, K, nD, 3)

                pos_mean = torch.sum(
                    w_joint[:, :, :, None] * pos,
                    dim=(1, 2),
                )

                vel_mean = torch.sum(
                    w_joint[:, :, :, None] * vel,
                    dim=(1, 2),
                )

            # =========================================================
            # Ordinary six-dimensional importance sampling
            # =========================================================
            else:
                sig = torch.exp(logsig)

                noise = torch.randn(
                    B,
                    K,
                    6,
                    dtype=TD,
                    device=DEVICE,
                )

                s = mu[:, None, :] + sig[:, None, :] * noise

                z = (
                    obs_mes[idx][:, None, :]
                    + s * err_mes[idx][:, None, :]
                )

                zf = z.reshape(B * K, 6)
                valid = zf[:, 2] > 1.0

                logp = phase_space_logdensity(
                    zf,
                    grids["logN"],
                )

                logp = torch.where(
                    valid,
                    logp,
                    torch.full_like(logp, -1.0e9),
                ).reshape(B, K)

                log_prior = (
                    -0.5 * torch.sum(s * s, dim=-1)
                    - 3.0 * LOG2PI
                )

                log_q = torch.sum(
                    -0.5 * noise ** 2
                    - logsig[:, None, :]
                    - 0.5 * LOG2PI,
                    dim=-1,
                )

                logw = torch.nan_to_num(
                    logp + log_prior - log_q,
                    nan=-1.0e9,
                    posinf=-1.0e9,
                    neginf=-1.0e9,
                )

                w = torch.softmax(logw, dim=1)

                ess = 1.0 / torch.sum(w * w, dim=1)
                ess_out[start:start + B] = ess.cpu().numpy()

                zs = torch.stack(
                    [
                        zf[:, 0],
                        zf[:, 1],
                        zf[:, 2].clamp_min(1.0),
                        zf[:, 3],
                        zf[:, 4],
                        zf[:, 5],
                    ],
                    dim=1,
                )

                pos, vel = obs_to_galactocentric_torch(zs)

                pos = pos.reshape(B, K, 3)
                vel = vel.reshape(B, K, 3)

                pos_mean = torch.sum(w[:, :, None] * pos, dim=1)
                vel_mean = torch.sum(w[:, :, None] * vel, dim=1)

            pos_out[start:start + B] = pos_mean.cpu().numpy()
            vel_out[start:start + B] = vel_mean.cpu().numpy()

    return pos_out, vel_out, ess_out

if CFG.training_mode == "observed":
    pos_fit_np, vel_fit_np, ess_np = reconstruct_posterior_means(
        K=POSTERIOR_DIAG_K,
        seed=CFG.seed + 9191,
    )
    r_fit_np = np.linalg.norm(pos_fit_np, axis=1)
elif CFG.training_mode == "projected":
    r_fit_np, ess_np = projected_radius_posterior()
    pos_fit_np = np.full((N, 3), np.nan)
    vel_fit_np = np.full((N, 3), np.nan)
    POSTERIOR_DIAG_K = CFG.proj_n_s          # ESS is over LOS s-nodes
else:
    pos_fit_np = truth_pos.cpu().numpy()
    vel_fit_np = truth_vel.cpu().numpy()
    ess_np = np.full(N, np.nan)
    r_fit_np = np.linalg.norm(pos_fit_np, axis=1)

POSTERIOR_DIAG_SUMMARY = None

if CFG.training_mode in ("observed", "projected"):
    ess_fraction = ess_np / float(POSTERIOR_DIAG_K)

    posterior_diag_df = pd.DataFrame({
        "star_index": np.arange(N, dtype=int),
        "seed": CFG.seed,
        "K": POSTERIOR_DIAG_K,
        "distance_marginalized": bool(CFG.marginalize_distance),

        "posterior_ess": ess_np,
        "posterior_ess_over_K": ess_fraction,

        "r_noisy_kpc": r_noisy_np,
        "r_posterior_mean_kpc": r_fit_np,

        "x_noisy_kpc": noisy_pos_np[:, 0],
        "y_noisy_kpc": noisy_pos_np[:, 1],
        "z_noisy_kpc": noisy_pos_np[:, 2],

        "x_posterior_mean_kpc": pos_fit_np[:, 0],
        "y_posterior_mean_kpc": pos_fit_np[:, 1],
        "z_posterior_mean_kpc": pos_fit_np[:, 2],
    })

    if r_truth_np is not None:
        posterior_diag_df["r_true_kpc"] = r_truth_np
        posterior_diag_df["log10_radius_error_noisy"] = np.log10(
            np.maximum(r_noisy_np, 1.0e-12)
            / np.maximum(r_truth_np, 1.0e-12)
        )
        posterior_diag_df["log10_radius_error_posterior"] = np.log10(
            np.maximum(r_fit_np, 1.0e-12)
            / np.maximum(r_truth_np, 1.0e-12)
        )

    posterior_diag_df.to_csv(
        os.path.join(
            CFG.output_dir,
            "posterior_star_diagnostics.csv",
        ),
        index=False,
    )

    POSTERIOR_DIAG_SUMMARY = {
        "K": int(POSTERIOR_DIAG_K),
        "estimator": (
            "LOS-depth quadrature posterior (projected mode)"
            if CFG.training_mode == "projected"
            else "5D importance sampling plus deterministic distance "
                 "quadrature"
            if CFG.marginalize_distance
            else "6D importance sampling"
        ),
        "ess_mean": float(np.nanmean(ess_np)),
        "ess_median": float(np.nanmedian(ess_np)),
        "ess_p05": float(np.nanpercentile(ess_np, 5)),
        "ess_p16": float(np.nanpercentile(ess_np, 16)),
        "ess_p84": float(np.nanpercentile(ess_np, 84)),
        "ess_min": float(np.nanmin(ess_np)),
        "ess_max": float(np.nanmax(ess_np)),
        "median_ess_over_K": float(np.nanmedian(ess_fraction)),
        "fraction_ess_below_K_over_4": float(
            np.mean(ess_np < POSTERIOR_DIAG_K / 4.0)
        ),
        "fraction_ess_below_K_over_10": float(
            np.mean(ess_np < POSTERIOR_DIAG_K / 10.0)
        ),
    }

    with open(
        os.path.join(
            CFG.output_dir,
            "posterior_diagnostic_summary.json",
        ),
        "w",
    ) as f:
        json.dump(POSTERIOR_DIAG_SUMMARY, f, indent=2)

if CFG.training_mode in ("observed", "projected"):
    print(f"Posterior ESS: mean={np.nanmean(ess_np):.1f}, "
          f"median={np.nanmedian(ess_np):.1f}, "
          f"p05={np.nanpercentile(ess_np, 5):.1f} "
          f"/ {POSTERIOR_DIAG_K}"
          + ("  [LOS-depth quadrature concentration, not IS quality]"
             if CFG.training_mode == "projected" else ""))

# final model grids and profiles
grids_f = compute_model_grids()

# ---- final projected-anchor diagnostics ----
with torch.no_grad():
    if CFG.use_vsp_anchor:
        VSP_MODEL_NP = model_vsp(grids_f).detach().cpu().numpy()
        VSP_PULL_NP = (VSP_MODEL_NP - VSP_OBS_NP) / VSP_ERR_NP
    else:
        VSP_MODEL_NP = None
        VSP_PULL_NP = None
if CFG.use_vsp_anchor:
    vsp_df = pd.DataFrame({
        "name": ["VSP1", "VSP2"],
        "data": VSP_OBS_NP,
        "model": VSP_MODEL_NP,
        "error": VSP_ERR_NP,
        "pull": VSP_PULL_NP,
        "chi2_contribution": VSP_PULL_NP ** 2,
    })

    vsp_df.to_csv(
        os.path.join(CFG.output_dir, "vsp_diagnostics.csv"),
        index=False,
    )
r_gnp = grids_f["r"].cpu().numpy()
nu_np = grids_f["nu"].cpu().numpy()
Nnorm = float(torch.exp(grids_f["logN"]))
rho_star_model = CFG.M_STAR_TOT * nu_np / Nnorm
sig_r_model = np.sqrt(grids_f["sig_r2"].cpu().numpy())
beta_model = grids_f["beta"].cpu().numpy()

# ============================================================
# 11b. GRAVSPHERE2-STYLE BANDS, SLOPE METRIC, AND FIGURE
# ============================================================
def potential_density_slope_bands(r_eval, n_samp=1500):
    """Propagate the Laplace covariance on the POTENTIAL parameters to
    95% bands on rho(r) and the 3D log-slope d ln rho/d ln r. Cheap for
    parametric potentials (nfw/gnfw/zhao: analytic density). Falls back to
    the MAP curve with zero width if no Laplace covariance is available
    (e.g. free_mass, --skip-laplace, or a failed Hessian)."""
    r_t = torch.tensor(r_eval, dtype=TD, device=DEVICE)
    params = [p for p in potential.parameters()]
    p0 = [p.detach().clone() for p in params]

    with torch.no_grad():
        rho_map = np.maximum(potential.density(r_t).cpu().numpy(), 1e-300)
    slope_map = np.gradient(np.log(rho_map), np.log(r_eval))

    if laplace is None or not np.all(np.isfinite(np.array(laplace["cov"], float))):
        tri=lambda a: np.vstack([a, a, a])
        return {"r": r_eval, "rho": tri(rho_map), "slope": tri(slope_map),
                "rho_samples": None, "slope_samples": None, "n_used": 0}

    mean = np.array(laplace["values"], float)
    cov = np.array(laplace["cov"], float)
    cov = 0.5 * (cov + cov.T)
    try:
        samples = np.random.multivariate_normal(mean, cov, size=n_samp)
    except np.linalg.LinAlgError:
        samples = np.random.multivariate_normal(
            mean, cov + 1e-8 * np.eye(len(mean)), size=n_samp)

    rho_s = np.empty((n_samp, len(r_eval)))
    slope_s = np.empty((n_samp, len(r_eval)))
    with torch.no_grad():
        for k in range(n_samp):
            for p, v in zip(params, samples[k]):
                p.copy_(torch.tensor(v, dtype=TD, device=p.device))
            rho = np.maximum(potential.density(r_t).cpu().numpy(), 1e-300)
            rho_s[k] = rho
            slope_s[k] = np.gradient(np.log(rho), np.log(r_eval))
        for p, v in zip(params, p0):
            p.copy_(v)

    return {"r": r_eval,
            "rho": np.percentile(rho_s, [2.5, 50, 97.5], axis=0),
            "slope": np.percentile(slope_s, [2.5, 50, 97.5], axis=0),
            "rho_samples": rho_s, "slope_samples": slope_s, "n_used": n_samp}


r_bench = np.logspace(np.log10(R_BENCH_LO), np.log10(R_BENCH_HI), 80)
bands = potential_density_slope_bands(r_bench)

rho_true_bench = np_rho_truth(r_bench)
slope_true_bench = np_dlnrho_dlnr_truth(r_bench)


def _interp_at(R, x, y):
    return float(np.interp(np.log(R), np.log(x), y))


slope_lo  = _interp_at(R_HALF_2D, r_bench, bands["slope"][0])
slope_med = _interp_at(R_HALF_2D, r_bench, bands["slope"][1])
slope_hi  = _interp_at(R_HALF_2D, r_bench, bands["slope"][2])

if bands["n_used"] > 0:
    s_samp = np.array([_interp_at(R_HALF_2D, r_bench, bands["slope_samples"][k])
                       for k in range(bands["slope_samples"].shape[0])])
    s16, s50, s84 = (float(x) for x in np.percentile(s_samp, [16, 50, 84]))
    sigma_slope = 0.5 * (s84 - s16)
    frac_prec = sigma_slope / max(abs(s50), 1e-6)
    src = "laplace_potential_only"
else:
    s16 = s50 = s84 = slope_med
    sigma_slope = 0.0
    frac_prec = 0.0
    src = "MAP_point_no_band"

slope_metrics = {
    "R_half_2d_kpc": R_HALF_2D,
    "benchmark_range_kpc": [R_BENCH_LO, R_BENCH_HI],
    "slope_at_Rhalf_true": slope_true_R12,
    "slope_at_Rhalf_median": s50,
    "slope_at_Rhalf_68CI": [s16, s84],
    "slope_at_Rhalf_95CI": [slope_lo, slope_hi],
    "slope_at_Rhalf_1sigma": sigma_slope,
    "slope_at_Rhalf_frac_precision": frac_prec,   # compare to paper 0.08-0.25
    "slope_at_Rhalf_signed_error": s50 - slope_true_R12,
    "posterior_source": src,
    "caveat": ("Laplace freezes DF/encoder -> profile-likelihood width, "
               "likely under-estimated. Bootstrap for the honest band."),
}
print("\nGravSphere2-style slope at R_1/2:")
print(f"  true   = {slope_true_R12:+.3f}")
print(f"  recov. = {s50:+.3f}  [68% {s16:+.3f}, {s84:+.3f}]  "
      f"(1sigma={sigma_slope:.3f}, frac={100*frac_prec:.0f}%)  src={src}")
if SCAN_SUMMARY is not None:
    print("\n[HEADLINE] profiled inner slope (scan):")
    print(f"  gamma0    = {SCAN_SUMMARY['gamma0_profiled']:+.3f} "
          f"68% {SCAN_SUMMARY['gamma0_68CI']}")
    print(f"  slope(Rh) = {SCAN_SUMMARY['slope_at_Rhalf_profiled']:+.3f}")
    print(f"  DeltaNLL(core-cusp) = {SCAN_SUMMARY['delta_nll_core_minus_cusp']:+.2f}"
          f"  ->  {SCAN_SUMMARY['verdict']}")

# save the band curves for your own re-plotting
beta_bench = np.interp(np.log(r_bench), np.log(r_gnp), beta_model)
beta_tilde_bench = beta_bench / (2.0 - beta_bench)
pd.DataFrame({
    "r": r_bench,
    "rho_lo": bands["rho"][0], "rho_med": bands["rho"][1], "rho_hi": bands["rho"][2],
    "rho_true": rho_true_bench,
    "slope_lo": bands["slope"][0], "slope_med": bands["slope"][1],
    "slope_hi": bands["slope"][2], "slope_true": slope_true_bench,
    "beta_tilde_map": beta_tilde_bench,
}).to_csv(os.path.join(CFG.output_dir, "gravsphere_bands.csv"), index=False)

# Figure: rho, log-slope, beta_tilde over 0.1-10 R_1/2 (paper Figs 1/2/4/6 style)
fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))

ax = axs[0]
ax.fill_between(r_bench, bands["rho"][0], bands["rho"][2],
                color="tab:red", alpha=0.25, label="95% band")
ax.plot(r_bench, bands["rho"][1], "r-", lw=2.0, label="fit median")
ax.plot(r_bench, rho_true_bench, "k-", lw=2.0, label=_TRUTH_LABEL)
ax.axvline(R_HALF_2D, color="0.3", ls="--", lw=1.2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(R_BENCH_LO, R_BENCH_HI)
ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$\rho_{\rm DM}$ [Msun/kpc$^3$]")
ax.set_title(r"(a) DM density"); ax.legend(); ax.grid(alpha=0.3, which="both")

ax = axs[1]
ax.fill_between(r_bench, bands["slope"][0], bands["slope"][2],
                color="tab:red", alpha=0.25, label="95% band")
ax.plot(r_bench, bands["slope"][1], "r-", lw=2.0, label="fit median")
ax.plot(r_bench, slope_true_bench, "k-", lw=2.0, label=_TRUTH_LABEL)
ax.axvline(R_HALF_2D, color="0.3", ls="--", lw=1.2)
ax.scatter([R_HALF_2D], [s50], color="r", zorder=6, s=45,
           label=r"slope @ $R_{1/2}$ (fit)")
ax.scatter([R_HALF_2D], [slope_true_R12], color="k", marker="x", zorder=6, s=55,
           label=r"slope @ $R_{1/2}$ (true)")
ax.set_xscale("log"); ax.set_xlim(R_BENCH_LO, R_BENCH_HI); ax.set_ylim(-3.2, 0.5)
ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$d\ln\rho/d\ln r$")
ax.set_title(r"(b) Log-density slope"); ax.legend(); ax.grid(alpha=0.3, which="both")

ax = axs[2]
ax.axhline(0.0, color="k", lw=1.6, label="isotropic truth")
ax.plot(r_bench, beta_tilde_bench, "b-", lw=2.0, label=r"fit $\tilde\beta$ (MAP)")
ax.axvline(R_HALF_2D, color="0.3", ls="--", lw=1.2)
ax.set_xscale("log"); ax.set_xlim(R_BENCH_LO, R_BENCH_HI); ax.set_ylim(-1.0, 1.0)
ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$\tilde\beta=\beta/(2-\beta)$")
ax.set_title(r"(c) Anisotropy (band needs bootstrap)")
ax.legend(); ax.grid(alpha=0.3, which="both")

plt.tight_layout()
save_pub(fig, "method")
plt.close(fig)

r_plot = np.logspace(math.log10(CFG.r_plot_min), math.log10(CFG.r_plot_max), CFG.n_r_plot)
r_plot_t = torch.tensor(r_plot, dtype=TD, device=DEVICE)
pos_eval = torch.zeros((len(r_plot), 3), dtype=TD, device=DEVICE)
pos_eval[:, 0] = r_plot_t
with torch.no_grad():
    mass_fit = potential.mass(r_plot_t).cpu().numpy()
    rho_fit = potential.density(r_plot_t).cpu().numpy()
    acc_fit = np.abs((CFG.G * mass_fit / np.maximum(r_plot**2, 1e-30)))
    sig_los_model_np = model_sigma_los(SIG_R_T, grids_f).cpu().numpy()
    R_dense = torch.tensor(np.logspace(math.log10(max(SIG_R_NP.min()*0.3, 1e-3)),
                                       math.log10(SIG_R_NP.max()*3.0), 60),
                           dtype=TD, device=DEVICE)
    sig_los_dense = model_sigma_los(R_dense, grids_f).cpu().numpy()
    R_dense_np = R_dense.cpu().numpy()

mass_true = np_mass_truth(r_plot)
rho_true = np_rho_truth(r_plot)
acc_true = np_force_truth(r_plot)

# support quantiles from the tracer model itself (mass-weighted radii)
cdf = cumulative_trapezoid(4*np.pi*r_gnp**2*nu_np, r_gnp, initial=0.0)
cdf /= max(cdf[-1], 1e-300)
def q_of(p): return float(np.interp(p, cdf, r_gnp))
r_sup = {"r005": q_of(0.005), "r05": q_of(0.05), "r50": q_of(0.5),
         "r95": q_of(0.95), "r995": q_of(0.995)}

# ---- metrics ----
def log_rmse(yp, yt, m):
    g = m & np.isfinite(yp) & np.isfinite(yt) & (yp > 0) & (yt > 0)
    if g.sum() < 3:
        return float("nan")
    d = np.log10(yp[g]) - np.log10(yt[g])
    return float(np.sqrt(np.mean(d * d)))


def log_bias(yp, yt, m):
    g = m & np.isfinite(yp) & np.isfinite(yt) & (yp > 0) & (yt > 0)
    if g.sum() < 3:
        return float("nan")
    return float(np.median(np.log10(yp[g]) - np.log10(yt[g])))


mask_5_95 = (r_plot >= r_sup["r05"]) & (r_plot <= r_sup["r95"])
mask_005_995 = (r_plot >= r_sup["r005"]) & (r_plot <= r_sup["r995"])

with torch.no_grad():
    M_half_fit = float(potential.mass(torch.tensor([r3], dtype=TD, device=DEVICE))[0])
M_half_true = float(np_mass_truth(r3))

metrics = {
    "seed": CFG.seed,
    "training_mode": CFG.training_mode,
    "potential_model": CFG.potential_model,
    "N": int(N),
    "potential_fit": potential.describe(),
    "truth_profile_source": (
        "truth_profile.csv" if _TRUTH_TBL is not None else "analytic_nfw"
    ),
    "truth_label": _TRUTH_LABEL,
    "support": r_sup,
    "use_pm_anchor": bool(CFG.use_pm_anchor),
    "use_vsp_anchor": bool(CFG.use_vsp_anchor),
    "sigma_pm_weight": float(CFG.sigma_pm_weight),
    "vsp_weight": float(CFG.vsp_weight),
    "mass_log_rmse_5_95": log_rmse(mass_fit, mass_true, mask_5_95),
    "mass_log_bias_5_95": log_bias(mass_fit, mass_true, mask_5_95),
    "acc_log_rmse_5_95": log_rmse(acc_fit, acc_true, mask_5_95),
    "acc_log_bias_5_95": log_bias(acc_fit, acc_true, mask_5_95),
    "rho_log_rmse_5_95": log_rmse(rho_fit, rho_true, mask_5_95),
    "rho_log_bias_5_95": log_bias(rho_fit, rho_true, mask_5_95),
    "mass_log_rmse_005_995": log_rmse(mass_fit, mass_true, mask_005_995),
    "M_wolf_radius_r3": r3,
    "M_r3_fit": M_half_fit,
    "M_r3_true": M_half_true,
    "M_r3_log10_ratio": float(np.log10(M_half_fit / M_half_true)),
    "sigma_los_chi2_per_bin": float(np.mean(((sig_los_model_np - SIG_T_NP) / SIG_E_NP) ** 2)),
    "sigma_pm_chi2_per_bin": (
        (lambda sR, sT: float(0.5 * (
            np.mean(((sR - PM_RT["sR"]) / PM_RT["eSR"]) ** 2)
          + np.mean(((sT - PM_RT["sT"]) / PM_RT["eST"]) ** 2))))(
            *[a.detach().cpu().numpy() for a in model_sigma_pm_RT(PM_R_T, grids_f)])
        if CFG.use_pm_anchor else None
    ),
    "vsp_chi2": (
        float(np.sum(((model_vsp(grids_f).detach().cpu().numpy() - VSP_OBS_NP) / VSP_ERR_NP) ** 2))
        if CFG.use_vsp_anchor else None
    ),
}
if _TRUTH_TBL is None:
    metrics["M0_true"] = CFG.M_0_NFW_TRUE
    metrics["Rs_true"] = CFG.R_S_NFW_TRUE
if laplace is not None:
    metrics["laplace"] = laplace
if CFG.training_mode in ("observed", "projected"):
    metrics["posterior_diagnostics"] = POSTERIOR_DIAG_SUMMARY

if r_truth_np is not None and CFG.training_mode in ("observed", "projected"):
    metrics["radius_log_rmse_fit_vs_truth"] = log_rmse(
        r_fit_np,
        r_truth_np,
        np.ones_like(r_fit_np, dtype=bool),
    )

    metrics["radius_log_rmse_noisy_vs_truth"] = log_rmse(
        r_noisy_np,
        r_truth_np,
        np.ones_like(r_fit_np, dtype=bool),
    )

# ---- save model + tables ----
torch.save({"df": df.state_dict(), "encoder": encoder.state_dict(),
            "potential": potential.state_dict(), "config": asdict(CFG),
            "history": history},
           os.path.join(CFG.output_dir, "model_state.pt"))

profile_df = pd.DataFrame({
    "r": r_plot,
    "mass_true": mass_true, "mass_fit": mass_fit,
    "acc_true": acc_true, "acc_fit": acc_fit,
    "rho_dm_true": rho_true, "rho_dm_fit": rho_fit,
    "inside_5_95_support": mask_5_95.astype(int),
    "inside_005_995_support": mask_005_995.astype(int),
})
profile_df.to_csv(os.path.join(CFG.output_dir, "final_profiles.csv"), index=False)

metrics["gravsphere_slope"] = slope_metrics

if SCAN_SUMMARY is not None:
    metrics["inner_slope_scan"] = SCAN_SUMMARY
    # Whether the scan determined its own answer, and -- under --scan-rs auto --
    # which R_s protocol was kept and why. Recorded for every run: the reported
    # argmin is meaningless when the span covers several grid steps, and nothing
    # else in the output says so.
    if SCAN_RS_CHOICE is not None:
        metrics["scan_resolution"] = SCAN_RS_CHOICE
    slope_metrics["scan"] = SCAN_SUMMARY

with open(os.path.join(CFG.output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ---- plots ----
def shade_support(ax, xlim):
    ax.axvspan(xlim[0], r_sup["r005"], color="0.9", alpha=0.7, zorder=0)
    ax.axvspan(r_sup["r995"], xlim[1], color="0.9", alpha=0.7, zorder=0)
    for k, ls in [("r05", ":"), ("r95", ":"), ("r50", "--")]:
        ax.axvline(r_sup[k], color="0.4", ls=ls, lw=0.9)


# Figure 1: training history
fig, axs = plt.subplots(2, 2, figsize=(14.5, 9.0))

# NLL
ax = axs[0, 0]
ax.plot(history["phaseA"]["nll"], label="phase A")
ax.plot(
    np.arange(len(history["phaseB"]["nll"])) + CFG.epochs_warmup,
    history["phaseB"]["nll"],
    label="phase B",
)
ax.set_xlabel("epoch")
ax.set_ylabel("NLL per star")
ax.set_title("Marginal likelihood (IWAE)")
ax.grid(alpha=0.3)
ax.legend()

# Projected anchor losses
ax = axs[0, 1]
ax.plot(history["phaseA"]["sig"], label=r"$\sigma_{\rm los}$ phase A")
ax.plot(
    np.arange(len(history["phaseB"]["sig"])) + CFG.epochs_warmup,
    history["phaseB"]["sig"],
    label=r"$\sigma_{\rm los}$ phase B",
)

if CFG.use_pm_anchor:
    ax.plot(history["phaseA"]["pm"], ls="--", label=r"$\sigma_{\rm pm}$ phase A")
    ax.plot(
        np.arange(len(history["phaseB"]["pm"])) + CFG.epochs_warmup,
        history["phaseB"]["pm"],
        ls="--",
        label=r"$\sigma_{\rm pm}$ phase B",
    )

if CFG.use_vsp_anchor:
    ax.plot(history["phaseA"]["vsp"], ls=":", label="VSP phase A")
    ax.plot(
        np.arange(len(history["phaseB"]["vsp"])) + CFG.epochs_warmup,
        history["phaseB"]["vsp"],
        ls=":",
        label="VSP phase B",
    )

ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("anchor loss")
ax.set_title("Projected anchor losses")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)

# ESS
ax = axs[1, 0]
ax.plot(history["phaseA"]["ess"], label="phase A")
ax.plot(
    np.arange(len(history["phaseB"]["ess"])) + CFG.epochs_warmup,
    history["phaseB"]["ess"],
    label="phase B",
)
ax.set_xlabel("epoch")
ax.set_ylabel(f"ESS / {CFG.n_iwae}")
ax.set_title("IWAE effective sample size")
ax.grid(alpha=0.3)
ax.legend()

# Mass-scale evolution
ax = axs[1, 1]
ax.plot(history["phaseB"]["M_at_Rs_true"], label="M(Rs_true) fit")
ax.axhline(np_mass_truth(CFG.R_S_NFW_TRUE), color="k", ls="--", label="truth")
ax.set_yscale("log")
ax.set_xlabel("phase B epoch")
ax.set_ylabel(r"$M(<R_s^{\rm true})$ [Msun]")
ax.set_title("Mass-scale evolution")
ax.grid(alpha=0.3, which="both")
ax.legend()

plt.tight_layout()
save_pub(fig, "loss_history")
plt.close(fig)

with torch.no_grad():
    if CFG.use_pm_anchor:
        pmR_dense, pmT_dense = model_sigma_pm_RT(R_dense, grids_f)
        pmR_dense = pmR_dense.cpu().numpy(); pmT_dense = pmT_dense.cpu().numpy()

# Figure 2: master physical diagnostics
fig, axs = plt.subplots(2, 3, figsize=(19, 10.5))
fig.suptitle("Joint f(E,L) + potential fit: physical diagnostics", fontsize=15)

ax = axs[0, 0]
ax.plot(r_plot, acc_true, "k-", lw=2, label=_TRUTH_LABEL)
ax.plot(r_plot, acc_fit, "r--", lw=2.4, label="fit")
shade_support(ax, (r_plot[0], r_plot[-1]))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$|a(r)|$ [(km/s)$^2$/kpc]")
ax.set_title("Acceleration"); ax.grid(alpha=0.3, which="both"); ax.legend()

ax = axs[0, 1]
ax.plot(r_plot, mass_true, "k-", lw=2, label=_TRUTH_LABEL)
ax.plot(r_plot, mass_fit, "r--", lw=2.4, label="fit")
ax.scatter([r3], [M_half_true], color="k", zorder=5, s=35,
           label="Wolf point (truth)")
ax.scatter([r3], [M_half_fit], color="r", zorder=5, s=35, marker="x",
           label="Wolf point (fit)")
shade_support(ax, (r_plot[0], r_plot[-1]))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$M(<r)$ [Msun]")
ax.set_title("Enclosed mass"); ax.grid(alpha=0.3, which="both"); ax.legend()

ax = axs[0, 2]
ax.plot(r_plot, rho_true, "k-", lw=2, label=f"{_TRUTH_LABEL} DM")
ax.plot(r_plot, np.maximum(rho_fit, 1e-30), "r--", lw=2.4, label="fit")
shade_support(ax, (r_plot[0], r_plot[-1]))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$\rho(r)$ [Msun/kpc$^3$]")
ax.set_title("Matter density"); ax.grid(alpha=0.3, which="both"); ax.legend()

ax = axs[1, 0]
ax.plot(r_plot, np_rho_plummer(r_plot), "k-", lw=2, label="Plummer truth")
ax.plot(r_gnp, rho_star_model, "b--", lw=2.2, label="DF-implied tracer")
shade_support(ax, (r_plot[0], r_plot[-1]))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(r_plot[0], r_plot[-1])
ax.set_ylim(1e-2, 5e8)
ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$\rho_\star(r)$")
ax.set_title("Tracer density from f(E,L)")
ax.grid(alpha=0.3, which="both"); ax.legend()

ax = axs[1, 1]
ax.errorbar(SIG_R_NP, SIG_T_NP, yerr=SIG_E_NP, fmt="o", color="tab:blue",
            capsize=2.5, label="data (noise-corrected)")
ax.plot(R_dense_np, sig_los_dense, "r-", lw=2.2, label="model projection")
ax.set_xscale("log")
ax.set_xlabel("R [kpc]"); ax.set_ylabel(r"$\sigma_{\rm los}$ [km/s]")
ax.set_title("Projected dispersion (FIX 2 anchor)")
if CFG.use_pm_anchor:
    ax.errorbar(PM_RT["Rc"], PM_RT["sR"], yerr=PM_RT["eSR"], fmt="s",
                color="tab:green", capsize=2.5, label=r"data $\sigma_{\rm pm,R}$")
    ax.errorbar(PM_RT["Rc"], PM_RT["sT"], yerr=PM_RT["eST"], fmt="^",
                color="tab:olive", capsize=2.5, label=r"data $\sigma_{\rm pm,T}$")
    ax.plot(R_dense_np, pmR_dense, color="tab:green", lw=2.0, ls="--",
            label=r"model $\sigma_{\rm pm,R}$")
    ax.plot(R_dense_np, pmT_dense, color="tab:olive", lw=2.0, ls=":",
            label=r"model $\sigma_{\rm pm,T}$")
ax.grid(alpha=0.3, which="both"); ax.legend()

ax = axs[1, 2]
ax.axhline(0.0, color="k", lw=1.8, label="isotropic truth")
ax.plot(r_gnp, beta_model, "b--", lw=2.2, label=r"model $\beta(r)$")
shade_support(ax, (r_plot[0], r_plot[-1]))
ax.set_xscale("log"); ax.set_ylim(-1.0, 1.0)
ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$\beta(r)$")
ax.set_title("Anisotropy of f(E,L)"); ax.grid(alpha=0.3, which="both"); ax.legend()

plt.tight_layout()
save_pub(fig, "master_physical_diagnostics")
plt.close(fig)

# Figure: VSP diagnostics
if CFG.use_vsp_anchor:
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))

    names = ["VSP1", "VSP2"]
    x = np.arange(2)

    # Panel 1: data/model comparison
    ax = axs[0]
    ax.errorbar(
        x - 0.08,
        VSP_OBS_NP,
        yerr=VSP_ERR_NP,
        fmt="o",
        capsize=4,
        label="data",
    )
    ax.scatter(
        x + 0.08,
        VSP_MODEL_NP,
        marker="s",
        s=55,
        label="model",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_yscale("symlog", linthresh=1.0e-6)
    ax.set_ylabel("VSP value")
    ax.set_title("VSP constraints: data vs model")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # Panel 2: normalized residuals
    ax = axs[1]
    ax.axhline(0.0, color="k", lw=1.2)
    ax.axhspan(-1.0, 1.0, alpha=0.15, label=r"$\pm 1\sigma$")
    ax.axhspan(-2.0, 2.0, alpha=0.08, label=r"$\pm 2\sigma$")

    ax.bar(x, VSP_PULL_NP, width=0.45)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"pull = $(\mathrm{model}-\mathrm{data})/\sigma$")
    ax.set_title("VSP normalized residuals")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()

    plt.tight_layout()
    save_pub(fig, "vsp_diagnostics")
    plt.close(fig)

# Figure 3: deconvolution check
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))
ax = axs[0]
if r_truth_np is not None:
    ax.scatter(r_truth_np, r_noisy_np, s=8, alpha=0.2, label="naive noisy")
    ax.scatter(r_truth_np, r_fit_np, s=8, alpha=0.4, label="posterior mean")
    rr = np.logspace(-2.3, 1.5, 50)
    ax.plot(rr, rr, "k--", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true r [kpc]"); ax.set_ylabel("reconstructed r [kpc]")
else:
    ax.hist(r_noisy_np, bins=40, alpha=0.4, label="naive noisy")
    ax.hist(r_fit_np, bins=40, alpha=0.4, label="posterior mean")
    ax.set_xlabel("r [kpc]")
ax.set_title("Deconvolution radius check"); ax.grid(alpha=0.3); ax.legend()

ax = axs[1]
if CFG.training_mode in ("observed", "projected"):
    ax.hist(ess_np, bins=40, color="tab:blue", alpha=0.7)
    ax.set_xlabel("posterior ESS (of 128)"); ax.set_ylabel("stars")
    ax.set_title("Importance-sampling ESS per star")
else:
    ax.text(0.5, 0.5, "truth mode: no deconvolution", ha="center",
            transform=ax.transAxes)
ax.grid(alpha=0.3)
plt.tight_layout()
save_pub(fig, "deconvolution_diagnostics")
plt.close(fig)

print("\n==================== FIT SUMMARY ====================")
for k, v in metrics.items():
    if k not in ("laplace",):
        print(f"{k}: {v}")


# ============================================================
# 12. OPTIONAL SBI / NPE STAGE
# ============================================================
def jeans_sigma_r_plummer_nfw(M0, Rs, a_star, r_grid):
    """Isotropic Jeans sigma_r for a Plummer tracer in an NFW halo (numpy)."""
    nu = (1.0 + (r_grid / a_star) ** 2) ** (-2.5)
    M = M0 * (np.log1p(r_grid / Rs) - (r_grid / Rs) / (1.0 + r_grid / Rs))
    integ = nu * CFG.G * M / np.maximum(r_grid**2, 1e-30)
    rev = cumulative_trapezoid(integ[::-1], r_grid[::-1], initial=0.0)
    J = -rev[::-1]
    sig2 = np.maximum(J / np.maximum(nu, 1e-300), 0.0)
    return np.sqrt(sig2)


def sbi_simulate(theta, err_pool, rng):
    """Fast forward model: Plummer positions, isotropic Maxwellian
    velocities with Jeans sigma_r(r), observer transform, bootstrapped
    observational noise. First two velocity moments match Jeans exactly;
    higher moments are approximate (Maxwellian)."""
    logM0, logRs, log_a = theta
    M0, Rs, a = math.exp(logM0), math.exp(logRs), math.exp(log_a)

    n = N
    # Plummer inverse CDF: P(<r) = r^3/(r^2+a^2)^{3/2}
    q = rng.random(n)
    u = q ** (2.0 / 3.0)
    r = a * np.sqrt(u / np.maximum(1.0 - u, 1e-12))
    r = np.clip(r, 1e-5, 100.0)

    rg = np.logspace(-4, np.log10(150.0), 220)
    sig_r = jeans_sigma_r_plummer_nfw(M0, Rs, a, rg)
    sig = np.interp(r, rg, sig_r)

    phi = 2 * np.pi * rng.random(n)
    ct = 2 * rng.random(n) - 1
    st = np.sqrt(1 - ct**2)
    pos = np.column_stack([r * st * np.cos(phi), r * st * np.sin(phi), r * ct])
    vel = rng.normal(size=(n, 3)) * sig[:, None]

    pos_abs = pos + np.asarray(CENTER_POS_NP)[None, :]
    vel_abs = vel + np.asarray(CENTER_VEL_NP)[None, :]
    RA, Dec, D, pmRA, pmDec, vlos = cartesian_to_observables_np(pos_abs, vel_abs)

    rows = rng.integers(0, len(err_pool), size=n)
    e = err_pool[rows]
    obs = {
        "RA": rng.normal(RA, e[:, 0]),
        "Dec": rng.normal(Dec, e[:, 1]),
        "vlos": rng.normal(vlos, e[:, 5]),
        "e_vlos": e[:, 5],
    }
    return obs


def sbi_summary(RA, Dec, vlos, e_vlos, R_edges):
    Rp = projected_radius_np(RA, Dec)
    w = 1.0 / np.maximum(e_vlos, 0.05) ** 2
    vbar = np.sum(w * vlos) / np.sum(w)
    dv = vlos - vbar

    qs = np.log10(np.maximum(np.quantile(Rp, np.linspace(0.1, 0.9, 9)), 1e-4))

    var_g = max(np.var(dv, ddof=1) - np.mean(e_vlos**2), CFG.sig_var_floor)
    sg = math.log10(math.sqrt(var_g))

    bins = []
    for i in range(len(R_edges) - 1):
        m = (Rp >= R_edges[i]) & (Rp < R_edges[i + 1])
        if m.sum() >= 10:
            v = dv[m] - np.mean(dv[m])
            var_c = max(np.var(v, ddof=1) - np.mean(e_vlos[m] ** 2),
                        CFG.sig_var_floor)
            bins.append(math.log10(math.sqrt(var_c)))
        else:
            bins.append(sg)
    return np.concatenate([qs, [sg], np.asarray(bins)])


def run_sbi():
    try:
        import zuko
    except ImportError:
        print("SBI skipped: `pip install zuko` to enable --run-sbi.")
        return None

    print(f"\n=== SBI/NPE stage: {CFG.sbi_n_sims} simulations ===")
    rng = np.random.default_rng(CFG.seed + 999)
    err_pool = obs_df[ERR_COLS].values

    R_edges = np.quantile(R_proj_np,
                          np.linspace(0.05, 0.95, CFG.sbi_n_summary_bins + 1))

    # priors (broad, log-uniform)
    lo = np.array([math.log(5.0e7), math.log(0.3), math.log(0.02)])
    hi = np.array([math.log(5.0e9), math.log(10.0), math.log(0.6)])

    thetas, summaries = [], []
    t0 = time.time()
    for i in range(CFG.sbi_n_sims):
        th = lo + (hi - lo) * rng.random(3)
        obs = sbi_simulate(th, err_pool, rng)
        s = sbi_summary(obs["RA"], obs["Dec"], obs["vlos"], obs["e_vlos"], R_edges)
        thetas.append(th)
        summaries.append(s)
        if (i + 1) % 500 == 0:
            print(f"  sims {i+1}/{CFG.sbi_n_sims} ({time.time()-t0:.0f}s)")
    thetas = np.asarray(thetas, dtype=np.float32)
    summaries = np.asarray(summaries, dtype=np.float32)

    s_data = sbi_summary(obs_df["RA"].values, obs_df["Dec"].values,
                         vlos_np, e_vlos_np, R_edges).astype(np.float32)

    th_mu, th_sd = thetas.mean(0), thetas.std(0) + 1e-8
    s_mu, s_sd = summaries.mean(0), summaries.std(0) + 1e-8
    th_n = torch.tensor((thetas - th_mu) / th_sd, device=DEVICE)
    s_n = torch.tensor((summaries - s_mu) / s_sd, device=DEVICE)
    s_obs_n = torch.tensor((s_data - s_mu) / s_sd, device=DEVICE)[None, :]

    flow = zuko.flows.NSF(features=3, context=s_n.shape[1],
                          transforms=3, hidden_features=[64, 64]).to(DEVICE)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    n_tr = int(0.9 * len(th_n))
    perm = torch.randperm(len(th_n), device=DEVICE)
    tr, va = perm[:n_tr], perm[n_tr:]

    best, best_state = 1e30, None
    for ep in range(CFG.sbi_epochs):
        flow.train()
        sh = tr[torch.randperm(len(tr), device=DEVICE)]
        for st in range(0, len(sh), 256):
            b = sh[st:st + 256]
            opt.zero_grad(set_to_none=True)
            loss = -flow(s_n[b]).log_prob(th_n[b]).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            flow.eval()
            vl = float(-flow(s_n[va]).log_prob(th_n[va]).mean())
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
        if (ep + 1) % 50 == 0:
            print(f"  NPE epoch {ep+1}/{CFG.sbi_epochs} | val NLL={vl:.4f}")
    if best_state is not None:
        flow.load_state_dict(best_state)

    with torch.no_grad():
        samp_n = flow(s_obs_n.expand(CFG.sbi_n_posterior, -1)).sample()
        samp = samp_n.cpu().numpy() * th_sd + th_mu

    M0_s = np.exp(samp[:, 0]); Rs_s = np.exp(samp[:, 1]); a_s = np.exp(samp[:, 2])
    out = pd.DataFrame({"M0": M0_s, "Rs": Rs_s, "a_star": a_s})
    out.to_csv(os.path.join(CFG.output_dir, "sbi_posterior_samples.csv"), index=False)

    def q3(x):
        return [float(np.percentile(x, p)) for p in (16, 50, 84)]
    res = {"M0_p16_p50_p84": q3(M0_s), "Rs_p16_p50_p84": q3(Rs_s),
           "a_star_p16_p50_p84": q3(a_s),
           "note": ("NPE with Maxwellian-Jeans simulator; "
                    "moment-level summaries only (R quantiles + corrected "
                    "sigma_los). Independent cross-check of the joint fit.")}
    print("SBI posterior (16/50/84%):")
    print(f"  M0 = {res['M0_p16_p50_p84']}  (truth {CFG.M_0_NFW_TRUE:.2e})")
    print(f"  Rs = {res['Rs_p16_p50_p84']}  (truth {CFG.R_S_NFW_TRUE})")
    print(f"  a* = {res['a_star_p16_p50_p84']}  (truth {CFG.A_STAR_TRUE})")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    axs[0].hist(np.log10(M0_s), bins=50, color="tab:blue", alpha=0.8)
    axs[0].axvline(np.log10(CFG.M_0_NFW_TRUE), color="k", ls="--", label="truth")
    axs[0].set_xlabel(r"$\log_{10} M_0$"); axs[0].legend()
    axs[1].hist(np.log10(Rs_s), bins=50, color="tab:blue", alpha=0.8)
    axs[1].axvline(np.log10(CFG.R_S_NFW_TRUE), color="k", ls="--")
    axs[1].set_xlabel(r"$\log_{10} R_s$")
    axs[2].hist2d(np.log10(M0_s), np.log10(Rs_s), bins=45, cmap="Blues")
    axs[2].scatter([np.log10(CFG.M_0_NFW_TRUE)], [np.log10(CFG.R_S_NFW_TRUE)],
                   color="r", marker="x", s=70, label="truth")
    axs[2].set_xlabel(r"$\log_{10} M_0$"); axs[2].set_ylabel(r"$\log_{10} R_s$")
    axs[2].legend()
    for a_ in axs:
        a_.grid(alpha=0.3)
    plt.suptitle("SBI/NPE posterior over halo parameters")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_pub(fig, "sbi_posterior")
    plt.close(fig)
    return res


sbi_res = run_sbi() if CFG.run_sbi else None

# ============================================================
# 13. FINAL SUMMARY
# ============================================================
summary = {
    "metrics": metrics,
    "sbi": sbi_res,
    "plots": {
        "loss_history_png": os.path.join(CFG.output_dir, "loss_history.png"),
        "loss_history_pdf": os.path.join(CFG.output_dir, "loss_history.pdf"),

        "master_physical_diagnostics_png": os.path.join(
            CFG.output_dir, "master_physical_diagnostics.png"),
        "master_physical_diagnostics_pdf": os.path.join(
            CFG.output_dir, "master_physical_diagnostics.pdf"),

        "deconvolution_diagnostics_png": os.path.join(
            CFG.output_dir, "deconvolution_diagnostics.png"),
        "deconvolution_diagnostics_pdf": os.path.join(
            CFG.output_dir, "deconvolution_diagnostics.pdf"),

        "method_png": os.path.join(
            CFG.output_dir, "gravsphere_comparison.png"),
        "method_pdf": os.path.join(
            CFG.output_dir, "gravsphere_comparison.pdf"),

        "vsp_diagnostics_png": (
            os.path.join(CFG.output_dir, "vsp_diagnostics.png")
            if CFG.use_vsp_anchor else None
        ),
        "vsp_diagnostics_pdf": (
            os.path.join(CFG.output_dir, "vsp_diagnostics.pdf")
            if CFG.use_vsp_anchor else None
        ),
        "sbi_posterior_png": (
            os.path.join(CFG.output_dir, "sbi_posterior.png")
            if CFG.run_sbi else None
        ),
        "sbi_posterior_pdf": (
            os.path.join(CFG.output_dir, "sbi_posterior.pdf")
            if CFG.run_sbi else None
        ),
    },
}
with open(os.path.join(CFG.output_dir, "fit_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nDone. Outputs in:", CFG.output_dir)
