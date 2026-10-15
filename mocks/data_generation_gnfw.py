import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.integrate import cumulative_trapezoid

# ============================================================
# Equilibrium noisy mock for a gNFW dwarf halo with TUNABLE
# inner slope -- the calibration ladder for the projected-mode
# slope estimator.
# ============================================================
#
# gNFW (Zhao 1996 with alpha=1, beta=3):
#     rho(r) = rho_s / [ (r/rs)^g (1 + r/rs)^(3-g) ]
#     inner: rho ~ r^-g          -> gamma0_truth = -g  EXACTLY
#     outer: rho ~ r^-3          -> M ~ ln r, Psi finite (as NFW/Burkert)
#     g = 1 reduces to NFW.
#
# DESIGN CHOICE THAT MAKES THIS A CALIBRATION LADDER:
# rho_s is solved for, per g, so that M(<r_piv) is IDENTICAL across
# the whole ladder, with r_piv = (4/3) a_star = the Wolf radius of a
# Plummer tracer. This is the same gauge the fitter uses, so the only
# thing varying along the ladder is the inner SHAPE -- the amplitude
# is held fixed. Without this, a slope calibration would be
# confounded by a simultaneous mass change.
#
# The default M_piv reproduces data_generation_nfw.py (M0=5e8, Rs=2)
# at g=1, so the ladder's cuspy endpoint IS the existing NFW class.
#
# Units: kpc, km/s, Msun, mas/yr, degrees. Psi(r) = -Phi(r), Psi(inf)=0.
# Output schema is byte-identical to the NFW/Burkert generators, so the
# fitter needs NO modification (it reads truth_profile.csv model-
# agnostically). Only the harness must learn the new class.

# ============================================================
# 1. CONFIGURATION
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="gNFW equilibrium dwarf mock with tunable inner slope.")

    p.add_argument("--output-dir", type=str,
                   default="realistic_mock_data_GNFW_equilibrium")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-stars", type=int, default=1000)

    # gNFW halo
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Inner logarithmic slope g in rho ~ r^-g. "
                        "gamma0_truth = -g. Ladder: 0, 0.25, 0.5, 0.75, 1.0. "
                        "Keep g < 2 (Eddington inversion degrades above that).")
    p.add_argument("--rs", type=float, default=2.0,
                   help="gNFW scale radius [kpc].")
    p.add_argument("--m-piv", type=float, default=1.0193e6,
                   help="Enclosed mass at r_piv=(4/3)a_star [Msun], held FIXED "
                        "across the ladder. Default reproduces the M0=5e8, "
                        "Rs=2 NFW mock at gamma=1.")
    p.add_argument("--rho-s", type=float, default=None,
                   help="Override: set rho_s [Msun/kpc^3] directly and ignore "
                        "--m-piv (breaks the matched-mass ladder; diagnostics only).")

    # Plummer tracer
    p.add_argument("--a-star", type=float, default=0.10,
                   help="Plummer scale radius of the stellar tracer [kpc]. "
                        "For Plummer the projected half-light radius is exactly a.")

    # comparison overlay only
    p.add_argument("--nfw-m0-companion", type=float, default=5.0e8)
    p.add_argument("--nfw-rs-companion", type=float, default=2.0)

    p.add_argument(
        "--noise-preset", type=str, default="learnable",
        choices=["learnable", "intermediate", "hard"],
        help="Observational-noise regime; identical definitions to the "
             "NFW/Burkert generators.")

    return p.parse_args()


ARGS = parse_args()

OUTPUT_DIR = ARGS.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = ARGS.seed
rng = np.random.default_rng(SEED)

N_STARS = ARGS.n_stars
M_STAR_TOT = 1.0e6
G = 4.3009e-6

GAM = float(ARGS.gamma)
RS = float(ARGS.rs)
A_STAR = float(ARGS.a_star)
R_PIV = (4.0 / 3.0) * A_STAR
M_PIV_TARGET = float(ARGS.m_piv)

if not (0.0 <= GAM < 2.5):
    raise ValueError(f"--gamma must be in [0, 2.5); got {GAM}")

NFW_M0_CMP = ARGS.nfw_m0_companion
NFW_RS_CMP = ARGS.nfw_rs_companion
NOISE_PRESET = ARGS.noise_preset

# --- Observer/systemic parameters (identical to the other generators) ---
RA_CENTER = 260.0
DEC_CENTER = 57.0
DIST_CENTER = 80.0
V_SYS_LOS = -290.0
PM_SYS_RA = 0.04
PM_SYS_DEC = -0.15
KAPPA = 4.74047  # km/s = KAPPA * mas/yr * kpc

if NOISE_PRESET == "learnable":
    ERR_ANGLE_MAS_MIN, ERR_ANGLE_MAS_MAX = 0.005, 0.050
    ERR_DIST_FRAC_MIN, ERR_DIST_FRAC_MAX = 0.005, 0.030
    ERR_PM_MAS_YR_MIN, ERR_PM_MAS_YR_MAX = 0.002, 0.020
    ERR_VLOS_MIN, ERR_VLOS_MAX = 0.2, 2.5
elif NOISE_PRESET == "intermediate":
    ERR_ANGLE_MAS_MIN, ERR_ANGLE_MAS_MAX = 0.010, 0.100
    ERR_DIST_FRAC_MIN, ERR_DIST_FRAC_MAX = 0.030, 0.100
    ERR_PM_MAS_YR_MIN, ERR_PM_MAS_YR_MAX = 0.010, 0.100
    ERR_VLOS_MIN, ERR_VLOS_MAX = 0.5, 5.0
else:
    ERR_ANGLE_MAS_MIN, ERR_ANGLE_MAS_MAX = 0.010, 0.200
    ERR_DIST_FRAC_MIN, ERR_DIST_FRAC_MAX = 0.100, 0.300
    ERR_PM_MAS_YR_MIN, ERR_PM_MAS_YR_MAX = 0.020, 0.400
    ERR_VLOS_MIN, ERR_VLOS_MAX = 0.5, 8.0

# Numerical settings (mirrors the Burkert generator)
R_DER_MIN, R_DER_MAX, N_R_DER = 1.0e-6, 1.0e6, 8000
R_SAMPLE_MIN, R_SAMPLE_MAX, N_R_SAMPLE = 1.0e-4, 1.0e2, 1200
N_EPS_GRID = 1400
N_U_EDDINGTON = 450
N_V_INT = 360
N_V_REJECTION_GRID = 280
R_PSI_MIN, R_PSI_MAX, N_R_PSI = 1.0e-6, 1.0e6, 9000
# mass-integral grid (dimensionless u = r/rs); very wide so no extrapolation
U_MASS_MIN, U_MASS_MAX, N_U_MASS = 1.0e-12, 1.0e12, 14000


# ============================================================
# 2. ANALYTIC gNFW HALO + TARGET TRACER
# ============================================================
def _build_mass_integral(gam):
    """I(x) = int_0^x u^(2-g) (1+u)^(g-3) du.

    Evaluated on a log-u grid with the exact inner seed
        I(u0) -> u0^(3-g)/(3-g)   for u0 << 1,
    which removes the r->0 integrable singularity analytically.
    Validated against ln(1+x) - x/(1+x) at g=1 to ~5e-6 relative.
    """
    u = np.logspace(np.log10(U_MASS_MIN), np.log10(U_MASS_MAX), N_U_MASS)
    integ = u ** (2.0 - gam) * (1.0 + u) ** (gam - 3.0) * u   # d(ln u) measure
    I0 = U_MASS_MIN ** (3.0 - gam) / (3.0 - gam)
    I = I0 + cumulative_trapezoid(integ, np.log(u), initial=0.0)
    return u, I, PchipInterpolator(np.log(u), np.log(I), extrapolate=False)


_U_M, _I_M, _I_INTERP = _build_mass_integral(GAM)


def _mass_integral(x):
    """I(x), with analytic asymptotics outside the tabulated range."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    lo = x < U_MASS_MIN
    hi = x > U_MASS_MAX
    mid = ~(lo | hi)
    out[lo] = x[lo] ** (3.0 - GAM) / (3.0 - GAM)
    if mid.any():
        out[mid] = np.exp(_I_INTERP(np.log(x[mid])))
    # large-x: rho -> rho_s (rs/r)^3  =>  I(x) -> I(Umax) + ln(x/Umax)
    out[hi] = _I_M[-1] + np.log(x[hi] / U_MASS_MAX)
    return out


# --- solve rho_s so that M(<r_piv) is the same for every gamma ---
_I_PIV = float(_mass_integral(R_PIV / RS)[0])
if ARGS.rho_s is not None:
    RHO_S = float(ARGS.rho_s)
    M_PIV_ACTUAL = 4.0 * np.pi * RHO_S * RS ** 3 * _I_PIV
else:
    RHO_S = M_PIV_TARGET / (4.0 * np.pi * RS ** 3 * _I_PIV)
    M_PIV_ACTUAL = M_PIV_TARGET

# amplitude of the logarithmic mass tail: M -> M(R) + M_TAIL*ln(r/R)
M_TAIL = 4.0 * np.pi * RHO_S * RS ** 3


def mass_gnfw(r):
    r = np.asarray(r, dtype=float)
    return M_TAIL * _mass_integral(r / RS).reshape(np.shape(r))


def rho_gnfw(r):
    x = np.maximum(np.asarray(r, dtype=float) / RS, 1.0e-30)
    return RHO_S / (x ** GAM * (1.0 + x) ** (3.0 - GAM))


def dlnrho_dlnr_gnfw(r):
    """Analytic 3D log-density slope: -g - (3-g) x/(1+x).
    Inner limit -> -g exactly; outer limit -> -3."""
    x = np.maximum(np.asarray(r, dtype=float) / RS, 1.0e-30)
    return -GAM - (3.0 - GAM) * x / (1.0 + x)


def force_gnfw(r):
    r = np.asarray(r, dtype=float)
    return G * mass_gnfw(r) / np.maximum(r * r, 1.0e-30)


def _build_psi():
    """Psi(r) = G int_r^inf M(s)/s^2 ds, with the analytic log tail
        int_R^inf [M(R) + M_TAIL ln(s/R)]/s^2 ds = [M(R) + M_TAIL]/R .
    """
    rg = np.logspace(np.log10(R_PSI_MIN), np.log10(R_PSI_MAX), N_R_PSI)
    integ = G * mass_gnfw(rg) / rg ** 2
    rev = cumulative_trapezoid(integ[::-1], rg[::-1], initial=0.0)
    psi_out = -rev[::-1]
    tail = G * (mass_gnfw(np.array([rg[-1]]))[0] + M_TAIL) / rg[-1]
    psi_grid = np.maximum(psi_out + tail, 0.0)
    return rg, psi_grid, PchipInterpolator(rg, psi_grid, extrapolate=False)


_PSI_RG, _PSI_GRID, _PSI_INTERP = _build_psi()
PSI_CENTRAL = float(_PSI_GRID[0])


def psi_gnfw(r):
    r = np.clip(np.asarray(r, dtype=float), _PSI_RG[0], _PSI_RG[-1])
    return np.nan_to_num(_PSI_INTERP(r), nan=0.0,
                         posinf=PSI_CENTRAL, neginf=0.0)


def dpsi_dr_gnfw(r):
    """Exact: dPsi/dr = -G M(r)/r^2 (analytic, no differencing)."""
    r = np.maximum(np.asarray(r, dtype=float), 1.0e-30)
    return -G * mass_gnfw(r) / r ** 2


def v_escape(r):
    return np.sqrt(2.0 * psi_gnfw(r))


def nu_plummer_shape(r):
    r = np.asarray(r)
    return (1.0 + (r / A_STAR) ** 2) ** (-2.5)


def dnu_plummer_dr(r):
    r = np.asarray(r)
    return -5.0 * r / A_STAR ** 2 * (1.0 + (r / A_STAR) ** 2) ** (-3.5)


def rho_plummer_mass_normalized(r):
    return (3.0 * M_STAR_TOT / (4.0 * np.pi * A_STAR ** 3)) * nu_plummer_shape(r)


def mass_nfw_cmp(r):
    x = np.asarray(r) / NFW_RS_CMP
    return NFW_M0_CMP * (np.log1p(x) - x / (1.0 + x))


def rho_nfw_cmp(r):
    x = np.maximum(np.asarray(r) / NFW_RS_CMP, 1.0e-30)
    return (NFW_M0_CMP / (4.0 * np.pi * NFW_RS_CMP ** 3)) / (x * (1.0 + x) ** 2)


SLOPE_AT_RHALF_TRUE = float(dlnrho_dlnr_gnfw(np.array([A_STAR]))[0])

print("=" * 68)
print(f"gNFW equilibrium mock | gamma={GAM:.3f} | preset={NOISE_PRESET!r}")
print(f"  rho_s={RHO_S:.4e} Msun/kpc^3 | rs={RS} kpc | a_star={A_STAR} kpc")
print(f"  r_piv=(4/3)a_star={R_PIV:.5f} kpc | M(<r_piv)={M_PIV_ACTUAL:.5e} Msun"
      f"  [matched across the ladder]")
print(f"  gamma0_truth = {-GAM:+.3f}   slope at R_1/2 = {SLOPE_AT_RHALF_TRUE:+.4f}")
print("=" * 68)


# ============================================================
# 3. EDDINGTON INVERSION -> POSITIVE EQUILIBRIUM DF f(epsilon)
# ============================================================
def safe_eval(interp, x):
    return np.nan_to_num(interp(x), nan=0.0, posinf=0.0, neginf=0.0)


def build_eddington_df():
    print("Building isotropic equilibrium DF (Eddington inversion, gNFW)...")
    r_der = np.logspace(np.log10(R_DER_MIN), np.log10(R_DER_MAX), N_R_DER)
    psi = psi_gnfw(r_der)
    dpsi = dpsi_dr_gnfw(r_der)                     # analytic, exact

    dnu_dpsi = dnu_plummer_dr(r_der) / dpsi
    d_dnu_dpsi_dr = np.gradient(dnu_dpsi, np.log(r_der), edge_order=2) / r_der
    d2nu_dpsi2 = d_dnu_dpsi_dr / dpsi

    psi_inc = psi[::-1]
    d2_inc = d2nu_dpsi2[::-1]
    mask = np.isfinite(psi_inc) & np.isfinite(d2_inc) & (psi_inc >= 0.0)
    psi_inc, d2_inc = psi_inc[mask], d2_inc[mask]
    order = np.argsort(psi_inc)
    psi_inc, d2_inc = psi_inc[order], d2_inc[order]
    uniq = np.r_[True, np.diff(psi_inc) > 0]
    psi_inc, d2_inc = psi_inc[uniq], d2_inc[uniq]

    d2_interp = PchipInterpolator(psi_inc, d2_inc, extrapolate=False)

    eps_max = psi_inc[-1] * (1.0 - 1.0e-8)
    eps_grid = np.linspace(0.0, eps_max, N_EPS_GRID)
    f_raw = np.zeros_like(eps_grid)
    prefactor = 1.0 / (np.sqrt(8.0) * np.pi ** 2)
    for j, eps in enumerate(eps_grid[1:], start=1):
        u = np.linspace(0.0, np.sqrt(eps), N_U_EDDINGTON)
        f_raw[j] = prefactor * 2.0 * np.trapezoid(
            safe_eval(d2_interp, eps - u * u), u)

    neg_fraction = float(np.mean(f_raw < 0.0))
    f_pos = np.maximum(f_raw, 0.0)
    print(f"  Raw DF min/max: {f_raw.min():.3e} / {f_raw.max():.3e}")
    print(f"  Negative grid points clipped: {100.0 * neg_fraction:.3f}%")
    if neg_fraction > 0.02:
        print("  *** WARNING: >2% of the DF grid was negative. The Plummer "
              "tracer is not comfortably supported by this potential; the "
              "sampled system will deviate from exact equilibrium. Consider "
              "a smaller --gamma or a larger --a-star. ***")

    f_interp = PchipInterpolator(eps_grid, f_pos, extrapolate=False)

    def f_eps(eps):
        return safe_eval(f_interp, np.clip(np.asarray(eps), 0.0, eps_grid[-1]))

    return eps_grid, f_raw, f_pos, f_eps


eps_grid, f_raw_grid, f_pos_grid, f_eps = build_eddington_df()


# ============================================================
# 4. DF-IMPLIED TRACER DENSITY AND RADIAL CDF
# ============================================================
def compute_nu_from_df(r_grid):
    nu = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        psi = float(psi_gnfw(r))
        vmax = np.sqrt(2.0 * psi)
        v = np.linspace(0.0, vmax, N_V_INT)
        nu[i] = 4.0 * np.pi * np.trapezoid(
            v * v * f_eps(psi - 0.5 * v * v), v)
    return nu


print("Computing DF-implied tracer density and radial CDF...")
r_sample_grid = np.logspace(np.log10(R_SAMPLE_MIN), np.log10(R_SAMPLE_MAX),
                            N_R_SAMPLE)
nu_df_shape_grid = compute_nu_from_df(r_sample_grid)

radial_pdf = 4.0 * np.pi * r_sample_grid ** 2 * nu_df_shape_grid
radial_cdf = cumulative_trapezoid(radial_pdf, r_sample_grid, initial=0.0)
radial_norm = radial_cdf[-1]
if not np.isfinite(radial_norm) or radial_norm <= 0.0:
    raise RuntimeError("Invalid radial normalization. Check DF construction.")
radial_cdf /= radial_norm

inv_radial_cdf = interp1d(radial_cdf, r_sample_grid, bounds_error=False,
                          fill_value=(r_sample_grid[0], r_sample_grid[-1]),
                          assume_sorted=True)

nu_df_mass_scale = M_STAR_TOT / radial_norm
rho_df_grid = nu_df_mass_scale * nu_df_shape_grid

r_check_mask = (r_sample_grid > 0.01) & (r_sample_grid < 10.0)
plum_grid_mass = rho_plummer_mass_normalized(r_sample_grid)
rel_shape = rho_df_grid[r_check_mask] / plum_grid_mass[r_check_mask]
print("  DF-implied tracer / Plummer target over 0.01-10 kpc: "
      f"min={np.nanmin(rel_shape):.4f}, median={np.nanmedian(rel_shape):.4f}, "
      f"max={np.nanmax(rel_shape):.4f}")
print("  Radial CDF quantiles: "
      f"r50={float(inv_radial_cdf(0.5)):.3f} kpc, "
      f"r90={float(inv_radial_cdf(0.9)):.3f} kpc, "
      f"r99={float(inv_radial_cdf(0.99)):.3f} kpc")


# ============================================================
# 5. SAMPLE EQUILIBRIUM PHASE-SPACE POINTS
# ============================================================
def sample_speed_given_radius(r, rng):
    psi = float(psi_gnfw(r))
    vmax = np.sqrt(2.0 * psi)
    v_grid = np.linspace(0.0, vmax, N_V_REJECTION_GRID)
    q_grid = v_grid * v_grid * f_eps(psi - 0.5 * v_grid * v_grid)
    q_max = float(np.max(q_grid))
    if not np.isfinite(q_max) or q_max <= 0.0:
        raise RuntimeError(f"Bad velocity proposal at r={r:.6e}, psi={psi:.6e}")
    for _ in range(20000):
        v_trial = rng.random() * vmax
        q_trial = v_trial * v_trial * float(
            np.asarray(f_eps(psi - 0.5 * v_trial * v_trial)))
        if rng.random() * q_max <= q_trial:
            return v_trial
    q_grid = np.maximum(q_grid, 0.0)
    cdf_v = cumulative_trapezoid(q_grid, v_grid, initial=0.0)
    if cdf_v[-1] <= 0.0:
        raise RuntimeError("Velocity fallback failed: zero local CDF.")
    cdf_v /= cdf_v[-1]
    return float(np.interp(rng.random(), cdf_v, v_grid))


print(f"Sampling {N_STARS} equilibrium stars...")
r = inv_radial_cdf(rng.random(N_STARS))

phi_pos = 2.0 * np.pi * rng.random(N_STARS)
cos_theta_pos = 2.0 * rng.random(N_STARS) - 1.0
sin_theta_pos = np.sqrt(1.0 - cos_theta_pos ** 2)
x_gal = r * sin_theta_pos * np.cos(phi_pos)
y_gal = r * sin_theta_pos * np.sin(phi_pos)
z_gal = r * cos_theta_pos
pos_gal = np.column_stack([x_gal, y_gal, z_gal])

speed = np.array([sample_speed_given_radius(ri, rng) for ri in r])
phi_vel = 2.0 * np.pi * rng.random(N_STARS)
cos_theta_vel = 2.0 * rng.random(N_STARS) - 1.0
sin_theta_vel = np.sqrt(1.0 - cos_theta_vel ** 2)
vx_gal = speed * sin_theta_vel * np.cos(phi_vel)
vy_gal = speed * sin_theta_vel * np.sin(phi_vel)
vz_gal = speed * cos_theta_vel
vel_gal = np.column_stack([vx_gal, vy_gal, vz_gal])

psi_star = psi_gnfw(r)
epsilon = psi_star - 0.5 * speed ** 2
energy = 0.5 * speed ** 2 - psi_star
if np.any(epsilon <= 0.0):
    raise RuntimeError("Generated unbound star. This should not happen.")


# ============================================================
# 6. TRANSFORM TO OBSERVER FRAME
# ============================================================
def center_cartesian_and_velocity():
    ra_c = np.deg2rad(RA_CENTER)
    dec_c = np.deg2rad(DEC_CENTER)
    X0 = DIST_CENTER * np.cos(dec_c) * np.cos(ra_c)
    Y0 = DIST_CENTER * np.cos(dec_c) * np.sin(ra_c)
    Z0 = DIST_CENTER * np.sin(dec_c)
    v_sys_perp_ra = PM_SYS_RA * KAPPA * DIST_CENTER
    v_sys_perp_dec = PM_SYS_DEC * KAPPA * DIST_CENTER
    VX0 = (V_SYS_LOS * np.cos(dec_c) * np.cos(ra_c)
           - v_sys_perp_ra * np.sin(ra_c)
           - v_sys_perp_dec * np.sin(dec_c) * np.cos(ra_c))
    VY0 = (V_SYS_LOS * np.cos(dec_c) * np.sin(ra_c)
           + v_sys_perp_ra * np.cos(ra_c)
           - v_sys_perp_dec * np.sin(dec_c) * np.sin(ra_c))
    VZ0 = V_SYS_LOS * np.sin(dec_c) + v_sys_perp_dec * np.cos(dec_c)
    return np.array([X0, Y0, Z0]), np.array([VX0, VY0, VZ0])


def cartesian_to_observables(pos_abs, vel_abs):
    X, Y, Z = pos_abs[:, 0], pos_abs[:, 1], pos_abs[:, 2]
    D = np.sqrt(X * X + Y * Y + Z * Z)
    RA = np.arctan2(Y, X)
    Dec = np.arcsin(Z / D)
    r_hat = np.column_stack([np.cos(Dec) * np.cos(RA),
                             np.cos(Dec) * np.sin(RA), np.sin(Dec)])
    ra_hat = np.column_stack([-np.sin(RA), np.cos(RA), np.zeros_like(RA)])
    dec_hat = np.column_stack([-np.sin(Dec) * np.cos(RA),
                               -np.sin(Dec) * np.sin(RA), np.cos(Dec)])
    V = vel_abs
    v_los = np.sum(V * r_hat, axis=1)
    pm_ra = np.sum(V * ra_hat, axis=1) / (KAPPA * D)
    pm_dec = np.sum(V * dec_hat, axis=1) / (KAPPA * D)
    return np.rad2deg(RA) % 360.0, np.rad2deg(Dec), D, pm_ra, pm_dec, v_los


def obs_to_galactocentric_np(ra_deg, dec_deg, dist, pm_ra, pm_dec, v_los):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    X = dist * np.cos(dec) * np.cos(ra)
    Y = dist * np.cos(dec) * np.sin(ra)
    Z = dist * np.sin(dec)
    center_pos, center_vel = center_cartesian_and_velocity()
    pos = np.column_stack([X, Y, Z]) - center_pos[None, :]
    r_hat = np.column_stack([np.cos(dec) * np.cos(ra),
                             np.cos(dec) * np.sin(ra), np.sin(dec)])
    ra_hat = np.column_stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)])
    dec_hat = np.column_stack([-np.sin(dec) * np.cos(ra),
                               -np.sin(dec) * np.sin(ra), np.cos(dec)])
    v_ra_phys = pm_ra * KAPPA * dist
    v_dec_phys = pm_dec * KAPPA * dist
    vel_abs = (v_los[:, None] * r_hat + v_ra_phys[:, None] * ra_hat
               + v_dec_phys[:, None] * dec_hat)
    vel = vel_abs - center_vel[None, :]
    return pos, vel


center_pos, center_vel = center_cartesian_and_velocity()
pos_abs = center_pos[None, :] + pos_gal
vel_abs = center_vel[None, :] + vel_gal
(RA_true, Dec_true, D_true,
 pmRA_true, pmDec_true, vlos_true) = cartesian_to_observables(pos_abs, vel_abs)

ra_c_rad = np.deg2rad(RA_CENTER)
dec_c_rad = np.deg2rad(DEC_CENTER)
cos_ang = (np.sin(np.deg2rad(Dec_true)) * np.sin(dec_c_rad)
           + np.cos(np.deg2rad(Dec_true)) * np.cos(dec_c_rad)
           * np.cos(np.deg2rad(RA_true) - ra_c_rad))
R_proj = DIST_CENTER * np.arccos(np.clip(cos_ang, -1.0, 1.0))


# ============================================================
# 7. HETEROSCEDASTIC OBSERVATIONAL NOISE
# ============================================================
print(f"Adding heteroscedastic noise (preset={NOISE_PRESET!r})...")
faintness = rng.gamma(shape=2.0, scale=1.0, size=N_STARS)
faintness = faintness / np.max(faintness)

err_angle_mas = ERR_ANGLE_MAS_MIN + faintness * (ERR_ANGLE_MAS_MAX - ERR_ANGLE_MAS_MIN)
err_dist_frac = ERR_DIST_FRAC_MIN + faintness * (ERR_DIST_FRAC_MAX - ERR_DIST_FRAC_MIN)
err_pm = ERR_PM_MAS_YR_MIN + faintness * (ERR_PM_MAS_YR_MAX - ERR_PM_MAS_YR_MIN)
err_vlos = ERR_VLOS_MIN + faintness * (ERR_VLOS_MAX - ERR_VLOS_MIN)

cos_dec_safe = np.maximum(np.cos(np.deg2rad(Dec_true)), 0.05)
err_ra = (err_angle_mas / 3.6e6) / cos_dec_safe
err_dec = err_angle_mas / 3.6e6
err_dist = D_true * err_dist_frac
err_pm_ra = err_pm.copy()
err_pm_dec = err_pm.copy()
err_v_los = err_vlos.copy()

obs_ra = rng.normal(RA_true, err_ra)
obs_dec = rng.normal(Dec_true, err_dec)
obs_dist = rng.normal(D_true, err_dist)
obs_pm_ra = rng.normal(pmRA_true, err_pm_ra)
obs_pm_dec = rng.normal(pmDec_true, err_pm_dec)
obs_vlos = rng.normal(vlos_true, err_v_los)

bad_d = obs_dist <= 0.0
if np.any(bad_d):
    obs_dist[bad_d] = D_true[bad_d]

pos_noisy_gal, vel_noisy_gal = obs_to_galactocentric_np(
    obs_ra, obs_dec, obs_dist, obs_pm_ra, obs_pm_dec, obs_vlos)
r_noisy = np.linalg.norm(pos_noisy_gal, axis=1)


# ============================================================
# 8. SAVE DATA PRODUCTS
# ============================================================
print("Saving CSV files...")
observed_df = pd.DataFrame({
    "RA": obs_ra, "Dec": obs_dec, "Dist": obs_dist,
    "pmRA": obs_pm_ra, "pmDec": obs_pm_dec, "v_los": obs_vlos,
    "e_RA": err_ra, "e_Dec": err_dec, "e_Dist": err_dist,
    "e_pmRA": err_pm_ra, "e_pmDec": err_pm_dec, "e_v_los": err_v_los,
    "faintness": faintness,
})
truth_gal_df = pd.DataFrame({
    "x": x_gal, "y": y_gal, "z": z_gal,
    "vx": vx_gal, "vy": vy_gal, "vz": vz_gal,
    "r": r, "v": speed, "epsilon": epsilon, "E": energy, "R_proj": R_proj,
})
truth_obs_df = pd.DataFrame({
    "RA_true": RA_true, "Dec_true": Dec_true, "Dist_true": D_true,
    "pmRA_true": pmRA_true, "pmDec_true": pmDec_true, "v_los_true": vlos_true,
})

# filenames identical to the NFW/Burkert generators -> fitter unchanged
observed_csv = os.path.join(OUTPUT_DIR, "draco_nfw_observations.csv")
truth_gal_csv = os.path.join(OUTPUT_DIR, "draco_nfw_truth_galactocentric.csv")
truth_obs_csv = os.path.join(OUTPUT_DIR, "draco_nfw_truth_observer.csv")
observed_df.to_csv(observed_csv, index=False)
truth_gal_df.to_csv(truth_gal_csv, index=False)
truth_obs_df.to_csv(truth_obs_csv, index=False)

r_truth_prof = np.logspace(-3, np.log10(50.0), 500)
truth_profile_df = pd.DataFrame({
    "r": r_truth_prof,
    "mass": mass_gnfw(r_truth_prof),
    "rho": rho_gnfw(r_truth_prof),
    "acc": force_gnfw(r_truth_prof),
    "dlnrho_dlnr": dlnrho_dlnr_gnfw(r_truth_prof),
})
truth_profile_csv = os.path.join(OUTPUT_DIR, "truth_profile.csv")
truth_profile_df.to_csv(truth_profile_csv, index=False)

R_half_2d_plummer = float(A_STAR)
R_half_2d_empirical = float(np.median(R_proj))
r05_3d = float(inv_radial_cdf(0.05))
r95_3d = float(inv_radial_cdf(0.95))

metadata = {
    "halo_type": "gnfw",
    "gamma_gnfw": GAM,
    # THE calibration targets -- the harness should harvest these
    "gamma0_truth": -GAM,
    "slope_at_Rhalf_truth": SLOPE_AT_RHALF_TRUE,
    "seed": SEED, "n_stars": N_STARS, "G": G,
    "RHO_S": RHO_S, "RS": RS,
    "r_piv": R_PIV, "M_piv": M_PIV_ACTUAL,
    "m_piv_matched_across_ladder": ARGS.rho_s is None,
    "A_STAR_target_plummer": A_STAR, "M_STAR_TOT_plot_only": M_STAR_TOT,
    "RA_CENTER": RA_CENTER, "DEC_CENTER": DEC_CENTER,
    "DIST_CENTER": DIST_CENTER,
    "V_SYS_LOS": V_SYS_LOS, "PM_SYS_RA": PM_SYS_RA, "PM_SYS_DEC": PM_SYS_DEC,
    "KAPPA": KAPPA, "noise_preset": NOISE_PRESET,
    "companion_nfw": {"M0": NFW_M0_CMP, "Rs": NFW_RS_CMP},
    "R_half_2d_plummer": R_half_2d_plummer,
    "R_half_2d_empirical": R_half_2d_empirical,
    "tracer_support": {
        "r05_3d": r05_3d, "r95_3d": r95_3d,
        "r_inner_tracer_3d": float(np.min(r)),
        "r_outer_tracer_3d": float(np.max(r)),
    },
    "benchmark_radial_range_kpc": [0.1 * R_half_2d_empirical,
                                   10.0 * R_half_2d_empirical],
    "noise": {
        "ERR_ANGLE_MAS_MIN": ERR_ANGLE_MAS_MIN, "ERR_ANGLE_MAS_MAX": ERR_ANGLE_MAS_MAX,
        "ERR_DIST_FRAC_MIN": ERR_DIST_FRAC_MIN, "ERR_DIST_FRAC_MAX": ERR_DIST_FRAC_MAX,
        "ERR_PM_MAS_YR_MIN": ERR_PM_MAS_YR_MIN, "ERR_PM_MAS_YR_MAX": ERR_PM_MAS_YR_MAX,
        "ERR_VLOS_MIN": ERR_VLOS_MIN, "ERR_VLOS_MAX": ERR_VLOS_MAX,
    },
    "df_diagnostics": {
        "raw_df_min": float(np.min(f_raw_grid)),
        "raw_df_max": float(np.max(f_raw_grid)),
        "fraction_negative_clipped": float(np.mean(f_raw_grid < 0.0)),
        "df_implied_over_plummer_0p01_10kpc_median": float(np.nanmedian(rel_shape)),
    },
}
with open(os.path.join(OUTPUT_DIR, "mock_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Observed noisy data:        {observed_csv}")
print(f"  Truth galactocentric:       {truth_gal_csv}")
print(f"  Truth observer frame:       {truth_obs_csv}")
print(f"  Model-agnostic truth table: {truth_profile_csv}")
print(f"  gamma0_truth={-GAM:+.3f} | slope@R_1/2={SLOPE_AT_RHALF_TRUE:+.4f} | "
      f"R_1/2(emp)={R_half_2d_empirical:.4f} kpc")


# ============================================================
# 9. DIAGNOSTIC PLOTS
# ============================================================
print("Building diagnostic plots...")


def _latex_usable():
    if os.environ.get("MOCK_NO_TEX", "") == "1":
        return False
    if shutil.which("latex") is None:
        return False
    try:
        import matplotlib as _mpl
        _mpl.rcParams["text.usetex"] = True
        _f = plt.figure()
        _f.text(0.5, 0.5, r"$\rho_\star$")
        import io as _io
        _f.savefig(_io.BytesIO(), format="png")
        plt.close(_f)
        return True
    except Exception:
        plt.close("all")
        matplotlib.rcParams["text.usetex"] = False
        return False


USE_TEX = _latex_usable()
plt.rcParams.update({
    "text.usetex": USE_TEX, "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.linewidth": 0.9, "axes.labelsize": 13.0, "axes.titlesize": 13.5,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 10.5,
    "figure.titlesize": 14.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 5.5, "ytick.major.size": 5.5,
    "legend.frameon": False, "axes.grid": False, "savefig.dpi": 300,
})
c_theory, c_data, c_noise, c_aux = "#1f77b4", "#2c3e50", "#e74c3c", "#27ae60"
c_base_line, c_cusp = "#b2bec3", "#8e44ad"

bins = np.logspace(np.log10(R_SAMPLE_MIN), np.log10(20.0), 34)
centers = np.sqrt(bins[:-1] * bins[1:])
rho_sample = np.full_like(centers, np.nan)
sigma_r_sample = np.full_like(centers, np.nan)
sigma_t_sample = np.full_like(centers, np.nan)
counts_sample = np.zeros_like(centers)

r_norm = np.linalg.norm(pos_gal, axis=1)
vr = np.sum(pos_gal * vel_gal, axis=1) / np.maximum(r_norm, 1.0e-12)
vt2 = np.sum(vel_gal * vel_gal, axis=1) - vr * vr
for i in range(len(centers)):
    m = (r_norm >= bins[i]) & (r_norm < bins[i + 1])
    n = int(np.sum(m))
    counts_sample[i] = n
    if n >= 15:
        vol = 4.0 / 3.0 * np.pi * (bins[i + 1] ** 3 - bins[i] ** 3)
        rho_sample[i] = (n / N_STARS) * M_STAR_TOT / vol
        sigma_r_sample[i] = np.std(vr[m] - np.mean(vr[m]))
        sigma_t_sample[i] = np.sqrt(0.5 * np.mean(vt2[m]))

r_jeans = r_sample_grid.copy()
nu_jeans = np.maximum(nu_df_shape_grid, 1.0e-300)
integ_j = nu_jeans * G * mass_gnfw(r_jeans) / (r_jeans * r_jeans)
rev_j = cumulative_trapezoid(integ_j[::-1], r_jeans[::-1], initial=0.0)
sigma_r_jeans = np.sqrt(np.maximum(-rev_j[::-1] / nu_jeans, 0.0))

fig, axs = plt.subplots(2, 2, figsize=(12.0, 9.2), constrained_layout=True)

ax = axs[0, 0]
ax.plot(r_sample_grid, rho_plummer_mass_normalized(r_sample_grid),
        color=c_base_line, lw=4.0, label=r"Target Plummer $\rho_\star$")
ax.plot(r_sample_grid, rho_df_grid, color=c_theory, lw=1.5,
        label=r"DF-implied $\rho_{\mathrm{DF}}$")
v = np.isfinite(rho_sample)
ax.errorbar(centers[v], rho_sample[v],
            yerr=rho_sample[v] / np.sqrt(counts_sample[v]),
            fmt="o", color=c_data, ms=4, elinewidth=0.9, capsize=1.5,
            markeredgecolor="black", markeredgewidth=0.5, label="Mock shells")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(5e-3, 3.0); ax.set_ylim(1e1, 8e8)
ax.set_xlabel(r"Galactocentric radius $r$ [kpc]")
ax.set_ylabel(r"Stellar density $\rho_\star$ [$M_\odot\,\mathrm{kpc}^{-3}$]")
ax.set_title(r"(a) Tracer density self-consistency", loc="left")
ax.legend(loc="lower left"); ax.grid(True, which="major", color="#eaeaea")

ax = axs[0, 1]
ax.plot(r_jeans, sigma_r_jeans, color="black", lw=1.6,
        label=r"Analytic Jeans $\sigma_r$")
v = np.isfinite(sigma_r_sample)
ax.errorbar(centers[v], sigma_r_sample[v],
            yerr=sigma_r_sample[v] / np.sqrt(2 * counts_sample[v]),
            fmt="s", color=c_theory, mfc="white", mew=1.5, ms=5,
            elinewidth=1.0, capsize=2.0, label=r"Sample $\sigma_r$")
ax.errorbar(centers[v], sigma_t_sample[v],
            yerr=sigma_t_sample[v] / np.sqrt(4 * counts_sample[v]),
            fmt="^", color=c_aux, mfc="white", mew=1.5, ms=5,
            elinewidth=1.0, capsize=2.0, label=r"Sample $\sigma_t$")
ax.set_xscale("log"); ax.set_xlim(5e-3, 3.0)
ax.set_xlabel(r"Galactocentric radius $r$ [kpc]")
ax.set_ylabel(r"Velocity dispersion $\sigma$ [km s$^{-1}$]")
ax.set_title(r"(b) Isotropic kinematic profile", loc="left")
ax.legend(loc="upper right"); ax.grid(True, which="major", color="#eaeaea")

# (c) the ladder in context: this gamma against the whole family
ax = axs[1, 0]
rr = np.logspace(-3, np.log10(30.0), 400)
for g in (0.0, 0.25, 0.5, 0.75, 1.0):
    _, Ig, itg = _build_mass_integral(g)
    Ipiv_g = float(np.exp(itg(np.log(R_PIV / RS))))
    rs_g = M_PIV_ACTUAL / (4.0 * np.pi * RS ** 3 * Ipiv_g)
    xg = np.maximum(rr / RS, 1e-30)
    rg = rs_g / (xg ** g * (1.0 + xg) ** (3.0 - g))
    hot = abs(g - GAM) < 1e-9
    ax.plot(rr, rg, lw=2.6 if hot else 1.0,
            color=c_theory if hot else "0.72", zorder=3 if hot else 1,
            label=(rf"$\gamma={g:g}$ (this mock)" if hot else None))
ax.plot(rr, rho_nfw_cmp(rr), color=c_cusp, lw=1.4, ls="--",
        label="NFW companion")
ax.axvspan(rr[0], r05_3d, color="0.92", zorder=0)
ax.axvspan(r95_3d, rr[-1], color="0.92", zorder=0)
ax.axvline(R_PIV, color="k", ls=":", lw=1.0)
ax.text(R_PIV * 1.06, 3e2, r"$r_{\rm piv}$", fontsize=10)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-3, 30.0); ax.set_ylim(1e2, 5e9)
ax.set_xlabel(r"Galactocentric radius $r$ [kpc]")
ax.set_ylabel(r"DM density $\rho_{\rm DM}$ [$M_\odot\,\mathrm{kpc}^{-3}$]")
ax.set_title(r"(c) Calibration ladder, matched $M(<r_{\rm piv})$", loc="left")
ax.legend(loc="upper right"); ax.grid(True, which="major", color="#eaeaea")

# (d) analytic log-slope, with the two reported truth values marked
ax = axs[1, 1]
ax.plot(rr, dlnrho_dlnr_gnfw(rr), color=c_theory, lw=2.4)
ax.axhline(-GAM, color="0.4", ls="--", lw=1.0)
ax.text(1.5e-3, -GAM + 0.08, rf"$\gamma_0={-GAM:+.2f}$ (asymptotic)",
        fontsize=10, color="0.3")
ax.plot([A_STAR], [SLOPE_AT_RHALF_TRUE], "o", color=c_noise, ms=7, zorder=5)
ax.text(A_STAR * 1.15, SLOPE_AT_RHALF_TRUE + 0.12,
        rf"slope$(R_{{1/2}})={SLOPE_AT_RHALF_TRUE:+.2f}$",
        fontsize=10, color=c_noise)
ax.axvspan(rr[0], r05_3d, color="0.92", zorder=0)
ax.axvspan(r95_3d, rr[-1], color="0.92", zorder=0)
ax.set_xscale("log"); ax.set_xlim(1e-3, 30.0); ax.set_ylim(-3.2, 0.3)
ax.set_xlabel(r"Galactocentric radius $r$ [kpc]")
ax.set_ylabel(r"$d\ln\rho/d\ln r$")
ax.set_title(r"(d) Truth log-slope (both calibration targets)", loc="left")
ax.grid(True, which="major", color="#eaeaea")

path = os.path.join(OUTPUT_DIR, "mock_diagnostics_master.png")
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

fig2, ax2 = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
ax = ax2[0]
eta = eps_grid / eps_grid[-1]
f_norm = f_pos_grid / np.nanmax(f_pos_grid)
m = (eta >= 0.80) & np.isfinite(f_norm)
ax.plot(eta[m], f_norm[m], color=c_theory, lw=2.2, label="Regularized DF")
neg = f_raw_grid < 0.0
if np.any(neg):
    ax.axvspan(np.min(eta[neg]), np.max(eta[neg]), color=c_noise, alpha=0.13,
               label="Endpoint clipping")
ax.axhline(0.0, color="black", lw=0.8)
ax.set_xlim(0.80, 1.005); ax.set_ylim(-0.04, 1.08)
ax.set_xlabel(r"Normalized binding energy $\varepsilon/\varepsilon_{\max}$")
ax.set_ylabel(r"$f(\varepsilon)/f_{\max}$")
ax.set_title(rf"(a) Eddington DF, $\gamma={GAM:g}$", loc="left")
ax.legend(loc="upper left"); ax.grid(True, which="major", color="#eaeaea")

ax = ax2[1]
resid = 100.0 * (rho_df_grid / rho_plummer_mass_normalized(r_sample_grid) - 1.0)
mr = (r_sample_grid >= 1e-2) & (r_sample_grid <= 1e1)
ax.plot(r_sample_grid[mr], resid[mr], color=c_theory, lw=2.2,
        label="DF-implied residual")
ax.axhline(0.0, color="black", ls="--", lw=1.0, label="Perfect recovery")
ax.axhspan(-0.5, 0.5, color=c_theory, alpha=0.10, label=r"$\pm0.5\%$")
ax.set_xscale("log"); ax.set_xlim(1e-2, 1e1); ax.set_ylim(-3.0, 3.0)
ax.set_xlabel(r"Galactocentric radius $r$ [kpc]")
ax.set_ylabel(r"Density residual [\%]" if USE_TEX else "Density residual [%]")
ax.set_title(r"(b) DF-implied density recovery", loc="left")
ax.legend(loc="lower right"); ax.grid(True, which="major", color="#eaeaea")

path2 = os.path.join(OUTPUT_DIR, "eddington_df_diagnostics.png")
plt.savefig(path2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved: {path2}")

plt.rcParams.update(plt.rcParamsDefault)
print("\nDone.")