import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.integrate import cumulative_trapezoid
# ============================================================
# Equilibrium noisy mock for a dwarf-galaxy-like tracer in a NFW halo
# ============================================================
#   - Units: kpc, km/s, Msun, mas/yr, degrees.
#   - The relative potential is Psi(r) = -Phi(r), with Phi(infinity)=0.
#   - Binding energy per unit mass is epsilon = Psi - v^2/2 > 0.

# ============================================================
# 1. CONFIGURATION
# ============================================================

import argparse

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--output-dir", type=str, default="realistic_mock_data_NFW_equilibrium")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-stars", type=int, default=1000)

    p.add_argument("--m0-nfw", type=float, default=5.0e8)
    p.add_argument("--rs-nfw", type=float, default=2.0)
    p.add_argument("--a-star", type=float, default=0.1)

    p.add_argument(
        "--noise-preset",
        type=str,
        default="learnable",
        choices=[
            "learnable",
            "intermediate",
            "hard",
            "draco_gaia_stress",
        ],
        help=(
            "Observational-noise regime. 'draco_gaia_stress' is a direct-distance "
            "stress test for the current latent-6D pipeline, not a full Gaia "
            "parallax-level forward model."
        ),
    )

    return p.parse_args()


ARGS = parse_args()

OUTPUT_DIR = ARGS.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = ARGS.seed
rng = np.random.default_rng(SEED)

N_STARS = ARGS.n_stars
M_STAR_TOT = 1.0e6

G = 4.3009e-6

M_0_NFW = ARGS.m0_nfw
R_S_NFW = ARGS.rs_nfw
A_STAR = ARGS.a_star

NOISE_PRESET = ARGS.noise_preset

# --- Observer/systemic parameters ---
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

elif NOISE_PRESET == "hard":
    ERR_ANGLE_MAS_MIN, ERR_ANGLE_MAS_MAX = 0.010, 0.200
    ERR_DIST_FRAC_MIN, ERR_DIST_FRAC_MAX = 0.100, 0.300
    ERR_PM_MAS_YR_MIN, ERR_PM_MAS_YR_MAX = 0.020, 0.400
    ERR_VLOS_MIN, ERR_VLOS_MAX = 0.5, 8.0

else:
    raise ValueError(f"Unknown NOISE_PRESET={NOISE_PRESET!r}")

# Numerical settings for the DF construction
R_DER_MIN, R_DER_MAX, N_R_DER = 1.0e-6, 1.0e6, 8000
R_SAMPLE_MIN, R_SAMPLE_MAX, N_R_SAMPLE = 1.0e-4, 1.0e2, 1200
N_EPS_GRID = 1400
N_U_EDDINGTON = 450
N_V_INT = 360
N_V_REJECTION_GRID = 280

# ============================================================
# 2. ANALYTIC NFW AND TARGET TRACER FUNCTIONS
# ============================================================
def mass_nfw(r):
    r = np.asarray(r)
    x = r / R_S_NFW
    return M_0_NFW * (np.log1p(x) - x / (1.0 + x))


def rho_nfw(r):
    r = np.asarray(r)
    x = np.maximum(r / R_S_NFW, 1.0e-30)
    return (M_0_NFW / (4.0 * np.pi * R_S_NFW**3)) / (x * (1.0 + x) ** 2)


def psi_nfw(r):
    """Relative potential Psi = -Phi, positive for bound particles."""
    r = np.asarray(r)
    x = np.maximum(r / R_S_NFW, 1.0e-30)
    return (G * M_0_NFW / R_S_NFW) * np.log1p(x) / x

def dlnrho_dlnr_nfw(r):
    """Analytic 3D logarithmic density slope d ln rho / d ln r for NFW.
    Inner limit -> -1 (cusp); outer limit -> -3. This is the quantity
    GravSphere2 reports at the projected half-light radius R_1/2."""
    x = np.maximum(np.asarray(r, dtype=float) / R_S_NFW, 1.0e-30)
    return -1.0 - 2.0 * x / (1.0 + x)

def phi_nfw(r):
    return -psi_nfw(r)


def dpsi_dr_nfw(r):
    r = np.asarray(r)
    x = np.maximum(r / R_S_NFW, 1.0e-30)
    amp = G * M_0_NFW / R_S_NFW
    return (amp / R_S_NFW) * (x / (1.0 + x) - np.log1p(x)) / (x * x)


def acceleration_nfw_cartesian(pos):
    """Acceleration vector in galaxy-centered Cartesian coordinates."""
    r = np.linalg.norm(pos, axis=1)
    r_safe = np.maximum(r, 1.0e-12)
    a_mag = -G * mass_nfw(r_safe) / (r_safe * r_safe)
    return pos * (a_mag / r_safe)[:, None]


def nu_plummer_shape(r):
    """Unnormalised Plummer-like 3D tracer density."""
    r = np.asarray(r)
    return (1.0 + (r / A_STAR) ** 2) ** (-2.5)


def dnu_plummer_dr(r):
    r = np.asarray(r)
    return -5.0 * r / A_STAR**2 * (1.0 + (r / A_STAR) ** 2) ** (-3.5)


def rho_plummer_mass_normalized(r):
    return (3.0 * M_STAR_TOT / (4.0 * np.pi * A_STAR**3)) * nu_plummer_shape(r)


def v_escape(r):
    return np.sqrt(2.0 * psi_nfw(r))

# ============================================================
# 3. EDDINGTON INVERSION: TARGET NU(r) -> POSITIVE EQUILIBRIUM DF f(epsilon)
# ============================================================
def safe_eval_interpolator(interpolator, x):
    y = interpolator(x)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def build_eddington_df():
    print("Building isotropic equilibrium DF with Eddington inversion...")

    r_der = np.logspace(np.log10(R_DER_MIN), np.log10(R_DER_MAX), N_R_DER)
    psi = psi_nfw(r_der)

    # dnu/dPsi = (dnu/dr) / (dPsi/dr)
    dnu_dpsi = dnu_plummer_dr(r_der) / dpsi_dr_nfw(r_der)

    # d/d r using log-r grid: d/dr = (1/r) d/dlnr
    d_dnu_dpsi_dr = np.gradient(dnu_dpsi, np.log(r_der), edge_order=2) / r_der
    d2nu_dpsi2 = d_dnu_dpsi_dr / dpsi_dr_nfw(r_der)

    # Reverse arrays so Psi is increasing from ~0 to Psi(0)
    psi_inc = psi[::-1]
    d2_inc = d2nu_dpsi2[::-1]

    mask = np.isfinite(psi_inc) & np.isfinite(d2_inc) & (psi_inc >= 0.0)
    psi_inc = psi_inc[mask]
    d2_inc = d2_inc[mask]

    # Psi should be strictly increasing after reversal.
    order = np.argsort(psi_inc)
    psi_inc = psi_inc[order]
    d2_inc = d2_inc[order]

    # Remove duplicate Psi values if any occur at numerical precision.
    unique = np.r_[True, np.diff(psi_inc) > 0]
    psi_inc = psi_inc[unique]
    d2_inc = d2_inc[unique]

    d2_interp = PchipInterpolator(psi_inc, d2_inc, extrapolate=False)

    eps_max = psi_inc[-1] * (1.0 - 1.0e-8)
    eps_grid = np.linspace(0.0, eps_max, N_EPS_GRID)
    f_raw = np.zeros_like(eps_grid)

    # Eddington formula, with change of variable Psi = eps - u^2:
    # integral_0^eps dPsi d2nu/dPsi2 / sqrt(eps-Psi)
    # = 2 integral_0^sqrt(eps) du d2nu/dPsi2(eps-u^2)
    prefactor = 1.0 / (np.sqrt(8.0) * np.pi**2)
    for j, eps in enumerate(eps_grid[1:], start=1):
        u = np.linspace(0.0, np.sqrt(eps), N_U_EDDINGTON)
        psi_eval = eps - u * u
        vals = safe_eval_interpolator(d2_interp, psi_eval)
        f_raw[j] = prefactor * 2.0 * np.trapezoid(vals, u)

    # Numerical Eddington inversions often develop tiny negative values near endpoints.
    # For this setup they are confined to the highest-energy endpoint and the implied
    # density remains essentially Plummer. We clip to enforce a positive DF.
    neg_mask = f_raw < 0.0
    neg_fraction = float(np.mean(neg_mask))
    min_raw = float(np.min(f_raw))
    max_raw = float(np.max(f_raw))
    f_pos = np.maximum(f_raw, 0.0)

    print(f"  Raw DF min/max: {min_raw:.3e} / {max_raw:.3e}")
    print(f"  Fraction of negative grid points clipped: {100.0 * neg_fraction:.2f}%")

    f_interp = PchipInterpolator(eps_grid, f_pos, extrapolate=False)

    def f_eps(eps):
        eps_arr = np.asarray(eps)
        eps_clip = np.clip(eps_arr, 0.0, eps_grid[-1])
        return safe_eval_interpolator(f_interp, eps_clip)

    return eps_grid, f_raw, f_pos, f_eps


eps_grid, f_raw_grid, f_pos_grid, f_eps = build_eddington_df()

# ============================================================
# 4. DF-IMPLIED TRACER DENSITY AND RADIAL CDF
# ============================================================
def compute_nu_from_df(r_grid):
    """nu(r) = 4pi int v^2 f(Psi - v^2/2) dv."""
    nu = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        psi = float(psi_nfw(r))
        vmax = np.sqrt(2.0 * psi)
        v = np.linspace(0.0, vmax, N_V_INT)
        eps = psi - 0.5 * v * v
        integrand = v * v * f_eps(eps)
        nu[i] = 4.0 * np.pi * np.trapezoid(integrand, v)
    return nu


print("Computing DF-implied tracer density and radial CDF...")
r_sample_grid = np.logspace(np.log10(R_SAMPLE_MIN), np.log10(R_SAMPLE_MAX), N_R_SAMPLE)
nu_df_shape_grid = compute_nu_from_df(r_sample_grid)

radial_pdf = 4.0 * np.pi * r_sample_grid**2 * nu_df_shape_grid
radial_cdf = cumulative_trapezoid(radial_pdf, r_sample_grid, initial=0.0)
radial_norm = radial_cdf[-1]
if not np.isfinite(radial_norm) or radial_norm <= 0.0:
    raise RuntimeError("Invalid radial normalization. Check DF construction.")
radial_cdf /= radial_norm

# Interpolator for inverse-CDF sampling
inv_radial_cdf = interp1d(
    radial_cdf,
    r_sample_grid,
    bounds_error=False,
    fill_value=(r_sample_grid[0], r_sample_grid[-1]),
    assume_sorted=True,
)

# Scale DF-implied nu to M_STAR_TOT for plots.
nu_df_mass_scale = M_STAR_TOT / radial_norm
rho_df_grid = nu_df_mass_scale * nu_df_shape_grid

# Check how close the positive DF-implied profile is to the Plummer target.
r_check_mask = (r_sample_grid > 0.01) & (r_sample_grid < 10.0)
plum_grid_mass = rho_plummer_mass_normalized(r_sample_grid)
rel_shape = rho_df_grid[r_check_mask] / plum_grid_mass[r_check_mask]
print(
    "  DF-implied tracer / Plummer target over 0.01-10 kpc: "
    f"min={np.nanmin(rel_shape):.4f}, median={np.nanmedian(rel_shape):.4f}, max={np.nanmax(rel_shape):.4f}"
)
print(
    "  Radial CDF quantiles: "
    f"r50={float(inv_radial_cdf(0.5)):.3f} kpc, "
    f"r90={float(inv_radial_cdf(0.9)):.3f} kpc, "
    f"r99={float(inv_radial_cdf(0.99)):.3f} kpc"
)

# ============================================================
# 5. SAMPLE EQUILIBRIUM PHASE-SPACE POINTS
# ============================================================
def sample_speed_given_radius(r, rng):
    psi = float(psi_nfw(r))
    vmax = np.sqrt(2.0 * psi)

    v_grid = np.linspace(0.0, vmax, N_V_REJECTION_GRID)
    eps_grid_local = psi - 0.5 * v_grid * v_grid
    q_grid = v_grid * v_grid * f_eps(eps_grid_local)
    q_max = float(np.max(q_grid))

    if not np.isfinite(q_max) or q_max <= 0.0:
        raise RuntimeError(f"Bad velocity proposal at r={r:.6e}, psi={psi:.6e}, q_max={q_max}")

    # Rejection sampling from p(v|r) proportional to v^2 f(Psi-v^2/2)
    for _ in range(20000):
        v_trial = rng.random() * vmax
        eps_trial = psi - 0.5 * v_trial * v_trial
        q_trial = v_trial * v_trial * float(np.asarray(f_eps(eps_trial)))
        if rng.random() * q_max <= q_trial:
            return v_trial

    # Extremely rare fallback: inverse-CDF on the proposal grid.
    q_grid = np.maximum(q_grid, 0.0)
    cdf_v = cumulative_trapezoid(q_grid, v_grid, initial=0.0)
    if cdf_v[-1] <= 0.0:
        raise RuntimeError("Velocity fallback failed: zero local CDF.")
    cdf_v /= cdf_v[-1]
    return float(np.interp(rng.random(), cdf_v, v_grid))


print(f"Sampling {N_STARS} equilibrium stars...")
r = inv_radial_cdf(rng.random(N_STARS))

# Isotropic positions
phi_pos = 2.0 * np.pi * rng.random(N_STARS)
cos_theta_pos = 2.0 * rng.random(N_STARS) - 1.0
sin_theta_pos = np.sqrt(1.0 - cos_theta_pos**2)

x_gal = r * sin_theta_pos * np.cos(phi_pos)
y_gal = r * sin_theta_pos * np.sin(phi_pos)
z_gal = r * cos_theta_pos
pos_gal = np.column_stack([x_gal, y_gal, z_gal])

# Isotropic velocities, with speed sampled from f(epsilon)
speed = np.array([sample_speed_given_radius(ri, rng) for ri in r])
phi_vel = 2.0 * np.pi * rng.random(N_STARS)
cos_theta_vel = 2.0 * rng.random(N_STARS) - 1.0
sin_theta_vel = np.sqrt(1.0 - cos_theta_vel**2)

vx_gal = speed * sin_theta_vel * np.cos(phi_vel)
vy_gal = speed * sin_theta_vel * np.sin(phi_vel)
vz_gal = speed * cos_theta_vel
vel_gal = np.column_stack([vx_gal, vy_gal, vz_gal])

psi_star = psi_nfw(r)
epsilon = psi_star - 0.5 * speed**2
energy = 0.5 * speed**2 - psi_star

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

    VX0 = (
        V_SYS_LOS * np.cos(dec_c) * np.cos(ra_c)
        - v_sys_perp_ra * np.sin(ra_c)
        - v_sys_perp_dec * np.sin(dec_c) * np.cos(ra_c)
    )
    VY0 = (
        V_SYS_LOS * np.cos(dec_c) * np.sin(ra_c)
        + v_sys_perp_ra * np.cos(ra_c)
        - v_sys_perp_dec * np.sin(dec_c) * np.sin(ra_c)
    )
    VZ0 = V_SYS_LOS * np.sin(dec_c) + v_sys_perp_dec * np.cos(dec_c)

    return np.array([X0, Y0, Z0]), np.array([VX0, VY0, VZ0])


def cartesian_to_observables(pos_abs, vel_abs):
    X, Y, Z = pos_abs[:, 0], pos_abs[:, 1], pos_abs[:, 2]
    VX, VY, VZ = vel_abs[:, 0], vel_abs[:, 1], vel_abs[:, 2]

    D = np.sqrt(X * X + Y * Y + Z * Z)
    RA = np.arctan2(Y, X)
    Dec = np.arcsin(Z / D)

    r_hat = np.column_stack([
        np.cos(Dec) * np.cos(RA),
        np.cos(Dec) * np.sin(RA),
        np.sin(Dec),
    ])
    ra_hat = np.column_stack([
        -np.sin(RA),
        np.cos(RA),
        np.zeros_like(RA),
    ])
    dec_hat = np.column_stack([
        -np.sin(Dec) * np.cos(RA),
        -np.sin(Dec) * np.sin(RA),
        np.cos(Dec),
    ])

    V = np.column_stack([VX, VY, VZ])
    v_los = np.sum(V * r_hat, axis=1)
    pm_ra = np.sum(V * ra_hat, axis=1) / (KAPPA * D)       # mu_alpha_star
    pm_dec = np.sum(V * dec_hat, axis=1) / (KAPPA * D)

    RA_deg = np.rad2deg(RA) % 360.0
    Dec_deg = np.rad2deg(Dec)

    return RA_deg, Dec_deg, D, pm_ra, pm_dec, v_los


def obs_to_galactocentric_np(ra_deg, dec_deg, dist, pm_ra, pm_dec, v_los):
    """Inverse transform used only for diagnostics; mirrors the fitting script."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    X = dist * np.cos(dec) * np.cos(ra)
    Y = dist * np.cos(dec) * np.sin(ra)
    Z = dist * np.sin(dec)

    center_pos, center_vel = center_cartesian_and_velocity()
    pos = np.column_stack([X, Y, Z]) - center_pos[None, :]

    r_hat = np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])
    ra_hat = np.column_stack([
        -np.sin(ra),
        np.cos(ra),
        np.zeros_like(ra),
    ])
    dec_hat = np.column_stack([
        -np.sin(dec) * np.cos(ra),
        -np.sin(dec) * np.sin(ra),
        np.cos(dec),
    ])

    v_ra_phys = pm_ra * KAPPA * dist
    v_dec_phys = pm_dec * KAPPA * dist
    vel_abs = (
        v_los[:, None] * r_hat
        + v_ra_phys[:, None] * ra_hat
        + v_dec_phys[:, None] * dec_hat
    )
    vel = vel_abs - center_vel[None, :]
    return pos, vel


center_pos, center_vel = center_cartesian_and_velocity()
pos_abs = center_pos[None, :] + pos_gal
vel_abs = center_vel[None, :] + vel_gal

RA_true, Dec_true, D_true, pmRA_true, pmDec_true, vlos_true = cartesian_to_observables(pos_abs, vel_abs)

# Projected radius on sky around the galaxy center
ra_c_rad = np.deg2rad(RA_CENTER)
dec_c_rad = np.deg2rad(DEC_CENTER)
ra_true_rad = np.deg2rad(RA_true)
dec_true_rad = np.deg2rad(Dec_true)
cos_ang = (
    np.sin(dec_true_rad) * np.sin(dec_c_rad)
    + np.cos(dec_true_rad) * np.cos(dec_c_rad) * np.cos(ra_true_rad - ra_c_rad)
)
ang_dist = np.arccos(np.clip(cos_ang, -1.0, 1.0))
R_proj = DIST_CENTER * ang_dist

# ============================================================
# 7. ADD HETEROSCEDASTIC OBSERVATIONAL NOISE
# ============================================================
print(f"Adding heteroscedastic observational noise: preset={NOISE_PRESET!r}...")

# A simple magnitude/faintness proxy in [0,1]
faintness = rng.gamma(shape=2.0, scale=1.0, size=N_STARS)
faintness = faintness / np.max(faintness)

err_angle_mas = ERR_ANGLE_MAS_MIN + faintness * (ERR_ANGLE_MAS_MAX - ERR_ANGLE_MAS_MIN)
err_dist_frac = ERR_DIST_FRAC_MIN + faintness * (ERR_DIST_FRAC_MAX - ERR_DIST_FRAC_MIN)
err_pm = ERR_PM_MAS_YR_MIN + faintness * (ERR_PM_MAS_YR_MAX - ERR_PM_MAS_YR_MIN)
err_vlos = ERR_VLOS_MIN + faintness * (ERR_VLOS_MAX - ERR_VLOS_MIN)

# Coordinate errors in degrees.
# err_angle_mas is an on-sky tangent-plane uncertainty. RA coordinate uncertainty
# is divided by cos(dec), because dtheta_RA = cos(dec) dRA.
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

# Avoid nonphysical negative distances in extremely unlucky draws.
# With the default noise this should almost never matter.
bad_d = obs_dist <= 0.0
if np.any(bad_d):
    obs_dist[bad_d] = D_true[bad_d]

# Noisy galactocentric reconstruction for diagnostics
pos_noisy_gal, vel_noisy_gal = obs_to_galactocentric_np(
    obs_ra, obs_dec, obs_dist, obs_pm_ra, obs_pm_dec, obs_vlos
)
r_noisy = np.linalg.norm(pos_noisy_gal, axis=1)
v_noisy = np.linalg.norm(vel_noisy_gal, axis=1)

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

observed_csv = os.path.join(OUTPUT_DIR, "draco_nfw_observations.csv")
truth_gal_csv = os.path.join(OUTPUT_DIR, "draco_nfw_truth_galactocentric.csv")
truth_obs_csv = os.path.join(OUTPUT_DIR, "draco_nfw_truth_observer.csv")
observed_df.to_csv(observed_csv, index=False)
truth_gal_df.to_csv(truth_gal_csv, index=False)
truth_obs_df.to_csv(truth_obs_csv, index=False)

# ---- model-agnostic ground-truth profile (NEW; mirrors the Burkert generator) ----
# Identical schema across both mocks so the comparison harness reads them the same way.
# dlnrho_dlnr is the ANALYTIC 3D log-density slope, reported at R_1/2 by GravSphere2.
r_truth_prof = np.logspace(-3, np.log10(50.0), 500)
truth_profile_df = pd.DataFrame({
    "r": r_truth_prof,
    "mass": mass_nfw(r_truth_prof),
    "rho": rho_nfw(r_truth_prof),
    "acc": G * mass_nfw(r_truth_prof) / r_truth_prof**2,
    "dlnrho_dlnr": dlnrho_dlnr_nfw(r_truth_prof),
})
truth_profile_csv = os.path.join(OUTPUT_DIR, "truth_profile.csv")
truth_profile_df.to_csv(truth_profile_csv, index=False)

# ---- characteristic radii for GravSphere2-style plots (NEW) ----
# Projected half-light radius R_1/2: for a Plummer tracer it equals the scale radius
# a_star exactly; we also store the empirical median projected radius, which is what
# the fitter uses (R_half = median(R_proj)) and what the plots should mark.
R_half_2d_plummer = float(A_STAR)
R_half_2d_empirical = float(np.median(R_proj))
r05_3d = float(inv_radial_cdf(0.05))
r95_3d = float(inv_radial_cdf(0.95))
r_inner_tracer_3d = float(np.min(r))
r_outer_tracer_3d = float(np.max(r))

metadata = {
    "halo_type": "nfw_cusped",
    "seed": SEED, "n_stars": N_STARS, "G": G,
    "M_0_NFW": M_0_NFW, "R_S_NFW": R_S_NFW,
    "A_STAR_target_plummer": A_STAR, "M_STAR_TOT_plot_only": M_STAR_TOT,
    "RA_CENTER": RA_CENTER, "DEC_CENTER": DEC_CENTER, "DIST_CENTER": DIST_CENTER,
    "V_SYS_LOS": V_SYS_LOS, "PM_SYS_RA": PM_SYS_RA, "PM_SYS_DEC": PM_SYS_DEC,
    "KAPPA": KAPPA, "noise_preset": NOISE_PRESET,
    # NEW: characteristic radii + benchmark range used by the comparison plots
    "R_half_2d_plummer": R_half_2d_plummer,
    "R_half_2d_empirical": R_half_2d_empirical,
    "tracer_support": {
        "r05_3d": r05_3d, "r95_3d": r95_3d,
        "r_inner_tracer_3d": r_inner_tracer_3d, "r_outer_tracer_3d": r_outer_tracer_3d,
    },
    "benchmark_radial_range_kpc": [0.1 * R_half_2d_empirical, 10.0 * R_half_2d_empirical],
    "noise": {
        "ERR_ANGLE_MAS_MIN": ERR_ANGLE_MAS_MIN, "ERR_ANGLE_MAS_MAX": ERR_ANGLE_MAS_MAX,
        "ERR_DIST_FRAC_MIN": ERR_DIST_FRAC_MIN, "ERR_DIST_FRAC_MAX": ERR_DIST_FRAC_MAX,
        "ERR_PM_MAS_YR_MIN": ERR_PM_MAS_YR_MIN, "ERR_PM_MAS_YR_MAX": ERR_PM_MAS_YR_MAX,
        "ERR_VLOS_MIN": ERR_VLOS_MIN, "ERR_VLOS_MAX": ERR_VLOS_MAX,
    },
    "df_diagnostics": {
        "raw_df_min": float(np.min(f_raw_grid)), "raw_df_max": float(np.max(f_raw_grid)),
        "fraction_negative_clipped": float(np.mean(f_raw_grid < 0.0)),
        "df_implied_over_plummer_0p01_10kpc_min": float(np.nanmin(rel_shape)),
        "df_implied_over_plummer_0p01_10kpc_median": float(np.nanmedian(rel_shape)),
        "df_implied_over_plummer_0p01_10kpc_max": float(np.nanmax(rel_shape)),
    },
}
with open(os.path.join(OUTPUT_DIR, "mock_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Observed noisy data:        {observed_csv}")
print(f"  Truth galactocentric data:  {truth_gal_csv}")
print(f"  Truth observer-frame data:  {truth_obs_csv}")
print(f"  Model-agnostic truth table: {truth_profile_csv}")
print(f"  R_1/2 (Plummer a_star)={R_half_2d_plummer:.4f} kpc | "
      f"empirical median R_proj={R_half_2d_empirical:.4f} kpc")

# ============================================================
# 9. DIAGNOSTIC PLOTS
# ============================================================
print("Building diagnostic plots...")

# Binned radial diagnostics
bins = np.logspace(np.log10(R_SAMPLE_MIN), np.log10(20.0), 34)
centers = np.sqrt(bins[:-1] * bins[1:])

rho_sample = np.full_like(centers, np.nan)
sigma_r_sample = np.full_like(centers, np.nan)
sigma_t_sample = np.full_like(centers, np.nan)
beta_sample = np.full_like(centers, np.nan)
counts_sample = np.zeros_like(centers)

r_norm = np.linalg.norm(pos_gal, axis=1)
vr = np.sum(pos_gal * vel_gal, axis=1) / np.maximum(r_norm, 1.0e-12)
vt2 = np.sum(vel_gal * vel_gal, axis=1) - vr * vr

for i in range(len(centers)):
    mask = (r_norm >= bins[i]) & (r_norm < bins[i + 1])
    n = int(np.sum(mask))
    counts_sample[i] = n
    if n >= 15:
        shell_volume = 4.0 / 3.0 * np.pi * (bins[i + 1] ** 3 - bins[i] ** 3)
        rho_sample[i] = (n / N_STARS) * M_STAR_TOT / shell_volume
        vr_c = vr[mask] - np.mean(vr[mask])
        sigma_r_sample[i] = np.std(vr_c)
        sigma_t_sample[i] = np.sqrt(0.5 * np.mean(vt2[mask]))
        beta_sample[i] = 1.0 - np.mean(vt2[mask]) / (2.0 * np.var(vr_c))

# Jeans prediction for the DF-implied tracer profile
r_jeans = r_sample_grid.copy()
nu_jeans = np.maximum(nu_df_shape_grid, 1.0e-300)
integrand_jeans = nu_jeans * G * mass_nfw(r_jeans) / (r_jeans * r_jeans)
rev_integral = cumulative_trapezoid(integrand_jeans[::-1], r_jeans[::-1], initial=0.0)
jeans_integral = -rev_integral[::-1]
sigma_r_jeans = np.sqrt(np.maximum(jeans_integral / nu_jeans, 0.0))

# True and noisy relative distance along observer line of sight
true_relative_distance = D_true - DIST_CENTER
obs_relative_distance = obs_dist - DIST_CENTER

# Velocity-equivalent PM error, useful for judging information content
err_pm_velocity_equiv = KAPPA * D_true * err_pm

# Configure journal quality matplotlib parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],

    "axes.linewidth": 0.9,
    "axes.labelsize": 13.0,
    "axes.titlesize": 13.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 14.0,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5.5,
    "ytick.major.size": 5.5,
    "xtick.minor.size": 3.0,
    "ytick.minor.size": 3.0,

    "legend.frameon": False,
    "axes.grid": False,
    "savefig.dpi": 400,
})

# Color palette: high-contrast minimalist academic style
c_theory = "#1f77b4"      # Deep Royal Blue
c_data = "#2c3e50"        # Charcoal Slate
c_noise = "#e74c3c"       # Muted Crimson
c_aux = "#27ae60"         # Forest Green
c_base_line = "#b2bec3"   # Clean light gray for backing curves

# --- FIGURE 1: 2x2 Master Physical and Observational Synthesis ---
fig, axs = plt.subplots(2, 2, figsize=(12.0, 9.2), constrained_layout=True)

# Panel (a): Density Profile Reconstruction
ax = axs[0, 0]
ax.plot(r_sample_grid, rho_plummer_mass_normalized(r_sample_grid), 
        color=c_base_line, linestyle="-", lw=4.0, label=r"Target Plummer $\rho_\star(r)$")
ax.plot(r_sample_grid, rho_df_grid, 
        color=c_theory, linestyle="-", lw=1.5, label=r"Isotropic DF-Implied $\rho_{\mathrm{DF}}(r)$")
valid_rho = np.isfinite(rho_sample)
ax.errorbar(centers[valid_rho], rho_sample[valid_rho], 
            yerr=rho_sample[valid_rho] / np.sqrt(counts_sample[valid_rho]),
            fmt="o", color=c_data, ms=4, elinewidth=0.9, capsize=1.5, markeredgecolor="black",
            markeredgewidth=0.5, label="Mock Sample Shells")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which='major', linestyle='-', color='#eaeaea', alpha=0.8)
ax.grid(False, which='minor') # Avoid crowded tick grids
ax.set_xlim(5e-3, 3.0)
ax.set_ylim(1e1, 8e8)  # Set explicitly to 8e8 to show the flat core at small r completely
ax.set_xlabel(r"Galactocentric Radius, $r$ [kpc]")
ax.set_ylabel(r"Stellar Mass Density, $\rho$ [$M_\odot\,\mathrm{kpc}^{-3}$]")
ax.set_title(r"\textbf{(a) Density Profile Self-Consistency}", loc="left")
ax.legend(loc="lower left")

# Panel (b): Internal Kinematics & Velocity Dispersion
ax = axs[0, 1]
ax.plot(r_jeans, sigma_r_jeans, color="black", linestyle="-", lw=1.6, label=r"Analytic Jeans $\sigma_r(r)$")
valid_sig = np.isfinite(sigma_r_sample)
ax.errorbar(centers[valid_sig], sigma_r_sample[valid_sig], 
            yerr=sigma_r_sample[valid_sig] / np.sqrt(2 * counts_sample[valid_sig]),
            fmt="s", color=c_theory, mfc="white", mew=1.5, ms=5, elinewidth=1.0, capsize=2.0, 
            label=r"Sample Radial, $\sigma_r$")
ax.errorbar(centers[valid_sig], sigma_t_sample[valid_sig], 
            yerr=sigma_t_sample[valid_sig] / np.sqrt(4 * counts_sample[valid_sig]),
            fmt="^", color=c_aux, mfc="white", mew=1.5, ms=5, elinewidth=1.0, capsize=2.0, 
            label=r"Sample Tangential, $\sigma_t$")
ax.set_xscale("log")
ax.grid(True, which='major', linestyle='-', color='#eaeaea', alpha=0.8)
ax.grid(False, which='minor')
ax.set_xlim(5e-3, 3.0)
ax.set_ylim(0, 10)
ax.set_xlabel(r"Galactocentric Radius, $r$ [kpc]")
ax.set_ylabel(r"Velocity Dispersion, $\sigma$ [$\mathrm{km}\,\mathrm{s}^{-1}$]")
ax.set_title(r"\textbf{(b) Isotropic Kinematic Profile}", loc="left")
ax.legend(loc="upper right")

# Panel (c): Distance errors dominate the intrinsic LOS depth
ax = axs[1, 0]

# True line-of-sight depth of each star relative to the galaxy centre.
los_depth_true = np.abs(D_true - DIST_CENTER)

# Observational distance error.
los_distance_error = np.abs(obs_dist - D_true)

# Avoid exact zeros on a logarithmic x-axis.
eps_plot = 1.0e-4
los_depth_true_plot = np.maximum(los_depth_true, eps_plot)
los_distance_error_plot = np.maximum(los_distance_error, eps_plot)

def empirical_cdf(x):
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

x_depth, y_depth = empirical_cdf(los_depth_true_plot)
x_err, y_err = empirical_cdf(los_distance_error_plot)

med_depth = np.median(los_depth_true)
med_err = np.median(los_distance_error)
ratio_err_depth = med_err / med_depth

# CDF curves
ax.plot(
    x_depth,
    y_depth,
    color=c_theory,
    lw=2.2,
    label=r"True LOS depth"
)

ax.plot(
    x_err,
    y_err,
    color=c_noise,
    lw=2.2,
    label=r"Observed distance error"
)

# Median markers as horizontal 50% guide + small ticks, not full vertical labels
ax.axhline(
    0.5,
    color="0.35",
    linestyle=":",
    lw=1.0,
    zorder=0
)

ax.plot(
    med_depth,
    0.5,
    marker="o",
    color=c_theory,
    mfc="white",
    mec=c_theory,
    mew=1.4,
    ms=6,
    zorder=5
)

ax.plot(
    med_err,
    0.5,
    marker="s",
    color=c_noise,
    mfc="white",
    mec=c_noise,
    mew=1.4,
    ms=6,
    zorder=5
)

# Short bottom ticks showing where the medians are
ax.vlines(
    med_depth,
    0.0,
    0.5,
    color=c_theory,
    linestyle="--",
    lw=1.0,
    alpha=0.75
)

ax.vlines(
    med_err,
    0.0,
    0.5,
    color=c_noise,
    linestyle="--",
    lw=1.0,
    alpha=0.75
)

# Clean summary box
ax.text(
    0.04,
    0.87,
    (
        rf"Median true depth: ${med_depth:.2f}\,\mathrm{{kpc}}$"
        "\n"
        rf"Median distance error: ${med_err:.2f}\,\mathrm{{kpc}}$"
        "\n"
        rf"Error/depth ratio: ${ratio_err_depth:.1f}$"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.4,
    bbox=dict(
        boxstyle="round,pad=0.28",
        facecolor="white",
        edgecolor="0.82",
        alpha=0.95
    )
)

ax.set_xscale("log")

# Robust x-range
xmax = np.nanpercentile(
    np.concatenate([los_depth_true_plot, los_distance_error_plot]),
    99.5
)
ax.set_xlim(eps_plot, 1.4 * xmax)
ax.set_ylim(0.0, 1.02)

ax.set_xlabel(r"Line-of-Sight Length Scale [kpc]")
ax.set_ylabel(r"Cumulative Fraction")
ax.set_title(r"\textbf{(c) Distance Errors Dominate LOS Structure}", loc="left")

ax.legend(
    loc="upper left",
    fontsize=8.0,
    handlelength=2.2,
    frameon=False
)

ax.grid(True, which="major", linestyle="-", color="#eaeaea", alpha=0.8)
ax.grid(False, which="minor")

# Panel (d): The Deconvolution Degeneracy
ax = axs[1, 1]
rline = np.logspace(-3, 2, 100)
ax.scatter(r_norm, r_noisy, s=12.0, color=c_noise, alpha=0.35, edgecolors="none", label="Noisy Reconstructed $r$")
ax.plot(rline, rline, color="black", linestyle="--", lw=1.2, label="Unbiased Deconvolution (1:1)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which='major', linestyle='-', color='#eaeaea', alpha=0.8)
ax.grid(False, which='minor')
ax.set_xlim(1e-2, 1e1)
ax.set_ylim(1e-2, 1e1)
ax.set_xlabel(r"True Physical Radius, $r_{\mathrm{true}}$ [kpc]")
ax.set_ylabel(r"Noisy Reconstructed Radius, $r_{\mathrm{noisy}}$ [kpc]")
ax.set_title(r"\textbf{(d) Projection \& Noise Degeneracy}", loc="left")
ax.legend(loc="upper left")

diag_path = os.path.join(OUTPUT_DIR, "mock_diagnostics_master.png")
plt.savefig(diag_path, dpi=300, bbox_inches="tight")
print(f"Diagnostic plot saved: {diag_path}")

# --- FIGURE 2: Clean Eddington Inversion and Density Recovery Diagnostic ---
fig_df, ax_df = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)

# ============================================================
# Panel (a): Zoomed normalized Eddington DF
# ============================================================
ax = ax_df[0]

eps_max_plot = eps_grid[-1]
eta_grid = eps_grid / eps_max_plot

fmax = np.nanmax(f_pos_grid)
if fmax <= 0.0 or not np.isfinite(fmax):
    raise RuntimeError("Invalid regularized DF normalization: fmax <= 0 or non-finite.")

f_norm = f_pos_grid / fmax

# The DF is visually informative only near the high-binding-energy end.
mask_eta = (eta_grid >= 0.88) & np.isfinite(f_norm)

ax.plot(
    eta_grid[mask_eta],
    f_norm[mask_eta],
    color=c_theory,
    lw=2.2,
    label=r"Regularized DF"
)

# Show the clipped endpoint region subtly, without plotting noisy raw negative points.
negative_raw = f_raw_grid < 0.0
neg_fraction_plot = 100.0 * np.mean(negative_raw)

if np.any(negative_raw):
    eta_neg_min = np.min(eta_grid[negative_raw])
    eta_neg_max = np.max(eta_grid[negative_raw])

    ax.axvspan(
        eta_neg_min,
        eta_neg_max,
        color=c_noise,
        alpha=0.13,
        lw=0.0,
        label=r"Endpoint clipping region"
    )

ax.axhline(
    0.0,
    color="black",
    lw=0.8
)

ax.text(
    0.04,
    0.86,
    (
        r"Regularized isotropic DF"
        "\n"
        rf"Endpoint clipping: ${neg_fraction_plot:.2f}\%$ of grid"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.7,
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="0.85",
        alpha=0.95
    )
)

ax.set_xlim(0.88, 1.002)
ax.set_ylim(-0.04, 1.08)

ax.set_xlabel(r"Normalized Binding Energy, $\varepsilon/\varepsilon_{\max}$")
ax.set_ylabel(r"Normalized DF, $f(\varepsilon)/f_{\max}$")
ax.set_title(r"\textbf{(a) Regularized Eddington DF}", loc="left")

ax.legend(loc="upper left", fontsize=8.0, frameon=False)
ax.grid(True, which="major", linestyle="-", color="#eaeaea", alpha=0.8)
ax.grid(False, which="minor")


# ============================================================
# Panel (b): Density recovery residual
# ============================================================
ax = ax_df[1]

ratio_df = rho_df_grid / rho_plummer_mass_normalized(r_sample_grid)
residual_percent = 100.0 * (ratio_df - 1.0)

# Same diagnostic range used in the metadata and textual description.
mask_ratio = (r_sample_grid >= 1e-2) & (r_sample_grid <= 1e1)
r_ratio = r_sample_grid[mask_ratio]
resid_plot = residual_percent[mask_ratio]

ax.plot(
    r_ratio,
    resid_plot,
    color=c_theory,
    lw=2.2,
    label=r"DF-implied residual"
)

ax.axhline(
    0.0,
    color="black",
    linestyle="--",
    lw=1.0,
    label=r"Perfect recovery"
)

# Tolerance band around perfect recovery.
ax.axhspan(
    -0.5,
    0.5,
    color=c_theory,
    alpha=0.10,
    lw=0.0,
    label=r"$\pm 0.5\%$"
)

median_resid = np.nanmedian(resid_plot)
max_abs_resid = np.nanmax(np.abs(resid_plot))

ax.text(
    0.04,
    0.92,
    (
        rf"Median residual: ${median_resid:.3f}\%$"
        "\n"
        rf"Max $|\mathrm{{residual}}|$: ${max_abs_resid:.2f}\%$"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.7,
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="0.85",
        alpha=0.95
    )
)

ax.set_xscale("log")
ax.set_xlim(1e-2, 1e1)
ax.set_ylim(-0.75, 1.75)

ax.set_xlabel(r"Galactocentric Radius, $r$ [kpc]")
ax.set_ylabel(r"Density Residual [percent]")
ax.set_title(r"\textbf{(b) DF-Implied Density Recovery}", loc="left")

ax.legend(loc="lower right", fontsize=8.0, frameon=False)
ax.grid(True, which="major", linestyle="-", color="#eaeaea", alpha=0.8)
ax.grid(False, which="minor")


# Save Figure 2
df_diag_path = os.path.join(OUTPUT_DIR, "eddington_df_diagnostics.png")
plt.savefig(df_diag_path, dpi=400, bbox_inches="tight")
print(f"DF diagnostic plot saved: {df_diag_path}")

# --- FIGURE 3: Local Speed Distributions across Radial Shells ---
fig_vel, axs_vel = plt.subplots(3, 1, figsize=(7.2, 9.2), constrained_layout=True)

shells = [
    {
        "r_min": 0.00,
        "r_max": 0.05,
        "title": r"\textbf{(a) Inner core}",
        "subtitle": r"$0 < r < 0.05\,\mathrm{kpc}$"
    },
    {
        "r_min": 0.08,
        "r_max": 0.15,
        "title": r"\textbf{(b) Near scale radius}",
        "subtitle": r"$0.08 < r < 0.15\,\mathrm{kpc}$"
    },
    {
        "r_min": 0.25,
        "r_max": 0.60,
        "title": r"\textbf{(c) Outer tracer body}",
        "subtitle": r"$0.25 < r < 0.60\,\mathrm{kpc}$"
    },
]

# Common velocity grid for all panels
v_plot_max = 15.0
v_eval_common = np.linspace(1.0e-3, v_plot_max, 500)

# Use common bins across panels for visual consistency
speed_bins = np.linspace(0.0, v_plot_max, 18)

for idx, shell in enumerate(shells):
    ax = axs_vel[idx]

    mask = (r_norm >= shell["r_min"]) & (r_norm < shell["r_max"])
    v_shell = speed[mask]
    n_shell = int(np.sum(mask))

    if n_shell > 0:
        r_median = np.median(r_norm[mask])
    else:
        r_median = 0.5 * (shell["r_min"] + shell["r_max"])

    psi_val = float(psi_nfw(r_median))
    v_esc_val = float(v_escape(r_median))

    # Histogram of sampled stars
    ax.hist(
        v_shell,
        bins=speed_bins,
        density=True,
        histtype="stepfilled",
        facecolor="#f1f5f9",
        edgecolor="#64748b",
        alpha=0.70,
        lw=1.0,
        label=r"Sampled stars" if idx == 0 else None
    )

    # Eddington speed distribution at the median radius
    v_eval = v_eval_common.copy()
    eps_eval = psi_val - 0.5 * v_eval**2

    # Speeds above escape are physically forbidden; here v_esc is outside range,
    # but keep this mask for correctness under other parameters.
    allowed = eps_eval > 0.0

    pdf_edd = np.zeros_like(v_eval)
    pdf_edd[allowed] = v_eval[allowed] ** 2 * f_eps(eps_eval[allowed])

    norm_edd = np.trapezoid(pdf_edd, v_eval)
    if norm_edd > 0.0:
        pdf_edd /= norm_edd

    # Maxwell-Boltzmann reference using Jeans sigma at the same radius
    sigma_val = float(np.interp(r_median, r_jeans, sigma_r_jeans))
    pdf_gauss = np.sqrt(2.0 / np.pi) * (v_eval**2 / sigma_val**3) * np.exp(
        -0.5 * v_eval**2 / sigma_val**2
    )

    norm_gauss = np.trapezoid(pdf_gauss, v_eval)
    if norm_gauss > 0.0:
        pdf_gauss /= norm_gauss

    # Plot analytical curves
    ax.plot(
        v_eval,
        pdf_edd,
        color=c_theory,
        lw=2.0,
        label=r"Eddington prediction" if idx == 0 else None
    )

    ax.plot(
        v_eval,
        pdf_gauss,
        color=c_noise,
        linestyle="--",
        lw=1.6,
        label=r"Local Maxwellian" if idx == 0 else None
    )

    # Mark the median sampled speed
    if n_shell > 0:
        v_med_shell = np.median(v_shell)
        ax.axvline(
            v_med_shell,
            color="0.25",
            linestyle=":",
            lw=1.0
        )

    # Small information box
    ax.text(
        0.97,
        0.93,
        (
            rf"$N={n_shell}$"
            "\n"
            rf"$r_{{\rm med}}={r_median:.2f}\,\mathrm{{kpc}}$"
            "\n"
            rf"$v_{{\rm esc}}\approx {v_esc_val:.0f}\,\mathrm{{km\,s^{{-1}}}}$"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="0.85",
            alpha=0.95
        )
    )

    ax.set_xlim(0.0, v_plot_max)
    ax.set_ylim(0.0, 0.25)

    ax.set_xlabel(r"Stellar Speed, $v$ [$\mathrm{km}\,\mathrm{s}^{-1}$]")

    if idx == 0:
        ax.set_ylabel(r"Probability Density, $P(v\mid r)$")

    ax.set_title(
        shell["title"] + "\n" + shell["subtitle"],
        loc="left"
    )

    ax.grid(True, which="major", linestyle="-", color="#eaeaea", alpha=0.8)
    ax.grid(False, which="minor")

# One shared legend, outside the data area
handles, labels = axs_vel[0].get_legend_handles_labels()
fig_vel.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
    fontsize=10.5
)

vel_diag_path = os.path.join(OUTPUT_DIR, "velocity_distribution_comparison.png")
plt.savefig(vel_diag_path, dpi=400, bbox_inches="tight")
print(f"Velocity distribution diagnostic plot saved: {vel_diag_path}")

# Restore default matplotlib settings for consecutive terminal scripts
plt.rcParams.update(plt.rcParamsDefault)

print("\nDone.")