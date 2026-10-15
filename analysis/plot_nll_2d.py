"""Original profile helpers. Hull flags are a shape heuristic, not a convergence test."""
import numpy as np
from scipy.optimize import minimize

def m_nfw(x):
    return np.log1p(x) - x / (1 + x)

def rho_c(r, M0, Rs, rc, n):
    x = r / Rs; f = np.tanh(r / rc)
    return f ** n * M0 / (4 * np.pi * Rs ** 3) / (x * (1 + x) ** 2) + n * f ** (n - 1) * (1 - f * f) * M0 * m_nfw(x) / (4 * np.pi * r * r * rc)

def truth_like(rh, gamma, rho0):
    """Meilleure approximation coreNFW (bornes du fitter) du halo vrai sur 0,1-30 R1/2 : (n, R_s/R1/2)."""
    r = rh * np.geomspace(.1, 30, 200); lt = np.log10(rho0 * r ** -gamma * (1 + r) ** (gamma - 3))
    lo, hi = .2 * rh, 5 * rh; sig = lambda z: 1 / (1 + np.exp(-z))
    un = lambda p: (10 ** p[0], 10 ** p[1], lo + (hi - lo) * sig(p[2]), -.5 + 1.5 * sig(p[3]))
    best = None
    for zrc in (-2, 0, 2):
        for lRs in (-0.5, 0, .5):
            for zn in (-1, 1, 3):
                res = minimize(lambda p: np.mean((np.log10(rho_c(r, *un(p))) - lt) ** 2), [9.5, lRs, zrc, zn],
                               method='Nelder-Mead', options=dict(maxiter=30000, xatol=1e-8, fatol=1e-12))
                if best is None or res.fun < best.fun:
                    best = res
    M0, Rs, rc, n = un(best.x)
    return n, Rs / rh

def lower_hull_flags(x, y, tol=3.0):
    """Nœuds au-dessus de l'enveloppe convexe basse de (x, y) de plus de tol."""
    o = np.argsort(x); xs, ys = x[o], y[o]; hull = []
    for p in zip(xs, ys):
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= 0:
            hull.pop()
        hull.append(p)
    hx, hy = np.array(hull).T
    flag = ys > np.interp(xs, hx, hy) + tol
    out = np.zeros_like(flag); out[o] = flag
    return out
