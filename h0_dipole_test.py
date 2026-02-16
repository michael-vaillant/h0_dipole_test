#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dipole Mapping of the Local Hubble Expansion (Analysis Pipeline)

This script performs a generalized least-squares (GLS) fit of the local Hubble
expansion field using Type Ia Supernovae data. It supports:
- Decomposition into Monopole, Kinematic Dipole (1/z), and Constant-amplitude (TEX) dipole (redshift-independent template).
- Tomographic binning.
- Full covariance matrix support.
- Robustness tests: Influence diagnostics, Directional Jackknife, Constant-N tests.
- Null hypothesis testing via footprint-aware permutations.

Usage:
    python h0_dipole.py --dat data.txt --cov cov.txt --zmin 0.0 --zmax 0.15 ...

Author:  Michaël Vaillant
Affil:   Meta-Connexions, Toulouse, France
License: MIT
Version: 6.5.0
DOI:     10.5281/zenodo.18603301
Paper:   "Dipole mapping of the local Hubble expansion", Vaillant (2026)

Dependencies:
    numpy, scipy, matplotlib
"""
import argparse, math
import numpy as np
import sys, time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

C_LIGHT = 299792.458  # km/s

# J2000 equatorial (ICRS-like) -> Galactic rotation matrix (unit vectors)
# Standard numerical values (IAU 1958 system realized in J2000 frame)
EQ2GAL = np.array([
    [-0.0548755604, -0.8734370902, -0.4838350155],
    [ 0.4941094279, -0.4448296300,  0.7469822445],
    [-0.8676661490, -0.1980763734,  0.4559837762],
], dtype=float)

# Anti–Great Attractor (approx.)
AGA_L = 145.3   # deg
AGA_B = 7.2     # deg

# Global column name cache, filled once in main()
RA_COL = DEC_COL = MU_COL = MUERR_COL = Z_COL = None

def parse_bins(s):
    """
    Parses a string like '0.020-0.025,0.025-0.030' into a list of tuples.
    Moved to global scope so it can be used by both main() and mocks.
    """
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
        elif ":" in part:
            a, b = part.split(":", 1)
        else:
            raise RuntimeError(f"Bad bin format in '{part}'. Use e.g. '0-0.03'")
        
        try:
            lo = float(a.strip())
            hi = float(b.strip())
            out.append((lo, hi))
        except ValueError:
            raise RuntimeError(f"Could not parse floats in bin '{part}'")
    return out
    
def _parse_lb(s):
    if not s: return None
    s = s.replace(" ", "")
    a, b = s.split(",")
    return float(a), float(b)

def _uvec_from_lb(l_deg, b_deg):
    l = np.deg2rad(l_deg); b = np.deg2rad(b_deg)
    return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)], dtype=float)

def _progress(i, n, t0, last_t, label="permute", every_sec=0.5):
    now = time.time()
    if (now - last_t) < every_sec and i < n:
        return last_t
    frac = i / n
    elapsed = now - t0
    rate = i / elapsed if elapsed > 0 else 0.0
    msg = f"\r[{label}] {100*frac:6.2f}%  ({i:6d}/{n})  elapsed={elapsed:7.1f}s  rate={rate:6.1f}/s"
    sys.stderr.write(msg)
    sys.stderr.flush()
    return now

def find_col(names, candidates):
    low = [n.lower() for n in names]
    for cand in candidates:
        if cand.lower() in low:
            return names[low.index(cand.lower())]
    return None

def load_table(path):
    data = np.genfromtxt(path, names=True, dtype=None, encoding=None)
    if data.dtype.names is None:
        raise RuntimeError("Could not read named columns. Open the .dat and check header formatting.")
    return data

def unitvec_from_radec(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T  # (N,3)

def radec_to_gal_l_b(ra_deg, dec_deg):
    # Convert equatorial (RA,Dec) degrees to Galactic (l,b) degrees, J2000 frame
    n_eq = unitvec_from_radec(np.array([ra_deg], dtype=float), np.array([dec_deg], dtype=float))[0]  # (3,)
    n_gal = EQ2GAL @ n_eq
    x, y, z = float(n_gal[0]), float(n_gal[1]), float(n_gal[2])
    l = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    b = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    return l, b

def hubble_mu_model(z, H0, q0=-0.55):
    # Low-z cosmography to O(z^2): D_L ≈ (c/H0) * z * [1 + (1 - q0) z / 2]
    dl_mpc = (C_LIGHT / H0) * z * (1.0 + 0.5*(1.0 - q0)*z)
    return 5.0*np.log10(dl_mpc) + 25.0

def wls_fit(X, y, w):
    XT_W = X.T * w
    A = XT_W @ X
    b = XT_W @ y
    beta = np.linalg.solve(A, b)
    return beta

def gls_prepare(C):
    # Cholesky factor for C (assumed SPD)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError as e:
        raise RuntimeError("Covariance is not SPD / Cholesky failed. Try adding --sigint or check cov file.") from e
    return L

def gls_solve(L, A):
    # Solve C^{-1} A using Cholesky: C = L L^T
    # returns X = C^{-1} A
    y = np.linalg.solve(L, A)
    return np.linalg.solve(L.T, y)

def gls_fit(L, X, y):
    # beta = (X^T C^{-1} X)^{-1} X^T C^{-1} y
    CiX = gls_solve(L, X)
    Ciy = gls_solve(L, y)
    A = X.T @ CiX
    b = X.T @ Ciy
    return np.linalg.solve(A, b)

def chi2_gls(L, r):
    # chi2 = r^T C^{-1} r
    v = np.linalg.solve(L, r)
    return float(v @ v)

def chi2_sf_df3(x):
    # Survival function for Chi^2 with 3 dof (exact, closed form)
    # For nu=3: SF(x) = erfc(sqrt(x/2)) + sqrt(2x/pi)*exp(-x/2)
    if x <= 0:
        return 1.0
    t = x / 2.0
    return math.erfc(math.sqrt(t)) + math.sqrt(2.0*x/math.pi) * math.exp(-t)

def chi2_sf_df4(x):
    # For nu=4: SF(x) = exp(-x/2) * (1 + x/2)
    if x <= 0:
        return 1.0
    t = x/2.0
    return math.exp(-t) * (1.0 + t)

def norm_ppf(p):
    # Acklam inverse normal CDF approximation (no scipy)
    # Robust to p=0/1 (returns +/-inf) and NaN.
    if not math.isfinite(p):
        return float("nan")
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")

    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

def norm_isf(p):
    # inverse survival: z such that P(Z>z)=p
    if not math.isfinite(p):
        return float("nan")
    if p <= 0.0:
        return float("inf")
    if p >= 1.0:
        return float("-inf")
    # Use symmetry: P(Z>z)=p  <=>  P(Z<-z)=p  <=>  -z = Φ^{-1}(p)
    if p <= 0.5:
        return -norm_ppf(p)
    # for p>0.5, z is negative; this branch is rarely used here
    return norm_ppf(1.0 - p)


def parse_list(s):
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out

def load_cov(path):
    # Reads Pantheon-style .cov (either N on first entry or raw N^2 entries).
    txt = open(path, "r", encoding="utf-8", errors="ignore").read().split()
    vals = np.array([float(x) for x in txt], dtype=float)
    if len(vals) < 4:
        raise RuntimeError("Cov file too small / unreadable.")
    n0 = int(round(vals[0]))
    if (len(vals)-1) == n0*n0:
        n = n0
        vals = vals[1:]
    else:
        n = int(round(math.sqrt(len(vals))))
        if n*n != len(vals):
            raise RuntimeError(f"Cannot infer N from cov length={len(vals)}")
    C = vals.reshape((n, n))
    return C

def whiten_from_cov(C):
    # Returns L such that C = L L^T (Cholesky). Raises if not PD.
    return np.linalg.cholesky(C)

def solve_lower(L, B):
    # L can be: (N,N) lower-tri (Cholesky) OR (N,) diagonal std vector
    if isinstance(L, np.ndarray) and L.ndim == 1:
        return B / L[:, None] if (isinstance(B, np.ndarray) and B.ndim == 2) else (B / L)
    return np.linalg.solve(L, B)

def gls_fit_and_chi2(L, X, y):
    yw = solve_lower(L, y)
    Xw = solve_lower(L, X)

    # Solve least squares in whitened space (QR/SVD-based internally)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    r = yw - Xw @ beta
    chi2 = float(r.T @ r)
    return beta, chi2

def chi2_only(L, y):
    yw = solve_lower(L, y)
    return float(yw.T @ yw)

def chi2_sf_df1(x):
    # For nu=1: SF(x) = erfc(sqrt(x/2))
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x/2.0))

def chi2_sf_df2(x):
    # For nu=2: SF(x) = exp(-x/2)
    if x <= 0:
        return 1.0
    return math.exp(-x/2.0)

def chi2_sf_df6(x):
    # For nu=6: SF(x) = exp(-x/2) * (1 + x/2 + (x/2)^2/2)
    if x <= 0:
        return 1.0
    t = x/2.0
    return math.exp(-t) * (1.0 + t + 0.5*t*t)

def chi2_sf(x, dof):
    if dof == 1: return chi2_sf_df1(x)
    if dof == 2: return chi2_sf_df2(x)
    if dof == 3: return chi2_sf_df3(x)
    if dof == 4: return chi2_sf_df4(x)
    if dof == 6: return chi2_sf_df6(x)
    raise RuntimeError(f"chi2_sf: unsupported dof={dof} (add a closed-form SF if needed)")


def _chi2_min_from_Xw(A, Xw, yw):
    # A = Xw.T @ Xw (small: 1x1 or 4x4)
    s = float(yw @ yw)
    b = Xw.T @ yw
    beta = np.linalg.solve(A, b)
    return s - float(b @ beta)

def rand_rotation_matrix(rng):
    # Uniform random rotation from random quaternion
    u1, u2, u3 = rng.random(3)
    q1 = math.sqrt(1-u1) * math.sin(2*math.pi*u2)
    q2 = math.sqrt(1-u1) * math.cos(2*math.pi*u2)
    q3 = math.sqrt(u1)   * math.sin(2*math.pi*u3)
    q4 = math.sqrt(u1)   * math.cos(2*math.pi*u3)
    x, y, z, w = q1, q2, q3, q4
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=float)
    return R


def fit_dipole(tab, idx, args, cov_full=None):

    global RA_COL, DEC_COL, MU_COL, MUERR_COL, Z_COL
    names = list(tab.dtype.names)

    ra_name  = RA_COL
    dec_name = DEC_COL
    mu_name  = MU_COL
    mue_name = MUERR_COL
    z_name   = Z_COL

    if args.surveycol:
        surv_name = args.surveycol
    else:
        surv_name = find_col(names, ["IDSURVEY", "idSURVEY", "survey", "SURVEY", "SURVEYID"])

    need = [("RA", ra_name), ("DEC", dec_name), ("MU", mu_name), ("MUERR", mue_name), ("z", z_name)]
    bad = [k for k, v in need if (v is None) or (v not in names)]
    if bad:
        raise RuntimeError("Missing required columns: " + ", ".join(bad) + "\nAvailable:\n" + ", ".join(names))

    ra  = np.array(tab[ra_name],  float)[idx]
    dec = np.array(tab[dec_name], float)[idx]
    mu  = np.array(tab[mu_name],  float)[idx]
    z   = np.array(tab[z_name],   float)[idx]
    muerr = np.array(tab[mue_name], float)[idx]
    muerr = np.maximum(muerr, 1e-6)

    # --- whitening L (diag or full cov) ---
    if (cov_full is not None) and args.use_cov:
        C = cov_full[np.ix_(idx, idx)].copy()
        if args.sigint > 0:
            C[np.diag_indices_from(C)] += args.sigint**2
        if getattr(args, "sigv", 0.0) > 0:
            # Approximate peculiar-velocity contribution to distance-modulus error (low-z):
            # sigma_mu ≈ (5/ln10) * (sigv / (c*z))
            z_safe = np.maximum(z, 1e-6)
            sigmu_v = (5.0/np.log(10.0)) * (args.sigv / (C_LIGHT * z_safe))
            C[np.diag_indices_from(C)] += sigmu_v**2
        L = whiten_from_cov(C)
        # NOTE: do NOT add muerr diag again here (assume cov already includes it)
    else:
        sig2 = muerr*muerr + args.sigint*args.sigint
        if getattr(args, "sigv", 0.0) > 0:
            z_safe = np.maximum(z, 1e-6)
            sigmu_v = (5.0/np.log(10.0)) * (args.sigv / (C_LIGHT * z_safe))
            sig2 = sig2 + sigmu_v*sigmu_v
        L = np.sqrt(sig2)   # (N,) std vector

    # --- H0 grid search (isotropic) ---
    H0_grid = np.linspace(args.h0min, args.h0max, args.h0n)
    best = (1e300, None)
    for H0 in H0_grid:
        mu_th = hubble_mu_model(z, H0, q0=args.q0)
        r = (mu - mu_th).astype(float)
        chi2 = chi2_only(L, r)
        if chi2 < best[0]:
            best = (chi2, H0)

    chi2_iso, H0_best = best
    mu_th = hubble_mu_model(z, H0_best, q0=args.q0)
    dmu = (mu - mu_th).astype(float)

    # --- unit vectors / helpers ---
    n = unitvec_from_radec(ra, dec)      # (N,3) equatorial unit vectors
    zinv = 1.0 / np.maximum(z, 1e-6)

    # Galactic latitude helpers (for MW templates)
    n_gal = n @ EQ2GAL.T
    sinb = n_gal[:, 2]

    # --- fixed axes (galactic input -> equatorial vector) ---
    tex_fix = _parse_lb(getattr(args, "fix_tex_axis", ""))   # (l,b) or None
    kin_fix = _parse_lb(getattr(args, "fix_kin_axis", ""))   # (l,b) or None

    deq_tex = None
    deq_kin = None
    proj_tex = None
    proj_kin = None

    if tex_fix is not None:
        l0, b0 = tex_fix
        dgal = _uvec_from_lb(l0, b0)         # (3,)
        deq_tex = (EQ2GAL.T @ dgal).astype(float)
        proj_tex = (n @ deq_tex).astype(float)  # (N,)

    if kin_fix is not None:
        l0, b0 = kin_fix
        dgal = _uvec_from_lb(l0, b0)
        deq_kin = (EQ2GAL.T @ dgal).astype(float)
        proj_kin = (n @ deq_kin).astype(float)

    # --- design matrices ---
    X0 = np.ones((len(dmu), 1), float)  # monopole only

    # Tex : either full D·n (3 params) or fixed-axis A*(d·n) (1 param)
    if tex_fix is None:
        X_tex = np.column_stack([np.ones(len(dmu)), n[:,0], n[:,1], n[:,2]])    # a0 + D·n
    else:
        X_tex = np.column_stack([np.ones(len(dmu)), proj_tex])                 # a0 + A*(d·n)

    # kinematic: either full (K·n)/z (3 params) or fixed-axis B*(d·n)/z (1 param)
    if kin_fix is None:
        X_kin = np.column_stack([np.ones(len(dmu)), n[:,0]*zinv, n[:,1]*zinv, n[:,2]*zinv])  # a0 + (K·n)/z
    else:
        X_kin = np.column_stack([np.ones(len(dmu)), proj_kin*zinv])                              # a0 + B*(d·n)/z

    # mix: concatenate the chosen tex + chosen kinematic blocks
    mix_cols = [np.ones(len(dmu))]
    if tex_fix is None:
        mix_cols += [n[:,0], n[:,1], n[:,2]]
    else:
        mix_cols += [proj_tex]
    if kin_fix is None:
        mix_cols += [n[:,0]*zinv, n[:,1]*zinv, n[:,2]*zinv]
    else:
        mix_cols += [proj_kin*zinv]
    X_mix = np.column_stack(mix_cols)

    # --- fits ---
    beta0,   chi2_mono = gls_fit_and_chi2(L, X0,    dmu)
    beta_tx, chi2_tex  = gls_fit_and_chi2(L, X_tex, dmu)
    beta_kn, chi2_kin  = gls_fit_and_chi2(L, X_kin, dmu)
    beta_mx, chi2_mix  = gls_fit_and_chi2(L, X_mix, dmu)

    a0 = float(beta0[0])

    # Keep legacy naming: "dip" = tex dipole component
    chi2_dip = chi2_tex
    dchi2_4dof = chi2_iso  - chi2_tex
    dchi2_tex  = chi2_mono - chi2_tex

    dchi2_kin = chi2_mono - chi2_kin
    dchi2_mix = chi2_mono - chi2_mix

    # incremental partition diagnostics (these χ² diffs are always valid; ddl depends on axis-fixing)
    dchi2_add_kin_given_tex = chi2_tex - chi2_mix
    dchi2_add_tex_given_kin = chi2_kin - chi2_mix

    # --- tex dipole params / vectors ---
    if tex_fix is None:
        D_vec = beta_tx[1:4].astype(float)
        A_mu = float(np.linalg.norm(D_vec))
        if A_mu > 0:
            Dx, Dy, Dz = D_vec
            ra_hat  = (math.degrees(math.atan2(Dy, Dx)) + 360.0) % 360.0
            dec_hat = math.degrees(math.asin(Dz / A_mu))
            l_hat, b_hat = radec_to_gal_l_b(ra_hat, dec_hat)
        else:
            ra_hat = dec_hat = l_hat = b_hat = float("nan")
    else:
        amp = float(beta_tx[1])      # signed amplitude along fixed axis
        # enforce positive amplitude by flipping axis if needed (pure convention)
        if amp < 0:
            amp = -amp
            deq_tex = -deq_tex
        D_vec = (amp * deq_tex).astype(float)
        A_mu  = float(amp)
        # direction from deq_tex
        x, y, zc = float(deq_tex[0]), float(deq_tex[1]), float(deq_tex[2])
        ra_hat  = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
        dec_hat = math.degrees(math.asin(max(-1.0, min(1.0, zc))))
        l_hat, b_hat = radec_to_gal_l_b(ra_hat, dec_hat)

    frac = (math.log(10.0)/5.0) * A_mu  # |δH0/H0|

    # --- kinematic dipole params / vectors ---
    if kin_fix is None:
        K_vec = beta_kn[1:4].astype(float)
        A_mu_kin = float(np.linalg.norm(K_vec))
        if A_mu_kin > 0:
            Kx, Ky, Kz = K_vec
            ra_k  = (math.degrees(math.atan2(Ky, Kx)) + 360.0) % 360.0
            dec_k = math.degrees(math.asin(Kz / A_mu_kin))
            l_k, b_k = radec_to_gal_l_b(ra_k, dec_k)
        else:
            ra_k = dec_k = l_k = b_k = float("nan")
    else:
        amp = float(beta_kn[1])  # signed
        if amp < 0:
            amp = -amp
            deq_kin = -deq_kin
        K_vec = (amp * deq_kin).astype(float)
        A_mu_kin = float(amp)
        x, y, zc = float(deq_kin[0]), float(deq_kin[1]), float(deq_kin[2])
        ra_k  = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
        dec_k = math.degrees(math.asin(max(-1.0, min(1.0, zc))))
        l_k, b_k = radec_to_gal_l_b(ra_k, dec_k)

    v_bulk = (math.log(10.0)/5.0) * C_LIGHT * A_mu_kin

    # --- mixed params (extract best-fit vectors in equatorial coords) ---
    # Note: columns depend on which axes are fixed; we rebuild by reading beta_mx accordingly.
    # beta_mx = [a0, (tex block...), (kin block...)]
    j = 1
    if tex_fix is None:
        Dm = beta_mx[j:j+3].astype(float); j += 3
    else:
        amp = float(beta_mx[j]); j += 1
        if amp < 0:
            amp = -amp
            deq_tex_m = -deq_tex
        else:
            deq_tex_m = deq_tex
        Dm = (amp * deq_tex_m).astype(float)

    if kin_fix is None:
        Km = beta_mx[j:j+3].astype(float); j += 3
    else:
        amp = float(beta_mx[j]); j += 1
        if amp < 0:
            amp = -amp
            deq_kin_m = -deq_kin
        else:
            deq_kin_m = deq_kin
        Km = (amp * deq_kin_m).astype(float)

    A_mu_mix_tex = float(np.linalg.norm(Dm))
    A_mu_mix_kin = float(np.linalg.norm(Km))
    v_bulk_mix = (math.log(10.0)/5.0) * C_LIGHT * A_mu_mix_kin

    if A_mu_mix_tex > 0:
        Dx, Dy, Dz = Dm
        ra_mD  = (math.degrees(math.atan2(Dy, Dx)) + 360.0) % 360.0
        dec_mD = math.degrees(math.asin(Dz / A_mu_mix_tex))
        l_mD, b_mD = radec_to_gal_l_b(ra_mD, dec_mD)
    else:
        ra_mD = dec_mD = l_mD = b_mD = float("nan")

    if A_mu_mix_kin > 0:
        Kx, Ky, Kz = Km
        ra_mK  = (math.degrees(math.atan2(Ky, Kx)) + 360.0) % 360.0
        dec_mK = math.degrees(math.asin(Kz / A_mu_mix_kin))
        l_mK, b_mK = radec_to_gal_l_b(ra_mK, dec_mK)
    else:
        ra_mK = dec_mK = l_mK = b_mK = float("nan")

    # --- p-values: ddl depend on whether axes are fixed ---
    dof_tex = 1 if (tex_fix is not None) else 3
    dof_kin = 1 if (kin_fix is not None) else 3
    
    # "mix vs mono": add (tex block) + (kin block)
    # df can be 2 (both fixed), 4 (one fixed, one free), or 6 (both free)
    dof_mix = (1 if tex_fix is not None else 3) + (1 if kin_fix is not None else 3)

    # --- P-values & Sigmas (Cleaned version) ---
    
    # TEX
    p_tex = chi2_sf(dchi2_tex, dof_tex)
    z1_tex = norm_isf(p_tex) if (0.0 < p_tex < 1.0 and math.isfinite(p_tex)) else float("nan")
    z2_tex = norm_isf(p_tex/2.0) if (0.0 < p_tex < 1.0 and math.isfinite(p_tex)) else float("nan")

    # KIN
    p_kin = chi2_sf(dchi2_kin, dof_kin)
    z1_kin = norm_isf(p_kin) if (0.0 < p_kin < 1.0 and math.isfinite(p_kin)) else float("nan")
    z2_kin = norm_isf(p_kin/2.0) if (0.0 < p_kin < 1.0 and math.isfinite(p_kin)) else float("nan")

    # MIX (df=2, 4, or 6)
    p_mix = chi2_sf(dchi2_mix, dof_mix)
    z1_mix = norm_isf(p_mix) if (0.0 < p_mix < 1.0 and math.isfinite(p_mix)) else float("nan")
    z2_mix = norm_isf(p_mix/2.0) if (0.0 < p_mix < 1.0 and math.isfinite(p_mix)) else float("nan")

    # Conditional adds (Nested models)
    dof_add_kin = 1 if (kin_fix is not None) else 3
    dof_add_tex = 1 if (tex_fix is not None) else 3

    p_add_kin_given_tex = chi2_sf(dchi2_add_kin_given_tex, dof_add_kin)
    p_add_tex_given_kin = chi2_sf(dchi2_add_tex_given_kin, dof_add_tex)
    
    # # --- p-values: ddl depend on whether axes are fixed ---
    # dof_tex = 1 if (tex_fix is not None) else 3
    # dof_kin = 1 if (kin_fix is not None) else 3
    # # "mix vs mono": add (tex block) + (kin block)
    # dof_mix = (1 if tex_fix is not None else 3) + (1 if kin_fix is not None else 3)

    # if dof_tex == 1:
    #     p_tex = chi2_sf_df1(dchi2_tex)
    # else:
    #     p_tex = chi2_sf_df3(dchi2_tex)

    # if dof_kin == 1:
    #     p_kin = chi2_sf_df1(dchi2_kin)
    # else:
    #     p_kin = chi2_sf_df3(dchi2_kin)

    # # mix: only df=6 or df=2 are possible here 
    # if dof_mix == 6:
    #     p_mix = chi2_sf_df6(dchi2_mix) 
    # elif dof_mix == 4:
    #     p_mix = chi2_sf_df4(dchi2_mix) 
    # elif dof_mix == 2:
    #     p_mix = math.exp(-0.5*max(0.0, dchi2_mix)) 
    # else:
    #     p_mix = float("nan")    

    # z1_tex = norm_isf(p_tex) if (0 < p_tex < 1 and math.isfinite(p_tex)) else float("nan")

    # # conditional adds: ddl are those of the added block
    # dof_add_kin = 1 if (kin_fix is not None) else 3
    # dof_add_tex = 1 if (tex_fix is not None) else 3

    # if dof_add_kin == 1:
    #     p_add_kin_given_tex = chi2_sf_df1(dchi2_add_kin_given_tex)
    # else:
    #     p_add_kin_given_tex = chi2_sf_df3(dchi2_add_kin_given_tex)

    # if dof_add_tex == 1:
    #     p_add_tex_given_kin = chi2_sf_df1(dchi2_add_tex_given_kin)
    # else:
    #     p_add_tex_given_kin = chi2_sf_df3(dchi2_add_tex_given_kin)

    # # --- p-values: ddl depend on whether axes are fixed ---
    # dof_tex = 1 if (tex_fix is not None) else 3
    # dof_kin = 1 if (kin_fix is not None) else 3
    # dof_mix = (1 if tex_fix is not None else 3) + (1 if kin_fix is not None else 3)  # 2 or 6

    # # TEX
    # p_tex = chi2_sf(dchi2_tex, dof_tex)
    # z1_tex = norm_isf(p_tex) if (0.0 < p_tex < 1.0 and math.isfinite(p_tex)) else float("nan")
    # z2_tex = norm_isf(p_tex/2.0) if (0.0 < p_tex < 1.0 and math.isfinite(p_tex)) else float("nan")

    # # KIN
    # p_kin = chi2_sf(dchi2_kin, dof_kin)
    # z1_kin = norm_isf(p_kin) if (0.0 < p_kin < 1.0 and math.isfinite(p_kin)) else float("nan")
    # z2_kin = norm_isf(p_kin/2.0) if (0.0 < p_kin < 1.0 and math.isfinite(p_kin)) else float("nan")

    # # MIX (df=2 or 6)  -> closed forms for both
    # p_mix = chi2_sf(dchi2_mix, dof_mix)
    # z1_mix = norm_isf(p_mix) if (0.0 < p_mix < 1.0 and math.isfinite(p_mix)) else float("nan")
    # z2_mix = norm_isf(p_mix/2.0) if (0.0 < p_mix < 1.0 and math.isfinite(p_mix)) else float("nan")

    # # conditional adds: ddl are those of the added block
    # dof_add_kin = 1 if (kin_fix is not None) else 3
    # dof_add_tex = 1 if (tex_fix is not None) else 3

    # p_add_kin_given_tex = chi2_sf(dchi2_add_kin_given_tex, dof_add_kin)
    # p_add_tex_given_kin = chi2_sf(dchi2_add_tex_given_kin, dof_add_tex)

    # H0-only -> (a0 + dip) comparison kept as before
    p4 = chi2_sf_df4(dchi2_4dof)
    z1_4 = norm_isf(p4) if (0.0 < p4 < 1.0) else float("nan")
    z2_4 = norm_isf(p4/2.0) if (0.0 < p4 < 1.0) else float("nan")
    a0_mix = float(beta_mx[0])
    
    # error bars
    # Mono (isotrope)
    # 1. For Monopole model (a0_mono)
    X0w = solve_lower(L, X0)
    H0_mat = X0w.T @ X0w
    
    # Robust scalar extraction (fixes NumPy 1.25+ DeprecationWarning)
    if np.ndim(H0_mat) == 0 or H0_mat.size == 1:
        val = H0_mat.item()
        var_a0_mono = 1.0 / val if val > 0 else float('inf')
    else:
        var_a0_mono = np.linalg.inv(H0_mat)[0, 0]
        
    err_a0_mono = math.sqrt(var_a0_mono)

    # 2. For Mixed model (a0_mix)
    Xmixw = solve_lower(L, X_mix)
    Hmix_mat = Xmixw.T @ Xmixw
    
    # Invert matrix (size ~7x7) to get variance of the first parameter (intercept)
    try:
        Cov_mix = np.linalg.inv(Hmix_mat)
        var_a0_mix = Cov_mix[0, 0] 
        err_a0_mix = math.sqrt(max(0.0, var_a0_mix))
    except np.linalg.LinAlgError:
        err_a0_mix = float('nan') 
    
    out = dict(
        N=len(dmu), H0=H0_best,
        chi2_iso=chi2_iso, chi2_mono=chi2_mono, chi2_dip=chi2_dip,

        dchi2_4=dchi2_4dof, p4=p4, z1_4=z1_4, z2_4=z2_4,

        # tex (canonical keys)
        dchi2_tex=dchi2_tex, p_tex=p_tex, z_tex=z1_tex, dof_tex=dof_tex,

        # legacy/compat keys for tex (kept)
        dchi2_3=dchi2_tex, p3=p_tex, z1_3=z1_tex, z2_3=z2_tex,

        a0=a0, 
        a0_mix=a0_mix,              # Intercept from the full MIX model
        err_a0_mono=err_a0_mono,    # Error on monopole intercept
        err_a0_mix=err_a0_mix,      # Error on mix intercept
        
        A_mu=A_mu, frac=frac,
        ra=ra_hat, dec=dec_hat, l=l_hat, b=b_hat,
        surveycol=surv_name,

        D_vec=D_vec,
        K_vec=K_vec,

        # kinematic
        chi2_kin=chi2_kin, dchi2_kin=dchi2_kin, p_kin=p_kin, dof_kin=dof_kin,
        A_mu_kin=A_mu_kin, ra_kin=ra_k, dec_kin=dec_k, l_kin=l_k, b_kin=b_k, v_bulk=v_bulk,

        # legacy/compat keys for kinematic
        p3_kin=p_kin, z1_3_kin=z1_kin, z2_3_kin=z2_kin,

        # mixed
        chi2_mix=chi2_mix, dchi2_mix=dchi2_mix, p_mix=p_mix, dof_mix=dof_mix,
        z1_mix=z1_mix, z2_mix=z2_mix,

        dchi2_add_kin_given_tex=dchi2_add_kin_given_tex,
        dchi2_add_tex_given_kin=dchi2_add_tex_given_kin,
        p_add_kin_given_tex=p_add_kin_given_tex,
        p_add_tex_given_kin=p_add_tex_given_kin,

        A_mu_mix_tex=A_mu_mix_tex, ra_mix_tex=ra_mD, dec_mix_tex=dec_mD, l_mix_tex=l_mD, b_mix_tex=b_mD,
        A_mu_mix_kin=A_mu_mix_kin, ra_mix_kin=ra_mK, dec_mix_kin=dec_mK, l_mix_kin=l_mK, b_mix_kin=b_mK,
        v_bulk_mix=v_bulk_mix
    )

    if surv_name and (surv_name in names):
        out["surveys"] = np.array(tab[surv_name])[idx]

    # --- MW template / "corset" checks (inchangés) ---
    if getattr(args, "mwcheck", False):
        tname = (getattr(args, "mwtemp", "p2") or "p2").lower()
        if tname == "sinb":
            t = sinb
        elif tname == "abs_sinb":
            t = np.abs(sinb)
        elif tname == "inv_abs_sinb":
            floor = float(getattr(args, "mw_sinb_min", 0.05))
            t = 1.0 / np.maximum(np.abs(sinb), floor)
        else:
            t = 0.5*(3.0*sinb*sinb - 1.0)

        Xmw    = np.column_stack([np.ones(len(dmu)), t])
        Xmwdip = np.column_stack([np.ones(len(dmu)), t, n[:,0], n[:,1], n[:,2]])

        beta_mw,    chi2_mw    = gls_fit_and_chi2(L, Xmw, dmu)
        beta_mwdip, chi2_mwdip = gls_fit_and_chi2(L, Xmwdip, dmu)

        dchi2_mw = chi2_mono - chi2_mw
        dchi2_dip_given_mw = chi2_mw - chi2_mwdip

        p_mw = chi2_sf_df1(dchi2_mw)
        p_dip_given_mw = chi2_sf_df3(dchi2_dip_given_mw)

        D_mw = beta_mwdip[-3:]
        A_mu_mw = float(np.linalg.norm(D_mw))

        if A_mu_mw > 0:
            Dx, Dy, Dz = D_mw
            ra_mw  = (math.degrees(math.atan2(Dy, Dx)) + 360.0) % 360.0
            dec_mw = math.degrees(math.asin(Dz / A_mu_mw))
            l_mw, b_mw = radec_to_gal_l_b(ra_mw, dec_mw)
        else:
            ra_mw = dec_mw = l_mw = b_mw = float("nan")

        dm = dmu - np.mean(dmu)
        tt = t - np.mean(t)
        corr_mw = float((dm @ tt) / (np.linalg.norm(dm)*np.linalg.norm(tt) + 1e-30))

        out.update(dict(
            mwtemp=tname,
            corr_mw=corr_mw,
            chi2_mw=chi2_mw,
            chi2_mwdip=chi2_mwdip,
            dchi2_mw=dchi2_mw,
            p_mw=p_mw,
            z_mw=norm_isf(p_mw),
            dchi2_dip_given_mw=dchi2_dip_given_mw,
            p_dip_given_mw=p_dip_given_mw,
            z_dip_given_mw=norm_isf(p_dip_given_mw),
            A_mu_mw=A_mu_mw,
            ra_mw=ra_mw, dec_mw=dec_mw, l_mw=l_mw, b_mw=b_mw
        ))

    # --- Dust analysis (inchangée) ---
    dust = None
    dustcol = None
    if args.dustcheck:
        dustcol = args.dustcol if args.dustcol else find_col(names, ["MWEBV","mwebv","E_BV","ebv"])
        if (dustcol is None) or (dustcol not in names):
            raise RuntimeError("Dust check requested but dust column not found. Use --dustcol.")
        dust = np.array(tab[dustcol], float)[idx]

    if dust is not None:
        Xd    = np.column_stack([np.ones(len(dmu)), dust])
        Xddip = np.column_stack([np.ones(len(dmu)), dust, n[:,0], n[:,1], n[:,2]])

        beta_d,    chi2_d    = gls_fit_and_chi2(L, Xd, dmu)
        beta_ddip, chi2_ddip = gls_fit_and_chi2(L, Xddip, dmu)

        dchi2_dust   = chi2_mono - chi2_d
        dchi2_dip_cd = chi2_d - chi2_ddip

        p_dust = chi2_sf_df1(dchi2_dust)
        p_cd   = chi2_sf_df3(dchi2_dip_cd)

        D_cd = beta_ddip[-3:]
        A_mu_cd = float(np.linalg.norm(D_cd))

        if A_mu_cd > 0:
            Dx, Dy, Dz = D_cd
            ra_cd = (math.degrees(math.atan2(Dy, Dx)) + 360.0) % 360.0
            dec_cd = math.degrees(math.asin(Dz / A_mu_cd))
            l_cd, b_cd = radec_to_gal_l_b(ra_cd, dec_cd)
        else:
            ra_cd = dec_cd = l_cd = b_cd = float("nan")

        dm = dmu - np.mean(dmu)
        ds = dust - np.mean(dust)
        corr_dust = float((dm @ ds) / (np.linalg.norm(dm)*np.linalg.norm(ds) + 1e-30))

        out.update(dict(
            corr_dust=corr_dust,
            ra_cd=ra_cd, dec_cd=dec_cd, l_cd=l_cd, b_cd=b_cd,
            dustcol=dustcol,
            chi2_dust=chi2_d,
            chi2_ddip=chi2_ddip,
            dchi2_dust=dchi2_dust,
            p_dust=p_dust,
            z_dust=norm_isf(p_dust),
            dchi2_dip_cd=dchi2_dip_cd,
            p_cd=p_cd,
            z_cd=norm_isf(p_cd),
            k_dust=float(beta_d[1]),
            A_mu_cd=A_mu_cd,
            frac_cd=(math.log(10.0)/5.0)*A_mu_cd
        ))

    return out



def permute_pvalue(dmu, L, X0, X, dchi2_obs, nperm, seed=0):
    rng = np.random.default_rng(seed)

    # Pre-whiten design matrices once
    X0w = solve_lower(L, X0)
    Xw  = solve_lower(L, X)
    A0  = X0w.T @ X0w
    A   = Xw.T  @ Xw

    ge = 0
    t0 = time.time()
    last = t0
        
    for i in range(1, nperm + 1):
        y = dmu[rng.permutation(len(dmu))]
        yw = solve_lower(L, y)
        chi2_mono = _chi2_min_from_Xw(A0, X0w, yw)
        chi2_dip  = _chi2_min_from_Xw(A,  Xw,  yw)
        if (chi2_mono - chi2_dip) >= dchi2_obs:
            ge += 1
            
        last = _progress(i, nperm, t0, last, label="permute(global)")
        
    sys.stderr.write("\n")
    sys.stderr.flush()        
    return (ge + 1.0) / (nperm + 1.0)

def permute_pvalue_within_survey(dmu, surveys, L, X0, X, dchi2_obs, nperm, seed=0):
    rng = np.random.default_rng(seed)

    surveys = np.asarray(surveys)
    uniq = np.unique(surveys)
    groups = [np.where(surveys == s)[0] for s in uniq]

    # Pre-whiten design matrices once
    X0w = solve_lower(L, X0)
    Xw  = solve_lower(L, X)
    A0  = X0w.T @ X0w
    A   = Xw.T  @ Xw

    perm_idx = np.arange(len(dmu))
    ge = 0
    t0 = time.time()
    last = t0
        
    for i in range(1, nperm + 1):
        for g in groups:
            perm_idx[g] = rng.permutation(g) if g.size > 1 else g

        y  = dmu[perm_idx]
        yw = solve_lower(L, y)

    
        chi2_mono = _chi2_min_from_Xw(A0, X0w, yw)
        chi2_dip  = _chi2_min_from_Xw(A,  Xw,  yw)
        if (chi2_mono - chi2_dip) >= dchi2_obs:
            ge += 1
        last = _progress(i, nperm, t0, last, label="permute(within)")
        
    sys.stderr.write("\n")
    sys.stderr.flush()    
    return (ge + 1.0) / (nperm + 1.0)

def precompute_Q(Xw):
    # QR reduced: Xw = Q R, with Q orthonormal columns
    Q, R = np.linalg.qr(Xw, mode="reduced")
    return Q

def chi2_from_Q(yw, Q):
    # chi2 = ||yw - Q(Q^T yw)||^2 = ||yw||^2 - ||Q^T yw||^2
    t = Q.T @ yw
    return float(yw @ yw - t @ t)

def save_and_plot_null(stats, obs_val, out_root):
    """
    Save permutation stats to CSV and generate the PNG plot.
    out_root: file prefix (e.g. 'output/fig_null_TEX')
    """
    from scipy.stats import chi2
    import matplotlib.pyplot as plt
    import numpy as np

    stats = np.asarray(stats, dtype=float)
    stats = stats[np.isfinite(stats)]

    # 1) Save CSV
    out_csv = out_root + ".csv"
    try:
        np.savetxt(out_csv, stats, header="dchi2_perm", fmt="%.6f")
        print(f"[INFO] Data saved to {out_csv}")
    except Exception as e:
        print(f"[WARN] Could not save CSV: {e}")

    # Empirical p (with clean "0 hits" handling)
    nperm = len(stats)
    hits = int(np.sum(stats >= obs_val))
    p_hat = (hits + 1.0) / (nperm + 1.0)
    if hits == 0:
        p_txt = f"< {1.0/(nperm+1.0):.1e}  (0/{nperm})"
    else:
        p_txt = f"= {p_hat:.1e}  ({hits}/{nperm})"

    # 2) Plot
    out_png = out_root + ".png"
    try:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.hist(stats, bins=50, density=True, alpha=0.6,
                label="Null (permutations)")

        # Optional "reference" curve 
        # Skip it for MAX (no simple df)
        show_ref = True
        df = 3
        if out_root.endswith("_MIX"):
            df = 6
        if out_root.endswith("_MAX"):
            show_ref = False

        limit = max(np.max(stats), obs_val) * 1.1
        x = np.linspace(0, limit, 200)

        if show_ref:
            ax.plot(x, chi2.pdf(x, df=df), "k--", lw=1.5,
                    label=fr"Reference $\chi^2_{{{df}}}$")

        ax.axvline(obs_val, color='#CC3311', linestyle="-", linewidth=2,
                   label=fr"Observed $\Delta\chi^2={obs_val:.1f}$")

        ax.set_yscale("log")
        ax.set_xlabel(r"$\Delta\chi^2$ (improvement over isotropy)")
        ax.set_ylabel("Probability density")
        ax.set_title(f"Null distribution (Nperm={nperm})")
        ax.legend()

        ax.text(0.95, 0.5, f"Empirical p {p_txt}",
                transform=ax.transAxes, ha="right", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8))

        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[INFO] Plot saved to {out_png}")

    except Exception as e:
        print(f"[WARN] Could not generate plot: {e}")


    except Exception as e:
        print(f"[WARN] Could not generate plot: {e}")

        
def permute_pvalue_within_survey_whitened(yw, surveys, Q0, Q, dchi2_obs, nperm, seed=0, progress=True, topk=8, plot_output=None):
    import numpy as np
    import sys, time, heapq

    rng = np.random.default_rng(seed)

    surveys = np.asarray(surveys)
    uniq = np.unique(surveys)
    groups = [np.where(surveys == s)[0] for s in uniq]
    perm_idx = np.arange(len(yw))

    # -----------------------------
    # MODE 1: single-stat (legacy)
    # -----------------------------
    if not isinstance(Q, dict):
        ge = 0
        t0 = time.time()
        step = max(1, nperm // 100)

        max_dchi2 = -1e300
        top = []
        all_stats = []

        for i in range(nperm):
            for g in groups:
                if g.size > 1:
                    perm_idx[g] = g[rng.permutation(g.size)]
                else:
                    perm_idx[g] = g

            y = yw[perm_idx]

            chi2_mono = chi2_from_Q(y, Q0)
            chi2_dip  = chi2_from_Q(y, Q)
            dchi2 = chi2_mono - chi2_dip
            all_stats.append(dchi2)

            if dchi2 > max_dchi2:
                max_dchi2 = dchi2

            if topk > 0:
                if len(top) < topk:
                    heapq.heappush(top, (dchi2, i))
                else:
                    if dchi2 > top[0][0]:
                        heapq.heapreplace(top, (dchi2, i))

            if dchi2 >= dchi2_obs:
                ge += 1

            if progress and ((i + 1) % step == 0 or (i + 1) == nperm):
                dt = time.time() - t0
                rate = (i + 1) / max(dt, 1e-9)
                sys.stdout.write(
                    f"\r[permute(within, whitened)] {100*(i+1)/nperm:6.2f}%  ({i+1}/{nperm})  "
                    f"hits={ge}  max_dchi2={max_dchi2:8.3f}  elapsed={dt:7.1f}s  rate={rate:5.1f}/s"
                )
                sys.stdout.flush()

        if progress:
            sys.stdout.write("\n")

        p_emp = (ge + 1.0) / (nperm + 1.0)
        gap = dchi2_obs - max_dchi2
        top_sorted = sorted(top, key=lambda x: x[0], reverse=True)

        if plot_output:
            save_and_plot_null(all_stats, dchi2_obs, plot_output)

        return p_emp, ge, max_dchi2, gap, top_sorted

    # -----------------------------
    # MODE 2: multi-stat + max
    # -----------------------------
    if not isinstance(dchi2_obs, dict):
        raise RuntimeError("Multi mode requires dchi2_obs as a dict matching Q keys (plus optional dk/kd/max).")

    req = [k for k in dchi2_obs.keys() if k != "max"]
    if len(req) == 0:
        raise RuntimeError("Multi mode: dchi2_obs must contain at least one key (e.g. 'tex').")

    Qkeys = list(Q.keys())

    ge = {k: 0 for k in req}
    max_seen = {k: -1e300 for k in req}
    heaps = {k: [] for k in req}
    all_stats = {k: [] for k in req}

    obs = {k: float(dchi2_obs[k]) for k in req}
    obs_max = float(dchi2_obs["max"]) if "max" in dchi2_obs else max(obs[k] for k in req)

    ge_max = 0
    max_seen_max = -1e300
    heap_max = []
    stats_max = []

    t0 = time.time()
    step = max(1, nperm // 100)

    for i in range(nperm):
        for g in groups:
            if g.size > 1:
                perm_idx[g] = g[rng.permutation(g.size)]
            else:
                perm_idx[g] = g

        y = yw[perm_idx]

        chi2_0 = chi2_from_Q(y, Q0)

        chi2_map = {}
        for k in Qkeys:
            chi2_map[k] = chi2_from_Q(y, Q[k])

        d = {}
        for k in Qkeys:
            d[k] = chi2_0 - chi2_map[k]

        if ("kin" in chi2_map) and ("mix" in chi2_map):
            d["dk"] = chi2_map["kin"] - chi2_map["mix"]
        if ("tex" in chi2_map) and ("mix" in chi2_map):
            d["kd"] = chi2_map["tex"] - chi2_map["mix"]

        for k in req:
            if k not in d:
                raise RuntimeError(f"Requested stat '{k}' not computable from Q. Available: {sorted(d.keys())}")

        for k in req:
            v = d[k]
            all_stats[k].append(v)

            if v > max_seen[k]:
                max_seen[k] = v

            if topk > 0:
                h = heaps[k]
                if len(h) < topk:
                    heapq.heappush(h, (v, i))
                else:
                    if v > h[0][0]:
                        heapq.heapreplace(h, (v, i))

            if v >= obs[k]:
                ge[k] += 1

        v_max = max(d[k] for k in req)
        stats_max.append(v_max)

        if v_max > max_seen_max:
            max_seen_max = v_max

        if topk > 0:
            if len(heap_max) < topk:
                heapq.heappush(heap_max, (v_max, i))
            else:
                if v_max > heap_max[0][0]:
                    heapq.heapreplace(heap_max, (v_max, i))

        if v_max >= obs_max:
            ge_max += 1

        if progress and ((i + 1) % step == 0 or (i + 1) == nperm):
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-9)
            sys.stdout.write(
                f"\r[permute(within, whitened, multi)] {100*(i+1)/nperm:6.2f}%  ({i+1}/{nperm})  "
                f"hits_max={ge_max}  max_stat={max_seen_max:8.3f}  elapsed={dt:7.1f}s  rate={rate:5.1f}/s"
            )
            sys.stdout.flush()

    if progress:
        sys.stdout.write("\n")

    p_emp = {k: (ge[k] + 1.0) / (nperm + 1.0) for k in req}
    gap = {k: obs[k] - max_seen[k] for k in req}
    top_sorted = {k: sorted(heaps[k], key=lambda x: x[0], reverse=True) for k in req}

    p_max = (ge_max + 1.0) / (nperm + 1.0)
    gap_max = obs_max - max_seen_max
    top_max = sorted(heap_max, key=lambda x: x[0], reverse=True)

    if plot_output:
        try:
            cols = req + ["max"]
            mat = np.column_stack([np.array(all_stats[k], dtype=float) for k in req] + [np.array(stats_max, dtype=float)])
            header = ",".join([f"dchi2_{k}" for k in cols])
            np.savetxt(plot_output + "_ALL.csv", mat, delimiter=",", header=header, fmt="%.6f")
            print(f"[INFO] Data saved to {plot_output + '_ALL.csv'}")
        except Exception as e:
            print(f"[WARN] Could not save multi CSV: {e}")

        if "tex" in all_stats:
            save_and_plot_null(all_stats["tex"], obs.get("tex", 0.0), plot_output + "_TEX")
        if "kin" in all_stats:
            save_and_plot_null(all_stats["kin"], obs.get("kin", 0.0), plot_output + "_KIN")
        save_and_plot_null(stats_max, obs_max, plot_output + "_MAX")

    p_emp["max"] = p_max
    ge["max"] = ge_max
    max_seen["max"] = max_seen_max
    gap["max"] = gap_max
    top_sorted["max"] = top_max

    return p_emp, ge, max_seen, gap, top_sorted


    
def permute_pvalue_whitened(yw, Q0, Q, dchi2_obs, nperm, seed=0, progress=True, topk=8, plot_output=None):
    import numpy as np
    import sys, time, heapq

    rng = np.random.default_rng(seed)

    # -----------------------------
    # MODE 1: single-stat (legacy)
    # -----------------------------
    if not isinstance(Q, dict):
        ge = 0
        t0 = time.time()
        step = max(1, nperm // 100)

        max_dchi2 = -1e300
        top = []      # min-heap of (dchi2, i)
        all_stats = []

        for i in range(nperm):
            y = yw[rng.permutation(len(yw))]

            chi2_mono = chi2_from_Q(y, Q0)
            chi2_dip  = chi2_from_Q(y, Q)
            dchi2 = chi2_mono - chi2_dip
            all_stats.append(dchi2)

            if dchi2 > max_dchi2:
                max_dchi2 = dchi2

            if topk > 0:
                if len(top) < topk:
                    heapq.heappush(top, (dchi2, i))
                else:
                    if dchi2 > top[0][0]:
                        heapq.heapreplace(top, (dchi2, i))

            if dchi2 >= dchi2_obs:
                ge += 1

            if progress and ((i + 1) % step == 0 or (i + 1) == nperm):
                dt = time.time() - t0
                rate = (i + 1) / max(dt, 1e-9)
                sys.stdout.write(
                    f"\r[permute(global, whitened)] {100*(i+1)/nperm:6.2f}%  ({i+1}/{nperm})  "
                    f"hits={ge}  max_dchi2={max_dchi2:8.3f}  elapsed={dt:7.1f}s  rate={rate:5.1f}/s"
                )
                sys.stdout.flush()

        if progress:
            sys.stdout.write("\n")

        p_emp = (ge + 1.0) / (nperm + 1.0)
        gap = dchi2_obs - max_dchi2
        top_sorted = sorted(top, key=lambda x: x[0], reverse=True)

        if plot_output:
            save_and_plot_null(all_stats, dchi2_obs, plot_output)

        return p_emp, ge, max_dchi2, gap, top_sorted

    # -----------------------------
    # MODE 2: multi-stat + max
    # Q is dict, dchi2_obs is dict
    # -----------------------------
    if not isinstance(dchi2_obs, dict):
        raise RuntimeError("Multi mode requires dchi2_obs as a dict matching Q keys (plus optional dk/kd/max).")

    # Which stats are requested? (everything in dchi2_obs except 'max')
    req = [k for k in dchi2_obs.keys() if k != "max"]
    if len(req) == 0:
        raise RuntimeError("Multi mode: dchi2_obs must contain at least one key (e.g. 'tex').")

    # We'll compute these primary chi2 for keys present in Q
    Qkeys = list(Q.keys())

    # Storage
    ge = {k: 0 for k in req}
    max_seen = {k: -1e300 for k in req}
    heaps = {k: [] for k in req}
    all_stats = {k: [] for k in req}

    # Observed values
    obs = {k: float(dchi2_obs[k]) for k in req}

    # If 'max' requested explicitly, use it; else define it consistently from requested stats
    if "max" in dchi2_obs:
        obs_max = float(dchi2_obs["max"])
    else:
        obs_max = max(obs[k] for k in req)

    ge_max = 0
    max_seen_max = -1e300
    heap_max = []
    stats_max = []

    t0 = time.time()
    step = max(1, nperm // 100)

    for i in range(nperm):
        y = yw[rng.permutation(len(yw))]

        # Baseline
        chi2_0 = chi2_from_Q(y, Q0)

        # Compute chi2 for each provided model in Q
        chi2_map = {}
        for k in Qkeys:
            chi2_map[k] = chi2_from_Q(y, Q[k])

        # Compute dchi2 map for primaries: Δχ²(model) = χ²0 - χ²model
        d = {}
        for k in Qkeys:
            d[k] = chi2_0 - chi2_map[k]

        # Optional nested gains if MIX exists with KIN/TEX
        # dk: TEX gain above KIN => χ²(KIN) - χ²(MIX)
        if ("kin" in chi2_map) and ("mix" in chi2_map):
            d["dk"] = chi2_map["kin"] - chi2_map["mix"]
        # kd: KIN gain above TEX => χ²(TEX) - χ²(MIX)
        if ("tex" in chi2_map) and ("mix" in chi2_map):
            d["kd"] = chi2_map["tex"] - chi2_map["mix"]

        # Make sure all requested stats exist
        for k in req:
            if k not in d:
                raise RuntimeError(f"Requested stat '{k}' not computable from Q. Available: {sorted(d.keys())}")

        # Update per-stat bookkeeping
        for k in req:
            v = d[k]
            all_stats[k].append(v)

            if v > max_seen[k]:
                max_seen[k] = v

            if topk > 0:
                h = heaps[k]
                if len(h) < topk:
                    heapq.heappush(h, (v, i))
                else:
                    if v > h[0][0]:
                        heapq.heapreplace(h, (v, i))

            if v >= obs[k]:
                ge[k] += 1

        # Define dmax consistently
        v_max = max(d[k] for k in req)
        stats_max.append(v_max)

        if v_max > max_seen_max:
            max_seen_max = v_max

        if topk > 0:
            if len(heap_max) < topk:
                heapq.heappush(heap_max, (v_max, i))
            else:
                if v_max > heap_max[0][0]:
                    heapq.heapreplace(heap_max, (v_max, i))

        if v_max >= obs_max:
            ge_max += 1

        if progress and ((i + 1) % step == 0 or (i + 1) == nperm):
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-9)
            sys.stdout.write(
                f"\r[permute(global, whitened, multi)] {100*(i+1)/nperm:6.2f}%  ({i+1}/{nperm})  "
                f"hits_max={ge_max}  max_stat={max_seen_max:8.3f}  elapsed={dt:7.1f}s  rate={rate:5.1f}/s"
            )
            sys.stdout.flush()

    if progress:
        sys.stdout.write("\n")

    # Convert hits to p-values
    p_emp = {k: (ge[k] + 1.0) / (nperm + 1.0) for k in req}
    gap = {k: obs[k] - max_seen[k] for k in req}
    top_sorted = {k: sorted(heaps[k], key=lambda x: x[0], reverse=True) for k in req}

    p_max = (ge_max + 1.0) / (nperm + 1.0)
    gap_max = obs_max - max_seen_max
    top_max = sorted(heap_max, key=lambda x: x[0], reverse=True)

    # Optional outputs: one CSV with all requested stats + MAX, plus plots.
    if plot_output:
        try:
            # Save ALL requested stats + MAX in a single file
            cols = req + ["max"]
            mat = np.column_stack([np.array(all_stats[k], dtype=float) for k in req] + [np.array(stats_max, dtype=float)])
            header = ",".join([f"dchi2_{k}" for k in cols])
            np.savetxt(plot_output + "_ALL.csv", mat, delimiter=",", header=header, fmt="%.6f")
            print(f"[INFO] Data saved to {plot_output + '_ALL.csv'}")
        except Exception as e:
            print(f"[WARN] Could not save multi CSV: {e}")

        # Plots: TEX/KIN if present + MAX always (A)
        if "tex" in all_stats:
            save_and_plot_null(all_stats["tex"], obs.get("tex", 0.0), plot_output + "_TEX")
        if "kin" in all_stats:
            save_and_plot_null(all_stats["kin"], obs.get("kin", 0.0), plot_output + "_KIN")
        save_and_plot_null(stats_max, obs_max, plot_output + "_MAX")

    # Return dicts (plus max packaged similarly)
    p_emp["max"] = p_max
    ge["max"] = ge_max
    max_seen["max"] = max_seen_max
    gap["max"] = gap_max
    top_sorted["max"] = top_max

    return p_emp, ge, max_seen, gap, top_sorted

def chi2_given_beta(L, X, y, beta):
    r = y - X @ beta
    return chi2_only(L, r)


# -----------------------------------------------------------------------------
# Robustness / diagnostics helpers (influence, directional jackknife, constant-N)
# -----------------------------------------------------------------------------

def angsep_lb_deg(l1, b1, l2, b2):
    """Angular separation between two galactic directions (degrees)."""
    l1 = np.deg2rad(l1); b1 = np.deg2rad(b1)
    l2 = np.deg2rad(l2); b2 = np.deg2rad(b2)
    s = (np.sin(b1) * np.sin(b2) +
         np.cos(b1) * np.cos(b2) * np.cos(l1 - l2))
    s = float(np.clip(s, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(s)))

def build_X(n_eq, z, mode):
    """
    Build the design matrix for the requested dipole model.
    mode ∈ {"mono","tex","kin","mix"}.
    """
    n_eq = np.asarray(n_eq, dtype=float)
    z = np.asarray(z, dtype=float)
    z_safe = np.maximum(z, 1e-6)

    cols = [np.ones(len(z_safe), dtype=float)]
    if mode in ("kin", "mix"):
        cols += [n_eq[:, 0] / z_safe, n_eq[:, 1] / z_safe, n_eq[:, 2] / z_safe]
    if mode in ("tex", "mix"):
        cols += [n_eq[:, 0], n_eq[:, 1], n_eq[:, 2]]
    return np.column_stack(cols)

def build_cov_for_idx(tab, idx0, args, cov_full=None):
    """
    Rebuild the effective covariance matrix (or diagonal surrogate) for idx0,
    matching the logic used in fit_dipole().

    Returns:
        C      : (N,N) ndarray (full cov if args.use_cov and cov_full is not None, else diagonal matrix)
        L      : Cholesky factor (N,N) or diagonal std vector (N,)
        is_cov : bool, True if full covariance was used
    """
    global Z_COL, MUERR_COL

    idx0 = np.asarray(idx0, dtype=int)

    # NOTE: tab is a structured array in this script; use safe column access.
    z = _col_as_float(tab, Z_COL, idx0)
    muerr = _col_as_float(tab, MUERR_COL, idx0)
    muerr = np.maximum(muerr, 1e-6)

    # base covariance
    use_cov = bool(getattr(args, "use_cov", False)) and (cov_full is not None)
    if use_cov:
        C = cov_full[np.ix_(idx0, idx0)].copy()
    else:
        C = np.diag(muerr * muerr)

    # intrinsic scatter
    if getattr(args, "sigint", 0.0) and getattr(args, "sigint", 0.0) > 0:
        C[np.diag_indices_from(C)] += float(args.sigint) ** 2

    # peculiar velocity term
    if getattr(args, "sigv", 0.0) and getattr(args, "sigv", 0.0) > 0:
        cz = C_LIGHT * z
        sigmu_v = (5.0 / math.log(10.0)) * (float(args.sigv) / np.maximum(cz, 1e-3))
        C[np.diag_indices_from(C)] += sigmu_v * sigmu_v

    # whitening
    if use_cov:
        L = whiten_from_cov(C)
    else:
        L = np.sqrt(np.maximum(np.diag(C), 1e-30))
    return C, L, use_cov

def whitened_ols_fit(L, X, y):
    """
    GLS via whitening + OLS.
    Works both when L is full lower-triangular Cholesky factor, and when L is
    a (N,) diagonal vector.
    Returns:
        beta, chi2, resid_w, Xw
    """
    Xw = solve_lower(L, X)
    yw = solve_lower(L, y)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid_w = yw - Xw @ beta
    chi2 = float(resid_w @ resid_w)
    return beta, chi2, resid_w, Xw

def influence_diagnostics(tab, idx0, args, cov_full, H0_ref, mode="mix",
                          top=20, dropone=0):
    """
    Influence diagnostics using whitening-based leverage + Cook's distance,
    plus a covariance-aware contribution score s_i.

    This routine is meant as a robustness *detector* (identify ultra-influential points),
    not as a strict per-observation probabilistic statement when C has off-diagonals.

    Parameters
    ----------
    mode : {"mono","tex","kin","mix"}
        Which design matrix to use for the whitened OLS leverage/Cook computation.
        The axis/Δχ² reported in the optional drop-one step are always computed with fit_dipole().
    """
    global Z_COL, RA_COL, DEC_COL, MU_COL, MUERR_COL

    idx0 = np.asarray(idx0, dtype=int)

    # --- columns (structured array safe access) ---
    z   = _col_as_float(tab, Z_COL,   idx0)
    ra  = _col_as_float(tab, RA_COL,  idx0)
    dec = _col_as_float(tab, DEC_COL, idx0)
    mu  = _col_as_float(tab, MU_COL,  idx0)

    # --- lock H0 to baseline isotropic best fit ---
    mu_th = hubble_mu_model(z, float(H0_ref), q0=args.q0)
    y = (mu - mu_th).astype(float)

    # --- design matrix (equatorial unit vectors) ---
    n_eq = unitvec_from_radec(ra, dec)
    X = build_X(n_eq, z, mode=mode)

    # --- covariance + whitening ---
    C, L, use_cov = build_cov_for_idx(tab, idx0, args, cov_full=cov_full)

    # --- whitened OLS fit ---
    beta, chi2, resid_w, Xw = whitened_ols_fit(L, X, y)

    n = int(len(y))
    p = int(X.shape[1])
    if n <= p + 1:
        print(f"Influence diagnostics: N too small (N={n}, p={p}).")
        return

    # --- leverage (diag of hat matrix in whitened space) ---
    XtX = Xw.T @ Xw
    XtX_inv = np.linalg.pinv(XtX, rcond=1e-15)
    h = np.sum((Xw @ XtX_inv) * Xw, axis=1)

    # --- Cook's distance (whitened) ---
    sigma2 = float(chi2) / float(max(1, n - p))
    denom = np.maximum(1.0 - h, 1e-12)
    cook = (resid_w * resid_w * h) / (float(p) * sigma2 * denom * denom + 1e-30)

    # --- covariance-aware contribution score s_i (sums to chi2) ---
    r = y - X @ beta
    if use_cov:
        w = np.linalg.solve(L.T, np.linalg.solve(L, r))
    else:
        w = r / np.maximum(np.diag(C), 1e-30)
    s = r * w

    # --- rank + display ---
    top = max(1, int(top))
    ridx = np.argsort(cook)[::-1][:top]

    print("")
    print(f"=== Influence diagnostics (mode={mode}, {'full-cov' if use_cov else 'diag'}; N={n}) ===")
    print("Top points by Cook's distance (whitened):")
    print("rank | i(idx0) | z | ra | dec | CookD | leverage(h) | s_i")
    for k, j in enumerate(ridx, start=1):
        i_glob = int(idx0[j])
        print(f"{k:>4} | {i_glob:>6} | {z[j]:.5f} | {ra[j]:.3f} | {dec[j]:.3f} | "
              f"{cook[j]:.4e} | {h[j]:.4e} | {s[j]:.4e}")

    # --- optional drop-one refit (exact pipeline via fit_dipole) ---
    if dropone and int(dropone) > 0:
        dropone = min(int(dropone), len(ridx))
        print("")
        print(f"Drop-one refits for top {dropone} CookD points (exact pipeline via fit_dipole):")

        base = fit_dipole(tab, idx0, args, cov_full=cov_full)
        base_tex = (base.get("l", float('nan')), base.get("b", float('nan')))
        base_kin = (base.get("l_kin", float('nan')), base.get("b_kin", float('nan')))

        for j in ridx[:dropone]:
            idx_sub = idx0[np.arange(len(idx0)) != j]
            rr = fit_dipole(tab, idx_sub, args, cov_full=cov_full)

            dtex = angsep_lb_deg(base_tex[0], base_tex[1], rr.get("l", float('nan')), rr.get("b", float('nan')))                    if (np.isfinite(base_tex[0]) and np.isfinite(rr.get("l", np.nan))) else float('nan')

            dkin = angsep_lb_deg(base_kin[0], base_kin[1], rr.get("l_kin", float('nan')), rr.get("b_kin", float('nan')))                    if (np.isfinite(base_kin[0]) and np.isfinite(rr.get("l_kin", np.nan))) else float('nan')

            print(f" - drop idx={int(idx0[j])}: Δχ²_tex={rr['dchi2_tex']-base['dchi2_tex']:+.3f} "
                  f"Δθ_tex={dtex:.2f}° | Δχ²_kin={rr['dchi2_kin']-base['dchi2_kin']:+.3f} "
                  f"Δθ_kin={dkin:.2f}°")

def fibonacci_sphere_dirs(n_dirs):
    """Approximately uniform directions on the sphere (unit vectors in EQ frame)."""
    n_dirs = int(n_dirs)
    if n_dirs <= 0:
        return np.zeros((0, 3), dtype=float)
    i = np.arange(n_dirs, dtype=float)
    phi = (1.0 + 5.0**0.5) / 2.0
    theta = 2.0 * np.pi * i / phi
    z = 1.0 - (2.0 * i + 1.0) / n_dirs
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])



def _col_as_float(tab, col, idx=None):
    """Return column `col` as float array.

    Supports:
      - numpy structured arrays (col is a field name str)
      - 2D numpy arrays (col is an int column index)
    """
    if isinstance(col, str):
        a = np.asarray(tab[col], dtype=float)
        return a if (idx is None) else a[idx]
    # fallback: assume 2D ndarray
    if idx is None:
        return np.asarray(tab[:, col], dtype=float)
    return np.asarray(tab[idx, col], dtype=float)

# Backward-compatible aliases used by robustness diagnostics
def unitvec_eq_from_radec(ra_deg, dec_deg):
    return unitvec_from_radec(ra_deg, dec_deg)

def gal_lb_from_eq_unitvec(u_eq):
    """(l,b) in degrees from an equatorial unit vector (or array of vectors)."""
    u_eq = np.asarray(u_eq, dtype=float)
    u_gal = u_eq @ EQ2GAL.T
    if u_gal.ndim == 1:
        x, y, z = float(u_gal[0]), float(u_gal[1]), float(u_gal[2])
        l = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
        b = math.degrees(math.asin(max(-1.0, min(1.0, z))))
        return float(l), float(b)
    x = u_gal[:, 0]
    y = u_gal[:, 1]
    z = np.clip(u_gal[:, 2], -1.0, 1.0)
    l = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    b = np.degrees(np.arcsin(z))
    return l, b

def directional_jackknife(tab, idx0, args, cov_full, n_dirs=48,
                          mode="cap", theta_deg=30.0, compare="tex",
                          dump_csv=None):
    """
    Directional jackknife (hemisphere or cap removal), tracking axis stability and Δχ².

    compare ∈ {"tex","kin","mix_tex","mix_kin"}.
    Notes:
      - Works with structured arrays (named columns).
      - The cut directions are generated in equatorial coordinates; we report them in Galactic (l,b).
    """
    global Z_COL, RA_COL, DEC_COL

    idx0 = np.asarray(idx0, dtype=int)

    z   = _col_as_float(tab, Z_COL,   idx0)
    ra  = _col_as_float(tab, RA_COL,  idx0)
    dec = _col_as_float(tab, DEC_COL, idx0)

    n_eq = unitvec_from_radec(ra, dec)

    base = fit_dipole(tab, idx0, args, cov_full=cov_full)

    def get_axis_and_dchi2(r):
        if compare == "tex":
            return (r.get("l", np.nan), r.get("b", np.nan)), r.get("dchi2_tex", np.nan)
        if compare == "kin":
            return (r.get("l_kin", np.nan), r.get("b_kin", np.nan)), r.get("dchi2_kin", np.nan)
        if compare == "mix_tex":
            return (r.get("l_mix_tex", np.nan), r.get("b_mix_tex", np.nan)), r.get("dchi2_mix", np.nan)
        if compare == "mix_kin":
            return (r.get("l_mix_kin", np.nan), r.get("b_mix_kin", np.nan)), r.get("dchi2_mix", np.nan)
        raise ValueError("unknown compare mode")

    base_axis, base_dchi2 = get_axis_and_dchi2(base)

    u_dirs = fibonacci_sphere_dirs(int(n_dirs))
    cos_cut = math.cos(math.radians(float(theta_deg)))

    rows = []
    for u in u_dirs:
        u = np.asarray(u, dtype=float)
        dots = n_eq @ u

        if mode == "hemisphere":
            keep = (dots <= 0.0)
        else:  # cap
            keep = (dots <= cos_cut)  # remove within theta (dot > cos(theta))

        idx_sub = idx0[keep]

        # skip pathological cases
        if len(idx_sub) < 20:
            continue

        rj = fit_dipole(tab, idx_sub, args, cov_full=cov_full)
        axis, dchi2 = get_axis_and_dchi2(rj)

        if np.isfinite(base_axis[0]) and np.isfinite(axis[0]):
            dtheta = float(angsep_lb_deg(base_axis[0], base_axis[1], axis[0], axis[1]))
        else:
            dtheta = float("nan")

        l_u, b_u = gal_lb_from_eq_unitvec(u)
        rows.append((l_u, b_u, len(idx0) - len(idx_sub), len(idx_sub), dtheta, dchi2))

    rows = np.array(rows, dtype=float)
    if len(rows) == 0:
        print("Directional jackknife: no valid realizations (too aggressive cuts?).")
        return

    dtheta = rows[:, 4]
    dchi2 = rows[:, 5]

    print("")
    print(f"=== Directional jackknife ({mode}, n_dirs={n_dirs}, theta={theta_deg}°, compare={compare}) ===")
    print(f"N baseline: {len(idx0)} | baseline axis (l,b)=({base_axis[0]:.1f},{base_axis[1]:.1f}) "
          f"| baseline Δχ²={base_dchi2:.2f}")
    print(f"Δθ axis: median={np.nanmedian(dtheta):.2f}°, 90%={np.nanpercentile(dtheta,90):.2f}°, "
          f"max={np.nanmax(dtheta):.2f}°")

    # show worst 10
    w = np.argsort(np.nan_to_num(dtheta, nan=-1.0))[::-1][:10]
    print("Worst directions (largest axis rotation):")
    print("rank | (l,b)_cut | N_removed | N_keep | Δθ_axis | Δχ²")
    for k, j in enumerate(w, start=1):
        print(f"{k:>4} | ({rows[j,0]:6.1f},{rows[j,1]:6.1f}) | {int(rows[j,2]):>9} | "
              f"{int(rows[j,3]):>6} | {rows[j,4]:7.2f}° | {rows[j,5]:7.2f}")

    if dump_csv:
        import csv
        with open(dump_csv, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["l_cut_deg", "b_cut_deg", "n_removed", "n_keep", "dtheta_deg", "dchi2"])
            for r in rows:
                wcsv.writerow([f"{r[0]:.6f}", f"{r[1]:.6f}", int(r[2]), int(r[3]), f"{r[4]:.6f}", f"{r[5]:.6f}"])
        print(f"(saved) {dump_csv}")

def constant_N_test(tab, args, cov_full, zmins, draws=200, N0=0, model="tex",
                    use_cov=False, seed=0, dump_csv=None):
    """
    Constant-N test across zmin values.

    For each zmin, draw N0 objects from the pool (z > zmin) and compute, on the SAME
    subsample:
        Δχ² = χ²_mono - χ²_model
    where "mono" is a0-only (intercept) and "model" ∈ {"tex","kin","mix"}.

    H0 is LOCKED to a single baseline value (best isotropic H0 on the baseline pool),
    so Δχ² measures only the dipole power at fixed H0.

    Important:
      - This function is self-contained (does NOT call fit_dipole inside the draws),
        so it stays consistent with the locked H0 and with use_cov/diag choice.
      - It also supports --fix_tex_axis / --fix_kin_axis (1-dof dipoles) consistently.
    """
    global Z_COL, RA_COL, DEC_COL, MU_COL, MUERR_COL

    # --- Read columns from structured array ---
    z     = np.array(tab[Z_COL], float)
    ra    = np.array(tab[RA_COL], float)
    dec   = np.array(tab[DEC_COL], float)
    mu    = np.array(tab[MU_COL], float)
    muerr = np.array(tab[MUERR_COL], float)

    # --- Unit vectors (equatorial) + Galactic latitude for masks ---
    n_eq_all  = unitvec_from_radec(ra, dec)              # (N,3) equatorial
    n_gal_all = n_eq_all @ EQ2GAL.T                      # (N,3) galactic
    b_gal = np.degrees(np.arcsin(np.clip(n_gal_all[:, 2], -1.0, 1.0)))

    # --- Dust (optional) ---
    ebv = np.zeros_like(z)
    if args.dustcut is not None and args.dustcut > 0:
        dcol = args.dustcol if args.dustcol else find_col(list(tab.dtype.names), ["MWEBV","mwebv","E_BV","ebv"])
        if dcol and (dcol in tab.dtype.names):
            ebv = np.array(tab[dcol], float)

    # --- Base mask (consistent with main) ---
    good = (np.isfinite(z) & np.isfinite(ra) & np.isfinite(dec) &
            np.isfinite(mu) & np.isfinite(muerr))
    good &= (z >= 0.0) & (z <= args.zmax)

    if args.bcut is not None and args.bcut > 0:
        good &= (np.abs(b_gal) >= args.bcut)

    if args.dustcut is not None and args.dustcut > 0:
        good &= (ebv <= args.dustcut)

    # --- Baseline selection: use the lowest zmin in the scan ---
    zmin_ref = float(np.min(np.array(zmins, dtype=float)))
    idx_base = np.where(good & (z > zmin_ref))[0]
    if len(idx_base) < 30:
        print("constN: baseline selection too small.")
        return

    # --- Build whitening for baseline (diag or full cov) ---
    z_base = z[idx_base]
    mu_base = mu[idx_base]
    muerr_base = np.maximum(muerr[idx_base], 1e-6)

    if use_cov and (cov_full is not None):
        Cb = cov_full[np.ix_(idx_base, idx_base)].copy()
        if args.sigint and args.sigint > 0:
            Cb[np.diag_indices_from(Cb)] += args.sigint**2
        if getattr(args, "sigv", 0.0) and getattr(args, "sigv", 0.0) > 0:
            cz = C_LIGHT * np.maximum(z_base, 1e-6)
            sigmu_v = (5.0 / math.log(10.0)) * (args.sigv / np.maximum(cz, 1e-3))
            Cb[np.diag_indices_from(Cb)] += sigmu_v**2
        Lb = whiten_from_cov(Cb)
    else:
        var = muerr_base**2
        if args.sigint and args.sigint > 0:
            var = var + args.sigint**2
        if getattr(args, "sigv", 0.0) and getattr(args, "sigv", 0.0) > 0:
            cz = C_LIGHT * np.maximum(z_base, 1e-6)
            sigmu_v = (5.0 / math.log(10.0)) * (args.sigv / np.maximum(cz, 1e-3))
            var = var + sigmu_v**2
        Lb = np.sqrt(np.maximum(var, 1e-30))

    # --- Baseline H0: isotropic grid search (same as fit_dipole's first step) ---
    H0_grid = np.linspace(args.h0min, args.h0max, args.h0n)
    best = (1e300, None)
    for H0 in H0_grid:
        mu_th = hubble_mu_model(z_base, H0, q0=args.q0)
        r = (mu_base - mu_th).astype(float)
        chi2 = chi2_only(Lb, r)
        if chi2 < best[0]:
            best = (chi2, float(H0))
    H0_ref = float(best[1])

    # --- Precompute residuals with locked H0 ---
    mu_th_all = hubble_mu_model(z, H0_ref, q0=args.q0)
    y_all = (mu - mu_th_all).astype(float)

    # --- Diag whitening (diag-only case) ---
    if not (use_cov and (cov_full is not None)):
        var_all = np.maximum(muerr, 1e-6)**2
        if args.sigint and args.sigint > 0:
            var_all = var_all + args.sigint**2
        if getattr(args, "sigv", 0.0) and getattr(args, "sigv", 0.0) > 0:
            cz = C_LIGHT * np.maximum(z, 1e-6)
            sigmu_v = (5.0 / math.log(10.0)) * (args.sigv / np.maximum(cz, 1e-3))
            var_all = var_all + sigmu_v**2
        Ldiag = np.sqrt(np.maximum(var_all, 1e-30))

    # --- Fixed axes support (same convention as fit_dipole) ---
    tex_fix = _parse_lb(getattr(args, "fix_tex_axis", ""))  # (l,b) or None
    kin_fix = _parse_lb(getattr(args, "fix_kin_axis", ""))  # (l,b) or None

    proj_tex_all = None
    proj_kin_all = None
    if tex_fix is not None:
        l0, b0 = tex_fix
        dgal = _uvec_from_lb(l0, b0)                 # (3,) in gal frame
        deq_tex = (EQ2GAL.T @ dgal).astype(float)    # back to equatorial
        proj_tex_all = (n_eq_all @ deq_tex).astype(float)
    if kin_fix is not None:
        l0, b0 = kin_fix
        dgal = _uvec_from_lb(l0, b0)
        deq_kin = (EQ2GAL.T @ dgal).astype(float)
        proj_kin_all = (n_eq_all @ deq_kin).astype(float)

    zinv_all = 1.0 / np.maximum(z, 1e-6)

    def _antipode_lb(lb):
        l0, b0 = lb
        return ((float(l0) + 180.0) % 360.0, -float(b0))

    def _vec_to_lb_equatorial(v):
        v = np.asarray(v, dtype=float)
        nrm = float(np.linalg.norm(v))
        if not np.isfinite(nrm) or nrm <= 0.0:
            return (float("nan"), float("nan"))
        ueq = v / nrm
        ugal = (EQ2GAL @ ueq).astype(float)
        l = (math.degrees(math.atan2(float(ugal[1]), float(ugal[0]))) + 360.0) % 360.0
        b = math.degrees(math.asin(max(-1.0, min(1.0, float(ugal[2])))))
        return (float(l), float(b))

    def _build_X_idx(idx_sub, mode_):
        nsub = n_eq_all[idx_sub]
        zinv = zinv_all[idx_sub]
        ones = np.ones(len(idx_sub), dtype=float)

        if mode_ == "mono":
            return ones[:, None]

        if mode_ == "tex":
            if tex_fix is None:
                return np.column_stack([ones, nsub[:, 0], nsub[:, 1], nsub[:, 2]])
            return np.column_stack([ones, proj_tex_all[idx_sub]])

        if mode_ == "kin":
            if kin_fix is None:
                return np.column_stack([ones, nsub[:, 0]*zinv, nsub[:, 1]*zinv, nsub[:, 2]*zinv])
            return np.column_stack([ones, proj_kin_all[idx_sub]*zinv])

        if mode_ == "mix":
            cols = [ones]
            # kinematic block first (same ordering as build_X)
            if kin_fix is None:
                cols += [nsub[:, 0]*zinv, nsub[:, 1]*zinv, nsub[:, 2]*zinv]
            else:
                cols += [proj_kin_all[idx_sub]*zinv]
            # tex block
            if tex_fix is None:
                cols += [nsub[:, 0], nsub[:, 1], nsub[:, 2]]
            else:
                cols += [proj_tex_all[idx_sub]]
            return np.column_stack(cols)

        raise ValueError(f"constant_N_test: unsupported mode={mode_}")

    def _axis_from_beta(beta, mode_):
        """
        Returns (l,b) in galactic deg, following fit_dipole conventions:
          - if axis is fixed, flip to enforce positive amplitude
          - if axis is free, use vector direction (max direction)
        """
        beta = np.asarray(beta, dtype=float)

        if mode_ == "tex":
            if tex_fix is not None:
                amp = float(beta[1])
                return tex_fix if (amp >= 0.0) else _antipode_lb(tex_fix)
            return _vec_to_lb_equatorial(beta[1:4])

        if mode_ == "kin":
            if kin_fix is not None:
                amp = float(beta[1])
                return kin_fix if (amp >= 0.0) else _antipode_lb(kin_fix)
            return _vec_to_lb_equatorial(beta[1:4])

        if mode_ == "mix":
            # We report TEX-axis by default (as the previous code did).
            # Figure out where the TEX block starts.
            kcols = 1 if (kin_fix is not None) else 3
            tex_start = 1 + kcols
            if tex_fix is not None:
                amp = float(beta[tex_start])
                return tex_fix if (amp >= 0.0) else _antipode_lb(tex_fix)
            return _vec_to_lb_equatorial(beta[tex_start:tex_start+3])

        return (float("nan"), float("nan"))

    # --- Base axis reference (single fit on the baseline pool) ---
    Xb = _build_X_idx(idx_base, model)
    yb = y_all[idx_base]

    # Use the same whitening choice as the draws:
    if use_cov and (cov_full is not None):
        # reuse baseline Cb/Lb already built above
        L_for_axis = Lb
    else:
        L_for_axis = Ldiag[idx_base]

    beta_b, _ = gls_fit_and_chi2(L_for_axis, Xb, yb)
    base_axis = _axis_from_beta(beta_b, model)

    # --- Pools per zmin ---
    idx_sets = [np.where(good & (z > float(zmin)))[0] for zmin in zmins]
    sizes = [len(x) for x in idx_sets]
    N0_eff = int(N0) if (N0 and N0 > 0) else int(min(sizes)) if sizes else 0

    if N0_eff < 30:
        print(f"constN: common N0 too small ({N0_eff}).")
        return

    rng = np.random.default_rng(seed)

    print("")
    print(f"=== Constant-N test (model={model}, draws={int(draws)}, N0={N0_eff}, "
          f"{'full-cov' if (use_cov and cov_full is not None) else 'diag'}; H0 locked={H0_ref:.2f}) ===")
    print("zmin    | N(zmin) | median Δχ² | 90% Δχ² | median Δθ_axis | 90% Δθ_axis")

    rows_out = []

    for zmin, idx_pool in zip(zmins, idx_sets):
        zmin = float(zmin)

        if len(idx_pool) < N0_eff:
            rows_out.append((zmin, int(len(idx_pool)), np.nan, np.nan, np.nan, np.nan))
            print(f"{zmin:0.5f} | {len(idx_pool):>6} | (skip: N < N0)")
            continue

        dchi2_list = []
        dtheta_list = []

        for _ in range(int(draws)):
            idx_sub = rng.choice(idx_pool, size=N0_eff, replace=False)

            y_sub = y_all[idx_sub]

            # Build X for mono + model on the SAME subsample
            X_mono  = _build_X_idx(idx_sub, "mono")
            X_model = _build_X_idx(idx_sub, model)

            # Whitening for this subsample
            if use_cov and (cov_full is not None):
                Csub = cov_full[np.ix_(idx_sub, idx_sub)].copy()
                if args.sigint and args.sigint > 0:
                    Csub[np.diag_indices_from(Csub)] += args.sigint**2
                if getattr(args, "sigv", 0.0) and getattr(args, "sigv", 0.0) > 0:
                    z_sub = z[idx_sub]
                    cz = C_LIGHT * np.maximum(z_sub, 1e-6)
                    sigmu_v = (5.0 / math.log(10.0)) * (args.sigv / np.maximum(cz, 1e-3))
                    Csub[np.diag_indices_from(Csub)] += sigmu_v**2
                Lsub = whiten_from_cov(Csub)
            else:
                Lsub = Ldiag[idx_sub]

            _, chi2_mono  = gls_fit_and_chi2(Lsub, X_mono,  y_sub)
            beta_md, chi2_model = gls_fit_and_chi2(Lsub, X_model, y_sub)

            dchi2_list.append(float(chi2_mono - chi2_model))

            # Axis stability (cheap: beta already computed)
            if np.isfinite(base_axis[0]):
                ax = _axis_from_beta(beta_md, model)
                if np.isfinite(ax[0]):
                    dtheta_list.append(float(angsep_lb_deg(base_axis[0], base_axis[1], ax[0], ax[1])))

        dchi2_arr = np.array(dchi2_list, dtype=float)
        dtheta_arr = np.array(dtheta_list, dtype=float)

        med  = float(np.median(dchi2_arr)) if dchi2_arr.size else np.nan
        p90  = float(np.percentile(dchi2_arr, 90)) if dchi2_arr.size else np.nan
        medA = float(np.median(dtheta_arr)) if dtheta_arr.size else np.nan
        p90A = float(np.percentile(dtheta_arr, 90)) if dtheta_arr.size else np.nan

        print(f"{zmin:0.5f} | {len(idx_pool):>6} | {med:>10.3f} | {p90:>8.3f} | "
              f"{medA:>13.2f}° | {p90A:>10.2f}°")

        rows_out.append((zmin, int(len(idx_pool)), med, p90, medA, p90A))

    if dump_csv and rows_out:
        import csv
        try:
            with open(dump_csv, "w", newline="", encoding="utf-8") as f:
                wcsv = csv.writer(f)
                wcsv.writerow(["zmin", "N_pool", "median_dchi2", "p90_dchi2", "median_dtheta_deg", "p90_dtheta_deg"])
                for r in rows_out:
                    wcsv.writerow(r)
            print(f"(saved) {dump_csv}")
        except IOError as e:
            print(f"Error writing CSV: {e}")

def run_null_mock_tomography(tab, base_mask, args, cov_full, zbin_results):
    """
    Null-footprint tomographic calibration (morphological / amplitude-based; bins 2&3 only).

    We run the locked-axis tomography on correlated Gaussian noise realizations drawn on the
    *same* footprint and with the *same* covariance model as the data, and we count how often
    noise reproduces a simple 'shell-like' morphology using only bins #2 and #3:

      - Peak (bin #2): signed A_tex is positive and >= observed A_tex in bin #2
      - Extinction (bin #3): |A_tex| <= thresh_outer (fixed threshold, not data-tuned)

    Notes:
      - Uses only args.seed (no extra seed arg).
      - Uses --nmock as the number of realizations; if --nmock is absent or <=1, defaults to 1000.
      - Fits TEX-only per bin (X = [1, proj_tex]) to match the reported A_tex_fit logic.
    """
    # --- Default zbins if not provided ---
    zbins_spec = getattr(args, "zbins", None)
    if (zbins_spec is None) or (str(zbins_spec).strip() == ""):
        zbins_spec = "0.020-0.025,0.025-0.030,0.030-0.035,0.035-0.040"
    zb = parse_bins(zbins_spec)
    if not zb:
        raise RuntimeError(f"Invalid zbins spec: {zbins_spec}")

    if len(zb) < 2:
        raise RuntimeError("--null_mock_tomo needs at least two zbins.")
    if len(zb) < 3:
        # If user provided only two bins, treat them as (peak, outer).
        i_peak, i_outer = 0, 1
    else:
        # Your intended definition: bins 2 and 3 of the provided list.
        i_peak, i_outer = 1, 2

    # --- Seed handling (use only args.seed) ---
    seed = getattr(args, "seed", 0)
    if seed is None:
        seed = 0
    seed = int(seed)
    rng = np.random.default_rng(seed + 555)

    # --- Number of mocks: reuse existing --nmock (no new CLI arg) ---
    n_mocks = getattr(args, "nmock", None)
    if n_mocks is None or int(n_mocks) <= 1:
        n_mocks = 1000
    n_mocks = int(n_mocks)

    # --- Sanity checks (locked-axis tomography path) ---
    if not getattr(args, "zbins_global", False):
        raise RuntimeError("--null_mock_tomo requires --zbins_global (locked axes + global H0 path).")
    if not (getattr(args, "fix_tex_axis", None) and getattr(args, "fix_kin_axis", None)):
        raise RuntimeError("--null_mock_tomo requires --fix_tex_axis and --fix_kin_axis.")

    # --- Build union selection (same union as zbins_global) ---
    z_all = np.array(tab[Z_COL], float)
    zlo_u = min(a for a, b in zb)
    zhi_u = max(b for a, b in zb)
    mU = base_mask & (z_all > zlo_u) & (z_all <= zhi_u)
    idxU = np.where(mU)[0]
    if idxU.size < 10:
        raise RuntimeError("Not enough SNe in union of bins for null mock tomography.")

    # --- Pull union arrays (geometry) ---
    raU    = np.array(tab[RA_COL], float)[idxU]
    decU   = np.array(tab[DEC_COL], float)[idxU]
    muerrU = np.maximum(np.array(tab[MUERR_COL], float)[idxU], 1e-6)

    # --- Sampling model: match covariance logic ---
    use_cov = (cov_full is not None) and bool(getattr(args, "use_cov", False))
    if use_cov:
        CU = cov_full[np.ix_(idxU, idxU)].copy()
        if args.sigint > 0:
            CU[np.diag_indices_from(CU)] += args.sigint**2
        LU = whiten_from_cov(CU)  # Cholesky factor L, so C = L L^T
    else:
        sig2U = muerrU * muerrU + args.sigint * args.sigint
        LU = np.sqrt(sig2U)  # diag std vector

    # --- Fixed TEX axis in equatorial frame ---
    l_tex, b_tex = _parse_lb(args.fix_tex_axis)
    dgal_tex = _uvec_from_lb(l_tex, b_tex)
    deq_tex  = (EQ2GAL.T @ dgal_tex).astype(float)

    nU = unitvec_from_radec(raU, decU)
    proj_tex_U = (nU @ deq_tex).astype(float)
    zU = np.array(tab[Z_COL], float)[idxU]

    # --- Precompute the two target bins (peak=bin2, outer=bin3) ---
    target_bins = [(i_peak, zb[i_peak]), (i_outer, zb[i_outer])]
    bin_pre = {}
    for (ibin, (zlo, zhi)) in target_bins:
        mB = (zU > zlo) & (zU <= zhi)
        loc = np.where(mB)[0]
        if loc.size < 10:
            raise RuntimeError(f"Null mock: not enough SNe in target bin #{ibin+1} ({zlo}-{zhi}).")

        if use_cov:
            idxB = idxU[loc]
            CB = cov_full[np.ix_(idxB, idxB)].copy()
            if args.sigint > 0:
                CB[np.diag_indices_from(CB)] += args.sigint**2
            LB = whiten_from_cov(CB)
        else:
            LB = LU[loc]

        # TEX-only model to match A_tex_fit: X = [1, proj_tex]
        Xtex = np.column_stack([np.ones(loc.size), proj_tex_U[loc]])
        bin_pre[ibin] = (loc, LB, Xtex)

    # --- Observed amplitudes (prefer keys that exist in your pipeline) ---
    def _get_obs_A(d):
        if "A_tex" in d:
            return float(d["A_tex"])
        if "A_tex_fit" in d:
            return float(d["A_tex_fit"])
        raise RuntimeError("Null mock: zbin_results missing A_tex (or A_tex_fit).")

    obs_map_A = {(float(d["zlo"]), float(d["zhi"])): _get_obs_A(d) for d in zbin_results}

    zlo2, zhi2 = zb[i_peak]
    zlo3, zhi3 = zb[i_outer]
    if (zlo2, zhi2) not in obs_map_A or (zlo3, zhi3) not in obs_map_A:
        raise RuntimeError("Null mock: missing observed A_tex for bin2/bin3 in zbin_results.")

    A2_obs = float(obs_map_A[(zlo2, zhi2)])  # bin 2 observed amplitude (positive in your table)
    # A3_obs not used for thresholding (we use a fixed extinction threshold)

    # --- Fixed, non-tuned thresholds for morphology ---
    thresh_peak  = 0.10   # generic "meaningful" peak (not tuned to the exact observed value)
    thresh_outer = 0.01   # extinction threshold (outer bin consistent with ~0)

    # --- Mock loop ---
    A2_m = np.zeros(n_mocks, float)
    A3_m = np.zeros(n_mocks, float)

    for i in range(n_mocks):
        g = rng.standard_normal(idxU.size)
        dmuU_mock = (LU @ g) if use_cov else (LU * g)

        # Bin 2
        loc2, LB2, X2 = bin_pre[i_peak]
        beta2, _ = gls_fit_and_chi2(LB2, X2, dmuU_mock[loc2])
        A2 = float(beta2[1])  # signed

        # Bin 3
        loc3, LB3, X3 = bin_pre[i_outer]
        beta3, _ = gls_fit_and_chi2(LB3, X3, dmuU_mock[loc3])
        A3 = float(beta3[1])  # signed

        A2_m[i] = A2
        A3_m[i] = A3

    # --- Empirical rates ---
    # How often noise produces a bin-2 peak >= observed (signed, so negative peaks don't count)
    p_peak = float(np.mean(A2_m >= A2_obs))

    shell_mask = (
        (A2_m >= max(A2_obs, thresh_peak)) &
        (A2_m > 0.0) &
        (np.abs(A3_m) <= thresh_outer)
    )
    n_shell = int(np.sum(shell_mask))
    p_shell = float(n_shell / float(n_mocks))

    print("\n=== Null Mock Tomography (Gaussian noise on real footprint; bins 2&3 only) ===")
    print(f"N_mocks={n_mocks} seed={seed} use_cov={use_cov}")
    print(f"zbins={zbins_spec}")
    print(f"Using bins: peak=bin#{i_peak+1} ({zlo2:.3f}-{zhi2:.3f}), outer=bin#{i_outer+1} ({zlo3:.3f}-{zhi3:.3f})")
    print(f"Observed: A2_obs={A2_obs:.6f}")
    print(f"Criteria: A2>=max(A2_obs,{thresh_peak}) & A2>0 & |A3|<={thresh_outer}")
    print(f"Counts: peak>=obs: {int(np.sum(A2_m >= A2_obs))}/{n_mocks}   shell: {n_shell}/{n_mocks}")
    print(f"Empirical: p(peak>=obs)={p_peak:.3e}  p(shell)={p_shell:.3e}")
    if n_shell == 0:
        print(f"Note: with 0 hits, a simple upper bound is p_shell < {1.0/n_mocks:.3e} (at 1/N resolution).")


# ---------- main ----------
def main():
    global RA_COL, DEC_COL, MU_COL, MUERR_COL, Z_COL
    
    ap = argparse.ArgumentParser()

    ap.add_argument("--make_pillar_plots", action="store_true", help="Generate 4-panel TEX/KIN cosmology figure for chapter 6.")
    ap.add_argument("--make_h0_residuals", action="store_true", help="Generate standalone sky map of Hubble residuals")
    
    ap.add_argument("--fix_tex_axis", default="", help="Fix tex dipole axis in galactic deg: 'l,b' (e.g. '135.54,5.99'). If set, no axis search; fit amplitude only.")
    ap.add_argument("--fix_kin_axis", default="", help="Fix kinematic (1/z) dipole axis in galactic deg: 'l,b'. If set, no axis search; fit v_bulk only.")
    ap.add_argument("--zbins_global", action="store_true", help="With --zbins and fixed axes: fit global parameters (mono / tex / kin / mix) on all selected SNe, then report per-bin χ² contributions and Δχ² splits.")

    ap.add_argument("--zbins", default="", help="Comma list of zmin-zmax bins (e.g. '0-0.03,0.03-0.06,0.06-0.10,0.10-0.15'); runs bins table then exits.")

    ap.add_argument("--mwcheck", action="store_true", help="Enable Milky Way template checks (axisymmetric).")
    ap.add_argument("--mwtemp", default="p2", help="MW template: sinb | abs_sinb | inv_abs_sinb | p2  (default p2).")
    ap.add_argument("--mw_sinb_min", type=float, default=0.05, help="Floor for |sin b| in inv_abs_sinb template (default 0.05).")

    ap.add_argument("--export_sky", default="", help="If non-empty, write CSV sky maps with given prefix (pred + resid).")
    ap.add_argument("--sky_step", type=float, default=10.0, help="Sky grid step in degrees for --export_sky (default 10).")
    ap.add_argument("--zstar", type=float, default=0.05, help="Reference z* used for predicted kinematic map (K·n/z*).")

    ap.add_argument("--null_gaussian", action="store_true", help="DEBUG: replace whitened residual vector yw by N(0,1) noise.")
    ap.add_argument("--nmock", type=int, default=1, help="Number of Gaussian-null mocks (only used with --null_gaussian).")

    ap.add_argument("--permute_within_survey", action="store_true", help="Permutation test, shuffling dmu within each IDSURVEY group (stratified). Requires IDSURVEY or --surveycol.")
    ap.add_argument("--permute_whitened", action="store_true", help="Permute whitened residuals yw=L^{-1} dmu (recommended with full covariance).")
    ap.add_argument("--permute_plot", default="", help="Prefix output for null distribution plot and CSV data (e.g. 'figs/null_dist').")

    ap.add_argument("--dustcheck", action="store_true", help="Enable Milky Way dust checks (corr + nuisance regressor).")
    ap.add_argument("--dustcol", default="", help="Dust column name (e.g. MWEBV). If empty, auto-detect.")
    ap.add_argument("--dustcut", type=float, default=-1.0, help="If >0, keep only SN with dust <= dustcut.")
    ap.add_argument("--bcut", type=float, default=-1.0, help="If >0, keep only SN with |galactic b| >= bcut (deg).")

    ap.add_argument("--dat", required=True)
    ap.add_argument("--cov", default="", help="Optional full covariance .cov (Pantheon style).")
    ap.add_argument("--use_cov", action="store_true", help="If set and --cov provided, use GLS with full cov.")
    ap.add_argument("--sigint", type=float, default=0.0, help="Add intrinsic scatter in mag (added to diag).")

    ap.add_argument("--sigv", type=float, default=0.0, help="Extra peculiar-velocity dispersion in km/s; converted to distance-modulus error sigma_mu≈(5/ln10)*(sigv/(c*z)) and added to the diagonal (and to full covariance diagonal if --use_cov).")

    ap.add_argument("--zmax", type=float, default=0.15)
    ap.add_argument("--zmin", type=float, default=0.0, help="Lower redshift cut (inclusive).")

    ap.add_argument("--scan", default="", help="Comma list of zmax values; runs scan table then exits.")
    ap.add_argument("--permute", type=int, default=0, help="N permutations (shuffle residuals on fixed positions) for empirical pval.")

    ap.add_argument("--zcol", default="")
    ap.add_argument("--q0", type=float, default=-0.55)

    ap.add_argument("--ra", default="")
    ap.add_argument("--dec", default="")
    ap.add_argument("--mu", default="")
    ap.add_argument("--muerr", default="")
    ap.add_argument("--surveycol", default="", help="Force survey id column name (e.g. IDSURVEY).")

    ap.add_argument("--h0min", type=float, default=60.0)
    ap.add_argument("--h0max", type=float, default=80.0)
    ap.add_argument("--h0n", type=int, default=2001)

    ap.add_argument("--seed", type=int, default=0)

    # Robustness / diagnostics
    
    ap.add_argument("--influence", action="store_true", help="Run influence diagnostics (whitened leverage + Cook's distance, plus drop-one refits).")
    ap.add_argument("--influence_mode", default="mix", choices=["mono","tex","kin","mix"], help="Model used for influence diagnostics. Default: mix.")
    ap.add_argument("--influence_top", type=int, default=20, help="Number of top influential points to print. Default: 20.")
    ap.add_argument("--influence_dropone", type=int, default=0, help="If >0, perform drop-one refits for the top N CookD points. Default: 0.")
    
    ap.add_argument("--jackknife", action="store_true", help="Leave-one-survey-out using IDSURVEY (or --surveycol).")
    ap.add_argument("--jackknife_dir", type=int, default=0, help="Directional jackknife: number of directions (e.g. 48 or 96). 0 disables.")
    ap.add_argument("--jackknife_dir_mode", choices=["cap","hemisphere"], default="cap",  help="Directional jackknife removal type. Default: cap.")
    ap.add_argument("--jackknife_dir_theta", type=float, default=30.0, help="Cap radius in degrees (only for mode=cap). Default: 30.")
    ap.add_argument("--jackknife_dir_compare", choices=["tex","kin","mix_tex","mix_kin"], default="tex", help="Which axis/Δχ² to track in directional jackknife. Default: tex.")
    ap.add_argument("--jackknife_dir_csv", default="",  help="Optional CSV output for directional jackknife results.")
    
    ap.add_argument("--constN", default="", help="Constant-N test: comma-separated zmin list (e.g. '0,0.003,0.005,0.01'). Empty disables.")
    ap.add_argument("--constN_draws", type=int, default=200, help="Constant-N Monte Carlo draws per zmin (diag-only). Default: 200.")
    ap.add_argument("--constN_N0", type=int, default=0, help="Constant-N common size. 0 uses min size across zmin list.")
    ap.add_argument("--constN_model", choices=["tex","kin","mix"], default="tex", help="Model used in constant-N test. Default: tex.")
    ap.add_argument("--constN_use_cov", action="store_true", help="Use full covariance in constant-N test (slower).")
    ap.add_argument("--constN_csv", default="", help="Optional CSV output for constant-N summary table.")
    
    ap.add_argument("--null_mock_tomo", action="store_true", help="Run null mock tomography test (noise injection) to check for spurious shell signals.")
    # ap.add_argument("--null_mock_tomo_n", type=int, default=1000, help="Number of Gaussian realizations for --null_mock_tomo (default: 1000).")
    ap.add_argument("--scan_zmin_h0", default="", help="Comma-separated zmin values to scan for effective H0 impact (e.g. '0,0.01,0.02,0.03,0.04').")

    args = ap.parse_args()

    tab = load_table(args.dat)
    names = list(tab.dtype.names)

    # Base columns for selection - Priority to zHD as fit_dipole
    Z_COL     = args.zcol   if args.zcol   else find_col(names, ["zHD","zhd","zCMB","zcmb","ZCMB","z"])
    MU_COL    = args.mu     if args.mu     else find_col(names, ["MU_SH0ES","MU","mu","mub","muobs","mures"])
    MUERR_COL = args.muerr  if args.muerr  else find_col(names, ["MU_SH0ES_ERR_DIAG","MUERR","muerr","dMU","MU_ERR","MUERR_FINAL"])
    RA_COL    = args.ra     if args.ra     else find_col(names, ["RA","ra","RAdeg","ra_deg"])
    DEC_COL   = args.dec    if args.dec    else find_col(names, ["DEC","Dec","dec","DEdeg","dec_deg"])

    if (Z_COL is None or MU_COL is None or MUERR_COL is None or
        RA_COL is None or DEC_COL is None):
        raise RuntimeError(
            "Could not auto-detect required columns (z, mu, muerr, ra, dec). "
            "Use --zcol/--mu/--muerr/--ra/--dec."
        )

    z_all     = np.array(tab[Z_COL],     float)
    muerr_all = np.array(tab[MUERR_COL], float)
    ra_all    = np.array(tab[RA_COL],    float)
    dec_all   = np.array(tab[DEC_COL],   float)

    base_mask = (
        np.isfinite(z_all) & (z_all > 0) &
        (z_all >= args.zmin) &
        np.isfinite(muerr_all) & (muerr_all > 0) &
        np.isfinite(ra_all) & np.isfinite(dec_all)
    )

    # optional |b| cut
    if args.bcut and args.bcut > 0:
        n_eq = unitvec_from_radec(ra_all, dec_all)         # (N,3)
        n_gal = n_eq @ EQ2GAL.T                            # (N,3)
        b_all = np.degrees(np.arcsin(np.clip(n_gal[:,2], -1.0, 1.0)))
        base_mask &= (np.abs(b_all) >= args.bcut)

    # optional dust cut
    if args.dustcut and args.dustcut > 0:
        dustcol = args.dustcol if args.dustcol else find_col(names, ["MWEBV","mwebv","E_BV","ebv"])
        if (dustcol is None) or (dustcol not in names):
            raise RuntimeError("Requested --dustcut but dust column not found. Use --dustcol.")
        dust_all = np.array(tab[dustcol], float)
        base_mask &= np.isfinite(dust_all) & (dust_all <= args.dustcut)

    # optional covariance
    cov_full = None
    if args.cov:
        cov_full = load_cov(args.cov)
        if cov_full.shape[0] != len(tab):
                    raise RuntimeError(f"FATAL: Covariance size ({cov_full.shape[0]}) does not match Data table length ({len(tab)}). Check inputs.")
        
    def gal_unitvec_from_lb(l_deg, b_deg):
        l = np.deg2rad(l_deg); b = np.deg2rad(b_deg)
        cb = np.cos(b)
        x = cb*np.cos(l); y = cb*np.sin(l); z = np.sin(b)
        return np.vstack([x,y,z]).T  # (N,3)
    
    zs = parse_list(args.scan)
    if zs:
        print("=== zmax scan (dipole-only test: Δχ² = χ²(mono) - χ²(mono+dip), dof depends on --fix_*_axis) ===")
        # print("=== zmax scan (dipole-only test: Δχ² = χ²(mono) - χ²(mono+dip), 3 dof) ===")
        print("zmax   N     H0     dChi2_tex   p_tex     sig_tex   "
              "dChi2_kin   p_kin     sig_kin   "
              "dChi2_mix   A_tex    A_kin(1/z)   v_bulk(km/s)   l_tex   b_tex   l_kin   b_kin")
        for zmax in zs:
            m = base_mask & (z_all <= zmax)
            idx = np.where(m)[0]
            r = fit_dipole(tab, idx, args, cov_full=cov_full)
            print(f"{zmax:0.3f}  {r['N']:4d}  {r['H0']:6.2f}  "
                  f"{r['dchi2_3']:9.2f}  {r['p3']:1.3e}  {r['z1_3']:7.2f}  "
                  f"{r.get('dchi2_kin', float('nan')):10.2f}  {r.get('p3_kin', float('nan')):1.3e}  {r.get('z1_3_kin', float('nan')):7.2f}  "
                  f"{r.get('dchi2_mix', float('nan')):10.2f}  "
                  f"{r.get('A_mu', float('nan')):0.6f}  {r.get('A_mu_kin', float('nan')):0.6f}  {r.get('v_bulk', float('nan')):10.1f}  "
                  f"{r.get('l', float('nan')):7.2f} {r.get('b', float('nan')):7.2f}  "
                  f"{r.get('l_kin', float('nan')):7.2f} {r.get('b_kin', float('nan')):7.2f}")
        return

    if args.scan_zmin_h0:
        try:
            zmins = [float(x) for x in args.scan_zmin_h0.split(",")]
        except:
            zmins = [0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
        
        zmins = sorted(list(set(zmins)))
        ln10_over_5 = math.log(10.0) / 5.0
        
        print("\n=== Effective H0 Impact Scan (with Error Propagation) ===")
        print("# Comparing Isotropic Monopole vs Directional Mix (KIN+TEX)")
        print("# Sigma(H0) ~ H0_grid * (ln10/5) * Sigma(a0)")
        print("-" * 130)
        header = (f"{'zmin':<8} {'N':<6} "
                  f"{'H0_mono':<10} {'err':<8} "
                  f"{'H0_mix':<10} {'err':<8} "
                  f"{'Delta_H0':<10} {'sig_Delta':<10}")
        print(header)
        print("-" * 130)

        rows_h0 = []

        for zmin_cut in zmins:
            # Local mask for this zmin threshold
            m = base_mask & (z_all >= zmin_cut) & (z_all <= args.zmax)
            idx = np.where(m)[0]
            
            if len(idx) < 10:
                continue

            # Run full fit pipeline
            r = fit_dipole(tab, idx, args, cov_full=cov_full)

            h0_grid = r['H0']
            
            # Retrieve intercepts
            a0_iso  = r['a0']
            a0_mix  = r.get('a0_mix', float('nan')) # Safe retrieval
            
            # Retrieve intercept errors (from GLS covariance)
            e_a0_iso = r.get('err_a0_mono', 0.0)
            e_a0_mix = r.get('err_a0_mix', 0.0)
            
            # Calculate effective H0
            h0_eff_mono = h0_grid * (1.0 - ln10_over_5 * a0_iso)
            h0_eff_mix  = h0_grid * (1.0 - ln10_over_5 * a0_mix)
            
            # Error propagation to H0 units
            # sig_H0 = H0_grid * (ln10/5) * sig_a0
            sig_h0_mono = h0_grid * ln10_over_5 * e_a0_iso
            sig_h0_mix  = h0_grid * ln10_over_5 * e_a0_mix
            
            # Difference and quadrature error (conservative)
            delta = h0_eff_mix - h0_eff_mono
            sig_delta = math.sqrt(sig_h0_mono**2 + sig_h0_mix**2)

            print(f"{zmin_cut:<8.4f} {r['N']:<6d} "
                  f"{h0_eff_mono:<10.3f} {sig_h0_mono:<8.3f} "
                  f"{h0_eff_mix:<10.3f} {sig_h0_mix:<8.3f} "
                  f"{delta:<+10.3f} {sig_delta:<10.3f}")
            
            # Save data for plotting
            rows_h0.append((zmin_cut, r['N'], 
                            h0_eff_mono, sig_h0_mono, 
                            h0_eff_mix, sig_h0_mix,
                            delta, sig_delta))

        if rows_h0:
            out_csv = "h0_impact_scan.csv"
            with open(out_csv, "w") as f:
                # Explicit CSV header
                f.write("zmin,N,H0_mono,err_mono,H0_mix,err_mix,Delta_H0,err_Delta\n")
                for row in rows_h0:
                    # Clean float formatting
                    line = ",".join(f"{x:.6f}" for x in row)
                    f.write(line + "\n")
            print(f"\n[INFO] Saved scan data to {out_csv}")
        
        return        
        # zmins = parse_list(args.scan_zmin_h0)
        # zmins = sorted(list(set(zmins)))
        
        # print("\n=== Effective H0 Impact Scan (vs zmin) ===")
        # print("# Comparing Isotropic Monopole vs Directional Mix (KIN+TEX)")
        # print("# Formula: H0_eff = H0_grid * (1 - (ln10/5)*a0)")
        # print(f"# Fixed zmax = {args.zmax}")
        # print("-" * 110) # Un peu plus large pour les erreurs
        # print(f"{'zmin':<8} {'N':<6} {'H0_mono':<10} {'err':<8} {'H0_mix':<10} {'err':<8} {'Delta_H0':<10}")
        # print("-" * 110)

        # ln10_over_5 = math.log(10.0) / 5.0
        # rows_h0 = []

        # for zmin_cut in zmins:
        #     m = base_mask & (z_all >= zmin_cut) & (z_all <= args.zmax)
        #     idx = np.where(m)[0]
            
        #     if len(idx) < 10:
        #         continue

        #     r = fit_dipole(tab, idx, args, cov_full=cov_full)

        #     h0_grid = r['H0']
        #     a0_iso  = r['a0']
        #     a0_mix  = r['a0_mix']
            
        #     # Récupération des erreurs (ajoutées à l'étape 1)
        #     e_iso = r.get('err_a0_mono', 0.0)
        #     e_mix = r.get('err_a0_mix', 0.0)
            
        #     # Calcul des H0 effectifs
        #     h0_eff_mono = h0_grid * (1.0 - ln10_over_5 * a0_iso)
        #     h0_eff_mix  = h0_grid * (1.0 - ln10_over_5 * a0_mix)
            
        #     # Propagation des erreurs
        #     # sigma_H = H * (ln10/5) * sigma_a0
        #     sig_h0_mono = h0_grid * ln10_over_5 * e_iso
        #     sig_h0_mix  = h0_grid * ln10_over_5 * e_mix
            
        #     delta = h0_eff_mix - h0_eff_mono

        #     print(f"{zmin_cut:<8.4f} {r['N']:<6d} "
        #           f"{h0_eff_mono:<10.3f} {sig_h0_mono:<8.3f} "
        #           f"{h0_eff_mix:<10.3f} {sig_h0_mix:<8.3f} "
        #           f"{delta:<+10.3f}")
            
        #     # On sauvegarde aussi les erreurs dans le CSV
        #     rows_h0.append((zmin_cut, r['N'], h0_eff_mono, sig_h0_mono, h0_eff_mix, sig_h0_mix))

        # if rows_h0:
        #     out_csv = "h0_impact_scan.csv"
        #     with open(out_csv, "w") as f:
        #         f.write("zmin,N,H0_mono,err_mono,H0_mix,err_mix\n")
        #         for row in rows_h0:
        #             f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n")
        #     print(f"\n[INFO] Saved scan data to {out_csv}")
        
        # return
    
    zbin_results = []
    zb = parse_bins(args.zbins)
    if zb:
        if args.zbins_global:
            # Require fixed axes for an auditable TEX(z)
            if not (args.fix_tex_axis and args.fix_kin_axis):
                raise RuntimeError(
                    "--zbins_global requires --fix_tex_axis and --fix_kin_axis "
                    "(axes must be locked)."
                )

            # Union selection over bins (auditable; not per-bin H0)
            zlo_u = min(a for a, b in zb)
            zhi_u = max(b for a, b in zb)
            mU = base_mask & (z_all > zlo_u) & (z_all <= zhi_u)
            idxU = np.where(mU)[0]
            if idxU.size < 10:
                raise RuntimeError("Not enough SNe in union of bins.")

            # Pull arrays on union
            raU    = np.array(tab[RA_COL],    float)[idxU]
            decU   = np.array(tab[DEC_COL],   float)[idxU]
            muU    = np.array(tab[MU_COL],    float)[idxU]
            zU     = np.array(tab[Z_COL],     float)[idxU]
            muerrU = np.maximum(
                np.array(tab[MUERR_COL], float)[idxU], 1e-6
            )

            # Whitening on union
            if (cov_full is not None) and args.use_cov:
                CU = cov_full[np.ix_(idxU, idxU)].copy()
                if args.sigint > 0:
                    CU[np.diag_indices_from(CU)] += args.sigint**2
                LU = whiten_from_cov(CU)
            else:
                sig2U = muerrU*muerrU + args.sigint*args.sigint
                LU = np.sqrt(sig2U)

            # Global H0 grid-search on union
            H0_grid = np.linspace(args.h0min, args.h0max, args.h0n)
            best = (1e300, None)
            for H0 in H0_grid:
                mu_th = hubble_mu_model(zU, H0, q0=args.q0)
                chi2 = chi2_only(LU, (muU - mu_th))
                if chi2 < best[0]:
                    best = (chi2, H0)
            chi2_iso_U, H0_best = best
            mu_th_U = hubble_mu_model(zU, H0_best, q0=args.q0)
            dmuU = (muU - mu_th_U).astype(float)

            # Geometry + fixed axes projections (gal -> eq)
            nU = unitvec_from_radec(raU, decU)     # (NU,3)
            zinvU = 1.0 / np.maximum(zU, 1e-6)

            tex_fix = _parse_lb(args.fix_tex_axis)
            kin_fix = _parse_lb(args.fix_kin_axis)

            l0, b0 = tex_fix
            dgal_tex = _uvec_from_lb(l0, b0)
            deq_tex = (EQ2GAL.T @ dgal_tex).astype(float)
            proj_tex_U = (nU @ deq_tex).astype(float)

            l1, b1 = kin_fix
            dgal_kin = _uvec_from_lb(l1, b1)
            deq_kin = (EQ2GAL.T @ dgal_kin).astype(float)
            proj_kin_U = (nU @ deq_kin).astype(float)

            # Design matrices (axes locked)
            X0U   = np.ones((len(dmuU), 1), float)
            XtexU = np.column_stack([np.ones(len(dmuU)), proj_tex_U])
            XkinU = np.column_stack([np.ones(len(dmuU)), proj_kin_U*zinvU])
            XmixU = np.column_stack(
                [np.ones(len(dmuU)), proj_tex_U, proj_kin_U*zinvU]
            )

            # Global fits (per model) -> used for diagnostic contributions
            beta0U,  chi2_mono_U = gls_fit_and_chi2(LU, X0U,   dmuU)
            betatU, chi2_tex_U   = gls_fit_and_chi2(LU, XtexU, dmuU)
            betakU, chi2_kin_U   = gls_fit_and_chi2(LU, XkinU, dmuU)
            betamU, chi2_mix_U   = gls_fit_and_chi2(LU, XmixU, dmuU)

            print("=== z bins (GLOBAL H0 + LOCKED axes; per-bin fits auditable) ===")
            print(
                f"# Union z in ({zlo_u:.3f},{zhi_u:.3f}]  "
                f"N={len(idxU)}  H0_best={H0_best:.2f}  (q0={args.q0})"
            )
            if args.use_cov and args.cov:
                print("# Using full covariance. Note: per-bin C-submatrix χ² "
                      "are diagnostic (cross-bin cov breaks additivity).")

            print(
                "zbin           N   H0glob "
                "dChi2_tex_fit  A_tex_fit    "
                "dChi2_kin_fit  v_bulk_fit   "
                "dChi2(D|K)_fit p(D|K)       "
                "dChi2(K|D)_fit p(K|D)       "
                "dChi2_tex_fix  dChi2_kin_fix  dChi2_mix_fix"
            )

            sum_fix_tex = 0.0
            sum_fix_kin = 0.0
            sum_fix_mix = 0.0

            # Per-bin loop (locked axes, locked H0)
            for (zlo, zhi) in zb:
                mB = base_mask & (z_all > zlo) & (z_all <= zhi)
                idxB = np.where(mB)[0]
                if idxB.size < 10:
                    print(f"{zlo:0.3f}-{zhi:0.3f}   {idxB.size:4d}  (skip)")
                    continue

                # Pull arrays in this bin (keep global H0_best)
                raB  = np.array(tab[RA_COL],    float)[idxB]
                decB = np.array(tab[DEC_COL],   float)[idxB]
                muB  = np.array(tab[MU_COL],    float)[idxB]
                zB   = np.array(tab[Z_COL],     float)[idxB]
                muerrB = np.maximum(
                    np.array(tab[MUERR_COL], float)[idxB], 1e-6
                )

                mu_th_B = hubble_mu_model(zB, H0_best, q0=args.q0)
                dmuB = (muB - mu_th_B).astype(float)

                # Bin whitening
                if (cov_full is not None) and args.use_cov:
                    CB = cov_full[np.ix_(idxB, idxB)].copy()
                    if args.sigint > 0:
                        CB[np.diag_indices_from(CB)] += args.sigint**2
                    LB = whiten_from_cov(CB)
                else:
                    sig2B = muerrB*muerrB + args.sigint*args.sigint
                    LB = np.sqrt(sig2B)

                # Bin geometry + projections on the same locked axes
                nB = unitvec_from_radec(raB, decB)
                zinvB = 1.0 / np.maximum(zB, 1e-6)
                proj_tex_B = (nB @ deq_tex).astype(float)
                proj_kin_B = (nB @ deq_kin).astype(float)

                X0B   = np.ones((len(dmuB), 1), float)
                XtexB = np.column_stack([np.ones(len(dmuB)), proj_tex_B])
                XkinB = np.column_stack(
                    [np.ones(len(dmuB)), proj_kin_B*zinvB]
                )
                XmixB = np.column_stack(
                    [np.ones(len(dmuB)), proj_tex_B, proj_kin_B*zinvB]
                )

                # (A) Per-bin GLS fits (axes locked, H0 locked)
                b0,  c0 = gls_fit_and_chi2(LB, X0B,   dmuB)
                bt,  ct = gls_fit_and_chi2(LB, XtexB, dmuB)
                bk,  ck = gls_fit_and_chi2(LB, XkinB, dmuB)
                bm,  cm = gls_fit_and_chi2(LB, XmixB, dmuB)

                dchi2_tex_fit = c0 - ct
                dchi2_kin_fit = c0 - ck
                dchi2_mix_fit = c0 - cm

                dchi2_DgK_fit = ck - cm   # add TEX given KIN
                dchi2_KgD_fit = ct - cm   # add KIN given TEX

                p_DgK = chi2_sf_df1(dchi2_DgK_fit)
                p_KgD = chi2_sf_df1(dchi2_KgD_fit)

                A_tex_fit = abs(float(bt[1]))
                v_bulk_fit = (
                    math.log(10.0)/5.0 * C_LIGHT * abs(float(bk[1]))
                )

                # (B) Diagnostics: contributions from global betas
                chi2_mono_fix = chi2_given_beta(LB, X0B,   dmuB, beta0U)
                chi2_tex_fix  = chi2_given_beta(LB, XtexB, dmuB, betatU)
                chi2_kin_fix  = chi2_given_beta(LB, XkinB, dmuB, betakU)
                chi2_mix_fix  = chi2_given_beta(LB, XmixB, dmuB, betamU)

                dchi2_tex_fix = chi2_mono_fix - chi2_tex_fix
                dchi2_kin_fix = chi2_mono_fix - chi2_kin_fix
                dchi2_mix_fix = chi2_mono_fix - chi2_mix_fix

                sum_fix_tex += dchi2_tex_fix
                sum_fix_kin += dchi2_kin_fix
                sum_fix_mix += dchi2_mix_fix

                print(
                    f"{zlo:0.3f}-{zhi:0.3f}  {len(idxB):4d}  {H0_best:6.2f}  "
                    f"{dchi2_tex_fit:11.2f}  {A_tex_fit:0.6f}  "
                    f"{dchi2_kin_fit:11.2f}  {v_bulk_fit:9.1f}  "
                    f"{dchi2_DgK_fit:12.2f}  {p_DgK:1.3e}  "
                    f"{dchi2_KgD_fit:12.2f}  {p_KgD:1.3e}  "
                    f"{dchi2_tex_fix:11.2f}  {dchi2_kin_fix:11.2f}  "
                    f"{dchi2_mix_fix:11.2f}"
                )

                # Store locked-axis per-bin summary for pillar plots
                zc    = 0.5 * (zlo + zhi)
                width = (zhi - zlo)
                zbin_results.append({
                    "zlo": zlo,
                    "zhi": zhi,
                    "zc": zc,
                    "width": width,
                    "dchi2_kin": dchi2_kin_fit,
                    "dchi2_tex_D_given_K": dchi2_DgK_fit,
                    "dchi2_tex_iso": dchi2_tex_fit,
                    "A_tex": A_tex_fit,
                    "v_bulk": v_bulk_fit,
                    "N": len(idxB),
                })

            # Global headline (locked axes)
            dchi2_tex_U = chi2_mono_U - chi2_tex_U
            dchi2_kin_U = chi2_mono_U - chi2_kin_U
            dchi2_mix_U = chi2_mono_U - chi2_mix_U

            print("\n# Global (union) with locked axes:")
            print(
                f"#  Δχ²_tex={dchi2_tex_U:.3f}  "
                f"Δχ²_kin={dchi2_kin_U:.3f}  Δχ²_mix={dchi2_mix_U:.3f}"
            )
            print(
                "#  Sum over bins (fixed-beta diagnostic): "
                f"Δχ²_tex={sum_fix_tex:.3f}  "
                f"Δχ²_kin={sum_fix_kin:.3f}  Δχ²_mix={sum_fix_mix:.3f}"
            )
            
            if args.null_mock_tomo:
                run_null_mock_tomography(tab, base_mask, args, cov_full, zbin_results)

        # Each bin fitted independently (axes may float unless --fix_*_axis)
        print("=== z bins (each bin fitted independently) ===")
        print(
            "zbin           N     H0     dChi2_tex  A_tex     l_tex   b_tex   "
            "dChi2_kin  v_bulk   l_kin   b_kin   dChi2(D|K)  p(D|K)   "
            "dChi2(K|D)  p(K|D)"
        )
        
        
        for (zlo, zhi) in zb:
            m = base_mask & (z_all > zlo) & (z_all <= zhi)
            idx = np.where(m)[0]
            if idx.size < 10:
                print(f"{zlo:0.3f}-{zhi:0.3f}   {idx.size:4d}  (skip)")
                continue

            r = fit_dipole(tab, idx, args, cov_full=cov_full)

            print(
                f"{zlo:0.3f}-{zhi:0.3f}  {r['N']:4d}  {r['H0']:6.2f}  "
                f"{r['dchi2_3']:9.2f}  {r['A_mu']:0.6f}  "
                f"{r['l']:7.2f} {r['b']:7.2f}  "
                f"{r.get('dchi2_kin', float('nan')):9.2f}  "
                f"{r.get('v_bulk', float('nan')):7.1f}  "
                f"{r.get('l_kin', float('nan')):7.2f} {r.get('b_kin', float('nan')):7.2f}  "
                f"{r.get('dchi2_add_tex_given_kin', float('nan')):10.2f}  "
                f"{r.get('p_add_tex_given_kin', float('nan')):1.3e}  "
                f"{r.get('dchi2_add_kin_given_tex', float('nan')):10.2f}  "
                f"{r.get('p_add_kin_given_tex', float('nan')):1.3e}"
            )

            # IMPORTANT: do not overwrite zbin_results when --zbins_global is used.
            if not args.zbins_global:
                zc    = 0.5 * (zlo + zhi)
                width = (zhi - zlo)
                zbin_results.append({
                    "zlo": zlo,
                    "zhi": zhi,
                    "zc": zc,
                    "width": width,
                    "dchi2_kin": r.get("dchi2_kin", float("nan")),
                    "dchi2_tex_D_given_K": r.get(
                        "dchi2_add_tex_given_kin", float("nan")
                    ),
                    "dchi2_tex_iso": r.get("dchi2_3", float("nan")),
                    "A_tex": r.get("A_mu", float("nan")),
                    "v_bulk": r.get("v_bulk", float("nan")),
                    "N": r['N'],
                })

        if not (args.make_pillar_plots or args.make_h0_residuals):
            return


    # single run
    m = base_mask & (z_all <= args.zmax)
    idx0 = np.where(m)[0]
    r0 = fit_dipole(tab, idx0, args, cov_full=cov_full)

    # ========= Mock  =========
    if args.null_mock_tomo:
            # Run the null test and exit
            run_null_mock_tomography(tab, idx0, args, cov_full, n_mocks=1000)
            return   
         
    # ========= Standalone sky map of Hubble residuals =========
    if args.make_h0_residuals:
        # Basic columns
        ra    = np.array(tab[RA_COL],    float)[idx0]
        dec   = np.array(tab[DEC_COL],   float)[idx0]
        mu    = np.array(tab[MU_COL],    float)[idx0]
        z     = np.array(tab[Z_COL],     float)[idx0]

        # Hubble residuals with respect to the best-fit isotropic H0
        H0_best = r0["H0"]
        mu_th   = hubble_mu_model(z, H0_best, q0=args.q0)
        dmu     = mu - mu_th

        # Convert equatorial coordinates to galactic coordinates
        n_eq  = unitvec_from_radec(ra, dec)
        n_gal = n_eq @ EQ2GAL.T
        l = (np.degrees(np.arctan2(n_gal[:, 1], n_gal[:, 0])) + 360.0) % 360.0
        b = np.degrees(np.arcsin(np.clip(n_gal[:, 2], -1.0, 1.0)))

        # Matplotlib Mollweide requires radians in range [-pi, pi] for longitude
        l_rad = np.radians(l)
        l_rad[l_rad > np.pi] -= 2 * np.pi  # Wrap 0..360 to -180..180

        # Enforce astronomical convention (longitude increases to the left)
        l_rad = -l_rad

        b_rad = np.radians(b)

        # Axis Locations (Convert to radians + wrap)
        def to_mollweide_rad(l_deg, b_deg):
            lr = np.radians(l_deg)
            if lr > np.pi:
                lr -= 2 * np.pi
            # >>> FIX: same mirror for markers
            lr = -lr
            br = np.radians(b_deg)
            return lr, br

        # Colors matching Figure S2
        c_tex = "#EE3333" # Red
        c_kin = "#0000BB" # Blue
        c_ref = 'black'

        # Create Figure
        fig, ax = plt.subplots(1, 1, figsize=(10.0, 6.0), subplot_kw={'projection': 'mollweide'})

        vmax = float(np.max(np.abs(dmu)))
        sc = ax.scatter(
            l_rad, b_rad, c=dmu, s=20,
            cmap="coolwarm", vmin=-vmax, vmax=+vmax,
            alpha=0.8, edgecolors="none", rasterized=True
        )

        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_title("Hubble residuals $\\Delta\\mu$ on the sky (Galactic)")
        
        # Correct Matplotlib graduations (astro labels)
        tick_pos_deg = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
        tick_pos_rad = np.deg2rad(tick_pos_deg)
        astronomical_xticks = ['150°','120°','90°','60°','30°','0°','330°','300°','270°','240°','210°']

        ax.set_xticks(tick_pos_rad)
        ax.set_xticklabels(astronomical_xticks)

        # Overlay TEX axis
        if math.isfinite(r0["l"]):
            lt, bt = to_mollweide_rad(r0["l"], r0["b"])
            ax.scatter(lt, bt, marker="*", s=200, facecolor=c_tex, edgecolor='k', 
                       linewidth=1.2, zorder=10, label="TEX axis (Baseline T1)")

        # Overlay KIN axis
        if "l_kin" in r0 and math.isfinite(r0["l_kin"]):
            lk, bk = to_mollweide_rad(r0["l_kin"], r0["b_kin"])
            ax.scatter(lk, bk, marker="D", s=100, facecolor=c_kin, edgecolor='k', 
                       linewidth=1.2, zorder=10, label="KIN axis (Baseline T1)")

        # Overlay Anti-GA
        l_aga, b_aga = to_mollweide_rad(AGA_L, AGA_B)
        ax.scatter(l_aga, b_aga, marker="x", s=100, color=c_ref, 
                   linewidth=2, zorder=9, label="Anti-Norma Direction")

        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

        cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04, shrink=0.6)
        cbar.set_label("$\\Delta\\mu$ [mag]")

        fig.tight_layout()
        fig.savefig("fig_1a_residuals_sky_mollweide.png", dpi=300)
        plt.close(fig)
        print("[INFO] Saved figure 'fig_1a_residuals_sky_mollweide.png' (Mollweide)")


    # ========= Pillar plots: 4 histograms over z-bins =========
    if args.make_pillar_plots:
        if not zbin_results:
            raise RuntimeError(
                "--make_pillar_plots need zbins option \n"
                "You may add:\n"
                "  --zbins 0.020-0.025,0.025-0.030,0.030-0.035,0.035-0.040 "
                "--make_pillar_plots"
            )

        zc_arr        = np.array([d["zc"]    for d in zbin_results], float)
        zwidth_arr    = np.array([d["width"] for d in zbin_results], float)
        dchi2_kin_arr = np.array([d["dchi2_kin"] for d in zbin_results], float)
        dchi2_tex_arr = np.array(
            [d["dchi2_tex_D_given_K"] for d in zbin_results], float
        )
        A_arr         = np.array([d["A_tex"]   for d in zbin_results], float)
        v_arr         = np.array([d["v_bulk"]  for d in zbin_results], float)
        N_arr         = np.array([d.get("N", np.nan) for d in zbin_results], float)

        if zc_arr.size == 0:
            print("[WARN] No z-bins with enough SNe for pillar plots.")
        else:
            # Flag low-statistics bins where v_bulk is poorly constrained
            # Threshold N_min can be adjusted if needed.
            N_min = 50
            lowN_mask = np.isfinite(N_arr) & (N_arr < N_min)

            # common scale for (a) and (b)
            finite_mask = np.isfinite(dchi2_kin_arr) & np.isfinite(dchi2_tex_arr)
            if np.any(finite_mask):
                ymax = np.max(
                    np.concatenate([
                        dchi2_kin_arr[finite_mask],
                        dchi2_tex_arr[finite_mask],
                    ])
                )
                ymax *= 1.05
            else:
                ymax = 1.0

            # Create 2x2 figure: all bar plots / histograms
            fig = plt.figure(figsize=(10.5, 8.5))
            gs  = fig.add_gridspec(2, 2)

            ax_a = fig.add_subplot(gs[0, 0])   # Δχ² from KIN
            ax_b = fig.add_subplot(gs[0, 1])   # Δχ² from TEX (on top of KIN)
            ax_c = fig.add_subplot(gs[1, 0])   # A_tex(z)
            ax_d = fig.add_subplot(gs[1, 1])   # v_bulk(z)

            width = 0.8 * zwidth_arr

            # Convenience masks
            good = ~lowN_mask
            bad  = lowN_mask

            # (a) χ² reduction due to bulk flow (KIN): iso → KIN
            ax_a.bar(zc_arr[good], dchi2_kin_arr[good],
                     width=width[good], align="center", alpha=0.85)
            ax_a.bar(zc_arr[bad], dchi2_kin_arr[bad],
                     width=width[bad], align="center", alpha=0.85,
                     hatch="///")
            ax_a.set_xlabel("$z$")
            ax_a.set_ylabel("$\\Delta\\chi^2$ per bin")
            ax_a.set_title("(a) $\\chi^2$ reduction from KIN (bulk flow)")
            ax_a.axhline(0.0, color="0.5", lw=1)
            ax_a.set_ylim(0.0, ymax)

            # (b) χ² reduction due to TEX on top of KIN: KIN → KIN+TEX
            ax_b.bar(zc_arr[good], dchi2_tex_arr[good],
                     width=width[good], align="center", alpha=0.85)
            ax_b.bar(zc_arr[bad], dchi2_tex_arr[bad],
                     width=width[bad], align="center", alpha=0.85,
                     hatch="///")
            ax_b.set_xlabel("$z$")
            ax_b.set_ylabel("$\\Delta\\chi^2$ per bin")
            ax_b.set_title("(b) $\\chi^2$ reduction from TEX (on top of KIN)")
            ax_b.axhline(0.0, color="0.5", lw=1)
            ax_b.set_ylim(0.0, ymax)

            # (c) TEX amplitude profile A_tex(z)
            A_tex_global = r0["A_mu"]
            ax_c.bar(zc_arr[good], A_arr[good],
                     width=width[good], align="center", alpha=0.85)
            ax_c.bar(zc_arr[bad], A_arr[bad],
                     width=width[bad], align="center", alpha=0.85,
                     hatch="///")
            ax_c.axhline(A_tex_global, color="0.3", ls="--", lw=1,
                         label="Global TEX amplitude")
            ax_c.set_xlabel("$z$")
            ax_c.set_ylabel("$A_\\mu^{\\rm TEX}(z)$ [mag]")
            ax_c.set_title("(c) TEX amplitude by z-bin")
            ax_c.legend(loc="best", fontsize=8)

            # (d) Bulk velocity per z-bin
            ax_d.bar(zc_arr[good], v_arr[good],
                     width=width[good], align="center", alpha=0.85)
            ax_d.bar(zc_arr[bad], v_arr[bad],
                     width=width[bad], align="center", alpha=0.85,
                     hatch="///")
            ax_d.set_xlabel("$z$")
            ax_d.set_ylabel("Bulk velocity [km/s]")
            ax_d.set_title("(d) Bulk velocity by z-bin")

            # Use 3 decimal places on all z-axes for consistency with the z-bins
            for ax in (ax_a, ax_b, ax_c, ax_d):
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

            fig.tight_layout()
            fig.savefig("fig_pillar_tex_kin.png", dpi=200)
            plt.close(fig)
            print("[INFO] Saved figure 'fig_pillar_tex_kin.png'")
        
    # ========= Global header =========
    print("\n=== Test A: SN low-z dipole on Hubble residuals ===")
    print(f"Sample: N = {r0['N']} SN  |  z ∈ [{args.zmin}, {args.zmax}]")
    print(f"Best-fit isotropic H0 (grid, q0={args.q0}): {r0['H0']:.2f} km/s/Mpc")
    if args.use_cov and args.cov:
        print("Fitting method: GLS with full covariance matrix.")
    else:
        print("Fitting method: diagonal errors only (MUERR).")

    # ========= Baseline fits (no dipole) =========
    print("\n--- Baseline (no dipole) ---")
    print(f"χ²_iso  (H0-only FRW, isotropic): {r0['chi2_iso']:.2f}")
    print(f"χ²_mono (a0-only, mean residual): {r0['chi2_mono']:.2f}")

    # ========= TEX: constant-in-z dipole (tex-like) =========
    dof_tex = r0.get('dof_tex', 3)
    print("\n--- TEX: constant-in-z dipole on μ (tex-like mode) ---")
    print(f"χ²_TEX (a0 + A_tex·n, const amp): {r0['chi2_dip']:.2f}")
    print(f"Δχ²_TEX (MONO → TEX, {dof_tex} dof): {r0['dchi2_3']:.2f}")
    print(f"p-value_TEX (χ², {dof_tex} dof): {r0['p3']:.3e}")
    print(f"σ_TEX (equiv.): {r0['z1_3']:.2f}σ (one-sided), {r0['z2_3']:.2f}σ (two-sided)")

    # Comparaison directe H0-only → (a0+dipole)
    print("\n[Alternative comparison: H0-only → (a0 + TEX dipole)]")
    print(f"Δχ²_total (4 dof): {r0['dchi2_4']:.2f}")
    print(f"p-value (χ², 4 dof): {r0['p4']:.3e}")
    print(f"σ (equiv.): {r0['z1_4']:.2f}σ (one-sided), {r0['z2_4']:.2f}σ (two-sided)")

    # Amplitude et direction TEX
    print("\n TEX dipole parameters (constant in z)")
    print(f"A_tex ≡ A_mu (const): {r0['A_mu']:.6f} mag")
    print(f"|δH0/H0| ≈ (ln10/5)·A_tex = {r0['frac']*100:.3f}%")
    print(f"TEX axis (equatorial): RA = {r0['ra']:.2f} deg, Dec = {r0['dec']:.2f} deg")
    print(f"TEX axis (galactic, J2000):  l = {r0['l']:.2f} deg,  b = {r0['b']:.2f} deg")
    print(f"Monopole offset a0 (mag): {r0['a0']:.6e}")

    # ========= KIN / MIX: kinematic 1/z partition (if present) =========
    if "chi2_kin" in r0:
        print("\n--- KIN: kinematic 1/z dipole on μ ---")
        print(
            f"χ²_KIN (a0 + (K·n)/z): {r0['chi2_kin']:.2f}   "
            f"Δχ²_KIN = {r0['dchi2_kin']:.2f} (3 dof)   "
            f"p_KIN = {r0['p3_kin']:.3e}   "
            f"z_KIN ≈ {r0['z1_3_kin']:.2f}σ"
        )
        print(
            f"A_kin (1/z coeff): {r0['A_mu_kin']:.6f} mag   "
            f"v_bulk ≈ {r0['v_bulk']:.1f} km/s   "
            f"\nKIN axis (galactic, J2000): l = {r0['l_kin']:.2f} deg, b = {r0['b_kin']:.2f} deg"
        )

        print("\n--- MIX: TEX + KIN combined ---")
        print(
            f"χ²_MIX (a0 + TEX + KIN): {r0['chi2_mix']:.2f}   "
            f"Δχ²_MIX = {r0['dchi2_mix']:.2f} (6 dof vs MONO)"
        )
        print(
            f"Incremental Δχ² (add KIN | TEX): {r0['dchi2_add_kin_given_tex']:.2f}   "
            f"(i.e. TEX → TEX+KIN)"
        )
        print(
            f"Incremental Δχ² (add TEX | KIN): {r0['dchi2_add_tex_given_kin']:.2f}   "
            f"(i.e. KIN → KIN+TEX)"
        )

    # ========= Dust test (if requested) =========
    if args.dustcheck and ("p_dust" in r0):
        print("\n=== Dust correlation test ===")
        print(f"Dust column proxy: {r0.get('dustcol', '?')}")
        print(f"corr(dμ, dust): {r0.get('corr_dust', float('nan')):.4f}")
        print(
            f"Δχ²_dust (add dust term, 1 dof): {r0['dchi2_dust']:.2f}   "
            f"p_dust = {r0['p_dust']:.3e}   z_dust ≈ {r0['z_dust']:.2f}σ"
        )
        print(f"Dust slope k (mag per unit dust): {r0['k_dust']:.6e}")

        print(
            f"Δχ²_dip|dust (add TEX dipole on top of dust, 3 dof): "
            f"{r0['dchi2_dip_cd']:.2f}   p = {r0['p_cd']:.3e}   z ≈ {r0['z_cd']:.2f}σ"
        )
        print(
            f"A_tex(dip|dust): {r0['A_mu_cd']:.6f} mag   "
            f"|δH0/H0|(dip|dust) ≈ {r0['frac_cd']*100:.3f}%"
        )
        print(
            f"TEX axis (dip|dust, galactic): "
            f"l = {r0.get('l_cd', float('nan')):.2f} deg (ℓtex), "
            f"b = {r0.get('b_cd', float('nan')):.2f} deg"
        )
        
    # permutation empirical pvals (footprint-aware)
    # =====================
    # Optional robustness / diagnostics
    # =====================
    if args.influence:
        influence_diagnostics(
            tab, idx0, args, cov_full=cov_full,
            H0_ref=r0["H0"], mode=args.influence_mode,
            top=args.influence_top, dropone=args.influence_dropone
        )

    if args.jackknife_dir and args.jackknife_dir > 0:
        dump_csv = args.jackknife_dir_csv if args.jackknife_dir_csv else None
        directional_jackknife(
            tab, idx0, args, cov_full=cov_full,
            n_dirs=args.jackknife_dir,
            mode=args.jackknife_dir_mode,
            theta_deg=args.jackknife_dir_theta,
            compare=args.jackknife_dir_compare,
            dump_csv=dump_csv
        )

    if args.constN:
        try:
            zmins = [float(s.strip()) for s in args.constN.split(",") if s.strip() != ""]
            zmins = sorted(set(zmins))
        except Exception as e:
            raise SystemExit(f"Could not parse --constN list: {e}")

        dump_csv = args.constN_csv if args.constN_csv else None
        constant_N_test(
            tab, args, cov_full=cov_full,
            zmins=zmins,
            draws=args.constN_draws,
            N0=args.constN_N0,
            model=args.constN_model,
            use_cov=args.constN_use_cov,
            seed=args.seed,
            dump_csv=dump_csv
        )

    if args.permute > 0:
        # pull columns again for this selection
        ra    = np.array(tab[RA_COL],    float)[idx0]
        dec   = np.array(tab[DEC_COL],   float)[idx0]
        mu    = np.array(tab[MU_COL],    float)[idx0]
        z     = np.array(tab[Z_COL],     float)[idx0]
        muerr = np.maximum(np.array(tab[MUERR_COL], float)[idx0], 1e-6)

        # # rebuild L
        # if (cov_full is not None) and args.use_cov:
        #     C = cov_full[np.ix_(idx0, idx0)].copy()
        #     if args.sigint > 0:
        #         C[np.diag_indices_from(C)] += args.sigint**2
        #     L = whiten_from_cov(C)
        # else:
        #     sig2 = muerr*muerr + args.sigint*args.sigint
        #     L = np.sqrt(sig2)
        # rebuild L
        if (cov_full is not None) and args.use_cov:
            C = cov_full[np.ix_(idx0, idx0)].copy()
            if args.sigint > 0:
                C[np.diag_indices_from(C)] += args.sigint**2
            
            if getattr(args, "sigv", 0.0) > 0:
                z_safe = np.maximum(z, 1e-6)
                sigmu_v = (5.0/np.log(10.0)) * (args.sigv / (C_LIGHT * z_safe))
                C[np.diag_indices_from(C)] += sigmu_v**2
        
            L = whiten_from_cov(C)
        else:
            sig2 = muerr*muerr + args.sigint*args.sigint
            
            if getattr(args, "sigv", 0.0) > 0:
                z_safe = np.maximum(z, 1e-6)
                sigmu_v = (5.0/np.log(10.0)) * (args.sigv / (C_LIGHT * z_safe))
                sig2 = sig2 + sigmu_v*sigmu_v
        
            L = np.sqrt(sig2)
        # recompute best H0 -> dmu
        H0_grid = np.linspace(args.h0min, args.h0max, args.h0n)
        best = (1e300, None)
        for H0 in H0_grid:
            mu_th = hubble_mu_model(z, H0, q0=args.q0)
            chi2 = chi2_only(L, (mu - mu_th))
            if chi2 < best[0]:
                best = (chi2, H0)
        mu_th = hubble_mu_model(z, best[1], q0=args.q0)
        dmu = (mu - mu_th).astype(float)

        n = unitvec_from_radec(ra, dec)
        zinv = 1.0 / np.maximum(z, 1e-6)

        tex_fix = _parse_lb(getattr(args, "fix_tex_axis", ""))
        kin_fix = _parse_lb(getattr(args, "fix_kin_axis", ""))

        proj_tex = proj_kin = None
        if tex_fix is not None:
            l0,b0 = tex_fix
            dgal = _uvec_from_lb(l0,b0)
            deq_tex = (EQ2GAL.T @ dgal).astype(float)
            proj_tex = (n @ deq_tex).astype(float)

        X0 = np.ones((len(dmu),1), float)

        if tex_fix is None:
            X_tex = np.column_stack([np.ones(len(dmu)), n[:,0], n[:,1], n[:,2]])
        else:
            X_tex = np.column_stack([np.ones(len(dmu)), proj_tex])

        if kin_fix is None:
            X_kin = np.column_stack([np.ones(len(dmu)), n[:,0]*zinv, n[:,1]*zinv, n[:,2]*zinv])
        else:
            X_kin = np.column_stack([np.ones(len(dmu)), proj_kin*zinv])

        mix_cols = [np.ones(len(dmu))]
        mix_cols += ([n[:,0],n[:,1],n[:,2]] if tex_fix is None else [proj_tex])
        mix_cols += ([n[:,0]*zinv,n[:,1]*zinv,n[:,2]*zinv] if kin_fix is None else [proj_kin*zinv])
        X_mix = np.column_stack(mix_cols)

        proj_tex = proj_kin = None
        if tex_fix is not None:
            l0,b0 = tex_fix
            dgal = _uvec_from_lb(l0,b0)
            deq_tex = (EQ2GAL.T @ dgal).astype(float)
            proj_tex = (n @ deq_tex).astype(float)

        if kin_fix is not None:
            l0,b0 = kin_fix
            dgal = _uvec_from_lb(l0,b0)
            deq_kin = (EQ2GAL.T @ dgal).astype(float)
            proj_kin = (n @ deq_kin).astype(float)

        X0 = np.ones((len(dmu),1), float)

        if tex_fix is None:
            X_tex = np.column_stack([np.ones(len(dmu)), n[:,0], n[:,1], n[:,2]])
        else:
            X_tex = np.column_stack([np.ones(len(dmu)), proj_tex])

        if kin_fix is None:
            X_kin = np.column_stack([np.ones(len(dmu)), n[:,0]*zinv, n[:,1]*zinv, n[:,2]*zinv])
        else:
            X_kin = np.column_stack([np.ones(len(dmu)), proj_kin*zinv])

        mix_cols = [np.ones(len(dmu))]
        mix_cols += ([n[:,0],n[:,1],n[:,2]] if tex_fix is None else [proj_tex])
        mix_cols += ([n[:,0]*zinv,n[:,1]*zinv,n[:,2]*zinv] if kin_fix is None else [proj_kin*zinv])
        X_mix = np.column_stack(mix_cols)


        # whitened objects
        yw  = solve_lower(L, dmu)    # (N,)
        X0w   = solve_lower(L, X0)
        Xtexw = solve_lower(L, X_tex)
        Xkinw = solve_lower(L, X_kin)
        Xmixw = solve_lower(L, X_mix)

        Q0   = precompute_Q(X0w)
        Qtex = precompute_Q(Xtexw)
        Qkin = precompute_Q(Xkinw)
        Qmix = precompute_Q(Xmixw)

        # observed dchi2 in whitened space 
        dchi2_tex_obs = chi2_from_Q(yw, Q0) - chi2_from_Q(yw, Qtex)
        dchi2_kin_obs = chi2_from_Q(yw, Q0) - chi2_from_Q(yw, Qkin)
        dchi2_mix_obs = chi2_from_Q(yw, Q0) - chi2_from_Q(yw, Qmix)
        
        # conditional delta observed
        chi2_tex_only = chi2_from_Q(yw, Qtex)
        chi2_kin_only = chi2_from_Q(yw, Qkin)
        chi2_mix_only = chi2_from_Q(yw, Qmix)

        dchi2_D_given_K_obs = chi2_kin_only - chi2_mix_only  # add D on top of K
        dchi2_K_given_D_obs = chi2_tex_only - chi2_mix_only  # add K on top of D

        print("[DEBUG] dchi2_D_given_K (QR) =", f"{dchi2_D_given_K_obs:.6f}", "   vs fit:", f"{r0.get('dchi2_add_tex_given_kin', float('nan')):.6f}")
        print("[DEBUG] dchi2_K_given_D (QR) =", f"{dchi2_K_given_D_obs:.6f}", "   vs fit:", f"{r0.get('dchi2_add_kin_given_tex', float('nan')):.6f}")

        Qmw = Qmwdip = None
        dchi2_mw_obs = dchi2_dip_given_mw_obs = None

        if args.mwcheck:
            n_gal = n @ EQ2GAL.T
            sinb = n_gal[:,2]
            tname = (args.mwtemp or "p2").lower()
            if tname == "sinb":
                t = sinb
            elif tname == "abs_sinb":
                t = np.abs(sinb)
            elif tname == "inv_abs_sinb":
                t = 1.0 / np.maximum(np.abs(sinb), float(args.mw_sinb_min))
            else:
                t = 0.5*(3.0*sinb*sinb - 1.0)

            Xmw    = np.column_stack([np.ones(len(dmu)), t])
            Xmwdip = np.column_stack([np.ones(len(dmu)), t, n[:,0], n[:,1], n[:,2]])

            Xmw_w    = solve_lower(L, Xmw)
            Xmwdip_w = solve_lower(L, Xmwdip)

            Qmw    = precompute_Q(Xmw_w)
            Qmwdip = precompute_Q(Xmwdip_w)

            dchi2_mw_obs = chi2_from_Q(yw, Q0) - chi2_from_Q(yw, Qmw)
            dchi2_dip_given_mw_obs = chi2_from_Q(yw, Qmw) - chi2_from_Q(yw, Qmwdip)

            print("[DEBUG] dchi2_mw (QR)          =", f"{dchi2_mw_obs:.6f}")
            print("[DEBUG] dchi2_dip_given_mw(QR) =", f"{dchi2_dip_given_mw_obs:.6f}")


        print("\n[DEBUG] dchi2_tex_obs (QR)  =", f"{dchi2_tex_obs:.6f}", "   vs fit:", f"{r0.get('dchi2_3', float('nan')):.6f}")
        print("[DEBUG] dchi2_kin_obs (QR)  =", f"{dchi2_kin_obs:.6f}", "   vs fit:", f"{r0.get('dchi2_kin', float('nan')):.6f}")
        print("[DEBUG] dchi2_mix_obs (QR)  =", f"{dchi2_mix_obs:.6f}", "   vs fit:", f"{r0.get('dchi2_mix', float('nan')):.6f}")

        # surveys if stratified
        surveys = None
        if args.permute_within_survey:
            surv_name = args.surveycol if args.surveycol else find_col(names, ["IDSURVEY","idSURVEY","survey","SURVEY","SURVEYID"])
            if (surv_name is None) or (surv_name not in names):
                raise RuntimeError("Need IDSURVEY for --permute_within_survey. Use --surveycol IDSURVEY if present.")
            surveys = np.array(tab[surv_name])[idx0]

        # Null Gaussian sanity check (whitened-space)
        if args.null_gaussian:
            if not args.permute_whitened:
                raise RuntimeError("--null_gaussian requires --permute_whitened (it operates in whitened space).")

            nm = max(1, int(args.nmock))
            rng_null = np.random.default_rng(args.seed + 999)

            # quick uniformity check for TEX only (enough to validate the engine)
            pvals = np.empty(nm, float)
            dchi2s = np.empty(nm, float)

            print(f"\n[Null Gaussian sanity check] nmock={nm}  nperm={args.permute}  N={len(yw)}")

            for j in range(nm):
                ywm = rng_null.standard_normal(len(yw))  # whitened: N(0,1)
                dchi2_obs_j = chi2_from_Q(ywm, Q0) - chi2_from_Q(ywm, Qtex)

                if args.permute_within_survey:
                    p_emp_j, ge_j, _, _, _ = permute_pvalue_within_survey_whitened(
                        ywm, surveys, Q0, Qtex, dchi2_obs_j, args.permute,
                        seed=args.seed + 1000 + j, progress=False, topk=0
                    )
                else:
                    p_emp_j, ge_j, _, _, _ = permute_pvalue_whitened(
                        ywm, Q0, Qtex, dchi2_obs_j, args.permute,
                        seed=args.seed + 1000 + j, progress=False, topk=0
                    )

                pvals[j] = p_emp_j
                dchi2s[j] = dchi2_obs_j

                if (j + 1) % max(1, nm // 10) == 0 or (j + 1) == nm:
                    print(f"  mock {j+1:4d}/{nm} done")

            print("\n[Null Gaussian summary]")
            print(f"p_emp: mean={pvals.mean():.3f}  median={np.median(pvals):.3f}  min={pvals.min():.3e}  max={pvals.max():.3e}")
            print(f"frac(p<0.05)={np.mean(pvals < 0.05):.3f}  frac(p<0.01)={np.mean(pvals < 0.01):.3f}")
            print(f"dchi2_obs: mean={dchi2s.mean():.3f}  std={dchi2s.std(ddof=1):.3f}")
            return

        # ======= Run empirical p-values =======
        if args.permute_whitened:

            # Joint permutation run (same permutations => valid MAX look-elsewhere)
            Qdict = {"tex": Qtex, "kin": Qkin, "mix": Qmix}

            obs_dict = {
                "tex": dchi2_tex_obs,
                "kin": dchi2_kin_obs,
                "mix": dchi2_mix_obs,
                "dk":  dchi2_D_given_K_obs,  # gain(MIX over KIN) == chi2_kin - chi2_mix
                "kd":  dchi2_K_given_D_obs,  # gain(MIX over TEX) == chi2_tex - chi2_mix
            }
            # Request MAX explicitly (so the function tracks hits/max/top and makes _MAX plot)
            obs_dict["max"] = max(obs_dict[k] for k in ["tex", "kin", "mix", "dk", "kd"])

            plot_prefix = args.permute_plot if args.permute_plot else None

            if args.permute_within_survey:
                if surveys is None:
                    raise RuntimeError("Need surveys for within-survey permutations.")
                p, ge, maxv, gap, top = permute_pvalue_within_survey_whitened(
                    yw, surveys, Q0, Qdict, obs_dict, args.permute,
                    seed=args.seed, progress=True, plot_output=plot_prefix
                )
                print(f"\n[Permutation, within-survey, whitened] Nperm={args.permute}")
            else:
                p, ge, maxv, gap, top = permute_pvalue_whitened(
                    yw, Q0, Qdict, obs_dict, args.permute,
                    seed=args.seed, progress=True, plot_output=plot_prefix
                )
                print(f"\n[Permutation, global, whitened] Nperm={args.permute}")

            # Reporting (same spirit as before)
            print(f"  TEX: hits={ge['tex']:6d}  p_emp≈{p['tex']:.8e}   dchi2_obs={dchi2_tex_obs:.3f}  max={maxv['tex']:.3f}  gap={gap['tex']:.3f}")
            print(f"  KIN: hits={ge['kin']:6d}  p_emp≈{p['kin']:.8e}   dchi2_obs={dchi2_kin_obs:.3f}  max={maxv['kin']:.3f}  gap={gap['kin']:.3f}")
            print(f"  MIX: hits={ge['mix']:6d}  p_emp≈{p['mix']:.8e}   dchi2_obs={dchi2_mix_obs:.3f}  max={maxv['mix']:.3f}  gap={gap['mix']:.3f}")
            print(f"  D|K: hits={ge['dk']:6d}   p_emp≈{p['dk']:.8e}    dchi2_obs={dchi2_D_given_K_obs:.3f}  max={maxv['dk']:.3f}  gap={gap['dk']:.3f}")
            print(f"  K|D: hits={ge['kd']:6d}   p_emp≈{p['kd']:.8e}    dchi2_obs={dchi2_K_given_D_obs:.3f}  max={maxv['kd']:.3f}  gap={gap['kd']:.3f}")

            print(f"  MAX: hits={ge['max']:6d}  p_emp≈{p['max']:.8e}   dchi2_obs={obs_dict['max']:.3f}  max={maxv['max']:.3f}  gap={gap['max']:.3f}")

            print("  top TEX dchi2:", ", ".join([f"{v[0]:.4f}" for v in top['tex'][:8]]))
            print("  top KIN dchi2:", ", ".join([f"{v[0]:.4f}" for v in top['kin'][:8]]))
            print("  top MIX dchi2:", ", ".join([f"{v[0]:.4f}" for v in top['mix'][:8]]))
            print("  top MAX dchi2:", ", ".join([f"{v[0]:.4f}" for v in top['max'][:8]]))

            # --- Keep MW checks ---
            if args.mwcheck and (Qmw is not None) and (Qmwdip is not None):
                if args.permute_within_survey:
                    p_mw, ge_mw, max_mw, gap_mw, top_mw = permute_pvalue_within_survey_whitened(
                        yw, surveys, Q0, Qmw, dchi2_mw_obs, args.permute,
                        seed=args.seed + 131, progress=True
                    )
                    p_dip_g_mw, ge_dgmw, max_dgmw, gap_dgmw, top_dgmw = permute_pvalue_within_survey_whitened(
                        yw, surveys, Qmw, Qmwdip, dchi2_dip_given_mw_obs, args.permute,
                        seed=args.seed + 137, progress=True
                    )
                else:
                    p_mw, ge_mw, max_mw, gap_mw, top_mw = permute_pvalue_whitened(
                        yw, Q0, Qmw, dchi2_mw_obs, args.permute,
                        seed=args.seed + 131, progress=True
                    )
                    p_dip_g_mw, ge_dgmw, max_dgmw, gap_dgmw, top_dgmw = permute_pvalue_whitened(
                        yw, Qmw, Qmwdip, dchi2_dip_given_mw_obs, args.permute,
                        seed=args.seed + 137, progress=True
                    )

                print(f"   MW: hits={ge_mw:6d}  p_emp≈{p_mw:.8e}   dchi2_obs={dchi2_mw_obs:.3f}  max={max_mw:.3f}  gap={gap_mw:.3f}")
                print(f"dip|MW: hits={ge_dgmw:6d}  p_emp≈{p_dip_g_mw:.8e}   dchi2_obs={dchi2_dip_given_mw_obs:.3f}  max={max_dgmw:.3f}  gap={gap_dgmw:.3f}")

    
        else:
            # unwhitened permutation (legacy)
            if args.permute_within_survey:
                if surveys is None:
                    raise RuntimeError("Need surveys for within-survey permutations.")
                p_tex = permute_pvalue_within_survey(dmu, surveys, L, X0, X_tex, dchi2_tex_obs, args.permute, seed=args.seed)
                p_kin = permute_pvalue_within_survey(dmu, surveys, L, X0, X_kin, dchi2_kin_obs, args.permute, seed=args.seed + 11)
                p_mix = permute_pvalue_within_survey(dmu, surveys, L, X0, X_mix, dchi2_mix_obs, args.permute, seed=args.seed + 23)
                print(f"\n[Permutation, within-survey] Nperm={args.permute}")
            else:
                p_tex = permute_pvalue(dmu, L, X0, X_tex, dchi2_tex_obs, args.permute, seed=args.seed)
                p_kin = permute_pvalue(dmu, L, X0, X_kin, dchi2_kin_obs, args.permute, seed=args.seed + 11)
                p_mix = permute_pvalue(dmu, L, X0, X_mix, dchi2_mix_obs, args.permute, seed=args.seed + 23)
                print(f"\n[Permutation, global] Nperm={args.permute}")

            print(f"  TEX p_emp ≈ {p_tex:.3e}")
            print(f"  KIN p_emp ≈ {p_kin:.3e}")
            print(f"  MIX p_emp ≈ {p_mix:.3e}")

    # ======= jackknife by survey =======
    if args.jackknife:
        surv = r0.get("surveys", None)
        col = r0.get("surveycol", None)
        if surv is None or col is None:
            print("\nJackknife requested but no IDSURVEY-like column found. Use --surveycol IDSURVEY if it exists.")
            return
        uniq = np.unique(surv)
        print("\n=== Jackknife by survey (dipole-only Δχ², 3 dof) ===")
        print("exclude  N     DeltaChi2_tex   p_tex     A_mu(mag)   |dH0/H0|%   l_tex   b_tex   "
              "DeltaChi2_kin   p_kin     v_bulk(km/s)   l_kin   b_kin")
        for sid in uniq:
            keep = (surv != sid)
            idx = idx0[keep]
            r = fit_dipole(tab, idx, args, cov_full=cov_full)
            print(f"{int(sid):7d} {r['N']:4d} "
                  f"{r['dchi2_3']:14.2f} {r['p3']:1.3e} "
                  f"{r['A_mu']:10.6f} {r['frac']*100:10.3f} {r['l']:7.2f} {r['b']:7.2f}  "
                  f"{r.get('dchi2_kin', float('nan')):14.2f} {r.get('p3_kin', float('nan')):1.3e} "
                  f"{r.get('v_bulk', float('nan')):12.1f} {r.get('l_kin', float('nan')):7.2f} {r.get('b_kin', float('nan')):7.2f}")


if __name__ == "__main__":
    main()
    
