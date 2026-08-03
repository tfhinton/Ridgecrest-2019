#!/usr/bin/env python3
"""Quick regularised least-squares slip inversion -- a sanity check for the
AlTar (Bayesian) triangular-mesh setup.

Loads the joint system assembled by prep_files_for_altar.py (the InversionManager
pickle + the fault) and solves it with a smoothed, zero-mean linear least
squares.  The point is not the slip model itself but to confirm the pipeline
produces a sensible fit before the expensive Bayesian run.

Regularisation
--------------
AlTar uses no smoothing (its prior does that job); a bare LSQ on this system is
under-determined, so we add a zero-mean Gaussian smoothing prior on slip, from
CSI Fault.buildCm (Radiguet et al. 2010):

    C_m(i, j) = (sigma * lam0 / lam)^2 * exp(-||c_i - c_j|| / lam)

on patch-centroid distances (km), applied independently to the strike-slip and
dip-slip blocks.  Ramp parameters get a loose diagonal prior.  With a zero prior
mean the regularised-LSQ estimate is

    m = (G^T Cd^-1 G + Cm^-1)^-1 G^T Cd^-1 d

Run:  python scripts/quick_lsq_far_field_inversion.py
"""
import pickle

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
from scipy.spatial.distance import cdist

import config
import optical_residual_check as optres


# ----- Config -----
# Smoothing prior (Radiguet Cm).  lam / lam0 are in KILOMETRES to match these
# tuned values; centroid distances are converted from metres accordingly.
CM_SIGMA = 1.317   # slip correlation amplitude (prior slip std ~ sigma*lam0/lam)
CM_LAM   = 5.0     # correlation length (km)
CM_LAM0  = 1.0     # normalising distance (km)
RAMP_VAR = 1.0e4   # loose diagonal prior variance for ramp params (m^2)

OUT_DIR = config.WORKING_DIR / "tmp/lsq"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----- Smoothing model covariance (CSI Fault.buildCm, Radiguet 2010) -----
def build_Cm_inv(fault, model_slices, n_params,
                 sigma=CM_SIGMA, lam=CM_LAM, lam0=CM_LAM0, ramp_var=RAMP_VAR):
    """Inverse prior covariance aligned with the InversionManager model vector.

    The strike-slip and dip-slip blocks each get the Radiguet exponential
    covariance on patch-centroid distances (km); ramp params get ``ramp_var`` on
    the diagonal.
    """
    ss = model_slices['fault_ss']
    ds = model_slices['fault_ds']
    n_patch = ss.stop - ss.start

    cen_km = np.asarray(fault.centroids, dtype=float) / 1.0e3
    Cmt    = (sigma * lam0 / lam) ** 2 * np.exp(-cdist(cen_km, cen_km) / lam)

    Cm_inv = np.zeros((n_params, n_params))
    Cmt_inv = scipy.linalg.inv(Cmt)                      # shared by SS and DS
    Cm_inv[ss, ss] = Cmt_inv
    Cm_inv[ds, ds] = Cmt_inv
    for j in range(2 * n_patch, n_params):               # ramp params
        Cm_inv[j, j] = 1.0 / ramp_var
    return Cm_inv


# ----- Regularised weighted least squares -----
def solve_regularised(inv, Cm_inv):
    """Zero-mean smoothed LSQ; writes m / m_std / residuals back onto ``inv`` so
    its summary/plot helpers work unchanged."""
    L   = scipy.linalg.cholesky(inv.Cd, lower=True)       # whiten data: Cd = L L^T
    G_w = scipy.linalg.solve_triangular(L, inv.G, lower=True)
    d_w = scipy.linalg.solve_triangular(L, inv.d, lower=True)

    A = G_w.T @ G_w + Cm_inv                              # G^T Cd^-1 G + Cm^-1
    b = G_w.T @ d_w                                       # G^T Cd^-1 d
    inv.m     = scipy.linalg.solve(A, b, assume_a='pos')
    inv.Cm    = scipy.linalg.inv(A)                       # posterior covariance
    inv.m_std = np.sqrt(np.diag(inv.Cm))

    predicted = inv.G @ inv.m
    inv.residuals = {label: inv.d[sl] - predicted[sl]
                     for label, sl in inv._dataset_slices.items()}


# ----- Main -----
# Load the exact AlTar system assembled by prep_files_for_altar.py.
print("[1/4] Loading assembled system + fault...")
with open(config.INVERSION_PICKLE, "rb") as fh:
    inv = pickle.load(fh)
with open(config.FAULT_PICKLE, "rb") as fh:
    fault = pickle.load(fh)

print(f"\n[2/4] Building smoothing prior Cm "
      f"(sigma={CM_SIGMA}, lam={CM_LAM} km, lam0={CM_LAM0} km)...")
Cm_inv = build_Cm_inv(fault, inv._model_slices, inv.G.shape[1])

print("[3/4] Solving regularised weighted least squares (m0 = 0)...")
solve_regularised(inv, Cm_inv)
inv.summary()

print("\n[4/4] Plotting + saving disposable result...")
ss  = inv.m[inv._model_slices['fault_ss']]
ds  = inv.m[inv._model_slices['fault_ds']]
mag = np.hypot(ss, ds)

# Slip model as 2D along-strike/depth sections (one subplot per strand).
for vals, label, cmap, fname in [
    (ss,  'Strike-slip (m)', cmc.vik,    'quick_lsq_slip_ss.png'),
    (ds,  'Dip-slip (m)',    cmc.vik,    'quick_lsq_slip_ds.png'),
    (mag, '|slip| (m)',      cmc.batlow, 'quick_lsq_slip_mag.png'),
]:
    vlim = np.nanpercentile(np.abs(vals), 99) or 1e-6
    vmin, vmax = (-vlim, vlim) if cmap is cmc.vik else (0.0, vlim)
    fig, _ = fault.plot_slip_2d(slip=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                                colorbar_label=label)
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"[fig] slip {label:<16} -> {OUT_DIR / fname}")

# Data vs post-fit residual map per InSAR track.
for track in config.INSAR_TRACKS:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    vlim = float(np.nanpercentile(np.abs(inv.datasets[track].vel), 98))
    inv.plot_map(ax=axes[0], raster_label=track, fault=fault, vlim=(-vlim, vlim),
                 colorbar_label='LOS (m)', title=f'{track}: data',
                 gnss_scale=1., gnss_arrow_length_m=0.1)
    inv.plot_residuals(ax=axes[1], raster_label=track, fault=fault,
                       vlim=(-vlim, vlim), title=f'{track}: residual',
                       gnss_scale=1., gnss_arrow_length_m=0.1)
    fig.savefig(OUT_DIR / f"quick_lsq_fit_{track}.png", dpi=150)
    plt.close(fig)
    print(f"[fig] fit {track} -> {OUT_DIR / f'quick_lsq_fit_{track}.png'}")

# Independent check: near-field optical residual (NOT in the inversion).
print("\n[optical] forward-predicting near-field optical residual...")
optres.optical_residual_figure(fault, ss, ds,
                               OUT_DIR / "quick_lsq_optical_residuals.png")

# Save the disposable result.
npz_path = OUT_DIR / "quick_lsq_result.npz"
np.savez(npz_path,
         m=inv.m, m_std=inv.m_std, predicted=inv.G @ inv.m,
         model_slice_names=np.array(list(inv._model_slices)),
         model_slice_bounds=np.array([[s.start, s.stop]
                                      for s in inv._model_slices.values()]),
         **{f"residual_{k}": v for k, v in inv.residuals.items()})
print(f"[out] result -> {npz_path}")
print("\n[done] Quick LSQ sanity check complete.")
