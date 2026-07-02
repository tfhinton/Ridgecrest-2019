#!/usr/bin/env python3
"""Quick-and-dirty regularised least-squares slip inversion -- a sanity check
for the AlTar (Bayesian) triangular-mesh setup.

Rebuilds *exactly* the joint system that ``prep_files_for_altar_triangles.py``
hands to AlTar (same fault, InSAR/GNSS Green's functions, Cd and per-track
ramps), then solves it with a smoothed, zero-mean linear least squares.  The
point is not the slip model itself but to confirm the pipeline produces a
sensible fit before the (expensive) Bayesian run.

Regularisation
--------------
AlTar uses no smoothing (the prior does that job); a bare LSQ on this system is
under-determined, so here we add a zero-mean Gaussian *smoothing* prior on slip,
copied from CSI ``Fault.buildCm`` (Radiguet et al. 2010):

    C_m(i, j) = (sigma * lam0 / lam)^2 * exp(-||c_i - c_j|| / lam)

on the patch centroid distances (km), applied independently to the strike-slip
and dip-slip blocks.  Ramp parameters get a loose diagonal prior (effectively
unconstrained).  With a zero prior mean the MAP / regularised-LSQ estimate is

    m = (G^T Cd^-1 G + Cm^-1)^-1 G^T Cd^-1 d          (initial slip guess = 0)

Outputs (to a temporary scratch dir -- this model is disposable):
  quick_lsq_result.npz              m, m_std, predicted, per-dataset residuals
  quick_lsq_slip.png                SS / DS / |slip| on the mesh
  quick_lsq_fit_<track>.png         data vs residual map per InSAR track

Run:  python scripts/quick_lsq_inversion_triangles.py
"""
import os
import sys

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
from scipy.spatial.distance import cdist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import prep_files_for_altar_triangles as prep     # noqa: E402  (path insert first)
import optical_residual_check as optres           # noqa: E402  (independent optical check)


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
# Smoothing prior (Radiguet Cm).  lam / lam0 are in KILOMETRES to match these
# tuned values; centroid distances are converted from metres accordingly.
CM_SIGMA = 1.317      # amplitude of the slip correlation (prior slip std ~ sigma*lam0/lam)
CM_LAM   = 5.0        # correlation length (km)
CM_LAM0  = 1.0        # normalising distance (km)

# Loose diagonal prior variance for ramp parameters (m^2) -- large => ~unconstrained.
RAMP_VAR = 1.0e4

# Where to drop the disposable result + figures.
OUT_DIR = ("/Users/hintont/Dev/projects/ridgecrest/results/inversion_results/lsq01")


# --------------------------------------------------------------------------- #
#  Smoothing model covariance (CSI Fault.buildCm, Radiguet 2010)
# --------------------------------------------------------------------------- #
def build_Cm(fault, model_slices, n_params,
             sigma=CM_SIGMA, lam=CM_LAM, lam0=CM_LAM0, ramp_var=RAMP_VAR):
    """Block-diagonal prior covariance aligned with the InversionManager model.

    Strike-slip and dip-slip blocks each get the Radiguet exponential
    covariance on patch-centroid distances; ramp parameters get ``ramp_var`` on
    the diagonal.  Returns ``(Cm, Cm_inv)`` (both ``(n_params, n_params)``).
    """
    ss = model_slices['fault_ss']
    ds = model_slices['fault_ds']
    n_patch = ss.stop - ss.start

    # Centroid distances in km (centroids are UTM metres; sign of z is irrelevant
    # to pairwise distance, so the z-up frame is fine).
    cen_km = np.asarray(fault.centroids, dtype=float) / 1.0e3
    dist   = cdist(cen_km, cen_km)                       # (n_patch, n_patch), km

    C   = (sigma * lam0 / lam) ** 2
    Cmt = C * np.exp(-dist / lam)                        # one slip-direction block

    Cm     = np.zeros((n_params, n_params))
    Cm_inv = np.zeros((n_params, n_params))
    Cmt_inv = scipy.linalg.inv(Cmt)                      # shared by SS and DS
    for sl in (ss, ds):
        Cm[sl, sl]     = Cmt
        Cm_inv[sl, sl] = Cmt_inv

    # Ramp (and any other) parameters: loose independent prior.
    for j in range(2 * n_patch, n_params):
        Cm[j, j]     = ramp_var
        Cm_inv[j, j] = 1.0 / ramp_var

    return Cm, Cm_inv


# --------------------------------------------------------------------------- #
#  Regularised weighted least squares
# --------------------------------------------------------------------------- #
def solve_regularised(inv, Cm_inv):
    """Zero-mean smoothed LSQ on the assembled system; writes m/m_std/residuals
    back onto ``inv`` so its summary/plot helpers work unchanged."""
    G, d, Cd = inv.G, inv.d, inv.Cd

    # Cholesky-whiten the data (Cd = L L^T) as InversionManager.run() does.
    L   = scipy.linalg.cholesky(Cd, lower=True)
    G_w = scipy.linalg.solve_triangular(L, G, lower=True)
    d_w = scipy.linalg.solve_triangular(L, d, lower=True)

    A = G_w.T @ G_w + Cm_inv                             # G^T Cd^-1 G + Cm^-1
    b = G_w.T @ d_w                                       # G^T Cd^-1 d
    m = scipy.linalg.solve(A, b, assume_a='pos')

    inv.Cm    = scipy.linalg.inv(A)                      # posterior covariance
    inv.m     = m
    inv.m_std = np.sqrt(np.diag(inv.Cm))

    predicted = G @ m
    inv.residuals = {}
    for label, sl in inv._dataset_slices.items():
        inv.residuals[label] = d[sl] - predicted[sl]

    return predicted


# --------------------------------------------------------------------------- #
#  Plots
# --------------------------------------------------------------------------- #
def plot_slip(fault, m, model_slices, path):
    """SS / DS / |slip| coloured on the 3D mesh."""
    ss = m[model_slices['fault_ss']]
    ds = m[model_slices['fault_ds']]
    mag = np.hypot(ss, ds)

    fig = plt.figure(figsize=(18, 6), layout='constrained')
    panels = [
        ("Strike-slip (m)", ss,  cmc.vik,   'diverging'),
        ("Dip-slip (m)",    ds,  cmc.vik,   'diverging'),
        ("|slip| (m)",      mag, cmc.batlow, 'seq'),
    ]
    for k, (title, vals, cmap, kind) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, k, projection='3d')
        if kind == 'diverging':
            vlim = np.nanpercentile(np.abs(vals), 99) or 1e-6
            vmin, vmax = -vlim, vlim
        else:
            vmin, vmax = 0.0, (np.nanpercentile(vals, 99) or 1e-6)
        fault.slips = vals
        fault.plot_fault3d(ax=ax, color_by='slip', cmap=cmap,
                           vmin=vmin, vmax=vmax, edgecolor='0.6', linewidth=0.15)
        ax.set_title(title)
    fault.slips = np.zeros(fault.n_patches)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[fig] slip model              -> {path}")


def plot_fit(inv, fault, track, path, gnss_scale=1.):
    """Data (left) vs post-fit residual (right) map for one InSAR track."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    # Shared colour limits from the data so data/residual are comparable.
    vlim = float(np.nanpercentile(np.abs(inv.datasets[track].vel), 98))
    inv.plot_map(ax=axes[0], raster_label=track, fault=fault,
                 vlim=(-vlim, vlim), colorbar_label='LOS (m)',
                 title=f'{track}: data', gnss_scale=gnss_scale, gnss_arrow_length_m=0.1)
    inv.plot_residuals(ax=axes[1], raster_label=track, fault=fault,
                       vlim=(-vlim, vlim), title=f'{track}: residual', gnss_scale=gnss_scale, gnss_arrow_length_m=0.1)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[fig] fit {track}                -> {path}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Rebuild the exact AlTar system (fault, GFs, Cd, ramps).
    print("[1/4] Assembling joint system via prep_files_for_altar_triangles...")
    inv, fault, insars, gnss = prep.main()

    print("\n[2/4] Building smoothing prior Cm "
          f"(sigma={CM_SIGMA}, lam={CM_LAM} km, lam0={CM_LAM0} km)...")
    n_params = inv.G.shape[1]
    _, Cm_inv = build_Cm(fault, inv._model_slices, n_params)

    print("[3/4] Solving regularised weighted least squares (m0 = 0)...")
    solve_regularised(inv, Cm_inv)
    inv.summary()

    print("\n[4/4] Plotting + saving disposable result...")
    plot_slip(fault, inv.m, inv._model_slices,
              os.path.join(OUT_DIR, "quick_lsq_slip.png"))
    # 2D along-strike/depth section of |slip|, one labelled subplot per strand.
    mag = np.hypot(inv.m[inv._model_slices['fault_ss']],
                   inv.m[inv._model_slices['fault_ds']])
    fig2d, _ = fault.plot_slip_2d(slip=mag, cmap=cmc.batlow,
                                  colorbar_label='|slip| (m)')
    path2d = os.path.join(OUT_DIR, "quick_lsq_slip_2d.png")
    fig2d.savefig(path2d, dpi=150)
    plt.close(fig2d)
    print(f"[fig] slip model (2D section) -> {path2d}")
    for track in insars:
        plot_fit(inv, fault, track,
                 os.path.join(OUT_DIR, f"quick_lsq_fit_{track}.png"))

    # Independent check: residual on the near-field optical (NOT in the inversion).
    print("\n[optical] forward-predicting near-field optical residual...")
    optres.optical_residual_figure(
        fault, inv.m[inv._model_slices['fault_ss']],
        inv.m[inv._model_slices['fault_ds']],
        os.path.join(OUT_DIR, "quick_lsq_optical_residuals.png"))

    npz_path = os.path.join(OUT_DIR, "quick_lsq_result.npz")
    np.savez(npz_path,
             m=inv.m, m_std=inv.m_std, predicted=inv.G @ inv.m,
             model_slice_names=np.array(list(inv._model_slices)),
             model_slice_bounds=np.array([[s.start, s.stop]
                                          for s in inv._model_slices.values()]),
             **{f"residual_{k}": v for k, v in inv.residuals.items()})
    print(f"[out] result                  -> {npz_path}")
    print("\n[done] Quick LSQ sanity check complete.")
    return inv, fault


if __name__ == "__main__":
    main()
