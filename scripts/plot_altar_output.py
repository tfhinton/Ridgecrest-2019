#!/usr/bin/env python3
"""Visualise an AlTar Bayesian slip-inversion run (triangular-mesh setup).

All the generalisable machinery lives in ``codes.AltarOutput``; this script just
wires up the fault + assembled system (coords, G, d) and calls the plotting
methods.  Comment lines out to skip figures.  Figures land in ``<run>/figs``.

This targets the archived tri01 run (outputs_16000_12000), which was inverted on
the 330-patch remesh -- the current fault pickle / cached InversionManager are on
a newer 424-patch mesh, so we reconstruct the matching fault from the *_remesh.npz
files and load G/d + dataset coords from the run's own inputs.

For a *fresh* run on the current mesh, drop all of that and just pass the cached
manager instead:

    with open(config.INVERSION_PICKLE, "rb") as fh: inv = pickle.load(fh)
    with open(config.FAULT_PICKLE, "rb") as fh:      fault = pickle.load(fh)
    ao = AltarOutput(RUN, fault, inv=inv)
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import cmcrameri.cm as cmc

from codes import FaultTriangles, AltarOutput
import optical_residual_check as optres
import config


RUN = config.ALTAR_DIR / "outputs_16000_12000"
INPUTS = config.ALTAR_DIR / "inputs"

# ---- 330-patch fault (mainshock = strand 0, foreshock = strand 1) ----
def _load_remesh(path):
    z = np.load(path, allow_pickle=True)
    return FaultTriangles(z["vertices"], z["triangles"], layers=z["layers"],
                          name=str(z["name"]))

fault = FaultTriangles.merge([
    _load_remesh(config.FAULT_DIR / "mainshock_fault_remesh.npz"),
    _load_remesh(config.FAULT_DIR / "foreshock_fault_remesh.npz")])

# ---- assembled system + dataset coords, from the run's own inputs ----
with h5py.File(INPUTS / "gf.h5", "r") as fh:
    G = fh["gf"][:]
with h5py.File(INPUTS / "data.h5", "r") as fh:
    d = fh["data"][:]
    dslices = {k: slice(*fh["datasets"][k][:]) for k in fh["datasets"]}
z = np.load(config.ALTAR_DIR / "dataset_coords.npz", allow_pickle=True)
coords = {k: z[k].item() for k in z.files}

ao = AltarOutput(RUN, fault, G=G, d=d, coords=coords, dataset_slices=dslices,
                 ss_keys=("strikeslipmain", "strikeslipsecond"),
                 ds_keys=("dipslip",), ramp_key="ramp")
ao.check_ordering()

# ---- slip (posterior mode) ----
ao.plot_slip_2d("total")
ao.plot_slip_2d("strikeslip", vmax=6.)
ao.plot_slip_2d("dipslip")
ao.plot_slip_3d()

# ---- bivariate slip + uncertainty ----
# A2: sigma axis, mid-grey uncertain.  C1: coefficient-of-variation axis (sigma/slip).
ao.plot_slip_bivariate(valmax=6., sigmamax=2.,
                       title="Total slip and posterior std",
                       path=f"{ao.figs_dir}/slip_bivariate_sigma.png",
                        grey=(0.55, 0.55, 0.55),
                    #    cmap=cmc.lipari_r, grey=(0.35, 0.35, 0.35),
                       )
ao.plot_slip_bivariate(valmax=6., sigmamax=1., uncertainty="cv", grey=(0.55, 0.55, 0.55),
                       title="Total slip and relative uncertainty (sigma / slip)",
                       path=f"{ao.figs_dir}/slip_bivariate_cv.png")

# ---- per-patch posterior PDFs ----
for i in range(3):
    ao.plot_patch_pdfs(ao.random_patches(8), subtitle=f"random selection {i+1}",
                       path=f"{ao.figs_dir}/patch_pdfs_random_{i+1}.png")
ao.plot_patch_pdfs(ao.depth_transect(), subtitle="down-dip transect through slip max",
                   path=f"{ao.figs_dir}/patch_pdfs_depth_transect.png")
ao.plot_patch_pdfs(ao.top_slip_patches(), subtitle="8 highest-slip patches",
                   path=f"{ao.figs_dir}/patch_pdfs_top_slip.png")
ao.plot_ramp_pdfs()

# ---- data / model / residual maps ----
ao.plot_fit_maps()

# independent near-field check: the optical (image correlation) is NOT in the
# inversion; forward-predict it from the mode slip model and map the residual.
optres.optical_residual_figure(fault, ao.summ["ss_mode"], ao.summ["ds_mode"],
                               f"{ao.figs_dir}/fit_optical_check.png")

# ---- trade-offs / moment / resolution / annealing ----
ao.plot_correlation_matrix()
ao.plot_correlation_map()          # peak patch by default
ao.plot_moment_magnitude()
ao.plot_uncertainty_vs_depth()
ao.plot_beta_annealing()

# ---- annealing convergence video (slow; uncomment to build) ----
# ao.plot_convergence_video()

print(f"\n[done] figures in {ao.figs_dir}")
