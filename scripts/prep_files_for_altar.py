#!/usr/bin/env python3
"""Prepare AlTar inversion inputs for the Ridgecrest doublet (triangular mesh).

Writes data.h5, covariance.h5, gf.h5 and patch_areas.h5 (plus diagnostic
figures) to OUT_DIR from the pre-merged fault, the InSAR tracks and the GNSS
offsets.
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt

from codes import InSAR, GNSS, InversionManager
import config


# ----- Config -----
OUT_DIR = config.WORKING_DIR / "inputs"           # AlTar .h5 inputs
FIG_DIR = config.TMP_DIR / "insar_preprocessing"  # diagnostic figures
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0   # covariance subsamples pixels at random; seed for reproducible inputs

INSAR_BBOX   = (-118.1, -117.0, 35.3, 36.2)
COV_MASK_OUT = [-117.88, -117.3, 35.53, 35.9]   # deforming region, out of covariance
COV_FRAC     = 0.005
COV_DISTMAX  = 25000.

# Downsampling (distance-based quad-tree; negative char_dist keeps near-fault dense)
DS_KW = dict(char_dist=-5000., expo_dist=1.1, min_size=500.,
             start_size=16000., scaler=0.7, reject_distance=1000.)

GNSS_FILE      = "unr_gps_offsets_full.txt"
GNSS_MAX_DIST  = 100000.
GNSS_SDE_SCALE = 20. # scale supplied GNSS standard deviations for more realistic covariance


# ----- Main -----
np.random.seed(SEED)

# Fault: pre-merged Little Lake strands.  The positive dip-slip sense is fixed by
# the "auto" convention inside compute_greens_functions (per-strand mean normal).
with open(config.FAULT_PICKLE, "rb") as fh:
    fault = pickle.load(fh)
print(f"[fault] {fault}")

# InSAR: per track -> covariance, downsample, Cd, LOS GFs.
insars = {}
for name, dir in config.INSAR_TRACKS.items():
    sar = InSAR(unw_filepath=str(dir / "unwrapped.grd"),
                los_filepaths=(str(dir / "east.grd"),
                               str(dir / "north.grd"),
                               str(dir / "up.grd")),
                utm_zone=config.UTM_ZONE, bbox=INSAR_BBOX)
    sar = sar.compute_covariance(mask_box=COV_MASK_OUT, frac=COV_FRAC,
                                 distmax=COV_DISTMAX)
    sar = sar.downsample(fault, **DS_KW)
    sar = sar.build_Cd()
    sar = sar.compute_greens_functions(fault)
    insars[name] = sar

# GNSS: offsets -> diagonal Cd, EN GFs.
gnss = GNSS().import_data(filepath=str(config.GNSS_DIR / GNSS_FILE),
                          utm_zone=config.UTM_ZONE, fault=fault,
                          max_dist=GNSS_MAX_DIST)
gnss = gnss.compute_covariance(sde_scale_factor=GNSS_SDE_SCALE)
gnss = gnss.compute_greens_functions(fault)

# Assemble joint system (one ramp per InSAR track).
inv = InversionManager().set_datasets({**insars, "gnss": gnss},
                                      solve_for_ramp=list(insars.keys()))
inv.summary()

# Write AlTar inputs, plus a pickle of the assembled system for the quick LSQ.
inv.save_to_hdf5(OUT_DIR)
fault.save_patch_areas(OUT_DIR / "patch_areas.h5")
with open(config.INVERSION_PICKLE, "wb") as fh:
    pickle.dump(inv, fh)

# Diagnostic figures.
for name, sar in insars.items():
    fig, ax = sar.plot_covariance()
    ax.set_title(f"{name}: spatial covariance fit")
    fig.savefig(FIG_DIR / f"covariance_{name}.png", dpi=150)
    plt.close(fig)

    fig, ax = sar.plot(fault=fault,
                       title=f"{name}: downsampled InSAR ({len(sar.vel)} px)")
    fig.savefig(FIG_DIR / f"downsampled_{name}.png", dpi=150)
    plt.close(fig)

print(f"\n[done] AlTar inputs -> {OUT_DIR}, diagnostics -> {FIG_DIR}")
