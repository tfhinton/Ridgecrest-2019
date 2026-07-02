#!/usr/bin/env python3
"""Optical residual check for the quick LSQ slip model.

The near-field optical (image-correlation) EW/NS displacements are NOT used in
the inversion; this script just forward-predicts them from the LSQ slip model
and maps the residual (data - model), as an independent look at how well the
geodetic slip explains the near-field.

Steps:
  * rebuild the merged LLFZ fault (same mesh as the inversion);
  * load the LSQ slip model m from results/inversion_results/lsq01/quick_lsq_result.npz;
  * load the 1 m EW/NS optical, decimate x100, flatten to pixels;
  * compute optical EW/NS Green's functions and predict displacement;
  * plot data / model / residual for EW and NS (shared colour scale per row).

Run:  python scripts/optical_residual_check.py
"""
import os
import sys

sys.path.insert(0, "/Users/hintont/Dev/codes/src")

import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

from codes import FaultTriangles, OpticalData


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
MAIN_DIR   = "/Users/hintont/Dev/projects/ridgecrest"
FAULT_DIR  = os.path.join(MAIN_DIR, "data/fault")
OPTICAL_DIR = os.path.join(MAIN_DIR, "data/optical")

LSQ_NPZ = os.path.join(MAIN_DIR, "results/inversion_results/lsq01/quick_lsq_result.npz")
OUT_DIR = os.path.join(MAIN_DIR, "results/inversion_results/lsq01")

FAULT_NPZ = [
    "SNFA-LLFZ-EAST-Eastern_Little_Lake_main_fault-CFM5_remesh.npz",
    "SNFA-LLFZ-SOUT-Southern_Little_Lake_main_fault-CFM5_remesh.npz",
]

EW_TIF = os.path.join(OPTICAL_DIR, "EW_Ridgecrest_1m_utm_detrended.tif")
NS_TIF = os.path.join(OPTICAL_DIR, "NS_Ridgecrest_1m_utm_detrended.tif")

DECIMATE = 500    # 1 m -> 1.3 km pixels; ~400 valid points (a few hundred, fast)


def load_slip(npz_path):
    """Return (ss, ds) slip arrays from a quick_lsq_result.npz."""
    d = np.load(npz_path, allow_pickle=True)
    m = d["m"]
    sl = {n: slice(int(a), int(b))
          for n, (a, b) in zip(d["model_slice_names"], d["model_slice_bounds"])}
    return m[sl["fault_ss"]], m[sl["fault_ds"]]


def predict(gfs, ss, ds):
    """Forward displacement from a (2, n_patches, n_pts) GF block: SS + DS."""
    return ss @ gfs[0] + ds @ gfs[1]        # (n_pts,)


def plot_residuals(opt, fault, res_ew, res_ns, pred_ew, pred_ns, path):
    """2x3 map grid: data / model / residual for EW (top) and NS (bottom)."""
    rows = [
        ("EW", opt.ew_vals, pred_ew, res_ew),
        ("NS", opt.ns_vals, pred_ns, res_ns),
    ]
    col_titles = ["data", "model", "residual (data - model)"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained",
                             sharex=True, sharey=True)
    for r, (comp, data, model, res) in enumerate(rows):
        vlim = float(np.nanpercentile(np.abs(data), 98)) or 1e-6
        for c, vals in enumerate((data, model, res)):
            ax = axes[r, c]
            sc = ax.scatter(opt.x / 1e3, opt.y / 1e3, c=vals, cmap=cmc.vik,
                            s=4, vmin=-vlim, vmax=vlim, rasterized=True)
            for line in fault.trace.geometry:
                xy = np.array(line.coords)
                ax.plot(xy[:, 0] / 1e3, xy[:, 1] / 1e3, "k-", lw=1.2)
            ax.set_aspect("equal")
            ax.set_title(f"{comp} {col_titles[c]}")
            if r == 1:
                ax.set_xlabel("Easting (km)")
            if c == 0:
                ax.set_ylabel(f"{comp}\nNorthing (km)")
        fig.colorbar(sc, ax=axes[r, :], label=f"{comp} displacement (m)",
                     shrink=0.7)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[fig] optical residuals       -> {path}")


def optical_residual_figure(fault, ss, ds, out_path, decimate=DECIMATE,
                            ew_tif=EW_TIF, ns_tif=NS_TIF, verbose=True):
    """Predict the near-field optical from a slip model and map the residual.

    Loads the EW/NS optical, decimates + flattens, computes optical Green's
    functions on ``fault``, forward-predicts with the (ss, ds) slip vectors, and
    writes the data/model/residual figure to ``out_path``.  Returns the
    (decimated) OpticalData object.  Callable both standalone and from the quick
    LSQ inversion so the same check runs every time the inversion does.
    """
    opt = OpticalData(verbose=verbose, ew_filepath=ew_tif, ns_filepath=ns_tif)
    opt = opt.clear_nan(clear_zero=True).decimate(decimate).flatten()
    print(f"[optical] {len(opt.x)} pixels after decimate x{decimate}")

    print("[optical] computing Green's functions...")
    opt = opt.compute_greens_functions(fault)

    pred_ew = predict(opt.gfs_ew, ss, ds)
    pred_ns = predict(opt.gfs_ns, ss, ds)
    res_ew  = opt.ew_vals - pred_ew
    res_ns  = opt.ns_vals - pred_ns

    def rms(a):
        return float(np.sqrt(np.nanmean(a ** 2)))
    print(f"[EW] data RMS {rms(opt.ew_vals):.4f} -> residual RMS {rms(res_ew):.4f} m")
    print(f"[NS] data RMS {rms(opt.ns_vals):.4f} -> residual RMS {rms(res_ns):.4f} m")

    plot_residuals(opt, fault, res_ew, res_ns, pred_ew, pred_ns, out_path)
    return opt


def main():
    # ---- Fault (same merged mesh the inversion used) ----
    strands = [FaultTriangles.from_npz(os.path.join(FAULT_DIR, f))
               for f in FAULT_NPZ]
    fault = FaultTriangles.merge(strands, name="LLFZ")
    print(f"[fault] {fault}")

    # ---- LSQ slip model ----
    ss, ds = load_slip(LSQ_NPZ)
    print(f"[slip] SS max {np.abs(ss).max():.2f} m, DS max {np.abs(ds).max():.2f} m")

    optical_residual_figure(fault, ss, ds,
                            os.path.join(OUT_DIR, "optical_residuals.png"))
    print("\n[done] Optical residual check complete.")


if __name__ == "__main__":
    main()
