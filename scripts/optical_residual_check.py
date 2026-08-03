#!/usr/bin/env python3
"""Optical residual check for the quick LSQ slip model.

The near-field optical (image-correlation) EW/NS displacements are NOT used in
the inversion; ``optical_residual_figure`` forward-predicts them from a slip
model and maps the residual (data - model), as an independent look at how well
the geodetic slip explains the near-field.  Called from
quick_lsq_far_field_inversion.py so the check runs every time the inversion does.
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

from codes import OpticalData
import config


DECIMATE = 500    # 1 m -> 1.3 km pixels; ~400 valid points (fast)


def _load_optical(decimate, verbose):
    """Decimated + flattened OpticalData, cached to skip the slow clear_empty /
    decimate on repeat runs (delete the cache if DECIMATE or the tifs change)."""
    cache = config.TMP_DIR / f"optical_decimated_x{decimate}.pickle"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    opt = OpticalData(verbose=verbose, ew_filepath=str(config.EW_TIF),
                      ns_filepath=str(config.NS_TIF))
    opt = opt.clear_empty(clear_zero=True).decimate(decimate).flatten()
    with open(cache, "wb") as fh:
        pickle.dump(opt, fh)
    return opt


def _plot_residuals(opt, fault, pred_ew, pred_ns, res_ew, res_ns, path):
    """2x3 map grid: data / model / residual for EW (top) and NS (bottom)."""
    trace = fault.get_surface_trace_xy()
    rows = [("EW", opt.ew_vals, pred_ew, res_ew),
            ("NS", opt.ns_vals, pred_ns, res_ns)]
    col_titles = ["data", "model", "residual (data - model)"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained",
                             sharex=True, sharey=True)
    for r, (comp, data, model, res) in enumerate(rows):
        vlim = float(np.nanpercentile(np.abs(data), 98)) or 1e-6
        for c, vals in enumerate((data, model, res)):
            ax = axes[r, c]
            sc = ax.scatter(opt.x / 1e3, opt.y / 1e3, c=vals, cmap=cmc.vik,
                            s=4, vmin=-vlim, vmax=vlim, rasterized=True)
            for line in trace.geometry:
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
    print(f"[fig] optical residuals -> {path}")


def optical_residual_figure(fault, ss, ds, out_path, decimate=DECIMATE,
                            verbose=True):
    """Predict the near-field optical from a slip model and map the residual.

    Loads (and caches) the decimated EW/NS optical, computes optical Green's
    functions on ``fault``, forward-predicts with the (ss, ds) slip vectors, and
    writes the data/model/residual figure to ``out_path``.
    """
    opt = _load_optical(decimate, verbose)
    print(f"[optical] {len(opt.x)} pixels after decimate x{decimate}")

    print("[optical] computing Green's functions...")
    opt = opt.compute_greens_functions(fault)

    pred_ew = ss @ opt.gfs_ew[0] + ds @ opt.gfs_ew[1]
    pred_ns = ss @ opt.gfs_ns[0] + ds @ opt.gfs_ns[1]
    res_ew  = opt.ew_vals - pred_ew
    res_ns  = opt.ns_vals - pred_ns

    rms = lambda a: float(np.sqrt(np.nanmean(a ** 2)))
    print(f"[EW] data RMS {rms(opt.ew_vals):.4f} -> residual RMS {rms(res_ew):.4f} m")
    print(f"[NS] data RMS {rms(opt.ns_vals):.4f} -> residual RMS {rms(res_ns):.4f} m")

    _plot_residuals(opt, fault, pred_ew, pred_ns, res_ew, res_ns, out_path)
    return opt
