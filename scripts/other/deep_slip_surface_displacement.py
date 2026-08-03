#!/usr/bin/env python3
"""Surface displacement due to *deep* slip in an AlTar posterior-mode model.

Loads the AlTar posterior (step_final.h5), takes the per-patch posterior-mode
slip model (same KDE-mode estimator as ``visualise_altar_triangles.py``), zeroes
every patch shallower than ``--min-depth`` (default 3 km, patch-centroid depth),
and forward-predicts the E / N / Up surface displacement on a square grid around
the fault with the same Meade (2007) TDE Green's functions used in the
inversion.  This isolates the part of the surface deformation field the deep
slip is responsible for -- the smooth, long-wavelength signal that survives
away from the fault trace.

Writes ``deep_slip_surface_displacement.png`` (and an .npz of the fields) into
``<output_dir>/figs``.

Run:  python scripts/deep_slip_surface_displacement.py results/tri01/outputs_40000_15000
"""
import argparse
import os
import sys

sys.path.insert(0, "/Users/hintont/Dev/codes/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

import scripts.prep_files_for_altar as prep
import scripts.plot_altar_output as viz
from codes import FaultTriangles


def build_fault():
    """Same merged LLFZ fault (and dip convention) as the AlTar prep."""
    strands = [FaultTriangles.from_npz(os.path.join(prep.FAULT_DIR, f))
               for f in prep.FAULT_NPZ]
    fault = FaultTriangles.merge(strands, name="LLFZ")
    fault.set_dip_convention(prep.DIP_NORMALS)
    return fault


def square_grid(fault, margin=20e3, n=101):
    """Square (n, n) grid of surface points centred on the fault (UTM m)."""
    xy = fault.vertices[:, :2]
    cx, cy = xy.mean(axis=0)
    half = 0.5 * float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))) + margin
    x = np.linspace(cx - half, cx + half, n)
    y = np.linspace(cy - half, cy + half, n)
    return np.meshgrid(x, y)


def deep_slip_displacement(fault, ss_mode, ds_mode, X, Y, min_depth):
    """E/N/U fields (m) from the mode slip restricted to depth > min_depth."""
    deep = fault.depths > min_depth
    ss = np.where(deep, ss_mode, 0.0)
    ds = np.where(deep, ds_mode, 0.0)
    print(f"[deep] {deep.sum()}/{fault.n_patches} patches below "
          f"{min_depth/1e3:.1f} km; |slip| there up to "
          f"{np.hypot(ss, ds).max():.2f} m")

    pts = np.vstack([X.ravel(), Y.ravel()])
    print(f"[gf  ] computing TDE Green's functions at {pts.shape[1]} points...")
    gfs = fault.compute_greens_functions(pts).gfs      # (2, n_patches, 3, n_pts)
    u = np.einsum("i,ikp->kp", ss, gfs[0]) + np.einsum("i,ikp->kp", ds, gfs[1])
    return (u[0].reshape(X.shape), u[1].reshape(X.shape),
            u[2].reshape(X.shape)), deep


def plot_fields(fault, X, Y, ue, un, uu, deep, min_depth, path):
    """E / N / Up panels + horizontal-displacement arrows on the Up panel."""
    Xk, Yk = X / 1e3, Y / 1e3
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), sharey=True,
                             layout="constrained")
    panels = [("East", ue), ("North", un), ("Up", uu)]
    vlim = max(float(np.abs(f).max()) for _, f in panels)
    for ax, (name, field) in zip(axes, panels):
        pc = ax.pcolormesh(Xk, Yk, field, cmap=cmc.vik, vmin=-vlim, vmax=vlim,
                           shading="auto", rasterized=True)
        for line in fault.trace.geometry:
            c = np.array(line.coords)
            ax.plot(c[:, 0] / 1e3, c[:, 1] / 1e3, "k-", lw=1.4)
        ax.set_aspect("equal")
        ax.set_title(f"{name} displacement")
        ax.set_xlabel("Easting (km)")
    # decimated horizontal arrows on the Up panel
    st = max(1, X.shape[0] // 18)
    axes[2].quiver(Xk[::st, ::st], Yk[::st, ::st],
                   ue[::st, ::st], un[::st, ::st],
                   angles="xy", color="k", width=0.004,
                   scale=8 * vlim, scale_units="inches")
    axes[0].set_ylabel("Northing (km)")
    fig.colorbar(pc, ax=axes, label="Displacement (m)", shrink=0.85)
    fig.suptitle(f"Surface displacement from deep slip only "
                 f"(posterior-mode model, patches deeper than "
                 f"{min_depth/1e3:.0f} km: {deep.sum()}/{fault.n_patches})",
                 fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig ] {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("output_dir", help="AlTar output directory (with step_final.h5)")
    p.add_argument("--min-depth", type=float, default=3000.,
                   help="keep slip only on patches deeper than this (m)")
    p.add_argument("--margin", type=float, default=20e3,
                   help="grid margin beyond the fault extent (m)")
    p.add_argument("--n-grid", type=int, default=101,
                   help="grid points per side")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    fault = build_fault()
    print(f"[fault] {fault}")

    ps, beta = viz.load_step(os.path.join(out_dir, "step_final.h5"))
    summ = viz.slip_summaries(ps, fault.n_patches, seed=args.seed)
    print(f"[post] mode |slip| max {summ['tot_mode'].max():.2f} m "
          f"(final beta = {beta:.4g})")

    X, Y = square_grid(fault, margin=args.margin, n=args.n_grid)
    (ue, un, uu), deep = deep_slip_displacement(
        fault, summ["ss_mode"], summ["ds_mode"], X, Y, args.min_depth)

    fig_path = os.path.join(figs_dir, "deep_slip_surface_displacement.png")
    plot_fields(fault, X, Y, ue, un, uu, deep, args.min_depth, fig_path)
    np.savez(os.path.join(figs_dir, "deep_slip_surface_displacement.npz"),
             X=X, Y=Y, ue=ue, un=un, uu=uu, deep_mask=deep,
             min_depth=args.min_depth)
    print("[done] deep-slip surface displacement complete.")


if __name__ == "__main__":
    main()
