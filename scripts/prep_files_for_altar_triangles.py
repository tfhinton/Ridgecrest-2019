#!/usr/bin/env python3
"""Prepare AlTar inversion inputs for the Ridgecrest doublet -- CSI-free,
triangular-mesh version of ``prep_files_for_altar.py``.

Replaces the CSI stack (``csi.insar`` / ``csi.gps`` / ``csi.TriangularPatches`` /
``csi.multifaultsolve``) with the in-house ``codes`` package:

  * fault      -> ``codes.FaultTriangles`` (Meade 2007 TDE Green's functions on
                  the ``scripts/remeshFault.py`` mesh), the two Little Lake strands
                  merged into one fault;
  * InSAR      -> ``codes.InSAR``  (GRD read, spatial-covariance fit, distance-
                  based quad-tree downsampling, Cd, LOS Green's functions);
  * GNSS       -> ``codes.GNSS``   (offsets, diagonal Cd, EN Green's functions);
  * assembly   -> ``codes.InversionManager`` (joint d / G / Cd, per-track ramps,
                  HDF5 writers) -- ``d`` = ``data.h5``, ``Cd`` = ``covariance.h5``,
                  ``G`` = ``gf.h5``; plus per-fault patch areas.

Outputs (to ``results/<run>/inputs/`` by default):
  data.h5, covariance.h5, gf.h5, patch_areas.h5  + three diagnostic figures.

Run:  python scripts/prep_files_for_altar_triangles.py
"""
import os
import sys

sys.path.insert(0, "/Users/hintont/Dev/codes/src")

import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

from codes import FaultTriangles, InSAR, GNSS, InversionManager


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
MAIN_DIR   = "/Users/hintont/Dev/projects/ridgecrest"
FAULT_DIR  = os.path.join(MAIN_DIR, "data/fault")
INSAR_DIR  = os.path.join(MAIN_DIR, "data/insar")
GNSS_DIR   = os.path.join(MAIN_DIR, "data/gnss")

OUT_DIR    = os.path.join(MAIN_DIR, "results/tri01/inputs")
FIG_DIR    = os.path.join(MAIN_DIR, "results/tri01/figures")

UTM_ZONE   = 11

# Covariance estimation subsamples pixels at random; seed it so the AlTar inputs
# are reproducible and the occasional degenerate covariogram fit is stable.
SEED = 0

FAULT_NPZ = [
    "mainshock_fault_remesh.npz",
    "foreshock_fault_remesh.npz",
]

# Reference normal(s) fixing a consistent positive dip-slip sense per strand (see
# FaultTriangles.set_dip_convention). "auto" = each strand's mean normal (keeps
# every strand internally uniform); replace with a per-strand z-up dict/array to
# control the cross-strand sense. Does not affect strike-slip.
DIP_NORMALS = "auto"

# One entry per InSAR track.  bbox / cov_mask_out are geographic (lon in -180..180).
INSAR_TRACKS = {
    # "A064": dict(subdir="A064_20190704-0710"),
    "D071": dict(subdir="D071_20190704-0716"),
}
INSAR_BBOX     = (-118.1, -117.0, 35.3, 36.2)
COV_MASK_OUT   = [-117.88, -117.3, 35.53, 35.9]   # deforming region, excluded from covariance
COV_FRAC       = 0.005
COV_DISTMAX    = 25000.

# Downsampling (distance-based quad-tree; negative char_dist keeps near-fault dense)
DS_KW = dict(char_dist=-5000., expo_dist=1.1, min_size=500.,
             start_size=16000., scaler=0.7, reject_distance=1000.)

GNSS_FILE       = "unr_gps_offsets_full.txt"
GNSS_MAX_DIST   = 100000.
GNSS_SDE_SCALE  = 20.

# Diagnostic figure: which patch to illustrate the Green's function for.
DIAG_PATCH = None    # None -> auto-pick a mid-depth patch on strand 0


# --------------------------------------------------------------------------- #
#  Diagnostic figures
# --------------------------------------------------------------------------- #
def print_dip_sense(fault, offset=2500.):
    """One-line-per-strand check that positive dip-slip has a consistent sense.

    For every patch we impose +1 unit dip-slip and read the vertical (Up) surface
    displacement just off the fault; a strand where "+DS" is consistent shows all
    patches with the same sign (all up or all down).  A mixed split means the
    per-strand ``set_dip_convention`` reference needs choosing/flipping.  (Sign
    only, so it costs one throwaway GF eval; ``fault.gfs`` is recomputed later.)
    """
    lines = fault._component_lines()
    cen = fault.centroids[:, :2]
    perp = np.zeros((fault.n_patches, 2))
    for k, ln in enumerate(lines):
        ln = np.asarray(ln, dtype=float)
        s = ln[-1] - ln[0]
        s = s / np.linalg.norm(s)
        perp[fault.fault_ids == k] = [-s[1], s[0]]
    pts = np.vstack([cen + offset * perp, cen - offset * perp]).T  # (2, 2N)
    gds = fault.compute_greens_functions(pts).gfs[1]               # (N, 3, 2N)
    n = fault.n_patches
    up = np.sign([gds[i, 2, i] for i in range(n)])                 # Up at +perp side
    print("[dip-sense] +1 DS -> vertical motion just off the fault (want uniform):")
    for k, name in enumerate(fault.component_names):
        m = fault.fault_ids == k
        nup, ndn = int((up[m] > 0).sum()), int((up[m] < 0).sum())
        flag = "" if nup == 0 or ndn == 0 else "  <-- MIXED: set a per-strand normal"
        print(f"    strand {k} ({name}): up {nup:3d} / down {ndn:3d}{flag}")


def fig_covariance(insars, path):
    """InSAR spatial-covariance empirical covariogram + fitted model, per track."""
    n = len(insars)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4), squeeze=False,
                             layout="constrained")
    for ax, (name, sar) in zip(axes[0], insars.items()):
        sar.plot_covariance(ax=ax)
        ax.set_title(f"{name}: spatial covariance fit")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[fig] covariance fit           -> {path}")


def fig_downsampled_data(insars, gnss, fault, path):
    """Downsampled InSAR LOS (dots) + GNSS horizontal offsets (arrows) per track."""
    n = len(insars)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6), squeeze=False,
                             layout="constrained")
    for ax, (name, sar) in zip(axes[0], insars.items()):
        vlim = np.nanpercentile(np.abs(sar.vel), 98)
        sc = ax.scatter(sar.lon, sar.lat, c=sar.vel, cmap=cmc.vik, s=8,
                        vmin=-vlim, vmax=vlim, zorder=2)
        plt.colorbar(sc, ax=ax, label="LOS displacement (m)", shrink=0.8)
        # GNSS arrows (reuse the track's projection to draw the trace consistently)
        lon_range = float(sar.lon.max() - sar.lon.min())
        max_disp = float(np.max(np.hypot(gnss.de, gnss.dn)))
        scale = max_disp / (0.15 * lon_range) if max_disp > 0. else 1.
        ax.quiver(gnss.lon, gnss.lat, gnss.de, gnss.dn, angles="xy",
                  scale_units="xy", scale=scale, color="k", width=0.004, zorder=3)
        # reference arrow (lower-right)
        ref_m = 10. ** np.floor(np.log10(max(max_disp, 1e-9)))
        lat_range = float(sar.lat.max() - sar.lat.min())
        rx = sar.lon.max() - 0.30 * lon_range
        ry = sar.lat.min() + 0.06 * lat_range
        ax.quiver(rx, ry, ref_m, 0., angles="xy", scale_units="xy", scale=scale,
                  color="k", width=0.004, zorder=5)
        ax.text(rx + (ref_m / scale) / 2., ry - 0.02 * lat_range,
                f"{ref_m*1000:.0f} mm GNSS", ha="center", va="top", fontsize=8)
        for line in fault.trace.geometry:
            c = np.array(line.coords)
            lo, la = sar._proj(c[:, 0], c[:, 1], inverse=True)
            ax.plot(lo, la, "r-", lw=1.5, zorder=4)
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        ax.set_title(f"{name}: downsampled InSAR ({len(sar.vel)} px) + GNSS "
                     f"({len(gnss.de)} sta)")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[fig] downsampled InSAR + GNSS -> {path}")


def fig_patch_greens(insar, fault, ipatch, path):
    """Green's function for one patch: LOS displacement field + patch on the fault.

    Left: unit strike-slip LOS Green's function at the downsampled InSAR points,
    with the chosen triangle outlined on the surface trace.
    Right: the mesh in 3D with that triangle highlighted.
    """
    proj = insar._proj
    gf_ss_los = insar.gfs[0, ipatch]     # (n_pts,) LOS response to unit SS on this patch

    fig = plt.figure(figsize=(13, 6), layout="constrained")

    # --- (a) map: LOS GF + patch highlight ---
    ax = fig.add_subplot(1, 2, 1)
    vlim = np.nanpercentile(np.abs(gf_ss_los), 99) or 1e-6
    sc = ax.scatter(insar.lon, insar.lat, c=gf_ss_los, cmap=cmc.vik, s=8,
                    vmin=-vlim, vmax=vlim, zorder=2)
    plt.colorbar(sc, ax=ax, label="LOS disp. per unit strike-slip (m/m)", shrink=0.8)
    for line in fault.trace.geometry:
        c = np.array(line.coords)
        lo, la = proj(c[:, 0], c[:, 1], inverse=True)
        ax.plot(lo, la, "k-", lw=1.2, zorder=3)
    # highlight the patch (project its 3 vertices to lon/lat)
    tri = fault.triangle_xyz(ipatch)
    lo, la = proj(tri[:, 0], tri[:, 1], inverse=True)
    ax.fill(np.r_[lo, lo[0]], np.r_[la, la[0]], facecolor="none",
            edgecolor="red", lw=2.0, zorder=5)
    cen = fault.centroids[ipatch]
    clo, cla = proj(cen[0], cen[1], inverse=True)
    ax.plot(clo, cla, "r*", ms=12, zorder=6)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.set_title(f"Strike-slip GF, patch {ipatch} "
                 f"(depth {fault.depths[ipatch]/1000:.1f} km)")

    # --- (b) 3D mesh with patch highlighted ---
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    slips = np.zeros(fault.n_patches)
    slips[ipatch] = 1.0
    fault.slips = slips
    fault.plot_fault3d(ax=ax3, color_by="slip", cmap="Reds",
                       vmin=0., vmax=1., edgecolor="0.6", linewidth=0.2)
    ax3.set_title(f"patch {ipatch} on the mesh")
    fault.slips = np.zeros(fault.n_patches)

    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[fig] patch Green's function   -> {path}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- Fault: merge the two Little Lake strands into one TDE fault ----
    strands = [FaultTriangles.from_npz(os.path.join(FAULT_DIR, f))
               for f in FAULT_NPZ]
    fault = FaultTriangles.merge(strands, name="LLFZ")
    # Consistent dip-slip sense per strand.  meade07 derives DS from each
    # element's forced-up normal, which flips where a strand's dip azimuth
    # reverses along strike or where elements are near-vertical -- so "+DS" would
    # otherwise not mean one physical thing across a strand.  Strike-slip is
    # unaffected (still right-lateral-positive everywhere).  "auto" makes each
    # strand internally uniform via its mean normal; to nail the cross-strand
    # sense, pass explicit per-strand z-up normals instead, e.g.
    #   fault.set_dip_convention({"SNFA-LLFZ-EAST-...": (nx, ny, nz), "SNFA-...SOUT...": (...)})
    # (flip a strand's reference sign if its +DS ends up meaning the wrong sense).
    fault.set_dip_convention(DIP_NORMALS)
    print(f"[fault] {fault} | patches per strand: "
          f"{[len(v) for v in fault.patch_areas_by_fault().values()]}")
    print_dip_sense(fault)

    # ---- InSAR: per track -> covariance, downsample, Cd, LOS GFs ----
    insars = {}
    for name, cfg in INSAR_TRACKS.items():
        sub = os.path.join(INSAR_DIR, cfg["subdir"])
        sar = InSAR(unw_filepath=os.path.join(sub, "unwrapped.grd"),
                    los_filepaths=(os.path.join(sub, "east.grd"),
                                   os.path.join(sub, "north.grd"),
                                   os.path.join(sub, "up.grd")),
                    utm_zone=UTM_ZONE, bbox=INSAR_BBOX)
        sar = sar.compute_covariance(mask_box=COV_MASK_OUT, frac=COV_FRAC,
                                     distmax=COV_DISTMAX)
        sar = sar.downsample(fault, **DS_KW)
        sar = sar.build_Cd()
        sar = sar.compute_greens_functions(fault)
        insars[name] = sar

    # ---- GNSS: offsets -> diagonal Cd, EN GFs ----
    gnss = GNSS().import_data(filepath=os.path.join(GNSS_DIR, GNSS_FILE),
                              utm_zone=UTM_ZONE, fault=fault,
                              max_dist=GNSS_MAX_DIST)
    gnss = gnss.compute_covariance(sde_scale_factor=GNSS_SDE_SCALE)
    gnss = gnss.compute_greens_functions(fault)

    # ---- Assemble joint system (one ramp per InSAR track) ----
    datasets = {**insars, "gnss": gnss}
    inv = InversionManager().set_datasets(datasets,
                                          solve_for_ramp=list(insars.keys()))
    inv.summary()

    # ---- Write AlTar inputs ----
    inv.save_to_hdf5(OUT_DIR)                                   # data / covariance / gf
    fault.save_patch_areas(os.path.join(OUT_DIR, "patch_areas.h5"))
    print(f"[out] patch_areas.h5 -> {OUT_DIR}")

    # ---- Diagnostic figures ----
    fig_covariance(insars, os.path.join(FIG_DIR, "covariance_fit.png"))
    any_insar = next(iter(insars.values()))
    fig_downsampled_data(insars, gnss, fault,
                         os.path.join(FIG_DIR, "downsampled_data.png"))
    ipatch = DIAG_PATCH
    if ipatch is None:
        # a mid-depth patch on strand 0, for a clear lobed GF
        strand0 = np.flatnonzero(fault.fault_ids == 0)
        d = fault.depths[strand0]
        ipatch = int(strand0[np.argmin(np.abs(d - np.median(d)))])
    fig_patch_greens(any_insar, fault, ipatch,
                     os.path.join(FIG_DIR, "patch_greens_function.png"))

    print("\n[done] AlTar inputs + diagnostics written.")
    return inv, fault, insars, gnss


if __name__ == "__main__":
    main() 