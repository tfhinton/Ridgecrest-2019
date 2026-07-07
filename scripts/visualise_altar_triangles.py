#!/usr/bin/env python3
"""Visualise AlTar Bayesian slip-inversion results (triangular-mesh setup).

Runs on an AlTar output directory (the one holding ``step_*.h5`` /
``step_final.h5``, e.g. ``results/tri01/outputs_16000_12000``), creates a
``figs/`` subdirectory inside it, and writes:

  * ``slip2d_{total,strikeslip,dipslip}.png``  -- 2D fault sections coloured by
    the per-patch posterior mode (max of a Gaussian KDE on each marginal);
  * ``slip3d_total.png``                       -- 3D mesh coloured by total slip;
  * ``slip_bivariate.png``                     -- bivariate slip + standard
    deviation with the wedge colorbar (after ``utils.plotSlipBivariate``);
  * ``patch_pdfs_random_<i>.png``              -- posterior PDFs (SS solid,
    DS dashed) for 8 random patches, highlighted on the fault above;
  * ``patch_pdfs_depth_transect.png``          -- same, for a down-dip column of
    patches through the slip maximum (resolution loss with depth);
  * ``patch_pdfs_top_slip.png``                -- same, for the 8 highest-slip patches;
  * ``fit_<dataset>.png``                      -- data / model / residual maps for
    every dataset (InSAR scatter, GNSS arrows, optical if present), where the
    model is G @ m with m the per-parameter posterior mode;
  * ``fit_optical_check.png``                  -- independent near-field check:
    the optical EW/NS data (NOT in the inversion) forward-predicted from the
    mode slip model (via ``optical_residual_check``);
  * ``ramp_pdfs.png``                          -- posteriors of the ramp parameters;
  * ``convergence_bivariate.mp4``              -- one bivariate frame per
    ``step_*.h5``, showing the annealing converge;
  * ``posterior_correlation_matrix.png``       -- full parameter correlation matrix;
  * ``correlation_map_patch<i>.png``           -- correlation of the peak patch's
    SS with every other parameter, painted on the fault;
  * ``moment_magnitude.png``                   -- posterior of Mw (total + per strand);
  * ``uncertainty_vs_depth.png``               -- posterior sigma vs patch depth;
  * ``beta_annealing.png``                     -- annealing schedule from
    BetaStatistics.txt (if present).

The AlTar model vector is assumed to be the ParameterSets concatenated as
``[strikeslipmain, strikeslipsecond, dipslip, ramp]``, matching the G columns
written by ``prep_files_for_altar_triangles.py`` (``fault_ss`` in merged patch
order, then ``fault_ds``, then ramps).  This is verified at runtime: the script
aborts if the posterior-mean prediction does not beat the data RMS.

Data coordinates are not stored in the AlTar inputs, so the InSAR downsampling
is re-run (seeded exactly as in the prep script -- RNG is consumed only by
``compute_covariance``) and verified against ``data.h5``; the result is cached
in ``dataset_coords.npz`` next to the inputs.

Run:   16000_12000
"""
import argparse
import os
import sys

sys.path.insert(0, "/Users/hintont/Dev/codes/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
import cmcrameri.cm as cmc

import prep_files_for_altar_triangles as prep
import optical_residual_check as optres
from codes import FaultTriangles, InSAR, GNSS

MU = 30e9          # shear modulus for moment (Pa)
KDE_MAX_SAMPLES = 4000
KDE_GRID = 256

# White -> red -> dark slip palette (the group's usual coseismic-slip colormap).
_COLORSCO = [(250, 250, 250), (255, 247, 236), (254, 232, 200), (253, 212, 158),
             (253, 187, 132), (252, 141, 89), (239, 101, 72), (215, 48, 31),
             (179, 0, 0), (127, 0, 0)]
SLIP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "cptslip", [(r / 255., g / 255., b / 255.) for r, g, b in _COLORSCO], N=256)

# Bivariate slip/uncertainty palette, base->top, left->right (from utils.py).
BIVCOLORS = [(208, 208, 208),
             (232, 232, 232), (164, 128, 128),
             (250, 250, 250), (237, 204, 187), (214, 137, 127), (149, 75, 75),
             (254, 248, 241), (254, 227, 190), (253, 173, 119), (248, 130, 84),
             (233, 87, 61), (210, 41, 27), (173, 0, 0), (127, 0, 0)]
BIVCOLORS_RGBA = [(r / 255., g / 255., b / 255.) for r, g, b in BIVCOLORS]


# --------------------------------------------------------------------------- #
#  Posterior loading & estimators
# --------------------------------------------------------------------------- #
def load_step(path):
    """Read one AlTar step file -> dict of ParameterSets arrays (+ beta)."""
    with h5py.File(path, "r") as fh:
        ps = {k: fh["ParameterSets"][k][:] for k in fh["ParameterSets"]}
        beta = float(fh["Annealer"]["beta"][()]) if "Annealer" in fh else np.nan
    return ps, beta


def model_matrix(ps):
    """(n_samples, n_params) samples ordered like the G columns:
    [strikeslipmain, strikeslipsecond, dipslip, ramp]."""
    return np.hstack([ps["strikeslipmain"], ps["strikeslipsecond"],
                      ps["dipslip"], ps["ramp"]])


def kde_mode(samples, seed=0):
    """Per-column posterior mode from a Gaussian KDE on each 1-D marginal.

    samples : (n_samples, n_params).  KDE is fit on a random subsample of
    KDE_MAX_SAMPLES (mode location converges long before that) and evaluated
    on a KDE_GRID-point grid spanning the sample range.
    """
    rng = np.random.default_rng(seed)
    n, p = samples.shape
    idx = rng.choice(n, size=min(n, KDE_MAX_SAMPLES), replace=False)
    out = np.empty(p)
    for j in range(p):
        x = samples[idx, j]
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-12:
            out[j] = lo
            continue
        grid = np.linspace(lo, hi, KDE_GRID)
        out[j] = grid[np.argmax(gaussian_kde(x)(grid))]
    return out


def slip_summaries(ps, n_patches, seed=0):
    """Posterior summaries per patch: modes and stds of SS, DS, |total|."""
    ss = np.hstack([ps["strikeslipmain"], ps["strikeslipsecond"]])
    ds = ps["dipslip"]
    tot = np.hypot(ss, ds)
    assert ss.shape[1] == n_patches, (ss.shape, n_patches)
    return dict(ss_mode=kde_mode(ss, seed), ds_mode=kde_mode(ds, seed),
                tot_mode=kde_mode(tot, seed),
                ss_std=ss.std(axis=0), ds_std=ds.std(axis=0),
                tot_std=tot.std(axis=0),
                ss=ss, ds=ds, tot=tot)


# --------------------------------------------------------------------------- #
#  Fault geometry -> 2D sections (shared by all along-strike/depth plots)
# --------------------------------------------------------------------------- #
def fault_sections(fault):
    """Per-strand projection onto the trace-defined vertical section (km).

    Returns a list (one entry per strand) of dicts:
      ids    : global patch indices (merged order) in this strand,
      polys  : (n_k, 3, 2) triangle vertices as (along-strike, depth) in km,
      depth  : (n_k,) centroid depths (km),
      width  : along-strike extent (km),
      name   : strand name.
    Same projection as FaultTriangles.plot_slip_2d.
    """
    lines = fault._component_lines()
    tri_xyz = fault.vertices[fault.triangles]
    sections = []
    for k, name in enumerate(fault.component_names):
        mask = fault.fault_ids == k
        line = np.asarray(lines[k], dtype=float)
        origin, strike = line[0], line[-1] - line[0]
        strike = strike / np.linalg.norm(strike)
        v = tri_xyz[mask]
        s = (v[:, :, :2] - origin) @ strike
        depth = -v[:, :, 2]
        sections.append(dict(ids=np.flatnonzero(mask),
                             polys=np.stack([s, depth], axis=-1) / 1e3,
                             depth=depth.mean(axis=1) / 1e3,
                             width=float(s.max() - s.min()) / 1e3,
                             name=str(name)))
    return sections


def draw_sections(axes, sections, facecolors=None, values=None, cmap=None,
                  norm=None, edgecolor="0.6", linewidth=0.15):
    """Draw each strand's triangles into its axes; returns the PolyCollections.

    Colour either by explicit per-patch RGBA (``facecolors``, merged order) or
    by ``values`` (merged order) through cmap/norm.  Deepest triangles are drawn
    first so shallow ones sit on top where dip projection overlaps.
    """
    colls = []
    for ax, sec in zip(axes, sections):
        order = np.argsort(-sec["depth"])
        polys = list(sec["polys"][order])
        gids = sec["ids"][order]
        if facecolors is not None:
            coll = PolyCollection(polys, facecolors=[facecolors[i] for i in gids],
                                  edgecolors=edgecolor, linewidths=linewidth)
        else:
            coll = PolyCollection(polys, array=np.asarray(values)[gids],
                                  cmap=cmap, norm=norm, edgecolors=edgecolor,
                                  linewidths=linewidth)
        ax.add_collection(coll)
        ax.autoscale_view()
        ax.set_aspect("equal")
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_xlabel("Along strike (km)")
        ax.set_title(sec["name"], fontsize=10)
        colls.append(coll)
    axes[0].set_ylabel("Depth (km)")
    return colls


def section_axes(fig, sections, subplot_spec=None):
    """A row of per-strand axes with widths proportional to strand length."""
    import matplotlib.gridspec as gridspec
    widths = [sec["width"] for sec in sections]
    n = len(sections)
    if subplot_spec is None:
        gs = fig.add_gridspec(1, n, width_ratios=widths)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=subplot_spec,
                                              width_ratios=widths)
    axes = [fig.add_subplot(gs[0, 0])]
    for i in range(1, n):
        axes.append(fig.add_subplot(gs[0, i], sharey=axes[0]))
        axes[-1].tick_params(labelleft=False)
    return axes


# --------------------------------------------------------------------------- #
#  Figure 1: 2D (and 3D) slip coloured by posterior mode
# --------------------------------------------------------------------------- #
def fig_slip2d(fault, values, label, cmap, path, vmin=None, vmax=None):
    fig, _ = fault.plot_slip_2d(slip=values, cmap=cmap, vmin=vmin, vmax=vmax,
                                colorbar_label=label)
    fig.suptitle("Posterior mode (per-patch KDE maximum)", fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)}")


def fig_slip3d(fault, values, path):
    fault.slips = values
    fig, ax = fault.plot_fault3d(color_by="slip", cmap=SLIP_CMAP, vmin=0.,
                                 vmax=float(values.max()), edgecolor="0.4",
                                 linewidth=0.15)
    ax.set_title("Total slip, posterior mode")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    fault.slips = np.zeros(fault.n_patches)
    print(f"[fig] {os.path.basename(path)}")


# --------------------------------------------------------------------------- #
#  Figure 2 (+5): bivariate slip / uncertainty with the wedge colorbar
# --------------------------------------------------------------------------- #
def bivariate_colval(slip, sigma, valmax, sigmamax):
    """Map (slip, sigma) to the 15 bivariate categories (utils.ColValSup)."""
    sigma1, sigma2 = sigmamax / 2., 3. * sigmamax / 4.
    colval = np.zeros(np.shape(slip))
    colval[sigma >= sigmamax] = 0
    colval[(slip < valmax / 2.) & (sigma >= sigma2) & (sigma < sigmamax)] = 1
    colval[(slip >= valmax / 2.) & (sigma >= sigma2) & (sigma < sigmamax)] = 2
    cat = 2
    for s in np.arange(0, valmax, valmax / 4.):
        cat += 1
        colval[(sigma < sigma2) & (sigma >= sigma1)
               & (slip >= s) & (slip < s + valmax / 4.)] = cat
    colval[(sigma < sigma2) & (sigma >= sigma1) & (slip >= valmax)] = 6
    cat = 6
    for s in np.arange(0, valmax, valmax / 8.):
        cat += 1
        colval[(sigma < sigma1) & (slip >= s) & (slip < s + valmax / 8.)] = cat
    colval[(sigma < sigma1) & (slip >= valmax)] = cat
    return colval


def _fmt(v):
    return f"{v:.2g}"


def draw_wedge_legend(ax, valmax, sigmamax, slip_label="Slip (m)",
                      sigma_label="Standard deviation (m)"):
    """The quarter-'colour wheel' bivariate legend, drawn in ``ax`` (0..1 data
    coords, equal aspect).  Radius = slip (finer bins at low sigma), angular
    rings = sigma.  Adapted from utils.plotSlipBivariate, kept wedge-for-wedge."""
    ax.set_axis_off()
    ax.set_aspect("equal")
    center, L = (0.5, 0.02), 0.70
    wdgs = []
    kw = dict(ec="white", lw=0.3)
    # ring 1 (sigma >= sigmamax)
    wdgs.append(mpatches.Wedge(center, L / 4, 60, 120, width=None,
                               fc=BIVCOLORS_RGBA[0], **kw))
    # ring 2
    wdgs.append(mpatches.Wedge(center, 2 * L / 4, 90, 120, width=L / 4,
                               fc=BIVCOLORS_RGBA[1], **kw))
    wdgs.append(mpatches.Wedge(center, 2 * L / 4, 60, 90, width=L / 4,
                               fc=BIVCOLORS_RGBA[2], **kw))
    # ring 3 (4 slip bins)
    for i, (a0, a1) in enumerate([(60, 75), (75, 90), (90, 105), (105, 120)]):
        wdgs.append(mpatches.Wedge(center, 3 * L / 4, a0, a1, width=L / 4,
                                   fc=BIVCOLORS_RGBA[6 - i], **kw))
    # ring 4 (8 slip bins)
    angles = np.linspace(60, 120, 9)
    for i in range(8):
        wdgs.append(mpatches.Wedge(center, L, angles[i], angles[i + 1],
                                   width=L / 4, fc=BIVCOLORS_RGBA[14 - i], **kw))
    for w in wdgs:
        ax.add_patch(w)
    # labels: slip along the outer arc, sigma down the right radial edge
    coords = [wdgs[i].get_path().vertices[6] for i in [14, 12, 10, 8]]
    coords += [wdgs[i].get_path().vertices[0] for i in [7, 3, 2, 0]]
    labels = ([_fmt(v) for v in np.arange(0, valmax, valmax / 4.)]
              + [_fmt(valmax), _fmt(sigmamax / 2.), _fmt(3 * sigmamax / 4.),
                 _fmt(sigmamax)])
    rots = [30, 15, 0, -15, -30, 60, 60, 60]
    vas = ["center"] * 5 + ["top"] * 3
    offs = [[0, 0.05 * L]] * 5 + [[0.05 * L, 0]] * 3
    for (x, y), lab, rot, va, off in zip(coords, labels, rots, vas, offs):
        ax.text(x + off[0], y + off[1], lab, rotation=rot,
                rotation_mode="anchor", ha="center", va=va, fontsize=8)
    ax.text(center[0], center[1] + 1.18 * L, slip_label, ha="center",
            va="center", fontsize=9)
    ax.text(center[0] + 0.42 * L, center[1] + 0.30 * L, sigma_label,
            rotation=60, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _bivariate_scaffold(fault, sections, valmax, sigmamax):
    """Figure with per-strand section axes + wedge legend; returns collections.

    The figure is sized from the fault extents (equal-aspect sections) so the
    fault panels fill the canvas, with a fixed-width column for the wedge.
    """
    widths = [sec["width"] for sec in sections]
    dmax = max(float(sec["polys"][:, :, 1].max()) for sec in sections)
    in_per_km = 0.15
    wedge_w = 2.6
    fig = plt.figure(figsize=(in_per_km * sum(widths) + wedge_w + 1.0,
                              in_per_km * dmax + 1.7),
                     layout="constrained")
    gs = fig.add_gridspec(1, len(sections) + 1,
                          width_ratios=[in_per_km * w for w in widths]
                                       + [wedge_w])
    axes = [fig.add_subplot(gs[0, 0])]
    for i in range(1, len(sections)):
        axes.append(fig.add_subplot(gs[0, i], sharey=axes[0]))
        axes[-1].tick_params(labelleft=False)
    grey = [BIVCOLORS_RGBA[0]] * fault.n_patches
    colls = draw_sections(axes, sections, facecolors=grey, edgecolor="white",
                          linewidth=0.25)
    draw_wedge_legend(fig.add_subplot(gs[0, -1]), valmax, sigmamax)
    return fig, axes, colls


def _set_bivariate_colors(colls, sections, slip, sigma, valmax, sigmamax):
    cmap = mcolors.ListedColormap(BIVCOLORS_RGBA)
    norm = mcolors.Normalize(0, 15)
    colval = bivariate_colval(np.asarray(slip), np.asarray(sigma),
                              valmax, sigmamax)
    for coll, sec in zip(colls, sections):
        order = np.argsort(-sec["depth"])
        coll.set_facecolor(cmap(norm(colval[sec["ids"][order]])))


def fig_slip_bivariate(fault, sections, slip, sigma, valmax, sigmamax, path,
                       title="Total slip and posterior uncertainty"):
    fig, axes, colls = _bivariate_scaffold(fault, sections, valmax, sigmamax)
    _set_bivariate_colors(colls, sections, slip, sigma, valmax, sigmamax)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)}")


def video_convergence_bivariate(fault, sections, step_files, valmax, sigmamax,
                                path, fps=2, seed=0):
    """One bivariate frame per AlTar step file (posterior mode + std of |slip|)."""
    frames = []
    for f in step_files:
        ps, beta = load_step(f)
        tot = np.hypot(np.hstack([ps["strikeslipmain"], ps["strikeslipsecond"]]),
                       ps["dipslip"])
        frames.append((os.path.splitext(os.path.basename(f))[0],
                       beta, kde_mode(tot, seed), tot.std(axis=0)))
        print(f"  [video] processed {frames[-1][0]} (beta={beta:.4g})")

    fig, axes, colls = _bivariate_scaffold(fault, sections, valmax, sigmamax)
    label = fig.text(0.02, 0.97, "", ha="left", va="top", fontsize=11)

    def update(i):
        name, beta, slip, sigma = frames[i]
        _set_bivariate_colors(colls, sections, slip, sigma, valmax, sigmamax)
        label.set_text(f"{name}   " + (f"beta = {beta:.3g}" if np.isfinite(beta)
                                       else ""))
        return colls

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(path, writer="ffmpeg", fps=fps, dpi=180)
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)} ({len(frames)} frames)")


# --------------------------------------------------------------------------- #
#  Figure 3: posterior PDFs for selected patches, highlighted on the fault
# --------------------------------------------------------------------------- #
def fig_patch_pdfs(fault, sections, summ, patch_ids, path, subtitle):
    """Top: fault sections (total-slip mode, grey scale) with the selected
    patches filled in qualitative colours.  Bottom: |strike-slip| posterior
    KDEs (peak-normalised) for those patches, in matching colours."""
    patch_ids = list(patch_ids)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:len(patch_ids)]

    fig = plt.figure(figsize=(11, 7))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.32)
    top_axes = section_axes(fig, sections, subplot_spec=outer[0])

    norm = mcolors.Normalize(0., max(float(summ["tot_mode"].max()), 1e-6))
    draw_sections(top_axes, sections, values=summ["tot_mode"],
                  cmap=plt.get_cmap("Greys"), norm=norm, edgecolor="0.75",
                  linewidth=0.15)
    # overlay the selected triangles
    for ax, sec in zip(top_axes, sections):
        lookup = {g: i for i, g in enumerate(sec["ids"])}
        for c, pid in zip(colors, patch_ids):
            if pid in lookup:
                poly = sec["polys"][lookup[pid]]
                ax.fill(poly[:, 0], poly[:, 1], facecolor=c, edgecolor="k",
                        linewidth=0.8, zorder=5)

    ax_pdf = fig.add_subplot(outer[1])
    for c, pid in zip(colors, patch_ids):
        x = np.abs(summ["ss"][:, pid])
        if x.max() - x.min() < 1e-12:
            continue
        grid = np.linspace(x.min(), x.max(), KDE_GRID)
        dens = gaussian_kde(x[:KDE_MAX_SAMPLES])(grid)
        dens = dens / dens.max()
        ax_pdf.plot(grid, dens, color=c, lw=1.6, label=f"patch {pid}")
        ax_pdf.fill_between(grid, dens, color=c, alpha=0.06)
    ax_pdf.set_ylim(0., 1.1)
    ax_pdf.set_xlabel("|Strike-slip| (m)")
    ax_pdf.set_ylabel("Normalised posterior density")
    ax_pdf.legend(fontsize=8, ncol=8, frameon=False, loc="lower center",
                  bbox_to_anchor=(0.5, 1.01))
    for spine in ("top", "right"):
        ax_pdf.spines[spine].set_visible(False)
    fig.suptitle(f"Per-patch posterior PDFs -- {subtitle}", fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)}")


def pick_depth_transect(fault, sections, summ, n=8):
    """Down-dip column of patches through the along-strike slip maximum."""
    peak = int(np.argmax(summ["tot_mode"]))
    strand = int(fault.fault_ids[peak])
    sec = sections[strand]
    local_peak = int(np.flatnonzero(sec["ids"] == peak)[0])
    s_cen = sec["polys"][:, :, 0].mean(axis=1)
    near = np.argsort(np.abs(s_cen - s_cen[local_peak]))
    # walk outward along strike, keeping one patch per depth band
    chosen, depths_taken = [], []
    for j in near:
        d = sec["depth"][j]
        if all(abs(d - t) > 0.8 for t in depths_taken):
            chosen.append(j)
            depths_taken.append(d)
        if len(chosen) == n:
            break
    chosen = sorted(chosen, key=lambda j: sec["depth"][j])
    return [int(sec["ids"][j]) for j in chosen]


# --------------------------------------------------------------------------- #
#  Figure 4: data / model / residual maps
# --------------------------------------------------------------------------- #
def rebuild_dataset_coords(fault, inputs_dir, cache_path):
    """Coordinates + data values per dataset, in data-vector order.

    Rebuilds the prep-script datasets (seeded; RNG is consumed only by
    ``InSAR.compute_covariance``, so stopping after ``downsample`` reproduces
    the same points) and verifies the data vector against ``data.h5``.
    Cached to ``cache_path`` as an .npz.
    """
    with h5py.File(os.path.join(inputs_dir, "data.h5"), "r") as fh:
        d = fh["data"][:]
        ds_slices = {k: slice(*fh["datasets"][k][:]) for k in fh["datasets"]}

    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        coords = {k: z[k].item() for k in z.files}
        ok = all(np.allclose(coords[lbl]["d"], d[sl])
                 for lbl, sl in ds_slices.items() if lbl in coords)
        if ok and set(coords) == set(ds_slices):
            print(f"[coords] cache hit -> {cache_path}")
            return coords
        print("[coords] cache stale, rebuilding")

    np.random.seed(prep.SEED)
    coords = {}
    for name, cfg in prep.INSAR_TRACKS.items():
        sub = os.path.join(prep.INSAR_DIR, cfg["subdir"])
        sar = InSAR(unw_filepath=os.path.join(sub, "unwrapped.grd"),
                    los_filepaths=(os.path.join(sub, "east.grd"),
                                   os.path.join(sub, "north.grd"),
                                   os.path.join(sub, "up.grd")),
                    utm_zone=prep.UTM_ZONE, bbox=prep.INSAR_BBOX)
        sar = sar.compute_covariance(mask_box=prep.COV_MASK_OUT,
                                     frac=prep.COV_FRAC,
                                     distmax=prep.COV_DISTMAX)
        sar = sar.downsample(fault, **prep.DS_KW)
        coords[name] = dict(type="insar", lon=sar.lon, lat=sar.lat, d=sar.vel)

    gnss = GNSS().import_data(filepath=os.path.join(prep.GNSS_DIR, prep.GNSS_FILE),
                              utm_zone=prep.UTM_ZONE, fault=fault,
                              max_dist=prep.GNSS_MAX_DIST)
    d_gnss = np.empty(2 * len(gnss.de))
    d_gnss[0::2], d_gnss[1::2] = gnss.de, gnss.dn
    coords["gnss"] = dict(type="gnss", lon=gnss.lon, lat=gnss.lat, d=d_gnss)

    # fault trace in lon/lat for map overlays (use the InSAR track's projection)
    proj = sar._proj
    trace = []
    for line in fault.trace.geometry:
        c = np.array(line.coords)
        lo, la = proj(c[:, 0], c[:, 1], inverse=True)
        trace.append(np.column_stack([lo, la]))
    for v in coords.values():
        v["trace"] = trace

    for lbl, sl in ds_slices.items():
        if not np.allclose(coords[lbl]["d"], d[sl]):
            raise RuntimeError(
                f"rebuilt dataset '{lbl}' does not match data.h5 -- the prep "
                f"config has changed since the AlTar inputs were written")
    print("[coords] rebuilt datasets match data.h5")
    np.savez(cache_path, **{k: np.array(v, dtype=object)
                            for k, v in coords.items()})
    return coords


def fig_fit_insar(label, c, d_obs, d_mod, path):
    resid = d_obs - d_mod
    vlim = float(np.nanpercentile(np.abs(d_obs), 98))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True,
                             layout="constrained")
    for ax, vals, title in zip(axes, (d_obs, d_mod, resid),
                               ("Data", "Model (posterior mode)", "Residual")):
        sc = ax.scatter(c["lon"], c["lat"], c=vals, cmap=cmc.vik, s=8,
                        vmin=-vlim, vmax=vlim, zorder=2)
        for t in c["trace"]:
            ax.plot(t[:, 0], t[:, 1], "k-", lw=1.2, zorder=3)
        rms = float(np.sqrt(np.mean(vals**2))) if title == "Residual" else None
        ax.set_title(title + (f"  (RMS {rms*100:.1f} cm)" if rms is not None
                              else ""))
        ax.set_xlabel("Longitude (°)")
        ax.set_aspect(1. / np.cos(np.radians(float(np.mean(c["lat"])))))
    axes[0].set_ylabel("Latitude (°)")
    fig.colorbar(sc, ax=axes, label="LOS displacement (m)", shrink=0.75)
    fig.suptitle(f"{label}: data vs. posterior-mode prediction", fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)}")


def fig_fit_gnss(label, c, d_obs, d_mod, path):
    """Left: data (black) vs model (red) arrows.  Right: residual arrows (scaled up)."""
    de_o, dn_o = d_obs[0::2], d_obs[1::2]
    de_m, dn_m = d_mod[0::2], d_mod[1::2]
    de_r, dn_r = de_o - de_m, dn_o - dn_m

    lon, lat = c["lon"], c["lat"]
    lon_range = float(lon.max() - lon.min()) or 1.
    max_disp = float(np.max(np.hypot(de_o, dn_o)))
    scale = max_disp / (0.15 * lon_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True,
                             layout="constrained")
    for ax in axes:
        for t in c["trace"]:
            ax.plot(t[:, 0], t[:, 1], "r-", lw=1.2, zorder=2)
        ax.set_xlabel("Longitude (°)")
        ax.set_aspect(1. / np.cos(np.radians(float(np.mean(lat)))))
        ax.locator_params(axis="x", nbins=6)
    axes[0].set_ylabel("Latitude (°)")

    q1 = axes[0].quiver(lon, lat, de_o, dn_o, angles="xy", scale_units="xy",
                        scale=scale, color="k", width=0.004, zorder=3)
    axes[0].quiver(lon, lat, de_m, dn_m, angles="xy", scale_units="xy",
                   scale=scale, color="crimson", width=0.0025, zorder=4)
    ref = 10. ** np.floor(np.log10(max(max_disp, 1e-9)))
    axes[0].quiverkey(q1, 0.85, 0.06, ref, f"{ref*1000:.0f} mm", labelpos="S",
                      coordinates="axes", fontproperties=dict(size=8))
    axes[0].set_title("Data (black) vs model (red)")

    q2 = axes[1].quiver(lon, lat, de_r, dn_r, angles="xy", scale_units="xy",
                        scale=scale, color="k", width=0.004, zorder=3)
    axes[1].quiverkey(q2, 0.85, 0.06, ref, f"{ref*1000:.0f} mm",
                      labelpos="S", coordinates="axes",
                      fontproperties=dict(size=8))
    rms = float(np.sqrt(np.mean(np.r_[de_r, dn_r]**2)))
    axes[1].set_title(f"Residuals, same scale  (RMS {rms*1000:.1f} mm)")
    fig.suptitle(f"{label}: horizontal offsets vs. posterior-mode prediction",
                 fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)}")


def fig_fit_optical(label, c, d_obs, d_mod, path):
    """Optical EW/NS: 2x3 (component x data/model/residual) scatter maps."""
    N = len(c["lon"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True,
                             layout="constrained")
    for row, comp in enumerate(("EW", "NS")):
        obs = d_obs[row * N:(row + 1) * N]
        mod = d_mod[row * N:(row + 1) * N]
        vlim = float(np.nanpercentile(np.abs(obs), 98))
        for ax, vals, title in zip(axes[row], (obs, mod, obs - mod),
                                   ("Data", "Model", "Residual")):
            sc = ax.scatter(c["lon"], c["lat"], c=vals, cmap=cmc.vik, s=8,
                            vmin=-vlim, vmax=vlim, zorder=2)
            for t in c["trace"]:
                ax.plot(t[:, 0], t[:, 1], "k-", lw=1.2, zorder=3)
            if row == 0:
                ax.set_title(title)
        fig.colorbar(sc, ax=axes[row], label=f"{comp} displacement (m)",
                     shrink=0.8)
    fig.suptitle(f"{label}: data vs. posterior-mode prediction", fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)}")


def fig_ramp_pdfs(ps, path):
    """Posterior KDEs of the ramp parameters (basis [1, x_norm, y_norm] per
    track), one subplot each -- a quick visual check they are resolved."""
    ramp = ps["ramp"]
    names = ["offset (m)", "x gradient (m/norm)", "y gradient (m/norm)"]
    n = ramp.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.2), layout="constrained")
    for j, ax in enumerate(np.atleast_1d(axes)):
        x = ramp[:, j]
        grid = np.linspace(x.min(), x.max(), KDE_GRID)
        dens = gaussian_kde(x[:KDE_MAX_SAMPLES])(grid)
        ax.plot(grid, dens, lw=1.6, color="#3266ad")
        ax.fill_between(grid, dens, color="#3266ad", alpha=0.08)
        ax.axvline(np.median(x), color="crimson", lw=1., ls="--",
                   label=f"median {np.median(x):.3g}")
        ax.set_xlabel(names[j % 3] if n % 3 == 0 else f"ramp[{j}]")
        ax.legend(frameon=False, fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Posterior density")
    fig.suptitle("Ramp parameter posteriors", fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)}")


# --------------------------------------------------------------------------- #
#  Figure 6+: trade-offs, moment, annealing
# --------------------------------------------------------------------------- #
def fig_correlation_matrix(samples, blocks, path):
    """Posterior correlation matrix with parameter-block separators.

    blocks : list of (name, start, stop) in model-vector order.
    """
    corr = np.corrcoef(samples.T)
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    im = ax.imshow(corr, cmap=cmc.vik, vmin=-1, vmax=1, interpolation="nearest")
    for _, start, stop in blocks:
        for edge in (start, stop):
            ax.axhline(edge - 0.5, color="k", lw=0.6)
            ax.axvline(edge - 0.5, color="k", lw=0.6)
    centers = [(start + stop) / 2 for _, start, stop in blocks]
    names = [name for name, _, _ in blocks]
    ax.set_xticks(centers, names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(centers, names, fontsize=9)
    fig.colorbar(im, ax=ax, label="Posterior correlation", shrink=0.8)
    ax.set_title("Parameter trade-offs: posterior correlation matrix")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)}")


def fig_correlation_map(fault, sections, summ, ref_patch, ref_comp, path):
    """Correlation of one patch's slip with every patch's SS and DS, painted on
    the fault -- the spatial trade-off structure around the reference patch."""
    ref = (summ["ss"] if ref_comp == "SS" else summ["ds"])[:, ref_patch]
    ref = (ref - ref.mean()) / (ref.std() or 1.)

    def corr_with(arr):
        a = (arr - arr.mean(axis=0)) / np.where(arr.std(axis=0) > 0,
                                                arr.std(axis=0), 1.)
        return (a * ref[:, None]).mean(axis=0)

    corr_ss = corr_with(summ["ss"])
    corr_ds = corr_with(summ["ds"])
    off_ref = np.r_[np.delete(corr_ss, ref_patch), corr_ds]
    vmax = float(np.ceil(np.abs(off_ref).max() * 10.) / 10.)

    fig = plt.figure(figsize=(11, 6))
    outer = fig.add_gridspec(2, 1, hspace=0.35)
    norm = mcolors.Normalize(-vmax, vmax)
    for row, (comp, vals) in enumerate((("strike-slip", corr_ss),
                                        ("dip-slip", corr_ds))):
        axes = section_axes(fig, sections, subplot_spec=outer[row])
        draw_sections(axes, sections, values=np.clip(vals, -vmax, vmax),
                      cmap=cmc.vik, norm=norm, edgecolor="0.7", linewidth=0.1)
        axes[0].set_ylabel(f"Depth (km)\n[vs {comp}]")
        # mark the reference patch
        for ax, sec in zip(axes, sections):
            hit = np.flatnonzero(sec["ids"] == ref_patch)
            if hit.size:
                poly = sec["polys"][hit[0]]
                ax.fill(poly[:, 0], poly[:, 1], facecolor="none",
                        edgecolor="limegreen", linewidth=1.6, zorder=6)
    sm = ScalarMappable(cmap=cmc.vik, norm=norm)
    fig.colorbar(sm, ax=fig.axes, label="Posterior correlation", shrink=0.8)
    fig.suptitle(f"Trade-offs with patch {ref_patch} {ref_comp} "
                 f"(green outline)", fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)}")


def fig_moment_magnitude(fault, summ, areas, path):
    """Posterior of moment magnitude, total and per strand (mu = MU)."""
    m0_strand = {}
    for k, name in enumerate(fault.component_names):
        m = fault.fault_ids == k
        m0_strand[name] = MU * (summ["tot"][:, m] * areas[m]).sum(axis=1)
    m0_tot = sum(m0_strand.values())

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for name, m0 in {**m0_strand, "combined": m0_tot}.items():
        mw = (np.log10(m0) - 9.1) * 2. / 3.
        grid = np.linspace(mw.min(), mw.max(), KDE_GRID)
        dens = gaussian_kde(mw[:KDE_MAX_SAMPLES])(grid)
        lw = 2.2 if name == "combined" else 1.4
        ax.plot(grid, dens, lw=lw, label=f"{name} (Mw {np.median(mw):.2f})")
        ax.fill_between(grid, dens, alpha=0.08)
    ax.set_xlabel("Moment magnitude Mw")
    ax.set_ylabel("Posterior density")
    ax.legend(frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(f"Posterior moment magnitude (mu = {MU/1e9:.0f} GPa)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)}")


def fig_uncertainty_vs_depth(fault, summ, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5), layout="constrained")
    depth = fault.depths / 1e3
    for k, name in enumerate(fault.component_names):
        m = fault.fault_ids == k
        ax.scatter(depth[m], summ["tot_std"][m], s=14, alpha=0.7, label=name)
    ax.set_xlabel("Patch depth (km)")
    ax.set_ylabel("Posterior std of |slip| (m)")
    ax.legend(frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Loss of resolution with depth")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)}")


def fig_beta_annealing(stats_path, path):
    rows = np.genfromtxt(stats_path, delimiter=",", skip_header=1,
                         usecols=(0, 1, 2))
    acc = []
    with open(stats_path) as fh:
        next(fh)
        for line in fh:
            a, i, r = (float(v) for v in
                       line.split("(")[1].rstrip(")\n").split(","))
            tot = a + i + r
            acc.append(a / tot if tot else np.nan)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.semilogy(rows[:, 0], np.maximum(rows[:, 1], 1e-12), "o-", ms=3,
                label="beta")
    ax.set_xlabel("Annealing iteration")
    ax.set_ylabel("beta (log)")
    ax2 = ax.twinx()
    ax2.plot(rows[:, 0], acc, "s-", ms=3, color="crimson",
             label="acceptance rate")
    ax2.set_ylabel("Acceptance rate", color="crimson")
    ax2.tick_params(axis="y", colors="crimson")
    ax.set_title("AlTar annealing schedule")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {os.path.basename(path)}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("output_dir", help="AlTar output directory (with step_*.h5)")
    p.add_argument("--inputs", default=None,
                   help="AlTar inputs directory (default: <run>/inputs, where "
                        "<run> is the parent of output_dir)")
    p.add_argument("--n-random", type=int, default=3,
                   help="number of random patch-PDF figures (default 3)")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-fit", action="store_true",
                   help="skip the data/model/residual maps (no dataset rebuild)")
    p.add_argument("--valmax", type=float, default=None,
                   help="bivariate colour scale: max slip (default: rounded "
                        "posterior-mode max)")
    p.add_argument("--sigmamax", type=float, default=None,
                   help="bivariate colour scale: max std (default: rounded "
                        "posterior std max)")
    p.add_argument("--fps", type=int, default=2, help="video frame rate")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    run_dir = os.path.dirname(out_dir)
    inputs_dir = args.inputs or os.path.join(run_dir, "inputs")
    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    print(f"[in ] outputs: {out_dir}\n[in ] inputs:  {inputs_dir}\n"
          f"[out] figures: {figs_dir}")

    # ---- fault (same construction as the prep script) ----
    strands = [FaultTriangles.from_npz(os.path.join(prep.FAULT_DIR, f))
               for f in prep.FAULT_NPZ]
    fault = FaultTriangles.merge(strands, name="LLFZ")
    fault.set_dip_convention(prep.DIP_NORMALS)
    sections = fault_sections(fault)
    print(f"[fault] {fault}")

    # ---- posterior samples ----
    final_path = os.path.join(out_dir, "step_final.h5")
    ps, beta = load_step(final_path)
    samples = model_matrix(ps)
    n_patches = fault.n_patches
    print(f"[post] {samples.shape[0]} samples x {samples.shape[1]} params "
          f"(final beta = {beta:.4g})")

    # sanity-check parameter ordering against G
    with h5py.File(os.path.join(inputs_dir, "gf.h5"), "r") as fh:
        G = fh["gf"][:]
    with h5py.File(os.path.join(inputs_dir, "data.h5"), "r") as fh:
        d = fh["data"][:]
        ds_slices = {k: slice(*fh["datasets"][k][:]) for k in fh["datasets"]}
    rms_d = float(np.sqrt(np.mean(d**2)))
    rms_r = float(np.sqrt(np.mean((d - G @ samples.mean(axis=0))**2)))
    print(f"[post] posterior-mean residual RMS {rms_r:.4f} m "
          f"(data RMS {rms_d:.4f} m)")
    if rms_r > 0.8 * rms_d:
        raise RuntimeError("posterior-mean prediction barely beats the data -- "
                           "parameter ordering vs G is probably wrong")

    print("[post] computing per-parameter posterior modes (KDE)...")
    summ = slip_summaries(ps, n_patches, seed=args.seed)
    m_mode = np.concatenate([summ["ss_mode"], summ["ds_mode"],
                             kde_mode(ps["ramp"], args.seed)])

    # ---- 1: 2D + 3D slip (posterior mode) ----
    ds_lim = float(np.abs(summ["ds_mode"]).max()) or 1e-3
    fig_slip2d(fault, summ["tot_mode"], "Total slip (m)", SLIP_CMAP,
               os.path.join(figs_dir, "slip2d_total.png"),
               vmin=0., vmax=float(summ["tot_mode"].max()))
    fig_slip2d(fault, np.abs(summ["ss_mode"]), "|Strike-slip| (m)", SLIP_CMAP,
               os.path.join(figs_dir, "slip2d_strikeslip.png"),
               vmin=0., vmax=6.)
    fig_slip2d(fault, summ["ds_mode"], "Dip-slip (m)", cmc.vik,
               os.path.join(figs_dir, "slip2d_dipslip.png"),
               vmin=-ds_lim, vmax=ds_lim)
    fig_slip3d(fault, summ["tot_mode"],
               os.path.join(figs_dir, "slip3d_total.png"))

    # ---- 2: bivariate slip + sigma ----
    def round_up_nice(v):
        e = 10. ** np.floor(np.log10(max(v, 1e-9)))
        return float(np.ceil(v / e) * e)
    valmax = args.valmax or round_up_nice(float(summ["tot_mode"].max()))
    sigmamax = args.sigmamax or round_up_nice(float(summ["tot_std"].max()))
    print(f"[biv ] valmax = {valmax}, sigmamax = {sigmamax}")
    fig_slip_bivariate(fault, sections, summ["tot_mode"], summ["tot_std"],
                       valmax, sigmamax,
                       os.path.join(figs_dir, "slip_bivariate.png"))

    # ---- 3: patch posterior PDFs ----
    rng = np.random.default_rng(args.seed)
    for i in range(args.n_random):
        ids = rng.choice(n_patches, 8, replace=False)
        fig_patch_pdfs(fault, sections, summ, ids,
                       os.path.join(figs_dir, f"patch_pdfs_random_{i+1}.png"),
                       subtitle=f"random selection {i+1}")
    fig_patch_pdfs(fault, sections, summ,
                   pick_depth_transect(fault, sections, summ),
                   os.path.join(figs_dir, "patch_pdfs_depth_transect.png"),
                   subtitle="down-dip transect through the slip maximum")
    top_ids = np.argsort(summ["tot_mode"])[-8:][::-1]
    fig_patch_pdfs(fault, sections, summ, top_ids,
                   os.path.join(figs_dir, "patch_pdfs_top_slip.png"),
                   subtitle="8 highest-slip patches")

    # ---- 4: data / model / residual ----
    if not args.skip_fit:
        cache = os.path.join(run_dir, "dataset_coords.npz")
        coords = rebuild_dataset_coords(fault, inputs_dir, cache)
        pred = G @ m_mode
        for lbl, sl in ds_slices.items():
            c = coords[lbl]
            fn = os.path.join(figs_dir, f"fit_{lbl}.png")
            if c["type"] == "insar":
                fig_fit_insar(lbl, c, d[sl], pred[sl], fn)
            elif c["type"] == "gnss":
                fig_fit_gnss(lbl, c, d[sl], pred[sl], fn)
            else:
                fig_fit_optical(lbl, c, d[sl], pred[sl], fn)

        # independent check: forward-predict the near-field optical (image
        # correlation), which is NOT in the inversion, from the mode slip model
        if os.path.exists(optres.EW_TIF):
            print("[optical] forward-predicting near-field optical residual...")
            optres.optical_residual_figure(
                fault, summ["ss_mode"], summ["ds_mode"],
                os.path.join(figs_dir, "fit_optical_check.png"))
        else:
            print(f"[optical] skipped (no {optres.EW_TIF})")

    # ---- 5: convergence video ----
    if not args.skip_video:
        step_files = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir)
                            if f.startswith("step_") and f.endswith(".h5"))
        video_convergence_bivariate(fault, sections, step_files, valmax,
                                    sigmamax,
                                    os.path.join(figs_dir,
                                                 "convergence_bivariate.mp4"),
                                    fps=args.fps, seed=args.seed)

    # ---- 6: trade-offs / moment / annealing ----
    n_main = ps["strikeslipmain"].shape[1]
    blocks = [("SS main", 0, n_main), ("SS second", n_main, n_patches),
              ("DS main", n_patches, n_patches + n_main),
              ("DS second", n_patches + n_main, 2 * n_patches),
              ("ramp", 2 * n_patches, samples.shape[1])]
    fig_correlation_matrix(samples, blocks,
                           os.path.join(figs_dir,
                                        "posterior_correlation_matrix.png"))
    peak = int(np.argmax(summ["tot_mode"]))
    fig_correlation_map(fault, sections, summ, peak, "SS",
                        os.path.join(figs_dir,
                                     f"correlation_map_patch{peak}.png"))

    with h5py.File(os.path.join(inputs_dir, "patch_areas.h5"), "r") as fh:
        areas = fh["patch_areas"][:]
    fig_moment_magnitude(fault, summ, areas,
                         os.path.join(figs_dir, "moment_magnitude.png"))
    fig_uncertainty_vs_depth(fault, summ,
                             os.path.join(figs_dir, "uncertainty_vs_depth.png"))
    fig_ramp_pdfs(ps, os.path.join(figs_dir, "ramp_pdfs.png"))

    stats = os.path.join(out_dir, "BetaStatistics.txt")
    if os.path.exists(stats):
        fig_beta_annealing(stats, os.path.join(figs_dir, "beta_annealing.png"))

    print(f"\n[done] figures in {figs_dir}")


if __name__ == "__main__":
    main()
