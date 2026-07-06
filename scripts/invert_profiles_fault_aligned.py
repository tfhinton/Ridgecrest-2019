#!/usr/bin/env python3
"""
Fault-aligned optical profiling -> 2D damage-zone slip inversion.

This is the successor to invert_profiles.py. Instead of hand-drawn CSI profiles
+ swathe averaging, it builds fault-perpendicular displacement profiles that
follow the *true* fault trace (located from peak shear strain) and stack
along-strike, using OpticalData.evaluate_profiles_fault_aligned. Each profile is
then fed through the same covariance -> centre -> forward-model -> invert loop as
the original script.

Two stages, run separately or together:
  1. GENERATE  all stacked, fault-aligned profiles once (the expensive step) and
     pickle them. A "profile extracted" figure is written per inverted profile.
  2. INVERT    each selected profile with a HamiltonianInversion, overwriting its
     figure with the model fit + posterior parameter distributions, and pickling
     the inversion. Finally a summary figure of every inverted parameter vs
     along-strike distance is written.

Modes (--mode):
  test   one central profile only, on a small cropped window. Fast; use this to
         sanity-check the whole pipeline before committing to the full run.
  full   ~N_PROFILES evenly spaced along the main fault (first trace linestring).

Linking to the far-field (AlTar triangular) inversion
-----------------------------------------------------
The far-field Bayesian inversion (prep_files_for_altar_triangles.py -> AlTar)
constrains the deep slip; here we take its most-probable model (per-patch KDE
mode of the posterior), forward-predict the surface displacement due to the
DEEP patches only (triangle centroid depth > DEEP_DEPTH_M = 2.5 km, both
strands, strike- and dip-slip), subtract it from each optical profile, and
invert the residual for SHALLOW slip only. The forward prediction is computed
with the same Meade (2007) triangular-dislocation Green's functions as the
far-field inversion, evaluated ONLY at the binned profile sample points (not
the full optical scene), so it costs seconds.

The shallow 2D model is discretised at depth interfaces VD = 0 / 250 / 750 /
1500 / 2500 m -> 4 slip parameters, ALL inverted, plus the damage-zone
half-width and modulus ratio (6 parameters total). There is no deep-slip
seeding any more: the deep field is removed from the data instead.

Note on the depth split: the triangular mesh has layer interfaces at 0 / 1500 /
3000 / 5500 ... m, so "centroid depth > 2500 m" selects exactly the triangles
lying entirely below 3000 m. Far-field slip attributed to the 1500-3000 m layer
is therefore NOT removed -- it is re-solved locally by the shallow 2D patches.

Notes
-----
* The new profiles already centre xs on the fault (x = 0), so the zero-crossing
  re-centring of the old script is dropped; only the data mean is removed
  (after the deep-field subtraction).
* The inverted component is the FAULT-PARALLEL (strike-slip) displacement, which
  is row 0 of p.displacements in the fault-aligned method (row 1 is fault-normal).
  PARALLEL_SIGN lets you flip its polarity if it disagrees with the forward
  model; the deep-field prediction is subtracted in the same signed frame, and
  the far-field step of data vs prediction is printed as a sign sanity check.
"""

####    IMPORTS    ####
import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas as gpd
from scipy.stats import gaussian_kde
from shapely.geometry import box, LineString
from shapely.ops import substring

from codes import (OpticalData, Fault, FaultTriangles, TwoDDzForwardModel,
                   PatchTwoD, UniformDist, HamiltonianInversion)

# fault mesh / dip-convention config shared with the far-field inversion
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep_files_for_altar_triangles as prep


####    FILEPATHS (local Mac paths)    ####
ROOT          = Path("/Users/hintont/Dev/projects/ridgecrest")
FAULT_SHP     = ROOT / "data/fault/little_lake_trace_multi.shp"
OPT_EW        = ROOT / "data/optical/EW_Ridgecrest_1m_utm_detrended.tif"
OPT_NS        = ROOT / "data/optical/NS_Ridgecrest_1m_utm_detrended.tif"
ALTAR_STEP    = ROOT / "results/tri01/outputs_16000_12000/step_final.h5"
RESDIR        = ROOT / "results/profile_inversion/fa02/"


####    PARAMETERS    ####
MAIN_FAULT_GEOM = 0       # which trace linestring is the main fault
N_PROFILES      = 20      # number of evenly spaced profiles to invert (full mode)

# --- fault-aligned profiling (all lengths in pixels == metres at 1 m/px) ---
PLEN        = 5000        # profile half-length (m). 5 km each side of the fault.
STACK       = 150         # along-strike half-window for stacking (m). Swathe = 2*STACK
TRACE_SMOOTH = 15
# Strain (used only to relocate the profile onto the true fault + estimate FZW) is
# computed only within this half-width of the fault, so its cost no longer scales
# with PLEN. ~the likely fault-trace error; raise if the fault zone is wider.
STRAIN_HALF_WIDTH = 200.  # m
# Parallel workers for the per-profile extraction (forked; Linux). None/1 = serial.
N_JOBS      = max(1, (os.cpu_count() or 2) - 1)
# float32 storage roughly halves the profile-buffer memory, which matters at
# PLEN=5000 over the whole fault. Set to np.float64 to reproduce the old run exactly.
PROF_DTYPE  = np.float32

# --- binning before inversion (reduces noise + data dimension) ---
N_BINS          = 200
N_NEAR_BINS     = 100
NEAR_FAULT_DIST = 400.    # m

# --- which displacement component, and its polarity ---
# The fault-aligned parallel component (row 0) steps UP left->right here, but the
# forward model + negative-slip priors expect a DOWN step (verified: slip=-1 gives
# left-positive / right-negative). So we flip the data to -1.0 to match the
# right-lateral convention. The deep-field prediction is subtracted in the same
# signed frame; the per-profile far-field step check below confirms consistency.
PARALLEL_SIGN = -1.0

# --- deep-field removal (far-field AlTar posterior mode) ---
DEEP_DEPTH_M    = 2500.   # triangles with centroid depth > this are "deep" (removed)
KDE_MAX_SAMPLES = 4000    # posterior subsample per patch for the KDE mode

# --- forward model depth layering (patch interfaces, m); ALL slips inverted ---
VD = [0., 250., 750., 1500., 2500.]
SLIP_LABELS = [f"slip{i}" for i in range(len(VD) - 1)]

# --- sampler ---
DRAWS, TUNE, CHAINS = 120, 60, 6
TIMEOUT_MINUTES = 15   # per-profile wall-clock cap; accepts partial draws

# --- summary plot: which inverted parameters to show along strike ---
SUMMARY_PARAMS = ["dz_halfwidth", "modulus_ratio"] + SLIP_LABELS

# --- test mode: length of the central trace segment used (m). Kept short so only
#     a handful of stacked profiles are produced (cost ~ segment length). ---
TEST_SEG_LEN = 2 * STACK + 120.


####    HELPERS    ####
def kde_mode(samples):
    """Most-probable value (KDE peak) of a 1-D posterior. scipy-only, so it is
    robust to arviz version changes; falls back to the median if the KDE fails."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size < 5 or np.ptp(samples) == 0:
        return float(np.median(samples)) if samples.size else np.nan
    try:
        grid = np.linspace(samples.min(), samples.max(), 512)
        return float(grid[np.argmax(gaussian_kde(samples)(grid))])
    except Exception:
        return float(np.median(samples))


def load_main_fault():
    """Return a Fault whose .trace is only the main-fault linestring."""
    trace = gpd.read_file(FAULT_SHP)
    main = trace.iloc[[MAIN_FAULT_GEOM]].reset_index(drop=True)
    fault = Fault()
    fault.trace = main
    return fault


def load_optical_window(bbox):
    """Load EW/NS rasters cropped to a shapely box (avoids loading the 20 GB whole scene)."""
    x0, y0, x1, y1 = bbox.bounds
    opt = OpticalData(ew_filepath=str(OPT_EW), ns_filepath=str(OPT_NS), verbose=True)
    opt = opt.clear_nan()
    # y is stored north->south, so the slice must go high -> low
    opt = opt.get_window(x0, x1, y1, y0)
    return opt


def generate_profiles(mode):
    """Run the (expensive) fault-aligned profiling and return the stacked profile list."""
    fault = load_main_fault()
    ls = fault.trace.geometry[0]
    margin = PLEN + STACK + 800.   # raster margin around the trace corridor (m)

    if mode == "test":
        # a SHORT central trace segment -> only a handful of stacked profiles.
        # Densify so the cubic trace resampling has enough vertices to work with.
        s_mid = ls.length / 2.
        seg = substring(ls, s_mid - TEST_SEG_LEN / 2., s_mid + TEST_SEG_LEN / 2.)
        seg = seg.segmentize(25.)
        fault.trace = gpd.GeoDataFrame(geometry=[LineString(seg.coords)],
                                       crs=fault.trace.crs)
        minx, miny, maxx, maxy = seg.bounds
        bbox = box(minx - margin, miny - margin, maxx + margin, maxy + margin)
        print(f"[gen] TEST central segment {TEST_SEG_LEN:.0f} m around "
              f"along-strike {s_mid / 1000:.1f} km")
    else:
        # corridor bounding box of the whole main fault + margin
        minx, miny, maxx, maxy = ls.bounds
        bbox = box(minx - margin, miny - margin, maxx + margin, maxy + margin)
        print(f"[gen] FULL corridor bbox {bbox.bounds}, fault length "
              f"{ls.length / 1000:.1f} km")

    print("[gen] loading + cropping optical rasters ...")
    opt = load_optical_window(bbox)
    print(f"[gen] window shape EW={opt.ew.shape} NS={opt.ns.shape} "
          f"(~{opt.ew.size / 1e6:.0f} Mpx)")

    print(f"[gen] running evaluate_profiles_fault_aligned "
          f"(plen={PLEN}, stack={STACK}, strain_half_width={STRAIN_HALF_WIDTH}, "
          f"n_jobs={N_JOBS}) ...")
    t0 = time.time()
    profiles = opt.evaluate_profiles_fault_aligned(
        fault, plen=PLEN, stack=STACK, trace_smooth=TRACE_SMOOTH,
        strain_half_width=STRAIN_HALF_WIDTH, n_jobs=N_JOBS, prof_dtype=PROF_DTYPE,
        attach_to_fault=True, store=True)
    print(f"[gen] produced {len(profiles)} stacked profiles in "
          f"{time.time() - t0:.0f} s")
    return profiles, opt


def select_profiles(profiles, mode):
    """Pick which stacked profiles to invert. Returns list of (idx, Profile)."""
    n = len(profiles)
    if mode == "test":
        c = n // 2
        return [(c, profiles[c])]
    idx = np.unique(np.linspace(0, n - 1, N_PROFILES).round().astype(int))
    return [(i, profiles[i]) for i in idx]


####    DEEP-FIELD REMOVAL (far-field AlTar posterior mode)    ####
def load_triangle_fault():
    """The merged triangular fault, built exactly as in the far-field prep script
    (same strands, same dip-slip sign convention -> same GF sense as AlTar's G)."""
    strands = [FaultTriangles.from_npz(os.path.join(prep.FAULT_DIR, f))
               for f in prep.FAULT_NPZ]
    fault = FaultTriangles.merge(strands, name="LLFZ")
    fault.set_dip_convention(prep.DIP_NORMALS)
    return fault


def posterior_mode_deep_slips(fault, seed=0):
    """Per-patch posterior-mode slip vectors (SS, DS) with SHALLOW patches zeroed.

    The AlTar model vector is the ParameterSets concatenated as
    [strikeslipmain, strikeslipsecond, dipslip, ramp]; SS main+second follow the
    merged patch order of `load_triangle_fault` (see visualise_altar_triangles).
    Ramps apply to the InSAR tracks only, so they are irrelevant here.
    """
    with h5py.File(ALTAR_STEP, "r") as fh:
        ps = {k: fh["ParameterSets"][k][:] for k in
              ("strikeslipmain", "strikeslipsecond", "dipslip")}
    ss = np.hstack([ps["strikeslipmain"], ps["strikeslipsecond"]])
    ds = ps["dipslip"]
    if ss.shape[1] != fault.n_patches or ds.shape[1] != fault.n_patches:
        raise RuntimeError(f"AlTar posterior has {ss.shape[1]} SS / {ds.shape[1]} "
                           f"DS patches but the fault has {fault.n_patches}")

    rng = np.random.default_rng(seed)
    idx = rng.choice(ss.shape[0], size=min(ss.shape[0], KDE_MAX_SAMPLES),
                     replace=False)
    ss_mode = np.array([kde_mode(ss[idx, j]) for j in range(ss.shape[1])])
    ds_mode = np.array([kde_mode(ds[idx, j]) for j in range(ds.shape[1])])

    deep = fault.depths > DEEP_DEPTH_M
    ss_mode[~deep] = 0.
    ds_mode[~deep] = 0.
    print(f"[deep] posterior mode from {ALTAR_STEP.parent.name}/step_final.h5: "
          f"{int(deep.sum())}/{fault.n_patches} patches deeper than "
          f"{DEEP_DEPTH_M:.0f} m kept "
          f"(|SS| up to {np.abs(ss_mode).max():.2f} m, "
          f"|DS| up to {np.abs(ds_mode).max():.2f} m)")
    return ss_mode, ds_mode


def profile_points(profile, xs):
    """UTM sample points + strike unit vector for one profile.

    The profile trace LineString runs from the xs = -plen end to the xs = +plen
    end (plen = profile.fault_x), so positions are linear in xs along it. The
    fault-parallel component (displacements row 0) is the displacement projected
    onto the local strike bearing, s_hat = (sin strike, cos strike) in (E, N).

    Returns (pts (n, 2) easting/northing, s_hat (2,)).
    """
    c = np.asarray(profile.linestring.coords, dtype=float)
    p1, p2 = c[0], c[-1]
    length = float(np.hypot(*(p2 - p1)))          # = 2 * plen
    t = (np.asarray(xs, dtype=float) + profile.fault_x) / length
    pts = p1[None, :] + t[:, None] * (p2 - p1)[None, :]

    theta = np.radians(profile.strike)
    s_hat = np.array([np.sin(theta), np.cos(theta)])
    # the profile direction must be ~perpendicular to the strike used for the
    # parallel/normal rotation; a large dot product means broken geometry
    u_hat = (p2 - p1) / length
    if abs(float(u_hat @ s_hat)) > 0.15:
        raise RuntimeError(f"profile/strike geometry inconsistent "
                           f"(|u.s| = {abs(float(u_hat @ s_hat)):.2f})")
    return pts, s_hat


def deep_predictions(fault, ss_mode, ds_mode, prof_pts):
    """Fault-parallel surface displacement of the deep slip model at each
    profile's sample points, via one batched TDE Green's-function evaluation.

    prof_pts : list of (pts, s_hat) from `profile_points`.
    Returns a list of (n_i,) predictions in the PHYSICAL (unsigned) frame of
    profile.displacements[0].
    """
    all_pts = np.vstack([pts for pts, _ in prof_pts])
    print(f"[deep] computing TDE Green's functions at {len(all_pts)} profile "
          f"points x {fault.n_patches} patches ...")
    t0 = time.time()
    fault.compute_greens_functions(all_pts.T)      # (2, n_patches, 3, n_pts)
    # E/N displacement of the mode model (shallow slips are zero)
    u_en = (np.tensordot(ss_mode, fault.gfs[0, :, :2, :], axes=(0, 0))
            + np.tensordot(ds_mode, fault.gfs[1, :, :2, :], axes=(0, 0)))
    print(f"[deep] done in {time.time() - t0:.0f} s")

    preds, k = [], 0
    for pts, s_hat in prof_pts:
        n = len(pts)
        preds.append(s_hat @ u_en[:, k:k + n])
        k += n
    return preds


####    INVERSION PIPELINE    ####
def build_model(profile_xs):
    """The layered shallow TwoDDzForwardModel; every patch's slip is inverted."""
    model = TwoDDzForwardModel()
    model.patches = [PatchTwoD(0., (VD[i] + VD[i + 1]) / 2., VD[i + 1] - VD[i])
                     for i in range(len(VD) - 1)]
    model.slips = np.zeros(len(model.patches))
    model.xs = profile_xs
    return model


def prepare_data(profile):
    """Bin-average, pick the fault-parallel component, apply PARALLEL_SIGN, drop NaNs.

    Returns (xs, data_signed) -- NOT yet deep-corrected or centred -- or None if
    degenerate.
    """
    binned = profile.bin_average(n_bins=N_BINS, n_near_fault_bins=N_NEAR_BINS,
                                 near_fault_dist=NEAR_FAULT_DIST)
    parallel = PARALLEL_SIGN * binned.displacements[0]
    finite = np.isfinite(parallel) & np.isfinite(binned.xs)
    xs = binned.xs[finite]
    data = parallel[finite].astype(float)
    if data.size < 40:
        return None
    return xs, data


def finalise_data(data_signed, pred_par, tag):
    """Subtract the deep-field prediction, estimate the covariance from the
    residual far field, and centre.

    pred_par is physical (same frame as displacements[0]); the data already
    carry PARALLEL_SIGN, so the prediction is subtracted in the same signed frame.
    Returns (data, pred_signed, data_covariance).
    """
    pred_signed = PARALLEL_SIGN * pred_par
    resid = data_signed - pred_signed

    # sign/convention check: the deep model should reproduce the far-field step
    step_d = np.mean(data_signed[-25:]) - np.mean(data_signed[:25])
    step_p = np.mean(pred_signed[-25:]) - np.mean(pred_signed[:25])
    print(f"[inv] {tag}: far-field step data {step_d:+.2f} m, "
          f"deep model {step_p:+.2f} m")
    if step_d * step_p < 0.:
        print(f"[inv] {tag}: WARNING -- deep-model step has the OPPOSITE sign "
              f"to the data; check PARALLEL_SIGN / slip conventions")

    # covariance from the residual far-field band at the negative-x end, detrended
    cd_est = resid[:25]
    t = np.arange(cd_est.size)
    detrended = cd_est - np.polyval(np.polyfit(t, cd_est, 1), t)
    std = np.std(detrended)
    if not np.isfinite(std) or std == 0.:
        std = np.std(resid) or 1e-3
    data_covariance = std ** 2 * np.eye(resid.size)

    # centre (the xs are already centred on the fault; only remove the mean)
    data = resid - np.mean(resid)
    return data, pred_signed, data_covariance


####    FIGURES    ####
def plot_profile_only(profile, path, title):
    """First-pass figure: the extracted profile (parallel + normal) before inversion."""
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(profile.xs, PARALLEL_SIGN * profile.displacements[0], lw=1.,
            color="crimson", label="fault-parallel (inverted)")
    ax.plot(profile.xs, profile.displacements[1], lw=1., color="steelblue",
            alpha=0.7, label="fault-normal")
    ax.axvline(0., color="lightgray", ls="--")
    ax.set_xlabel("Distance from fault (m)")
    ax.set_ylabel("Displacement (m)")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_inversion(inversion, model, xs, data, data_signed, pred_signed,
                   path, title):
    """Second-pass figure: model fit (top) + posterior parameter distributions
    (bottom). Self-contained (matplotlib only) so it does not depend on the
    arviz plotting API, which varies between versions.
    """
    post = inversion.result.posterior
    labels = ["dz_halfwidth", "modulus_ratio"] + SLIP_LABELS
    samp = {l: post[l].values.flatten() for l in labels}
    med = {l: np.median(s) for l, s in samp.items()}

    # best-fit model curve (posterior medians)
    fit = model._copy()
    fit.dz_half_width = med["dz_halfwidth"]
    fit.modulus_ratio = med["modulus_ratio"]
    fit.slips = np.array([med[l] for l in SLIP_LABELS], dtype=float)
    fit = fit.run(xs)

    fig = plt.figure(figsize=(12, 7), layout="constrained")
    gs = gridspec.GridSpec(2, len(labels), height_ratios=[2, 1], figure=fig)
    fig.suptitle(title)

    # --- slip vs depth (all patches inverted) ---
    ax_s = fig.add_subplot(gs[0, 0])
    depth, slip = [], []
    for i, p in enumerate(fit.patches):
        depth += [p.top, p.bottom]
        slip += [fit.slips[i]] * 2
    ax_s.plot([-s for s in slip], depth, color="navy")
    ax_s.axvline(0, color="lightgray", ls="--")
    ax_s.invert_yaxis()
    ax_s.set_xlabel("Right-lateral slip (m)")
    ax_s.set_ylabel("Depth (m)")
    ax_s.set_title("Shallow slip (median)")

    # --- data, deep-field removal, and model fit ---
    ax_f = fig.add_subplot(gs[0, 1:])
    ax_f.plot(xs, data_signed - np.mean(data_signed), color="0.8", lw=1.,
              label="data (pre-removal)")
    ax_f.plot(xs, pred_signed - np.mean(data_signed), color="seagreen", lw=1.2,
              ls="--", label="deep model (AlTar mode)")
    ax_f.plot(xs, data, color="0.35", lw=1.2, label="residual data (inverted)")
    ax_f.plot(xs, fit.sol - np.mean(fit.sol), color="crimson", lw=1.5,
              label="shallow model fit")
    ax_f.axvline(0, color="lightgray", ls="--")
    ax_f.set_xlabel("Distance from fault (m)")
    ax_f.set_ylabel("Fault-parallel displacement (m)")
    ax_f.set_title("Deep-field removal + shallow model fit")
    ax_f.legend(fontsize=8)

    # --- posterior histograms ---
    for k, l in enumerate(labels):
        ax = fig.add_subplot(gs[1, k])
        ax.hist(samp[l], bins=30, color="steelblue", alpha=0.8)
        ax.axvline(med[l], color="crimson", lw=1.2)
        ax.axvline(np.percentile(samp[l], 16), color="black", ls=":", lw=0.8)
        ax.axvline(np.percentile(samp[l], 84), color="black", ls=":", lw=0.8)
        ax.set_title(l, fontsize=9)
        ax.set_yticks([])

    fig.savefig(path, dpi=200)
    plt.close(fig)


def summary_plot(records, path):
    """Scatter of each inverted parameter vs along-strike distance.

    records: list of dicts with keys 'along_km', and per-param 'map'/'lo'/'hi'.
    Mirrors plot_dz_along_strike.py: most-probable value (KDE mode) as the marker,
    16th/84th percentiles as error bars.
    """
    params = SUMMARY_PARAMS
    fig, axes = plt.subplots(len(params), 1, figsize=(9, 2.4 * len(params)),
                             sharex=True, layout="constrained")
    axes = np.atleast_1d(axes)
    xs = np.array([r["along_km"] for r in records])
    for ax, p in zip(axes, params):
        med = np.array([r[p]["map"] for r in records])
        lo = np.array([r[p]["lo"] for r in records])
        hi = np.array([r[p]["hi"] for r in records])
        # the most-probable value can sit outside [16, 84] when the posterior
        # rails against a prior bound; clip the bars so they stay non-negative
        errs = np.clip(np.vstack([med - lo, hi - med]), 0., None)
        ax.errorbar(xs, med, yerr=errs, fmt="s", color="steelblue",
                    ecolor="black", elinewidth=1., capsize=3., ms=6., zorder=5)
        ax.set_ylabel(p)
        ax.grid(True, ls=":", color="gray", alpha=0.7)
    axes[-1].set_xlabel("Along-strike distance (km)")
    axes[0].set_title("Inverted parameters along strike "
                      "(marker = most probable, bars = 16-84th pct)")
    fig.savefig(path, dpi=250)
    plt.close(fig)


####    MAIN    ####
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["test", "full"], default="test")
    ap.add_argument("--skip-invert", action="store_true",
                    help="generate + plot profiles only, no inversion")
    ap.add_argument("--regenerate", action="store_true",
                    help="force re-generation even if a cached profile pickle exists")
    args = ap.parse_args()

    figdir = RESDIR / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    prof_pickle = RESDIR / f"profiles_{args.mode}.pickle"

    ####    1. GENERATE (with caching)    ####
    if prof_pickle.exists() and not args.regenerate:
        print(f"[gen] loading cached profiles from {prof_pickle}")
        profiles = pickle.load(open(prof_pickle, "rb"))
    else:
        profiles, _ = generate_profiles(args.mode)
        with open(prof_pickle, "wb") as f:
            pickle.dump(profiles, f)
        print(f"[gen] cached {len(profiles)} profiles to {prof_pickle}")

    selected = select_profiles(profiles, args.mode)
    print(f"[sel] inverting {len(selected)} profile(s): "
          f"indices {[i for i, _ in selected]}")

    # first-pass "profile extracted" figures
    for i, p in selected:
        plot_profile_only(p, figdir / f"profile_{i:03d}.png",
                          f"Profile {i}  (along-strike {p.x_along_fault / 1000:.2f} km)")
    print(f"[fig] wrote {len(selected)} profile-only figures to {figdir}")

    if args.skip_invert:
        print("[done] --skip-invert set; stopping before inversion.")
        return

    ####    2. DEEP-FIELD PREDICTION (far-field AlTar posterior mode)    ####
    print("\n[deep] loading triangular fault + AlTar posterior mode ...")
    tri_fault = load_triangle_fault()
    ss_mode, ds_mode = posterior_mode_deep_slips(tri_fault)

    # bin every selected profile first, so the TDE Green's functions can be
    # computed in one batch at just the binned sample points
    prepped = []
    for i, profile in selected:
        pre = prepare_data(profile)
        if pre is None:
            print(f"[inv] profile {i}: too few finite data points, skipping.")
            continue
        xs, data_signed = pre
        pts, s_hat = profile_points(profile, xs)
        prepped.append((i, profile, xs, data_signed, (pts, s_hat)))
    if not prepped:
        print("[done] no usable profiles.")
        return

    preds = deep_predictions(tri_fault, ss_mode, ds_mode,
                             [pp[-1] for pp in prepped])

    ####    3. INVERT    ####
    priors = ([UniformDist("dz_halfwidth", 0., 2500.),
               UniformDist("modulus_ratio", 0.2, 0.9),
               UniformDist("slip0", -5., 0.)]
              + [UniformDist(l, -10., 0.) for l in SLIP_LABELS[1:]])

    records = []
    for n, ((i, profile, xs, data_signed, _), pred_par) in \
            enumerate(zip(prepped, preds), 1):
        tag = f"profile {i} ({n}/{len(prepped)})"
        try:
            print(f"\n[inv] === {tag} ===")
            data, pred_signed, Cd = finalise_data(data_signed, pred_par, tag)
            along_km = profile.x_along_fault / 1000.
            model = build_model(xs)
            print(f"[inv] {tag}: along-strike {along_km:.2f} km, "
                  f"{data.size} data pts")

            inversion = HamiltonianInversion(model, priors, data, Cd)
            inversion = inversion.run(draws=DRAWS, tune=TUNE, chains=CHAINS,
                                       timeout_minutes=TIMEOUT_MINUTES)

            with open(RESDIR / f"profile_{i:03d}.pickle", "wb") as f:
                pickle.dump(inversion, f)
            plot_inversion(inversion, model, xs, data, data_signed, pred_signed,
                           figdir / f"profile_{i:03d}_inv.png",
                           f"Profile {i} inversion (along-strike {along_km:.2f} km)")

            # collect for the along-strike summary (most-probable value + 16/84 pct)
            post = inversion.result.posterior
            rec = {"along_km": along_km}
            for lbl in SUMMARY_PARAMS:
                s = post[lbl].values.flatten()
                rec[lbl] = {"map": kde_mode(s),
                            "lo": np.percentile(s, 16),
                            "hi": np.percentile(s, 84)}
            records.append(rec)
            print(f"[inv] {tag}: done, saved pickle + figure.")
        except (Exception, KeyboardInterrupt) as e:
            print(f"[inv] {tag} FAILED: {type(e).__name__}: {e}")

    ####    4. SUMMARY    ####
    if len(records) > 1:
        records.sort(key=lambda r: r["along_km"])
        summary_plot(records, figdir / "summary_along_strike.png")
        print(f"\n[done] wrote summary figure to {figdir / 'summary_along_strike.png'}")
    else:
        print("\n[done] not enough successful inversions for a summary plot.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
