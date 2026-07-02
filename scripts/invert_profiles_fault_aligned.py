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

Notes
-----
* The new profiles already centre xs on the fault (x = 0), so the zero-crossing
  re-centring of the old script is dropped; only the data mean is removed.
* The inverted component is the FAULT-PARALLEL (strike-slip) displacement, which
  is row 0 of p.displacements in the fault-aligned method (row 1 is fault-normal).
  PARALLEL_SIGN lets you flip its polarity if it disagrees with the seeded slip.
* Deep-patch slip is seeded from the AlTar posterior exactly as before, by
  projecting each profile's fault midpoint onto the CSI trace to get its
  along-strike position, then reading the strikeslipmain patches that the local
  vertical section intersects.
"""

####    IMPORTS    ####
import argparse
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas as gpd
from scipy.stats import gaussian_kde
from shapely.geometry import box, LineString
from shapely.ops import substring

from codes import (OpticalData, Fault, TwoDDzForwardModel, PatchTwoD,
                   UniformDist, HamiltonianInversion, AltarOutput)


####    FILEPATHS (local Mac paths)    ####
ROOT          = Path("/Users/hintont/Dev/projects/ridgecrest")
FAULT_SHP     = ROOT / "data/fault/little_lake_trace_multi.shp"
OPT_EW        = ROOT / "data/optical/EW_Ridgecrest_1m_utm_detrended.tif"
OPT_NS        = ROOT / "data/optical/NS_Ridgecrest_1m_utm_detrended.tif"
CSI_PICKLE    = ROOT / "results/inversion_results/in15/inputs/csi.pickle"
ALTAR_DIR     = ROOT / "results/inversion_results/in15/outputs/"
RESDIR        = ROOT / "results/profile_inversion/fa01/"


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
# right-lateral convention. CONFIRM this against the real AlTar seed sign on the
# first full run -- if the deep seed is positive, set this back to +1.0.
PARALLEL_SIGN = -1.0

# --- forward model depth layering (patch interfaces, m) ---
VD = [0., 500., 1000., 1500., 2750., 4000., 6000., 8000., 11000., 14000.]
N_SHALLOW_INVERTED = 3    # slip on the 3 shallowest patches is inverted; rest seeded

# --- sampler ---
DRAWS, TUNE, CHAINS = 120, 60, 6

# --- summary plot: which inverted parameters to show along strike ---
SUMMARY_PARAMS = ["dz_halfwidth", "modulus_ratio", "slip0", "slip1", "slip2"]

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
    opt = OpticalData(ew_filepath=str(OPT_EW), ns_filepath=str(OPT_NS), verbose=False)
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


def csi_along_strike_km(csi, easting_m, northing_m, fault_idx=0):
    """Arc-length (km) of the nearest point on the CSI surface trace to a UTM point (m)."""
    f = csi.faults[fault_idx]
    xf = np.asarray(f.xf) * 1000.   # km -> m
    yf = np.asarray(f.yf) * 1000.
    best_s, best_d, acc = 0., np.inf, 0.
    for i in range(len(xf) - 1):
        ax, ay, bx, by = xf[i], yf[i], xf[i + 1], yf[i + 1]
        seg2 = (bx - ax) ** 2 + (by - ay) ** 2
        t = 0. if seg2 == 0 else np.clip(
            ((easting_m - ax) * (bx - ax) + (northing_m - ay) * (by - ay)) / seg2, 0., 1.)
        cx, cy = ax + t * (bx - ax), ay + t * (by - ay)
        d = np.hypot(easting_m - cx, northing_m - cy)
        seg = np.sqrt(seg2)
        if d < best_d:
            best_d, best_s = d, acc + t * seg
        acc += seg
    return best_s / 1000.


def seed_deep_slips(model, csi, altar, along_km):
    """Seed slip on the deep (non-inverted) patches from the AlTar strikeslipmain posterior."""
    vp = csi.vertical_profile(along_km, fault_idx=0)
    ss = altar.final["ParameterSets"]["strikeslipmain"]
    for i in range(N_SHALLOW_INVERTED, len(model.patches)):
        z = model.patches[i].z
        match = next((m for m in vp
                      if m["intersection_top"] * 1000. <= z <= m["intersection_bot"] * 1000.),
                     None)
        if match is not None:
            model.slips[i] = np.median(ss[:, match["patch_idx"]])
    return model


def build_model(profile_xs, csi, altar, along_km):
    """Construct the layered TwoDDzForwardModel with deep slip seeded from AlTar."""
    model = TwoDDzForwardModel()
    model.patches = [PatchTwoD(0., (VD[i] + VD[i + 1]) / 2., VD[i + 1] - VD[i])
                     for i in range(len(VD) - 1)]
    model.slips = np.zeros(len(model.patches))
    model.xs = profile_xs
    seed_deep_slips(model, csi, altar, along_km)
    return model


def prepare_data(profile):
    """Bin-average, pick the fault-parallel component, drop NaNs, estimate covariance, centre.

    Returns (xs, data, data_covariance) ready for the inversion, or None if degenerate.
    """
    binned = profile.bin_average(n_bins=N_BINS, n_near_fault_bins=N_NEAR_BINS,
                                 near_fault_dist=NEAR_FAULT_DIST)
    parallel = PARALLEL_SIGN * binned.displacements[0]
    finite = np.isfinite(parallel) & np.isfinite(binned.xs)
    xs = binned.xs[finite]
    data = parallel[finite].astype(float)
    if data.size < 40:
        return None

    # covariance from the far-field band at the negative-x end, detrended
    cd_est = data[:25]
    t = np.arange(cd_est.size)
    detrended = cd_est - np.polyval(np.polyfit(t, cd_est, 1), t)
    std = np.std(detrended)
    if not np.isfinite(std) or std == 0.:
        std = np.std(data) or 1e-3
    data_covariance = std ** 2 * np.eye(data.size)

    # centre (the new xs is already centred on the fault; only remove the data mean)
    data = data - np.mean(data)
    return xs, data, data_covariance


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


def plot_inversion(inversion, model, xs, data, path, title):
    """Second-pass figure: model fit (top) + posterior parameter distributions
    (bottom). Self-contained (matplotlib only) so it does not depend on the
    arviz plotting API, which varies between versions.
    """
    post = inversion.result.posterior
    labels = ["dz_halfwidth", "modulus_ratio", "slip0", "slip1", "slip2"]
    samp = {l: post[l].values.flatten() for l in labels}
    med = {l: np.median(s) for l, s in samp.items()}

    # best-fit model curve (posterior medians; deep slip kept at its seeded value)
    fit = model._copy()
    fit.dz_half_width = med["dz_halfwidth"]
    fit.modulus_ratio = med["modulus_ratio"]
    fit.slips = np.array(fit.slips, dtype=float)
    fit.slips[:N_SHALLOW_INVERTED] = [med["slip0"], med["slip1"], med["slip2"]]
    fit = fit.run(xs)

    fig = plt.figure(figsize=(11, 7), layout="constrained")
    gs = gridspec.GridSpec(2, 5, height_ratios=[2, 1], figure=fig)
    fig.suptitle(title)

    # --- slip vs depth ---
    ax_s = fig.add_subplot(gs[0, 0])
    depth, slip = [], []
    for i, p in enumerate(fit.patches):
        depth += [p.top, p.bottom]
        slip += [fit.slips[i]] * 2
    ax_s.plot(slip, depth, color="navy")
    ax_s.axhspan(0, fit.patches[N_SHALLOW_INVERTED - 1].bottom, color="gold",
                 alpha=0.15, label="inverted")
    ax_s.axvline(0, color="lightgray", ls="--")
    ax_s.invert_yaxis()
    ax_s.set_xlabel("Slip (m)")
    ax_s.set_ylabel("Depth (m)")
    ax_s.set_title("Slip (median)")
    ax_s.legend(fontsize=8)

    # --- data vs model fit ---
    ax_f = fig.add_subplot(gs[0, 1:])
    ax_f.plot(xs, data, color="0.5", lw=1.2, label="data (centred)")
    ax_f.plot(xs, fit.sol - np.mean(fit.sol), color="crimson", lw=1.5,
              label="model fit")
    ax_f.axvline(0, color="lightgray", ls="--")
    ax_f.set_xlabel("Distance from fault (m)")
    ax_f.set_ylabel("Fault-parallel displacement (m)")
    ax_f.set_title("Model fit")
    ax_f.legend()

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

    ####    2. INVERT    ####
    # csi.pickle unpickling needs the csi package (-> okada4py); import lazily so
    # the generation/plotting stage above works even where that is unavailable.
    print("[inv] loading CSI + AlTar (needed to seed deep slip) ...")
    from codes import CSIWrapper  # noqa: F401  (ensures csi class is importable)
    multi, faults, datasets, trans = pickle.load(open(CSI_PICKLE, "rb"))
    csi = CSIWrapper(multi, faults, datasets, trans)
    altar = AltarOutput(str(ALTAR_DIR))

    priors = [
        UniformDist("dz_halfwidth", 0., 1000.),
        UniformDist("modulus_ratio", 0.01, 0.9),
        UniformDist("slip0", -5., 0.),
        UniformDist("slip1", -10., 0.),
        UniformDist("slip2", -10., 0.),
    ]

    records = []
    for n, (i, profile) in enumerate(selected, 1):
        tag = f"profile {i} ({n}/{len(selected)})"
        try:
            print(f"\n[inv] === {tag} ===")
            prep = prepare_data(profile)
            if prep is None:
                print(f"[inv] {tag}: too few finite data points, skipping.")
                continue
            xs, data, Cd = prep

            east, north = profile.fault_utm
            along_km = csi_along_strike_km(csi, east, north)
            model = build_model(xs, csi, altar, along_km)
            print(f"[inv] {tag}: along-strike {along_km:.2f} km, {data.size} data pts, "
                  f"seeded deep slip {np.round(model.slips[N_SHALLOW_INVERTED:], 2)}")

            inversion = HamiltonianInversion(model, priors, data, Cd)
            inversion = inversion.run(draws=DRAWS, tune=TUNE, chains=CHAINS)

            with open(RESDIR / f"profile_{i:03d}.pickle", "wb") as f:
                pickle.dump(inversion, f)
            plot_inversion(inversion, model, xs, data, figdir / f"profile_{i:03d}.png",
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
        except Exception as e:
            print(f"[inv] {tag} FAILED: {type(e).__name__}: {e}")

    ####    3. SUMMARY    ####
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
