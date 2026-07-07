#!/usr/bin/env python3
"""Quick-look: fault-aligned optical profile + deep-slip removal, NO inversion.

Give one or more along-fault distances (km, measured along the main-fault trace
used by invert_profiles_fault_aligned.py). For each one the script

  1. gets the stacked fault-aligned profile there -- either freshly generated
     from a short trace segment around the requested point (fast: cost scales
     with the segment, not the whole fault), or, with --cache, the nearest
     already-generated profile from a profiles_*.pickle,
  2. bins it exactly as the inversion script would (same N_BINS / near-fault
     layout / PARALLEL_SIGN),
  3. forward-predicts the deep (> DEEP_DEPTH_M) AlTar posterior-mode slip at the
     binned points and subtracts it,
  4. writes a figure with the raw profile, the binned signed data, the deep-slip
     model, and the residual, and prints centring diagnostics (far-field datum,
     data mean, step midpoint x).

Everything convention-critical (PARALLEL_SIGN, binning, DEEP_DEPTH_M, posterior
mode, GF batching) is imported from invert_profiles_fault_aligned.py, so what
you see here is exactly what the inversion would invert.

Examples:
  python scripts/profile_quicklook.py 22.5
  python scripts/profile_quicklook.py 10 20 30 --cache
  python scripts/profile_quicklook.py 22.5 --plen 2000     # cheaper short profile
  python scripts/profile_quicklook.py 22.5 --no-deep       # skip deep-slip removal
"""

####    IMPORTS    ####
import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas as gpd
from shapely.geometry import box, LineString
from shapely.ops import substring

try:
    import cmcrameri.cm as cmc
    CMAP_DIV = cmc.vik
except ImportError:
    CMAP_DIV = "RdBu_r"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import invert_profiles_fault_aligned as inv

OUTDIR_DEFAULT = inv.ROOT / "results/profile_quicklook"
CACHE_DEFAULT = inv.RESDIR / "profiles_full.pickle"
# forked worker pools are only safe on Linux; default to serial elsewhere
N_JOBS_DEFAULT = inv.N_JOBS if sys.platform.startswith("linux") else 1


####    PROFILE ACQUISITION    ####
def generate_profile_at(s_m, plen, n_jobs, shift_cap=8, shift_cap_final=4,
                        search_half_width=None):
    """Freshly extract the stacked profile nearest along-fault distance s_m (m).

    Uses a short trace segment centred on s_m (like the inversion script's test
    mode) so only ~2*STACK+120 profiles are sampled and the optical window stays
    small. Returns (profile, opt): the central stacked Profile with
    x_along_fault made absolute (i.e. measured along the FULL main-fault trace,
    not the segment), and the loaded OpticalData window (for the map panel).
    """
    fault = inv.load_main_fault()
    ls = fault.trace.geometry[0]
    if not 0. <= s_m <= ls.length:
        raise ValueError(f"along-fault distance {s_m / 1000:.2f} km outside "
                         f"trace [0, {ls.length / 1000:.2f}] km")
    seg_len = 2 * inv.STACK + 120.
    s0 = float(np.clip(s_m - seg_len / 2., 0., ls.length - seg_len))
    seg = substring(ls, s0, s0 + seg_len).segmentize(25.)
    fault.trace = gpd.GeoDataFrame(geometry=[LineString(seg.coords)],
                                   crs=fault.trace.crs)

    margin = plen + inv.STACK + 800.
    minx, miny, maxx, maxy = seg.bounds
    bbox = box(minx - margin, miny - margin, maxx + margin, maxy + margin)
    print(f"[gen] segment [{s0 / 1000:.2f}, {(s0 + seg_len) / 1000:.2f}] km, "
          f"loading optical window ...")
    opt = inv.load_optical_window(bbox)

    t0 = time.time()
    profiles = opt.evaluate_profiles_fault_aligned(
        fault, plen=plen, stack=inv.STACK, trace_smooth=inv.TRACE_SMOOTH,
        strain_half_width=inv.STRAIN_HALF_WIDTH, n_jobs=n_jobs,
        prof_dtype=inv.PROF_DTYPE, attach_to_fault=False, store=False,
        shift_cap=shift_cap, shift_cap_final=shift_cap_final,
        search_half_width=search_half_width)
    print(f"[gen] {len(profiles)} stacked profiles in {time.time() - t0:.0f} s")
    p = profiles[len(profiles) // 2]
    # x_along_fault from the segment run is measured along the segment; the
    # segment starts at s0 on the full trace (segmentize changes length by <<1%)
    p.x_along_fault += s0
    return p, opt


def profile_from_cache(cache_path, s_m):
    """Nearest cached stacked profile to along-fault distance s_m (m)."""
    with open(cache_path, "rb") as f:
        profiles = pickle.load(f)
    along = np.array([p.x_along_fault for p in profiles])
    i = int(np.argmin(np.abs(along - s_m)))
    print(f"[cache] nearest of {len(profiles)} profiles: index {i} at "
          f"{along[i] / 1000:.2f} km (requested {s_m / 1000:.2f} km)")
    return profiles[i]


def profile_map_layer(opt, profile, margin=400., max_px=900):
    """Small display copy of the EW scene around the profile line.

    opt : OpticalData whose window covers the profile (generate mode reuses the
        extraction window; cache mode loads a thin corridor). Returns
        (arr, extent) with extent in UTM km, ready for imshow (origin upper).
    """
    x0, y0, x1, y1 = profile.linestring.bounds
    win = opt.get_window(x0 - margin, x1 + margin,
                         y1 + margin, y0 - margin)   # y slices high -> low
    k = max(1, int(np.ceil(max(win.ew.shape) / max_px)))
    ew = win.ew.isel(y=slice(0, None, k), x=slice(0, None, k))
    arr = np.asarray(ew.values, dtype=float)
    x, y = ew.x.values, ew.y.values
    extent = (x[0] / 1e3, x[-1] / 1e3, y[-1] / 1e3, y[0] / 1e3)
    return arr, extent


def load_map_window(profile, margin=400.):
    """Corridor optical window around a cached profile (map panel only)."""
    x0, y0, x1, y1 = profile.linestring.bounds
    return inv.load_optical_window(box(x0 - margin, y0 - margin,
                                       x1 + margin, y1 + margin))


####    DIAGNOSTICS    ####
def centring_diagnostics(xs, resid_signed, n_band=25):
    """Datum / centring numbers for a deep-removed (uncentred) residual.

    datum      : symmetric far-field level, 0.5 * (left band + right band) --
                 the even far-field component the odd forward model cannot fit.
    mean       : plain mean over all bins (what finalise_data subtracts).
    step       : right band minus left band.
    x_cross    : x where the datum-centred residual crosses zero nearest the
                 fault, i.e. the apparent step midpoint (should be ~0 if the
                 strain relocation put x=0 on the fault).
    """
    left = float(np.mean(resid_signed[:n_band]))
    right = float(np.mean(resid_signed[-n_band:]))
    datum = 0.5 * (left + right)
    centred = resid_signed - datum
    cross = np.where(np.diff(np.sign(centred)) != 0)[0]
    if cross.size:
        k = cross[np.argmin(np.abs(xs[cross]))]
        x_cross = float(xs[k] - centred[k] * (xs[k + 1] - xs[k])
                        / (centred[k + 1] - centred[k]))
    else:
        x_cross = np.nan
    return dict(left=left, right=right, datum=datum,
                mean=float(np.mean(resid_signed)),
                step=right - left, x_cross=x_cross)


####    FIGURE    ####
def plot_quicklook(profile, xs, data_signed, pred_signed, diag, map_layer,
                   trace, path, title):
    """Top: location map (EW scene + profile line). Middle: raw stacked
    profile. Bottom: binned data, deep model, residual."""
    fig = plt.figure(figsize=(9, 12), layout="constrained")
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.5, 1, 1])
    fig.suptitle(title)

    # --- location map: EW displacement + profile geometry ---
    ax_m = fig.add_subplot(gs[0])
    arr, extent = map_layer
    vmax = float(np.nanpercentile(np.abs(arr), 98))
    im = ax_m.imshow(arr, extent=extent, vmin=-vmax, vmax=vmax,
                     cmap=CMAP_DIV, interpolation="nearest")
    if trace is not None:
        for gi, geom in enumerate(trace.geometry):
            c = np.asarray(geom.coords) / 1e3
            ax_m.plot(c[:, 0], c[:, 1], color="0.15", lw=0.8, ls=":",
                      label="drawn trace" if gi == 0 else None)
    c = np.asarray(profile.linestring.coords) / 1e3
    ax_m.plot(c[:, 0], c[:, 1], color="black", lw=1.4, label="profile")
    for end, lbl in ((c[0], "$-x$"), (c[-1], "$+x$")):
        ax_m.annotate(lbl, end, textcoords="offset points", xytext=(4, 4),
                      fontsize=9)
    fx, fy = np.asarray(profile.fault_utm) / 1e3
    ax_m.plot(fx, fy, "o", ms=7, mfc="none", mec="black", mew=1.4,
              label="drawn-trace crossing (x ≈ 0)")
    ax_m.set_xlim(extent[0], extent[1])
    ax_m.set_ylim(extent[2], extent[3])
    ax_m.set_aspect("equal")
    ax_m.set_xlabel("UTM E (km)")
    ax_m.set_ylabel("UTM N (km)")
    ax_m.set_title("Profile location", fontsize=10)
    ax_m.legend(fontsize=8, loc="lower left")
    fig.colorbar(im, ax=ax_m, shrink=0.9, pad=0.01,
                 label="E–W displacement (m)")

    # --- raw profile (full resolution) ---
    ax_r = fig.add_subplot(gs[1])
    ax_b = fig.add_subplot(gs[2], sharex=ax_r)
    ax_r.plot(profile.xs, inv.PARALLEL_SIGN * profile.displacements[0], lw=1.,
              color="crimson", label="fault-parallel (signed)")
    ax_r.plot(profile.xs, profile.displacements[1], lw=1., color="steelblue",
              alpha=0.7, label="fault-normal")
    ax_r.axvline(0., color="lightgray", ls="--")
    ax_r.set_ylabel("Displacement (m)")
    ax_r.set_title("Raw stacked profile", fontsize=10)
    ax_r.legend(fontsize=8)

    # --- binned data, deep model, residual (NO vertical adjustment: the
    #     residual curve is exactly data minus deep model) ---
    resid = data_signed - pred_signed
    ax_b.plot(xs, data_signed, ".-", ms=3, lw=0.8, color="0.55",
              label="binned data (signed)")
    ax_b.plot(xs, pred_signed, lw=1.5, ls="--", color="seagreen",
              label="deep-slip model (AlTar mode)")
    ax_b.plot(xs, resid, lw=1.2, color="crimson",
              label="residual (= data − deep model)")
    ax_b.axvline(0., color="lightgray", ls="--")
    ax_b.axhline(0., color="lightgray", ls="--")
    if np.isfinite(diag["x_cross"]):
        ax_b.axvline(diag["x_cross"], color="crimson", ls=":", lw=1.,
                     label=f"step midpoint x = {diag['x_cross']:+.0f} m")
    ax_b.set_xlabel("Distance from fault (m)")
    ax_b.set_ylabel("Fault-parallel displacement (m)")
    ax_b.set_title("Deep-slip removal (binned, signed frame)", fontsize=10)
    ax_b.legend(fontsize=8)
    box_txt = (f"far-field: left {diag['left']:+.2f}, right {diag['right']:+.2f} m\n"
               f"datum (sym. far-field) {diag['datum']:+.2f} m, "
               f"mean {diag['mean']:+.2f} m\n"
               f"residual step {diag['step']:+.2f} m")
    ax_b.text(0.02, 0.03, box_txt, transform=ax_b.transAxes, fontsize=8,
              va="bottom", bbox=dict(fc="white", alpha=0.8, ec="0.7"))

    fig.savefig(path, dpi=200)
    plt.close(fig)


####    MAIN    ####
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("along_km", type=float, nargs="+",
                    help="along-fault distance(s) in km on the main-fault trace")
    ap.add_argument("--cache", nargs="?", const=str(CACHE_DEFAULT), default=None,
                    metavar="PICKLE",
                    help="pick the nearest profile from a cached profiles pickle "
                         f"instead of generating (default file: {CACHE_DEFAULT})")
    ap.add_argument("--plen", type=int, default=inv.PLEN,
                    help="profile half-length in m when generating (default %(default)s)")
    ap.add_argument("--n-jobs", type=int, default=N_JOBS_DEFAULT,
                    help="extraction workers when generating (default %(default)s; "
                         "forked pool, >1 is Linux-only)")
    ap.add_argument("--shift-cap", type=float, default=8,
                    help="max first-pass relocation shift in px (=m) when "
                         "generating (default %(default)s)")
    ap.add_argument("--shift-cap-final", type=float, default=20,
                    help="max second-pass relocation shift in px (default %(default)s)")
    ap.add_argument("--search-half-width", type=float,
                    default=inv.SEARCH_HALF_WIDTH,
                    help="half-width (px) of the peak-strain search window used "
                         "to realign each profile on the refined (strain-peak) "
                         "fault location (default %(default)s, matching the "
                         "inversion script). Must stay below the distance to "
                         "the nearest parallel strand")
    ap.add_argument("--no-deep", action="store_true",
                    help="skip the deep-slip removal (prediction set to zero)")
    ap.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    ####    1. profiles    ####
    trace = inv.load_main_fault().trace   # map-panel context
    entries = []   # (s_m, profile, xs, data_signed, (pts, s_hat), map_layer)
    for km in args.along_km:
        s_m = km * 1000.
        if args.cache:
            profile = profile_from_cache(args.cache, s_m)
            opt = load_map_window(profile)
        else:
            profile, opt = generate_profile_at(
                s_m, args.plen, args.n_jobs, args.shift_cap,
                args.shift_cap_final, args.search_half_width)
        map_layer = profile_map_layer(opt, profile)
        del opt   # only the small display copy is kept
        pre = inv.prepare_data(profile)
        if pre is None:
            print(f"[skip] {km:.2f} km: too few finite data points.")
            continue
        xs, data_signed = pre
        entries.append((s_m, profile, xs, data_signed,
                        inv.profile_points(profile, xs), map_layer))
    if not entries:
        print("[done] no usable profiles.")
        return

    ####    2. deep-slip prediction (one batched GF evaluation)    ####
    if args.no_deep:
        preds = [np.zeros_like(e[2]) for e in entries]
    else:
        tri_fault = inv.load_triangle_fault()
        ss_mode, ds_mode = inv.posterior_mode_deep_slips(tri_fault)
        preds = inv.deep_predictions(tri_fault, ss_mode, ds_mode,
                                     [e[4] for e in entries])

    ####    3. figures + diagnostics    ####
    for (s_m, profile, xs, data_signed, _, map_layer), pred_par in \
            zip(entries, preds):
        along_km = profile.x_along_fault / 1000.
        tag = f"{along_km:.2f} km"
        pred_signed = inv.PARALLEL_SIGN * np.asarray(pred_par, dtype=float)
        diag = centring_diagnostics(xs, data_signed - pred_signed)
        print(f"[diag] {tag}: far-field left {diag['left']:+.2f} m / right "
              f"{diag['right']:+.2f} m -> datum {diag['datum']:+.2f} m "
              f"(plain mean {diag['mean']:+.2f} m), residual step "
              f"{diag['step']:+.2f} m, step midpoint x = {diag['x_cross']:+.1f} m")

        stem = f"quicklook_{along_km:06.2f}km"
        plot_quicklook(profile, xs, data_signed, pred_signed, diag,
                       map_layer, trace, args.outdir / f"{stem}.png",
                       f"Profile at {tag} along-strike"
                       + ("  [deep removal OFF]" if args.no_deep else ""))
        with open(args.outdir / f"{stem}.pickle", "wb") as f:
            pickle.dump(dict(profile=profile, xs=xs, data_signed=data_signed,
                             pred_signed=pred_signed, diag=diag), f)
        print(f"[fig ] wrote {args.outdir / (stem + '.png')}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
