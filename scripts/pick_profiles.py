#!/usr/bin/env python3
'''Quick across-fault profiles along the Ridgecrest rupture + interactive picking.

Takes ~N_PROFILES fault-perpendicular profiles along the main and secondary
fault traces, evaluates each with the quick swathe scheme, then opens the
interactive picker: keep/reject each profile by eye (drag to keep only part of
its extent). Kept profiles go to PICKED_PICKLE for accurate re-evaluation
(fault-aligned profiling) and inversion later.

Evaluated profiles and the minimap background are cached in TMP_DIR; pass
--regenerate to rebuild them.
'''
import argparse
import pickle
import numpy as np
import rasterio

from codes import OpticalData
from codes.profile_picking import (profiles_along_trace, evaluate_profiles_quick,
                                   ProfilePicker, plot_picked_profiles)
import config

N_PROFILES        = [100, 30]   # per strand: [main, secondary]
HALF_LENGTH       = 4000.   # m each side of the fault
SWATHE_HALF_WIDTH = 150.    # m
N_BINS            = 400
MINIMAP_DECIMATE  = 64

QUICK_PICKLE  = config.TMP_DIR / 'quick_profiles.pickle'
MINIMAP_NPZ   = config.TMP_DIR / 'optical_minimap.npz'
PICKED_PICKLE = config.TMP_DIR / 'picked_profiles.pickle'
PICKED_FIG    = config.TMP_DIR / 'picked_profiles.png'


def get_profiles(trace, regenerate):
    if QUICK_PICKLE.exists() and not regenerate:
        print(f'loading cached profiles from {QUICK_PICKLE}')
        return pickle.load(open(QUICK_PICKLE, 'rb'))
    opt = OpticalData(ew_filepath=str(config.EW_TIF), ns_filepath=str(config.NS_TIF),
                      verbose=False)
    profiles = profiles_along_trace(trace, n_profiles=N_PROFILES,
                                    half_length=HALF_LENGTH)
    print(f'evaluating {len(profiles)} profiles '
          f'(swathe half-width {SWATHE_HALF_WIDTH:.0f} m) ...')
    profiles = evaluate_profiles_quick(opt, profiles,
                                       swathe_half_width=SWATHE_HALF_WIDTH,
                                       n_bins=N_BINS)
    pickle.dump(profiles, open(QUICK_PICKLE, 'wb'))
    print(f'cached to {QUICK_PICKLE}')
    return profiles


def get_minimap(regenerate):
    if MINIMAP_NPZ.exists() and not regenerate:
        z = np.load(MINIMAP_NPZ)
        return z['bg'], z['extent']
    print('building minimap (one decimated read of the full EW scene) ...')
    with rasterio.open(config.EW_TIF) as src:
        bg = src.read(1, out_shape=(src.height // MINIMAP_DECIMATE,
                                    src.width // MINIMAP_DECIMATE))
        b = src.bounds
        extent = np.array([b.left, b.right, b.bottom, b.top])
    bg[bg == 0.] = np.nan
    np.savez(MINIMAP_NPZ, bg=bg, extent=extent)
    return bg, extent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--regenerate', action='store_true',
                    help='rebuild the cached profiles and minimap')
    args = ap.parse_args()

    fault = pickle.load(open(config.FAULT_PICKLE, 'rb'))
    trace = fault.trace

    profiles = get_profiles(trace, args.regenerate)
    bg, extent = get_minimap(args.regenerate)

    picker = ProfilePicker(profiles, save_path=PICKED_PICKLE,
                           background=bg, bg_extent=extent, trace=trace)
    picked = picker.run()
    if picked:
        plot_picked_profiles(picked, background=bg, bg_extent=extent,
                             trace=trace, path=PICKED_FIG)
        print(f'wrote map of kept profiles to {PICKED_FIG}')
    print(f'done: {len(picked)} profiles kept.')


if __name__ == '__main__':
    main()
