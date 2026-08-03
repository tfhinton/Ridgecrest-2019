#!/usr/bin/env python3
'''Accurate re-evaluation of the picked profiles.

Takes the interactively picked profiles (pick_profiles.py) and re-evaluates
each with the fault-aligned scheme (codes.fault_aligned_profiles): profiles
perpendicular to the local strike of the drawn trace, relocated onto the peak
shear strain, and median-stacked +/-STACK m along strike. The picked profile's
fault_utm / x_along_fault give the expected fault location the strain search
is centred on; the relocation can correct it by up to SEARCH_HALF_WIDTH m.

Output: list of evaluated Profiles (xs=0 on the relocated fault,
displacements=[fault-parallel, fault-normal], analysis metadata, original pick
attached as .picked) pickled to TMP_DIR/evaluated_profiles.pickle. Step
through them with check_evaluated_profiles.py.
'''
import pickle
import time

from codes import OpticalData
from codes.fault_aligned_profiles import evaluate_picked_profiles
import config

PLEN              = 4000    # m half-length; matches pick_profiles HALF_LENGTH
STACK             = 150     # along-strike half-window for stacking (m)
TRACE_SMOOTH      = 15
STRAIN_HALF_WIDTH = 200.    # m; must exceed SEARCH_HALF_WIDTH (search is clamped to it)
SEARCH_HALF_WIDTH = 100.    # m; narrow strain search around the step-refined position
STEP_SEARCH_HALF_WIDTH = 350.  # m; stage-0 refinement onto the displacement step
                               # (robust to off-fault strain; handles imprecise trace)

PICKED_PICKLE    = config.TMP_DIR / 'picked_profiles.pickle'
EVALUATED_PICKLE = config.TMP_DIR / 'evaluated_profiles.pickle'


def main():
    picked = pickle.load(open(PICKED_PICKLE, 'rb'))
    fault = pickle.load(open(config.FAULT_PICKLE, 'rb'))
    opt = OpticalData(ew_filepath=str(config.EW_TIF), ns_filepath=str(config.NS_TIF),
                      verbose=False)

    print(f'evaluating {len(picked)} picked profiles '
          f'(plen={PLEN}, stack={STACK}) ...')
    t0 = time.time()
    evaluated = evaluate_picked_profiles(
        opt, fault, picked, plen=PLEN, stack=STACK, trace_smooth=TRACE_SMOOTH,
        strain_half_width=STRAIN_HALF_WIDTH, search_half_width=SEARCH_HALF_WIDTH,
        step_search_half_width=STEP_SEARCH_HALF_WIDTH)
    print(f'done in {time.time() - t0:.0f} s')

    pickle.dump(evaluated, open(EVALUATED_PICKLE, 'wb'))
    print(f'wrote {len(evaluated)} evaluated profiles to {EVALUATED_PICKLE}')


if __name__ == '__main__':
    main()
