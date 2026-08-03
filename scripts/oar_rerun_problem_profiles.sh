#!/bin/bash
# Submit with : oarsub -S ./oar_rerun_problem_profiles.sh
#
# Re-run 4 profiles from the twod01 weekend run under different sampler conditions,
# each targeting a specific diagnosis from the results (see twod_inversion_handover.md
# for the full writeup). Doesn't touch results/profile_inversion/twod01/ -- each
# condition gets its own --outtag subdir.
#
#   profile 15 (fault 0, 38.5 km): clean single-mode convergence but mr pinned exactly
#     at the 0.2 floor -- does it want to go lower, or is 0.2 close to the truth?
#       --mr-lower 0.02
#   profile 24 (fault 0, 50.3 km): strong dz_halfwidth/modulus_ratio bimodality
#     (two well-separated posterior clusters) -- does fixing mr collapse it to one
#     well-determined dz, with the fit still acceptable?
#       --fix-mr 0.5
#   profile 8  (fault 0, 24.3 km): the worst bimodality in the run (rhat up to 111) --
#     does a large population of independently-initialised chains reveal which mode
#     actually dominates, rather than 8 chains landing wherever jitter put them?
#       --dispersed-init --chains 128 --draws 300 --tune 500
#   profile 7  (fault 0, 21.8 km): converges tightly but fits badly (chi2_red ~ 190)
#     with a real dip (61/67 deg) -- does letting dip float resolve the near/far-field
#     shape mismatch, or is this a genuine data/geometry issue outside the model?
#       --free-dip

#OAR --name twod_rerun_problem_profiles
#OAR --resource /nodes=1/core=8,walltime=06:00:00
#OAR --project iste-equ-cycle
#OAR --stderr oar_rerun_problem_profiles.err
#OAR --stdout oar_rerun_problem_profiles.out
#OAR --notify mail:thomas.hinton@univ-grenoble-alpes.fr

source /soft/env.bash

cd /home/hintont/projects/ridgecrest
source .venv/bin/activate
cd /data/cycle/hintont/projects/ridgecrest
export PYTHONPATH=/home/hintont/packages/okada4py-install:$PYTHONPATH
export OMP_NUM_THREADS=1   # chains parallelise across cores; keep BLAS single-threaded

LOGDIR="results/profile_inversion/logs"
mkdir -p "$LOGDIR"

run() {
    local name="$1"; shift
    echo "[rerun] $name started $(date)"
    python3 -u scripts/invert_twod_profiles.py --skip-done "$@" > "$LOGDIR/${name}.log" 2>&1
    echo "[rerun] $name finished $(date)"
}

run mrfloor002_p15 --profile 15 --mr-lower 0.02 \
    --outtag exp_mrfloor002

run fixmr05_p24 --profile 24 --fix-mr 0.5 \
    --outtag exp_fixmr05

run dispersed128_p8 --profile 8 --dispersed-init --chains 128 --draws 300 --tune 500 \
    --target-accept 0.95 --timeout 180 \
    --outtag exp_dispersed128

run freedip_p7 --profile 7 --free-dip \
    --outtag exp_freedip

echo "[rerun] all done $(date)"
