#!/bin/bash
# Submit with : oarsub -S ./oar_rerun_discretisation.sh
#
# Slip-discretisation sensitivity test: 3 profiles x 3 depth-patch schemes, ALL under
# matching sampler settings (mr-lower 0.02, dispersed-init, 8 chains) so discretisation
# is the only thing varying between arms -- see twod_inversion_handover.md. Deliberately
# re-runs the current 3-patch scheme under these settings too, rather than comparing
# against the original twod01 result (mr-lower=0.2, non-dispersed init), which would
# confound discretisation with the sampler-settings change.
#
#   patches3  (current default): 0,500,1500,3000               -- 3 slip parameters
#   patches5:                    0,100,350,800,1700,3000       -- 5 slip parameters
#   patches10: 0,100,200,300,500,700,1000,1400,1800,2300,3000  -- 10 slip parameters
#
# Profiles: 15 (f0, well-fit, mr near the old 0.2 floor), 0 (f0, well-fit, mr well
# interior ~0.6), 28 (f1 foreshock strand, weak signal) -- chosen for good baseline
# fit/convergence so discretisation effects aren't confounded with the dz/mr
# multimodality or misfit issues seen on other profiles.

#OAR --name twod_rerun_discretisation
#OAR --resource /nodes=1/core=8,walltime=08:00:00
#OAR --project iste-equ-cycle
#OAR --stderr oar_rerun_discretisation.err
#OAR --stdout oar_rerun_discretisation.out
#OAR --notify mail:thomas.hinton@univ-grenoble-alpes.fr

source /soft/env.bash
cd /home/hintont/projects/ridgecrest
source .venv/bin/activate
cd /data/cycle/hintont/projects/ridgecrest
export PYTHONPATH=/home/hintont/packages/okada4py-install:$PYTHONPATH
export OMP_NUM_THREADS=1   # chains parallelise across cores; keep BLAS single-threaded

LOGDIR="results/profile_inversion/logs"
mkdir -p "$LOGDIR"

VD3="0,500,1500,3000"
VD5="0,100,350,800,1700,3000"
VD10="0,100,200,300,500,700,1000,1400,1800,2300,3000"

run() {
    local name="$1"; shift
    echo "[rerun] $name started $(date)"
    python3 -u scripts/invert_twod_profiles.py --skip-done --mr-lower 0.02 \
        --dispersed-init --chains 8 --target-accept 0.95 "$@" \
        > "$LOGDIR/${name}.log" 2>&1
    echo "[rerun] $name finished $(date)"
}

for p in 15 0 28; do
    run disc3_p${p}  --profile $p --vd "$VD3"  --outtag exp_disc3
    run disc5_p${p}  --profile $p --vd "$VD5"  --outtag exp_disc5
    run disc10_p${p} --profile $p --vd "$VD10" --outtag exp_disc10
done

echo "[rerun] all done $(date)"
