#!/bin/bash
# Submit with : oarsub -S ./oar_twod02_production.sh
#
# Production re-run of all 34 profiles, applying the lessons from the 2026-07-21 /
# 2026-07-27 experiments (see twod_inversion_handover.md):
#   - mr-lower 0.02 (was 0.2 -- the floor was a truncation artifact for several
#     profiles, e.g. profile 15's true modulus_ratio is ~0.14, not pinned at 0.20)
#   - dispersed-init (was PyMC's default small-jitter init -- profile 8's 128-chain
#     test showed the dz_halfwidth/modulus_ratio bimodality's dominant mode is often
#     missed by jittered chains landing near the same starting point)
#   - 4 slip patches: 0,400,1000,1800,3000 (was 3; the discretisation experiment found
#     no benefit beyond ~5 patches with this unregularised per-patch prior, so this is
#     a modest bump for resolution, not a jump to 10)
#
# --skip-done makes this resumable: if the 20h walltime is hit before all 34 profiles
# finish, just resubmit and it picks up where it left off.
# Results -> results/profile_inversion/twod03/ (does not touch twod01/ or any exp_*/).

#OAR --name twod03_production
#OAR --resource /nodes=1/core=8,walltime=20:00:00
#OAR --project iste-equ-cycle
#OAR --stderr oar_twod03_production.err
#OAR --stdout oar_twod03_production.out
#OAR --notify mail:thomas.hinton@univ-grenoble-alpes.fr

source /soft/env.bash
cd /home/hintont/projects/ridgecrest
source .venv/bin/activate
cd /data/cycle/hintont/projects/ridgecrest
export PYTHONPATH=/home/hintont/packages/okada4py-install:$PYTHONPATH
export OMP_NUM_THREADS=1   # chains parallelise across cores; keep BLAS single-threaded

LOGDIR="results/profile_inversion/logs"
mkdir -p "$LOGDIR"

echo "[twod03] started $(date)"
python3 -u scripts/invert_twod_profiles.py --skip-done \
    --mr-lower 0.02 --dispersed-init --chains 8 --target-accept 0.95 \
    --sheath --draws 800 --tune 400 \
    --vd "0,400,1000,1800,3000" --timeout 45 \
    --outtag twod03 \
    > "$LOGDIR/twod03_production.log" 2>&1
echo "[twod03] finished $(date)"
