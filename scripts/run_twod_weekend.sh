#!/usr/bin/env bash
# Weekend run: convergence study on two profiles first, then the full 2D profile
# inversion (production settings are the invert_twod_profiles.py defaults:
# 8 chains, 4000 draws, 2000 tune). Logs land in results/profile_inversion/twod01/
# so `transfer_twod_inversion.sh fetch` brings them back with the results.
#
# On the cluster:  nohup ./scripts/run_twod_weekend.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1   # chains parallelise across cores; keep BLAS single-threaded

LOGDIR="results/profile_inversion/twod01"
mkdir -p "$LOGDIR"

echo "[weekend] convergence study started $(date)"
python scripts/twod_convergence_test.py > "$LOGDIR/convergence.log" 2>&1

echo "[weekend] full inversion started $(date)"
python scripts/invert_twod_profiles.py --skip-done > "$LOGDIR/inversion.log" 2>&1

echo "[weekend] finished $(date)"
