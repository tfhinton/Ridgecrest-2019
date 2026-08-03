#!/usr/bin/env bash
set -euo pipefail

# Push the 2D profile inversion inputs to the cluster, or fetch its results back.
#
#   ./scripts/transfer_twod_inversion.sh push    # before the run
#   ./scripts/transfer_twod_inversion.sh fetch   # Monday morning
#
# NB the codes package syncs via git (PhD-Shared-Codes) -- push/pull it separately.
# The dipping forward model lives in codes/src/codes/TwoDDzForwardModel.py + Patch.py.

MODE="${1:?usage: $0 push|fetch}"

LOCAL_BASE="/Users/hintont/Dev/projects/ridgecrest"
REMOTE_USER="hintont"
REMOTE_HOST="ist-oar.u-ga.fr"
REMOTE_BASE="/data/cycle/hintont/projects/ridgecrest"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

if [[ "$MODE" == "push" ]]; then
    ssh "$REMOTE" "mkdir -p '${REMOTE_BASE}/results/working/tmp' \
                            '${REMOTE_BASE}/data/fault' \
                            '${REMOTE_BASE}/scripts' \
                            '${REMOTE_BASE}/results/profile_inversion'"
    rsync -avz --progress \
        "${LOCAL_BASE}/results/working/tmp/deep_removed_profiles.pickle" \
        "${LOCAL_BASE}/results/working/tmp/evaluated_profiles.pickle" \
        "${REMOTE}:${REMOTE_BASE}/results/working/tmp/"
    rsync -avz --progress \
        "${LOCAL_BASE}/data/fault/ridgecrest_faults.pickle" \
        "${REMOTE}:${REMOTE_BASE}/data/fault/"
    rsync -avz --progress \
        "${LOCAL_BASE}/scripts/invert_twod_profiles.py" \
        "${LOCAL_BASE}/scripts/twod_convergence_test.py" \
        "${LOCAL_BASE}/scripts/run_twod_weekend.sh" \
        "${LOCAL_BASE}/scripts/oar_rerun_problem_profiles.sh" \
        "${LOCAL_BASE}/scripts/oar_rerun_discretisation.sh" \
        "${LOCAL_BASE}/scripts/oar_twod02_production.sh" \
        "${LOCAL_BASE}/scripts/oar_twod03_production.sh" \
        "${LOCAL_BASE}/scripts/result_io.py" \
        "${LOCAL_BASE}/scripts/config.py" \
        "${REMOTE}:${REMOTE_BASE}/scripts/"
    ssh "$REMOTE" "chmod +x '${REMOTE_BASE}/scripts/run_twod_weekend.sh' \
                            '${REMOTE_BASE}/scripts/oar_rerun_problem_profiles.sh' \
                            '${REMOTE_BASE}/scripts/oar_rerun_discretisation.sh' \
                            '${REMOTE_BASE}/scripts/oar_twod02_production.sh' \
                            '${REMOTE_BASE}/scripts/oar_twod03_production.sh'"
    echo
    echo "Done. On the cluster (with codes pulled + env active), launch one of:"
    echo "  nohup ./scripts/run_twod_weekend.sh > /dev/null 2>&1 &"
    echo "  (ORIGINAL production settings: convergence study on profiles 15 + 28,"
    echo "   then all 34 profiles; results/profile_inversion/twod01/)"
    echo "  oarsub -S ./scripts/oar_rerun_problem_profiles.sh"
    echo "  (4 problem profiles re-run under different conditions; results in"
    echo "   results/profile_inversion/exp_*/)"
    echo "  oarsub -S ./scripts/oar_rerun_discretisation.sh"
    echo "  (3 profiles x 3 slip discretisations; results in"
    echo "   results/profile_inversion/exp_disc{3,5,10}/)"
    echo "  oarsub -S ./scripts/oar_twod02_production.sh"
    echo "  (NEW production settings, all 34 profiles: mr-lower 0.02, dispersed-init,"
    echo "   4 slip patches; results/profile_inversion/twod02/)"

elif [[ "$MODE" == "fetch" ]]; then
    rsync -avz --progress \
        "${REMOTE}:${REMOTE_BASE}/results/profile_inversion/" \
        "${LOCAL_BASE}/results/profile_inversion/"
    echo "Done. Results in ${LOCAL_BASE}/results/profile_inversion/{twod01,twod02,exp_*}"

else
    echo "usage: $0 push|fetch" >&2
    exit 1
fi
