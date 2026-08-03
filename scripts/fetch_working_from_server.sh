#!/usr/bin/env bash
set -euo pipefail

# ---- Argument handling ----
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <dir_name>" >&2
    echo "Example: $0 01" >&2
    exit 1
fi
DIR_NAME="$1"

# ---- Configuration ----
LOCAL_BASE="/Users/hintont/Dev/projects/ridgecrest/results"
REMOTE_USER="hintont"
REMOTE_HOST="ist-oar.u-ga.fr"
REMOTE_BASE="/data/cycle/hintont/projects/ridgecrest/results"

REMOTE_OUTPUTS="${REMOTE_BASE}/${DIR_NAME}/outputs"
LOCAL_TARGET_DIR="${LOCAL_BASE}/${DIR_NAME}"

# ---- Make sure local dir exists ----
mkdir -p "$LOCAL_TARGET_DIR"

# ---- Fetch only the outputs folder ----
rsync -avz --progress \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_OUTPUTS}" \
    "${LOCAL_TARGET_DIR}/"

echo "Done. Fetched ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_OUTPUTS} -> ${LOCAL_TARGET_DIR}/outputs"