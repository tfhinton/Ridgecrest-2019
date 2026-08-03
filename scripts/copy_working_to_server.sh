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

LOCAL_INPUTS="${LOCAL_BASE}/${DIR_NAME}/inputs"
REMOTE_TARGET_DIR="${REMOTE_BASE}/${DIR_NAME}"

# ---- Sanity checks ----
if [[ ! -d "$LOCAL_INPUTS" ]]; then
    echo "Error: local inputs directory not found: $LOCAL_INPUTS" >&2
    exit 1
fi

# ---- Make sure remote dir exists ----
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_TARGET_DIR}'"

# ---- Copy only the inputs folder ----
rsync -avz --progress \
    "$LOCAL_INPUTS" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TARGET_DIR}/"

echo "Done. Copied ${LOCAL_INPUTS} -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TARGET_DIR}/inputs"