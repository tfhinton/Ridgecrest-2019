#!/bin/bash
# Submit with : oarsub -S ./oar_invert_profiles.sh

#OAR --name profile_and_inversion
#OAR --resource /nodes=1/core=6,walltime=08:00:00
#OAR --project iste-equ-cycle
#OAR --stderr oarfull.err
#OAR --stdout oarfull.out
#OAR --notify mail:thomas.hinton@univ-grenoble-alpes.fr

source /soft/env.bash
cd /home/hintont/projects/ridgecrest
source .venv/bin/activate
export PYTHONPATH=/home/hintont/packages/okada4py-install:$PYTHONPATH
echo " running Python slip inversion "
python -u scripts/invert_profiles_fault_aligned.py --mode full
