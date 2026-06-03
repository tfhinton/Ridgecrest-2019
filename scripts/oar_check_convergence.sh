#!/bin/bash
# Submit with : oarsub -S ./oar_check_convergence.sh

#OAR --name python_slip_inversion
#OAR -p network_address='ist-calcul30.u-ga.fr'
#OAR --resource /nodes=1/gpu=1/core=4,walltime=04:00:00
#OAR -t gpu
#OAR --project iste-equ-cycle
#OAR --stderr oar-checkconvergence.err
#OAR --stdout oar-checkconvergence.out
#OAR --notify mail:thomas.hinton@univ-grenoble-alpes.fr

source /soft/env.bash
cd /home/hintont/projects/ridgecrest
source .venv/bin/activate
export PYTHONPATH=/home/hintont/packages/okada4py-install:$PYTHONPATH
echo " running Python slip inversion "
python3.11 -u ./scripts/profile_inversion_check_convergence.py