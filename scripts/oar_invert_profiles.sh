#!/bin/bash
# Submit with : oarsub -S ./oar_invert_profiles.sh

#OAR --name python_slip_inversion
#OAR -p network_address='ist-calcul30.u-ga.fr'
#OAR --resource /nodes=1/gpu=1/core=6,walltime=08:00:00
#OAR -t gpu
#OAR --project iste-equ-cycle
#OAR --stderr oar.err
#OAR --stdout oar.out
#OAR --notify mail:thomas.hinton@univ-grenoble-alpes.fr

source /soft/env.bash
module load python/python3.11
cd /home/hintont/projects/ridgecrest
source .venv/bin/activate
echo " running Python slip inversion "
python3.11 ./scripts/invert_profiles.py