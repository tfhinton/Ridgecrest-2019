#!/bin/bash
cd /Users/hintont/Dev/projects/ridgecrest/results/inversion_results
mkdir -p $1/outputs $1/inputs

sftp hintont@ist-oar.u-ga.fr << EOF
  get /data/cycle/hintont/projects/ridgecrest/inversion_results/record.md ./record.md
  get /data/cycle/hintont/projects/ridgecrest/inversion_results/$1/outputs/step_final.h5 ./$1/outputs/step_final.h5
  get /data/cycle/hintont/projects/ridgecrest/inversion_results/$1/inputs/csi.pickle ./$1/inputs/csi.pickle
EOF
