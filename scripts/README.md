Workflow:
1) Build fault mesh: mesh_fault.py
2) Pre-process far-field data: prep_files_for_altar.py
3) Quick least-squares inversion: quick_lsq_far_field_inversion.py
4) Pass to Altar for Bayesian inversion:
   copy_working_to_server.sh
   fetch_working_from_server.sh
5) Visualise Altar results: plot_altar_output.py
6) Interactive profile picking: pick_profiles.py
7) Accurate re-evaluation of picked profiles: evaluate_profiles.py
   check them out with check_evaluated_profiles.py
8) Remove deep slip contribution from profiles: remove_deep_slip_contribution.py
9) Hamiltonian inversion of 2D profiles for DZ width: invert_twod_profiles.py
   (forward-model validation: test_twod_dz_forward.py)