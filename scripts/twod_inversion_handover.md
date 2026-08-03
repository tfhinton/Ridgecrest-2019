# Handover: 2D damage-zone profile inversion — weekend run + analysis

## STATUS (2026-07-10, Friday)
Workflow step 9 (the final step) is implemented and validated. A weekend batch
run has been prepared for the compute cluster (ist-oar.u-ga.fr); the owner
launches it via the checklist below. Next session: fetch + analyse the results,
re-run problem profiles, and build a results-summary Jupyter notebook
(existing notebook home: `scripts/notebooks/`, e.g. `ridgecrest_workflow.ipynb`).

## Project in one paragraph
Slip inversion of the 2019 Ridgecrest earthquake doublet. Stage 1: far-field
InSAR + GNSS inverted with a 3D triangular-mesh fault model (AlTar Bayesian
sampler on the cluster) — constrains deep slip. Stage 2 (the science target):
2D profiles of near-field 1 m optical image correlation data, taken
perpendicular to the fault, inverted with a 2D antiplane elastic model that has
a reduced-modulus damage zone (Segall 2010 / Ragon & Simons 2021 image-series
solutions) — resolves shallow slip, damage-zone half-width, and modulus ratio
along strike. The far-field posterior's deep-slip contribution is removed from
each profile before the 2D inversion.

## Workflow (scripts/README.md; steps 1–8 done, outputs cached)
1. mesh_fault.py — build FaultTriangles mesh -> `data/fault/ridgecrest_faults.pickle`
   (424 patches, 2 strands: 0 = main 61.6 km, 1 = foreshock 22.1 km;
   mesh layer 0 spans 0–1500 m depth, layer 1 spans 1500–3000 m)
2. prep_files_for_altar.py — far-field data/GF/cov h5 for AlTar
3. quick_lsq_far_field_inversion.py — sanity LSQ
4. copy_working_to_server.sh / fetch_working_from_server.sh — AlTar run on cluster
   (archived run used downstream: `results/tri01/`, 330-patch mesh)
5. plot_altar_output.py
6. pick_profiles.py — interactive profile picking
7. evaluate_profiles.py — accurate fault-aligned re-evaluation
   -> `results/working/tmp/evaluated_profiles.pickle` (34 profiles)
8. remove_deep_slip_contribution.py — subtract deep (>2.5 km centroid) slip
   predicted from the AlTar posterior ensemble; propagates posterior uncertainty
   -> `results/working/tmp/deep_removed_profiles.pickle`
   (list of dicts: i, fault_id, x_along_fault, strike, xs, data_par, resid,
   cov_model, cov_total = ensemble cov + noise^2*I, ...)
9. **invert_twod_profiles.py — THIS STEP.** Hamiltonian (PyMC NUTS) inversion of
   each deep-removed residual for [dz_halfwidth, modulus_ratio, offset,
   slip0, slip1, slip2]. Results -> `results/profile_inversion/twod01/`.

## The 2D forward model (new this session — dipping-fault extension)
`codes/src/codes/TwoDDzForwardModel.py` (repo ~/Dev/codes, github
PhD-Shared-Codes). The compliant zone stays vertical (|x| < dz_half_width,
centred on the surface trace, x=0 = relocated rupture trace); each fault patch
is an inclined segment between endpoint dislocation LINES. New njit
`dislocation_line_dz(xs, x0, d, h, k, m_max)`: image series for a screw
dislocation line at horizontal offset x0 (in-zone images at ±2mh + (−1)^m x0,
strength k^m; separate verified series for sources outside the zone), branch
cut runs vertically to the surface at x0 (−½sgn(x−x0) term) so buried lines are
continuous and only the trace carries the step. `build_dipping_patches
(interfaces, x_offsets)` builds the geometry; vertical case reproduces the old
`compute_two_d_dz` exactly. `PatchTwoD.dip` bug fixed (was hardcoded 90).
**Validation: `scripts/test_twod_dz_forward.py` (all passing)** — vertical
regression, analytic homogeneous dipping limit (exact), wall
continuity/traction (~1e-6), source crossing the wall, and an independent
finite-difference solve of div(mu grad u)=0 with the slip jump (agrees ~1% rms,
= discretisation error). NB these codes-repo changes must be committed/pushed
and pulled on the cluster (see checklist).

## invert_twod_profiles.py — key conventions
- Depth interfaces VD = 0/500/1500/3000 m -> 3 slip patches. Only strike-slip,
  only the fault-parallel component is inverted.
- Priors: slips U(0,10) m each; dz_halfwidth U(0,2500) m; modulus_ratio
  U(0.2,0.9); offset U(−3,3) m (datum nuisance — profiles are NOT mean-centred).
- Sign: per profile, data = sign * resid with sign chosen so the far-field step
  (|xs| > 2 km bands) is positive; positive model slip = up-step. Main strand
  all +1, foreshock (left-lateral) −1 (except profile 33, ~zero step). Stored as
  'sign' in the result pickle.
- Dip: local shallow dip from `config.FAULT_PICKLE` FaultTriangles — area-
  weighted mean triangle normal within 3 km of the profile, per mesh layer
  (layer 0 -> 0–1500 m, layer 1 -> 1500–3000 m), projected into the profile
  frame (+xs from the evaluated profile's linestring, index 0 = −xs end).
  Offsets are large where dip ~61° (−1.4 km at 3 km depth, 20–30 km along
  strike). Data covariance = record's cov_total (full matrix, MvNormal).
- Outputs per profile (saved as each finishes; `--skip-done` resumes):
  `twod01/profile_XX.pickle` = dict(i, fault_id, x_along_fault, strike, sign,
  vd, x_offsets, dips_deg, xs, data, sigma, summary (per-param map/med/16/84),
  best (max-likelihood posterior sample — use THIS for fit curves, marginal
  medians are meaningless when the posterior is bimodal), stats (convergence),
  sampler, idata (arviz InferenceData)) + `figs/profile_XX.png` + along-strike
  `figs/summary_along_strike.png`.
- Convergence diagnostics built in (hand-rolled, arviz-version-proof):
  split_rhat, ess_bulk, per-chain means per parameter; printed per profile,
  stored in 'stats', end-of-run recap flags rhat>1.05 or ess<400. Diverging
  per-chain means = chains stuck in different modes: smoke tests showed a real
  narrow-DZ vs wide-DZ bimodality on profile 15.
- CLI: `--profile i` (single record, cluster job-array friendly), `--draws
  --tune --chains --timeout --skip-done --list`. Defaults = production:
  8 chains / 4000 draws / 2000 tune / 360 min timeout, cores = min(chains, cpus).

## Weekend run (prepared; owner launches)
`scripts/run_twod_weekend.sh` (nohup on the cluster, 8 CPUs, OMP_NUM_THREADS=1):
1. `twod_convergence_test.py`: profiles 15 (strong signal, bimodal) + 28 (weak
   foreshock, slow mixing) at 4 settings — (4ch,500/500), (8ch,1000/1000),
   (8ch,2000/1500), (8ch,4000/2000) — report.txt + posterior-stability overlay
   figures -> `twod01/convergence/`.
2. Full 34-profile run at production defaults, logs `twod01/{convergence,inversion}.log`.
Launch checklist (Friday):
1. Commit + push codes changes (TwoDDzForwardModel.py, Patch.py — mixed in with
   the owner's other uncommitted codes-repo work) and pull on cluster.
2. `./scripts/transfer_twod_inversion.sh push` (inputs + fault pickle + scripts;
   config.py now path-portable, resolves from __file__).
3. On cluster: `nohup ./scripts/run_twod_weekend.sh > /dev/null 2>&1 &`
4. Monday: `./scripts/transfer_twod_inversion.sh fetch`.

## Monday / next session
1. Fetch; read `twod01/inversion.log` `[conv]` recap + `convergence/report.txt`:
   are posteriors stable between the (8,2000,1500) and (8,4000,2000) settings?
   Which profiles have poor rhat/ess or split modes?
2. Check whether modulus_ratio rails at the 0.2 prior floor across profiles —
   if so, the prior lower bound needs revisiting before interpreting DZ results.
3. Re-run problem profiles individually (`--profile i`, more tune/draws or
   higher target_accept — the latter needs exposing through the CLI).
4. Results-summary notebook in `scripts/notebooks/`: along-strike dz_halfwidth /
   modulus_ratio / shallow slip (use 'best' + 16–84 marginals; compare FZW
   estimates from the profile analysis metadata), example profile fits,
   convergence appendix.

## Gotchas
- Old local smoke-test pickles for profiles 15/28 may sit in twod01/ until the
  fetch overwrites them (60–200-draw runs — do not analyse).
- x=0 exactly makes the OLD compute_two_d_dz raise ZeroDivisionError (numba);
  the new line function guards it. Binned profile xs never hit 0 exactly.
- arviz 1.x: idata.posterior is a DataTree — no .stack; use
  post[label].values.reshape(-1). az.kde / get_maps_from_arviz broken (known).
- **Local-vs-cluster arviz version mismatch**: the cluster's weekend-run pickles were
  written with an arviz whose `InferenceData` still lives at
  `arviz.data.inference_data.InferenceData`; the local Mac .venv now has arviz 1.1.0,
  which split into arviz_base/arviz_stats/arviz_plots and has no `arviz.data` module at
  all, so plain `pickle.load` on a twod01/exp_* result fails with
  `ModuleNotFoundError: No module named 'arviz.data'`. Workaround: a custom
  `pickle.Unpickler.find_class` that substitutes a bare stub class for
  `arviz.data.inference_data.InferenceData` unpickles fine — the stub's `__dict__` then
  has a real, usable `.posterior` xarray Dataset on it (`idata.posterior['label'].values`
  works as normal). Checked in as `scripts/result_io.py` (`load_result(path)`,
  `param_labels(r)`) — a generalised version (shims *any* missing class, not just
  InferenceData) that `invert_twod_profiles.py`'s own along-strike summary loader
  now uses too.
- **Cluster env: don't `module load python/python3.11`.** Doing so before `source
  .venv/bin/activate` injects `/soft/python/lib/python3.11/site-packages` onto
  PYTHONPATH, which survives venv activation and takes priority over the venv's own
  packages for anything not explicitly shadowed. Symptom: `numpy` resolves to
  `/soft/python/...` (module-provided, upgraded to 2.4.2) while `rasterio` — not
  installed in the venv, so it too falls through to `/soft/python/...` — is an older
  build still compiled against numpy 1.x, giving `ImportError: numpy.core.multiarray
  failed to import`. The venv (`pyvenv.cfg`: `home = /usr/bin`, `uv = 0.11.17`,
  `include-system-site-packages = false`) is fully self-contained and doesn't need the
  module at all; just `source /soft/env.bash; cd ~/projects/ridgecrest; source
  .venv/bin/activate`. This venv is uv-managed and has **no pip inside it** (by uv
  design) — use `uv sync` / `uv pip install ...`, not `pip`/`python3 -m pip`.
- `scripts/oar_invert_profiles.sh` (unrelated script, fault-aligned inversion) had an
  accidental one-line edit that clobbered its `cd .../ridgecrest` + `source
  .venv/bin/activate` lines into a stray `b` — owner fixed it independently
  (2026-07-21); not touched by anything above.
- Auto-memory has full background: twod-dipping-dz-inversion, deep-slip-removal,
  fault-aligned-profiles-module, faulttriangles-gfs, altar-mesh-versions.

## STATUS (2026-07-20) — weekend run analysed, re-run of 4 problem profiles prepared

Fetched + loaded all 34 twod01 results (see arviz-mismatch gotcha above) and found two
**separate** problems, not one:

1. **A genuine dz_halfwidth/modulus_ratio degeneracy, with a hard-prior-boundary
   artifact on top.** A narrow-but-very-soft damage zone and a wide-but-less-soft one
   produce almost the same near-fault step, diverging only in the far field where S/N
   is often weak — the likelihood is genuinely bimodal for several profiles (profiles 8
   and 24 show two well-separated posterior clusters, e.g. dz≈0–250m/mr≈0.2–0.4 vs
   dz≈2200–2500m/mr≈0.85–0.9). PyMC's default init (`jitter+adapt_diag`) starts all
   chains near the same point, so which basin a chain lands in is mostly luck — that's
   why per-chain means diverge and rhat blows up to 5–100+ (14/34 profiles flagged
   poor). 17/34 profiles have mr pinned at the 0.2 floor (MAP *and* 16th pctile) —
   some (15, 7: all 8 chains agree, rhat=1.0) look like a genuine preference at/below
   0.2; others (8, 24, 6, 14, 16, 18, 19, 23...) are chains split between the wall and
   an interior mode. The "unusually tight" posteriors are chains clipped against the
   hard Uniform wall — within-chain variance collapses to ~0, which *looks* like a
   confident answer but is a truncation artifact, not real information.
2. **Separately, some profiles have a forward-model shape mismatch no re-sampling will
   fix.** Reduced chi2 (recomputed from `best` + `cov_total`, dof = n − 6) ranges from
   ~1 up to 316 (profile 32) and 192 (profile 7). Profile 32's data has a sharp
   near-fault bump/plateau the smooth monotonic antiplane solution structurally cannot
   reproduce at any parameter values. Profile 7's far-field data never decays the way
   the model predicts. These need a look at the input data / deep-slip-removal step
   (or should be flagged unreliable), not different priors/inits.

**invert_twod_profiles.py new CLI flags** (all backward-compatible; defaults reproduce
the original twod01 run bit-for-bit — verified via `--list`):
- `--mr-lower FLOAT` (default 0.2): modulus_ratio prior lower bound.
- `--fix-mr FLOAT`: hold modulus_ratio fixed, dropped from the free parameters
  entirely (not just a tight prior) — `ProfileModel.param_labels` / `make_priors()`
  are now dynamic per-run instead of a module-level `PARAM_LABELS` constant.
- `--free-dip`: dip0/dip1 (one per mesh layer) become free params with a Gaussian
  prior (mesh estimate, sigma=`DIP_PRIOR_SIGMA`=8 deg); `ProfileModel.pred_func`
  rebuilds patch geometry from `dip_to_offsets()` each call in this case (only —
  the fixed-geometry path is untouched and just as cheap as before).
- `--dispersed-init`: each chain starts from an independent prior draw
  (`dispersed_initvals()`) instead of PyMC's small-jitter default, with `init`
  switched to `'adapt_diag'` so PyMC doesn't re-cluster them; pairs with a large
  `--chains` to map mode occupancy. `plot_chain_modes()` (auto-triggered when
  `--chains > 16`) scatters per-chain-mean dz vs mr/offset — one point per chain.
- `--target-accept FLOAT` (was hardcoded 0.9 in HamiltonianInversion.run, now
  exposed both there and via CLI).
- `--outtag NAME` (default `twod01`): output subdir, so experiment re-runs don't
  clobber the production results.
- `codes/src/codes/Dist.py`: `GaussianDist` was an unused stub (no `label`, no `.pm`)
  — completed to match `UniformDist`'s interface for `--free-dip`. Was unused
  elsewhere, so this was a safe signature change (added `label` as 1st arg).
- **NB**: these codes-repo changes (Dist.py, HamiltonianInversion.py) must be
  committed/pushed and pulled on the cluster before the re-run, same as the dipping
  forward-model changes from the previous session.

**`scripts/oar_rerun_problem_profiles.sh`** (prepared, not yet launched; OAR-submitted
like `oar_invert_profiles.sh`, `/nodes=1/core=8,walltime=06:00:00`) — 4 profiles, one
condition each, `--skip-done`, 8 CPUs (same budget as the weekend run):
- profile 15 (f0, 38.5km, clean rail) → `--mr-lower 0.02`
- profile 24 (f0, 50.3km, strong bimodality) → `--fix-mr 0.5`
- profile 8 (f0, 24.3km, worst rhat blowup, 111) → `--dispersed-init --chains 128
  --draws 300 --tune 500 --target-accept 0.95 --timeout 180`
- profile 7 (f0, 21.8km, poor fit + real dip) → `--free-dip`

Results land in `results/profile_inversion/exp_{mrfloor002,fixmr05,dispersed128,freedip}/`.
`transfer_twod_inversion.sh` push/fetch updated to include the new script and to fetch
the whole `results/profile_inversion/` tree (was `twod01` only).

## STATUS (2026-07-21) — 4-condition re-run analysed; discretisation experiment launched

All 4 conditions from `oar_rerun_problem_profiles.sh` completed (after fixing the
cluster env — see gotcha above — and an accidental bug in an unrelated OAR script).
Results in `results/profile_inversion/exp_{mrfloor002,fixmr05,dispersed128,freedip}/`.
Compared against the `twod01` baseline for the same 4 profiles (script: load both with
`result_io.load_result`, recompute chi2/dof from `best` + the matching record's
`cov_total`, diff `summary`/`stats`). Also copied into
`scripts/notebooks/ridgecrest_workflow.ipynb` as a short dated section with the 4
figures.

1. **Profile 15, `--mr-lower 0.02`: confirms the boundary artifact and gives the real
   number.** Dominant mode moved cleanly off the wall — mr=0.141 [0.133,0.157],
   dz=371m [358,393] (was mr=0.20-pinned, dz=399m) — touching neither the old nor new
   floor, fit still excellent. A small (~10-15%) secondary hump at the wide/high-mr
   branch appears too (source of the new 'poor' flags) but doesn't change the
   headline answer. **Recommends lowering the production `--mr-lower` globally**
   (0.05-0.1 rather than all the way to 0.02, to avoid re-exposing that secondary
   hump as strongly).
2. **Profile 24, `--fix-mr 0.5`: doesn't resolve the degeneracy, just relocates the
   wall.** dz_halfwidth piles up almost entirely against its own prior *ceiling*
   (2500m) — the model wants an even wider zone than allowed once mr is forced to
   0.5, which is physically implausible. slip0/slip1 stayed visibly bimodal. Useful
   negative result: 0.5 is not a good universal fixed value; the wide-dz branch wants
   mr≈0.85-0.9 (matches both the original run and the profile-8 dispersed-init read).
3. **Profile 8, `--dispersed-init --chains 128 --draws 300 --tune 500`: the wide mode
   is real and more probable, not a fluke.** Chain-mode occupancy (per-chain
   posterior means, `plot_chain_modes`): narrow mode (dz<500,mr<0.4) 50/128 (39%),
   wide mode (dz>1500,mr>0.7) 73/128 (**57%**), 5/128 transitional. The original
   8-chain production run reported the *minority* basin (dz≈247m, mr=0.20) just from
   init-jitter luck. Implication: any of the other bimodal/poor-rhat profiles (6, 14,
   16, 18, 19, 23...) could have the same problem — 8 non-dispersed chains isn't
   enough to trust which mode "won" for those either.
4. **Profile 7, `--free-dip`: dip is very unlikely to be the explanation for the poor
   fit.** dip0 moved modestly (60.8→64.2°, tight posterior); dip1 stayed essentially
   at its prior (data has no opinion on the deeper interface's tilt). The core
   problem persists regardless: far-field data on the right side never decays (stays
   flat ~0.65-0.7m to 3.7km) while any physically sensible model must decay there.
   chi2/dof got numerically worse (400 vs 191) but that's likely a harder posterior
   to sample (dz/offset/slip1/dip0 flagged poor) rather than a genuinely worse true
   optimum — a free-dip model can't have a worse best fit than a fixed-dip one by
   construction. Either way, no dip value fixes the far-field shape mismatch, so this
   points *away* from geometry and *toward* an upstream issue — most likely
   incomplete deep-slip removal for this profile specifically. Deprioritise further
   sampler tuning on profile 7/32-like misfits; look at their
   `remove_deep_slip_contribution.py` inputs directly instead.

**New `--vd` CLI flag** (`invert_twod_profiles.py`): comma-separated depth interfaces
overriding the default `0,500,1500,3000` (3 slip patches), e.g.
`--vd 0,100,350,800,1700,3000` for 5. Implemented as a `global VD, SLIP_LABELS`
reassignment right after `argparse` (both are read as module globals throughout —
`local_dip`, `make_priors`, `ProfileModel.pred_func`, the slip-vs-depth panel in
`plot_inversion` — so this was the minimal-diff way to thread a custom discretisation
through all of them without a wider refactor). `MESH_INTERFACES` (the FaultTriangles
dip-layer boundaries, 0/1500/3000) is independent and unaffected.

**Slip-discretisation sensitivity experiment (launched, not yet fetched)**:
`scripts/oar_rerun_discretisation.sh` — 3 profiles x 3 discretisations, **all under
matching settings** (`--mr-lower 0.02 --dispersed-init --chains 8 --target-accept
0.95`) so discretisation is the only thing varying between arms (deliberately re-runs
the 3-patch case under the new settings too, rather than reusing the old twod01
result, which used mr-lower=0.2 + non-dispersed init — comparing against that would
confound discretisation with sampler-settings differences):
- patches3 (current default): `0,500,1500,3000`
- patches5: `0,100,350,800,1700,3000`
- patches10: `0,100,200,300,500,700,1000,1400,1800,2300,3000`
- Profiles: 15 (f0, well-fit, mr near the old 0.2 floor), 0 (f0, well-fit, mr well
  interior ~0.6), 28 (f1 foreshock strand, weak signal) — chosen for good baseline
  fit/convergence so discretisation effects aren't confounded with the dz/mr
  multimodality or misfit issues above.
- Results -> `results/profile_inversion/exp_disc{3,5,10}/profile_{00,15,28}.pickle`.
- `/nodes=1/core=8,walltime=08:00:00` (9 sequential single-profile runs; higher-dim
  posteriors for the 10-patch case may be slower/need the higher target_accept).

## STATUS (2026-07-27) — discretisation experiment analysed: no benefit beyond ~5 patches

Fetched `exp_disc{3,5,10}` (9 runs: profiles 15/0/28 x 3/5/10 slip patches, all under
matching mr-lower=0.02/dispersed-init/8-chains). Compared chi2/dof (recomputed from
`best`) and the fraction of slip patches sitting within 0.15m of a prior bound (0 or
10m) as a proxy for "individually unresolved":

| profile | 3 patches | 5 patches | 10 patches |
|---|---|---|---|
| 15 (chi2/dof, %bound-pinned) | 7.25, 33% | 4.49, 40% | 4.58, 30% |
| 0  | 2.72, 67% | 2.82, 60% | 2.66, 80% |
| 28 | 4.01, 33% | 3.89, 40% | 3.87, 40% |

**No conclusive benefit beyond ~5 patches.** chi2/dof drops meaningfully 3->5 for
profile 15 (real improvement) but is flat-to-marginal for 0/28, and going 5->10 is
flat or slightly *worse* everywhere. Meanwhile 30-80% of slip patches sit pinned at a
prior bound regardless of discretisation — and profile 0 gets *more* bound-pinned
(60%->80%) at 10 patches, not less. Visual confirmation on profile 15: the 3-patch
slip-vs-depth profile is smooth and sensible; the 10-patch one alternates between ~0
and the 10m ceiling on adjacent patches, with several marginal posteriors piled at
*both* ends and nothing in between — the classic signature of an unidentifiable
direction in parameter space (the data constrains some smoothed combination of
neighbouring patches, not their individual values) rather than genuine added
resolution. This is expected: `make_priors()` puts an independent, unregularised
`U(0,10)` on every slip patch, so finer patches just give the sampler more freedom to
redistribute slip arbitrarily wherever the fit is insensitive to it.
**Decision: keep 3 patches (5 at most) for the production re-run** — using 10 would
need a smoothing/regularising prior between adjacent patches (random-walk or
Laplacian-type penalty) to be meaningful, which isn't implemented.
Figures + this writeup also copied into `ridgecrest_workflow.ipynb` §9.7 (including a
new `discretisation_chi2_vs_npatch.png` summary plot, script inline in that session —
not yet a checked-in analysis script).

**`offset` parameter, for reference** (asked this session): NOT a vertical/topographic
offset — it's a constant additive shift applied to the *entire predicted
fault-parallel-displacement curve* (`ProfileModel.pred_func`: `model.run(xs).sol +
offset`, prior `U(-3,3)` m). A nuisance datum/reference-level parameter, needed
because the deep-slip-removed residuals going into this inversion are deliberately
*not* mean-centred (see `invert_twod_profiles.py`'s conventions section) — `offset`
absorbs whatever constant bias is left between the data's absolute level and the
model's (which has no free additive constant of its own). Visually: it's what lets
the whole red model curve in each profile figure slide up/down as one to match the
data's actual y-level, independent of dz/mr/slip which control its *shape*.

## STATUS (2026-07-27, later) — twod02 production re-run prepared, not yet launched

`scripts/oar_twod02_production.sh` — all 34 profiles, one OAR job
(`/nodes=1/core=8,walltime=20:00:00`), `--skip-done` (resubmit-safe if walltime is hit
before finishing):
- `--mr-lower 0.02` — the exact value already validated in the 2026-07-21 experiment
  (profile 15's true mr=0.14, clear of both the old 0.20 floor and this one), used in
  preference to the untested "0.05-0.1 would probably be safer" guess from that
  session's writeup.
- `--dispersed-init --chains 8 --target-accept 0.95` — chain count is the owner's
  explicit choice for this run (the 128-chain profile-8 test was a diagnostic to
  characterise mode occupancy, not a production chain count); note 8 dispersed chains
  will be noisier at catching rare/small secondary modes than that 128-chain read
  (e.g. profile 15's ~10-15% secondary hump might or might not get a chain by luck),
  but is still strictly better than PyMC's small-jitter default for this bimodal
  problem.
- `--vd 0,400,1000,1800,3000` — 4 slip patches (owner's choice; between the original 3
  and the 5 that still showed a real chi2 improvement in the discretisation
  experiment, comfortably clear of the resolution collapse seen at 10).
- `--timeout 45` (minutes, per-profile safety net so one pathological profile can't
  eat the whole 20h budget; not expected to bind — 8ch/4000d/2000t runs have
  historically taken 13-20 min).
- Output -> `results/profile_inversion/twod02/` (does not touch `twod01/` or any
  `exp_*/`). `transfer_twod_inversion.sh` push/fetch updated accordingly.

Launch: `./scripts/transfer_twod_inversion.sh push`, then on the cluster
`oarsub -S ./scripts/oar_twod02_production.sh`.

## Next session
1. `./scripts/transfer_twod_inversion.sh fetch`; if not all 34 profiles finished
   within the 20h walltime, resubmit `oar_twod02_production.sh` (`--skip-done` picks
   up where it left off).
2. Read `results/profile_inversion/logs/twod02_production.log` `[conv]` recap — how
   many profiles still flagged poor/railed with the new settings vs. the original
   twod01 run (17/34 railed, 14/34 poor)?
3. Still open: profiles 32/7/5's chi2/dof (316/192/126) are enormous even setting
   multimodality aside — worth checking whether `cov_total` from
   remove_deep_slip_contribution.py is underestimated (small ensemble? related to the
   InSAR-covariance RNG-seed sensitivity noted for A064) as well as / instead of a
   genuine model-shape problem.
4. Results-summary notebook in `scripts/notebooks/` — dated sections through §9.7;
   still needs the final along-strike dz_halfwidth/modulus_ratio/shallow-slip summary,
   now against `twod02` once fetched (§9.3's is still the old twod01 run).
