# Ridgecrest 2019 Doublet: Fault-Zone Damage Inversion Workflow

Reference notes for report writing — methodology + current results, drawn from
`scripts/notebooks/ridgecrest_workflow.ipynb`, `scripts/`, `scripts/README.md`,
`scripts/twod_inversion_handover.md`, and the `codes` package
(`~/Dev/codes/src/codes`). Bullet-point reference manual, not prose for
publication. Status as of 2026-07-28; several results below are explicitly
marked in-progress/not-yet-analysed.

## Contents
1. Scientific objective
2. Data
3. Fault geometry model
4. Stage 1 — far-field Bayesian slip inversion (AlTar)
5. Stage 2 — near-field 2D damage-zone inversion (the science target)
6. Current status, open issues, next steps
7. Appendix: script/output reference

---

## 1. Scientific objective

- 2019 Ridgecrest, CA earthquake **doublet** (Mw 6.4 foreshock + Mw 7.1
  mainshock) ruptured a **conjugate fault system**: a long NW-striking main
  strand (Little Lake Fault Zone / mainshock) and a shorter, more southerly
  foreshock strand — see the 3D mesh / map figures (§3).
- **Two-stage inversion strategy**, coarse-to-fine:
  1. **Far field** (InSAR + GNSS, km-scale resolution): 3D triangular-mesh
     elastic dislocation model, Bayesian (AlTar). Resolves the regional slip
     distribution, in particular **deep slip** (> ~2.5–3 km).
  2. **Near field** (1 m optical image correlation, resolved *across* the
     fault at cm–dm scale): a 2D antiplane elastic model with a
     **reduced-modulus damage zone** straddling the fault, inverted
     per-profile at picked locations along strike.
- **Science target**: along-strike variation of (a) damage-zone half-width,
  (b) damage-zone modulus reduction, (c) shallow (0–3 km) slip — using the
  near-field optical data, which is sensitive to the fault-zone rheology in a
  way InSAR/GNSS (too coarse) are not.
- The far-field model's **deep-slip contribution is forward-predicted and
  subtracted** from each near-field profile before the 2D inversion, so the
  two stages are not independent: stage 2 only has to explain the *shallow*
  residual.

---

## 2. Data

| Dataset | Source / file | Role |
|---|---|---|
| InSAR descending | `D071_20190704-0716` (unwrapped + LOS components) | **Used** in AlTar production inversion |
| InSAR ascending | `A064_20190704-0710` | Tested in quick-LSQ comparison only; **dropped** from production (see §4.2) |
| GNSS | `unr_gps_offsets_full.txt` (UNR network), horizontal offsets, max 100 km from fault | Used in all inversions; formal std. errors scaled ×20 (`GNSS_SDE_SCALE`) — raw UNR errors are unrealistically tight for a joint fit |
| Optical image correlation | 1 m pixel, EW + NS components, detrended GeoTIFFs (`EW/NS_Ridgecrest_1m_utm_detrended.tif`) | **Never inverted** in Stage 1 (used only as an independent near-field check). **Is** the primary data for Stage 2 (fault-perpendicular profiles). |

- CRS: UTM zone 11; Poisson's ratio ν = 0.25 throughout (standard elastic
  half-space assumption for all GF calculations).
- InSAR preprocessing (`prep_files_for_altar.py`, `codes.InSAR`):
  1. **Covariance estimation** (`compute_covariance`): random subsample of
     pixels *outside* the deforming region (`COV_MASK_OUT` box), bilinear-ramp
     detrend, empirical covariogram, fit exponential `C(r) = σ²·exp(−r/λ) +
     sill`. **RNG-sensitive** — must be seeded (`SEED=0`); an unseeded run can
     converge to a visibly worse fit (flagged in memory as a real gotcha).
  2. **Downsampling**: distance-based quadtree (`downsample`), denser near the
     fault trace (`char_dist=-5000` triggers the near-fault-dense branch),
     `start_size=16 km` down to `min_size=500 m`.
  3. **Green's functions**: `compute_greens_functions(fault)` — Meade (2007)
     triangular dislocation element (TDE) solution, implementation translated
     from Romain Jolivet's CSI (`codes/meade07.py`, ported from Brendan
     Meade's MATLAB code, *Computers & Geosciences* 2007,
     doi:10.1016/j.cageo.2006.12.003).
- GNSS preprocessing: diagonal `Cd` from (scaled) formal errors; same TDE GF
  machinery as InSAR.
- Joint system assembly: `codes.InversionManager` — stacks all datasets, adds
  a per-InSAR-track 3-parameter linear ramp (`Ramp` class: `[1, x_norm,
  y_norm]`, centred/scaled design matrix), writes `data.h5` / `covariance.h5`
  / `gf.h5` / `patch_areas.h5` for AlTar and pickles the assembled system
  (`INVERSION_PICKLE`) for the quick-LSQ check.

---

## 3. Fault geometry model

- Source geometry: California Fault Model (CFM5) GOCAD TSurf files for the
  two Little Lake Fault Zone strands (`SNFA-LLFZ-EAST...`,
  `SNFA-LLFZ-SOUT...`).
- **Meshing pipeline** (`scripts/mesh_fault.py` →
  `codes.helpers.{fault_from_cfm, remesh_fault}`, refactored away from the
  older ad-hoc `RemeshedFault`/`.npz` scheme):
  1. `fault_from_cfm`: read CFM TSurf, shift so the **top-trace datum sits at
     z = 0** (Meade/Okada assume a flat free surface — the topographic wedge
     above datum is discarded).
  2. `remesh_fault`: rebuild as a **depth-layered triangular mesh** between
     horizontal contour depths `[0, 1500, 3000, 5500, 8500, 11500, 15000]` m
     — 6 layers, coarsening with depth (`n_top`→`n_bottom` point counts per
     contour, forced even). Contours below `project_below=10 km` are
     projected vertically rather than sliced (handles the CFM's uneven bottom
     edge). Adjacent layers share exact contour vertices → clean per-layer
     splitting.
  3. Two strands meshed independently then `FaultTriangles.merge`d into one
     object; saved as `data/fault/ridgecrest_faults.pickle` (+ GOCAD `.txt`).
- **Current mesh** (as loaded in the notebook, §1): mainshock 284 triangles /
  172 vertices, foreshock 140 / 88 → **424 triangles total**.
  - Per-layer depth ranges identical for both strands: L0 0–1.5 km, L1
    1.5–3 km, L2 3–5.5 km, L3 5.5–8.5 km, L4 8.5–11.5 km, L5 11.5–15 km.
- **Version note**: the **archived AlTar run (`tri01`) used an earlier
  330-patch mesh** (reconstructed from `*_fault_remesh.npz` snapshots), not
  the current 424-patch mesh — downstream scripts that must match the AlTar
  posterior (`plot_altar_output.py`, `remove_deep_slip_contribution.py`)
  reconstruct the 330-patch version explicitly. Only the local-dip lookup in
  the 2D inversion uses the current mesh (discretisation-independent).
- Dip-slip sign fixed by an "auto" per-strand mean-normal convention inside
  `compute_greens_functions`; not an issue for this geometry.

---

## 4. Stage 1 — far-field Bayesian slip inversion (AlTar)

### 4.1 Quick least-squares sanity check (`quick_lsq_far_field_inversion.py`)

- Purpose: confirm the assembled system (G, d, Cd) is sane *before* the
  expensive Bayesian run — not a science product itself.
- Method: zero-mean regularised weighted LSQ,
  `m = (GᵀCd⁻¹G + Cm⁻¹)⁻¹ GᵀCd⁻¹d`, data pre-whitened via Cholesky of Cd.
- **Smoothing prior** `Cm`: CSI-style (Radiguet et al. 2010)
  exponential-covariance Laplacian smoothing on patch centroids,
  `Cm(i,j) = (σ·λ₀/λ)² exp(−‖cᵢ−cⱼ‖/λ)`, same kernel applied independently to
  strike-slip and dip-slip blocks (σ=1.317, λ=5 km, λ₀=1 km); ramp params get
  a loose diagonal prior (var=1e4).
- **Five data-combination tests** run (ascending only / descending only /
  each + GNSS / all three), each producing slip-2D figures + data-fit maps +
  a **near-field optical residual check** (`optical_residual_check.py` —
  forward-predicts the optical displacement from the LSQ slip model; optical
  is *never* part of the fit, purely a hold-out check).
- **Post-fit residual RMS** (from the notebook, §3.6):

  | Case | A064 LOS (cm) | D071 LOS (cm) | GNSS (mm) | Optical EW (m) | Optical NS (m) |
  |---|---|---|---|---|---|
  | Ascending only | 10.4 | — | 57.2 | 0.51 | 0.63 |
  | Descending only | — | 7.6 | 17.3 | 0.40 | 0.46 |
  | Ascending + GNSS | 11.2 | — | 15.4 | 0.43 | 0.52 |
  | **Descending + GNSS** | — | **6.9** | **7.7** | 0.37 | 0.43 |
  | Ascending + descending + GNSS | 12.3 | 7.5 | 9.8 | 0.36 | 0.40 |

  **Interpretation / decision**: descending + GNSS gives both the lowest D071
  LOS residual *and* the lowest GNSS residual of any combination — adding
  ascending (A064) does not improve either and makes GNSS measurably worse
  (7.7→9.8 mm). This is the basis for dropping A064 from the production AlTar
  run (`config.py`'s `INSAR_TRACKS` has A064 commented out) — ascending data
  appears to be somewhat inconsistent with the rest of the joint system.
  - Optical residuals sit at the 0.4–0.6 m level across *all* combinations
    and barely move with the choice of geodetic data — this is the
    near-field signature the far-field model structurally cannot explain
    (no shallow damage zone, no fine along-strike slip resolution), and is
    the direct motivation for Stage 2.

### 4.2 AlTar Bayesian inversion

- **AlTar**: external Bayesian sampler, CATMIP-family
  transitional/simulated-annealing MCMC (not part of `codes`; run on the
  `ist-oar.u-ga.fr` cluster). Anneals a temperature parameter β from ≈0 to 1
  over discrete steps (archived run: **84 steps**, `BetaStatistics.txt`),
  resampling a fixed-size particle ensemble each step so the final ensemble
  (`step_final.h5`) approximates the posterior without needing gradients.
- Two cluster configs tried: `outputs_16000_12000` (archived/analysed,
  **`tri01`**) and `outputs_40000_15000` (larger, unused downstream). Free
  parameters: strike-slip + dip-slip per patch (330-patch mesh) + one
  3-parameter ramp per InSAR track.
- **Outputs** (`plot_altar_output.py`, `codes.AltarOutput`): data fits
  (D071 LOS, GNSS, + **optical near-field check** — posterior model
  forward-predicted at optical pixels, same hold-out role as §4.1, and the
  quantitative motivation for Stage 2); slip distribution (2D along-strike/
  depth sections + 3D); **bivariate slip + uncertainty** plots (colour =
  slip, second channel = posterior σ or σ/slip — shows where slip is
  actually constrained vs. prior-dominated); per-patch posterior PDFs.
- **Deep-slip-only surface displacement**: posterior slip below the depth
  cutoff, forward-propagated to the surface — exactly the field subtracted
  from each near-field profile in §5.3, shown standalone for inspection.

---

## 5. Stage 2 — near-field 2D damage-zone inversion (the science target)

Pipeline: **pick profiles → re-evaluate accurately (fault-aligned) → remove
deep-slip contribution → invert each residual with a 2D damage-zone model
(Bayesian, PyMC/NUTS)**. Implemented across
`pick_profiles.py` → `evaluate_profiles.py` → `remove_deep_slip_contribution.py`
→ `invert_twod_profiles.py`, all driven off `config.py`.

### 5.1 Profile picking (`pick_profiles.py`, `codes.profile_picking`)

- 123 candidate fault-perpendicular profiles generated automatically along
  both strands (`profiles_along_trace`, local strike from a ±125m chord),
  ±4 km half-length; each quick-evaluated with a swathe scheme (150m
  half-width, 400 bins, `evaluate_profiles_quick`), cropped to the
  well-populated span, dropped if too little data.
- **Interactive keep/reject GUI** (`ProfilePicker`): step through
  candidates, keep/reject (y/n), optionally **drag-select** a sub-extent to
  keep (`keep_extent`); minimap shows location on the optical scene.
- Result: **34 kept of 123** (27 main strand, 7 foreshock) —
  `picked_profiles.pickle`, in **pick order = the profile index (0–33)
  used throughout every later step/figure**.
- Rejections mostly aren't data-quality (dense initial spacing → picking
  selects a representative subset); a few true rejects lack any visible
  fault-parallel step (e.g. index 0, off the active trace).

### 5.2 Fault-aligned re-evaluation (`evaluate_profiles.py`, `codes.fault_aligned_profiles`)

- Accurate re-evaluation of the 34 kept profiles
  (`evaluate_picked_profiles`), materially different from the quick pass:
  resample+smooth the trace; **local strike** at each point; per-point
  perpendicular profile (displacement **and finite strain**); **sub-pixel
  relocation** onto the peak shear-strain ridge (coarse step-detection pass,
  `step_search_half_width=350m`, robust to off-fault strain/imprecise
  trace, then a narrower strain-based refinement, `search_half_width=100m`);
  **along-strike median stacking** (±150m window); per-profile analysis
  metadata including a strain-based fault-zone-width (FZW) estimate
  independent of the later mechanical `dz_halfwidth` (planned cross-check,
  not yet done, §6).
- Output: 34 `Profile` objects (`evaluated_profiles.pickle`); `xs=0` is the
  *relocated* fault, `displacements = [fault-parallel, fault-normal]`.
- QC: `check_evaluated_profiles.py` — interactive stepper vs. the quick-pick
  result + relocation strain profile + minimap.

### 5.3 Deep-slip contribution removal (`remove_deep_slip_contribution.py`)

- For each of the 34 profiles, isolates the **shallow residual** Stage 2
  must explain: draw **1000 posterior slip samples** from the archived AlTar
  ensemble (seeded); zero out shallow patches (centroid depth ≤ 2.5km,
  deliberately *below* the 1.5–3km mesh boundary so that layer is left for
  the 2D inversion to re-solve, not assumed known); batched TDE-GF forward
  prediction at the profile's binned points for all 1000 samples; subtract
  the **ensemble mean** from the data → residual; **propagate uncertainty**
  as `cov_total = cov_model` (ensemble covariance of the deep predictions)
  `+ σ_noise²·I` (noise estimated from detrending the far-field band,
  `|xs|>2km`).
- Model uncertainty stays comfortably below the data noise floor for almost
  every profile (`figures/deep_removal/summary_model_vs_noise.png`) — the
  far-field posterior is well-constrained enough that it isn't the limiting
  factor for the near-field fit.
- Output: 34 dicts (`deep_removed_profiles.pickle`) — `xs`, `data_par`,
  `resid`, `cov_model`, `cov_total`, + metadata. Stays in the physical
  displacement frame; the sign flip is applied downstream (§5.5).

### 5.4 2D forward model — damage-zone antiplane elasticity

- Base physics: **Segall (2010)** / **Ragon & Simons (2021)** antiplane
  (screw-dislocation) image-series solution for a vertical strike-slip fault
  with a **reduced-modulus damage zone**: a compliant column of half-width
  `dz_half_width`, modulus ratio `modulus_ratio` = μ_damage/μ_host, straddling
  the fault trace. Modulus contrast parameter
  `k = (1 − modulus_ratio)/(1 + modulus_ratio)`; the surface displacement
  from a buried screw dislocation is an image series in powers of `k` that
  converges geometrically (series truncated once `k^m < tol=1e-6`).
- **`codes/TwoDDzForwardModel.py`** — dipping-fault extension (this
  project's original contribution to the forward model). Compliant zone
  stays **vertical** (`|x| < dz_half_width`, centred on the surface trace);
  each fault **patch** is an *inclined* segment between two endpoint
  dislocation **lines** at different horizontal offsets/depths — uniform
  slip on the patch = difference of the two endpoint line fields. New
  `dislocation_line_dz(xs, x0, d, h, k, m_max)` generalises the image series
  to a line at an arbitrary offset `x0` (in-zone images at
  `±2mh+(−1)^m·x0`, strength `k^m`; a separate verified series for sources
  outside the zone); branch cut runs vertically to the surface at `x0` so a
  buried line stays continuous and only the trace carries the step.
  `build_dipping_patches(interfaces, x_offsets)` builds the geometry
  (all-zero offsets exactly reproduces the original vertical-fault
  expressions — regression-tested). **Validation**
  (`test_twod_dz_forward.py`, all passing): vertical-fault regression;
  analytic homogeneous dipping limit (exact); wall continuity/traction
  (~1e-6); source crossing the wall; independent finite-difference solve of
  `div(μ∇u)=0` with the slip jump (agrees ~1% RMS = discretisation error).
- **Sheath extension** (`TwoDDzSheathForwardModel.py`, `--sheath`, in active
  testing): a damage-zone sheath of constant *perpendicular* width that dips
  *with* the fault — the image-series trick needs the two zone walls
  parallel to each other and to the free surface, which a dipping sheath
  breaks (no finite image set). Solved instead as a **boundary-integral/
  equivalent-source-density** problem on the sheath walls (same idea as
  building slip from dislocation cores, applied to the modulus jump).
  Reduces to the base class in the vertical limit; cross-validated against
  an independent FD solve (`test_twod_dz_sheath_forward.py`). **~27× slower
  per evaluation** — sampler settings were cut back accordingly (§5.9).

### 5.5 Bayesian inversion setup (`invert_twod_profiles.py`)

- **Parameters** (original/default): `dz_halfwidth`, `modulus_ratio`,
  `offset` (nuisance datum shift, **not** vertical/topographic — see below),
  `slip0..slip_{n-1}` (one per depth patch).
- **Depth discretisation**: originally 3 patches, interfaces
  `VD = [0, 500, 1500, 3000]` m; now CLI-configurable (`--vd`, comma-separated
  interfaces) after the discretisation experiment (§5.8).
- **Priors** (original defaults, all independent uniforms):
  slip patches `U(0, 10)` m each; `dz_halfwidth` `U(0, 2500)` m;
  `modulus_ratio` `U(0.2, 0.9)` — **the 0.2 floor turned out to be too
  high, see §5.7**; `offset` `U(−3, 3)` m.
- **`offset` clarified**: a constant additive shift on the whole predicted
  curve (`model.run(xs).sol + offset`), absorbing the datum bias between the
  (deliberately not mean-centred) deep-removed data and the model — lets the
  predicted curve slide up/down independent of the shape parameters.
- **Sign convention**: data × per-profile `sign` so the far-field step
  (`|xs| > 2 km`) is positive, matching "positive slip = up-step". Main
  strand `+1`; foreshock (left-lateral) `−1` (except profile 33, ~zero
  step). Stored in the result.
- **Local dip**: per profile, area-weighted mean triangle normal within
  3 km along-strike (current mesh), per shallow mesh layer, projected into
  the profile frame as each patch-interface's horizontal offset. Reaches
  ~1.4 km at 3 km depth where dip ≈ 61° — the dipping extension is not a
  small correction here.
- **Likelihood**: `pm.MvNormal` on the full `cov_total` from §5.3 (not
  diagonal) — correlated data noise.
- **Sampler**: PyMC **NUTS**, via `codes.HamiltonianInversion`. The
  numba-jitted forward model isn't auto-differentiable, so it's wrapped as a
  **black-box PyTensor Op with a finite-difference Jacobian**
  (`eps=1e-5`, one forward call per parameter per gradient) — the dominant
  per-draw cost.
- Priors as objects (`codes.UniformDist`/`GaussianDist`, `Dist.py`) give
  `HamiltonianInversion` a uniform `.label`/`.val`/`.pm` interface; the
  Gaussian class was completed this project for `--free-dip` (Normal prior,
  μ = mesh dip estimate, σ = 8°).
- **Convergence diagnostics**: hand-rolled (arviz-version-proof, §6)
  split-Rhat + bulk-ESS + per-chain means per parameter, stored per run;
  flags `rhat > 1.05` or `ess < 400`.
- Production defaults (original): 8 chains / 4000 draws / 2000 tune /
  360 min timeout.
- Result pickle per profile: `idata`, `summary` (MAP/median/16th/84th),
  **`best`** (max-likelihood sample — use for fit curves; marginal medians
  are meaningless once bimodal), `stats`, geometry/sign metadata.
  `--skip-done` (resumable), `--profile i` (single record, job-array
  friendly).

### 5.6 Convergence testing methodology (`twod_convergence_test.py`)

- Before committing to production settings: ran 2 representative profiles
  (15 = strong signal, known-bimodal; 28 = weak foreshock signal,
  slow-mixing) at 4 increasing settings, (4ch,500d,500t) →
  (8ch,1000d,1000t) → (8ch,2000d,1500t) → (8ch,4000d,2000t), comparing
  overlaid posteriors + Rhat/ESS.
- **Profile 28**: Rhat ≈ 1.00, posteriors overlay near-perfectly at every
  setting — well-behaved, unimodal.
- **Profile 15**: Rhat blows up to 3–7 at intermediate settings (chains
  split between a `dz≈400 m` mode most find and a spurious `dz≈2500 m`
  outlier mode) — a genuine secondary mode, only reliably suppressed (not
  eliminated) at the largest setting. Correctly predicted the systemic
  bimodality problem diagnosed properly in §5.7.
- Production settings adopted: 8 chains / 4000 draws / 2000 tune.

### 5.7 Baseline production run (`twod01`) — results and diagnosis

- `run_twod_weekend.sh`: convergence test + full 34-profile run at
  production settings, unattended on the cluster.
- **Headline numbers** (all 34 profiles, `twod01`): **17/34 rail at the 0.2
  `modulus_ratio` floor**; **14/34 flagged poor convergence** (Rhat > 1.05
  or ESS < 400). Along-strike summary: `figures/twod/summary_along_strike.png`.
- Range of fit quality seen (`figures/twod/profile_{12,15,09}_*.png`):
  well-resolved (profile 12, mr=0.48, interior of prior, tight/excellent);
  excellent fit but mr pinned at floor (profile 15); dz/mr essentially
  unconstrained (profile 9).
- **Root-cause diagnosis (2026-07-20/21) — two separate problems**:
  1. **Genuine dz/mr degeneracy + a hard-prior-boundary truncation
     artifact.** A narrow-but-soft zone and a wide-but-stiffer zone give
     almost the same near-fault step, diverging only in the (often weak
     S/N) far field → real bimodal likelihood for several profiles (e.g.
     profile 8: `dz≈0–250m,mr≈0.2–0.4` vs `dz≈2200–2500m,mr≈0.85–0.9`).
     PyMC's default jittered init starts all chains near the same point, so
     which basin wins is mostly luck — explains both the Rhat blowups and
     the "unusually tight" posteriors flagged initially: chains clipped
     against the hard `U(0.2,·)` wall have collapsed within-chain
     variance, which looks confident but is a **truncation artifact, not
     real information**.
  2. **Separately, a forward-model shape mismatch for a few profiles that
     no re-sampling fixes.** Reduced χ²/dof up to **316 (profile 32)** and
     **192 (profile 7)** — e.g. profile 32 has a sharp near-fault bump the
     smooth monotonic model can't reproduce at any parameters; profile 7's
     far field never decays as the model predicts. Points to an upstream
     data/deep-removal issue for these specific profiles.

### 5.8 Follow-up experiments

**4-condition re-run** (`oar_rerun_problem_profiles.sh`, results
`results/profile_inversion/exp_{mrfloor002,fixmr05,dispersed128,freedip}/`),
one representative profile per new CLI capability:

| Profile | Condition | Finding |
|---|---|---|
| 15 (mr pinned) | `--mr-lower 0.02` | **Confirms the boundary artifact.** True mode: mr=0.141 [0.133,0.157], dz=371m [358,393] — clear of both floors, fit unchanged. Small (~10–15%) secondary hump at the wide/high-mr branch too. |
| 24 (strong bimodality) | `--fix-mr 0.5` | **Relocates the wall, doesn't resolve it.** dz piles against its own 2500m *ceiling* instead — implausible. Wide-dz branch actually wants mr≈0.85–0.9. |
| 8 (worst Rhat, 111) | `--dispersed-init --chains 128` | **Wide/high-mr mode is the majority (57% vs 39%), not a fluke.** The original 8-chain run reported the minority basin from init-jitter luck alone — likely true for other bimodal profiles (6, 14, 16, 18, 19, 23...). |
| 7 (poor fit) | `--free-dip` (±8°) | **Rules out dip.** dip0 moved modestly, dip1 stayed at prior. Far-field mismatch persists → points to an upstream issue (likely incomplete deep-slip removal). |

- New CLI flags on `invert_twod_profiles.py` (backward-compatible):
  `--mr-lower`, `--fix-mr`, `--free-dip`, `--dispersed-init` (+ auto
  `plot_chain_modes()` for `--chains > 16`), `--target-accept`, `--outtag`,
  `--vd`, `--sheath`.
- Recommendation: lower the production mr floor (later superseded, §5.9);
  default to dispersed-init.

**Slip-discretisation sensitivity** (`oar_rerun_discretisation.sh`, results
`exp_disc{3,5,10}/`): 3 profiles (15, 0, 28) × 3 patch schemes (3 / 5 / 10),
all under **matching** settings (`--mr-lower 0.02 --dispersed-init --chains 8
--target-accept 0.95`) so only discretisation varies:

| profile | 3 patches (χ²/dof, % pinned) | 5 patches | 10 patches |
|---|---|---|---|
| 15 | 7.25, 33% | 4.49, 40% | 4.58, 30% |
| 0 | 2.72, 67% | 2.82, 60% | 2.66, **80%** |
| 28 | 4.01, 33% | 3.89, 40% | 3.87, 40% |

- **No conclusive benefit beyond ~5 patches**: χ²/dof drops meaningfully
  3→5 for profile 15 (real), flat-to-marginal for 0/28; 5→10 is flat or
  slightly worse everywhere, and profile 0 gets *more* bound-pinned patches
  (not fewer) at 10.
- At 10 patches, profile 15's slip-vs-depth posterior alternates between ~0
  and the 10m ceiling on adjacent patches, marginals piled at both ends —
  the signature of an unidentifiable direction (data constrains a smoothed
  combination of neighbours, not each patch), not genuine resolution.
  Expected: `make_priors()` puts an independent, unregularised `U(0,10)` on
  every slip patch, so finer patches just add freedom the data can't use.
- **Decision: keep 3–5 slip patches.** Finer needs a smoothing/regularising
  prior between adjacent patches (random-walk/Laplacian) — not implemented.

### 5.9 Current production re-runs (in progress, not yet fully analysed)

- **`twod02`** (`oar_twod02_production.sh`): all 34 profiles,
  `--mr-lower 0.02 --dispersed-init --chains 8 --target-accept 0.95
  --vd 0,400,1000,1800,3000` (4 patches — between the tested 3 and 5),
  `--timeout 45` min/profile, `--skip-done` (resumable).
  **Status: 26/34 profiles completed** — run was cut off (last profile in
  the log landed only 217 draws on 1 chain before the walltime hit);
  resubmittable with `--skip-done`.
- **`twod03`** (`oar_twod03_production.sh`): same settings as `twod02` plus
  **`--sheath`** (the dip-following damage-zone sheath model, §5.4) and
  reduced sampler budget (`--draws 800 --tune 400`, down from 4000/2000) to
  compensate for the ~27× per-evaluation cost. **Status: 14/34 profiles
  completed**, actively running.
- **Not yet done**: fetching + analysing either run to completion; the
  along-strike summary in the notebook (§9.3) is **still the original
  `twod01` baseline**, not `twod02`/`twod03`. Treat `twod01`'s *numbers* as
  superseded (mr floor was a truncation artifact) but its *figures/machinery*
  as representative of the method.

---

## 6. Current status, open issues, next steps

- **Headline finding**: the original `modulus_ratio` floor (0.2) was too
  tight and produced a truncation artifact masquerading as a confident
  result for ~half the profiles; the true posterior is often **bimodal**
  between a narrow/soft and a wide/stiff damage zone, and a non-dispersed
  8-chain run can silently report either mode by init luck. The single most
  important caveat for any `twod01`-based number.
- **Open, unresolved**: profiles 32, 7 (and 5) have very large χ²/dof (up
  to 316) that no re-sampling or dip freedom fixes — likely a forward-model
  shape mismatch or an upstream issue in `remove_deep_slip_contribution.py`
  inputs (possibly the known InSAR-covariance RNG-seed sensitivity, or an
  underestimated `cov_total`) — flagged, not yet investigated.
- **Discretisation**: settled at 3–5 slip patches without a regularising
  prior; finer needs one first.
- **Sheath model** (`--sheath`): implemented, FD-validated, being tested in
  `twod03` — the more physically defensible model wherever dip is large
  (offsets up to 1.4 km at 3 km depth here), the natural refinement over the
  vertical-column approximation used for all baseline results.
- **Still to do**: fetch/resubmit `twod02`/`twod03` to completion; re-check
  the railed/poor counts against the new settings; investigate profiles
  32/7/5's χ²/dof; final along-strike summary against `twod02`/`twod03`
  once fetched, **including the still-pending FZW cross-check** (independent
  strain-based fault-zone-width from profile picking, §5.2, vs. the
  model's `dz_halfwidth`).
- **Engineering notes**: non-differentiable forward model handled via a
  black-box PyTensor Op with finite-difference gradients (§5.5, dominant
  per-draw cost); local arviz (1.1.0, restructured) can't unpickle
  `InferenceData` from the cluster's older arviz — handled by a custom
  unpickler (`result_io.py`, `load_result`) that stubs unresolvable classes,
  safe since only plain dict/array fields are needed downstream; InSAR
  covariance estimation is RNG-sensitive, must be seeded.

---

## 7. Appendix: script / output reference

| Step | Script | Key output |
|---|---|---|
| 1. Fault mesh | `mesh_fault.py` | `data/fault/ridgecrest_faults.pickle` (424 patches, 2 strands) |
| 2. Far-field data prep | `prep_files_for_altar.py` | `results/working/inputs/{data,covariance,gf,patch_areas}.h5` |
| 3. Quick LSQ sanity check | `quick_lsq_far_field_inversion.py` | `results/working/tmp/lsq/*.png`, `.npz` |
| 4. AlTar (external, on cluster) | — | `results/tri01/outputs_16000_12000/step_*.h5` |
| 5. Visualise AlTar | `plot_altar_output.py` (`codes.AltarOutput`) | `results/tri01/outputs_16000_12000/figs/` |
| 6. Profile picking | `pick_profiles.py` (`codes.profile_picking`) | `results/working/tmp/picked_profiles.pickle` (34) |
| 7. Fault-aligned re-eval | `evaluate_profiles.py` (`codes.fault_aligned_profiles`) | `results/working/tmp/evaluated_profiles.pickle` |
| — QC | `check_evaluated_profiles.py` | interactive only |
| 8. Deep-slip removal | `remove_deep_slip_contribution.py` | `results/working/tmp/deep_removed_profiles.pickle` |
| 9. 2D forward model | `codes/TwoDDzForwardModel.py`, `TwoDDzSheathForwardModel.py` | — |
| — forward-model validation | `test_twod_dz_forward.py`, `test_twod_dz_sheath_forward.py` | pass/fail only |
| 9. 2D Bayesian inversion | `invert_twod_profiles.py` | `results/profile_inversion/{twod01,twod02,twod03,exp_*}/profile_XX.pickle` |
| — convergence test | `twod_convergence_test.py` | `results/profile_inversion/twod01/convergence/` |
| — safe result loading | `result_io.py` (`load_result`, `param_labels`) | — |
| Notebook (this report's source) | `scripts/notebooks/ridgecrest_workflow.ipynb` | figures copied to `scripts/notebooks/figures/` |
| Detailed running log / handover | `scripts/twod_inversion_handover.md` | dated status entries, most detail of any single file |

**Citations to track down**: Meade (2007) TDE Green's functions; Segall
(2010) (antiplane damage-zone solution); Ragon & Simons (2021) (image-series
generalisation, eq. A12); Radiguet et al. (2010) (smoothing prior); Jolivet
et al., CSI software (Zenodo doi:10.5281/zenodo.14170822, TDE source); AlTar
sampler (CATMIP-family transitional MCMC — confirm exact citation, not
resolved from this codebase).
