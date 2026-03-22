# h0_dipole_test v7.2.1

Code repository for the analysis in:

- **Vaillant (2026)**, *“A single-flow model fails to account for localized low-z anisotropy in the Hubble residual field”*  
  DOI (Zenodo): **10.5281/zenodo.19154067**. Older version of this preprint (without CF4++): *“Dipole mapping of the local Hubble expansion”* (Vaillant, 2026)

Author: **Michaël Vaillant**  
Affiliation: **Meta-Connexions (Toulouse, France)**  
License: **MIT**

---

## 1) What this code does (updated overview)

This script performs a **generalized least-squares (GLS)** fit of the **local Hubble residual field** using Type Ia Supernovae (SNe Ia) in a low-redshift sample (typically Pantheon+SH0ES low-z).

### Core observable

The pipeline builds **Hubble residuals** (distance-modulus residuals):

**Δμ ≡ μ_obs − μ_th(z; H₀, q₀)**

where **H₀** is obtained by grid minimization (and **q₀** is fixed by `--q0`).

The fit can be done either:
- with **diagonal errors** (WLS-style), or
- with **full covariance** (GLS), using the Pantheon-style STAT+SYS covariance (`--cov` + `--use_cov`).

### Signal model: TEX / KIN / MIX decomposition

The script supports a decomposition into nested models (all include a monopole/intercept **a₀**):

- **MONO**:  Δμ = a₀ + ε
- **TEX** (constant-in-z dipole):  Δμ = a₀ + A_TEX · (n · d̂_TEX) + ε
- **KIN** (kinematic bulk-flow-like dipole scaling as 1/z):  Δμ = a₀ + A_KIN · (n · d̂_KIN) / z + ε
- **MIX** (TEX + KIN together)

It reports:
- best-fit amplitudes and axes (equatorial + galactic),
- χ² and **Δχ²** for nested comparisons (e.g., MONO → TEX),
- nominal χ² p-values and empirical p-values (permutations),
- and derived bulk velocity scale (from the KIN coefficient; order-of-magnitude conversion).

### Footprint-aware empirical significance (permutations)

For low-z anisotropy on a **non-uniform footprint**, the code implements **empirical p-values** by permutation tests:
- global permutations (`--permute N`)
- within-survey permutations (`--permute_within_survey`)
- and (recommended with full covariance) **whitened-space permutations** (`--permute_whitened`).

### Tomography (redshift shells) and “locked-axis” morphology

With `--zbins`, the code runs redshift-shell fits.  
With `--fix_tex_axis` and `--fix_kin_axis`, it can enforce **locked axes** (from the global baseline) so that shell behavior reflects **amplitude evolution**, not direction wandering.

The “pillar plot” mode (`--make_pillar_plots`) produces the 4-panel diagnostic figure used in the paper:
- Δχ² from KIN by bin
- Δχ² from TEX (on top of KIN) by bin
- A_TEX(z)
- v_bulk(z)

### Robustness / diagnostics included

- **Influence diagnostics** (Cook’s distance in whitened space, plus optional exact drop-one refits)
- **Directional jackknife** (cap or hemisphere masking over many sky directions)
- **Constant-N control** across zmin (power vs geometry disentangling by forced equal N)
- **Milky Way (MW) dust / template checks** (optional nuisance regressors and correlations)

### CF4++ external cross-validation (T12/T13) and conjunctive null calibration (T14)

The script can sample an external reconstructed velocity field and convert it into a SN template t_CF4, then test:
- **T12**: global regression and absorption tests (does CF4++ absorb TEX/KIN?)
- **T13**: the same question **tomographically** on the locked-axis shell morphology
- **T14**: a **joint single-flow null** calibration on the Pantheon–CF4 conjunction using correlated mocks on the shell union.

CF4 input modes supported:
- a precomputed template with SN-aligned row order (`--cf4_template`)
- a public CF4++ NPZ grid product sampled at SN positions (`--cf4pp_npz`)
- FITS cubes for SGX/SGY/SGZ sampled at SN positions (`--cf4_sgx_grid`, etc.)

> Note: CF4 sampling and FITS reading require `astropy` (optional dependency).

### Dependencies

- Required: `numpy`, `scipy`, `matplotlib`
- Optional (CF4++ / FITS / geometry): `astropy`

---

## 2) Reproducibility protocols (Variants T0–T14)

Below are the **exact command-line protocols** used in the manuscript Supplementary test catalogue (S10).

### Path variables used in the paper examples

```powershell
$script   = ".\h0_dipole_v7.2.py"
$dat      = ".\DataRelease\Pantheon_Data\4_DISTANCES_AND_COVAR\Pantheon+SH0ES.dat"
$covFull  = ".\DataRelease\Pantheon_Data\4_DISTANCES_AND_COVAR\Pantheon+SH0ES_STAT+SYS.cov"
$covStat  = ".\DataRelease\Pantheon_Data\4_DISTANCES_AND_COVAR\Pantheon+SH0ES_STATONLY.cov"
$cf4pp    = ".\DataRelease\CF4\CF4pp_mean_std_grids.npz"
```

### Variant T0 — No MW cuts; full cov (effect of MW cuts)

Summary: run the global pipeline without the conservative Milky Way cuts to show their impact.

```text
python -X utf8 $script --dat $dat --cov $covFull --use_cov   --q0 -0.55 --sigint 0.07 --zmax 0.10 --zcol zHD --seed 0   --permute 200000 --permute_whitened --permute_within_survey
```

### Variant T1 — Baseline global fit + footprint-aware permutations (global axes + p_emp)

Summary: baseline discovery configuration; defines the reference TEX/KIN axes and empirical significance.

```text
python -X utf8 $script --dat $dat --cov $covFull --use_cov   --q0 -0.55 --bcut 20 --dustcut 0.10 --sigint 0.07   --zmax 0.10 --zcol zHD --seed 0   --permute 200000 --permute_whitened --permute_within_survey
```

Figure hook (sky map): add `--make_h0_residuals`.

### Variant T2 — Locked-axis tomography in redshift shells (“pillar” bins)

Summary: tomographic amplitude evolution with TEX/KIN axes locked to T1.

```text
python -X utf8 $script --dat $dat --cov $covFull --use_cov   --q0 -0.55 --sigint 0.07 --bcut 20 --dustcut 0.10 --zcol zHD   --zbins "0.020-0.025,0.025-0.030,0.030-0.035,0.035-0.040"   --zbins_global --seed 0   --fix_tex_axis "143.98,7.13" --fix_kin_axis "172.15,23.45"   --make_pillar_plots
```

### Variant T3 — Influence diagnostics (Cook’s distance + drop-one refits)

Summary: tests whether a small number of SNe dominate TEX/KIN/MIX fits.

```text
python -X utf8 $script --dat $dat --cov $covFull --use_cov   --q0 -0.55 --sigint 0.07 --bcut 20 --dustcut 0.10   --zmax 0.10 --zcol zHD --seed 0 --influence   --influence_mode mix --influence_top 30 --influence_dropone 20
```

### Variant T4 — Directional jackknife (cap masking)

Summary: masks a cap over many sky directions and refits, tracking axis rotation and Δχ² stability.

- **T4a (TEX)**

```text
python -X utf8 $script --dat $dat --cov $covFull --use_cov   --q0 -0.55 --sigint 0.07 --bcut 20 --dustcut 0.10   --zmax 0.10 --zcol zHD --seed 0 --jackknife_dir 96   --jackknife_dir_mode cap --jackknife_dir_theta 30   --jackknife_dir_compare tex   --jackknife_dir_csv jackknife_cap30_tex_fullcov.csv
```

- **T4b (KIN)**

```text
python -X utf8 $script --dat $dat --cov $covFull --use_cov   --q0 -0.55 --sigint 0.07 --bcut 20 --dustcut 0.10   --zmax 0.10 --zcol zHD --seed 0 --jackknife_dir 96   --jackknife_dir_mode cap --jackknife_dir_theta 30   --jackknife_dir_compare kin   --jackknife_dir_csv jackknife_cap30_kin_fullcov.csv
```

### Variant T5 — Constant-N control across zmin (power vs geometry)

Summary: keeps N fixed while raising zmin, to separate loss of leverage from loss of power.

```text
python -X utf8 $script --dat $dat --cov $covFull --use_cov   --q0 -0.55 --sigint 0.07 --bcut 20 --dustcut 0.10   --zmax 0.10 --zcol zHD --seed 0   --constN "0,0.003,0.005,0.01" --constN_draws 50   --constN_model tex --constN_use_cov   --constN_csv constN_tex_fullcov.csv
```

### Variant T6 — Frame test: z=zCMB (full cov)

Summary: tests sensitivity to redshift definition by using zCMB instead of zHD.

```text
python -X utf8 $script --dat $dat --cov $covFull --use_cov   --q0 -0.55 --bcut 20 --dustcut 0.10 --sigint 0.07   --zmax 0.10 --zcol zCMB --seed 0   --permute 200000 --permute_whitened --permute_within_survey
```

### Variant T7 — Diagonal covariance only (WLS stress test)

Summary: turns off full covariance and fits using diagonal errors only.

Fit only:

```text
python -X utf8 $script --dat $dat --q0 -0.55 --bcut 20   --dustcut 0.10 --sigint 0.07 --zmax 0.10 --zcol zHD --seed 0
```

Optional empirical p-value (non-whitened permutations):

```text
python -X utf8 $script --dat $dat --q0 -0.55 --bcut 20   --dustcut 0.10 --sigint 0.07 --zmax 0.10 --zcol zHD   --seed 0 --permute 200000 --permute_within_survey
```

### Variant T8 — sigma_v=200 km/s (full cov)

Summary: robustness to the assumed peculiar-velocity dispersion (adds sigma_mu ∝ sigv/(c z)).

```text
python -X utf8 $script --dat $dat --cov $covFull --use_cov   --q0 -0.55 --bcut 20 --dustcut 0.10 --sigint 0.07   --sigv 200 --zmax 0.10 --zcol zHD --seed 0   --permute 200000 --permute_whitened --permute_within_survey
```

### Variant T9 — sigma_v=350 km/s (full cov)

Summary: same as T8 with a larger peculiar-velocity dispersion.

```text
python -X utf8 $script --dat $dat --cov $covFull --use_cov   --q0 -0.55 --bcut 20 --dustcut 0.10 --sigint 0.07   --sigv 350 --zmax 0.10 --zcol zHD --seed 0   --permute 200000 --permute_whitened --permute_within_survey
```

### Variant T10 — Null mock tomography (correlated-noise shell false-positive rate)

Summary: correlated Gaussian mocks on the same footprint/covariance to estimate shell-morphology false positives.

```text
python -X utf8 $script --dat $dat --cov $covFull --use_cov   --q0 -0.55 --sigint 0.07 --bcut 20 --dustcut 0.10 --zcol zHD   --zbins "0.020-0.025,0.025-0.030,0.030-0.035,0.035-0.040"   --zbins_global --seed 0 --fix_tex_axis "143.98,7.13"   --fix_kin_axis "172.15,23.45" --null_mock_tomo --nmock 100000
```

### Variant T11 — Survey-marginalized global fit + within-survey, whitened permutations

Summary: adds survey intercept nuisance parameters and refits them in each permutation to test survey-heterogeneity absorption.

```text
python -X utf8 $script --dat $dat --cov $covFull --use_cov   --q0 -0.55 --bcut 20 --dustcut 0.10 --sigint 0.07   --zmax 0.10 --zcol zHD --seed 0   --permute 200000 --permute_whitened --permute_within_survey   --marginalize_surveys
```

### Variant T12 — External CF4++ full-sample regression

Summary: regress Pantheon residuals on an external CF4++ template and test TEX/KIN absorption.

- **T12a (default CF4 geometry zCMB)**

```text
python $script --dat $dat --cov $covFull --use_cov --q0 -0.55 --sigint 0.07   --bcut 20 --dustcut 0.10 --zmax 0.10 --zcol zHD --cf4_zcol zCMB   --cf4pp_npz $cf4pp --cf4_grid_units Mpc/h --cf4_h 0.746   --cf4_vel_scale 1 --cf4_h0_geom 74.6 --cf4_test   --marginalize_surveys --seed 0
```

- **T12b (CF4 geometry sampled with zHD)**

```text
python $script --dat $dat --cov $covFull --use_cov --q0 -0.55 --sigint 0.07   --bcut 20 --dustcut 0.10 --zmax 0.10 --zcol zHD --cf4_zcol zHD   --cf4pp_npz $cf4pp --cf4_grid_units Mpc/h --cf4_h 0.746   --cf4_vel_scale 1 --cf4_h0_geom 74.6 --cf4_test   --marginalize_surveys --seed 0
```

### Variant T13 — CF4++ tomography on the T2 shell morphology

Summary: tomographic CF4++ absorption test on the locked-axis shell bins from T2/T1.

- **T13a (default CF4 geometry zCMB)**

```text
python $script --dat $dat --cov $covFull --use_cov --q0 -0.55 --sigint 0.07   --bcut 20 --dustcut 0.10 --zmax 0.10 --zcol zHD --cf4_zcol zCMB   --cf4pp_npz $cf4pp --cf4_grid_units Mpc/h --cf4_h 0.746   --cf4_vel_scale 1 --cf4_h0_geom 74.6 --cf4_tomo   --zbins "0.020-0.025,0.025-0.030,0.030-0.035,0.035-0.040"   --zbins_global --fix_tex_axis "143.98,7.13"   --fix_kin_axis "172.15,23.45" --seed 0
```

- **T13b (PV-stress: sigv=250; CF4 geometry zCMB)**

```text
python $script --dat $dat --cov $covFull --use_cov --q0 -0.55 --sigint 0.07   --sigv 250 --bcut 20 --dustcut 0.10 --zmax 0.10 --zcol zHD   --cf4_zcol zCMB --cf4pp_npz $cf4pp --cf4_grid_units Mpc/h   --cf4_h 0.746 --cf4_vel_scale 1 --cf4_h0_geom 74.6 --cf4_tomo   --zbins "0.020-0.025,0.025-0.030,0.030-0.035,0.035-0.040"   --zbins_global --fix_tex_axis "143.98,7.13"   --fix_kin_axis "172.15,23.45" --seed 0
```

- **T13c (sensitivity: CF4 geometry zHD + sigv=250)**

```text
python $script --dat $dat --cov $covFull --use_cov --q0 -0.55 --sigint 0.07   --sigv 250 --bcut 20 --dustcut 0.10 --zmax 0.10 --zcol zHD   --cf4_zcol zHD --cf4pp_npz $cf4pp --cf4_grid_units Mpc/h   --cf4_h 0.746 --cf4_vel_scale 1 --cf4_h0_geom 74.6 --cf4_tomo   --zbins "0.020-0.025,0.025-0.030,0.030-0.035,0.035-0.040"   --zbins_global --fix_tex_axis "143.98,7.13"   --fix_kin_axis "172.15,23.45" --seed 0
```

### Variant T14 — Joint single-flow null on the Pantheon–CF4 conjunction

Summary: correlated mocks under the simple null M0 = MONO + β_CF4 t_CF4 on the shell union, estimating the joint probability of the shell morphology + non-absorption pattern.

```text
python $script --dat $dat --cov $covFull --use_cov --q0 -0.55 --sigint 0.07 --bcut 20 --dustcut 0.10 --zmax 0.10 --zcol zHD --cf4_zcol zCMB --cf4pp_npz $cf4pp --cf4_grid_units Mpc/h --cf4_h 0.746 --cf4_vel_scale 1 --cf4_h0_geom 74.6 --zbins "0.020-0.025,0.025-0.030,0.030-0.035,0.035-0.040" --zbins_global --fix_tex_axis "143.98,7.13" --fix_kin_axis "172.15,23.45" --t14_joint_null --nmock 10000000 --seed 0 --t14_csv t14_joint.csv
```

---

## 3) CLI options reference (grouped)

Below is a curated list of the most useful options, grouped by domain.

### I/O and column mapping

- `--dat <path>` *(required)*: input SN table (named columns).
- `--cov <path>`: Pantheon-style covariance `.cov` (STAT+SYS).
- `--use_cov`: enable GLS with full covariance (requires `--cov`).
- `--zcol <name>`: redshift column for residual fit (typically `zHD`; variants use `zCMB`).
- `--mu <name>`, `--muerr <name>`, `--ra <name>`, `--dec <name>`: override auto-detected columns.
- `--surveycol <name>`: survey-ID column (e.g. `IDSURVEY`) for stratified permutations/jackknife.

### Cosmology / fit controls

- `--q0 <float>`: fixed deceleration parameter used in μ_th(z).
- `--h0min <float>`, `--h0max <float>`, `--h0n <int>`: H0 grid search range and resolution.
- `--sigint <float>`: intrinsic scatter (added to diagonal; also to GLS diagonal if `--use_cov`).
- `--sigv <float>`: additional peculiar velocity dispersion in km/s (converted to σ_μ ∝ sigv/(c z)).

### Sample selection cuts

- `--zmin <float>`, `--zmax <float>`: redshift cuts.
- `--bcut <deg>`: keep only SNe with |b| ≥ bcut (galactic latitude).
- `--dustcut <float>`: keep only SNe with dust proxy ≤ dustcut.
- `--dustcol <name>`: dust column name (auto-detected if omitted).

### TEX/KIN axis handling and tomography

- `--fix_tex_axis "l,b"`: lock TEX axis in Galactic degrees.
- `--fix_kin_axis "l,b"`: lock KIN axis in Galactic degrees.
- `--zbins "z1-z2,z2-z3,..."`: define tomography bins (then prints/uses bin table).
- `--zbins_global`: with fixed axes + zbins, fits a global model on the union and reports per-bin χ² splits.
- `--make_pillar_plots`: generate the 4-panel pillar figure.
- `--make_h0_residuals`: generate a sky map of Δμ residuals.

### Empirical significance (permutations) and debug nulls

- `--permute <int>`: number of permutations (0 disables).
- `--permute_within_survey`: stratified permutations within each survey (needs `IDSURVEY` or `--surveycol`).
- `--permute_whitened`: permute whitened residuals (recommended with `--use_cov`).
- `--permute_plot <prefix>`: save null distributions (CSV/PNG) for the permutation tests.
- `--null_gaussian`: debug mode (replaces whitened data with N(0,1) noise).
- `--nmock <int>`: number of Gaussian debug mocks for `--null_gaussian`.

### Survey marginalization / robustness diagnostics

- `--marginalize_surveys`: add survey intercept nuisance parameters (and refit them in each permutation when enabled).
- `--influence`: run influence diagnostics (Cook’s distance + optional exact drop-one refits).
- `--influence_mode {mono,tex,kin,mix}`
- `--influence_top <int>`, `--influence_dropone <int>`
- `--jackknife`: leave-one-survey-out.
- `--jackknife_dir <int>`: directional jackknife with Ndir directions.
- `--jackknife_dir_mode {cap,hemisphere}`, `--jackknife_dir_theta <deg>`
- `--jackknife_dir_compare {tex,kin,mix_tex,mix_kin}`
- `--jackknife_dir_csv <path>`: save jackknife outputs to CSV.
- `--constN "<zmin1,zmin2,...>"`: constant-N control across zmin list.
- `--constN_draws <int>`, `--constN_N0 <int>`, `--constN_model {tex,kin,mix}`, `--constN_use_cov`, `--constN_csv <path>`
- `--null_mock_tomo`: correlated-noise shell morphology false-positive test (tomography mocks).

### Milky Way (MW) checks and sky exports

- `--mwcheck`: enable MW template checks.
- `--mwtemp {sinb,abs_sinb,inv_abs_sinb,p2}`, `--mw_sinb_min <float>`
- `--dustcheck`: enable dust correlation + nuisance regression.
- `--export_sky <prefix>`: export sky grids (pred + resid).
- `--sky_step <deg>`, `--zstar <float>`

### CF4++ external template (T12/T13) options

- `--cf4_test`: run global external-template regression (T12 logic).
- `--cf4_tomo`: run locked-axis tomography with CF4 template (T13 logic).
- `--cf4_template <path>`: precomputed SN-aligned template with columns like `CF4_DMU` or `CF4_VRAD`.
- `--cf4pp_npz <path>`: CF4++ NPZ grid product sampled at SN positions.
- `--cf4_sgx_grid/--cf4_sgy_grid/--cf4_sgz_grid <path>`: FITS cubes for SG velocity components.
- `--cf4_export_template <path>`: export sampled CF4 template table.
- `--cf4_zcol <name>`: redshift column used only for CF4 geometry/template conversion (recommended `zCMB`).
- `--cf4_h0_geom <float>`, `--cf4_q0_geom <float>`: geometry conversion parameters.
- `--cf4_grid_units {Mpc,Mpc/h}`, `--cf4_h <float>`
- `--cf4_vel_scale <float>`: scale applied to sampled CF4 velocities before conversion to Δμ (note in help: some official CF4 products may require 52).

### T14 joint null options

- `--t14_joint_null`: enable the joint single-flow null calibration (T14).
- `--t14_peak_bin <int>`, `--t14_adj_bin <int>`: 1-based bin indices for peak and adjacent shells.
- `--t14_peak_min <float>`, `--t14_adj_max <float>`, `--t14_cf4_union_max <float>`, `--t14_tex_cf4_min <float>`: thresholds (default: use observed values from the run).
- `--t14_batch <int>`: mock batch size.
- `--t14_csv <path>`: export per-mock T14 metrics to CSV.

---

## Quick sanity run (suggested)

If you just want to confirm everything is wired correctly before running the full battery, run the baseline (T1) with a small permutation count first, e.g. `--permute 1000`, then scale up.
