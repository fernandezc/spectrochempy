# MCR-ALS Analysis

**template_id:** `mcrals_analysis`
**version:** `0.1.0`
**compatible_registry:** `0.1`

---

## Scientific Objective

Resolve a multivariate spectral mixture into chemically meaningful
concentration profiles and pure component spectra using Multivariate Curve
Resolution by Alternating Least Squares (MCR-ALS).

Unlike PCA, which produces orthogonal mathematical components, MCR-ALS
enforces physical constraints (non-negativity, unimodality) and yields
profiles that can be interpreted as real chemical components. MCR-ALS is
the standard method for mixture resolution in analytical chemistry when the
goal is to recover physically interpretable concentration and spectral
profiles.

---

## Methodology

This template implements a complete MCR-ALS workflow consisting of seven
ordered steps:

| Step | Operation | Purpose |
|---|---|---|
| 1 | `load` | Load spectral mixture data from a portable file |
| 2 | `inspect` | Validate dataset quality before processing |
| 3 | `baseline` | Remove additive baseline drift |
| 4 | `mcrals_init` | Generate initial concentration guess |
| 5 | `mcrals` | Fit MCR-ALS model to resolve mixture |
| 6 | `mcrals_conc_plot` | Visualise resolved concentration profiles |
| 7 | `mcrals_spec_plot` | Visualise resolved pure spectra |

---

## Step Rationales

### Step 1: Load spectral mixture

Real spectral mixture data must be loaded from an external source. The `load`
operation reads a portable file format (default: SCP) and produces a spectral
dataset. MCR-ALS operates on the full mixture matrix, so the dataset must
contain all observations and all spectral variables.

**Parameters:**
- `filename` (default: `"data.scp"`): Path to input file.
- `format` (default: `"scp"`): File format specification.

### Step 2: Inspect mixture quality

Before any processing, inspect the dataset dimensions, coordinate ranges, and
value distribution. Early detection of anomalies — missing values, negative
absorbance, irregular sampling — prevents misleading downstream results and
saves time. For MCR-ALS in particular, check that the data are approximately
non-negative and that the mixture is additive.

### Step 3: Baseline correction

Baseline drift is an additive artefact that can distort concentration
profiles and prevent physically meaningful MCR-ALS solutions. Residual
baseline inflates the apparent number of components and can introduce
unphysical oscillations in resolved profiles.

The Asymmetric Least Squares (ASLS) method is used by default. It models the
baseline as a smoothly varying signal, weighted asymmetrically to fit below
the spectrum.

**Parameter:**
- `method` (default: `"asls"`): Baseline correction algorithm.

**Reference:** Eilers & Boelens, "Baseline correction with asymmetric least
squares smoothing," Leiden University Medical Centre Report, 2005.

### Step 4: Initial concentration guess

MCR-ALS requires an initial estimate for either concentration profiles or
pure spectra. The algorithm alternates between estimating C (concentrations)
and S^T (spectra) from this initial guess.

This template generates a simple initial guess using spaced Gaussian profiles
along the observation axis. This is a scientifically plausible starting point
for chromatographic or kinetic data where concentration profiles are expected
to be unimodal and non-overlapping.

**Important:** For production work, replace this simple guess with a
chemically informed estimate from:
- Evolving Factor Analysis (EFA)
- SIMPLISMA
- Pure variable methods
- Domain knowledge

**Parameter:**
- `n_components` (default: `3`): Number of chemical components.

### Step 5: MCR-ALS mixture resolution

Fit the MCR-ALS model to resolve the mixture into concentration profiles (C)
and pure component spectra (S^T). MCR-ALS iteratively alternates between:

1. Fixing S^T and solving for C (concentration update)
2. Fixing C and solving for S^T (spectral update)

At each iteration, physical constraints are applied:
- **Non-negativity:** Both C and S^T are constrained to be non-negative.
- **Unimodality:** Concentration profiles are constrained to have a single
  maximum.

The iteration continues until convergence (change below tolerance) or until
the maximum number of iterations is reached.

**Parameters:**
- `n_components` (default: `3`): Number of chemical components.
- `max_iter` (default: `100`): Maximum number of ALS iterations.

**References:**
- Tauler, "Multivariate curve resolution applied to second order data,"
  *Chemom. Intell. Lab. Syst.* 30(1), 133–146, 1995.
- Jaumot et al., "A graphical user-friendly interface for MCR-ALS: a new
tool for multivariate curve resolution in MATLAB," *Chemom. Intell. Lab.
Syst.* 76(1), 101–110, 2005.

### Step 6: Concentration profiles plot

Visualise the resolved concentration profiles for each component. These
profiles should be:

- **Non-negative:** All values ≥ 0 (enforced by constraint).
- **Unimodal:** Single maximum per profile (enforced by constraint).
- **Chemically interpretable:** Peaks should correspond to expected
  concentration changes (e.g., elution peaks, reaction progress).

Check for unexpected oscillations, negative values, or multiple maxima that
may indicate convergence issues or an incorrect number of components.

### Step 7: Resolved pure spectra plot

Visualise the resolved pure component spectra. These spectra should show:

- **Expected spectral features:** Peaks, bands, or patterns consistent with
  the chemical system.
- **Non-negativity:** All values ≥ 0 (enforced by constraint).
- **Distinctness:** Different components should have different spectral
  signatures.

Unphysical features (negative peaks, sharp oscillations, flat spectra) may
indicate an incorrect number of components, poor initial guess, or
insufficient constraints.

---

## Assumptions

The template makes the following assumptions about the input data:

1. Data are two-dimensional (observations × spectral variables).
2. The spectral axis is continuous.
3. The mixture is approximately additive: X ≈ C · S^T + E.
4. The number of chemical components is known or can be estimated a priori.
5. Concentration profiles are non-negative and unimodal.
6. Pure component spectra are non-negative.
7. Baseline has been removed or is negligible.
8. The observation axis represents a meaningful ordering (time, elution,
   temperature, etc.).

---

## Limitations

1. **Initial guess dependency.** MCR-ALS is not guaranteed to converge to the
global optimum. Different initial guesses can lead to different solutions
(rotational ambiguity). The simple Gaussian profiles used in this template
are a starting point; production work requires chemically informed
initialization.

2. **Rotational ambiguity.** Even with constraints, MCR-ALS solutions are not
unique. Additional constraints (closure, known pure spectra, kinetic models)
may be needed to resolve ambiguity.

3. **Component count sensitivity.** Underestimation merges components;
overestimation introduces artefacts. The number of components must be
specified correctly. Use EFA or SIMPLISMA to estimate the rank before
running MCR-ALS.

4. **Baseline sensitivity.** Residual baseline can distort concentration
profiles. Ensure thorough baseline correction before MCR-ALS.

5. **No closure or kinetic constraints.** This template applies only
non-negativity and unimodality. If the chemical system requires closure
(mass balance) or kinetic constraints, add them manually.

6. **Outlier sensitivity.** Outliers or strongly varying baselines across
observations can prevent convergence or produce unphysical profiles.

7. **No uncertainty quantification.** MCR-ALS does not provide confidence
intervals for resolved profiles. Bootstrap or Monte Carlo approaches are
needed for statistical inference.

8. **Simplified initialization.** The Gaussian initial guess assumes
non-overlapping, equally spaced concentration profiles. This is appropriate
for some chromatographic data but not for all chemical systems.

---

## Expected Outputs

| Output | Type | Description |
|---|---|---|
| `dataset_corrected` | dataset | Baseline-corrected mixture data |
| `conc_guess` | dataset | Initial concentration guess |
| `mcrals_result` | result | MCR-ALS model with C, S^T, and diagnostics |

Additionally, the following plots are produced as side effects:

| Plot | Description |
|---|---|
| Concentration profiles | Resolved concentration profiles for each component |
| Pure spectra | Resolved pure component spectra |

The `mcrals_result.result` object exposes:
- `C`: Resolved concentration profiles (observations × components)
- `components` / `St`: Resolved pure spectra (components × variables)
- Diagnostics: iterations, convergence status, residual standard deviation

---

## Interpretation Guidance

1. Start with the **concentration profiles**. Check that they are non-negative,
   unimodal, and chemically interpretable.
2. Examine the **pure spectra**. They should show expected spectral features
   and be non-negative.
3. Check **convergence** via `mcrals_result.result.diagnostics`. If
   `converged` is False, increase `max_iter` or improve the initial guess.
4. If profiles look unphysical (oscillations, negative values), consider:
   - Increasing the number of components
   - Improving the initial guess
   - Adding additional constraints (closure, hard constraints)
   - Removing outliers

---

## Related Templates

- `efa` — Rank estimation to determine the number of components
- `simplisma` — Pure variable selection for improved initial guess
- `exploratory_pca` — Exploratory analysis when mixture resolution is not
  needed
- `nmf_exploration` — Blind source separation without hard constraints

---

## When Not to Use This Template

- The data are not a mixture (single component system).
- The number of components is unknown and cannot be estimated.
- The chemical system requires closure or kinetic constraints not provided.
- The goal is exploratory dimensionality reduction (use PCA instead).
- The goal is blind source separation without physical constraints (use NMF).

---

## Reproducibility Guarantees

Identical template version, registry version, parameters, and input data
produce identical `WorkflowPlan` objects. The framework guarantees
deterministic plan generation.

The numerical results of the underlying operations (MCR-ALS fit, baseline
correction) depend on the specific library implementations and may vary
across versions or platforms. Reproducibility of numerical results should
be verified independently.

The MCR-ALS solution is sensitive to the initial guess. The Gaussian initial
guess used in this template is deterministic (fixed seed and spacing), but
chemically informed initial guesses will yield different results.

---

## References

- Tauler, "Multivariate curve resolution applied to second order data,"
  *Chemom. Intell. Lab. Syst.* 30(1), 133–146, 1995.
- Jaumot et al., "A graphical user-friendly interface for MCR-ALS: a new
tool for multivariate curve resolution in MATLAB," *Chemom. Intell. Lab.
Syst.* 76(1), 101–110, 2005.
- de Juan & Tauler, "Multivariate Curve Resolution (MCR) from 2000: progress
in concepts and applications," *Crit. Rev. Anal. Chem.* 36(3–4), 163–176, 2006.
- Eilers & Boelens, "Baseline correction with asymmetric least squares
  smoothing," Leiden University Medical Centre Report, 2005.
