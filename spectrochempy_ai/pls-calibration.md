# PLS Calibration

**template_id:** `pls_calibration`
**version:** `0.1.0`
**compatible_registry:** `0.1`

---

## Scientific Objective

Build a quantitative calibration model relating spectral measurements to
reference analytical values using Partial Least Squares (PLS) regression.

PLS is the standard chemometric method for multivariate calibration when
spectral predictors are collinear. Unlike ordinary least squares, PLS finds
latent variables (components) that maximise covariance between the spectra
and the property of interest, producing a model that is robust to
collinearity and can handle more variables than observations.

---

## Methodology

This template implements a complete PLS calibration workflow consisting of
seven ordered steps:

| Step | Operation | Purpose |
|---|---|---|
| 1 | `load` | Load spectral data (X matrix) |
| 2 | `load` | Load reference values (y vector) |
| 3 | `inspect` | Validate spectral dataset |
| 4 | `inspect` | Validate reference values |
| 5 | `baseline` | Remove additive baseline drift |
| 6 | `pls` | Fit PLS regression model |
| 7 | `pls_predict_plot` | Display predicted values |

---

## Step Rationales

### Step 1: Load spectral data

Load the spectral dataset from a portable file. This is the predictor
matrix X for the PLS model. Each row is an observation (sample); each
column is a spectral variable (wavelength, wavenumber, chemical shift).

**Parameters:**
- `filename` (default: `"spectra.scp"`): Path to spectral data file.
- `format` (default: `"scp"`): File format specification.

### Step 2: Load reference values

Load the reference analytical values from a separate file. These are the
response values y that the PLS model will predict from spectra. The
reference values must correspond one-to-one with the spectral
observations.

**Parameters:**
- `filename` (default: `"reference.csv"`): Path to reference values file.
- `format` (default: `"csv"`): File format specification.

### Step 3–4: Inspect datasets

Validate both datasets before model fitting:
- Check dimensions match (same number of observations in X and y)
- Verify no missing values or anomalies
- Confirm reference values span the calibration range of interest

### Step 5: Baseline correction

Remove additive baseline drift from spectra before PLS fitting. Baseline
artefacts add uninformative variance that can degrade model performance
and make coefficients less interpretable.

**Parameter:**
- `method` (default: `"asls"`): Baseline correction algorithm.

### Step 6: PLS regression

Fit a Partial Least Squares regression model. PLS finds latent variables
(components) that maximise covariance between the spectral predictors and
the reference values.

The algorithm iteratively:
1. Finds the direction in X-space that has maximum covariance with y
2. Deflates X and y by removing the variance explained by this component
3. Repeats until the desired number of components is extracted

**Parameter:**
- `n_components` (default: `3`): Number of latent variables. Should be
  chosen via cross-validation; 3 is a starting point for exploration.

**Reference:** Wold et al., "PLS-regression: a basic tool of chemometrics,"
*Chemom. Intell. Lab. Syst.* 58(2), 109–130, 2001.

### Step 7: Predicted values

Generate predictions from the fitted PLS model and display them. Compare
predicted values to reference values to assess calibration quality. The
R² score indicates the proportion of variance in y explained by the
model.

---

## Assumptions

1. Spectra (X) are 2D (observations × spectral variables) with a
   continuous spectral axis.
2. Reference values (y) are 1D or 2D (observations × targets).
3. The relationship between spectra and reference is approximately linear
   in the latent-variable space.
4. The number of observations exceeds the number of PLS components.
5. Reference values span the calibration range of interest.
6. Observations in X and y are aligned (same order, same samples).

---

## Limitations

1. **Linearity.** PLS assumes a linear relationship in latent-variable
   space. Strong non-linearities require non-linear methods (SVM, neural
   networks, polynomial PLS).

2. **Component selection.** The default `n_components=3` is arbitrary.
   The optimal number should be chosen via cross-validation. No automatic
   selection is performed in this template.

3. **No outlier detection.** Influential samples can distort the model.
   Leverage and residual diagnostics should be inspected manually.

4. **No variable selection.** Uninformative spectral regions are included
   in the model. Variable selection (e.g., competitive adaptive reweighted
   sampling, interval PLS) can improve performance.

5. **Single dataset use.** The template uses the calibration set for both
   fitting and prediction. A proper validation requires an independent
   test set.

6. **No cross-validation.** Cross-validation for component selection is
   not implemented in this template. Add it manually for robust model
   validation.

7. **Known SpectroChemPy bug.** `PLSRegression` has a coordinate-wrapping
   issue when `y` is a 1D `NDDataset` created from scratch. This affects
   notebook execution with synthetic data. The template itself is correct;
   the bug is in the underlying SPeC PLS implementation and should be
   resolved upstream.

---

## Expected Outputs

| Output | Type | Description |
|---|---|---|
| `dataset_corrected` | dataset | Baseline-corrected spectral dataset |
| `pls_result` | result | PLS model with coefficients, scores, loadings, weights, and intercept |

The `pls_result.result` object exposes:
- `coef`: Regression coefficients (variables × targets)
- `intercept`: Model intercept
- `x_scores`, `y_scores`: Latent variable scores
- `x_loadings`, `y_loadings`: Variable loadings
- `x_weights`, `y_weights`: PLS weights
- `diagnostics`: Number of iterations, convergence info

---

## Interpretation Guidance

1. Examine the **R² score** (via `pls.score(X, y)`) to assess overall
   fit quality.
2. Inspect **coefficients** to identify which spectral regions contribute
   most to the prediction.
3. Compare **predicted vs reference** values to check for systematic bias
   or outliers.
4. If R² is low, consider: more components, better preprocessing,
   variable selection, or non-linear methods.

---

## When Not to Use This Template

- The relationship between spectra and reference is strongly non-linear.
- The number of observations is smaller than the number of components.
- The goal is classification rather than regression (use PLS-DA instead).
- The goal is exploratory analysis without a reference variable (use PCA).

---

## Reproducibility Guarantees

Identical template version, registry version, parameters, and input data
produce identical `WorkflowPlan` objects. The framework guarantees
deterministic plan generation.

The numerical results of the underlying PLS fit depend on the specific
library implementation and may vary across versions or platforms.
Reproducibility of numerical results should be verified independently.

---

## References

- Wold et al., "PLS-regression: a basic tool of chemometrics,"
  *Chemom. Intell. Lab. Syst.* 58(2), 109–130, 2001.
- Geladi & Kowalski, "Partial least-squares regression: a tutorial,"
  *Anal. Chim. Acta* 185, 1–17, 1986.
- Hoskuldsson, "PLS regression methods," *J. Chemom.* 2(3), 211–228, 1988.
