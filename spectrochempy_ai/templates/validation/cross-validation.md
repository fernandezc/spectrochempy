# Cross-Validation

**template_id:** `cross_validation`
**version:** `0.1.0`
**compatible_registry:** `0.1`

---

## Scientific Objective

Estimate the predictive performance of a calibration model by
systematically partitioning the dataset into training and test
subsets. Cross-validation provides a more reliable estimate of
prediction error than the calibration fit itself and is the standard
method for selecting the optimal number of latent variables.

---

## Methodology (Provisional)

| Step | Operation | Purpose |
|---|---|---|
| 1 | `load` | Load spectral data |
| 2 | `load` | Load reference values |
| 3 | `inspect` | Validate datasets |
| 4 | `baseline` | Baseline correction |
| 5 | `pls` | PLS model |
| 6 | `crossval` | Cross-validation |

---

## Related Templates

- `pls_calibration` — Calibration model to validate
- `residual_analysis` — Post-validation diagnostics
- `model_comparison` — Compare multiple models

---

## Status

**Draft.** Not yet implemented.
