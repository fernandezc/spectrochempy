# Calibration (Quantification)

## Scientific Objective

Build a quantitative model that predicts an analytical property (concentration,
purity, hardness, etc.) from spectral measurements. Calibration is the core
of applied chemometrics: once validated, a spectral model can replace slower,
more expensive reference analyses.

## Typical Questions

- Can this analyte be quantified from the spectral fingerprint?
- Which spectral regions are most predictive?
- How many latent variables are needed for a robust model?
- What is the expected prediction error?

## Available Methodologies

| Status | Template | Description |
|---|---|---|
| ✓ | `pls_calibration` | PLS regression with reference values |
| □ | `pcr` | Principal Component Regression |
| □ | `svm_regression` | Support Vector Regression (non-linear) |

## How to Choose

1. `pls_calibration` is the default for linear spectral calibration.
   It handles collinear predictors and works well when the number of
   variables exceeds the number of samples.
2. `pcr` is an alternative that separates dimensionality reduction
   (PCA) from regression. It may underperform PLS when the predictive
   information is in low-variance components.
3. `svm_regression` is recommended when the relationship between
   spectra and the response is non-linear.
