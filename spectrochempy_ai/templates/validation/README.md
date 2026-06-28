# Validation

## Scientific Objective

Assess the reliability, robustness, and predictive power of a chemometric
model before deploying it on new data. Validation is the bridge between
calibration and application: a model that fits the calibration data well
may fail on unseen samples if it captures noise (overfitting) or if the
calibration set does not represent future observations.

## Typical Questions

- How many latent variables should I retain to avoid overfitting?
- What is the expected prediction error on new samples?
- Are there spectral residuals that indicate unmodelled phenomena?
- Which model (PLS, PCR, SVM) performs best on this dataset?

## Available Methodologies

| Status | Template | Description |
|---|---|---|
| □ | `cross_validation` | Venetian blinds, random subsets, leave-one-out |
| □ | `residual_analysis` | Spectral residuals, Q residuals, Hotelling T² |
| □ | `model_comparison` | Compare PLS, PCR, SVM on the same dataset |

## How to Choose

1. `cross_validation` is the first step for any calibration model —
   it estimates the optimal complexity and expected prediction error.
2. `residual_analysis` diagnoses whether the model captures all
   systematic variation or whether unmodelled phenomena remain.
3. `model_comparison` when you need to select among competing
   modelling approaches for the same analytical problem.
