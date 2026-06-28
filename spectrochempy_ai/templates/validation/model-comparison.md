# Model Comparison

**template_id:** `model_comparison`
**version:** `0.1.0`
**compatible_registry:** `0.1`

---

## Scientific Objective

Compare the predictive performance of multiple modelling approaches
(PLS, PCR, SVM) on the same calibration dataset. Model comparison
provides evidence for selecting the most appropriate method: if PLS
and PCR give equivalent performance, the simpler model is preferred;
if the non-linear method (SVM) significantly outperforms linear
methods, systematic non-linearity is indicated.

---

## Related Templates

- `pls_calibration` — Linear baseline model
- `pcr` — Alternative linear model
- `svm_regression` — Non-linear model
- `cross_validation` — Consistent validation across models

---

## Status

**Draft.** Not yet implemented.
