# Principal Component Regression (PCR)

**template_id:** `pcr`
**version:** `0.1.0`
**compatible_registry:** `0.1`

---

## Scientific Objective

Build a calibration model by combining PCA dimensionality reduction
with linear regression. PCR first decomposes the spectral matrix into
principal components, then regresses the response variable against a
subset of those components. Unlike PLS, the dimension reduction step
is blind to the response.

---

## Related Templates

- `pls_calibration` — Supervised latent variable regression
- `exploratory_pca` — PCA decomposition step
- `cross_validation` — Component selection

---

## Status

**Draft.** Not yet implemented.
