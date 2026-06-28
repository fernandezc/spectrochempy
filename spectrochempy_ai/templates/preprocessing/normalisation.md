# Normalisation

**template_id:** `normalisation`
**version:** `0.1.0`
**compatible_registry:** `0.1`

---

## Scientific Objective

Standardise spectral data to correct for multiplicative artefacts
(path-length variation, sample thickness, scattering). Common methods
include Standard Normal Variate (SNV), Multiplicative Scatter
Correction (MSC), and area normalisation.

---

## Methodology (Provisional)

| Step | Operation | Purpose |
|---|---|---|
| 1 | `load` | Load spectral data |
| 2 | `inspect` | Validate dataset |
| 3 | `snv` | SNV normalisation |

---

## Related Templates

- `baseline_integrate` — Baseline removal before or after normalisation
- `smoothing` — Noise reduction before normalisation
- `pls_calibration` — Downstream calibration after preprocessing

---

## Status

**Draft.** Not yet implemented.
