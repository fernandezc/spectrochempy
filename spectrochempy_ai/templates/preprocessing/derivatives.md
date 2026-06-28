# Derivatives

**template_id:** `derivatives`
**version:** `0.1.0`
**compatible_registry:** `0.1`

---

## Scientific Objective

Enhance spectral resolution and remove additive baseline artefacts
using first and second derivatives. The first derivative removes
constant baseline offset; the second derivative removes linear
baseline while sharpening spectral features. Derivatives are typically
computed via the Savitzky–Golay algorithm, which combines smoothing
and differentiation in a single step.

---

## Methodology (Provisional)

| Step | Operation | Purpose |
|---|---|---|
| 1 | `load` | Load spectral data |
| 2 | `inspect` | Validate dataset |
| 3 | `derivative` | Savitzky–Golay differentiation |

---

## Related Templates

- `smoothing` — Noise reduction without differentiation
- `baseline_integrate` — Alternative baseline removal strategy
- `pls_calibration` — Downstream calibration on derivative spectra

---

## Status

**Draft.** Not yet implemented.
