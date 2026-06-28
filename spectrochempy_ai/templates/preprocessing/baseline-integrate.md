# Baseline Correction and Integration

**template_id:** `baseline_integrate`
**version:** `0.1.0`
**compatible_registry:** `0.1`

---

## Scientific Objective

Remove additive baseline drift from one-dimensional spectra and
quantify specific spectral bands by peak area integration. This is the
most fundamental preprocessing workflow: many downstream analyses
(calibration, classification, monitoring) depend on reliable baseline
removal and reproducible integration.

---

## Methodology (Provisional)

| Step | Operation | Purpose |
|---|---|---|
| 1 | `load` | Load spectral data |
| 2 | `inspect` | Validate dataset |
| 3 | `baseline` | Baseline correction |
| 4 | `integrate` | Peak area integration |

---

## Related Templates

- `smoothing` — Noise reduction before integration
- `derivatives` — Alternative baseline removal
- `exploratory_pca` — Downstream analysis after preprocessing

---

## Status

**Draft.** Template definition, rules, and operations defined in
`template_planner.py`. Needs renderer and real-dataset validation.
