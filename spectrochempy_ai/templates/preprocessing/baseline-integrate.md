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

## Methodology

| Step | Operation | Purpose |
|---|---|---|
| 1 | `load` | Load spectral data |
| 2 | `inspect` | Validate dataset |
| 3 | `baseline` | Baseline correction |
| 4 | `integrate` | Peak area integration |
| 5 | `plot` | Plot corrected spectrum with annotated area |
| 6 | `inspect` | Report integrated value |

---

## Related Templates

- `smoothing` — Noise reduction before integration
- `derivatives` — Alternative baseline removal
- `exploratory_pca` — Downstream analysis after preprocessing

---

## Notes

- The default workflow does **not** apply smoothing. For single spectra,
  smoothing should only be introduced when noise clearly impairs
  integration and the band shape is broad enough to preserve.
- The default integration uses the full selected `x` range. Restrict the
  spectral window before running the workflow if only one band should be
  quantified.

## Status

**Stable.** Template, rules, renderer support, tests, and real-dataset
validation are implemented.
