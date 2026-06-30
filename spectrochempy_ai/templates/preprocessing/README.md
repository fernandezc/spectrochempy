# Preprocessing

## Scientific Objective

Remove artefacts and standardise spectral data before modelling.
Preprocessing is rarely the goal itself — it is a prerequisite for reliable
exploration, calibration, or resolution. The right preprocessing depends on
the physical nature of the artefact and the requirements of the downstream
model.

## Typical Questions

- Does the baseline drift across observations?
- Are the spectra affected by scatter or path-length variation?
- Should I smooth or differentiate to enhance resolution?
- Which preprocessing combination gives the best model performance?

## Available Methodologies

| Status | Template | Description |
|---|---|---|
| ✓ | `baseline_integrate` | Baseline correction + peak area integration on a single spectrum |
| □ | `smoothing` | Savitzky–Golay, moving average |
| □ | `normalisation` | SNV, MSC, area normalisation |
| □ | `derivatives` | First and second derivative spectra |

## How to Choose

1. `baseline_integrate` for subtractive baseline drift followed by
   quantification of specific spectral bands.
2. `smoothing` when the signal-to-noise ratio is low and spectral
   features are broader than the noise.
3. `normalisation` when path-length or sample thickness varies across
   observations.
4. `derivatives` to remove additive baseline (first derivative) or
   linear baseline (second derivative) while enhancing spectral
   resolution.
