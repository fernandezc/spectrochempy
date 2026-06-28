# NMF Exploration

**template_id:** `nmf_exploration`
**version:** `0.1.0`
**compatible_registry:** `0.1`

---

## Scientific Objective

Decompose a spectral dataset into additive, non-negative parts using
Non-negative Matrix Factorization (NMF). Unlike PCA, NMF enforces
non-negativity on both the components and their coefficients, which
can produce more interpretable parts-based representations when the
data are naturally non-negative (absorbance, counts).

---

## Methodology

| Step | Operation | Purpose |
|---|---|---|
| 1 | `load` | Load spectral data |
| 2 | `inspect` | Validate dataset |
| 3 | `nmf` | Fit NMF model |
| 4 | `plot` | Display components |

---

## Related Templates

- `exploratory_pca` — Linear decomposition without non-negativity
- `mcrals_analysis` — Constrained mixture resolution
- `clustering` — Grouping rather than decomposition

---

## Status

**Experimental.** Registered and renders but not yet aligned with current
template standards. Needs operation refinements, parameter validation,
and scientific documentation.
