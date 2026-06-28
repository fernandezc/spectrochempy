# Evolving Factor Analysis (EFA)

**template_id:** `efa`
**version:** `0.1.0`
**compatible_registry:** `0.1`

---

## Scientific Objective

Estimate the number of chemical components in a mixture by tracking the
rank of the data matrix along the observation axis. EFA computes the
singular values of forward and backward expanding windows: the emergence
of a new component is indicated by a rise in the corresponding singular
value. EFA is a standard pre-processing step before MCR-ALS.

---

## Related Templates

- `mcrals_analysis` — Downstream resolution using EFA-based initial guess
- `simplisma` — Alternative pure-variable approach for rank estimation

---

## Status

**Draft.** Not yet implemented.
