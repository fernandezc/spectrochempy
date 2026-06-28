# Mixture Resolution

## Scientific Objective

Separate a spectroscopic mixture into the pure contributions of each chemical
component — both their concentration profiles and their individual spectra.
Unlike exploratory methods (PCA), resolution methods enforce physical
constraints (non-negativity, unimodality) that produce chemically meaningful
profiles rather than orthogonal mathematical axes.

## Typical Questions

- How many chemical components are present in this mixture?
- What are the pure spectra of each component?
- How do the concentrations evolve over time, temperature, or elution?
- Are there intermediate or transient species?

## Available Methodologies

| Status | Template | Description |
|---|---|---|
| ✓ | `mcrals_analysis` | MCR-ALS with initial guess and constraints |
| □ | `efa` | Evolving Factor Analysis (rank estimation) |
| □ | `simplisma` | SIMPLISMA (pure variable selection) |

## How to Choose

1. Use `efa` first to estimate the number of chemical components from
   the evolving rank of the data matrix.
2. Use `simplisma` to identify pure variables (spectral channels that
   carry signal from only one component) as an alternative initial guess.
3. Use `mcrals_analysis` to perform the actual resolution with
   non-negativity and unimodality constraints.
