# Exploration

## Scientific Objective

Discover the underlying structure of a spectral dataset when no prior
information about sample groupings or response variables is available.
Exploration is the first step in most spectroscopic workflows: it reveals
whether the data contain meaningful chemical variation, which variables
drive that variation, and whether unexpected patterns (outliers, gradients,
batch effects) require attention before proceeding to calibration or
resolution.

## Typical Questions

- How many latent sources of variation exist in this dataset?
- Are there natural clusters of samples (e.g., by material, treatment,
  batch)?
- Which spectral regions carry the most information?
- Are there outliers or measurement artefacts?
- Is the variance structure compatible with a linear model?

## Available Methodologies

| Status | Template | Description |
|---|---|---|
| ✓ | `exploratory_pca` | PCA with score, loading, and scree plots |
| □ | `nmf_exploration` | Non-negative Matrix Factorization |
| □ | `clustering` | k-means, hierarchical clustering on score space |
| □ | `outlier_detection` | Hotelling T², Q residuals, leverage |

## How to Choose

1. Start with `exploratory_pca` for any new dataset — it reveals the
   global variance structure with minimal assumptions.
2. If PCA components are not chemically interpretable (e.g., mixed signs,
   no clear separation), try `nmf_exploration` for parts-based
   decomposition.
3. If the score plot shows well-separated groups, `clustering` can
   formalise the grouping.
4. If the score plot shows extreme points, `outlier_detection`
   provides statistical cutoffs.
