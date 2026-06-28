# Exploratory PCA

**template_id:** `exploratory_pca`
**version:** `0.1.0`
**compatible_registry:** `0.1`

---

## Scientific Objective

Identify the dominant variance directions in a spectral dataset. Principal
Component Analysis (PCA) after baseline correction is the standard first step
in exploratory analysis of multivariate spectral data. It reduces the
dimensionality of the data while preserving the variance structure, enabling
visualisation of sample relationships and identification of the spectral
features that drive sample separation.

---

## Methodology

This template implements a complete exploratory PCA workflow consisting of
seven ordered steps:

| Step | Operation | Purpose |
|---|---|---|
| 1 | `load` | Load spectral data from a portable file |
| 2 | `inspect` | Validate dataset quality before processing |
| 3 | `baseline` | Remove additive baseline drift |
| 4 | `pca` | Decompose variance into principal components |
| 5 | `scree_plot` | Visualise explained variance per component |
| 6 | `score_plot` | Visualise sample distribution in PC space |
| 7 | `loading_plot` | Visualise variable contributions to each PC |

---

## Step Rationales

### Step 1: Load spectral dataset

Real spectral data must be loaded from an external source. The `load`
operation reads a portable file format (default: SCP) and produces a
spectral dataset. Starting from external data rather than synthetic
generation ensures the template represents a genuine scientific workflow.

**Parameter:**
- `filename` (default: `"data.scp"`): Path to input file.
- `format` (default: `"scp"`): File format specification.

### Step 2: Inspect dataset quality

Before any processing, inspect the dataset shape, coordinate ranges, and
value distribution. Early detection of anomalies — missing values, negative
absorbance, irregular sampling — prevents misleading downstream results and
saves time.

### Step 3: Baseline correction

Baseline drift is an additive artefact that inflates the variance captured
by early principal components, potentially masking genuine chemical
variation. Removal is essential before variance analysis.

The Asymmetric Least Squares (ASLS) method (Eilers & Boelens, 2005) is used
by default. It models the baseline as a smoothly varying signal, weighted
asymmetrically to fit below the spectrum. ASLS is widely adopted for
vibrational spectra because it adapts to varying baselines without requiring
user-specified anchor points.

**Parameter:**
- `method` (default: `"asls"`): Baseline correction algorithm.

**Reference:** Eilers & Boelens, "Baseline correction with asymmetric least
squares smoothing," Leiden University Medical Centre Report, 2005.

### Step 4: Principal Component Analysis

PCA transforms the baseline-corrected spectral variables into a reduced set
of orthogonal latent variables (principal components, PCs) ranked by
explained variance. The first few PCs typically capture chemical information
(variance shared across many variables), while later PCs increasingly
capture noise.

PCA is a linear method: it finds directions of maximum variance in the
original variable space. Each PC is a linear combination of the original
spectral variables, defined by a loading vector.

**Parameter:**
- `n_components` (default: `5`): Number of PCs to compute. Choose based on
  the scree plot (see Step 5); 5 is a sensible default for exploration.

**Reference:** Jolliffe, *Principal Component Analysis*, 2nd ed., Springer,
2002.

### Step 5: Scree plot (explained variance)

The scree plot shows the variance explained by each PC as a bar plot, with a
cumulative variance curve overlaid. It is the standard diagnostic for
determining how many PCs to retain:

- Look for the "elbow" where additional components contribute little
  additional variance.
- Consider how many components are needed to explain 80–95% of the total
  variance.
- Components beyond the elbow typically represent noise.

### Step 6: Score plot

The score plot projects each observation onto the first two principal
components. It reveals:

- **Clusters:** Groups of samples with similar spectral characteristics.
- **Gradients:** Continuous variation along a PC axis (e.g., concentration
  gradient).
- **Outliers:** Samples far from the main distribution; these may be
  measurement artefacts or chemically distinct.
- **High-leverage points:** Samples far from the origin in PC space that
  may dominate the model.

### Step 7: Loading plot

The loading plot displays the contribution (weight) of each spectral
variable to the first two principal components. Variables with high absolute
loading values correspond to spectral regions that drive the observed sample
separation. Loading plots can be interpreted as the "spectral signature"
captured by each PC: a positive loading indicates variables that are high
when the PC score is high, and vice versa.

---

## Assumptions

The template makes the following assumptions about the input data:

1. Data are two-dimensional (observations × spectral variables).
2. The spectral axis is continuous and evenly spaced.
3. Baseline varies slowly relative to spectral features.
4. The dominant structure in the data can be captured by linear
   combinations of the original variables.
5. The number of observations exceeds the number of informative components.

---

## Limitations

1. **Linearity.** PCA captures only linear relationships. Non-linear
   chemical or physical effects (peak shifts, saturation, resonance) will
   distort the PC space. Consider non-linear methods (MDS, t-SNE, UMAP)
   when non-linearity is suspected.

2. **Component selection.** The default `n_components=5` is arbitrary. The
   scree plot must be inspected to choose a defensible number of PCs. No
   automatic selection is performed.

3. **Outlier sensitivity.** A few extreme observations can dominate the
   first PC. Score plots should always be inspected for outliers before
   interpreting loadings.

4. **Spectral alignment.** PCA assumes that spectral features are
   approximately aligned across all observations. Peak shifts require
   alignment or warping before PCA.

5. **Interpretation.** PCA components are mathematical axes that maximise
   explained variance. They are not necessarily chemically pure components.
   For mixture resolution, consider NMF or MCR-ALS.

6. **No uncertainty quantification.** PCA does not provide confidence
   intervals for scores or loadings. Bootstrap or cross-validation
   approaches are needed for statistical inference.

---

## Expected Outputs

| Output | Type | Description |
|---|---|---|
| `dataset_corrected` | dataset | Baseline-corrected spectral data |
| `pca_result` | result | PCA model with scores, loadings, explained variance |

Additionally, the following plots are produced as side effects:

| Plot | Description |
|---|---|
| Scree plot | Variance per PC (bars) + cumulative variance (line) |
| Score plot | Observations projected onto PC1–PC2 |
| Loading plot | Variable weights for PC1–PC2 |

---

## Interpretation Guidance

1. Start with the **scree plot** to decide how many PCs to retain.
2. Examine the **score plot** for clustering, gradients, and outliers.
3. Examine the **loading plot** to identify which spectral regions drive
   each PC.
4. If strong outliers are visible, consider removing them and re-running.

---

## When Not to Use This Template

- The data are one-dimensional (single spectrum) — PCA needs multiple
  observations.
- The data contain strong non-linearities (e.g., peak shifts, saturation).
- The spectral baseline is already corrected and verification is not needed.
- The goal is mixture resolution (use NMF or MCR-ALS instead).

---

## Reproducibility Guarantees

Identical template version, registry version, parameters, and input data
produce identical `WorkflowPlan` objects. The framework guarantees
deterministic plan generation.

The numerical results of the underlying operations (PCA fit, baseline
correction) depend on the specific library implementations and may vary
across versions or platforms. Reproducibility of numerical results should
be verified independently.

---

## References

- Eilers & Boelens, "Baseline correction with asymmetric least squares
  smoothing," Leiden University Medical Centre Report, 2005.
- Jolliffe, *Principal Component Analysis*, 2nd ed., Springer, 2002.
- Wold, Esbensen & Geladi, "Principal component analysis," *Chemom. Intell.
  Lab. Syst.* 2(1–3), 37–52, 1987.
