# Scientific Methodology Roadmap

**Drives template development, rule evolution, and long-term project
direction for the SpectroChemPy Scientific Workflow Assistant.**

This is a *scientific* roadmap, not a software plan.  Each item represents
a capability a spectroscopist or chemometrician should be able to express
through the assistant.

> **Implementation phases derive from this document, not the reverse.**

The assistant is developed as a **scientific methodology library**, not as
an AI application.  AI components — when they arrive — will serve the
methodology, never replace it.

---

## 1. Principles

1. **Methodology first.**  Every template implements a documented scientific
   workflow.  The framework serves the methodology, not the other way around.
2. **Reproducibility by construction.**  Templates, rules, and evidence are
   deterministic and testable.  Any user can trace a recommendation back to
   the data characteristics that produced it.
3. **Expertise capture, not replacement.**  The assistant encodes the
   decision process of an experienced spectroscopist.  It does not replace
   that expertise — it makes it accessible, auditable, and extensible.
4. **LLM readiness without LLM dependency.**  The deterministic baseline
   (RulePlanner) must be complete enough that an LLM planner can be
   validated against it.  The roadmap prioritises the baseline.
5. **Maturity before breadth.**  A template is not "done" when it generates
   a notebook.  It is done when its scientific assumptions, limitations,
   and evidence rules are documented, tested, and stable across real
   datasets.
6. **A template is not the methodology — it is an implementation of one.**
   A methodology (e.g. "exploratory PCA") can have multiple renderers:
   notebook, Python script, GUI dashboard, PDF report.  The template is
   one representation, optimised for execution and reproducibility.

---

## 2. Methodology Categories

```
Exploration           →  dimensionality reduction, pattern discovery
Calibration           →  quantitative models from spectral data
Mixture Resolution    →  pure component separation
Preprocessing         →  baseline, normalisation, derivatives
Validation            →  outlier detection, cross-validation, residuals
Integration           →  peak fitting, area quantification
Kinetics              →  time-resolved analysis
Hyperspectral         →  imaging, mapping, spatial exploration
```

---

## 3. Template Maturity Matrix

| Template | Category | Maturity | Rules | Evidence | Tests | Real-data | Notebook quality |
|---|---|---|---|---|---|---|---|
| `exploratory_pca` | Exploration | **Stable** | ✓ | ✓ | 39+ | Validated | Idiomatic |
| `mcrals_analysis` | Mixture Resolution | **Stable** | ✓ | ✓ | 10+ | Validated | Idiomatic |
| `pls_calibration` | Calibration | **Stable** | ✓ | ✓ | 7+ | Validated | Idiomatic |
| `baseline_integrate` | Preprocessing / Integration | **Stable** | ✓ | ✓ | 10+ | Validated | Idiomatic |
| `nmf_exploration` | Exploration | **Experimental** | — | — | — | — | Prototype |
| `smoothing_pca` | Preprocessing / Exploration | **Experimental** | — | — | — | — | Prototype |

**Legend:**
- **Stable** — template registered, tested, executed, documented, rules defined,
  evidence generated, real-dataset validated, notebook at *Idiomatic* quality
- **Draft** — plan exists, no implementation
- **Experimental** — legacy or partial implementation, not yet aligned with current standards

**Real-data validation levels:**
- *Multi-domain validated* — tested on IR, Raman, NMR, UV-Vis
- *Validated* — tested on at least one real dataset (not synthetic)
- *Planned* — validation dataset identified but not yet run

**Notebook quality levels:**
- *Gallery quality* — indistinguishable from a hand-written tutorial by an experienced SPeC user
- *Idiomatic* — follows SPeC conventions, no internal module paths, no `scp.show()`, native display
- *Prototype* — functional but may contain raw matplotlib, internal imports, or debug prints

---

## 4. Workflow Collections

Collections group related templates, documentation, examples, and gallery
notebooks into methodology-level navigation hubs.  They do not require code
changes — they are a documentation and UX layer that can be implemented
progressively.

Each collection may contain:
- **Templates** — registered workflows the assistant can generate
- **Documentation** — methodology guides (what, why, when)
- **Examples** — real-dataset walkthroughs
- **Gallery** — rendered notebooks for quick preview

### Exploration

| Component | Status |
|---|---|
| **Templates** | |
| ✓ `exploratory_pca` — PCA with score, loading, scree plots | Stable |
| □ `nmf_exploration` — Non-negative Matrix Factorization | Experimental |
| □ `clustering` — k-means, hierarchical on score space | Draft |
| □ `outlier_detection` — Hotelling T², Q residuals | Draft |
| **Documentation** | |
| □ PCA methodology guide | Planned |
| □ When to explore vs. calibrate | Planned |
| **Examples** | |
| □ IR exploration walkthrough | Planned |
| □ Raman PCA example | Planned |
| **Gallery** | |
| □ Rendered PCA notebook preview | Planned |

### Calibration (Quantification)

| Component | Status |
|---|---|
| **Templates** | |
| ✓ `pls_calibration` — PLS regression | Stable |
| □ `pcr` — Principal Component Regression | Draft |
| □ `svm_regression` — Support Vector Regression | Draft |
| **Documentation** | |
| □ PLS methodology guide | Planned |
| □ Reference values best practices | Planned |

### Mixture Resolution

| Component | Status |
|---|---|
| **Templates** | |
| ✓ `mcrals_analysis` — MCR-ALS | Stable |
| □ `efa` — Evolving Factor Analysis | Draft |
| □ `simplisma` — SIMPLISMA (pure variable selection) | Draft |
| **Documentation** | |
| □ MCR-ALS methodology guide | Planned |
| □ Rotational ambiguity explained | Planned |

### Preprocessing

| Component | Status |
|---|---|
| **Templates** | |
| ✓ `baseline_integrate` — baseline + peak integration | Stable |
| □ `smoothing` — Savitzky–Golay, moving average | Draft |
| □ `normalisation` — SNV, MSC, area normalisation | Draft |
| □ `derivatives` — first and second derivative | Draft |
| **Documentation** | |
| □ Preprocessing decision tree | Planned |
| □ When to correct baseline vs. normalise | Planned |

### Validation

| Component | Status |
|---|---|
| **Templates** | |
| □ `cross_validation` — venetian blinds, random subsets, LOO | Draft |
| □ `residual_analysis` — Q residuals, Hotelling T² | Draft |
| □ `model_comparison` — compare PLS, PCR, SVM | Draft |
| **Documentation** | |
| □ Validation strategies for calibration | Planned |

---

## 5. Rule Evolution Path

The RulePlanner's evidence rules grow with each new template.

| Template added | New rules / evidence |
|---|---|
| `baseline_integrate` | 1D or single-spectrum + continuous x → baseline_integrate (0.85) |
| `clustering` | n_obs > 50 + intent="explore" → also suggest clustering (0.5) |
| `cross_validation` | reference_path + n_components > 1 → also suggest CV (0.6) |

A template cannot reach **Stable** maturity without:
- at least one RulePlanner rule that can recommend it
- at least one evidence generator that explains the recommendation
- a fallback confidence lower than the primary recommendation for the same profile
- at least one real spectroscopic dataset that produces that recommendation
  (the template must have been recommended in practice, not only in tests)

---

## 7. LLMPlanner Design Principles (preparation)

These principles are documented now, before any LLM code is written, to
ensure the LLM planner is constrained by the same scientific rigour as the
RulePlanner.

### 7.1 Responsibilities

1. **Interpret free-text intent** — translate "I want to quantify
   glucose in NIR spectra" into structured parameters (calibration,
   PLS, preprocessing).
2. **Combine rules with experience** — use the RulePlanner as the
   deterministic core, but adjust confidence based on learned patterns
   (e.g. "dataset looks like a time series → prefer MCR-ALS").
3. **Generate explanations** — produce natural-language explanations
   from the same evidence list the RulePlanner uses, so the user cannot
   tell whether the recommendation came from rules or an LLM.
4. **Flag uncertainty** — when the LLM is unsure, it must defer to the
   RulePlanner rather than inventing a methodology.

### 7.2 Constraints

1. **Never modify the WorkflowPlan schema.**
2. **Never generate unregistered operations.**  Every operation in a
   plan must exist in the operation registry.
3. **Never override explicit user intent.**  If the user says "PLS with
   5 components", the LLM must not change it.
4. **Never lower reproducibility.**  Every plan must pass the same
   validation and render deterministically.
5. **Evidence is the contract.**  An LLM recommendation without evidence
   is not accepted.  The evidence list must be verifiable against the
   DatasetProfile.  The LLM never replaces evidence — it may enrich its
   interpretation, but every claim must trace back to an observable
   characteristic of the data.

### 7.3 Interaction with existing components

- **DatasetProfile** — read-only input.  The LLM may request additional
  profiling (e.g. "check if the x axis is linear"), but never modifies
  the profile.
- **RulePlanner** — the baseline.  The LLM must explain any deviation
  from the RulePlanner's top recommendation.
- **TemplatePlanner** — the LLM calls `create_plan()`, same as
  `explore()`.  No special API.
- **WorkflowPlan** — the LLM consumes plans, never produces them
  directly.  Production is always through `create_plan()`.

### 7.4 When NOT to use an LLM

1. When the RulePlanner confidence is ≥ 0.7 for the top recommendation
   and the user provides no contradictory intent.
2. When the dataset profile is incomplete (readable=False).
3. When the user requests a specific template by name.

---

## 8. How to use this roadmap

1. **New template?**  Assign it to a category, set an initial maturity
   (Draft), and define when it will reach Stable.
2. **New rule?**  Add it to the Rule Evolution Path table before
   implementing the template.
3. **New collection?**  Add it to Workflow Collections with at least
   one concrete template planned.
4. **LLM work?**  Read section 7 first.  Do not implement LLM features
   before the RulePlanner is complete enough to serve as baseline.

---

## 9. Architecture overview

The hierarchy from methodology to renderer, as defined by this roadmap:

```
Scientific Methodology
         │
         ▼
Workflow Collection     (navigation: templates + docs + examples + gallery)
         │
         ▼
Workflow Template       (implementation of a methodology for a specific
         │               planner and renderer)
         ▼
WorkflowPlan            (executable step sequence)
         │
         ▼
Operation Library       (atomic building blocks: load, inspect, baseline,
         │               pca, pls, mcrals, …)
         ▼
Renderer                (notebook, Python script, GUI, PDF report, …)
```

Each layer is independently versionable, testable, and extensible.  A new
renderer does not require new operations.  A new methodology does not
require new planners.

---

## 10. Current priorities

1. **Real-dataset validation campaign.**  Test each stable template
   against real spectroscopic data from different domains (IR, Raman,
   NMR, UV-Vis).  For each dataset:
   - run `suggest()` → verify recommendation is correct
   - run `explore()` → verify notebook executes
   - review notebook output for scientific plausibility

   This campaign will reveal remaining gaps in the RulePlanner,
   templates, and renderer before adding new templates.

2. **`smoothing`** — next preprocessing extension after the now-stable
   `baseline_integrate` single-spectrum workflow.

3. **Scientific documentation** — for each template, publish the
   methodology document in `templates/` so a new user can understand
   the technique without reading the code.

---

---

## Appendix A — Operation Library (evolving)

Operations are the atomic building blocks of a workflow.  Each operation
has a spec in the registry, a renderer in `notebook_renderer.py`, and
(ideally) a RulePlanner evidence generator.  This table tracks their
maturity separately from templates.

### Current operations

| Operation | Category | Status | Renderer | Evidence |
|---|---|---|---|---|
| `load` | I/O | **Stable** | ✓ | ✓ (via profile) |
| `inspect` | Diagnostic | **Stable** | ✓ | — |
| `baseline` | Preprocessing | **Stable** | ✓ | — |
| `pca` | Decomposition | **Stable** | ✓ | — |
| `score_plot` | Visualisation | **Stable** | ✓ | — |
| `loading_plot` | Visualisation | **Stable** | ✓ | — |
| `scree_plot` | Visualisation | **Stable** | ✓ | — |
| `mcrals_init` | Initialisation | **Stable** | ✓ | — |
| `mcrals` | Decomposition | **Stable** | ✓ | — |
| `mcrals_conc_plot` | Visualisation | **Stable** | ✓ | — |
| `mcrals_spec_plot` | Visualisation | **Stable** | ✓ | — |
| `pls` | Regression | **Stable** | ✓ | — |
| `pls_predict_plot` | Visualisation | **Stable** | ✓ | — |
| `nmf` | Decomposition | **Experimental** | ✓ | — |
| `plot` | Visualisation | **Experimental** | ✓ | — |
| `export` | I/O | **Experimental** | ✓ | — |

### Planned operations

| Operation | Category | When |
|---|---|---|
| `snv` | Preprocessing | With normalisation collection |
| `smooth` | Preprocessing | With smoothing template |
| `derivative` | Preprocessing | With derivatives template |
| `integrate` | Quantification | With baseline_integrate |
| `peak_fit` | Quantification | Future |
| `crossval` | Validation | With cross_validation collection |
| `cluster` | Exploration | With clustering collection |

### Lifecycle

- **Experimental** — registered, renders, but not yet idiomatic or tested
- **Stable** — idiomatic renderer, parameter validation, tested, used by
  at least one Stable template
- **Deprecated** — replaced by a better operation, kept for backward
  compatibility

---

---

## Related architecture documents

| Document | Scope |
|---|---|
| `maintainers/architecture/workflow-assistant-architecture.md` | Components, interfaces, module responsibilities |
| `maintainers/architecture/workflow-plan.md` | WorkflowPlan contract, execution model, serialization |
| `maintainers/architecture/operation-specification.md` | Operation registry, parameter spec, validation |
| *(this document)* | Scientific methodology, template catalog, rule evolution, roadmap |

Together these four documents form the complete architecture corpus of the
SpectroChemPy Scientific Workflow Assistant.  They cover four distinct
layers — system, execution, operations, and methodology — without
overlapping.

*Last updated: 2026-06-28*
