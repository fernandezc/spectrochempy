# Phase 0.5 Report — WorkflowPlan Language Stress Test

**Date:** 2026-06-26
**Branch:** `feature/workflow-prototype`
**Scope:** Stress-test the WorkflowPlan schema with multiple realistic fixtures.

---

## Summary

Phase 0 proved that one WorkflowPlan could be validated and rendered.
Phase 0.5 adds five new hand-written fixtures covering different SpectroChemPy
workflow families. All six fixtures (including the original exploratory PCA)
load, validate, and render deterministically. The schema survived the test,
but several gaps and emerging patterns were identified.

---

## 1. Fixtures Added

| Fixture | Workflow Family | New Concepts Tested |
|---|---|---|
| `exploratory_pca.json` | PCA exploratory analysis | baseline → PCA → plots *(existing)* |
| `baseline_integrate_plot.json` | Signal integration | integration, generic `plot`, 1D output |
| `smoothing_pca.json` | Preprocessing + PCA | smoothing (`savgol`), parameter variation |
| `nmf_workflow.json` | Non-negative decomposition | NMF, component/reconstruction plots |
| `mcrals_workflow.json` | Curve resolution | MCR-ALS, **multiple input refs per step**, resolved profiles |
| `simple_export.json` | Inspection and export | `inspect`, `export`, side-effect-only steps |

All fixtures are hand-written JSON. No AI generated them.

---

## 2. Validator Changes

Extended the hard-coded `ALLOWLIST` from 5 to 15 operation IDs:

```text
read, baseline, smooth, pca, score_plot, loading_plot,
integrate, plot, nmf, nmf_components_plot, nmf_reconstruction_plot,
mcrals, mcrals_conc_plot, mcrals_spec_plot, inspect, export
```

No structural changes to validation rules. The existing rules (schema fields,
scientific context, allowlist, input/output reference resolution, duplicate
step IDs) were sufficient for all new fixtures.

One observation: **MCR-ALS requires two input refs** (`dataset` and
`conc_guess`). The current validator resolves all `input_refs` against the
available variable set. This worked without modification because both inputs
are declared as `InputReference` objects (one in `inputs`, one produced by an
earlier `read` step).

---

## 3. Renderer Changes

Added 10 hard-coded code generators to `notebook_renderer.py`:

- `_generate_smooth` — Savitzky-Golay (`scp.processing.filter.filter.savgol`)
- `_generate_integrate` — trapezoidal integration
- `_generate_plot` — generic line plot
- `_generate_nmf` — NMF fit
- `_generate_nmf_components_plot` — NMF components visualisation
- `_generate_nmf_reconstruction_plot` — NMF reconstruction visualisation
- `_generate_mcrals` — MCR-ALS fit (consumes two input refs)
- `_generate_mcrals_conc_plot` — concentration profile plot
- `_generate_mcrals_spec_plot` — resolved spectra plot
- `_generate_inspect` — inline dataset summary
- `_generate_export` — `scp.write()` call

No structural changes to the rendering pipeline. The same markdown/code cell
pattern works for all workflow families.

---

## 4. WorkflowPlan Strengths

After exercising the schema on six different workflows, these strengths are
confirmed:

### ScientificContext is expressive
Every fixture uses `goal`, `analytical_strategy`, `data_assumptions`,
`validation_criteria`, `expected_outputs`, and `limitations`. The resulting
notebooks are readable as scientific protocols, not just execution scripts.

### Input/output reference model is sufficient
Chaining steps through `input_refs` / `output_var` works for linear pipelines
(PCA), branching outputs (integration → plot), and multi-input steps
(MCR-ALS). The validator's simple reachability check caught no false positives.

### Step-level rationale improves readability
Every operation step carries a `rationale` field. In rendered notebooks, this
becomes a markdown paragraph explaining *why* the step exists. This is
valuable for review and education.

### ReproducibilityMetadata is future-proof
`package_versions` and `random_seeds` appear in every fixture. The renderer
copies them into the notebook manifest. This is a strong foundation for
reproducibility auditing.

---

## 5. WorkflowPlan Gaps

### Gap 1: No explicit plot vs. dataset distinction in outputs
`OutputReference.type` allows `"dataset" | "notebook" | "file" | "plot"`, but
plotting operations in the fixtures have empty `output_var` because they are
side effects. The *plot* is not captured as an output variable, yet it is
listed in `expected_outputs`. This is a mild inconsistency.

**Implication:** The renderer treats plots as side-effect cells. If we ever
want to capture plot objects (e.g. for batch export), the plan cannot express
it cleanly.

### Gap 2: No diagnostic / scalar output type
NMF and PCA produce scalar diagnostics (explained variance, convergence info).
These are printed in code cells but are not represented in the plan schema.
`OutputReference` cannot represent a "scalar table" or "diagnostic text".

**Implication:** Reproducibility metadata is incomplete. A user cannot see
from the plan alone what diagnostics were expected.

### Gap 3: Operation parameters are untyped
`OperationStep.parameters` is `dict[str, Any]`. The renderer assumes specific
keys (`method`, `window_length`, `polyorder`, `n_components`, etc.) but the
validator does not verify them. A misspelled parameter is silently ignored by
the renderer.

**Implication:** Parameter validation requires an operation registry with
`ParameterSpec`.

### Gap 4: No conditional or optional steps
`mcrals_workflow.json` includes an optional preprocessing concept in its
scientific context, but the plan cannot express optional or conditional steps.
Every step is executed unconditionally.

**Implication:** Future workflows (e.g. "baseline-correct only if drift is
detected") will need branching or conditional constructs.

### Gap 5: Multi-input steps are implicit
MCR-ALS consumes two inputs, but this is only visible by reading `input_refs`.
There is no `signature` declaring *which* inputs are required, which are
optional, and what types they must have.

**Implication:** Validation cannot check that the *right* variables are passed
to MCR-ALS (e.g. `conc_guess` must be 2D).

### Gap 6: No coordinate / axis metadata
Integration operates along a specific axis. The fixture assumes the default
axis. There is no way to declare "integrate along the spectral axis" vs
"integrate along the sample axis" in the plan without overloading parameters.

**Implication:** Scientific intent is partially lost. A plan editor would need
to show axis choices, but the schema does not support them natively.

---

## 6. Scientific Types Observed

From the fixtures, these scientific types appear naturally:

| Type | Example | Current Representation |
|---|---|---|
| `dataset` | `dataset`, `dataset_corrected` | `OutputReference.type = "dataset"` |
| `result` | `pca_result`, `nmf_result` | `OutputReference.type = "dataset"` *(imprecise)* |
| `fitted_model` | `pca_result`, `nmf_result` | same as above |
| `plot` | score plot, component plot | side effect, not in outputs |
| `diagnostic_table` | explained variance | printed, not structured |
| `scalar` | area profile after integration | `dataset` (1D NDDataset) |
| `file` | exported `.scp` | `OutputReference.type = "file"` |

**Observation:** `result` and `fitted_model` are currently represented as
`dataset` because SpectroChemPy estimators expose their state through
attributes (`.scores`, `.loadings`, `.components`). The plan does not capture
the *semantic* type of the output.

---

## 7. Emerging OperationSignature Concept

Every operation in the fixtures naturally exposes a signature:

```text
operation_id: str
inputs: list[str]           # names of required input variables
optional_inputs: list[str]  # names of optional input variables
parameters: dict[str, ParamSpec]
outputs: list[str]          # names of produced variables
side_effects: list[str]     # e.g. "plot", "file_write"
```

Examples from the fixtures:

| Operation | Inputs | Parameters | Outputs | Side Effects |
|---|---|---|---|---|
| `read` | — | `shape`, `random_seed`, `non_negative` | `dataset` | — |
| `baseline` | `dataset` | `method` | `dataset_corrected` | — |
| `smooth` | `dataset` | `method`, `window_length`, `polyorder` | `dataset_smoothed` | — |
| `pca` | `dataset` | `n_components` | `pca_result` | — |
| `score_plot` | `pca_result` | — | — | `plot` |
| `integrate` | `dataset` | `method` | `area_profile` | — |
| `nmf` | `dataset` | `n_components`, `max_iter` | `nmf_result` | — |
| `mcrals` | `dataset`, `conc_guess` | `max_iter` | `mcrals_result` | — |
| `inspect` | `dataset` | — | — | `print` |
| `export` | `dataset` | `filename`, `format` | — | `file_write` |

**Key insight:** The renderer already implements a crude version of this
signature mapping (the `_OPERATION_GENERATORS` dict). The validator does not
yet use signatures, but it could. A formal `OperationSignature` would replace
the hard-coded allowlist and enable parameter validation, type checking, and
auto-completion in a plan editor.

---

## 8. Recommended Changes Before Phase 1

Based on the gaps observed, the following changes are recommended **before**
building a full OperationRegistry:

1. **Add `side_effects` to `OperationStep`**
   Explicitly list side effects (`plot`, `file_write`, `print`) so the
   renderer and validator know what to expect.

2. **Distinguish `dataset` from `result` in `OutputReference.type`**
   Add `"result"` and `"diagnostic"` to the type enum. This makes the plan
   self-describing.

3. **Add `diagnostics` list to `WorkflowPlan`**
   A list of expected diagnostic outputs (scalars, tables, convergence info)
   that are not primary datasets.

4. **Add `axis` or `dim` parameter convention**
   Standardise how operations declare which axis they operate on, so the
   renderer can generate correct code without guessing.

5. **Add `optional` flag to `OperationStep`**
   Even if conditionals are not implemented, marking a step as optional is
   useful for template authors.

These are small schema changes. They do not require a registry, but they make
the registry easier to design.

---

## 9. Is OperationRegistry Now Justified?

**Yes.**

The hard-coded allowlist and code generators have grown from 5 to 15
operations. The mapping from `operation_id` to:
- validation rule (is it allowed?)
- code generator (how is it rendered?)
- expected parameters (what keys are valid?)
- expected inputs/outputs (what signature?)

is now repeated in three places:
1. `validator.py` (`ALLOWLIST`)
2. `notebook_renderer.py` (`_OPERATION_GENERATORS`)
3. Each fixture JSON (implicitly, through parameter keys)

This duplication is manageable at 15 operations but will become a maintenance
burden quickly. An `OperationRegistry` that unifies allowlist, signature, and
code template is the logical next abstraction.

**However**, the registry should be **minimal** in Phase 1:
- a Python module with `OperationSignature` dataclasses;
- a hard-coded dictionary of signatures (not dynamic discovery);
- code templates as simple Python string templates or functions (not Jinja yet);
- used by both validator and renderer.

This is justified because:
- the signatures are now clear from the fixtures;
- the duplication is already visible;
- parameter validation is the next blocker for robustness.

---

## Test Results

All 55 tests pass:
- 36 fixture coverage tests (6 fixtures × 6 checks)
- 3 invalid-plan regression tests
- 8 original renderer tests
- 8 original validator tests

```bash
conda run -n scpy-core python -m pytest spectrochempy_ai/tests/ -v
```

---

## Files Changed

```text
spectrochempy_ai/
    validator.py                 # extended ALLOWLIST
    notebook_renderer.py         # 10 new code generators
    fixtures/
        baseline_integrate_plot.json   # new
        smoothing_pca.json             # new
        nmf_workflow.json              # new
        mcrals_workflow.json           # new
        simple_export.json             # new
    tests/
        test_fixtures.py               # new parametrised stress tests
    phase-0.5-report.md            # new
```

---

## Next Steps

1. Review this report and the recommended schema changes.
2. If accepted, begin Phase 1: minimal `OperationRegistry` with
   `OperationSignature` dataclasses and code templates.
3. Apply the recommended schema changes (side_effects, result type,
   diagnostics) before or alongside the registry.
