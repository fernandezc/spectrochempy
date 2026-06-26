# Phase 0 Prototype Report

**Date:** 2026-06-26
**Branch:** `feature/workflow-prototype`
**Scope:** End-to-end validation of WorkflowPlan → Validator → NotebookRenderer

---

## Summary

This prototype demonstrates that a structured, hand-written WorkflowPlan can
be validated deterministically and rendered into a runnable Jupyter notebook
without any AI, provider, or prompt infrastructure.

The pipeline is:

```text
WorkflowPlan (JSON fixture)
    ↓
Validator (hard-coded allowlist)
    ↓
NotebookRenderer (nbformat, deterministic)
    ↓
Runnable notebook (.ipynb)
```

---

## Implemented Components

### 1. WorkflowPlan schema (`workflow_plan.py`)

Minimal dataclass-based schema with:
- `ScientificContext` — goal, analytical_strategy, data_assumptions,
  validation_criteria, expected_outputs, limitations
- `InputReference` — named inputs with type and source
- `OperationStep` — structured step (operation_id, parameters, input_refs,
  output_var, rationale)
- `OutputReference` — named outputs
- `ReproducibilityMetadata` — package versions, random seeds
- `WorkflowPlan` — top-level container with schema_version, planner_id, etc.

Serialization: `to_dict()` / `from_dict()` for JSON round-tripping.

### 2. Fixture (`fixtures/exploratory_pca.json`)

One hand-written fixture representing a complete exploratory PCA workflow:

```text
Generate synthetic dataset → Baseline correction → PCA → Score plot → Loading plot
```

The fixture includes full `scientific_context`, structured steps with
parameters, and `reproducibility` metadata. No AI generated this file.

### 3. Validator (`validator.py`)

Deterministic validation with a hard-coded allowlist:

- schema required fields (schema_version, spectrochempy_version, plugin_version)
- scientific_context required fields (goal, analytical_strategy)
- operation allowlist (`read`, `baseline`, `pca`, `score_plot`, `loading_plot`)
- input reference resolution (all refs must be available before use)
- output reference resolution (outputs must be produced by steps or available as inputs)
- duplicate step_id detection

Fails closed: raises `ValidationError` with a list of violations.

### 4. Notebook Renderer (`notebook_renderer.py`)

Deterministic renderer using `nbformat` v4. Generates:

1. Title and analysis goal (markdown)
2. Reproducibility manifest (markdown)
3. Scientific context and assumptions (markdown)
4. Imports cell (code)
5. One markdown + one code cell per operation step
6. Summary and caveats (markdown)
7. Notebook metadata manifest

Operation code generators are hard-coded functions mapping each allowlisted
operation_id to SpectroChemPy code. This is intentionally simple.

### 5. Tests (`tests/`)

Eight tests covering:
- fixture loading
- valid plan passes validation
- missing schema_version fails
- missing goal fails
- unknown operation fails
- unresolved input reference fails
- duplicate step_id fails
- empty plan fails

Renderer tests covering:
- render produces a notebook
- title cell present
- import cell present
- all steps represented
- manifest metadata present
- determinism (same plan → same notebook)
- write roundtrip

All 14 tests pass (8 validator + 6 renderer).

---

## Verification

### Determinism

Rendering the same fixture twice produces identical cell types, sources, and
metadata. Verified by `test_determinism`.

### Notebook runs with SpectroChemPy

The generated notebook code was manually executed in the `scpy-core`
environment. All operations execute successfully:

- Synthetic dataset creation with `scp.NDDataset`
- Baseline correction with `scp.processing.baselineprocessing.baselineprocessing.asls`
- PCA with `scp.analysis.decomposition.pca.PCA`
- Score and loading plots via `pca.scoreplot()` and `pca.loadings.plot()`

Note: `nbconvert --execute` failed because the execution kernel did not have
SpectroChemPy on its path. This is an environment issue, not a notebook issue.
The notebook runs correctly inside the project's conda environment.

---

## Deliberately Omitted Components

These belong to later phases and were intentionally not built:

| Component | Phase |
|---|---|
| Operation registry (dynamic discovery) | 1 |
| `WorkflowPlan` schema refinements (full parameter validation) | 2 |
| Fixture-based testing expansion | 2 |
| TemplatePlanner / RulePlanner | 3 |
| LLMPlanner | 4 |
| Provider adapters (Ollama, OpenAI, etc.) | 4–6 |
| Jupyter integration / plan editor UI | 5 |
| Chat UI | deferred |
| MCP / agent surface | deferred |
| WorkflowTemplate library | 2–3 |
| Prompt packs | 4+ |
| Redaction / privacy policy | 4+ |
| Safety levels per operation | 2+ |

---

## Architecture Decisions

### Dataclasses over Pydantic

Pydantic was considered but dataclasses were chosen for Phase 0 because:
- zero additional dependency
- simple `asdict()` / manual reconstruction for JSON
- easy to migrate to Pydantic later if schema validation needs grow

### Hard-coded allowlist over dynamic registry

An operation registry is architecturally correct but too heavy for Phase 0.
A `dict[str, callable]` mapping operation_id to code generators is sufficient
to prove the renderer concept.

### No plugin packaging yet

The prototype lives in `spectrochempy_ai/` at the repository root. It is not
yet a proper Python package with `pyproject.toml`, entry points, or namespace
registration. Packaging is deferred until the architecture stabilises.

---

## Lessons Learned

1. **The pipeline works.** A structured plan can be validated and rendered
deterministically. This validates the core architectural hypothesis.

2. **ScientificContext is valuable.** Even in a hand-written fixture, the
structured scientific metadata (goal, strategy, assumptions, limitations)
makes the plan readable as a protocol, not just an execution graph.

3. **Dataclasses are sufficient for Phase 0.** The schema is small enough
that manual `from_dict()` reconstruction is trivial. Pydantic can be adopted
later without breaking the conceptual model.

4. **The renderer is the riskiest part.** Mapping abstract operation steps
to real SpectroChemPy API calls requires knowledge of the actual library
surface. The current hard-coded generators are brittle. A real operation
registry with parameter specs and code templates is needed for robustness.

5. **Validation rules are easy to enumerate.** Schema validation, allowlist
checks, and reference resolution are straightforward. Parameter validation
(are the parameters valid for the operation?) is harder and requires the
operation registry.

6. **Fixture plans are excellent test assets.** The exploratory PCA fixture
already serves as a regression test for the validator and renderer. Future
planners can be tested against the same fixtures.

---

## Recommended Phase 1

Build the **Operation Registry**:

```text
OperationEntry
    operation_id: str
    display_name: str
    description: str
    parameters: list[ParameterSpec]
    code_template: str  # Jinja2 or similar
    input_type: str
    output_type: str
```

This unblocks:
- dynamic operation allowlist (no more hard-coded dict)
- parameter validation (type checking against ParameterSpec)
- code template-based rendering (no more hard-coded generator functions)
- future TemplatePlanner (templates reference operations from the registry)

Phase 1 should also:
- add more fixture plans (Peak Analysis, NMR Processing)
- add parameter validation rules to the validator
- expand renderer tests to compare rendered code against expected snippets

---

## Files Added

```text
spectrochempy_ai/
    __init__.py
    workflow_plan.py
    validator.py
    notebook_renderer.py
    fixtures/
        exploratory_pca.json
    tests/
        test_validator.py
        test_renderer.py
prototype-report.md
```

---

## Risks

- **Renderer brittleness:** Hard-coded code generators will break when
  SpectroChemPy APIs change. The operation registry with code templates is
  the mitigation.
- **Schema evolution:** `from_dict()` uses manual field mapping. Adding new
  fields requires updating both the dataclass and the deserializer. Pydantic
  would handle this more gracefully.
- **No notebook execution in CI:** The generated notebooks are not yet
  executed as part of CI. Adding notebook execution tests would catch
  renderer/API mismatches early.

---

## Next Steps

1. Review this prototype with maintainers.
2. If accepted, begin Phase 1 (Operation Registry).
3. Keep the prototype branch unmerged until Phase 1 stabilises, or merge it
   as an experimental `spectrochempy_ai` package behind a clear
   "work in progress" marker.
