# Phase 1 Report — Operation Registry Implementation

**Date:** 2026-06-26
**Branch:** `feature/workflow-prototype`
**Scope:** Replace duplicated operation metadata with OperationSpecification +
OperationRegistry.

---

## Summary

Phase 1 replaced the prototype's duplicated operation metadata (hard-coded
allowlist in validator, hard-coded code generators in renderer, implicit
parameters in fixtures) with a single authoritative source:
`OperationSpecification` + `OperationRegistry`.

The prototype remains fully functional. All 68 tests pass. No fixtures were
modified. No WorkflowPlan schema changes.

---

## Implementation Steps

### 1. OperationSpecification dataclasses

Created `operation_specification.py` with pure metadata dataclasses:

- `OperationSpecification` — operation_id, display_name, description, inputs,
  outputs, parameters, constraints, side_effects, category
- `InputSpec` — name, type, required, description
- `OutputSpec` — name, type, description
- `ParameterSpec` — name, type, default, description, constraints
- `Constraint` — predicate, description (declarative, not executable)

No rendering logic. No validation rules. No runtime state.

### 2. Minimal OperationRegistry

Created `operation_registry.py` with a hard-coded dictionary of 16
OperationSpecifications covering all prototype operations:

- read, baseline, smooth, pca, score_plot, loading_plot, integrate, plot,
  nmf, nmf_components_plot, nmf_reconstruction_plot, mcrals,
  mcrals_conc_plot, mcrals_spec_plot, inspect, export

Registry API:
- `get_spec(operation_id) -> OperationSpecification`
- `is_registered(operation_id) -> bool`
- `list_specs(category=None) -> list[OperationSpecification]`
- `list_operation_ids() -> list[str]`

No dynamic discovery. No plugin loading. No metaclasses.

### 3. Refactored Validator

Removed the hard-coded `ALLOWLIST` set. The validator now:

1. Checks `is_registered(operation_id)` instead of set membership.
2. Validates required input counts against `InputSpec.required`.
3. Validates parameter names against `ParameterSpec` names.
4. Validates output variable presence against `OutputSpec` and `side_effects`.

### 4. Refactored NotebookRenderer

The renderer keeps its existing code generators (it owns rendering logic).
It now consumes `OperationSpecification` from the registry to:

1. Verify that each step's operation is registered.
2. Enrich markdown cells with the spec's `description` alongside the step's
   `display_label` and `rationale`.

### 5. Updated Tests

- Modified `test_validator.py` to expect "registry" instead of "allowlist".
- Added `test_registry.py` with 13 tests covering lookup, discovery, and
  specification contents.

### 6. Removed Obsolete Structures

- Deleted `ALLOWLIST` from `validator.py`.
- Updated comments in `notebook_renderer.py` to reflect registry usage.

---

## Files Changed

```text
spectrochempy_ai/
    operation_specification.py     # new
    operation_registry.py          # new
    validator.py                   # refactored: uses registry instead of ALLOWLIST
    notebook_renderer.py           # refactored: consumes specs for documentation
    tests/
        test_validator.py          # updated: "registry" instead of "allowlist"
        test_registry.py           # new: 13 tests for registry and specs
```

---

## Validation Results

68 tests pass (0 failures):
- 36 fixture coverage tests
- 3 invalid-plan regression tests
- 13 registry tests
- 8 renderer tests
- 8 validator tests

```bash
conda run -n scpy-core python -m pytest spectrochempy_ai/tests/ -v
```

---

## Architectural Verification

| Principle | Status |
|---|---|
| OperationSpecification is pure metadata | Yes |
| OperationSpecification contains no rendering logic | Yes |
| OperationSpecification contains no executable validation logic | Yes |
| OperationRegistry only performs lookup and discovery | Yes |
| Validator uses OperationSpecifications | Yes |
| Renderer consumes OperationSpecifications | Yes |
| Renderer owns rendering logic | Yes |
| WorkflowPlan remains independent from registry | Yes |

---

## Design Decisions

### Input validation by count, not by name

The validator checks that the number of `input_refs` matches the number of
required inputs, rather than matching names exactly. This is because
`input_refs` contains runtime variable names (e.g. `dataset_corrected`),
while `InputSpec.name` describes the semantic slot (e.g. `dataset`).

Name-based matching would require either:
- renaming all intermediate variables to match spec slot names (brittle);
- introducing a mapping abstraction (overkill for Phase 1).

Count validation is sufficient and correct.

### Renderer keeps code generators

The renderer still maintains `_OPERATION_GENERATORS`, a hard-coded mapping
from `operation_id` to Python code generator functions. This is correct
because:
- the specification declares *what* the operation does;
- the renderer decides *how* to express it in Python.

A future script renderer would have its own `_OPERATION_GENERATORS` mapping.

---

## Known Limitations

1. **No parameter type validation yet.** The validator checks parameter names
   against the spec, but does not yet validate types or ranges. This requires
   a small type interpreter that is deferred to Phase 2.

2. **No constraint enforcement.** `OperationSpecification.constraints` are
   populated (e.g. `requires_positive_values` for NMF) but not checked by the
   validator. A `ConstraintInterpreter` is needed but deferred to Phase 2.

3. **OperationStep.output_var remains singular.** The Phase 0.75 review
   recommended changing to `output_vars: dict[str, str]` for multi-output
   operations. This was not implemented in Phase 1 because no current
   operation produces multiple independent variables. It is a non-breaking
   schema change that can be added when needed.

4. **Registry is hard-coded.** Dynamic discovery, plugin loading, and
   metaclasses were intentionally deferred.

---

## Next Steps

Phase 2 should focus on:

1. **Parameter type validation** — validate that parameter values match their
   `ParameterSpec.type` (int, float, str, bool, list).

2. **Schema refinements** — add `side_effects` to `OperationStep` or keep it
   in spec only; decide on `output_vars` multi-output support.

3. **Fixture expansion** — add more workflow families to stress-test the
   registry.

---

## Conclusion

Phase 1 successfully replaced duplicated operation metadata with a single
authoritative OperationRegistry. The validator and renderer now consume
OperationSpecifications instead of hard-coded data. The architecture is
stable and ready for Phase 2.
