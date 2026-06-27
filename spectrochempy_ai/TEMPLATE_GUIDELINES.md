# Template Guidelines

## Reference Template

The `exploratory-pca` template is the reference implementation for all
future WorkflowTemplates. It established what a production-quality template
looks like.

## Required Components

Every template must provide:

| Component | Location | Purpose |
|---|---|---|
| `WorkflowTemplate` definition | `template_planner.py` method | Executable template with all metadata |
| Standalone documentation | `templates/<template-id>.md` | Scientific methodology document |
| Comprehensive tests | `tests/test_planner.py` | Structural, data-flow, and pipeline tests |

## Documentation Standard

The documentation must:

- State the **scientific objective** — what question does this workflow answer?
- List **assumptions** about input data.
- List **limitations** — when should this template NOT be used?
- Describe **expected outputs** — both data and plots.
- Provide a **step-by-step methodology** table.
- Justify every operation with a **scientific rationale** (not a software
  rationale).
- Provide **parameter recommendations** with justification.
- Include **interpretation guidance** — how to read the results.
- Cite **references** to published methodology.

The document should describe methodology, not software. It should be
understandable to an experienced spectroscopist who does not know
SpectroChemPy's API.

## Test Standard

At minimum, test:

- Registration (including idempotency).
- Step count and operation IDs match expectations.
- Data flow between steps (input_refs chain).
- Parameter defaults are correct.
- Per-step parameter overrides.
- Scientific context completeness.
- Input/output references.
- Template metadata (version, registry version).
- Full pipeline: register → instantiate → validate → render.
- Deterministic plan generation.
- Serialization roundtrip.
- Error paths (unknown parameters, bad step IDs).

## Template Versioning

- Initial version: `0.1.0`.
- MAJOR bump: structural change that breaks backward compatibility.
- MINOR bump: new steps, optional parameters, improved defaults.
- PATCH bump: documentation fixes, corrected references.

## Registry Compatibility

`compatible_registry_version` must be set to the `REGISTRY_VERSION` that
the template was validated against. Update it when new operation specs are
required.

## Scientific Review

Before finalising a template, verify:

- Is this a recognised scientific methodology?
- Would an experienced spectroscopist recognise and accept this workflow?
- Are all parameter defaults justified (by reference or reasoning)?
- Are limitations documented?
- Could the template produce misleading results if used outside its
  assumptions?

## Reference

See `exploratory-pca.md` for the canonical example.
