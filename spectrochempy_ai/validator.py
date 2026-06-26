"""Deterministic validator for WorkflowPlan.

The validator enforces a hard-coded allowlist and structural rules.
It is deterministic, provider-independent, and fails closed.

No AI repair loop. No runtime execution.
"""

from __future__ import annotations

from spectrochempy_ai.workflow_plan import WorkflowPlan

# Hard-coded allowlist for the Phase 0 prototype.
# This is intentionally minimal. A real operation registry belongs to Phase 1.
ALLOWLIST: set[str] = {
    "read",
    "baseline",
    "smooth",
    "pca",
    "score_plot",
    "loading_plot",
    "integrate",
    "plot",
    "nmf",
    "nmf_components_plot",
    "nmf_reconstruction_plot",
    "mcrals",
    "mcrals_conc_plot",
    "mcrals_spec_plot",
    "inspect",
    "export",
}


class ValidationError(Exception):
    """Raised when a WorkflowPlan fails validation."""

    def __init__(self, message: str, violations: list[str]) -> None:
        super().__init__(message)
        self.violations = violations


def validate(plan: WorkflowPlan) -> None:
    """Validate a WorkflowPlan against the Phase 0 rules.

    Raises:
        ValidationError: if any rule is violated.
    """
    violations: list[str] = []

    # 1. Schema-level required fields
    if not plan.schema_version:
        violations.append("schema_version is required")
    if not plan.spectrochempy_version:
        violations.append("spectrochempy_version is required")
    if not plan.plugin_version:
        violations.append("plugin_version is required")

    # 2. Scientific context required fields
    sci = plan.scientific_context
    if sci is None:
        violations.append("scientific_context is required")
    else:
        if not sci.goal:
            violations.append("scientific_context.goal is required")
        if not sci.analytical_strategy:
            violations.append("scientific_context.analytical_strategy is required")

    # 3. Operation allowlist
    step_ids: set[str] = set()
    for step in plan.steps:
        if not step.step_id:
            violations.append(f"step_id is required for operation {step.operation_id}")
            continue
        if step.step_id in step_ids:
            violations.append(f"duplicate step_id: {step.step_id}")
        step_ids.add(step.step_id)

        if step.operation_id not in ALLOWLIST:
            violations.append(
                f"operation_id '{step.operation_id}' is not in the allowlist"
            )

    # 4. Input reference resolution
    available_vars = {inp.name for inp in plan.inputs}
    for step in plan.steps:
        for ref in step.input_refs:
            if ref not in available_vars:
                violations.append(
                    f"step '{step.step_id}' references unresolved input '{ref}'"
                )
        if step.output_var:
            available_vars.add(step.output_var)

    # 5. Output references must match produced variables
    output_names = {out.name for out in plan.outputs}
    produced = {step.output_var for step in plan.steps if step.output_var}
    for out in plan.outputs:
        if out.name not in produced and out.name not in {inp.name for inp in plan.inputs}:
            violations.append(
                f"output '{out.name}' is not produced by any step or available as input"
            )

    if violations:
        raise ValidationError(
            f"WorkflowPlan validation failed with {len(violations)} violation(s)",
            violations,
        )
