"""
Deterministic validator for WorkflowPlan.

Validates WorkflowPlans using OperationSpecifications from the registry.
No hard-coded allowlists. No provider-dependent logic.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from spectrochempy_ai.operation_registry import get_spec
from spectrochempy_ai.operation_registry import is_registered
from spectrochempy_ai.workflow_plan import WorkflowPlan


class ValidationError(Exception):
    """Raised when a WorkflowPlan fails validation."""

    def __init__(self, message: str, violations: list[str]) -> None:
        super().__init__(message)
        self.violations = violations


def validate(plan: WorkflowPlan) -> None:
    """
    Validate a WorkflowPlan against OperationSpecifications.

    Raises
    ------
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

    # 3. Operation validation using registry specs
    step_ids: set[str] = set()
    for step in plan.steps:
        if not step.step_id:
            violations.append(f"step_id is required for operation {step.operation_id}")
            continue
        if step.step_id in step_ids:
            violations.append(f"duplicate step_id: {step.step_id}")
        step_ids.add(step.step_id)

        if not is_registered(step.operation_id):
            violations.append(
                f"operation_id '{step.operation_id}' is not in the registry"
            )
            continue

        spec = get_spec(step.operation_id)

        # 3a. Required inputs present
        required_count = sum(1 for inp in spec.inputs if inp.required)
        provided_count = len(step.input_refs)
        if provided_count < required_count:
            violations.append(
                f"step '{step.step_id}' ({spec.operation_id}) "
                f"has {provided_count} input(s) but requires {required_count}"
            )

        # 3b. Parameters match spec
        spec_params = {p.name for p in spec.parameters}
        for param_name in step.parameters:
            if param_name not in spec_params:
                violations.append(
                    f"step '{step.step_id}' ({spec.operation_id}) "
                    f"has unknown parameter '{param_name}'"
                )

        # 3c. Output variable naming
        # If spec declares outputs, the step should bind at least one
        # Side-effect-only steps (plot, inspect, export) are allowed to have
        # no output_var.
        if spec.outputs and not step.output_var and not spec.side_effects:
            violations.append(
                f"step '{step.step_id}' ({spec.operation_id}) "
                f"produces outputs but has no output_var"
            )

    # 4. Input reference resolution (variables available in pipeline)
    available_vars = {inp.name for inp in plan.inputs}
    for step in plan.steps:
        for ref in step.input_refs:
            if ref not in available_vars:
                violations.append(
                    f"step '{step.step_id}' references unresolved variable '{ref}'"
                )
        if step.output_var:
            available_vars.add(step.output_var)

    # 5. Output references must match produced variables
    produced = {step.output_var for step in plan.steps if step.output_var}
    for out in plan.outputs:
        if out.name not in produced and out.name not in {
            inp.name for inp in plan.inputs
        }:
            violations.append(
                f"output '{out.name}' is not produced by any step or available as input"
            )

    if violations:
        raise ValidationError(
            f"WorkflowPlan validation failed with {len(violations)} violation(s)",
            violations,
        )


class NotebookExecutionError(Exception):
    """Raised when notebook execution validation fails."""

    def __init__(self, message: str, cell_errors: list[tuple[int, str, str]]) -> None:
        super().__init__(message)
        self.cell_errors = cell_errors


_CELL_EXCLUDE_PREFIXES = ("get_ipython",)


def _should_skip(code: str) -> bool:
    code = code.strip()
    if not code:
        return True
    return any(code.startswith(p) for p in _CELL_EXCLUDE_PREFIXES)


def validate_notebook_execution(notebook_path: str | Path) -> None:
    """
    Execute all code cells from a generated notebook and report errors.

    Skips:
    - markdown cells
    - ``get_ipython`` magic

    Forces ``Agg`` backend so plots are never shown on screen.

    Raises
    ------
    NotebookExecutionError
        If any executed cell raises an exception.
    """
    import matplotlib as mpl

    mpl.use("Agg")

    import nbformat

    nb = nbformat.read(str(notebook_path), as_version=4)

    cell_errors: list[tuple[int, str, str]] = []
    ns: dict = {}

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        code = cell.source.strip()
        if _should_skip(code):
            continue
        try:
            exec(code, ns)
        except Exception:
            cls, exc, tb = sys.exc_info()
            tb_str = "".join(traceback.format_exception(cls, exc, tb))
            cell_errors.append((i, code.split(chr(10))[0], tb_str))

    if cell_errors:
        raise NotebookExecutionError(
            f"Notebook execution failed in {len(cell_errors)} cell(s)",
            cell_errors,
        )
