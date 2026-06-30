"""
High-level exploration API for Phase 6.

Thin orchestration only. Does not duplicate planner, validator, or
renderer logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spectrochempy_ai.dataset_profile import resolve_dataset_source
from spectrochempy_ai.notebook_renderer import write_notebook
from spectrochempy_ai.rule_planner import suggest
from spectrochempy_ai.template_planner import TemplatePlanner
from spectrochempy_ai.validator import validate as validate_plan


def explore(
    input_path: str,
    output_path: str | None = None,
    *,
    template_id: str | None = None,
    n_components: int | None = None,
    baseline_method: str | None = None,
    file_format: str | None = None,
    reference_path: str | None = None,
) -> Path:
    """
    Create a validated, reproducible exploratory notebook.

    This is the minimal user-facing entry point. It selects a template,
    overrides parameters, generates a WorkflowPlan, validates it, and
    writes a runnable Jupyter notebook.

    Args:
    ----
        input_path: Path to the spectral dataset file.
        output_path: Path for the output notebook. Derived from input
            filename if not provided (data.scp -> data-exploratory-pca.ipynb).
        template_id: Template to use. If omitted, select the top
            recommendation from ``suggest()``.
        n_components: Override component count.
        baseline_method: Override baseline correction method.
        file_format: File format override (default from template).
        reference_path: Path to reference values file (for multi-input
            templates like pls_calibration).

    Returns:
    -------
        Path to the written notebook.

    Raises:
    ------
        FileNotFoundError: If input_path or reference_path do not exist.
        TemplateNotFoundError: If template_id is unknown.
        UnknownParameterError: If a parameter override is invalid.
        ValidationError: If the generated plan fails validation.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(
            f"Input file not found: {src}\n"
            f"Provide a valid path to a spectral dataset file."
        )

    selection = resolve_dataset_source(src)
    if selection.source_was_multi_object and selection.dataset is None:
        raise ValueError(
            selection.error or "No suitable dataset found in multi-object file"
        )

    resolved_template_id = template_id
    if resolved_template_id is None:
        recommendations = suggest(str(src), reference_path=reference_path)
        resolved_template_id = recommendations[0].template_id

    if output_path is None:
        stem = src.stem
        slug = resolved_template_id.replace("_", "-")
        output_path = f"{stem}-{slug}.ipynb"
    dst = Path(output_path)

    planner = TemplatePlanner()
    template = planner.get_template(resolved_template_id)
    valid_ops = {step.operation_id for step in template.steps}

    parameter_overrides: dict[str, dict[str, Any]] = {}

    primary_input_name = template.inputs[0].name if template.inputs else "dataset"
    primary_load_step_id = next(
        (
            step.step_id
            for step in template.steps
            if step.operation_id == "load" and step.output_var == primary_input_name
        ),
        None,
    )
    if primary_load_step_id is not None:
        parameter_overrides.setdefault(primary_load_step_id, {})["filename"] = str(src)
        if selection.source_was_multi_object:
            parameter_overrides[primary_load_step_id]["selected_index"] = (
                selection.selected_object_index
            )
            parameter_overrides[primary_load_step_id]["selected_name"] = (
                selection.selected_object_name
            )
            parameter_overrides[primary_load_step_id]["source_object_count"] = (
                selection.source_object_count
            )
        if file_format is not None:
            parameter_overrides[primary_load_step_id]["format"] = file_format

    # Build overrides by operation_id — only for operations that exist
    # in the chosen template.
    operation_overrides: dict[str, dict[str, Any]] = {}
    if n_components is not None:
        # Determine which decomposition operation this template uses
        for op in ("pca", "nmf", "mcrals", "pls"):
            if op in valid_ops:
                operation_overrides.setdefault(op, {})["n_components"] = n_components
                break
    if baseline_method is not None and "baseline" in valid_ops:
        operation_overrides.setdefault("baseline", {})["method"] = baseline_method

    if reference_path is not None:
        ref_src = Path(reference_path)
        if not ref_src.exists():
            raise FileNotFoundError(
                f"Reference file not found: {ref_src}\n"
                f"Provide a valid path to the reference values file."
            )
        # Find non-primary load steps (those whose output_var is not the
        # first template input).  For pls_calibration this matches s2
        # (output_var="reference"), overriding its filename to the
        # reference file while s1 keeps the spectral data file.
        for step in template.steps:
            if step.operation_id == "load" and step.output_var != primary_input_name:
                parameter_overrides.setdefault(step.step_id, {})["filename"] = str(
                    ref_src
                )

    plan = planner.create_plan(
        resolved_template_id,
        operation_overrides=operation_overrides,
        parameter_overrides=parameter_overrides or None,
    )
    validate_plan(plan)
    write_notebook(plan, str(dst))
    return dst.resolve()


# Internal alias — preserved for compatibility and direct access
create_exploration_notebook = explore
