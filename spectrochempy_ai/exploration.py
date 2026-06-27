"""
High-level exploration API for Phase 6.

Thin orchestration only. Does not duplicate planner, validator, or
renderer logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spectrochempy_ai.notebook_renderer import write_notebook
from spectrochempy_ai.template_planner import TemplatePlanner
from spectrochempy_ai.validator import validate as validate_plan


def explore(
    input_path: str,
    output_path: str | None = None,
    *,
    template_id: str = "exploratory_pca",
    n_components: int | None = None,
    baseline_method: str | None = None,
    file_format: str | None = None,
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
        template_id: Template to use (default exploratory_pca).
        n_components: Override PCA component count.
        baseline_method: Override baseline correction method.
        file_format: File format override (default from template).

    Returns:
    -------
        Path to the written notebook.

    Raises:
    ------
        FileNotFoundError: If input_path does not exist.
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

    if output_path is None:
        stem = src.stem
        output_path = f"{stem}-exploratory-pca.ipynb"
    dst = Path(output_path)

    planner = TemplatePlanner()

    # Build overrides by operation_id — no step_id knowledge here
    operation_overrides: dict[str, dict[str, Any]] = {}
    operation_overrides.setdefault("load", {})["filename"] = str(src)
    if file_format is not None:
        operation_overrides["load"]["format"] = file_format
    if n_components is not None:
        operation_overrides.setdefault("pca", {})["n_components"] = n_components
    if baseline_method is not None:
        operation_overrides.setdefault("baseline", {})["method"] = baseline_method

    plan = planner.create_plan(
        template_id,
        operation_overrides=operation_overrides,
    )
    validate_plan(plan)
    write_notebook(plan, str(dst))
    return dst.resolve()


# Internal alias — preserved for compatibility and direct access
create_exploration_notebook = explore
