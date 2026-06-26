"""Deterministic notebook renderer for WorkflowPlan.

Translates a validated WorkflowPlan into a Jupyter notebook using nbformat.
The same plan produces the same notebook under the same renderer version.

No AI. No providers. No prompts. Only deterministic cell generation.
"""

from __future__ import annotations

import json
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from spectrochempy_ai.workflow_plan import OperationStep, WorkflowPlan

# Mapping from operation_id to the Python code generator function.
# This is a hard-coded prototype mapping. A real operation registry
# belongs to Phase 1.


def _generate_read(step: OperationStep) -> str:
    """Generate code for the 'read' operation."""
    params = step.parameters
    shape = params.get("shape", [50, 100])
    seed = params.get("random_seed", 42)
    return (
        f"import numpy as np\n"
        f"import spectrochempy as scp\n\n"
        f"# Reproducible synthetic dataset\n"
        f"np.random.seed({seed})\n"
        f"_data = np.random.rand({shape[0]}, {shape[1]})\n"
        f"_x = np.linspace(0, 100, {shape[1]})\n"
        f"_y = np.linspace(0, {shape[0]}, {shape[0]})\n"
        f"{step.output_var} = scp.NDDataset(\n"
        f"    _data,\n"
        f"    coordset=[\n"
        f"        scp.Coord(_y, title='samples'),\n"
        f"        scp.Coord(_x, title='wavelength')\n"
        f"    ]\n"
        f")\n"
        f"print('Dataset shape:', {step.output_var}.shape)"
    )


def _generate_baseline(step: OperationStep) -> str:
    """Generate code for baseline correction."""
    inp = step.input_refs[0] if step.input_refs else "dataset"
    method = step.parameters.get("method", "asls")
    return (
        f"# Baseline correction ({method})\n"
        f"{step.output_var} = scp.processing.baselineprocessing.baselineprocessing.{method}({inp})\n"
        f"print('Baseline corrected shape:', {step.output_var}.shape)"
    )


def _generate_pca(step: OperationStep) -> str:
    """Generate code for PCA."""
    inp = step.input_refs[0] if step.input_refs else "dataset"
    n_components = step.parameters.get("n_components", 3)
    return (
        f"# Principal Component Analysis\n"
        f"pca = scp.analysis.decomposition.pca.PCA(n_components={n_components})\n"
        f"pca.fit({inp})\n"
        f"{step.output_var} = pca\n"
        f"print('Explained variance ratio:', pca.explained_variance_ratio.data)"
    )


def _generate_score_plot(step: OperationStep) -> str:
    """Generate code for PCA score plot."""
    inp = step.input_refs[0] if step.input_refs else "pca_result"
    return (
        f"# PCA score plot\n"
        f"{inp}.scoreplot(cmap='viridis')\n"
        f"scp.show()"
    )


def _generate_loading_plot(step: OperationStep) -> str:
    """Generate code for PCA loading plot."""
    inp = step.input_refs[0] if step.input_refs else "pca_result"
    return (
        f"# PCA loading plot\n"
        f"{inp}.loadings.plot(cmap='viridis')\n"
        f"scp.show()"
    )


_OPERATION_GENERATORS: dict[str, Any] = {
    "read": _generate_read,
    "baseline": _generate_baseline,
    "pca": _generate_pca,
    "score_plot": _generate_score_plot,
    "loading_plot": _generate_loading_plot,
}


def render(plan: WorkflowPlan) -> nbformat.NotebookNode:
    """Render a validated WorkflowPlan into a deterministic Jupyter notebook.

    Returns:
        An nbformat NotebookNode ready to be written to ``.ipynb``.
    """
    cells: list[Any] = []

    # 1. Title and goal
    cells.append(
        new_markdown_cell(f"# {plan.scientific_context.goal}\n\n"
                          f"**Strategy:** {plan.scientific_context.analytical_strategy}")
    )

    # 2. Reproducibility manifest
    repro = plan.reproducibility
    repro_lines = [
        "## Reproducibility Manifest",
        "",
        f"- **Schema version:** {plan.schema_version}",
        f"- **SpectroChemPy version:** {plan.spectrochempy_version}",
        f"- **Plugin version:** {plan.plugin_version}",
        f"- **Planner:** {plan.planner_id}",
    ]
    if repro.package_versions:
        repro_lines.append("- **Package versions:**")
        for pkg, ver in repro.package_versions.items():
            repro_lines.append(f"  - {pkg}: {ver}")
    if repro.random_seeds:
        repro_lines.append("- **Random seeds:**")
        for name, seed in repro.random_seeds.items():
            repro_lines.append(f"  - {name}: {seed}")
    repro_lines.append(f"- **Timestamp:** {plan.timestamp}")
    cells.append(new_markdown_cell("\n".join(repro_lines)))

    # 3. Scientific context and assumptions
    ctx = plan.scientific_context
    ctx_lines = ["## Scientific Context", ""]
    ctx_lines.append(f"**Goal:** {ctx.goal}")
    ctx_lines.append(f"**Strategy:** {ctx.analytical_strategy}")
    if ctx.data_assumptions:
        ctx_lines.append("")
        ctx_lines.append("**Data assumptions:**")
        for assumption in ctx.data_assumptions:
            ctx_lines.append(f"- {assumption}")
    if ctx.validation_criteria:
        ctx_lines.append("")
        ctx_lines.append("**Validation criteria:**")
        for criterion in ctx.validation_criteria:
            ctx_lines.append(f"- {criterion}")
    if ctx.expected_outputs:
        ctx_lines.append("")
        ctx_lines.append("**Expected outputs:**")
        for out in ctx.expected_outputs:
            ctx_lines.append(f"- {out}")
    if ctx.limitations:
        ctx_lines.append("")
        ctx_lines.append("**Limitations:**")
        for limitation in ctx.limitations:
            ctx_lines.append(f"- {limitation}")
    cells.append(new_markdown_cell("\n".join(ctx_lines)))

    # 4. Imports
    cells.append(
        new_code_cell(
            "import numpy as np\n"
            "import spectrochempy as scp\n\n"
            "print('SpectroChemPy version:', scp.__version__)"
        )
    )

    # 5. Processing steps
    for step in plan.steps:
        # Markdown cell with rationale
        rationale_md = f"## {step.display_label}\n\n**Rationale:** {step.rationale}"
        cells.append(new_markdown_cell(rationale_md))

        # Code cell
        generator = _OPERATION_GENERATORS.get(step.operation_id)
        if generator is None:
            code = f"# Unknown operation: {step.operation_id}\nraise NotImplementedError('{step.operation_id}')"
        else:
            code = generator(step)
        cells.append(new_code_cell(code))

    # 6. Limitations and validation criteria (recap)
    if ctx.limitations or ctx.validation_criteria:
        recap_lines = ["## Summary and Caveats", ""]
        if ctx.validation_criteria:
            recap_lines.append("**Validation criteria:**")
            for criterion in ctx.validation_criteria:
                recap_lines.append(f"- {criterion}")
            recap_lines.append("")
        if ctx.limitations:
            recap_lines.append("**Limitations:**")
            for limitation in ctx.limitations:
                recap_lines.append(f"- {limitation}")
        cells.append(new_markdown_cell("\n".join(recap_lines)))

    # 7. Notebook metadata / manifest
    manifest = {
        "spectrochempy_workflow_assistant": {
            "schema_version": plan.schema_version,
            "planner_id": plan.planner_id,
            "timestamp": plan.timestamp,
        }
    }

    notebook = new_notebook(cells=cells)
    notebook.metadata.update(manifest)
    return notebook


def write_notebook(plan: WorkflowPlan, path: str) -> None:
    """Render a plan and write it to a ``.ipynb`` file."""
    notebook = render(plan)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)
