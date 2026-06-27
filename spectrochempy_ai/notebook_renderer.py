"""
Deterministic notebook renderer for WorkflowPlan.

Translates a validated WorkflowPlan into a Jupyter notebook using nbformat.
The same plan produces the same notebook under the same renderer version.

No AI. No providers. No prompts. Only deterministic cell generation.
"""

from __future__ import annotations

from typing import Any

import nbformat
from nbformat.v4 import new_code_cell
from nbformat.v4 import new_markdown_cell
from nbformat.v4 import new_notebook

from spectrochempy_ai.operation_registry import get_spec
from spectrochempy_ai.operation_registry import is_registered
from spectrochempy_ai.workflow_plan import OperationStep
from spectrochempy_ai.workflow_plan import WorkflowPlan

# Mapping from operation_id to the Python code generator function.
# The renderer owns rendering logic. It consumes OperationSpecifications
# from the registry to validate and document operations, but the actual
# code generation remains here.


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
    return f"# PCA score plot\n" f"{inp}.plot_score(cmap='viridis')"


def _generate_loading_plot(step: OperationStep) -> str:
    """Generate code for PCA loading plot."""
    inp = step.input_refs[0] if step.input_refs else "pca_result"
    return f"# PCA loading plot\n{inp}.loadings.plot(cmap='viridis')"


def _generate_smooth(step: OperationStep) -> str:
    """Generate code for smoothing."""
    inp = step.input_refs[0] if step.input_refs else "dataset"
    method = step.parameters.get("method", "savgol")
    size = step.parameters.get("window_length", 5)
    order = step.parameters.get("polyorder", 2)
    return (
        f"# Smoothing ({method})\n"
        f"{step.output_var} = scp.processing.filter.filter.{method}({inp}, size={size}, order={order})\n"
        f"print('Smoothed shape:', {step.output_var}.shape)"
    )


def _generate_integrate(step: OperationStep) -> str:
    """Generate code for integration."""
    inp = step.input_refs[0] if step.input_refs else "dataset"
    method = step.parameters.get("method", "trapezoid")
    return (
        f"# Integration ({method})\n"
        f"{step.output_var} = scp.analysis.integration.integrate.{method}({inp})\n"
        f"print('Integrated shape:', {step.output_var}.shape)"
    )


def _generate_plot(step: OperationStep) -> str:
    """Generate generic plot code."""
    inp = step.input_refs[0] if step.input_refs else "dataset"
    plot_type = step.parameters.get("plot_type", "line")
    if plot_type == "line":
        return f"# Plot ({plot_type})\n{inp}.plot()"
    return f"# Plot ({plot_type})\n{inp}.plot()"


def _generate_nmf(step: OperationStep) -> str:
    """Generate code for NMF."""
    inp = step.input_refs[0] if step.input_refs else "dataset"
    n_components = step.parameters.get("n_components", 2)
    max_iter = step.parameters.get("max_iter", 500)
    return (
        f"# Non-negative Matrix Factorisation\n"
        f"nmf = scp.analysis.decomposition.nmf.NMF(n_components={n_components}, max_iter={max_iter})\n"
        f"nmf.fit({inp})\n"
        f"{step.output_var} = nmf\n"
        f"print('NMF components shape:', nmf.components.shape)"
    )


def _generate_nmf_components_plot(step: OperationStep) -> str:
    """Generate code for NMF components plot."""
    inp = step.input_refs[0] if step.input_refs else "nmf_result"
    return f"# NMF components plot\n{inp}.components.plot(cmap='viridis')"


def _generate_nmf_reconstruction_plot(step: OperationStep) -> str:
    """Generate code for NMF reconstruction plot."""
    inp = step.input_refs[0] if step.input_refs else "nmf_result"
    return f"# NMF reconstruction plot\n{inp}.reconstruct().plot(cmap='viridis')"


def _generate_mcrals(step: OperationStep) -> str:
    """Generate code for MCR-ALS."""
    inp = step.input_refs[0] if step.input_refs else "dataset"
    y = step.input_refs[1] if len(step.input_refs) > 1 else "conc_guess"
    max_iter = step.parameters.get("max_iter", 100)
    return (
        f"# MCR-ALS decomposition\n"
        f"mcr = scp.analysis.decomposition.mcrals.MCRALS()\n"
        f"mcr.fit({inp}, {y})\n"
        f"{step.output_var} = mcr\n"
        f"print('MCR-ALS C shape:', mcr.C.shape)\n"
        f"print('MCR-ALS St shape:', mcr.St.shape)"
    )


def _generate_mcrals_conc_plot(step: OperationStep) -> str:
    """Generate code for MCR-ALS concentration plot."""
    inp = step.input_refs[0] if step.input_refs else "mcrals_result"
    return f"# MCR-ALS concentration profiles\n{inp}.C.plot(cmap='viridis')"


def _generate_mcrals_spec_plot(step: OperationStep) -> str:
    """Generate code for MCR-ALS spectra plot."""
    inp = step.input_refs[0] if step.input_refs else "mcrals_result"
    return f"# MCR-ALS resolved spectra\n{inp}.St.plot(cmap='viridis')"


def _generate_load(step: OperationStep) -> str:
    """Generate code for loading a dataset from file."""
    filename = step.parameters.get("filename", "data.scp")
    fmt = step.parameters.get("format", "scp")
    return (
        f"import spectrochempy as scp\n\n"
        f"# Load spectral dataset from file\n"
        f"{step.output_var} = scp.read('{filename}', format='{fmt}')\n"
        f"print('Loaded: {filename}')"
    )


def _generate_scree_plot(step: OperationStep) -> str:
    """Generate code for a PCA scree plot with explained variance."""
    inp = step.input_refs[0] if step.input_refs else "pca_result"
    return (
        f"# Scree plot: explained variance per component\n"
        f"{inp}.plot_scree()"
    )


def _generate_inspect(step: OperationStep) -> str:
    """Generate code for dataset inspection."""
    inp = step.input_refs[0] if step.input_refs else "dataset"
    return (
        f"# Dataset inspection\n"
        f"print('Shape:', {inp}.shape)\n"
        f"print('Dimensions:', {inp}.dims)\n"
        f"print('Coordset:', {inp}.coordset)"
    )


def _generate_export(step: OperationStep) -> str:
    """Generate code for dataset export."""
    inp = step.input_refs[0] if step.input_refs else "dataset"
    filename = step.parameters.get("filename", "output.scp")
    fmt = step.parameters.get("format", "scp")
    return (
        f"# Export dataset\n"
        f"scp.write({inp}, '{filename}')\n"
        f"print('Exported to: {filename}')"
    )


_OPERATION_GENERATORS: dict[str, Any] = {
    "load": _generate_load,
    "read": _generate_read,
    "baseline": _generate_baseline,
    "smooth": _generate_smooth,
    "pca": _generate_pca,
    "score_plot": _generate_score_plot,
    "loading_plot": _generate_loading_plot,
    "scree_plot": _generate_scree_plot,
    "integrate": _generate_integrate,
    "plot": _generate_plot,
    "nmf": _generate_nmf,
    "nmf_components_plot": _generate_nmf_components_plot,
    "nmf_reconstruction_plot": _generate_nmf_reconstruction_plot,
    "mcrals": _generate_mcrals,
    "mcrals_conc_plot": _generate_mcrals_conc_plot,
    "mcrals_spec_plot": _generate_mcrals_spec_plot,
    "inspect": _generate_inspect,
    "export": _generate_export,
}


def render(plan: WorkflowPlan) -> nbformat.NotebookNode:
    """
    Render a validated WorkflowPlan into a deterministic Jupyter notebook.

    Returns
    -------
        An nbformat NotebookNode ready to be written to ``.ipynb``.
    """
    cells: list[Any] = []

    # 1. Short title + goal summary
    # Derive a display name from template_id
    _DISPLAY_NAMES = {
        "exploratory_pca": "Exploratory PCA",
        "baseline_integrate": "Baseline + Integrate",
        "nmf_exploration": "NMF Exploration",
    }
    template_id = plan.planner_config.get("template_id", "workflow")
    display_name = _DISPLAY_NAMES.get(
        template_id, template_id.replace("_", " ").title()
    )

    # Try to extract the dataset filename from the first step that has one
    dataset_label = ""
    for step in plan.steps:
        if "filename" in step.parameters and step.parameters["filename"]:
            dataset_label = step.parameters["filename"]
            break

    title = display_name
    if dataset_label:
        title += f" — {dataset_label}"

    cells.append(
        new_markdown_cell(
            f"# {title}\n\n"
            f"{plan.scientific_context.goal}\n\n"
            f"**Strategy:** {plan.scientific_context.analytical_strategy}"
        )
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
        # Consume OperationSpecification from registry for documentation
        spec_description = ""
        if is_registered(step.operation_id):
            spec = get_spec(step.operation_id)
            spec_description = spec.description

        # Markdown cell with rationale and optional spec description
        rationale_md = f"## {step.display_label}\n\n**Rationale:** {step.rationale}"
        if spec_description:
            rationale_md += f"\n\n*Description:* {spec_description}"
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
