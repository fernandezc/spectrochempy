"""Minimal OperationRegistry for Phase 1.

A hard-coded dictionary of OperationSpecifications.
No dynamic discovery. No plugin loading. No metaclasses.
"""

from __future__ import annotations

from spectrochempy_ai.operation_specification import (
    Constraint,
    InputSpec,
    OperationSpecification,
    OutputSpec,
    ParameterSpec,
)

# ---------------------------------------------------------------------------
# Operation specifications
# ---------------------------------------------------------------------------

_READ = OperationSpecification(
    operation_id="read",
    display_name="Read / Generate Dataset",
    description="Generate a reproducible synthetic NDDataset for testing and exploration.",
    inputs=[],
    outputs=[
        OutputSpec(name="dataset", type="dataset", description="Generated synthetic dataset"),
    ],
    parameters=[
        ParameterSpec(name="shape", type="list", default=[50, 100], description="Dataset shape as [observations, variables]"),
        ParameterSpec(name="random_seed", type="int", default=42, description="Random seed for reproducibility"),
        ParameterSpec(name="non_negative", type="bool", default=False, description="Generate non-negative data (useful for NMF)"),
    ],
    side_effects=[],
    category="io",
)

_BASELINE = OperationSpecification(
    operation_id="baseline",
    display_name="Baseline Correction",
    description="Remove baseline drift from a dataset using a chosen baseline algorithm.",
    inputs=[
        InputSpec(name="dataset", type="dataset", required=True, description="Dataset to correct"),
    ],
    outputs=[
        OutputSpec(name="dataset_corrected", type="dataset", description="Baseline-corrected dataset"),
    ],
    parameters=[
        ParameterSpec(name="method", type="str", default="asls", description="Baseline correction method (asls, detrend, etc.)"),
    ],
    side_effects=[],
    category="preprocessing",
)

_SMOOTH = OperationSpecification(
    operation_id="smooth",
    display_name="Smoothing",
    description="Apply a smoothing filter to reduce noise along the spectral axis.",
    inputs=[
        InputSpec(name="dataset", type="dataset", required=True, description="Dataset to smooth"),
    ],
    outputs=[
        OutputSpec(name="dataset_smoothed", type="dataset", description="Smoothed dataset"),
    ],
    parameters=[
        ParameterSpec(name="method", type="str", default="savgol", description="Smoothing method"),
        ParameterSpec(name="window_length", type="int", default=5, description="Window length for Savitzky-Golay filter"),
        ParameterSpec(name="polyorder", type="int", default=2, description="Polynomial order for Savitzky-Golay filter"),
    ],
    side_effects=[],
    category="preprocessing",
)

_PCA = OperationSpecification(
    operation_id="pca",
    display_name="Principal Component Analysis",
    description="Decompose variance into principal components using PCA.",
    inputs=[
        InputSpec(name="dataset", type="dataset", required=True, description="Dataset to decompose"),
    ],
    outputs=[
        OutputSpec(name="pca_result", type="result", description="PCA estimator with scores, loadings, and explained variance"),
    ],
    parameters=[
        ParameterSpec(name="n_components", type="int", default=3, description="Number of principal components to compute"),
    ],
    side_effects=[],
    category="analysis",
)

_SCORE_PLOT = OperationSpecification(
    operation_id="score_plot",
    display_name="PCA Score Plot",
    description="Visualise sample distribution in principal component space.",
    inputs=[
        InputSpec(name="pca_result", type="result", required=True, description="Fitted PCA estimator"),
    ],
    outputs=[],
    parameters=[],
    side_effects=["plot"],
    category="plotting",
)

_LOADING_PLOT = OperationSpecification(
    operation_id="loading_plot",
    display_name="PCA Loading Plot",
    description="Visualise variable contributions to each principal component.",
    inputs=[
        InputSpec(name="pca_result", type="result", required=True, description="Fitted PCA estimator"),
    ],
    outputs=[],
    parameters=[],
    side_effects=["plot"],
    category="plotting",
)

_INTEGRATE = OperationSpecification(
    operation_id="integrate",
    display_name="Integration",
    description="Integrate signal along an axis using a numerical method.",
    inputs=[
        InputSpec(name="dataset", type="dataset", required=True, description="Dataset to integrate"),
    ],
    outputs=[
        OutputSpec(name="area_profile", type="dataset", description="Integrated result (often 1D)"),
    ],
    parameters=[
        ParameterSpec(name="method", type="str", default="trapezoid", description="Integration method"),
    ],
    side_effects=[],
    category="analysis",
)

_PLOT = OperationSpecification(
    operation_id="plot",
    display_name="Plot",
    description="Generic line or image plot of a dataset.",
    inputs=[
        InputSpec(name="dataset", type="dataset", required=True, description="Dataset to plot"),
    ],
    outputs=[],
    parameters=[
        ParameterSpec(name="plot_type", type="str", default="line", description="Plot type (line, image, etc.)"),
    ],
    side_effects=["plot"],
    category="plotting",
)

_NMF = OperationSpecification(
    operation_id="nmf",
    display_name="Non-negative Matrix Factorisation",
    description="Decompose a non-negative dataset into components and loadings.",
    inputs=[
        InputSpec(name="dataset", type="dataset", required=True, description="Non-negative dataset to decompose"),
    ],
    outputs=[
        OutputSpec(name="nmf_result", type="result", description="NMF estimator with components and reconstruction"),
    ],
    parameters=[
        ParameterSpec(name="n_components", type="int", default=2, description="Number of components"),
        ParameterSpec(name="max_iter", type="int", default=500, description="Maximum iterations"),
    ],
    constraints=[
        Constraint(predicate="requires_positive_values", description="Data must be non-negative for NMF"),
    ],
    side_effects=[],
    category="analysis",
)

_NMF_COMPONENTS_PLOT = OperationSpecification(
    operation_id="nmf_components_plot",
    display_name="NMF Components Plot",
    description="Visualise the extracted spectral components from NMF.",
    inputs=[
        InputSpec(name="nmf_result", type="result", required=True, description="Fitted NMF estimator"),
    ],
    outputs=[],
    parameters=[],
    side_effects=["plot"],
    category="plotting",
)

_NMF_RECONSTRUCTION_PLOT = OperationSpecification(
    operation_id="nmf_reconstruction_plot",
    display_name="NMF Reconstruction Plot",
    description="Visualise the dataset reconstructed from NMF components.",
    inputs=[
        InputSpec(name="nmf_result", type="result", required=True, description="Fitted NMF estimator"),
    ],
    outputs=[],
    parameters=[],
    side_effects=["plot"],
    category="plotting",
)

_MCRALS = OperationSpecification(
    operation_id="mcrals",
    display_name="MCR-ALS",
    description="Multivariate Curve Resolution by Alternating Least Squares.",
    inputs=[
        InputSpec(name="dataset", type="dataset", required=True, description="Mixture dataset to resolve"),
        InputSpec(name="conc_guess", type="dataset", required=True, description="Initial concentration guess"),
    ],
    outputs=[
        OutputSpec(name="mcrals_result", type="result", description="MCR-ALS estimator with C and St matrices"),
    ],
    parameters=[
        ParameterSpec(name="max_iter", type="int", default=100, description="Maximum iterations"),
    ],
    side_effects=[],
    category="analysis",
)

_MCRALS_CONC_PLOT = OperationSpecification(
    operation_id="mcrals_conc_plot",
    display_name="MCR-ALS Concentration Plot",
    description="Visualise resolved concentration profiles.",
    inputs=[
        InputSpec(name="mcrals_result", type="result", required=True, description="Fitted MCR-ALS estimator"),
    ],
    outputs=[],
    parameters=[],
    side_effects=["plot"],
    category="plotting",
)

_MCRALS_SPEC_PLOT = OperationSpecification(
    operation_id="mcrals_spec_plot",
    display_name="MCR-ALS Spectra Plot",
    description="Visualise resolved pure spectra.",
    inputs=[
        InputSpec(name="mcrals_result", type="result", required=True, description="Fitted MCR-ALS estimator"),
    ],
    outputs=[],
    parameters=[],
    side_effects=["plot"],
    category="plotting",
)

_INSPECT = OperationSpecification(
    operation_id="inspect",
    display_name="Inspect Dataset",
    description="Print a summary of dataset shape, dimensions, and coordinates.",
    inputs=[
        InputSpec(name="dataset", type="dataset", required=True, description="Dataset to inspect"),
    ],
    outputs=[],
    parameters=[],
    side_effects=["print"],
    category="inspection",
)

_EXPORT = OperationSpecification(
    operation_id="export",
    display_name="Export Dataset",
    description="Write a dataset to a portable file format.",
    inputs=[
        InputSpec(name="dataset", type="dataset", required=True, description="Dataset to export"),
    ],
    outputs=[],
    parameters=[
        ParameterSpec(name="filename", type="str", default="output.scp", description="Output file name"),
        ParameterSpec(name="format", type="str", default="scp", description="Export format"),
    ],
    side_effects=["file_write"],
    category="export",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, OperationSpecification] = {
    spec.operation_id: spec
    for spec in [
        _READ,
        _BASELINE,
        _SMOOTH,
        _PCA,
        _SCORE_PLOT,
        _LOADING_PLOT,
        _INTEGRATE,
        _PLOT,
        _NMF,
        _NMF_COMPONENTS_PLOT,
        _NMF_RECONSTRUCTION_PLOT,
        _MCRALS,
        _MCRALS_CONC_PLOT,
        _MCRALS_SPEC_PLOT,
        _INSPECT,
        _EXPORT,
    ]
}


class RegistryLookupError(KeyError):
    """Raised when an operation_id is not found in the registry."""

    pass


def get_spec(operation_id: str) -> OperationSpecification:
    """Lookup an OperationSpecification by operation_id.

    Raises:
        RegistryLookupError: if the operation_id is not registered.
    """
    try:
        return _REGISTRY[operation_id]
    except KeyError as exc:
        raise RegistryLookupError(
            f"Operation '{operation_id}' is not in the registry"
        ) from exc


def list_specs(category: str | None = None) -> list[OperationSpecification]:
    """List all registered specifications, optionally filtered by category."""
    specs = list(_REGISTRY.values())
    if category is not None:
        specs = [s for s in specs if s.category == category]
    return specs


def list_operation_ids() -> list[str]:
    """Return all registered operation IDs."""
    return list(_REGISTRY.keys())


def is_registered(operation_id: str) -> bool:
    """Return True if the operation_id is registered."""
    return operation_id in _REGISTRY
