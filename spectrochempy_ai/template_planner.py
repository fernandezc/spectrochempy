"""
Deterministic TemplatePlanner for Phase 2.

Templates are predefined step sequences. The planner instantiates them into
WorkflowPlan instances using OperationSpecifications from the registry.

No AI. No LLM. No prompts. Only deterministic template instantiation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

from spectrochempy_ai.operation_registry import REGISTRY_VERSION
from spectrochempy_ai.operation_registry import get_spec
from spectrochempy_ai.operation_registry import is_registered
from spectrochempy_ai.workflow_plan import InputReference
from spectrochempy_ai.workflow_plan import OperationStep
from spectrochempy_ai.workflow_plan import OutputReference
from spectrochempy_ai.workflow_plan import ReproducibilityMetadata
from spectrochempy_ai.workflow_plan import ScientificContext
from spectrochempy_ai.workflow_plan import WorkflowPlan

_PACKAGES = ("spectrochempy", "numpy")


def _detect_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


class TemplateNotFoundError(KeyError):
    """Raised when a template_id is not found."""


class TemplateOperationError(ValueError):
    """Raised when a template references an unregistered operation."""


class UnknownParameterError(ValueError):
    """Raised when a parameter override does not match the spec."""


class UnknownOverrideTarget(ValueError):
    """Raised when an override references an operation_id or step_id that
    does not exist in the template."""


class UnresolvedInputError(ValueError):
    """Raised when a step references an unavailable variable."""


@dataclass
class TemplateStep:
    """A single step definition within a workflow template."""

    step_id: str
    operation_id: str
    display_label: str
    rationale: str
    input_refs: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    output_var: str = ""


@dataclass
class WorkflowTemplate:
    """
    A named, reusable workflow template.

    Templates define:
    - Scientific context (goal, strategy, assumptions, etc.)
    - An ordered list of steps with wiring
    - Input/output references
    - Reproducibility metadata

    Version fields enable independent versioning and compatibility checking.
    """

    template_id: str
    description: str
    scientific_context: ScientificContext
    steps: list[TemplateStep] = field(default_factory=list)
    inputs: list[InputReference] = field(default_factory=list)
    outputs: list[OutputReference] = field(default_factory=list)
    reproducibility: ReproducibilityMetadata = field(
        default_factory=ReproducibilityMetadata
    )
    template_version: str = "0.1.0"
    compatible_registry_version: str = REGISTRY_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-compatible)."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowTemplate:
        """Deserialize from a plain dict."""
        sci_ctx = ScientificContext(**data["scientific_context"])
        steps = [TemplateStep(**s) for s in data.get("steps", [])]
        inputs = [InputReference(**i) for i in data.get("inputs", [])]
        outputs = [OutputReference(**o) for o in data.get("outputs", [])]
        repro = ReproducibilityMetadata(**data.get("reproducibility", {}))
        return cls(
            template_id=data["template_id"],
            description=data["description"],
            scientific_context=sci_ctx,
            steps=steps,
            inputs=inputs,
            outputs=outputs,
            reproducibility=repro,
            template_version=data.get("template_version", "0.1.0"),
            compatible_registry_version=data.get(
                "compatible_registry_version", REGISTRY_VERSION
            ),
        )


class TemplatePlanner:
    """
    Creates WorkflowPlan instances from predefined templates.

    The planner:
    - Looks up templates by template_id
    - Fills parameter defaults from OperationSpecifications
    - Applies user-provided parameter overrides
    - Returns a complete, ready-to-validate WorkflowPlan
    """

    def __init__(self) -> None:
        self._templates: dict[str, WorkflowTemplate] = {}
        self._register_default_templates()

    def register_template(self, template: WorkflowTemplate) -> None:
        """
        Register a template.

        Validates at registration time:
        - Every operation_id is registered
        - Every template parameter name matches the operation spec

        Raises
        ------
            TemplateOperationError: if an operation_id is unknown.
            UnknownParameterError: if a parameter name does not match the spec.
        """
        for step in template.steps:
            if not is_registered(step.operation_id):
                raise TemplateOperationError(
                    f"Template '{template.template_id}' references unknown "
                    f"operation '{step.operation_id}'"
                )
            spec = get_spec(step.operation_id)
            spec_param_names = {p.name for p in spec.parameters}
            for key in step.parameters:
                if key not in spec_param_names:
                    raise UnknownParameterError(
                        f"Template '{template.template_id}' step "
                        f"'{step.step_id}' has unknown parameter "
                        f"'{key}' for operation '{step.operation_id}'"
                    )
        self._templates[template.template_id] = template

    def list_templates(self) -> list[str]:
        """Return all registered template IDs."""
        return list(self._templates.keys())

    def get_template(self, template_id: str) -> WorkflowTemplate:
        """Look up a template by ID. Raises TemplateNotFoundError."""
        try:
            return self._templates[template_id]
        except KeyError as exc:
            raise TemplateNotFoundError(
                f"Template '{template_id}' not found. "
                f"Available: {list(self._templates.keys())}"
            ) from exc

    @staticmethod
    def _resolve_operation_overrides(
        template: WorkflowTemplate,
        operation_overrides: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """
        Resolve operation_id-based overrides to step_id-based overrides.

        If multiple steps share the same operation_id (e.g. two plot steps),
        the override is applied to each matching step.
        """
        resolved: dict[str, dict[str, Any]] = {}
        for step in template.steps:
            if step.operation_id in operation_overrides:
                merged = resolved.get(step.step_id, {})
                merged.update(operation_overrides[step.operation_id])
                resolved[step.step_id] = merged
        return resolved

    def create_plan(
        self,
        template_id: str,
        parameter_overrides: dict[str, dict[str, Any]] | None = None,
        operation_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> WorkflowPlan:
        """
        Instantiate a template into a complete WorkflowPlan.

        Args:
        ----
            template_id: The template to instantiate.
            parameter_overrides: Per-step parameter overrides keyed by step_id,
                e.g. {"s3": {"n_components": 5}}.
            operation_overrides: Per-operation parameter overrides keyed by
                operation_id, e.g. {"pca": {"n_components": 5}}.
                Resolved to step_id-based overrides internally.
                If both parameter_overrides and operation_overrides target the
                same step, parameter_overrides takes precedence.

        Returns:
        -------
            A complete WorkflowPlan ready for validation and rendering.
        """
        template = self.get_template(template_id)
        valid_ops = {step.operation_id for step in template.steps}
        valid_ids = {step.step_id for step in template.steps}

        if operation_overrides:
            unknown = set(operation_overrides) - valid_ops
            if unknown:
                raise UnknownOverrideTarget(
                    f"operation_overrides target unknown operation(s): "
                    f"{', '.join(sorted(unknown))}. "
                    f"Valid operations for '{template_id}': "
                    f"{', '.join(sorted(valid_ops))}."
                )

        if parameter_overrides:
            unknown = set(parameter_overrides) - valid_ids
            if unknown:
                raise UnknownOverrideTarget(
                    f"parameter_overrides target unknown step(s): "
                    f"{', '.join(sorted(unknown))}. "
                    f"Valid step_ids for '{template_id}': "
                    f"{', '.join(sorted(valid_ids))}."
                )

        overrides = dict(parameter_overrides or {})
        if operation_overrides:
            resolved = self._resolve_operation_overrides(template, operation_overrides)
            # operation_overrides fill in, parameter_overrides take precedence
            for step_id, params in resolved.items():
                if step_id not in overrides:
                    overrides[step_id] = {}
                for key, value in params.items():
                    if key not in overrides[step_id]:
                        overrides[step_id][key] = value

        steps: list[OperationStep] = []
        available_vars = {inp.name for inp in template.inputs}

        for tstep in template.steps:
            spec = get_spec(tstep.operation_id)

            # Merge: spec defaults -> template defaults -> user overrides
            merged_params: dict[str, Any] = {}
            for p in spec.parameters:
                if p.default is not None:
                    merged_params[p.name] = deepcopy(p.default)
            merged_params.update(tstep.parameters)
            if tstep.step_id in overrides:
                for key, value in overrides[tstep.step_id].items():
                    if key not in merged_params and key not in {
                        p.name for p in spec.parameters
                    }:
                        raise UnknownParameterError(
                            f"Parameter '{key}' is not valid for operation "
                            f"'{tstep.operation_id}' (step '{tstep.step_id}')"
                        )
                    merged_params[key] = value

            # Resolve input refs
            input_refs: list[str] = []
            for ref in tstep.input_refs:
                if ref not in available_vars:
                    raise UnresolvedInputError(
                        f"Step '{tstep.step_id}' ({tstep.operation_id}) "
                        f"references unavailable variable '{ref}'"
                    )
                input_refs.append(ref)

            step = OperationStep(
                step_id=tstep.step_id,
                operation_id=tstep.operation_id,
                display_label=tstep.display_label,
                rationale=tstep.rationale,
                input_refs=input_refs,
                parameters=merged_params,
                output_var=tstep.output_var,
            )
            steps.append(step)
            if tstep.output_var:
                available_vars.add(tstep.output_var)

        versions = _detect_package_versions()
        repro = deepcopy(template.reproducibility)
        repro.package_versions = versions
        plan = WorkflowPlan(
            schema_version="0.1.0",
            spectrochempy_version=versions.get("spectrochempy", "unknown"),
            plugin_version="0.1.0",  # TODO: detect dynamically once this is a proper plugin
            planner_id="TemplatePlanner",
            planner_config={"template_id": template_id},
            scientific_context=template.scientific_context,
            inputs=template.inputs,
            steps=steps,
            outputs=template.outputs,
            reproducibility=repro,
            timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        return plan

    # ------------------------------------------------------------------
    # Built-in templates
    # ------------------------------------------------------------------

    def _register_default_templates(self) -> None:
        self._register_exploratory_pca()
        self._register_baseline_integrate()
        self._register_nmf_exploration()
        self._register_mcrals_analysis()
        self._register_pls_calibration()

    def _register_exploratory_pca(self) -> None:
        template = WorkflowTemplate(
            template_id="exploratory_pca",
            description=(
                "Exploratory PCA: load spectral data, correct baseline, "
                "decompose variance, and diagnose via scree, score, and "
                "loading plots."
            ),
            scientific_context=ScientificContext(
                goal=(
                    "Identify the dominant variance directions in a spectral "
                    "dataset. PCA after baseline correction is the standard "
                    "first step in exploratory analysis of multivariate "
                    "spectral data."
                ),
                analytical_strategy=(
                    "Load real spectral data from file, inspect its "
                    "structure, correct additive baseline drift (which would "
                    "otherwise inflate early PC variance and mask chemical "
                    "information), decompose with PCA, inspect explained "
                    "variance via a scree plot, then visualise sample "
                    "distributions (score plot) and variable contributions "
                    "(loading plot)."
                ),
                data_assumptions=[
                    "Data are 2D (observation × spectral variable) with a "
                    "continuous spectral axis",
                    "Baseline varies slowly relative to spectral features",
                    "Variance is dominated by a small number of latent factors",
                    "Linear combinations of original variables adequately "
                    "capture the structure of interest",
                ],
                validation_criteria=[
                    "PCA converges without numerical error",
                    "Scree plot shows a decreasing variance profile (if not, "
                    "check data quality or preprocessing)",
                    "Score and loading plots render successfully",
                ],
                expected_outputs=[
                    "Baseline-corrected dataset",
                    "PCA model with scores, loadings, and explained variance",
                    "Scree plot (bar + cumulative variance) for component "
                    "selection decisions",
                    "Score plot showing sample distribution in PC1–PC2 space, "
                    "useful for clustering and outlier detection",
                    "Loading plot identifying spectral variables that "
                    "contribute most to each PC",
                ],
                limitations=[
                    "PCA is a linear method; non-linear chemical or physical "
                    "effects (e.g., peak shifts, saturation) will not be "
                    "captured and may distort the PC space",
                    "The number of components is set to 5 by default; users "
                    "should inspect the scree plot and adjust",
                    "Outliers can dominate variance; inspect score plots "
                    "before interpreting loadings",
                    "Spectral alignment is assumed; peak shifts across "
                    "observations require warping before PCA",
                    "PCA components are mathematical axes, not necessarily "
                    "chemically pure components (see NMF or MCR-ALS for "
                    "mixture resolution)",
                ],
            ),
            inputs=[
                InputReference(
                    name="dataset",
                    type="dataset",
                    source="external",
                    summary=(
                        "Real spectral dataset loaded via the `load` step. "
                        "Alternatively, an in-memory dataset may be passed "
                        "directly as an input."
                    ),
                ),
            ],
            steps=[
                TemplateStep(
                    step_id="s1",
                    operation_id="load",
                    display_label="Load spectral dataset",
                    rationale=(
                        "Load real spectral data from a portable file "
                        "format. Starting from external data (rather than "
                        "synthetic generation) ensures the template "
                        "represents a genuine scientific workflow."
                    ),
                    input_refs=[],
                    parameters={"filename": "data.scp", "format": "scp"},
                    output_var="dataset",
                ),
                TemplateStep(
                    step_id="s2",
                    operation_id="inspect",
                    display_label="Inspect dataset quality",
                    rationale=(
                        "Validate dataset shape, coordinate ranges, and "
                        "value distribution before processing. Early "
                        "detection of anomalies (NaN, negative values, "
                        "irregular sampling) prevents misleading results."
                    ),
                    input_refs=["dataset"],
                    parameters={},
                    output_var="",
                ),
                TemplateStep(
                    step_id="s3",
                    operation_id="baseline",
                    display_label="Baseline correction",
                    rationale=(
                        "Remove additive baseline drift. Baseline artifacts "
                        "inflate variance in early principal components and "
                        "can mask genuine chemical variation. The Asymmetric "
                        "Least Squares (asls) method is widely used for "
                        "vibrational spectra because it adapts to smoothly "
                        "varying baselines without requiring user-specified "
                        "anchor points."
                    ),
                    input_refs=["dataset"],
                    parameters={"method": "asls"},
                    output_var="dataset_corrected",
                ),
                TemplateStep(
                    step_id="s4",
                    operation_id="pca",
                    display_label="Principal Component Analysis",
                    rationale=(
                        "Decompose the baseline-corrected dataset into "
                        "orthogonal principal components ranked by explained "
                        "variance. PCA transforms the original spectral "
                        "variables into a reduced set of uncorrelated "
                        "latent variables (PCs) that capture the dominant "
                        "patterns in the data. The first few PCs typically "
                        "capture chemical information while later PCs "
                        "capture noise."
                    ),
                    input_refs=["dataset_corrected"],
                    parameters={"n_components": 5},
                    output_var="pca_result",
                ),
                TemplateStep(
                    step_id="s5",
                    operation_id="scree_plot",
                    display_label="Variance explained (scree plot)",
                    rationale=(
                        "Visualise the variance explained by each PC as a "
                        "bar plot with a cumulative variance overlay. The "
                        "scree plot is the standard diagnostic for choosing "
                        "the number of components to retain: look for the "
                        "'elbow' where additional components contribute "
                        "little additional variance, and consider components "
                        "needed to reach a target cumulative variance "
                        "(typically 80–95%)."
                    ),
                    input_refs=["pca_result"],
                    parameters={},
                    output_var="",
                ),
                TemplateStep(
                    step_id="s6",
                    operation_id="score_plot",
                    display_label="PCA score plot",
                    rationale=(
                        "Visualise each observation projected onto the "
                        "first two principal components. Score plots reveal "
                        "sample groupings, gradients, and outliers. Samples "
                        "that cluster together in PC space share similar "
                        "spectral characteristics; samples far from the "
                        "origin may be strong outliers or high-leverage "
                        "points requiring investigation."
                    ),
                    input_refs=["pca_result"],
                    parameters={},
                    output_var="",
                ),
                TemplateStep(
                    step_id="s7",
                    operation_id="loading_plot",
                    display_label="PCA loading plot",
                    rationale=(
                        "Visualise the contribution (weight) of each "
                        "spectral variable to the first two principal "
                        "components. Loading plots identify which spectral "
                        "regions drive the observed sample separation: "
                        "variables with high absolute loading values "
                        "correspond to chemically relevant absorption "
                        "bands. Loadings can be interpreted as the "
                        "'spectral signature' captured by each PC."
                    ),
                    input_refs=["pca_result"],
                    parameters={},
                    output_var="",
                ),
            ],
            outputs=[
                OutputReference(
                    name="dataset_corrected",
                    type="dataset",
                    description=(
                        "Baseline-corrected dataset. Ready for further "
                        "analysis or export."
                    ),
                ),
                OutputReference(
                    name="pca_result",
                    type="dataset",
                    description=(
                        "PCA result object containing scores, loadings, "
                        "explained variance per component, and the fitted "
                        "PCA model."
                    ),
                ),
            ],
            reproducibility=ReproducibilityMetadata(
                package_versions={"spectrochempy": "0.9.4", "numpy": "1.26.4"},
                random_seeds={},
            ),
        )
        self._templates["exploratory_pca"] = template

    def _register_baseline_integrate(self) -> None:
        template = WorkflowTemplate(
            template_id="baseline_integrate",
            description="Correct baseline, integrate signal, and plot the area profile.",
            scientific_context=ScientificContext(
                goal=(
                    "Correct baseline drift, integrate signals, and "
                    "visualise the integrated result."
                ),
                analytical_strategy=(
                    "Baseline-correct with asls, then integrate along the "
                    "spectral axis and plot the area profile."
                ),
                data_assumptions=[
                    "2D NDDataset with observation x spectral dimensions",
                    "Signal area is meaningful after baseline removal",
                ],
                validation_criteria=[
                    "Integration returns a 1D area profile",
                    "Plot renders without error",
                ],
                expected_outputs=[
                    "Baseline-corrected dataset",
                    "Integrated area profile",
                    "Area profile plot",
                ],
                limitations=[
                    "Assumes uniform spectral spacing for trapezoidal integration",
                    "Does not account for masked regions",
                ],
            ),
            inputs=[
                InputReference(
                    name="dataset",
                    type="dataset",
                    source="synthetic",
                    summary="Synthetic 2D NDDataset for integration workflow",
                ),
            ],
            steps=[
                TemplateStep(
                    step_id="s1",
                    operation_id="read",
                    display_label="Generate synthetic dataset",
                    rationale="Create reproducible 2D data for baseline + integration.",
                    input_refs=[],
                    parameters={"shape": [30, 80], "random_seed": 123},
                    output_var="dataset",
                ),
                TemplateStep(
                    step_id="s2",
                    operation_id="baseline",
                    display_label="Baseline correction",
                    rationale="Remove baseline before integrating signal areas.",
                    input_refs=["dataset"],
                    parameters={"method": "asls"},
                    output_var="dataset_corrected",
                ),
                TemplateStep(
                    step_id="s3",
                    operation_id="integrate",
                    display_label="Trapezoidal integration",
                    rationale="Compute signal area for each observation.",
                    input_refs=["dataset_corrected"],
                    parameters={"method": "trapezoid"},
                    output_var="area_profile",
                ),
                TemplateStep(
                    step_id="s4",
                    operation_id="plot",
                    display_label="Plot integrated area profile",
                    rationale="Visualise the integrated area as a function of "
                    "sample index.",
                    input_refs=["area_profile"],
                    parameters={"plot_type": "line"},
                    output_var="",
                ),
            ],
            outputs=[
                OutputReference(
                    name="dataset_corrected",
                    type="dataset",
                    description="Baseline-corrected dataset",
                ),
                OutputReference(
                    name="area_profile",
                    type="dataset",
                    description="1D area profile after integration",
                ),
            ],
            reproducibility=ReproducibilityMetadata(
                package_versions={"spectrochempy": "0.9.4", "numpy": "1.26.4"},
                random_seeds={"dataset_generation": 123},
            ),
        )
        self._templates["baseline_integrate"] = template

    def _register_nmf_exploration(self) -> None:
        template = WorkflowTemplate(
            template_id="nmf_exploration",
            description="Decompose a non-negative mixture using NMF and "
            "visualise components and reconstruction.",
            scientific_context=ScientificContext(
                goal=(
                    "Decompose a synthetic mixture into non-negative "
                    "components using NMF."
                ),
                analytical_strategy=(
                    "Generate a mixed dataset, apply NMF, and visualise "
                    "components and reconstruction."
                ),
                data_assumptions=[
                    "Data are non-negative and approximately additive",
                    "Number of components is known a priori",
                ],
                validation_criteria=[
                    "NMF converges within max_iter",
                    "Reconstruction approximates input shape",
                ],
                expected_outputs=[
                    "NMF component matrix",
                    "Component plot",
                    "Reconstruction plot",
                ],
                limitations=[
                    "NMF is not unique; different initialisations yield "
                    "different solutions",
                    "Requires non-negative data",
                ],
            ),
            inputs=[
                InputReference(
                    name="dataset",
                    type="dataset",
                    source="synthetic",
                    summary="Synthetic non-negative 2D mixture for NMF",
                ),
            ],
            steps=[
                TemplateStep(
                    step_id="s1",
                    operation_id="read",
                    display_label="Generate non-negative synthetic mixture",
                    rationale="Create reproducible non-negative 2D data.",
                    input_refs=[],
                    parameters={
                        "shape": [50, 80],
                        "random_seed": 99,
                        "non_negative": True,
                    },
                    output_var="dataset",
                ),
                TemplateStep(
                    step_id="s2",
                    operation_id="nmf",
                    display_label="Non-negative Matrix Factorisation",
                    rationale="Decompose mixture into non-negative components "
                    "and loadings.",
                    input_refs=["dataset"],
                    parameters={"n_components": 3, "max_iter": 500},
                    output_var="nmf_result",
                ),
                TemplateStep(
                    step_id="s3",
                    operation_id="nmf_components_plot",
                    display_label="NMF components plot",
                    rationale="Visualise the extracted spectral components.",
                    input_refs=["nmf_result"],
                    parameters={},
                    output_var="",
                ),
                TemplateStep(
                    step_id="s4",
                    operation_id="nmf_reconstruction_plot",
                    display_label="NMF reconstruction plot",
                    rationale="Visualise the reconstructed dataset for quality control.",
                    input_refs=["nmf_result"],
                    parameters={},
                    output_var="",
                ),
            ],
            outputs=[
                OutputReference(
                    name="nmf_result",
                    type="dataset",
                    description="NMF result object with components and reconstruction",
                ),
            ],
            reproducibility=ReproducibilityMetadata(
                package_versions={"spectrochempy": "0.9.4", "numpy": "1.26.4"},
                random_seeds={"dataset_generation": 99},
            ),
        )
        self._templates["nmf_exploration"] = template

    def _register_mcrals_analysis(self) -> None:
        template = WorkflowTemplate(
            template_id="mcrals_analysis",
            description=(
                "MCR-ALS mixture resolution: load spectral mixture data, "
                "correct baseline, generate an initial concentration guess, "
                "and resolve pure concentration profiles and spectra using "
                "Multivariate Curve Resolution by Alternating Least Squares."
            ),
            scientific_context=ScientificContext(
                goal=(
                    "Resolve a multivariate spectral mixture into chemically "
                    "meaningful concentration profiles and pure component spectra "
                    "using MCR-ALS. Unlike PCA, MCR-ALS enforces physical constraints "
                    "(non-negativity, unimodality) and yields profiles that can be "
                    "interpreted as real chemical components."
                ),
                analytical_strategy=(
                    "Load spectral mixture data, inspect its structure, correct "
                    "additive baseline drift, generate a simple initial concentration "
                    "guess (spaced Gaussian profiles), and fit an MCR-ALS model. "
                    "Visualise the resolved concentration profiles and pure spectra "
                    "to assess whether the solution is chemically meaningful."
                ),
                data_assumptions=[
                    "Data are 2D (observation x spectral variable) with a "
                    "continuous spectral axis",
                    "The mixture is approximately additive: X ≈ C · S^T + E",
                    "Number of chemical components is known or can be estimated "
                    "a priori",
                    "Concentration profiles are non-negative and unimodal",
                    "Pure component spectra are non-negative",
                    "Baseline has been removed or is negligible",
                ],
                validation_criteria=[
                    "MCR-ALS converges within max_iter iterations",
                    "Resolved concentration profiles are non-negative and "
                    "physically plausible",
                    "Resolved spectra are non-negative and show expected "
                    "spectral features",
                    "Residuals are unstructured (no systematic patterns)",
                    "Reconstruction approximates the original dataset shape",
                ],
                expected_outputs=[
                    "Baseline-corrected dataset",
                    "Resolved concentration profiles (C matrix)",
                    "Resolved pure component spectra (S^T matrix)",
                    "MCR-ALS result object with diagnostics (iterations, "
                    "convergence, residual standard deviation)",
                    "Concentration profile plot",
                    "Pure spectra plot",
                ],
                limitations=[
                    "MCR-ALS requires a good initial guess; the simple Gaussian "
                    "profiles used here are a starting point and may not be "
                    "optimal for all datasets. For production work, use "
                    "chemically informed initial estimates or EFA/SIMPLISMA "
                    "initialization.",
                    "The solution is not unique; different initial guesses can "
                    "lead to different resolved profiles (rotational ambiguity).",
                    "The number of components must be specified correctly; "
                    "underestimation merges components, overestimation introduces "
                    "artefacts.",
                    "MCR-ALS is sensitive to baseline residuals; incomplete "
                    "baseline correction can distort concentration profiles.",
                    "No closure or kinetic constraints are applied in this "
                    "template; add them if the chemical system requires them.",
                    "Outliers or strongly varying baselines across observations "
                    "can prevent convergence.",
                ],
            ),
            inputs=[
                InputReference(
                    name="dataset",
                    type="dataset",
                    source="external",
                    summary=(
                        "Real spectral mixture dataset loaded via the `load` step. "
                        "Alternatively, synthetic data may be used for demonstration."
                    ),
                ),
            ],
            steps=[
                TemplateStep(
                    step_id="s1",
                    operation_id="load",
                    display_label="Load spectral mixture",
                    rationale=(
                        "Load the spectral mixture dataset from a portable file "
                        "format. MCR-ALS operates on the full mixture matrix, "
                        "so the dataset must contain all observations and all "
                        "spectral variables."
                    ),
                    input_refs=[],
                    parameters={"filename": "data.scp", "format": "scp"},
                    output_var="dataset",
                ),
                TemplateStep(
                    step_id="s2",
                    operation_id="inspect",
                    display_label="Inspect mixture quality",
                    rationale=(
                        "Validate dataset dimensions, coordinate ranges, and "
                        "check for anomalies (NaN, negative values) before "
                        "proceeding with mixture resolution."
                    ),
                    input_refs=["dataset"],
                    parameters={},
                    output_var="",
                ),
                TemplateStep(
                    step_id="s3",
                    operation_id="baseline",
                    display_label="Baseline correction",
                    rationale=(
                        "Remove additive baseline drift before MCR-ALS fitting. "
                        "Residual baseline can distort concentration profiles "
                        "and prevent physically meaningful solutions."
                    ),
                    input_refs=["dataset"],
                    parameters={"method": "asls"},
                    output_var="dataset_corrected",
                ),
                TemplateStep(
                    step_id="s4",
                    operation_id="mcrals_init",
                    display_label="Initial concentration guess",
                    rationale=(
                        "Generate a simple initial guess for concentration profiles. "
                        "MCR-ALS requires an initial estimate for either concentrations "
                        "or pure spectra. This template uses spaced Gaussian profiles "
                        "as a chemically plausible starting point. For real data, "
                        "replace with a chemically informed estimate from EFA, "
                        "SIMPLISMA, or domain knowledge."
                    ),
                    input_refs=["dataset_corrected"],
                    parameters={"n_components": 3},
                    output_var="conc_guess",
                ),
                TemplateStep(
                    step_id="s5",
                    operation_id="mcrals",
                    display_label="MCR-ALS mixture resolution",
                    rationale=(
                        "Fit the MCR-ALS model to resolve the mixture into "
                        "concentration profiles and pure component spectra. "
                        "MCR-ALS iteratively alternates between estimating C "
                        "(concentrations) and S^T (spectra) under non-negativity "
                        "and unimodality constraints until convergence."
                    ),
                    input_refs=["dataset_corrected", "conc_guess"],
                    parameters={"n_components": 3, "max_iter": 100},
                    output_var="mcrals_result",
                ),
                TemplateStep(
                    step_id="s6",
                    operation_id="mcrals_conc_plot",
                    display_label="Concentration profiles",
                    rationale=(
                        "Visualise the resolved concentration profiles for each "
                        "component. These profiles should be non-negative, unimodal, "
                        "and chemically interpretable. Check for unexpected "
                        "oscillations or negative values that indicate convergence "
                        "issues or an incorrect number of components."
                    ),
                    input_refs=["mcrals_result"],
                    parameters={},
                    output_var="",
                ),
                TemplateStep(
                    step_id="s7",
                    operation_id="mcrals_spec_plot",
                    display_label="Resolved pure spectra",
                    rationale=(
                        "Visualise the resolved pure component spectra. These "
                        "spectra should show expected spectral features (peaks, "
                        "bands) and be non-negative. Unphysical features may "
                        "indicate an incorrect number of components or poor "
                        "initial guess."
                    ),
                    input_refs=["mcrals_result"],
                    parameters={},
                    output_var="",
                ),
            ],
            outputs=[
                OutputReference(
                    name="dataset_corrected",
                    type="dataset",
                    description="Baseline-corrected mixture dataset",
                ),
                OutputReference(
                    name="mcrals_result",
                    type="result",
                    description=(
                        "MCR-ALS result object with resolved concentration "
                        "profiles (C), pure spectra (S^T), and convergence "
                        "diagnostics"
                    ),
                ),
            ],
            reproducibility=ReproducibilityMetadata(
                package_versions={},
                random_seeds={},
            ),
        )
        self._templates["mcrals_analysis"] = template

    def _register_pls_calibration(self) -> None:
        template = WorkflowTemplate(
            template_id="pls_calibration",
            description=(
                "PLS calibration: load spectral data and reference values, "
                "preprocess spectra, fit a Partial Least Squares regression "
                "model, and assess predictive performance."
            ),
            scientific_context=ScientificContext(
                goal=(
                    "Build a quantitative calibration model relating spectral "
                    "measurements to reference analytical values using Partial "
                    "Least Squares (PLS) regression. PLS handles collinear "
                    "spectral predictors by finding latent variables that "
                    "maximise covariance between spectra and the property of "
                    "interest."
                ),
                analytical_strategy=(
                    "Load spectral data (X) and reference values (y) from "
                    "separate sources, inspect both datasets for quality, "
                    "correct baseline drift in spectra, fit a PLS model with "
                    "a chosen number of components, display the model result "
                    "(coefficients, scores, loadings), and visualise "
                    "predictions against reference values."
                ),
                data_assumptions=[
                    "Spectra (X) are 2D (observations x spectral variables) "
                    "with a continuous spectral axis",
                    "Reference values (y) are 1D or 2D (observations x targets)",
                    "The relationship between spectra and reference is "
                    "approximately linear in the latent-variable space",
                    "The number of observations exceeds the number of PLS "
                    "components",
                    "Reference values span the calibration range of interest",
                ],
                validation_criteria=[
                    "PLS model converges without numerical error",
                    "R² (coefficient of determination) is reported",
                    "Predicted vs reference plot shows agreement",
                    "Number of components is justified (not overfitted)",
                ],
                expected_outputs=[
                    "Baseline-corrected spectral dataset",
                    "PLS model with coefficients, scores, loadings, and weights",
                    "Predicted values for the calibration set",
                    "R² score assessing model fit",
                ],
                limitations=[
                    "PLS assumes a linear relationship in latent-variable "
                    "space; strong non-linearities require non-linear methods "
                    "(SVM, neural networks, polynomial PLS)",
                    "The default n_components=3 is arbitrary; it should be "
                    "chosen via cross-validation (not implemented in this "
                    "template)",
                    "No outlier detection is performed; influential samples "
                    "can distort the model",
                    "No variable selection is applied; uninformative spectral "
                    "regions are included in the model",
                    "The template uses the calibration set for both fitting "
                    "and prediction; a proper validation requires an "
                    "independent test set",
                    "Cross-validation (CV) is not implemented in this "
                    "template; add it manually for robust component selection",
                ],
            ),
            inputs=[
                InputReference(
                    name="dataset",
                    type="dataset",
                    source="external",
                    summary="Spectral dataset (X matrix) for calibration",
                ),
                InputReference(
                    name="reference",
                    type="dataset",
                    source="external",
                    summary="Reference analytical values (y vector)",
                ),
            ],
            steps=[
                TemplateStep(
                    step_id="s1",
                    operation_id="load",
                    display_label="Load spectral data",
                    rationale=(
                        "Load the spectral dataset from a portable file. "
                        "This is the predictor matrix X for the PLS model."
                    ),
                    input_refs=[],
                    parameters={"filename": "spectra.scp", "format": "scp"},
                    output_var="dataset",
                ),
                TemplateStep(
                    step_id="s2",
                    operation_id="load",
                    display_label="Load reference values",
                    rationale=(
                        "Load the reference analytical values from a separate "
                        "file. These are the response values y that the PLS "
                        "model will predict from spectra."
                    ),
                    input_refs=[],
                    parameters={"filename": "reference.csv", "format": "csv"},
                    output_var="reference",
                ),
                TemplateStep(
                    step_id="s3",
                    operation_id="inspect",
                    display_label="Inspect spectral data",
                    rationale=(
                        "Validate spectral dataset dimensions and check for "
                        "anomalies before model fitting."
                    ),
                    input_refs=["dataset"],
                    parameters={},
                    output_var="",
                ),
                TemplateStep(
                    step_id="s4",
                    operation_id="inspect",
                    display_label="Inspect reference values",
                    rationale=(
                        "Validate reference values: check range, missing data, "
                        "and consistency with the spectral dataset."
                    ),
                    input_refs=["reference"],
                    parameters={},
                    output_var="",
                ),
                TemplateStep(
                    step_id="s5",
                    operation_id="baseline",
                    display_label="Baseline correction",
                    rationale=(
                        "Remove additive baseline drift from spectra before "
                        "PLS fitting. Baseline artefacts add uninformative "
                        "variance that can degrade model performance."
                    ),
                    input_refs=["dataset"],
                    parameters={"method": "asls"},
                    output_var="dataset_corrected",
                ),
                TemplateStep(
                    step_id="s6",
                    operation_id="pls",
                    display_label="PLS regression",
                    rationale=(
                        "Fit a Partial Least Squares regression model. PLS "
                        "finds latent variables (components) that maximise "
                        "covariance between the spectral predictors and the "
                        "reference values, producing a calibration model that "
                        "is robust to collinearity in the spectral data."
                    ),
                    input_refs=["dataset_corrected", "reference"],
                    parameters={"n_components": 3},
                    output_var="pls_result",
                ),
                TemplateStep(
                    step_id="s7",
                    operation_id="pls_predict_plot",
                    display_label="Predicted values",
                    rationale=(
                        "Generate predictions from the fitted PLS model and "
                        "display them. Compare predicted values to reference "
                        "values to assess calibration quality."
                    ),
                    input_refs=["pls_result", "dataset_corrected"],
                    parameters={},
                    output_var="",
                ),
            ],
            outputs=[
                OutputReference(
                    name="dataset_corrected",
                    type="dataset",
                    description="Baseline-corrected spectral dataset",
                ),
                OutputReference(
                    name="pls_result",
                    type="result",
                    description=(
                        "PLS model with coefficients, scores, loadings, "
                        "weights, and intercept"
                    ),
                ),
            ],
            reproducibility=ReproducibilityMetadata(
                package_versions={},
                random_seeds={},
            ),
        )
        self._templates["pls_calibration"] = template
