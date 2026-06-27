"""Deterministic TemplatePlanner for Phase 2.

Templates are predefined step sequences. The planner instantiates them into
WorkflowPlan instances using OperationSpecifications from the registry.

No AI. No LLM. No prompts. Only deterministic template instantiation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from spectrochempy_ai.operation_registry import REGISTRY_VERSION, get_spec, is_registered
from spectrochempy_ai.workflow_plan import (
    InputReference,
    OperationStep,
    OutputReference,
    ReproducibilityMetadata,
    ScientificContext,
    WorkflowPlan,
)


class TemplateNotFoundError(KeyError):
    """Raised when a template_id is not found."""


class TemplateOperationError(ValueError):
    """Raised when a template references an unregistered operation."""


class UnknownParameterError(ValueError):
    """Raised when a parameter override does not match the spec."""


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
    """A named, reusable workflow template.

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
    """Creates WorkflowPlan instances from predefined templates.

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
        """Register a template.

        Validates at registration time:
        - Every operation_id is registered
        - Every template parameter name matches the operation spec

        Raises:
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
        """Resolve operation_id-based overrides to step_id-based overrides.

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
        """Instantiate a template into a complete WorkflowPlan.

        Args:
            template_id: The template to instantiate.
            parameter_overrides: Per-step parameter overrides keyed by step_id,
                e.g. {"s3": {"n_components": 5}}.
            operation_overrides: Per-operation parameter overrides keyed by
                operation_id, e.g. {"pca": {"n_components": 5}}.
                Resolved to step_id-based overrides internally.
                If both parameter_overrides and operation_overrides target the
                same step, parameter_overrides takes precedence.

        Returns:
            A complete WorkflowPlan ready for validation and rendering.
        """
        template = self.get_template(template_id)
        overrides = dict(parameter_overrides or {})
        if operation_overrides:
            resolved = self._resolve_operation_overrides(
                template, operation_overrides
            )
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

        plan = WorkflowPlan(
            schema_version="0.1.0",
            spectrochempy_version="0.9.4",
            plugin_version="0.1.0",
            planner_id="TemplatePlanner",
            planner_config={"template_id": template_id},
            scientific_context=template.scientific_context,
            inputs=template.inputs,
            steps=steps,
            outputs=template.outputs,
            reproducibility=template.reproducibility,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        return plan

    # ------------------------------------------------------------------
    # Built-in templates
    # ------------------------------------------------------------------

    def _register_default_templates(self) -> None:
        self._register_exploratory_pca()
        self._register_baseline_integrate()
        self._register_nmf_exploration()

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
