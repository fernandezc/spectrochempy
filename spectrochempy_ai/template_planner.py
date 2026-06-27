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

from spectrochempy_ai.operation_registry import get_spec, is_registered
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
        """Register a template. Raises if any operation_id is unknown."""
        for step in template.steps:
            if not is_registered(step.operation_id):
                raise TemplateOperationError(
                    f"Template '{template.template_id}' references unknown "
                    f"operation '{step.operation_id}'"
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

    def create_plan(
        self,
        template_id: str,
        parameter_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> WorkflowPlan:
        """Instantiate a template into a complete WorkflowPlan.

        Args:
            template_id: The template to instantiate.
            parameter_overrides: Per-step parameter overrides keyed by step_id,
                e.g. {"s3": {"n_components": 5}}.

        Returns:
            A complete WorkflowPlan ready for validation and rendering.
        """
        template = self.get_template(template_id)
        overrides = parameter_overrides or {}

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
            description="Baseline-correct, decompose with PCA, and visualise scores and loadings.",
            scientific_context=ScientificContext(
                goal="Perform an exploratory Principal Component Analysis (PCA) "
                "on a synthetic NDDataset to identify the main variance directions.",
                analytical_strategy=(
                    "Baseline-correct the data, then apply PCA and "
                    "visualise scores and loadings."
                ),
                data_assumptions=[
                    "Dataset is a 2D NDDataset with observation x variable dimensions",
                    "Data contains meaningful variance after baseline removal",
                ],
                validation_criteria=[
                    "PCA converges without error",
                    "Score and loading plots render successfully",
                ],
                expected_outputs=[
                    "Baseline-corrected dataset",
                    "PCA score plot",
                    "PCA loading plot",
                ],
                limitations=[
                    "Does not perform cross-validation or determine optimal "
                    "number of components",
                    "Assumes linear structure; nonlinear patterns will not be captured",
                ],
            ),
            inputs=[
                InputReference(
                    name="dataset",
                    type="dataset",
                    source="synthetic",
                    summary="Synthetic 2D NDDataset for PCA exploration",
                ),
            ],
            steps=[
                TemplateStep(
                    step_id="s1",
                    operation_id="read",
                    display_label="Generate synthetic dataset",
                    rationale="Create a reproducible 2D test dataset for PCA exploration.",
                    input_refs=[],
                    parameters={"shape": [50, 100], "random_seed": 42},
                    output_var="dataset",
                ),
                TemplateStep(
                    step_id="s2",
                    operation_id="baseline",
                    display_label="Baseline correction",
                    rationale="Remove baseline drift before variance analysis.",
                    input_refs=["dataset"],
                    parameters={"method": "asls"},
                    output_var="dataset_corrected",
                ),
                TemplateStep(
                    step_id="s3",
                    operation_id="pca",
                    display_label="Principal Component Analysis",
                    rationale="Decompose variance into principal components.",
                    input_refs=["dataset_corrected"],
                    parameters={"n_components": 3},
                    output_var="pca_result",
                ),
                TemplateStep(
                    step_id="s4",
                    operation_id="score_plot",
                    display_label="PCA score plot",
                    rationale="Visualise sample distribution in PC space.",
                    input_refs=["pca_result"],
                    parameters={},
                    output_var="",
                ),
                TemplateStep(
                    step_id="s5",
                    operation_id="loading_plot",
                    display_label="PCA loading plot",
                    rationale="Visualise variable contributions to each "
                    "principal component.",
                    input_refs=["pca_result"],
                    parameters={},
                    output_var="",
                ),
            ],
            outputs=[
                OutputReference(
                    name="dataset_corrected",
                    type="dataset",
                    description="Baseline-corrected dataset ready for analysis",
                ),
                OutputReference(
                    name="pca_result",
                    type="dataset",
                    description="PCA result object containing scores, loadings, "
                    "and explained variance",
                ),
            ],
            reproducibility=ReproducibilityMetadata(
                package_versions={"spectrochempy": "0.9.4", "numpy": "1.26.4"},
                random_seeds={"dataset_generation": 42},
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
