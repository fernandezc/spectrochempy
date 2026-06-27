"""Tests for Phase 2 TemplatePlanner.

Every generated plan must:
- be a valid WorkflowPlan (passes validator)
- render deterministically (passes renderer checks)
- use the expected step structure
"""

from __future__ import annotations

import pytest

from spectrochempy_ai.notebook_renderer import render
from spectrochempy_ai.template_planner import (
    TemplateNotFoundError,
    TemplatePlanner,
    UnknownParameterError,
    UnresolvedInputError,
)
from spectrochempy_ai.validator import validate


@pytest.fixture
def planner() -> TemplatePlanner:
    return TemplatePlanner()


# ---------------------------------------------------------------------------
# Template discovery
# ---------------------------------------------------------------------------


class TestTemplateDiscovery:
    def test_lists_default_templates(self, planner: TemplatePlanner) -> None:
        ids = planner.list_templates()
        assert "exploratory_pca" in ids
        assert "baseline_integrate" in ids
        assert "nmf_exploration" in ids

    def test_get_template_exists(self, planner: TemplatePlanner) -> None:
        template = planner.get_template("exploratory_pca")
        assert template.template_id == "exploratory_pca"
        assert len(template.steps) == 5

    def test_get_template_unknown_raises(self, planner: TemplatePlanner) -> None:
        with pytest.raises(TemplateNotFoundError):
            planner.get_template("nonexistent")

    def test_get_template_baseline_integrate(self, planner: TemplatePlanner) -> None:
        template = planner.get_template("baseline_integrate")
        assert len(template.steps) == 4
        step_ids = [s.step_id for s in template.steps]
        assert step_ids == ["s1", "s2", "s3", "s4"]

    def test_get_template_nmf_exploration(self, planner: TemplatePlanner) -> None:
        template = planner.get_template("nmf_exploration")
        assert len(template.steps) == 4
        op_ids = [s.operation_id for s in template.steps]
        assert op_ids == ["read", "nmf", "nmf_components_plot", "nmf_reconstruction_plot"]


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------


class TestPlanGeneration:
    def test_create_plan_exploratory_pca(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("exploratory_pca")
        assert plan.schema_version == "0.1.0"
        assert plan.planner_id == "TemplatePlanner"
        assert plan.planner_config == {"template_id": "exploratory_pca"}
        assert len(plan.steps) == 5
        assert plan.timestamp != ""

    def test_create_plan_baseline_integrate(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("baseline_integrate")
        assert len(plan.steps) == 4
        assert plan.steps[2].operation_id == "integrate"

    def test_create_plan_nmf_exploration(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("nmf_exploration")
        assert len(plan.steps) == 4
        assert plan.steps[1].operation_id == "nmf"

    def test_unknown_template_raises(self, planner: TemplatePlanner) -> None:
        with pytest.raises(TemplateNotFoundError):
            planner.create_plan("nonexistent")

    def test_plan_has_scientific_context(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("exploratory_pca")
        ctx = plan.scientific_context
        assert "PCA" in ctx.goal
        assert ctx.analytical_strategy != ""
        assert len(ctx.data_assumptions) > 0
        assert len(ctx.validation_criteria) > 0
        assert len(ctx.expected_outputs) > 0
        assert len(ctx.limitations) > 0

    def test_plan_has_reproducibility(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("nmf_exploration")
        assert "spectrochempy" in plan.reproducibility.package_versions
        assert "dataset_generation" in plan.reproducibility.random_seeds

    def test_plan_baseline_integrate_has_expected_steps(
        self, planner: TemplatePlanner
    ) -> None:
        plan = planner.create_plan("baseline_integrate")
        step_ids = [s.step_id for s in plan.steps]
        assert step_ids == ["s1", "s2", "s3", "s4"]
        assert plan.steps[0].operation_id == "read"
        assert plan.steps[1].operation_id == "baseline"
        assert plan.steps[2].operation_id == "integrate"
        assert plan.steps[3].operation_id == "plot"


# ---------------------------------------------------------------------------
# Parameter overrides
# ---------------------------------------------------------------------------


class TestParameterOverrides:
    def test_override_n_components(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan(
            "exploratory_pca", parameter_overrides={"s3": {"n_components": 5}}
        )
        pca_step = plan.steps[2]
        assert pca_step.parameters["n_components"] == 5

    def test_override_unknown_param_raises(self, planner: TemplatePlanner) -> None:
        with pytest.raises(UnknownParameterError):
            planner.create_plan(
                "exploratory_pca",
                parameter_overrides={"s3": {"nonexistent_param": 42}},
            )

    def test_override_baseline_method(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan(
            "exploratory_pca", parameter_overrides={"s2": {"method": "detrend"}}
        )
        assert plan.steps[1].parameters["method"] == "detrend"

    def test_override_read_shape(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan(
            "exploratory_pca", parameter_overrides={"s1": {"shape": [20, 50]}}
        )
        assert plan.steps[0].parameters["shape"] == [20, 50]

    def test_override_nmf_n_components(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan(
            "nmf_exploration", parameter_overrides={"s2": {"n_components": 5}}
        )
        assert plan.steps[1].parameters["n_components"] == 5

    def test_override_multiple_steps(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan(
            "exploratory_pca",
            parameter_overrides={
                "s1": {"shape": [10, 30], "random_seed": 7},
                "s3": {"n_components": 2},
            },
        )
        assert plan.steps[0].parameters["shape"] == [10, 30]
        assert plan.steps[0].parameters["random_seed"] == 7
        assert plan.steps[2].parameters["n_components"] == 2


# ---------------------------------------------------------------------------
# Validation and rendering
# ---------------------------------------------------------------------------


class TestPlanValidation:
    def test_exploratory_pca_validates(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("exploratory_pca")
        validate(plan)  # should not raise

    def test_baseline_integrate_validates(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("baseline_integrate")
        validate(plan)  # should not raise

    def test_nmf_exploration_validates(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("nmf_exploration")
        validate(plan)  # should not raise

    def test_overridden_plan_validates(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan(
            "exploratory_pca",
            parameter_overrides={
                "s1": {"shape": [10, 20], "random_seed": 1},
                "s3": {"n_components": 2},
            },
        )
        validate(plan)  # should not raise


class TestPlanRendering:
    def test_exploratory_pca_renders(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("exploratory_pca")
        notebook = render(plan)
        assert notebook.nbformat >= 4
        assert len(notebook.cells) > 0

    def test_baseline_integrate_renders(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("baseline_integrate")
        notebook = render(plan)
        assert len(notebook.cells) > 0
        sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        assert any("integrate" in s for s in sources)

    def test_nmf_exploration_renders(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("nmf_exploration")
        notebook = render(plan)
        assert len(notebook.cells) > 0
        sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        assert any("NMF" in s for s in sources)

    def test_rendered_plan_has_title_cell(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("exploratory_pca")
        notebook = render(plan)
        first = notebook.cells[0]
        assert first.cell_type == "markdown"
        assert "PCA" in first.source

    def test_rendered_plan_deterministic(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("exploratory_pca")
        nb1 = render(plan)
        nb2 = render(plan)
        assert len(nb1.cells) == len(nb2.cells)
        for c1, c2 in zip(nb1.cells, nb2.cells):
            assert c1.cell_type == c2.cell_type
            assert c1.source == c2.source

    def test_rendered_plan_has_import_cell(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("exploratory_pca")
        notebook = render(plan)
        sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        assert any("import spectrochempy" in s for s in sources)

    def test_rendered_plan_has_manifest(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan("exploratory_pca")
        notebook = render(plan)
        manifest = notebook.metadata.get("spectrochempy_workflow_assistant", {})
        assert manifest.get("planner_id") == "TemplatePlanner"

    def test_rendered_plan_overrides_preserved(self, planner: TemplatePlanner) -> None:
        plan = planner.create_plan(
            "exploratory_pca",
            parameter_overrides={"s3": {"n_components": 6}},
        )
        notebook = render(plan)
        sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        # The PCA code cell should mention n_components=6
        pca_code = [s for s in sources if "n_components" in s]
        assert len(pca_code) > 0
        assert "n_components=6" in pca_code[0]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestPlannerErrors:
    def test_unregistered_operation_in_template_raises(self, planner: TemplatePlanner) -> None:
        from spectrochempy_ai.template_planner import (
            TemplateOperationError,
            TemplateStep,
            WorkflowTemplate,
        )
        from spectrochempy_ai.workflow_plan import ScientificContext

        bad_template = WorkflowTemplate(
            template_id="bad",
            description="References unknown operation.",
            scientific_context=ScientificContext(
                goal="test", analytical_strategy="test"
            ),
            steps=[
                TemplateStep(
                    step_id="s1",
                    operation_id="nonexistent_op",
                    display_label="Bad step",
                    rationale="",
                    input_refs=[],
                    parameters={},
                    output_var="",
                ),
            ],
        )
        with pytest.raises(TemplateOperationError):
            planner.register_template(bad_template)

    def test_create_plan_with_unknown_template_id(
        self, planner: TemplatePlanner
    ) -> None:
        with pytest.raises(TemplateNotFoundError):
            planner.create_plan("does_not_exist")
