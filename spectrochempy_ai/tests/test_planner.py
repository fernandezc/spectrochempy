"""Tests for Phase 3 WorkflowTemplate consolidation.

Every generated plan must:
- be a valid WorkflowPlan (passes validator)
- render deterministically (passes renderer checks)
- use the expected step structure

Template metadata:
- carries template_version and compatible_registry_version
- serializes to dict and back
- catches bad parameter names at registration time
"""

from __future__ import annotations

import pytest

from spectrochempy_ai.notebook_renderer import render
from spectrochempy_ai.operation_registry import REGISTRY_VERSION
from spectrochempy_ai.template_planner import (
    TemplateNotFoundError,
    TemplateOperationError,
    TemplatePlanner,
    UnknownParameterError,
    UnresolvedInputError,
)
from spectrochempy_ai.validator import validate


TEMPLATE_IDS = ["exploratory_pca", "baseline_integrate", "nmf_exploration"]


@pytest.fixture(params=TEMPLATE_IDS)
def template_id(request: pytest.FixtureRequest) -> str:
    return request.param


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

    def test_bad_parameter_name_at_registration_raises(
        self, planner: TemplatePlanner
    ) -> None:
        from spectrochempy_ai.template_planner import (
            TemplateStep,
            WorkflowTemplate,
        )
        from spectrochempy_ai.workflow_plan import ScientificContext

        bad_template = WorkflowTemplate(
            template_id="bad_params",
            description="Has bad parameter names.",
            scientific_context=ScientificContext(
                goal="test", analytical_strategy="test"
            ),
            steps=[
                TemplateStep(
                    step_id="s1",
                    operation_id="pca",
                    display_label="PCA",
                    rationale="",
                    input_refs=[],
                    parameters={"typo_n_components": 3},
                    output_var="result",
                ),
            ],
        )
        with pytest.raises(UnknownParameterError):
            planner.register_template(bad_template)

    def test_register_template_with_no_params_succeeds(
        self, planner: TemplatePlanner
    ) -> None:
        from spectrochempy_ai.template_planner import (
            TemplateStep,
            WorkflowTemplate,
        )
        from spectrochempy_ai.workflow_plan import ScientificContext

        # A step with zero parameters declared and zero expected should pass
        template = WorkflowTemplate(
            template_id="no_params",
            description="Step with no parameters.",
            scientific_context=ScientificContext(
                goal="test", analytical_strategy="test"
            ),
            steps=[
                TemplateStep(
                    step_id="s1",
                    operation_id="score_plot",
                    display_label="Score plot",
                    rationale="",
                    input_refs=[],
                    parameters={},
                    output_var="",
                ),
            ],
        )
        planner.register_template(template)  # should not raise


# ---------------------------------------------------------------------------
# Template metadata (Phase 3 consolidation)
# ---------------------------------------------------------------------------


class TestTemplateMetadata:
    def test_template_has_version(self, planner: TemplatePlanner) -> None:
        for tid in TEMPLATE_IDS:
            t = planner.get_template(tid)
            assert t.template_version == "0.1.0"

    def test_template_has_registry_version(self, planner: TemplatePlanner) -> None:
        for tid in TEMPLATE_IDS:
            t = planner.get_template(tid)
            assert t.compatible_registry_version == REGISTRY_VERSION

    def test_template_serialize_roundtrip(self, planner: TemplatePlanner) -> None:
        from spectrochempy_ai.template_planner import WorkflowTemplate

        for tid in TEMPLATE_IDS:
            t = planner.get_template(tid)
            data = t.to_dict()
            restored = WorkflowTemplate.from_dict(data)
            assert restored.template_id == t.template_id
            assert restored.template_version == t.template_version
            assert restored.compatible_registry_version == t.compatible_registry_version
            assert len(restored.steps) == len(t.steps)
            for rs, ts in zip(restored.steps, t.steps):
                assert rs.step_id == ts.step_id
                assert rs.operation_id == ts.operation_id
                assert rs.parameters == ts.parameters

    def test_template_from_dict_respects_defaults(self) -> None:
        from spectrochempy_ai.template_planner import WorkflowTemplate
        from spectrochempy_ai.workflow_plan import ScientificContext

        minimal = {
            "template_id": "minimal",
            "description": "A minimal template",
            "scientific_context": {
                "goal": "test",
                "analytical_strategy": "test",
            },
        }
        t = WorkflowTemplate.from_dict(minimal)
        assert t.template_version == "0.1.0"
        assert t.compatible_registry_version == REGISTRY_VERSION
        assert len(t.steps) == 0


# ---------------------------------------------------------------------------
# Parametrized pipeline tests (Phase 3 consolidation)
# ---------------------------------------------------------------------------


class TestParametrizedTemplates:
    def test_template_registers(self, planner: TemplatePlanner, template_id: str) -> None:
        t = planner.get_template(template_id)
        planner.register_template(t)  # idempotent; should not raise

    def test_template_instantiation(self, planner: TemplatePlanner, template_id: str) -> None:
        plan = planner.create_plan(template_id)
        assert plan.planner_id == "TemplatePlanner"
        assert plan.planner_config == {"template_id": template_id}
        assert len(plan.steps) > 0

    def test_template_validates(self, planner: TemplatePlanner, template_id: str) -> None:
        plan = planner.create_plan(template_id)
        validate(plan)

    def test_template_renders(self, planner: TemplatePlanner, template_id: str) -> None:
        plan = planner.create_plan(template_id)
        notebook = render(plan)
        assert notebook.nbformat >= 4
        assert len(notebook.cells) > 0

    def test_template_deterministic(self, planner: TemplatePlanner, template_id: str) -> None:
        plan = planner.create_plan(template_id)
        nb1 = render(plan)
        nb2 = render(plan)
        assert len(nb1.cells) == len(nb2.cells)
        for c1, c2 in zip(nb1.cells, nb2.cells):
            assert c1.cell_type == c2.cell_type
            assert c1.source == c2.source

    def test_template_overrides(
        self, planner: TemplatePlanner, template_id: str
    ) -> None:
        plan = planner.create_plan(template_id, parameter_overrides={})
        validate(plan)  # no-op overrides should not break
