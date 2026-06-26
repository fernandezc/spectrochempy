"""Phase 0.5 stress tests for WorkflowPlan fixtures.

Every fixture must load, validate, render deterministically, and survive
basic structural checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectrochempy_ai.notebook_renderer import render
from spectrochempy_ai.validator import ValidationError, validate
from spectrochempy_ai.workflow_plan import WorkflowPlan


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

FIXTURE_NAMES = [
    "exploratory_pca.json",
    "baseline_integrate_plot.json",
    "smoothing_pca.json",
    "nmf_workflow.json",
    "mcrals_workflow.json",
    "simple_export.json",
]


def load_fixture(name: str) -> WorkflowPlan:
    path = FIXTURES_DIR / name
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return WorkflowPlan.from_dict(data)


class TestFixtureCoverage:
    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_fixture_loads(self, name: str) -> None:
        plan = load_fixture(name)
        assert plan.schema_version == "0.1.0"
        assert plan.scientific_context.goal != ""
        assert len(plan.steps) > 0

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_fixture_validates(self, name: str) -> None:
        plan = load_fixture(name)
        validate(plan)  # should not raise

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_fixture_renders(self, name: str) -> None:
        plan = load_fixture(name)
        notebook = render(plan)
        assert notebook.nbformat >= 4
        assert len(notebook.cells) > 0

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_fixture_deterministic(self, name: str) -> None:
        plan = load_fixture(name)
        nb1 = render(plan)
        nb2 = render(plan)
        assert len(nb1.cells) == len(nb2.cells)
        for c1, c2 in zip(nb1.cells, nb2.cells):
            assert c1.cell_type == c2.cell_type
            assert c1.source == c2.source

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_fixture_has_title_cell(self, name: str) -> None:
        plan = load_fixture(name)
        notebook = render(plan)
        first = notebook.cells[0]
        assert first.cell_type == "markdown"
        assert plan.scientific_context.goal in first.source

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_fixture_has_import_cell(self, name: str) -> None:
        plan = load_fixture(name)
        notebook = render(plan)
        sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        assert any("import spectrochempy" in s for s in sources)


class TestInvalidPlansStillFail:
    def test_unknown_operation_still_fails(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        plan.steps[0].operation_id = "magic_op"
        with pytest.raises(ValidationError) as exc_info:
            validate(plan)
        assert any("magic_op" in v for v in exc_info.value.violations)

    def test_unresolved_input_still_fails(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        plan.steps[2].input_refs = ["ghost_var"]
        with pytest.raises(ValidationError) as exc_info:
            validate(plan)
        assert any("ghost_var" in v for v in exc_info.value.violations)

    def test_orphan_output_still_fails(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        from spectrochempy_ai.workflow_plan import OutputReference
        plan.outputs = [OutputReference(name="orphan", type="dataset")]
        with pytest.raises(ValidationError) as exc_info:
            validate(plan)
        assert any("orphan" in v for v in exc_info.value.violations)
