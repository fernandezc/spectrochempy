"""
Tests for the Phase 0 validator.

All tests use the hand-written fixture plan. No AI required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spectrochempy_ai.validator import NotebookExecutionError
from spectrochempy_ai.validator import ValidationError
from spectrochempy_ai.validator import validate
from spectrochempy_ai.validator import validate_notebook_execution
from spectrochempy_ai.workflow_plan import WorkflowPlan

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_fixture(name: str) -> WorkflowPlan:
    path = FIXTURES_DIR / name
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return WorkflowPlan.from_dict(data)


class TestValidateExploratoryPCA:
    def test_fixture_loads(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        assert plan.schema_version == "0.1.0"
        assert plan.scientific_context.goal != ""
        assert len(plan.steps) == 5

    def test_valid_plan_passes(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        validate(plan)  # should not raise

    def test_missing_schema_version_fails(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        plan.schema_version = ""
        with pytest.raises(ValidationError) as exc_info:
            validate(plan)
        assert "schema_version is required" in exc_info.value.violations

    def test_missing_goal_fails(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        plan.scientific_context.goal = ""
        with pytest.raises(ValidationError) as exc_info:
            validate(plan)
        assert "scientific_context.goal is required" in exc_info.value.violations

    def test_unknown_operation_fails(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        plan.steps[0].operation_id = "unknown_op"
        with pytest.raises(ValidationError) as exc_info:
            validate(plan)
        assert any(
            "unknown_op" in v and "registry" in v for v in exc_info.value.violations
        )

    def test_unresolved_input_ref_fails(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        plan.steps[2].input_refs = ["nonexistent_var"]
        with pytest.raises(ValidationError) as exc_info:
            validate(plan)
        assert any("nonexistent_var" in v for v in exc_info.value.violations)

    def test_duplicate_step_id_fails(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        plan.steps[0].step_id = "same"
        plan.steps[1].step_id = "same"
        with pytest.raises(ValidationError) as exc_info:
            validate(plan)
        assert any("duplicate step_id" in v for v in exc_info.value.violations)


class TestValidateEmptyPlan:
    def test_empty_plan_fails(self) -> None:
        plan = WorkflowPlan(
            schema_version="",
            spectrochempy_version="",
            plugin_version="",
            scientific_context=None,  # type: ignore[arg-type]
        )
        with pytest.raises(ValidationError) as exc_info:
            validate(plan)
        violations = exc_info.value.violations
        assert any("schema_version" in v for v in violations)
        assert any("spectrochempy_version" in v for v in violations)
        assert any("plugin_version" in v for v in violations)


class TestNotebookExecution:
    def test_exploratory_pca_executes(self, tmp_path: Path) -> None:
        import spectrochempy as scp
        from spectrochempy_ai.exploration import explore

        rng = np.random.default_rng(42)
        data = scp.NDDataset(
            rng.normal(size=(50, 100)).astype(np.float32),
            title="synthetic",
        )
        data.y = np.arange(50)
        data.x = np.linspace(1000, 4000, 100)
        ds_path = tmp_path / "test_data.scp"
        data.save_as(str(ds_path), confirm=False)

        nb_path = explore(
            input_path=str(ds_path),
            output_path=str(tmp_path / "test_notebook.ipynb"),
            n_components=3,
        )

        validate_notebook_execution(nb_path)

    def test_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_notebook_execution(tmp_path / "nonexistent.ipynb")
