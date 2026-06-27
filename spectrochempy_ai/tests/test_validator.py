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

    def test_mcrals_analysis_executes(self, tmp_path: Path) -> None:
        import spectrochempy as scp
        from spectrochempy_ai.exploration import explore

        # Create a synthetic mixture: 3 components, 50 observations, 20 variables
        rng = np.random.default_rng(42)
        t = np.linspace(0, 10, 50)
        x = np.linspace(0, 100, 20)

        # Pure spectra (3 gaussians)
        pure_spectra = np.array([
            np.exp(-((x - 30) ** 2) / 200),
            np.exp(-((x - 50) ** 2) / 200),
            np.exp(-((x - 70) ** 2) / 200),
        ])

        # Concentration profiles (3 gaussians in time)
        conc = np.array([
            np.exp(-((t - 3) ** 2) / 2),
            np.exp(-((t - 5) ** 2) / 2),
            np.exp(-((t - 7) ** 2) / 2),
        ]).T

        # Mixture with small noise
        mixture_data = conc @ pure_spectra + rng.normal(scale=0.01, size=(50, 20))
        data = scp.NDDataset(
            mixture_data.astype(np.float32),
            title="synthetic_mixture",
        )
        data.y = np.arange(50)
        data.x = x
        ds_path = tmp_path / "mcrals_data.scp"
        data.save_as(str(ds_path), confirm=False)

        nb_path = explore(
            input_path=str(ds_path),
            output_path=str(tmp_path / "mcrals_notebook.ipynb"),
            template_id="mcrals_analysis",
        )

        validate_notebook_execution(nb_path)

    @pytest.mark.skip(reason="SPeC PLSRegression has a coordinate-wrapping bug with 1D y; see issue TBD")
    def test_pls_calibration_executes(self, tmp_path: Path) -> None:
        import spectrochempy as scp
        from spectrochempy_ai.notebook_renderer import write_notebook
        from spectrochempy_ai.template_planner import TemplatePlanner
        from spectrochempy_ai.validator import validate

        rng = np.random.default_rng(42)

        # Create synthetic spectra (30 obs x 20 vars) correlated with reference
        n_obs, n_vars = 30, 20
        x_axis = np.linspace(1000, 4000, n_vars)
        true_coef = np.exp(-((x_axis - 2000) ** 2) / 1e5)
        X_base = rng.normal(size=(n_obs, n_vars))
        y_values = X_base @ true_coef + rng.normal(scale=0.1, size=n_obs)
        X_data = X_base + rng.normal(scale=0.1, size=(n_obs, n_vars))

        spectra = scp.NDDataset(X_data.astype(np.float32), title="synthetic_spectra")
        spectra.y = np.arange(n_obs)
        spectra.x = x_axis
        spectra_path = tmp_path / "spectra.scp"
        spectra.save_as(str(spectra_path), confirm=False)

        reference = scp.NDDataset(y_values.astype(np.float32), title="reference")
        reference.x = np.arange(n_obs)
        ref_path = tmp_path / "reference.scp"
        reference.save_as(str(ref_path), confirm=False)

        planner = TemplatePlanner()
        plan = planner.create_plan(
            "pls_calibration",
            operation_overrides={
                "load": {"filename": str(spectra_path), "format": "scp"},
            },
        )
        # Override the second load step (reference) with the actual path
        plan.steps[1].parameters["filename"] = str(ref_path)
        plan.steps[1].parameters["format"] = "scp"
        validate(plan)
        nb_path = tmp_path / "pls_notebook.ipynb"
        write_notebook(plan, str(nb_path))

        validate_notebook_execution(nb_path)
