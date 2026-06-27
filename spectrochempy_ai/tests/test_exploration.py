"""
Tests for Phase 6 exploration API.

Tests the high-level ``explore`` function and its internal alias.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

from spectrochempy_ai import create_exploration_notebook
from spectrochempy_ai import explore
from spectrochempy_ai.template_planner import TemplateNotFoundError


class TestExplore:
    """Tests for the public ``explore`` function."""

    def test_creates_notebook_from_existing_file(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy spectral data")
        output = explore(str(input_file), None)
        assert output.exists()
        assert output.suffix == ".ipynb"

    def test_output_contains_valid_notebook(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy spectral data")
        output = explore(str(input_file))
        with open(output, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        assert nb.nbformat >= 4
        assert len(nb.cells) > 0

    def test_output_path_derived_from_input(self, tmp_path: Path) -> None:
        input_file = tmp_path / "my_spectra.scp"
        input_file.write_text("dummy")
        output = explore(str(input_file))
        assert output.name == "my_spectra-exploratory-pca.ipynb"

    def test_custom_output_path(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy")
        custom = tmp_path / "custom.ipynb"
        output = explore(str(input_file), str(custom))
        assert output == custom.resolve()

    def test_notebook_has_expected_content(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy")
        output = explore(str(input_file))
        with open(output, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        sources = [c.source for c in nb.cells if c.cell_type == "code"]
        all_sources = "".join(sources)
        assert "import spectrochempy" in all_sources
        assert "read(" in all_sources or "scp.read(" in all_sources

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            explore("/nonexistent/path.scp")

    def test_override_n_components(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy")
        output = explore(str(input_file), n_components=3)
        with open(output, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        code_sources = [c.source for c in nb.cells if c.cell_type == "code"]
        assert any("N_COMPONENTS = 3" in s for s in code_sources)

    def test_override_baseline_method(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy")
        output = explore(str(input_file), baseline_method="detrend")
        with open(output, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        sources = [c.source for c in nb.cells if c.cell_type == "code"]
        assert any("detrend" in s for s in sources)

    def test_override_file_format(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.csv"
        input_file.write_text("x,y\n1,2")
        output = explore(str(input_file), file_format="csv")
        with open(output, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        sources = [c.source for c in nb.cells if c.cell_type == "code"]
        assert any("format='csv'" in s for s in sources)

    def test_notebook_has_scientific_context(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy")
        output = explore(str(input_file))
        with open(output, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        markdown_cells = [c.source for c in nb.cells if c.cell_type == "markdown"]
        combined = "\n".join(markdown_cells)
        assert "Goal" in combined or "goal" in combined
        assert "PCA" in combined

    def test_notebook_has_manifest_metadata(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy")
        output = explore(str(input_file))
        with open(output, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        manifest = nb.metadata.get("spectrochempy_workflow_assistant", {})
        assert manifest.get("planner_id") == "TemplatePlanner"
        assert manifest.get("schema_version") == "0.1.0"

    def test_no_template_found_raises(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy")
        with pytest.raises(TemplateNotFoundError):
            explore(str(input_file), template_id="nonexistent_template")

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy")
        output = explore(str(input_file))
        assert output.is_absolute()

    def test_returned_path_is_resolved(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy")
        output = explore(str(input_file), str(input_file))
        assert output.exists()


class TestCreateExplorationNotebookAlias:
    """The internal alias must still work."""

    def test_alias_returns_same_type(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy")
        result = create_exploration_notebook(str(input_file))
        assert isinstance(result, Path)
        assert result.exists()


class TestOperationOverridesInPlanner:
    """Verify operation_overrides resolve correctly in TemplatePlanner."""

    def test_operation_overrides_applied(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        plan = planner.create_plan(
            "exploratory_pca",
            operation_overrides={"pca": {"n_components": 4}},
        )
        pca_step = [s for s in plan.steps if s.operation_id == "pca"][0]
        assert pca_step.parameters["n_components"] == 4

    def test_step_overrides_take_precedence(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        plan = planner.create_plan(
            "exploratory_pca",
            parameter_overrides={"s4": {"n_components": 5}},
            operation_overrides={"pca": {"n_components": 3}},
        )
        pca_step = [s for s in plan.steps if s.operation_id == "pca"][0]
        # step-level should win
        assert pca_step.parameters["n_components"] == 5

    def test_operation_override_multiple_steps(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        plan = planner.create_plan(
            "exploratory_pca",
            operation_overrides={"baseline": {"method": "detrend"}},
        )
        baseline_step = [s for s in plan.steps if s.operation_id == "baseline"][0]
        assert baseline_step.parameters["method"] == "detrend"

    def test_operation_override_combined_with_defaults(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        plan = planner.create_plan(
            "exploratory_pca",
            operation_overrides={
                "load": {"filename": "custom.scp"},
                "pca": {"n_components": 6},
            },
        )
        assert plan.steps[0].parameters["filename"] == "custom.scp"
        assert plan.steps[3].parameters["n_components"] == 6
