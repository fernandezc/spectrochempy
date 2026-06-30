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

    def test_reference_file_not_found_raises(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy")
        with pytest.raises(FileNotFoundError):
            explore(
                str(input_file),
                reference_path="/nonexistent/ref.csv",
                template_id="pls_calibration",
            )

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

    @pytest.mark.parametrize("suffix", [".spg", ".wdf", ".srs", ".mat"])
    def test_default_load_omits_format_argument(
        self, tmp_path: Path, suffix: str
    ) -> None:
        input_file = tmp_path / f"data{suffix}"
        input_file.write_text("dummy spectral data")
        output = explore(str(input_file))
        with open(output, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        sources = [c.source for c in nb.cells if c.cell_type == "code"]
        load_cells = [s for s in sources if "scp.read(" in s]
        assert load_cells
        assert all("format=" not in cell for cell in load_cells)

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

    def test_unknown_operation_id_raises(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner
        from spectrochempy_ai.template_planner import UnknownOverrideTarget

        planner = TemplatePlanner()
        with pytest.raises(UnknownOverrideTarget):
            planner.create_plan(
                "exploratory_pca",
                operation_overrides={"magic_op": {"param": 1}},
            )


class TestMcralsAnalysisTemplate:
    """Tests for the mcrals_analysis template (Phase 8)."""

    def test_template_registered(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        assert "mcrals_analysis" in planner.list_templates()

    def test_plan_instantiation(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        plan = planner.create_plan("mcrals_analysis")
        assert plan.planner_config["template_id"] == "mcrals_analysis"
        assert len(plan.steps) == 7
        op_ids = [s.operation_id for s in plan.steps]
        assert op_ids == [
            "load",
            "inspect",
            "baseline",
            "mcrals_init",
            "mcrals",
            "mcrals_conc_plot",
            "mcrals_spec_plot",
        ]

    def test_dataflow_chain(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        plan = planner.create_plan("mcrals_analysis")
        # s4 (mcrals_init) takes dataset_corrected from s3
        init_step = [s for s in plan.steps if s.operation_id == "mcrals_init"][0]
        assert init_step.input_refs == ["dataset_corrected"]
        # s5 (mcrals) takes dataset_corrected and conc_guess
        mcrals_step = [s for s in plan.steps if s.operation_id == "mcrals"][0]
        assert mcrals_step.input_refs == ["dataset_corrected", "conc_guess"]

    def test_parameter_defaults(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        plan = planner.create_plan("mcrals_analysis")
        mcrals_step = [s for s in plan.steps if s.operation_id == "mcrals"][0]
        assert mcrals_step.parameters["n_components"] == 3
        assert mcrals_step.parameters["max_iter"] == 100
        init_step = [s for s in plan.steps if s.operation_id == "mcrals_init"][0]
        assert init_step.parameters["n_components"] == 3

    def test_operation_override_n_components(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        plan = planner.create_plan(
            "mcrals_analysis",
            operation_overrides={"mcrals": {"n_components": 4}},
        )
        mcrals_step = [s for s in plan.steps if s.operation_id == "mcrals"][0]
        assert mcrals_step.parameters["n_components"] == 4

    def test_notebook_generation(self, tmp_path: Path) -> None:
        input_file = tmp_path / "mixture.scp"
        input_file.write_text("dummy mixture data")
        output = explore(
            str(input_file),
            str(tmp_path / "mcrals.ipynb"),
            template_id="mcrals_analysis",
        )
        assert output.exists()
        with open(output, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        assert nb.nbformat >= 4
        # Check title
        first = nb.cells[0]
        assert "MCR-ALS" in first.source
        # Check public API calls
        code_sources = [c.source for c in nb.cells if c.cell_type == "code"]
        all_code = "\n".join(code_sources)
        assert "scp.MCRALS()" in all_code
        assert "scp.Baseline(model='asls')" in all_code
        assert "mcrals_result.result" in all_code
        assert "mcrals_result.C.plot" in all_code
        assert "mcrals_result.St.plot" in all_code

    def test_notebook_has_editable_parameters(self, tmp_path: Path) -> None:
        input_file = tmp_path / "mixture.scp"
        input_file.write_text("dummy")
        output = explore(
            str(input_file),
            str(tmp_path / "mcrals.ipynb"),
            template_id="mcrals_analysis",
        )
        with open(output, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        code_sources = [c.source for c in nb.cells if c.cell_type == "code"]
        all_code = "\n".join(code_sources)
        assert "N_COMPONENTS = 3" in all_code
        assert "MAX_ITER = 100" in all_code

    def test_output_filename_derived_from_template(self, tmp_path: Path) -> None:
        input_file = tmp_path / "my_mixture.scp"
        input_file.write_text("dummy")
        output = explore(str(input_file), template_id="mcrals_analysis")
        assert output.name == "my_mixture-mcrals-analysis.ipynb"

    def test_scientific_context_present(self, tmp_path: Path) -> None:
        input_file = tmp_path / "mixture.scp"
        input_file.write_text("dummy")
        output = explore(
            str(input_file),
            str(tmp_path / "mcrals.ipynb"),
            template_id="mcrals_analysis",
        )
        with open(output, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        markdowns = [c.source for c in nb.cells if c.cell_type == "markdown"]
        combined = "\n".join(markdowns)
        assert "MCR-ALS" in combined
        assert "non-negative" in combined
        assert "rotational ambiguity" in combined


class TestPlsCalibrationTemplate:
    """Tests for the pls_calibration template (Phase 9)."""

    def test_template_registered(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        assert "pls_calibration" in planner.list_templates()

    def test_plan_instantiation(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        plan = planner.create_plan("pls_calibration")
        assert plan.planner_config["template_id"] == "pls_calibration"
        assert len(plan.steps) == 7
        op_ids = [s.operation_id for s in plan.steps]
        assert op_ids == [
            "load",
            "load",
            "inspect",
            "inspect",
            "baseline",
            "pls",
            "pls_predict_plot",
        ]

    def test_two_input_workflow(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        plan = planner.create_plan("pls_calibration")
        pls_step = [s for s in plan.steps if s.operation_id == "pls"][0]
        assert pls_step.input_refs == ["dataset_corrected", "reference"]

    def test_parameter_defaults(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        plan = planner.create_plan("pls_calibration")
        pls_step = [s for s in plan.steps if s.operation_id == "pls"][0]
        assert pls_step.parameters["n_components"] == 3

    def test_operation_override_n_components(self) -> None:
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        plan = planner.create_plan(
            "pls_calibration",
            operation_overrides={"pls": {"n_components": 5}},
        )
        pls_step = [s for s in plan.steps if s.operation_id == "pls"][0]
        assert pls_step.parameters["n_components"] == 5

    def test_notebook_generation(self, tmp_path: Path) -> None:
        from spectrochempy_ai.notebook_renderer import render
        from spectrochempy_ai.template_planner import TemplatePlanner
        from spectrochempy_ai.validator import validate

        planner = TemplatePlanner()
        plan = planner.create_plan("pls_calibration")
        validate(plan)
        notebook = render(plan)
        assert notebook.nbformat >= 4
        assert len(notebook.cells) > 0
        # Check title
        first = notebook.cells[0]
        assert first.cell_type == "markdown"
        assert "PLS" in first.source
        # Check public API
        code_sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        all_code = "\n".join(code_sources)
        assert "scp.PLSRegression" in all_code
        assert "pls_result.result" in all_code
        assert "pls_result.predict" in all_code

    def test_notebook_has_editable_parameters(self, tmp_path: Path) -> None:
        from spectrochempy_ai.notebook_renderer import render
        from spectrochempy_ai.template_planner import TemplatePlanner

        planner = TemplatePlanner()
        plan = planner.create_plan("pls_calibration")
        notebook = render(plan)
        code_sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        all_code = "\n".join(code_sources)
        assert "N_COMPONENTS = 3" in all_code

    def test_reference_path_wired_to_second_load(self, tmp_path: Path) -> None:
        input_file = tmp_path / "spectra.scp"
        input_file.write_text("dummy")
        ref_file = tmp_path / "reference.csv"
        ref_file.write_text("dummy reference")
        output = explore(
            str(input_file),
            str(tmp_path / "pls.ipynb"),
            template_id="pls_calibration",
            reference_path=str(ref_file),
        )
        with open(output, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        code_sources = [c.source for c in nb.cells if c.cell_type == "code"]
        all_code = "\n".join(code_sources)
        assert "reference.csv" in all_code
        assert "spectra.scp" in all_code

    def test_reference_path_ignored_for_single_input(self, tmp_path: Path) -> None:
        input_file = tmp_path / "data.scp"
        input_file.write_text("dummy")
        ref_file = tmp_path / "ref.csv"
        ref_file.write_text("dummy")
        output = explore(
            str(input_file),
            str(tmp_path / "pca.ipynb"),
            reference_path=str(ref_file),
        )
        with open(output, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        code_sources = [c.source for c in nb.cells if c.cell_type == "code"]
        all_code = "\n".join(code_sources)
        # Only one load — reference file should not appear
        assert "ref.csv" not in all_code
