"""
Tests for the Phase 0 notebook renderer.

All tests use the hand-written fixture plan. No AI required.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

from spectrochempy_ai.notebook_renderer import render
from spectrochempy_ai.notebook_renderer import write_notebook
from spectrochempy_ai.template_planner import TemplatePlanner
from spectrochempy_ai.validator import validate
from spectrochempy_ai.workflow_plan import WorkflowPlan

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_fixture(name: str) -> WorkflowPlan:
    path = FIXTURES_DIR / name
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return WorkflowPlan.from_dict(data)


class TestRenderExploratoryPCA:
    def test_render_produces_notebook(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        validate(plan)
        notebook = render(plan)
        assert notebook.nbformat >= 4
        assert len(notebook.cells) > 0

    def test_notebook_has_title_cell(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        notebook = render(plan)
        first = notebook.cells[0]
        assert first.cell_type == "markdown"
        assert "PCA" in first.source

    def test_notebook_has_import_cell(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        notebook = render(plan)
        sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        assert any("import spectrochempy" in s for s in sources)
        assert any("import numpy" in s for s in sources)

    def test_notebook_has_all_steps(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        notebook = render(plan)
        markdowns = [c.source for c in notebook.cells if c.cell_type == "markdown"]
        for step in plan.steps:
            assert any(step.display_label in m for m in markdowns)

    def test_notebook_has_manifest_metadata(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        notebook = render(plan)
        manifest = notebook.metadata.get("spectrochempy_workflow_assistant", {})
        assert manifest.get("schema_version") == plan.schema_version
        assert manifest.get("planner_id") == plan.planner_id

    def test_determinism(self) -> None:
        """Rendering the same plan twice must produce equivalent notebooks."""
        plan = load_fixture("exploratory_pca.json")
        nb1 = render(plan)
        nb2 = render(plan)
        # Compare cell sources
        assert len(nb1.cells) == len(nb2.cells)
        for c1, c2 in zip(nb1.cells, nb2.cells, strict=False):
            assert c1.cell_type == c2.cell_type
            assert c1.source == c2.source

    def test_write_notebook_roundtrip(self, tmp_path: Path) -> None:
        plan = load_fixture("exploratory_pca.json")
        out = tmp_path / "test.ipynb"
        write_notebook(plan, str(out))
        assert out.exists()
        with open(out, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        assert len(nb.cells) > 0


class TestRendererEdgeCases:
    def test_unknown_operation_generates_placeholder(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        plan.steps[0].operation_id = "unknown_op"
        # We do not validate here — renderer should handle unknown ops gracefully
        notebook = render(plan)
        sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        assert any("Unknown operation" in s for s in sources)

    def test_load_omits_default_format(self) -> None:
        plan = TemplatePlanner().create_plan("exploratory_pca")
        plan.steps[0].parameters["filename"] = "sample.wdf"
        plan.steps[0].parameters["format"] = None
        notebook = render(plan)
        sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        load_cell = next(s for s in sources if "scp.read(" in s)
        assert "scp.read('sample.wdf')" in load_cell
        assert "format=" not in load_cell

    def test_load_keeps_explicit_format_override(self) -> None:
        plan = TemplatePlanner().create_plan("exploratory_pca")
        plan.steps[0].parameters["filename"] = "sample.csv"
        plan.steps[0].parameters["format"] = "csv"
        notebook = render(plan)
        sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        load_cell = next(s for s in sources if "scp.read(" in s)
        assert "format='csv'" in load_cell

    def test_asls_warning_filter_is_present_and_narrow(self) -> None:
        plan = load_fixture("exploratory_pca.json")
        notebook = render(plan)
        sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        import_cell = sources[0]
        assert "import warnings" in import_cell
        assert "from scipy.sparse import SparseEfficiencyWarning" in import_cell
        assert "category=SparseEfficiencyWarning" in import_cell
        assert "spsolve requires A be CSC or CSR matrix format" in import_cell

    def test_non_asls_baseline_does_not_add_warning_filter(self) -> None:
        plan = TemplatePlanner().create_plan("exploratory_pca")
        baseline_step = next(step for step in plan.steps if step.operation_id == "baseline")
        baseline_step.parameters["method"] = "detrend"
        notebook = render(plan)
        sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        import_cell = sources[0]
        assert "SparseEfficiencyWarning" not in import_cell
