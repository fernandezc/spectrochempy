"""Tests for the Phase 0 notebook renderer.

All tests use the hand-written fixture plan. No AI required.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

from spectrochempy_ai.notebook_renderer import render, write_notebook
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
        for c1, c2 in zip(nb1.cells, nb2.cells):
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
