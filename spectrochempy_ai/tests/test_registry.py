"""
Tests for OperationRegistry and OperationSpecification.

These tests verify that the registry contains the expected specifications
and that lookup/discovery work correctly.
"""

from __future__ import annotations

import pytest

from spectrochempy_ai.operation_registry import RegistryLookupError
from spectrochempy_ai.operation_registry import get_spec
from spectrochempy_ai.operation_registry import is_registered
from spectrochempy_ai.operation_registry import list_operation_ids
from spectrochempy_ai.operation_registry import list_specs


class TestRegistryLookup:
    def test_all_prototype_operations_registered(self) -> None:
        expected = {
            "read",
            "baseline",
            "smooth",
            "pca",
            "score_plot",
            "loading_plot",
            "integrate",
            "plot",
            "nmf",
            "nmf_components_plot",
            "nmf_reconstruction_plot",
            "mcrals",
            "mcrals_conc_plot",
            "mcrals_spec_plot",
            "inspect",
            "export",
        }
        registered = set(list_operation_ids())
        assert expected <= registered

    def test_get_spec_returns_specification(self) -> None:
        spec = get_spec("pca")
        assert spec.operation_id == "pca"
        assert spec.display_name == "Principal Component Analysis"
        assert len(spec.inputs) == 1
        assert len(spec.outputs) == 1
        assert spec.outputs[0].type == "result"

    def test_get_spec_unknown_raises(self) -> None:
        with pytest.raises(RegistryLookupError):
            get_spec("magic_op")

    def test_is_registered_true(self) -> None:
        assert is_registered("baseline")

    def test_is_registered_false(self) -> None:
        assert not is_registered("magic_op")


class TestSpecificationContents:
    def test_pca_has_parameters(self) -> None:
        spec = get_spec("pca")
        param_names = {p.name for p in spec.parameters}
        assert "n_components" in param_names

    def test_nmf_has_constraint(self) -> None:
        spec = get_spec("nmf")
        assert len(spec.constraints) == 1
        assert spec.constraints[0].predicate == "requires_positive_values"

    def test_plot_has_side_effect(self) -> None:
        spec = get_spec("plot")
        assert "plot" in spec.side_effects

    def test_mcrals_has_two_inputs(self) -> None:
        spec = get_spec("mcrals")
        assert len(spec.inputs) == 2
        assert spec.inputs[0].name == "dataset"
        assert spec.inputs[1].name == "conc_guess"

    def test_export_has_file_write_side_effect(self) -> None:
        spec = get_spec("export")
        assert "file_write" in spec.side_effects

    def test_inspect_has_no_outputs(self) -> None:
        spec = get_spec("inspect")
        assert len(spec.outputs) == 0
        assert "print" in spec.side_effects


class TestRegistryDiscovery:
    def test_list_specs_by_category(self) -> None:
        preprocessing = list_specs(category="preprocessing")
        ids = {s.operation_id for s in preprocessing}
        assert "baseline" in ids
        assert "smooth" in ids

    def test_list_specs_all(self) -> None:
        all_specs = list_specs()
        assert len(all_specs) >= 16
