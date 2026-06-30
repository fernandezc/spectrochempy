"""
Tests for Phase 12 RulePlanner and suggest() API.

Covers DatasetProfile, RulePlanner rules, and the public suggest() entry
point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spectrochempy_ai.dataset_profile import DatasetProfile
from spectrochempy_ai.dataset_profile import profile_dataset
from spectrochempy_ai.rule_planner import RecommendationEvidence
from spectrochempy_ai.rule_planner import RulePlanner
from spectrochempy_ai.rule_planner import TemplateRecommendation
from spectrochempy_ai.rule_planner import suggest

# ---------------------------------------------------------------------------
# DatasetProfile
# ---------------------------------------------------------------------------


class TestDatasetProfile:
    def _make_dataset(self, shape: tuple[int, ...], name: str, spectral: bool = True):
        import spectrochempy as scp

        data = np.random.randn(*shape)
        dataset = scp.NDDataset(data, name=name)
        if len(shape) == 2:
            if spectral:
                dataset.x = np.linspace(1000, 2000, shape[1])
            dataset.y = np.arange(shape[0])
        elif len(shape) == 1 and spectral:
            dataset.x = np.linspace(1000, 2000, shape[0])
        return dataset

    def test_profile_readable_2d(self, tmp_path: Path) -> None:
        import spectrochempy as scp

        data = np.random.randn(10, 100)
        d = scp.NDDataset(data, name="test")
        d.x = np.linspace(1000, 2000, 100)
        path = tmp_path / "test.scp"
        d.save_as(str(path))

        profile = profile_dataset(str(path))
        assert profile.readable is True
        assert profile.ndim == 2
        assert profile.shape == (10, 100)
        assert profile.dims == ("y", "x")
        assert profile.has_continuous_x is True
        assert profile.n_observations == 10
        assert profile.n_variables == 100

    def test_profile_non_existent(self) -> None:
        profile = profile_dataset("/nonexistent/file.scp")
        assert profile.readable is False
        assert profile.error == "File does not exist"

    def test_profile_unreadable_format(self, tmp_path: Path) -> None:
        path = tmp_path / "data.txt"
        path.write_text("not spectral data")
        profile = profile_dataset(str(path))
        assert profile.readable is False

    def test_profile_1d(self, tmp_path: Path) -> None:
        import spectrochempy as scp

        data = np.random.randn(100)
        d = scp.NDDataset(data, name="test")
        d.x = np.linspace(1000, 2000, 100)
        path = tmp_path / "test_1d.scp"
        d.save_as(str(path))

        profile = profile_dataset(str(path))
        assert profile.readable is True
        assert profile.ndim == 1
        assert profile.shape == (100,)
        assert profile.is_1d is True
        assert profile.n_observations == 1
        assert profile.n_variables == 100

    def test_profile_no_continuous_x(self, tmp_path: Path) -> None:
        import spectrochempy as scp

        data = np.random.randn(10, 100)
        d = scp.NDDataset(data, name="test")
        path = tmp_path / "no_coord.scp"
        d.save_as(str(path))

        profile = profile_dataset(str(path))
        # Without explicit x coord, has_continuous_x may be non-None
        # but the dataset still has ndim/shape
        assert profile.readable is True
        assert profile.ndim == 2

    def test_profile_properties(self) -> None:
        p = DatasetProfile(
            path=Path("/dummy.scp"),
            readable=True,
            ndim=2,
            shape=(5, 100),
        )
        assert p.is_2d is True
        assert p.is_1d is False
        assert p.is_spectral is False  # has_continuous_x is None

    def test_profile_spectral_property(self) -> None:
        p = DatasetProfile(
            path=Path("/dummy.scp"),
            readable=True,
            has_continuous_x=True,
        )
        assert p.is_spectral is True

    def test_profile_multi_object_selects_largest_2d_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import spectrochempy as scp

        path = tmp_path / "multi.mat"
        path.write_text("dummy")
        datasets = scp.ScpObjectList(
            [
                self._make_dataset((8,), "trace"),
                self._make_dataset((4, 20), "small"),
                self._make_dataset((12, 50), "largest"),
            ]
        )
        monkeypatch.setattr(scp, "read", lambda _: datasets)

        profile = profile_dataset(str(path))
        assert profile.readable is True
        assert profile.source_was_multi_object is True
        assert profile.source_object_count == 3
        assert profile.selected_object_index == 2
        assert profile.selected_object_name == "largest"
        assert profile.shape == (12, 50)

    def test_profile_summary_mentions_multi_object_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import spectrochempy as scp

        path = tmp_path / "multi.mat"
        path.write_text("dummy")
        datasets = scp.ScpObjectList(
            [
                self._make_dataset((3, 10), "a"),
                self._make_dataset((5, 40), "b"),
            ]
        )
        monkeypatch.setattr(scp, "read", lambda _: datasets)

        profile = profile_dataset(str(path))
        assert "multi-object" in profile.summary
        assert "selected index 1" in profile.summary
        assert "name 'b'" in profile.summary

    def test_profile_multi_object_without_suitable_dataset_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import spectrochempy as scp

        path = tmp_path / "multi.mat"
        path.write_text("dummy")
        datasets = scp.ScpObjectList(
            [
                self._make_dataset((8,), "trace1"),
                self._make_dataset((12,), "trace2"),
            ]
        )
        monkeypatch.setattr(scp, "read", lambda _: datasets)

        profile = profile_dataset(str(path))
        assert profile.readable is False
        assert profile.source_was_multi_object is True
        assert "no suitable 2D dataset" in (profile.error or "")


# ---------------------------------------------------------------------------
# RulePlanner — rules only (no real files)
# ---------------------------------------------------------------------------


class TestRulePlanner:
    def make_profile(
        self,
        *,
        ndim: int = 2,
        shape: tuple[int, ...] = (10, 100),
        has_continuous_x: bool = True,
        readable: bool = True,
    ) -> DatasetProfile:
        dims = ("y", "x") if ndim == 2 else ("x",)
        return DatasetProfile(
            path=Path("/dummy.scp"),
            readable=readable,
            ndim=ndim,
            shape=shape,
            dims=dims,
            has_continuous_x=has_continuous_x,
            n_observations=shape[0] if ndim == 2 else 1,
            n_variables=shape[-1] if ndim >= 1 else None,
            summary=f"{ndim}D, shape {shape}",
        )

    def test_pca_recommendation(self) -> None:
        profile = self.make_profile()
        planner = RulePlanner()
        recs = planner.suggest(profile)
        assert len(recs) >= 1
        top = recs[0]
        assert top.template_id == "exploratory_pca"
        assert top.confidence == 0.7

    def test_pls_with_reference(self) -> None:
        profile = self.make_profile()
        planner = RulePlanner()
        recs = planner.suggest(profile, reference_path="ref.csv")
        assert len(recs) >= 1
        top = recs[0]
        assert top.template_id == "pls_calibration"
        assert top.confidence == 0.8

    def test_pls_calibrate_intent(self) -> None:
        profile = self.make_profile()
        planner = RulePlanner()
        recs = planner.suggest(profile, reference_path="ref.csv", intent="calibrate")
        assert len(recs) >= 1
        top = recs[0]
        assert top.template_id == "pls_calibration"
        # calibrate intent gives higher confidence
        assert top.confidence == 0.85

    def test_mcrals_with_resolve_intent(self) -> None:
        profile = self.make_profile(shape=(5, 100))
        planner = RulePlanner()
        recs = planner.suggest(profile, intent="resolve")
        assert len(recs) >= 1
        top = recs[0]
        assert top.template_id == "mcrals_analysis"
        assert top.confidence == 0.7

    def test_mcrals_small_dataset(self) -> None:
        profile = self.make_profile(shape=(5, 100))
        planner = RulePlanner()
        recs = planner.suggest(profile)
        # Small dataset should suggest PCA (0.7) above MCR-ALS (0.5)
        assert len(recs) >= 2
        # MCR-ALS should appear with lower confidence
        mcrals = [r for r in recs if r.template_id == "mcrals_analysis"]
        assert len(mcrals) == 1
        assert mcrals[0].confidence == 0.5
        assert len(mcrals[0].warnings) >= 1

    def test_1d_no_template(self) -> None:
        profile = self.make_profile(ndim=1, shape=(100,))
        planner = RulePlanner()
        recs = planner.suggest(profile)
        assert len(recs) == 1
        top = recs[0]
        assert top.template_id == "baseline_integrate"
        assert top.confidence == 0.85

    def test_fallback_for_unreadable(self) -> None:
        profile = DatasetProfile(
            path=Path("/broken.scp"),
            readable=False,
            summary="Cannot read file",
            error="File format not recognised",
        )
        planner = RulePlanner()
        recs = planner.suggest(profile)
        assert len(recs) == 1
        top = recs[0]
        assert top.template_id == "exploratory_pca"
        assert top.confidence == 0.3
        assert len(top.warnings) >= 1

    def test_suggestions_sorted_by_confidence(self) -> None:
        profile = self.make_profile(shape=(5, 100))
        planner = RulePlanner()
        recs = planner.suggest(profile)
        confidences = [r.confidence for r in recs]
        assert confidences == sorted(confidences, reverse=True)

    def test_template_recommendation_fields(self) -> None:
        rec = TemplateRecommendation(
            template_id="test",
            confidence=0.5,
            rationale="Some reason",
            dataset_summary="2D, shape (10, 100)",
            warnings=["Caution"],
        )
        assert rec.template_id == "test"
        assert rec.confidence == 0.5
        assert len(rec.warnings) == 1

    def test_pca_evidence_populated(self) -> None:
        profile = self.make_profile()
        planner = RulePlanner()
        recs = planner.suggest(profile)
        top = recs[0]
        assert len(top.evidence) > 0
        facts = [e.fact for e in top.evidence]
        assert any("readable" in f for f in facts)
        assert any("2D" in f for f in facts)
        assert any("continuous" in f for f in facts)
        assert any("observations" in f for f in facts)

    def test_pls_evidence_includes_reference(self) -> None:
        profile = self.make_profile()
        planner = RulePlanner()
        recs = planner.suggest(profile, reference_path="ref.csv")
        top = recs[0]
        facts = [e.fact for e in top.evidence]
        assert any("reference" in f for f in facts)
        assert any("PLS" in f for f in facts)

    def test_mcrals_evidence_includes_intent(self) -> None:
        profile = self.make_profile(shape=(5, 100))
        planner = RulePlanner()
        recs = planner.suggest(profile, intent="resolve")
        top = recs[0]
        facts = [e.fact for e in top.evidence]
        assert any("resolve" in f for f in facts)
        assert any("MCR-ALS" in f for f in facts) or any("mixture" in f for f in facts)

    def test_fallback_evidence_not_supportive(self) -> None:
        profile = DatasetProfile(
            path=Path("/broken.scp"),
            readable=False,
            summary="Cannot read file",
            error="File format not recognised",
        )
        planner = RulePlanner()
        recs = planner.suggest(profile)
        top = recs[0]
        # Fallback evidence should be non-supportive or empty
        if top.evidence:
            assert all(not e.supportive for e in top.evidence)

    def test_explain_contains_profile_details(self) -> None:
        profile = self.make_profile(shape=(10, 100))
        planner = RulePlanner()
        recs = planner.suggest(profile)
        top = recs[0]
        explanation = top.explain()
        assert "Recommended because" in explanation
        assert "✓" in explanation
        # Should mention profile-derived facts
        assert any(
            keyword in explanation for keyword in ["2D", "continuous", "observations"]
        )
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_explain_with_warnings(self) -> None:
        profile = self.make_profile(shape=(5, 100))
        planner = RulePlanner()
        recs = planner.suggest(profile)
        mcrals = [r for r in recs if r.template_id == "mcrals_analysis"][0]
        explanation = mcrals.explain()
        assert "Warnings" in explanation

    def test_evidence_dataclass(self) -> None:
        ev = RecommendationEvidence(fact="dataset is 2D", supportive=True)
        assert ev.fact == "dataset is 2D"
        assert ev.supportive is True

        ev2 = RecommendationEvidence(fact="few observations", supportive=False)
        assert ev2.supportive is False

    def test_explain_no_warnings(self) -> None:
        """Explain should not mention Warnings section when there are none."""
        rec = TemplateRecommendation(
            template_id="test",
            confidence=0.5,
            rationale="reason",
            dataset_summary="summary",
            evidence=[
                RecommendationEvidence(fact="good", supportive=True),
            ],
        )
        explanation = rec.explain()
        assert "Warnings" not in explanation
        assert "Considerations" not in explanation

    def test_explain_empty_evidence(self) -> None:
        """Recommendation with no evidence should produce empty string."""
        rec = TemplateRecommendation(
            template_id="test",
            confidence=0.5,
            rationale="reason",
            dataset_summary="summary",
        )
        assert rec.explain() == ""

    def test_explain_adverse_evidence(self) -> None:
        """Non-supportive evidence should appear under Considerations."""
        rec = TemplateRecommendation(
            template_id="test",
            confidence=0.3,
            rationale="reason",
            dataset_summary="summary",
            evidence=[
                RecommendationEvidence(fact="dataset is 1D", supportive=False),
            ],
        )
        explanation = rec.explain()
        assert "Considerations" in explanation
        assert "⚠" in explanation or "26A0" in explanation


# ---------------------------------------------------------------------------
# Public suggest() API — integration tests with real files
# ---------------------------------------------------------------------------


class TestSuggestAPI:
    def _create_2d_scp(self, tmp_path: Path, name: str = "spectra.scp") -> Path:
        import spectrochempy as scp

        data = np.random.randn(10, 100)
        d = scp.NDDataset(data, name="test")
        d.x = np.linspace(1000, 2000, 100)
        path = tmp_path / name
        d.save_as(str(path))
        return path

    def _create_reference_csv(self, tmp_path: Path) -> Path:
        path = tmp_path / "reference.csv"
        path.write_text("y\n1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n7.0\n8.0\n9.0\n10.0\n")
        return path

    def test_suggest_pca(self, tmp_path: Path) -> None:
        src = self._create_2d_scp(tmp_path)
        recs = suggest(str(src))
        assert len(recs) >= 1
        assert recs[0].template_id == "exploratory_pca"

    def test_suggest_pls_with_reference(self, tmp_path: Path) -> None:
        src = self._create_2d_scp(tmp_path)
        ref = self._create_reference_csv(tmp_path)
        recs = suggest(str(src), reference_path=str(ref))
        assert len(recs) >= 1
        assert recs[0].template_id == "pls_calibration"

    def test_suggest_mcrals_with_intent(self, tmp_path: Path) -> None:
        src = self._create_2d_scp(tmp_path)
        recs = suggest(str(src), intent="resolve")
        assert len(recs) >= 1
        assert recs[0].template_id == "mcrals_analysis"

    def test_suggest_non_existent_file(self) -> None:
        recs = suggest("/nonexistent/file.scp")
        assert len(recs) == 1
        assert recs[0].confidence == 0.3  # fallback confidence

    def test_suggest_roundtrip_explore(self, tmp_path: Path) -> None:
        """suggest() top recommendation should be explorable."""
        from spectrochempy_ai.exploration import explore

        src = self._create_2d_scp(tmp_path)
        recs = suggest(str(src))
        assert len(recs) >= 1
        top = recs[0]
        assert top.template_id == "exploratory_pca"

        output = explore(
            str(src),
            str(tmp_path / "roundtrip.ipynb"),
            template_id=top.template_id,
        )
        assert output.exists()
        assert output.suffix == ".ipynb"

    def test_suggest_baseline_integrate_for_single_spectrum(self, tmp_path: Path) -> None:
        import spectrochempy as scp

        data = np.random.randn(100)
        spectrum = scp.NDDataset(data, name="single_spectrum")
        spectrum.x = np.linspace(1000, 2000, 100)
        path = tmp_path / "single.scp"
        spectrum.save_as(str(path))

        recs = suggest(str(path))
        assert len(recs) >= 1
        assert recs[0].template_id == "baseline_integrate"
        assert recs[0].confidence == 0.85

    def test_suggest_baseline_integrate_for_single_row_spectrum(
        self, tmp_path: Path
    ) -> None:
        import spectrochempy as scp

        data = np.random.randn(1, 100)
        spectrum = scp.NDDataset(data, name="single_row")
        spectrum.x = np.linspace(1000, 2000, 100)
        spectrum.y = np.arange(1)
        path = tmp_path / "single_row.scp"
        spectrum.save_as(str(path))

        recs = suggest(str(path))
        assert len(recs) >= 1
        assert recs[0].template_id == "baseline_integrate"

    def test_suggest_uses_selected_multi_object_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import spectrochempy as scp

        path = tmp_path / "als2004dataset.MAT"
        path.write_text("dummy")
        datasets = scp.ScpObjectList(
            [
                scp.NDDataset(np.random.randn(4), name="meta"),
                scp.NDDataset(np.random.randn(4, 12), name="small"),
                scp.NDDataset(np.random.randn(55, 96), name="spectra"),
            ]
        )
        datasets[2].x = np.linspace(1000, 2000, 96)
        datasets[2].y = np.arange(55)
        monkeypatch.setattr(scp, "read", lambda _: datasets)

        recs = suggest(str(path))
        assert len(recs) >= 1
        assert recs[0].template_id == "exploratory_pca"
        assert recs[0].confidence == 0.7
        assert "multi-object" in recs[0].dataset_summary
        assert any("automatically selected" in fact.fact for fact in recs[0].evidence)
        assert any("automatically selected" in warning for warning in recs[0].warnings)

    def test_matlab_als2004dataset_regression_case(self) -> None:
        import spectrochempy as scp
        from spectrochempy import preferences as prefs

        path = prefs.datadir / "matlabdata" / "als2004dataset.MAT"
        if not path.exists():
            pytest.skip("MATLAB regression data not available")

        recs = suggest(str(path))
        assert len(recs) >= 1
        assert recs[0].confidence > 0.3
        assert recs[0].template_id in {"exploratory_pca", "mcrals_analysis"}
