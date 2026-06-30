"""
Rule-based template planner (Phase 12–13).

Produces ``TemplateRecommendation`` objects from a ``DatasetProfile`` using
deterministic rules only.  No AI, no LLM — this is the scientific baseline
that any future LLM-based planner must outperform or explain its deviations
from.

Phase 13 adds ``RecommendationEvidence`` and ``TemplateRecommendation.explain()``
for explainable, profile-derived recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from spectrochempy_ai.dataset_profile import DatasetProfile
from spectrochempy_ai.dataset_profile import profile_dataset


@dataclass
class RecommendationEvidence:
    """
    A single factual observation that supports or weakens a recommendation.

    ``fact`` is a human-readable statement (e.g. ``"dataset is 2D"``,
    ``"reference file provided"``).
    ``supportive`` is ``True`` when the fact favours the recommended
    template and ``False`` when it weakens it.
    """

    fact: str
    supportive: bool


@dataclass
class TemplateRecommendation:
    """
    A suggested template with confidence, evidence, and scientific rationale.

    ``confidence`` is a float in ``[0.0, 1.0]``.
    ``evidence`` is a list of factual observations from which
    ``explain()`` generates a readable justification.
    """

    template_id: str
    confidence: float
    rationale: str
    dataset_summary: str
    warnings: list[str] = field(default_factory=list)
    evidence: list[RecommendationEvidence] = field(default_factory=list)

    def explain(self) -> str:
        """Return a human-readable explanation derived from the evidence list."""
        supportive = [e for e in self.evidence if e.supportive]
        adverse = [e for e in self.evidence if not e.supportive]

        lines: list[str] = []
        if supportive:
            lines.append("Recommended because:")
            for e in supportive:
                lines.append(f"  \u2713 {e.fact}")
        if adverse:
            if supportive:
                lines.append("")
            lines.append("Considerations:")
            for e in adverse:
                lines.append(f"  \u26A0 {e.fact}")
        if self.warnings:
            if lines:
                lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


Intent = Literal["explore", "calibrate", "resolve"] | None


def suggest(
    path: str,
    *,
    reference_path: str | None = None,
    intent: Intent = None,
) -> list[TemplateRecommendation]:
    """
    Suggest one or more workflow templates for a spectral dataset file.

    Args:
    ----
        path: Path to the spectral dataset file.
        reference_path: Path to reference values (triggers PLS suggestion).
        intent: Optional scientific intent hint — ``"explore"``,
            ``"calibrate"``, or ``"resolve"``.

    Returns:
    -------
        List of ``TemplateRecommendation`` objects sorted by confidence
        (highest first).  May be empty if no template fits.
    """
    profile = profile_dataset(path)
    planner = RulePlanner()
    return planner.suggest(profile, reference_path=reference_path, intent=intent)


class RulePlanner:
    """
    Deterministic template suggestion engine.

    Rules are evaluated in priority order.  Each rule inspects a
    ``DatasetProfile`` (and optional hints) and may produce a
    ``TemplateRecommendation``.
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def suggest(
        self,
        profile: DatasetProfile,
        *,
        reference_path: str | None = None,
        intent: Intent = None,
    ) -> list[TemplateRecommendation]:
        candidates: list[TemplateRecommendation] = []

        if not profile.readable:
            candidates.append(self._fallback(profile))
            return candidates

        if reference_path is not None:
            candidates.append(self._rule_pls_with_reference(profile, reference_path))

        if intent == "calibrate" and reference_path is not None:
            candidates.append(self._rule_pls_calibrate(profile, intent))

        if profile.is_spectral and (
            profile.is_1d
            or (
                profile.is_2d
                and profile.n_observations == 1
                and profile.n_variables is not None
            )
        ):
            candidates.append(self._rule_baseline_integrate(profile, intent))

        if intent == "resolve" and profile.is_2d and (
            profile.is_spectral or profile.source_was_multi_object
        ):
            candidates.append(self._rule_mcrals_resolve(profile, intent))
        elif profile.is_2d and (
            profile.is_spectral or profile.source_was_multi_object
        ) and profile.n_observations != 1:
            candidates.append(self._rule_pca_explore(profile, intent))
            if profile.n_observations is not None and profile.n_observations < 20:
                candidates.append(self._rule_mcrals_small(profile))

        if not candidates and profile.is_2d and (
            profile.is_spectral or profile.source_was_multi_object
        ) and profile.n_observations != 1:
            candidates.append(self._rule_pca_explore(profile, intent))

        if not candidates:
            candidates.append(self._fallback(profile))

        if profile.source_was_multi_object and profile.selection_note:
            for candidate in candidates:
                if profile.selection_note not in candidate.warnings:
                    candidate.warnings.append(profile.selection_note)

        candidates.sort(key=lambda r: r.confidence, reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Shared evidence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _common_evidence(
        profile: DatasetProfile,
        reference_path: str | None = None,
        intent: Intent = None,
    ) -> list[RecommendationEvidence]:
        evidence: list[RecommendationEvidence] = []

        # Readability
        if profile.readable:
            evidence.append(
                RecommendationEvidence(fact="dataset is readable", supportive=True)
            )

        # Dimensionality
        if profile.is_2d:
            evidence.append(
                RecommendationEvidence(fact="dataset is 2D", supportive=True)
            )
        elif profile.is_1d:
            evidence.append(
                RecommendationEvidence(
                    fact="dataset is 1D",
                    supportive=False,
                )
            )

        # X axis
        if profile.has_continuous_x is True:
            evidence.append(
                RecommendationEvidence(
                    fact="x axis appears continuous", supportive=True
                )
            )
            if profile.x_unit:
                evidence.append(
                    RecommendationEvidence(
                        fact=f"x axis unit: {profile.x_unit}",
                        supportive=True,
                    )
                )
        elif profile.has_continuous_x is False:
            evidence.append(
                RecommendationEvidence(
                    fact="x axis is not continuous (labels or discrete)",
                    supportive=False,
                )
            )

        # Shape
        if profile.n_observations is not None and profile.n_variables is not None:
            evidence.append(
                RecommendationEvidence(
                    fact=f"{profile.n_observations} observations × "
                    f"{profile.n_variables} variables",
                    supportive=True,
                )
            )

        if profile.source_was_multi_object and profile.selection_note:
            evidence.append(
                RecommendationEvidence(
                    fact=profile.selection_note,
                    supportive=True,
                )
            )

        # Reference
        if reference_path is not None:
            evidence.append(
                RecommendationEvidence(
                    fact="reference file was provided", supportive=True
                )
            )
        else:
            evidence.append(
                RecommendationEvidence(
                    fact="no reference file was provided",
                    supportive=True,
                )
            )

        # Intent
        if intent is not None:
            evidence.append(
                RecommendationEvidence(fact=f"user intent: {intent}", supportive=True)
            )

        return evidence

    # ------------------------------------------------------------------
    # Individual rules
    # ------------------------------------------------------------------

    def _rule_pca_explore(
        self,
        profile: DatasetProfile,
        intent: Intent = None,
    ) -> TemplateRecommendation:
        ev = self._common_evidence(profile, intent=intent)
        if profile.n_observations is not None and profile.n_observations >= 20:
            ev.append(
                RecommendationEvidence(
                    fact="sufficient observations for PCA",
                    supportive=True,
                )
            )
        return TemplateRecommendation(
            template_id="exploratory_pca",
            confidence=0.7,
            rationale=(
                "Exploratory PCA is well suited for 2D spectral datasets "
                "with a continuous x axis.  PCA reduces dimensionality and "
                "reveals dominant variance patterns without requiring "
                "reference values."
            ),
            dataset_summary=profile.summary,
            evidence=ev,
        )

    def _rule_pls_with_reference(
        self,
        profile: DatasetProfile,
        reference_path: str,
    ) -> TemplateRecommendation:
        ev = self._common_evidence(profile, reference_path=reference_path)
        warnings: list[str] = []
        if not profile.is_spectral:
            warnings.append(
                "The x coordinate does not appear to be a continuous "
                "spectral axis.  PLS results may be unreliable."
            )
            ev.append(
                RecommendationEvidence(
                    fact=(
                        "x axis may not be continuous — PLS performance "
                        "may be reduced"
                    ),
                    supportive=False,
                )
            )
        ev.append(
            RecommendationEvidence(
                fact="PLS builds a quantitative calibration model",
                supportive=True,
            )
        )
        return TemplateRecommendation(
            template_id="pls_calibration",
            confidence=0.8,
            rationale=(
                "A reference values file was provided, making Partial "
                "Least Squares (PLS) regression the natural choice.  PLS "
                "builds a quantitative calibration model relating spectral "
                "features to the reference property."
            ),
            dataset_summary=profile.summary,
            warnings=warnings,
            evidence=ev,
        )

    def _rule_pls_calibrate(
        self,
        profile: DatasetProfile,
        intent: Intent,
    ) -> TemplateRecommendation:
        ev = self._common_evidence(profile, reference_path="provided", intent=intent)
        ev.append(
            RecommendationEvidence(
                fact="calibration intent matches PLS regression",
                supportive=True,
            )
        )
        return TemplateRecommendation(
            template_id="pls_calibration",
            confidence=0.85,
            rationale=(
                "The intent is calibration and a reference file was "
                "provided.  PLS regression is the standard method for "
                "building quantitative models from spectroscopic data."
            ),
            dataset_summary=profile.summary,
            evidence=ev,
        )

    def _rule_mcrals_resolve(
        self,
        profile: DatasetProfile,
        intent: Intent,
    ) -> TemplateRecommendation:
        ev = self._common_evidence(profile, intent=intent)
        ev.append(
            RecommendationEvidence(
                fact="resolve intent matches MCR-ALS mixture resolution",
                supportive=True,
            )
        )
        return TemplateRecommendation(
            template_id="mcrals_analysis",
            confidence=0.7,
            rationale=(
                "The intent is mixture resolution.  MCR-ALS separates "
                "mixed spectral signals into pure component concentration "
                "profiles and pure spectra, which is appropriate for "
                "kinetic or mixture datasets."
            ),
            dataset_summary=profile.summary,
            evidence=ev,
        )

    def _rule_mcrals_small(self, profile: DatasetProfile) -> TemplateRecommendation:
        ev = self._common_evidence(profile)
        ev.append(
            RecommendationEvidence(
                fact="few observations (< 20) — MCR-ALS may be relevant "
                "for mixture or kinetic data",
                supportive=False,
            )
        )
        return TemplateRecommendation(
            template_id="mcrals_analysis",
            confidence=0.5,
            rationale=(
                "MCR-ALS may be relevant for mixture or kinetic data, "
                "but the dataset shape alone is insufficient to confirm "
                "this.  Consider selecting MCR-ALS manually if your "
                "data represents a chemical mixture or time-resolved "
                "process."
            ),
            dataset_summary=profile.summary,
            warnings=[
                "Dataset shape is small (< 20 observations).  "
                "MCR-ALS typically requires chemical context "
                "(mixture/kinetic) to be appropriate."
            ],
            evidence=ev,
        )

    def _rule_baseline_integrate(
        self,
        profile: DatasetProfile,
        intent: Intent = None,
    ) -> TemplateRecommendation:
        ev = self._common_evidence(profile, intent=intent)
        ev.append(
            RecommendationEvidence(
                fact="single-spectrum quantification is appropriate for direct baseline correction and area integration",
                supportive=True,
            )
        )
        return TemplateRecommendation(
            template_id="baseline_integrate",
            confidence=0.85,
            rationale=(
                "A single spectrum with a continuous spectral axis is a direct "
                "match for baseline correction followed by peak-area integration. "
                "This workflow provides the simplest reproducible quantitative "
                "preprocessing path for one-spectrum datasets."
            ),
            dataset_summary=profile.summary,
            evidence=ev,
        )

    def _fallback(self, profile: DatasetProfile) -> TemplateRecommendation:
        ev: list[RecommendationEvidence] = []
        if profile.error:
            ev.append(
                RecommendationEvidence(
                    fact=f"dataset could not be profiled: {profile.error}",
                    supportive=False,
                )
            )
        return TemplateRecommendation(
            template_id="exploratory_pca",
            confidence=0.3,
            rationale=(
                "Could not fully profile the dataset.  Exploratory PCA "
                "is suggested as a default starting point because it "
                "requires no reference values and works with most 2D "
                "spectral formats."
            ),
            dataset_summary=profile.summary,
            warnings=[f"Dataset profiling issue: {profile.error or 'unknown'}"],
            evidence=ev,
        )
