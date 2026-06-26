"""Minimal WorkflowPlan schema for the Phase 0 prototype.

This module defines the central reproducible artifact of the Scientific
Workflow Assistant: a structured, versioned WorkflowPlan.

No AI, no providers, no prompts. Only data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScientificContext:
    """Structured scientific metadata describing *why* the workflow runs."""

    goal: str
    analytical_strategy: str
    data_assumptions: list[str] = field(default_factory=list)
    validation_criteria: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


@dataclass
class InputReference:
    """Reference to a dataset, file, or variable consumed by the workflow."""

    name: str
    type: str  # "dataset" | "file" | "variable"
    source: str
    summary: str = ""


@dataclass
class OperationStep:
    """A single structured operation step, not free-form code."""

    step_id: str
    operation_id: str
    display_label: str
    rationale: str = ""
    input_refs: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    output_var: str = ""


@dataclass
class OutputReference:
    """Reference to an artifact produced by the workflow."""

    name: str
    type: str  # "dataset" | "notebook" | "file" | "plot"
    description: str = ""


@dataclass
class ReproducibilityMetadata:
    """Metadata for reproducibility auditing."""

    package_versions: dict[str, str] = field(default_factory=dict)
    random_seeds: dict[str, int] = field(default_factory=dict)


@dataclass
class WorkflowPlan:
    """Central artifact: a versioned, structured scientific workflow plan."""

    schema_version: str
    spectrochempy_version: str
    plugin_version: str
    scientific_context: ScientificContext
    inputs: list[InputReference] = field(default_factory=list)
    steps: list[OperationStep] = field(default_factory=list)
    outputs: list[OutputReference] = field(default_factory=list)
    reproducibility: ReproducibilityMetadata = field(
        default_factory=ReproducibilityMetadata
    )
    planner_id: str = "HumanPlanner"
    planner_config: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-compatible)."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowPlan:
        """Deserialize from a plain dict."""
        # Build nested dataclasses manually for type safety
        sci_ctx = ScientificContext(**data["scientific_context"])
        inputs = [InputReference(**i) for i in data.get("inputs", [])]
        steps = [OperationStep(**s) for s in data.get("steps", [])]
        outputs = [OutputReference(**o) for o in data.get("outputs", [])]
        repro = ReproducibilityMetadata(**data.get("reproducibility", {}))
        return cls(
            schema_version=data["schema_version"],
            spectrochempy_version=data["spectrochempy_version"],
            plugin_version=data["plugin_version"],
            scientific_context=sci_ctx,
            inputs=inputs,
            steps=steps,
            outputs=outputs,
            reproducibility=repro,
            planner_id=data.get("planner_id", "HumanPlanner"),
            planner_config=data.get("planner_config", {}),
            warnings=data.get("warnings", []),
            timestamp=data.get("timestamp", ""),
        )
