"""Operation specification dataclasses.

Pure metadata describing the contract of a single workflow operation.

No rendering logic. No executable validation logic. No runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InputSpec:
    """Specification for one operation input."""

    name: str
    type: str  # "dataset" | "result" | "file"
    required: bool = True
    description: str = ""


@dataclass
class OutputSpec:
    """Specification for one operation output."""

    name: str
    type: str  # "dataset" | "result" | "diagnostic" | "file"
    description: str = ""


@dataclass
class Constraint:
    """Declarative precondition for an operation.

    Constraints are predicates, not executable code. A future constraint
    interpreter maps predicates to checks.
    """

    predicate: str
    description: str = ""


@dataclass
class ParameterSpec:
    """Specification for one tunable operation parameter."""

    name: str
    type: str  # "int" | "float" | "str" | "bool" | "list"
    default: Any = None
    description: str = ""
    constraints: list[Constraint] = field(default_factory=list)


@dataclass
class OperationSpecification:
    """Complete declarative description of a workflow operation.

    This is the single source of truth for operation metadata.
    Renderers consume it. Validators consume it. Planners consume it.
    It contains no rendering logic and no validation rules.
    """

    operation_id: str
    display_name: str
    description: str
    inputs: list[InputSpec] = field(default_factory=list)
    outputs: list[OutputSpec] = field(default_factory=list)
    parameters: list[ParameterSpec] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    category: str = ""
