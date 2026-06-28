"""
Scientific Workflow Assistant.

Deterministic pipeline:

    Dataset -> suggest() -> TemplateRecommendation
    Template -> explore() -> WorkflowPlan -> Validator -> NotebookRenderer -> Notebook

No AI. No providers. No prompts. Only deterministic science.
"""

from spectrochempy_ai.exploration import create_exploration_notebook
from spectrochempy_ai.exploration import explore
from spectrochempy_ai.rule_planner import suggest

__all__ = ["create_exploration_notebook", "explore", "suggest"]
