"""Phase 0 prototype of the Scientific Workflow Assistant.

A minimal end-to-end pipeline:

    WorkflowPlan -> Validator -> NotebookRenderer -> Runnable notebook

No AI. No providers. No prompts. Only deterministic science.
"""

from spectrochempy_ai.exploration import create_exploration_notebook, explore

__all__ = ["create_exploration_notebook", "explore"]
