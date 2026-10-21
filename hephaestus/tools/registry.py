"""Tool schemas (``TOOLS``), dispatch, ``ToolError`` and the per-run ``Workspace``.

The contract between the harness and the tools; see ``docs/TOOLS.md``.
"""


class ToolError(Exception):
    """A tool failure reported to the LLM as ``{"error": "..."}``, never raised past dispatch."""
