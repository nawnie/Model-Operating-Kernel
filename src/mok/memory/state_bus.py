"""
src/mok/memory/state_bus.py

Inter-expert context passing (P5.4).

ExpertContext carries shared state across a multi-step task so each
expert in a chain can see what prior experts produced.  The runtime
assembles the context on the first call and threads it through each
subsequent expert call in the chain.

Design notes
------------
* Pure dataclass — no external dependencies.
* History format matches the ChatML / OpenAI messages schema so it can
  be appended directly to a backend payload's ``messages`` list.
* Artifacts are keyed text blobs (e.g. "code", "summary").  The runtime
  sets them; experts read them via the payload's ``context`` field.
* ``task_plan`` is set by the coordinator during decomposition;
  ``step_index`` tracks which step is currently executing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExpertContext:
    """
    Shared state passed between experts in a multi-step task.

    Fields
    ------
    request_id   : ties context to its originating request
    history      : list of {role, content} message dicts (ChatML style)
    artifacts    : free-form text outputs keyed by label
                   e.g. {"code": "def sort…", "summary": "Sorts a list…"}
    task_plan    : ordered list of expert roles from coordinator decomposition
                   e.g. ["code", "general", "vision"]
    step_index   : which step in task_plan is currently executing (0-based)
    metadata     : catch-all for extension fields
    """

    request_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    task_plan: list[str] | None = None
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the history."""
        self.history.append({"role": role, "content": content})

    def last_message(self) -> dict[str, str] | None:
        """Return the most recent history entry, or None if empty."""
        return self.history[-1] if self.history else None

    # ------------------------------------------------------------------
    # Artifact helpers
    # ------------------------------------------------------------------

    def set_artifact(self, key: str, content: str) -> None:
        """Store or overwrite a named text artifact."""
        self.artifacts[key] = content

    def get_artifact(self, key: str, default: str = "") -> str:
        """Retrieve a named artifact, returning *default* if not found."""
        return self.artifacts.get(key, default)

    # ------------------------------------------------------------------
    # Task-plan helpers
    # ------------------------------------------------------------------

    @property
    def current_step_role(self) -> str | None:
        """Role of the currently-executing step, or None if no plan."""
        if self.task_plan and 0 <= self.step_index < len(self.task_plan):
            return self.task_plan[self.step_index]
        return None

    @property
    def is_final_step(self) -> bool:
        """True when the current step is the last in the plan."""
        if not self.task_plan:
            return True
        return self.step_index >= len(self.task_plan) - 1

    def advance(self) -> "ExpertContext":
        """
        Increment step_index and return self (fluent).

        Safe to call past the end of task_plan — step_index will not
        exceed len(task_plan).
        """
        if self.task_plan:
            self.step_index = min(self.step_index + 1, len(self.task_plan))
        return self

    # ------------------------------------------------------------------
    # Serialisation (for tracing)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "history_len": len(self.history),
            "artifact_keys": list(self.artifacts.keys()),
            "task_plan": self.task_plan,
            "step_index": self.step_index,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        plan_str = f"step={self.step_index}/{len(self.task_plan)}" if self.task_plan else "no-plan"
        return (
            f"ExpertContext(request_id={self.request_id!r}, "
            f"history={len(self.history)}, "
            f"artifacts={list(self.artifacts.keys())!r}, "
            f"{plan_str})"
        )
