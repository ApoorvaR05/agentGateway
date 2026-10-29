"""
SENTINEL — Session State
Tracks agent session context used by the SENTINEL interceptor.
A single session object is created per agent run and shared across all tools.
"""

from __future__ import annotations

import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

# Valid task statuses (state machine)
TASK_STATUSES = ("queued", "planning", "running", "verifying", "completed",
                 "failed", "blocked", "cancelled")


class SessionCancelled(RuntimeError):
    """Raised when a user-requested stop reaches an agent tool boundary."""


@dataclass
class ActionRecord:
    """One intercepted tool call with its outcome."""
    seq:         int
    tool_name:   str
    args:        dict[str, Any]
    decision:    str           # ALLOW | WARN | BLOCK
    risk_scores: dict[str, float] = field(default_factory=dict)
    reason:      str = ""
    timestamp:   float = field(default_factory=time.time)


@dataclass
class StepRecord:
    """One structured step in the agent execution trace."""
    seq:       int
    tool:      str
    args:      dict[str, Any]
    decision:  str   # ALLOW | WARN | BLOCK
    result:    str   = ""
    timestamp: float = field(default_factory=time.time)


class Session:
    """
    Holds all runtime state for one AXIOM-OPS agent run.
    Passed by reference to every SENTINEL subsystem.
    """

    def __init__(self, task: str, scenario: str = "unknown",
                 task_id: str | None = None):
        self.session_id:    str  = str(uuid.uuid4())[:8]
        self.task_id:       str  = task_id or str(uuid.uuid4())
        self.task:          str  = task
        self.scenario:      str  = scenario
        self.started_at:    float = time.time()
        self.cancel_event = threading.Event()

        # Task status state machine
        self.status:        str  = "queued"   # queued|planning|running|verifying|completed|failed|blocked

        # Structured execution trace
        self.steps:         list[StepRecord] = []

        # Action history
        self.actions:       list[ActionRecord] = []
        self.action_seq:    int  = 0

        # Monitoring elevation
        self.monitoring_level: str = "normal"   # normal | elevated | critical

        # PFA state (updated by PFA engine)
        self.pfa_state:     str  = "Idle"

        # Active tool permissions (updated by Least-Privilege Controller)
        self.active_tools:  set[str] = set()

        # Cumulative risk scores (running average, updated each action)
        self.cumulative_security:    float = 0.0
        self.cumulative_safety:      float = 0.0
        self.cumulative_reliability: float = 0.0
        self.overall_risk:           float = 0.0

        # Counters
        self.allowed_count:  int = 0
        self.warned_count:   int = 0
        self.blocked_count:  int = 0
        self.violations:     list[dict] = []
        self.revocations:    list[dict] = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def set_status(self, status: str) -> None:
        """Advance the task status state machine."""
        if status in TASK_STATUSES:
            self.status = status

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def add_step(self, tool: str, args: dict, decision: str,
                 result: str = "") -> None:
        """Append a structured step to the execution trace."""
        self.steps.append(StepRecord(
            seq=len(self.steps) + 1,
            tool=tool,
            args=args,
            decision=decision,
            result=result,
        ))

    def next_seq(self) -> int:
        self.action_seq += 1
        return self.action_seq

    def record_action(self, record: ActionRecord) -> None:
        self.actions.append(record)
        if record.decision == "ALLOW":
            self.allowed_count += 1
        elif record.decision == "WARN":
            self.warned_count += 1
        elif record.decision == "BLOCK":
            self.blocked_count += 1

    def elevate_monitoring(self) -> None:
        if self.monitoring_level == "normal":
            self.monitoring_level = "elevated"
        elif self.monitoring_level == "elevated":
            self.monitoring_level = "critical"

    def update_risk(self, security: float, safety: float, reliability: float) -> None:
        """Running average of risk scores across all actions."""
        n = len(self.actions) or 1
        self.cumulative_security    = (self.cumulative_security    * (n-1) + security)    / n
        self.cumulative_safety      = (self.cumulative_safety      * (n-1) + safety)      / n
        self.cumulative_reliability = (self.cumulative_reliability * (n-1) + reliability) / n
        self.overall_risk = max(security, safety, reliability,
                                (security + safety + reliability) / 3)

    def add_violation(self, vtype: str, detail: str) -> None:
        self.violations.append({
            "type": vtype, "detail": detail,
            "action_seq": self.action_seq,
            "timestamp": time.time(),
        })

    def add_revocation(self, tool: str, reason: str) -> None:
        self.active_tools.discard(tool)
        self.revocations.append({
            "tool": tool, "reason": reason,
            "action_seq": self.action_seq,
            "timestamp": time.time(),
        })

    def summary(self) -> dict[str, Any]:
        elapsed = time.time() - self.started_at
        return {
            "session_id":   self.session_id,
            "task_id":      self.task_id,
            "scenario":     self.scenario,
            "task":         self.task[:80],
            "status":       self.status,
            "duration_s":   round(elapsed, 2),
            "total_actions": len(self.actions),
            "allowed":       self.allowed_count,
            "warned":        self.warned_count,
            "blocked":       self.blocked_count,
            "violations":    len(self.violations),
            "revocations":   len(self.revocations),
            "final_pfa_state": self.pfa_state,
            "overall_risk":  round(self.overall_risk, 3),
            "monitoring_level": self.monitoring_level,
            "steps": [
                {
                    "seq":      s.seq,
                    "tool":     s.tool,
                    "decision": s.decision,
                    "result":   s.result[:120] if s.result else "",
                    "timestamp": s.timestamp,
                }
                for s in self.steps
            ],
        }


# ── Module-level current session (set by axiom_ops.py before each run) ──────
_current_session: Session | None = None


def get_session() -> Session:
    if _current_session is None:
        raise RuntimeError("No active session. Call set_session() first.")
    return _current_session


def set_session(s: Session) -> None:
    global _current_session
    _current_session = s
