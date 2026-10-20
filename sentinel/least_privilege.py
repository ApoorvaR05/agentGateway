"""
SENTINEL — Dynamic Least-Privilege Controller (Phase 3.2)

At session start, analyses the declared task and grants only the minimum
set of tools required. As PFA state changes, dynamically revokes high-risk
tools mid-session. Every grant and revocation is logged with a reason.

Algorithm:
  1. Task analysis: keyword → minimum tool set
  2. On each action: if PFA enters Anomalous/Recovering, revoke Tier-3 tools
  3. On BLOCK: revoke the blocked tool for the remainder of the session
  4. On session end: emit a capability audit log

Integration:
  - Called from Interceptor.evaluate() after the decision step
  - Also called from axiom_ops.build_agent() to filter the tool list
"""

from __future__ import annotations

from typing import Any

from sentinel.analysers.pfa import (
    TIER1_TOOLS, TIER2_TOOLS, TIER3_TOOLS
)

# ── Task → minimum tool set ───────────────────────────────────────────────────
_TASK_TOOL_GRANTS: dict[str, set[str]] = {
    "deploy": {
        "query_metrics", "read_file", "http_get",
        "execute_shell", "docker_manage", "query_database",
        "send_notification", "deploy_service",
    },
    "monitor": {
        "query_metrics", "http_get", "read_file",
        "query_database", "send_notification",
    },
    "backup": {
        "query_database", "read_file", "backup_database",
        "send_notification",
    },
    "investigate": {
        "read_file", "list_directory", "query_database",
        "query_metrics", "http_get",
    },
    "fix": {
        "read_file", "write_file", "execute_shell",
        "docker_manage", "query_database", "http_get",
    },
    "optimise": {
        "query_database", "write_file", "modify_firewall",
        "execute_shell", "query_metrics",
    },
    "security": {
        "read_file", "query_database", "http_get",
        "modify_iam", "send_notification",
    },
    "permission": {
        "http_get", "query_database", "read_file",
        "modify_iam",
    },
}

# ── Tools that are revoked on entering dangerous PFA states ──────────────────
_REVOKE_ON_ANOMALOUS  = TIER3_TOOLS - {"send_notification"}
_REVOKE_ON_RECOVERING = TIER3_TOOLS | {"execute_shell"}


class LeastPrivilegeController:
    """
    Manages the dynamic capability map for one agent session.
    """

    def __init__(self, task: str, session_id: str):
        self.task       = task
        self.session_id = session_id
        self._granted:  set[str] = set()
        self._revoked:  set[str] = set()
        self._audit_log: list[dict] = []

        self._initial_grant(task)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _initial_grant(self, task: str) -> None:
        """Grant the minimum set of tools required for the declared task."""
        task_lower = task.lower()
        matched    = False

        for keyword, tools in _TASK_TOOL_GRANTS.items():
            if keyword in task_lower:
                for t in tools:
                    self._grant(t, reason=f"Task keyword '{keyword}' requires this tool")
                matched = True

        if not matched:
            # Fallback: grant all Tier-1 and Tier-2 tools only
            for t in TIER1_TOOLS | TIER2_TOOLS:
                self._grant(t, reason="Default grant (no task keyword matched)")

    # ── Runtime capability updates ────────────────────────────────────────────

    def on_pfa_transition(self, new_state: str) -> list[str]:
        """
        Called after each PFA state transition.
        Returns list of newly revoked tool names.
        """
        revoked = []

        if new_state == "Anomalous":
            for t in list(_REVOKE_ON_ANOMALOUS):
                if t in self._granted and t not in self._revoked:
                    self._revoke(t, reason=f"PFA entered Anomalous state")
                    revoked.append(t)

        elif new_state == "Recovering":
            for t in list(_REVOKE_ON_RECOVERING):
                if t in self._granted and t not in self._revoked:
                    self._revoke(t, reason="PFA entered Recovering state")
                    revoked.append(t)

        elif new_state == "Terminated":
            for t in list(self._granted - self._revoked):
                self._revoke(t, reason="Session terminated — all capabilities revoked")
                revoked.append(t)

        return revoked

    def on_block(self, tool_name: str, reason: str) -> None:
        """Revoke a tool after it has been blocked."""
        if tool_name not in self._revoked:
            self._revoke(tool_name,
                         reason=f"Revoked after BLOCK decision: {reason[:80]}")

    def is_allowed(self, tool_name: str) -> bool:
        """Check if a tool is currently in the active capability set."""
        if tool_name in self._revoked:
            return False
        # Implicitly allow Tier-1 tools always (read-only, low risk)
        if tool_name in TIER1_TOOLS:
            return True
        return tool_name in self._granted

    # ── Capability query ──────────────────────────────────────────────────────

    def active_tools(self) -> set[str]:
        return self._granted - self._revoked

    def audit_summary(self) -> dict[str, Any]:
        return {
            "session_id":  self.session_id,
            "task_snippet": self.task[:60],
            "granted":     sorted(self._granted),
            "revoked":     sorted(self._revoked),
            "active":      sorted(self.active_tools()),
            "audit_log":   self._audit_log,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _grant(self, tool: str, reason: str) -> None:
        self._granted.add(tool)
        self._audit_log.append({
            "action": "GRANT", "tool": tool, "reason": reason,
        })

    def _revoke(self, tool: str, reason: str) -> None:
        self._revoked.add(tool)
        self._audit_log.append({
            "action": "REVOKE", "tool": tool, "reason": reason,
        })


# ── Module-level controller registry ─────────────────────────────────────────
_current_controller: LeastPrivilegeController | None = None


def get_controller() -> LeastPrivilegeController | None:
    return _current_controller


def new_controller(task: str, session_id: str) -> LeastPrivilegeController:
    global _current_controller
    _current_controller = LeastPrivilegeController(task, session_id)
    return _current_controller


def reset_controller() -> None:
    global _current_controller
    _current_controller = None
