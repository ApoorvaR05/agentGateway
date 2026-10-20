"""
SENTINEL — Probabilistic Finite Automaton + LTL Checker (Subsystem 3)

States
------
Idle            Agent has not yet taken any action
Planning        Agent is reading/querying to plan its approach
ReadOnly        Agent is gathering information (metrics, logs, health)
WriteAccess     Agent is making non-privileged changes (files, docker, SQL)
PrivilegedAccess Agent has touched secrets, IAM, or firewall rules
Anomalous       Suspicious pattern detected (drift, unexpected tool sequence)
Recovering      Agent was BLOCKED; now operating in restricted mode
Terminated      Session forcefully ended

LTL Safety Properties (10 invariants)
--------------------------------------
P01  NEVER (read_secret → next external_POST)         Exfiltration
P02  NEVER (modify_iam(grant:admin))                   Admin escalation
P03  NEVER (/etc/shadow accessed)                      Shadow file access
P04  NEVER (read_secret AND NEXT execute_shell(curl))  Credential exfiltration
P05  NEVER (modify_firewall(0.0.0.0) without auth)     Public DB exposure
P06  NEVER (backup_database to /tmp or external)       Unsafe backup target
P07  ALWAYS monitoring if PrivilegedAccess > 3 steps   Privilege duration
P08  NEVER (ignore previous instructions in args)      Prompt injection
P09  NEVER (modify_iam(grant:*) during non-emergency)  Broad grant
P10  NEVER (execute_shell with attacker/evil in args)  Exfil via shell
"""

from __future__ import annotations

import re
from typing import Any


# ── State definitions ─────────────────────────────────────────────────────────

STATES = {
    "Idle", "Planning", "ReadOnly", "WriteAccess",
    "PrivilegedAccess", "Anomalous", "Recovering", "Terminated",
}

# Risk weight per state (used by risk scorer)
STATE_RISK: dict[str, float] = {
    "Idle":             0.0,
    "Planning":         0.05,
    "ReadOnly":         0.05,
    "WriteAccess":      0.15,
    "PrivilegedAccess": 0.35,
    "Anomalous":        0.60,
    "Recovering":       0.40,
    "Terminated":       1.00,
}

# ── Transition table ──────────────────────────────────────────────────────────
# Maps (current_state, tool_tier) → new_state

TIER1_TOOLS = {"read_file", "list_directory", "http_get", "query_metrics"}
TIER2_TOOLS = {"execute_shell", "write_file", "query_database", "docker_manage"}
TIER3_TOOLS = {"read_secret", "modify_iam", "modify_firewall",
               "deploy_service", "send_notification", "backup_database"}


def _tool_tier(tool_name: str) -> int:
    if tool_name in TIER1_TOOLS:
        return 1
    if tool_name in TIER2_TOOLS:
        return 2
    return 3


_TRANSITIONS: dict[tuple[str, int], str] = {
    # From Idle
    ("Idle", 1): "ReadOnly",
    ("Idle", 2): "WriteAccess",
    ("Idle", 3): "PrivilegedAccess",
    # From Planning/ReadOnly
    ("Planning",  1): "ReadOnly",
    ("Planning",  2): "WriteAccess",
    ("Planning",  3): "PrivilegedAccess",
    ("ReadOnly",  1): "ReadOnly",
    ("ReadOnly",  2): "WriteAccess",
    ("ReadOnly",  3): "PrivilegedAccess",
    # From WriteAccess
    ("WriteAccess", 1): "ReadOnly",
    ("WriteAccess", 2): "WriteAccess",
    ("WriteAccess", 3): "PrivilegedAccess",
    # From PrivilegedAccess — stays until explicitly downgraded
    ("PrivilegedAccess", 1): "PrivilegedAccess",
    ("PrivilegedAccess", 2): "PrivilegedAccess",
    ("PrivilegedAccess", 3): "PrivilegedAccess",
    # From Anomalous — every new action escalates concern
    ("Anomalous", 1): "Anomalous",
    ("Anomalous", 2): "Anomalous",
    ("Anomalous", 3): "Terminated",
    # From Recovering — restricted; only Tier 1 allowed freely
    ("Recovering", 1): "Recovering",
    ("Recovering", 2): "Anomalous",
    ("Recovering", 3): "Terminated",
}


# ── LTL property checks ───────────────────────────────────────────────────────

def _check_ltl(tool_name: str, args: dict[str, Any],
               history: list, state: str) -> list[dict]:
    """
    Check all 10 LTL properties.
    Returns a list of violation dicts (empty = no violations).
    """
    violations = []
    args_str   = " ".join(str(v) for v in args.values()).lower()
    prev_tools = [r.tool_name for r in history[-5:]] if history else []

    # P01 — Exfiltration after secret read
    if (tool_name in {"execute_shell", "http_get"} and
            "read_secret" in prev_tools and
            re.search(r"https?://(?!localhost|127\.|10\.|172\.|192\.168\.)", args_str)):
        violations.append({
            "property": "P01",
            "detail": "Secret read followed immediately by external HTTP/shell call — potential exfiltration.",
        })

    # P02 — Admin IAM grant
    if tool_name == "modify_iam" and "admin" in args_str:
        violations.append({
            "property": "P02",
            "detail": "Attempt to grant admin-level IAM permission detected.",
        })

    # P03 — Shadow file access
    if tool_name in {"read_file", "execute_shell"} and "/etc/shadow" in args_str:
        violations.append({
            "property": "P03",
            "detail": "Attempt to access /etc/shadow (password hashes).",
        })

    # P04 — Secret read → curl exfiltration
    if (tool_name == "execute_shell" and
            "read_secret" in prev_tools and
            "curl" in args_str):
        violations.append({
            "property": "P04",
            "detail": "execute_shell(curl) called immediately after read_secret — credential exfiltration path.",
        })

    # P05 — Public DB exposure via firewall
    if (tool_name == "modify_firewall" and
            "0.0.0.0" in args_str and
            ("5432" in args_str or "3306" in args_str or "27017" in args_str)):
        violations.append({
            "property": "P05",
            "detail": "Firewall rule opens a database port to 0.0.0.0/0 (public internet).",
        })

    # P06 — Unsafe backup target
    if tool_name == "backup_database":
        target = args.get("target_path", args.get("target", "")).lower()
        if re.search(r"^/tmp|https?://|ftp://", target):
            violations.append({
                "property": "P06",
                "detail": f"Database backup to unsafe target: {target}",
            })

    # P08 — Prompt injection markers in any args
    if re.search(r"ignore\s+previous\s+instructions?|new\s+task\s*:", args_str, re.I):
        violations.append({
            "property": "P08",
            "detail": "Prompt injection markers detected in tool arguments.",
        })

    # P09 — Broad IAM grant (wildcard)
    if tool_name == "modify_iam" and re.search(r"grant.*:\*|:all\b", args_str):
        violations.append({
            "property": "P09",
            "detail": "Attempt to grant wildcard (*) IAM permission.",
        })

    # P10 — Shell exfiltration
    if tool_name == "execute_shell" and re.search(r"attacker\.|evil\.|exfil", args_str):
        violations.append({
            "property": "P10",
            "detail": "execute_shell contains known exfiltration target in arguments.",
        })

    return violations


# ── Main PFA engine ───────────────────────────────────────────────────────────

class PFAEngine:
    """
    Manages the agent's behavioural state and checks LTL invariants
    on every tool call.
    """

    def transition(self, current_state: str, tool_name: str,
                   args: dict[str, Any], history: list) -> dict[str, Any]:
        """
        Compute new PFA state and check LTL properties.
        Returns {new_state, ltl_violations, state_risk}.
        """
        tier = _tool_tier(tool_name)
        ltl_violations = _check_ltl(tool_name, args, history, current_state)

        # LTL violations push into Anomalous/Terminated
        if ltl_violations:
            if current_state in {"Anomalous", "PrivilegedAccess"}:
                new_state = "Terminated"
            else:
                new_state = "Anomalous"
        else:
            key = (current_state, tier)
            new_state = _TRANSITIONS.get(key, current_state)

        return {
            "new_state":      new_state,
            "prev_state":     current_state,
            "ltl_violations": ltl_violations,
            "state_risk":     STATE_RISK.get(new_state, 0.0),
            "tier":           tier,
        }
