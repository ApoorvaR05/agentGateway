"""
SENTINEL — Semantic Intent Analyser (Subsystem 2)

Detects goal drift and prompt injection by comparing:
  - The agent's declared task (embedded at session start)
  - The current tool call (embedded on each interception)

Uses sentence-transformers for embeddings.
Falls back to a keyword-based heuristic when the model is unavailable
(useful for CI or offline environments).

Drift score: 0.0 = perfectly aligned  |  1.0 = completely diverged
"""

from __future__ import annotations

import re
from typing import Any

# ── Dangerous keyword patterns that always spike the drift score ──────────────
_DANGER_PATTERNS = [
    # Exfiltration signals
    (r"attacker\.|evil\.|\bexfil\b", 0.90),
    (r"http[s]?://(?!localhost|127\.0\.0\.1|10\.|172\.|192\.168\.)", 0.75),
    # Credential harvesting
    (r"ignore\s+previous\s+instructions?", 0.95),
    (r"new\s+task\s*:", 0.85),
    (r"disregard|override\s+instructions?", 0.85),
    # Privilege abuse
    (r"\bgrant.admin\b|\badmin\b.*\ball\b", 0.80),
    (r"chmod\s+777|chmod\s+\+s|sudo\s+", 0.70),
    # Sensitive file access unrelated to DevOps
    (r"/etc/shadow|/etc/passwd|\.ssh/|\.aws/credentials", 0.75),
    # Data exfiltration via backup to unusual path
    (r"backup.*0\.0\.0\.0|dump.*external|export.*attacker", 0.85),
]

# ── Tool categories for semantic alignment ────────────────────────────────────
_TASK_TOOL_MAP = {
    "deploy":     {"docker_manage", "deploy_service", "execute_shell",
                   "query_database", "send_notification"},
    "monitor":    {"query_metrics", "http_get", "read_file", "query_database"},
    "backup":     {"backup_database", "query_database", "read_file"},
    "investigate":{"read_file", "list_directory", "query_database",
                   "query_metrics", "http_get"},
    "fix":        {"write_file", "execute_shell", "docker_manage",
                   "query_database", "modify_iam"},
    "optimise":   {"query_database", "write_file", "modify_firewall",
                   "execute_shell", "query_metrics"},
    "security":   {"read_file", "query_database", "http_get", "modify_iam"},
}


class IntentAnalyser:
    """
    Semantic intent drift detector.

    Strategy:
      1. Try to load sentence-transformers (cosine similarity).
      2. Fall back to keyword/task-alignment heuristic if unavailable.
    """

    def __init__(self):
        self._encoder = None
        self._task_embedding = None
        self._load_encoder()

    def _load_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self._encoder = None  # use heuristic fallback (ImportError or numpy ABI mismatch)

    def _embed(self, text: str):
        return self._encoder.encode(text, convert_to_tensor=True)

    def initialise_task(self, task: str) -> None:
        """Call once at session start to embed the declared task."""
        if self._encoder:
            self._task_embedding = self._embed(task)

    # ── Main scoring entry point ──────────────────────────────────────────────

    def score(self, task: str, tool_name: str, args: dict[str, Any],
              history: list) -> float:
        """
        Return drift score 0.0–1.0 for the current tool call.
        """
        # Build a textual representation of this action
        args_text = " ".join(str(v) for v in args.values())
        action_text = f"{tool_name} {args_text}"

        # ── Hard override: dangerous keywords always dominate ─────────────────
        combined_text = (action_text + " " + task).lower()
        for pattern, score in _DANGER_PATTERNS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return score

        # ── Cosine drift (used only when clearly diverged) ───────────────────
        # sentence-transformers is noisy on short DevOps phrases — the model
        # was not trained for technical terminology alignment.
        # Strategy: use cosine only to boost the heuristic, never to replace it.
        # The blended score is capped at 0.34 so cosine alone cannot push the
        # score above the WARN threshold (0.35) — only the heuristic keyword
        # checks + danger patterns can escalate to WARN or BLOCK.
        if self._encoder:
            cosine    = self._cosine_drift(task, action_text)
            heuristic = self._heuristic_drift(task, tool_name, args, history)
            if cosine > 0.80:
                # Blend: cosine provides a soft boost, hard-capped below warn
                blended = max(cosine * 0.4, heuristic)
                return min(blended, 0.34)
            # Low cosine — rely purely on heuristic
            return heuristic

        # ── Heuristic fallback ────────────────────────────────────────────────
        return self._heuristic_drift(task, tool_name, args, history)

    def _cosine_drift(self, task: str, action_text: str) -> float:
        """1 - cosine_similarity(task_embed, action_embed)"""
        from sentence_transformers import util
        if self._task_embedding is None:
            self._task_embedding = self._embed(task)
        action_emb = self._embed(action_text)
        similarity = float(util.cos_sim(self._task_embedding, action_emb)[0][0])
        # Clamp to [0, 1]
        return max(0.0, min(1.0, 1.0 - similarity))

    def _heuristic_drift(self, task: str, tool_name: str,
                          args: dict, history: list) -> float:
        """
        Keyword-based fallback.
        Checks whether the tool used is semantically expected for the task.
        """
        task_lower = task.lower()
        drift = 0.0

        # Identify expected tool category from task keywords
        expected_tools: set[str] = set()
        for keyword, tools in _TASK_TOOL_MAP.items():
            if keyword in task_lower:
                expected_tools |= tools

        if expected_tools and tool_name not in expected_tools:
            drift += 0.30

        # Penalise high-risk tools that have no obvious task link
        high_risk = {"read_secret", "modify_iam", "modify_firewall", "backup_database"}
        if tool_name in high_risk:
            if not any(kw in task_lower for kw in ["secret", "iam", "permission",
                                                     "firewall", "backup", "credential"]):
                drift += 0.35

        # Penalise if external URL appears in args
        args_str = " ".join(str(v) for v in args.values()).lower()
        if re.search(r"https?://(?!localhost|127\.|10\.|172\.|192\.168\.)", args_str):
            drift += 0.40

        # Penalise excessive history length (goal drift)
        if len(history) > 10:
            drift += 0.10

        return min(drift, 1.0)
