"""
SENTINEL — SQLite Persistence Layer
Zero external dependencies — uses stdlib sqlite3 only.
Tables: tasks, decisions, audit_log, incidents
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any

# ── DB location — placed in project root ──────────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "sentinel_ops.db")
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    """Open a connection with WAL mode and row_factory for dict-like rows."""
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── Schema creation ────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id      TEXT PRIMARY KEY,
    objective    TEXT NOT NULL,
    scenario     TEXT NOT NULL DEFAULT 'unknown',
    status       TEXT NOT NULL DEFAULT 'queued',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id         TEXT,
    task_id          TEXT,
    session_id       TEXT,
    seq              INTEGER,
    tool             TEXT,
    args_json        TEXT,
    decision         TEXT,
    scores_json      TEXT,
    reason           TEXT,
    pfa_state        TEXT,
    drift            REAL,
    ltl_violations_json TEXT,
    timestamp        REAL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     TEXT,
    task_id      TEXT,
    action       TEXT,
    details_json TEXT,
    prev_hash    TEXT,
    curr_hash    TEXT,
    timestamp    REAL
);

CREATE TABLE IF NOT EXISTS incidents (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    severity             TEXT NOT NULL DEFAULT 'medium',
    status               TEXT NOT NULL DEFAULT 'open',
    resource             TEXT,
    title                TEXT NOT NULL,
    first_seen           REAL NOT NULL,
    last_seen            REAL NOT NULL,
    evidence_json        TEXT,
    related_task_ids_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_task    ON decisions(task_id);
CREATE INDEX IF NOT EXISTS idx_decisions_session ON decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_task        ON audit_log(task_id);
CREATE INDEX IF NOT EXISTS idx_incidents_status  ON incidents(status);
"""


def init_db() -> None:
    """Create tables if they do not exist (idempotent)."""
    with _lock:
        conn = _conn()
        try:
            conn.executescript(_DDL)
            conn.commit()
        finally:
            conn.close()


# ── Task CRUD ──────────────────────────────────────────────────────────────────

def upsert_task(task_id: str, objective: str, scenario: str,
                status: str = "queued",
                summary: dict | None = None) -> None:
    now = time.time()
    summary_json = json.dumps(summary) if summary else None
    with _lock:
        conn = _conn()
        try:
            conn.execute(
                """
                INSERT INTO tasks (task_id, objective, scenario, status,
                                   created_at, updated_at, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status       = excluded.status,
                    updated_at   = excluded.updated_at,
                    summary_json = excluded.summary_json
                """,
                (task_id, objective, scenario, status, now, now, summary_json),
            )
            conn.commit()
        finally:
            conn.close()


def update_task_status(task_id: str, status: str,
                       summary: dict | None = None) -> None:
    now = time.time()
    with _lock:
        conn = _conn()
        try:
            if summary is not None:
                conn.execute(
                    "UPDATE tasks SET status=?, updated_at=?, summary_json=? WHERE task_id=?",
                    (status, now, json.dumps(summary), task_id),
                )
            else:
                conn.execute(
                    "UPDATE tasks SET status=?, updated_at=? WHERE task_id=?",
                    (status, now, task_id),
                )
            conn.commit()
        finally:
            conn.close()


def get_tasks(limit: int = 50) -> list[dict]:
    with _lock:
        conn = _conn()
        try:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_active_task() -> dict | None:
    with _lock:
        conn = _conn()
        try:
            row = conn.execute(
                "SELECT * FROM tasks WHERE status NOT IN ('completed','failed','blocked') "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


# ── Decision persistence ────────────────────────────────────────────────────────

def save_decision(event: dict) -> None:
    """Persist one SENTINEL decision event."""
    with _lock:
        conn = _conn()
        try:
            conn.execute(
                """
                INSERT INTO decisions
                    (event_id, task_id, session_id, seq, tool, args_json,
                     decision, scores_json, reason, pfa_state, drift,
                     ltl_violations_json, timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event.get("event_id"),
                    event.get("task_id"),
                    event.get("session_id"),
                    event.get("seq"),
                    event.get("tool"),
                    json.dumps(event.get("args", {})),
                    event.get("decision"),
                    json.dumps(event.get("scores", {})),
                    event.get("reason", ""),
                    event.get("pfa_state"),
                    event.get("drift"),
                    json.dumps(event.get("ltl_violations", [])),
                    event.get("timestamp", time.time()),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def get_decisions(limit: int = 100,
                  decision_filter: str | None = None,
                  tool_filter: str | None = None,
                  task_id: str | None = None) -> list[dict]:
    clauses = []
    params: list[Any] = []
    if decision_filter:
        clauses.append("decision = ?")
        params.append(decision_filter.upper())
    if tool_filter:
        clauses.append("tool = ?")
        params.append(tool_filter)
    if task_id:
        clauses.append("task_id = ?")
        params.append(task_id)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with _lock:
        conn = _conn()
        try:
            rows = conn.execute(
                f"SELECT * FROM decisions {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                for field in ("args_json", "scores_json", "ltl_violations_json"):
                    try:
                        d[field] = json.loads(d[field]) if d[field] else {}
                    except Exception:
                        pass
                result.append(d)
            return result
        finally:
            conn.close()


# ── Audit log ─────────────────────────────────────────────────────────────────

def _last_audit_hash() -> str:
    """Return the hash of the most recent audit entry (or genesis hash)."""
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT curr_hash FROM audit_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["curr_hash"] if row else "0" * 64
    finally:
        conn.close()


def append_audit(event_id: str, task_id: str | None,
                 action: str, details: dict) -> str:
    """Append one audit record; returns the new hash."""
    with _lock:
        prev_hash = _last_audit_hash()
        details_json = json.dumps(details, default=str)
        payload = f"{event_id}|{action}|{details_json}|{prev_hash}"
        curr_hash = hashlib.sha256(payload.encode()).hexdigest()
        conn = _conn()
        try:
            conn.execute(
                """
                INSERT INTO audit_log
                    (event_id, task_id, action, details_json, prev_hash, curr_hash, timestamp)
                VALUES (?,?,?,?,?,?,?)
                """,
                (event_id, task_id, action, details_json,
                 prev_hash, curr_hash, time.time()),
            )
            conn.commit()
        finally:
            conn.close()
        return curr_hash


def get_audit(limit: int = 100, task_id: str | None = None) -> list[dict]:
    params: list[Any] = []
    where = ""
    if task_id:
        where = "WHERE task_id = ?"
        params.append(task_id)
    params.append(limit)
    with _lock:
        conn = _conn()
        try:
            rows = conn.execute(
                f"SELECT * FROM audit_log {where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def verify_audit_chain() -> dict:
    """Walk every audit record and verify the hash chain. Returns pass/fail."""
    with _lock:
        conn = _conn()
        try:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id ASC"
            ).fetchall()
        finally:
            conn.close()

    if not rows:
        return {"valid": True, "records": 0, "message": "No audit records yet"}

    prev = "0" * 64
    for i, row in enumerate(rows):
        r = dict(row)
        payload = f"{r['event_id']}|{r['action']}|{r['details_json']}|{prev}"
        expected = hashlib.sha256(payload.encode()).hexdigest()
        if expected != r["curr_hash"]:
            return {
                "valid": False,
                "records": len(rows),
                "broken_at": r["id"],
                "message": f"Chain broken at record id={r['id']} (entry {i+1})",
            }
        prev = r["curr_hash"]

    return {"valid": True, "records": len(rows), "message": "Hash chain intact"}


# ── Incidents ─────────────────────────────────────────────────────────────────

def create_incident(severity: str, resource: str, title: str,
                    evidence: dict | None = None,
                    related_task_ids: list[str] | None = None) -> int:
    now = time.time()
    with _lock:
        conn = _conn()
        try:
            cur = conn.execute(
                """
                INSERT INTO incidents
                    (severity, status, resource, title, first_seen, last_seen,
                     evidence_json, related_task_ids_json)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (severity, "open", resource, title, now, now,
                 json.dumps(evidence or {}),
                 json.dumps(related_task_ids or [])),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_incidents(limit: int = 100) -> list[dict]:
    with _lock:
        conn = _conn()
        try:
            rows = conn.execute(
                "SELECT * FROM incidents ORDER BY last_seen DESC LIMIT ?", (limit,)
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                for field in ("evidence_json", "related_task_ids_json"):
                    try:
                        d[field] = json.loads(d[field]) if d[field] else {}
                    except Exception:
                        pass
                result.append(d)
            return result
        finally:
            conn.close()


# ── Initialise on import ──────────────────────────────────────────────────────
init_db()
