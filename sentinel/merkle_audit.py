"""
SENTINEL — Merkle Audit Trail (Phase 3.5)

Every risk decision is hashed with SHA-256 and committed to a Merkle
accumulator. At session end a proof token is generated that allows
anyone to verify the audit log was not tampered with.

Properties:
  - Each leaf = SHA-256( event_id | session_id | seq | tool | decision | scores | timestamp )
  - Merkle tree built over all leaves in insertion order
  - Root hash changes if any decision is altered
  - Proof token = base64( root_hash | leaf_count | session_id | timestamp )
  - Stored in reports/audit/ as {session_id}_merkle.json

Cryptographic guarantee:
  Given a proof token, any auditor can re-hash the raw events and verify
  the root matches — without access to SENTINEL internals.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from typing import Any

_AUDIT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "audit"
)
os.makedirs(_AUDIT_DIR, exist_ok=True)


# ── Leaf hash ─────────────────────────────────────────────────────────────────

def _leaf_hash(event: dict[str, Any]) -> str:
    """SHA-256 of the canonical event fields."""
    canonical = json.dumps({
        "event_id":   event.get("event_id",   ""),
        "session_id": event.get("session_id", ""),
        "seq":        event.get("seq",        0),
        "tool":       event.get("tool",       ""),
        "decision":   event.get("decision",   ""),
        "scores":     event.get("scores",     {}),
        "timestamp":  event.get("published_at", 0),
    }, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _merkle_root(leaves: list[str]) -> str:
    """Compute the Merkle root from a list of leaf hashes."""
    if not leaves:
        return hashlib.sha256(b"empty").hexdigest()
    layer = list(leaves)
    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            left  = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else left
            combined = hashlib.sha256((left + right).encode()).hexdigest()
            next_layer.append(combined)
        layer = next_layer
    return layer[0]


# ── Merkle Accumulator ────────────────────────────────────────────────────────

class MerkleAuditTrail:
    """
    Accumulates SHA-256 leaf hashes and maintains a rolling Merkle root.
    One instance per session.
    """

    def __init__(self, session_id: str):
        self.session_id  = session_id
        self._leaves:    list[str]       = []
        self._events:    list[dict]      = []
        self._root:      str             = ""
        self._created_at = time.time()

    # ── Public API ────────────────────────────────────────────────────────────

    def commit(self, event: dict[str, Any]) -> str:
        """Add an event to the audit trail. Returns the leaf hash."""
        leaf = _leaf_hash(event)
        self._leaves.append(leaf)
        self._events.append({
            "seq":       event.get("seq", 0),
            "event_id":  event.get("event_id", ""),
            "tool":      event.get("tool", ""),
            "decision":  event.get("decision", ""),
            "leaf_hash": leaf,
        })
        self._root = _merkle_root(self._leaves)
        return leaf

    def proof_token(self) -> str:
        """
        Generate a base64-encoded proof token.
        Token = base64( JSON { root, leaf_count, session_id, finalized_at } )
        """
        payload = json.dumps({
            "root":         self._root,
            "leaf_count":   len(self._leaves),
            "session_id":   self.session_id,
            "finalized_at": time.time(),
        }, sort_keys=True)
        return base64.b64encode(payload.encode()).decode()

    def verify(self, proof_token: str, events: list[dict]) -> dict[str, Any]:
        """
        Re-compute the Merkle root from raw events and compare to the proof token.
        Returns { valid: bool, root_expected, root_computed, leaf_count }.
        """
        try:
            payload = json.loads(base64.b64decode(proof_token).decode())
        except Exception as e:
            return {"valid": False, "error": str(e)}

        recomputed_leaves = [_leaf_hash(e) for e in events]
        recomputed_root   = _merkle_root(recomputed_leaves)

        return {
            "valid":          recomputed_root == payload["root"],
            "root_expected":  payload["root"],
            "root_computed":  recomputed_root,
            "leaf_count":     payload["leaf_count"],
            "events_provided": len(events),
            "session_id":     payload.get("session_id"),
        }

    def save(self) -> str:
        """Persist the audit trail to disk. Returns the file path."""
        out = {
            "session_id":   self.session_id,
            "created_at":   self._created_at,
            "finalized_at": time.time(),
            "leaf_count":   len(self._leaves),
            "merkle_root":  self._root,
            "proof_token":  self.proof_token(),
            "leaves":       self._events,
        }
        path = os.path.join(_AUDIT_DIR, f"{self.session_id}_merkle.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return path

    def summary(self) -> dict[str, Any]:
        return {
            "session_id":  self.session_id,
            "leaf_count":  len(self._leaves),
            "merkle_root": self._root[:16] + "…",
            "proof_token": self.proof_token()[:32] + "…",
        }


# ── Module-level registry ─────────────────────────────────────────────────────
_current_trail: MerkleAuditTrail | None = None


def get_trail() -> MerkleAuditTrail | None:
    return _current_trail


def new_trail(session_id: str) -> MerkleAuditTrail:
    global _current_trail
    _current_trail = MerkleAuditTrail(session_id)
    return _current_trail


def reset_trail() -> None:
    global _current_trail
    _current_trail = None
