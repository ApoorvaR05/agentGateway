"""
SENTINEL — Core Interceptor
The single gate between every AXIOM-OPS tool call and execution.

On each tool call:
  1. Publishes an event to the event bus
  2. Runs each SENTINEL analyser in sequence (fast path)
  3. Combines scores into a risk decision
  4. Returns ALLOW / WARN / BLOCK with a reason

This module is intentionally kept thin — each subsystem lives in its own file.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import config
from agent import session as session_mod
from sentinel import event_bus
from sentinel.analysers.intent      import IntentAnalyser
from sentinel.analysers.pfa         import PFAEngine
from sentinel.analysers.risk_scorer import RiskScorer
from sentinel import knowledge_graph as kg_mod
from sentinel import gnn_harm_detector as gnn_mod
from sentinel import report_card as rc_mod
from sentinel import least_privilege as lp_mod
from sentinel import merkle_audit as merkle_mod
from sentinel import rl_tuner as rl_mod

_logger = logging.getLogger(__name__)


class Interceptor:
    """Singleton interceptor. One instance per process."""

    _instance: "Interceptor | None" = None

    # Shared encoder instance — kept alive across session resets
    _shared_intent_analyser: "IntentAnalyser | None" = None

    def __init__(self):
        # Re-use the shared encoder to avoid reloading weights every session
        if Interceptor._shared_intent_analyser is None:
            Interceptor._shared_intent_analyser = IntentAnalyser()
        self.intent_analyser = Interceptor._shared_intent_analyser
        self.pfa_engine      = PFAEngine()
        self.risk_scorer     = RiskScorer()
        self._monitoring_elevated = False
        self._kg: "kg_mod.KnowledgeGraph | None" = None

    @classmethod
    def get_instance(cls) -> "Interceptor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Call between sessions to clear per-session state (not the encoder)."""
        cls._instance = None
        kg_mod.reset_graph()
        lp_mod.reset_controller()
        merkle_mod.reset_trail()

    # ── Main evaluation method ────────────────────────────────────────────────

    def evaluate(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate a tool call through all SENTINEL subsystems.
        Returns a decision dict: {action, reason, scores, report_card}.
        """
        sess = session_mod.get_session()
        seq  = sess.next_seq()
        t0   = time.time()

        # ── 1. Intent drift analysis ─────────────────────────────────────────
        intent_analysis = self.intent_analyser.score_with_details(
            task=sess.task,
            tool_name=tool_name,
            args=args,
            history=sess.actions,
        )
        drift_score = intent_analysis["drift_score"]

        # ── 2. PFA state transition + LTL check ──────────────────────────────
        pfa_result = self.pfa_engine.transition(
            current_state=sess.pfa_state,
            tool_name=tool_name,
            args=args,
            history=sess.actions,
        )
        sess.pfa_state = pfa_result["new_state"]
        ltl_violations = pfa_result["ltl_violations"]

        # ── 3. ML risk scoring ────────────────────────────────────────────────
        scores = self.risk_scorer.score(
            tool_name=tool_name,
            args=args,
            drift_score=drift_score,
            pfa_state=pfa_result["new_state"],
            ltl_violations=ltl_violations,
            history=sess.actions,
            monitoring_level=sess.monitoring_level,
        )

        security    = scores["security"]
        safety      = scores["safety"]
        reliability = scores["reliability"]
        overall     = scores["overall"]

        # ── 4. Apply thresholds → decision ───────────────────────────────────
        warn_t  = config.RISK_WARN_THRESHOLD
        block_t = config.RISK_BLOCK_THRESHOLD

        # LTL violation is always a block
        if ltl_violations:
            action = "BLOCK"
            reason = f"LTL safety property violated: {ltl_violations[0]['property']}. {ltl_violations[0]['detail']}"
        elif overall >= block_t or security >= block_t or drift_score >= config.DRIFT_BLOCK_THRESHOLD:
            action = "BLOCK"
            reason = _build_block_reason(overall, security, safety, drift_score, ltl_violations)
        elif overall >= warn_t or drift_score >= config.DRIFT_WARN_THRESHOLD:
            action = "WARN"
            reason = f"Elevated risk detected (overall={overall:.2f}, drift={drift_score:.2f}). Proceeding under elevated monitoring."
        else:
            action = "ALLOW"
            reason = "All risk scores within normal thresholds."

        # ── 5. Record to session ──────────────────────────────────────────────
        risk_scores = {
            "security": round(security, 3),
            "safety":   round(safety,   3),
            "reliability": round(reliability, 3),
            "overall":  round(overall,  3),
            "drift":    round(drift_score, 3),
        }

        record = session_mod.ActionRecord(
            seq=seq,
            tool_name=tool_name,
            args=args,
            decision=action,
            risk_scores=risk_scores,
            reason=reason,
        )
        sess.record_action(record)
        sess.update_risk(security, safety, reliability)

        if ltl_violations:
            for v in ltl_violations:
                sess.add_violation("LTL", v["detail"])

        # ── 6. Update Knowledge Graph ─────────────────────────────────────────
        kg_stats: dict = {}
        if self._kg is None:
            self._kg = kg_mod.new_graph(sess.session_id)
        self._kg.add_action(seq, tool_name, args, action, risk_scores)
        kg_stats = self._kg.stats()

        # ── 7. GNN harm chain detection ───────────────────────────────────────
        # GNN needs: (a) at least 3 actions, AND (b) at least one harm edge in
        # the KG — if the KG has no causal paths to harm nodes, the GNN cannot
        # detect an emergent chain (avoids false positives on safe sessions).
        gnn_prob    = 0.0
        harm_scores = kg_stats.get("harm_scores", {})
        kg_has_harm = any(v > 0 for v in harm_scores.values())

        if seq >= 3 and kg_has_harm:
            try:
                detector  = gnn_mod.get_detector()
                gnn_prob  = detector.predict({
                    **self._kg.to_dict(),
                    "harm_scores": harm_scores,
                })
            except Exception:
                pass

        # Escalate only when GNN AND KG both agree on harm
        if gnn_prob >= 0.70 and kg_has_harm and action == "ALLOW":
            action = "WARN"
            reason = (f"GNN harm chain detected (prob={gnn_prob:.2f}). "
                      f"Top harm: {kg_stats.get('top_harm', '?')}. "
                      f"Proceeding under elevated monitoring.")
            record.decision = action
            sess.warned_count  += 1
            sess.allowed_count -= 1
        elif gnn_prob >= 0.90 and kg_has_harm and action != "BLOCK":
            action = "BLOCK"
            reason = (f"GNN harm chain blocked (prob={gnn_prob:.2f}). "
                      f"Causal path leads to: {kg_stats.get('top_harm', '?')}.")
            record.decision = action
            if record.decision == "WARN":
                sess.warned_count -= 1
            else:
                sess.allowed_count -= 1
            sess.blocked_count += 1

        # ── 8. Least-privilege controller ─────────────────────────────────────
        lp = lp_mod.get_controller()
        if lp is None:
            lp = lp_mod.new_controller(sess.task, sess.session_id)
        revoked = lp.on_pfa_transition(pfa_result["new_state"])
        if action == "BLOCK":
            lp.on_block(tool_name, reason)

        # ── 9. Build event dict ───────────────────────────────────────────────
        event_id    = str(uuid.uuid4())[:8]
        published_at = time.time()   # stamp before commit so hash is stable
        event = {
            "event_id":    event_id,
            "session_id":  sess.session_id,
            "seq":         seq,
            "tool":        tool_name,
            "args":        {k: str(v)[:100] for k, v in args.items()},
            "decision":    action,
            "scores":      risk_scores,
            "model_inference": {
                "sentence_transformer_available":
                    intent_analysis["sentence_transformer_available"],
                "embedding_inference_performed":
                    intent_analysis["embedding_inference_performed"],
                "embedding_similarity": intent_analysis["embedding_similarity"],
                "drift_source": intent_analysis["drift_source"],
                "drift_score": round(drift_score, 3),
                "xgboost_available": scores["xgb_model_available"],
                "xgboost_inference_performed": scores["xgb_inference_performed"],
                "xgboost_status": scores["xgb_status"],
                "model_generated_risk_score": scores["xgb_model_risk_score"],
            },
            "pfa_state":   pfa_result["new_state"],
            "drift":       round(drift_score, 3),
            "ltl_violations": [v["property"] for v in ltl_violations],
            "reason":      reason,
            "latency_ms":  round((time.time() - t0) * 1000, 1),
            "kg_stats":    kg_stats,
            "gnn_prob":    round(gnn_prob, 3),
            "lp_revoked":  revoked,
            "published_at": published_at,
        }

        # ── 10. Merkle audit commit ───────────────────────────────────────────
        trail = merkle_mod.get_trail()
        if trail is None:
            trail = merkle_mod.new_trail(sess.session_id)
        trail.commit(event)

        # ── 11. Generate report card for BLOCK/WARN decisions ─────────────────
        if action in ("BLOCK", "WARN"):
            try:
                kg_chains: dict = {}
                if self._kg:
                    for harm in ("CredentialExfiltration", "PrivilegeEscalation",
                                 "DataExfiltration", "PublicDatabaseExposure"):
                        chains = self._kg.get_causal_chain(harm)
                        if chains:
                            kg_chains[harm] = chains
                rc_mod.generate_report_card(event, kg_chains=kg_chains)
            except Exception:
                pass

        # ── 12. RL threshold tuning (every 20 actions) ───────────────────────
        if seq % 20 == 0:
            try:
                tuner = rl_mod.get_tuner()
                new_t = tuner.tune()
                if new_t:
                    config.RISK_WARN_THRESHOLD   = new_t["warn_t"]
                    config.RISK_BLOCK_THRESHOLD  = new_t["block_t"]
                    config.DRIFT_WARN_THRESHOLD  = new_t["drift_w"]
                    config.DRIFT_BLOCK_THRESHOLD = new_t["drift_b"]
            except Exception:
                pass

        # ── 13. Publish event to bus ──────────────────────────────────────────
        _logger.info(
            "SENTINEL_ACTION %s",
            json.dumps({
                "seq": seq,
                "tool": tool_name,
                "sentence_transformer_available":
                    intent_analysis["sentence_transformer_available"],
                "embedding_inference_performed":
                    intent_analysis["embedding_inference_performed"],
                "embedding_similarity": intent_analysis["embedding_similarity"],
                "drift_score": round(drift_score, 3),
                "drift_source": intent_analysis["drift_source"],
                "xgboost_available": scores["xgb_model_available"],
                "xgboost_inference_performed": scores["xgb_inference_performed"],
                "xgboost_status": scores["xgb_status"],
                "model_generated_risk_score": scores["xgb_model_risk_score"],
                "final_decision": action,
            }, sort_keys=True),
        )
        event_bus.publish(event)

        return {
            "action":         action,
            "reason":         reason,
            "scores":         risk_scores,
            "pfa_state":      pfa_result["new_state"],
            "ltl_violations": ltl_violations,
            "seq":            seq,
            "kg_stats":       kg_stats,
            "gnn_prob":       round(gnn_prob, 3),
            "lp_revoked":     revoked,
        }

    def elevate_monitoring(self) -> None:
        sess = session_mod.get_session()
        sess.elevate_monitoring()


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_block_reason(overall: float, security: float, safety: float,
                         drift: float, ltl: list) -> str:
    parts = []
    if security >= config.RISK_BLOCK_THRESHOLD:
        parts.append(f"security risk critical ({security:.2f})")
    if safety >= config.RISK_BLOCK_THRESHOLD:
        parts.append(f"safety risk critical ({safety:.2f})")
    if overall >= config.RISK_BLOCK_THRESHOLD:
        parts.append(f"overall risk critical ({overall:.2f})")
    if drift >= config.DRIFT_BLOCK_THRESHOLD:
        parts.append(f"intent drift too high ({drift:.2f} — declared task vs. observed action diverged)")
    return "Action blocked: " + "; ".join(parts) + "."
