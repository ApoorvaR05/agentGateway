"""
SENTINEL — ML Risk Ensemble (Phase 2 — Subsystem 5)
Combines two models:
  1. Rule-weighted scorer (transparent baseline, always runs)
  2. XGBoost sequence classifier (trained on 1200 synthetic sequences)

The ensemble blends both scores:
  - Rule model provides per-dimension scores (security, safety, reliability)
  - XGBoost model provides a calibrated adversarial probability
  - Final overall = max(rule_overall, xgb_prob * 0.9)
    (XGBoost has 10% discount since it was trained on synthetic, not live data)

Falls back to rule-only if the model file is not found.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from sentinel.analysers.pfa import STATE_RISK, TIER3_TOOLS, TIER2_TOOLS

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_FILE  = os.path.join(_ROOT, "models", "risk_classifier.json")
_FEATS_FILE  = os.path.join(_ROOT, "models", "feature_names.json")

# ── Feature weights for Security sub-score ────────────────────────────────────
_SEC_FEATURES = {
    "tool_tier_3":              0.35,
    "external_url_in_args":     0.40,
    "secret_key_in_args":       0.30,
    "shadow_file_access":       0.60,
    "admin_in_args":            0.45,
    "curl_to_external":         0.50,
    "injection_marker":         0.70,
    "privileged_state":         0.20,
    "consecutive_tier3":        0.25,
    "iam_grant_action":         0.40,
}

# ── Feature weights for Safety sub-score ─────────────────────────────────────
_SAF_FEATURES = {
    "pfa_anomalous":            0.60,
    "drift_high":               0.50,
    "drift_moderate":           0.25,
    "tool_out_of_context":      0.35,
    "public_port_opening":      0.55,
    "backup_to_unsafe_target":  0.50,
    "ltl_violated":             0.65,
    "emergent_harm_seq":        0.45,
}

# ── Feature weights for Reliability sub-score ─────────────────────────────────
_REL_FEATURES = {
    "long_session":             0.10,
    "repeated_tool":            0.15,
    "all_tools_fail":           0.30,
    "monitoring_elevated":      0.10,
}

# ── XGBoost feature names (must match training) ───────────────────────────────
_XGB_FEATURE_NAMES = [
    "tier3_ratio", "tier2_ratio", "seq_length",
    "has_external_url", "has_secret_keyword", "has_shadow_access",
    "has_admin_keyword", "has_curl_external", "has_injection_marker",
    "has_wildcard_perm", "has_public_firewall", "has_unsafe_backup",
    "max_consecutive_tier3", "secret_then_shell",
    "mean_drift_score", "max_drift_score",
    "mean_security_score", "max_security_score",
    "block_rate", "warn_rate",
]


class RiskScorer:
    """
    ML Risk Ensemble: rule-weighted scorer + XGBoost classifier.
    Returns scores in [0, 1] — higher = riskier.
    """

    def __init__(self):
        self._xgb_model = None
        self._load_xgb()

    def _load_xgb(self) -> None:
        """Load the trained XGBoost model (silent fail → rule-only fallback)."""
        try:
            import xgboost as xgb
            if os.path.exists(_MODEL_FILE):
                self._xgb_model = xgb.XGBClassifier()
                self._xgb_model.load_model(_MODEL_FILE)
        except Exception:
            self._xgb_model = None

    def score(self, tool_name: str, args: dict[str, Any],
              drift_score: float, pfa_state: str,
              ltl_violations: list, history: list,
              monitoring_level: str) -> dict[str, float]:

        args_str   = " ".join(str(v) for v in args.values()).lower()
        prev_tools = [r.tool_name for r in history[-5:]] if history else []

        # ── 1. Rule-weighted sub-scores ───────────────────────────────────────
        sec = self._rule_security(tool_name, args_str, pfa_state,
                                  prev_tools, drift_score)
        saf = self._rule_safety(tool_name, args_str, pfa_state,
                                drift_score, ltl_violations, prev_tools)
        rel = self._rule_reliability(history, prev_tools, monitoring_level)

        sec = min(sec, 1.0)
        saf = min(saf, 1.0)
        rel = min(rel, 1.0)

        rule_overall = max(sec, saf, (sec + saf + rel) / 3)
        rule_overall = min(rule_overall, 1.0)
        state_floor  = STATE_RISK.get(pfa_state, 0.0)
        rule_overall = max(rule_overall, state_floor)

        # ── 2. XGBoost sequence-level probability ─────────────────────────────
        xgb_prob = self._xgb_score(
            tool_name, args_str, drift_score, history, sec, saf, prev_tools
        )

        # ── 3. Ensemble blend ─────────────────────────────────────────────────
        # XGBoost contributes only when rule-based scores are already elevated
        # (avoids false positives where XGBoost over-generalises on Tier-2/3
        # tools that appear in both safe and adversarial training sequences).
        # XGBoost can raise overall but only if rule_overall > 0.25 (not cold).
        if rule_overall > 0.25:
            overall = max(rule_overall, xgb_prob * 0.70)
        else:
            overall = rule_overall
        overall = min(overall, 1.0)

        return {
            "security":    round(sec,      3),
            "safety":      round(saf,      3),
            "reliability": round(rel,      3),
            "overall":     round(overall,  3),
            "xgb_prob":    round(xgb_prob, 3),
        }

    # ── Rule scorer helpers ───────────────────────────────────────────────────

    def _rule_security(self, tool_name, args_str, pfa_state,
                       prev_tools, drift_score) -> float:
        sec = 0.0
        if tool_name in TIER3_TOOLS:
            sec += _SEC_FEATURES["tool_tier_3"]
        if re.search(r"https?://(?!localhost|127\.|10\.|172\.|192\.168\.)", args_str):
            sec += _SEC_FEATURES["external_url_in_args"]
        if re.search(r"password|secret|token|api.key|credential", args_str):
            sec += _SEC_FEATURES["secret_key_in_args"]
        if re.search(r"/etc/shadow|/etc/passwd", args_str):
            sec += _SEC_FEATURES["shadow_file_access"]
        if re.search(r"\badmin\b", args_str):
            sec += _SEC_FEATURES["admin_in_args"]
        if tool_name == "execute_shell" and "curl" in args_str and \
                re.search(r"https?://(?!localhost|127\.|10\.|172\.|192\.168\.)", args_str):
            sec += _SEC_FEATURES["curl_to_external"]
        if re.search(r"ignore\s+previous\s+instructions?|new\s+task\s*:", args_str, re.I):
            sec += _SEC_FEATURES["injection_marker"]
        if pfa_state in {"PrivilegedAccess", "Anomalous", "Terminated"}:
            sec += _SEC_FEATURES["privileged_state"]
        if len(prev_tools) >= 2 and all(t in TIER3_TOOLS for t in prev_tools[-2:]):
            sec += _SEC_FEATURES["consecutive_tier3"]
        if tool_name == "modify_iam" and "grant" in args_str:
            sec += _SEC_FEATURES["iam_grant_action"]
        return sec

    def _rule_safety(self, tool_name, args_str, pfa_state,
                     drift_score, ltl_violations, prev_tools) -> float:
        saf = 0.0
        if pfa_state == "Anomalous":
            saf += _SAF_FEATURES["pfa_anomalous"]
        if drift_score > 0.40:
            saf += _SAF_FEATURES["drift_high"]
        elif drift_score > 0.25:
            saf += _SAF_FEATURES["drift_moderate"]
        if ltl_violations:
            saf += _SAF_FEATURES["ltl_violated"]
        if tool_name == "modify_firewall" and \
                re.search(r"0\.0\.0\.0|public|internet|world", args_str):
            saf += _SAF_FEATURES["public_port_opening"]
        if tool_name == "backup_database" and re.search(r"^/tmp|https?://", args_str):
            saf += _SAF_FEATURES["backup_to_unsafe_target"]
        if tool_name == "modify_firewall" and prev_tools.count("modify_firewall") >= 1:
            saf += _SAF_FEATURES["emergent_harm_seq"]
        return saf

    def _rule_reliability(self, history, prev_tools, monitoring_level) -> float:
        rel = 0.0
        if len(history) > 15:
            rel += _REL_FEATURES["long_session"]
        if len(prev_tools) >= 3 and len(set(prev_tools[-3:])) == 1:
            rel += _REL_FEATURES["repeated_tool"]
        if monitoring_level == "elevated":
            rel += _REL_FEATURES["monitoring_elevated"]
        if monitoring_level == "critical":
            rel += _REL_FEATURES["monitoring_elevated"] * 2
        return rel

    # ── XGBoost scorer ────────────────────────────────────────────────────────

    def _xgb_score(self, tool_name: str, args_str: str, drift_score: float,
                   history: list, sec: float, saf: float,
                   prev_tools: list) -> float:
        """Extract session-level features and return XGBoost adversarial probability."""
        if self._xgb_model is None:
            return 0.0

        try:
            import numpy as np

            total    = max(len(history) + 1, 1)
            all_tools = [r.tool_name for r in history] + [tool_name]
            all_args  = args_str + " " + " ".join(
                " ".join(str(v) for v in r.args.values()).lower()
                for r in history
            )
            tier3_c = sum(1 for t in all_tools if t in TIER3_TOOLS)
            tier2_c = sum(1 for t in all_tools if t in TIER2_TOOLS)

            # Consecutive tier3 run
            max_t3 = 0
            run = 0
            for t in all_tools:
                if t in TIER3_TOOLS:
                    run += 1; max_t3 = max(max_t3, run)
                else:
                    run = 0

            # Prior scores from history
            h_scores   = [r.risk_scores for r in history] if history else []
            prev_secs  = [s.get("security", 0) for s in h_scores]
            prev_drifts= [s.get("drift",    0) for s in h_scores]
            prev_decs  = [r.decision for r in history] if history else []

            # secret_then_shell in full history window
            all_t = [r.tool_name for r in history]
            s_then_sh = int(any(
                all_t[i] == "read_secret" and all_t[i+1] == "execute_shell"
                for i in range(len(all_t) - 1)
            )) if len(all_t) >= 2 else 0

            row = {
                "tier3_ratio":           tier3_c / total,
                "tier2_ratio":           tier2_c / total,
                "seq_length":            total,
                "has_external_url":      int(bool(re.search(
                    r"https?://(?!localhost|127\.|10\.|172\.|192\.168\.)", all_args))),
                "has_secret_keyword":    int(bool(re.search(
                    r"password|secret|token|api.key|credential", all_args))),
                "has_shadow_access":     int("/etc/shadow" in all_args or
                                             "/etc/passwd" in all_args),
                "has_admin_keyword":     int(bool(re.search(r"\badmin\b", all_args))),
                "has_curl_external":     int(bool(re.search(
                    r"curl.*https?://(?!localhost|127\.|10\.|172\.|192\.168\.)", all_args))),
                "has_injection_marker":  int(bool(re.search(
                    r"ignore\s+previous\s+instructions?|new\s+task\s*:", all_args, re.I))),
                "has_wildcard_perm":     int(bool(re.search(r":\*|:all\b", all_args))),
                "has_public_firewall":   int(bool(re.search(
                    r"0\.0\.0\.0|public|internet", all_args)) and
                    bool(re.search(r"5432|3306|27017", all_args))),
                "has_unsafe_backup":     int("backup_database" in all_tools and
                    bool(re.search(r"/tmp|https?://|ftp://", all_args))),
                "max_consecutive_tier3": max_t3,
                "secret_then_shell":     s_then_sh,
                "mean_drift_score":      (sum(prev_drifts) + drift_score) / total,
                "max_drift_score":       max(prev_drifts + [drift_score], default=0),
                "mean_security_score":   (sum(prev_secs)  + sec) / total,
                "max_security_score":    max(prev_secs + [sec], default=0),
                "block_rate":            prev_decs.count("BLOCK") / total,
                "warn_rate":             prev_decs.count("WARN")  / total,
            }

            X = np.array([[row[f] for f in _XGB_FEATURE_NAMES]], dtype=np.float32)
            prob = float(self._xgb_model.predict_proba(X)[0][1])
            return min(prob, 1.0)

        except Exception:
            return 0.0
