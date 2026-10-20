"""
sentinel/xai.py — Explainability module (Phase 4.1 · 4.2 · 4.3)

Three classes:
  SHAPExplainer        — SHAP feature-attribution using the trained XGBoost model
  DiCECounterfactual   — minimum-change counterfactual for blocked actions
  NarrativeExplainer   — human-readable paragraph (template or LLM-backed)

Public helper:
  get_xai_for_event(event_dict) -> {"shap": ..., "counterfactual": ..., "narrative": ...}
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import config

# ── Feature registry ──────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    "tier3_ratio",
    "tier2_ratio",
    "seq_length",
    "has_external_url",
    "has_secret_keyword",
    "has_shadow_access",
    "has_admin_keyword",
    "has_curl_external",
    "has_injection_marker",
    "has_wildcard_perm",
    "has_public_firewall",
    "has_unsafe_backup",
    "max_consecutive_tier3",
    "secret_then_shell",
    "mean_drift_score",
    "max_drift_score",
    "mean_security_score",
    "max_security_score",
    "block_rate",
    "warn_rate",
]

# Security features drive blocking; safety features drive warning
_SEC_FEATURES: list[str] = [
    "has_injection_marker",
    "has_shadow_access",
    "has_secret_keyword",
    "has_wildcard_perm",
    "has_admin_keyword",
    "has_curl_external",
    "secret_then_shell",
    "has_external_url",
    "max_security_score",
    "mean_security_score",
]
_SAF_FEATURES: list[str] = [
    "has_public_firewall",
    "has_unsafe_backup",
    "max_drift_score",
    "mean_drift_score",
    "tier3_ratio",
    "tier2_ratio",
    "block_rate",
    "warn_rate",
    "max_consecutive_tier3",
    "seq_length",
]

_MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "risk_classifier.json"

# ── SHAP Explainer ─────────────────────────────────────────────────────────────


class SHAPExplainer:
    """
    Wraps shap.TreeExplainer around the trained XGBoost risk classifier.
    Falls back to a rule-based approximation when shap or xgboost is absent.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._explainer: Any = None
        self._use_shap: bool = False
        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        try:
            import xgboost as xgb  # type: ignore
            import shap  # type: ignore

            if not _MODEL_PATH.exists():
                return

            booster = xgb.Booster()
            booster.load_model(str(_MODEL_PATH))
            self._model = booster
            self._explainer = shap.TreeExplainer(booster)
            self._use_shap = True
        except Exception:
            # shap / xgboost not installed or model missing → rule-based fallback
            self._use_shap = False

    # ------------------------------------------------------------------
    def _feature_vector(self, features_dict: dict) -> list[float]:
        return [float(features_dict.get(f, 0.0)) for f in FEATURE_NAMES]

    # ------------------------------------------------------------------
    def _rule_based_explain(self, features_dict: dict) -> dict:
        """
        Simple rule-based approximation: impact = raw feature value
        weighted by a hand-crafted importance heuristic.
        """
        weights = {
            "has_injection_marker": 0.20,
            "has_shadow_access": 0.18,
            "secret_then_shell": 0.15,
            "has_wildcard_perm": 0.12,
            "has_admin_keyword": 0.10,
            "has_secret_keyword": 0.08,
            "has_curl_external": 0.07,
            "max_security_score": 0.06,
            "tier3_ratio": 0.05,
            "has_public_firewall": 0.04,
        }
        base_value = 0.10
        importances = []
        total = base_value
        for name in FEATURE_NAMES:
            val = float(features_dict.get(name, 0.0))
            impact = round(val * weights.get(name, 0.01), 6)
            importances.append({"name": name, "value": val, "impact": impact})
            total += impact

        prediction = min(max(total, 0.0), 1.0)
        importances.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return {
            "feature_importances": importances,
            "base_value": round(base_value, 6),
            "prediction": round(prediction, 6),
        }

    # ------------------------------------------------------------------
    def explain(self, features_dict: dict) -> dict:
        """
        Returns:
            {
              "feature_importances": [{"name": str, "value": float, "impact": float}, ...],
              "base_value": float,
              "prediction": float,
            }
        """
        if not self._use_shap:
            return self._rule_based_explain(features_dict)

        import numpy as np  # type: ignore
        import xgboost as xgb  # type: ignore

        vec = self._feature_vector(features_dict)
        dmatrix = xgb.DMatrix([vec], feature_names=FEATURE_NAMES)
        shap_values = self._explainer.shap_values(dmatrix)
        expected = float(self._explainer.expected_value)

        # shap_values shape: (1, n_features) for binary classifiers
        sv = shap_values[0] if hasattr(shap_values[0], "__len__") else shap_values[0]
        prediction_raw = expected + float(np.sum(sv))

        importances = [
            {
                "name": FEATURE_NAMES[i],
                "value": round(vec[i], 6),
                "impact": round(float(sv[i]), 6),
            }
            for i in range(len(FEATURE_NAMES))
        ]
        importances.sort(key=lambda x: abs(x["impact"]), reverse=True)

        return {
            "feature_importances": importances,
            "base_value": round(expected, 6),
            "prediction": round(min(max(prediction_raw, 0.0), 1.0), 6),
        }


# ── DiCE Counterfactual ────────────────────────────────────────────────────────


class DiCECounterfactual:
    """
    Returns the minimum set of feature changes that would push the risk score
    below the safe threshold (< 0.40).

    Strategy: rank features by SHAP impact (largest positive impact first),
    then suggest zeroing them out until the projected score drops below 0.40.
    """

    SAFE_THRESHOLD: float = 0.40

    def __init__(self, shap_explainer: SHAPExplainer | None = None) -> None:
        self._shap = shap_explainer or SHAPExplainer()

    # ------------------------------------------------------------------
    def generate(self, features_dict: dict, current_score: float) -> dict:
        """
        Returns:
            {
              "changes": [{"feature": str, "from_value": float, "to_value": float}, ...],
              "counterfactual_score": float,
            }
        """
        if current_score < self.SAFE_THRESHOLD:
            return {"changes": [], "counterfactual_score": round(current_score, 6)}

        shap_result = self._shap.explain(features_dict)
        base = shap_result["base_value"]

        # Build (name, current_value, shap_impact) sorted by positive impact desc
        candidates = [
            (fi["name"], fi["value"], fi["impact"])
            for fi in shap_result["feature_importances"]
            if fi["impact"] > 0 and fi["value"] > 0
        ]
        # Prefer security features first, then safety features, then the rest
        def _priority(name: str) -> int:
            if name in _SEC_FEATURES:
                return 0
            if name in _SAF_FEATURES:
                return 1
            return 2

        candidates.sort(key=lambda x: (_priority(x[0]), -x[2]))

        projected_score = current_score
        changes: list[dict] = []

        for name, from_val, impact in candidates:
            if projected_score < self.SAFE_THRESHOLD:
                break
            # Suggest zeroing the feature → removes its positive SHAP contribution
            changes.append(
                {
                    "feature": name,
                    "from_value": round(from_val, 6),
                    "to_value": 0.0,
                }
            )
            projected_score -= impact

        projected_score = round(min(max(projected_score, 0.0), 1.0), 6)
        return {"changes": changes, "counterfactual_score": projected_score}


# ── Narrative Explainer ────────────────────────────────────────────────────────


class NarrativeExplainer:
    """
    Produces a human-readable explanation paragraph.

    If config.OPENAI_API_KEY is set and non-empty, delegates to
    langchain_openai.ChatOpenAI for a richer, context-aware narrative.
    Otherwise renders a deterministic template.
    """

    def __init__(self) -> None:
        self._use_llm: bool = bool(getattr(config, "OPENAI_API_KEY", ""))

    # ------------------------------------------------------------------
    def _top_factors(self, event_dict: dict, n: int = 3) -> list[str]:
        """Extract top risk factor names from SHAP or raw scores."""
        shap_data = event_dict.get("shap") or {}
        importances = shap_data.get("feature_importances", [])
        if importances:
            return [fi["name"] for fi in importances[:n]]

        # Fallback: rank by raw feature value among the known feature set
        feats = event_dict.get("features", {})
        ranked = sorted(
            FEATURE_NAMES,
            key=lambda f: float(feats.get(f, 0.0)),
            reverse=True,
        )
        return ranked[:n]

    # ------------------------------------------------------------------
    def _render_template(self, event_dict: dict) -> str:
        tool = event_dict.get("tool_name", "unknown tool")
        decision = event_dict.get("decision", "unknown").upper()
        score = event_dict.get("risk_score", event_dict.get("score", 0.0))
        ltl_violations = event_dict.get("ltl_violations", [])
        pfa_state = event_dict.get("pfa_state", "unknown")
        cf = event_dict.get("counterfactual") or {}
        cf_changes = cf.get("changes", [])
        cf_score = cf.get("counterfactual_score")
        top_factors = self._top_factors(event_dict)

        # Decision prose
        if decision == "BLOCK":
            decision_phrase = f"was **blocked** (risk score {score:.3f})"
        elif decision == "WARN":
            decision_phrase = f"received a **warning** (risk score {score:.3f})"
        else:
            decision_phrase = f"was **allowed** (risk score {score:.3f})"

        # LTL prose
        if ltl_violations:
            ltl_phrase = (
                f"The following safety-property violations were detected: "
                + ", ".join(f"`{v}`" for v in ltl_violations)
                + "."
            )
        else:
            ltl_phrase = "No LTL safety-property violations were detected."

        # Top factors prose
        if top_factors:
            factors_phrase = (
                "The primary risk drivers were: "
                + ", ".join(f"`{f}`" for f in top_factors)
                + "."
            )
        else:
            factors_phrase = "No dominant risk factors were identified."

        # Counterfactual prose
        if cf_changes:
            hints = "; ".join(
                f"`{c['feature']}` {c['from_value']} → {c['to_value']}"
                for c in cf_changes[:3]
            )
            cf_phrase = (
                f"To reduce the risk score below 0.40 (currently projected at "
                f"{cf_score:.3f} after changes), consider: {hints}."
            )
        else:
            cf_phrase = "No counterfactual adjustments are required."

        return (
            f"Tool `{tool}` {decision_phrase}. "
            f"The PFA monitor reported state `{pfa_state}`. "
            f"{factors_phrase} "
            f"{ltl_phrase} "
            f"{cf_phrase}"
        )

    # ------------------------------------------------------------------
    def _render_llm(self, event_dict: dict) -> str:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

            model_name = getattr(config, "OPENAI_MODEL", "gpt-4o")
            llm = ChatOpenAI(
                model=model_name,
                api_key=config.OPENAI_API_KEY,
                temperature=0.3,
                max_tokens=300,
            )

            template_text = self._render_template(event_dict)
            system_msg = SystemMessage(
                content=(
                    "You are a security analyst explaining AI agent risk decisions "
                    "to engineers. Be concise, specific, and use plain English. "
                    "Do not repeat the raw numbers unless they add clarity."
                )
            )
            human_msg = HumanMessage(
                content=(
                    f"Expand the following machine-generated summary into a clear, "
                    f"professional two-to-three sentence explanation:\n\n{template_text}"
                )
            )
            response = llm.invoke([system_msg, human_msg])
            return response.content.strip()
        except Exception:
            # Any LLM failure → fall back to template
            return self._render_template(event_dict)

    # ------------------------------------------------------------------
    def explain(self, event_dict: dict) -> str:
        """
        Returns a human-readable narrative string describing the event,
        the decision, top risk factors, LTL violations, PFA state, and
        a counterfactual hint.
        """
        if self._use_llm:
            return self._render_llm(event_dict)
        return self._render_template(event_dict)


# ── Public helper ──────────────────────────────────────────────────────────────


def get_xai_for_event(event_dict: dict) -> dict:
    """
    Runs all three explainers and returns a unified dict:
    {
        "shap":            {...},   # SHAPExplainer.explain output
        "counterfactual":  {...},   # DiCECounterfactual.generate output
        "narrative":       str,     # NarrativeExplainer.explain output
    }

    event_dict expected keys (all optional with sensible defaults):
        tool_name     str
        features      dict[str, float]   — the 20 feature values
        risk_score    float
        decision      str  ("BLOCK" | "WARN" | "ALLOW")
        ltl_violations list[str]
        pfa_state     str
    """
    features = event_dict.get("features", {})
    risk_score = float(event_dict.get("risk_score", event_dict.get("score", 0.0)))

    shap_exp = SHAPExplainer()
    shap_result = shap_exp.explain(features)

    cf_gen = DiCECounterfactual(shap_explainer=shap_exp)
    cf_result = cf_gen.generate(features, risk_score)

    # Augment event_dict with intermediate results so the narrative can use them
    enriched = dict(event_dict)
    enriched["shap"] = shap_result
    enriched["counterfactual"] = cf_result

    narrative_exp = NarrativeExplainer()
    narrative = narrative_exp.explain(enriched)

    return {
        "shap": shap_result,
        "counterfactual": cf_result,
        "narrative": narrative,
    }
