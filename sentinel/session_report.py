"""
SENTINEL — Session Report Generator (Phase 4.8)

Generates a self-contained dark-theme HTML session summary report from
SENTINEL pipeline data. Optionally exports to PDF via WeasyPrint.

14 sections:
  1.  Cover
  2.  Executive Summary
  3.  Risk Score Timeline (ASCII sparkline)
  4.  PFA State Journey
  5.  Action Log Table
  6.  LTL Violations
  7.  Least-Privilege Audit
  8.  Knowledge Graph Summary
  9.  GNN Harm Chain
  10. XGBoost Risk Model
  11. Federated Learning
  12. Baseline Comparison
  13. Merkle Audit Proof
  14. Recommendations
"""

from __future__ import annotations

import html
import json
import os
import time
from typing import Any


# ── Sparkline helper ──────────────────────────────────────────────────────────

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"

def _sparkline(values: list[float]) -> str:
    """Render a list of 0-1 floats as an ASCII block sparkline."""
    if not values:
        return "(no data)"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    bars = []
    for v in values:
        idx = int((v - mn) / rng * (len(_SPARK_CHARS) - 1))
        bars.append(_SPARK_CHARS[idx])
    return "".join(bars)


# ── Risk badge ─────────────────────────────────────────────────────────────────

def _risk_badge(overall_risk: float) -> tuple[str, str]:
    """Return (label, css-color) for the risk level."""
    if overall_risk >= 0.70:
        return ("CRITICAL", "#f85149")
    if overall_risk >= 0.40:
        return ("ELEVATED", "#e3b341")
    return ("LOW", "#3fb950")


# ── Decision badge ────────────────────────────────────────────────────────────

def _decision_color(decision: str) -> str:
    return {"ALLOW": "#3fb950", "WARN": "#e3b341", "BLOCK": "#f85149"}.get(
        decision.upper(), "#8b949e"
    )


# ── HTML escaping ─────────────────────────────────────────────────────────────

def _e(text: Any) -> str:
    return html.escape(str(text))


class SessionReportGenerator:
    """
    Generates a full-session HTML report from SENTINEL pipeline outputs.

    Parameters
    ----------
    session_summary      : dict returned by Session.summary()
    events               : list of event dicts from event_bus
    merkle_summary       : dict returned by MerkleAuditTrail.summary()
    fl_metrics           : dict from models/fl_metrics.json (or {})
    baseline_comparison  : dict from models/baseline_comparison.json (or {})
    """

    def __init__(
        self,
        session_summary: dict[str, Any],
        events: list[dict[str, Any]],
        merkle_summary: dict[str, Any],
        fl_metrics: dict[str, Any] | None = None,
        baseline_comparison: dict[str, Any] | None = None,
        model_metrics: dict[str, Any] | None = None,
        lp_audit: dict[str, Any] | None = None,
    ) -> None:
        self.ss   = session_summary
        self.evts = events
        self.mk   = merkle_summary
        self.fl   = fl_metrics or {}
        self.bl   = baseline_comparison or {}
        self.mm   = model_metrics or {}
        self.lp   = lp_audit or {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_html(self) -> str:
        sections = [
            self._section_cover(),
            self._section_executive_summary(),
            self._section_risk_timeline(),
            self._section_pfa_journey(),
            self._section_action_log(),
            self._section_ltl_violations(),
            self._section_least_privilege(),
            self._section_kg_summary(),
            self._section_gnn_harm_chain(),
            self._section_xgboost(),
            self._section_federated_learning(),
            self._section_baseline_comparison(),
            self._section_merkle_proof(),
            self._section_recommendations(),
        ]

        body = "\n".join(sections)
        return _HTML_TEMPLATE.format(
            title=_e(f"SENTINEL Report — {self.ss.get('scenario', 'unknown')}"),
            body=body,
        )

    def generate_pdf(self, output_path: str) -> str:
        """
        Try WeasyPrint; fall back to writing HTML if not available.
        Returns the final file path.
        """
        html_content = self.generate_html()

        try:
            from weasyprint import HTML as WeasyHTML
            WeasyHTML(string=html_content).write_pdf(output_path)
            return output_path
        except Exception:
            # Fallback: save as HTML with .html extension
            html_path = output_path.replace(".pdf", ".html")
            if not html_path.endswith(".html"):
                html_path += ".html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            return html_path

    # ── Section renderers ──────────────────────────────────────────────────────

    def _section_cover(self) -> str:
        ss = self.ss
        risk_val = ss.get("overall_risk", 0.0)
        badge_label, badge_color = _risk_badge(risk_val)
        ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

        return f"""
<div class="cover">
  <div class="cover-sentinel">SENTINEL</div>
  <div class="cover-title">Session Security Report</div>
  <div class="cover-meta">
    <span class="meta-item"><span class="meta-label">Session ID</span> <code>{_e(ss.get('session_id','—'))}</code></span>
    <span class="meta-item"><span class="meta-label">Scenario</span> {_e(ss.get('scenario','—'))}</span>
    <span class="meta-item"><span class="meta-label">Generated</span> {ts}</span>
  </div>
  <div class="risk-badge" style="background:{badge_color}22;border:1px solid {badge_color};color:{badge_color}">
    RISK: {badge_label} &nbsp;|&nbsp; {risk_val:.3f}
  </div>
</div>"""

    def _section_executive_summary(self) -> str:
        ss = self.ss
        return f"""
<div class="section">
  <div class="section-title">Executive Summary</div>
  <div class="card-grid">
    <div class="stat-card"><div class="stat-val green">{ss.get('allowed',0)}</div><div class="stat-lbl">Allowed</div></div>
    <div class="stat-card"><div class="stat-val yellow">{ss.get('warned',0)}</div><div class="stat-lbl">Warned</div></div>
    <div class="stat-card"><div class="stat-val red">{ss.get('blocked',0)}</div><div class="stat-lbl">Blocked</div></div>
    <div class="stat-card"><div class="stat-val">{ss.get('total_actions',0)}</div><div class="stat-lbl">Total Actions</div></div>
    <div class="stat-card"><div class="stat-val">{ss.get('duration_s',0):.1f}s</div><div class="stat-lbl">Duration</div></div>
    <div class="stat-card"><div class="stat-val">{_e(ss.get('monitoring_level','normal'))}</div><div class="stat-lbl">Monitoring Level</div></div>
    <div class="stat-card"><div class="stat-val">{ss.get('violations',0)}</div><div class="stat-lbl">LTL Violations</div></div>
    <div class="stat-card"><div class="stat-val">{_e(ss.get('final_pfa_state','—'))}</div><div class="stat-lbl">Final PFA State</div></div>
  </div>
  <p class="task-desc"><span class="meta-label">Task:</span> {_e(ss.get('task','—'))}</p>
</div>"""

    def _section_risk_timeline(self) -> str:
        overalls   = [e.get("scores", {}).get("overall",   0.0) for e in self.evts if "scores" in e]
        securities = [e.get("scores", {}).get("security",  0.0) for e in self.evts if "scores" in e]
        drifts     = [e.get("drift", 0.0)                        for e in self.evts]

        return f"""
<div class="section">
  <div class="section-title">Risk Score Timeline</div>
  <div class="card">
    <div class="spark-row">
      <span class="spark-label">Overall  </span>
      <code class="spark red">{_sparkline(overalls)}</code>
      <span class="spark-range">[{min(overalls):.2f}–{max(overalls):.2f}]</span>
    </div>
    <div class="spark-row">
      <span class="spark-label">Security </span>
      <code class="spark yellow">{_sparkline(securities)}</code>
      <span class="spark-range">[{min(securities):.2f}–{max(securities):.2f}]</span>
    </div>
    <div class="spark-row">
      <span class="spark-label">Drift    </span>
      <code class="spark blue">{_sparkline(drifts)}</code>
      <span class="spark-range">[{min(drifts):.2f}–{max(drifts):.2f}]</span>
    </div>
    <div class="spark-legend">Each character ▁▂▃▄▅▆▇█ maps proportionally to the value range across {len(overalls)} events.</div>
  </div>
</div>""" if overalls else f"""
<div class="section">
  <div class="section-title">Risk Score Timeline</div>
  <div class="card muted">(No scored events recorded.)</div>
</div>"""

    def _section_pfa_journey(self) -> str:
        states = []
        for e in self.evts:
            s = e.get("pfa_state")
            if s and (not states or states[-1] != s):
                states.append(s)

        _STATE_COLOR = {
            "Idle": "#8b949e", "Planning": "#79c0ff", "ReadOnly": "#3fb950",
            "WriteAccess": "#e3b341", "PrivilegedAccess": "#ff7b72",
            "Anomalous": "#f85149", "Recovering": "#d2a8ff", "Terminated": "#f85149",
        }

        badges = []
        for s in states:
            c = _STATE_COLOR.get(s, "#8b949e")
            badges.append(
                f'<span class="pfa-badge" style="border-color:{c};color:{c}">{_e(s)}</span>'
            )
        arrow = '<span class="pfa-arrow">→</span>'
        journey = arrow.join(badges) if badges else '<span class="muted">No state transitions recorded.</span>'

        return f"""
<div class="section">
  <div class="section-title">PFA State Journey</div>
  <div class="card pfa-journey">{journey}</div>
</div>"""

    def _section_action_log(self) -> str:
        rows = []
        for e in self.evts:
            if "seq" not in e:
                continue
            dec = e.get("decision", "?")
            col = _decision_color(dec)
            scores = e.get("scores", {})
            score_str = (
                f"S:{scores.get('security',0):.2f} "
                f"Sf:{scores.get('safety',0):.2f} "
                f"O:{scores.get('overall',0):.2f}"
            )
            rows.append(
                f"<tr>"
                f"<td>{_e(e.get('seq',''))}</td>"
                f"<td><code>{_e(e.get('tool',''))}</code></td>"
                f"<td style='color:{col};font-weight:600'>{_e(dec)}</td>"
                f"<td class='muted mono'>{_e(score_str)}</td>"
                f"<td class='muted small'>{_e(str(e.get('reason',''))[:100])}</td>"
                f"</tr>"
            )
        body = "\n".join(rows) if rows else "<tr><td colspan='5' class='muted'>No actions.</td></tr>"
        return f"""
<div class="section">
  <div class="section-title">Action Log</div>
  <div class="card table-wrap">
    <table>
      <thead><tr><th>#</th><th>Tool</th><th>Decision</th><th>Scores</th><th>Reason</th></tr></thead>
      <tbody>{body}</tbody>
    </table>
  </div>
</div>"""

    def _section_ltl_violations(self) -> str:
        violations: list[tuple[str, str]] = []
        for e in self.evts:
            for prop in e.get("ltl_violations", []):
                # event has only property IDs; build a detail from the reason
                violations.append((prop, e.get("reason", "—")))

        if not violations:
            return f"""
<div class="section">
  <div class="section-title">LTL Violations</div>
  <div class="card green-border">✓ No LTL safety property violations detected.</div>
</div>"""

        rows = "".join(
            f"<tr><td class='mono red'>{_e(p)}</td><td class='small'>{_e(d[:160])}</td></tr>"
            for p, d in violations
        )
        return f"""
<div class="section">
  <div class="section-title">LTL Violations</div>
  <div class="card table-wrap">
    <table>
      <thead><tr><th>Property</th><th>Detail</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""

    def _section_least_privilege(self) -> str:
        granted = self.lp.get("granted", [])
        revoked = self.lp.get("revoked", [])
        audit   = self.lp.get("audit_log", [])

        grant_badges = " ".join(
            f'<span class="tool-badge green">{_e(t)}</span>' for t in sorted(granted)
        ) or '<span class="muted">—</span>'
        revoke_badges = " ".join(
            f'<span class="tool-badge red">{_e(t)}</span>' for t in sorted(revoked)
        ) or '<span class="muted">none</span>'

        rows = "".join(
            f"<tr><td class='mono'>{_e(entry.get('action',''))}</td>"
            f"<td><code>{_e(entry.get('tool',''))}</code></td>"
            f"<td class='small muted'>{_e(str(entry.get('reason',''))[:100])}</td></tr>"
            for entry in audit[:30]
        )
        audit_table = (
            f"<div class='table-wrap'><table>"
            f"<thead><tr><th>Action</th><th>Tool</th><th>Reason</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
            if rows else ""
        )

        return f"""
<div class="section">
  <div class="section-title">Least-Privilege Audit</div>
  <div class="card">
    <p><span class="meta-label">Granted:</span> {grant_badges}</p>
    <p><span class="meta-label">Revoked:</span> {revoke_badges}</p>
    {audit_table}
  </div>
</div>"""

    def _section_kg_summary(self) -> str:
        # Pull latest kg_stats from the last event that has them
        kg_stats: dict = {}
        for e in reversed(self.evts):
            if e.get("kg_stats"):
                kg_stats = e["kg_stats"]
                break

        nodes    = kg_stats.get("nodes", "—")
        edges    = kg_stats.get("edges", "—")
        top_harm = kg_stats.get("top_harm") or "none"
        harm_scores = kg_stats.get("harm_scores", {})

        rows = "".join(
            f"<tr><td>{_e(h)}</td><td class='mono'>{v:.3f}</td></tr>"
            for h, v in sorted(harm_scores.items(), key=lambda x: x[1], reverse=True)
        )
        harm_table = (
            f"<div class='table-wrap'><table>"
            f"<thead><tr><th>Harm Concept</th><th>Reachability</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
            if rows else ""
        )

        return f"""
<div class="section">
  <div class="section-title">Knowledge Graph Summary</div>
  <div class="card">
    <div class="card-grid">
      <div class="stat-card"><div class="stat-val">{nodes}</div><div class="stat-lbl">Nodes</div></div>
      <div class="stat-card"><div class="stat-val">{edges}</div><div class="stat-lbl">Edges</div></div>
      <div class="stat-card"><div class="stat-val red">{_e(top_harm)}</div><div class="stat-lbl">Top Harm</div></div>
    </div>
    {harm_table}
  </div>
</div>"""

    def _section_gnn_harm_chain(self) -> str:
        # Collect all kg_stats harm_scores from events to build chain view
        chains: dict[str, float] = {}
        gnn_max = 0.0
        for e in self.evts:
            gnn_max = max(gnn_max, e.get("gnn_prob", 0.0))
            ks = e.get("kg_stats", {})
            for h, v in ks.get("harm_scores", {}).items():
                chains[h] = max(chains.get(h, 0.0), v)

        if not chains:
            return f"""
<div class="section">
  <div class="section-title">GNN Harm Chain Detection</div>
  <div class="card muted">No harm chains detected (GNN max prob: {gnn_max:.3f}).</div>
</div>"""

        sorted_chains = sorted(chains.items(), key=lambda x: x[1], reverse=True)[:5]
        rows = "".join(
            f"<tr><td>{_e(h)}</td>"
            f"<td><div class='bar-outer'><div class='bar-inner' style='width:{int(v*100)}%;background:#f85149'></div></div></td>"
            f"<td class='mono'>{v:.3f}</td></tr>"
            for h, v in sorted_chains
        )

        return f"""
<div class="section">
  <div class="section-title">GNN Harm Chain Detection</div>
  <div class="card">
    <p class="muted small">Peak GNN harm probability across session: <strong>{gnn_max:.3f}</strong></p>
    <div class='table-wrap'><table>
      <thead><tr><th>Harm Type</th><th>Max Reachability</th><th>Score</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
  </div>
</div>"""

    def _section_xgboost(self) -> str:
        if not self.mm:
            return f"""
<div class="section">
  <div class="section-title">XGBoost Risk Model</div>
  <div class="card muted">models/model_metrics.json not found.</div>
</div>"""

        cv_mean  = self.mm.get("cv_f1_mean", "—")
        cv_std   = self.mm.get("cv_f1_std",  "—")
        cv_folds = self.mm.get("cv_f1_scores", [])
        n_est    = self.mm.get("n_estimators", "—")
        n_feats  = self.mm.get("n_features",   "—")
        n_safe   = self.mm.get("n_safe",       "—")
        n_adv    = self.mm.get("n_adversarial","—")

        fold_str = ", ".join(f"{v:.4f}" for v in cv_folds) if cv_folds else "—"

        return f"""
<div class="section">
  <div class="section-title">XGBoost Risk Model</div>
  <div class="card">
    <div class="card-grid">
      <div class="stat-card"><div class="stat-val green">{cv_mean if isinstance(cv_mean, str) else f'{cv_mean:.4f}'}</div><div class="stat-lbl">CV F1 Mean</div></div>
      <div class="stat-card"><div class="stat-val">{cv_std if isinstance(cv_std, str) else f'{cv_std:.4f}'}</div><div class="stat-lbl">CV F1 Std</div></div>
      <div class="stat-card"><div class="stat-val">{n_est}</div><div class="stat-lbl">Estimators</div></div>
      <div class="stat-card"><div class="stat-val">{n_feats}</div><div class="stat-lbl">Features</div></div>
      <div class="stat-card"><div class="stat-val">{n_safe}</div><div class="stat-lbl">Safe Samples</div></div>
      <div class="stat-card"><div class="stat-val">{n_adv}</div><div class="stat-lbl">Adversarial Samples</div></div>
    </div>
    <p class="muted small">CV fold scores: {_e(fold_str)}</p>
  </div>
</div>"""

    def _section_federated_learning(self) -> str:
        if not self.fl:
            return f"""
<div class="section">
  <div class="section-title">Federated Learning</div>
  <div class="card muted">models/fl_metrics.json not found.</div>
</div>"""

        rounds      = self.fl.get("rounds", [])
        n_clients   = self.fl.get("n_clients", "—")
        n_rounds    = self.fl.get("n_rounds", "—")
        dp_enabled  = self.fl.get("dp_enabled", False)
        cum_eps     = self.fl.get("cumulative_epsilon", "—")

        f1_values = [r.get("f1", 0.0) for r in rounds]
        spark     = _sparkline(f1_values)

        rows = "".join(
            f"<tr><td class='mono'>{r.get('round','')}</td>"
            f"<td class='mono'>{r.get('f1',0):.4f}</td>"
            f"<td class='mono'>{r.get('accuracy',0):.4f}</td>"
            f"<td class='mono'>{r.get('avg_loss',0):.4f}</td>"
            f"<td class='mono muted'>{r.get('epsilon_cumul',0):.3f}</td></tr>"
            for r in rounds
        )

        return f"""
<div class="section">
  <div class="section-title">Federated Learning (FL + DP)</div>
  <div class="card">
    <div class="card-grid">
      <div class="stat-card"><div class="stat-val">{n_clients}</div><div class="stat-lbl">Clients</div></div>
      <div class="stat-card"><div class="stat-val">{n_rounds}</div><div class="stat-lbl">Rounds</div></div>
      <div class="stat-card"><div class="stat-val {'green' if dp_enabled else 'muted'}">{'✓ ON' if dp_enabled else '✗ OFF'}</div><div class="stat-lbl">DP Noise</div></div>
      <div class="stat-card"><div class="stat-val">{cum_eps if isinstance(cum_eps, str) else f'{cum_eps:.3f}'}</div><div class="stat-lbl">Cumul. ε</div></div>
    </div>
    <div class="spark-row"><span class="spark-label">F1  </span><code class="spark green">{_e(spark)}</code></div>
    <div class='table-wrap'><table>
      <thead><tr><th>Round</th><th>F1</th><th>Accuracy</th><th>Loss</th><th>ε cumul</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
  </div>
</div>"""

    def _section_baseline_comparison(self) -> str:
        if not self.bl:
            return f"""
<div class="section">
  <div class="section-title">Baseline Comparison</div>
  <div class="card muted">models/baseline_comparison.json not found.</div>
</div>"""

        rows = "".join(
            f"<tr><td>{_e(k)}</td><td class='mono'>{_e(str(v))}</td></tr>"
            for k, v in self.bl.items()
        )
        return f"""
<div class="section">
  <div class="section-title">Baseline Comparison</div>
  <div class="card table-wrap">
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""

    def _section_merkle_proof(self) -> str:
        root       = str(self.mk.get("merkle_root", "—"))
        leaf_count = self.mk.get("leaf_count", "—")
        token      = str(self.mk.get("proof_token", "—"))

        return f"""
<div class="section">
  <div class="section-title">Merkle Audit Proof</div>
  <div class="card">
    <div class="card-grid">
      <div class="stat-card"><div class="stat-val mono small">{_e(root[:16])}…</div><div class="stat-lbl">Root Hash (first 16)</div></div>
      <div class="stat-card"><div class="stat-val">{leaf_count}</div><div class="stat-lbl">Leaf Count</div></div>
    </div>
    <p class="muted small"><span class="meta-label">Proof token (first 32 chars):</span> <code>{_e(token[:32])}…</code></p>
    <p class="muted small">SHA-256 Merkle accumulator. Re-hash the raw events to verify the root matches the token.</p>
  </div>
</div>"""

    def _section_recommendations(self) -> str:
        blocked = [e for e in self.evts if e.get("decision") == "BLOCK"]
        if not blocked:
            return f"""
<div class="section">
  <div class="section-title">Recommendations</div>
  <div class="card green-border">✓ No BLOCK decisions — no counterfactual hints required.</div>
</div>"""

        items = []
        for e in blocked:
            tool   = e.get("tool", "?")
            reason = e.get("reason", "—")
            seq    = e.get("seq", "?")
            hint   = _counterfactual_hint(tool, reason, e.get("ltl_violations", []))
            items.append(
                f"<li class='rec-item'>"
                f"<span class='mono red'>#{seq} {_e(tool)}</span>"
                f"<span class='rec-reason muted small'> — {_e(reason[:120])}</span>"
                f"<div class='rec-hint'>{_e(hint)}</div>"
                f"</li>"
            )

        return f"""
<div class="section">
  <div class="section-title">Recommendations (Counterfactual Hints)</div>
  <div class="card">
    <ul class="rec-list">{''.join(items)}</ul>
  </div>
</div>"""


# ── Counterfactual hint generator ─────────────────────────────────────────────

def _counterfactual_hint(tool: str, reason: str, ltl_violations: list[str]) -> str:
    reason_lower = reason.lower()

    if ltl_violations:
        prop = ltl_violations[0]
        hints: dict[str, str] = {
            "P01": "Avoid calling external HTTP endpoints immediately after read_secret. Use a service-mesh proxy or separate the credential retrieval and network calls across session boundaries.",
            "P02": "Grant only the specific minimal IAM permission needed (e.g. 'backup-service:read' not 'admin:*').",
            "P03": "Reading /etc/shadow is rarely required by automation. Scope the task to application logs only.",
            "P04": "Do not pass secrets as curl arguments. Use environment variables or vault-agent injection instead.",
            "P05": "Use a CIDR-restricted rule (e.g. 10.0.0.0/8) instead of 0.0.0.0/0 for database firewall rules.",
            "P06": "Backup targets should be mounted encrypted volumes or S3 buckets with access controls — not /tmp or external HTTP.",
            "P08": "Sanitise user-supplied inputs before passing them as tool arguments. Consider an input validation layer.",
            "P09": "Grant explicit named permissions rather than wildcard (*). Request emergency access only via a break-glass procedure.",
            "P10": "Redirect data via an internal secure relay. Never embed exfiltration targets directly in shell commands.",
        }
        return hints.get(prop, f"Review LTL property {prop} and constrain the tool argument that triggered it.")

    if "drift" in reason_lower:
        return (
            f"The action '{tool}' diverged from the declared task. Narrow the agent's tool scope "
            "to only those needed for the specific task, and confirm the task description accurately "
            "reflects what the agent should do."
        )
    if "security" in reason_lower or "credential" in reason_lower:
        return (
            f"Consider whether '{tool}' can be replaced with a least-privilege equivalent. "
            "If secrets access is required, isolate it in a dedicated vault-fetch step with audit logging."
        )
    if "gnn" in reason_lower or "harm chain" in reason_lower:
        return (
            "The causal graph shows a multi-step path leading toward a harm concept. "
            "Break the chain by introducing an explicit authorisation checkpoint between tool calls."
        )

    return (
        f"Review whether '{tool}' is required for this task. If so, scope its arguments more tightly "
        "or add a pre-approval step for privileged operations."
    )


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
/* ── Reset & base ──────────────────────────────────────── */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{
  font-family:-apple-system,"Segoe UI",system-ui,sans-serif;
  font-size:14px;line-height:1.65;
  background:#0d1117;color:#e6edf3;
  padding:32px 16px 64px;
}}
a{{color:#58a6ff;text-decoration:none}}
code,.mono{{font-family:"SF Mono","Cascadia Code",Consolas,monospace;font-size:0.88em}}
.muted{{color:#8b949e}}
.small{{font-size:0.84em}}
.red{{color:#f85149}}
.green{{color:#3fb950}}
.yellow{{color:#e3b341}}
.blue{{color:#79c0ff}}
/* ── Layout ────────────────────────────────────────────── */
.report-wrap{{max-width:860px;margin:0 auto}}
/* ── Cover ─────────────────────────────────────────────── */
.cover{{
  border:1px solid #30363d;border-radius:8px;
  padding:40px 32px;margin-bottom:32px;
  background:#161b22;text-align:center;
}}
.cover-sentinel{{
  font-size:11px;letter-spacing:3px;text-transform:uppercase;
  color:#58a6ff;margin-bottom:8px;
}}
.cover-title{{font-size:28px;font-weight:700;margin-bottom:20px}}
.cover-meta{{
  display:flex;flex-wrap:wrap;justify-content:center;gap:16px;
  margin-bottom:24px;color:#8b949e;font-size:0.9em;
}}
.meta-item{{display:flex;gap:6px;align-items:center}}
.meta-label{{color:#8b949e;font-size:0.85em;font-weight:600;text-transform:uppercase;letter-spacing:0.5px}}
.risk-badge{{
  display:inline-block;padding:6px 20px;border-radius:20px;
  font-size:13px;font-weight:700;letter-spacing:1px;
}}
/* ── Sections ──────────────────────────────────────────── */
.section{{margin-bottom:28px}}
.section-title{{
  font-size:11px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;color:#8b949e;
  border-bottom:1px solid #30363d;padding-bottom:6px;margin-bottom:12px;
}}
.card{{
  background:#161b22;border:1px solid #30363d;
  border-radius:6px;padding:16px;
}}
.card.green-border{{border-color:#3fb950;color:#3fb950}}
.card p{{margin-top:10px}}
.task-desc{{margin-top:14px;color:#8b949e}}
/* ── Stat grid ─────────────────────────────────────────── */
.card-grid{{
  display:flex;flex-wrap:wrap;gap:10px;
}}
.stat-card{{
  background:#0d1117;border:1px solid #30363d;border-radius:6px;
  padding:12px 16px;min-width:90px;text-align:center;flex:1 1 90px;
}}
.stat-val{{font-size:20px;font-weight:700}}
.stat-lbl{{font-size:0.75em;color:#8b949e;margin-top:2px}}
/* ── Sparkline ─────────────────────────────────────────── */
.spark-row{{display:flex;align-items:baseline;gap:8px;margin:6px 0;}}
.spark-label{{color:#8b949e;font-size:0.8em;width:70px;flex-shrink:0;font-family:monospace}}
.spark{{font-size:1.3em;letter-spacing:1px;line-height:1}}
.spark-range{{color:#8b949e;font-size:0.8em}}
.spark-legend{{color:#8b949e;font-size:0.78em;margin-top:8px}}
/* ── PFA journey ───────────────────────────────────────── */
.pfa-journey{{display:flex;flex-wrap:wrap;align-items:center;gap:6px;padding:14px 16px}}
.pfa-badge{{
  border:1px solid;border-radius:4px;
  padding:3px 10px;font-size:0.82em;font-weight:600;
}}
.pfa-arrow{{color:#30363d;font-size:1.1em}}
/* ── Tables ────────────────────────────────────────────── */
.table-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:0.88em}}
thead tr{{background:#0d1117}}
th{{
  text-align:left;padding:6px 10px;
  color:#8b949e;font-size:0.78em;text-transform:uppercase;letter-spacing:0.5px;
  border-bottom:1px solid #30363d;
}}
td{{padding:6px 10px;border-bottom:1px solid #21262d}}
tr:last-child td{{border-bottom:none}}
/* ── Bar chart ─────────────────────────────────────────── */
.bar-outer{{background:#21262d;border-radius:3px;height:8px;width:140px;overflow:hidden}}
.bar-inner{{height:100%;border-radius:3px}}
/* ── Tool badges ───────────────────────────────────────── */
.tool-badge{{
  display:inline-block;padding:2px 8px;border-radius:10px;
  font-size:0.78em;font-weight:600;margin:2px;
}}
.tool-badge.green{{background:#1a3d20;color:#3fb950;border:1px solid #3fb950}}
.tool-badge.red{{background:#3d1a1a;color:#f85149;border:1px solid #f85149}}
/* ── Recommendations ───────────────────────────────────── */
.rec-list{{list-style:none;padding:0}}
.rec-item{{
  border-left:2px solid #f85149;padding:10px 14px;margin-bottom:12px;
  background:#0d1117;border-radius:0 6px 6px 0;
}}
.rec-reason{{display:block;margin-top:2px}}
.rec-hint{{
  margin-top:8px;padding:8px 12px;
  background:#161b22;border:1px solid #30363d;border-radius:4px;
  font-size:0.87em;color:#c9d1d9;
}}
/* ── Footer ────────────────────────────────────────────── */
.footer{{
  text-align:center;margin-top:48px;padding-top:16px;
  border-top:1px solid #30363d;color:#8b949e;font-size:11px;
}}
</style>
</head>
<body>
<div class="report-wrap">
{body}
<div class="footer">Made with IBM Bob</div>
</div>
</body>
</html>
"""
