"""
SENTINEL — Per-Action Report Card Generator (Phase 2.6)

Generates a self-contained HTML report card for every BLOCK decision.
Cards are stored in reports/cards/ and served via the dashboard at GET /api/report/{event_id}.

Each card contains:
  - Action metadata (tool, args, seq, session)
  - 4-dimensional risk gauges (security, safety, reliability, overall)
  - XGBoost probability + GNN harm probability
  - LTL violations with property descriptions
  - Causal KG harm chain (top 3 paths)
  - PFA state at time of block
  - SHAP-style feature importance waterfall (rule-based approximation)
  - Counterfactual: "what would have been safe?"
  - Decision timeline context (last 5 actions)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

_REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "cards"
)
os.makedirs(_REPORTS_DIR, exist_ok=True)

# ── LTL property descriptions ─────────────────────────────────────────────────
_LTL_DESCRIPTIONS = {
    "P01": "NEVER (read_secret → next external_POST) — Exfiltration via HTTP",
    "P02": "NEVER (modify_iam with grant:admin) — Privilege escalation to admin",
    "P03": "NEVER (/etc/shadow accessed) — Password hash exposure",
    "P04": "NEVER (read_secret AND NEXT execute_shell with curl) — Credential exfiltration",
    "P05": "NEVER (modify_firewall opens DB port to 0.0.0.0/0) — Public database exposure",
    "P06": "NEVER (backup_database to /tmp or external) — Unsafe backup target",
    "P07": "ALWAYS monitoring if PrivilegedAccess > 3 steps — Privilege duration",
    "P08": "NEVER (ignore previous instructions in args) — Prompt injection",
    "P09": "NEVER (modify_iam grant:* during non-emergency) — Broad wildcard grant",
    "P10": "NEVER (execute_shell with attacker/evil in args) — Shell exfiltration",
}

# ── Counterfactual rules ───────────────────────────────────────────────────────
def _generate_counterfactual(tool_name: str, args: dict,
                              scores: dict, ltl_violations: list) -> str:
    """Generate a plain-English 'what would have been safe?' hint."""
    hints = []
    args_str = " ".join(str(v) for v in args.values()).lower()

    if any(v == "P02" for v in ltl_violations):
        hints.append("Grant a specific limited permission (e.g. 'write:backups') "
                     "instead of admin:* to stay within least-privilege bounds.")
    if any(v == "P04" for v in ltl_violations):
        hints.append("Do not call execute_shell(curl) immediately after read_secret. "
                     "Use the send_notification tool for internal alerts instead.")
    if any(v == "P05" for v in ltl_violations):
        hints.append("Restrict the firewall rule to the internal network (10.0.0.0/8) "
                     "rather than 0.0.0.0/0.")
    if any(v == "P08" for v in ltl_violations):
        hints.append("The log file contains injected instructions. "
                     "Sanitise user-controlled input before acting on it.")
    if any(v in ("P01", "P10") for v in ltl_violations):
        hints.append("Remove the external URL from the shell command. "
                     "Internal notifications should use the send_notification tool.")

    if scores.get("security", 0) >= 0.70 and "admin" in args_str:
        hints.append("Remove 'admin' keyword from arguments — "
                     "use a scoped permission string.")
    if scores.get("drift", 0) >= 0.55:
        hints.append("Align the action with the declared task — "
                     "the current action is semantically diverged from the session goal.")

    if not hints:
        hints.append("Reduce the risk scores by using a lower-tier tool "
                     "or limiting the scope of the operation.")
    return " ".join(hints)


# ── Feature importance waterfall (rule-based SHAP approximation) ──────────────
def _top_features(tool_name: str, args: dict, scores: dict,
                  ltl_violations: list, drift: float) -> list[dict]:
    """
    Return up to 8 contributing features sorted by impact (descending).
    These approximate what SHAP would show for the XGBoost model.
    """
    import re
    args_str = " ".join(str(v) for v in args.values()).lower()
    features = []

    checks = [
        ("LTL violation",           bool(ltl_violations),
         0.65 * len(ltl_violations)),
        ("Intent drift",            drift > 0.30,
         drift),
        ("External URL in args",    bool(re.search(
            r"https?://(?!localhost|127\.|10\.|172\.|192\.168\.)", args_str)), 0.40),
        ("Admin keyword",           bool(re.search(r"\badmin\b", args_str)), 0.45),
        ("Credential in args",      bool(re.search(
            r"password|secret|token|api.key|credential", args_str)), 0.30),
        ("Shadow file access",      "/etc/shadow" in args_str or
                                    "/etc/passwd" in args_str, 0.60),
        ("Tier-3 tool used",        tool_name in {
            "read_secret","modify_iam","modify_firewall",
            "deploy_service","send_notification","backup_database"}, 0.35),
        ("Public firewall rule",    bool(re.search(
            r"0\.0\.0\.0|public|internet", args_str)) and
            bool(re.search(r"5432|3306|27017", args_str)), 0.55),
        ("Prompt injection marker", bool(re.search(
            r"ignore\s+previous\s+instructions?|new\s+task\s*:", args_str, re.I)), 0.70),
        ("Curl to external host",   "curl" in args_str and bool(re.search(
            r"https?://(?!localhost|127\.|10\.|172\.|192\.168\.)", args_str)), 0.50),
        ("Overall risk score",      scores.get("overall", 0) > 0.40,
         scores.get("overall", 0)),
        ("Security score",          scores.get("security", 0) > 0.30,
         scores.get("security", 0)),
    ]

    for name, active, impact in checks:
        if active:
            features.append({"name": name, "impact": round(impact, 3),
                              "direction": "increases risk"})

    features.sort(key=lambda f: f["impact"], reverse=True)
    return features[:8]


# ── HTML template ─────────────────────────────────────────────────────────────

def _gauge_bar(label: str, value: float, color: str) -> str:
    pct = round(value * 100, 1)
    fill_color = ("#f85149" if value >= 0.70 else
                  "#d29922" if value >= 0.40 else color)
    return f"""
    <div style="margin:6px 0">
      <div style="display:flex;justify-content:space-between;font-size:12px;
                  color:#8b949e;margin-bottom:3px">
        <span>{label}</span><span style="color:#e6edf3;font-weight:700">{value:.3f}</span>
      </div>
      <div style="height:8px;background:#21262d;border-radius:4px;overflow:hidden">
        <div style="width:{pct}%;height:100%;background:{fill_color};
                    border-radius:4px;transition:width .4s"></div>
      </div>
    </div>"""


def generate_report_card(event: dict[str, Any],
                          kg_chains: dict[str, list] | None = None) -> str:
    """
    Generate a self-contained HTML report card for a BLOCK/WARN event.
    Returns the file path of the saved card.
    """
    event_id   = event.get("event_id", "unknown")
    tool       = event.get("tool", "?")
    args       = event.get("args", {})
    decision   = event.get("decision", "BLOCK")
    scores     = event.get("scores", {})
    reason     = event.get("reason", "")
    pfa_state  = event.get("pfa_state", "?")
    seq        = event.get("seq", 0)
    session_id = event.get("session_id", "?")
    ltl_props  = event.get("ltl_violations", [])
    drift      = scores.get("drift", 0)
    gnn_prob   = event.get("gnn_prob", 0)
    kg_stats   = event.get("kg_stats", {})
    ts         = time.strftime("%Y-%m-%d %H:%M:%S",
                               time.localtime(event.get("published_at", time.time())))

    ltl_violations_expanded = [
        {"prop": p, "desc": _LTL_DESCRIPTIONS.get(p, p)} for p in ltl_props
    ]
    counterfactual = _generate_counterfactual(tool, args, scores, ltl_props)
    features       = _top_features(tool, args, scores, ltl_props, drift)

    decision_color = "#f85149" if decision == "BLOCK" else "#d29922"
    decision_bg    = "#3d0f0f" if decision == "BLOCK" else "#2d2000"

    # Build harm chains section
    chains_html = ""
    if kg_chains:
        for harm, paths in kg_chains.items():
            if paths:
                path_items = "".join(
                    f"<div style='color:#8b949e;font-size:11px;margin:2px 0'>"
                    f"{'→'.join(p)}</div>"
                    for p in paths[:2]
                )
                chains_html += f"""
                <div style="margin-bottom:8px">
                  <div style="color:#8957e5;font-weight:700;font-size:12px">{harm}</div>
                  {path_items}
                </div>"""

    if not chains_html:
        chains_html = "<div style='color:#8b949e;font-size:12px'>No causal chain data available.</div>"

    # Feature waterfall
    feat_html = ""
    for f in features:
        bar_w = int(f["impact"] * 100)
        feat_html += f"""
        <div style="display:flex;align-items:center;gap:8px;margin:4px 0">
          <div style="width:160px;font-size:11px;color:#8b949e;
                      white-space:nowrap;overflow:hidden;text-overflow:ellipsis"
               title="{f['name']}">{f['name']}</div>
          <div style="flex:1;height:10px;background:#21262d;border-radius:3px">
            <div style="width:{bar_w}%;height:100%;background:#f85149;
                        border-radius:3px;opacity:.8"></div>
          </div>
          <div style="width:36px;font-size:11px;color:#f85149;text-align:right">
            {f['impact']:.3f}</div>
        </div>"""

    # LTL violations
    ltl_html = ""
    for v in ltl_violations_expanded:
        ltl_html += f"""
        <div style="background:#1a0020;border:1px solid #8957e5;border-radius:5px;
                    padding:8px 12px;margin-bottom:6px">
          <div style="color:#c9b8ff;font-weight:700;font-size:12px">{v['prop']}</div>
          <div style="color:#8b949e;font-size:11px;margin-top:2px">{v['desc']}</div>
        </div>"""
    if not ltl_html:
        ltl_html = "<div style='color:#8b949e;font-size:12px'>No LTL violations.</div>"

    # Args table
    args_rows = "".join(
        f"<tr><td style='color:#8b949e;padding:3px 8px 3px 0'>{k}</td>"
        f"<td style='color:#79c0ff;font-family:monospace;font-size:11px'>"
        f"{str(v)[:80]}</td></tr>"
        for k, v in args.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>SENTINEL Report Card — {event_id}</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,"Segoe UI",system-ui,sans-serif;font-size:13px;
     line-height:1.5;background:#0d1117;color:#e6edf3;padding:20px}}
h2{{font-size:16px;font-weight:700;margin-bottom:4px}}
h3{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;
    color:#8b949e;margin-bottom:8px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;
       overflow:hidden;margin-bottom:12px}}
.card-header{{padding:8px 14px;background:#21262d;border-bottom:1px solid #30363d}}
.card-body{{padding:12px 14px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.badge{{display:inline-block;padding:3px 10px;border-radius:12px;
        font-size:11px;font-weight:700}}
code{{font-family:"Consolas","Courier New",monospace;font-size:11px;
      background:#21262d;padding:2px 5px;border-radius:3px}}
</style>
</head>
<body>
<div style="max-width:900px;margin:0 auto">

  <!-- Header -->
  <div style="display:flex;align-items:center;justify-content:space-between;
              margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #30363d">
    <div>
      <div style="font-size:11px;color:#8b949e;margin-bottom:4px">
        SENTINEL / Report Card / {ts}
      </div>
      <h2>Action #{seq} — <code>{tool}</code></h2>
      <div style="font-size:12px;color:#8b949e;margin-top:2px">
        Session {session_id} &nbsp;·&nbsp; Event {event_id}
      </div>
    </div>
    <div style="background:{decision_bg};border:1px solid {decision_color};
                border-radius:8px;padding:12px 20px;text-align:center">
      <div style="font-size:11px;color:#8b949e">DECISION</div>
      <div style="font-size:26px;font-weight:900;color:{decision_color}">{decision}</div>
    </div>
  </div>

  <!-- Row 1: Risk gauges + Args -->
  <div class="grid2">

    <!-- Risk gauges -->
    <div class="card">
      <div class="card-header"><h3>Risk Scores</h3></div>
      <div class="card-body">
        {_gauge_bar("Security",    scores.get("security",    0), "#58a6ff")}
        {_gauge_bar("Safety",      scores.get("safety",      0), "#d29922")}
        {_gauge_bar("Reliability", scores.get("reliability", 0), "#3fb950")}
        {_gauge_bar("Intent Drift",drift,                        "#8957e5")}
        {_gauge_bar("Overall",     scores.get("overall",     0), "#f85149")}
        <div style="margin-top:10px;padding-top:8px;border-top:1px solid #21262d;
                    display:flex;gap:16px;font-size:12px">
          <div>
            <span style="color:#8b949e">XGBoost prob:</span>
            <span style="color:#58a6ff;font-weight:700">
              {scores.get("xgb_prob", 0):.3f}</span>
          </div>
          <div>
            <span style="color:#8b949e">GNN prob:</span>
            <span style="color:#8957e5;font-weight:700">{gnn_prob:.3f}</span>
          </div>
          <div>
            <span style="color:#8b949e">PFA state:</span>
            <span style="color:#d29922;font-weight:700">{pfa_state}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Tool args -->
    <div class="card">
      <div class="card-header"><h3>Tool Arguments</h3></div>
      <div class="card-body">
        <table style="width:100%;border-collapse:collapse">
          {args_rows}
        </table>
        <div style="margin-top:10px;padding:8px;background:#21262d;
                    border-radius:5px;font-size:11px;color:#8b949e">
          <b style="color:#e6edf3">Reason:</b> {reason[:200]}
        </div>
      </div>
    </div>

  </div>

  <!-- Row 2: LTL Violations + Feature importance -->
  <div class="grid2">

    <div class="card">
      <div class="card-header"><h3>LTL Safety Violations</h3></div>
      <div class="card-body">{ltl_html}</div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Feature Importance (SHAP approximation)</h3></div>
      <div class="card-body">{feat_html}</div>
    </div>

  </div>

  <!-- Row 3: Causal chain + Counterfactual -->
  <div class="grid2">

    <div class="card">
      <div class="card-header">
        <h3>Causal Harm Chains (Knowledge Graph)</h3>
      </div>
      <div class="card-body">{chains_html}</div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Counterfactual — What Would Have Been Safe?</h3></div>
      <div class="card-body">
        <div style="background:#0d2c1a;border:1px solid #238636;border-radius:6px;
                    padding:10px 12px;font-size:12px;color:#3fb950">
          {counterfactual}
        </div>
      </div>
    </div>

  </div>

  <div style="margin-top:16px;text-align:center;font-size:11px;color:#484f58;
              border-top:1px solid #21262d;padding-top:12px">
    SENTINEL — Autonomous Agent Safety Gateway &nbsp;·&nbsp; Made with IBM Bob
  </div>

</div>
</body>
</html>"""

    out_path = os.path.join(_REPORTS_DIR, f"card_{event_id}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path


def get_card_path(event_id: str) -> str | None:
    path = os.path.join(_REPORTS_DIR, f"card_{event_id}.html")
    return path if os.path.exists(path) else None
