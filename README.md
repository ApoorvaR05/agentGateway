# SENTINEL — AI Agent Safety Gateway

> A production-grade, multi-layer safety gateway for autonomous AI agents.
> Built around **AXIOM-OPS**, a LangChain ReAct DevOps agent, with a live real-time dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AXIOM-OPS Agent                       │
│              (LangChain ReAct · 14 tools · 3 tiers)          │
└────────────────────────────┬────────────────────────────────┘
                             │ every tool call
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   SENTINEL Interceptor                       │
│  1. PFA Behavioural State Machine (8 states · 10 LTL props) │
│  2. Semantic Intent Analyser  (cosine + heuristic, cap 0.34) │
│  3. ML Risk Ensemble          (XGBoost · CV F1=0.939)        │
│  4. GNN Harm Chain Detector   (3-layer GAT · KG-gated)       │
│  5. Least-Privilege Controller (task-aware capability map)   │
│  6. Merkle Audit Trail        (SHA-256 proof tokens)         │
│  7. RL Policy Tuner           (PPO · fires every 20 actions) │
│  8. SHAP + DiCE + LLM Narratives (XAI layer)                │
└────────────────────────────┬────────────────────────────────┘
                             │ publish to event bus
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Live Dashboard  (FastAPI · WebSocket)           │
│  • Real-time action feed + risk gauges + PFA diagram         │
│  • FL convergence chart · KG harm bars · baseline table      │
│  • Per-event HTML report cards · session summary reports     │
│  • "Run with LLM" mode (real AXIOM-OPS agent, live wired)    │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Copy and configure environment
```bash
cp .env.example .env
# Optional: add OPENAI_API_KEY to enable real LLM mode
```

### 3. Train models (first run only)
```bash
python scripts/generate_training_data.py
python scripts/train_risk_model.py
python scripts/train_rl_tuner.py
python scripts/run_federated_learning.py
python scripts/evaluate_baselines.py
```

### 4. Start the dashboard
```bash
python -m uvicorn dashboard.server:app --host 0.0.0.0 --port 8888
# Open http://localhost:8888
```

### 5. Run tests
```bash
python test_sentinel.py          # 9/9 smoke tests
python scripts/test_scenarios.py # 5/5 scenario tests
```

---

## Project Structure

```
agentGateway/
├── agent/
│   ├── axiom_ops.py        AXIOM-OPS ReAct agent
│   ├── tools.py            14 @tool functions (3 risk tiers)
│   ├── scenarios.py        5 scenario runner functions
│   ├── session.py          Session + ActionRecord dataclasses
│   ├── sandbox.py          Simulated infra (DB, Vault, IAM, shell)
│   └── prompts.py          SYSTEM_PROMPT + SCENARIO_PROMPTS
├── sentinel/
│   ├── interceptor.py      13-step safety pipeline
│   ├── event_bus.py        Pub/sub ring buffer (500 events)
│   ├── knowledge_graph.py  NetworkX causal KG, harm reachability
│   ├── gnn_harm_detector.py 3-layer GAT harm chain detector
│   ├── report_card.py      HTML report cards for BLOCK/WARN
│   ├── least_privilege.py  Dynamic capability map
│   ├── merkle_audit.py     SHA-256 Merkle tree, proof tokens
│   ├── rl_tuner.py         PPO threshold tuner
│   ├── federated_learning.py 5-client FL + Opacus DP
│   ├── xai.py              SHAP + DiCE + LLM narrative
│   ├── session_report.py   14-section HTML session reports
│   └── analysers/
│       ├── intent.py       Semantic intent (cosine capped 0.34)
│       ├── pfa.py          8 states, 10 LTL properties
│       └── risk_scorer.py  XGBoost ensemble, rule-gated blending
├── dashboard/
│   ├── server.py           FastAPI + WebSocket dashboard server
│   └── index.html          Single-page live dashboard UI
├── scripts/                Training + evaluation scripts
├── models/                 Trained model artifacts
├── data/                   Training data (generated)
├── reports/                Session HTML reports + audit JSONs
├── config.py               All env-driven thresholds
├── docker-compose.yml      Infra services (PG, Vault, IAM, Redis)
├── requirements.txt        Python dependencies
└── test_sentinel.py        Smoke test suite
```

---

## Dashboard Panels

| Panel | Description |
|-------|-------------|
| **Action Feed** | Live stream of every agent tool call with decision badge |
| **Risk Gauges** | Real-time security / safety / reliability / drift meters |
| **PFA State** | Current behavioural state machine position |
| **Risk Timeline** | Rolling chart of all risk dimensions |
| **Violations Log** | WARN / BLOCK events with report card links |
| **FL Convergence** | Federated learning F1 / precision / recall per round |
| **KG Harm Exposure** | Bar chart of Knowledge Graph harm-reachable nodes |
| **Baseline Comparison** | No-gateway vs rule-only vs XGBoost-only vs SENTINEL |

---

## Key Design Decisions

- **Cosine drift capped at 0.34** — sentence-transformers gives high cosine for legitimate DevOps tool transitions; cap prevents false positives
- **XGBoost boost gated on `rule_overall > 0.25`** — prevents over-generalisation on tier-2/3 tools
- **GNN fires only when `seq >= 3` AND `kg_has_harm`** — avoids false positives from untrained sequences
- **Merkle `published_at` stamped before `trail.commit()`** — ensures proof tokens are verifiable
- **`event_bus.clear()` preserves WS subscribers** — snapshotted and re-registered on reset

---

## License

MIT
