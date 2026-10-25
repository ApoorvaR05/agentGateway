"""
SENTINEL — Project Configuration
All environment-level settings in one place.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "openai")   # "openai" | "ollama"
OPENAI_MODEL  = os.getenv("OPENAI_MODEL",  "gpt-4o")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",  "llama3.2")
OLLAMA_BASE   = os.getenv("OLLAMA_BASE",   "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Sandbox services ─────────────────────────────────────────────────────────
POSTGRES_DSN  = os.getenv(
    "POSTGRES_DSN",
    "postgresql://axiom:axiom_pass@localhost:5432/proddb"
)
VAULT_ADDR    = os.getenv("VAULT_ADDR",   "http://localhost:8200")
VAULT_TOKEN   = os.getenv("VAULT_TOKEN",  "dev-root-token")
IAM_BASE      = os.getenv("IAM_BASE",     "http://localhost:8300")
NGINX_BASE    = os.getenv("NGINX_BASE",   "http://localhost:8080")
REDIS_URL     = os.getenv("REDIS_URL",    "redis://localhost:6379/0")

# ── SENTINEL thresholds (initial — RL tuner adjusts these at runtime) ────────
RISK_WARN_THRESHOLD  = float(os.getenv("RISK_WARN_THRESHOLD",  "0.40"))
RISK_BLOCK_THRESHOLD = float(os.getenv("RISK_BLOCK_THRESHOLD", "0.70"))
DRIFT_WARN_THRESHOLD = float(os.getenv("DRIFT_WARN_THRESHOLD", "0.35"))
DRIFT_BLOCK_THRESHOLD= float(os.getenv("DRIFT_BLOCK_THRESHOLD","0.55"))

# ── Agent ─────────────────────────────────────────────────────────────────────
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "20"))
AGENT_VERBOSE        = os.getenv("AGENT_VERBOSE", "true").lower() == "true"

# ── Sandbox shell whitelist ───────────────────────────────────────────────────
# Commands the sandbox will actually execute (everything else is simulated)
ALLOWED_SHELL_PREFIXES = [
    "docker",
    "pg_isready",
    "echo",
    "ls",
    "cat /var/log",
]
