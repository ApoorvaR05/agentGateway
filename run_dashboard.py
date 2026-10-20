"""
SENTINEL — Dashboard Launcher
Run with:  python run_dashboard.py
Then open: http://localhost:8888
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "8888"))
    print(f"\n  SENTINEL Dashboard starting on http://localhost:{port}")
    print("  Open that URL in your browser.\n")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "dashboard.server:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload",
        "--log-level", "warning",
    ])
