"""
SENTINEL — Federated Learning Runner
Run: python scripts/run_federated_learning.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.sandbox as sb
sb.run_sql = lambda q: [{"id": 1}]
sb._seed_vault_secrets = lambda: None

from sentinel.federated_learning import run_federated_learning

if __name__ == "__main__":
    print("\n=== SENTINEL Federated Learning (5 clients, 10 rounds, DP enabled) ===\n")
    result = run_federated_learning(use_dp=True)
    final  = result["final"]
    print(f"\nFinal: F1={final['f1']:.4f}  recall={final['recall']:.4f}  "
          f"precision={final['precision']:.4f}  "
          f"cumulative_eps={result['cumulative_epsilon']:.3f}")
