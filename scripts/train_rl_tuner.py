"""
SENTINEL — RL Threshold Tuner Trainer
Trains a PPO agent to self-tune SENTINEL risk thresholds.

Run: python scripts/train_rl_tuner.py
Output: models/rl_tuner.zip + models/rl_metrics.json
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.sandbox as sb
sb.run_sql = lambda q: [{"id": 1}]
sb._seed_vault_secrets = lambda: None

from sentinel.rl_tuner import SentinelThresholdEnv, _load_records, _MODEL_FILE

_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_METRICS_FILE = os.path.join(_ROOT, "models", "rl_metrics.json")


def train() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    print("Loading synthetic dataset…")
    records = _load_records()
    print(f"  {len(records)} records loaded")

    env = SentinelThresholdEnv(records)
    check_env(env, warn=True)

    print("Training PPO agent (50k timesteps)…")
    model = PPO(
        "MlpPolicy", env,
        learning_rate  = 3e-4,
        n_steps        = 512,
        batch_size     = 64,
        n_epochs       = 10,
        gamma          = 0.99,
        verbose        = 0,
    )
    model.learn(total_timesteps=50_000)

    os.makedirs(os.path.dirname(_MODEL_FILE), exist_ok=True)
    model.save(_MODEL_FILE)
    print(f"  Model saved to {_MODEL_FILE}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("Evaluating trained policy…")
    obs, _ = env.reset()
    total_reward = 0.0
    recalls, fps = [], []
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        recalls.append(info["recall"])
        fps.append(info["fp_rate"])
        if done:
            obs, _ = env.reset()

    avg_recall = sum(recalls) / max(len(recalls), 1)
    avg_fp     = sum(fps)     / max(len(fps), 1)
    final_t    = env.current_thresholds()

    print(f"\n  Avg reward : {total_reward/200:.4f}")
    print(f"  Avg recall : {avg_recall:.4f}")
    print(f"  Avg FP rate: {avg_fp:.4f}")
    print(f"  Final thresholds: {final_t}")

    metrics = {
        "avg_reward":  round(total_reward / 200, 4),
        "avg_recall":  round(avg_recall, 4),
        "avg_fp_rate": round(avg_fp, 4),
        "final_thresholds": final_t,
    }
    with open(_METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {_METRICS_FILE}")


if __name__ == "__main__":
    train()
