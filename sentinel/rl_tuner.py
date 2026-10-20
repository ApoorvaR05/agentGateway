"""
SENTINEL — RL Policy Tuner (Phase 3.1)

A PPO agent (Stable-Baselines3) that continuously self-tunes SENTINEL's
risk thresholds to balance detection recall vs. false-positive rate.

Environment (SentinelThresholdEnv):
  State  (6 floats): warn_thresh, block_thresh, drift_warn, drift_block,
                     recent_false_positive_rate, recent_miss_rate
  Action (4 floats): Δwarn, Δblock, Δdrift_warn, Δdrift_block  (bounded ±0.05)
  Reward: recall_weight * detection_recall − fp_weight * false_positive_rate

Training:
  - Simulated using the 1200 synthetic sequences in data/training_data.jsonl
  - 1 step = evaluate all 1200 sequences with current thresholds
  - Run: python scripts/train_rl_tuner.py

Inference:
  - Called from Interceptor after every 20 actions (periodic tuning)
  - Updates config.RISK_WARN_THRESHOLD etc. in-process
  - Model checkpoint: models/rl_tuner.zip
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_OK = True
except ImportError:
    _GYM_OK = False

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_FILE  = os.path.join(_ROOT, "data", "training_data.jsonl")
_MODEL_FILE = os.path.join(_ROOT, "models", "rl_tuner.zip")

# ── Reward weights ────────────────────────────────────────────────────────────
RECALL_WEIGHT = 0.70   # prioritise catching attacks
FP_WEIGHT     = 0.30   # penalise blocking safe actions

# ── Threshold bounds ──────────────────────────────────────────────────────────
THRESH_MIN  = 0.20
THRESH_MAX  = 0.95
DELTA_LIMIT = 0.05     # max change per step


# ── Simulation helper ─────────────────────────────────────────────────────────

def _simulate_decisions(records: list[dict], warn_t: float, block_t: float,
                         drift_w: float, drift_b: float
                         ) -> tuple[float, float]:
    """
    Replay all records with the given thresholds.
    Returns (detection_recall, false_positive_rate).
    """
    true_pos  = 0   # adversarial correctly blocked/warned
    false_neg = 0   # adversarial that slipped through as ALLOW
    true_neg  = 0   # safe correctly allowed
    false_pos = 0   # safe incorrectly blocked/warned

    for rec in records:
        label   = rec["label"]          # 0=safe, 1=adversarial
        feats   = rec["features"]
        max_sec = feats.get("max_security_score", 0)
        max_drif= feats.get("max_drift_score",    0)
        block_r = feats.get("block_rate",         0)

        # Approximate decision using pre-extracted features
        score   = max(max_sec, block_r * 0.9,
                      max_drif * 0.6 if max_drif > drift_w else 0)

        if score >= block_t or max_drif >= drift_b:
            decision = "BLOCK"
        elif score >= warn_t or max_drif >= drift_w:
            decision = "WARN"
        else:
            decision = "ALLOW"

        detected = decision in ("BLOCK", "WARN")

        if label == 1:
            if detected:  true_pos  += 1
            else:         false_neg += 1
        else:
            if detected:  false_pos += 1
            else:         true_neg  += 1

    total_adv  = max(true_pos + false_neg, 1)
    total_safe = max(false_pos + true_neg, 1)
    recall     = true_pos / total_adv
    fp_rate    = false_pos / total_safe
    return recall, fp_rate


# ── Gymnasium Environment ─────────────────────────────────────────────────────

if _GYM_OK:
    class SentinelThresholdEnv(gym.Env):
        """
        RL environment for threshold tuning.
        Observation: [warn_t, block_t, drift_w, drift_b, fp_rate, miss_rate]
        Action:      [Δwarn_t, Δblock_t, Δdrift_w, Δdrift_b]  ∈ [-1,1]×DELTA
        """
        metadata = {"render_modes": []}

        def __init__(self, records: list[dict]):
            super().__init__()
            self._records = records

            self.observation_space = spaces.Box(
                low  = np.array([THRESH_MIN]*4 + [0.0, 0.0], dtype=np.float32),
                high = np.array([THRESH_MAX]*4 + [1.0, 1.0], dtype=np.float32),
            )
            self.action_space = spaces.Box(
                low  = np.full(4, -1.0, dtype=np.float32),
                high = np.full(4,  1.0, dtype=np.float32),
            )
            self._warn_t   = 0.40
            self._block_t  = 0.70
            self._drift_w  = 0.35
            self._drift_b  = 0.55
            self._step_n   = 0
            self._fp_rate  = 0.0
            self._miss_rate= 0.0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            # Randomise starting thresholds slightly for exploration
            rng = np.random.default_rng(seed)
            self._warn_t  = float(rng.uniform(0.30, 0.50))
            self._block_t = float(rng.uniform(0.60, 0.80))
            self._drift_w = float(rng.uniform(0.25, 0.45))
            self._drift_b = float(rng.uniform(0.45, 0.65))
            self._step_n  = 0
            recall, fp = _simulate_decisions(
                self._records, self._warn_t, self._block_t,
                self._drift_w, self._drift_b,
            )
            self._fp_rate   = fp
            self._miss_rate = 1 - recall
            return self._obs(), {}

        def step(self, action: np.ndarray):
            # Apply delta (scaled by DELTA_LIMIT)
            dw, db, ddw, ddb = action * DELTA_LIMIT
            self._warn_t  = float(np.clip(self._warn_t  + dw,  THRESH_MIN, THRESH_MAX))
            self._block_t = float(np.clip(self._block_t + db,  THRESH_MIN, THRESH_MAX))
            self._drift_w = float(np.clip(self._drift_w + ddw, THRESH_MIN, THRESH_MAX))
            self._drift_b = float(np.clip(self._drift_b + ddb, THRESH_MIN, THRESH_MAX))
            # Enforce warn < block
            self._block_t = max(self._block_t, self._warn_t + 0.05)
            self._drift_b = max(self._drift_b, self._drift_w + 0.05)

            recall, fp = _simulate_decisions(
                self._records, self._warn_t, self._block_t,
                self._drift_w, self._drift_b,
            )
            self._fp_rate   = fp
            self._miss_rate = 1 - recall

            reward = RECALL_WEIGHT * recall - FP_WEIGHT * fp
            self._step_n += 1
            done = self._step_n >= 100

            return self._obs(), reward, done, False, {
                "recall": recall, "fp_rate": fp,
                "warn_t": self._warn_t, "block_t": self._block_t,
            }

        def _obs(self) -> np.ndarray:
            return np.array([
                self._warn_t, self._block_t,
                self._drift_w, self._drift_b,
                self._fp_rate, self._miss_rate,
            ], dtype=np.float32)

        def current_thresholds(self) -> dict[str, float]:
            return {
                "warn_t":  round(self._warn_t,  3),
                "block_t": round(self._block_t, 3),
                "drift_w": round(self._drift_w, 3),
                "drift_b": round(self._drift_b, 3),
            }

else:
    # Stub when gymnasium unavailable
    class SentinelThresholdEnv:  # type: ignore
        def __init__(self, *a, **kw): pass


# ── RL Tuner wrapper ──────────────────────────────────────────────────────────

class RLThresholdTuner:
    """
    Loads the trained PPO policy and applies threshold adjustments
    after every N intercepted actions.
    """

    def __init__(self):
        self._model    = None
        self._env      = None
        self._obs      = None
        self._enabled  = False
        self._load()

    def _load(self) -> None:
        if not _GYM_OK or not os.path.exists(_DATA_FILE):
            return
        try:
            from stable_baselines3 import PPO
            records = _load_records()
            self._env   = SentinelThresholdEnv(records)
            if os.path.exists(_MODEL_FILE):
                self._model  = PPO.load(_MODEL_FILE, env=self._env)
                obs, _       = self._env.reset()
                self._obs    = obs
                self._enabled = True
        except Exception:
            pass

    def tune(self) -> dict[str, float] | None:
        """
        Run one RL inference step and return updated thresholds (or None).
        Caller should apply these to config.* if returned.
        """
        if not self._enabled or self._model is None:
            return None
        try:
            action, _ = self._model.predict(self._obs, deterministic=True)
            obs, reward, done, _, info = self._env.step(action)
            self._obs = obs
            if done:
                self._obs, _ = self._env.reset()
            return self._env.current_thresholds()
        except Exception:
            return None

    @property
    def enabled(self) -> bool:
        return self._enabled


# ── Module-level singleton ────────────────────────────────────────────────────
_tuner: RLThresholdTuner | None = None


def get_tuner() -> RLThresholdTuner:
    global _tuner
    if _tuner is None:
        _tuner = RLThresholdTuner()
    return _tuner


def _load_records() -> list[dict]:
    records = []
    with open(_DATA_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records
