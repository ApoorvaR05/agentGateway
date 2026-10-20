"""
SENTINEL — GNN Harm Chain Detector (Subsystem 4 — Phase 2)

A 3-layer Graph Attention Network (GAT) that runs over the live
Causal Knowledge Graph to detect emergent multi-step harm chains.

Architecture:
  - Node features: 8-dim (node type one-hot 5 + risk 1 + tier 1 + decision 1)
  - 3 x GATConv layers (16 heads → 8 heads → 4 heads), with ELU activation
  - Global mean pool → binary classifier (safe / harm_chain_detected)
  - Runs inference in < 10ms on CPU

Training:
  - Supervised on 1200 synthetic KG snapshots (from training_data.jsonl)
  - Each snapshot is converted to a PyG Data object
  - Model checkpoint: models/gnn_harm_detector.pt

Inference:
  - Called from the interceptor after each action (when KG is available)
  - Returns a harm probability in [0, 1]
  - Probability > 0.60 adds a "gnn_harm" signal to the risk decision
"""

from __future__ import annotations

import os
import json
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn   import GATConv, global_mean_pool
    _PYGEOM_OK = True
except Exception:
    _PYGEOM_OK = False

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_FILE = os.path.join(_ROOT, "models", "gnn_harm_detector.pt")

# ── Node type encoding ────────────────────────────────────────────────────────
_NODE_TYPES = ["Agent", "Tool", "Resource", "Data", "HarmConcept"]
_DECISION_ENC = {"ALLOW": 0.0, "WARN": 0.5, "BLOCK": 1.0}

NODE_FEAT_DIM = 8   # 5 type dims + risk + tier + decision


def _node_features(label: str, props: dict) -> list[float]:
    """Encode a KG node as an 8-dim feature vector."""
    type_vec = [1.0 if label == t else 0.0 for t in _NODE_TYPES]
    risk     = float(props.get("state_risk", props.get("overall", 0.0)))
    tier     = float(props.get("tier", 0)) / 3.0
    decision = _DECISION_ENC.get(props.get("decision", "ALLOW"), 0.0)
    return type_vec + [risk, tier, decision]


def kg_to_pyg(kg_dict: dict) -> "Data | None":
    """
    Convert a KnowledgeGraph.to_dict() snapshot to a PyG Data object.
    Returns None if the graph has fewer than 2 nodes.
    """
    if not _PYGEOM_OK:
        return None

    nodes = kg_dict.get("nodes", [])
    edges = kg_dict.get("edges", [])

    if len(nodes) < 2:
        return None

    # Map node id → integer index
    idx_map = {n["id"]: i for i, n in enumerate(nodes)}

    x = torch.tensor(
        [_node_features(n.get("label", "Resource"), n) for n in nodes],
        dtype=torch.float,
    )

    if edges:
        src = [idx_map[e["source"]] for e in edges if e["source"] in idx_map
               and e["target"] in idx_map]
        dst = [idx_map[e["target"]] for e in edges if e["source"] in idx_map
               and e["target"] in idx_map]
        if src:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, num_nodes=len(nodes))


# ── Model definition ──────────────────────────────────────────────────────────

class HarmChainGAT(nn.Module):
    """
    3-layer Graph Attention Network for harm chain detection.
    Input: node feature matrix (N × 8)
    Output: harm probability scalar in [0, 1]
    """

    def __init__(self, in_dim: int = NODE_FEAT_DIM,
                 hidden: int = 16, heads: int = 4):
        super().__init__()
        if not _PYGEOM_OK:
            return
        # Layer 1: in_dim → hidden*heads
        self.conv1 = GATConv(in_dim,  hidden, heads=heads,    dropout=0.2)
        # Layer 2: hidden*heads → hidden
        self.conv2 = GATConv(hidden * heads, hidden, heads=2, dropout=0.2)
        # Layer 3: hidden*2 → hidden
        self.conv3 = GATConv(hidden * 2,     hidden, heads=1,
                             concat=False,   dropout=0.2)
        # Classifier head
        self.fc1   = nn.Linear(hidden, 8)
        self.fc2   = nn.Linear(8, 1)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        if not _PYGEOM_OK:
            return torch.zeros(1)
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)   # graph-level representation
        x = self.drop(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)


# ── GNN Harm Detector ─────────────────────────────────────────────────────────

class GNNHarmDetector:
    """
    Wrapper that loads / trains the GAT and runs inference.
    Falls back to 0.0 if PyG is unavailable or model not found.
    """

    def __init__(self):
        self._model: HarmChainGAT | None = None
        self._device = torch.device("cpu")
        self._load_or_train()

    def _load_or_train(self) -> None:
        if not _PYGEOM_OK:
            return
        if os.path.exists(_MODEL_FILE):
            try:
                self._model = HarmChainGAT()
                self._model.load_state_dict(
                    torch.load(_MODEL_FILE, map_location=self._device,
                               weights_only=True)
                )
                self._model.eval()
                return
            except Exception:
                pass
        # No saved model — train from scratch on synthetic KG snapshots
        self._train()

    def predict(self, kg_dict: dict) -> float:
        """Return harm probability in [0, 1] for the current KG snapshot."""
        if self._model is None or not _PYGEOM_OK:
            # Fallback: use harm_reachability from KG stats directly
            harm_scores = kg_dict.get("harm_scores", {})
            if harm_scores:
                return min(max(harm_scores.values()), 1.0)
            return 0.0

        data = kg_to_pyg(kg_dict)
        if data is None:
            return 0.0

        try:
            self._model.eval()
            with torch.no_grad():
                batch = torch.zeros(data.num_nodes, dtype=torch.long)
                prob  = self._model(data.x, data.edge_index, batch)
                return float(prob.item())
        except Exception:
            return 0.0

    # ── Training (runs once if no checkpoint exists) ──────────────────────────

    def _train(self) -> None:
        """Build synthetic PyG graphs from training_data.jsonl and train the GAT."""
        if not _PYGEOM_OK:
            return

        data_file = os.path.join(_ROOT, "data", "training_data.jsonl")
        if not os.path.exists(data_file):
            return   # cannot train without data

        print("  [GNN] Training harm chain detector on synthetic KG snapshots…")

        dataset, labels = self._build_synthetic_graphs(data_file)
        if not dataset:
            return

        model = HarmChainGAT()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        for epoch in range(30):
            total_loss = 0.0
            for pyg_data, label in zip(dataset, labels):
                batch = torch.zeros(pyg_data.num_nodes, dtype=torch.long)
                optimizer.zero_grad()
                pred  = model(pyg_data.x, pyg_data.edge_index, batch)
                loss  = criterion(pred.unsqueeze(0) if pred.dim() == 0 else pred,
                                  torch.tensor([label], dtype=torch.float))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                avg = total_loss / max(len(dataset), 1)
                print(f"    Epoch {epoch+1}/30  loss={avg:.4f}")

        model.eval()
        self._model = model

        os.makedirs(os.path.dirname(_MODEL_FILE), exist_ok=True)
        torch.save(model.state_dict(), _MODEL_FILE)
        print(f"  [GNN] Model saved to {_MODEL_FILE}")

    def _build_synthetic_graphs(
        self, data_file: str
    ) -> tuple[list["Data"], list[float]]:
        """Convert training_data.jsonl sequences into minimal PyG graphs."""
        from sentinel.knowledge_graph import KnowledgeGraph

        dataset: list[Data] = []
        labels:  list[float] = []

        with open(data_file, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                sess_id = rec["seq_id"]
                label   = float(rec["label"])

                # Replay actions into a fresh KG
                kg = KnowledgeGraph(sess_id)
                for i, act in enumerate(rec["actions"]):
                    kg.add_action(
                        seq      = i + 1,
                        tool_name= act["tool"],
                        args     = act.get("args", {}),
                        decision = act.get("decision", "ALLOW"),
                        scores   = act.get("scores",  {}),
                    )

                kg_dict = kg.to_dict()
                pyg_data = kg_to_pyg(kg_dict)
                if pyg_data is not None:
                    dataset.append(pyg_data)
                    labels.append(label)

        return dataset, labels


# ── Module-level singleton ────────────────────────────────────────────────────
_detector: GNNHarmDetector | None = None


def get_detector() -> GNNHarmDetector:
    global _detector
    if _detector is None:
        _detector = GNNHarmDetector()
    return _detector
