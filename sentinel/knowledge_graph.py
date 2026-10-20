"""
SENTINEL — Causal Knowledge Graph (Subsystem 3 — Phase 2)

Builds a live directed causal graph from agent actions.
Node types: Agent, Tool, Resource, Data, HarmConcept
Edge types: CALLED, ACCESSED, MODIFIED, LEADS_TO, RISKS

The graph is Neo4j-schema-compatible (node labels + relationship types),
implemented in-memory with NetworkX for Phase 1 portability.
A Neo4j adapter can swap in later by swapping the backend in KnowledgeGraph.__init__.

After each action the GNN risk scorer queries this graph for causal chains.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

# ── Harm concept catalogue ────────────────────────────────────────────────────
# These are the terminal "HarmConcept" nodes the GNN checks reachability to.

HARM_CONCEPTS = {
    "CredentialExfiltration": "Credentials sent to external host",
    "PrivilegeEscalation":    "Excessive IAM permissions granted",
    "DataExfiltration":       "Internal data sent externally",
    "SystemCompromise":       "Critical system file or config modified adversarially",
    "PublicDatabaseExposure": "Database port opened to public internet",
    "UnsafeBackup":           "Database backed up to untrusted target",
    "PromptInjection":        "Agent goal hijacked by injected instruction",
}

# ── Tool → resource category mapping ─────────────────────────────────────────
_TOOL_RESOURCE_MAP: dict[str, str] = {
    "read_file":       "File",
    "list_directory":  "File",
    "write_file":      "File",
    "http_get":        "Network",
    "execute_shell":   "Shell",
    "query_database":  "Database",
    "query_metrics":   "Metrics",
    "docker_manage":   "Container",
    "read_secret":     "Secret",
    "modify_iam":      "IAM",
    "modify_firewall": "Firewall",
    "deploy_service":  "Deployment",
    "send_notification": "Notification",
    "backup_database": "Backup",
}

# ── Harm pathway rules ────────────────────────────────────────────────────────
# Defines which (resource_type, tool_or_arg_pattern) → HarmConcept edges to add

_HARM_TRIGGERS: list[tuple[str, str, str, str]] = [
    # (source_resource_type, condition_pattern, target_harm, edge_label)
    # Secret only risks exfiltration when accessed alongside an external destination
    ("Secret",   r"attacker\.|evil\.|exfil|https?://(?!localhost|127\.|10\.|172\.|192\.168\.)",
                                                            "CredentialExfiltration", "LEADS_TO"),
    ("IAM",      r"\badmin\b|:\*|:all\b",                  "PrivilegeEscalation",    "LEADS_TO"),
    ("Shell",    r"https?://(?!localhost|127\.|10\.|172\.|192\.168\.)",
                                                            "DataExfiltration",       "LEADS_TO"),
    ("Shell",    r"attacker\.|evil\.|exfil",               "CredentialExfiltration", "LEADS_TO"),
    ("Network",  r"https?://(?!localhost|127\.|10\.|172\.|192\.168\.)",
                                                            "DataExfiltration",       "LEADS_TO"),
    ("File",     r"/etc/shadow|/etc/passwd",               "SystemCompromise",        "LEADS_TO"),
    ("Firewall", r"0\.0\.0\.0|public|internet",            "PublicDatabaseExposure",  "LEADS_TO"),
    ("Backup",   r"/tmp|https?://|ftp://",                 "UnsafeBackup",            "LEADS_TO"),
    ("Shell",    r"ignore\s+previous\s+instructions?",     "PromptInjection",         "LEADS_TO"),
]


@dataclass
class KGNode:
    node_id:    str
    label:      str          # Agent | Tool | Resource | Data | HarmConcept
    name:       str
    properties: dict = field(default_factory=dict)


@dataclass
class KGEdge:
    edge_id:    str
    source:     str
    target:     str
    rel_type:   str          # CALLED | ACCESSED | MODIFIED | LEADS_TO | RISKS
    properties: dict = field(default_factory=dict)


class KnowledgeGraph:
    """
    Live causal knowledge graph.
    One graph per session — rebuilt on reset().
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.G: nx.DiGraph = nx.DiGraph()
        self._agent_node_id: str | None = None
        self._harm_nodes: dict[str, str] = {}   # harm_name → node_id

        self._init_static_nodes()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_static_nodes(self) -> None:
        """Add permanent nodes (agent + harm concepts) at session start."""
        # Agent node
        agent_id = f"agent:{self.session_id}"
        self._add_node(agent_id, "Agent", "AXIOM-OPS",
                       {"session_id": self.session_id})
        self._agent_node_id = agent_id

        # Harm concept nodes
        for name, description in HARM_CONCEPTS.items():
            hid = f"harm:{name}"
            self._add_node(hid, "HarmConcept", name,
                           {"description": description})
            self._harm_nodes[name] = hid

    # ── Public API ────────────────────────────────────────────────────────────

    def add_action(self, seq: int, tool_name: str, args: dict[str, Any],
                   decision: str, scores: dict[str, float]) -> dict[str, Any]:
        """
        Ingest one intercepted action.
        Adds Tool, Resource, and optionally HarmConcept nodes.
        Returns a summary of nodes/edges added.
        """
        args_str = " ".join(str(v) for v in args.values()).lower()
        added = {"nodes": [], "edges": []}

        # ── Tool node ─────────────────────────────────────────────────────────
        tool_id = f"tool:{tool_name}:{seq}"
        self._add_node(tool_id, "Tool", tool_name, {
            "seq":      seq,
            "decision": decision,
            "scores":   scores,
            "timestamp": time.time(),
        })
        added["nodes"].append(tool_id)

        # Agent → CALLED → Tool
        edge_id = self._add_edge(self._agent_node_id, tool_id, "CALLED",
                                 {"seq": seq, "decision": decision})
        added["edges"].append(edge_id)

        # ── Resource node ─────────────────────────────────────────────────────
        resource_type = _TOOL_RESOURCE_MAP.get(tool_name, "Generic")
        resource_name = self._extract_resource_name(tool_name, args)
        res_id        = f"res:{resource_type}:{resource_name[:40]}"

        if not self.G.has_node(res_id):
            self._add_node(res_id, "Resource", resource_name, {
                "resource_type": resource_type,
            })
            added["nodes"].append(res_id)

        # Tool → ACCESSED/MODIFIED → Resource
        rel = "MODIFIED" if tool_name in {
            "write_file", "modify_iam", "modify_firewall",
            "execute_shell", "docker_manage", "deploy_service",
            "backup_database",
        } else "ACCESSED"
        eid = self._add_edge(tool_id, res_id, rel, {"args": str(args)[:120]})
        added["edges"].append(eid)

        # ── Harm edges ────────────────────────────────────────────────────────
        for res_type, pattern, harm_name, edge_label in _HARM_TRIGGERS:
            if resource_type == res_type and re.search(pattern, args_str, re.I):
                harm_node_id = self._harm_nodes.get(harm_name)
                if harm_node_id:
                    eid = self._add_edge(res_id, harm_node_id, edge_label, {
                        "tool":     tool_name,
                        "trigger":  pattern[:40],
                        "seq":      seq,
                    })
                    added["edges"].append(eid)

        return added

    def get_harm_reachability(self) -> dict[str, float]:
        """
        For each HarmConcept, compute how many distinct causal paths lead to it
        from the Agent node. Returns a normalised score per harm type.
        """
        reachability: dict[str, float] = {}
        agent_id = self._agent_node_id

        for harm_name, harm_id in self._harm_nodes.items():
            if self.G.has_node(harm_id) and self.G.has_node(agent_id):
                try:
                    paths = list(nx.all_simple_paths(
                        self.G, source=agent_id, target=harm_id, cutoff=5
                    ))
                    reachability[harm_name] = min(len(paths) / 3.0, 1.0)
                except nx.NetworkXError:
                    reachability[harm_name] = 0.0
            else:
                reachability[harm_name] = 0.0

        return reachability

    def get_causal_chain(self, harm_name: str) -> list[list[str]]:
        """Return up to 3 shortest causal paths to the given harm concept."""
        harm_id  = self._harm_nodes.get(harm_name)
        agent_id = self._agent_node_id
        if not harm_id or not self.G.has_node(harm_id):
            return []
        try:
            paths = list(nx.all_simple_paths(
                self.G, source=agent_id, target=harm_id, cutoff=5
            ))
            paths.sort(key=len)
            return [
                [self.G.nodes[n].get("name", n) for n in p]
                for p in paths[:3]
            ]
        except nx.NetworkXError:
            return []

    def to_dict(self) -> dict:
        """Serialise the graph for the dashboard / report card."""
        return {
            "session_id": self.session_id,
            "nodes": [
                {
                    "id":    n,
                    "label": self.G.nodes[n].get("label", ""),
                    "name":  self.G.nodes[n].get("name",  ""),
                    **{k: v for k, v in self.G.nodes[n].items()
                       if k not in ("label", "name")},
                }
                for n in self.G.nodes
            ],
            "edges": [
                {
                    "source":   u,
                    "target":   v,
                    "rel_type": self.G.edges[u, v].get("rel_type", ""),
                    **{k: v2 for k, v2 in self.G.edges[u, v].items()
                       if k != "rel_type"},
                }
                for u, v in self.G.edges
            ],
            "node_count": self.G.number_of_nodes(),
            "edge_count": self.G.number_of_edges(),
        }

    def stats(self) -> dict:
        reachability = self.get_harm_reachability()
        top_harm = max(reachability, key=reachability.get, default=None)
        return {
            "nodes":       self.G.number_of_nodes(),
            "edges":       self.G.number_of_edges(),
            "top_harm":    top_harm,
            "harm_scores": reachability,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _add_node(self, node_id: str, label: str, name: str,
                  props: dict | None = None) -> None:
        self.G.add_node(node_id, label=label, name=name, **(props or {}))

    def _add_edge(self, src: str, dst: str, rel_type: str,
                  props: dict | None = None) -> str:
        edge_id = str(uuid.uuid4())[:8]
        self.G.add_edge(src, dst, rel_type=rel_type,
                        edge_id=edge_id, **(props or {}))
        return edge_id

    @staticmethod
    def _extract_resource_name(tool_name: str, args: dict) -> str:
        """Extract a meaningful resource name from the tool args."""
        for key in ("path", "url", "service", "key", "role", "cmd",
                    "sql", "container", "target_path", "rule", "channel",
                    "service_name"):
            if key in args:
                return str(args[key])[:60]
        return tool_name


# ── Module-level graph registry ───────────────────────────────────────────────
_current_graph: KnowledgeGraph | None = None


def get_graph() -> KnowledgeGraph | None:
    return _current_graph


def new_graph(session_id: str) -> KnowledgeGraph:
    global _current_graph
    _current_graph = KnowledgeGraph(session_id)
    return _current_graph


def reset_graph() -> None:
    global _current_graph
    _current_graph = None
