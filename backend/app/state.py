from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from app.config import Settings


@dataclass
class AppState:
    incidents: list[dict[str, Any]] = field(default_factory=list)
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    communities: list[dict[str, Any]] = field(default_factory=list)
    incident_ids: list[str] = field(default_factory=list)
    incident_embeddings: np.ndarray | None = None
    community_ids: list[str] = field(default_factory=list)
    community_embeddings: np.ndarray | None = None
    sessions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def load(cls, settings: Settings) -> "AppState":
        st = cls()
        data_path = settings.data_path / "incidents.json"
        if data_path.exists():
            st.incidents = json.loads(data_path.read_text(encoding="utf-8"))
        idx = settings.index_path
        g_path = idx / "graph.pickle"
        if g_path.exists():
            with g_path.open("rb") as fh:
                st.graph = pickle.load(fh)
        c_path = idx / "communities.json"
        if c_path.exists():
            st.communities = json.loads(c_path.read_text(encoding="utf-8"))
        emb_path = idx / "embeddings.npz"
        if emb_path.exists():
            z = np.load(emb_path, allow_pickle=False)
            if "incident_ids" in z:
                st.incident_ids = list(z["incident_ids"])
                st.incident_embeddings = z["incident_embeddings"]
            if "community_ids" in z:
                st.community_ids = list(z["community_ids"])
                st.community_embeddings = z["community_embeddings"]
        return st

    def incident_by_id(self, incident_id: str) -> dict[str, Any] | None:
        for inc in self.incidents:
            if inc.get("incidentId") == incident_id:
                return inc
        return None
