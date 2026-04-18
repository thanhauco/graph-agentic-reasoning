# Plan: GraphRAG Agentic Reasoning for IcM

Full-stack demo: FastAPI backend with an agentic GraphRAG pipeline over a synthetic 2026 Azure incident knowledge graph, and a React+Vite+Tailwind+shadcn/ui light-theme frontend with Cytoscape (KG) and React Flow (agent trace).

## Stack (confirmed)
- Backend: Python 3.11, FastAPI, Uvicorn, LangGraph + LangChain, Azure OpenAI (chat + embeddings), NetworkX (in-memory graph), scikit-learn/numpy (vector search), python-louvain (community detection), Pydantic v2.
- Frontend: React 18 + Vite + TypeScript, Tailwind, shadcn/ui, Cytoscape.js + `react-cytoscapejs`, React Flow, TanStack Query, SSE streaming.
- Data: 400 synthetic Azure incidents (2026) generated deterministically with Faker seed.

## Architecture

```
frontend (Vite) ──SSE/REST──> FastAPI
                                 │
                    ┌────────────┴───────────────┐
                    │ LangGraph Agent            │
                    │  nodes: plan → act → reflect│
                    │  tools: graph_search,       │
                    │         neighbor_expand,    │
                    │         community_summary,  │
                    │         vector_search,      │
                    │         incident_lookup,    │
                    │         temporal_filter     │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ GraphRAG Index          │
                    │  - NetworkX KG           │
                    │  - Community reports     │
                    │  - Embedding store (npy) │
                    └─────────────────────────┘
```

### Knowledge Graph Schema
Node types: Incident, Service, Region, Team, Owner, RootCauseCategory, Mitigation, Component, Customer.
Edge types: AFFECTS, LOCATED_IN, OWNED_BY, ASSIGNED_TO, CAUSED_BY, MITIGATED_BY, DEPENDS_ON, IMPACTS, RELATED_TO.

### GraphRAG Index Build (offline)
1. Load 400 incidents JSON.
2. Extract entities/relationships from structured fields + lightweight LLM NER on `description` and `rca`.
3. Build NetworkX graph, persist to `graph.pickle`.
4. Louvain community detection → per-community LLM summaries → `communities.json`.
5. Embed incident title+summary and community reports with Azure OpenAI → `embeddings.npz`.

### Agentic Reasoning (LangGraph)
- State: `messages`, `plan`, `scratchpad`, `tool_calls`, `citations`.
- Flow: `planner` → `router` (global/local/hybrid) → `executor` (tool-calling loop, max 6 iters) → `critic` → `synthesizer` (final answer with inline `[INC-xxxx]` citations).
- SSE events: `plan`, `tool_call`, `tool_result`, `thought`, `token`, `final`.

## Synthetic Data (400 incidents, 2026)
`backend/data/incidents.json`. Deterministic Faker-seeded generator. Includes 8–12 "storm" clusters (e.g., Jan 2026 AOAI capacity, Mar 2026 Front Door cert rotation).

## Phases
1. **Scaffold** — FastAPI + Vite/React/Tailwind/shadcn + root README + docker-compose.
2. **Data & Index** — generate incidents → extract entities → build NetworkX + communities → embed → `build_index.py`.
3. **Retrieval & Tools** — local/global/drift search + LangChain tools.
4. **Agent** — LangGraph state machine + SSE streaming + session store.
5. **Backend API** — `/api/graph`, `/api/incidents`, `/api/stats`, `/api/chat/stream`.
6. **Frontend** — shell → Dashboard → Incidents → KG Explorer → Chat with live agent trace.
7. **Polish & Verify** — evals (ragas, ~15 gold Q/A), seeded example prompts, README.

## Confirmed Addenda
- Azure OpenAI defaults: chat `gpt-4o`, embeddings `text-embedding-3-large`.
- `backend/evals/` with ragas harness.
- Root `docker-compose.yml` alongside dev scripts.
