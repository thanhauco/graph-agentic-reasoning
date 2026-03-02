# IcM GraphRAG — Agentic Reasoning

A full-stack demo that applies **GraphRAG** and **agentic reasoning** (LangGraph) over **400 synthetic 2026 Azure incidents** for Incident Management (IcM).

- **Backend:** FastAPI, LangGraph + LangChain, Azure OpenAI (chat + embeddings), NetworkX knowledge graph, Louvain community detection, SSE streaming.
- **Frontend:** React + Vite + TypeScript, Tailwind + shadcn/ui (light theme), Cytoscape.js (knowledge graph), React Flow (live agent trace), TanStack Query.
- **Data:** 400 deterministic synthetic incidents with realistic 2026 "storm" clusters.

## Quick start

### 1. Backend
```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,evals]"
copy .env.example .env    # edit Azure OpenAI values
python -m scripts.gen_incidents
python -m scripts.build_index
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend
```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

### Docker Compose (optional)
```powershell
docker compose up --build
```

## Project Layout
```
backend/
  app/           FastAPI app, routes, graphrag pipeline, agent
  scripts/       gen_incidents.py, build_index.py
  evals/         ragas harness (15 gold Q/A)
  data/          generated incidents + index artifacts
frontend/
  src/
    pages/       Dashboard, Chat, Explorer, Incidents
    components/  KnowledgeGraph, AgentTrace, ChatPanel
    lib/         api client, utils
plan.md         Implementation plan
```

See [plan.md](plan.md) for architecture details.
