# IcM GraphRAG Backend

FastAPI backend that serves an agentic GraphRAG pipeline over synthetic 2026 Azure incidents.

## Setup

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,evals]"
cp .env.example .env   # then edit Azure OpenAI values
```

## Generate data + build index

```powershell
python -m scripts.gen_incidents
python -m scripts.build_index
```

## Run

```powershell
uvicorn app.main:app --reload --port 8000
```

Health check: http://localhost:8000/api/health

## Evals

```powershell
python -m evals.run
```
