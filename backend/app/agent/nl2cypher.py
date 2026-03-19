"""LLM-powered NL → Cypher pipeline with embedding re-ranking + answer synthesis.

Flow for `POST /api/graph/query`:

    1. LLM(intent) → JSON {intent, cypher, params, explanation}
    2. Validate Cypher is read-only, execute against Memgraph
    3. Pull incident rows out of the result, embed the question with
       Azure OpenAI embeddings, cosine-rerank vs. pre-computed incident
       embeddings, blend with the DB score
    4. LLM(synthesis) → short grounded answer citing incident IDs
    5. Return everything to the UI

Graceful degradation: if Azure OpenAI or Memgraph is not configured, fall
back to the heuristic `parse_query` path already used in the repo.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.agent.nl_query import parse_query
from app.config import get_settings
from app.graphdb import get_client, is_memgraph_enabled
from app.state import AppState

log = logging.getLogger("icm.agent.nl2cypher")

# ---------- guardrails ----------

# Any of these tokens in the Cypher → reject. Memgraph/Neo4j write + admin verbs.
_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD|FOREACH|CALL\s+dbms|CALL\s+db\.)\b",
    re.IGNORECASE,
)

_MAX_ROWS = 200


# ---------- data classes ----------

@dataclass
class CypherPlan:
    intent: str
    cypher: str
    params: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    source: str = "llm"  # "llm" | "heuristic"


@dataclass
class PipelineResult:
    question: str
    plan: CypherPlan
    rows: list[dict[str, Any]]
    matches: list[dict[str, Any]]   # flattened / reranked incidents (+ entities)
    match_ids: list[str]
    anchor_ids: list[str]
    answer: str
    total: int


# ---------- step 1: plan ----------

_SYSTEM_PROMPT = """You translate natural-language questions about an Azure
incident-management knowledge graph into read-only Memgraph Cypher.

Graph schema (Memgraph / Neo4j compatible):

  (:Incident {incidentId, title, description, severity, status, service,
              region, team, owner, rootCauseCategory, mitigation,
              createdAt, impactedCustomers})
  (:Service {name})
  (:Region  {name})
  (:Team    {name})
  (:Owner   {name})
  (:RootCauseCategory {name})
  (:Mitigation {name})
  (:Component {name})
  (:Community {communityId, size, summary})

  Relationships (directions shown are canonical; undirected matches work):
    (i:Incident)-[:BELONGS_TO]->(s:Service)
    (i:Incident)-[:IN_REGION]->(r:Region)
    (i:Incident)-[:OWNED_BY]->(t:Team)
    (i:Incident)-[:ASSIGNED_TO]->(o:Owner)
    (i:Incident)-[:HAS_ROOT_CAUSE]->(rc:RootCauseCategory)
    (i:Incident)-[:MITIGATED_BY]->(m:Mitigation)
    (i:Incident)-[:AFFECTS]->(c:Component)
    (i:Incident)-[:LINKED_TO]->(j:Incident)
    (i:Incident)-[:IN_COMMUNITY]->(com:Community)

Rules:
  - OUTPUT MUST BE A SINGLE JSON OBJECT — no prose, no markdown fences.
  - JSON keys: {"intent": string, "cypher": string, "params": object, "explanation": string}.
  - `intent` is one of: lookup | list | aggregate | path | compare | trend.
  - `cypher` MUST be READ-ONLY. Allowed: MATCH, OPTIONAL MATCH, WHERE, WITH,
    RETURN, ORDER BY, LIMIT, UNWIND. FORBIDDEN: CREATE, MERGE, DELETE, SET,
    REMOVE, DROP, CALL dbms, CALL db.
  - ALWAYS include a LIMIT (≤ 100). Prefer LIMIT 50.
  - Prefer parameterized values via $params (e.g. $service) over inline strings.
  - When the question is about incidents, RETURN must include at minimum
    `i.incidentId AS incidentId, i.title AS title, i.severity AS severity,
     i.service AS service, i.region AS region, i.status AS status,
     i.rootCauseCategory AS rootCauseCategory, i.createdAt AS createdAt`.
  - Dates are ISO-8601 strings; compare lexicographically.
  - For "top N" / "most …", use ORDER BY + LIMIT.
  - For "related to X" or "similar to X", use LINKED_TO or shared
    RootCauseCategory / Service.
  - Do NOT invent labels, relationships, or properties outside the schema above.
  - If the question is ambiguous, choose the most useful list query.

Example:
  Q: "sev1 Front Door incidents in westus2 last month"
  {
    "intent": "list",
    "cypher": "MATCH (i:Incident) WHERE i.severity = $sev AND i.service = $service AND i.region = $region AND i.createdAt >= $start AND i.createdAt < $end RETURN i.incidentId AS incidentId, i.title AS title, i.severity AS severity, i.service AS service, i.region AS region, i.status AS status, i.rootCauseCategory AS rootCauseCategory, i.createdAt AS createdAt ORDER BY i.createdAt DESC LIMIT 50",
    "params": {"sev": 1, "service": "Front Door", "region": "westus2", "start": "2026-03-01", "end": "2026-04-01"},
    "explanation": "Filter Incidents by severity, service, region, and creation month."
  }
"""


def _vocab_hint(store: AppState) -> str:
    services = sorted({d.get("service") for _, d in store.graph.nodes(data=True) if d.get("type") == "Incident" and d.get("service")})
    regions = sorted({d.get("region") for _, d in store.graph.nodes(data=True) if d.get("type") == "Incident" and d.get("region")})
    causes = sorted({d.get("rootCauseCategory") for _, d in store.graph.nodes(data=True) if d.get("type") == "Incident" and d.get("rootCauseCategory")})
    teams = sorted({d.get("team") for _, d in store.graph.nodes(data=True) if d.get("type") == "Incident" and d.get("team")})
    return (
        "Known services: " + ", ".join(services[:30]) + "\n"
        "Known regions: " + ", ".join(regions[:30]) + "\n"
        "Known rootCauseCategories: " + ", ".join(causes[:30]) + "\n"
        "Known teams: " + ", ".join(teams[:30])
    )


def _llm_plan(question: str, store: AppState) -> CypherPlan | None:
    settings = get_settings()
    if not settings.has_azure_openai:
        return None
    try:
        from app.graphrag.llm import chat as llm_chat
    except Exception:  # pragma: no cover
        return None
    user = f"Vocabulary:\n{_vocab_hint(store)}\n\nQuestion: {question}\n\nReturn ONLY the JSON object."
    try:
        raw = llm_chat(_SYSTEM_PROMPT, user, max_tokens=700, temperature=0.1)
    except Exception as e:  # noqa: BLE001
        log.warning("LLM plan failed: %s", e)
        return None
    # Strip code fences if the model ignored instructions.
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Grab first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            log.warning("LLM plan returned non-JSON: %s", raw[:200])
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    cypher = str(obj.get("cypher") or "").strip()
    if not cypher or _FORBIDDEN.search(cypher):
        log.warning("LLM returned unsafe or empty Cypher; rejecting.")
        return None
    return CypherPlan(
        intent=str(obj.get("intent") or "list"),
        cypher=cypher,
        params=dict(obj.get("params") or {}),
        explanation=str(obj.get("explanation") or ""),
        source="llm",
    )


def _heuristic_plan(question: str, store: AppState) -> CypherPlan:
    gq = parse_query(question, store)
    where: list[str] = []
    params: dict[str, Any] = {}
    if gq.service:
        where.append("i.service = $service")
        params["service"] = gq.service
    if gq.region:
        where.append("i.region = $region")
        params["region"] = gq.region
    if gq.team:
        where.append("i.team = $team")
        params["team"] = gq.team
    if gq.rootCause:
        where.append("i.rootCauseCategory = $rootCause")
        params["rootCause"] = gq.rootCause
    if gq.status:
        where.append("i.status = $status")
        params["status"] = gq.status
    if gq.severity is not None:
        where.append("i.severity = $severity")
        params["severity"] = gq.severity
    if gq.startIso:
        where.append("i.createdAt >= $start")
        params["start"] = gq.startIso
    if gq.endIso:
        where.append("i.createdAt <= $end")
        params["end"] = gq.endIso
    if gq.incidentId:
        where.append("i.incidentId = $incidentId")
        params["incidentId"] = gq.incidentId.upper()
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    cypher = (
        "MATCH (i:Incident) "
        f"{clause} "
        "RETURN i.incidentId AS incidentId, i.title AS title, i.severity AS severity, "
        "i.service AS service, i.region AS region, i.status AS status, "
        "i.rootCauseCategory AS rootCauseCategory, i.createdAt AS createdAt "
        "ORDER BY i.severity ASC, i.createdAt DESC "
        "LIMIT 50"
    )
    return CypherPlan(
        intent=gq.intent or "list",
        cypher=cypher,
        params=params,
        explanation="Heuristic rule-based translation (LLM unavailable).",
        source="heuristic",
    )


def plan_cypher(question: str, store: AppState) -> CypherPlan:
    plan = _llm_plan(question, store)
    if plan is not None:
        return plan
    return _heuristic_plan(question, store)


# ---------- step 2: execute ----------

def _execute(plan: CypherPlan) -> list[dict[str, Any]]:
    if not is_memgraph_enabled():
        return []
    client = get_client()
    if _FORBIDDEN.search(plan.cypher):
        raise ValueError("Cypher failed guardrail check.")
    rows = client.read(plan.cypher, **plan.params)
    return rows[:_MAX_ROWS]


# ---------- step 3: embedding rerank ----------

def _cosine(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (Bn @ an).astype(np.float32)


def _rerank(question: str, rows: list[dict[str, Any]], store: AppState) -> list[dict[str, Any]]:
    """Blend DB order (implicit rank) with semantic similarity to the question
    using the pre-computed incident embeddings. Incidents only; non-incident
    rows pass through unchanged at the end."""
    if not rows:
        return rows
    settings = get_settings()
    if (
        not settings.has_azure_openai
        or store.incident_embeddings is None
        or not store.incident_ids
    ):
        return rows

    inc_rows = [r for r in rows if r.get("incidentId")]
    other_rows = [r for r in rows if not r.get("incidentId")]
    if not inc_rows:
        return rows

    try:
        from app.graphrag.llm import embed_texts
        qv = embed_texts([question])[0]
    except Exception as e:  # noqa: BLE001
        log.warning("Embedding query failed: %s", e)
        return rows

    id_to_idx = {iid: i for i, iid in enumerate(store.incident_ids)}
    sims: list[tuple[float, dict[str, Any]]] = []
    all_vecs = store.incident_embeddings
    # Per-row similarity (default 0 if not embedded — shouldn't happen often).
    for pos, row in enumerate(inc_rows):
        idx = id_to_idx.get(str(row["incidentId"]))
        if idx is None:
            sem = 0.0
        else:
            vec = all_vecs[idx : idx + 1]
            sem = float(_cosine(qv, vec)[0])
        # rank prior: prefer earlier DB ordering slightly (0..1, decayed)
        rank_prior = max(0.0, 1.0 - pos / max(1, len(inc_rows)))
        score = 0.7 * sem + 0.3 * rank_prior
        sims.append((score, {**row, "_score": round(score, 4), "_semantic": round(sem, 4)}))

    sims.sort(key=lambda t: t[0], reverse=True)
    reranked = [r for _, r in sims]
    return reranked + other_rows


# ---------- step 4: synthesize ----------

_ANSWER_SYSTEM = """You are an Azure IcM analyst. Write a concise 2-4 sentence
answer to the user's question grounded ONLY in the provided rows. Cite the most
relevant incident IDs inline (e.g. "INC-2026-0033"). If the rows are empty,
say so plainly. Do not invent data. Prefer concrete numbers (counts,
severities, services, regions).
"""


def _answer(question: str, rows: list[dict[str, Any]], plan: CypherPlan) -> str:
    settings = get_settings()
    if not settings.has_azure_openai or not rows:
        # Minimal deterministic fallback.
        if not rows:
            return f"No matching rows for: {question}"
        top = rows[: min(5, len(rows))]
        bullets = ", ".join(str(r.get("incidentId") or r.get("name") or r.get("communityId") or "?") for r in top)
        return f"Found {len(rows)} result(s). Top: {bullets}."
    try:
        from app.graphrag.llm import chat as llm_chat
    except Exception:  # pragma: no cover
        return f"Found {len(rows)} result(s)."
    rows_text = json.dumps(rows[:25], default=str, ensure_ascii=False, indent=2)
    user = (
        f"Question: {question}\n\n"
        f"Intent: {plan.intent}\nCypher used:\n{plan.cypher}\n\n"
        f"Rows ({len(rows)} total, first 25 shown):\n{rows_text}"
    )
    try:
        return llm_chat(_ANSWER_SYSTEM, user, max_tokens=280, temperature=0.2).strip()
    except Exception as e:  # noqa: BLE001
        log.warning("Answer synthesis failed: %s", e)
        return f"Found {len(rows)} result(s)."


# ---------- orchestrator ----------

def _flatten_match(row: dict[str, Any], store: AppState) -> dict[str, Any] | None:
    iid = row.get("incidentId")
    if iid and store.graph.has_node(str(iid)):
        d = store.graph.nodes[str(iid)]
        return {
            "id": str(iid),
            "type": "Incident",
            "label": str(iid),
            "severity": d.get("severity"),
            "service": d.get("service"),
            "region": d.get("region"),
            "status": d.get("status"),
            "title": d.get("title"),
            "score": row.get("_score"),
            "semantic": row.get("_semantic"),
        }
    # Non-incident entity row (e.g. from a path/compare query).
    for key_field, label in (("team", "Team"), ("service", "Service"), ("region", "Region"),
                              ("rootCauseCategory", "RootCauseCategory"), ("team_name", "Team")):
        val = row.get(key_field)
        if val:
            nid = f"{label}:{val}"
            if store.graph.has_node(nid):
                return {"id": nid, "type": label, "label": val}
    return None


def run_pipeline(question: str, store: AppState, *, limit: int = 50) -> PipelineResult:
    plan = plan_cypher(question, store)
    try:
        rows = _execute(plan)
    except Exception as e:  # noqa: BLE001
        log.warning("Cypher execution failed (%s); retrying with heuristic plan.", e)
        plan = _heuristic_plan(question, store)
        try:
            rows = _execute(plan)
        except Exception:  # noqa: BLE001
            rows = []

    rows = _rerank(question, rows, store)[:limit]

    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        m = _flatten_match(r, store)
        if m and m["id"] not in seen:
            matches.append(m)
            seen.add(m["id"])

    match_ids = [m["id"] for m in matches if m["type"] == "Incident"]

    # Anchor entities from the question's vocabulary (reuse heuristic parser).
    gq = parse_query(question, store)
    anchor_ids: list[str] = []
    for val, label in (
        (gq.service, "Service"),
        (gq.region, "Region"),
        (gq.team, "Team"),
        (gq.rootCause, "RootCauseCategory"),
    ):
        if val:
            nid = f"{label}:{val}"
            if store.graph.has_node(nid):
                anchor_ids.append(nid)

    answer = _answer(question, rows, plan)

    return PipelineResult(
        question=question,
        plan=plan,
        rows=rows,
        matches=matches,
        match_ids=match_ids,
        anchor_ids=anchor_ids,
        answer=answer,
        total=len(rows),
    )
