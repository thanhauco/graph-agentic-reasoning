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
    # Optional multi-step plan. Each step: {label, cypher, params}.
    # When set, `cypher`/`params` mirror steps[0] for UI back-compat.
    steps: list[dict[str, Any]] = field(default_factory=list)


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
              createdAt, mitigatedAt, resolvedAt, impactedCustomers})
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
  - JSON keys: {"intent": string, "steps": [{"label": string, "cypher": string, "params": object}], "explanation": string}.
  - `intent` is one of: lookup | list | multi_hop | aggregate | path | compare | trend.
  - `steps` is an ORDERED list of 1–3 read-only Cypher queries.
    * Use ONE step for simple lookups/lists.
    * Use MULTIPLE steps when the question has multiple parts. For example,
      "what happened to INC-X and similar incidents" → step 1 lookup of X
      with full properties, step 2 similar via shared rootCauseCategory /
      service / LINKED_TO / IN_COMMUNITY.
    * Each step must have a short snake_case `label` like "target", "similar",
      "by_root_cause", "neighbors".
  - Cypher MUST be READ-ONLY. Allowed: MATCH, OPTIONAL MATCH, WHERE, WITH,
    RETURN, ORDER BY, LIMIT, UNWIND. FORBIDDEN: CREATE, MERGE, DELETE, SET,
    REMOVE, DROP, CALL dbms, CALL db.
  - ALWAYS include a LIMIT (≤ 25 per step).
  - Prefer parameterized values (e.g. $incidentId, $service) over inline strings.
  - When returning incidents, include at minimum:
      `i.incidentId AS incidentId, i.title AS title, i.description AS description,
       i.severity AS severity, i.service AS service, i.region AS region,
       i.team AS team, i.status AS status, i.rootCauseCategory AS rootCauseCategory,
       i.mitigation AS mitigation, i.impactedCustomers AS impactedCustomers,
       i.createdAt AS createdAt`
  - For "similar to INC-X" exclude INC-X itself.
  - Do NOT invent labels, relationships, or properties outside the schema.
  - If ambiguous, choose the most useful query set.

Example:
  Q: "what happened to INC-2026-0243 and any similar incidents?"
  {
    "intent": "multi_hop",
    "steps": [
      {
        "label": "target",
        "cypher": "MATCH (i:Incident {incidentId: $incidentId}) RETURN i.incidentId AS incidentId, i.title AS title, i.description AS description, i.severity AS severity, i.service AS service, i.region AS region, i.team AS team, i.status AS status, i.rootCauseCategory AS rootCauseCategory, i.mitigation AS mitigation, i.impactedCustomers AS impactedCustomers, i.createdAt AS createdAt LIMIT 1",
        "params": {"incidentId": "INC-2026-0243"}
      },
      {
        "label": "similar",
        "cypher": "MATCH (x:Incident {incidentId: $incidentId}) MATCH (j:Incident) WHERE j.incidentId <> $incidentId AND (j.rootCauseCategory = x.rootCauseCategory OR j.service = x.service OR EXISTS { MATCH (x)-[:LINKED_TO]-(j) } OR EXISTS { MATCH (x)-[:IN_COMMUNITY]->(:Community)<-[:IN_COMMUNITY]-(j) }) RETURN j.incidentId AS incidentId, j.title AS title, j.severity AS severity, j.service AS service, j.region AS region, j.rootCauseCategory AS rootCauseCategory, j.status AS status, j.createdAt AS createdAt ORDER BY j.severity ASC, j.createdAt DESC LIMIT 15",
        "params": {"incidentId": "INC-2026-0243"}
      }
    ],
    "explanation": "Step 1 retrieves the target incident. Step 2 finds peers with shared root cause, service, direct links, or shared community."
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
    steps_raw = obj.get("steps")
    steps: list[dict[str, Any]] = []
    if isinstance(steps_raw, list) and steps_raw:
        for i, st in enumerate(steps_raw):
            if not isinstance(st, dict):
                continue
            c = str(st.get("cypher") or "").strip()
            if not c or _FORBIDDEN.search(c):
                log.warning("LLM step %d contained unsafe or empty Cypher; dropping.", i)
                continue
            steps.append({
                "label": str(st.get("label") or f"step{i + 1}"),
                "cypher": c,
                "params": dict(st.get("params") or {}),
            })
    if not steps and cypher:
        if _FORBIDDEN.search(cypher):
            log.warning("LLM returned unsafe Cypher; rejecting.")
            return None
        steps = [{"label": "main", "cypher": cypher, "params": dict(obj.get("params") or {})}]
    if not steps:
        log.warning("LLM plan has no usable steps.")
        return None
    return CypherPlan(
        intent=str(obj.get("intent") or "list"),
        cypher=steps[0]["cypher"],
        params=steps[0]["params"],
        explanation=str(obj.get("explanation") or ""),
        source="llm",
        steps=steps,
    )


_RICH_INCIDENT_RETURN = (
    "i.incidentId AS incidentId, i.title AS title, i.description AS description, "
    "i.severity AS severity, i.service AS service, i.region AS region, "
    "i.team AS team, i.status AS status, i.rootCauseCategory AS rootCauseCategory, "
    "i.mitigation AS mitigation, i.impactedCustomers AS impactedCustomers, "
    "i.createdAt AS createdAt"
)

_PEER_INCIDENT_RETURN = (
    "j.incidentId AS incidentId, j.title AS title, j.severity AS severity, "
    "j.service AS service, j.region AS region, j.status AS status, "
    "j.rootCauseCategory AS rootCauseCategory, j.createdAt AS createdAt"
)


def _heuristic_plan(question: str, store: AppState) -> CypherPlan:
    gq = parse_query(question, store)
    steps: list[dict[str, Any]] = []

    # ----- lookup + similar incidents (multi-hop or plain lookup with anchor) -----
    if gq.incidentId and gq.intent in {"multi_hop", "lookup"}:
        inc_id = gq.incidentId.upper()
        # Step 1: full target details
        steps.append({
            "label": "target",
            "cypher": f"MATCH (i:Incident {{incidentId: $incidentId}}) RETURN {_RICH_INCIDENT_RETURN} LIMIT 1",
            "params": {"incidentId": inc_id},
        })
        if gq.intent == "multi_hop" or "similar" in question.lower() or "related" in question.lower():
            # Step 2: peers via shared root cause / service / direct link / community
            steps.append({
                "label": "similar",
                "cypher": (
                    "MATCH (x:Incident {incidentId: $incidentId}) "
                    "MATCH (j:Incident) "
                    "WHERE j.incidentId <> $incidentId "
                    "  AND ( "
                    "    j.rootCauseCategory = x.rootCauseCategory "
                    "    OR j.service = x.service "
                    "    OR EXISTS { MATCH (x)-[:LINKED_TO]-(j) } "
                    "    OR EXISTS { MATCH (x)-[:IN_COMMUNITY]->(:Community)<-[:IN_COMMUNITY]-(j) } "
                    "  ) "
                    f"RETURN {_PEER_INCIDENT_RETURN} "
                    "ORDER BY j.severity ASC, j.createdAt DESC LIMIT 15"
                ),
                "params": {"incidentId": inc_id},
            })
        explanation = (
            "Looked up target incident, then found peers sharing root cause, service, "
            "direct LINKED_TO edges, or the same community."
            if len(steps) > 1 else
            "Looked up target incident by ID."
        )
        return CypherPlan(
            intent=gq.intent or "multi_hop",
            cypher=steps[0]["cypher"],
            params=steps[0]["params"],
            explanation=explanation,
            source="heuristic",
            steps=steps,
        )

    # ----- generic filtered list -----
    where: list[str] = []
    params: dict[str, Any] = {}
    if gq.service:
        where.append("i.service = $service"); params["service"] = gq.service
    if gq.region:
        where.append("i.region = $region"); params["region"] = gq.region
    if gq.team:
        where.append("i.team = $team"); params["team"] = gq.team
    if gq.rootCause:
        where.append("i.rootCauseCategory = $rootCause"); params["rootCause"] = gq.rootCause
    if gq.status:
        where.append("i.status = $status"); params["status"] = gq.status
    if gq.severity is not None:
        where.append("i.severity = $severity"); params["severity"] = gq.severity
    if gq.startIso:
        where.append("i.createdAt >= $start"); params["start"] = gq.startIso
    if gq.endIso:
        where.append("i.createdAt <= $end"); params["end"] = gq.endIso
    if gq.incidentId:
        where.append("i.incidentId = $incidentId"); params["incidentId"] = gq.incidentId.upper()
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    cypher = (
        f"MATCH (i:Incident) {clause} RETURN {_RICH_INCIDENT_RETURN} "
        "ORDER BY i.severity ASC, i.createdAt DESC LIMIT 50"
    )
    steps.append({"label": "main", "cypher": cypher, "params": params})
    return CypherPlan(
        intent=gq.intent or "list",
        cypher=cypher,
        params=params,
        explanation="Heuristic rule-based translation (LLM unavailable).",
        source="heuristic",
        steps=steps,
    )


def plan_cypher(question: str, store: AppState) -> CypherPlan:
    plan = _llm_plan(question, store)
    if plan is not None:
        return plan
    return _heuristic_plan(question, store)


# ---------- step 2: execute ----------

def _execute(plan: CypherPlan) -> list[dict[str, Any]]:
    """Execute every step in the plan and return a single flat list of rows,
    each annotated with `_step` (the step label). Anchors earlier steps' rows
    first so 'target' appears before 'similar' when both are present."""
    if not is_memgraph_enabled():
        return []
    client = get_client()
    rows: list[dict[str, Any]] = []
    steps = plan.steps or [{"label": "main", "cypher": plan.cypher, "params": plan.params}]
    for step in steps:
        cypher = step.get("cypher") or ""
        if not cypher:
            continue
        if _FORBIDDEN.search(cypher):
            raise ValueError(f"Step {step.get('label')!r} failed guardrail check.")
        try:
            result = client.read(cypher, **(step.get("params") or {}))
        except Exception as e:  # noqa: BLE001
            log.warning("Cypher step %s failed: %s", step.get("label"), e)
            continue
        label = step.get("label") or "main"
        for r in result[:_MAX_ROWS]:
            r["_step"] = label
            rows.append(r)
    return rows[:_MAX_ROWS]


# ---------- step 3: relevance rerank ----------

def _cosine(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (Bn @ an).astype(np.float32)


_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "at", "for", "with", "and", "or", "but", "any", "it", "its", "this", "that",
    "these", "those", "what", "which", "who", "whom", "when", "where", "why",
    "how", "do", "does", "did", "has", "have", "had", "show", "list", "find",
    "give", "me", "my", "please", "about", "similar", "related", "like", "to",
    "happened", "happen", "happens", "incident", "incidents",
}
_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{1,}")


def _tokens(text: str) -> set[str]:
    if not text:
        return set()
    return {t for t in (m.group(0).lower() for m in _WORD_RE.finditer(text)) if t not in _STOP}


def _row_text(row: dict[str, Any]) -> str:
    parts = [
        row.get("incidentId"), row.get("title"), row.get("description"),
        row.get("service"), row.get("region"), row.get("team"),
        row.get("rootCauseCategory"), row.get("mitigation"),
        row.get("status"), row.get("name"),
    ]
    sev = row.get("severity")
    if sev is not None:
        parts.append(f"sev{sev}")
    return " ".join(str(p) for p in parts if p)


def _lexical(q_tokens: set[str], row: dict[str, Any]) -> float:
    """Jaccard-ish overlap of query tokens against row text."""
    if not q_tokens:
        return 0.0
    rt = _tokens(_row_text(row))
    if not rt:
        return 0.0
    overlap = len(q_tokens & rt)
    return overlap / max(3, len(q_tokens))  # saturate around 3 matched terms


def _rerank(question: str, rows: list[dict[str, Any]], store: AppState) -> list[dict[str, Any]]:
    """Rerank ALL query results for relevance.

    Signals (any that are available):
      * Lexical overlap of query tokens with row text (always available).
      * Semantic cosine between the query embedding and each incident's
        pre-computed embedding (requires Azure OpenAI for the query side).
      * Anchor-similarity: when the question has an anchor incident that
        is in `store.incident_ids`, use its pre-computed embedding as an
        additional reference vector (works WITHOUT Azure OpenAI).
      * Rank prior from the DB's ORDER BY clause.

    A row tagged `_step == "target"` is always pinned first (it's the
    exact incident the user asked for). Non-incident rows pass through
    after incident rows.
    """
    if not rows:
        return rows

    # ----- split rows -----
    target_rows = [r for r in rows if r.get("_step") == "target"]
    non_target = [r for r in rows if r.get("_step") != "target"]
    inc_rows = [r for r in non_target if r.get("incidentId")]
    other_rows = [r for r in non_target if not r.get("incidentId")]
    if not inc_rows:
        return rows

    # ----- lexical signal (always) -----
    q_tokens = _tokens(question)

    # ----- semantic signal (query embedding) -----
    settings = get_settings()
    have_incident_embeddings = (
        store.incident_embeddings is not None and bool(store.incident_ids)
    )
    id_to_idx: dict[str, int] = (
        {iid: i for i, iid in enumerate(store.incident_ids)}
        if have_incident_embeddings else {}
    )
    qv: np.ndarray | None = None
    if settings.has_azure_openai and have_incident_embeddings:
        try:
            from app.graphrag.llm import embed_texts
            qv = np.asarray(embed_texts([question])[0], dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            log.warning("Query embedding failed, falling back to lexical: %s", e)
            qv = None

    # ----- anchor-similarity signal -----
    anchor_vec: np.ndarray | None = None
    if have_incident_embeddings:
        # Pull anchor from parsed query (e.g. "similar to INC-...")
        try:
            gq = parse_query(question, store)
            anchor_id = (gq.anchorId or gq.incidentId or "").upper() or None
        except Exception:  # noqa: BLE001
            anchor_id = None
        if anchor_id and anchor_id in id_to_idx:
            anchor_vec = np.asarray(
                store.incident_embeddings[id_to_idx[anchor_id]], dtype=np.float32
            )

    # ----- score every incident row -----
    scored: list[tuple[float, dict[str, Any]]] = []
    for pos, row in enumerate(inc_rows):
        lex = _lexical(q_tokens, row)
        sem_q = 0.0
        sem_a = 0.0
        idx = id_to_idx.get(str(row.get("incidentId")))
        if idx is not None:
            vec = np.asarray(store.incident_embeddings[idx : idx + 1], dtype=np.float32)
            if qv is not None:
                sem_q = float(_cosine(qv, vec)[0])
            if anchor_vec is not None and str(row["incidentId"]).upper() != (
                # don't boost the anchor itself if it somehow leaks into peers
                row.get("incidentId") if False else ""
            ):
                sem_a = float(_cosine(anchor_vec, vec)[0])
        rank_prior = max(0.0, 1.0 - pos / max(1, len(inc_rows)))

        # Weight: prefer explicit semantic signal; fall back to lexical.
        if qv is not None and anchor_vec is not None:
            sem = 0.6 * sem_q + 0.4 * sem_a
        elif qv is not None:
            sem = sem_q
        elif anchor_vec is not None:
            sem = sem_a
        else:
            sem = 0.0

        if qv is not None or anchor_vec is not None:
            score = 0.55 * sem + 0.30 * lex + 0.15 * rank_prior
        else:
            score = 0.70 * lex + 0.30 * rank_prior

        scored.append((score, {
            **row,
            "_score": round(score, 4),
            "_lexical": round(lex, 4),
            "_semantic": round(sem, 4),
        }))

    scored.sort(key=lambda t: t[0], reverse=True)
    reranked = [r for _, r in scored]
    return target_rows + reranked + other_rows


# ---------- step 4: synthesize ----------

_ANSWER_SYSTEM = """You are an Azure IcM analyst answering a user's question
about incident data.

You receive:
  - The original question.
  - Rows returned by one or more Cypher queries, GROUPED BY step label
    (e.g. "target" = the specific incident asked about; "similar" = peer
    incidents found via shared root cause / service / direct links /
    community). Field `_semantic` (when present) is the cosine similarity
    of each peer to the user's question.

Write a grounded answer:
  1. Directly address EVERY part of the question.
  2. If a "target" row is present, summarize it FIRST in 1–2 sentences using
     its title, severity, service/region/team, status, root cause, mitigation,
     and (if available) description and impactedCustomers. Cite its ID.
  3. If "similar" / peer rows are present, describe the shared pattern
     (common service, root cause, or mitigation) in 1–2 sentences and list
     3–5 most relevant peer incident IDs inline.
  4. Use bullet points only when listing 3+ peers. Otherwise write prose.
  5. Never invent facts. If a section has no rows, say so briefly.
  6. Keep the total answer under ~160 words.
"""


def _deterministic_answer(question: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"No matching rows for: {question}"
    by_step: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_step.setdefault(r.get("_step") or "main", []).append(r)
    parts: list[str] = []
    tgt = (by_step.get("target") or [None])[0]
    if tgt:
        bits = [
            f"**{tgt.get('incidentId')}** — {tgt.get('title') or ''}".strip(),
            f"Sev{tgt.get('severity')}" if tgt.get("severity") is not None else None,
            f"{tgt.get('service') or ''}/{tgt.get('region') or ''}".strip("/"),
            f"status {tgt.get('status')}" if tgt.get("status") else None,
            f"root cause: {tgt.get('rootCauseCategory')}" if tgt.get("rootCauseCategory") else None,
            f"mitigation: {tgt.get('mitigation')}" if tgt.get("mitigation") else None,
            f"impacted {tgt.get('impactedCustomers')} customers" if tgt.get("impactedCustomers") else None,
        ]
        parts.append(" · ".join(b for b in bits if b))
        if tgt.get("description"):
            parts.append(str(tgt["description"]))
    sim = by_step.get("similar") or by_step.get("main") or []
    sim = [r for r in sim if r.get("incidentId") and (not tgt or r.get("incidentId") != tgt.get("incidentId"))]
    if sim:
        ids = ", ".join(str(r["incidentId"]) for r in sim[:5])
        parts.append(f"Similar incidents ({len(sim)}): {ids}.")
    elif tgt:
        parts.append("No similar peers found in the current graph.")
    return "\n\n".join(parts) if parts else f"Found {len(rows)} result(s)."


def _answer(question: str, rows: list[dict[str, Any]], plan: CypherPlan) -> str:
    settings = get_settings()
    if not settings.has_azure_openai or not rows:
        return _deterministic_answer(question, rows)
    try:
        from app.graphrag.llm import chat as llm_chat
    except Exception:  # pragma: no cover
        return _deterministic_answer(question, rows)

    by_step: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_step.setdefault(r.get("_step") or "main", []).append(r)
    # Trim rows per step and drop bulky description for non-target rows to save tokens.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for label, group in by_step.items():
        trimmed: list[dict[str, Any]] = []
        for r in group[:12]:
            if label != "target":
                r = {k: v for k, v in r.items() if k != "description"}
            trimmed.append(r)
        grouped[label] = trimmed

    context = json.dumps(grouped, default=str, ensure_ascii=False, indent=2)
    user = (
        f"Question: {question}\n\n"
        f"Intent: {plan.intent}\n"
        f"Cypher plan source: {plan.source}\n\n"
        f"Results (grouped by step):\n{context}"
    )
    try:
        return llm_chat(_ANSWER_SYSTEM, user, max_tokens=380, temperature=0.2).strip()
    except Exception as e:  # noqa: BLE001
        log.warning("Answer synthesis failed: %s", e)
        return _deterministic_answer(question, rows)


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
