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
  - `steps` is an ORDERED list of 1–4 read-only Cypher queries.
    * Use ONE step for simple lookups/lists.
    * Use MULTIPLE steps when the question has multiple parts. For example,
      "what happened to INC-X and similar incidents" → step 1 lookup of X
      with full properties, step 2 similar via shared rootCauseCategory /
      service / LINKED_TO / IN_COMMUNITY.
    * For questions asking "how many nodes connect" / "how many neighbors"
      / "connected to it" add steps labeled "neighbors_count" (single row
      with `count(DISTINCT n) AS neighborCount`) and "neighbors_by_type"
      (breakdown `nodeType, relation, cnt`).
    * Each step must have a short snake_case `label`. Known labels:
      "target", "similar", "neighbors_count", "neighbors_by_type", "main".
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


def _llm_plan(
    question: str,
    store: AppState,
    *,
    history: list[dict[str, Any]] | None = None,
) -> CypherPlan | None:
    settings = get_settings()
    if not settings.has_azure_openai:
        return None
    try:
        from app.graphrag.llm import chat as llm_chat
    except Exception:  # pragma: no cover
        return None
    hist_block = ""
    if history:
        turns: list[str] = []
        for t in history[-4:]:
            q = str(t.get("question") or "").strip()
            a = str(t.get("answer") or "").strip()
            mids = ", ".join(list(t.get("matchIds") or [])[:5])
            if q:
                turns.append(
                    f"- Q: {q}\n  A: {a[:240]}" + (f"\n  matches: {mids}" if mids else "")
                )
        if turns:
            hist_block = (
                "Recent conversation (oldest → newest). Use it to resolve "
                "pronouns/follow-ups like 'those', 'these', 'the same', 'more like that':\n"
                + "\n".join(turns)
                + "\n\n"
            )
    user = (
        f"Vocabulary:\n{_vocab_hint(store)}\n\n"
        f"{hist_block}"
        f"Question: {question}\n\nReturn ONLY the JSON object."
    )
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
    ql = question.lower()
    wants_similar = any(k in ql for k in ("similar", "related", "like this", "like it"))
    wants_cause = any(
        k in ql for k in (
            "root cause", "what cause", "what caused", "cause this", "cause it",
            "why this", "why did", "reason", "reason why",
        )
    )
    wants_explain = any(
        k in ql for k in (
            "explain", "long text", "long-form", "long form", "detailed",
            "detail", "in depth", "deep dive", "full context", "elaborate",
        )
    )
    wants_neighbors = any(
        k in ql for k in (
            "how many", "neighbor", "neighbour", "connect", "connected",
            "connection", "linked", "related nodes", "relationships",
            "edges", "around it", "around this",
        )
    )

    # ----- relationship / path between TWO incidents -----
    inc_ids_found = [m.upper() for m in _INC_ID_RE.findall(question)]
    # Deduplicate preserving order.
    seen: set[str] = set()
    inc_ids_unique: list[str] = []
    for iid in inc_ids_found:
        if iid not in seen:
            seen.add(iid)
            inc_ids_unique.append(iid)
    if len(inc_ids_unique) >= 2:
        src, dst = inc_ids_unique[0], inc_ids_unique[1]
        # Step 1: direct LINKED_TO edge (and any direct relationship).
        steps.append({
            "label": "direct_link",
            "cypher": (
                "MATCH (a:Incident {incidentId: $src}), (b:Incident {incidentId: $dst}) "
                "OPTIONAL MATCH p = (a)-[r]-(b) "
                "RETURN a.incidentId AS src, b.incidentId AS dst, "
                "       collect(DISTINCT type(r)) AS directRelations, "
                "       count(r) AS directEdgeCount"
            ),
            "params": {"src": src, "dst": dst},
        })
        # Step 2: shortest path (up to 6 hops) through any nodes.
        steps.append({
            "label": "shortest_path",
            "cypher": (
                "MATCH (a:Incident {incidentId: $src}), (b:Incident {incidentId: $dst}) "
                "MATCH p = (a)-[*BFS..6]-(b) "
                "WITH p, nodes(p) AS ns, relationships(p) AS rs "
                "RETURN size(rs) AS pathLength, "
                "       [n IN ns | coalesce(n.incidentId, n.name, n.communityId)] AS nodeLabels, "
                "       [n IN ns | labels(n)[0]] AS nodeTypes, "
                "       [r IN rs | type(r)] AS relations "
                "ORDER BY pathLength ASC LIMIT 1"
            ),
            "params": {"src": src, "dst": dst},
        })
        # Step 3: shared neighbors / context (service/region/team/rootCause/community).
        steps.append({
            "label": "shared_context",
            "cypher": (
                "MATCH (a:Incident {incidentId: $src}), (b:Incident {incidentId: $dst}) "
                "OPTIONAL MATCH (a)-[:BELONGS_TO]->(sa:Service), (b)-[:BELONGS_TO]->(sb:Service) "
                "OPTIONAL MATCH (a)-[:IN_REGION]->(ra:Region), (b)-[:IN_REGION]->(rb:Region) "
                "OPTIONAL MATCH (a)-[:OWNED_BY]->(ta:Team), (b)-[:OWNED_BY]->(tb:Team) "
                "OPTIONAL MATCH (a)-[:HAS_ROOT_CAUSE]->(rca:RootCauseCategory), "
                "               (b)-[:HAS_ROOT_CAUSE]->(rcb:RootCauseCategory) "
                "OPTIONAL MATCH (a)-[:IN_COMMUNITY]->(ca:Community)<-[:IN_COMMUNITY]-(b) "
                "RETURN a.service AS srcService, b.service AS dstService, "
                "       a.region AS srcRegion, b.region AS dstRegion, "
                "       a.team AS srcTeam, b.team AS dstTeam, "
                "       a.rootCauseCategory AS srcRootCause, b.rootCauseCategory AS dstRootCause, "
                "       collect(DISTINCT ca.communityId) AS sharedCommunities, "
                "       (a.service = b.service) AS sameService, "
                "       (a.region = b.region) AS sameRegion, "
                "       (a.team = b.team) AS sameTeam, "
                "       (a.rootCauseCategory = b.rootCauseCategory) AS sameRootCause"
            ),
            "params": {"src": src, "dst": dst},
        })
        # Step 4: detailed rows for both incidents so the answerer can cite them.
        steps.append({
            "label": "endpoints",
            "cypher": (
                "MATCH (i:Incident) WHERE i.incidentId IN [$src, $dst] "
                f"RETURN {_RICH_INCIDENT_RETURN} "
                "ORDER BY i.incidentId"
            ),
            "params": {"src": src, "dst": dst},
        })
        return CypherPlan(
            intent="path",
            cypher=steps[0]["cypher"],
            params=steps[0]["params"],
            explanation=(
                f"Checked for a direct edge between {src} and {dst}, then "
                "computed the shortest path (≤6 hops) and surfaced any shared "
                "service/region/team/root-cause/community context."
            ),
            source="heuristic",
            steps=steps,
        )

    # ----- lookup + optional similar + optional neighbor count -----
    if gq.incidentId and gq.intent in {"multi_hop", "lookup"}:
        inc_id = gq.incidentId.upper()
        steps.append({
            "label": "target",
            "cypher": f"MATCH (i:Incident {{incidentId: $incidentId}}) RETURN {_RICH_INCIDENT_RETURN} LIMIT 1",
            "params": {"incidentId": inc_id},
        })
        if gq.intent == "multi_hop" or wants_similar or wants_explain or wants_cause:
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
        if wants_neighbors or wants_explain or wants_cause:
            # Total count across all relationship types.
            steps.append({
                "label": "neighbors_count",
                "cypher": (
                    "MATCH (i:Incident {incidentId: $incidentId})--(n) "
                    "RETURN count(DISTINCT n) AS neighborCount"
                ),
                "params": {"incidentId": inc_id},
            })
            # Breakdown by neighbor label and relationship type.
            steps.append({
                "label": "neighbors_by_type",
                "cypher": (
                    "MATCH (i:Incident {incidentId: $incidentId})-[r]-(n) "
                    "WITH labels(n)[0] AS nodeType, type(r) AS relation, "
                    "     count(DISTINCT n) AS cnt "
                    "RETURN nodeType, relation, cnt "
                    "ORDER BY cnt DESC LIMIT 20"
                ),
                "params": {"incidentId": inc_id},
            })
        parts = []
        if any(s["label"] == "target" for s in steps):
            parts.append("target incident details")
        if any(s["label"] == "similar" for s in steps):
            parts.append("peers sharing root cause/service/community or linked")
        if any(s["label"] == "neighbors_count" for s in steps):
            parts.append("neighbor count and breakdown by type")
        explanation = "Looked up " + ", ".join(parts) + "."
        return CypherPlan(
            intent=gq.intent or ("multi_hop" if len(steps) > 1 else "lookup"),
            cypher=steps[0]["cypher"],
            params=steps[0]["params"],
            explanation=explanation,
            source="heuristic",
            steps=steps,
        )

    # ----- entity anchor + neighbor count (e.g. Service/Team/Region node) -----
    # The question came in with no incident ID, but run_pipeline may have
    # appended "(about service 'Storage')" context when an entity was selected.
    # Heuristic parser will populate gq.service/team/region from that hint.
    if wants_neighbors and (gq.service or gq.team or gq.region or gq.rootCause):
        if gq.service:
            anchor_label, anchor_name = "Service", gq.service
        elif gq.team:
            anchor_label, anchor_name = "Team", gq.team
        elif gq.region:
            anchor_label, anchor_name = "Region", gq.region
        else:
            anchor_label, anchor_name = "RootCauseCategory", gq.rootCause  # type: ignore[assignment]
        steps.append({
            "label": "neighbors_count",
            "cypher": (
                f"MATCH (a:{anchor_label} {{name: $name}})--(n) "
                "RETURN count(DISTINCT n) AS neighborCount"
            ),
            "params": {"name": anchor_name},
        })
        steps.append({
            "label": "neighbors_by_type",
            "cypher": (
                f"MATCH (a:{anchor_label} {{name: $name}})-[r]-(n) "
                "WITH labels(n)[0] AS nodeType, type(r) AS relation, "
                "     count(DISTINCT n) AS cnt "
                "RETURN nodeType, relation, cnt "
                "ORDER BY cnt DESC LIMIT 20"
            ),
            "params": {"name": anchor_name},
        })
        return CypherPlan(
            intent="neighbors",
            cypher=steps[0]["cypher"],
            params=steps[0]["params"],
            explanation=f"Counted nodes connected to {anchor_label} '{anchor_name}'.",
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


def plan_cypher(
    question: str,
    store: AppState,
    *,
    history: list[dict[str, Any]] | None = None,
) -> CypherPlan:
    plan = _llm_plan(question, store, history=history)
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
  - Rows returned by one or more Cypher queries, GROUPED BY step label.
    Known step labels:
      * "target"            — the specific incident asked about
      * "similar"           — peers found via shared root cause / service /
                              LINKED_TO / community
      * "neighbors_count"   — one row with { neighborCount: <int> } counting
                              distinct nodes directly connected to the target
      * "neighbors_by_type" — breakdown rows { nodeType, relation, cnt }
      * "main"              — a generic filtered list
    Field `_semantic` (when present) is the cosine similarity of each peer
    to the user's question.

Write a grounded answer:
  1. Directly address EVERY part of the question — if the user asked
     multiple things (e.g. "what happened to X and how many nodes connect
     to it"), answer EACH part in its own paragraph or sentence.
  2. If a "target" row is present, summarize it FIRST in 1–2 sentences using
     title, severity, service/region/team, status, root cause, mitigation,
     and (if available) description and impactedCustomers. Cite its ID.
  3. If "neighbors_count" is present, explicitly state the total neighbor
     count. If "neighbors_by_type" is present, list the top 3–5 breakdowns
     as "<cnt> <nodeType> via <relation>".
  4. If "similar" / peer rows are present, describe the shared pattern in
     1–2 sentences and list 3–5 most relevant peer incident IDs inline.
  5. Use bullet points only when listing 3+ peers or 3+ breakdown rows.
     Otherwise write prose.
  6. Never invent facts. If a section has no rows, say so briefly.
  7. Keep the total answer under ~180 words.
    8. If the user asks causal intent (e.g., "what caused this", "why this"),
         provide: (a) primary cause from target data, (b) supporting evidence from
         similar/neighbor rows, and (c) a short confidence qualifier.
"""


def _deterministic_answer(question: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"No matching rows for: {question}"
    ql = question.lower()
    wants_long_text = any(
        k in ql for k in (
            "explain", "long text", "long-form", "long form", "detailed",
            "detail", "in depth", "deep dive", "full context", "elaborate",
        )
    )
    wants_cause = any(
        k in ql for k in (
            "root cause", "what cause", "what caused", "cause this", "cause it",
            "why this", "why did", "reason", "reason why",
        )
    )
    by_step: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_step.setdefault(r.get("_step") or "main", []).append(r)
    parts: list[str] = []

    # --- path / relationship between two incidents ---
    if "direct_link" in by_step or "shortest_path" in by_step or "shared_context" in by_step:
        direct = (by_step.get("direct_link") or [{}])[0]
        src = direct.get("src")
        dst = direct.get("dst")
        if not src or not dst:
            # Fallback: pull endpoints.
            ep = by_step.get("endpoints") or []
            if len(ep) >= 2:
                src, dst = ep[0].get("incidentId"), ep[1].get("incidentId")

        direct_rels = direct.get("directRelations") or []
        direct_rels = [r for r in direct_rels if r]
        if direct_rels:
            parts.append(f"**{src} ↔ {dst}** are directly connected via `{', '.join(direct_rels)}`.")
        else:
            parts.append(f"**{src} ↔ {dst}**: no direct edge between them.")

        sp = (by_step.get("shortest_path") or [{}])[0]
        plen = sp.get("pathLength")
        labels = sp.get("nodeLabels") or []
        types = sp.get("nodeTypes") or []
        rels = sp.get("relations") or []
        if plen is not None and labels:
            hop_parts: list[str] = []
            for i, lab in enumerate(labels):
                typ = types[i] if i < len(types) else ""
                hop_parts.append(f"{lab}[{typ}]")
                if i < len(rels):
                    hop_parts.append(f"-({rels[i]})-")
            parts.append(f"Shortest path (length {plen}): " + " ".join(hop_parts))
        elif direct_rels:
            pass  # already covered
        else:
            parts.append("No path (≤ 6 hops) found between them in the graph.")

        ctx = (by_step.get("shared_context") or [{}])[0]
        shared: list[str] = []
        if ctx.get("sameService") and ctx.get("srcService"):
            shared.append(f"same service **{ctx['srcService']}**")
        elif ctx.get("srcService") or ctx.get("dstService"):
            shared.append(f"services {ctx.get('srcService')} vs {ctx.get('dstService')}")
        if ctx.get("sameRegion") and ctx.get("srcRegion"):
            shared.append(f"same region **{ctx['srcRegion']}**")
        elif ctx.get("srcRegion") or ctx.get("dstRegion"):
            shared.append(f"regions {ctx.get('srcRegion')} vs {ctx.get('dstRegion')}")
        if ctx.get("sameTeam") and ctx.get("srcTeam"):
            shared.append(f"same team **{ctx['srcTeam']}**")
        if ctx.get("sameRootCause") and ctx.get("srcRootCause"):
            shared.append(f"same root cause **{ctx['srcRootCause']}**")
        elif ctx.get("srcRootCause") or ctx.get("dstRootCause"):
            shared.append(f"root causes {ctx.get('srcRootCause')} vs {ctx.get('dstRootCause')}")
        sc = [c for c in (ctx.get("sharedCommunities") or []) if c]
        if sc:
            shared.append(f"shared community {', '.join(sc)}")
        if shared:
            parts.append("Shared context: " + "; ".join(shared) + ".")

        endpoints = by_step.get("endpoints") or []
        for ep in endpoints[:2]:
            bits = [
                f"**{ep.get('incidentId')}** — {ep.get('title') or ''}".strip(),
                f"Sev{ep.get('severity')}" if ep.get("severity") is not None else None,
                f"{ep.get('service') or ''}/{ep.get('region') or ''}".strip("/"),
                f"root cause: {ep.get('rootCauseCategory')}" if ep.get("rootCauseCategory") else None,
                f"status {ep.get('status')}" if ep.get("status") else None,
            ]
            parts.append(" · ".join(b for b in bits if b))
        return "\n\n".join(parts)

    tgt = (by_step.get("target") or [None])[0]
    if tgt:
        sim_rows_for_target = by_step.get("similar") or []
        sim_rows_for_target = [
            r for r in sim_rows_for_target
            if r.get("incidentId") and r.get("incidentId") != tgt.get("incidentId")
        ]
        nc_rows_for_target = by_step.get("neighbors_count") or []
        nb_rows_for_target = by_step.get("neighbors_by_type") or []

        if wants_cause:
            incident_id = str(tgt.get("incidentId") or "this incident")
            sev = tgt.get("severity")
            service = str(tgt.get("service") or "unknown service")
            region = str(tgt.get("region") or "unknown region")
            team = str(tgt.get("team") or "unknown team")
            status = str(tgt.get("status") or "unknown")
            root_cause = str(tgt.get("rootCauseCategory") or "unknown")
            mitigation = str(tgt.get("mitigation") or "not recorded")
            impacted = tgt.get("impactedCustomers")

            cause_parts: list[str] = []
            cause_parts.append(
                f"Likely primary cause for **{incident_id}** is **{root_cause}**. "
                f"This is a Sev{sev} incident in {service}/{region} (team: {team}, status: {status})."
            )
            cause_parts.append(
                f"Recorded mitigation was: {mitigation}."
                + (f" Reported impact: {impacted} customers." if impacted is not None else "")
            )

            if sim_rows_for_target:
                rc_count: dict[str, int] = {}
                svc_count: dict[str, int] = {}
                for r in sim_rows_for_target:
                    rc = str(r.get("rootCauseCategory") or "Unknown")
                    rc_count[rc] = rc_count.get(rc, 0) + 1
                    svc = str(r.get("service") or "Unknown")
                    svc_count[svc] = svc_count.get(svc, 0) + 1
                top_rc = ", ".join(
                    f"{k} ({v})"
                    for k, v in sorted(rc_count.items(), key=lambda kv: kv[1], reverse=True)[:3]
                )
                top_svc = ", ".join(
                    f"{k} ({v})"
                    for k, v in sorted(svc_count.items(), key=lambda kv: kv[1], reverse=True)[:3]
                )
                ids = ", ".join(str(r["incidentId"]) for r in sim_rows_for_target[:5])
                cause_parts.append(
                    f"Inference from peer incidents ({ids}): dominant peer causes are {top_rc}; "
                    f"peer services are {top_svc}."
                )

            if nc_rows_for_target and nc_rows_for_target[0].get("neighborCount") is not None:
                line = f"Graph context: {nc_rows_for_target[0].get('neighborCount')} directly connected nodes"
                top_nb = [
                    f"{r.get('cnt')} {r.get('nodeType')} via {r.get('relation')}"
                    for r in nb_rows_for_target[:4]
                    if r.get("cnt") and r.get("nodeType")
                ]
                if top_nb:
                    line += "; strongest links: " + ", ".join(top_nb)
                cause_parts.append(line + ".")

            cause_parts.append(
                "Inference confidence: medium, based on recorded root-cause label plus similarity and graph-neighborhood signals; "
                "validate against postmortem/change timeline for final attribution."
            )
            return "\n\n".join(cause_parts)

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

        if wants_long_text:
            incident_id = str(tgt.get("incidentId") or "this incident")
            service = str(tgt.get("service") or "unknown service")
            region = str(tgt.get("region") or "unknown region")
            team = str(tgt.get("team") or "unknown team")
            status = str(tgt.get("status") or "unknown")
            root_cause = str(tgt.get("rootCauseCategory") or "unknown")
            mitigation = str(tgt.get("mitigation") or "not recorded")
            impacted = tgt.get("impactedCustomers")
            created_at = str(tgt.get("createdAt") or "")

            long_sections: list[str] = []
            long_sections.append(
                f"{incident_id} is a Sev{tgt.get('severity')} incident in {service} / {region}, "
                f"owned by {team}. It is currently marked {status}."
            )
            long_sections.append(
                f"The dominant failure signal is {root_cause}. "
                f"Mitigation recorded: {mitigation}."
            )
            if impacted is not None:
                long_sections.append(
                    f"Customer impact was recorded as {impacted} potentially affected customers."
                )
            if created_at:
                long_sections.append(f"Timeline anchor: createdAt = {created_at}.")

            nc_rows = nc_rows_for_target
            nb_rows = nb_rows_for_target
            if nc_rows and nc_rows[0].get("neighborCount") is not None:
                nb_text = f"Graph context shows {nc_rows[0].get('neighborCount')} directly connected nodes"
                if nb_rows:
                    top = [
                        f"{r.get('cnt')} {r.get('nodeType')} via {r.get('relation')}"
                        for r in nb_rows[:5]
                        if r.get("cnt") and r.get("nodeType")
                    ]
                    if top:
                        nb_text += "; strongest links are " + ", ".join(top)
                long_sections.append(nb_text + ".")

            sim_rows = sim_rows_for_target
            if sim_rows:
                ids = ", ".join(str(r["incidentId"]) for r in sim_rows[:6])
                rc_count: dict[str, int] = {}
                svc_count: dict[str, int] = {}
                for r in sim_rows:
                    rc = str(r.get("rootCauseCategory") or "Unknown")
                    rc_count[rc] = rc_count.get(rc, 0) + 1
                    svc = str(r.get("service") or "Unknown")
                    svc_count[svc] = svc_count.get(svc, 0) + 1
                top_rc = sorted(rc_count.items(), key=lambda kv: kv[1], reverse=True)[:2]
                top_svc = sorted(svc_count.items(), key=lambda kv: kv[1], reverse=True)[:2]
                long_sections.append(
                    f"Related incidents include {ids}. "
                    f"Most common peer root causes: {', '.join(f'{k} ({v})' for k, v in top_rc)}; "
                    f"top peer services: {', '.join(f'{k} ({v})' for k, v in top_svc)}."
                )

            long_sections.append(
                "Operationally, prioritize validating configuration drift and dependency health in the same "
                "service/team boundary, then monitor for recurrence in the linked peer incidents."
            )

            parts.append("\n".join(long_sections))

    # --- neighbor count / breakdown ---
    nc_rows = by_step.get("neighbors_count") or []
    nb_rows = by_step.get("neighbors_by_type") or []
    if nc_rows:
        total = nc_rows[0].get("neighborCount")
        if total is not None:
            line = f"Connected to **{total}** nodes"
            if nb_rows:
                top = [
                    f"{r.get('cnt')} {r.get('nodeType')} via {r.get('relation')}"
                    for r in nb_rows[:5]
                    if r.get("cnt") and r.get("nodeType")
                ]
                if top:
                    line += " — " + "; ".join(top)
            parts.append(line + ".")

    # --- similar peers (explicit similar step) ---
    sim_rows = by_step.get("similar") or []
    sim_rows = [
        r for r in sim_rows
        if r.get("incidentId") and (not tgt or r.get("incidentId") != tgt.get("incidentId"))
    ]
    if sim_rows:
        ids = ", ".join(str(r["incidentId"]) for r in sim_rows[:5])
        parts.append(f"Similar incidents ({len(sim_rows)}): {ids}.")
    elif tgt and "similar" in by_step:
        parts.append("No similar peers found in the current graph.")

    # --- filtered list / generic main step ---
    main_rows = by_step.get("main") or []
    if main_rows and not tgt and not nc_rows and not sim_rows:
        main_inc = [r for r in main_rows if r.get("incidentId")]
        if main_inc:
            ids = ", ".join(str(r["incidentId"]) for r in main_inc[:6])
            svc_count: dict[str, int] = {}
            rc_count: dict[str, int] = {}
            region_count: dict[str, int] = {}
            status_count: dict[str, int] = {}
            sev_count: dict[str, int] = {}
            for r in main_inc:
                s = str(r.get("service") or "Unknown")
                svc_count[s] = svc_count.get(s, 0) + 1
                rc = str(r.get("rootCauseCategory") or "Unknown")
                rc_count[rc] = rc_count.get(rc, 0) + 1
                rg = str(r.get("region") or "Unknown")
                region_count[rg] = region_count.get(rg, 0) + 1
                st = str(r.get("status") or "Unknown")
                status_count[st] = status_count.get(st, 0) + 1
                sev = str(r.get("severity") if r.get("severity") is not None else "Unknown")
                sev_count[sev] = sev_count.get(sev, 0) + 1
            top_svc = ", ".join(
                f"{k} ({v})"
                for k, v in sorted(svc_count.items(), key=lambda kv: kv[1], reverse=True)[:3]
            )
            top_rc = ", ".join(
                f"{k} ({v})"
                for k, v in sorted(rc_count.items(), key=lambda kv: kv[1], reverse=True)[:3]
            )
            parts.append(f"Matching incidents ({len(main_inc)}): {ids}.")
            parts.append(f"Top services: {top_svc}. Top root causes: {top_rc}.")
            if wants_cause:
                top_causes = sorted(rc_count.items(), key=lambda kv: kv[1], reverse=True)[:3]
                total = sum(rc_count.values()) or 1
                likely = ", ".join(
                    f"{k} ({v}/{total})" for k, v in top_causes
                )
                parts.append(
                    f"Likely causes for this selection are: {likely}. "
                    "Inference is based on distribution across matched incidents; "
                    "validate with per-incident postmortem timelines for final attribution."
                )
            if wants_long_text:
                top_region = ", ".join(
                    f"{k} ({v})"
                    for k, v in sorted(region_count.items(), key=lambda kv: kv[1], reverse=True)[:3]
                )
                top_status = ", ".join(
                    f"{k} ({v})"
                    for k, v in sorted(status_count.items(), key=lambda kv: kv[1], reverse=True)[:3]
                )
                sev_mix = ", ".join(
                    f"Sev{k}:{v}"
                    for k, v in sorted(sev_count.items(), key=lambda kv: kv[0])[:5]
                )
                parts.append(
                    f"Detailed context: this selection maps to {len(main_inc)} incident records. "
                    f"Top regions are {top_region}; status distribution is {top_status}; "
                    f"severity mix is {sev_mix}."
                )
                parts.append(
                    "Interpretation: this node is associated with a repeated operational pattern rather than "
                    "a one-off event. Use the cited incidents to trace recurring failure modes and validate "
                    "whether mitigations are consistently applied across regions."
                )

    return "\n\n".join(parts) if parts else f"Found {len(rows)} result(s)."


def _answer(
    question: str,
    rows: list[dict[str, Any]],
    plan: CypherPlan,
    *,
    history: list[dict[str, Any]] | None = None,
) -> str:
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

    hist_block = ""
    if history:
        turns: list[str] = []
        for t in history[-4:]:
            q = str(t.get("question") or "").strip()
            a = str(t.get("answer") or "").strip()
            if q:
                turns.append(f"- Q: {q}\n  A: {a[:240]}")
        if turns:
            hist_block = "Previous conversation:\n" + "\n".join(turns) + "\n\n"

    context = json.dumps(grouped, default=str, ensure_ascii=False, indent=2)
    user = (
        f"{hist_block}"
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


_PRONOUN_RE = re.compile(
    # Only anaphoric pronouns that clearly refer back to something the user
    # already brought up. We deliberately omit bare "that" / "it" since they
    # are ubiquitous relative pronouns in English ("incidents that impact...",
    # "show the service it belongs to") and cause false positives on
    # aggregate / structural questions.
    r"\b("
    r"selected(?:\s+(?:node|incident|one))?|"
    r"currently\s+selected|"
    r"this\s+(?:node|incident|one|service|team|region|issue|problem)|"
    r"that\s+(?:node|incident|one|service|team|region|issue|problem)|"
    r"the\s+(?:selected|current|above|previous)(?:\s+(?:node|incident|one))?|"
    r"the\s+node|the\s+incident|the\s+one|"
    r"(?:like|about|to)\s+(?:this|that|it)|"
    r"(?:similar|related|more)\s+(?:to\s+)?(?:this|that|it|these|those|ones?)|"
    r"(?:similar|related)\s+ones?|"
    r"these|those"
    r")\b",
    re.IGNORECASE,
)
_INC_ID_RE = re.compile(r"\bINC-\d{4}-\d{3,5}\b", re.IGNORECASE)


def _resolve_selected_context(
    question: str,
    selected_id: str | None,
    history: list[dict[str, Any]] | None = None,
) -> tuple[str, str | None]:
    """Inject the selected-node / conversation context into the question so
    the LLM and heuristic parser can resolve pronouns like "this", "it",
    "selected node", "those", "the same".

    Precedence when the question contains a pronoun and no explicit ID:
      1. `selected_id` (current node in the UI)
      2. Most recent incident ID mentioned in `history[-1].matchIds`
      3. No change.

    Returns `(rewritten_question, resolved_incident_id_or_None)`.
    """
    if _INC_ID_RE.search(question):
        return question, None

    # ---- 1. explicit selected node ----
    # Only inject selected-node context when the question is clearly ABOUT
    # the selected node. Otherwise the selection silently pollutes aggregate
    # / structural questions like "incidents impacting >= 2 teams".
    has_pronoun = bool(_PRONOUN_RE.search(question))
    # Short follow-up questions (≤ 7 words) without their own structural
    # keywords are almost always about the currently selected node.
    word_count = len(re.findall(r"\w+", question))
    has_structural = bool(re.search(
        r"\b(find|list|all|any|top|most|min|at least|at most|more than|"
        r"greater than|fewer than|less than|how many|count|where|with|"
        r"impact(?:s|ing|ed)?|involve|involves|group|aggregate)\b",
        question, re.IGNORECASE,
    ))
    short_anchor_fallback = (
        selected_id is not None
        and not has_pronoun
        and not has_structural
        and word_count <= 7
    )

    if selected_id and (has_pronoun or short_anchor_fallback):
        if _INC_ID_RE.fullmatch(selected_id.strip()):
            inc_id = selected_id.strip().upper()
            if has_pronoun:
                rewritten = _PRONOUN_RE.sub(inc_id, question, count=1)
            else:
                rewritten = f"{question} (about {inc_id})"
            return rewritten, inc_id
        if ":" in selected_id:
            label, _, name = selected_id.partition(":")
            return f"{question} (about {label.lower()} '{name}')", None
        return f"{question} (context: {selected_id})", None

    # ---- 2. infer anchor from conversation history ----
    if history and _PRONOUN_RE.search(question):
        for turn in reversed(history):
            for mid in (turn.get("matchIds") or []):
                if mid and _INC_ID_RE.fullmatch(str(mid)):
                    inc_id = str(mid).upper()
                    rewritten = _PRONOUN_RE.sub(inc_id, question, count=1)
                    return rewritten, inc_id
            for aid in (turn.get("anchorIds") or []):
                if aid and _INC_ID_RE.fullmatch(str(aid)):
                    inc_id = str(aid).upper()
                    rewritten = _PRONOUN_RE.sub(inc_id, question, count=1)
                    return rewritten, inc_id
            break  # only last turn

    return question, None


def run_pipeline(
    question: str,
    store: AppState,
    *,
    limit: int = 50,
    selected_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> PipelineResult:
    # Resolve pronouns like "selected node" / "it" / "this" / "those" using
    # both the UI's current selection and the prior conversation.
    effective_q, forced_anchor = _resolve_selected_context(
        question, selected_id, history=history
    )

    plan = plan_cypher(effective_q, store, history=history)
    try:
        rows = _execute(plan)
    except Exception as e:  # noqa: BLE001
        log.warning("Cypher execution failed (%s); retrying with heuristic plan.", e)
        plan = _heuristic_plan(effective_q, store)
        try:
            rows = _execute(plan)
        except Exception:  # noqa: BLE001
            rows = []

    rows = _rerank(effective_q, rows, store)[:limit]

    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        m = _flatten_match(r, store)
        if m and m["id"] not in seen:
            matches.append(m)
            seen.add(m["id"])

    match_ids = [m["id"] for m in matches if m["type"] == "Incident"]

    gq = parse_query(effective_q, store)
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
    if forced_anchor and forced_anchor not in anchor_ids:
        anchor_ids.append(forced_anchor)
    if selected_id and ":" in selected_id and store.graph.has_node(selected_id) \
            and selected_id not in anchor_ids:
        anchor_ids.append(selected_id)

    answer = _answer(effective_q, rows, plan, history=history)

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
