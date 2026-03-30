"""Agentic reasoning pipeline.

Implemented as an explicit async generator instead of LangGraph primitives so
we can stream SSE events directly. Flow:
    planner -> router -> executor (tool loop, max N) -> critic -> synthesizer
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator

from app.agent import prompts
from app.agent.nl_query import GraphQuery, parse_query, to_filters
from app.agent.nl2cypher import _resolve_selected_context
from app.config import get_settings
from app.graphrag import tools as tool_registry
from app.state import AppState

log = logging.getLogger("icm.agent")

MAX_TOOL_STEPS = 5


# ---------- LLM adapter (with heuristic fallback) ----------

def _llm_chat(system: str, user: str, *, max_tokens: int = 600, temperature: float = 0.1) -> str:
    s = get_settings()
    if s.has_azure_openai:
        try:
            from app.graphrag.llm import chat
            return chat(system, user, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM chat failed, using heuristic: %s", e)
    return _heuristic_reply(system, user)


def _llm_stream(system: str, user: str, *, max_tokens: int = 700, temperature: float = 0.2) -> AsyncIterator[str]:
    async def _gen() -> AsyncIterator[str]:
        s = get_settings()
        if s.has_azure_openai:
            try:
                from app.graphrag.llm import stream_chat
                for tok in stream_chat(system, user, max_tokens=max_tokens, temperature=temperature):
                    yield tok
                return
            except Exception as e:  # noqa: BLE001
                log.warning("LLM stream failed, using heuristic: %s", e)
        text = _heuristic_synthesis(user)
        for chunk in _split_stream(text):
            yield chunk
    return _gen()


def _split_stream(text: str, chunk: int = 18) -> list[str]:
    return [text[i : i + chunk] for i in range(0, len(text), chunk)]


def _heuristic_reply(system: str, user: str) -> str:
    """Deterministic fallback so the agent works without Azure OpenAI configured.

    NOTE: the planner path here is intentionally minimal — the real routing
    happens in `_plan_from_query()` using the parsed `GraphQuery`. This fallback
    is only used if the LLM planner is enabled and fails mid-stream.
    """
    if "executor" in system.lower():
        return json.dumps({"action": "answer", "thought": "Sufficient evidence already gathered."})
    if "critic" in system.lower():
        return json.dumps({"ok": True, "issues": [], "revisedHint": ""})
    return _heuristic_synthesis(user)


def _plan_from_query(gq: GraphQuery) -> dict[str, Any]:
    """Translate the structured NL query into a retrieval plan + first tool call."""
    # 1. Direct incident lookup.
    if gq.intent == "lookup" and gq.incidentId:
        return {
            "mode": "local",
            "plan": [
                f"Look up incident {gq.incidentId} in the graph.",
                "Expand to service, region, team, and root-cause neighbors.",
                "Return the full record with citations.",
            ],
            "firstTool": {"name": "incident_lookup", "args": {"incident_id": gq.incidentId}},
            "toolChain": [
                {"name": "incident_lookup", "args": {"incident_id": gq.incidentId}},
            ],
        }
    # 1b. Multi-hop: anchored at an incident, find siblings via shared dims.
    if gq.intent == "multi_hop":
        anchor = gq.anchorId or gq.incidentId
        chain: list[dict[str, Any]] = []
        plan_steps: list[str] = []
        if anchor:
            chain.append({"name": "incident_lookup", "args": {"incident_id": anchor}})
            chain.append({"name": "related_incidents",
                          "args": {"incident_id": anchor, "hops": 2, "min_shared": 2, "limit": 12}})
            plan_steps = [
                f"Anchor on {anchor} and pull its service/region/team/root-cause nodes.",
                "Traverse 2 hops to find OTHER incidents sharing >=2 of those dimensions.",
                "Group related incidents by shared-dimension signature.",
                "Explain the recurring pattern with citations.",
            ]
        else:
            # No anchor yet — use filtered local_search to identify candidates, then multi-hop from top.
            args = {"q": gq.question, "top_k": 5}
            args.update(to_filters(gq))
            chain.append({"name": "local_search", "args": args})
            plan_steps = [
                "No explicit anchor — run filtered local_search first.",
                "Pick the top match as the anchor for multi-hop expansion.",
                "Traverse shared service/root-cause dimensions.",
            ]
        return {"mode": "multi_hop", "plan": plan_steps, "firstTool": chain[0], "toolChain": chain}
    # 1c. Compare two entities.
    if gq.intent == "compare" and gq.compareLeft and gq.compareRight:
        # Infer dimension: if both are services in vocab we pick service; else service default.
        dim = "service"
        args = {"left": gq.compareLeft, "right": gq.compareRight, "dimension": dim}
        chain = [{"name": "compare_entities", "args": args}]
        # Also pull co-occurrence for the left anchor so the answer can say "both share cluster C-X".
        chain.append({"name": "cooccurrence", "args": {"anchor": gq.compareLeft, "top": 5}})
        return {
            "mode": "compare",
            "plan": [
                f"Break down incidents for {gq.compareLeft} and {gq.compareRight} side-by-side.",
                "List shared root-causes and divergent patterns.",
                "Surface communities where both appear.",
            ],
            "firstTool": chain[0],
            "toolChain": chain,
        }
    # 1d. Path between two entities.
    if gq.intent == "path" and gq.pathSrc and gq.pathDst:
        args = {"src": gq.pathSrc, "dst": gq.pathDst, "max_len": 6}
        return {
            "mode": "path",
            "plan": [
                f"Find the shortest relational path between {gq.pathSrc} and {gq.pathDst}.",
                "Report each hop with its relation.",
                "Complement with incidents sitting on that path.",
            ],
            "firstTool": {"name": "shortest_path", "args": args},
            "toolChain": [
                {"name": "shortest_path", "args": args},
                {"name": "cooccurrence", "args": {"anchor": gq.pathSrc, "top": 5}},
            ],
        }
    # 1e. Co-occurrence / dependency question.
    if gq.intent == "cooccur":
        anchor = gq.service or gq.region or gq.rootCause or (gq.keywords[0] if gq.keywords else "")
        if anchor:
            return {
                "mode": "cooccur",
                "plan": [
                    f"Find Louvain communities containing {anchor}.",
                    "Count peer services, regions, and root-causes in those communities.",
                    "Pair with filtered local_search for incident citations.",
                ],
                "firstTool": {"name": "cooccurrence", "args": {"anchor": anchor, "top": 5}},
                "toolChain": [
                    {"name": "cooccurrence", "args": {"anchor": anchor, "top": 5}},
                    {"name": "local_search",
                     "args": {"q": gq.question, "top_k": 6, **to_filters(gq)}},
                ],
            }
    # 2. Cluster / storm / cascade.
    if gq.intent == "cluster":
        args: dict[str, Any] = {"q": gq.question, "top_communities": 3, "top_k": 8}
        plan = ["Detect the cluster via Louvain community summaries.",
                "Drift-search inside the top communities for citation incidents."]
        if gq.service or gq.region or gq.rootCause or gq.startIso:
            plan.append(
                "Filter candidates by "
                + ", ".join(f"{k}={v}" for k, v in to_filters(gq).items())
            )
        plan.append("Summarize shared root cause + mitigation.")
        return {"mode": "drift", "plan": plan,
                "firstTool": {"name": "drift_search", "args": args},
                "toolChain": [{"name": "drift_search", "args": args}]}
    # 3. Summary / trend / executive overview.
    if gq.intent == "summarize":
        args = {"q": gq.question, "top_k": 5}
        return {
            "mode": "global",
            "plan": [
                "Map over community summaries matching the question.",
                "Reduce into thematic observations.",
                "Cite communities and anchor incidents.",
            ],
            "firstTool": {"name": "global_search", "args": args},
            "toolChain": [{"name": "global_search", "args": args}],
        }
    # 4. Temporal-only filter (month/quarter with no service/region/cluster).
    if (gq.startIso or gq.endIso) and not (gq.service or gq.region or gq.rootCause or gq.team):
        args = {"start": gq.startIso, "end": gq.endIso, "service": gq.service}
        return {
            "mode": "local",
            "plan": [
                f"Filter incidents by time window {gq.startIso or '*'} → {gq.endIso or '*'}.",
                "Rank by severity and impacted customers.",
                "Summarize with citations.",
            ],
            "firstTool": {"name": "temporal_filter", "args": args},
            "toolChain": [{"name": "temporal_filter", "args": args}],
        }
    # 4b. Aggregate / structural question — delegate to read-only Cypher.
    ql = (gq.question or "").lower()
    agg_pat = re.compile(
        r"\b(how many|count of|count\(|at least|at most|min\s+\d+|minimum of|"
        r"more than|greater than|>=|<=|fewer than|less than|"
        r"impact(?:s|ing|ed)?\s+\d+|impact(?:s|ing|ed)?\s+(?:multiple|more|min|at)|"
        r"spanning|span(?:s|ned)?|across\s+\d+|per\s+(?:month|team|service|region))\b"
    )
    if agg_pat.search(ql):
        # Extract threshold N (default 2) and dimension (team/service/region).
        m_n = re.search(r"\b(?:>=|min|at\s*least|more\s+than|greater\s+than)\s*(\d+)", ql)
        threshold = int(m_n.group(1)) if m_n else 2
        # Strict "more than N" means >= N+1.
        if m_n and "more than" in ql:
            threshold = threshold + 1
        # Pick dimension (teams default).
        dim = "teams"
        dim_rel = "OWNED_BY"
        dim_label = "Team"
        if re.search(r"\b(services?|products?)\b", ql):
            dim, dim_rel, dim_label = "services", "BELONGS_TO", "Service"
        elif re.search(r"\b(regions?|geographies?)\b", ql):
            dim, dim_rel, dim_label = "regions", "IN_REGION", "Region"
        cypher_main = (
            f"MATCH (i:Incident)-[:{dim_rel}]->(x:{dim_label}) "
            f"WITH i, collect(DISTINCT x.name) AS {dim}, count(DISTINCT x) AS {dim[:-1]}Count "
            f"WHERE {dim[:-1]}Count >= $n "
            f"RETURN i.incidentId AS incidentId, i.title AS title, i.severity AS severity, "
            f"       i.service AS service, i.region AS region, i.status AS status, "
            f"       {dim}, {dim[:-1]}Count "
            f"ORDER BY {dim[:-1]}Count DESC, i.severity ASC LIMIT 25"
        )
        cypher_fallback = (
            f"MATCH (i:Incident)-[:IN_COMMUNITY]->(c:Community) "
            f"MATCH (i)-[:{dim_rel}]->(x:{dim_label}) "
            f"WITH c, collect(DISTINCT x.name) AS {dim}, collect(DISTINCT i.incidentId) AS incidents "
            f"WHERE size({dim}) >= $n "
            f"RETURN c.communityId AS communityId, size({dim}) AS {dim[:-1]}Count, "
            f"       {dim}[..8] AS sample{dim_label}s, incidents[..6] AS sampleIncidents, c.summary AS summary "
            f"ORDER BY {dim[:-1]}Count DESC LIMIT 10"
        )
        return {
            "mode": "local",
            "plan": [
                f"Translate the aggregate question into read-only Cypher (threshold: {dim[:-1]}Count >= {threshold}).",
                "Execute against Memgraph and collect concrete incident IDs.",
                f"If no single incident spans >= {threshold} {dim}, fall back to communities that do.",
                "Answer with honest, grounded counts and citations.",
            ],
            "firstTool": {
                "name": "cypher_query",
                "args": {"cypher": cypher_main, "params": {"n": threshold}},
            },
            "toolChain": [
                {"name": "cypher_query",
                 "args": {"cypher": cypher_main, "params": {"n": threshold}}},
                {"name": "cypher_query",
                 "args": {"cypher": cypher_fallback, "params": {"n": threshold}}},
            ],
        }
    # 5. Default: filtered local search.
    args = {"q": gq.question, "top_k": 10}
    filters = to_filters(gq)
    args.update(filters)
    plan = []
    if filters:
        plan.append("Parsed filters: " + ", ".join(f"{k}={v}" for k, v in filters.items()))
    plan += [
        "Anchor on matched entities in the KG.",
        "Rank candidates with embedding similarity.",
        "Answer with inline incident citations.",
    ]
    return {"mode": "local", "plan": plan,
            "firstTool": {"name": "local_search", "args": args},
            "toolChain": [{"name": "local_search", "args": args}]}


def _heuristic_synthesis(user: str) -> str:
    """Data-driven fallback: parse the EVIDENCE block embedded in `user` and
    produce a query-specific answer (not a boilerplate template)."""
    from collections import Counter

    # Evidence lines look like:
    #   [INC-2026-0137] Sev1 Front Door/westus — <title> | rc=Certificate mit=<text> status=Resolved
    #   [C-003 size=12] <summary>
    incident_re = re.compile(
        r"\[(?P<id>INC-2026-\d{4})\]\s+Sev(?P<sev>\d)\s+(?P<svc>[^/]+)/(?P<reg>\S+)\s+—\s+"
        r"(?P<title>[^|]+)\|\s+rc=(?P<rc>[^ ]+(?:\s+[^ |]+)*?)\s+mit=(?P<mit>.+?)\s+status=(?P<st>\S+)"
    )
    comm_re = re.compile(r"\[(?P<id>C-\d{3})(?:\s+size=(?P<sz>\d+))?\]\s+(?P<sum>.+)")

    question_match = re.search(r"QUESTION:\s*(.+?)\n", user)
    question = question_match.group(1).strip() if question_match else ""
    ql = question.lower()
    wants_cause = any(
        k in ql for k in (
            "root cause", "what cause", "what caused", "cause this", "cause it",
            "why this", "why did", "reason", "reason why",
        )
    )

    # ---- multi-hop / compare / path markers inserted by _evidence_text ----
    related_groups = re.search(r"RELATED-GROUPS:\s*(.+)", user)
    compare_line = re.search(r"COMPARE\s+(.+)", user)
    path_line = re.search(r"PATH (?:length=\d+:|not found) .+", user)
    cooccur_line = re.search(r"COOCCUR\s+(.+)", user)

    incs: list[dict[str, str]] = []
    comms: list[dict[str, str]] = []
    anchor_id: str | None = None
    anchor_match = re.search(r"^ANCHOR:\s*(INC-2026-\d{4})", user, re.MULTILINE)
    if anchor_match:
        anchor_id = anchor_match.group(1)
    for line in user.splitlines():
        mi = incident_re.search(line)
        if mi:
            g = mi.groupdict()
            incs.append(g)
            continue
        mc = comm_re.search(line)
        if mc:
            comms.append(mc.groupdict())

    # ---- CAUSE answer (shared sectioned style) ----
    if wants_cause and incs:
        from collections import Counter as _C
        focus = incs[0]
        rcs = _C(i["rc"].strip() for i in incs)
        svcs = _C(i["svc"].strip() for i in incs)
        regs = _C(i["reg"].strip() for i in incs)
        top_causes = ", ".join(f"{k} ({v})" for k, v in rcs.most_common(3))
        top_svcs = ", ".join(f"{k} ({v})" for k, v in svcs.most_common(3))
        top_regs = ", ".join(f"{k} ({v})" for k, v in regs.most_common(3))
        peer_ids = ", ".join(f"[{i['id']}]" for i in incs[1:6])

        lines = ["**Cause**"]
        lines.append(
            f"Likely primary cause is **{focus['rc'].strip()}** for [{focus['id']}] "
            f"(Sev{focus['sev']} {focus['svc'].strip()}/{focus['reg'].strip()})."
        )
        lines.append("\n**Evidence**")
        lines.append(
            f"Observed cause distribution in retrieved evidence: {top_causes}. "
            f"Service spread: {top_svcs}. Region spread: {top_regs}."
        )
        if peer_ids:
            lines.append(f"Supporting peer incidents: {peer_ids}.")
        lines.append("\n**Confidence**")
        lines.append(
            "Medium. Inference is grounded in retrieved incident labels and frequencies, "
            "not a full postmortem chain."
        )
        lines.append("\n**Next Checks**")
        lines.append(
            "1. Validate with timeline-level evidence (deployments, config changes, dependency events)."
        )
        lines.append(
            "2. Confirm whether the same cause pattern recurs in the cited peer incidents."
        )
        return "\n".join(lines)

    # ---- PATH answer ----
    if path_line:
        chain = path_line.group(0)
        return (
            f"## Summary\nThe knowledge graph connects the two entities in your question "
            f"via the following relational path.\n\n## Path\n`{chain}`\n\n"
            "## Interpretation\nEach hop is an explicit KG edge (hosted_in, owns, caused_by, "
            "in_community). The path length is the minimum number of relationships required "
            "to traverse from source to destination.\n"
        )

    # ---- COMPARE answer ----
    if compare_line and not related_groups:
        from collections import Counter as _C
        left_rows = [i for i in incs[:5]]
        right_rows = [i for i in incs[5:10]]
        lines = [f"## Summary\n{compare_line.group(0).strip()}", "\n## Side-by-side incidents"]
        if left_rows:
            lines.append("### Left")
            for i in left_rows:
                lines.append(f"- [{i['id']}] Sev{i['sev']} {i['svc'].strip()}/{i['reg'].strip()} — {i['title'].strip()} (rc={i['rc'].strip()})")
        if right_rows:
            lines.append("### Right")
            for i in right_rows:
                lines.append(f"- [{i['id']}] Sev{i['sev']} {i['svc'].strip()}/{i['reg'].strip()} — {i['title'].strip()} (rc={i['rc'].strip()})")
        lines.append("\n## Recommended Next Steps")
        lines.append("1. Focus mitigation investment on the shared root-cause categories.")
        lines.append("2. Verify whether divergent causes reflect genuinely different failure modes or gaps in tagging.")
        return "\n".join(lines)

    # ---- MULTI-HOP answer ----
    if related_groups and anchor_id:
        from collections import Counter as _C
        rel_rows = [i for i in incs if i["id"] != anchor_id]
        svcs = _C(i["svc"].strip() for i in rel_rows)
        rcs = _C(i["rc"].strip() for i in rel_rows)
        regs = _C(i["reg"].strip() for i in rel_rows)
        anchor = next((i for i in incs if i["id"] == anchor_id), None)
        lines = [f"## Summary"]
        if anchor:
            lines.append(
                f"Anchor incident **[{anchor_id}]** (Sev{anchor['sev']} {anchor['svc'].strip()}/{anchor['reg'].strip()}, "
                f"rc={anchor['rc'].strip()}) has **{len(rel_rows)}** related incidents in the KG "
                "that share at least 2 of its {service, region, team, rootCause} dimensions."
            )
        lines.append("\n## Shared-signature groups")
        lines.append(related_groups.group(1).strip())
        lines.append("\n## Related incidents")
        for i in rel_rows[:8]:
            lines.append(
                f"- [{i['id']}] Sev{i['sev']} {i['svc'].strip()}/{i['reg'].strip()} — "
                f"{i['title'].strip()} (rc={i['rc'].strip()}, mit={i['mit'].strip()})"
            )
        if rcs:
            top_rc = rcs.most_common(1)[0][0]
            lines.append(f"\n## Pattern\nThe dominant shared failure mode is **{top_rc}** "
                         f"(appears in {rcs[top_rc]} of {len(rel_rows)} related incidents).")
        lines.append("\n## Recommended Next Steps")
        if anchor:
            lines.append(f"1. Treat {anchor_id} as an exemplar; apply its mitigation across the related group.")
        if svcs:
            lines.append(f"2. Inspect the most-affected service ({svcs.most_common(1)[0][0]}) for a systemic defect.")
        lines.append("3. Add a correlation rule on the shared signature so future occurrences auto-link.")
        return "\n".join(lines)

    # ---- COOCCUR answer ----
    if cooccur_line and not related_groups:
        lines = [f"## Summary", cooccur_line.group(1).strip(), "\n## Community context"]
        for c in comms[:3]:
            summary = c["sum"].strip()
            if len(summary) > 220:
                summary = summary[:220] + "…"
            lines.append(f"- [{c['id']}] {summary}")
        if incs:
            lines.append("\n## Citations")
            for i in incs[:6]:
                lines.append(f"- [{i['id']}] Sev{i['sev']} {i['svc'].strip()}/{i['reg'].strip()} — {i['title'].strip()}")
        return "\n".join(lines)

    # ---- CYPHER / aggregate answer (honest grounded response) ----
    cypher_header_re = re.compile(r"^CYPHER\s+(\d+)/(\d+)\s+rows\b", re.MULTILINE)
    cypher_headers = cypher_header_re.findall(user)
    if cypher_headers:
        row_re = re.compile(r"^ROW\s+(.+)$", re.MULTILINE)
        cypher_rows = row_re.findall(user)

        def _parse_row(line: str) -> dict[str, str]:
            out: dict[str, str] = {}
            # Split on ', ' but preserve lists in [..]
            depth = 0
            buf: list[str] = []
            parts: list[str] = []
            for ch in line:
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                if ch == "," and depth == 0:
                    parts.append("".join(buf).strip())
                    buf = []
                    continue
                buf.append(ch)
            if buf:
                parts.append("".join(buf).strip())
            for p in parts:
                if "=" in p:
                    k, _, v = p.partition("=")
                    out[k.strip()] = v.strip()
            return out

        parsed_rows = [_parse_row(r) for r in cypher_rows]
        total_first = int(cypher_headers[0][0]) if cypher_headers else 0

        # Case: no incidents match, but community fallback has rows.
        if total_first == 0 and parsed_rows:
            lines = [f"## Summary"]
            lines.append(
                f"No single incident in the graph matches `{question}`. "
                f"However, **{len(parsed_rows)}** Louvain community cluster(s) do meet the threshold, "
                "so the cross-team / cross-service pattern lives at the community level."
            )
            lines.append("\n## Communities")
            for r in parsed_rows[:6]:
                cid = r.get("communityId") or r.get("id") or "?"
                cnt = r.get("teamCount") or r.get("serviceCount") or r.get("regionCount") or "?"
                samples = (r.get("sampleTeams") or r.get("sampleServices") or r.get("sampleRegions") or "").strip("[]")
                incs_s = (r.get("sampleIncidents") or "").strip("[]")
                summary = r.get("summary") or ""
                if len(summary) > 180:
                    summary = summary[:180] + "…"
                bullet = f"- [{cid}] count={cnt}"
                if samples:
                    bullet += f" · members: {samples}"
                if incs_s:
                    bullet += f" · incidents: {incs_s}"
                if summary:
                    bullet += f"\n  {summary}"
                lines.append(bullet)
            lines.append("\n## Interpretation")
            lines.append(
                "Individual incidents in this dataset are each owned by a single team / service, "
                "so team-level co-impact only emerges when you group related incidents into "
                "communities. The communities above are the places to look for cross-team blast radius."
            )
            lines.append("\n## Recommended Next Steps")
            lines.append("1. Inspect the top community for shared root-cause signatures.")
            lines.append("2. Use `drift_search` or `cooccurrence` on the top community's anchor service for narrative context.")
            lines.append("3. If you need incident-level joins, run a graph query that traverses `LINKED_TO` between incidents.")
            return "\n".join(lines)

        # Case: incidents do match the aggregate.
        if parsed_rows and any("incidentId" in r for r in parsed_rows):
            lines = [f"## Summary",
                     f"**{total_first}** incident(s) match `{question}`."]
            lines.append("\n## Matching incidents")
            for r in parsed_rows[:12]:
                iid = r.get("incidentId", "?")
                title = r.get("title", "")
                svc = r.get("service", "")
                reg = r.get("region", "")
                cnt = r.get("teamCount") or r.get("serviceCount") or r.get("regionCount") or ""
                members = (r.get("teams") or r.get("services") or r.get("regions") or "").strip("[]")
                bullet = f"- [{iid}] {svc}/{reg} — {title}"
                if cnt:
                    bullet += f" · count={cnt}"
                if members:
                    bullet += f" · {members}"
                lines.append(bullet)
            return "\n".join(lines)

    if not incs and not comms:
        ids = list(dict.fromkeys(re.findall(r"INC-2026-\d{4}", user)))
        cites = ", ".join(ids[:4]) or "(no matching incidents)"
        return (
            f"## Summary\nNo evidence rows matched the query `{question}`. "
            f"Known IDs in context: {cites}.\n"
        )

    # Aggregate structured stats.
    svcs = Counter(i["svc"].strip() for i in incs)
    regs = Counter(i["reg"].strip() for i in incs)
    rcs = Counter(i["rc"].strip() for i in incs)
    sevs = Counter(i["sev"] for i in incs)
    statuses = Counter(i["st"] for i in incs)
    mits = Counter(i["mit"].strip() for i in incs)

    top_svc = ", ".join(f"{s} ({n})" for s, n in svcs.most_common(3))
    top_reg = ", ".join(f"{r} ({n})" for r, n in regs.most_common(3))
    top_rc = ", ".join(f"{r} ({n})" for r, n in rcs.most_common(3))
    sev_dist = ", ".join(f"Sev{k}:{v}" for k, v in sorted(sevs.items()))
    status_dist = ", ".join(f"{k}:{v}" for k, v in statuses.most_common())
    top_mit = ", ".join(f"{m} ({n})" for m, n in mits.most_common(2))

    cite_list = [i["id"] for i in incs[:6]]
    inc_cites = ", ".join(f"[{cid}]" for cid in cite_list)

    lines: list[str] = []
    lines.append(f"## Summary")
    if question:
        lines.append(f"For **{question}**, the graph returned **{len(incs)}** matching incident(s)"
                     + (f" across **{len(comms)}** community cluster(s)" if comms else "") + ".")
    else:
        lines.append(f"Retrieved {len(incs)} incidents" + (f" and {len(comms)} community cluster(s)." if comms else "."))

    lines.append("\n## Observations")
    if svcs:
        lines.append(f"- **Services**: {top_svc}")
    if regs:
        lines.append(f"- **Regions**: {top_reg}")
    if rcs:
        lines.append(f"- **Root causes**: {top_rc}")
    if sevs:
        lines.append(f"- **Severity mix**: {sev_dist}")
    if statuses:
        lines.append(f"- **Status**: {status_dist}")

    # Show up to 5 incidents with context, each one different.
    lines.append("\n## Evidence")
    for i in incs[:6]:
        title = i["title"].strip()
        lines.append(
            f"- [{i['id']}] Sev{i['sev']} {i['svc'].strip()} / {i['reg'].strip()} — {title} "
            f"(cause: {i['rc'].strip()}, mitigation: {i['mit'].strip()}, status: {i['st']})"
        )

    if comms:
        lines.append("\n## Community context")
        for c in comms[:3]:
            summary = c["sum"].strip()
            if len(summary) > 220:
                summary = summary[:220] + "…"
            size = f" (size={c['sz']})" if c.get("sz") else ""
            lines.append(f"- [{c['id']}]{size} {summary}")

    lines.append("\n## Likely Cause & Mitigations")
    rc_top = rcs.most_common(1)[0][0] if rcs else "mixed"
    lines.append(
        f"The dominant root-cause signal is **{rc_top}** "
        + (f"(out of {sum(rcs.values())} observations). " if rcs else ". ")
        + (f"Most common mitigations: {top_mit}." if mits else "")
    )

    lines.append("\n## Recommended Next Steps")
    lines.append(f"1. Drill into the top incidents ({inc_cites}) for shared component signatures.")
    if regs:
        lines.append(f"2. Confirm whether the issue is regional (top regions: {top_reg}) or global.")
    if rcs:
        lines.append(f"3. Check change-management windows for **{rc_top}** categories.")
    return "\n".join(lines)


# ---------- JSON helper ----------

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json(s: str) -> dict[str, Any]:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).rstrip("`").strip()
    m = _JSON_RE.search(s)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {}


# ---------- Evidence ----------

def _evidence_text(evidence: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    cited_ids: set[str] = set()

    def _row(inc: dict[str, Any]) -> str | None:
        iid = inc.get("incidentId")
        if not iid or iid in cited_ids:
            return None
        cited_ids.add(iid)
        extra = ""
        if inc.get("sharedWithAnchor"):
            extra = f" shared={'+'.join(inc['sharedWithAnchor'])}"
        return (
            f"[{iid}] Sev{inc.get('severity')} {inc.get('service')}/{inc.get('region')} "
            f"— {inc.get('title')} | rc={inc.get('rootCauseCategory')} "
            f"mit={inc.get('mitigation')} status={inc.get('status')}{extra}"
        )

    for ev in evidence:
        result = ev.get("result")
        tool = ev.get("tool")
        if result is None:
            continue
        # related_incidents result.
        if tool == "related_incidents" and isinstance(result, dict):
            anchor = result.get("anchor") or {}
            if anchor.get("incidentId"):
                lines.append(f"ANCHOR: {anchor['incidentId']}")
                anchor_line = _row(anchor)
                if anchor_line:
                    lines.append(anchor_line)
            for inc in result.get("related", []) or []:
                line = _row(inc)
                if line:
                    lines.append(line)
            groups = result.get("groups") or {}
            if groups:
                lines.append("RELATED-GROUPS: " + "; ".join(
                    f"{k}={','.join(v[:4])}" for k, v in groups.items()
                ))
            continue
        # compare_entities result.
        if tool == "compare_entities" and isinstance(result, dict):
            L, R = result.get("left") or {}, result.get("right") or {}
            lines.append(
                f"COMPARE {L.get('value')}={L.get('total')} incidents "
                f"rc={list(L.get('byRootCause', {}).keys())[:3]} "
                f"VS {R.get('value')}={R.get('total')} incidents "
                f"rc={list(R.get('byRootCause', {}).keys())[:3]} "
                f"shared={result.get('sharedRootCauses')}"
            )
            for inc in (L.get("topIncidents") or []) + (R.get("topIncidents") or []):
                line = _row(inc)
                if line:
                    lines.append(line)
            continue
        # shortest_path result.
        if tool == "shortest_path" and isinstance(result, dict):
            if result.get("found"):
                steps = result.get("steps") or []
                chain = " -> ".join(
                    f"{s.get('fromLabel') or s['from']}[{s.get('fromType')}] -({s.get('relation')})-> "
                    f"{s.get('toLabel') or s['to']}[{s.get('toType')}]"
                    for s in steps
                )
                lines.append(f"PATH length={result.get('length')}: {chain}")
            else:
                lines.append(f"PATH not found between {result.get('src')} and {result.get('dst')}")
            continue
        # cooccurrence result.
        if tool == "cooccurrence" and isinstance(result, dict):
            lines.append(
                f"COOCCUR anchor={result.get('anchor')} "
                f"coServices={list(result.get('coServices', {}).items())[:5]} "
                f"coRootCauses={list(result.get('coRootCauses', {}).items())[:5]}"
            )
            for c in result.get("communities", []) or []:
                cid = c.get("communityId")
                if not cid or cid in cited_ids:
                    continue
                cited_ids.add(cid)
                lines.append(f"[{cid} size={c.get('size')}] {c.get('summary')}")
            continue
        # cypher_query result (read-only, schema-free rows).
        if tool == "cypher_query" and isinstance(result, dict):
            if result.get("error"):
                lines.append(f"CYPHER ERROR: {str(result['error'])[:200]}")
                continue
            rows = result.get("rows") or []
            total = result.get("total", len(rows))
            lines.append(
                f"CYPHER {len(rows)}/{total} rows — cypher: {str(result.get('cypher', ''))[:240]}"
            )
            # Surface incident rows through the standard formatter so the
            # synthesizer can cite them; also show raw rows for structural data.
            for row in rows[:20]:
                if isinstance(row, dict) and row.get("incidentId"):
                    line = _row(row)
                    if line:
                        # Append structural extras (teams, services, communities) if present.
                        extras = []
                        for k in ("teams", "services", "communities", "neighborCount",
                                  "nodeType", "relation", "cnt"):
                            if k in row and row[k] not in (None, [], ""):
                                extras.append(f"{k}={row[k]}")
                        if extras:
                            line = line + " | " + " ".join(extras)
                        lines.append(line)
                else:
                    # Non-incident rows: dump as compact key=value pairs.
                    pairs = ", ".join(f"{k}={v}" for k, v in (row.items() if isinstance(row, dict) else []))
                    if pairs:
                        lines.append(f"ROW {pairs[:240]}")
            continue
        # Shape A: list of incidents (temporal_filter).
        if isinstance(result, list):
            for inc in result:
                line = _row(inc) if isinstance(inc, dict) else None
                if line:
                    lines.append(line)
            continue
        # Shape B: single incident dict (incident_lookup).
        if isinstance(result, dict) and result.get("incidentId") and "incidents" not in result:
            line = _row(result)
            if line:
                lines.append(line)
            continue
        # Shape C: structured retrieval result.
        if isinstance(result, dict):
            for inc in result.get("incidents", []) or []:
                line = _row(inc)
                if line:
                    lines.append(line)
            for c in result.get("communities", []) or []:
                cid = c.get("communityId")
                if not cid or cid in cited_ids:
                    continue
                cited_ids.add(cid)
                lines.append(f"[{cid} size={c.get('size')}] {c.get('summary')}")
    return "\n".join(lines[:40])


def _collect_citations(evidence: list[dict[str, Any]]) -> list[str]:
    cited: list[str] = []

    def _add(val: str | None) -> None:
        if val and val not in cited:
            cited.append(val)

    for ev in evidence:
        result = ev.get("result")
        tool = ev.get("tool")
        if result is None:
            continue
        if tool == "related_incidents" and isinstance(result, dict):
            anchor = result.get("anchor") or {}
            _add(anchor.get("incidentId"))
            for inc in result.get("related", []) or []:
                _add(inc.get("incidentId"))
            continue
        if tool == "compare_entities" and isinstance(result, dict):
            for side in ("left", "right"):
                for inc in (result.get(side) or {}).get("topIncidents") or []:
                    _add(inc.get("incidentId"))
            continue
        if tool == "cooccurrence" and isinstance(result, dict):
            for c in result.get("communities") or []:
                _add(c.get("communityId"))
            continue
        if tool == "cypher_query" and isinstance(result, dict):
            for row in result.get("rows") or []:
                if isinstance(row, dict):
                    _add(row.get("incidentId"))
                    _add(row.get("communityId"))
            continue
        if isinstance(result, list):
            for inc in result:
                if isinstance(inc, dict):
                    _add(inc.get("incidentId"))
            continue
        if isinstance(result, dict):
            if result.get("incidentId") and "incidents" not in result:
                _add(result.get("incidentId"))
                continue
            for inc in result.get("incidents", []) or []:
                _add(inc.get("incidentId"))
            for c in result.get("communities", []) or []:
                _add(c.get("communityId"))
    return cited


# ---------- Public entrypoint ----------

async def run_agent(
    store: AppState,
    question: str,
    *,
    history: list[dict[str, Any]] | None = None,
    selected_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    session_id = uuid.uuid4().hex[:12]
    started = time.time()
    tools = tool_registry.make_tools(store)
    tool_desc = tool_registry.describe_tools(tools)
    trace: list[dict[str, Any]] = []

    def _emit(event: dict[str, Any]) -> dict[str, Any]:
        event["sessionId"] = session_id
        event["t"] = round(time.time() - started, 3)
        trace.append(event)
        return event

    # Resolve pronouns ("it", "this", "selected", "those") using both the UI's
    # current selection and the prior conversation. The rewritten question is
    # what every downstream prompt / parser / tool sees; the original remains
    # in the session event for transparency.
    effective_q, forced_anchor = _resolve_selected_context(
        question, selected_id, history=history
    )
    yield _emit({
        "type": "session",
        "question": question,
        "effectiveQuestion": effective_q if effective_q != question else None,
        "selectedId": selected_id,
        "historyTurns": len(history or []),
    })

    # Build a compact conversation block reused by planner/executor/synth prompts.
    conv_block = ""
    if history:
        lines: list[str] = []
        for t in history[-4:]:
            q = str(t.get("question") or "").strip()
            a = str(t.get("answer") or "").strip()
            cites = ", ".join(list(t.get("citations") or t.get("matchIds") or [])[:5])
            if q:
                line = f"- Q: {q}\n  A: {a[:240]}"
                if cites:
                    line += f"\n  cited: {cites}"
                lines.append(line)
        if lines:
            conv_block = (
                "Recent conversation (oldest → newest). Use it to resolve "
                "pronouns/follow-ups like 'those', 'these', 'the same', "
                "'more like that':\n" + "\n".join(lines) + "\n\n"
            )

    question_for_prompts = (
        conv_block + ("Question: " + effective_q if conv_block else effective_q)
    )

    # 0. Parse NL → structured graph query (always runs; grounds the whole agent).
    gq: GraphQuery = parse_query(effective_q, store)
    if forced_anchor and not gq.incidentId:
        gq.incidentId = forced_anchor
        if not gq.anchorId:
            gq.anchorId = forced_anchor
    yield _emit({
        "type": "graph_query",
        "query": gq.to_dict(),
        "describe": gq.describe(),
    })

    # 1. Planner — prefer the LLM when configured, else use the structured query.
    heur_plan = _plan_from_query(gq)
    if get_settings().has_azure_openai:
        plan_raw = _llm_chat(prompts.PLANNER_SYSTEM, question_for_prompts, max_tokens=400)
        plan = _parse_json(plan_raw) or heur_plan
    else:
        plan = heur_plan
    mode = plan.get("mode", heur_plan["mode"])
    steps = plan.get("plan", heur_plan["plan"])
    yield _emit({"type": "plan", "mode": mode, "steps": steps})

    # 2. Initial tool call from planner
    evidence: list[dict[str, Any]] = []
    first_tool = plan.get("firstTool") or heur_plan["firstTool"]
    tool_chain: list[dict[str, Any]] = list(plan.get("toolChain") or heur_plan.get("toolChain") or [first_tool])
    has_llm = get_settings().has_azure_openai

    # Heuristic mode: execute the pre-baked chain deterministically.
    if not has_llm:
        for step, tool_call in enumerate(tool_chain[:MAX_TOOL_STEPS]):
            tool_name = tool_call.get("name")
            tool_args = dict(tool_call.get("args") or {})
            if tool_name not in tools:
                break
            yield _emit({"type": "tool_call", "step": step, "tool": tool_name, "args": tool_args})
            try:
                result = tool_registry.call_tool(tools, tool_name, tool_args)
            except Exception as e:  # noqa: BLE001
                result = {"error": str(e)}
            evidence.append({"tool": tool_name, "args": tool_args, "result": result})
            yield _emit({
                "type": "tool_result",
                "step": step,
                "tool": tool_name,
                "summary": _summarize_result(tool_name, result),
            })
            # Auto-chain: if multi_hop but anchor was missing, promote top local_search hit.
            if (gq.intent == "multi_hop" and tool_name == "local_search"
                    and isinstance(result, dict) and result.get("incidents")
                    and not any(tc.get("name") == "related_incidents" for tc in tool_chain)):
                anchor = result["incidents"][0]["incidentId"]
                tool_chain.append({
                    "name": "related_incidents",
                    "args": {"incident_id": anchor, "hops": 2, "min_shared": 2, "limit": 12},
                })
                yield _emit({"type": "thought", "step": step,
                             "thought": f"Anchor selected from top hit: {anchor}."})
    else:
        for step in range(MAX_TOOL_STEPS):
            tool_name = first_tool.get("name")
            tool_args = first_tool.get("args") or {}
            if tool_name not in tools:
                break
            yield _emit({"type": "tool_call", "step": step, "tool": tool_name, "args": tool_args})
            try:
                result = tool_registry.call_tool(tools, tool_name, tool_args)
            except Exception as e:  # noqa: BLE001
                result = {"error": str(e)}
            evidence.append({"tool": tool_name, "args": tool_args, "result": result})
            yield _emit({
                "type": "tool_result",
                "step": step,
                "tool": tool_name,
                "summary": _summarize_result(tool_name, result),
            })

            # Decide next action.
            exec_prompt = prompts.EXECUTOR_SYSTEM.format(tools=tool_desc, evidence=_evidence_text(evidence) or "(none)")
            decision_raw = _llm_chat(exec_prompt, question_for_prompts, max_tokens=220)
            decision = _parse_json(decision_raw)
            thought = decision.get("thought", "")
            if thought:
                yield _emit({"type": "thought", "step": step, "thought": thought})
            if decision.get("action") == "answer":
                break
            nxt = decision.get("tool") or {}
            if nxt.get("name") and nxt["name"] in tools:
                first_tool = nxt
            else:
                break

    citations = _collect_citations(evidence)

    # 3. Synthesis (streamed)
    ctx = _evidence_text(evidence)
    synth_user = (
        (conv_block if conv_block else "")
        + f"QUESTION:\n{effective_q}\n\n"
        + (f"SELECTED NODE: {selected_id}\n\n" if selected_id else "")
        + f"EVIDENCE (only use IDs from this list):\n{ctx or '(no evidence)'}\n"
    )
    answer_chunks: list[str] = []
    async for tok in _llm_stream(prompts.SYNTHESIZER_SYSTEM, synth_user, max_tokens=700):
        answer_chunks.append(tok)
        yield _emit({"type": "token", "text": tok})
    answer = "".join(answer_chunks)

    yield _emit({
        "type": "final",
        "answer": answer,
        "citations": citations,
        "evidence": evidence,
    })

    store.sessions[session_id] = trace


def _summarize_result(tool_name: str, result: Any) -> str:
    if tool_name == "cypher_query" and isinstance(result, dict):
        if result.get("error"):
            return f"cypher error: {str(result['error'])[:120]}"
        rows = result.get("rows") or []
        if not rows:
            return "cypher returned 0 rows"
        # Try to surface first-row keys + a preview
        keys = list(rows[0].keys())[:6]
        first = ", ".join(f"{k}={rows[0].get(k)}" for k in keys)
        return f"{len(rows)} rows (keys: {keys}); first: {first[:160]}"
    if tool_name == "related_incidents" and isinstance(result, dict):
        n = len(result.get("related") or [])
        anchor = (result.get("anchor") or {}).get("incidentId")
        groups = result.get("groups") or {}
        top_group = max(groups, key=lambda k: len(groups[k])) if groups else None
        extra = f", top group={top_group}" if top_group else ""
        return f"{n} related incidents from anchor {anchor}{extra}"
    if tool_name == "compare_entities" and isinstance(result, dict):
        L = (result.get("left") or {}).get("total", 0)
        R = (result.get("right") or {}).get("total", 0)
        shared = result.get("sharedRootCauses") or []
        return f"{(result.get('left') or {}).get('value')}={L} vs {(result.get('right') or {}).get('value')}={R}, shared rc={shared}"
    if tool_name == "shortest_path" and isinstance(result, dict):
        if result.get("found"):
            return f"path length={result.get('length')} ({len(result.get('steps') or [])} hops)"
        return "no path"
    if tool_name == "cooccurrence" and isinstance(result, dict):
        return (f"{len(result.get('communities') or [])} communities contain {result.get('anchor')}; "
                f"top co-services={list((result.get('coServices') or {}).keys())[:3]}")
    if isinstance(result, dict):
        incs = result.get("incidents") or []
        comms = result.get("communities") or []
        if incs or comms:
            return f"{len(incs)} incidents, {len(comms)} communities"
        if "nodes" in result and "edges" in result:
            return f"{len(result['nodes'])} nodes, {len(result['edges'])} edges"
        if result.get("incidentId"):
            return f"incident {result['incidentId']} (Sev{result.get('severity')} {result.get('service')}/{result.get('region')})"
    if isinstance(result, list):
        return f"{len(result)} rows"
    if result is None:
        return "no result"
    return str(result)[:120]
