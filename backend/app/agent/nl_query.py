"""Natural-language → structured graph query translator.

Parses a user question against the current KG vocabulary (services, regions,
teams, root causes) and returns a `GraphQuery` describing:
  - intent: lookup | list | summarize | cluster | trend
  - filters: service, region, team, severity, status, rootCause, incidentId
  - time window: startIso, endIso (ISO-8601)
  - keywords: remaining tokens for embedding-ranking

This gives the heuristic (no-LLM) path real understanding of the query so the
agent's tool call and answer differ per question, not by template.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app.state import AppState


# ---------- vocabulary ----------

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_CLUSTER_WORDS = ("cluster", "storm", "cascade", "outage", "wave", "incident surge", "surge")
_SUMMARY_WORDS = ("summarize", "summary", "overall", "trend", "trends", "themes", "overview", "executive")
_LIST_WORDS = ("list", "show", "which", "what incidents", "top", "most", "all")
_LOOKUP_WORDS = ("details", "detail", "look up", "pull", "fetch", "tell me about", "what happened in")
_MULTIHOP_WORDS = (
    "related to", "similar to", "like this", "like inc", "recurring", "repeat",
    "why does this keep", "root cause pattern", "pattern", "linked to",
)
_COMPARE_WORDS = (" vs ", "versus", "compare", "compared to", "difference between", "vs.")
_PATH_WORDS = (
    "how are", "connected", "connection between", "path between", "linked between",
    "relationship between", "related via",
)
_COOCCUR_WORDS = (
    "fails alongside", "fail alongside", "fails with", "fail together", "fail with",
    "co-occur", "cooccur", "alongside",
    "along with", "together with", "depends on", "dependencies of", "dependency of",
    "most often fail", "often fail",
)

_SEV_RE = re.compile(r"\bsev(?:erity)?\s*[- ]?\s*([0-4])\b", re.IGNORECASE)
_INC_RE = re.compile(r"\bINC-2026-\d{4}\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(2026)\b")


@dataclass
class GraphQuery:
    question: str
    intent: str = "list"  # lookup | list | summarize | cluster | trend
    incidentId: str | None = None
    service: str | None = None
    region: str | None = None
    team: str | None = None
    rootCause: str | None = None
    status: str | None = None
    severity: int | None = None
    startIso: str | None = None
    endIso: str | None = None
    keywords: list[str] = field(default_factory=list)
    # For multi-hop / compare / path intents:
    anchorId: str | None = None
    compareLeft: str | None = None
    compareRight: str | None = None
    pathSrc: str | None = None
    pathDst: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, [], "")}

    def describe(self) -> str:
        parts: list[str] = [f"intent={self.intent}"]
        for k in ("incidentId", "service", "region", "team", "rootCause", "status", "severity",
                  "anchorId", "compareLeft", "compareRight", "pathSrc", "pathDst"):
            v = getattr(self, k)
            if v is not None:
                parts.append(f"{k}={v}")
        if self.startIso or self.endIso:
            parts.append(f"time={self.startIso or '*'}..{self.endIso or '*'}")
        if self.keywords:
            parts.append("kw=" + ",".join(self.keywords))
        return " ".join(parts)


# ---------- vocab from the KG ----------

def _collect_vocab(store: AppState) -> dict[str, list[str]]:
    services, regions, teams, causes, statuses = set(), set(), set(), set(), set()
    for _, d in store.graph.nodes(data=True):
        t = d.get("type")
        label = d.get("label")
        if not label:
            continue
        if t == "Service":
            services.add(label)
        elif t == "Region":
            regions.add(label)
        elif t == "Team":
            teams.add(label)
        elif t == "RootCauseCategory":
            causes.add(label)
    for inc in store.incidents:
        if inc.get("status"):
            statuses.add(inc["status"])
    return {
        "services": sorted(services, key=len, reverse=True),
        "regions": sorted(regions, key=len, reverse=True),
        "teams": sorted(teams, key=len, reverse=True),
        "causes": sorted(causes, key=len, reverse=True),
        "statuses": sorted(statuses),
    }


_SERVICE_ALIASES = {
    "azure openai": "Azure OpenAI",
    "aoai": "Azure OpenAI",
    "apim": "API Management",
    "api management": "API Management",
    "front door": "Front Door",
    "afd": "Front Door",
    "cosmos": "Cosmos DB",
    "cosmos db": "Cosmos DB",
    "sql": "SQL DB",
    "sql db": "SQL DB",
    "entra": "Entra ID",
    "entra id": "Entra ID",
    "aad": "Entra ID",
    "aks": "AKS",
    "kubernetes": "AKS",
    "functions": "Functions",
    "azure functions": "Functions",
    "app service": "App Service",
    "storage": "Storage",
    "monitor": "Monitor",
    "networking": "Networking",
}


def _find_alias(ql: str, aliases: dict[str, str], vocab: list[str]) -> str | None:
    # Longest alias wins to avoid partial matches.
    for alias in sorted(aliases.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", ql):
            return aliases[alias]
    for v in vocab:
        if re.search(rf"\b{re.escape(v.lower())}\b", ql):
            return v
    return None


def _find_vocab(ql: str, vocab: list[str]) -> str | None:
    for v in vocab:
        if re.search(rf"\b{re.escape(v.lower())}\b", ql):
            return v
    return None


def _parse_time(ql: str) -> tuple[str | None, str | None]:
    """Extract a time window as ISO strings. Defaults to 2026."""
    # Explicit month + 2026: "March 2026", "in March", "Mar 2026"
    month_hit = None
    year = 2026
    for name, num in _MONTHS.items():
        if re.search(rf"\b{name}\b", ql):
            month_hit = num
            break
    ym = re.search(r"\b(20\d{2})\b", ql)
    if ym:
        year = int(ym.group(1))
    if month_hit is not None:
        start = datetime(year, month_hit, 1)
        # end = first of next month
        end = datetime(year + (month_hit // 12), (month_hit % 12) + 1, 1)
        return start.isoformat() + "Z", end.isoformat() + "Z"
    # Quarter handling: "Q1 2026", "first quarter"
    qmatch = re.search(r"\bq([1-4])\b", ql)
    if qmatch:
        q = int(qmatch.group(1))
        sm = (q - 1) * 3 + 1
        start = datetime(year, sm, 1)
        end_month = sm + 3
        end_year = year + (end_month // 13)
        end_month = end_month if end_month <= 12 else end_month - 12
        end = datetime(end_year, end_month, 1)
        return start.isoformat() + "Z", end.isoformat() + "Z"
    # "last week", "last month" (relative to today)
    if "last week" in ql:
        now = datetime.utcnow()
        return (now - timedelta(days=7)).isoformat() + "Z", now.isoformat() + "Z"
    if "last month" in ql:
        now = datetime.utcnow()
        return (now - timedelta(days=30)).isoformat() + "Z", now.isoformat() + "Z"
    if "this year" in ql or re.search(r"\b2026\b", ql):
        return "2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z"
    return None, None


def _detect_intent(ql: str, has_id: bool) -> str:
    if any(w in ql for w in _COMPARE_WORDS):
        return "compare"
    if any(w in ql for w in _PATH_WORDS):
        return "path"
    if any(w in ql for w in _COOCCUR_WORDS):
        return "cooccur"
    if any(w in ql for w in _MULTIHOP_WORDS):
        return "multi_hop"
    if any(w in ql for w in _CLUSTER_WORDS):
        return "cluster"
    if any(w in ql for w in _SUMMARY_WORDS):
        return "summarize"
    if has_id:
        return "lookup"
    if any(w in ql for w in _LOOKUP_WORDS):
        return "lookup"
    if any(w in ql for w in _LIST_WORDS):
        return "list"
    return "list"


# ---------- public ----------

def parse_query(question: str, store: AppState) -> GraphQuery:
    ql = question.lower().strip()
    vocab = _collect_vocab(store)

    inc_match = _INC_RE.search(question)
    incident_id = inc_match.group(0).upper() if inc_match else None

    sev_match = _SEV_RE.search(ql)
    severity = int(sev_match.group(1)) if sev_match else None

    service = _find_alias(ql, _SERVICE_ALIASES, [s.lower() for s in vocab["services"]])
    # _find_alias may return alias value or vocab match; normalize to actual vocab entry:
    if service and service.lower() not in [s.lower() for s in vocab["services"]]:
        # Alias already maps to canonical
        pass

    region = _find_vocab(ql, [r.lower() for r in vocab["regions"]])
    if region:
        # Restore exact casing from vocab
        for r in vocab["regions"]:
            if r.lower() == region:
                region = r
                break

    team = _find_vocab(ql, [t.lower() for t in vocab["teams"]])
    if team:
        for t in vocab["teams"]:
            if t.lower() == team:
                team = t
                break

    cause = _find_vocab(ql, [c.lower() for c in vocab["causes"]])
    if cause:
        for c in vocab["causes"]:
            if c.lower() == cause:
                cause = c
                break
    # Extra cause synonyms:
    if cause is None:
        cause_alias = {
            "capacity": "Capacity", "throttle": "Capacity", "throttling": "Capacity",
            "cert": "Certificate", "certificate": "Certificate", "tls": "Certificate", "ssl": "Certificate",
            "dns": "DNS", "network": "Networking", "networking": "Networking",
            "deploy": "Deployment", "deployment": "Deployment", "rollout": "Deployment",
            "config": "Config", "configuration": "Config",
            "bug": "Code Defect", "defect": "Code Defect", "regression": "Code Defect",
            "security": "Security", "auth": "Security", "attack": "Security",
            "dependency": "Dependency", "upstream": "Dependency",
        }
        for k, v in cause_alias.items():
            if re.search(rf"\b{k}\b", ql):
                cause = v
                break

    status = None
    for s in vocab["statuses"]:
        if re.search(rf"\b{re.escape(s.lower())}\b", ql):
            status = s
            break

    start, end = _parse_time(ql)
    intent = _detect_intent(ql, has_id=bool(incident_id))

    # Remaining keywords: strip known tokens.
    stripped = ql
    for tok in [incident_id, service, region, team, cause, status]:
        if tok:
            stripped = stripped.replace(tok.lower(), " ")
    stripped = _SEV_RE.sub(" ", stripped)
    stripped = _YEAR_RE.sub(" ", stripped)
    for m in _MONTHS:
        stripped = re.sub(rf"\b{m}\b", " ", stripped)
    keywords = [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", stripped)
                if w not in _STOPWORDS][:8]

    # ---- multi-hop / compare / path extraction ----
    anchor_id = incident_id if intent == "multi_hop" else None
    compare_left = compare_right = path_src = path_dst = None

    all_named = vocab["services"] + vocab["regions"] + vocab["teams"] + vocab["causes"]

    def _find_all_named(text: str) -> list[str]:
        found: list[tuple[int, str]] = []
        for v in all_named:
            m = re.search(rf"\b{re.escape(v.lower())}\b", text)
            if m:
                found.append((m.start(), v))
        # Also honor service aliases.
        for alias, canonical in _SERVICE_ALIASES.items():
            m = re.search(rf"\b{re.escape(alias)}\b", text)
            if m:
                found.append((m.start(), canonical))
        # Dedup preserving earliest position.
        seen: dict[str, int] = {}
        for pos, v in found:
            seen.setdefault(v, pos)
        return sorted(seen, key=seen.get)  # type: ignore[arg-type]

    if intent == "compare":
        names = _find_all_named(ql)
        if len(names) >= 2:
            compare_left, compare_right = names[0], names[1]
    elif intent == "path":
        names = _find_all_named(ql)
        if len(names) >= 2:
            path_src, path_dst = names[0], names[1]

    return GraphQuery(
        question=question,
        intent=intent,
        incidentId=incident_id,
        service=service,
        region=region,
        team=team,
        rootCause=cause,
        status=status,
        severity=severity,
        startIso=start,
        endIso=end,
        keywords=keywords,
        anchorId=anchor_id,
        compareLeft=compare_left,
        compareRight=compare_right,
        pathSrc=path_src,
        pathDst=path_dst,
    )


_STOPWORDS = {
    "the", "what", "who", "how", "why", "which", "when", "where", "show", "list", "give",
    "me", "tell", "about", "please", "with", "and", "for", "that", "this", "these", "those",
    "any", "all", "some", "most", "top", "summary", "summarize", "overall", "trend", "trends",
    "themes", "overview", "executive", "pull", "fetch", "details", "detail", "incident",
    "incidents", "azure", "sev", "severity", "year", "month", "quarter", "happened",
    "cluster", "storm", "cascade", "outage", "wave", "issue", "issues", "owning",
    "involved", "involving", "team", "owns", "region", "service", "related", "relating",
}


def to_filters(gq: GraphQuery) -> dict[str, Any]:
    """Subset of GraphQuery to pass as structured tool args."""
    f: dict[str, Any] = {}
    for k in ("service", "region", "team", "rootCause", "status"):
        v = getattr(gq, k)
        if v:
            f[k] = v
    if gq.severity is not None:
        f["severity"] = gq.severity
    if gq.startIso:
        f["start"] = gq.startIso
    if gq.endIso:
        f["end"] = gq.endIso
    return f
