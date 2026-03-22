import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, keepPreviousData } from "@tanstack/react-query";
import { api } from "@/lib/api";
import KnowledgeGraph from "@/components/KnowledgeGraph";
import {
  Badge,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Select,
  SeverityBadge,
} from "@/components/ui";
import { Sparkles } from "lucide-react";

const STATE_KEY = "explorer.state.v1";

type PersistedState = {
  service: string;
  region: string;
  selected: string | null;
  limit: number;
  search: string;
};

function loadState(): Partial<PersistedState> {
  try {
    const raw = sessionStorage.getItem(STATE_KEY);
    return raw ? (JSON.parse(raw) as PersistedState) : {};
  } catch {
    return {};
  }
}

// Render a grounded answer: support **bold**, preserve newlines, and turn
// any INC-YYYY-#### occurrence into a clickable link that selects the node.
function renderAnswer(text: string, onSelect: (id: string) => void) {
  const tokenRe = /(\*\*[^*]+\*\*|INC-\d{4}-\d{3,5})/g;
  const parts = text.split(tokenRe);
  return parts.map((p, i) => {
    if (!p) return null;
    if (/^\*\*[^*]+\*\*$/.test(p)) {
      return <strong key={i} className="font-semibold">{p.slice(2, -2)}</strong>;
    }
    if (/^INC-\d{4}-\d{3,5}$/.test(p)) {
      return (
        <button
          key={i}
          type="button"
          onClick={() => onSelect(p)}
          className="font-mono text-primary hover:underline"
        >
          {p}
        </button>
      );
    }
    return <span key={i}>{p}</span>;
  });
}

export default function Explorer() {
  const initial = loadState();
  const [service, setService] = useState(initial.service ?? "");
  const [region, setRegion] = useState(initial.region ?? "");
  const [selected, setSelected] = useState<string | null>(initial.selected ?? null);
  const [limit, setLimit] = useState(initial.limit ?? 100);
  const [search, setSearch] = useState(initial.search ?? "");
  const [searchOpen, setSearchOpen] = useState(false);
  const [nlQuery, setNlQuery] = useState("");
  const [queryHighlight, setQueryHighlight] = useState<string[]>([]);

  type ChatTurn = {
    question: string;
    answer: string;
    intent: string;
    total: number;
    matchIds: string[];
    anchorIds: string[];
    cypherSource: "llm" | "heuristic";
    cypherSteps: Array<{ label: string; cypher: string; params: Record<string, unknown> }>;
    explanation: string;
  };
  const [conversation, setConversation] = useState<ChatTurn[]>([]);

  const runNlQuery = useMutation({
    mutationFn: (q: string) =>
      api.graphQuery(
        q,
        50,
        selected,
        // Send only the compact fields the backend uses for pronoun
        // resolution and grounding. Trim to last 4 turns to keep prompts
        // small.
        conversation.slice(-4).map((t) => ({
          question: t.question,
          answer: t.answer,
          matchIds: t.matchIds,
          anchorIds: t.anchorIds,
        })),
      ),
    onSuccess: (res, question) => {
      const ids = [...(res.matchIds ?? []), ...(res.anchorIds ?? [])];
      setQueryHighlight(ids);
      if (res.matchIds && res.matchIds.length > 0) {
        setSelected(res.matchIds[0]);
      }
      setConversation((prev) => [
        ...prev,
        {
          question,
          answer: res.answer,
          intent: res.intent,
          total: res.total,
          matchIds: res.matchIds ?? [],
          anchorIds: res.anchorIds ?? [],
          cypherSource: res.cypherSource,
          cypherSteps:
            res.cypherSteps && res.cypherSteps.length > 0
              ? res.cypherSteps
              : [{ label: "main", cypher: res.cypher, params: res.cypherParams }],
          explanation: res.explanation,
        },
      ]);
      setNlQuery("");
    },
  });

  useEffect(() => {
    try {
      sessionStorage.setItem(
        STATE_KEY,
        JSON.stringify({ service, region, selected, limit, search }),
      );
    } catch {
      /* ignore quota / disabled storage */
    }
  }, [service, region, selected, limit, search]);

  const { data: stats } = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const { data: graph, isLoading, isFetching } = useQuery({
    queryKey: ["graph", service, region, limit],
    queryFn: () => api.graph({ service: service || undefined, region: region || undefined, limit_incidents: limit }),
    placeholderData: keepPreviousData,
  });
  const { data: neighbors } = useQuery({
    queryKey: ["neighbors", selected],
    queryFn: () => (selected ? api.neighbors(selected, 25) : Promise.resolve(null)),
    enabled: !!selected,
  });
  const { data: incident } = useQuery({
    queryKey: ["incident-detail", selected],
    queryFn: () => (selected && selected.startsWith("INC-") ? api.incident(selected) : Promise.resolve(null)),
    enabled: !!selected && selected.startsWith("INC-"),
  });

  const services = useMemo(() => (stats ? Object.keys(stats.byService) : []), [stats]);
  const regions = useMemo(() => (stats ? Object.keys(stats.byRegion) : []), [stats]);

  const searchMatches = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q || !graph) return [];
    const out: { id: string; label: string; type: string; sub?: string }[] = [];
    for (const n of graph.nodes) {
      const hay = [n.id, n.label, n.title, n.service, n.region]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      if (hay.includes(q)) {
        out.push({
          id: n.id,
          label: n.label ?? n.id,
          type: n.type,
          sub: [n.service, n.region].filter(Boolean).join(" · ") || n.title,
        });
        if (out.length >= 15) break;
      }
    }
    return out;
  }, [search, graph]);

  const pickMatch = (id: string) => {
    setSelected(id);
    setSearch("");
    setSearchOpen(false);
  };

  const selectedNode = useMemo(
    () => (selected && graph ? graph.nodes.find((n) => n.id === selected) ?? null : null),
    [selected, graph],
  );

  const entitySummary = useMemo(() => {
    if (!selected || !neighbors) return null;
    const nbNodes = neighbors.nodes ?? [];
    const nbEdges = neighbors.edges ?? [];
    const byRel: Record<string, number> = {};
    const byType: Record<string, number> = {};
    const incidents: string[] = [];
    const sevCount: Record<number, number> = {};
    const nodeMap = new Map(nbNodes.map((n) => [n.id, n]));
    for (const e of nbEdges) {
      byRel[e.relation] = (byRel[e.relation] ?? 0) + 1;
      const otherId = e.source === selected ? e.target : e.source;
      const other = nodeMap.get(otherId);
      if (other) {
        byType[other.type] = (byType[other.type] ?? 0) + 1;
        if (other.type === "Incident") {
          incidents.push(other.id);
          if (typeof other.severity === "number") {
            sevCount[other.severity] = (sevCount[other.severity] ?? 0) + 1;
          }
        }
      }
    }
    return {
      total: nbEdges.length,
      relations: Object.entries(byRel).sort((a, b) => b[1] - a[1]),
      types: Object.entries(byType).sort((a, b) => b[1] - a[1]),
      incidents,
      sevCount,
    };
  }, [selected, neighbors]);

  return (
    <div className="h-full flex flex-col bg-slate-50">
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-white">
        <div>
          <h1 className="text-xl font-semibold">Knowledge Graph Explorer</h1>
          <p className="text-xs text-muted-foreground">
            Incidents connected via services, regions, teams, root causes, and mitigations.
          </p>
        </div>
        <div className="flex gap-2">
          <div className="relative w-64">
            <input
              type="text"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setSearchOpen(true);
              }}
              onFocus={() => setSearchOpen(true)}
              onBlur={() => setTimeout(() => setSearchOpen(false), 150)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && searchMatches[0]) pickMatch(searchMatches[0].id);
                if (e.key === "Escape") {
                  setSearch("");
                  setSearchOpen(false);
                }
              }}
              placeholder="Search nodes (id, title, service…)"
              className="w-full h-9 rounded-md border border-slate-200 bg-white px-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
            />
            {searchOpen && search.trim() && (
              <div className="absolute top-full left-0 right-0 mt-1 max-h-80 overflow-auto rounded-md border border-slate-200 bg-white shadow-lg z-20">
                {searchMatches.length === 0 ? (
                  <div className="px-3 py-2 text-xs text-muted-foreground">
                    No matches in current graph
                  </div>
                ) : (
                  searchMatches.map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      onMouseDown={(e) => {
                        e.preventDefault();
                        pickMatch(m.id);
                      }}
                      className="flex w-full items-center gap-2 px-3 py-1.5 text-left hover:bg-slate-50"
                    >
                      <Badge variant="outline" className="shrink-0 text-[10px]">
                        {m.type}
                      </Badge>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-xs font-medium text-slate-900">
                          {m.label}
                        </div>
                        {m.sub && (
                          <div className="truncate text-[10px] text-muted-foreground">
                            {m.sub}
                          </div>
                        )}
                      </div>
                      <span className="shrink-0 font-mono text-[10px] text-slate-400">
                        {m.id}
                      </span>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>
          <Select value={service} onChange={(e) => setService(e.target.value)} className="w-48">
            <option value="">All services</option>
            {services.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
          <Select value={region} onChange={(e) => setRegion(e.target.value)} className="w-48">
            <option value="">All regions</option>
            {regions.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </Select>
        </div>
      </div>

      <div className="flex items-center gap-3 px-6 py-2 border-b border-slate-200 bg-white">
        <span className="text-xs font-medium text-slate-700 whitespace-nowrap">Incidents shown</span>
        <input
          type="range"
          min={50}
          max={500}
          step={50}
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="flex-1 max-w-xl accent-primary"
        />
        <span className="text-xs font-mono text-slate-700 w-28 text-right">
          {graph?.incidents ?? limit} / {graph?.totalIncidents ?? "…"}
          {isFetching && <span className="ml-2 text-muted-foreground">loading…</span>}
        </span>
      </div>

      <div className="flex items-center gap-2 px-6 py-2 border-b border-slate-200 bg-white">
        <Sparkles className="h-4 w-4 text-primary shrink-0" />
        <input
          type="text"
          value={nlQuery}
          onChange={(e) => setNlQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && nlQuery.trim()) {
              runNlQuery.mutate(nlQuery.trim());
            }
            if (e.key === "Escape") {
              setNlQuery("");
            }
          }}
          placeholder={
            conversation.length > 0
              ? "Follow up — e.g. 'only sev1', 'similar to this', 'by Storage team'"
              : "Ask the graph in natural language — e.g. 'sev1 Front Door incidents in westus2'"
          }
          className="flex-1 h-9 rounded-md border border-slate-200 bg-white px-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
        />
        <button
          type="button"
          onClick={() => nlQuery.trim() && runNlQuery.mutate(nlQuery.trim())}
          disabled={!nlQuery.trim() || runNlQuery.isPending}
          className="h-9 rounded-md bg-primary px-3 text-xs font-semibold text-primary-foreground shadow-sm hover:bg-primary/90 disabled:opacity-50"
        >
          {runNlQuery.isPending ? "Querying…" : "Query"}
        </button>
        {(queryHighlight.length > 0 || conversation.length > 0) && (
          <button
            type="button"
            onClick={() => {
              setQueryHighlight([]);
              setConversation([]);
              runNlQuery.reset();
            }}
            className="h-9 rounded-md border border-slate-200 bg-white px-3 text-xs font-medium text-slate-600 hover:bg-slate-50"
          >
            New chat
          </button>
        )}
        {conversation.length > 0 && (
          <span className="text-xs text-muted-foreground whitespace-nowrap">
            {conversation.length} turn{conversation.length === 1 ? "" : "s"}
            {runNlQuery.data && ` · ${runNlQuery.data.intent}`}
          </span>
        )}
        {runNlQuery.isError && (
          <span className="text-xs text-red-600">Query failed</span>
        )}
      </div>

      {conversation.length > 0 && (
        <div className="px-6 py-3 border-b border-slate-200 bg-gradient-to-r from-primary/5 via-white to-white max-h-72 overflow-auto space-y-3">
          {conversation.map((turn, idx) => {
            const isLast = idx === conversation.length - 1;
            return (
              <div key={idx} className="space-y-1.5">
                <div className="flex items-start gap-2">
                  <span className="mt-0.5 inline-flex h-5 items-center rounded-full bg-slate-200 px-2 text-[10px] font-semibold uppercase tracking-wide text-slate-600">
                    You
                  </span>
                  <p className="text-sm text-slate-700">{turn.question}</p>
                </div>
                <div className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 mt-0.5 text-primary shrink-0" />
                  <div className="flex-1 min-w-0 space-y-1">
                    <div className="text-sm leading-relaxed text-slate-800 whitespace-pre-line">
                      {renderAnswer(turn.answer, (id) => setSelected(id))}
                    </div>
                    {isLast && (
                      <details className="group">
                        <summary className="cursor-pointer text-[11px] font-semibold uppercase tracking-wide text-slate-500 hover:text-slate-700">
                          Generated Cypher ({turn.cypherSource}
                          {turn.cypherSteps.length > 1 ? ` · ${turn.cypherSteps.length} steps` : ""})
                        </summary>
                        {turn.cypherSteps.map((step, i) => (
                          <div key={i} className="mt-2">
                            <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                              Step {i + 1}: {step.label}
                            </div>
                            <pre className="mt-0.5 overflow-auto rounded-md bg-slate-900 p-2 text-[11px] leading-relaxed text-slate-100 max-h-48">
                              {step.cypher}
                            </pre>
                            {Object.keys(step.params ?? {}).length > 0 && (
                              <pre className="mt-0.5 overflow-auto rounded-md bg-slate-100 p-2 text-[11px] leading-relaxed text-slate-700 max-h-28">
                                {JSON.stringify(step.params, null, 2)}
                              </pre>
                            )}
                          </div>
                        ))}
                        {turn.explanation && (
                          <p className="mt-2 text-[11px] italic text-slate-500">{turn.explanation}</p>
                        )}
                      </details>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
          {runNlQuery.isPending && (
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Sparkles className="h-3 w-3 animate-pulse text-primary" />
              Thinking…
            </div>
          )}
        </div>
      )}

      <div className="flex-1 grid grid-cols-[1fr_360px] gap-4 p-4 min-h-0">
        <div className="min-h-0">
          {isLoading || !graph ? (
            <div className="grid h-full place-items-center text-sm text-muted-foreground">
              Loading graph…
            </div>
          ) : (
            <KnowledgeGraph
              nodes={graph.nodes}
              edges={graph.edges}
              selected={selected}
              onSelect={setSelected}
              highlightIds={queryHighlight}
            />
          )}
        </div>
        <Card className="min-h-0 flex flex-col">
          <CardHeader>
            <CardTitle>Node Details</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-auto space-y-3 text-sm">
            {!selected && (
              <div className="text-muted-foreground">
                Click a node to inspect its incident details and local neighborhood.
              </div>
            )}
            {selected && incident && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <SeverityBadge sev={incident.severity} />
                  <Badge variant="outline">{incident.status}</Badge>
                  <span className="font-mono text-xs text-primary">{incident.incidentId}</span>
                </div>
                <div className="text-sm font-semibold text-slate-900">{incident.title}</div>
                <div className="text-xs text-muted-foreground">
                  {incident.service} · {incident.region} · {incident.team}
                </div>
                <p className="text-sm leading-relaxed text-slate-700">{incident.description}</p>
                <div className="text-xs">
                  <div>
                    <span className="font-semibold">Root cause:</span> {incident.rootCauseCategory}
                  </div>
                  <div>
                    <span className="font-semibold">Mitigation:</span> {incident.mitigation}
                  </div>
                  <div>
                    <span className="font-semibold">Impacted:</span>{" "}
                    {incident.impactedCustomers.toLocaleString()}
                  </div>
                </div>
              </div>
            )}
            {selected && !incident && (
              <div className="space-y-3">
                <div className="flex items-center gap-2 flex-wrap">
                  <Badge variant="outline">{selectedNode?.type ?? "Entity"}</Badge>
                  <span className="font-mono text-[11px] text-slate-500">{selected}</span>
                </div>
                {selectedNode?.label && selectedNode.label !== selected && (
                  <div className="text-sm font-semibold text-slate-900">
                    {selectedNode.label}
                  </div>
                )}
                {(selectedNode?.service || selectedNode?.region) && (
                  <div className="text-xs text-muted-foreground">
                    {[selectedNode.service, selectedNode.region].filter(Boolean).join(" · ")}
                  </div>
                )}
                {entitySummary && (
                  <div className="space-y-2 rounded-md bg-slate-50 p-2">
                    <div className="text-xs leading-relaxed text-slate-700">
                      Connected to{" "}
                      <span className="font-semibold">{entitySummary.total}</span> node
                      {entitySummary.total === 1 ? "" : "s"}
                      {entitySummary.incidents.length > 0 && (
                        <>
                          {" "}including{" "}
                          <span className="font-semibold">
                            {entitySummary.incidents.length}
                          </span>{" "}
                          incident
                          {entitySummary.incidents.length === 1 ? "" : "s"}
                        </>
                      )}
                      .
                    </div>
                    {entitySummary.types.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {entitySummary.types.map(([t, c]) => (
                          <Badge key={t} variant="outline" className="text-[10px]">
                            {t} · {c}
                          </Badge>
                        ))}
                      </div>
                    )}
                    {Object.keys(entitySummary.sevCount).length > 0 && (
                      <div className="flex flex-wrap items-center gap-1">
                        {Object.entries(entitySummary.sevCount)
                          .sort((a, b) => Number(a[0]) - Number(b[0]))
                          .map(([s, c]) => (
                            <span key={s} className="flex items-center gap-1">
                              <SeverityBadge sev={Number(s)} />
                              <span className="text-[10px] text-slate-600">×{c}</span>
                            </span>
                          ))}
                      </div>
                    )}
                    {entitySummary.relations.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {entitySummary.relations.slice(0, 6).map(([r, c]) => (
                          <Badge key={r} className="text-[10px]">
                            {r} · {c}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            {neighbors && (
              <div className="pt-2 border-t border-slate-100">
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  Neighbors ({neighbors.edges?.length ?? 0})
                </div>
                <ul className="space-y-1 text-xs">
                  {(neighbors.edges ?? []).slice(0, 20).map((e, i) => {
                    const otherId = e.source === selected ? e.target : e.source;
                    return (
                      <li key={i}>
                        <button
                          type="button"
                          onClick={() => setSelected(otherId)}
                          className="flex w-full items-center gap-2 rounded px-1 py-1 text-left hover:bg-slate-50"
                          title={`Go to ${otherId}`}
                        >
                          <Badge variant="outline" className="shrink-0">
                            {e.relation}
                          </Badge>
                          <span className="font-mono text-[11px] text-primary hover:underline truncate">
                            {otherId}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
