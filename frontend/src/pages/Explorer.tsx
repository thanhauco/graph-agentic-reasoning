import { useEffect, useMemo, useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
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

export default function Explorer() {
  const initial = loadState();
  const [service, setService] = useState(initial.service ?? "");
  const [region, setRegion] = useState(initial.region ?? "");
  const [selected, setSelected] = useState<string | null>(initial.selected ?? null);
  const [limit, setLimit] = useState(initial.limit ?? 100);
  const [search, setSearch] = useState(initial.search ?? "");
  const [searchOpen, setSearchOpen] = useState(false);

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
                  {(neighbors.edges ?? []).slice(0, 20).map((e, i) => (
                    <li key={i} className="flex gap-2">
                      <Badge variant="outline">{e.relation}</Badge>
                      <span className="font-mono text-[11px] text-slate-600 truncate">
                        {e.source === selected ? e.target : e.source}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
