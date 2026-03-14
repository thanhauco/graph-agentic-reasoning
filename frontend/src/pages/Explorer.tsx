import { useMemo, useState } from "react";
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

export default function Explorer() {
  const [service, setService] = useState("");
  const [region, setRegion] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [limit, setLimit] = useState(100);

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
              <div>
                <div className="font-mono text-xs">{selected}</div>
                <div className="text-xs text-muted-foreground">Entity node</div>
              </div>
            )}
            {neighbors && (
              <div className="pt-2 border-t border-slate-100">
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  Neighbors ({neighbors.edges.length})
                </div>
                <ul className="space-y-1 text-xs">
                  {neighbors.edges.slice(0, 20).map((e, i) => (
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
