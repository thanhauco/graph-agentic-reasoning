import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type IncidentRow } from "@/lib/api";
import {
  Badge,
  Card,
  CardContent,
  Input,
  Select,
  SeverityBadge,
} from "@/components/ui";

const PAGE_SIZE = 50;

export default function Incidents() {
  const [q, setQ] = useState("");
  const [service, setService] = useState("");
  const [severity, setSeverity] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [offset, setOffset] = useState(0);

  const { data: stats } = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const { data, isLoading } = useQuery({
    queryKey: ["incidents", q, service, severity, status, offset],
    queryFn: () =>
      api.incidents({
        q: q || undefined,
        service: service || undefined,
        severity: severity !== "" ? Number(severity) : undefined,
        status: status || undefined,
        limit: PAGE_SIZE,
        offset,
      }),
  });

  const services = useMemo(
    () => (stats ? Object.keys(stats.byService) : []),
    [stats],
  );
  const statuses = useMemo(
    () => (stats ? Object.keys(stats.byStatus) : []),
    [stats],
  );

  const rows: IncidentRow[] = data?.rows ?? [];
  const total = data?.total ?? 0;

  return (
    <div className="h-full overflow-auto bg-slate-50">
      <div className="p-8 space-y-4">
        <div>
          <h1 className="text-2xl font-semibold">Incidents</h1>
          <p className="text-sm text-muted-foreground">
            {total.toLocaleString()} match{total === 1 ? "" : "es"} of 400 synthetic 2026 incidents.
          </p>
        </div>

        <Card>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
              <Input
                placeholder="Search by ID, title, description…"
                value={q}
                onChange={(e) => {
                  setQ(e.target.value);
                  setOffset(0);
                }}
                className="md:col-span-2"
              />
              <Select
                value={service}
                onChange={(e) => {
                  setService(e.target.value);
                  setOffset(0);
                }}
              >
                <option value="">All services</option>
                {services.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </Select>
              <Select
                value={severity}
                onChange={(e) => {
                  setSeverity(e.target.value);
                  setOffset(0);
                }}
              >
                <option value="">All severities</option>
                {[0, 1, 2, 3, 4].map((s) => (
                  <option key={s} value={s}>
                    Sev{s}
                  </option>
                ))}
              </Select>
              <Select
                value={status}
                onChange={(e) => {
                  setStatus(e.target.value);
                  setOffset(0);
                }}
              >
                <option value="">All statuses</option>
                {statuses.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </Select>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2">ID</th>
                    <th className="px-4 py-2">Sev</th>
                    <th className="px-4 py-2">Status</th>
                    <th className="px-4 py-2">Service</th>
                    <th className="px-4 py-2">Region</th>
                    <th className="px-4 py-2">Title</th>
                    <th className="px-4 py-2">Root cause</th>
                    <th className="px-4 py-2 text-right">Impacted</th>
                    <th className="px-4 py-2">Created</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {isLoading && (
                    <tr>
                      <td colSpan={9} className="px-4 py-6 text-center text-muted-foreground">
                        Loading…
                      </td>
                    </tr>
                  )}
                  {!isLoading && rows.length === 0 && (
                    <tr>
                      <td colSpan={9} className="px-4 py-6 text-center text-muted-foreground">
                        No incidents match your filters.
                      </td>
                    </tr>
                  )}
                  {rows.map((r) => (
                    <tr key={r.incidentId} className="hover:bg-slate-50">
                      <td className="px-4 py-2 font-mono text-xs text-primary">{r.incidentId}</td>
                      <td className="px-4 py-2">
                        <SeverityBadge sev={r.severity} />
                      </td>
                      <td className="px-4 py-2">
                        <Badge variant="outline">{r.status}</Badge>
                      </td>
                      <td className="px-4 py-2">{r.service}</td>
                      <td className="px-4 py-2 text-xs text-slate-600">{r.region}</td>
                      <td className="px-4 py-2 max-w-[360px] truncate" title={r.title}>
                        {r.title}
                      </td>
                      <td className="px-4 py-2 text-xs">{r.rootCauseCategory}</td>
                      <td className="px-4 py-2 text-right tabular-nums">
                        {r.impactedCustomers.toLocaleString()}
                      </td>
                      <td className="px-4 py-2 text-xs text-slate-500">
                        {r.createdAt.slice(0, 10)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex items-center justify-between border-t border-slate-100 px-4 py-3 text-xs text-muted-foreground">
              <div>
                Showing {rows.length} of {total.toLocaleString()} matches
              </div>
              <div className="flex gap-2">
                <button
                  disabled={offset === 0}
                  onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                  className="rounded-md border border-slate-200 px-3 py-1 disabled:opacity-50"
                >
                  Prev
                </button>
                <button
                  disabled={offset + PAGE_SIZE >= total}
                  onClick={() => setOffset(offset + PAGE_SIZE)}
                  className="rounded-md border border-slate-200 px-3 py-1 disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
