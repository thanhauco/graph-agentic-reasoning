import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui";
import { Activity, AlertTriangle, Boxes, Network } from "lucide-react";

const SEV_COLORS = ["#b91c1c", "#dc2626", "#ea580c", "#d97706", "#2563eb"];

function Kpi({
  title,
  value,
  hint,
  icon: Icon,
}: {
  title: string;
  value: string | number;
  hint?: string;
  icon: React.ComponentType<{ className?: string }>;
}) {
  return (
    <Card>
      <CardContent className="flex items-center gap-4">
        <div className="grid h-11 w-11 place-items-center rounded-lg bg-primary/10 text-primary">
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            {title}
          </div>
          <div className="text-2xl font-semibold text-slate-900">{value}</div>
          {hint && <div className="text-[11px] text-muted-foreground">{hint}</div>}
        </div>
      </CardContent>
    </Card>
  );
}

export default function Dashboard() {
  const { data, isLoading } = useQuery({ queryKey: ["stats"], queryFn: api.stats });

  if (isLoading || !data) {
    return (
      <div className="p-8">
        <div className="text-sm text-muted-foreground">Loading stats…</div>
      </div>
    );
  }

  const sevData = Object.entries(data.bySeverity).map(([k, v]) => ({
    name: `Sev${k}`,
    value: v,
    color: SEV_COLORS[Number(k)] ?? "#64748b",
  }));
  const svcData = Object.entries(data.byService)
    .map(([name, value]) => ({ name, value }))
    .slice(0, 10);
  const monthData = Object.entries(data.byMonth).map(([month, value]) => ({ month, value }));
  const rcData = Object.entries(data.byRootCause).map(([name, value]) => ({ name, value }));
  const activeSev01 = (data.bySeverity["0"] ?? 0) + (data.bySeverity["1"] ?? 0);

  return (
    <div className="h-full overflow-auto bg-slate-50">
      <div className="p-8 space-y-6">
        <div>
          <h1 className="text-2xl font-semibold">IcM Operations Overview</h1>
          <p className="text-sm text-muted-foreground">
            Synthetic 2026 Azure incidents — ingested, graph-indexed, agent-ready.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
          <Kpi title="Total incidents" value={data.total} icon={Activity} hint="2026 dataset" />
          <Kpi title="High severity" value={activeSev01} icon={AlertTriangle} hint="Sev0 + Sev1" />
          <Kpi title="Communities" value={data.communities} icon={Network} hint="Louvain clusters" />
          <Kpi
            title="Services covered"
            value={Object.keys(data.byService).length}
            icon={Boxes}
            hint="Top-level Azure services"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Incidents per Month (2026)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={monthData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="month" stroke="#64748b" fontSize={12} />
                  <YAxis stroke="#64748b" fontSize={12} />
                  <Tooltip />
                  <Line type="monotone" dataKey="value" stroke="#2563eb" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Severity Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                  <Pie data={sevData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                    {sevData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Top Services by Incident Count</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={svcData} layout="vertical" margin={{ left: 28 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" stroke="#64748b" fontSize={12} />
                  <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={11} width={120} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#2563eb" radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Root Cause Breakdown</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={rcData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="name" stroke="#64748b" fontSize={11} angle={-20} textAnchor="end" height={60} />
                  <YAxis stroke="#64748b" fontSize={12} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#0ea5e9" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
