export type IncidentRow = {
  incidentId: string;
  title: string;
  description?: string;
  severity: number;
  status: string;
  service: string;
  region: string;
  team: string;
  owner: string;
  createdAt: string;
  mitigatedAt?: string | null;
  resolvedAt?: string | null;
  rootCauseCategory: string;
  mitigation: string;
  impactedCustomers: number;
  tags?: string[];
  linkedIncidents?: string[];
  componentSignatures?: string[];
};

export type Stats = {
  total: number;
  byService: Record<string, number>;
  bySeverity: Record<string, number>;
  byRegion: Record<string, number>;
  byStatus: Record<string, number>;
  byRootCause: Record<string, number>;
  byMonth: Record<string, number>;
  communities: number;
};

export type GraphNode = {
  id: string;
  type: string;
  label: string;
  severity?: number;
  service?: string;
  region?: string;
  status?: string;
  title?: string;
};

export type GraphEdge = {
  source: string;
  target: string;
  relation: string;
};

const BASE = "/api";

async function j<T>(p: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${p}`, init);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} (${p})`);
  return r.json() as Promise<T>;
}

export const api = {
  stats: () => j<Stats>("/stats"),
  incidents: (params: Record<string, string | number | undefined> = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== "") qs.set(k, String(v));
    });
    return j<{ total: number; offset: number; limit: number; rows: IncidentRow[] }>(
      `/incidents?${qs.toString()}`,
    );
  },
  incident: (id: string) => j<IncidentRow>(`/incidents/${encodeURIComponent(id)}`),
  graph: (params: { limit_incidents?: number; service?: string; region?: string } = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== "") qs.set(k, String(v));
    });
    return j<{ nodes: GraphNode[]; edges: GraphEdge[]; incidents: number; totalIncidents?: number }>(
      `/graph?${qs.toString()}`,
    );
  },
  neighbors: (id: string, limit = 50) =>
    j<{ node: string; nodes: GraphNode[]; edges: GraphEdge[] }>(
      `/graph/neighbors/${encodeURIComponent(id)}?limit=${limit}`,
    ),
  graphQuery: (question: string, limit = 50) =>
    j<{
      question: string;
      intent: string;
      filters: Record<string, unknown>;
      matches: GraphNode[];
      matchIds: string[];
      anchorIds: string[];
      total: number;
    }>(`/graph/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, limit }),
    }),
  communities: (limit = 20) =>
    j<Array<{
      communityId: string;
      size: number;
      summary: string;
      incidentIds: string[];
      topServices?: string[];
      topRootCauses?: string[];
      topRegions?: string[];
    }>>(`/communities?limit=${limit}`),
};

export type AgentEvent =
  | { type: "session"; sessionId: string; t: number; question: string }
  | { type: "graph_query"; sessionId: string; t: number; query: Record<string, any>; describe: string }
  | { type: "plan"; sessionId: string; t: number; mode: string; steps: string[] }
  | { type: "tool_call"; sessionId: string; t: number; step: number; tool: string; args: any }
  | { type: "tool_result"; sessionId: string; t: number; step: number; tool: string; summary: string }
  | { type: "thought"; sessionId: string; t: number; step: number; thought: string }
  | { type: "token"; sessionId: string; t: number; text: string }
  | { type: "final"; sessionId: string; t: number; answer: string; citations: string[]; evidence: any[] };

export async function streamChat(
  question: string,
  onEvent: (ev: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!resp.body) throw new Error("No response body");
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const findFrameEnd = (s: string): { idx: number; len: number } => {
    // SSE frames may be separated by \n\n, \r\n\r\n, or \r\r.
    const candidates = [
      { sep: "\r\n\r\n", len: 4 },
      { sep: "\n\n", len: 2 },
      { sep: "\r\r", len: 2 },
    ];
    let best = { idx: -1, len: 0 };
    for (const { sep, len } of candidates) {
      const i = s.indexOf(sep);
      if (i !== -1 && (best.idx === -1 || i < best.idx)) best = { idx: i, len };
    }
    return best;
  };
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { idx, len } = findFrameEnd(buf);
      if (idx === -1) break;
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + len);
      const dataLines = block
        .split(/\r?\n/)
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).replace(/^ /, ""));
      if (!dataLines.length) continue;
      try {
        const parsed = JSON.parse(dataLines.join("\n")) as AgentEvent;
        onEvent(parsed);
      } catch {
        // ignore malformed frame
      }
    }
  }
}
