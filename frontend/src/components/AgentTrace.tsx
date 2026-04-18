import { useMemo } from "react";
import ReactFlow, {
  Background,
  Controls,
  Handle,
  Position,
  type Edge,
  type Node,
} from "reactflow";
import "reactflow/dist/style.css";
import type { AgentEvent } from "@/lib/api";

const STEP_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  session: { bg: "#eff6ff", border: "#2563eb", text: "#1e40af" },
  plan: { bg: "#faf5ff", border: "#a855f7", text: "#6b21a8" },
  tool_call: { bg: "#ecfeff", border: "#0ea5e9", text: "#075985" },
  tool_result: { bg: "#ecfdf5", border: "#10b981", text: "#065f46" },
  thought: { bg: "#fff7ed", border: "#f97316", text: "#9a3412" },
  final: { bg: "#f0fdf4", border: "#16a34a", text: "#14532d" },
};

function TraceNode({ data }: { data: any }) {
  const c = STEP_COLORS[data.kind] ?? STEP_COLORS.session;
  return (
    <div
      className="rounded-lg border px-3 py-2 shadow-sm max-w-[260px]"
      style={{ backgroundColor: c.bg, borderColor: c.border, color: c.text }}
    >
      <Handle type="target" position={Position.Left} />
      <div className="text-[10px] uppercase tracking-wider font-semibold">{data.kind}</div>
      <div className="text-xs font-medium truncate">{data.title}</div>
      {data.subtitle && <div className="text-[11px] opacity-80 truncate">{data.subtitle}</div>}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { trace: TraceNode };

export default function AgentTrace({ events }: { events: AgentEvent[] }) {
  const { nodes, edges } = useMemo(() => {
    const ns: Node[] = [];
    const es: Edge[] = [];
    let x = 0;
    const stepY: Record<number, number> = {};
    let lastId: string | null = null;
    events.forEach((ev, i) => {
      if (ev.type === "token") return;
      let title = "";
      let subtitle = "";
      let y = 0;
      const kind = ev.type;

      switch (ev.type) {
        case "session":
          title = "Question";
          subtitle = ev.question.slice(0, 80);
          break;
        case "plan":
          title = `Plan (${ev.mode})`;
          subtitle = ev.steps[0] ?? "";
          break;
        case "tool_call":
          title = `tool: ${ev.tool}`;
          subtitle = JSON.stringify(ev.args).slice(0, 70);
          stepY[ev.step] = 0;
          break;
        case "tool_result":
          title = `${ev.tool} → result`;
          subtitle = ev.summary;
          y = 70;
          break;
        case "thought":
          title = "Thought";
          subtitle = ev.thought.slice(0, 80);
          y = 140;
          break;
        case "final":
          title = "Final answer";
          subtitle = `${ev.citations.length} citations`;
          break;
      }

      const id = `n${i}`;
      ns.push({
        id,
        type: "trace",
        position: { x, y },
        data: { kind, title, subtitle },
      });
      if (lastId) {
        es.push({ id: `e${i}`, source: lastId, target: id, animated: ev.type === "tool_call" });
      }
      lastId = id;
      x += 280;
    });
    return { nodes: ns, edges: es };
  }, [events]);

  if (!events.length) {
    return (
      <div className="grid h-full place-items-center text-sm text-muted-foreground">
        Ask a question to see the agent reasoning trace here.
      </div>
    );
  }

  return (
    <div className="h-full w-full rounded-lg border border-slate-200 bg-white">
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView proOptions={{ hideAttribution: true }}>
        <Background gap={20} color="#e2e8f0" />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
