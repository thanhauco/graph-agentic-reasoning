import { useState, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send, Sparkles } from "lucide-react";
import AgentTrace from "@/components/AgentTrace";
import { streamChat, type AgentEvent } from "@/lib/api";
import { Badge, Button, Card, CardContent } from "@/components/ui";

const DEMO_QUERIES: { group: string; items: string[] }[] = [
  {
    group: "Lookup & filter",
    items: [
      "Tell me everything about INC-2026-0137.",
      "List all Sev1 Front Door certificate incidents in Q1 2026.",
    ],
  },
  {
    group: "Multi-hop similarity",
    items: [
      "Show incidents similar to INC-2026-0050 and explain the pattern.",
      "Find incidents related to INC-2026-0137 — what pattern do they share?",
    ],
  },
  {
    group: "Cross-entity reasoning",
    items: [
      "Compare Front Door vs API Management incidents this year.",
      "What services most often fail alongside Azure OpenAI?",
      "How are Cosmos DB and AKS connected in the knowledge graph?",
    ],
  },
  {
    group: "Cluster & executive",
    items: [
      "Was there an incident storm in March 2026? What was the shared root cause?",
      "Which regions had a cascade of Networking or DNS failures, and which teams own them?",
      "Give me an executive overview of the top Azure reliability themes across 2026 with incident citations.",
    ],
  },
];

type Message = {
  role: "user" | "assistant";
  text: string;
  events?: AgentEvent[];
  citations?: string[];
  streaming?: boolean;
};

export default function Chat() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const send = async (q: string) => {
    if (!q.trim() || busy) return;
    setBusy(true);
    setEvents([]);
    setMessages((m) => [
      ...m,
      { role: "user", text: q },
      { role: "assistant", text: "", streaming: true, events: [] },
    ]);
    setInput("");

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    let buffer = "";
    const localEvents: AgentEvent[] = [];
    try {
      await streamChat(
        q,
        (ev) => {
          localEvents.push(ev);
          setEvents([...localEvents]);
          if (ev.type === "token") {
            buffer += ev.text;
            setMessages((m) => {
              const copy = [...m];
              const last = copy[copy.length - 1];
              copy[copy.length - 1] = { ...last, text: buffer, events: [...localEvents] };
              return copy;
            });
          } else if (ev.type === "final") {
            setMessages((m) => {
              const copy = [...m];
              copy[copy.length - 1] = {
                role: "assistant",
                text: ev.answer || buffer,
                citations: ev.citations,
                events: [...localEvents],
                streaming: false,
              };
              return copy;
            });
          }
        },
        ctrl.signal,
      );
    } catch (e: any) {
      setMessages((m) => {
        const copy = [...m];
        const last = copy[copy.length - 1];
        copy[copy.length - 1] = {
          ...last,
          text: buffer || `⚠️ Error: ${e?.message ?? String(e)}`,
          streaming: false,
        };
        return copy;
      });
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  return (
    <div className="h-full grid grid-cols-[1.1fr_1fr] bg-slate-50">
      <div className="flex flex-col min-h-0 border-r border-slate-200 bg-white">
        <div className="px-6 py-4 border-b border-slate-200">
          <h1 className="text-xl font-semibold flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            Agentic Reasoning
          </h1>
          <p className="text-xs text-muted-foreground">
            Ask anything about 2026 Azure incidents — the agent plans, retrieves on the graph, and cites.
          </p>
        </div>

        <div className="flex-1 overflow-auto px-6 py-5 space-y-4 scrollbar-thin">
          {messages.length === 0 && (
            <div className="space-y-4">
              <div className="text-sm text-muted-foreground">
                10 demo queries showcasing multi-hop graph reasoning — click any to run:
              </div>
              {DEMO_QUERIES.map((section) => (
                <div key={section.group} className="space-y-1.5">
                  <div className="text-[11px] uppercase tracking-wider font-semibold text-slate-500">
                    {section.group}
                  </div>
                  <div className="grid grid-cols-1 gap-2">
                    {section.items.map((ex) => (
                      <button
                        key={ex}
                        onClick={() => send(ex)}
                        className="text-left rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm hover:bg-slate-50 hover:border-primary/40 transition-colors"
                      >
                        {ex}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={m.role === "user" ? "flex justify-end" : ""}>
              <Card className={m.role === "user" ? "max-w-[80%] bg-primary text-primary-foreground border-primary" : "max-w-[92%] w-full"}>
                <CardContent className={m.role === "user" ? "py-2 px-3 text-sm" : "prose prose-slate prose-sm max-w-none py-3"}>
                  {m.role === "user" ? (
                    m.text
                  ) : (
                    <>
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {m.text || (m.streaming ? "_thinking…_" : "")}
                      </ReactMarkdown>
                      {m.citations && m.citations.length > 0 && (
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {m.citations.map((c) => (
                            <Badge key={c} variant="outline" className="font-mono">
                              {c}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </CardContent>
              </Card>
            </div>
          ))}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
          className="border-t border-slate-200 p-4 flex gap-2"
        >
          <input
            className="flex-1 h-10 rounded-md border border-slate-200 bg-white px-3 text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            placeholder="Ask about a cluster, service, region, or timeframe…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={busy}
          />
          <Button type="submit" disabled={busy || !input.trim()}>
            <Send className="h-4 w-4" />
            {busy ? "Thinking…" : "Send"}
          </Button>
        </form>
      </div>

      <div className="flex flex-col min-h-0 p-4">
        <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2 px-2">
          Agent reasoning trace
        </div>
        <div className="flex-1 min-h-0">
          <AgentTrace events={events} />
        </div>
      </div>
    </div>
  );
}
