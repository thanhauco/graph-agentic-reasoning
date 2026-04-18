import { useEffect, useMemo, useRef } from "react";
import cytoscape, { type Core, type ElementDefinition } from "cytoscape";
import fcose from "cytoscape-fcose";
import type { GraphEdge, GraphNode } from "@/lib/api";

cytoscape.use(fcose as any);

const TYPE_COLOR: Record<string, string> = {
  Incident: "#2563eb",
  Service: "#0ea5e9",
  Region: "#10b981",
  Team: "#a855f7",
  Owner: "#f59e0b",
  RootCauseCategory: "#ef4444",
  Mitigation: "#14b8a6",
  Component: "#64748b",
};

const SEV_COLOR = ["#b91c1c", "#dc2626", "#ea580c", "#d97706", "#2563eb"];

export type Props = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selected?: string | null;
  onSelect?: (id: string) => void;
};

export default function KnowledgeGraph({ nodes, edges, selected, onSelect }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);

  const elements: ElementDefinition[] = useMemo(() => {
    const nodeEls: ElementDefinition[] = nodes.map((n) => ({
      data: {
        id: n.id,
        label: n.type === "Incident" ? n.label : n.label ?? n.id,
        type: n.type,
        severity: n.severity ?? null,
        service: n.service,
        region: n.region,
        title: n.title,
      },
    }));
    const edgeEls: ElementDefinition[] = edges.map((e, i) => ({
      data: {
        id: `e${i}`,
        source: e.source,
        target: e.target,
        relation: e.relation,
      },
    }));
    return [...nodeEls, ...edgeEls];
  }, [nodes, edges]);

  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      wheelSensitivity: 0.2,
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "font-size": 9,
            "text-valign": "center",
            "text-halign": "center",
            color: "#0f172a",
            "background-color": (ele: any) => {
              if (ele.data("type") === "Incident") {
                const sev = ele.data("severity");
                return SEV_COLOR[sev] ?? TYPE_COLOR.Incident;
              }
              return TYPE_COLOR[ele.data("type")] ?? "#94a3b8";
            },
            "border-width": 1,
            "border-color": "#ffffff",
            width: (ele: any) => (ele.data("type") === "Incident" ? 14 : 20),
            height: (ele: any) => (ele.data("type") === "Incident" ? 14 : 20),
          } as any,
        },
        {
          selector: "node[type != 'Incident']",
          style: {
            "font-weight": 600,
            "font-size": 10,
            "text-outline-color": "#ffffff",
            "text-outline-width": 2,
          } as any,
        },
        {
          selector: "edge",
          style: {
            width: 1,
            "line-color": "#cbd5e1",
            "target-arrow-color": "#cbd5e1",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            opacity: 0.65,
          } as any,
        },
        {
          selector: "node:selected",
          style: {
            "border-width": 3,
            "border-color": "#2563eb",
          } as any,
        },
      ],
      layout: {
        name: "fcose",
        animate: false,
        randomize: true,
        nodeRepulsion: 4500,
        idealEdgeLength: 60,
        nodeDimensionsIncludeLabels: true,
      } as any,
    });
    cyRef.current = cy;

    cy.on("tap", "node", (e) => onSelect?.(e.target.id()));

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [elements]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !selected) return;
    cy.elements().unselect();
    const n = cy.getElementById(selected);
    if (n && n.length) {
      n.select();
      cy.animate({ center: { eles: n }, zoom: 1.1 }, { duration: 350 });
    }
  }, [selected]);

  return <div ref={containerRef} className="h-full w-full rounded-lg border border-slate-200 bg-white" />;
}
