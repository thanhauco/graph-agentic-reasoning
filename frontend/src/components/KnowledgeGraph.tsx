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
  Community: "#9333ea",
};

// Sev0 deep red → Sev4 calm blue.
const SEV_COLOR = ["#b91c1c", "#dc2626", "#f97316", "#eab308", "#3b82f6"];

export type Props = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selected?: string | null;
  onSelect?: (id: string) => void;
  highlightIds?: string[];
};

export default function KnowledgeGraph({
  nodes,
  edges,
  selected,
  onSelect,
  highlightIds,
}: Props) {
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
      wheelSensitivity: 0.25,
      minZoom: 0.2,
      maxZoom: 3,
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "font-size": 9,
            "font-family": "Inter, system-ui, sans-serif",
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
            "border-width": 1.5,
            "border-color": "#ffffff",
            width: (ele: any) => (ele.data("type") === "Incident" ? 16 : 24),
            height: (ele: any) => (ele.data("type") === "Incident" ? 16 : 24),
            "transition-property":
              "background-color, border-color, border-width, opacity",
            "transition-duration": 220,
          } as any,
        },
        {
          selector: "node[type != 'Incident']",
          style: {
            "font-weight": 600,
            "font-size": 10,
            "text-outline-color": "#ffffff",
            "text-outline-width": 2,
            shape: "round-rectangle",
          } as any,
        },
        {
          selector: "edge",
          style: {
            width: 1,
            "line-color": "#cbd5e1",
            "target-arrow-color": "#cbd5e1",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.8,
            "curve-style": "bezier",
            opacity: 0.55,
            "transition-property": "line-color, target-arrow-color, opacity, width",
            "transition-duration": 220,
          } as any,
        },
        { selector: ".dim", style: { opacity: 0.08, "text-opacity": 0 } as any },
        {
          selector: ".hl-node",
          style: {
            "border-width": 4,
            "border-color": "#fbbf24",
            "z-index": 9999,
          } as any,
        },
        {
          selector: ".hl-edge",
          style: {
            "line-color": "#fbbf24",
            "target-arrow-color": "#fbbf24",
            opacity: 1,
            width: 2.5,
          } as any,
        },
        {
          selector: ".query-hit",
          style: {
            "border-width": 4,
            "border-color": "#22c55e",
            "background-blacken": -0.15,
            "z-index": 9998,
          } as any,
        },
        {
          selector: "node:selected, node.focused",
          style: {
            "border-width": 5,
            "border-color": "#2563eb",
            "border-opacity": 1,
            "background-blacken": -0.1,
            "z-index": 9999,
          } as any,
        },
        {
          selector: "edge.focused-edge",
          style: {
            "line-color": "#2563eb",
            "target-arrow-color": "#2563eb",
            width: 2,
            opacity: 0.9,
          } as any,
        },
      ],
      layout: {
        name: "fcose",
        animate: true,
        animationDuration: 500,
        animationEasing: "ease-out",
        randomize: true,
        nodeRepulsion: 5500,
        idealEdgeLength: 70,
        nodeDimensionsIncludeLabels: true,
        packComponents: true,
        fit: true,
        padding: 30,
      } as any,
    });
    cyRef.current = cy;

    cy.on("tap", "node", (e) => onSelect?.(e.target.id()));

    cy.on("mouseover", "node", (e) => {
      const node = e.target;
      const nb = node.closedNeighborhood();
      // Don't dim focused (persistently-selected) node or its neighborhood.
      const focused = cy.nodes(".focused");
      const keep = nb.union(focused.closedNeighborhood());
      cy.elements().difference(keep).addClass("dim");
      nb.nodes().addClass("hl-node");
      nb.edges().addClass("hl-edge");
      if (containerRef.current) containerRef.current.style.cursor = "pointer";
    });
    cy.on("mouseout", "node", () => {
      cy.elements().removeClass("dim hl-node hl-edge");
      if (containerRef.current) containerRef.current.style.cursor = "";
    });

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [elements]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().unselect();
    cy.nodes().removeClass("focused");
    cy.edges().removeClass("focused-edge");
    if (!selected) return;
    const n = cy.getElementById(selected);
    if (n && n.length) {
      n.select();
      n.addClass("focused");
      n.connectedEdges().addClass("focused-edge");
      cy.animate(
        { center: { eles: n }, zoom: 1.4 },
        { duration: 450, easing: "ease-in-out" },
      );
    }
    // Re-run when elements change (e.g. after remount / graph reload)
    // so the focused visual is reapplied on a freshly-built cytoscape.
  }, [selected, elements]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().removeClass("query-hit");
    const ids = highlightIds ?? [];
    if (!ids.length) return;
    const hits = cy.collection();
    ids.forEach((id) => {
      const n = cy.getElementById(id);
      if (n && n.length) {
        n.addClass("query-hit");
        hits.merge(n as any);
      }
    });
    if (hits.length > 0) {
      cy.animate(
        { fit: { eles: hits, padding: 60 } as any },
        { duration: 500, easing: "ease-in-out" },
      );
    }
  }, [highlightIds, elements]);

  return <div ref={containerRef} className="h-full w-full rounded-lg border border-slate-200 bg-white" />;
}
