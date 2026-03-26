export const COMPLEX_MULTI_HOP_QUERIES: string[] = [
  "For INC-2026-0243, trace a 2-hop path through service/team/root cause and find the top 3 most similar incidents with shared dimensions.",
  "Find incidents that are connected to both Storage and Front Door through any path <= 3 hops, then rank by severity and impacted customers.",
  "Which incidents bridge two different communities via LINKED_TO edges, and what shared root-cause pattern appears across those bridges?",
  "Compare the graph neighborhood of INC-2026-0286 vs INC-2026-0183: shared nodes, unique nodes, and shortest connecting path.",
  "Find cross-region cascades where an incident in one region is linked to incidents in at least 2 other regions via service or community relationships.",
  "Identify incidents where team ownership differs but root cause is shared; return clusters with at least 3 incidents.",
  "Show path-based dependencies from Azure OpenAI incidents to downstream affected services and summarize the most frequent failure chain.",
  "Find incidents similar to INC-2026-0137, then expand one more hop to related incidents and explain what changes between hop-1 and hop-2.",
  "Which communities contain incidents from at least 3 teams and 2 services, and what incident IDs best represent each community?",
  "For the selected node, explain what happened, count connected nodes by type, and highlight the strongest multi-hop relationship pattern.",
];