"""Gold Q/A pairs spanning local, global, and drift retrieval modes."""

from __future__ import annotations

GOLD_QA: list[dict[str, object]] = [
    # --- local ---
    {
        "q": "What happened in incident INC-2026-0001?",
        "mode": "local",
        "must_include": ["INC-2026-0001"],
    },
    {
        "q": "List Sev1 incidents in the westus region involving Front Door",
        "mode": "local",
        "must_include_any": ["Front Door"],
    },
    {
        "q": "Which team owns the Cosmos DB incidents in eastus?",
        "mode": "local",
        "must_include_any": ["Cosmos", "team"],
    },
    {
        "q": "Show incidents assigned to the AOAI-Inference team",
        "mode": "local",
        "must_include_any": ["AOAI", "Azure OpenAI"],
    },
    {
        "q": "Pull the details for INC-2026-0137",
        "mode": "local",
        "must_include_any": ["INC-2026-0137"],
    },
    # --- global ---
    {
        "q": "What were the top recurring themes across 2026 incidents?",
        "mode": "global",
        "must_include_any": ["cluster", "community", "service", "root cause"],
    },
    {
        "q": "Summarize the overall Azure OpenAI health in 2026",
        "mode": "global",
        "must_include_any": ["Azure OpenAI", "capacity", "AOAI"],
    },
    {
        "q": "Give me an executive summary of certificate-related incidents",
        "mode": "global",
        "must_include_any": ["Certificate", "cert"],
    },
    {
        "q": "Which services saw the most capacity-related issues?",
        "mode": "global",
        "must_include_any": ["Capacity", "capacity"],
    },
    {
        "q": "Broad trends in DNS failures this year",
        "mode": "global",
        "must_include_any": ["DNS"],
    },
    # --- drift (cluster / storm) ---
    {
        "q": "What caused the March 2026 Front Door outage cluster?",
        "mode": "drift",
        "must_include_any": ["Front Door", "Certificate", "cert"],
    },
    {
        "q": "Explain the January 2026 Azure OpenAI capacity storm",
        "mode": "drift",
        "must_include_any": ["Azure OpenAI", "Capacity", "AOAI"],
    },
    {
        "q": "Analyze the May 2026 AKS CrashLoopBackOff cascade",
        "mode": "drift",
        "must_include_any": ["AKS", "kubelet", "Crash"],
    },
    {
        "q": "What happened with APIM in November 2026?",
        "mode": "drift",
        "must_include_any": ["APIM", "API Management", "policy"],
    },
    {
        "q": "Summarize the regional DNS storm in September 2026",
        "mode": "drift",
        "must_include_any": ["DNS", "Networking"],
    },
]
