"""Generate 400 synthetic Azure incidents for 2026.

Deterministic: uses Faker + random with a fixed seed. Produces realistic
"storm" clusters (e.g., Jan 2026 AOAI capacity, Mar 2026 Front Door cert
rotation) so graph queries surface meaningful patterns.

Usage:
    python -m scripts.gen_incidents
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from faker import Faker

from app.config import get_settings
from app.models import Incident

SERVICES = [
    "AKS", "App Service", "Cosmos DB", "Storage", "Azure OpenAI",
    "Front Door", "Entra ID", "Monitor", "SQL DB", "Networking",
    "API Management", "Functions",
]

REGIONS = [
    "eastus", "eastus2", "westus", "westus2", "westus3", "centralus",
    "northcentralus", "southcentralus", "westeurope", "northeurope",
    "uksouth", "ukwest", "francecentral", "germanywestcentral",
    "switzerlandnorth", "swedencentral", "eastasia", "southeastasia",
    "japaneast", "australiaeast",
]

TEAMS = [
    "AKS-Control-Plane", "AKS-Node-Team", "App-Service-Platform",
    "Cosmos-DB-Core", "Storage-Blob", "AOAI-Inference", "AOAI-Capacity",
    "Front-Door-Edge", "Entra-Identity", "Monitor-Ingestion",
    "SQL-Engine", "Network-Fabric", "APIM-Gateway", "Functions-Runtime",
]

ROOT_CAUSES = [
    "Config", "Deployment", "Dependency", "Capacity",
    "Code Defect", "Certificate", "DNS", "Networking", "Security",
]

MITIGATIONS = [
    "Rolled back deployment",
    "Scaled out capacity",
    "Rotated certificate",
    "Failed over to secondary region",
    "Disabled faulty feature flag",
    "Restarted affected nodes",
    "Applied hotfix",
    "Cleared cache and reindexed",
    "Updated DNS records",
    "Tightened rate limits",
]

COMPONENT_SIGNATURES = {
    "AKS": ["kubelet", "coredns", "containerd", "api-server", "etcd", "kube-proxy"],
    "App Service": ["kudu", "easyauth", "w3wp", "dwas", "arr-affinity"],
    "Cosmos DB": ["gateway", "partition-master", "replica-set", "backend-service"],
    "Storage": ["blob-front-end", "storage-stamp", "partition-layer", "stream-layer"],
    "Azure OpenAI": ["router", "inference-pool", "capacity-manager", "tokenizer"],
    "Front Door": ["edge-pop", "tls-terminator", "origin-probe", "waf-engine"],
    "Entra ID": ["sts", "token-signer", "mfa-service", "ccs"],
    "Monitor": ["ingestion-pipe", "kusto-cluster", "log-analytics-agent"],
    "SQL DB": ["sql-engine", "availability-group", "resource-governor"],
    "Networking": ["slb", "vnet-gateway", "dns-resolver", "nsg-engine"],
    "API Management": ["gateway-proxy", "dev-portal", "policy-engine"],
    "Functions": ["host-runtime", "scale-controller", "dotnet-worker", "python-worker"],
}


# Storm cluster scenarios (month, service, root-cause, title-pattern, count, severity).
STORMS = [
    ("2026-01-15", "Azure OpenAI", "Capacity",
     "AOAI inference pool throttling ({region})", 14, 1),
    ("2026-02-07", "Cosmos DB", "Capacity",
     "Cosmos DB 429 throttling surge ({region})", 10, 2),
    ("2026-03-12", "Front Door", "Certificate",
     "Front Door TLS handshake failures after cert rotation ({region})", 16, 1),
    ("2026-04-03", "Entra ID", "Dependency",
     "Entra token issuance latency spike ({region})", 9, 1),
    ("2026-05-22", "AKS", "Code Defect",
     "AKS kubelet CrashLoopBackOff after 1.31 upgrade ({region})", 12, 2),
    ("2026-06-18", "Storage", "Networking",
     "Blob storage elevated latency ({region})", 8, 3),
    ("2026-07-09", "App Service", "Deployment",
     "App Service 5xx after platform rollout ({region})", 11, 2),
    ("2026-08-14", "Functions", "Dependency",
     "Functions cold-start regressions ({region})", 7, 3),
    ("2026-09-05", "Networking", "DNS",
     "Regional DNS resolution failures ({region})", 13, 1),
    ("2026-10-27", "SQL DB", "Capacity",
     "SQL DB connection pool exhaustion ({region})", 9, 2),
    ("2026-11-11", "API Management", "Config",
     "APIM policy misconfiguration blocking requests ({region})", 8, 2),
    ("2026-12-03", "Monitor", "Dependency",
     "Monitor ingestion backlog ({region})", 10, 3),
]


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mk_description(fake: Faker, service: str, root_cause: str, region: str, comps: list[str]) -> str:
    sig = ", ".join(comps[:2]) if comps else "multiple components"
    return (
        f"Customers reported elevated error rates on {service} in {region}. "
        f"Initial triage indicates a {root_cause.lower()} issue affecting {sig}. "
        f"Telemetry shows degraded availability and increased latency during the impact window. "
        f"{fake.sentence(nb_words=18)}"
    )


def _pick_weighted(rng: random.Random, weights: dict[str, float]) -> str:
    items = list(weights.keys())
    w = list(weights.values())
    return rng.choices(items, weights=w, k=1)[0]


def _severity_weight(rng: random.Random) -> int:
    # Sev3 most common; Sev0 rare.
    return rng.choices([0, 1, 2, 3, 4], weights=[2, 10, 25, 45, 18], k=1)[0]


def _status_weight(rng: random.Random) -> str:
    return rng.choices(
        ["Active", "Mitigated", "Resolved", "Postmortem"],
        weights=[5, 15, 70, 10], k=1,
    )[0]


def _times(rng: random.Random, base: datetime, status: str) -> tuple[str, str | None, str | None]:
    created = base + timedelta(minutes=rng.randint(0, 59))
    mitigated = resolved = None
    if status in ("Mitigated", "Resolved", "Postmortem"):
        mitigated_dt = created + timedelta(minutes=rng.randint(20, 600))
        mitigated = _iso(mitigated_dt)
        if status in ("Resolved", "Postmortem"):
            resolved_dt = mitigated_dt + timedelta(hours=rng.randint(1, 48))
            resolved = _iso(resolved_dt)
    return _iso(created), mitigated, resolved


def _team_for_service(service: str, rng: random.Random) -> str:
    candidates = [t for t in TEAMS if service.split()[0].lower()[:3] in t.lower()]
    return rng.choice(candidates or TEAMS)


def _make_incident(
    idx: int,
    rng: random.Random,
    fake: Faker,
    *,
    when: datetime,
    service: str | None = None,
    root_cause: str | None = None,
    title: str | None = None,
    region: str | None = None,
    severity: int | None = None,
    storm_tag: str | None = None,
) -> dict:
    svc = service or rng.choice(SERVICES)
    reg = region or rng.choice(REGIONS)
    rc = root_cause or rng.choice(ROOT_CAUSES)
    sev = severity if severity is not None else _severity_weight(rng)
    status = _status_weight(rng)
    created, mitigated, resolved = _times(rng, when, status)
    comps = rng.sample(COMPONENT_SIGNATURES.get(svc, ["generic-worker"]),
                       k=min(2, len(COMPONENT_SIGNATURES.get(svc, ["x"]))))
    desc = _mk_description(fake, svc, rc, reg, comps)
    mit = rng.choice(MITIGATIONS)
    tags = [svc.lower().replace(" ", "-"), rc.lower().replace(" ", "-"), reg]
    if storm_tag:
        tags.append(storm_tag)
    inc_id = f"INC-2026-{idx:04d}"
    default_title = f"{svc} {rc.lower()} issue in {reg}"
    return Incident(
        incidentId=inc_id,
        title=title or default_title,
        description=desc,
        severity=sev,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        service=svc,
        region=reg,
        team=_team_for_service(svc, rng),
        owner=fake.name(),
        createdAt=created,
        mitigatedAt=mitigated,
        resolvedAt=resolved,
        rootCauseCategory=rc,
        mitigation=mit,
        impactedCustomers=rng.randint(1, 5000) if sev <= 2 else rng.randint(0, 400),
        tags=tags,
        linkedIncidents=[],
        componentSignatures=comps,
    ).model_dump()


def generate(count: int = 400, seed: int = 2026) -> list[dict]:
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)
    incidents: list[dict] = []
    idx = 1

    # Storm clusters first.
    for date_str, service, rc, title_pat, n, sev in STORMS:
        storm_base = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        storm_tag = f"storm:{service.lower().replace(' ', '-')}-{date_str}"
        cluster_ids: list[str] = []
        for _ in range(n):
            reg = rng.choice(REGIONS)
            when = storm_base + timedelta(minutes=rng.randint(0, 240))
            inc = _make_incident(
                idx, rng, fake,
                when=when, service=service, root_cause=rc,
                title=title_pat.format(region=reg),
                region=reg, severity=sev, storm_tag=storm_tag,
            )
            cluster_ids.append(inc["incidentId"])
            incidents.append(inc)
            idx += 1
        # Cross-link within the storm cluster.
        for inc in incidents[-n:]:
            others = [x for x in cluster_ids if x != inc["incidentId"]]
            inc["linkedIncidents"] = rng.sample(others, k=min(3, len(others)))

    # Fill remainder with random incidents across 2026.
    year_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(2026, 12, 31, tzinfo=timezone.utc)
    year_span = int((year_end - year_start).total_seconds())
    while idx <= count:
        when = year_start + timedelta(seconds=rng.randint(0, year_span))
        inc = _make_incident(idx, rng, fake, when=when)
        # 15% chance to link to a prior random incident.
        if incidents and rng.random() < 0.15:
            inc["linkedIncidents"] = [rng.choice(incidents)["incidentId"]]
        incidents.append(inc)
        idx += 1

    incidents.sort(key=lambda i: i["createdAt"])
    return incidents


def main() -> None:
    settings = get_settings()
    out_dir = settings.data_path
    out_dir.mkdir(parents=True, exist_ok=True)
    data = generate(settings.incident_count, settings.random_seed)
    out_path = out_dir / "incidents.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {len(data)} incidents to {out_path}")


if __name__ == "__main__":
    main()
