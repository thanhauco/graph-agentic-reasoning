from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Severity = Literal[0, 1, 2, 3, 4]
Status = Literal["Active", "Mitigated", "Resolved", "Postmortem"]


class Incident(BaseModel):
    incidentId: str
    title: str
    description: str
    severity: Severity
    status: Status
    service: str
    region: str
    team: str
    owner: str
    createdAt: str  # ISO-8601
    mitigatedAt: str | None = None
    resolvedAt: str | None = None
    rootCauseCategory: str
    mitigation: str
    impactedCustomers: int = Field(ge=0)
    tags: list[str] = Field(default_factory=list)
    linkedIncidents: list[str] = Field(default_factory=list)
    componentSignatures: list[str] = Field(default_factory=list)
