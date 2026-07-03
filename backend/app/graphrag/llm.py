"""Azure OpenAI helpers for chat + embeddings.

Safe to import even if Azure OpenAI is not configured; callers must check
`get_settings().has_azure_openai` before using.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from app.config import get_settings

log = logging.getLogger("icm.graphrag.llm")


def _client():
    from openai import AzureOpenAI
    s = get_settings()
    return AzureOpenAI(
        api_key=s.azure_openai_api_key,
        api_version=s.azure_openai_api_version,
        azure_endpoint=s.azure_openai_endpoint,
    )


def _chat_client_and_model():
    """Return (client, model) for chat. Prefer NVIDIA Build (OpenAI-compatible)."""
    s = get_settings()
    if getattr(s, "nvidia_api_key", ""):
        from openai import OpenAI
        return OpenAI(api_key=s.nvidia_api_key, base_url=s.nvidia_base_url), s.nvidia_model
    return _client(), s.azure_openai_chat_deployment


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    s = get_settings()
    client = _client()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(model=s.azure_openai_embedding_deployment, input=chunk)
        out.extend(d.embedding for d in resp.data)
    return np.array(out, dtype=np.float32)


def chat(
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    s = get_settings()
    client, model = _chat_client_and_model()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def stream_chat(
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> Iterable[str]:
    s = get_settings()
    client, model = _chat_client_and_model()
    stream = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        if delta and delta.content:
            yield delta.content


def summarize_community(incidents: list[dict]) -> str:
    if not incidents:
        return ""
    bullets = "\n".join(
        f"- {i['incidentId']} Sev{i['severity']} {i['service']}/{i['region']}: {i['title']} — rc={i['rootCauseCategory']}"
        for i in incidents[:25]
    )
    system = (
        "You are an Azure IcM analyst. Summarize an incident cluster in 3-5 sentences, "
        "calling out shared services, root causes, regions, customer impact, and "
        "recurring mitigation patterns. Be concrete and cite 2-3 incident IDs."
    )
    user = f"Cluster has {len(incidents)} incidents. Representative subset:\n{bullets}"
    return chat(system, user, max_tokens=300)
