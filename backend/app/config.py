from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:4173"

    data_dir: str = "./data"
    index_dir: str = "./data/index"

    incident_count: int = 400
    random_seed: int = 2026
    use_llm_for_index: bool = True

    # Graph database backend.
    # - "networkx": in-process NetworkX (zero-infra, pickle on disk).
    # - "neo4j":    Bolt-connected Neo4j 5 (Cypher-native reasoning).
    graph_backend: str = "networkx"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4jlocal"
    neo4j_database: str = "neo4j"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def use_neo4j(self) -> bool:
        return self.graph_backend.lower() == "neo4j"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).resolve()

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir).resolve()

    @property
    def has_azure_openai(self) -> bool:
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
