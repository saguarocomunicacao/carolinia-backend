"""Configurações via variáveis de ambiente."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Anthropic
    anthropic_api_key: str = ""
    model_orchestrator: str = "claude-opus-4-7"
    model_subagent: str = "claude-sonnet-4-6"
    
    # Lovable Cloud
    lovable_project_url: str = ""
    
    # Auth interno entre Lovable ↔ Railway
    shared_secret: str = ""
    
    # Workspace dos agentes
    workspaces_dir: str = "/workspaces"


settings = Settings()
