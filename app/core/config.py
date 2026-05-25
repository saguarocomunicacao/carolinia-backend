"""Configurações via variáveis de ambiente."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ============================================================
    # Anthropic (Claude)
    # ============================================================
    anthropic_api_key: str = ""
    model_orchestrator: str = "claude-opus-4-7"
    model_subagent: str = "claude-sonnet-4-6"
    
    # ============================================================
    # OpenAI (GPT, DALL-E)
    # ============================================================
    openai_api_key: str = ""
    model_openai_text: str = "gpt-4o"
    model_openai_text_fast: str = "gpt-4o-mini"
    model_openai_image: str = "dall-e-3"
    
    # ============================================================
    # Google (Gemini, Nano Banana)
    # ============================================================
    google_api_key: str = ""
    model_google_text: str = "gemini-2.5-flash"
    model_google_text_pro: str = "gemini-2.5-pro"
    model_google_image: str = "gemini-2.5-flash-image"
    
    # ============================================================
    # Lovable Cloud
    # ============================================================
    lovable_project_url: str = ""
    
    # ============================================================
    # Auth interno Lovable ↔ Railway
    # ============================================================
    shared_secret: str = ""
    
    # ============================================================
    # Workspace dos agentes
    # ============================================================
    workspaces_dir: str = "/workspaces"


settings = Settings()
