"""Rotas da API."""
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str = "ok"
    workspaces_dir: str
    has_anthropic_key: bool
    has_shared_secret: bool
    has_lovable_url: bool


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        workspaces_dir=settings.workspaces_dir,
        has_anthropic_key=bool(settings.anthropic_api_key),
        has_shared_secret=bool(settings.shared_secret),
        has_lovable_url=bool(settings.lovable_project_url),
    )


def _verify_secret(provided: str | None) -> None:
    if not settings.shared_secret:
        raise HTTPException(status_code=500, detail="SHARED_SECRET não configurado no backend")
    if provided != settings.shared_secret:
        raise HTTPException(status_code=401, detail="Shared secret inválido")


class ProcessDemandRequest(BaseModel):
    demand_id: str
    demand_title: str
    demand_description: str
    project: dict = {}
    project_id: str = ""
    context_entries: list[dict] = []


class ProcessDemandResponse(BaseModel):
    accepted: bool = True
    demand_id: str


@router.post("/process-demand", response_model=ProcessDemandResponse)
async def process_demand_endpoint(
    payload: ProcessDemandRequest,
    x_shared_secret: str | None = Header(default=None, alias="X-Shared-Secret"),
) -> ProcessDemandResponse:
    _verify_secret(x_shared_secret)
    
    # Esqueleto: aceita o request mas ainda não processa.
    # A integração com Claude Agent SDK virá nos prompts seguintes.
    print(f"Demanda recebida: {payload.demand_id} - {payload.demand_title}")
    
    return ProcessDemandResponse(accepted=True, demand_id=payload.demand_id)
