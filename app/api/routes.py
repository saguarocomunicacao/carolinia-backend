"""Rotas da API."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path as PathLib

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.services.git_service import clone_repo, list_files
from app.services.lovable_client import lovable_client
from app.services.text_extractor import estimate_tokens, extract_text

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# Schemas
# ============================================================

class HealthResponse(BaseModel):
    status: str = "ok"
    workspaces_dir: str
    has_anthropic_key: bool
    has_shared_secret: bool
    has_lovable_url: bool


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


class CloneRepoRequest(BaseModel):
    repo_id: str
    project_id: str
    repo_full_name: str
    access_token_secret_id: str
    default_branch: str = "main"


class CloneRepoResponse(BaseModel):
    accepted: bool = True
    repo_id: str


class ProcessFileRequest(BaseModel):
    file_id: str
    project_id: str
    file_name: str
    signed_url: str
    mime_type: str | None = None
    category: str = "other"


class ProcessFileResponse(BaseModel):
    accepted: bool = True
    file_id: str


# ============================================================
# Auth helper
# ============================================================

def _verify_secret(provided: str | None) -> None:
    if not settings.shared_secret:
        raise HTTPException(status_code=500, detail="SHARED_SECRET não configurado no backend")
    if provided != settings.shared_secret:
        raise HTTPException(status_code=401, detail="Shared secret inválido")


# ============================================================
# Health
# ============================================================

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        workspaces_dir=settings.workspaces_dir,
        has_anthropic_key=bool(settings.anthropic_api_key),
        has_shared_secret=bool(settings.shared_secret),
        has_lovable_url=bool(settings.lovable_project_url),
    )


# ============================================================
# /process-demand (esqueleto — agente IA virá no P07)
# ============================================================

@router.post("/process-demand", response_model=ProcessDemandResponse)
async def process_demand_endpoint(
    payload: ProcessDemandRequest,
    x_shared_secret: str | None = Header(default=None, alias="X-Shared-Secret"),
) -> ProcessDemandResponse:
    _verify_secret(x_shared_secret)
    
    print(f"Demanda recebida: {payload.demand_id} - {payload.demand_title}")
    print(f"Contexto: {len(payload.context_entries)} entradas")
    
    return ProcessDemandResponse(accepted=True, demand_id=payload.demand_id)


# ============================================================
# /clone-repo
# ============================================================

@router.post("/clone-repo", response_model=CloneRepoResponse)
async def clone_repo_endpoint(
    payload: CloneRepoRequest,
    background_tasks: BackgroundTasks,
    x_shared_secret: str | None = Header(default=None, alias="X-Shared-Secret"),
) -> CloneRepoResponse:
    _verify_secret(x_shared_secret)
    
    background_tasks.add_task(
        _clone_and_index,
        repo_id=payload.repo_id,
        project_id=payload.project_id,
        repo_full_name=payload.repo_full_name,
        access_token_secret_id=payload.access_token_secret_id,
        default_branch=payload.default_branch,
    )
    
    return CloneRepoResponse(accepted=True, repo_id=payload.repo_id)


async def _clone_and_index(
    repo_id: str,
    project_id: str,
    repo_full_name: str,
    access_token_secret_id: str,
    default_branch: str,
):
    """Background task: busca token, clona, indexa arquivos relevantes."""
    print(f"[clone-repo] Iniciando clone: {repo_full_name}")
    
    # 1. Busca o PAT no vault via Edge Function
    access_token = await lovable_client.get_project_secret(access_token_secret_id)
    if not access_token:
        await lovable_client.send_repo_status(
            repo_id=repo_id,
            status="error",
            error_message="Token GitHub não encontrado no vault",
        )
        return
    
    # 2. Clona o repo
    success, error, path = clone_repo(
        project_id=project_id,
        repo_full_name=repo_full_name,
        access_token=access_token,
        default_branch=default_branch,
    )
    
    await lovable_client.send_repo_status(
        repo_id=repo_id,
        status="cloned" if success else "error",
        error_message=error,
    )
    
    if not success or not path:
        print(f"[clone-repo] Falhou: {repo_full_name} — {error}")
        return
    
    print(f"[clone-repo] Clone OK: {repo_full_name}. Indexando...")
    
    # 3. Indexa arquivos relevantes em project_context
    files = list_files(path)
    context_entries = []
    
    for f in files[:100]:
        text = extract_text(f)
        if not text:
            continue
        relative = f.relative_to(path)
        context_entries.append({
            "source_type": "repo",
            "source_id": repo_id,
            "title": str(relative),
            "content": text[:10_000],
            "tokens_approx": estimate_tokens(text[:10_000]),
            "importance": _calculate_importance(relative),
        })
    
    if context_entries:
        await lovable_client.bulk_insert_context(
            project_id=project_id,
            entries=context_entries,
        )
        print(f"[clone-repo] Indexou {len(context_entries)} arquivos de {repo_full_name}")


def _calculate_importance(relative_path: PathLib) -> int:
    """Importância 1-10 baseada no nome do arquivo."""
    name = relative_path.name.lower()
    if name in ("readme.md", "readme"):
        return 10
    if name in ("package.json", "pyproject.toml", "requirements.txt", "go.mod", "cargo.toml"):
        return 9
    if name.endswith(".md"):
        return 8
    return 5


# ============================================================
# /process-file
# ============================================================

@router.post("/process-file", response_model=ProcessFileResponse)
async def process_file_endpoint(
    payload: ProcessFileRequest,
    background_tasks: BackgroundTasks,
    x_shared_secret: str | None = Header(default=None, alias="X-Shared-Secret"),
) -> ProcessFileResponse:
    _verify_secret(x_shared_secret)
    
    background_tasks.add_task(
        _download_and_index_file,
        file_id=payload.file_id,
        project_id=payload.project_id,
        file_name=payload.file_name,
        signed_url=payload.signed_url,
        mime_type=payload.mime_type,
        category=payload.category,
    )
    
    return ProcessFileResponse(accepted=True, file_id=payload.file_id)


async def _download_and_index_file(
    file_id: str,
    project_id: str,
    file_name: str,
    signed_url: str,
    mime_type: str | None,
    category: str,
):
    """Background task: baixa arquivo, extrai texto, insere em project_context."""
    print(f"[process-file] Baixando: {file_name}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(signed_url)
            response.raise_for_status()
            content = response.content
        
        # Salva em temp pra processar com pypdf/Pillow/Tesseract
        with tempfile.NamedTemporaryFile(
            suffix=PathLib(file_name).suffix,
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp_path = PathLib(tmp.name)
        
        text = extract_text(tmp_path, mime_type)
        tmp_path.unlink(missing_ok=True)
        
        if not text:
            print(f"[process-file] Sem texto extraível: {file_name}")
            await lovable_client.mark_file_indexed(file_id=file_id)
            return
        
        importance_map = {
            "spec": 9, "document": 8, "code": 7, "image": 5, "other": 4,
        }
        importance = importance_map.get(category, 5)
        
        entry = {
            "source_type": "file",
            "source_id": file_id,
            "title": file_name,
            "content": text[:10_000],
            "tokens_approx": estimate_tokens(text[:10_000]),
            "importance": importance,
        }
        
        await lovable_client.bulk_insert_context(
            project_id=project_id,
            entries=[entry],
        )
        await lovable_client.mark_file_indexed(file_id=file_id)
        print(f"[process-file] Indexado: {file_name}")
    
    except Exception:
        logger.exception("Erro processando arquivo %s", file_id)
