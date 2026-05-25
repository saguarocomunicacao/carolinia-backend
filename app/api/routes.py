"""Rotas da API."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path as PathLib
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel

from app.agents.analista_briefing import AnalistaBriefing
from app.agents.base import AgentContext
from app.core.config import settings
from app.services.context_loader import fetch_project_context
from app.services.git_service import clone_repo, list_files
from app.services.llm_router import (
    ModelChoice,
    Provider,
    call_llm,
    generate_image,
)
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
    has_openai_key: bool
    has_google_key: bool


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


class RouterTestRequest(BaseModel):
    prompt: str = "Diga 'olá' em português, espanhol e inglês, separados por vírgula."


class ProviderTestResult(BaseModel):
    provider: str
    model: str
    success: bool
    text: str = ""
    error: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


class RouterTestResponse(BaseModel):
    results: list[ProviderTestResult]


class AnalyzeBriefingRequest(BaseModel):
    project_id: str
    analysis_run_id: str
    window_id: str | None = None
    previous_run_id: str | None = None


class AnalyzeBriefingResponse(BaseModel):
    accepted: bool = True
    analysis_run_id: str
    message: str = "Análise iniciada em background"


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
        has_openai_key=bool(settings.openai_api_key),
        has_google_key=bool(settings.google_api_key),
    )


# ============================================================
# /router-test — valida cada um dos 3 providers
# ============================================================

@router.post("/router-test", response_model=RouterTestResponse)
async def router_test_endpoint(
    payload: RouterTestRequest,
    x_shared_secret: str | None = Header(default=None, alias="X-Shared-Secret"),
) -> RouterTestResponse:
    """Testa os 3 providers individualmente. Não usa fallback — testa cada um."""
    _verify_secret(x_shared_secret)
    
    test_cases = [
        (Provider.ANTHROPIC, settings.model_orchestrator),
        (Provider.OPENAI, settings.model_openai_text),
        (Provider.GOOGLE, settings.model_google_text),
    ]
    
    results = []
    
    for provider, model in test_cases:
        try:
            response = call_llm(
                system_prompt="Você é um assistente conciso. Responda em uma linha.",
                user_message=payload.prompt,
                model_preferences=[ModelChoice(provider, model)],
                max_tokens=200,
                temperature=0.7,
            )
            results.append(ProviderTestResult(
                provider=provider.value,
                model=model,
                success=True,
                text=response.text[:300],
                tokens_input=response.tokens_input,
                tokens_output=response.tokens_output,
                cost_usd=round(response.cost_usd, 6),
                latency_ms=response.latency_ms,
            ))
        except Exception as e:
            results.append(ProviderTestResult(
                provider=provider.value,
                model=model,
                success=False,
                error=str(e)[:500],
            ))
    
    return RouterTestResponse(results=results)


# ============================================================
# /process-demand (esqueleto — agente IA vem nos próximos passos)
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
    print(f"[clone-repo] Iniciando clone: {repo_full_name}")
    
    access_token = await lovable_client.get_project_secret(access_token_secret_id)
    if not access_token:
        await lovable_client.send_repo_status(
            repo_id=repo_id,
            status="error",
            error_message="Token GitHub não encontrado no vault",
        )
        return
    
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
    print(f"[process-file] Baixando: {file_name}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(signed_url)
            response.raise_for_status()
            content = response.content
        
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


# ============================================================
# /analyze-briefing — endpoint do Analista de Briefing
# ============================================================

@router.post("/analyze-briefing", response_model=AnalyzeBriefingResponse)
async def analyze_briefing_endpoint(
    payload: AnalyzeBriefingRequest,
    background_tasks: BackgroundTasks,
    x_shared_secret: str | None = Header(default=None, alias="X-Shared-Secret"),
) -> AnalyzeBriefingResponse:
    """Recebe pedido de análise, responde 200 imediato, processa em background."""
    _verify_secret(x_shared_secret)
    
    logger.info(
        "[analyze-briefing] Recebido: project=%s run=%s window=%s previous=%s",
        payload.project_id, payload.analysis_run_id, payload.window_id, payload.previous_run_id,
    )
    
    background_tasks.add_task(
        _analyze_briefing_task,
        project_id=payload.project_id,
        analysis_run_id=payload.analysis_run_id,
        window_id=payload.window_id,
        previous_run_id=payload.previous_run_id,
    )
    
    return AnalyzeBriefingResponse(
        accepted=True,
        analysis_run_id=payload.analysis_run_id,
    )


async def _analyze_briefing_task(
    project_id: str,
    analysis_run_id: str,
    window_id: str | None,
    previous_run_id: str | None,
) -> None:
    """Task assíncrona — busca contexto, roda Analista, cria demandas, finaliza."""
    marker = f"[task project={project_id} run={analysis_run_id}]"
    logger.info("%s Iniciando", marker)
    
    # 1. Busca contexto do projeto
    ctx_data = await fetch_project_context(project_id, min_importance=1)
    if not ctx_data.get("ok"):
        logger.error("%s Falha buscando project_context", marker)
        await _finalize_failed(
            analysis_run_id, window_id, project_id,
            "Falha ao buscar project_context",
        )
        return
    
    context_entries = ctx_data.get("entries", [])
    if not context_entries:
        logger.warning("%s Sem entries de contexto — análise pode ficar pobre", marker)
    
    # 2. Busca meta do projeto
    project_meta = await _fetch_project_meta(project_id)
    
    # 3. Monta AgentContext
    agent_context = AgentContext(
        project_id=project_id,
        project_name=project_meta.get("name", ""),
        project_stack=project_meta.get("stack", ""),
        project_description=project_meta.get("description", ""),
        context_entries=context_entries,
        extra_context={
            "analysis_run_id": analysis_run_id,
            "previous_run_id": previous_run_id or "primeira análise",
            "total_context_entries": len(context_entries),
        },
    )
    
    # 4. Roda o Analista
    instruction = (
        "Analise TODO o material do projeto e produza um JSON estruturado "
        "com os gaps que precisam ser respondidos antes do time começar. "
        "Siga ESTRITAMENTE o formato JSON descrito no seu system prompt."
    )
    
    try:
        analista = AnalistaBriefing()
        result = await analista.run(context=agent_context, instruction=instruction)
    except Exception as e:
        logger.exception("%s Erro rodando Analista", marker)
        await _finalize_failed(
            analysis_run_id, window_id, project_id,
            f"Erro rodando Analista: {str(e)[:500]}",
        )
        return
    
    if not result.success:
        logger.error("%s Analista success=False: %s", marker, result.error)
        await _finalize_failed(
            analysis_run_id, window_id, project_id,
            f"Analista falhou: {result.error}",
        )
        return
    
    logger.info(
        "%s Analista completou: model=%s cost=$%.4f tokens_in=%d tokens_out=%d",
        marker, result.model, result.cost_usd, result.tokens_input, result.tokens_output,
    )
    
    # 5. Parseia output JSON
    try:
        analysis_output = AnalistaBriefing.parse_output(result.output_text)
    except Exception as e:
        logger.exception("%s Falha parseando JSON do Analista", marker)
        logger.error("%s Output bruto: %s", marker, result.output_text[:500])
        await _finalize_failed(
            analysis_run_id, window_id, project_id,
            f"Output do Analista não é JSON válido: {str(e)[:300]}",
        )
        return
    
    logger.info(
        "%s Parseou: %d gaps, confidence=%s",
        marker, len(analysis_output.gaps), analysis_output.confidence_level,
    )
    
    # 6. Cria uma demanda pra cada gap
    created_demands, failed_demands = await _create_demands_for_gaps(
        project_id=project_id,
        analysis_run_id=analysis_run_id,
        gaps=analysis_output.gaps,
        marker=marker,
    )
    
    # 7. Determina briefing_status do projeto
    if len(analysis_output.gaps) == 0:
        new_briefing_status = "consolidated"
    else:
        new_briefing_status = "awaiting_clarifications"
    
    # 8. Finaliza
    await _finalize_completed(
        analysis_run_id=analysis_run_id,
        window_id=window_id,
        project_id=project_id,
        new_briefing_status=new_briefing_status,
        analysis_output=analysis_output,
        total_cost_usd=result.cost_usd,
        total_tokens=result.tokens_input + result.tokens_output,
        demands_added=len(created_demands),
        demands_failed=len(failed_demands),
    )
    
    logger.info(
        "%s Concluído: %d demandas criadas, %d falharam, briefing_status=%s",
        marker, len(created_demands), len(failed_demands), new_briefing_status,
    )


# ============================================================
# Helpers do /analyze-briefing
# ============================================================

async def _fetch_project_meta(project_id: str) -> dict[str, Any]:
    """Busca name/description/stack do projeto via Edge Function get-project-meta."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.post(
                f"{settings.lovable_project_url}/functions/v1/get-project-meta",
                headers={
                    "X-Shared-Secret": settings.shared_secret,
                    "Content-Type": "application/json",
                },
                json={"project_id": project_id},
            )
            if response.status_code == 200:
                return response.json().get("project", {})
            logger.warning("get-project-meta retornou %d: %s", response.status_code, response.text[:200])
    except Exception:
        logger.exception("Falha buscando meta do projeto %s", project_id)
    return {}


async def _create_demands_for_gaps(
    project_id: str,
    analysis_run_id: str,
    gaps: list,
    marker: str,
) -> tuple[list[str], list[str]]:
    """Cria uma demanda pra cada gap via agent-create-demand."""
    created: list[str] = []
    failed: list[str] = []
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        for gap in gaps:
            try:
                payload = {
                    "project_id": project_id,
                    "title": f"[{gap.priority}] {gap.title}",
                    "description": f"**Pergunta:** {gap.question}\n\n**Por quê:** {gap.why_matters}",
                    "source": "analista_briefing",
                    "analysis_run_id": analysis_run_id,
                    "approval_required": True,
                    "metadata": {
                        "category": gap.category,
                        "priority": gap.priority,
                        "question": gap.question,
                        "why_matters": gap.why_matters,
                    },
                }
                response = await client.post(
                    f"{settings.lovable_project_url}/functions/v1/agent-create-demand",
                    headers={
                        "X-Shared-Secret": settings.shared_secret,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if response.status_code == 200:
                    body = response.json()
                    demand_id = body.get("demand_id", "")
                    created.append(demand_id)
                    logger.info("%s Demanda criada: %s — %s", marker, demand_id, gap.title)
                else:
                    failed.append(gap.title)
                    logger.error(
                        "%s Falha criando demanda '%s': %s %s",
                        marker, gap.title, response.status_code, response.text[:200],
                    )
            except Exception:
                failed.append(gap.title)
                logger.exception("%s Exceção criando demanda '%s'", marker, gap.title)
    
    return created, failed


async def _finalize_completed(
    analysis_run_id: str,
    window_id: str | None,
    project_id: str,
    new_briefing_status: str,
    analysis_output: Any,
    total_cost_usd: float,
    total_tokens: int,
    demands_added: int,
    demands_failed: int,
) -> None:
    """Chama finalize-analysis-run pra marcar como completed."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            await client.post(
                f"{settings.lovable_project_url}/functions/v1/finalize-analysis-run",
                headers={
                    "X-Shared-Secret": settings.shared_secret,
                    "Content-Type": "application/json",
                },
                json={
                    "analysis_run_id": analysis_run_id,
                    "window_id": window_id,
                    "project_id": project_id,
                    "status": "completed",
                    "new_briefing_status": new_briefing_status,
                    "analysis_output_text": getattr(analysis_output, "notes", "") or "",
                    "briefing_summary": getattr(analysis_output, "briefing_summary", ""),
                    "identified_gaps_count": len(getattr(analysis_output, "gaps", [])),
                    "confidence_level": getattr(analysis_output, "confidence_level", "medium"),
                    "total_cost_usd": total_cost_usd,
                    "total_tokens": total_tokens,
                    "demands_added": demands_added,
                    "demands_failed": demands_failed,
                },
            )
    except Exception:
        logger.exception("Falha em finalize-analysis-run (completed) pra run=%s", analysis_run_id)


async def _finalize_failed(
    analysis_run_id: str,
    window_id: str | None,
    project_id: str,
    error_message: str,
) -> None:
    """Chama finalize-analysis-run pra marcar como failed."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            await client.post(
                f"{settings.lovable_project_url}/functions/v1/finalize-analysis-run",
                headers={
                    "X-Shared-Secret": settings.shared_secret,
                    "Content-Type": "application/json",
                },
                json={
                    "analysis_run_id": analysis_run_id,
                    "window_id": window_id,
                    "project_id": project_id,
                    "status": "failed",
                    "error_message": error_message,
                },
            )
    except Exception:
        logger.exception("Falha em finalize-analysis-run (failed) pra run=%s", analysis_run_id)
