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
from app.agents.planejador import Planejador
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


class PlanRoadmapRequest(BaseModel):
    project_id: str
    analysis_run_id: str
    source: str = "manual"


class PlanRoadmapResponse(BaseModel):
    accepted: bool = True
    analysis_run_id: str
    message: str = "Planejador iniciado em background"


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


# ============================================================
# /plan-roadmap — endpoint do Planejador
# ============================================================

@router.post("/plan-roadmap", response_model=PlanRoadmapResponse)
async def plan_roadmap_endpoint(
    payload: PlanRoadmapRequest,
    background_tasks: BackgroundTasks,
    x_shared_secret: str | None = Header(default=None, alias="X-Shared-Secret"),
) -> PlanRoadmapResponse:
    """Recebe pedido de planejamento, responde 200 imediato, processa em background."""
    _verify_secret(x_shared_secret)
    
    logger.info(
        "[plan-roadmap] Recebido: project=%s run=%s source=%s",
        payload.project_id, payload.analysis_run_id, payload.source,
    )
    
    background_tasks.add_task(
        _plan_roadmap_task,
        project_id=payload.project_id,
        analysis_run_id=payload.analysis_run_id,
        source=payload.source,
    )
    
    return PlanRoadmapResponse(
        accepted=True,
        analysis_run_id=payload.analysis_run_id,
    )


async def _plan_roadmap_task(
    project_id: str,
    analysis_run_id: str,
    source: str,
) -> None:
    """Task assíncrona — busca contexto + clarificações, roda Planejador, persiste roadmap."""
    marker = f"[planejador project={project_id} run={analysis_run_id}]"
    logger.info("%s Iniciando", marker)
    
    # 1. Busca contexto do projeto
    ctx_data = await fetch_project_context(project_id, min_importance=1)
    if not ctx_data.get("ok"):
        logger.error("%s Falha buscando project_context", marker)
        await _finalize_planning_failed(analysis_run_id, project_id, "Falha buscando project_context")
        return
    
    context_entries = ctx_data.get("entries", [])
    if not context_entries:
        logger.warning("%s Sem context_entries — planejamento pode ficar pobre", marker)
    
    # 2. Busca clarificações respondidas (do Analista)
    clarifications = await _fetch_consolidated_briefing(project_id)
    logger.info("%s Recebidas %d clarificações respondidas", marker, len(clarifications))
    
    # 3. Busca meta do projeto
    project_meta = await _fetch_project_meta(project_id)
    
    # 4. Monta AgentContext
    agent_context = AgentContext(
        project_id=project_id,
        project_name=project_meta.get("name", ""),
        project_stack=project_meta.get("stack", ""),
        project_description=project_meta.get("description", ""),
        context_entries=context_entries,
        extra_context={
            "analysis_run_id": analysis_run_id,
            "source": source,
            "clarifications": clarifications,
            "total_clarifications": len(clarifications),
            "total_context_entries": len(context_entries),
        },
    )
    
    # 5. Instrução do Planejador
    instruction = (
        "Analise o briefing consolidado (contexto do projeto + clarificações respondidas) "
        "e produza um roadmap estruturado em fases. Cada fase deve agrupar demandas "
        "concretas com critérios de aceite, complexidade estimada e dependências entre si. "
        "Siga ESTRITAMENTE o formato JSON descrito no seu system prompt."
    )
    
    # 6. Roda o Planejador
    try:
        planejador = Planejador()
        result = await planejador.run(context=agent_context, instruction=instruction)
    except Exception as e:
        logger.exception("%s Erro rodando Planejador", marker)
        await _finalize_planning_failed(
            analysis_run_id, project_id,
            f"Erro rodando Planejador: {str(e)[:500]}",
        )
        return
    
    if not result.success:
        logger.error("%s Planejador success=False: %s", marker, result.error)
        await _finalize_planning_failed(
            analysis_run_id, project_id,
            f"Planejador falhou: {result.error}",
        )
        return
    
    logger.info(
        "%s Planejador completou: model=%s cost=$%.4f tokens_in=%d tokens_out=%d",
        marker, result.model, result.cost_usd, result.tokens_input, result.tokens_output,
    )
    
    # 7. Parseia output
    try:
        roadmap_output = Planejador.parse_output(result.output_text)
    except Exception as e:
        logger.exception("%s Falha parseando JSON do Planejador", marker)
        logger.error("%s Output bruto: %s", marker, result.output_text[:500])
        await _finalize_planning_failed(
            analysis_run_id, project_id,
            f"Output do Planejador não é JSON válido: {str(e)[:300]}",
        )
        return
    
    if not roadmap_output.phases:
        logger.error("%s Planejador retornou 0 fases — roadmap vazio", marker)
        await _finalize_planning_failed(
            analysis_run_id, project_id,
            "Planejador retornou roadmap vazio (0 fases)",
        )
        return
    
    total_demands = sum(len(p.demands) for p in roadmap_output.phases)
    logger.info(
        "%s Parseou: %d fases, %d demandas total",
        marker, len(roadmap_output.phases), total_demands,
    )
    
    # 8. Persiste no banco via Edge Function persist-roadmap
    persist_payload = {
        "project_id": project_id,
        "analysis_run_id": analysis_run_id,
        "roadmap_summary": roadmap_output.roadmap_summary,
        "phases": [
            {
                "title": p.title,
                "description": p.description,
                "rationale": p.rationale,
                "order": p.order,
                "demands": [
                    {
                        "title": d.title,
                        "description": d.description,
                        "acceptance_criteria": d.acceptance_criteria,
                        "complexity": d.complexity,
                        "order": d.order,
                        "depends_on_titles": d.depends_on_titles,
                    }
                    for d in p.demands
                ],
            }
            for p in roadmap_output.phases
        ],
        "out_of_scope": roadmap_output.out_of_scope,
        "risks": roadmap_output.risks,
        "total_cost_usd": result.cost_usd,
        "total_tokens": result.tokens_input + result.tokens_output,
    }
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            persist_resp = await client.post(
                f"{settings.lovable_project_url}/functions/v1/persist-roadmap",
                headers={
                    "X-Shared-Secret": settings.shared_secret,
                    "Content-Type": "application/json",
                },
                json=persist_payload,
            )
            if persist_resp.status_code != 200:
                logger.error(
                    "%s persist-roadmap retornou %d: %s",
                    marker, persist_resp.status_code, persist_resp.text[:300],
                )
                await _finalize_planning_failed(
                    analysis_run_id, project_id,
                    f"persist-roadmap falhou: {persist_resp.status_code} {persist_resp.text[:200]}",
                )
                return
            persist_data = persist_resp.json()
    except Exception as e:
        logger.exception("%s Exceção chamando persist-roadmap", marker)
        await _finalize_planning_failed(
            analysis_run_id, project_id,
            f"Erro chamando persist-roadmap: {str(e)[:300]}",
        )
        return
    
    logger.info(
        "%s Roadmap persistido: %d phases, %d demands, %d dependências resolvidas, %d não-resolvidas",
        marker,
        persist_data.get("phases_created", 0),
        persist_data.get("demands_created", 0),
        persist_data.get("dependencies_resolved", 0),
        len(persist_data.get("unresolved_dependencies", [])),
    )
    
    logger.info("%s Concluído com sucesso", marker)


async def _fetch_consolidated_briefing(project_id: str) -> list[dict[str, Any]]:
    """Busca clarificações respondidas via Edge Function get-consolidated-briefing."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            response = await client.post(
                f"{settings.lovable_project_url}/functions/v1/get-consolidated-briefing",
                headers={
                    "X-Shared-Secret": settings.shared_secret,
                    "Content-Type": "application/json",
                },
                json={"project_id": project_id},
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("clarifications", [])
            logger.warning("get-consolidated-briefing retornou %d", response.status_code)
    except Exception:
        logger.exception("Falha buscando clarificações do projeto %s", project_id)
    return []


async def _finalize_planning_failed(
    analysis_run_id: str,
    project_id: str,
    error_message: str,
) -> None:
    """Marca analysis_run como failed em caso de falha do Planejador.
    
    Nota: NÃO reverte projects.roadmap_status de 'planning' pra 'not_started'.
    Se UI ficar travada, rodar manualmente no Lovable:
      UPDATE projects SET roadmap_status='not_started' WHERE id='X';
    """
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
                    "project_id": project_id,
                    "status": "failed",
                    "error_message": error_message,
                },
            )
    except Exception:
        logger.exception("Falha em finalize (planning failed) pra run=%s", analysis_run_id)

# ============================================================
# /debug/workspace — endpoint temporário pra validar workspace_manager
# REMOVER após P07.5.b validado
# ============================================================

from app.services.workspace_manager import (
    ensure_workspace_ready,
    create_worktree,
    delete_worktree,
    list_worktrees,
    workspace_health_check,
    get_worktree_path,
    get_branch_name,
)


class WorkspaceDebugRequest(BaseModel):
    operation: str  # 'ensure_ready' | 'create_worktree' | 'delete_worktree' | 'list' | 'health' | 'full_test'
    project_id: str
    repo_id: str
    demand_id: str | None = None
    repo_full_name: str | None = None
    github_token: str | None = None
    default_branch: str = "main"


@router.post("/debug/workspace")
async def workspace_debug_endpoint(
    payload: WorkspaceDebugRequest,
    x_shared_secret: str | None = Header(default=None, alias="X-Shared-Secret"),
) -> dict:
    """ENDPOINT TEMPORÁRIO — valida workspace_manager. REMOVER após P07.5.b."""
    _verify_secret(x_shared_secret)
    
    op = payload.operation
    
    if op == "ensure_ready":
        if not payload.repo_full_name or not payload.github_token:
            raise HTTPException(400, "ensure_ready requer repo_full_name e github_token")
        success, error, paths = await ensure_workspace_ready(
            project_id=payload.project_id,
            repo_id=payload.repo_id,
            repo_full_name=payload.repo_full_name,
            github_token=payload.github_token,
            default_branch=payload.default_branch,
        )
        return {
            "operation": op,
            "success": success,
            "error": error,
            "paths": {
                "root": str(paths.root),
                "main_clone": str(paths.main_clone),
                "worktrees_dir": str(paths.worktrees_dir),
            },
        }
    
    elif op == "create_worktree":
        if not payload.demand_id:
            raise HTTPException(400, "create_worktree requer demand_id")
        success, error, info = await create_worktree(
            project_id=payload.project_id,
            repo_id=payload.repo_id,
            demand_id=payload.demand_id,
            default_branch=payload.default_branch,
        )
        return {
            "operation": op,
            "success": success,
            "error": error,
            "worktree": {
                "demand_id": info.demand_id,
                "path": str(info.path),
                "branch_name": info.branch_name,
            } if info else None,
        }
    
    elif op == "delete_worktree":
        if not payload.demand_id:
            raise HTTPException(400, "delete_worktree requer demand_id")
        success, error = await delete_worktree(
            project_id=payload.project_id,
            repo_id=payload.repo_id,
            demand_id=payload.demand_id,
            force=True,
        )
        return {"operation": op, "success": success, "error": error}
    
    elif op == "list":
        worktrees = await list_worktrees(payload.project_id, payload.repo_id)
        return {"operation": op, "worktrees": worktrees, "count": len(worktrees)}
    
    elif op == "health":
        health = await workspace_health_check(payload.project_id, payload.repo_id)
        return {"operation": op, "health": health}
    
    elif op == "full_test":
        if not payload.repo_full_name or not payload.github_token:
            raise HTTPException(400, "full_test requer repo_full_name e github_token")
        if not payload.demand_id:
            raise HTTPException(400, "full_test requer demand_id (será criada worktree de teste)")
        
        results = {}
        
        # 1. ensure_ready
        success, error, paths = await ensure_workspace_ready(
            project_id=payload.project_id,
            repo_id=payload.repo_id,
            repo_full_name=payload.repo_full_name,
            github_token=payload.github_token,
            default_branch=payload.default_branch,
        )
        results["1_ensure_ready"] = {
            "success": success, "error": error,
            "main_clone": str(paths.main_clone),
        }
        if not success:
            return {"operation": op, "results": results, "stopped_at_step": 1}
        
        # 2. create_worktree
        success, error, info = await create_worktree(
            project_id=payload.project_id,
            repo_id=payload.repo_id,
            demand_id=payload.demand_id,
            default_branch=payload.default_branch,
        )
        results["2_create_worktree"] = {
            "success": success, "error": error,
            "worktree_path": str(info.path) if info else None,
            "branch_name": info.branch_name if info else None,
        }
        if not success:
            return {"operation": op, "results": results, "stopped_at_step": 2}
        
        # 3. health check
        health = await workspace_health_check(payload.project_id, payload.repo_id)
        results["3_health"] = health
        
        # 4. list worktrees
        worktrees = await list_worktrees(payload.project_id, payload.repo_id)
        results["4_list_worktrees"] = {"count": len(worktrees), "worktrees": worktrees}
        
        # 5. delete worktree
        success, error = await delete_worktree(
            project_id=payload.project_id,
            repo_id=payload.repo_id,
            demand_id=payload.demand_id,
            force=True,
        )
        results["5_delete_worktree"] = {"success": success, "error": error}
        
        # 6. health check final
        health_after = await workspace_health_check(payload.project_id, payload.repo_id)
        results["6_health_after"] = health_after
        
        return {"operation": op, "results": results}
    
    else:
        raise HTTPException(
            400,
            f"Operação inválida: {op}. Aceitos: ensure_ready, create_worktree, delete_worktree, list, health, full_test",
        )
