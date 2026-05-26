"""Entrypoint do backend FastAPI da CarolinIA."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.agents.orquestrador import (
    list_active_projects,
    orchestrate_project,
    recover_orphaned_demands,
)
from app.api.routes import router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Worker em background — polling 30s
# ============================================================

WORKER_POLL_INTERVAL_SECONDS = 30
_worker_task: asyncio.Task | None = None
_worker_stop_event: asyncio.Event | None = None


async def _orchestration_worker() -> None:
    """Loop infinito que orquestra projetos ativos a cada 30s.
    
    Como funciona:
    1. Lista projetos ativos (execution_status=running, roadmap approved/executing)
    2. Pra cada projeto, chama orchestrate_project()
    3. Aguarda 30s
    4. Repete
    
    Termina quando _worker_stop_event é setado (shutdown do FastAPI).
    """
    logger.info("[worker] Iniciando loop de orquestração (poll=%ds)", WORKER_POLL_INTERVAL_SECONDS)
    
    while _worker_stop_event and not _worker_stop_event.is_set():
        try:
            project_ids = await list_active_projects()
            
            if not project_ids:
                logger.debug("[worker] Nenhum projeto ativo no momento")
            else:
                logger.info("[worker] Orquestrando %d projetos ativos", len(project_ids))
                
                for project_id in project_ids:
                    try:
                        result = await orchestrate_project(project_id)
                        if result.dispatched_count > 0:
                            logger.info(
                                "[worker] project=%s: %d despachadas, %d bloqueadas (paralelismo=%d, file=%d, cost=%d)",
                                project_id,
                                result.dispatched_count,
                                result.blocked_by_parallelism + result.blocked_by_file_conflict + result.blocked_by_cost_limit,
                                result.blocked_by_parallelism,
                                result.blocked_by_file_conflict,
                                result.blocked_by_cost_limit,
                            )
                    except Exception:
                        logger.exception("[worker] Erro orquestrando project=%s", project_id)
                        # Continua pros outros projetos
                        continue
        
        except Exception:
            logger.exception("[worker] Erro no loop principal — segue")
        
        # Espera 30s OU stop event (o que vier primeiro)
        try:
            await asyncio.wait_for(
                _worker_stop_event.wait(),
                timeout=WORKER_POLL_INTERVAL_SECONDS,
            )
            # Se chegou aqui sem timeout, o stop event foi setado
            break
        except asyncio.TimeoutError:
            # Normal — timeout significa "passaram 30s, vamos repetir"
            continue
    
    logger.info("[worker] Loop encerrado")


# ============================================================
# Lifespan — startup e shutdown
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle do FastAPI: startup e shutdown."""
    global _worker_task, _worker_stop_event
    
    print(f"CarolinIA backend iniciando. Workspaces: {settings.workspaces_dir}")
    
    # 1. Recovery: reverte demands órfãs (in_progress → pending)
    try:
        recovered = await recover_orphaned_demands()
        if recovered > 0:
            logger.warning(
                "[startup] Recovery: %d demands órfãs foram revertidas pra pending",
                recovered,
            )
        else:
            logger.info("[startup] Recovery: nenhuma demand órfã encontrada")
    except Exception:
        logger.exception("[startup] Erro em recovery — seguindo")
    
    # 2. Inicia worker em background
    _worker_stop_event = asyncio.Event()
    _worker_task = asyncio.create_task(_orchestration_worker())
    logger.info("[startup] Worker em background iniciado")
    
    yield
    
    # Shutdown
    print("CarolinIA backend encerrando.")
    if _worker_stop_event:
        _worker_stop_event.set()
    
    if _worker_task:
        try:
            await asyncio.wait_for(_worker_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("[shutdown] Worker não terminou em 5s, cancelando")
            _worker_task.cancel()
            try:
                await _worker_task
            except asyncio.CancelledError:
                pass
    
    logger.info("[shutdown] Encerrado")


# ============================================================
# App
# ============================================================

app = FastAPI(
    title="CarolinIA Backend",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(router)


# ============================================================
# Endpoint raiz
# ============================================================

@app.get("/")
async def root():
    return {"service": "carolinia-backend", "status": "ok", "version": "0.2.0"}


# ============================================================
# Webhook de orquestração — disparo imediato (latência < 1s)
# ============================================================

class WebhookOrchestrateRequest(BaseModel):
    project_id: str
    trigger_source: str = "webhook"  # 'webhook' | 'phase_approved' | 'demand_completed' | 'manual'


@app.post("/webhook/orchestrate")
async def webhook_orchestrate(
    payload: WebhookOrchestrateRequest,
    x_shared_secret: str | None = Header(default=None, alias="X-Shared-Secret"),
):
    """Webhook chamado por trigger SQL do Supabase quando:
    - Phase é aprovada
    - Demand completa (libera dependentes)
    - User aciona manualmente
    
    Em vez de esperar o polling de 30s, dispara orquestração imediatamente.
    Fire-and-forget: responde 200 logo, processa em background.
    """
    if not settings.shared_secret:
        raise HTTPException(500, "SHARED_SECRET não configurado")
    if x_shared_secret != settings.shared_secret:
        raise HTTPException(401, "Shared secret inválido")
    
    logger.info(
        "[webhook] Recebido: project=%s source=%s",
        payload.project_id, payload.trigger_source,
    )
    
    # Fire-and-forget: cria task assíncrona e retorna imediatamente
    asyncio.create_task(_run_orchestration_safe(payload.project_id))
    
    return {
        "ok": True,
        "project_id": payload.project_id,
        "trigger_source": payload.trigger_source,
        "message": "Orquestração disparada em background",
    }


async def _run_orchestration_safe(project_id: str) -> None:
    """Wrapper safe pra orquestração disparada via webhook."""
    try:
        result = await orchestrate_project(project_id)
        logger.info(
            "[webhook] Orquestração completou: project=%s dispatched=%d",
            project_id, result.dispatched_count,
        )
    except Exception:
        logger.exception("[webhook] Erro na orquestração de project=%s", project_id)
