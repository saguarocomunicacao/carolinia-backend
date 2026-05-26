"""Orquestrador — decide quais demandas executar e quando.

NÃO é um agente IA (não chama LLM). É um worker estrutural que aplica
regras de elegibilidade, paralelismo e limites de custo pra orquestrar
as demandas aprovadas do roadmap.

Responsabilidades:
1. Identificar demandas elegíveis pra execução:
   - phase.status='approved'
   - demand.status='pending'
   - todas demands em demand.depends_on estão 'completed'
   - project.execution_status='running'
2. Aplicar regras de paralelismo:
   - máximo settings.max_parallel_demands_per_project demandas simultaneamente
   - demandas com expected_files comuns viram sequenciais
3. Verificar limites de custo:
   - project.max_cost_usd_total NÃO foi atingido
4. Despachar demanda (marcar como in_progress + criar task_run)
5. Simular execução (P07.5.d temporário) ou plugar agentes reais (P07.5.e+)

Pra P07.5.d (esse arquivo) a execução é SIMULADA:
- Marca demand 'in_progress'
- Sleep settings.simulated_execution_seconds
- Marca demand 'completed'
- Libera dependentes

Isso valida toda a lógica de orquestração ANTES de plugar PM/Dev/Tester reais.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from app.agents.db_client import agent_db
from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# Tipos
# ============================================================

@dataclass
class EligibleDemand:
    """Demand pronta pra execução."""
    id: str
    project_id: str
    phase_id: str
    title: str
    complexity: str
    expected_files: list[str]
    depends_on: list[str]
    phase_order: int
    demand_order: int


@dataclass
class OrchestrationResult:
    """Resultado de uma rodada do orquestrador pra um projeto."""
    project_id: str
    eligible_count: int
    dispatched_count: int
    blocked_by_parallelism: int
    blocked_by_file_conflict: int
    blocked_by_cost_limit: int
    error: str | None = None


# ============================================================
# Helpers de banco — via Edge Functions
# ============================================================

async def _fetch_eligible_demands(project_id: str) -> list[EligibleDemand]:
    """Busca demandas elegíveis via Edge Function get-eligible-demands."""
    url = f"{settings.lovable_project_url}/functions/v1/get-eligible-demands"
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            response = await client.post(
                url,
                json={"project_id": project_id},
                headers={
                    "X-Shared-Secret": settings.shared_secret,
                    "Content-Type": "application/json",
                },
            )
            if response.status_code != 200:
                logger.error(
                    "[orquestrador] get-eligible-demands retornou %d: %s",
                    response.status_code, response.text[:200],
                )
                return []
            data = response.json()
            
            demands_raw = data.get("demands", [])
            return [
                EligibleDemand(
                    id=d["id"],
                    project_id=d["project_id"],
                    phase_id=d["phase_id"],
                    title=d["title"],
                    complexity=d.get("complexity") or "M",
                    expected_files=d.get("expected_files") or [],
                    depends_on=d.get("depends_on") or [],
                    phase_order=d.get("phase_order_in_project") or 0,
                    demand_order=d.get("phase_order") or 0,
                )
                for d in demands_raw
            ]
    except Exception:
        logger.exception("[orquestrador] Erro buscando eligible demands")
        return []


async def _fetch_in_progress_demands(project_id: str) -> list[dict]:
    """Busca demands com status='in_progress' do projeto."""
    url = f"{settings.lovable_project_url}/functions/v1/get-in-progress-demands"
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            response = await client.post(
                url,
                json={"project_id": project_id},
                headers={
                    "X-Shared-Secret": settings.shared_secret,
                    "Content-Type": "application/json",
                },
            )
            if response.status_code != 200:
                logger.error(
                    "[orquestrador] get-in-progress-demands retornou %d",
                    response.status_code,
                )
                return []
            return response.json().get("demands", [])
    except Exception:
        logger.exception("[orquestrador] Erro buscando in-progress demands")
        return []


async def _fetch_project_cost_status(project_id: str) -> dict[str, Any]:
    """Busca status de custo do projeto."""
    url = f"{settings.lovable_project_url}/functions/v1/get-project-cost-status"
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.post(
                url,
                json={"project_id": project_id},
                headers={
                    "X-Shared-Secret": settings.shared_secret,
                    "Content-Type": "application/json",
                },
            )
            if response.status_code != 200:
                return {"cost_limit_reached": False, "total_cost_usd": 0, "max_cost_usd_total": 50}
            return response.json()
    except Exception:
        logger.exception("[orquestrador] Erro buscando cost status")
        return {"cost_limit_reached": False, "total_cost_usd": 0, "max_cost_usd_total": 50}


async def _mark_demand_in_progress(demand_id: str) -> bool:
    """Marca demand como in_progress."""
    url = f"{settings.lovable_project_url}/functions/v1/update-demand-status"
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.post(
                url,
                json={
                    "demand_id": demand_id,
                    "new_status": "in_progress",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                headers={
                    "X-Shared-Secret": settings.shared_secret,
                    "Content-Type": "application/json",
                },
            )
            return response.status_code == 200
    except Exception:
        logger.exception("[orquestrador] Erro marcando demand=%s in_progress", demand_id)
        return False


async def _mark_demand_completed(demand_id: str) -> bool:
    """Marca demand como completed."""
    url = f"{settings.lovable_project_url}/functions/v1/update-demand-status"
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.post(
                url,
                json={
                    "demand_id": demand_id,
                    "new_status": "completed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                headers={
                    "X-Shared-Secret": settings.shared_secret,
                    "Content-Type": "application/json",
                },
            )
            return response.status_code == 200
    except Exception:
        logger.exception("[orquestrador] Erro marcando demand=%s completed", demand_id)
        return False


# ============================================================
# Lógica de paralelismo e conflitos
# ============================================================

def _has_file_conflict(
    demand: EligibleDemand,
    in_progress: list[dict],
) -> bool:
    """True se a demand tem expected_files em comum com alguma demand 
    já em execução.
    """
    if not demand.expected_files:
        return False
    
    demand_files = set(demand.expected_files)
    
    for ip in in_progress:
        ip_files = set(ip.get("expected_files") or [])
        if not ip_files:
            continue
        if demand_files & ip_files:
            return True
    
    return False


def _filter_dispatchable(
    eligible: list[EligibleDemand],
    in_progress: list[dict],
    max_parallel: int,
) -> tuple[list[EligibleDemand], dict[str, int]]:
    """Filtra demands elegíveis aplicando paralelismo e file conflicts."""
    counters = {
        "blocked_by_parallelism": 0,
        "blocked_by_file_conflict": 0,
    }
    
    available_slots = max_parallel - len(in_progress)
    if available_slots <= 0:
        counters["blocked_by_parallelism"] = len(eligible)
        return [], counters
    
    files_being_worked: set[str] = set()
    for ip in in_progress:
        files_being_worked.update(ip.get("expected_files") or [])
    
    to_dispatch = []
    
    for demand in eligible:
        if len(to_dispatch) >= available_slots:
            counters["blocked_by_parallelism"] += 1
            continue
        
        demand_files = set(demand.expected_files or [])
        if demand_files & files_being_worked:
            counters["blocked_by_file_conflict"] += 1
            continue
        
        to_dispatch.append(demand)
        files_being_worked.update(demand_files)
    
    return to_dispatch, counters


# ============================================================
# Simulação de execução (P07.5.d — temporário)
# ============================================================

async def _simulate_demand_execution(demand: EligibleDemand) -> None:
    """Simula execução: sleep + completed. Vai ser substituída por dispatch 
    real ao PM em P07.5.e.
    """
    logger.info(
        "[orquestrador] SIMULANDO execução demand=%s (%s) — %ds",
        demand.id, demand.title, settings.simulated_execution_seconds,
    )
    
    success = await _mark_demand_in_progress(demand.id)
    if not success:
        logger.error("[orquestrador] Falha marcando demand=%s in_progress", demand.id)
        return
    
    action_id = await agent_db.create_action(
        project_id=demand.project_id,
        agent_role="orquestrador",
        instruction=f"[SIMULAÇÃO] Despachar demand '{demand.title}'",
        demand_id=demand.id,
        status="running",
    )
    
    started_at = datetime.now(timezone.utc).isoformat()
    if action_id:
        await agent_db.update_action(
            action_id=action_id,
            status="running",
            started_at=started_at,
        )
    
    await asyncio.sleep(settings.simulated_execution_seconds)
    
    success = await _mark_demand_completed(demand.id)
    if not success:
        logger.error("[orquestrador] Falha marcando demand=%s completed", demand.id)
        if action_id:
            await agent_db.update_action(
                action_id=action_id,
                status="failed",
                error_message="Falha marcando demand completed",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        return
    
    completed_at = datetime.now(timezone.utc).isoformat()
    if action_id:
        await agent_db.update_action(
            action_id=action_id,
            status="completed",
            output_summary=f"[SIMULAÇÃO] Demand '{demand.title}' completou em {settings.simulated_execution_seconds}s",
            completed_at=completed_at,
        )
    
    logger.info(
        "[orquestrador] Demand=%s completed (simulação)",
        demand.id,
    )


# ============================================================
# API pública
# ============================================================

async def orchestrate_project(project_id: str) -> OrchestrationResult:
    """Executa uma rodada de orquestração pra um projeto."""
    logger.info("[orquestrador] Iniciando rodada pra projeto=%s", project_id)
    
    eligible = await _fetch_eligible_demands(project_id)
    if not eligible:
        return OrchestrationResult(
            project_id=project_id,
            eligible_count=0,
            dispatched_count=0,
            blocked_by_parallelism=0,
            blocked_by_file_conflict=0,
            blocked_by_cost_limit=0,
        )
    
    cost_status = await _fetch_project_cost_status(project_id)
    if cost_status.get("cost_limit_reached"):
        logger.warning(
            "[orquestrador] Projeto=%s atingiu limite de custo ($%.2f de $%.2f). Pausando.",
            project_id,
            cost_status.get("total_cost_usd", 0),
            cost_status.get("max_cost_usd_total", 0),
        )
        return OrchestrationResult(
            project_id=project_id,
            eligible_count=len(eligible),
            dispatched_count=0,
            blocked_by_parallelism=0,
            blocked_by_file_conflict=0,
            blocked_by_cost_limit=len(eligible),
        )
    
    in_progress = await _fetch_in_progress_demands(project_id)
    
    to_dispatch, counters = _filter_dispatchable(
        eligible=eligible,
        in_progress=in_progress,
        max_parallel=settings.max_parallel_demands_per_project,
    )
    
    logger.info(
        "[orquestrador] Projeto=%s: %d elegíveis, %d em execução, %d a despachar "
        "(bloqueados: %d paralelismo, %d file conflict)",
        project_id,
        len(eligible), len(in_progress), len(to_dispatch),
        counters["blocked_by_parallelism"],
        counters["blocked_by_file_conflict"],
    )
    
    for demand in to_dispatch:
        asyncio.create_task(_simulate_demand_execution(demand))
    
    return OrchestrationResult(
        project_id=project_id,
        eligible_count=len(eligible),
        dispatched_count=len(to_dispatch),
        blocked_by_parallelism=counters["blocked_by_parallelism"],
        blocked_by_file_conflict=counters["blocked_by_file_conflict"],
        blocked_by_cost_limit=0,
    )


async def recover_orphaned_demands() -> int:
    """Reverte demands órfãs (in_progress → pending) na inicialização."""
    url = f"{settings.lovable_project_url}/functions/v1/recover-orphaned-demands"
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.post(
                url,
                json={},
                headers={
                    "X-Shared-Secret": settings.shared_secret,
                    "Content-Type": "application/json",
                },
            )
            if response.status_code != 200:
                logger.error(
                    "[orquestrador] recover-orphaned-demands retornou %d",
                    response.status_code,
                )
                return 0
            data = response.json()
            count = data.get("recovered_count", 0)
            if count > 0:
                logger.warning(
                    "[orquestrador] Recovery: %d demands órfãs voltaram pra pending",
                    count,
                )
            return count
    except Exception:
        logger.exception("[orquestrador] Erro em recover_orphaned_demands")
        return 0


async def list_active_projects() -> list[str]:
    """Lista project_ids ativos pro worker."""
    url = f"{settings.lovable_project_url}/functions/v1/list-active-projects"
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.post(
                url,
                json={},
                headers={
                    "X-Shared-Secret": settings.shared_secret,
                    "Content-Type": "application/json",
                },
            )
            if response.status_code != 200:
                return []
            return response.json().get("project_ids", [])
    except Exception:
        logger.exception("[orquestrador] Erro listando active projects")
        return []
