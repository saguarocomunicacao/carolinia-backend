"""AgentDBClient — cliente HTTP unificado pros agentes lerem/escreverem 
no banco via Edge Functions do Lovable Cloud.

Os agentes NUNCA falam direto com o Postgres. Sempre via Edge Functions,
que rodam com service_role e bypassam RLS.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class AgentDBClient:
    """Cliente HTTP pros agentes interagirem com o banco via Edge Functions."""
    
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={
                "X-Shared-Secret": settings.shared_secret,
                "Content-Type": "application/json",
            },
        )
    
    # ============================================================
    # agent_actions — registrar e atualizar ações dos agentes
    # ============================================================
    
    async def create_action(
        self,
        project_id: str,
        agent_role: str,
        instruction: str,
        demand_id: str | None = None,
        task_id: str | None = None,
        parent_action_id: str | None = None,
        input_context: str | None = None,
        status: str = "pending",
    ) -> str | None:
        """Cria uma nova agent_action e retorna o ID."""
        url = f"{settings.lovable_project_url}/functions/v1/agent-record-action"
        try:
            response = await self._client.post(url, json={
                "operation": "create",
                "project_id": project_id,
                "agent_role": agent_role,
                "instruction": instruction,
                "demand_id": demand_id,
                "task_id": task_id,
                "parent_action_id": parent_action_id,
                "input_context": input_context,
                "status": status,
            })
            response.raise_for_status()
            data = response.json()
            return data.get("action_id")
        except Exception as e:
            logger.error("Falha criando agent_action: %s", e)
            return None
    
    async def update_action(
        self,
        action_id: str,
        status: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        output_text: str | None = None,
        output_summary: str | None = None,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
        cost_usd: float | None = None,
        latency_ms: int | None = None,
        error_message: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> bool:
        """Atualiza uma agent_action existente. Só envia campos não-None."""
        url = f"{settings.lovable_project_url}/functions/v1/agent-record-action"
        payload: dict[str, Any] = {
            "operation": "update",
            "action_id": action_id,
        }
        for k, v in {
            "status": status, "provider": provider, "model": model,
            "output_text": output_text, "output_summary": output_summary,
            "tokens_input": tokens_input, "tokens_output": tokens_output,
            "cost_usd": cost_usd, "latency_ms": latency_ms,
            "error_message": error_message,
            "started_at": started_at, "completed_at": completed_at,
        }.items():
            if v is not None:
                payload[k] = v
        
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error("Falha atualizando agent_action %s: %s", action_id, e)
            return False
    
    # ============================================================
    # project_context — ler contexto pra alimentar agentes
    # ============================================================
    
    async def fetch_project_context(
        self,
        project_id: str,
        limit: int = 30,
        min_importance: int = 1,
    ) -> list[dict]:
        """Busca top N entradas de contexto do projeto, ordenadas por importância.
        
        Usa a Edge Function get-project-context (a criar) OU faz fallback
        chamando insert-context com modo 'read' se ainda não existir.
        
        Por enquanto retorna vazio — vamos adicionar esse endpoint no
        próximo passo se precisarmos. Os agentes geralmente recebem o
        contexto JÁ INCLUÍDO no payload (a Edge Function process-demand
        já busca os top-30 e envia ao backend), então essa função é
        opcional.
        """
        logger.info(
            "fetch_project_context não implementado ainda. "
            "Use o context_entries recebido em process-demand."
        )
        return []
    
    # ============================================================
    # approvals — criar pontos de aprovação humana
    # ============================================================
    
    async def create_approval(
        self,
        project_id: str,
        approval_type: str,
        title: str,
        summary: str,
        details: str | None = None,
        demand_id: str | None = None,
    ) -> str | None:
        """Cria um ponto de aprovação humana.
        
        Usa a Edge Function agent-create-approval (a criar quando 
        chegarmos no Analista de Briefing — P07.3).
        
        Por enquanto stub — só loga.
        """
        logger.info(
            "[STUB] create_approval: project=%s type=%s title=%s",
            project_id, approval_type, title,
        )
        return None
    
    # ============================================================
    # cleanup
    # ============================================================
    
    async def aclose(self) -> None:
        await self._client.aclose()


# Instância global pros agentes
agent_db = AgentDBClient()
