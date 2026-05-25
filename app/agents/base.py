"""BaseAgent — classe base que todos os 8 agentes do CarolinIA estendem.

Cada agente especializado:
1. Define seu `role` (enum agent_role)
2. Define seus modelos preferidos (preset do LLMRouter)
3. Define seu system prompt
4. Opcionalmente sobrescreve build_user_message() pra customizar o input

O BaseAgent cuida de:
- Registrar a ação em agent_actions (status pending → running → completed/failed)
- Chamar o LLMRouter
- Gravar resultado com tokens, custo e latência
- Tratar erros gracefully
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.agents.db_client import agent_db
from app.services.llm_router import LLMResponse, ModelChoice, call_llm

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Contexto que todo agente recebe pra trabalhar.
    
    project_id: obrigatório — agentes sempre rodam no contexto de um projeto
    demand_id: opcional — null pro Analista de Briefing (que roda antes de qualquer demanda)
    task_id: opcional — só presente quando PM, Devs ou Tester estão executando uma task
    parent_action_id: opcional — pra rastreabilidade do hand-off (qual agente chamou esse)
    context_entries: lista de entradas de project_context relevantes
    extra_context: dict livre pra cada agente adicionar info específica
    """
    project_id: str
    project_name: str = ""
    project_stack: str = ""
    project_description: str = ""
    demand_id: str | None = None
    demand_title: str = ""
    demand_description: str = ""
    task_id: str | None = None
    parent_action_id: str | None = None
    context_entries: list[dict] = None
    extra_context: dict[str, Any] = None
    
    def __post_init__(self):
        if self.context_entries is None:
            self.context_entries = []
        if self.extra_context is None:
            self.extra_context = {}


@dataclass
class AgentResult:
    """Resultado de uma execução de agente."""
    success: bool
    output_text: str = ""
    output_summary: str = ""
    action_id: str | None = None
    error: str | None = None
    cost_usd: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: int = 0
    provider: str = ""
    model: str = ""


class BaseAgent:
    """Classe base pros agentes do CarolinIA.
    
    Subclasses devem definir:
    - role: str (valor do enum agent_role)
    - model_preferences: list[ModelChoice]
    - system_prompt: str
    
    E opcionalmente sobrescrever:
    - build_user_message(context) -> str
    - summarize_output(output_text) -> str
    """
    
    # Subclasses sobrescrevem
    role: str = ""
    model_preferences: list[ModelChoice] = []
    system_prompt: str = ""
    
    # Configuração de chamada (defaults razoáveis, subclasses podem mudar)
    max_tokens: int = 4096
    temperature: float = 1.0
    
    def __init__(self) -> None:
        if not self.role:
            raise ValueError(f"{self.__class__.__name__}: role não definido")
        if not self.model_preferences:
            raise ValueError(f"{self.__class__.__name__}: model_preferences vazio")
        if not self.system_prompt:
            raise ValueError(f"{self.__class__.__name__}: system_prompt não definido")
    
    # ============================================================
    # Métodos pra subclasses customizarem
    # ============================================================
    
    def build_user_message(self, context: AgentContext) -> str:
        """Monta a mensagem do usuário a partir do contexto.
        
        Default: junta projeto + demanda + entradas de contexto num
        formato legível. Subclasses podem sobrescrever pra estruturar
        diferente.
        """
        parts = []
        
        # Contexto do projeto
        parts.append("## Projeto")
        parts.append(f"Nome: {context.project_name or 'Sem nome'}")
        if context.project_stack:
            parts.append(f"Stack: {context.project_stack}")
        if context.project_description:
            parts.append(f"Descrição: {context.project_description}")
        
        # Contexto da demanda (se houver)
        if context.demand_id:
            parts.append("\n## Demanda atual")
            parts.append(f"Título: {context.demand_title}")
            parts.append(f"Descrição: {context.demand_description}")
        
        # Entradas de contexto relevantes (top 20 por importância)
        if context.context_entries:
            sorted_entries = sorted(
                context.context_entries,
                key=lambda x: x.get("importance", 5),
                reverse=True,
            )[:20]
            parts.append(f"\n## Contexto disponível ({len(sorted_entries)} entradas)")
            for entry in sorted_entries:
                title = entry.get("title", "sem título")
                imp = entry.get("importance", 5)
                content = entry.get("content", "")[:3000]
                parts.append(f"\n### {title} (importance: {imp})")
                parts.append(f"```\n{content}\n```")
        
        # Extra context específico do agente
        if context.extra_context:
            parts.append("\n## Contexto adicional")
            for k, v in context.extra_context.items():
                parts.append(f"- **{k}**: {v}")
        
        return "\n".join(parts)
    
    def summarize_output(self, output_text: str) -> str:
        """Gera um resumo curto do output pra UI (300 chars).
        
        Default: primeiros 300 chars da resposta.
        Subclasses podem sobrescrever pra extrair título/conclusão.
        """
        text = output_text.strip()
        if len(text) <= 300:
            return text
        return text[:297] + "..."
    
    # ============================================================
    # Execução (não sobrescrever)
    # ============================================================
    
    async def run(
        self,
        context: AgentContext,
        instruction: str,
    ) -> AgentResult:
        """Executa o agente: registra no banco, chama LLM, grava resultado.
        
        Args:
            context: contexto completo da execução
            instruction: o que esse agente deve fazer especificamente
                         (ex: "Analise o briefing e liste perguntas")
        
        Returns:
            AgentResult com sucesso/erro, output, métricas
        """
        # 1. Cria registro inicial em agent_actions
        action_id = await agent_db.create_action(
            project_id=context.project_id,
            agent_role=self.role,
            instruction=instruction,
            demand_id=context.demand_id,
            task_id=context.task_id,
            parent_action_id=context.parent_action_id,
            status="pending",
        )
        
        if not action_id:
            logger.error(
                "[%s] Falha ao criar agent_action — abortando execução",
                self.role,
            )
            return AgentResult(
                success=False,
                error="Falha ao criar registro de agent_action",
            )
        
        # 2. Marca como running
        started_at = datetime.now(timezone.utc).isoformat()
        await agent_db.update_action(
            action_id=action_id,
            status="running",
            started_at=started_at,
        )
        
        logger.info(
            "[%s] Iniciando execução (action_id=%s, project=%s, demand=%s)",
            self.role, action_id, context.project_id, context.demand_id,
        )
        
        # 3. Monta o user_message a partir do contexto + instrução
        user_message = self.build_user_message(context)
        user_message += f"\n\n## Sua tarefa agora\n{instruction}"
        
        # 4. Chama o LLMRouter
        try:
            llm_response: LLMResponse = call_llm(
                system_prompt=self.system_prompt,
                user_message=user_message,
                model_preferences=self.model_preferences,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            error_msg = str(e)[:1000]
            logger.exception("[%s] Falha na chamada LLM", self.role)
            
            await agent_db.update_action(
                action_id=action_id,
                status="failed",
                error_message=error_msg,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            
            return AgentResult(
                success=False,
                action_id=action_id,
                error=error_msg,
            )
        
        # 5. Grava resultado
        output_summary = self.summarize_output(llm_response.text)
        completed_at = datetime.now(timezone.utc).isoformat()
        
        await agent_db.update_action(
            action_id=action_id,
            status="completed",
            provider=llm_response.provider.value,
            model=llm_response.model,
            output_text=llm_response.text,
            output_summary=output_summary,
            tokens_input=llm_response.tokens_input,
            tokens_output=llm_response.tokens_output,
            cost_usd=llm_response.cost_usd,
            latency_ms=llm_response.latency_ms,
            completed_at=completed_at,
        )
        
        logger.info(
            "[%s] Completou (action_id=%s, provider=%s, model=%s, "
            "cost=$%.4f, lat=%dms, in=%d, out=%d)",
            self.role, action_id, llm_response.provider.value, llm_response.model,
            llm_response.cost_usd, llm_response.latency_ms,
            llm_response.tokens_input, llm_response.tokens_output,
        )
        
        return AgentResult(
            success=True,
            output_text=llm_response.text,
            output_summary=output_summary,
            action_id=action_id,
            cost_usd=llm_response.cost_usd,
            tokens_input=llm_response.tokens_input,
            tokens_output=llm_response.tokens_output,
            latency_ms=llm_response.latency_ms,
            provider=llm_response.provider.value,
            model=llm_response.model,
        )
