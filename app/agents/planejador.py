"""Planejador — agente que propõe roadmap de demandas em fases.

É o segundo agente do pipeline. Roda depois que o briefing está
consolidado (todas as clarificações do Analista foram respondidas).

Sua função é ler:
- Briefing consolidado (perguntas + respostas)
- Todo project_context (PDFs, repos)
- Demandas existentes (clarificações já respondidas)

E produzir:
- Roadmap dividido em fases lógicas
- Cada fase com N demandas estruturadas
- Cada demanda com: título, descrição, critérios de aceite,
  complexidade (S/M/L/XL) e dependências

IMPORTANTE: o Planejador NÃO propõe arquitetura técnica. Foca em
"o-que-fazer", deixa o "como-fazer" pro Senior Dev.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.agents.base import AgentContext, BaseAgent
from app.services.llm_router import PRESET_ANALISE_DENSA

logger = logging.getLogger(__name__)


# ============================================================
# Output esperado do modelo
# ============================================================

@dataclass
class PlannedDemand:
    """Uma demanda proposta pelo Planejador, dentro de uma fase."""
    title: str
    description: str
    acceptance_criteria: str
    complexity: str          # 'S' | 'M' | 'L' | 'XL'
    order: int               # Ordem dentro da fase
    depends_on_titles: list[str] = field(default_factory=list)


@dataclass
class PlannedPhase:
    """Uma fase do roadmap."""
    title: str
    description: str
    rationale: str           # Por que essa fase existe / o que ela entrega
    order: int               # Ordem da fase no roadmap
    demands: list[PlannedDemand] = field(default_factory=list)


@dataclass
class RoadmapOutput:
    """Estrutura completa do output do Planejador."""
    roadmap_summary: str
    phases: list[PlannedPhase]
    out_of_scope: list[str]
    risks: list[str]
    notes: str


# ============================================================
# Planejador
# ============================================================

class Planejador(BaseAgent):
    """Agente que propõe roadmap de demandas em fases."""
    
    role = "planejador"
    model_preferences = PRESET_ANALISE_DENSA  # Opus → GPT-4o → Gemini Pro
    
    system_prompt = """Você é o PLANEJADOR da CarolinIA, uma plataforma multi-agente.

# Seu papel

Você é o segundo agente que entra em ação. O Analista de Briefing já 
trabalhou: leu o material do projeto e fez perguntas críticas; o cliente 
respondeu. Agora o briefing está consolidado.

Sua função é transformar esse briefing num ROADMAP DE EXECUÇÃO — uma 
sequência clara de fases, cada uma com demandas concretas que o time 
de agentes (Senior Dev, Junior Dev, Tester, Revisor) vai executar.

Você é pragmático. NÃO é arquiteto de software — não decide stack, libs, 
padrões de código ou estrutura técnica. Isso é tarefa do Senior Dev 
DENTRO de cada demanda. Você decide O QUE precisa ser feito e EM QUE 
ORDEM.

# Como você pensa em fases

Uma boa fase:
- Tem um OBJETIVO claro ("Setup e fundações", "MVP funcional", 
  "Refinamento e polish")
- Entrega VALOR INCREMENTAL — se o projeto parar no fim de uma fase, 
  algo útil já foi produzido
- Tem entre 3 e 10 demandas (menos é fase trivial, mais é fase inchada)
- Depende SÓ de fases anteriores, não das próximas

Heurísticas comuns de divisão em fases:
1. **Setup** — autenticação, base do projeto, conexões com serviços externos
2. **Core** — funcionalidade principal que justifica o projeto
3. **Periféricos** — features que enriquecem mas não são bloqueadores
4. **Polish** — UX, performance, refinamento visual, documentação

Mas NÃO siga essas heurísticas cegamente. Cada projeto tem sua lógica.

Projeto pequeno (< 10 demandas total) → pode ser 1 fase única.
Projeto médio (10-30 demandas) → 2-4 fases.
Projeto grande (30-100 demandas) → 4-8 fases.
Projeto muito grande → quebra em fases menores em vez de inchar uma.

# Como você pensa em demandas

Uma demanda boa:
- Tem TÍTULO ACIONÁVEL ("Implementar login com Google" — não 
  "Autenticação")
- Tem ESCOPO FECHADO — sabe quando termina
- Tem CRITÉRIOS DE ACEITE concretos (testáveis)
- Tem COMPLEXIDADE estimada:
  - **S** (Small) — algumas horas, low-risk, padrão. Junior Dev consegue.
  - **M** (Medium) — 1-2 dias, requer entendimento mais profundo. Senior 
    ou Junior+revisão.
  - **L** (Large) — semana, design técnico necessário. Senior Dev.
  - **XL** (Extra Large) — múltiplas semanas, ARQUITETURA pesada. 
    QUEBRE em demandas menores em vez de marcar XL. Use XL só quando 
    quebrar não faz sentido.

# Como você pensa em dependências

Cada demanda pode depender de outras (que precisam estar PRONTAS antes). 
Use isso pra:
- Sinalizar bloqueios reais ("Implementar dashboard depende de "Setup 
  autenticação")
- Permitir paralelismo: demandas sem dependências comuns podem rodar 
  em paralelo

Liste dependências pelo TÍTULO da demanda dependida (não por ID — você 
não tem IDs ainda).

NÃO crie dependências artificiais. Se duas demandas podem rodar em 
paralelo, deixe sem depender uma da outra. Paralelismo é bom.

# O que NÃO fazer

- NÃO propor arquitetura técnica ("usar React + Vite + Tailwind"). 
  Não é teu papel.
- NÃO repetir as perguntas que o Analista fez. Elas já viraram demandas 
  do tipo "Briefing — clarificações" e foram respondidas. Foca no QUE 
  FAZER agora.
- NÃO inflar com demandas filler ("Documentar tudo" sem razão clara, 
  "Setup CI/CD" se o projeto é pequeno). Cada demanda deve ter motivo.
- NÃO inventar features que o cliente não pediu nem implicou. Se o 
  briefing diz "site landing page", não propõe "área administrativa 
  com login" porque seria bonito.

# Saída em JSON estruturado

Sua resposta DEVE ser um único bloco JSON válido, sem nenhum texto 
fora do JSON. Estrutura:

```json
{
  "roadmap_summary": "3-5 frases resumindo a estratégia. Seja específico.",
  "phases": [
    {
      "title": "Nome da fase — 3-6 palavras",
      "description": "1-2 frases sobre o que essa fase entrega",
      "rationale": "Por que essa fase existe e essa ordem específica",
      "order": 1,
      "demands": [
        {
          "title": "Título acionável da demanda — 5-10 palavras",
          "description": "Descrição detalhada do que precisa ser feito (2-5 frases)",
          "acceptance_criteria": "Lista de critérios verificáveis. Use marcadores. Ex: '- Login funciona com email/senha\\n- Senhas hashadas com bcrypt\\n- JWT expira em 24h'",
          "complexity": "S|M|L|XL",
          "order": 1,
          "depends_on_titles": ["Título de outra demanda já listada"]
        }
      ]
    }
  ],
  "out_of_scope": [
    "Item que você entendeu que NÃO faz parte (com base no briefing)"
  ],
  "risks": [
    "Risco identificado. Pode ser técnico ('integração com API X tem rate limit baixo') ou de negócio ('público parece amplo demais, MVP pode não validar todas hipóteses')"
  ],
  "notes": "Observações livres. Pode ficar vazio."
}
```

# Regras finais

- Máximo de 12 fases por roadmap. Idealmente 3-7.
- Máximo de 15 demandas por fase. Idealmente 5-10.
- Total geral: máximo 50 demandas. Mais que isso ESGOTA SEU LIMITE DE TOKENS 
  e o JSON é truncado — fica inválido e perde tudo. Se o projeto é gigantesco, 
  proponha PRIMEIRA ITERAÇÃO do roadmap (Fundações + Core) e indica em 
  out_of_scope que "Polish, integrações secundárias e features avançadas 
  serão planejadas em iteração futura após validação do MVP".

- BUDGET DE TOKENS: cada demand consome ~300-400 tokens (title + description 
  + acceptance_criteria). Com max_tokens=16k disponível pro JSON, você tem 
  margem confortável pra ~40 demands BEM DESCRITAS ou ~60 demands COMPACTAS. 
  Sempre PRIORIZE QUALIDADE sobre quantidade.

- FINALIZE O JSON SEMPRE. Se você sentir que está chegando perto do limite, 
  REDUZA o detalhamento das próximas demands em vez de truncar. Um JSON 
  completo com 25 demands é INFINITAMENTE melhor que um JSON truncado com 
  40 demands.

- Português brasileiro, tom direto, sem clichê.
- Trata o cliente por "você".
- NÃO INCLUA TEXTO FORA DO JSON.
"""

    max_tokens: int = 16384  # Roadmap denso pode passar de 8k; Opus 4.7 aceita até 32k
    temperature: float = 0.7
    
    # ============================================================
    # Customização específica do Planejador
    # ============================================================
    
    def build_user_message(self, context: AgentContext) -> str:
        """Monta input com TODO contexto + clarificações respondidas."""
        parts = []
        
        # Cabeçalho
        parts.append("# Briefing consolidado do projeto")
        parts.append(f"\n**Projeto:** {context.project_name or 'Sem nome'}")
        if context.project_stack:
            parts.append(f"**Stack/contexto técnico declarado:** {context.project_stack}")
        if context.project_description:
            parts.append(f"**Descrição inicial:** {context.project_description}")
        
        # Material indexado
        if context.context_entries:
            sorted_entries = sorted(
                context.context_entries,
                key=lambda x: x.get("importance", 5),
                reverse=True,
            )
            parts.append(f"\n## Materiais indexados ({len(sorted_entries)} entradas)\n")
            for entry in sorted_entries:
                title = entry.get("title", "sem título")
                imp = entry.get("importance", 5)
                source = entry.get("source_type", "?")
                content = entry.get("content", "")
                if len(content) > 5000:
                    content = content[:5000] + "\n[... conteúdo truncado ...]"
                parts.append(f"\n### {title}")
                parts.append(f"_Origem: {source} · Importância: {imp}/10_\n")
                parts.append(f"```\n{content}\n```")
        
        # Clarificações respondidas (vêm via extra_context)
        clarifications = context.extra_context.get("clarifications", []) if context.extra_context else []
        if clarifications:
            parts.append("\n## Clarificações do briefing (Analista perguntou, cliente respondeu)")
            for c in clarifications:
                question = c.get("question", "")
                answer = c.get("answer", "")
                category = c.get("category", "other")
                priority = c.get("priority", "medium")
                parts.append(f"\n### [{priority}/{category}] {question}")
                parts.append(f"**Resposta do cliente:** {answer}")
        
        # Contexto adicional
        if context.extra_context:
            extra = {k: v for k, v in context.extra_context.items() if k != "clarifications"}
            if extra:
                parts.append("\n## Contexto da análise")
                for k, v in extra.items():
                    parts.append(f"- **{k}**: {v}")
        
        return "\n".join(parts)
    
    def summarize_output(self, output_text: str) -> str:
        """Extrai resumo + contagem de fases/demandas."""
        try:
            parsed = self.parse_output(output_text)
            total_demands = sum(len(p.demands) for p in parsed.phases)
            summary = parsed.roadmap_summary
            if len(summary) > 250:
                summary = summary[:247] + "..."
            return f"[{len(parsed.phases)} fases, {total_demands} demandas] {summary}"
        except Exception:
            return output_text[:300] + ("..." if len(output_text) > 300 else "")
    
    @staticmethod
    def parse_output(output_text: str) -> RoadmapOutput:
        """Parseia JSON do modelo numa estrutura tipada.
        
        Tolera:
        - JSON dentro de bloco markdown ```json ... ```
        - Texto antes/depois do JSON
        - Campos faltantes (preenche com defaults)
        """
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            start = output_text.find("{")
            end = output_text.rfind("}")
            if start == -1 or end == -1 or end < start:
                raise ValueError(f"Output não contém JSON parseável: {output_text[:200]}")
            json_str = output_text[start:end + 1]
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido: {e}. Conteúdo: {json_str[:300]}")
        
        # Parseia fases
        phases_raw = data.get("phases", [])
        phases: list[PlannedPhase] = []
        for idx, p in enumerate(phases_raw):
            demands_raw = p.get("demands", [])
            demands: list[PlannedDemand] = []
            for didx, d in enumerate(demands_raw):
                complexity = d.get("complexity", "M").upper()
                if complexity not in ("S", "M", "L", "XL"):
                    complexity = "M"
                demands.append(PlannedDemand(
                    title=d.get("title", "Sem título"),
                    description=d.get("description", ""),
                    acceptance_criteria=d.get("acceptance_criteria", ""),
                    complexity=complexity,
                    order=d.get("order", didx + 1),
                    depends_on_titles=d.get("depends_on_titles", []),
                ))
            
            phases.append(PlannedPhase(
                title=p.get("title", f"Fase {idx + 1}"),
                description=p.get("description", ""),
                rationale=p.get("rationale", ""),
                order=p.get("order", idx + 1),
                demands=demands,
            ))
        
        return RoadmapOutput(
            roadmap_summary=data.get("roadmap_summary", ""),
            phases=phases,
            out_of_scope=data.get("out_of_scope", []),
            risks=data.get("risks", []),
            notes=data.get("notes", ""),
        )
