"""AnalistaBriefing — agente que detecta gaps em briefings.

É o primeiro agente do pipeline. Sua função é ler todo o contexto disponível
do projeto e identificar perguntas críticas que precisam ser respondidas
antes de qualquer execução começar.

Output esperado: JSON estruturado com lista de gaps identificados, cada um
virando uma "demanda de clarificação" no banco.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.agents.base import AgentContext, BaseAgent
from app.services.llm_router import PRESET_ANALISE_DENSA

logger = logging.getLogger(__name__)


# ============================================================
# Output esperado do modelo
# ============================================================

@dataclass
class IdentifiedGap:
    """Um gap identificado no briefing."""
    category: str          # 'audience' | 'problem' | 'success_criteria' | 'constraints' | 'delivery' | 'out_of_scope' | 'other'
    title: str             # Título curto da pergunta/gap
    question: str          # Pergunta específica pro cliente responder
    why_matters: str       # Por que isso é crítico saber antes de começar
    priority: str          # 'high' | 'medium' | 'low'


@dataclass
class AnalysisOutput:
    """Estrutura completa do output do Analista."""
    briefing_summary: str  # 2-4 frases resumindo o que foi entendido
    gaps: list[IdentifiedGap]
    confidence_level: str  # 'high' | 'medium' | 'low' — quanto material o Analista teve
    notes: str             # Observações livres do Analista pro usuário


# ============================================================
# AnalistaBriefing
# ============================================================

class AnalistaBriefing(BaseAgent):
    """Agente que analisa briefing e identifica gaps."""
    
    role = "analista_briefing"
    model_preferences = PRESET_ANALISE_DENSA  # Opus → GPT-4o → Gemini Pro
    
    system_prompt = """Você é o ANALISTA DE BRIEFING da CarolinIA, uma plataforma multi-agente.

# Seu papel

Você é o primeiro agente que recebe um projeto novo. Sua função é ler TODO 
o material disponível e identificar o que está faltando antes do time 
começar a trabalhar.

Você é direto, rigoroso e prático. NÃO inventa contexto. NÃO faz 30 
perguntas — faz as 5 a 10 mais críticas. NÃO usa jargão de consultoria 
("alinhar expectativas", "sinergia", "disruptivo" — proibido).

# Dimensões que você sempre analisa

Pra cada projeto, você verifica gaps em 6 dimensões:

1. **PÚBLICO-ALVO** — Quem vai consumir/usar/comprar o resultado?
   - Idade? Localização? Renda? Comportamento? Dor específica?
   - "Brasileiros" não é resposta — "Mães de classe C em SP, 25-40, 
     que compram via WhatsApp" é.

2. **PROBLEMA/OPORTUNIDADE** — Por que esse projeto existe?
   - O que o cliente está tentando resolver ou alcançar?
   - O que muda na vida do público se isso der certo?

3. **CRITÉRIOS DE SUCESSO** — Como saber se deu certo?
   - Métricas concretas: conversão? leads? CTR? receita? NPS?
   - Sem métrica: "Como você vai saber se eu te entreguei algo bom?"

4. **RESTRIÇÕES** — O que limita as escolhas?
   - Prazo (data específica)
   - Budget (R$ ou faixa)
   - Stack técnica (se aplica)
   - Marca (cores, tom, fontes — voltar pro brand book?)
   - Compliance (LGPD, regulamentações setoriais)

5. **ENTREGA** — Formato e canais
   - Onde vai aparecer (site? Instagram? PDF? loja física?)
   - Quantos formatos? Em quais idiomas?
   - Quem vai distribuir/publicar?

6. **FORA DO ESCOPO** — O que claramente NÃO faz parte
   - Importante pra não inflar e gerar fricção depois
   - "Esse projeto inclui produção de vídeo?" é uma boa pergunta

# Como você prioriza gaps

- **high**: sem responder isso, é impossível começar com qualidade. 
  Ex: "Qual é o público?", "Qual a data limite?"
- **medium**: é importante mas a equipe pode começar e refinar no 
  caminho. Ex: "Qual o tom de voz preferido?"
- **low**: nice-to-have, pode esperar. Ex: "Tem preferência de fonte?"

Foco em high e medium. Low só se sobrar bandwidth.

# Output OBRIGATÓRIO em JSON

Sua resposta DEVE ser um único bloco JSON válido, sem nenhum texto fora 
do JSON. Estrutura:

```json
{
  "briefing_summary": "2-4 frases resumindo o que entendi do projeto. Seja específico.",
  "confidence_level": "high|medium|low",
  "gaps": [
    {
      "category": "audience|problem|success_criteria|constraints|delivery|out_of_scope|other",
      "title": "Título curto — 4-8 palavras",
      "question": "Pergunta direta e específica pro cliente responder",
      "why_matters": "Por que isso bloqueia ou prejudica o trabalho",
      "priority": "high|medium|low"
    }
  ],
  "notes": "Observações livres pro usuário. Pode ficar vazio."
}
```

# Regras finais

- Máximo 10 gaps. Idealmente 5-7.
- Se o briefing estiver completo, retorna gaps vazio e confidence_level="high".
- Se faltar TUDO (briefing muito curto), confidence_level="low" e foca 
  nos 3 gaps mais essenciais.
- NUNCA invente dados do projeto. Se o material não menciona prazo, 
  pergunta sobre prazo — não chute uma data.
- Português brasileiro, tom direto, sem clichê.
- Trata o cliente por "você".
- NÃO INCLUA TEXTO FORA DO JSON. Nenhum "Aqui está minha análise:" ou 
  "Espero ter ajudado". Só o JSON.
"""

    # Modelos podem ser mais caros pra análise densa
    max_tokens: int = 4096
    temperature: float = 0.7
    
    # ============================================================
    # Customização específica do Analista
    # ============================================================
    
    def build_user_message(self, context: AgentContext) -> str:
        """Monta o input com TODO o contexto disponível (sem truncar duro).
        
        Diferente do BaseAgent default que pega top-20 por importância,
        o Analista quer ver tudo que tem do projeto, ordenado por importância.
        """
        parts = []
        
        # Contexto básico do projeto
        parts.append("# Material do projeto")
        parts.append(f"\n**Nome do projeto:** {context.project_name or 'Sem nome'}")
        if context.project_stack:
            parts.append(f"**Stack/contexto técnico:** {context.project_stack}")
        if context.project_description:
            parts.append(f"**Descrição inicial:** {context.project_description}")
        
        # Entradas de contexto — TODAS, ordenadas por importance
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
                # Trunca cada entrada em 5000 chars pra não estourar context window
                if len(content) > 5000:
                    content = content[:5000] + "\n[... conteúdo truncado ...]"
                parts.append(f"\n### {title}")
                parts.append(f"_Origem: {source} · Importância: {imp}/10_\n")
                parts.append(f"```\n{content}\n```")
        else:
            parts.append("\n_Nenhum material indexado ainda._")
        
        # Extra context (info da análise, run anterior, etc)
        if context.extra_context:
            parts.append("\n## Contexto da análise")
            for k, v in context.extra_context.items():
                parts.append(f"- **{k}**: {v}")
        
        return "\n".join(parts)
    
    def summarize_output(self, output_text: str) -> str:
        """Extrai o briefing_summary do JSON como resumo curto."""
        try:
            parsed = self.parse_output(output_text)
            summary = parsed.briefing_summary
            if len(summary) > 300:
                summary = summary[:297] + "..."
            return f"[{len(parsed.gaps)} gaps identificados] {summary}"
        except Exception:
            # Fallback se não conseguir parsear
            return output_text[:300] + ("..." if len(output_text) > 300 else "")
    
    @staticmethod
    def parse_output(output_text: str) -> AnalysisOutput:
        """Parseia o JSON retornado pelo modelo numa estrutura tipada.
        
        Tolera:
        - JSON dentro de bloco markdown ```json ... ```
        - Texto antes/depois do JSON
        - Campos faltantes (preenche com defaults)
        """
        # Tenta extrair JSON de dentro de ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Procura o primeiro { até o último }
            start = output_text.find("{")
            end = output_text.rfind("}")
            if start == -1 or end == -1 or end < start:
                raise ValueError(f"Output não contém JSON parseável: {output_text[:200]}")
            json_str = output_text[start:end + 1]
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido: {e}. Conteúdo: {json_str[:300]}")
        
        gaps_raw = data.get("gaps", [])
        gaps = []
        for g in gaps_raw:
            gaps.append(IdentifiedGap(
                category=g.get("category", "other"),
                title=g.get("title", "Sem título"),
                question=g.get("question", ""),
                why_matters=g.get("why_matters", ""),
                priority=g.get("priority", "medium"),
            ))
        
        return AnalysisOutput(
            briefing_summary=data.get("briefing_summary", ""),
            gaps=gaps,
            confidence_level=data.get("confidence_level", "medium"),
            notes=data.get("notes", ""),
        )
