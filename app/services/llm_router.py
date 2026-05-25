"""LLMRouter — abstração unificada pra Anthropic, OpenAI e Google.

Cada agente declara uma lista de modelos preferidos (em ordem).
O Router tenta cada um até conseguir uma resposta.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# Tipos e enums
# ============================================================

class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


@dataclass
class ModelChoice:
    """Representa uma escolha de modelo: provider + nome do modelo."""
    provider: Provider
    model: str


@dataclass
class LLMResponse:
    """Resposta padronizada de qualquer provider."""
    text: str
    provider: Provider
    model: str
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    raw: Any = None  # Resposta crua do SDK, pra debug


@dataclass
class ImageResponse:
    """Resposta padronizada de geração de imagem."""
    image_url: str | None = None  # URL se for hosted (OpenAI)
    image_b64: str | None = None  # Base64 se for inline (Google)
    provider: Provider = Provider.OPENAI
    model: str = ""
    cost_usd: float = 0.0
    latency_ms: int = 0
    raw: Any = None


# ============================================================
# Pricing (em USD por 1M tokens, snapshot atual — ajustar quando precisar)
# ============================================================

PRICING = {
    # Anthropic
    "claude-opus-4-7": {"input": 15.0, "output": 75.0},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    # OpenAI
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Google (Gemini)
    "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Preview, grátis no AI Studio
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
}

# Pricing pra imagens (preço por imagem, não por token)
IMAGE_PRICING = {
    "dall-e-3": 0.040,           # 1024x1024 standard
    "imagen-3.0-generate-001": 0.040,
}


def _calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p = PRICING.get(model)
    if not p:
        return 0.0
    return (tokens_in / 1_000_000 * p["input"]) + (tokens_out / 1_000_000 * p["output"])


# ============================================================
# Clients lazy (só inicializa quando precisar)
# ============================================================

_anthropic_client = None
_openai_client = None
_google_configured = False


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY não configurada")
        _anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY não configurada")
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _get_google():
    global _google_configured
    if not _google_configured:
        import google.generativeai as genai
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY não configurada")
        genai.configure(api_key=settings.google_api_key)
        _google_configured = True
    import google.generativeai as genai
    return genai


# ============================================================
# Chamadas por provider (texto)
# ============================================================

def _call_anthropic(
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 4096,
    temperature: float = 1.0,
) -> LLMResponse:
    client = _get_anthropic()
    start = time.time()
    
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    
    latency = int((time.time() - start) * 1000)
    text = response.content[0].text if response.content else ""
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens
    
    return LLMResponse(
        text=text,
        provider=Provider.ANTHROPIC,
        model=model,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        cost_usd=_calculate_cost(model, tokens_in, tokens_out),
        latency_ms=latency,
        raw=response,
    )


def _call_openai(
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 4096,
    temperature: float = 1.0,
) -> LLMResponse:
    client = _get_openai()
    start = time.time()
    
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    
    latency = int((time.time() - start) * 1000)
    text = response.choices[0].message.content or ""
    tokens_in = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens
    
    return LLMResponse(
        text=text,
        provider=Provider.OPENAI,
        model=model,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        cost_usd=_calculate_cost(model, tokens_in, tokens_out),
        latency_ms=latency,
        raw=response,
    )


def _call_google(
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 4096,
    temperature: float = 1.0,
) -> LLMResponse:
    genai = _get_google()
    start = time.time()
    
    # Gemini junta system + user numa única instrução
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    
    response = gemini_model.generate_content(user_message)
    
    latency = int((time.time() - start) * 1000)
    text = response.text if hasattr(response, "text") else ""
    
    # Gemini usage_metadata tem prompt_token_count e candidates_token_count
    usage = getattr(response, "usage_metadata", None)
    tokens_in = getattr(usage, "prompt_token_count", 0) if usage else 0
    tokens_out = getattr(usage, "candidates_token_count", 0) if usage else 0
    
    return LLMResponse(
        text=text,
        provider=Provider.GOOGLE,
        model=model,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        cost_usd=_calculate_cost(model, tokens_in, tokens_out),
        latency_ms=latency,
        raw=response,
    )


# ============================================================
# Geração de imagem
# ============================================================

def _generate_image_openai(prompt: str, size: str = "1024x1024") -> ImageResponse:
    client = _get_openai()
    start = time.time()
    
    response = client.images.generate(
        model=settings.model_openai_image,
        prompt=prompt,
        size=size,
        quality="standard",
        n=1,
    )
    
    latency = int((time.time() - start) * 1000)
    image_url = response.data[0].url if response.data else None
    
    return ImageResponse(
        image_url=image_url,
        provider=Provider.OPENAI,
        model=settings.model_openai_image,
        cost_usd=IMAGE_PRICING.get(settings.model_openai_image, 0.04),
        latency_ms=latency,
        raw=response,
    )


def _generate_image_google(prompt: str, size: str = "1024x1024") -> ImageResponse:
    """Gera imagem com Imagen 3. Requer projeto Google Cloud com billing."""
    genai = _get_google()
    start = time.time()
    
    # Imagen API é diferente do Gemini — usa generate_images
    # Nota: pode falhar se billing do Google Cloud não estiver ativo
    model = genai.ImageGenerationModel(settings.model_google_image)
    response = model.generate_images(prompt=prompt, number_of_images=1)
    
    latency = int((time.time() - start) * 1000)
    image_b64 = None
    if response and response.images:
        image_b64 = response.images[0]._image_bytes  # bytes raw
    
    return ImageResponse(
        image_b64=image_b64,
        provider=Provider.GOOGLE,
        model=settings.model_google_image,
        cost_usd=IMAGE_PRICING.get(settings.model_google_image, 0.04),
        latency_ms=latency,
        raw=response,
    )


# ============================================================
# Router principal — função pública
# ============================================================

def call_llm(
    system_prompt: str,
    user_message: str,
    model_preferences: list[ModelChoice],
    max_tokens: int = 4096,
    temperature: float = 1.0,
) -> LLMResponse:
    """Chama o primeiro modelo da lista que conseguir responder.
    
    Se um falhar (rate limit, erro de API), tenta o próximo.
    Se TODOS falharem, levanta exceção.
    """
    last_error: Exception | None = None
    
    for choice in model_preferences:
        try:
            logger.info(
                "[LLMRouter] Tentando %s/%s (sys=%d chars, user=%d chars)",
                choice.provider.value, choice.model,
                len(system_prompt), len(user_message),
            )
            
            if choice.provider == Provider.ANTHROPIC:
                response = _call_anthropic(
                    choice.model, system_prompt, user_message,
                    max_tokens, temperature,
                )
            elif choice.provider == Provider.OPENAI:
                response = _call_openai(
                    choice.model, system_prompt, user_message,
                    max_tokens, temperature,
                )
            elif choice.provider == Provider.GOOGLE:
                response = _call_google(
                    choice.model, system_prompt, user_message,
                    max_tokens, temperature,
                )
            else:
                raise ValueError(f"Provider desconhecido: {choice.provider}")
            
            logger.info(
                "[LLMRouter] OK %s/%s (in=%d out=%d cost=$%.4f lat=%dms)",
                choice.provider.value, choice.model,
                response.tokens_input, response.tokens_output,
                response.cost_usd, response.latency_ms,
            )
            return response
            
        except Exception as e:
            logger.warning(
                "[LLMRouter] Falha em %s/%s: %s. Tentando próximo.",
                choice.provider.value, choice.model, e,
            )
            last_error = e
            continue
    
    raise RuntimeError(
        f"Todos os providers falharam. Último erro: {last_error}"
    )


def generate_image(
    prompt: str,
    provider_preferences: list[Provider] = None,
    size: str = "1024x1024",
) -> ImageResponse:
    """Gera imagem usando o primeiro provider disponível.
    
    Default: tenta Google (Imagen 3) primeiro, fallback OpenAI (DALL-E 3).
    """
    if provider_preferences is None:
        provider_preferences = [Provider.GOOGLE, Provider.OPENAI]
    
    last_error: Exception | None = None
    
    for provider in provider_preferences:
        try:
            logger.info("[LLMRouter] Gerando imagem com %s", provider.value)
            
            if provider == Provider.OPENAI:
                return _generate_image_openai(prompt, size)
            elif provider == Provider.GOOGLE:
                return _generate_image_google(prompt, size)
            else:
                raise ValueError(f"Provider de imagem inválido: {provider}")
                
        except Exception as e:
            logger.warning(
                "[LLMRouter] Falha gerando imagem em %s: %s. Tentando próximo.",
                provider.value, e,
            )
            last_error = e
            continue
    
    raise RuntimeError(
        f"Geração de imagem falhou em todos os providers. Último erro: {last_error}"
    )


# ============================================================
# Presets de model_preferences para os agentes
# ============================================================

# Análise textual densa (briefing, planejamento)
PRESET_ANALISE_DENSA = [
    ModelChoice(Provider.ANTHROPIC, settings.model_orchestrator),  # claude-opus-4-7
    ModelChoice(Provider.OPENAI, settings.model_openai_text),       # gpt-4o
    ModelChoice(Provider.GOOGLE, settings.model_google_text_pro),   # gemini-1.5-pro
]

# Decisão de orquestração (raciocínio estruturado)
PRESET_ORQUESTRACAO = [
    ModelChoice(Provider.ANTHROPIC, settings.model_orchestrator),
    ModelChoice(Provider.OPENAI, settings.model_openai_text),
]

# Tarefas rápidas, código simples (Junior Dev)
PRESET_RAPIDO = [
    ModelChoice(Provider.ANTHROPIC, settings.model_subagent),       # claude-sonnet-4-6
    ModelChoice(Provider.OPENAI, settings.model_openai_text_fast),  # gpt-4o-mini
    ModelChoice(Provider.GOOGLE, settings.model_google_text),       # gemini-2.0-flash-exp
]

# Copywriting / texto criativo (OpenAI tem reputação aqui)
PRESET_COPY = [
    ModelChoice(Provider.OPENAI, settings.model_openai_text),
    ModelChoice(Provider.ANTHROPIC, settings.model_orchestrator),
]

# Análise visual (entender imagem que o usuário mandou)
PRESET_VISUAL = [
    ModelChoice(Provider.GOOGLE, settings.model_google_text_pro),   # Gemini é forte em visão
    ModelChoice(Provider.ANTHROPIC, settings.model_orchestrator),
    ModelChoice(Provider.OPENAI, settings.model_openai_text),
]
