"""ContextLoader — busca project_context via Edge Function do Lovable.

Backend Railway não tem JWT do user, então usa a Edge Function 
get-project-context (autenticada por X-Shared-Secret) pra buscar 
entries indexadas do projeto.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


async def fetch_project_context(
    project_id: str,
    min_importance: int = 1,
) -> dict[str, Any]:
    """Busca entries de project_context via Edge Function.
    
    Returns:
        {
            "ok": bool,
            "entries": list[dict],  # cada dict tem id, title, content, source_type, importance, etc
            "total_count": int,
            "returned_count": int,
            "truncated": bool,
        }
    
    Em caso de erro, retorna {"ok": False, "entries": [], ...}.
    """
    url = f"{settings.lovable_project_url}/functions/v1/get-project-context"
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.post(
                url,
                json={
                    "project_id": project_id,
                    "min_importance": min_importance,
                },
                headers={
                    "X-Shared-Secret": settings.shared_secret,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get("ok"):
                logger.error(
                    "get-project-context retornou ok=false: %s",
                    data.get("error", "sem erro"),
                )
                return {
                    "ok": False,
                    "entries": [],
                    "total_count": 0,
                    "returned_count": 0,
                    "truncated": False,
                }
            
            logger.info(
                "[ContextLoader] project=%s entries=%d (total=%d, truncated=%s)",
                project_id,
                data.get("returned_count", 0),
                data.get("total_count", 0),
                data.get("truncated", False),
            )
            
            return data
    except Exception as e:
        logger.exception("Falha ao buscar project_context pro project=%s", project_id)
        return {
            "ok": False,
            "entries": [],
            "total_count": 0,
            "returned_count": 0,
            "truncated": False,
            "error": str(e),
        }
