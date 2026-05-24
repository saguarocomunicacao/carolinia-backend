"""Cliente HTTP pra falar com as Edge Functions do Lovable Cloud."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class LovableClient:
    """Cliente compartilhado pra chamar Edge Functions com X-Shared-Secret."""
    
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={
                "X-Shared-Secret": settings.shared_secret,
                "Content-Type": "application/json",
            },
        )
    
    async def get_project_secret(self, secret_id: str) -> str | None:
        """Busca um secret (ex: PAT do GitHub) do vault do Supabase."""
        url = f"{settings.lovable_project_url}/functions/v1/get-secret"
        try:
            response = await self._client.post(
                url, json={"secret_id": secret_id}
            )
            response.raise_for_status()
            return response.json().get("value")
        except Exception as e:
            logger.error("Falha ao buscar secret %s: %s", secret_id, e)
            return None
    
    async def send_repo_status(
        self,
        repo_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Atualiza status de um project_repo (pending → cloned/error)."""
        url = f"{settings.lovable_project_url}/functions/v1/update-repo-status"
        try:
            await self._client.post(
                url,
                json={
                    "repo_id": repo_id,
                    "status": status,
                    "error_message": error_message,
                },
            )
        except Exception as e:
            logger.error("Falha ao atualizar status do repo %s: %s", repo_id, e)
    
    async def bulk_insert_context(
        self,
        project_id: str,
        entries: list[dict[str, Any]],
    ) -> None:
        """Insere várias entradas em project_context numa chamada."""
        if not entries:
            return
        url = f"{settings.lovable_project_url}/functions/v1/insert-context"
        try:
            await self._client.post(
                url,
                json={"project_id": project_id, "entries": entries},
            )
        except Exception as e:
            logger.error("Falha ao inserir contexto no projeto %s: %s", project_id, e)
    
    async def mark_file_indexed(self, file_id: str) -> None:
        """Marca um project_file como indexado (indexed=true)."""
        url = f"{settings.lovable_project_url}/functions/v1/mark-file-indexed"
        try:
            await self._client.post(url, json={"file_id": file_id})
        except Exception as e:
            logger.error("Falha ao marcar arquivo %s indexado: %s", file_id, e)
    
    async def aclose(self) -> None:
        await self._client.aclose()


# Instância global usada pelos endpoints
lovable_client = LovableClient()
