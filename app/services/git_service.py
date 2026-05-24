"""Serviço de operações Git pra clonar/atualizar repos no workspace."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from git import GitCommandError, Repo

from app.core.config import settings

logger = logging.getLogger(__name__)


def repo_workspace_path(project_id: str, repo_full_name: str) -> Path:
    """Path onde o repo será clonado dentro do workspace."""
    safe_name = repo_full_name.replace("/", "__")
    return Path(settings.workspaces_dir) / project_id / "repos" / safe_name


def clone_repo(
    project_id: str,
    repo_full_name: str,
    access_token: str,
    default_branch: str = "main",
) -> tuple[bool, str | None, Path | None]:
    """Clona ou atualiza um repo do GitHub.
    
    Retorna: (success, error_message, path)
    """
    target = repo_workspace_path(project_id, repo_full_name)
    auth_url = f"https://x-access-token:{access_token}@github.com/{repo_full_name}.git"
    
    try:
        if target.exists() and (target / ".git").exists():
            repo = Repo(target)
            repo.remotes.origin.set_url(auth_url)
            repo.remotes.origin.fetch()
            repo.git.checkout(default_branch)
            repo.remotes.origin.pull(default_branch)
            logger.info("Repo atualizado: %s", repo_full_name)
        else:
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            Repo.clone_from(auth_url, target, branch=default_branch, depth=1)
            logger.info("Repo clonado: %s", repo_full_name)
        
        # Remove URL com token (não fica gravada no .git/config)
        repo = Repo(target)
        repo.remotes.origin.set_url(f"https://github.com/{repo_full_name}.git")
        
        return True, None, target
    
    except GitCommandError as e:
        msg = str(e).replace(access_token, "***") if access_token else str(e)
        logger.error("Erro clonando %s: %s", repo_full_name, msg)
        return False, msg, None
    except Exception as e:
        logger.exception("Erro inesperado clonando %s", repo_full_name)
        return False, str(e), None


def list_files(repo_path: Path, max_files: int = 500) -> list[Path]:
    """Lista arquivos relevantes do repo (ignora binários e diretórios grandes)."""
    ignore_dirs = {
        ".git", "node_modules", "venv", ".venv", "__pycache__",
        "dist", "build", ".next", ".nuxt", "target", "vendor", ".turbo",
    }
    ignore_extensions = {
        ".pyc", ".pyo", ".class", ".o", ".so", ".dll", ".exe",
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".ico", ".svg",
        ".mp4", ".mp3", ".wav", ".pdf",
        ".zip", ".tar", ".gz", ".rar", ".7z",
        ".woff", ".woff2", ".ttf", ".eot",
    }
    
    files = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignore_dirs for part in path.parts):
            continue
        if path.suffix.lower() in ignore_extensions:
            continue
        try:
            if path.stat().st_size > 200_000:
                continue
        except OSError:
            continue
        files.append(path)
        if len(files) >= max_files:
            break
    
    return files
