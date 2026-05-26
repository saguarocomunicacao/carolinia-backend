"""WorkspaceManager — gerencia clones Git + worktrees por demand.

Cada projeto tem 1 clone "principal" do repo (branch default). Cada
demand em execução tem 1 worktree isolado dentro desse clone. Isso 
permite paralelismo Git: várias branches sendo trabalhadas ao mesmo
tempo sem race condition no .git/.

Layout no filesystem:
  /workspaces/<project_id>/<repo_id>/
    ├── main-clone/             clone principal (HEAD em default_branch)
    └── worktrees/
        ├── demand-<id8>/       worktree pra demand A (branch isolada)
        ├── demand-<id8>/       worktree pra demand B
        └── ...

Operações são atômicas onde possível: erro a meio do clone limpa parcial.
Locks por (project_id, repo_id) garantem que múltiplas demands não 
corrompam o clone principal ao mesmo tempo.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# Constantes
# ============================================================

# Prefixo das branches criadas pelo CarolinIA
BRANCH_PREFIX = "carolinia/demand-"

# Timeout pra comandos git (em segundos)
GIT_TIMEOUT_SECONDS = 300  # 5 minutos


# ============================================================
# Tipos
# ============================================================

@dataclass
class WorkspacePaths:
    """Caminhos de um workspace de projeto+repo."""
    root: Path                    # /workspaces/<project_id>/<repo_id>
    main_clone: Path              # /workspaces/<project_id>/<repo_id>/main-clone
    worktrees_dir: Path           # /workspaces/<project_id>/<repo_id>/worktrees


@dataclass
class WorktreeInfo:
    """Info de uma worktree de demand."""
    demand_id: str
    path: Path
    branch_name: str


@dataclass
class GitCommandResult:
    """Resultado de um comando git executado."""
    success: bool
    stdout: str
    stderr: str
    returncode: int


# ============================================================
# Locks por (project_id, repo_id) — evita race em operações no main-clone
# ============================================================

_workspace_locks: dict[str, asyncio.Lock] = {}


def _get_lock(project_id: str, repo_id: str) -> asyncio.Lock:
    """Retorna lock asyncio único pro par (project, repo)."""
    key = f"{project_id}:{repo_id}"
    if key not in _workspace_locks:
        _workspace_locks[key] = asyncio.Lock()
    return _workspace_locks[key]


# ============================================================
# Helpers de filesystem
# ============================================================

def _workspace_paths(project_id: str, repo_id: str) -> WorkspacePaths:
    """Calcula paths de um workspace sem criar nada."""
    root = Path(settings.workspaces_dir) / project_id / repo_id
    return WorkspacePaths(
        root=root,
        main_clone=root / "main-clone",
        worktrees_dir=root / "worktrees",
    )


def _branch_name_for_demand(demand_id: str) -> str:
    """Gera nome de branch pra uma demand. Usa primeiros 8 chars do uuid."""
    short_id = demand_id.replace("-", "")[:8]
    return f"{BRANCH_PREFIX}{short_id}"


def _worktree_dir_name(demand_id: str) -> str:
    """Nome do diretório da worktree (sem o path completo)."""
    short_id = demand_id.replace("-", "")[:8]
    return f"demand-{short_id}"


# ============================================================
# Execução de comandos git (subprocess assíncrono)
# ============================================================

async def _run_git(
    args: list[str],
    cwd: Path | None = None,
    timeout: int = GIT_TIMEOUT_SECONDS,
    env_extra: dict[str, str] | None = None,
) -> GitCommandResult:
    """Roda um comando git via subprocess assíncrono.
    
    Args:
        args: lista de argumentos (sem o 'git' inicial). Ex: ['clone', 'url', 'dest']
        cwd: working directory (None = pasta atual)
        timeout: segundos pra abortar
        env_extra: variáveis de ambiente adicionais (ex: GITHUB_TOKEN)
    
    Returns:
        GitCommandResult com stdout/stderr capturados
    """
    cmd = ["git"] + args
    
    # Constrói env base + adicionais
    import os as _os
    env = _os.environ.copy()
    if env_extra:
        env.update(env_extra)
    
    # Evita prompt interativo do git (ex: pedindo senha)
    env["GIT_TERMINAL_PROMPT"] = "0"
    
    cmd_str = " ".join(cmd)
    cwd_str = str(cwd) if cwd else "cwd"
    logger.info("[git] %s (in %s)", cmd_str, cwd_str)
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error("[git] TIMEOUT após %ds: %s", timeout, cmd_str)
            return GitCommandResult(
                success=False,
                stdout="",
                stderr=f"Comando excedeu timeout de {timeout}s",
                returncode=-1,
            )
        
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        success = proc.returncode == 0
        
        if not success:
            logger.warning(
                "[git] FAILED (rc=%d): %s\nstderr: %s",
                proc.returncode, cmd_str, stderr[:500]
            )
        
        return GitCommandResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            returncode=proc.returncode or 0,
        )
    
    except Exception as e:
        logger.exception("[git] Exceção rodando: %s", cmd_str)
        return GitCommandResult(
            success=False,
            stdout="",
            stderr=str(e),
            returncode=-1,
        )


# ============================================================
# Operações públicas
# ============================================================

async def ensure_workspace_ready(
    project_id: str,
    repo_id: str,
    repo_full_name: str,
    github_token: str,
    default_branch: str = "main",
) -> tuple[bool, str | None, WorkspacePaths]:
    """Garante que o main-clone do workspace existe e está atualizado.
    
    - Se não existe: clona do GitHub
    - Se existe: faz pull do default_branch
    
    Retorna: (success, error_message, paths)
    
    Operação protegida por lock (project_id, repo_id).
    """
    paths = _workspace_paths(project_id, repo_id)
    lock = _get_lock(project_id, repo_id)
    
    async with lock:
        # Cria estrutura base se não existir
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.worktrees_dir.mkdir(parents=True, exist_ok=True)
        
        clone_url_with_token = f"https://x-access-token:{github_token}@github.com/{repo_full_name}.git"
        clone_url_clean = f"https://github.com/{repo_full_name}.git"
        
        # Cenário 1: main-clone não existe → faz clone fresh
        if not paths.main_clone.exists() or not (paths.main_clone / ".git").exists():
            logger.info(
                "[workspace] Clonando %s em %s",
                repo_full_name, paths.main_clone,
            )
            
            # Remove qualquer resto parcial
            if paths.main_clone.exists():
                shutil.rmtree(paths.main_clone, ignore_errors=True)
            
            result = await _run_git([
                "clone",
                "--branch", default_branch,
                clone_url_with_token,
                str(paths.main_clone),
            ])
            
            if not result.success:
                # Sanitiza stderr pra não vazar token
                safe_stderr = result.stderr.replace(github_token, "***") if github_token else result.stderr
                return False, f"Falha clonando: {safe_stderr[:500]}", paths
            
            # Remove URL com token da config do remote (segurança)
            await _run_git(
                ["remote", "set-url", "origin", clone_url_clean],
                cwd=paths.main_clone,
            )
            
            logger.info("[workspace] Clone concluído: %s", paths.main_clone)
            return True, None, paths
        
        # Cenário 2: main-clone existe → atualiza
        logger.info("[workspace] Atualizando %s", paths.main_clone)
        
        # Set URL com token temporariamente pra fetch funcionar
        await _run_git(
            ["remote", "set-url", "origin", clone_url_with_token],
            cwd=paths.main_clone,
        )
        
        # Garante que está em default_branch
        checkout_result = await _run_git(
            ["checkout", default_branch],
            cwd=paths.main_clone,
        )
        if not checkout_result.success:
            await _run_git(
                ["remote", "set-url", "origin", clone_url_clean],
                cwd=paths.main_clone,
            )
            return False, f"Falha checkout {default_branch}: {checkout_result.stderr[:300]}", paths
        
        # Fetch + pull
        pull_result = await _run_git(
            ["pull", "origin", default_branch],
            cwd=paths.main_clone,
        )
        
        # Volta URL sem token
        await _run_git(
            ["remote", "set-url", "origin", clone_url_clean],
            cwd=paths.main_clone,
        )
        
        if not pull_result.success:
            safe_stderr = pull_result.stderr.replace(github_token, "***") if github_token else pull_result.stderr
            return False, f"Falha pull: {safe_stderr[:500]}", paths
        
        logger.info("[workspace] Atualização concluída")
        return True, None, paths


async def create_worktree(
    project_id: str,
    repo_id: str,
    demand_id: str,
    default_branch: str = "main",
) -> tuple[bool, str | None, WorktreeInfo | None]:
    """Cria worktree isolada pra uma demand.
    
    Pré-requisito: ensure_workspace_ready já foi chamada e teve sucesso.
    
    Cria branch nova partir de default_branch e cria worktree apontando 
    pra essa branch. Idempotente: se worktree já existe, retorna ela.
    
    Retorna: (success, error_message, worktree_info)
    """
    paths = _workspace_paths(project_id, repo_id)
    
    if not paths.main_clone.exists():
        return False, "Main clone não existe. Chame ensure_workspace_ready primeiro.", None
    
    branch_name = _branch_name_for_demand(demand_id)
    worktree_path = paths.worktrees_dir / _worktree_dir_name(demand_id)
    
    info = WorktreeInfo(
        demand_id=demand_id,
        path=worktree_path,
        branch_name=branch_name,
    )
    
    lock = _get_lock(project_id, repo_id)
    
    async with lock:
        # Idempotência: se worktree já existe E é Git válido, retorna ela
        if worktree_path.exists() and (worktree_path / ".git").exists():
            logger.info("[workspace] Worktree já existe: %s", worktree_path)
            return True, None, info
        
        # Limpa resto parcial se houver
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)
        
        # Cria worktree com branch nova (-b cria branch)
        # Equivalente: git worktree add -b carolinia/demand-xxx ../worktrees/demand-xxx main
        logger.info(
            "[workspace] Criando worktree %s (branch %s, base %s)",
            worktree_path, branch_name, default_branch,
        )
        
        result = await _run_git(
            [
                "worktree", "add",
                "-b", branch_name,
                str(worktree_path),
                default_branch,
            ],
            cwd=paths.main_clone,
        )
        
        if not result.success:
            # Pode ter falhado porque branch já existe (resíduo de tentativa anterior).
            # Tenta deletar branch e refazer.
            if "already exists" in result.stderr.lower() or "already used" in result.stderr.lower():
                logger.warning(
                    "[workspace] Branch %s já existe, removendo e refazendo",
                    branch_name,
                )
                
                # Force delete da branch
                await _run_git(
                    ["branch", "-D", branch_name],
                    cwd=paths.main_clone,
                )
                
                # Tenta de novo
                result = await _run_git(
                    [
                        "worktree", "add",
                        "-b", branch_name,
                        str(worktree_path),
                        default_branch,
                    ],
                    cwd=paths.main_clone,
                )
            
            if not result.success:
                return False, f"Falha criando worktree: {result.stderr[:500]}", None
        
        logger.info("[workspace] Worktree criada: %s", worktree_path)
        return True, None, info


async def delete_worktree(
    project_id: str,
    repo_id: str,
    demand_id: str,
    force: bool = False,
) -> tuple[bool, str | None]:
    """Remove worktree de uma demand (após PR mergeado ou demand cancelada).
    
    Args:
        force: se True, força remoção mesmo se houver mudanças não-commitadas
    
    Retorna: (success, error_message)
    """
    paths = _workspace_paths(project_id, repo_id)
    branch_name = _branch_name_for_demand(demand_id)
    worktree_path = paths.worktrees_dir / _worktree_dir_name(demand_id)
    
    lock = _get_lock(project_id, repo_id)
    
    async with lock:
        if not worktree_path.exists():
            # Nada pra deletar
            return True, None
        
        if not paths.main_clone.exists():
            # Main clone sumiu — não dá pra usar git worktree remove
            # Apaga manualmente
            shutil.rmtree(worktree_path, ignore_errors=True)
            return True, None
        
        # Tenta remover via git worktree remove
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(worktree_path))
        
        result = await _run_git(args, cwd=paths.main_clone)
        
        if not result.success:
            # Força remoção manual
            logger.warning(
                "[workspace] git worktree remove falhou, removendo manualmente: %s",
                result.stderr[:200],
            )
            shutil.rmtree(worktree_path, ignore_errors=True)
            
            # Prune referências órfãs
            await _run_git(["worktree", "prune"], cwd=paths.main_clone)
        
        # Deleta a branch local também (não precisa preservar)
        await _run_git(
            ["branch", "-D", branch_name],
            cwd=paths.main_clone,
        )
        
        logger.info("[workspace] Worktree e branch removidas: %s", worktree_path)
        return True, None


def get_worktree_path(project_id: str, repo_id: str, demand_id: str) -> Path:
    """Retorna o path da worktree (sem garantir que existe)."""
    paths = _workspace_paths(project_id, repo_id)
    return paths.worktrees_dir / _worktree_dir_name(demand_id)


def get_branch_name(demand_id: str) -> str:
    """Retorna o nome da branch que será usada pra essa demand."""
    return _branch_name_for_demand(demand_id)


async def list_worktrees(project_id: str, repo_id: str) -> list[dict]:
    """Lista worktrees existentes pra um repo.
    
    Retorna lista de dicts com path, branch, HEAD commit. Vazia se 
    workspace não existe.
    """
    paths = _workspace_paths(project_id, repo_id)
    
    if not paths.main_clone.exists():
        return []
    
    result = await _run_git(
        ["worktree", "list", "--porcelain"],
        cwd=paths.main_clone,
    )
    
    if not result.success:
        logger.warning("[workspace] git worktree list falhou: %s", result.stderr[:200])
        return []
    
    # Parse do output porcelain
    worktrees = []
    current = {}
    for line in result.stdout.split("\n"):
        line = line.strip()
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            current["path"] = line[len("worktree "):]
        elif line.startswith("HEAD "):
            current["head"] = line[len("HEAD "):]
        elif line.startswith("branch "):
            current["branch"] = line[len("branch "):].replace("refs/heads/", "")
    
    if current:
        worktrees.append(current)
    
    return worktrees


async def workspace_health_check(
    project_id: str,
    repo_id: str,
) -> dict:
    """Diagnóstico do workspace pra debugging/UI.
    
    Retorna: {
        exists: bool,
        main_clone_exists: bool,
        current_branch: str,
        worktrees_count: int,
        worktrees: list,
        disk_usage_mb: int,
    }
    """
    paths = _workspace_paths(project_id, repo_id)
    
    info: dict = {
        "workspace_path": str(paths.root),
        "exists": paths.root.exists(),
        "main_clone_exists": (paths.main_clone / ".git").exists() if paths.main_clone.exists() else False,
        "current_branch": None,
        "worktrees_count": 0,
        "worktrees": [],
        "disk_usage_mb": 0,
    }
    
    if not info["main_clone_exists"]:
        return info
    
    # Branch atual
    branch_result = await _run_git(
        ["rev-parse", "--abbrev-ref", "HEAD"],
        cwd=paths.main_clone,
        timeout=10,
    )
    if branch_result.success:
        info["current_branch"] = branch_result.stdout.strip()
    
    # Worktrees
    worktrees = await list_worktrees(project_id, repo_id)
    info["worktrees"] = worktrees
    info["worktrees_count"] = max(0, len(worktrees) - 1)  # -1 porque o main-clone aparece também
    
    # Uso de disco (estimativa via du)
    try:
        proc = await asyncio.create_subprocess_exec(
            "du", "-sm", str(paths.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        size_str = stdout_bytes.decode().split()[0]
        info["disk_usage_mb"] = int(size_str)
    except Exception:
        pass
    
    return info
