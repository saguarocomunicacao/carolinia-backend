# CarolinIA Backend

Backend FastAPI dos agentes da CarolinIA. Roda no Railway.

## Endpoints

- `GET /health` — healthcheck
- `POST /process-demand` — processa uma demanda (chamado por Edge Function do Lovable)

## Variáveis de ambiente

- `ANTHROPIC_API_KEY` — chave da Anthropic API
- `SHARED_SECRET` — segredo compartilhado com Edge Functions
- `LOVABLE_PROJECT_URL` — URL do projeto Lovable Cloud (ex: https://xxx.supabase.co)
- `WORKSPACES_DIR` — diretório onde ficam workspaces (default: /workspaces)
- `MODEL_ORCHESTRATOR` — modelo do orquestrador (default: claude-opus-4-7)
- `MODEL_SUBAGENT` — modelo dos subagentes (default: claude-sonnet-4-6)
