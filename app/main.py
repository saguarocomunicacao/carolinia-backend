"""Entrypoint do backend FastAPI da CarolinIA."""
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"CarolinIA backend iniciando. Workspaces: {settings.workspaces_dir}")
    yield
    print("CarolinIA backend encerrando.")


app = FastAPI(
    title="CarolinIA Backend",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
async def root():
    return {"service": "carolinia-backend", "status": "ok"}
