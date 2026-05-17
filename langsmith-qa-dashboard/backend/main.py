import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from config import get_settings
from routers.api import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Validate settings at startup — fail fast if LANGSMITH_API_KEY is missing
try:
    _settings = get_settings()
    logger.info("LangSmith project: %s", _settings.langsmith_project)
except ValidationError as exc:
    logger.error("Configuration error: %s", exc)
    sys.exit(1)

app = FastAPI(
    title="LangSmith QA Dashboard API",
    version="0.1.0",
    description="Backend API for extracting and serving LangSmith QA datasets",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
