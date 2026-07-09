import logging
import os
import sys

from pydantic import ValidationError
from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Validate settings at startup and configure LangSmith tracing environment variables BEFORE importing router/services
try:
    _settings = get_settings()
    logger.info("LangSmith project: %s", _settings.langsmith_project)
    
    if _settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        api_key = _settings.langchain_api_key or _settings.langsmith_api_key
        if api_key:
            os.environ["LANGCHAIN_API_KEY"] = api_key
        if _settings.langchain_project:
            os.environ["LANGCHAIN_PROJECT"] = _settings.langchain_project
        logger.info("LangChain tracing enabled for project: %s", os.environ.get("LANGCHAIN_PROJECT"))
except ValidationError as exc:
    logger.error("Configuration error: %s", exc)
    sys.exit(1)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.api import router

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
