import json
import logging
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import get_settings, Settings
from models.schemas import AnalysisResult, Interaction, QAItem
from services.langsmith_extractor import LangSmithExtractor
from services.qa_analyst import QAAnalyst

logger = logging.getLogger(__name__)

router = APIRouter()


def get_extractor(settings: Annotated[Settings, Depends(get_settings)]) -> LangSmithExtractor:
    return LangSmithExtractor(
        api_key=settings.langsmith_api_key,
        project=settings.langsmith_project,
    )


@router.get("/interactions", response_model=list[Interaction])
async def get_interactions(
    project: str | None = Query(default=None),
    start_time: datetime | None = Query(default=None),
    end_time: datetime | None = Query(default=None),
    extractor: LangSmithExtractor = Depends(get_extractor),
) -> list[Interaction]:
    try:
        return extractor.get_interactions(
            project=project,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as exc:
        logger.exception("Error fetching interactions")
        # Mejorar mensaje de error para proyectos no encontrados
        error_msg = str(exc)
        if "not found" in error_msg.lower():
            project_name = project or extractor.default_project
            raise HTTPException(
                status_code=404, 
                detail={"error": f"El proyecto '{project_name}' no existe en LangSmith. Verifica el nombre del proyecto."}
            )
        raise HTTPException(status_code=500, detail={"error": error_msg})


@router.get("/qa-dataset", response_model=list[QAItem])
async def get_qa_dataset(
    project: str | None = Query(default=None),
    start_time: datetime | None = Query(default=None),
    end_time: datetime | None = Query(default=None),
    extractor: LangSmithExtractor = Depends(get_extractor),
) -> list[QAItem]:
    try:
        interactions = extractor.get_interactions(
            project=project,
            start_time=start_time,
            end_time=end_time,
        )
        return [
            QAItem(
                question=ix.question,
                context=ix.retrieval_context.strip(),
                professor_response=ix.agent_response,
            )
            for ix in interactions
        ]
    except Exception as exc:
        logger.exception("Error generating QA dataset")
        # Mejorar mensaje de error para proyectos no encontrados
        error_msg = str(exc)
        if "not found" in error_msg.lower():
            project_name = project or extractor.default_project
            raise HTTPException(
                status_code=404, 
                detail={"error": f"El proyecto '{project_name}' no existe en LangSmith. Verifica el nombre del proyecto."}
            )
        raise HTTPException(status_code=500, detail={"error": error_msg})


@router.get("/debug/runs")
async def debug_runs(
    project: str | None = Query(default=None),
    extractor: LangSmithExtractor = Depends(get_extractor),
) -> dict:
    """Debug endpoint: shows raw inputs/outputs structure of first ChatGroq runs."""
    project_name = project or extractor.default_project
    now = datetime.now(timezone.utc)
    root_runs = list(extractor.client.list_runs(
        project_name=project_name,
        is_root=True,
        start_time=now - timedelta(days=7),
        end_time=now,
    ))
    result = []
    for root_run in root_runs[:3]:  # inspect first 3 traces
        child_runs = list(extractor.client.list_runs(
            project_name=project_name,
            trace_id=str(root_run.id),
            filter='eq(name, "ChatGroq")',
        ))
        for run in child_runs[:2]:  # first 2 ChatGroq runs per trace
            # Safely serialize — truncate long strings
            def truncate(v, n=200):
                s = str(v)
                return s[:n] + "..." if len(s) > n else s

            inputs_raw = run.inputs or {}
            outputs_raw = run.outputs or {}

            # Show keys and first message structure
            messages = inputs_raw.get("messages", [])
            first_msg = messages[0] if messages else None
            first_msg_info = None
            if first_msg is not None:
                if isinstance(first_msg, dict):
                    first_msg_info = {k: truncate(v) for k, v in first_msg.items()}
                else:
                    first_msg_info = {
                        "type": getattr(first_msg, "type", "?"),
                        "class": type(first_msg).__name__,
                        "content_preview": truncate(getattr(first_msg, "content", "")),
                    }

            result.append({
                "run_id": str(run.id),
                "run_name": run.name,
                "inputs_keys": list(inputs_raw.keys()),
                "messages_count": len(messages),
                "first_message": first_msg_info,
                "all_message_roles": [
                    m.get("role") or m.get("type") or getattr(m, "type", "?")
                    if isinstance(m, dict) else getattr(m, "type", type(m).__name__)
                    for m in messages
                ],
                "outputs_keys": list(outputs_raw.keys()),
                "outputs_preview": {k: truncate(v) for k, v in list(outputs_raw.items())[:3]},
            })
    return {"traces_inspected": len(root_runs), "runs": result}


@router.get("/debug/projects")
async def debug_projects(
    extractor: LangSmithExtractor = Depends(get_extractor),
) -> dict:
    """Debug endpoint: lists all available projects for the configured API key."""
    try:
        projects = list(extractor.client.list_projects())
        return {
            "total_projects": len(projects),
            "projects": [
                {
                    "name": p.name,
                    "id": str(p.id),
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                }
                for p in projects
            ]
        }
    except Exception as exc:
        logger.exception("Error listing projects")
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@router.post("/qa-dataset/export")
async def export_qa_dataset(items: list[QAItem]) -> FileResponse:
    try:
        content = json.dumps(
            [item.model_dump() for item in items],
            ensure_ascii=False,
            indent=2,
        )
        # Write to a temp file (FileResponse needs a real path)
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        )
        tmp.write(content)
        tmp.close()
        return FileResponse(
            path=tmp.name,
            media_type="application/json",
            filename="qa_dataset.json",
            headers={"Content-Disposition": 'attachment; filename="qa_dataset.json"'},
        )
    except Exception as exc:
        logger.exception("Error exporting QA dataset")
        raise HTTPException(status_code=500, detail={"error": str(exc)})


# ── /analyze ─────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    question: str
    agent_response: str
    retrieval_context: str = ""


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_interaction(
    body: AnalyzeRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AnalysisResult:
    """Run the 4-node LangGraph/Gemini analysis pipeline on a QA pair."""
    if not settings.effective_gemini_key:
        raise HTTPException(
            status_code=503,
            detail={"error": "GEMINI_API_KEY or GOOGLE_API_KEY not configured in backend .env"},
        )
    try:
        analyst = QAAnalyst(gemini_api_key=settings.effective_gemini_key)
        result = analyst.analyze(
            question=body.question,
            agent_response=body.agent_response,
            retrieval_context=body.retrieval_context,
            langsmith_extra={
                "metadata": {
                    "source_project": settings.langsmith_project,
                    "target_trace": "qa_evaluation",
                }
            }
        )
        return AnalysisResult(**result)
    except Exception as exc:
        err_str = str(exc)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            import re as _re
            m = _re.search(r'retry in ([\d.]+)s', err_str, _re.IGNORECASE)
            wait = int(float(m.group(1))) if m else 60
            logger.warning("Gemini quota exhausted after retries: %s", exc)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": f"Cuota de Gemini agotada. Reintenta en ~{wait} segundos.",
                    "retry_after": wait,
                },
            )
        logger.exception("Error running QA analysis")
        raise HTTPException(status_code=500, detail={"error": str(exc)})

