from datetime import datetime

from pydantic import BaseModel


class HistoryTurn(BaseModel):
    question: str
    response: str


class Interaction(BaseModel):
    trace_id: str
    trace_timestamp: datetime
    turn_index: int
    question: str
    agent_response: str
    history: list[HistoryTurn]


class QAItem(BaseModel):
    question: str
    context: str
    professor_response: str


class AnalysisResult(BaseModel):
    """Output of the 4-node LangGraph QA analyst pipeline."""
    question_analysis: str
    response_analysis: str
    retrieval_analysis: str
    improvement_suggestions: str
