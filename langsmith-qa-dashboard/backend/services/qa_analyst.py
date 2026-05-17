"""
QA Analyst — 4-node LangGraph pipeline powered by Gemini Flash.

Graph topology (linear):
    question_analyst
        → response_analyst
            → retrieval_analyst
                → improvement_synthesizer
                    → END

Each node is a standalone Gemini call; the state accumulates all outputs so
the final node has the full picture to produce improvement suggestions.
"""
from __future__ import annotations

import logging
import re
import time
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

# ── Shared LLM ──────────────────────────────────────────────────────────────
# gemini-2.5-flash — best available model on this API key.


def _make_llm(api_key: str, model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.3,
    )


# ── State ────────────────────────────────────────────────────────────────────

class AnalystState(TypedDict):
    """Shared state passed between nodes."""
    # Inputs (provided by the caller)
    question: str
    agent_response: str
    retrieval_context: str       # whatever Qdrant returned (context field)

    # Outputs produced by each node
    question_analysis: str
    response_analysis: str
    retrieval_analysis: str
    improvement_suggestions: str

    # Internal: the Gemini LLM instance
    _llm: ChatGoogleGenerativeAI


def _invoke_with_retry(llm: ChatGoogleGenerativeAI, messages: list, max_retries: int = 3):
    """Call llm.invoke with automatic retry on 429 RESOURCE_EXHAUSTED.

    Parses the 'retry in Xs' delay from the error message when available;
    otherwise uses exponential backoff (30 → 60 → 120 s).
    """
    backoff = 30
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            err_str = str(exc)
            if "429" not in err_str and "RESOURCE_EXHAUSTED" not in err_str:
                raise  # not a rate-limit error — re-raise immediately
            if attempt >= max_retries:
                raise
            # Try to parse the suggested retry delay from the error message
            m = re.search(r'retry in ([\d.]+)s', err_str, re.IGNORECASE)
            wait = float(m.group(1)) if m else backoff
            logger.warning(
                "Gemini 429 rate-limit on attempt %d/%d — waiting %.0fs before retry",
                attempt + 1, max_retries, wait,
            )
            time.sleep(wait)
            backoff = min(backoff * 2, 120)


# ── Node implementations ─────────────────────────────────────────────────────

def _node_question_analyst(state: AnalystState) -> dict:
    """Agent 1 — Analyzes the clarity and didactic quality of the question."""
    llm = state["_llm"]
    messages = [
        SystemMessage(content=(
            "Eres un experto en pedagogía universitaria de física. "
            "Tu tarea es evaluar la calidad de la pregunta de un estudiante. "
            "Responde en español en 3-5 oraciones concisas."
        )),
        HumanMessage(content=(
            f"Pregunta del estudiante:\n{state['question']}\n\n"
            "Evalúa: ¿Es la pregunta clara? ¿Muestra comprensión del tema? "
            "¿Qué conceptos físicos está intentando entender el estudiante?"
        )),
    ]
    response = _invoke_with_retry(llm, messages)
    return {"question_analysis": response.content}


def _node_response_analyst(state: AnalystState) -> dict:
    """Agent 2 — Analyzes the pedagogical quality of the agent's response."""
    llm = state["_llm"]
    messages = [
        SystemMessage(content=(
            "Eres un experto en didáctica de ciencias. "
            "Tu tarea es evaluar la calidad pedagógica de la respuesta de un agente IA. "
            "Responde en español en 3-5 oraciones concisas."
        )),
        HumanMessage(content=(
            f"Pregunta del estudiante:\n{state['question']}\n\n"
            f"Respuesta del agente:\n{state['agent_response']}\n\n"
            "Evalúa: ¿Es la respuesta completa? ¿Es pedagógicamente adecuada? "
            "¿Usa correctamente el método socrático? ¿Hay imprecisiones?"
        )),
    ]
    response = _invoke_with_retry(llm, messages)
    return {"response_analysis": response.content}


def _node_retrieval_analyst(state: AnalystState) -> dict:
    """Agent 3 — Analyzes the relevance of what Qdrant retrieved."""
    llm = state["_llm"]
    retrieval = state.get("retrieval_context", "").strip()
    if not retrieval:
        return {"retrieval_analysis": "No se encontró contexto de recuperación (Qdrant) para esta interacción."}

    messages = [
        SystemMessage(content=(
            "Eres un experto en sistemas de recuperación de información (RAG) para física. "
            "Tu tarea es evaluar si el contexto recuperado por Qdrant fue relevante. "
            "Responde en español en 3-5 oraciones concisas."
        )),
        HumanMessage(content=(
            f"Pregunta del estudiante:\n{state['question']}\n\n"
            f"Contexto recuperado por Qdrant:\n{retrieval}\n\n"
            "Evalúa: ¿El contexto recuperado es relevante para la pregunta? "
            "¿Faltó algún concepto importante? ¿Hay ruido o información irrelevante?"
        )),
    ]
    response = _invoke_with_retry(llm, messages)
    return {"retrieval_analysis": response.content}


def _node_improvement_synthesizer(state: AnalystState) -> dict:
    """Agent 4 — Synthesizes all analyses and produces actionable suggestions."""
    llm = state["_llm"]
    messages = [
        SystemMessage(content=(
            "Eres un coordinador de calidad pedagógica para un sistema de tutoría IA de física. "
            "Recibirás tres análisis previos y debes sintetizarlos en sugerencias de mejora concretas. "
            "Responde en español. Usa viñetas (•) para cada sugerencia."
        )),
        HumanMessage(content=(
            f"## Análisis de la pregunta\n{state['question_analysis']}\n\n"
            f"## Análisis de la respuesta\n{state['response_analysis']}\n\n"
            f"## Análisis de la recuperación (Qdrant)\n{state['retrieval_analysis']}\n\n"
            "Genera 3-5 sugerencias accionables para mejorar la calidad de las interacciones futuras. "
            "Cada sugerencia debe ser específica y práctica."
        )),
    ]
    response = _invoke_with_retry(llm, messages)
    return {"improvement_suggestions": response.content}


# ── Graph builder ────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(AnalystState)
    g.add_node("question_analyst", _node_question_analyst)
    g.add_node("response_analyst", _node_response_analyst)
    g.add_node("retrieval_analyst", _node_retrieval_analyst)
    g.add_node("improvement_synthesizer", _node_improvement_synthesizer)

    g.set_entry_point("question_analyst")
    g.add_edge("question_analyst", "response_analyst")
    g.add_edge("response_analyst", "retrieval_analyst")
    g.add_edge("retrieval_analyst", "improvement_synthesizer")
    g.add_edge("improvement_synthesizer", END)

    return g.compile()


# ── Public API ───────────────────────────────────────────────────────────────

class QAAnalyst:
    """Runs the 4-node LangGraph pipeline for a single QA interaction."""

    def __init__(self, gemini_api_key: str, model: str = "gemini-2.5-flash") -> None:
        self._llm = _make_llm(gemini_api_key, model)
        self._graph = _build_graph()

    def analyze(
        self,
        question: str,
        agent_response: str,
        retrieval_context: str = "",
    ) -> dict:
        """
        Run the full pipeline and return a dict with keys:
          - question_analysis
          - response_analysis
          - retrieval_analysis
          - improvement_suggestions
        """
        initial_state: AnalystState = {
            "question": question,
            "agent_response": agent_response,
            "retrieval_context": retrieval_context,
            "question_analysis": "",
            "response_analysis": "",
            "retrieval_analysis": "",
            "improvement_suggestions": "",
            "_llm": self._llm,
        }
        result = self._graph.invoke(initial_state)
        return {
            "question_analysis": result["question_analysis"],
            "response_analysis": result["response_analysis"],
            "retrieval_analysis": result["retrieval_analysis"],
            "improvement_suggestions": result["improvement_suggestions"],
        }
