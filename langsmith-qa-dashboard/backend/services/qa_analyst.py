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

import json
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
    """Agent 1 — Runs the combined analysis in a single Gemini call."""
    llm = state["_llm"]
    messages = [
        SystemMessage(content=(
            "Eres un experto en pedagogía y didáctica universitaria, y un coordinador de calidad "
            "para sistemas de tutoría de inteligencia artificial.\n"
            "Tu tarea es analizar una interacción entre un estudiante y un tutor de IA, evaluando "
            "la pregunta, la respuesta del agente, y el contexto recuperado (RAG) de los manuales.\n"
            "Identifica el dominio temático a partir del contenido (puede ser histología, física, "
            "matemáticas, biología u otro) y proporciona análisis y sugerencias específicas y accionables.\n\n"
            "Responde EXCLUSIVAMENTE con un objeto JSON que contenga las siguientes llaves (todos los valores deben ser cadenas de texto - strings):\n"
            "{\n"
            "  \"question_analysis\": \"(3-5 oraciones en español evaluando la claridad y comprensión de la pregunta)\",\n"
            "  \"response_analysis\": \"(3-5 oraciones en español evaluando la calidad didáctica/pedagógica de la respuesta del agente IA)\",\n"
            "  \"retrieval_analysis\": \"(3-5 oraciones en español evaluando la relevancia del contexto recuperado)\",\n"
            "  \"improvement_suggestions\": \"(3-5 sugerencias de mejora accionables y específicas en formato de cadena de texto simple con saltos de línea, donde cada línea empiece con • y un espacio. Ejemplo: • Sugerencia 1\\n• Sugerencia 2\\n• Sugerencia 3)\"\n"
            "}\n"
            "No incluyas markdown (como ```json) ni texto adicional fuera del JSON."
        )),
        HumanMessage(content=(
            f"### Interacción a Analizar:\n"
            f"**Pregunta del Estudiante:**\n{state['question']}\n\n"
            f"**Respuesta del Agente:**\n{state['agent_response']}\n\n"
            f"**Contexto Recuperado (RAG):**\n{state.get('retrieval_context', '').strip() or 'No se encontró contexto de recuperación.'}\n\n"
            f"Por favor, analiza la interacción anterior y responde con el objeto JSON solicitado."
        ))
    ]
    response = _invoke_with_retry(llm, messages)
    
    # Parse JSON cleanly
    try:
        content_str = response.content.strip()
        if content_str.startswith("```"):
            # Strip markdown block if model generated it anyway
            lines = content_str.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            content_str = "\n".join(lines).strip()
            
        data = json.loads(content_str)
        return {
            "question_analysis": str(data.get("question_analysis", "")),
            "response_analysis": str(data.get("response_analysis", "")),
            "retrieval_analysis": str(data.get("retrieval_analysis", "")),
            "improvement_suggestions": str(data.get("improvement_suggestions", "")),
        }
    except Exception as exc:
        logger.exception("Error parsing combined JSON response: %s", response.content)
        # Fallback in case of json parsing error
        return {
            "question_analysis": "Error al analizar la pregunta.",
            "response_analysis": "Error al analizar la respuesta.",
            "retrieval_analysis": "Error al analizar el contexto recuperado.",
            "improvement_suggestions": f"Error: No se pudo generar la sugerencia. ({str(exc)})",
        }


def _node_response_analyst(state: AnalystState) -> dict:
    """Agent 2 — Pass-through (runs combined in node 1)"""
    return {}


def _node_retrieval_analyst(state: AnalystState) -> dict:
    """Agent 3 — Pass-through (runs combined in node 1)"""
    return {}


def _node_improvement_synthesizer(state: AnalystState) -> dict:
    """Agent 4 — Pass-through (runs combined in node 1)"""
    return {}


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
