"""
ContextInferrer: heuristic-based topic and concept extractor for physics/math Q&A.

Algorithm (two steps):
1. Identify the main topic by counting keyword matches per domain (case-insensitive).
   Fallback to "Física y Matemáticas" when no matches or tie.
2. Extract up to 5 unique key concepts from:
   a. Short LaTeX expressions ($...$  or $$...$$, < 30 chars)
   b. Technical nouns matching the identified topic's keywords
   c. Terms in uppercase or with special notation (e.g. F = ma)
Output format: "<Topic>. <Concept1>, <Concept2>, ..."
"""

import re
from collections import Counter

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Mecánica Clásica": [
        "fuerza", "masa", "aceleración", "velocidad", "torque",
        "momento", "fricción", "newton", "cinemática", "dinámica",
        "trabajo", "energía cinética", "potencial gravitatorio",
        "plano inclinado", "rozamiento", "normal", "peso",
    ],
    "Termodinámica": [
        "temperatura", "calor", "entropía", "entalpía", "presión",
        "volumen", "gas ideal", "ciclo carnot", "primera ley",
        "segunda ley", "isotérmica", "adiabática",
    ],
    "Electromagnetismo": [
        "campo eléctrico", "campo magnético", "carga", "voltaje",
        "corriente", "resistencia", "capacitor", "inductor",
        "ley de gauss", "maxwell", "faraday", "onda electromagnética",
    ],
    "Óptica": [
        "luz", "refracción", "reflexión", "lente", "espejo", "difracción",
        "interferencia", "polarización", "índice de refracción", "snell",
    ],
    "Mecánica Cuántica": [
        "función de onda", "hamiltoniano", "eigenvalor", "operador",
        "schrödinger", "heisenberg", "superposición", "espín",
        "cuanto", "fotón", "probabilidad cuántica",
    ],
    "Álgebra Lineal": [
        "matriz", "vector", "determinante", "eigenvalor", "eigenvector",
        "transformación lineal", "espacio vectorial", "base", "rango",
        "producto escalar", "producto vectorial",
    ],
    "Cálculo": [
        "derivada", "integral", "límite", "serie de taylor", "gradiente",
        "divergencia", "rotacional", "ecuación diferencial", "laplaciano",
    ],
    "Estadística y Probabilidad": [
        "probabilidad", "distribución", "media", "varianza",
        "desviación estándar", "bayes", "esperanza",
        "variable aleatoria", "histograma",
    ],
    "Trabajo y Energía": [
        "trabajo", "energía potencial", "energía cinética", "conservativa",
        "no conservativa", "trayectoria cerrada", "integral de línea",
        "fuerza conservativa", "energía mecánica",
    ],
}

_FALLBACK_TOPIC = "Física y Matemáticas"
_MAX_CONCEPTS = 5
_MAX_LATEX_LEN = 30

# Regex patterns
_LATEX_BLOCK = re.compile(r'\$\$([^$]+)\$\$')
_LATEX_INLINE = re.compile(r'\$([^$\n]+)\$')
_SPECIAL_NOTATION = re.compile(
    r'\b([A-Z][a-z]?\s*[=<>≤≥]\s*[A-Za-z0-9\s\+\-\*\/\^\.]+)\b'
)


class ContextInferrer:
    """Infers topic and key concepts from a student question and agent response."""

    def infer(self, question: str, agent_response: str) -> str:
        """Return context string: '<Topic>. <Concept1>, <Concept2>, ...'"""
        combined = f"{question} {agent_response}"
        topic = self._identify_topic(combined)
        concepts = self._extract_concepts(combined, topic)
        return f"{topic}. {', '.join(concepts)}"

    def _identify_topic(self, text: str) -> str:
        text_lower = text.lower()
        scores: Counter = Counter()
        for topic, keywords in TOPIC_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    scores[topic] += 1
        if not scores:
            return _FALLBACK_TOPIC
        max_score = max(scores.values())
        top_topics = [t for t, s in scores.items() if s == max_score]
        # Deterministic tie-break: first in TOPIC_KEYWORDS order
        for topic in TOPIC_KEYWORDS:
            if topic in top_topics:
                return topic
        return _FALLBACK_TOPIC

    def _extract_concepts(self, text: str, topic: str) -> list[str]:
        seen: set[str] = set()
        concepts: list[str] = []

        def add(c: str) -> None:
            c = c.strip()
            if c and c not in seen and len(concepts) < _MAX_CONCEPTS:
                seen.add(c)
                concepts.append(c)

        # 1. Short LaTeX block expressions $$...$$
        for m in _LATEX_BLOCK.finditer(text):
            expr = m.group(1).strip()
            if len(expr) < _MAX_LATEX_LEN:
                add(expr)

        # 2. Short LaTeX inline expressions $...$
        for m in _LATEX_INLINE.finditer(text):
            expr = m.group(1).strip()
            if len(expr) < _MAX_LATEX_LEN:
                add(expr)

        # 3. Topic keywords found in text
        text_lower = text.lower()
        for kw in TOPIC_KEYWORDS.get(topic, []):
            if kw.lower() in text_lower:
                add(kw)

        # 4. Special notation patterns (e.g. F = ma)
        for m in _SPECIAL_NOTATION.finditer(text):
            add(m.group(1).strip())

        # Ensure at least one concept (fallback to topic name)
        if not concepts:
            add(topic)

        return concepts
