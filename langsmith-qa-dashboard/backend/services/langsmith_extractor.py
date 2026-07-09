import json
import logging
from datetime import datetime, timedelta, timezone

from langsmith import Client

from models.schemas import HistoryTurn, Interaction

logger = logging.getLogger(__name__)

# Fields we actually need — avoids fetching heavy default fields.
_SELECT = ["id", "name", "run_type", "start_time", "inputs", "outputs", "trace_id",
           "parent_run_id"]


class LangSmithExtractor:
    def __init__(self, api_key: str, project: str = "socratico_test") -> None:
        self.client = Client(api_key=api_key)
        self.default_project = project
        # Cached project UUIDs — keyed by project name to support multiple projects
        self._project_id_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_interactions(
        self,
        project: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Interaction]:
        """Fetch all student-agent interaction pairs from LangSmith traces."""
        project_name = project or self.default_project
        project_id = self._resolve_project_id(project_name)
        start, end = self._compute_default_time_range(start_time, end_time)

        # ── SINGLE bulk fetch ──────────────────────────────────────────
        # Fetch root runs and all child LLM + retriever runs in bulk,
        # then group by trace_id in-memory.
        root_runs = self._get_root_runs(project_id, start, end)
        root_ids = {str(r.id) for r in root_runs}
        if not root_ids:
            return []

        root_run_map = {str(r.id): r for r in root_runs}

        # Fetch ALL llm runs (no name filter — works for any model).
        all_llm_runs = self._get_all_llm_runs(project_id, start, end)

        # Fetch ALL retriever runs (e.g. search_qdrant) for vector DB context.
        all_retriever_runs = self._get_all_retriever_runs(project_id, start, end)

        # Group LLM runs: trace_id → sorted list of llm runs
        trace_to_runs: dict[str, list] = {}
        for run in all_llm_runs:
            tid = str(run.trace_id) if run.trace_id else str(run.id)
            trace_to_runs.setdefault(tid, []).append(run)

        # For root runs that ARE the LLM call (trace_id == run.id), also
        # include them directly if they didn't appear as children.
        for run in all_llm_runs:
            if str(run.id) in root_ids and str(run.id) not in trace_to_runs:
                trace_to_runs[str(run.id)] = [run]

        # Group retriever runs: trace_id → list of retriever runs
        trace_to_retrievers: dict[str, list] = {}
        for run in all_retriever_runs:
            tid = str(run.trace_id) if run.trace_id else str(run.id)
            trace_to_retrievers.setdefault(tid, []).append(run)

        interactions: list[Interaction] = []

        # Process root runs in chronological order
        sorted_root_runs = sorted(root_runs, key=lambda r: r.start_time)
        for root_run in sorted_root_runs:
            root_id = str(root_run.id)

            child_runs = trace_to_runs.get(root_id, [])
            # Exclude the root run itself when it reappears as a child.
            child_runs = [r for r in child_runs if str(r.id) != root_id or len(child_runs) == 1]
            child_runs.sort(key=lambda r: r.start_time)

            # Pre-scan for retrieval context from multiple sources:
            # 1. Root run outputs (directly from pipeline state output e.g. contexto_documentos)
            # 2. Retriever runs (search_qdrant outputs — vector DB results)
            # 3. LLM message markers (SECCIONES DEL MANUAL — rag-histologia style)
            # 4. Embedded tracing context in HumanMessage (standalone pipeline steps)
            trace_retrieval_context = self._extract_root_context(root_run)

            # Source 2: Retriever run outputs (vector DB results)
            if not trace_retrieval_context:
                retriever_runs = trace_to_retrievers.get(root_id, [])
                for rr in reversed(retriever_runs):
                    ctx = self._extract_retriever_context(rr)
                    if ctx:
                        trace_retrieval_context = ctx
                        break

            # Source 3: LLM message markers (fallback)
            if not trace_retrieval_context:
                for run in reversed(child_runs):
                    ctx = self._extract_retrieval_context(run)
                    if ctx:
                        trace_retrieval_context = ctx
                        break

            # Source 4: Embedded tracing context in HumanMessage (standalone runs)
            if not trace_retrieval_context:
                for run in reversed(child_runs):
                    ctx = self._extract_embedded_tracing_context(run)
                    if ctx:
                        trace_retrieval_context = ctx
                        break

            # Direct root Q&A extraction (best for single-turn LangGraph/pipeline runs)
            root_inputs = root_run.inputs or {}
            root_outputs = root_run.outputs or {}
            root_question = None
            for key in ["consulta_in", "consulta_usuario", "question", "query", "input", "prompt"]:
                val = root_inputs.get(key)
                if val and isinstance(val, str) and val.strip():
                    root_question = val.strip()
                    break
            if not root_question:
                for key in ["consulta_usuario", "question", "query", "input"]:
                    val = root_outputs.get(key)
                    if val and isinstance(val, str) and val.strip():
                        root_question = val.strip()
                        break

            root_response = None
            for key in ["respuesta_final", "output", "response", "final_response", "agent_response"]:
                val = root_outputs.get(key)
                if val and isinstance(val, str) and val.strip():
                    root_response = val.strip()
                    break

            if root_question and root_response:
                interactions.append(
                    Interaction(
                        trace_id=root_id,
                        trace_timestamp=root_run.start_time,
                        turn_index=0,
                        question=root_question,
                        agent_response=root_response,
                        retrieval_context=trace_retrieval_context,
                        history=[],
                    )
                )
                continue

            history: list[HistoryTurn] = []
            for turn_index, run in enumerate(child_runs):
                question = self._extract_question(run)
                if question is None:
                    logger.warning("Skipping run %s: no valid human message", run.id)
                    continue

                agent_response = self._extract_response(run)
                if agent_response is None:
                    logger.warning("Skipping run %s: no valid agent response", run.id)
                    continue

                # Skip internal pipeline steps that aren't real interactions
                if self._is_pipeline_intermediate_step(run, question, agent_response):
                    continue

                interactions.append(
                    Interaction(
                        trace_id=root_id,
                        trace_timestamp=root_run.start_time,
                        turn_index=turn_index,
                        question=question,
                        agent_response=agent_response,
                        retrieval_context=trace_retrieval_context,
                        history=list(history),
                    )
                )
                history.append(HistoryTurn(question=question, response=agent_response))

        return interactions

    # ------------------------------------------------------------------
    # Project ID caching
    # ------------------------------------------------------------------

    def _resolve_project_id(self, project_name: str) -> str:
        """Return the project UUID, resolved once per project name and cached."""
        if project_name not in self._project_id_cache:
            session = self.client.read_project(project_name=project_name)
            self._project_id_cache[project_name] = str(session.id)
        return self._project_id_cache[project_name]

    # ------------------------------------------------------------------
    # Bulk run fetching (2 HTTP calls total instead of N+1)
    # ------------------------------------------------------------------

    def _compute_default_time_range(
        self,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> tuple[datetime, datetime]:
        now = datetime.now(timezone.utc)
        start = start_time if start_time is not None else now - timedelta(days=7)
        end = end_time if end_time is not None else now
        return start, end

    def _get_root_runs(self, project_id: str, start: datetime, end: datetime):
        """HTTP call 1: fetch root runs (llm + chain only)."""
        # end_time must be an ISO string: the SDK auto-serializes start_time
        # but forwards extra kwargs raw → TypeError for datetime objects.
        all_roots = list(
            self.client.list_runs(
                project_id=project_id,
                is_root=True,
                start_time=start,
                end_time=end.isoformat(),
                select=_SELECT,
            )
        )
        return [r for r in all_roots if r.run_type in ("llm", "chain")]

    def _get_all_llm_runs(self, project_id: str, start: datetime, end: datetime):
        """HTTP call 2: fetch ALL llm runs in the window at once.

        No name filter — works with any LLM provider (ChatGroq, Gemini, etc.).
        """
        return list(
            self.client.list_runs(
                project_id=project_id,
                run_type="llm",
                start_time=start,
                end_time=end.isoformat(),
                select=_SELECT,
            )
        )

    def _get_all_retriever_runs(self, project_id: str, start: datetime, end: datetime):
        """HTTP call 3: fetch ALL retriever runs (e.g. search_qdrant)."""
        return list(
            self.client.list_runs(
                project_id=project_id,
                run_type="retriever",
                start_time=start,
                end_time=end.isoformat(),
                select=_SELECT,
            )
        )

    # ------------------------------------------------------------------
    # Message parsing helpers
    # ------------------------------------------------------------------

    def _parse_message(self, msg) -> tuple[str, str]:
        """Return (role, content) from any message representation.

        Handles three formats:
        1. Plain dict: {'role': 'user', 'content': '...'}  /  {'type': 'human', ...}
        2. LangChain serialised dict:
               {'id': [..., 'HumanMessage'], 'kwargs': {'content': '...'}}
        3. LangChain message object with .type / .content attributes.
        """
        if isinstance(msg, dict):
            # LangChain serialised format: {'id': [..., 'HumanMessage'], 'kwargs': {...}}
            if "id" in msg and "kwargs" in msg:
                lc_id: list = msg["id"]
                class_name: str = lc_id[-1] if lc_id else ""
                role = self._class_name_to_role(class_name)
                kwargs = msg.get("kwargs", {})
                content = kwargs.get("content", "")
                return role, str(content) if content else ""

            # Plain dict format
            role = msg.get("role") or msg.get("type", "unknown")
            content = msg.get("content", "")
            return role, str(content) if content else ""

        # Object with attributes (langchain_core message objects)
        role = getattr(msg, "type", type(msg).__name__)
        content = getattr(msg, "content", "")
        return role, str(content) if content else ""

    @staticmethod
    def _class_name_to_role(class_name: str) -> str:
        mapping = {
            "HumanMessage": "human",
            "AIMessage": "ai",
            "SystemMessage": "system",
            "FunctionMessage": "function",
            "ToolMessage": "tool",
            "ChatMessage": "chat",
        }
        return mapping.get(class_name, class_name.lower())

    def _flatten_messages(self, raw_messages) -> list:
        """LangChain serialises messages as [[msg1, msg2]] (list-of-lists)
        when the prompt template wraps them.  Flatten one level if needed."""
        if not raw_messages:
            return []
        if isinstance(raw_messages[0], list):
            flat: list = []
            for item in raw_messages:
                flat.extend(item)
            return flat
        return raw_messages

    def _extract_question(self, run) -> str | None:
        """Extract the student's actual question from run inputs.

        The agent constructs a compound message like:

            HALLAZGOS VISUALES: ...
            CONTEXTO: ...
            CONSULTA: <student question>

        We try, in order:
        1. The text after 'CONSULTA:' in the last human message.
        2. The last 'Usuario:' line in the 'Historial' block.
        3. The raw last human message (fallback for simple turns).
        """
        try:
            raw = run.inputs.get("messages", [])
            messages = self._flatten_messages(raw)
            human_messages = []
            for msg in messages:
                role, content = self._parse_message(msg)
                if role in ("human", "user") and content:
                    human_messages.append(content)
            if not human_messages:
                return None

            last_human = human_messages[-1]

            # Strategy 1: extract the CONSULTA block (everything after the marker)
            import re
            consulta_match = re.search(
                r'CONSULTA:\s*(.+)',
                last_human,
                re.DOTALL | re.IGNORECASE,
            )
            if consulta_match:
                consulta = consulta_match.group(1).strip()
                if consulta:
                    return consulta

            # Strategy 2: last 'Usuario: ...' line in Historial block
            historial_match = re.findall(
                r'(?:^|\n)Usuario:\s*(.+?)(?=\nAsistente:|\nUsuario:|$)',
                last_human,
                re.DOTALL,
            )
            if historial_match:
                last_user_turn = historial_match[-1].strip()
                if last_user_turn:
                    return last_user_turn

            # Strategy 3: raw fallback — strip preamble keywords so at least
            # the question section is cleaner.
            for marker in ("HALLAZGOS VISUALES:", "CONTEXTO:"):
                if marker in last_human:
                    # Return the portion before the first marker (usually empty)
                    # or just the full text — better than nothing.
                    pass
            return last_human
        except Exception:
            logger.exception("Error extracting question from run %s", run.id)
            return None

    def _extract_response(self, run) -> str | None:
        """Extract the agent response text from run outputs."""
        try:
            outputs = run.outputs or {}
            generations = outputs.get("generations", [])
            if generations and generations[0]:
                first = generations[0][0]
                if isinstance(first, dict):
                    text = first.get("text") or first.get("message", {}).get("content")
                    if text:
                        return text
            return outputs.get("output") or outputs.get("content") or None
        except Exception:
            logger.exception("Error extracting response from run %s", run.id)
            return None

    @staticmethod
    def _raw_text_from_message(msg) -> tuple[str, str]:
        """Extract (role, full_text) from a message, handling list-of-dicts content.

        Unlike _parse_message (which calls str() on non-string content),
        this method properly joins multimodal text parts so that marker
        searches work reliably.
        """
        if not isinstance(msg, dict):
            role = getattr(msg, "type", type(msg).__name__)
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                return role, " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            return role, str(content) if content else ""

        # LangChain serialised format
        if "id" in msg and "kwargs" in msg:
            lc_id: list = msg["id"]
            class_name = lc_id[-1] if lc_id else ""
            role_map = {
                "HumanMessage": "human", "AIMessage": "ai",
                "SystemMessage": "system",
            }
            role = role_map.get(class_name, class_name.lower())
            content = msg.get("kwargs", {}).get("content", "")
        else:
            role = msg.get("role") or msg.get("type", "unknown")
            content = msg.get("content", "")

        if isinstance(content, list):
            return role, " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        return role, str(content) if content else ""

    def _extract_retriever_context(self, run) -> str:
        """Extract context from a retriever run (e.g. search_qdrant outputs).

        Retriever runs store vector DB results in outputs.text (list of
        scored chunks) and outputs.image (list of scored image matches).
        """
        try:
            outputs = run.outputs if run.outputs else {}
            parts: list[str] = []

            # Text chunks from vector DB
            text_results = outputs.get("text", [])
            if isinstance(text_results, list):
                for i, item in enumerate(text_results[:8]):  # limit to top 8
                    if isinstance(item, dict):
                        payload = item.get("payload", {})
                        score = item.get("score", 0)
                        text = payload.get("text", "")
                        pdf = payload.get("pdf_name", "").split("/")[-1] if payload.get("pdf_name") else ""
                        chunk_id = payload.get("chunk_id", "")
                        if text:
                            header = f"[Sección {i+1} | Fuente: {pdf} | Tipo: texto | Sim: {score:.3f}]"
                            # Truncate very long chunks
                            snippet = text[:500] + "..." if len(text) > 500 else text
                            parts.append(f"{header}\n{snippet}")

            # Image matches from vector DB
            image_results = outputs.get("image", [])
            if isinstance(image_results, list):
                for i, item in enumerate(image_results[:3]):  # limit to top 3
                    if isinstance(item, dict):
                        payload = item.get("payload", {})
                        score = item.get("score", 0)
                        img_path = payload.get("image_path", "")
                        pdf = payload.get("pdf_name", "").split("/")[-1] if payload.get("pdf_name") else ""
                        if img_path:
                            img_name = img_path.split("/")[-1] if img_path else ""
                            parts.append(
                                f"[Sección {len(text_results)+i+1} ⭐ MEJOR MATCH VISUAL | "
                                f"Fuente: {pdf} | Tipo: imagen | Sim: {score:.3f} | "
                                f"Imagen: {img_name}]"
                            )

            return "\n\n".join(parts) if parts else ""
        except Exception:
            logger.exception("Error extracting retriever context from run %s", run.id)
            return ""

    def _extract_retrieval_context(self, run) -> str:
        """Extract the retrieved manual sections from LLM run inputs (marker-based)."""
        try:
            raw = run.inputs.get("messages", []) if run.inputs else []
            messages = self._flatten_messages(raw)
            for msg in messages:
                role, content_str = self._raw_text_from_message(msg)
                if role not in ("human", "user") or not content_str:
                    continue

                # Try several marker variants (bold markdown, plain, etc.)
                markers_to_try = [
                    "**SECCIONES DEL MANUAL:**",   # bold markdown (most common)
                    "SECCIONES DEL MANUAL:**",
                    "SECCIONES DEL MANUAL:",
                ]
                idx = -1
                used_marker = ""
                for m in markers_to_try:
                    idx = content_str.find(m)
                    if idx != -1:
                        used_marker = m
                        break

                if idx == -1:
                    continue

                context_part = content_str[idx + len(used_marker):].strip()
                next_markers = [
                    "**ANÁLISIS COMPARATIVO:**",
                    "**HISTORIAL DE CONVERSACIÓN:**",
                    "ESTILO DE RESPUESTA:",
                    "INSTRUCCIONES PARA REFERENCIAS",
                    "REGLAS CRÍTICAS:",
                    "Responde EXCLUSIVAMENTE",
                    "**CONSULTA:**",
                    "CONSULTA:",
                ]
                min_next_idx = len(context_part)
                for next_marker in next_markers:
                    nxt_idx = context_part.find(next_marker)
                    if nxt_idx != -1 and nxt_idx < min_next_idx:
                        min_next_idx = nxt_idx

                retrieved = context_part[:min_next_idx].strip()
                # Clean leading markdown artifacts
                retrieved = retrieved.lstrip("*").lstrip(":").strip()
                if retrieved:
                    return retrieved
            return ""
        except Exception:
            logger.exception("Error extracting retrieval context from run %s", run.id)
            return ""

    def _extract_embedded_tracing_context(self, run) -> str:
        """Extract tracing context embedded in standalone LLM run messages.

        For runs that are standalone pipeline steps (root ChatGroq calls),
        the HumanMessage contains the full tracing context from previous
        pipeline stages: classification, visual findings, conversation
        context, medical keywords, etc.

        We extract the full HumanMessage content MINUS the original user
        query, which gives us the pipeline's assembled context.
        """
        try:
            raw = run.inputs.get("messages", []) if run.inputs else []
            messages = self._flatten_messages(raw)

            for msg in messages:
                role, content_str = self._raw_text_from_message(msg)
                if role not in ("human", "user") or not content_str:
                    continue

                # Check for known tracing markers that indicate this message
                # contains embedded pipeline context
                tracing_markers = [
                    "CLASIFICACIÓN MÉDICA:",
                    "CLASIFICACIÓN:",
                    "HALLAZGOS VISUALES",
                    "HALLAZGOS_RELEVANTES:",
                    "KEYWORDS_MEDICAS:",
                    "CONTEXTO DE CONVERSACIÓN",
                    "CONTEXTO PREVIO:",
                    "CONTEXTO:",
                ]

                found_marker = False
                for marker in tracing_markers:
                    if marker in content_str:
                        found_marker = True
                        break

                if not found_marker:
                    continue

                # Extract everything after **CONSULTA ORIGINAL** header
                # (the rest is the tracing context)
                # Strip the user's original query to leave only context
                query_markers = [
                    "**CONSULTA ORIGINAL DEL USUARIO:**",
                    "**CONSULTA ORIGINAL:**",
                    "CONSULTA ORIGINAL DEL USUARIO:",
                    "CONSULTA ORIGINAL:",
                ]

                context_parts = []

                # Find and skip the original query line
                for qm in query_markers:
                    qm_idx = content_str.find(qm)
                    if qm_idx != -1:
                        # Get the query line (next line after the marker)
                        after_qm = content_str[qm_idx + len(qm):].strip()
                        query_end = after_qm.find("\n\n")
                        if query_end == -1:
                            query_end = after_qm.find("\n")
                        # Everything after the query is context
                        if query_end != -1:
                            remaining = after_qm[query_end:].strip()
                            if remaining:
                                context_parts.append(remaining)
                        break

                if not context_parts:
                    # Fallback: return the full human message content as context
                    # (better than nothing)
                    context_parts.append(content_str)

                result = "\n".join(context_parts).strip()
                # Truncate if excessively long
                if len(result) > 3000:
                    result = result[:3000] + "\n... [contexto truncado]"
                return result

            return ""
        except Exception:
            logger.exception("Error extracting embedded tracing context from run %s", run.id)
            return ""

    def _extract_root_context(self, root_run) -> str:
        """Extract retrieved context from the root run's outputs (e.g. state keys)."""
        if not root_run or not root_run.outputs:
            return ""

        outputs = root_run.outputs
        
        onto = outputs.get("contexto_ontologico")
        onto_str = ""
        if onto and isinstance(onto, str) and onto.strip():
            onto_str = f"**CONTEXTO ONTOLÓGICO:**\n{onto.strip()}"

        doc_context = ""
        # Try various keys for retrieved document context
        for key in ["contexto_documentos", "retrieved_context", "retrieval_context", "context"]:
            val = outputs.get(key)
            if val and isinstance(val, str) and val.strip():
                doc_context = val.strip()
                break

        if not doc_context:
            busqueda = outputs.get("resultados_busqueda")
            if busqueda and isinstance(busqueda, list):
                parts = []
                for i, item in enumerate(busqueda):
                    if isinstance(item, dict):
                        payload = item.get("payload", {})
                        score = item.get("score", 0)
                        text = payload.get("text") or payload.get("texto", "")
                        pdf = payload.get("pdf_name", "").split("/")[-1] if payload.get("pdf_name") else ""
                        if text:
                            header = f"[Sección {i+1} | Fuente: {pdf} | Sim: {score}]"
                            parts.append(f"{header}\n{text}")
                if parts:
                    doc_context = "\n\n".join(parts)

        # Merge them
        merged = []
        if doc_context:
            merged.append(doc_context)
        if onto_str:
            merged.append(onto_str)

        return "\n\n".join(merged) if merged else ""

    @staticmethod
    def _is_pipeline_intermediate_step(run, question: str, response: str) -> bool:
        """Return True if this run is an internal pipeline step, not a real interaction.

        Internal steps include routing decisions, intent classifiers, visual
        analysis, and search query generators that get logged as standalone
        LLM runs but don't represent actual student-agent conversations.
        """
        # Filter by run name (named pipeline steps)
        run_name = (run.name or "").lower()
        intermediate_names = {
            "classify_query", "generate_search_query",
            "generate_socratic_question",
        }
        if run_name in intermediate_names:
            return True

        # Filter by question content patterns (system prompts leaked as questions)
        q_lower = question.lower()
        intermediate_question_patterns = [
            "you are a routing system",
            "eres un clasificador de intención",
            "clasificador de intención",
            "analiza estas",  # "Analiza estas N imágenes médicas"
            "analyze the user's request and decide",
        ]
        for pattern in intermediate_question_patterns:
            if pattern in q_lower:
                return True

        # Filter by very short responses that are classification outputs
        r_stripped = response.strip()
        classification_responses = {
            "CONTINUAR", "SALIR", "SI", "NO", "TEXTO",
        }
        if r_stripped in classification_responses:
            return True

        # Filter routing responses (agent name selections)
        routing_responses = {
            "tutor socrático de física multimodal",
            "asistente médico",
        }
        if r_stripped.lower() in routing_responses:
            return True

        return False

    def _build_history(
        self, turns: list[tuple[str, str]], current_index: int
    ) -> list[HistoryTurn]:
        """Return accumulated history of turns before current_index."""
        return [
            HistoryTurn(question=q, response=r)
            for q, r in turns[:current_index]
        ]
