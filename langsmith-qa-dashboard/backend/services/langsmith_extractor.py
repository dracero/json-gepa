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
        # Cached project UUID — resolved once to skip the name→id HTTP call
        # that the SDK makes internally on every list_runs() invocation.
        self._project_id: str | None = None

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
        # Instead of 1 call per root-run to get its children (N×HTTP), we
        # make ONE call that fetches every ChatGroq llm run in the time
        # window, then group them by trace_id in-memory.  Dramatically
        # faster for projects with many short traces.
        root_ids = {
            str(r.id)
            for r in self._get_root_runs(project_id, start, end)
        }
        if not root_ids:
            return []

        # Fetch ALL ChatGroq llm runs in the window at once.
        all_llm_runs = self._get_all_llm_runs(project_id, start, end)

        # Group: trace_id → sorted list of llm runs
        trace_to_runs: dict[str, list] = {}
        for run in all_llm_runs:
            tid = str(run.trace_id) if run.trace_id else str(run.id)
            trace_to_runs.setdefault(tid, []).append(run)

        # For root runs that ARE the LLM call (trace_id == run.id), also
        # include them directly if they didn't appear as children.
        for run in all_llm_runs:
            if str(run.id) in root_ids and str(run.id) not in trace_to_runs:
                trace_to_runs[str(run.id)] = [run]

        interactions: list[Interaction] = []
        # We need root run objects (for trace_timestamp) — rebuild from
        # the root_ids set using what we already fetched.
        root_run_map = {
            str(r.id): r
            for r in self._get_root_runs(project_id, start, end)
        }

        for root_id in root_ids:
            root_run = root_run_map.get(root_id)
            if root_run is None:
                continue

            child_runs = trace_to_runs.get(root_id, [])
            # Exclude the root run itself when it reappears as a child.
            child_runs = [r for r in child_runs if str(r.id) != root_id or len(child_runs) == 1]
            child_runs.sort(key=lambda r: r.start_time)

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

                interactions.append(
                    Interaction(
                        trace_id=root_id,
                        trace_timestamp=root_run.start_time,
                        turn_index=turn_index,
                        question=question,
                        agent_response=agent_response,
                        history=list(history),
                    )
                )
                history.append(HistoryTurn(question=question, response=agent_response))

        return interactions

    # ------------------------------------------------------------------
    # Project ID caching
    # ------------------------------------------------------------------

    def _resolve_project_id(self, project_name: str) -> str:
        """Return the project UUID, resolved once and cached."""
        if self._project_id is None:
            session = self.client.read_project(project_name=project_name)
            self._project_id = str(session.id)
        return self._project_id

    # ------------------------------------------------------------------
    # Bulk run fetching (2 HTTP calls total instead of N+1)
    # ------------------------------------------------------------------

    def _compute_default_time_range(
        self,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> tuple[datetime, datetime]:
        now = datetime.now(timezone.utc)
        start = start_time if start_time is not None else now - timedelta(hours=48)
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
        """HTTP call 2: fetch ALL ChatGroq llm runs in the window at once.

        By filtering with filter='eq(name, "ChatGroq")' we get every LLM
        turn across all traces in a single paginated request.  The SDK
        handles pagination automatically via its cursor iterator.
        """
        return list(
            self.client.list_runs(
                project_id=project_id,
                run_type="llm",
                filter='eq(name, "ChatGroq")',
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

    def _build_history(
        self, turns: list[tuple[str, str]], current_index: int
    ) -> list[HistoryTurn]:
        """Return accumulated history of turns before current_index."""
        return [
            HistoryTurn(question=q, response=r)
            for q, r in turns[:current_index]
        ]
