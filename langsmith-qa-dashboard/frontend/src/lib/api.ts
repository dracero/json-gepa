export interface HistoryTurn {
  question: string;
  response: string;
}

export interface Interaction {
  trace_id: string;
  trace_timestamp: string;
  turn_index: number;
  question: string;
  agent_response: string;
  history: HistoryTurn[];
}

export interface QAItem {
  question: string;
  context: string;
  professor_response: string;
}

export interface QueryParams {
  project?: string;
  start_time?: string;
  end_time?: string;
}

const BASE_URL =
  (typeof import.meta !== "undefined" && import.meta.env?.PUBLIC_API_URL) ||
  "http://localhost:8000";

function buildUrl(path: string, params?: QueryParams): string {
  const url = new URL(path, BASE_URL);
  if (params?.project) url.searchParams.set("project", params.project);
  if (params?.start_time) url.searchParams.set("start_time", params.start_time);
  if (params?.end_time) url.searchParams.set("end_time", params.end_time);
  return url.toString();
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      message = body?.error ?? body?.detail ?? message;
    } catch {
      // ignore parse errors
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export async function fetchQADataset(params?: QueryParams): Promise<QAItem[]> {
  const res = await fetch(buildUrl("/qa-dataset", params));
  return handleResponse<QAItem[]>(res);
}

export async function fetchInteractions(params?: QueryParams): Promise<Interaction[]> {
  const res = await fetch(buildUrl("/interactions", params));
  return handleResponse<Interaction[]>(res);
}

export async function exportDataset(items: QAItem[]): Promise<Blob> {
  const res = await fetch(`${BASE_URL}/qa-dataset/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(items),
  });
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      message = body?.error ?? body?.detail ?? message;
    } catch {
      // ignore
    }
    throw new Error(message);
  }
  return res.blob();
}
