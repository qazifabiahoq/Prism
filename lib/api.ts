import type { AnalysisResponse, AssistantResponse } from "./types";

export async function analyzeTransactions(
  rows: Record<string, unknown>[]
): Promise<AnalysisResponse> {
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rows }),
  });

  if (!res.ok) {
    let message = `Analysis failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.error) message = body.error;
    } catch {
      // ignore
    }
    return { ok: false, error: message };
  }

  return res.json();
}

export async function askAssistant(
  question: string,
  context: Record<string, unknown>
): Promise<AssistantResponse> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, context }),
  });

  if (!res.ok) {
    let message = `Assistant unavailable (${res.status})`;
    try {
      const body = await res.json();
      if (body?.error) message = body.error;
    } catch {
      // ignore
    }
    return { ok: false, error: message };
  }

  return res.json();
}
