"use client";

import { useRef, useState } from "react";
import { Bot, Loader2, Send, User } from "lucide-react";
import type { AnalysisResult } from "@/lib/types";
import { askAssistant } from "@/lib/api";

const SUGGESTIONS = [
  "How can I reduce my spending?",
  "What's my biggest expense category?",
  "Am I on track with my budget?",
  "Where should I cut back?",
];

interface Message {
  role: "user" | "assistant";
  text: string;
}

export function Assistant({ result }: { result: AnalysisResult }) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      text: "Hi, I'm Prism. Ask me anything about your financial data and I'll provide personalized advice grounded in your actual transactions.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const send = async (question: string) => {
    if (!question.trim() || loading) return;
    setError(null);
    setMessages((m) => [...m, { role: "user", text: question }]);
    setInput("");
    setLoading(true);

    const topCategory = result.categoryBreakdown[0]?.category ?? "Unknown";
    const context = {
      transactionCount: result.summary.transactionCount,
      averageAmount: result.summary.averageAmount,
      wellnessScore: result.wellness.score,
      anomalyCount: result.anomalies.count,
      topCategory,
    };

    const res = await askAssistant(question, context);
    setLoading(false);

    if (res.ok && res.answer) {
      setMessages((m) => [...m, { role: "assistant", text: res.answer as string }]);
    } else {
      setError(res.error ?? "The assistant is temporarily unavailable. Please try again.");
    }
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
  };

  return (
    <div className="mx-auto max-w-3xl">
      <div className="text-center">
        <h2 className="font-display text-2xl font-extrabold tracking-tight text-ink-950 sm:text-3xl">
          Financial Assistant
        </h2>
        <p className="mt-1.5 text-sm text-slate-500 sm:text-base">
          Powered by Llama 3.3 70B, grounded in your real spending data.
        </p>
      </div>

      <div className="card-surface mt-7 flex h-[28rem] flex-col overflow-hidden rounded-2xl shadow-card sm:h-[32rem]">
        <div className="flex-1 space-y-4 overflow-y-auto p-5 sm:p-6">
          {messages.map((m, i) => (
            <div key={i} className={`flex gap-3 ${m.role === "user" ? "flex-row-reverse" : ""}`}>
              <div
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
                  m.role === "user" ? "bg-ink-950 text-white" : "bg-accent-500/15 text-ink-700"
                }`}
              >
                {m.role === "user" ? <User size={15} /> : <Bot size={15} />}
              </div>
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  m.role === "user"
                    ? "rounded-tr-sm bg-ink-950 text-white"
                    : "rounded-tl-sm bg-surface-100 text-ink-900"
                }`}
              >
                {m.text}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent-500/15 text-ink-700">
                <Bot size={15} />
              </div>
              <div className="flex items-center gap-2 rounded-2xl rounded-tl-sm bg-surface-100 px-4 py-3 text-sm text-slate-500">
                <Loader2 size={14} className="animate-spin" /> Thinking…
              </div>
            </div>
          )}
          {error && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-2.5 text-sm text-rose-600">
              {error}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {messages.length <= 1 && (
          <div className="flex flex-wrap gap-2 border-t border-slate-100 px-5 py-3 sm:px-6">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => send(s)}
                className="focus-ring rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:border-accent-400 hover:text-ink-800"
              >
                {s}
              </button>
            ))}
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
          className="flex items-center gap-2 border-t border-slate-100 p-3 sm:p-4"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your spending…"
            className="focus-ring min-w-0 flex-1 rounded-full border border-slate-200 bg-surface-50 px-4 py-2.5 text-sm text-ink-950 outline-none placeholder:text-slate-400"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="focus-ring flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-ink-950 text-white transition hover:bg-ink-800 disabled:opacity-40"
          >
            <Send size={16} />
          </button>
        </form>
      </div>
    </div>
  );
}
