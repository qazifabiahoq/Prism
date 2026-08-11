"use client";

import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { AlertOctagon, ShieldCheck } from "lucide-react";
import type { AnalysisResult } from "@/lib/types";
import { formatCurrency, formatDateLong } from "@/lib/format";

export function Alerts({ result }: { result: AnalysisResult }) {
  const { anomalies } = result;

  const normalPoints = anomalies.scatter
    .filter((p) => !p.isAnomaly)
    .map((p) => ({ ...p, x: p.date ? new Date(p.date).getTime() : 0 }));
  const anomalyPoints = anomalies.scatter
    .filter((p) => p.isAnomaly)
    .map((p) => ({ ...p, x: p.date ? new Date(p.date).getTime() : 0 }));

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <div>
        <h2 className="font-display text-2xl font-extrabold tracking-tight text-navy-950 sm:text-3xl">
          Activity Alerts
        </h2>
        <p className="mt-1.5 text-sm text-slate-500 sm:text-base">
          Flagged by an Isolation Forest model trained on amount, timing, and volatility.
        </p>
      </div>

      {anomalies.count === 0 ? (
        <div className="flex flex-col items-center gap-3 rounded-2xl border border-emerald-200 bg-emerald-50 px-6 py-14 text-center">
          <ShieldCheck size={36} className="text-emerald-500" />
          <p className="text-base font-bold text-emerald-800">
            No unusual activity detected in your transactions.
          </p>
          <p className="text-sm text-emerald-700/80">Your spending pattern looks consistent. Nice work.</p>
        </div>
      ) : (
        <>
          <div className="flex items-center gap-3 rounded-2xl border border-rose-200 bg-rose-50 px-6 py-4">
            <AlertOctagon size={20} className="shrink-0 text-rose-500" />
            <p className="text-sm font-bold text-rose-700">
              {anomalies.count} unusual transaction{anomalies.count === 1 ? "" : "s"} detected (
              {anomalies.rate.toFixed(1)}% of activity)
            </p>
          </div>

          <div className="card-surface rounded-2xl p-6 shadow-card sm:p-7">
            <div className="mb-4 text-[0.7rem] font-bold uppercase tracking-wider text-slate-400">
              Normal vs Unusual Transactions
            </div>
            <div className="h-72 sm:h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ left: -10, right: 10, top: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F3" />
                  <XAxis
                    dataKey="x"
                    type="number"
                    domain={["dataMin", "dataMax"]}
                    tickFormatter={(v) => new Date(v).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                    tick={{ fontSize: 11, fill: "#94A3B8" }}
                    axisLine={{ stroke: "#E2E8F3" }}
                    tickLine={false}
                  />
                  <YAxis
                    dataKey="amount"
                    tickFormatter={(v) => `$${v}`}
                    tick={{ fontSize: 11, fill: "#94A3B8" }}
                    axisLine={false}
                    tickLine={false}
                    width={56}
                  />
                  <ZAxis range={[50, 50]} />
                  <Tooltip
                    formatter={(value: number) => formatCurrency(value)}
                    labelFormatter={() => ""}
                    contentStyle={{ borderRadius: 12, border: "1px solid #E2E8F3", fontSize: 13 }}
                  />
                  <Scatter data={normalPoints} fill="#0F2A63" fillOpacity={0.6} name="Normal" />
                  <Scatter data={anomalyPoints} fill="#E11D48" shape="cross" name="Unusual" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-3 flex items-center justify-center gap-6 text-xs font-medium text-slate-500">
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-navy-800" /> Normal
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-rose-500" /> Unusual
              </span>
            </div>
          </div>

          <div className="card-surface overflow-hidden rounded-2xl shadow-card">
            <div className="border-b border-slate-100 px-6 py-4 text-[0.7rem] font-bold uppercase tracking-wider text-slate-400 sm:px-7">
              Flagged Transactions
            </div>
            <div className="divide-y divide-slate-100">
              {anomalies.items.map((item, i) => (
                <div key={i} className="flex items-center justify-between gap-4 px-6 py-4 sm:px-7">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold text-navy-950">
                      {item.description || "Unlabeled transaction"}
                    </div>
                    <div className="mt-0.5 text-xs text-slate-400">
                      {formatDateLong(item.date)} · {item.category}
                    </div>
                  </div>
                  <div className="shrink-0 text-right">
                    <div className="font-display text-sm font-bold tabular text-rose-600">
                      {formatCurrency(item.amount)}
                    </div>
                    <div className="text-[0.7rem] text-slate-400">z-score {item.zScore.toFixed(1)}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
