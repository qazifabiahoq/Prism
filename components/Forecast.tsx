"use client";

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BarChart3, CalendarClock, TrendingDown, TrendingUp } from "lucide-react";
import type { AnalysisResult } from "@/lib/types";
import { formatCurrency, formatDateShort } from "@/lib/format";
import { StatCard } from "./StatCard";

export function Forecast({ result }: { result: AnalysisResult }) {
  const forecast = result.forecast;

  if (!forecast || !forecast.available) {
    return (
      <div className="mx-auto max-w-3xl rounded-2xl border border-amber-200 bg-amber-50 p-8 text-center">
        <p className="text-sm font-semibold text-amber-800">
          We need at least 20 transactions to train a reliable forecast model.
          Upload more history to unlock this tab.
        </p>
      </div>
    );
  }

  const chartData = [
    ...forecast.history.map((h) => ({ date: h.date, historical: h.amount, forecast: null as number | null })),
    ...forecast.predictions.map((p) => ({ date: p.date, historical: null as number | null, forecast: p.amount })),
  ];
  // Bridge the two series at the seam so the line connects visually
  if (forecast.history.length && forecast.predictions.length) {
    chartData[forecast.history.length - 1].forecast = forecast.history[forecast.history.length - 1].amount;
  }

  const changeIsGood = forecast.vsCurrentAveragePct < 0;

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <div>
        <h2 className="font-display text-2xl font-extrabold tracking-tight text-navy-950 sm:text-3xl">
          7-Day Spending Forecast
        </h2>
        <p className="mt-1.5 text-sm text-slate-500 sm:text-base">
          Predicted using a RandomForest ensemble trained on your transaction history.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard label="Week Total" value={formatCurrency(forecast.weekTotal)} icon={CalendarClock} />
        <StatCard label="Daily Average" value={formatCurrency(forecast.dailyAverage)} icon={BarChart3} />
        <StatCard
          label="vs Current Average"
          value={`${forecast.vsCurrentAveragePct > 0 ? "+" : ""}${forecast.vsCurrentAveragePct.toFixed(1)}%`}
          icon={changeIsGood ? TrendingDown : TrendingUp}
          tone={changeIsGood ? "positive" : "negative"}
        />
      </div>

      <div className="card-surface rounded-2xl p-6 shadow-card sm:p-7">
        <div className="mb-1 text-[0.7rem] font-bold uppercase tracking-wider text-slate-400">
          Historical Spending + 7-Day Forecast
        </div>
        <div className="mt-4 h-72 sm:h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ left: -10, right: 10, top: 10 }}>
              <defs>
                <linearGradient id="histGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#0F2A63" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#0F2A63" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="fcGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#14B8A6" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#14B8A6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F3" vertical={false} />
              <XAxis
                dataKey="date"
                tickFormatter={(v) => formatDateShort(v)}
                tick={{ fontSize: 11, fill: "#94A3B8" }}
                axisLine={{ stroke: "#E2E8F3" }}
                tickLine={false}
              />
              <YAxis
                tickFormatter={(v) => `$${v}`}
                tick={{ fontSize: 11, fill: "#94A3B8" }}
                axisLine={false}
                tickLine={false}
                width={56}
              />
              <Tooltip
                labelFormatter={(v) => formatDateShort(v as string)}
                formatter={(value: number, name: string) => [
                  formatCurrency(value),
                  name === "historical" ? "Historical" : "Forecast",
                ]}
                contentStyle={{ borderRadius: 12, border: "1px solid #E2E8F3", fontSize: 13 }}
              />
              <Area
                type="monotone"
                dataKey="historical"
                stroke="#0F2A63"
                strokeWidth={2.5}
                fill="url(#histGrad)"
                connectNulls={false}
              />
              <Area
                type="monotone"
                dataKey="forecast"
                stroke="#14B8A6"
                strokeWidth={2.5}
                strokeDasharray="5 4"
                fill="url(#fcGrad)"
                connectNulls
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-3 flex items-center justify-center gap-6 text-xs font-medium text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="h-0.5 w-4 rounded-full bg-navy-700" /> Historical
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-0.5 w-4 rounded-full border-t-2 border-dashed border-accent-600" /> Forecast
          </span>
        </div>
      </div>

      <div className="card-surface overflow-hidden rounded-2xl shadow-card">
        <div className="border-b border-slate-100 px-6 py-4 text-[0.7rem] font-bold uppercase tracking-wider text-slate-400 sm:px-7">
          Daily Predictions
        </div>
        <div className="divide-y divide-slate-100">
          {forecast.predictions.map((p) => (
            <div key={p.date} className="flex items-center justify-between px-6 py-3.5 sm:px-7">
              <span className="text-sm font-semibold text-navy-900">{p.label}</span>
              <span className="font-display text-sm font-bold tabular text-navy-950">
                {formatCurrency(p.amount)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {forecast.r2 != null && (
        <div className="rounded-2xl border border-slate-200 bg-surface-50 p-5 text-center text-xs font-medium text-slate-500 sm:text-sm">
          Model confidence: R² {forecast.r2.toFixed(2)} · RMSE {formatCurrency(forecast.rmse ?? 0)} — trained
          fresh on your uploaded transactions.
        </div>
      )}
    </div>
  );
}
