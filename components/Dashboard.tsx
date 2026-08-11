"use client";

import { AlertTriangle, Gauge, PieChart as PieIcon, Users } from "lucide-react";
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import type { AnalysisResult } from "@/lib/types";
import { formatCurrency, formatNumber } from "@/lib/format";
import { StatCard } from "./StatCard";

const CATEGORY_COLORS = ["#4143C7", "#7C7FF2", "#10B981", "#F59E0B", "#FB7185", "#94A3B8"];

const WELLNESS_TONE: Record<string, string> = {
  Excellent: "#10B981",
  Good: "#5457E5",
  Fair: "#F59E0B",
  "Needs Attention": "#E11D48",
  Unknown: "#94A3B8",
};

export function Dashboard({ result }: { result: AnalysisResult }) {
  const { wellness, categoryBreakdown, patterns, anomalies } = result;
  const scoreColor = WELLNESS_TONE[wellness.category] ?? WELLNESS_TONE.Unknown;
  const circumference = 2 * Math.PI * 54;
  const dash = (wellness.score / 100) * circumference;

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <div>
        <h2 className="font-display text-2xl font-extrabold tracking-tight text-ink-950 sm:text-3xl">
          Financial Dashboard
        </h2>
        <p className="mt-1.5 text-sm text-slate-500 sm:text-base">
          Your complete spending overview, generated from trained models on your data.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="card-surface rounded-2xl p-6 shadow-card sm:p-7">
          <div className="flex items-center gap-2 text-[0.7rem] font-bold uppercase tracking-wider text-slate-400">
            <Gauge size={14} /> Financial Wellness
          </div>
          <div className="mt-4 flex items-center gap-6">
            <svg width="120" height="120" viewBox="0 0 120 120" className="shrink-0 -rotate-90">
              <circle cx="60" cy="60" r="54" fill="none" stroke="#F4F5F9" strokeWidth="10" />
              <circle
                cx="60"
                cy="60"
                r="54"
                fill="none"
                stroke={scoreColor}
                strokeWidth="10"
                strokeLinecap="round"
                strokeDasharray={`${dash} ${circumference}`}
                className="transition-all duration-1000 ease-out"
              />
            </svg>
            <div>
              <div className="font-display text-4xl font-extrabold tabular text-ink-950">
                {wellness.score.toFixed(0)}
                <span className="text-lg font-semibold text-slate-400">/100</span>
              </div>
              <div
                className="mt-1 inline-block rounded-full px-2.5 py-0.5 text-xs font-bold"
                style={{ backgroundColor: `${scoreColor}1A`, color: scoreColor }}
              >
                {wellness.category}
              </div>
              <div className="mt-2 text-xs text-slate-500">
                Consistency {wellness.consistency.toFixed(0)}% · Unusual rate{" "}
                {wellness.unusualRate.toFixed(1)}%
              </div>
            </div>
          </div>
        </div>

        <StatCard
          label="Activity Alerts"
          value={formatNumber(anomalies.count)}
          description="Unusual transactions flagged"
          icon={AlertTriangle}
          tone={anomalies.count > 5 ? "negative" : "positive"}
        />
      </div>

      <div className="card-surface rounded-2xl p-6 shadow-card sm:p-7">
        <div className="mb-4 flex items-center gap-2 text-[0.7rem] font-bold uppercase tracking-wider text-slate-400">
          <PieIcon size={14} /> Where Your Money Goes
        </div>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="h-72 sm:h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={categoryBreakdown}
                  dataKey="amount"
                  nameKey="category"
                  innerRadius="55%"
                  outerRadius="85%"
                  paddingAngle={2}
                >
                  {categoryBreakdown.map((entry, i) => (
                    <Cell key={entry.category} fill={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value: number, name: string) => [formatCurrency(value), name]}
                  contentStyle={{ borderRadius: 12, border: "1px solid #ECEEF4", fontSize: 13 }}
                />
                <Legend
                  layout="horizontal"
                  verticalAlign="bottom"
                  formatter={(value) => <span className="text-xs text-slate-600">{value}</span>}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-col justify-center gap-2.5">
            {categoryBreakdown.map((c, i) => (
              <div key={c.category} className="flex items-center justify-between rounded-lg bg-surface-50 px-4 py-2.5">
                <div className="flex items-center gap-2.5">
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: CATEGORY_COLORS[i % CATEGORY_COLORS.length] }}
                  />
                  <span className="text-sm font-semibold text-ink-950">{c.category}</span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-bold tabular text-ink-950">
                    {formatCurrency(c.amount)}
                  </div>
                  <div className="text-[0.7rem] text-slate-400">{c.pct.toFixed(1)}%</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {patterns.length > 0 && (
        <div className="card-surface rounded-2xl p-6 shadow-card sm:p-7">
          <div className="mb-4 flex items-center gap-2 text-[0.7rem] font-bold uppercase tracking-wider text-slate-400">
            <Users size={14} /> Your Spending Patterns
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {patterns.map((p) => (
              <div key={p.cluster} className="rounded-xl border border-slate-200 bg-surface-50 p-5">
                <div className="text-xs font-bold uppercase tracking-wide text-ink-600">
                  {p.label}
                </div>
                <div className="mt-2 font-display text-2xl font-extrabold tabular text-ink-950">
                  {formatCurrency(p.avgAmount)}
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  avg per transaction · {p.count} transactions
                </div>
                <div className="mt-2 inline-block rounded-full bg-white px-2.5 py-0.5 text-[0.7rem] font-semibold text-slate-500 shadow-sm">
                  Mostly {p.topCategory}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
