import type { LucideIcon } from "lucide-react";

export function StatCard({
  label,
  value,
  description,
  icon: Icon,
  tone = "neutral",
}: {
  label: string;
  value: string;
  description?: string;
  icon?: LucideIcon;
  tone?: "neutral" | "positive" | "negative" | "accent";
}) {
  const toneClass = {
    neutral: "text-navy-950",
    positive: "text-emerald-600",
    negative: "text-rose-600",
    accent: "text-navy-700",
  }[tone];

  return (
    <div className="card-surface group relative overflow-hidden rounded-2xl p-5 shadow-card transition hover:-translate-y-0.5 hover:shadow-card-hover sm:p-6">
      <div className="flex items-start justify-between">
        <div className="text-[0.7rem] font-bold uppercase tracking-wider text-slate-400">
          {label}
        </div>
        {Icon && (
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-surface-100 text-navy-700">
            <Icon size={16} />
          </div>
        )}
      </div>
      <div className={`mt-3 font-display text-3xl font-extrabold tabular ${toneClass}`}>
        {value}
      </div>
      {description && (
        <div className="mt-1.5 text-sm font-medium text-slate-500">{description}</div>
      )}
    </div>
  );
}
