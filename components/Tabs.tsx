"use client";

import { AlertTriangle, LayoutDashboard, MessageSquareText, TrendingUp, Upload } from "lucide-react";
import type { TabKey } from "@/lib/types";

const TABS: Array<{ key: TabKey; label: string; icon: typeof Upload }> = [
  { key: "upload", label: "Upload", icon: Upload },
  { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { key: "forecast", label: "Forecast", icon: TrendingUp },
  { key: "alerts", label: "Alerts", icon: AlertTriangle },
  { key: "assistant", label: "Assistant", icon: MessageSquareText },
];

export function Tabs({
  active,
  onChange,
  alertCount,
}: {
  active: TabKey;
  onChange: (key: TabKey) => void;
  alertCount?: number;
}) {
  return (
    <div className="sticky top-16 z-40 border-b border-slate-200 bg-white/85 backdrop-blur-md sm:top-20">
      <div className="container-shell">
        <div className="scrollbar-none flex gap-1 overflow-x-auto py-2">
          {TABS.map((tab) => {
            const isActive = active === tab.key;
            return (
              <button
                key={tab.key}
                onClick={() => onChange(tab.key)}
                className={`focus-ring relative flex shrink-0 items-center gap-2 rounded-full px-4 py-2.5 text-sm font-semibold transition ${
                  isActive
                    ? "bg-ink-950 text-white shadow-md"
                    : "text-slate-500 hover:bg-surface-100 hover:text-ink-900"
                }`}
              >
                <tab.icon size={15} />
                {tab.label}
                {tab.key === "alerts" && !!alertCount && alertCount > 0 && (
                  <span
                    className={`ml-0.5 inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded-full px-1 text-[0.65rem] font-bold ${
                      isActive ? "bg-rose-400 text-ink-950" : "bg-rose-100 text-rose-600"
                    }`}
                  >
                    {alertCount}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
