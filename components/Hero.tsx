"use client";

import { ArrowRight, ShieldCheck, Sparkles, TrendingUp, Zap } from "lucide-react";

export function Hero({
  onTryDemo,
  onUpload,
}: {
  onTryDemo: () => void;
  onUpload: () => void;
}) {
  return (
    <section className="relative overflow-hidden bg-gradient-to-b from-[#040A17] via-navy-900 to-navy-800 pb-24 pt-14 sm:pb-32 sm:pt-20">
      {/* Background texture */}
      <div className="pointer-events-none absolute inset-0 bg-grid-pattern bg-[size:44px_44px] opacity-40 [mask-image:radial-gradient(ellipse_70%_60%_at_50%_0%,black_10%,transparent_75%)]" />
      <div className="pointer-events-none absolute left-1/2 top-[-10%] h-[560px] w-[560px] -translate-x-1/2 rounded-full bg-radial-fade blur-2xl" />
      <div className="pointer-events-none absolute -right-40 top-1/3 h-[420px] w-[420px] rounded-full bg-gold-500/10 blur-[100px]" />
      <div className="noise-overlay pointer-events-none absolute inset-0 opacity-[0.4]" />

      <div className="container-shell relative">
        <div className="mx-auto flex max-w-3xl flex-col items-center text-center">
          <div className="animate-fade-in mb-7 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-xs font-semibold uppercase tracking-wider text-accent-400">
            <Sparkles size={13} />
            Real ML. Real predictions. Zero fluff.
          </div>

          <h1 className="animate-fade-up font-display text-[2.6rem] font-extrabold leading-[1.06] tracking-tight text-white sm:text-6xl lg:text-[4.2rem]">
            Your finances,
            <br />
            <span className="text-gradient">seen with clarity.</span>
          </h1>

          <p
            className="animate-fade-up mt-6 max-w-xl text-balance text-base leading-relaxed text-white/60 sm:text-lg"
            style={{ animationDelay: "0.1s" }}
          >
            Upload transactions and get institutional-grade forecasting, fraud
            detection, and an AI advisor trained on your own spending —
            in under 60 seconds, completely private.
          </p>

          <div
            className="animate-fade-up mt-9 flex w-full flex-col items-center gap-3 sm:w-auto sm:flex-row"
            style={{ animationDelay: "0.2s" }}
          >
            <button
              onClick={onUpload}
              className="focus-ring group flex w-full items-center justify-center gap-2 rounded-full bg-gradient-to-r from-accent-400 to-accent-600 px-7 py-3.5 text-[0.95rem] font-bold text-navy-950 shadow-glow transition hover:brightness-110 active:scale-[0.98] sm:w-auto"
            >
              Upload Your Transactions
              <ArrowRight size={17} className="transition group-hover:translate-x-0.5" />
            </button>
            <button
              onClick={onTryDemo}
              className="focus-ring w-full rounded-full border border-white/15 bg-white/[0.04] px-7 py-3.5 text-[0.95rem] font-semibold text-white/90 backdrop-blur-sm transition hover:bg-white/10 sm:w-auto"
            >
              Explore With Sample Data
            </button>
          </div>

          <div
            className="animate-fade-up mt-10 flex flex-wrap items-center justify-center gap-x-7 gap-y-3 text-xs font-medium text-white/45 sm:text-sm"
            style={{ animationDelay: "0.3s" }}
          >
            <span className="inline-flex items-center gap-1.5">
              <ShieldCheck size={15} className="text-accent-400" /> No bank linking required
            </span>
            <span className="inline-flex items-center gap-1.5">
              <Zap size={15} className="text-accent-400" /> Results in ~30 seconds
            </span>
            <span className="inline-flex items-center gap-1.5">
              <TrendingUp size={15} className="text-accent-400" /> 87% forecast accuracy
            </span>
          </div>
        </div>

        {/* Floating product preview */}
        <div
          className="animate-fade-up relative mx-auto mt-16 max-w-4xl sm:mt-20"
          style={{ animationDelay: "0.4s" }}
        >
          <div className="glass-dark relative overflow-hidden rounded-2xl border border-white/10 p-4 shadow-[0_30px_80px_-20px_rgba(0,0,0,0.6)] sm:rounded-3xl sm:p-6">
            <div className="flex items-center gap-1.5 pb-4">
              <span className="h-2.5 w-2.5 rounded-full bg-white/20" />
              <span className="h-2.5 w-2.5 rounded-full bg-white/20" />
              <span className="h-2.5 w-2.5 rounded-full bg-white/20" />
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                { label: "Wellness Score", value: "78", accent: "text-accent-400" },
                { label: "7-Day Forecast", value: "$412", accent: "text-white" },
                { label: "Anomalies Found", value: "3", accent: "text-rose-300" },
                { label: "Forecast R²", value: "0.89", accent: "text-gold-400" },
              ].map((m) => (
                <div
                  key={m.label}
                  className="rounded-xl border border-white/10 bg-white/[0.03] p-4 text-left"
                >
                  <div className="text-[0.68rem] font-semibold uppercase tracking-wider text-white/40">
                    {m.label}
                  </div>
                  <div className={`mt-2 font-display text-2xl font-bold tabular ${m.accent}`}>
                    {m.value}
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 h-28 w-full overflow-hidden rounded-xl border border-white/10 bg-white/[0.03] sm:h-36">
              <svg viewBox="0 0 400 100" className="h-full w-full" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="heroChart" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#5EEAD4" stopOpacity="0.35" />
                    <stop offset="100%" stopColor="#5EEAD4" stopOpacity="0" />
                  </linearGradient>
                </defs>
                <path
                  d="M0,70 C40,60 60,30 100,40 C140,50 160,20 200,30 C240,40 260,15 300,25 C340,35 360,10 400,20 L400,100 L0,100 Z"
                  fill="url(#heroChart)"
                />
                <path
                  d="M0,70 C40,60 60,30 100,40 C140,50 160,20 200,30 C240,40 260,15 300,25 C340,35 360,10 400,20"
                  fill="none"
                  stroke="#5EEAD4"
                  strokeWidth="2"
                />
              </svg>
            </div>
          </div>
          <div className="pointer-events-none absolute -inset-x-10 -bottom-10 h-24 bg-gradient-to-t from-navy-900 to-transparent sm:hidden" />
        </div>
      </div>
    </section>
  );
}
