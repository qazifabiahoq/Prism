import { AlertTriangle, Brain, LineChart, Lock, MessageSquareText, Sparkles } from "lucide-react";

const FEATURES = [
  {
    icon: LineChart,
    title: "Predictive Forecasting",
    desc: "RandomForest ensemble learning projects your next 7 days of spending with 85–90% R² accuracy.",
  },
  {
    icon: AlertTriangle,
    title: "Fraud Detection",
    desc: "Isolation Forest flags unusual transactions in real time, no labeled fraud data required.",
  },
  {
    icon: Brain,
    title: "Behavioral Clustering",
    desc: "K-Means uncovers the spending personas hiding in your data: necessities, social, and major expenses.",
  },
  {
    icon: Sparkles,
    title: "Wellness Scoring",
    desc: "A single 0–100 score combining volatility and anomaly rate, so you can track improvement over time.",
  },
  {
    icon: MessageSquareText,
    title: "AI Financial Advisor",
    desc: "Llama 3.3 70B answers your questions in plain English, grounded in your actual transaction history.",
  },
  {
    icon: Lock,
    title: "Privacy by Design",
    desc: "No bank linking, no account required, zero data retention. Your CSV never leaves the analysis session.",
  },
];

export function Features() {
  return (
    <section className="relative bg-surface-50 py-20 sm:py-28">
      <div className="container-shell">
        <div className="mx-auto max-w-2xl text-center">
          <span className="text-xs font-bold uppercase tracking-[0.2em] text-ink-600">
            Capabilities
          </span>
          <h2 className="mt-3 font-display text-3xl font-extrabold tracking-tight text-ink-950 sm:text-4xl">
            Data-scientist-grade analysis,
            <br className="hidden sm:block" /> zero spreadsheets required
          </h2>
          <p className="mt-4 text-base leading-relaxed text-slate-500">
            Every metric in Prism comes from a trained statistical model on
            your own data, not a generic chatbot guessing at advice.
          </p>
        </div>

        <div className="mt-14 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f, i) => (
            <div
              key={f.title}
              className="group card-surface animate-fade-up relative rounded-2xl p-6 shadow-card transition duration-300 hover:-translate-y-1 hover:shadow-card-hover"
              style={{ animationDelay: `${i * 0.06}s` }}
            >
              <div className="inline-flex h-11 w-11 items-center justify-center rounded-xl bg-ink-950 text-accent-400 transition group-hover:bg-accent-500 group-hover:text-white">
                <f.icon size={20} strokeWidth={2} />
              </div>
              <h3 className="mt-4 font-display text-base font-bold text-ink-950">
                {f.title}
              </h3>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-500">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
