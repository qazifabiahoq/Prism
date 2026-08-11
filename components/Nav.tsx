"use client";

import { useEffect, useState } from "react";
import { Logo } from "./Logo";
import type { TabKey } from "@/lib/types";

const LINKS: Array<{ key: TabKey | "top"; label: string }> = [
  { key: "top", label: "Overview" },
  { key: "dashboard", label: "Dashboard" },
  { key: "forecast", label: "Forecast" },
  { key: "alerts", label: "Alerts" },
  { key: "assistant", label: "Assistant" },
];

export function Nav({
  hasAnalysis,
  onNavigate,
}: {
  hasAnalysis: boolean;
  onNavigate: (key: TabKey | "top") => void;
}) {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const handleNav = (key: TabKey | "top") => {
    setMenuOpen(false);
    onNavigate(key);
  };

  return (
    <header
      className={`sticky top-0 z-50 transition-all duration-300 ${
        scrolled
          ? "glass-dark border-b border-white/10 shadow-[0_4px_30px_rgba(0,0,0,0.25)]"
          : "bg-transparent"
      }`}
    >
      <div className="container-shell flex h-16 items-center justify-between sm:h-20">
        <button onClick={() => handleNav("top")} className="focus-ring rounded-lg">
          <Logo light />
        </button>

        <nav className="hidden items-center gap-1 md:flex">
          {LINKS.filter((l) => l.key === "top" || hasAnalysis).map((link) => (
            <button
              key={link.key}
              onClick={() => handleNav(link.key)}
              className="focus-ring rounded-full px-4 py-2 text-sm font-medium text-white/70 transition hover:bg-white/10 hover:text-white"
            >
              {link.label}
            </button>
          ))}
        </nav>

        <div className="hidden items-center gap-3 md:flex">
          <button
            onClick={() => handleNav("upload" as TabKey)}
            className="focus-ring rounded-full bg-gradient-to-r from-accent-500 to-accent-600 px-5 py-2.5 text-sm font-bold text-navy-950 shadow-glow transition hover:brightness-110 active:scale-[0.98]"
          >
            {hasAnalysis ? "My Data" : "Try Free Demo"}
          </button>
        </div>

        <button
          className="focus-ring flex h-10 w-10 items-center justify-center rounded-lg text-white md:hidden"
          onClick={() => setMenuOpen((v) => !v)}
          aria-label="Toggle menu"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            {menuOpen ? (
              <path d="M6 6l12 12M18 6L6 18" strokeLinecap="round" />
            ) : (
              <path d="M3 6h18M3 12h18M3 18h18" strokeLinecap="round" />
            )}
          </svg>
        </button>
      </div>

      {menuOpen && (
        <div className="glass-dark border-t border-white/10 px-5 pb-6 pt-2 md:hidden">
          <div className="flex flex-col gap-1">
            {LINKS.filter((l) => l.key === "top" || hasAnalysis).map((link) => (
              <button
                key={link.key}
                onClick={() => handleNav(link.key)}
                className="focus-ring rounded-lg px-3 py-3 text-left text-base font-medium text-white/80 hover:bg-white/10 hover:text-white"
              >
                {link.label}
              </button>
            ))}
            <button
              onClick={() => handleNav("upload" as TabKey)}
              className="focus-ring mt-2 rounded-full bg-gradient-to-r from-accent-500 to-accent-600 px-5 py-3 text-center text-sm font-bold text-navy-950"
            >
              {hasAnalysis ? "My Data" : "Try Free Demo"}
            </button>
          </div>
        </div>
      )}
    </header>
  );
}
