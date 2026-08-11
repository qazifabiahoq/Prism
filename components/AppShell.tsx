"use client";

import { useEffect, useRef, useState } from "react";
import { Nav } from "./Nav";
import { Hero } from "./Hero";
import { Features } from "./Features";
import { Footer } from "./Footer";
import { UploadPanel } from "./UploadPanel";
import { Tabs } from "./Tabs";
import { Dashboard } from "./Dashboard";
import { Forecast } from "./Forecast";
import { Alerts } from "./Alerts";
import { Assistant } from "./Assistant";
import { analyzeTransactions } from "@/lib/api";
import { generateSampleTransactions } from "@/lib/sample-data";
import type { AnalysisResult, TabKey } from "@/lib/types";

type Row = Record<string, unknown>;

export function AppShell() {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabKey>("upload");
  const [showApp, setShowApp] = useState(false);

  const heroRef = useRef<HTMLDivElement>(null);
  const appRef = useRef<HTMLDivElement>(null);

  const scrollToApp = () => {
    setShowApp(true);
    requestAnimationFrame(() => {
      appRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  const handleNav = (key: TabKey | "top") => {
    if (key === "top") {
      setShowApp(false);
      requestAnimationFrame(() => window.scrollTo({ top: 0, behavior: "smooth" }));
      return;
    }
    setActiveTab(key);
    scrollToApp();
  };

  const handleRowsParsed = (parsed: Row[], name: string) => {
    setRows(parsed);
    setFileName(name);
    setResult(null);
    setError(null);
  };

  const handleTryDemo = () => {
    handleRowsParsed(generateSampleTransactions(120), "sample_transactions.csv");
    scrollToApp();
  };

  const handleAnalyze = async () => {
    if (!rows) return;
    setAnalyzing(true);
    setError(null);
    const res = await analyzeTransactions(rows);
    setAnalyzing(false);

    if (res.ok) {
      setResult(res);
      setActiveTab("dashboard");
    } else {
      setError(res.error);
    }
  };

  const handleReset = () => {
    setRows(null);
    setFileName(null);
    setResult(null);
    setError(null);
    setActiveTab("upload");
  };

  useEffect(() => {
    if (showApp) {
      requestAnimationFrame(() => appRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    }
  }, [showApp]);

  return (
    <div ref={heroRef}>
      <Nav hasAnalysis={!!result} onNavigate={handleNav} onTryDemo={handleTryDemo} />
      {!showApp && (
        <>
          <Hero onTryDemo={handleTryDemo} onUpload={scrollToApp} />
          <Features />
          <Footer />
        </>
      )}

      {showApp && (
        <div ref={appRef} className="min-h-screen bg-surface-50">
          <Tabs active={activeTab} onChange={setActiveTab} alertCount={result?.anomalies.count} />
          <div className="container-shell py-10 sm:py-14">
            {activeTab === "upload" && (
              <UploadPanel
                rows={rows}
                fileName={fileName}
                onRowsParsed={handleRowsParsed}
                onAnalyze={handleAnalyze}
                analyzing={analyzing}
                analysisComplete={!!result}
                onReset={handleReset}
                error={error}
              />
            )}

            {activeTab === "dashboard" &&
              (result ? (
                <Dashboard result={result} />
              ) : (
                <EmptyState onGoUpload={() => setActiveTab("upload")} message="Upload your transactions and analyze to view your dashboard." />
              ))}

            {activeTab === "forecast" &&
              (result ? (
                <Forecast result={result} />
              ) : (
                <EmptyState onGoUpload={() => setActiveTab("upload")} message="Upload transactions and analyze to see your forecast." />
              ))}

            {activeTab === "alerts" &&
              (result ? (
                <Alerts result={result} />
              ) : (
                <EmptyState onGoUpload={() => setActiveTab("upload")} message="Upload transactions and analyze to see alerts." />
              ))}

            {activeTab === "assistant" &&
              (result ? (
                <Assistant result={result} />
              ) : (
                <EmptyState onGoUpload={() => setActiveTab("upload")} message="Analyze your transactions first to get personalized advice." />
              ))}
          </div>
          <Footer />
        </div>
      )}
    </div>
  );
}

function EmptyState({ message, onGoUpload }: { message: string; onGoUpload: () => void }) {
  return (
    <div className="mx-auto flex max-w-md flex-col items-center rounded-2xl border border-slate-200 bg-white px-8 py-16 text-center shadow-card">
      <p className="text-sm font-medium text-slate-500">{message}</p>
      <button
        onClick={onGoUpload}
        className="focus-ring mt-5 rounded-full bg-ink-950 px-5 py-2.5 text-sm font-bold text-white transition hover:bg-ink-800"
      >
        Go to Upload
      </button>
    </div>
  );
}
