"use client";

import { useCallback, useRef, useState } from "react";
import Papa from "papaparse";
import {
  CheckCircle2,
  FileSpreadsheet,
  ImageIcon,
  Loader2,
  RefreshCcw,
  ScanEye,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import { generateSampleTransactions } from "@/lib/sample-data";
import { extractReceipt } from "@/lib/api";
import { formatCurrency, formatNumber } from "@/lib/format";

type Row = Record<string, unknown>;

function findAmountColumn(rows: Row[]): string | null {
  if (!rows.length) return null;
  const cols = Object.keys(rows[0]);
  return (
    cols.find((c) => ["amount", "value", "total", "price"].some((k) => c.toLowerCase().includes(k))) ??
    null
  );
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result.split(",")[1] ?? "");
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export function UploadPanel({
  rows,
  fileName,
  onRowsParsed,
  onAnalyze,
  analyzing,
  analysisComplete,
  onReset,
  error,
}: {
  rows: Row[] | null;
  fileName: string | null;
  onRowsParsed: (rows: Row[], fileName: string) => void;
  onAnalyze: () => void;
  analyzing: boolean;
  analysisComplete: boolean;
  onReset: () => void;
  error: string | null;
}) {
  const [method, setMethod] = useState<"csv" | "photo">("csv");
  const [dragActive, setDragActive] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const photoInputRef = useRef<HTMLInputElement>(null);

  const parseFile = useCallback(
    (file: File) => {
      setParseError(null);
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          const data = (results.data as Row[]).filter((r) =>
            Object.values(r).some((v) => v !== null && v !== undefined && v !== "")
          );
          if (!data.length) {
            setParseError("We couldn't find any rows in that file. Please check the CSV and try again.");
            return;
          }
          onRowsParsed(data, file.name);
        },
        error: (err) => setParseError(err.message),
      });
    },
    [onRowsParsed]
  );

  const scanPhoto = useCallback(
    async (file: File) => {
      setParseError(null);
      setScanning(true);
      try {
        const base64 = await fileToBase64(file);
        const res = await extractReceipt(base64, file.type || "image/jpeg");
        if (res.ok && res.rows) {
          onRowsParsed(res.rows, file.name);
        } else {
          setParseError(res.error ?? "Could not read that photo. Please try a clearer image or use CSV upload.");
        }
      } catch {
        setParseError("Could not read that photo. Please try a clearer image or use CSV upload.");
      } finally {
        setScanning(false);
      }
    },
    [onRowsParsed]
  );

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    if (method === "csv") parseFile(file);
    else scanPhoto(file);
  };

  const handleTryDemo = () => {
    const demo = generateSampleTransactions(120);
    onRowsParsed(demo, "sample_transactions.csv");
  };

  const amountCol = rows ? findAmountColumn(rows) : null;
  const total = amountCol && rows ? rows.reduce((s, r) => s + (Number(r[amountCol]) || 0), 0) : 0;
  const avg = rows && rows.length ? total / rows.length : 0;

  return (
    <div className="mx-auto max-w-3xl">
      <div className="text-center">
        <h2 className="font-display text-2xl font-extrabold tracking-tight text-ink-950 sm:text-3xl">
          Upload Your Transactions
        </h2>
        <p className="mt-2 text-sm text-slate-500 sm:text-base">
          Get instant, ML-powered insights from your spending data.
        </p>
      </div>

      <div className="mt-7 flex justify-center gap-2 rounded-full bg-surface-100 p-1 sm:mx-auto sm:w-fit">
        <button
          onClick={() => setMethod("csv")}
          className={`focus-ring flex flex-1 items-center justify-center gap-1.5 rounded-full px-4 py-2 text-sm font-semibold transition sm:flex-none ${
            method === "csv" ? "bg-white text-ink-950 shadow-sm" : "text-slate-500"
          }`}
        >
          <FileSpreadsheet size={15} /> CSV File
        </button>
        <button
          onClick={() => setMethod("photo")}
          className={`focus-ring flex flex-1 items-center justify-center gap-1.5 rounded-full px-4 py-2 text-sm font-semibold transition sm:flex-none ${
            method === "photo" ? "bg-white text-ink-950 shadow-sm" : "text-slate-500"
          }`}
        >
          <ImageIcon size={15} /> Photo Upload
        </button>
      </div>

      <div className="mt-6">
        {method === "csv" ? (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            className={`group cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition sm:p-14 ${
              dragActive
                ? "border-accent-500 bg-accent-50"
                : "border-slate-300 bg-white hover:border-accent-400 hover:bg-surface-50"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) parseFile(file);
                e.target.value = "";
              }}
            />
            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-ink-950 text-accent-400 transition group-hover:scale-105">
              <UploadCloud size={26} />
            </div>
            <p className="mt-4 text-base font-semibold text-ink-950">
              Drop your CSV here, or click to browse
            </p>
            <p className="mt-1.5 text-sm text-slate-500">
              Needs Date, Amount, and Description columns
            </p>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                handleTryDemo();
              }}
              className="focus-ring mt-5 inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-bold text-ink-700 shadow-sm transition hover:border-accent-400"
            >
              <Sparkles size={13} /> Or try instantly with sample data
            </button>
          </div>
        ) : (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            onClick={() => !scanning && photoInputRef.current?.click()}
            className={`group cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition sm:p-14 ${
              dragActive
                ? "border-accent-500 bg-accent-50"
                : "border-slate-300 bg-white hover:border-accent-400 hover:bg-surface-50"
            }`}
          >
            <input
              ref={photoInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) scanPhoto(file);
                e.target.value = "";
              }}
            />
            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-ink-950 text-accent-400 transition group-hover:scale-105">
              {scanning ? <Loader2 size={26} className="animate-spin" /> : <ScanEye size={26} />}
            </div>
            <p className="mt-4 text-base font-semibold text-ink-950">
              {scanning ? "Reading your receipt…" : "Drop a receipt or statement photo here"}
            </p>
            <p className="mx-auto mt-1.5 max-w-sm text-sm text-slate-500">
              {scanning
                ? "This can take a few seconds."
                : "Our AI reads the image and pulls out each transaction automatically."}
            </p>
          </div>
        )}
      </div>

      {(parseError || error) && (
        <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-medium text-rose-700">
          {parseError || error}
        </div>
      )}

      {rows && rows.length > 0 && (
        <div className="animate-fade-up mt-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle2 size={17} className="text-emerald-500" />
              <span className="text-sm font-semibold text-ink-950">{fileName}</span>
            </div>
            <span className="text-xs font-medium text-slate-400">{formatNumber(rows.length)} rows</span>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MiniStat label="Transactions" value={formatNumber(rows.length)} />
            <MiniStat label="Data Fields" value={formatNumber(Object.keys(rows[0] ?? {}).length)} />
            {amountCol && <MiniStat label="Total Spending" value={formatCurrency(total)} />}
            {amountCol && <MiniStat label="Average Amount" value={formatCurrency(avg)} />}
          </div>

          <div className="mt-4 overflow-x-auto rounded-xl border border-slate-200">
            <table className="w-full min-w-[420px] text-left text-sm">
              <thead className="bg-surface-50 text-xs font-bold uppercase tracking-wide text-slate-500">
                <tr>
                  {Object.keys(rows[0]).slice(0, 5).map((c) => (
                    <th key={c} className="px-4 py-2.5">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 5).map((r, i) => (
                  <tr key={i} className="border-t border-slate-100">
                    {Object.keys(rows[0]).slice(0, 5).map((c) => (
                      <td key={c} className="px-4 py-2.5 text-slate-600">
                        {String(r[c] ?? "")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-6">
            {analysisComplete ? (
              <div className="flex flex-col items-center gap-3 rounded-xl bg-emerald-50 px-5 py-4 text-center sm:flex-row sm:justify-between sm:text-left">
                <span className="text-sm font-semibold text-emerald-700">
                  Analysis complete. Explore the Dashboard, Forecast, Alerts, and Assistant tabs.
                </span>
                <button
                  onClick={onReset}
                  className="focus-ring inline-flex items-center gap-1.5 whitespace-nowrap rounded-full bg-white px-4 py-2 text-sm font-bold text-emerald-700 shadow-sm"
                >
                  <RefreshCcw size={14} /> Upload New Data
                </button>
              </div>
            ) : (
              <button
                onClick={onAnalyze}
                disabled={analyzing}
                className="focus-ring flex w-full items-center justify-center gap-2 rounded-full bg-gradient-to-r from-accent-500 to-accent-600 px-6 py-4 text-base font-bold text-white shadow-card transition hover:brightness-110 disabled:opacity-70"
              >
                {analyzing ? (
                  <>
                    <Loader2 size={18} className="animate-spin" /> Analyzing your financial data…
                  </>
                ) : (
                  <>Analyze My Spending</>
                )}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="card-surface rounded-xl p-4 text-center shadow-card">
      <div className="text-[0.65rem] font-bold uppercase tracking-wider text-slate-400">{label}</div>
      <div className="mt-1.5 font-display text-lg font-extrabold tabular text-ink-950 sm:text-xl">
        {value}
      </div>
    </div>
  );
}
