export interface WellnessScore {
  score: number;
  category: "Excellent" | "Good" | "Fair" | "Needs Attention" | "Unknown";
  consistency: number;
  unusualRate: number;
}

export interface CategoryBreakdown {
  category: string;
  amount: number;
  count: number;
  pct: number;
}

export interface SpendingPattern {
  cluster: number;
  label: string;
  avgAmount: number;
  count: number;
  topCategory: string;
}

export interface ForecastPoint {
  date: string;
  label: string;
  amount: number;
}

export interface HistoryPoint {
  date: string;
  amount: number;
}

export interface ForecastResult {
  available: boolean;
  r2: number | null;
  rmse: number | null;
  featureImportance: Array<{ feature: string; importance: number }>;
  predictions: ForecastPoint[];
  history: HistoryPoint[];
  weekTotal: number;
  dailyAverage: number;
  vsCurrentAveragePct: number;
}

export interface AnomalyItem {
  date: string | null;
  amount: number;
  description: string;
  category: string;
  zScore: number;
}

export interface ScatterPoint {
  date: string | null;
  amount: number;
  isAnomaly: boolean;
}

export interface AnomalyResult {
  count: number;
  rate: number;
  items: AnomalyItem[];
  scatter: ScatterPoint[];
}

export interface TransactionRow {
  date: string | null;
  amount: number;
  description: string;
  category: string;
  isAnomaly: boolean;
}

export interface AnalysisSummary {
  transactionCount: number;
  dateRange: { start: string | null; end: string | null };
  totalSpending: number;
  averageAmount: number;
}

export interface AnalysisResult {
  ok: true;
  summary: AnalysisSummary;
  wellness: WellnessScore;
  categoryBreakdown: CategoryBreakdown[];
  patterns: SpendingPattern[];
  forecast: ForecastResult | null;
  anomalies: AnomalyResult;
  transactions: TransactionRow[];
}

export interface AnalysisError {
  ok: false;
  error: string;
}

export type AnalysisResponse = AnalysisResult | AnalysisError;

export interface AssistantResponse {
  ok: boolean;
  answer?: string;
  error?: string;
}

export interface ReceiptResponse {
  ok: boolean;
  rows?: Array<{ Date: string | null; Amount: number; Description: string }>;
  error?: string;
}

export type UploadMethod = "csv" | "photo";
export type TabKey = "upload" | "dashboard" | "forecast" | "alerts" | "assistant";
