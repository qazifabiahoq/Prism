import json
from http.server import BaseHTTPRequestHandler
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

FOOD_KEYWORDS = ["restaurant", "cafe", "coffee", "food", "grocery", "starbucks", "mcdonald", "pizza", "chipotle", "whole foods", "trader joe"]
TRANSPORT_KEYWORDS = ["uber", "lyft", "gas", "fuel", "parking", "transit", "bus", "taxi", "shell"]
SHOPPING_KEYWORDS = ["amazon", "store", "shop", "retail", "clothing", "mall"]
BILLS_KEYWORDS = ["utility", "electric", "water", "rent", "mortgage", "insurance", "phone", "internet", "verizon", "con edison"]
ENTERTAINMENT_KEYWORDS = ["movie", "theater", "game", "spotify", "netflix", "concert", "gym", "amc", "equinox"]

CATEGORY_MAP = {"Food": 0, "Transportation": 1, "Shopping": 2, "Bills": 3, "Entertainment": 4, "Other": 5}


def find_column(columns, keywords):
    for col in columns:
        low = str(col).lower()
        if any(k in low for k in keywords):
            return col
    return None


def categorize(description):
    if description is None or (isinstance(description, float) and pd.isna(description)):
        return "Other"
    desc = str(description).lower()
    if any(k in desc for k in FOOD_KEYWORDS):
        return "Food"
    if any(k in desc for k in TRANSPORT_KEYWORDS):
        return "Transportation"
    if any(k in desc for k in SHOPPING_KEYWORDS):
        return "Shopping"
    if any(k in desc for k in BILLS_KEYWORDS):
        return "Bills"
    if any(k in desc for k in ENTERTAINMENT_KEYWORDS):
        return "Entertainment"
    return "Other"


def process_data(df):
    df = df.copy()

    date_col = find_column(df.columns, ["date", "time"])
    if date_col is not None:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(float)
    else:
        df["date"] = pd.NaT

    amount_col = find_column(df.columns, ["amount", "value", "total", "price"])
    if amount_col is not None:
        df["amount"] = pd.to_numeric(df[amount_col], errors="coerce").abs()
    else:
        df["amount"] = 0.0
    df["amount"] = df["amount"].fillna(0.0)

    df = df.dropna(subset=["date"]) if date_col is not None else df
    df = df.sort_values("date") if date_col is not None else df
    df = df.reset_index(drop=True)

    df["rolling_mean_7d"] = df["amount"].rolling(window=7, min_periods=1).mean()
    df["rolling_std_7d"] = df["amount"].rolling(window=7, min_periods=1).std().fillna(0)
    df["rolling_max_7d"] = df["amount"].rolling(window=7, min_periods=1).max()

    mean_amount = df["amount"].mean()
    std_amount = df["amount"].std()
    df["z_score"] = ((df["amount"] - mean_amount) / std_amount) if std_amount and std_amount > 0 else 0.0

    df["spending_velocity"] = df["amount"].diff().fillna(0)
    df["cumulative_spending"] = df["amount"].cumsum()

    desc_col = find_column(df.columns, ["description", "desc", "merchant", "memo"])
    if desc_col is not None:
        df["description"] = df[desc_col].astype(str)
        df["category"] = df[desc_col].apply(categorize)
    else:
        df["description"] = ""
        df["category"] = "Other"

    df["category_encoded"] = df["category"].map(CATEGORY_MAP).fillna(5)

    return df


def build_forecast_model(df):
    features = ["day_of_week", "day_of_month", "month", "is_weekend", "rolling_mean_7d", "category_encoded"]
    features = [f for f in features if f in df.columns]

    if len(df) < 20 or "date" not in df.columns or df["date"].isna().all():
        return None, None, None

    X = df[features].fillna(0)
    y = df["amount"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    importance = dict(zip(features, model.feature_importances_.tolist()))

    return model, {"r2": float(r2), "rmse": rmse, "importance": importance}, features


def detect_unusual_activity(df):
    features = ["amount", "rolling_mean_7d", "rolling_std_7d", "z_score"]
    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0)

    contamination = min(0.25, max(0.02, 10.0 / max(len(df), 1)))
    iso = IsolationForest(contamination=0.1 if len(df) >= 40 else contamination, random_state=42)
    df["is_anomaly"] = iso.fit_predict(X) == -1
    return df


def discover_patterns(df):
    features = ["amount", "day_of_week", "category_encoded"]
    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0)

    n_clusters = min(3, max(1, df["amount"].nunique()))
    if len(df) < n_clusters:
        df["spending_pattern"] = 0
        return df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["spending_pattern"] = kmeans.fit_predict(X_scaled)
    return df


def calculate_health_score(df):
    avg_spending = df["amount"].mean()
    std_spending = df["amount"].std()
    cv = (std_spending / avg_spending) * 100 if avg_spending and avg_spending > 0 else 0
    if pd.isna(cv):
        cv = 0

    anomaly_rate = (df["is_anomaly"].sum() / len(df) * 100) if "is_anomaly" in df.columns and len(df) else 0

    risk_score = min(100, (cv * 0.5) + (anomaly_rate * 5))
    wellness_score = max(0, 100 - risk_score)

    if wellness_score >= 70:
        category = "Excellent"
    elif wellness_score >= 50:
        category = "Good"
    elif wellness_score >= 30:
        category = "Fair"
    else:
        category = "Needs Attention"

    return {
        "score": round(float(wellness_score), 0),
        "category": category,
        "consistency": round(float(max(0, 100 - cv)), 1),
        "unusualRate": round(float(anomaly_rate), 1),
    }


PATTERN_LABELS = ["Small & Frequent", "Everyday Spending", "Major Transactions"]


def build_patterns(df):
    if "spending_pattern" not in df.columns:
        return []

    groups = []
    for cluster_id, group in df.groupby("spending_pattern"):
        top_cat = group["category"].mode()
        groups.append({
            "cluster": int(cluster_id),
            "avgAmount": float(group["amount"].mean()),
            "count": int(len(group)),
            "topCategory": str(top_cat.iloc[0]) if len(top_cat) else "Other",
        })

    groups.sort(key=lambda g: g["avgAmount"])
    for i, g in enumerate(groups):
        g["label"] = PATTERN_LABELS[min(i, len(PATTERN_LABELS) - 1)]

    return groups


def iso_date(ts):
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def build_forecast(df, model, metrics, features):
    if model is None or not features:
        return {
            "available": False,
            "r2": None,
            "rmse": None,
            "featureImportance": [],
            "predictions": [],
            "history": [],
            "weekTotal": 0,
            "dailyAverage": 0,
            "vsCurrentAveragePct": 0,
        }

    last_date = df["date"].max()
    rolling_mean_last = float(df["rolling_mean_7d"].iloc[-1]) if "rolling_mean_7d" in df.columns else float(df["amount"].mean())
    cat_mode = df["category_encoded"].mode()
    cat_mode_val = float(cat_mode.iloc[0]) if len(cat_mode) else 0.0

    predictions = []
    for i in range(1, 8):
        future_date = last_date + timedelta(days=i)
        row = {
            "day_of_week": future_date.dayofweek,
            "day_of_month": future_date.day,
            "month": future_date.month,
            "is_weekend": 1 if future_date.dayofweek >= 5 else 0,
            "rolling_mean_7d": rolling_mean_last,
            "category_encoded": cat_mode_val,
        }
        X_pred = pd.DataFrame([[row[f] for f in features]], columns=features)
        pred = float(model.predict(X_pred)[0])
        predictions.append({
            "date": iso_date(future_date),
            "label": future_date.strftime("%A, %b %d"),
            "amount": round(max(0.0, pred), 2),
        })

    history_df = df.tail(14)
    history = [
        {"date": iso_date(row["date"]), "amount": round(float(row["amount"]), 2)}
        for _, row in history_df.iterrows()
    ]

    week_total = sum(p["amount"] for p in predictions)
    daily_avg = week_total / 7
    current_avg = float(df["amount"].mean())
    vs_current = ((daily_avg - current_avg) / current_avg * 100) if current_avg > 0 else 0

    importance_sorted = sorted(metrics["importance"].items(), key=lambda kv: kv[1], reverse=True)

    return {
        "available": True,
        "r2": metrics["r2"],
        "rmse": metrics["rmse"],
        "featureImportance": [{"feature": k, "importance": round(float(v), 4)} for k, v in importance_sorted],
        "predictions": predictions,
        "history": history,
        "weekTotal": round(week_total, 2),
        "dailyAverage": round(daily_avg, 2),
        "vsCurrentAveragePct": round(vs_current, 1),
    }


def build_anomalies(df):
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = False

    anomalies_df = df[df["is_anomaly"] == True].sort_values("amount", ascending=False)  # noqa: E712
    count = int(len(anomalies_df))
    rate = round((count / len(df) * 100) if len(df) else 0, 1)

    items = []
    for _, row in anomalies_df.head(50).iterrows():
        items.append({
            "date": iso_date(row.get("date")),
            "amount": round(float(row["amount"]), 2),
            "description": str(row.get("description", ""))[:80],
            "category": str(row.get("category", "Other")),
            "zScore": round(float(row.get("z_score", 0)) if not pd.isna(row.get("z_score", 0)) else 0, 2),
        })

    scatter = []
    sample_df = df if len(df) <= 2000 else df.sample(2000, random_state=42)
    for _, row in sample_df.iterrows():
        scatter.append({
            "date": iso_date(row.get("date")),
            "amount": round(float(row["amount"]), 2),
            "isAnomaly": bool(row["is_anomaly"]),
        })

    return {"count": count, "rate": rate, "items": items, "scatter": scatter}


def build_category_breakdown(df):
    total = float(df["amount"].sum())
    grouped = df.groupby("category")["amount"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    out = []
    for cat, row in grouped.iterrows():
        amount = float(row["sum"])
        out.append({
            "category": str(cat),
            "amount": round(amount, 2),
            "count": int(row["count"]),
            "pct": round((amount / total * 100) if total > 0 else 0, 1),
        })
    return out


def run_analysis(raw_rows):
    if not raw_rows:
        return {"ok": False, "error": "No transaction rows were provided."}

    df = pd.DataFrame(raw_rows)
    df = process_data(df)

    if df["amount"].sum() == 0:
        return {"ok": False, "error": "We couldn't find a usable amount column in your file. Make sure it includes a column like Amount, Total, or Price."}

    model, metrics, features = build_forecast_model(df)
    df = detect_unusual_activity(df)
    df = discover_patterns(df)

    wellness = calculate_health_score(df)
    patterns = build_patterns(df)
    forecast = build_forecast(df, model, metrics, features)
    anomalies = build_anomalies(df)
    category_breakdown = build_category_breakdown(df)

    valid_dates = df["date"].dropna()
    date_range = {
        "start": iso_date(valid_dates.min()) if len(valid_dates) else None,
        "end": iso_date(valid_dates.max()) if len(valid_dates) else None,
    }

    transactions = []
    for _, row in df.tail(500).iterrows():
        transactions.append({
            "date": iso_date(row.get("date")),
            "amount": round(float(row["amount"]), 2),
            "description": str(row.get("description", ""))[:80],
            "category": str(row.get("category", "Other")),
            "isAnomaly": bool(row.get("is_anomaly", False)),
        })

    return {
        "ok": True,
        "summary": {
            "transactionCount": int(len(df)),
            "dateRange": date_range,
            "totalSpending": round(float(df["amount"].sum()), 2),
            "averageAmount": round(float(df["amount"].mean()), 2),
        },
        "wellness": wellness,
        "categoryBreakdown": category_breakdown,
        "patterns": [{"cluster": p["cluster"], "label": p["label"], "avgAmount": round(p["avgAmount"], 2), "count": p["count"], "topCategory": p["topCategory"]} for p in patterns],
        "forecast": forecast,
        "anomalies": anomalies,
        "transactions": transactions,
    }


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw_body or b"{}")
            rows = payload.get("rows", [])
            result = run_analysis(rows)
            self._send_json(200 if result.get("ok") else 400, result)
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"ok": False, "error": f"Analysis failed: {exc}"})
