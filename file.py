import io
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 避免負號變方塊

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, request, send_from_directory


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class MomentumConfig:
    db_name: str = "taiwan_momentum_pr_backtest.db"
    start_date: str = "2025-01-01"
    end_date: str = "2026-04-30"
    quarter_window: int = 63
    pr_threshold: float = 80.0
    yfinance_sleep: float = 0.15
    max_price_gap_days: int = 14
    min_price_coverage_ratio: float = 0.60
    universe_lookback_days: int = 10


def script_path(filename: str) -> str:
    if os.path.isabs(filename):
        return filename
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_date(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def clean_number(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace(",", "").replace("%", "").strip()
    if text in {"", "--", "---", "nan", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clean_int(value) -> Optional[int]:
    number = clean_number(value)
    if number is None:
        return None
    return int(number)


def to_twse_date(date_str: str) -> str:
    return date_str.replace("-", "")


def make_ticker(code_or_ticker: str) -> tuple[str, str]:
    text = str(code_or_ticker).strip().upper()
    if "." in text:
        code = text.split(".")[0].zfill(4)
        return code, text
    code = text.zfill(4)
    return code, f"{code}.TW"


def parse_twse_csv(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    start_index = None
    for idx, line in enumerate(lines):
        if "證券代號" in line and "證券名稱" in line:
            start_index = idx
            break

    if start_index is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[start_index:])))
    except Exception:
        return pd.DataFrame()

    df = df.dropna(how="all")
    df.columns = [str(col).strip().replace('"', "") for col in df.columns]
    return df


class MomentumDatabase:
    def __init__(self, db_name: str):
        self.db_path = script_path(db_name)

    def connect(self):
        return sqlite3.connect(self.db_path)

    def init(self):
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS listed_stocks (
                    code TEXT NOT NULL,
                    name TEXT,
                    market TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    source_date TEXT NOT NULL,
                    updated_at TEXT,
                    PRIMARY KEY (code, market)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT NOT NULL,
                    code TEXT,
                    name TEXT,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    updated_at TEXT,
                    PRIMARY KEY (ticker, date)
                )
            """)

    def load_listed_stocks(self) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query("""
                SELECT code, name, market, ticker, source_date
                FROM listed_stocks
                WHERE market = '上市'
                ORDER BY code
            """, conn)

    def save_listed_stocks(self, stocks_df: pd.DataFrame, source_date: str) -> int:
        if stocks_df.empty:
            return 0

        saved = 0
        with self.connect() as conn:
            for _, row in stocks_df.iterrows():
                code = str(row.get("code", "")).strip().zfill(4)
                name = str(row.get("name", "")).strip()
                ticker = str(row.get("ticker", f"{code}.TW")).strip()
                if not code or not ticker:
                    continue

                conn.execute("""
                    INSERT OR REPLACE INTO listed_stocks
                    (code, name, market, ticker, source_date, updated_at)
                    VALUES (?, ?, '上市', ?, ?, ?)
                """, (code, name, ticker, source_date, now_text()))
                saved += 1

        return saved

    def has_price_range(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_gap_days: int,
        min_coverage_ratio: float,
    ) -> bool:
        with self.connect() as conn:
            df = pd.read_sql_query("""
                SELECT date
                FROM daily_prices
                WHERE ticker = ?
                  AND date BETWEEN ? AND ?
                ORDER BY date
            """, conn, params=(ticker, start_date, end_date))

        if df.empty:
            return False

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        if df.empty:
            return False

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if df["date"].min() > start_dt + pd.Timedelta(days=7):
            return False
        if df["date"].max() < end_dt - pd.Timedelta(days=7):
            return False

        expected_days = len(pd.bdate_range(start_dt, end_dt))
        if expected_days and len(df) / expected_days < min_coverage_ratio:
            return False

        max_gap = df["date"].diff().dt.days.max()
        if pd.notna(max_gap) and max_gap > max_gap_days:
            return False

        return True

    def save_price_history(self, code: str, name: str, ticker: str, hist: pd.DataFrame) -> int:
        if hist.empty:
            return 0

        saved = 0
        with self.connect() as conn:
            for price_date, row in hist.iterrows():
                date_value = normalize_date(price_date)
                if date_value is None:
                    continue

                conn.execute("""
                    INSERT OR REPLACE INTO daily_prices
                    (ticker, code, name, date, open, high, low, close, adj_close, volume, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    code,
                    name,
                    date_value,
                    clean_number(row.get("Open")),
                    clean_number(row.get("High")),
                    clean_number(row.get("Low")),
                    clean_number(row.get("Close")),
                    clean_number(row.get("Adj Close")),
                    clean_int(row.get("Volume")),
                    now_text(),
                ))
                saved += 1

        return saved

    def load_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query("""
                SELECT ticker, code, name, date, open, high, low, close, adj_close, volume
                FROM daily_prices
                WHERE date BETWEEN ? AND ?
                ORDER BY ticker, date
            """, conn, params=(start_date, end_date))


class TaiwanListedClient:
    def fetch_listed_stocks(self, date_str: str) -> pd.DataFrame:
        url = (
            "https://www.twse.com.tw/exchangeReport/MI_INDEX"
            f"?response=csv&date={to_twse_date(date_str)}&type=ALLBUT0999"
        )
        print(f"抓 TWSE 上市股票清單：{date_str}")

        try:
            res = requests.get(url, headers=HEADERS, timeout=20)
        except requests.RequestException as exc:
            print(f"  TWSE 請求失敗：{exc}")
            return pd.DataFrame()

        if res.status_code != 200 or not res.text.strip():
            return pd.DataFrame()

        raw_df = parse_twse_csv(res.text)
        if raw_df.empty or "證券代號" not in raw_df.columns or "證券名稱" not in raw_df.columns:
            return pd.DataFrame()

        df = pd.DataFrame()
        df["code"] = raw_df["證券代號"].astype(str).str.strip().str.extract(r"(\d{4})", expand=False)
        df["name"] = raw_df["證券名稱"].astype(str).str.strip()
        df = df.dropna(subset=["code"])
        df = df[df["code"].str.fullmatch(r"\d{4}")]
        df = df.drop_duplicates("code").copy()
        df["market"] = "上市"
        df["ticker"] = df["code"] + ".TW"
        return df[["code", "name", "market", "ticker"]]


class MomentumPRAnalyzer:
    def __init__(self, config: MomentumConfig):
        self.config = config
        self.db = MomentumDatabase(config.db_name)
        self.twse = TaiwanListedClient()

    def fetch_listed_stocks_if_needed(self):
        listed_df = self.db.load_listed_stocks()
        if not listed_df.empty:
            print(f"上市股票清單已存在 SQL：{len(listed_df)} 檔，略過 TWSE 清單抓取")
            return

        end_dt = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        for offset in range(self.config.universe_lookback_days + 1):
            date_str = (end_dt - timedelta(days=offset)).strftime("%Y-%m-%d")
            listed_df = self.twse.fetch_listed_stocks(date_str)
            if listed_df.empty:
                continue
            saved = self.db.save_listed_stocks(listed_df, date_str)
            print(f"已存入上市股票清單：{saved} 檔")
            return

        print("無法取得上市股票清單")

    def fetch_prices_if_needed(self):
        listed_df = self.db.load_listed_stocks()
        if listed_df.empty:
            print("沒有上市股票清單，無法抓股價")
            return

        for idx, row in listed_df.iterrows():
            code = row["code"]
            name = row["name"]
            ticker = row["ticker"]

            if self.db.has_price_range(
                ticker,
                self.config.start_date,
                self.config.end_date,
                self.config.max_price_gap_days,
                self.config.min_price_coverage_ratio,
            ):
                print(f"[{idx + 1}/{len(listed_df)}] {ticker} 股價已存在 SQL，略過")
                continue

            print(f"[{idx + 1}/{len(listed_df)}] 抓股價：{code} {name} ({ticker})")
            try:
                hist = yf.Ticker(ticker).history(
                    start=self.config.start_date,
                    end=self.config.end_date,
                    interval="1d",
                    auto_adjust=False,
                )
            except Exception as exc:
                print(f"  yfinance 失敗：{exc}")
                continue

            if hist.empty:
                print("  無資料")
                continue

            saved = self.db.save_price_history(code, name, ticker, hist)
            print(f"  已存入 {saved} 筆")
            time.sleep(self.config.yfinance_sleep)

    def build_close_pivot(self) -> pd.DataFrame:
        price_df = self.db.load_prices(self.config.start_date, self.config.end_date)
        if price_df.empty:
            return pd.DataFrame()

        price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
        price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
        price_df = price_df.dropna(subset=["date", "ticker", "close"])
        return price_df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()

    def calculate_quarter_returns(self, close_pivot: pd.DataFrame) -> pd.DataFrame:
        window = self.config.quarter_window
        required_days = window * 4 + 1
        valid = close_pivot.dropna(axis=1, thresh=required_days)
        if valid.empty or len(valid) < required_days:
            return pd.DataFrame()

        end_prices = valid.iloc[-1]
        q1_start = valid.iloc[-1 - window]
        q2_start = valid.iloc[-1 - window * 2]
        q3_start = valid.iloc[-1 - window * 3]
        q4_start = valid.iloc[-1 - window * 4]

        q1_return = end_prices / q1_start - 1
        q2_return = q1_start / q2_start - 1
        q3_return = q2_start / q3_start - 1
        q4_return = q3_start / q4_start - 1

        result = pd.DataFrame({
            "ticker": valid.columns,
            "q1_return_recent": q1_return.values,
            "q2_return": q2_return.values,
            "q3_return": q3_return.values,
            "q4_return_oldest": q4_return.values,
        })
        return result.dropna()

    def score_momentum(self) -> pd.DataFrame:
        close_pivot = self.build_close_pivot()
        quarter_df = self.calculate_quarter_returns(close_pivot)
        if quarter_df.empty:
            print("股價資料不足，無法計算四季度 PR")
            return pd.DataFrame()

        return_cols = ["q1_return_recent", "q2_return", "q3_return", "q4_return_oldest"]
        for col in return_cols:
            quarter_df[f"{col}_pr"] = quarter_df[col].rank(pct=True) * 100

        quarter_df["weighted_pr_score"] = (
            quarter_df["q1_return_recent_pr"] * 0.40
            + quarter_df["q2_return_pr"] * 0.20
            + quarter_df["q3_return_pr"] * 0.20
            + quarter_df["q4_return_oldest_pr"] * 0.20
        )

        listed_df = self.db.load_listed_stocks()[["ticker", "code", "name"]]
        scored_df = quarter_df.merge(listed_df, on="ticker", how="left")
        scored_df = scored_df.sort_values("weighted_pr_score", ascending=False).reset_index(drop=True)
        scored_df["rank"] = scored_df.index + 1

        all_output = script_path("market_momentum_pr_scores.csv")
        strong_output = script_path("strong_stocks_pr80.csv")
        scored_df.to_csv(all_output, index=False, encoding="utf-8-sig")

        strong_df = scored_df[scored_df["weighted_pr_score"] > self.config.pr_threshold].copy()
        strong_df.to_csv(strong_output, index=False, encoding="utf-8-sig")

        print("\n========== 動能 PR 篩選 ==========")
        print(f"可計算股票數：{len(scored_df)}")
        print(f"PR > {self.config.pr_threshold:.0f} 強勢股：{len(strong_df)}")
        print(f"已輸出：{all_output}")
        print(f"已輸出：{strong_output}")
        return scored_df

    def run_market_pr(self) -> dict:
        self.db.init()
        self.fetch_listed_stocks_if_needed()
        self.fetch_prices_if_needed()
        scored_df = self.score_momentum()
        if scored_df.empty:
            return {
                "ok": False,
                "message": "股價資料不足，無法計算市場 PR。",
                "total_count": 0,
                "strong_count": 0,
                "rows": [],
            }

        strong_df = scored_df[scored_df["weighted_pr_score"] > self.config.pr_threshold].copy()
        preview_cols = [
            "rank", "code", "name", "ticker", "weighted_pr_score",
            "q1_return_recent", "q2_return", "q3_return", "q4_return_oldest",
        ]
        rows = strong_df[preview_cols].head(100).copy()
        for col in ["weighted_pr_score", "q1_return_recent", "q2_return", "q3_return", "q4_return_oldest"]:
            rows[col] = rows[col].astype(float).round(4)

        return {
            "ok": True,
            "message": f"完成：PR > {self.config.pr_threshold:.0f} 強勢股 {len(strong_df)} 檔。",
            "total_count": int(len(scored_df)),
            "strong_count": int(len(strong_df)),
            "rows": rows.to_dict(orient="records"),
            "score_csv": "market_momentum_pr_scores.csv",
            "strong_csv": "strong_stocks_pr80.csv",
        }

    def fetch_market_index(self) -> pd.DataFrame:
        print("抓大盤指數：^TWII")
        try:
            hist = yf.Ticker("^TWII").history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval="1d",
                auto_adjust=False,
            )
        except Exception as exc:
            print(f"  大盤資料抓取失敗：{exc}")
            return pd.DataFrame()

        if hist.empty:
            return pd.DataFrame()

        df = hist.reset_index()
        df["date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
        df["twii_close"] = pd.to_numeric(df["Close"], errors="coerce")
        return df[["date", "twii_close"]].dropna()

    @staticmethod
    def normalize_to_100(series: pd.Series) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty or series.iloc[0] == 0:
            return series
        return series / series.iloc[0] * 100

    def build_momentum_pr_curve(self, close_pivot: pd.DataFrame, ticker: str) -> pd.Series:
        rolling_returns = close_pivot / close_pivot.shift(self.config.quarter_window) - 1
        pr_table = rolling_returns.rank(axis=1, pct=True) * 100
        if ticker not in pr_table.columns:
            return pd.Series(dtype=float)
        return pr_table[ticker].dropna()

    def visualize_stock(self, code_or_ticker: str):
        code, ticker = make_ticker(code_or_ticker)
        close_pivot = self.build_close_pivot()
        if close_pivot.empty or ticker not in close_pivot.columns:
            print(f"找不到 {ticker} 的股價資料，無法繪圖")
            return

        market_df = self.fetch_market_index()
        stock_series = close_pivot[ticker].dropna()
        stock_df = stock_series.rename("stock_close").reset_index().rename(columns={"index": "date"})
        stock_df["date"] = pd.to_datetime(stock_df["date"], errors="coerce")

        chart_df = stock_df.merge(market_df, on="date", how="inner")
        if chart_df.empty:
            print("股票與大盤沒有重疊日期，無法繪圖")
            return

        chart_df["stock_index"] = self.normalize_to_100(chart_df["stock_close"]).values
        chart_df["twii_index"] = self.normalize_to_100(chart_df["twii_close"]).values

        pr_curve = self.build_momentum_pr_curve(close_pivot, ticker)
        pr_df = pr_curve.rename("momentum_pr").reset_index().rename(columns={"index": "date"})
        pr_df["date"] = pd.to_datetime(pr_df["date"], errors="coerce")
        chart_df = chart_df.merge(pr_df, on="date", how="left")

        listed_df = self.db.load_listed_stocks()
        name = ticker
        matched = listed_df[listed_df["ticker"] == ticker]
        if not matched.empty:
            name = f"{matched.iloc[0]['code']} {matched.iloc[0]['name']}"

        fig, (ax_price, ax_pr) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        ax_price.plot(chart_df["date"], chart_df["stock_index"], label=name, linewidth=2)
        ax_price.plot(chart_df["date"], chart_df["twii_index"], label="^TWII", linewidth=2, alpha=0.8)
        ax_price.set_title(f"{name} vs 台股指數（基期為 100）")
        ax_price.set_ylabel("Normalized Index")
        ax_price.grid(True, alpha=0.25)
        ax_price.legend()

        ax_pr.plot(chart_df["date"], chart_df["momentum_pr"], color="#2563eb", linewidth=1.8, label="63日動能PR")
        ax_pr.fill_between(
            chart_df["date"],
            chart_df["momentum_pr"],
            80,
            where=chart_df["momentum_pr"] >= 80,
            color="#16a34a",
            alpha=0.20,
            interpolate=True,
        )
        ax_pr.fill_between(
            chart_df["date"],
            chart_df["momentum_pr"],
            20,
            where=chart_df["momentum_pr"] <= 20,
            color="#dc2626",
            alpha=0.20,
            interpolate=True,
        )
        ax_pr.axhline(80, color="#16a34a", linestyle="--", linewidth=1)
        ax_pr.axhline(20, color="#dc2626", linestyle="--", linewidth=1)
        ax_pr.set_ylim(0, 100)
        ax_pr.set_ylabel("Momentum PR")
        ax_pr.set_title("63日動能PR")
        ax_pr.grid(True, alpha=0.25)
        ax_pr.legend()

        fig.tight_layout()
        output_path = script_path(f"{code}_momentum_visualization.png")
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        print(f"已輸出視覺化圖表：{output_path}")
        return output_path

    def run_visualization(self, code_or_ticker: str) -> dict:
        self.db.init()
        code, ticker = make_ticker(code_or_ticker)
        with self.db.connect() as conn:
            row = conn.execute("""
                SELECT code, name, ticker
                FROM listed_stocks
                WHERE ticker = ?
            """, (ticker,)).fetchone()

        if row is not None:
            listed = pd.DataFrame([{"code": row[0], "name": row[1], "ticker": row[2]}])
        else:
            listed = pd.DataFrame([{"code": code, "name": code, "ticker": ticker}])

        for _, item in listed.iterrows():
            if not self.db.has_price_range(
                item["ticker"],
                self.config.start_date,
                self.config.end_date,
                self.config.max_price_gap_days,
                self.config.min_price_coverage_ratio,
            ):
                try:
                    hist = yf.Ticker(item["ticker"]).history(
                        start=self.config.start_date,
                        end=self.config.end_date,
                        interval="1d",
                        auto_adjust=False,
                    )
                    self.db.save_price_history(item["code"], item["name"], item["ticker"], hist)
                except Exception as exc:
                    return {"ok": False, "message": f"{item['ticker']} 股價抓取失敗：{exc}"}

        output_path = self.visualize_stock(code_or_ticker)
        if not output_path:
            return {"ok": False, "message": f"無法產生 {ticker} 圖表。"}

        return {
            "ok": True,
            "message": f"已產生 {ticker} 動能視覺化圖。",
            "image": os.path.basename(output_path),
        }

APP_HTML = """
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>台股動能 PR 分析</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #d8dee8;
      --brand: #2563eb;
      --good: #16a34a;
    }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px;
    }
    h1 {
      margin: 0 0 18px;
      font-size: 28px;
      letter-spacing: 0;
    }
    .tabs {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 18px;
    }
    .tab {
      border: 0;
      background: transparent;
      padding: 12px 16px;
      font-size: 15px;
      cursor: pointer;
      border-bottom: 3px solid transparent;
      color: var(--muted);
    }
    .tab.active {
      color: var(--brand);
      border-bottom-color: var(--brand);
      font-weight: 700;
    }
    section {
      display: none;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
    }
    section.active { display: block; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 16px;
    }
    button.primary {
      border: 0;
      background: var(--brand);
      color: white;
      border-radius: 6px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }
    input {
      width: min(280px, 100%);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      font-size: 15px;
    }
    .status {
      color: var(--muted);
      min-height: 24px;
      margin: 8px 0 14px;
      white-space: pre-wrap;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: #fbfdff;
    }
    .metric strong {
      display: block;
      font-size: 22px;
      margin-top: 4px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: right;
      white-space: nowrap;
    }
    th:nth-child(2), th:nth-child(3),
    td:nth-child(2), td:nth-child(3) {
      text-align: left;
    }
    .table-wrap {
      overflow: auto;
      max-height: 560px;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .links {
      display: flex;
      gap: 12px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    a {
      color: var(--brand);
      font-weight: 600;
      text-decoration: none;
    }
    .chart {
      width: 100%;
      max-width: 1080px;
      border: 1px solid var(--line);
      border-radius: 8px;
      display: none;
      margin-top: 14px;
    }
  </style>
</head>
<body>
<main>
  <h1>台股動能 PR 分析</h1>
  <div class="tabs">
    <button class="tab active" data-tab="market">市場 PR 篩選</button>
    <button class="tab" data-tab="chart">股票代號製圖</button>
  </div>

  <section id="market" class="active">
    <div class="toolbar">
      <button class="primary" id="runPr">執行市場 PR 篩選</button>
    </div>
    <div class="status" id="marketStatus">讀取全上市清單與股價快取，已存在 SQL 的資料會略過。</div>
    <div class="summary">
      <div class="metric">可計算股票數<strong id="totalCount">-</strong></div>
      <div class="metric">PR &gt; 80 強勢股<strong id="strongCount">-</strong></div>
    </div>
    <div class="links" id="csvLinks"></div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>名稱</th><th>代號</th><th>PR</th>
            <th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th>
          </tr>
        </thead>
        <tbody id="resultRows"></tbody>
      </table>
    </div>
  </section>

  <section id="chart">
    <div class="toolbar">
      <input id="symbol" placeholder="輸入股票代號，例如 2330">
      <button class="primary" id="drawChart">產生圖表</button>
    </div>
    <div class="status" id="chartStatus">會產生與加權指數的基期 100 對比圖，以及 63 日動能 PR 曲線。</div>
    <img id="chartImage" class="chart" alt="momentum chart">
  </section>
</main>

<script>
const tabs = document.querySelectorAll(".tab");
tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    tabs.forEach(t => t.classList.remove("active"));
    document.querySelectorAll("section").forEach(s => s.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(tab.dataset.tab).classList.add("active");
  });
});

function pct(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  return (Number(v) * 100).toFixed(2) + "%";
}

document.getElementById("runPr").addEventListener("click", async () => {
  const status = document.getElementById("marketStatus");
  status.textContent = "執行中，第一次抓全市場資料會花比較久...";
  const res = await fetch("/api/run-pr", { method: "POST" });
  const data = await res.json();
  status.textContent = data.message || "";
  document.getElementById("totalCount").textContent = data.total_count ?? "-";
  document.getElementById("strongCount").textContent = data.strong_count ?? "-";

  const links = document.getElementById("csvLinks");
  links.innerHTML = data.ok ? `
    <a href="/outputs/${data.score_csv}" target="_blank">全部 PR 報表</a>
    <a href="/outputs/${data.strong_csv}" target="_blank">PR80 強勢股</a>
  ` : "";

  const body = document.getElementById("resultRows");
  body.innerHTML = "";
  (data.rows || []).forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.rank}</td>
      <td>${row.name || ""}</td>
      <td>${row.ticker || ""}</td>
      <td>${Number(row.weighted_pr_score).toFixed(2)}</td>
      <td>${pct(row.q1_return_recent)}</td>
      <td>${pct(row.q2_return)}</td>
      <td>${pct(row.q3_return)}</td>
      <td>${pct(row.q4_return_oldest)}</td>
    `;
    body.appendChild(tr);
  });
});

document.getElementById("drawChart").addEventListener("click", async () => {
  const symbol = document.getElementById("symbol").value.trim();
  const status = document.getElementById("chartStatus");
  const img = document.getElementById("chartImage");
  if (!symbol) {
    status.textContent = "請先輸入股票代號。";
    return;
  }
  status.textContent = "產生圖表中...";
  img.style.display = "none";
  const res = await fetch("/api/visualize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol })
  });
  const data = await res.json();
  status.textContent = data.message || "";
  if (data.ok) {
    img.src = `/outputs/${data.image}?t=${Date.now()}`;
    img.style.display = "block";
  }
});
</script>
</body>
</html>
"""


def create_app() -> Flask:
    analyzer = MomentumPRAnalyzer(MomentumConfig())
    app = Flask(__name__)

    @app.get("/")
    def index():
        return APP_HTML

    @app.post("/api/run-pr")
    def run_pr():
        try:
            return jsonify(analyzer.run_market_pr())
        except Exception as exc:
            return jsonify({"ok": False, "message": f"執行失敗：{exc}", "rows": []}), 500

    @app.post("/api/visualize")
    def visualize():
        payload = request.get_json(silent=True) or {}
        symbol = str(payload.get("symbol", "")).strip()
        if not symbol:
            return jsonify({"ok": False, "message": "請輸入股票代號。"}), 400
        try:
            return jsonify(analyzer.run_visualization(symbol))
        except Exception as exc:
            return jsonify({"ok": False, "message": f"製圖失敗：{exc}"}), 500

    @app.get("/outputs/<path:filename>")
    def outputs(filename):
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), filename)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5003, debug=False)
