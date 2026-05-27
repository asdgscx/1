"""
台股動能 PR 分析工具 (包含 CAN SLIM 7% 絕對停損機制)。

這支程式同時提供兩個功能：
1. 市場 PR 篩選：讀取指定日期附近的 TWSE 全上市股票清單，抓取股價，
   計算四個季度區間報酬率，並用 40/20/20/20 權重得到綜合 PR 分數。
2. 股票代號製圖：輸入股票代號與日期區間，產生個股相對大盤的基期 100 圖，
   以及 63 日動能 PR 曲線。
3. 買進視窗回測：模擬買進後的走勢，並執行停利停損策略。
"""

import io
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

MPL_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)

import matplotlib

# 🛡️ 強化字體設定：依序尋找 微軟正黑體(Win) -> Arial Unicode MS(Mac) -> 系統預設黑體
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Arial Unicode MS", "sans-serif"]
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
    """集中管理回測、抓資料、PR 計算會用到的可調參數。"""

    db_name: str = "taiwan_momentum_pr_backtest.db"
    start_date: str = "2025-01-01"
    end_date: str = "2026-04-30"
    quarter_window: int = 63
    rolling_window: int = 63
    pr_threshold: float = 80.0
    yfinance_sleep: float = 0.15
    max_price_gap_days: int = 14
    min_price_coverage_ratio: float = 0.60
    universe_lookback_days: int = 10
    min_pr_universe_count: int = 50
    # 🛡️ 新增：CAN SLIM 絕對停損比例 (7%)
    stop_loss_pct: float = 0.07 


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

def add_days(date_str: str, days: int) -> str:
    return (pd.to_datetime(date_str) + pd.Timedelta(days=days)).strftime("%Y-%m-%d")

def parse_date_or_default(value: Optional[str], default: str) -> str:
    parsed = pd.to_datetime(value or default, errors="coerce")
    if pd.isna(parsed):
        return default
    return parsed.strftime("%Y-%m-%d")

def market_pr_start_date(end_date: str) -> str:
    return (pd.to_datetime(end_date) - pd.Timedelta(days=460)).strftime("%Y-%m-%d")

def rolling_window_fetch_start(start_date: str, window: int) -> str:
    return (pd.to_datetime(start_date) - pd.offsets.BDay(max(window, 100))).strftime("%Y-%m-%d")

def validate_market_pr_end_date(end_date: str) -> Optional[str]:
    parsed = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(parsed):
        return "日期格式錯誤。"
    today = pd.Timestamp.today().normalize()
    min_end = today - pd.DateOffset(years=3)
    if parsed.normalize() > today:
        return f"市場 PR 日期不能晚於今天 {today.strftime('%Y-%m-%d')}。"
    if parsed.normalize() < min_end:
        return f"市場 PR 需要前面約一年資料，因此最早只能選到 {min_end.strftime('%Y-%m-%d')}。"
    return None

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
                CREATE TABLE IF NOT EXISTS listed_stock_snapshots (
                    source_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT,
                    market TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    updated_at TEXT,
                    PRIMARY KEY (source_date, code, market)
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_fetch_failures (
                    ticker TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    reason TEXT,
                    failed_at TEXT,
                    PRIMARY KEY (ticker, start_date, end_date)
                )
            """)

    def load_listed_stocks(self, source_date: Optional[str] = None) -> pd.DataFrame:
        if source_date:
            with self.connect() as conn:
                return pd.read_sql_query("""
                    SELECT code, name, market, ticker, source_date
                    FROM listed_stock_snapshots
                    WHERE market = '上市'
                      AND source_date = ?
                    ORDER BY code
                """, conn, params=(source_date,))

        with self.connect() as conn:
            return pd.read_sql_query("""
                SELECT code, name, market, ticker, source_date
                FROM listed_stocks
                WHERE market = '上市'
                ORDER BY code
            """, conn)

    def load_recent_listed_snapshot(self, as_of_date: str, lookback_days: int) -> tuple[pd.DataFrame, Optional[str]]:
        end_dt = pd.to_datetime(as_of_date)
        start_dt = end_dt - pd.Timedelta(days=lookback_days)
        with self.connect() as conn:
            row = conn.execute("""
                SELECT source_date
                FROM listed_stock_snapshots
                WHERE source_date BETWEEN ? AND ?
                GROUP BY source_date
                ORDER BY source_date DESC
                LIMIT 1
            """, (start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))).fetchone()

        if row is None:
            return pd.DataFrame(), None
        return self.load_listed_stocks(row[0]), row[0]

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
                conn.execute("""
                    INSERT OR REPLACE INTO listed_stock_snapshots
                    (source_date, code, name, market, ticker, updated_at)
                    VALUES (?, ?, ?, '上市', ?, ?)
                """, (source_date, code, name, ticker, now_text()))
                saved += 1
        return saved

    def has_price_fetch_failure(self, ticker: str, start_date: str, end_date: str) -> bool:
        with self.connect() as conn:
            row = conn.execute("""
                SELECT 1
                FROM price_fetch_failures
                WHERE ticker = ?
                  AND start_date = ?
                  AND end_date = ?
                LIMIT 1
            """, (ticker, start_date, end_date)).fetchone()
        return row is not None 

    def save_price_fetch_failure(self, ticker: str, start_date: str, end_date: str, reason: str):
        with self.connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO price_fetch_failures
                (ticker, start_date, end_date, reason, failed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker, start_date, end_date, reason[:500], now_text()))

    def clear_price_fetch_failure(self, ticker: str, start_date: str, end_date: str):
        with self.connect() as conn:
            conn.execute("""
                DELETE FROM price_fetch_failures
                WHERE ticker = ?
                  AND start_date = ?
                  AND end_date = ?
            """, (ticker, start_date, end_date))

    def has_price_range(self, ticker: str, start_date: str, end_date: str, max_gap_days: int, min_coverage_ratio: float) -> bool:
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
        effective_end_dt = min(end_dt, pd.Timestamp.today().normalize())
        if df["date"].min() > start_dt + pd.Timedelta(days=7):
            return False
        if df["date"].max() < start_dt:
            return False
        required_dates = pd.bdate_range(start_dt, effective_end_dt)
        required_last_date = required_dates[-1] if len(required_dates) else effective_end_dt
        if df["date"].max() < required_last_date:
            return False
        expected_days = len(pd.bdate_range(start_dt, effective_end_dt))
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
                    ticker, code, name, date_value,
                    clean_number(row.get("Open")),
                    clean_number(row.get("High")),
                    clean_number(row.get("Low")),
                    clean_number(row.get("Close")),
                    clean_number(row.get("Adj Close")),
                    clean_int(row.get("Volume")),
                    now_text(),
                ))
                saved += 1
        if saved:
            dates = [normalize_date(idx) for idx in hist.index]
            dates = [value for value in dates if value]
            if dates:
                self.clear_price_fetch_failure(ticker, min(dates), max(dates))
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

    def fetch_listed_stocks_if_needed(self, as_of_date: Optional[str] = None) -> pd.DataFrame:
        as_of_date = parse_date_or_default(as_of_date, self.config.end_date)
        listed_df, source_date = self.db.load_recent_listed_snapshot(as_of_date, self.config.universe_lookback_days)
        if not listed_df.empty:
            print(f"TWSE {source_date} 上市股票清單已存在 SQL：{len(listed_df)} 檔，略過清單抓取")
            return listed_df
        end_dt = datetime.strptime(as_of_date or self.config.end_date, "%Y-%m-%d")
        for offset in range(self.config.universe_lookback_days + 1):
            date_str = (end_dt - timedelta(days=offset)).strftime("%Y-%m-%d")
            listed_df = self.twse.fetch_listed_stocks(date_str)
            if listed_df.empty:
                continue
            saved = self.db.save_listed_stocks(listed_df, date_str)
            print(f"已存入上市股票清單：{saved} 檔")
            return self.db.load_listed_stocks(date_str)
        print("無法取得上市股票清單")
        return pd.DataFrame()

    def fetch_prices_if_needed(self, start_date: Optional[str] = None, end_date: Optional[str] = None, listed_df: Optional[pd.DataFrame] = None):
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        if listed_df is None:
            listed_df = self.db.load_listed_stocks()
        if listed_df.empty:
            print("沒有上市股票清單，無法抓股價")
            return
        for idx, row in listed_df.iterrows():
            code = row["code"]
            name = row["name"]
            ticker = row["ticker"]
            if self.db.has_price_range(ticker, start_date, end_date, self.config.max_price_gap_days, self.config.min_price_coverage_ratio):
                print(f"[{idx + 1}/{len(listed_df)}] {ticker} 股價已存在 SQL，略過")
                continue
            if self.db.has_price_fetch_failure(ticker, start_date, end_date):
                print(f"[{idx + 1}/{len(listed_df)}] {ticker} 先前抓不到這段股價，略過重試")
                continue
            print(f"[{idx + 1}/{len(listed_df)}] 抓股價：{code} {name} ({ticker})")
            try:
                hist = yf.Ticker(ticker).history(start=start_date, end=add_days(end_date, 1), interval="1d", auto_adjust=False)
            except Exception as exc:
                print(f"  yfinance 失敗：{exc}")
                continue
            if hist.empty:
                print("  無資料")
                self.db.save_price_fetch_failure(ticker, start_date, end_date, "empty history")
                continue
            saved = self.db.save_price_history(code, name, ticker, hist)
            print(f"  已存入 {saved} 筆")
            time.sleep(self.config.yfinance_sleep) 

    def build_close_pivot(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        price_df = self.db.load_prices(start_date, end_date)
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

    def score_momentum(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        close_pivot = self.build_close_pivot(start_date, end_date)
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
        suffix = end_date.replace("-", "")
        all_output = script_path(f"market_momentum_pr_scores_{suffix}.csv")
        strong_output = script_path(f"strong_stocks_pr80_{suffix}.csv")
        scored_df.to_csv(all_output, index=False, encoding="utf-8-sig")
        strong_df = scored_df[scored_df["weighted_pr_score"] > self.config.pr_threshold].copy()
        strong_df.to_csv(strong_output, index=False, encoding="utf-8-sig")
        return scored_df

    def run_market_pr(self, end_date: Optional[str] = None) -> dict:
        end_date = parse_date_or_default(end_date, self.config.end_date)
        error = validate_market_pr_end_date(end_date)
        if error:
            return {"ok": False, "message": error, "total_count": 0, "strong_count": 0, "rows": []}
        start_date = market_pr_start_date(end_date)
        self.db.init()
        listed_df = self.fetch_listed_stocks_if_needed(end_date)
        self.fetch_prices_if_needed(start_date, end_date, listed_df)
        scored_df = self.score_momentum(start_date, end_date)
        if scored_df.empty:
            return {"ok": False, "message": "股價資料不足，無法計算市場 PR。", "total_count": 0, "strong_count": 0, "rows": []}
        strong_df = scored_df[scored_df["weighted_pr_score"] > self.config.pr_threshold].copy()
        preview_cols = ["rank", "code", "name", "ticker", "weighted_pr_score", "q1_return_recent", "q2_return", "q3_return", "q4_return_oldest"]
        rows = strong_df[preview_cols].head(100).copy()
        for col in ["weighted_pr_score", "q1_return_recent", "q2_return", "q3_return", "q4_return_oldest"]:
            rows[col] = rows[col].astype(float).round(4)
        return {
            "ok": True,
            "message": f"完成：PR > {self.config.pr_threshold:.0f} 強勢股 {len(strong_df)} 檔。",
            "total_count": int(len(scored_df)),
            "strong_count": int(len(strong_df)),
            "rows": rows.to_dict(orient="records"),
            "score_csv": f"market_momentum_pr_scores_{end_date.replace('-', '')}.csv",
            "strong_csv": f"strong_stocks_pr80_{end_date.replace('-', '')}.csv",
            "selected_date": end_date,
        }

    def fetch_market_index(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        try:
            hist = yf.Ticker("^TWII").history(start=start_date, end=add_days(end_date, 1), interval="1d", auto_adjust=False)
        except Exception:
            return pd.DataFrame()
        if hist.empty:
            return pd.DataFrame()
        df = hist.reset_index()
        date_col = pd.to_datetime(df["Date"], errors="coerce")
        try:
            df["date"] = date_col.dt.tz_localize(None)
        except TypeError:
            df["date"] = date_col.dt.tz_convert(None)
        df["twii_close"] = pd.to_numeric(df["Close"], errors="coerce")
        return df[["date", "twii_close"]].dropna()

    @staticmethod
    def normalize_to_100(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.dropna()
        if valid.empty or valid.iloc[0] == 0:
            return numeric
        return numeric / valid.iloc[0] * 100

    def build_momentum_pr_curve(self, close_pivot: pd.DataFrame, ticker: str) -> pd.Series:
        rolling_returns = close_pivot / close_pivot.shift(self.config.rolling_window) - 1
        pr_table = rolling_returns.rank(axis=1, pct=True) * 100
        valid_counts = rolling_returns.notna().sum(axis=1)
        pr_table = pr_table.where(valid_counts >= self.config.min_pr_universe_count)
        if ticker not in pr_table.columns:
            return pd.Series(dtype=float)
        return pr_table[ticker].dropna()

    def visualize_stock(self, code_or_ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None, price_start_date: Optional[str] = None):
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        price_start_date = price_start_date or start_date
        code, ticker = make_ticker(code_or_ticker)
        close_pivot = self.build_close_pivot(price_start_date, end_date)
        if close_pivot.empty or ticker not in close_pivot.columns:
            return
        market_df = self.fetch_market_index(start_date, end_date)
        stock_series = close_pivot[ticker].dropna()
        stock_df = stock_series.rename("stock_close").reset_index().rename(columns={"index": "date"})
        stock_df["date"] = pd.to_datetime(stock_df["date"], errors="coerce")
        stock_df = stock_df[stock_df["date"] >= pd.to_datetime(start_date)]
        if market_df.empty:
            chart_df = stock_df.copy()
            chart_df["twii_close"] = pd.NA
        else:
            chart_df = stock_df.merge(market_df, on="date", how="left")
        if chart_df.empty:
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
        ax_price.set_title(f"{name} vs 台股指數（基期為100）")
        ax_price.set_ylabel("Normalized Index")
        ax_price.grid(True, alpha=0.25)
        ax_price.legend()

        ax_pr.plot(chart_df["date"], chart_df["momentum_pr"], color="#2563eb", linewidth=1.8, label="63日動能PR")
        ax_pr.fill_between(chart_df["date"], chart_df["momentum_pr"], 80, where=chart_df["momentum_pr"] >= 80, color="#16a34a", alpha=0.20, interpolate=True)
        ax_pr.fill_between(chart_df["date"], chart_df["momentum_pr"], 20, where=chart_df["momentum_pr"] <= 20, color="#dc2626", alpha=0.20, interpolate=True)
        ax_pr.axhline(80, color="#16a34a", linestyle="--", linewidth=1)
        ax_pr.axhline(20, color="#dc2626", linestyle="--", linewidth=1)
        ax_pr.set_ylim(0, 100)
        ax_pr.set_ylabel("Momentum PR")
        ax_pr.set_title("63日動能PR")
        ax_pr.grid(True, alpha=0.25)
        ax_pr.legend()

        fig.tight_layout()
        output_path = script_path(f"{code}_momentum_visualization_{start_date.replace('-', '')}_{end_date.replace('-', '')}.png")
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path

    def run_visualization(self, code_or_ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict:
        start_date = parse_date_or_default(start_date, self.config.start_date)
        end_date = parse_date_or_default(end_date, self.config.end_date)
        if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
            return {"ok": False, "message": "開始日期必須早於結束日期。"}
        price_start_date = rolling_window_fetch_start(start_date, self.config.rolling_window)
        self.db.init()
        listed_df = self.fetch_listed_stocks_if_needed(start_date)
        self.fetch_prices_if_needed(price_start_date, end_date, listed_df)
        code, ticker = make_ticker(code_or_ticker)
        with self.db.connect() as conn:
            row = conn.execute("SELECT code, name, ticker FROM listed_stocks WHERE ticker = ?", (ticker,)).fetchone()
        if row is not None:
            listed = pd.DataFrame([{"code": row[0], "name": row[1], "ticker": row[2]}])
        else:
            listed = pd.DataFrame([{"code": code, "name": code, "ticker": ticker}])

        for _, item in listed.iterrows():
            if not self.db.has_price_range(item["ticker"], price_start_date, end_date, self.config.max_price_gap_days, self.config.min_price_coverage_ratio):
                if self.db.has_price_fetch_failure(item["ticker"], price_start_date, end_date):
                    return {"ok": False, "message": f"{item['ticker']} 先前已確認抓不到資料，略過重試。"}
                try:
                    hist = yf.Ticker(item["ticker"]).history(start=price_start_date, end=add_days(end_date, 1), interval="1d", auto_adjust=False)
                    if hist.empty:
                        self.db.save_price_fetch_failure(item["ticker"], price_start_date, end_date, "empty history")
                        return {"ok": False, "message": f"{item['ticker']} 無股價資料。"}
                    self.db.save_price_history(item["code"], item["name"], item["ticker"], hist)
                except Exception as exc:
                    return {"ok": False, "message": f"{item['ticker']} 股價抓取失敗：{exc}"}

        output_path = self.visualize_stock(code_or_ticker, start_date, end_date, price_start_date)
        if not output_path:
            return {"ok": False, "message": f"無法產生 {ticker} 圖表。"}
        return {"ok": True, "message": f"已產生 {ticker} 動能視覺化圖。", "image": os.path.basename(output_path)}

    def run_buy_date_window(self, code_or_ticker: str, buy_date: str, stop_loss_pct: Optional[float] = None, take_profit_pct: Optional[float] = None) -> dict:
        """
        🔥 升級版買入視窗回測：加入 CAN SLIM 絕對停損機制
        """
        buy_date = parse_date_or_default(buy_date, self.config.end_date)
        buy_dt = pd.to_datetime(buy_date)
        start_date = (buy_dt - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        end_date = (buy_dt + pd.DateOffset(months=2)).strftime("%Y-%m-%d")
        stop_loss_pct = self.config.stop_loss_pct if stop_loss_pct is None else max(float(stop_loss_pct), 0.0) / 100
        take_profit_pct = None if take_profit_pct is None or float(take_profit_pct) <= 0 else float(take_profit_pct) / 100

        self.db.init()
        code, ticker = make_ticker(code_or_ticker)
        with self.db.connect() as conn:
            row = conn.execute("SELECT code, name, ticker FROM listed_stocks WHERE ticker = ?", (ticker,)).fetchone()
        name = code
        if row is not None:
            code, name, ticker = row

        if not self.db.has_price_range(ticker, start_date, end_date, self.config.max_price_gap_days, self.config.min_price_coverage_ratio):
            if self.db.has_price_fetch_failure(ticker, start_date, end_date):
                return {"ok": False, "message": f"{ticker} 先前確認抓不到資料，略過重試。"}
            try:
                hist = yf.Ticker(ticker).history(start=start_date, end=add_days(end_date, 1), interval="1d", auto_adjust=False)
                if hist.empty:
                    self.db.save_price_fetch_failure(ticker, start_date, end_date, "empty history")
                    return {"ok": False, "message": f"{ticker} 無股價資料。"}
                self.db.save_price_history(code, name, ticker, hist)
            except Exception as exc:
                return {"ok": False, "message": f"{ticker} 股價抓取失敗：{exc}"}

        price_df = self.db.load_prices(start_date, end_date)
        price_df = price_df[price_df["ticker"] == ticker].copy()
        if price_df.empty:
            return {"ok": False, "message": f"找不到 {ticker} 的回測視窗資料。"}

        price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
        price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
        price_df["open"] = pd.to_numeric(price_df["open"], errors="coerce")
        price_df["high"] = pd.to_numeric(price_df["high"], errors="coerce")
        price_df["low"] = pd.to_numeric(price_df["low"], errors="coerce") # 載入盤中最低價
        price_df = price_df.dropna(subset=["date", "close"]).sort_values("date")
        price_df["buy_line_price"] = price_df["open"].fillna(price_df["close"])

        buy_rows = price_df[price_df["date"] >= buy_dt]
        if buy_rows.empty:
            return {"ok": False, "message": f"{ticker} 在 {buy_date} 之後沒有可買入交易日。"}

        # 1. 確立買進成本與停損價位
        buy_row = buy_rows.iloc[0]
        buy_price = buy_row["open"] if pd.notna(buy_row["open"]) else buy_row["close"]
        stop_price = buy_price * (1 - stop_loss_pct) if stop_loss_pct > 0 else None
        take_profit_price = buy_price * (1 + take_profit_pct) if take_profit_pct is not None else None

        # 2. 模擬持有期間 (從買入後的下一天開始檢查停損/停利)
        post_buy_df = price_df[price_df["date"] > buy_row["date"]]
        
        stopped_out = False
        took_profit = False
        exit_reason = "window_end"
        exit_date = price_df.iloc[-1]["date"]
        exit_price = price_df.iloc[-1]["close"]

        # 每天用高低價檢查是否觸發停損或停利；同日都碰到時採保守假設，先視為停損。
        for _, daily_row in post_buy_df.iterrows():
            current_low = daily_row["low"] if pd.notna(daily_row["low"]) else daily_row["close"]
            current_high = daily_row["high"] if pd.notna(daily_row["high"]) else daily_row["close"]
            hit_stop = stop_price is not None and current_low <= stop_price
            hit_profit = take_profit_price is not None and current_high >= take_profit_price
            if hit_stop:
                stopped_out = True
                exit_date = daily_row["date"]
                exit_price = stop_price
                exit_reason = "stop_loss"
                break
            if hit_profit:
                took_profit = True
                exit_date = daily_row["date"]
                exit_price = take_profit_price
                exit_reason = "take_profit"
                break

        window_return = (exit_price / buy_price) - 1 if buy_price else 0.0
        display_name = f"{code} {name}"

        # 製圖準備：只使用實際交易日，避免補週末/休市日造成線圖看起來每天跳空。
        plot_df = price_df[["date", "open", "high", "low", "close", "buy_line_price"]].copy()
        plot_df = plot_df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        plot_df["x"] = range(len(plot_df))
        buy_x = int(plot_df.loc[plot_df["date"] == buy_row["date"], "x"].iloc[0])
        exit_x = int(plot_df.loc[plot_df["date"] == exit_date, "x"].iloc[0]) if exit_date in set(plot_df["date"]) else int(plot_df["x"].iloc[-1])

        fig, ax = plt.subplots(figsize=(12, 5.8))
        ax.vlines(plot_df["x"], plot_df["low"], plot_df["high"], color="#cbd5e1", linewidth=1, alpha=0.85, label="當日高低")
        ax.plot(plot_df["x"], plot_df["open"], color="#2563eb", linewidth=1.8, label="開盤價")
        ax.plot(plot_df["x"], plot_df["close"], color="#475569", linewidth=1.8, label="收盤價")
        ax.axvline(buy_x, color="#dc2626", linestyle="--", linewidth=1.4, label="買進日")
        ax.scatter([buy_x], [buy_price], color="#dc2626", s=55, zorder=4)
        
        # 🛡️ 畫上停損/停利線與出場點位
        if stop_price is not None:
            ax.hlines(y=stop_price, xmin=buy_x, xmax=plot_df["x"].max(), color="#f59e0b", linestyle=":", linewidth=2, label=f"停損線 (-{stop_loss_pct*100:.1f}%)")
        if take_profit_price is not None:
            ax.hlines(y=take_profit_price, xmin=buy_x, xmax=plot_df["x"].max(), color="#16a34a", linestyle=":", linewidth=2, label=f"停利線 (+{take_profit_pct*100:.1f}%)")
        if stopped_out or took_profit:
            marker_color = "#000000" if stopped_out else "#16a34a"
            marker_label = "觸發停損出場" if stopped_out else "觸發停利出場"
            ax.scatter([exit_x], [exit_price], color=marker_color, marker="v", s=80, zorder=4, label=marker_label)
            ax.axvline(exit_x, color=marker_color, linestyle=":", linewidth=1, alpha=0.6)

        ax.set_title(f"{display_name} 買入視窗回測")
        ax.set_ylabel("Price")
        tick_count = min(8, len(plot_df))
        tick_indexes = sorted(set(round(i * (len(plot_df) - 1) / max(tick_count - 1, 1)) for i in range(tick_count)))
        ax.set_xticks(tick_indexes)
        ax.set_xticklabels([plot_df.loc[i, "date"].strftime("%Y-%m-%d") for i in tick_indexes], rotation=0)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        output_path = script_path(f"{code}_buy_window_{buy_date.replace('-', '')}.png")
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

        # 動態產生回報訊息
        msg = f"{display_name}：買入 {buy_row['date'].strftime('%Y-%m-%d')} ({buy_price:.2f})。"
        if stopped_out:
            msg += f" 於 {exit_date.strftime('%Y-%m-%d')} 觸發停損出場，結算報酬率 {window_return * 100:.2f}%"
        elif took_profit:
            msg += f" 於 {exit_date.strftime('%Y-%m-%d')} 觸發停利出場，結算報酬率 {window_return * 100:.2f}%"
        else:
            msg += f" 視窗期末未觸發停損/停利，結算報酬率 {window_return * 100:.2f}%"

        return {
            "ok": True,
            "message": msg,
            "image": os.path.basename(output_path),
            "buy_date": buy_row["date"].strftime("%Y-%m-%d"),
            "buy_price": round(float(buy_price), 2),
            "window_return": round(float(window_return), 4) if window_return is not None else None,
            "stopped_out": stopped_out,
            "took_profit": took_profit,
            "exit_reason": exit_reason,
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
      --bad: #dc2626;
      --warn: #f59e0b;
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
    input[type="date"] {
      width: 170px;
    }
    input[type="number"] {
      width: 96px;
    }
    label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    button.small {
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--brand);
      border-radius: 6px;
      padding: 6px 10px;
      font-weight: 700;
      cursor: pointer;
    }
    .status {
      color: var(--text);
      min-height: 24px;
      margin: 8px 0 14px;
      white-space: pre-wrap;
      font-weight: bold;
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
    th:nth-child(2), th:nth-child(3), th:nth-child(4),
    td:nth-child(2), td:nth-child(3), td:nth-child(4) {
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
      <label>查詢日期 <input type="date" id="prDate"></label>
      <label>停損% <input type="number" id="stopLossPct" min="0" step="0.5" value="7"></label>
      <label>停利% <input type="number" id="takeProfitPct" min="0" step="0.5" value="20"></label>
      <button class="primary" id="runPr">執行市場 PR 篩選</button>
    </div>
    <div class="status" id="marketStatus" style="color: var(--muted); font-weight: normal;">讀取全上市清單與股價快取，查詢日最早為三年前。</div>
    <div class="summary">
      <div class="metric">可計算股票數<strong id="totalCount">-</strong></div>
      <div class="metric">PR &gt; 80 強勢股<strong id="strongCount">-</strong></div>
    </div>
    <div class="links" id="csvLinks"></div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>名稱</th><th>代號</th><th>綜合分數</th>
            <th>Q1報酬</th><th>Q2報酬</th><th>Q3報酬</th><th>Q4報酬</th><th>回測</th>
          </tr>
        </thead>
        <tbody id="resultRows"></tbody>
      </table>
    </div>
    <div class="status" id="backtestStatus"></div>
    <img id="backtestImage" class="chart" alt="buy date backtest chart">
  </section>

  <section id="chart">
    <div class="toolbar">
      <input id="symbol" placeholder="輸入股票代號，例如 2330">
      <label>開始 <input type="date" id="chartStart"></label>
      <label>結束 <input type="date" id="chartEnd"></label>
      <button class="primary" id="drawChart">產生圖表</button>
    </div>
    <div class="status" id="chartStatus" style="color: var(--muted); font-weight: normal;">會產生與加權指數的基期 100 對比圖，以及 63 日動能 PR 曲線。</div>
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

function isoDate(d) {
  return d.toISOString().slice(0, 10);
}

function addYears(d, years) {
  const next = new Date(d);
  next.setFullYear(next.getFullYear() + years);
  return next;
}

function addMonths(d, months) {
  const next = new Date(d);
  next.setMonth(next.getMonth() + months);
  return next;
}

const today = new Date();
const prDate = document.getElementById("prDate");
const chartStart = document.getElementById("chartStart");
const chartEnd = document.getElementById("chartEnd");
prDate.min = isoDate(addYears(today, -3));
prDate.max = isoDate(today);
prDate.value = prDate.max;
chartEnd.max = isoDate(today);
chartEnd.value = isoDate(today);
chartStart.max = chartEnd.value;
chartStart.value = isoDate(addMonths(today, -16));

chartEnd.addEventListener("change", () => {
  chartStart.max = chartEnd.value;
});

document.getElementById("runPr").addEventListener("click", async () => {
  const runButton = document.getElementById("runPr");
  const status = document.getElementById("marketStatus");
  const selectedDate = document.getElementById("prDate").value;
  if (runButton.disabled) return;
  runButton.disabled = true;
  status.textContent = "執行中，第一次需捕捉較多資料，請耐心等候...";
  try {
    const res = await fetch("/api/run-pr", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ end_date: selectedDate })
    });
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
        <td><button class="small backtest" data-ticker="${row.ticker || ""}">模擬回測</button></td>
      `;
      body.appendChild(tr);
    });

    document.querySelectorAll(".backtest").forEach(button => {
      button.addEventListener("click", async () => {
        const ticker = button.dataset.ticker;
        const backtestStatus = document.getElementById("backtestStatus");
        const backtestImage = document.getElementById("backtestImage");
        if (button.disabled) return;
        button.disabled = true;
        backtestStatus.textContent = `${ticker} 執行停損回測運算中...`;
        backtestStatus.style.color = "var(--text)";
        backtestImage.style.display = "none";
        try {
          const btRes = await fetch("/api/backtest-window", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              symbol: ticker,
              buy_date: document.getElementById("prDate").value,
              stop_loss_pct: document.getElementById("stopLossPct").value,
              take_profit_pct: document.getElementById("takeProfitPct").value
            })
          });
          const btData = await btRes.json();
          backtestStatus.textContent = btData.message || "";
          if (btData.stopped_out) {
            backtestStatus.style.color = "var(--bad)";
          } else if (btData.took_profit) {
            backtestStatus.style.color = "var(--good)";
          } else if (btData.window_return > 0) {
            backtestStatus.style.color = "var(--good)";
          }
          if (btData.ok) {
            backtestImage.src = `/outputs/${btData.image}?t=${Date.now()}`;
            backtestImage.style.display = "block";
          }
        } finally {
          button.disabled = false;
        }
      });
    });
  } finally {
    runButton.disabled = false;
  }
});

document.getElementById("drawChart").addEventListener("click", async () => {
  const drawButton = document.getElementById("drawChart");
  const symbol = document.getElementById("symbol").value.trim();
  const status = document.getElementById("chartStatus");
  const img = document.getElementById("chartImage");
  if (!symbol) {
    status.textContent = "請先輸入股票代號。";
    return;
  }
  if (drawButton.disabled) return;
  drawButton.disabled = true;
  status.textContent = "產生圖表中；第一次需補捉較多資料，請耐心等候...";
  img.style.display = "none";
  try {
    const res = await fetch("/api/visualize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbol,
        start_date: document.getElementById("chartStart").value,
        end_date: document.getElementById("chartEnd").value
      })
    });
    const data = await res.json();
    status.textContent = data.message || "";
    if (data.ok) {
      img.src = `/outputs/${data.image}?t=${Date.now()}`;
      img.style.display = "block";
    }
  } finally {
    drawButton.disabled = false;
  }
});
</script>
</body>
</html>
"""

def create_app() -> Flask:
    analyzer = MomentumPRAnalyzer(MomentumConfig())
    app = Flask(__name__)
    task_lock = threading.Lock()

    @app.get("/")
    def index():
        return APP_HTML

    @app.post("/api/run-pr")
    def run_pr():
        payload = request.get_json(silent=True) or {}
        end_date = str(payload.get("end_date", "")).strip()
        if not task_lock.acquire(blocking=False):
            return jsonify({"ok": False, "message": "已有任務執行中，請等目前抓取完成。", "rows": []}), 409
        try:
            return jsonify(analyzer.run_market_pr(end_date))
        except Exception as exc:
            return jsonify({"ok": False, "message": f"執行失敗：{exc}", "rows": []}), 500
        finally:
            task_lock.release()

    @app.post("/api/visualize")
    def visualize():
        payload = request.get_json(silent=True) or {}
        symbol = str(payload.get("symbol", "")).strip()
        start_date = str(payload.get("start_date", "")).strip()
        end_date = str(payload.get("end_date", "")).strip()
        if not symbol:
            return jsonify({"ok": False, "message": "請輸入股票代號。"}), 400
        if not task_lock.acquire(blocking=False):
            return jsonify({"ok": False, "message": "已有任務執行中，請等目前抓取完成。"}), 409
        try:
            return jsonify(analyzer.run_visualization(symbol, start_date, end_date))
        except Exception as exc:
            return jsonify({"ok": False, "message": f"製圖失敗：{exc}"}), 500
        finally:
            task_lock.release()

    @app.post("/api/backtest-window")
    def backtest_window():
        payload = request.get_json(silent=True) or {}
        symbol = str(payload.get("symbol", "")).strip()
        buy_date = str(payload.get("buy_date", "")).strip()
        if not symbol:
            return jsonify({"ok": False, "message": "缺少股票代號。"}), 400
        if not buy_date:
            return jsonify({"ok": False, "message": "缺少買入日期。"}), 400
        if not task_lock.acquire(blocking=False):
            return jsonify({"ok": False, "message": "已有任務執行中，請等目前抓取完成。"}), 409
        try:
            stop_loss_pct = payload.get("stop_loss_pct")
            take_profit_pct = payload.get("take_profit_pct")
            stop_loss_pct = None if stop_loss_pct in (None, "") else float(stop_loss_pct)
            take_profit_pct = None if take_profit_pct in (None, "") else float(take_profit_pct)
            return jsonify(analyzer.run_buy_date_window(symbol, buy_date, stop_loss_pct, take_profit_pct))
        except Exception as exc:
            return jsonify({"ok": False, "message": f"回測製圖失敗：{exc}"}), 500
        finally:
            task_lock.release()

    @app.get("/outputs/<path:filename>")
    def outputs(filename):
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), filename)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5006, debug=False)
