"""
台股週K線過濾工具 v4
條件：
  1. 週 MA20 向上
  2. 過去 N 週內曾有大正乖離
  3. 目前股價已修正到 MA20 附近

圖表：週K線 + 布林通道（Bollinger Bands）+ 成交量
附加：三大法人週買賣超（大戶）+ 融資餘額週增減（散戶）
資料來源：TWSE + Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import requests
import sys
import os
import json
import time
from datetime import datetime, timedelta

# ── 參數設定 ──────────────────────────────────────────────────────────────────
LOOKBACK_WEEKS    = 16    # 往回看幾週判斷「曾有大正乖離」
BIG_DEV_THRESHOLD = 0.15  # 大正乖離門檻 (15%)
NEAR_MA20_MAX     = 0.05  # 目前乖離上限 +5%
NEAR_MA20_MIN     = -0.03 # 目前乖離下限 -3%
MA20_SLOPE_WEEKS  = 4     # MA20 斜率計算週數
CHART_WEEKS       = 60    # 圖表顯示週數
BB_STD            = 2     # 布林通道標準差倍數

# ── 1. 取得台股上市清單 ───────────────────────────────────────────────────────
def get_twse_stock_list():
    print("取得台股上市清單...", flush=True)
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        stocks = []
        for item in data:
            code = item.get("Code", "")
            name = item.get("Name", "")
            if code.isdigit() and len(code) == 4:
                stocks.append({"code": code, "name": name})
        print(f"   共 {len(stocks)} 檔上市股票", flush=True)
        return stocks
    except Exception as e:
        print(f"   無法取得清單：{e}", flush=True)
        return []

# ── 2. 計算布林通道 ───────────────────────────────────────────────────────────
def calc_bollinger(close_series, window=20, std_mult=2):
    ma   = close_series.rolling(window).mean()
    std  = close_series.rolling(window).std(ddof=0)
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    return ma, upper, lower

# ── 3. 批次下載週K線並套用篩選條件 ───────────────────────────────────────────
def scan_stocks(stocks, batch_size=150):
    results = []
    total = len(stocks)

    for i in range(0, total, batch_size):
        batch = stocks[i : i + batch_size]
        tickers = [f"{s['code']}.TW" for s in batch]
        name_map = {f"{s['code']}.TW": s["name"] for s in batch}

        pct = (i + len(batch)) / total * 100
        print(f"   下載中 {i+1}~{i+len(batch)}/{total}（{pct:.0f}%）...", flush=True)

        try:
            raw = yf.download(
                tickers,
                period="2y",
                interval="1wk",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as e:
            print(f"   批次下載失敗：{e}", flush=True)
            continue

        for ticker in tickers:
            try:
                single = len(tickers) == 1
                close  = raw["Close"]          if single else raw[ticker]["Close"]
                open_  = raw["Open"]           if single else raw[ticker]["Open"]
                high   = raw["High"]           if single else raw[ticker]["High"]
                low    = raw["Low"]            if single else raw[ticker]["Low"]
                volume = raw["Volume"]         if single else raw[ticker]["Volume"]

                close  = close.dropna()
                if len(close) < 25:
                    continue

                # 篩選邏輯
                ma20 = close.rolling(20).mean()
                ma20_clean = ma20.dropna()
                if len(ma20_clean) < MA20_SLOPE_WEEKS + 2:
                    continue

                last_close = float(close.iloc[-1])
                last_ma20  = float(ma20_clean.iloc[-1])
                old_ma20   = float(ma20_clean.iloc[-1 - MA20_SLOPE_WEEKS])

                if not (last_ma20 > old_ma20):
                    continue

                recent_close = close.iloc[-LOOKBACK_WEEKS:]
                recent_ma20  = ma20.iloc[-LOOKBACK_WEEKS:]
                dev_series   = (recent_close - recent_ma20) / recent_ma20
                max_dev = float(dev_series.max())
                if max_dev < BIG_DEV_THRESHOLD:
                    continue

                current_dev = (last_close - last_ma20) / last_ma20
                if not (NEAR_MA20_MIN <= current_dev <= NEAR_MA20_MAX):
                    continue

                ma20_slope_pct   = (last_ma20 - old_ma20) / old_ma20 * 100
                peak_idx         = int(dev_series.argmax())
                weeks_since_peak = LOOKBACK_WEEKS - 1 - peak_idx

                # 計算布林通道
                _, bb_upper, bb_lower = calc_bollinger(close, window=20, std_mult=BB_STD)

                # 取最近 CHART_WEEKS 週的 OHLCV + 指標，對齊索引
                df = pd.DataFrame({
                    "open":     open_,
                    "high":     high,
                    "low":      low,
                    "close":    close,
                    "volume":   volume,
                    "ma20":     ma20,
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                }).dropna(subset=["close"]).tail(CHART_WEEKS)

                candles, vol_bars, ma_line, bbu_line, bbl_line = [], [], [], [], []
                for dt, row in df.iterrows():
                    t = dt.strftime("%Y-%m-%d")
                    candles.append({
                        "time":  t,
                        "open":  round(float(row["open"]),  2),
                        "high":  round(float(row["high"]),  2),
                        "low":   round(float(row["low"]),   2),
                        "close": round(float(row["close"]), 2),
                    })
                    vol_bars.append({
                        "time":  t,
                        "value": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                        "color": "#f87171" if float(row["close"]) >= float(row["open"]) else "#34d399",
                    })
                    if not pd.isna(row["ma20"]):
                        ma_line.append( {"time": t, "value": round(float(row["ma20"]),  2)})
                        bbu_line.append({"time": t, "value": round(float(row["bb_upper"]), 2)})
                        bbl_line.append({"time": t, "value": round(float(row["bb_lower"]), 2)})

                code = ticker.replace(".TW", "")
                results.append({
                    "code":             code,
                    "name":             name_map.get(ticker, ""),
                    "close":            round(last_close, 2),
                    "ma20":             round(last_ma20,  2),
                    "current_dev":      round(current_dev * 100, 2),
                    "peak_dev":         round(max_dev * 100,     2),
                    "weeks_since_peak": weeks_since_peak,
                    "ma20_slope_pct":   round(ma20_slope_pct,    2),
                    "chart": {
                        "candles":  candles,
                        "volume":   vol_bars,
                        "ma20":     ma_line,
                        "bb_upper": bbu_line,
                        "bb_lower": bbl_line,
                    },
                })

            except Exception:
                continue

    return results

# ── 4. 取得近期有資料的交易日清單 ─────────────────────────────────────────────
def get_trading_days(n=10):
    """找最近 n 個有三大法人資料的交易日"""
    headers = {"User-Agent": "Mozilla/5.0"}
    days = []
    d = datetime.now() - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:
            date_str = d.strftime("%Y%m%d")
            try:
                r = requests.get(
                    "https://www.twse.com.tw/rwd/zh/fund/T86",
                    params={"response": "json", "date": date_str, "selectType": "ALL"},
                    headers=headers, timeout=10
                )
                if r.json().get("total", 0) > 0:
                    days.append(date_str)
            except Exception:
                pass
        d -= timedelta(days=1)
    return days  # 由新到舊

# ── 5. 三大法人週買賣超（大戶）─────────────────────────────────────────────────
def fetch_institutional_weekly(trading_days):
    """
    取最近兩週共 10 個交易日的 T86 資料
    回傳 dict: code -> {"this_week": 本週合計萬股, "prev_week": 上週合計萬股}
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    this_week_days = trading_days[:5]   # 最新 5 天
    prev_week_days = trading_days[5:10] # 前一週 5 天

    def sum_week(days):
        totals = {}
        for date_str in days:
            try:
                r = requests.get(
                    "https://www.twse.com.tw/rwd/zh/fund/T86",
                    params={"response": "json", "date": date_str, "selectType": "ALL"},
                    headers=headers, timeout=10
                )
                data = r.json()
                for row in data.get("data", []):
                    code = row[0]
                    try:
                        net = int(row[18].replace(",", "").replace(" ", ""))
                        totals[code] = totals.get(code, 0) + net
                    except Exception:
                        pass
                time.sleep(0.3)
            except Exception:
                pass
        return totals

    print("   抓三大法人本週資料…", flush=True)
    this_w = sum_week(this_week_days)
    print("   抓三大法人上週資料…", flush=True)
    prev_w = sum_week(prev_week_days)

    result = {}
    all_codes = set(this_w) | set(prev_w)
    for code in all_codes:
        result[code] = {
            "this_week": round(this_w.get(code, 0) / 10000, 1),   # 萬股
            "prev_week": round(prev_w.get(code, 0) / 10000, 1),
        }
    return result

# ── 6. 融資餘額週增減（散戶代理指標）─────────────────────────────────────────
MARGIN_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "margin_cache.json")

def fetch_margin_current():
    """從 TWSE openapi 取得當日融資餘額（含前日）"""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(
            "https://openapi.twse.com.tw/v1/exchangeReport/MI_MARGN",
            headers=headers, timeout=15
        )
        data = r.json()
        result = {}
        for item in data:
            code = item.get("股票代號", "")
            if code.isdigit() and len(code) == 4:
                try:
                    result[code] = int(item.get("融資今日餘額", "0").replace(",", "") or "0")
                except Exception:
                    result[code] = 0
        return result
    except Exception as e:
        print(f"   ⚠ 融資資料抓取失敗：{e}", flush=True)
        return {}

def get_margin_weekly_change():
    """
    比對本次與上次快取的融資餘額
    回傳 dict: code -> 週增減（張，= 本週餘額 - 上週餘額）
    """
    # 讀上次快取
    prev_margin = {}
    if os.path.exists(MARGIN_CACHE_PATH):
        try:
            with open(MARGIN_CACHE_PATH, "r", encoding="utf-8") as f:
                prev_margin = json.load(f)
        except Exception:
            pass

    print("   抓融資餘額資料…", flush=True)
    curr_margin = fetch_margin_current()

    # 計算週增減（單位：張）
    weekly_chg = {}
    if prev_margin:
        for code in curr_margin:
            curr = curr_margin.get(code, 0)
            prev = prev_margin.get(code, curr)  # 若上週無資料則視為 0 增減
            weekly_chg[code] = curr - prev
    else:
        # 第一次執行，沒有歷史資料
        for code in curr_margin:
            weekly_chg[code] = None  # None = 無法計算

    # 儲存本次快取（供下週比對）
    try:
        with open(MARGIN_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(curr_margin, f, ensure_ascii=False)
    except Exception as e:
        print(f"   ⚠ 快取儲存失敗：{e}", flush=True)

    return weekly_chg

# ── 7. 產生 HTML ──────────────────────────────────────────────────────────────
def gen_html(results, scan_time):
    results.sort(key=lambda x: abs(x["current_dev"]))

    # 把圖表資料分離，只把 code→chart 的映射嵌入 JS
    chart_data_map = {r["code"]: r["chart"] for r in results}
    chart_data_json = json.dumps(chart_data_map, ensure_ascii=False)

    rows = ""
    for r in results:
        dev_color = "pos-green" if r["current_dev"] >= 0 else "neg-red"
        dev_str   = f"{r['current_dev']:+.2f}%"
        peak_str  = f"{r['peak_dev']:+.2f}%"
        slope_str = f"{'▲' if r['ma20_slope_pct'] > 0 else '▼'} {r['ma20_slope_pct']:.2f}%"

        # 三大法人週買賣超（大戶）
        inst = r.get("institutional") or {}
        tw = inst.get("this_week")
        if tw is not None:
            inst_color = "pos-red" if tw > 0 else ("neg-green" if tw < 0 else "neutral")
            inst_str   = f"{tw:+,.1f}"
        else:
            inst_color, inst_str = "neutral", "—"

        # 融資週增減（散戶）
        mc = r.get("margin_chg")
        if mc is None:
            margin_color, margin_str = "neutral", "初次"
        else:
            margin_color = "neg-green" if mc > 0 else ("pos-red" if mc < 0 else "neutral")
            margin_str   = f"{mc:+,}"

        rows += f"""
        <tr data-code="{r['code']}" data-name="{r['name']}">
          <td class="code">{r['code']}</td>
          <td>{r['name']}</td>
          <td class="num">{r['close']}</td>
          <td class="num">{r['ma20']}</td>
          <td class="num {dev_color}">{dev_str}</td>
          <td class="num peak">{peak_str}</td>
          <td class="num">{r['weeks_since_peak']} 週前</td>
          <td class="num up">{slope_str}</td>
          <td class="num {inst_color}">{inst_str}</td>
          <td class="num {margin_color}">{margin_str}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>台股週K MA20 回測清單｜正乖離修正</title>
  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #080e18;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.7;
      padding: 32px 20px;
    }}
    h1 {{ font-size: 1.6rem; color: #3b82f6; margin-bottom: 6px; }}
    .subtitle {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 24px; }}

    .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }}
    .stat-card {{
      background: #131f2e; border: 1px solid #2d3f55;
      border-radius: 10px; padding: 14px 22px; min-width: 140px;
    }}
    .stat-label {{ font-size: 0.78rem; color: #94a3b8; margin-bottom: 4px; }}
    .stat-value {{ font-size: 1.5rem; font-weight: 700; color: #3b82f6; }}

    .callout {{
      background: #0f2040; border: 1px solid rgba(59,130,246,0.3);
      border-radius: 8px; color: #93c5fd; font-size: 0.85rem;
      padding: 14px 18px; margin-bottom: 24px; line-height: 1.8;
    }}
    .callout strong {{ color: #60a5fa; }}

    .params {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 24px; }}
    .param-tag {{
      background: #0f2040; border: 1px solid rgba(59,130,246,0.25);
      border-radius: 20px; color: #7dd3fc; font-size: 0.78rem; padding: 4px 12px;
    }}

    .search-bar {{ margin-bottom: 16px; }}
    .search-bar input {{
      background: #131f2e; border: 1px solid #2d3f55; border-radius: 6px;
      color: #e2e8f0; font-size: 0.9rem; padding: 8px 14px; width: 280px; outline: none;
    }}
    .search-bar input:focus {{ border-color: #3b82f6; }}

    .table-wrap {{ overflow-x: auto; border-radius: 10px; border: 1px solid #2d3f55; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    thead tr {{ background: #0f2040; }}
    th {{
      padding: 12px 16px; text-align: left; color: #93c5fd;
      font-weight: 600; font-size: 0.82rem; white-space: nowrap;
      cursor: pointer; user-select: none;
    }}
    th:hover {{ color: #3b82f6; }}
    tbody tr {{
      border-top: 1px solid #1a2d42; transition: background 0.15s; cursor: pointer;
    }}
    tbody tr:hover {{ background: #162030; }}
    tbody tr.active {{ background: #0f2040; outline: 1px solid #3b82f6; }}
    td {{ padding: 10px 16px; white-space: nowrap; }}
    .code {{ color: #fb923c; font-weight: 600; font-family: monospace; font-size: 0.95rem; }}
    .num       {{ text-align: right; }}
    .pos-green  {{ color: #34d399; }}
    .neg-red    {{ color: #f87171; }}
    .pos-red    {{ color: #f87171; }}
    .neg-green  {{ color: #34d399; }}
    .neutral    {{ color: #94a3b8; }}
    .peak       {{ color: #fb923c; }}
    .up         {{ color: #3b82f6; }}

    /* ── 圖表 Modal ────────────────────────────────────── */
    .modal-overlay {{
      display: none;
      position: fixed; inset: 0;
      background: rgba(0,0,0,0.75);
      z-index: 1000;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }}
    .modal-overlay.open {{ display: flex; }}

    .modal {{
      background: #0d1826;
      border: 1px solid #2d3f55;
      border-radius: 14px;
      width: 100%;
      max-width: 900px;
      max-height: 90vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}

    .modal-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 20px;
      border-bottom: 1px solid #2d3f55;
      gap: 12px;
      flex-shrink: 0;
    }}
    .modal-title {{
      font-size: 1.05rem;
      font-weight: 600;
      color: #e2e8f0;
    }}
    .modal-title span {{ color: #fb923c; margin-right: 8px; font-family: monospace; }}

    .modal-legend {{
      display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.78rem;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 5px; color: #94a3b8; }}
    .legend-dot {{
      width: 20px; height: 3px; border-radius: 2px; flex-shrink: 0;
    }}

    .modal-close {{
      background: none; border: none; color: #94a3b8;
      font-size: 1.4rem; cursor: pointer; line-height: 1;
      padding: 2px 6px; border-radius: 4px;
      flex-shrink: 0;
    }}
    .modal-close:hover {{ color: #e2e8f0; background: #1e2d40; }}

    .chart-area {{
      padding: 0 4px 4px;
      flex: 1;
      min-height: 0;
    }}

    #main-chart  {{ width: 100%; height: 340px; }}
    #vol-chart   {{ width: 100%; height: 100px; margin-top: 2px; }}

    .modal-footer {{
      padding: 8px 20px;
      border-top: 1px solid #1a2d42;
      font-size: 0.75rem;
      color: #4a6080;
      display: flex;
      justify-content: space-between;
      flex-shrink: 0;
    }}
    .modal-footer a {{ color: #3b82f6; text-decoration: none; }}
    .modal-footer a:hover {{ text-decoration: underline; }}

    footer {{
      margin-top: 36px; padding-top: 16px;
      border-top: 1px solid #2d3f55; font-size: 0.78rem; color: #94a3b8;
    }}
  </style>
</head>
<body>

<h1>台股週K MA20 ｜正乖離修正回測清單</h1>
<p class="subtitle">資料來源：Yahoo Finance 週K線（還原權值）　｜　點擊列查看週K圖表</p>

<div class="stats">
  <div class="stat-card" style="border-color: rgba(59,130,246,0.5);">
    <div class="stat-label">最後更新</div>
    <div class="stat-value" style="font-size:0.95rem; color:#60a5fa; padding-top:4px; letter-spacing:0.02em;">{scan_time}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">符合條件股票</div>
    <div class="stat-value">{len(results)}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">大正乖離門檻</div>
    <div class="stat-value" style="color:#fb923c">≥ {int(BIG_DEV_THRESHOLD*100)}%</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">目前乖離範圍</div>
    <div class="stat-value" style="font-size:1rem; padding-top:6px; color:#34d399">
      {int(NEAR_MA20_MIN*100)}% ～ +{int(NEAR_MA20_MAX*100)}%
    </div>
  </div>
  <div class="stat-card">
    <div class="stat-label">回看週期</div>
    <div class="stat-value" style="color:#7dd3fc">{LOOKBACK_WEEKS} 週</div>
  </div>
</div>

<div class="callout">
  <strong>篩選邏輯說明</strong><br>
  ① <strong>週 MA20 向上（Moving Average 20-week）</strong>：當前 MA20 高於 {MA20_SLOPE_WEEKS} 週前的 MA20，確認均線處於上升趨勢<br>
  ② <strong>曾有大正乖離（Positive Deviation）</strong>：過去 {LOOKBACK_WEEKS} 週內，收盤價曾超越 MA20 達 <strong>+{int(BIG_DEV_THRESHOLD*100)}% 以上</strong><br>
  ③ <strong>修正到 MA20 附近（Pullback to MA）</strong>：目前乖離率介於 <strong>{int(NEAR_MA20_MIN*100)}% 至 +{int(NEAR_MA20_MAX*100)}%</strong><br>
  📊 點擊任一列可查看 <strong>週K線圖 + 布林通道（Bollinger Bands, {BB_STD}σ）</strong><br>
  👥 <strong>大戶週買賣（萬股）</strong>：三大法人（外資＋投信＋自營）本週合計買賣超，正值紅色＝買超，負值綠色＝賣超<br>
  💳 <strong>散戶融資週增減（張）</strong>：融資餘額與上週比較，正值紅色＝融資增加（散戶積極做多），負值綠色＝融資減少（健康去槓桿）
</div>

<div class="params">
  <span class="param-tag">回看 {LOOKBACK_WEEKS} 週</span>
  <span class="param-tag">大乖離門檻 +{int(BIG_DEV_THRESHOLD*100)}%</span>
  <span class="param-tag">MA20 附近 {int(NEAR_MA20_MIN*100)}%～+{int(NEAR_MA20_MAX*100)}%</span>
  <span class="param-tag">布林通道 MA20 ± {BB_STD}σ</span>
  <span class="param-tag">圖表顯示 {CHART_WEEKS} 週</span>
</div>

<div class="search-bar">
  <input type="text" id="search" placeholder="搜尋代碼或名稱..." oninput="filterTable()">
</div>

<div class="table-wrap">
  <table id="mainTable">
    <thead>
      <tr>
        <th onclick="sortTable(0)">股票代碼 ↕</th>
        <th onclick="sortTable(1)">名稱 ↕</th>
        <th onclick="sortTable(2)">收盤價 ↕</th>
        <th onclick="sortTable(3)">週MA20 ↕</th>
        <th onclick="sortTable(4)">目前乖離 ↕</th>
        <th onclick="sortTable(5)">高峰正乖離 ↕</th>
        <th onclick="sortTable(6)">距高點 ↕</th>
        <th onclick="sortTable(7)">MA20斜率 ↕</th>
        <th onclick="sortTable(8)" title="三大法人本週買賣超合計（萬股）紅=買超 綠=賣超">大戶週買賣(萬股) ↕</th>
        <th onclick="sortTable(9)" title="融資餘額週增減（張）綠=減少(健康) 紅=增加(注意)">散戶融資週增減(張) ↕</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</div>

<!-- ── 圖表 Modal ────────────────────────────────────────────────────────── -->
<div class="modal-overlay" id="modalOverlay" onclick="closeModalOnBg(event)">
  <div class="modal">
    <div class="modal-header">
      <div class="modal-title" id="modalTitle"></div>
      <div class="modal-legend">
        <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>MA20</div>
        <div class="legend-item"><div class="legend-dot" style="background:#fb923c; opacity:.8"></div>BB 上軌</div>
        <div class="legend-item"><div class="legend-dot" style="background:#a78bfa; opacity:.8"></div>BB 下軌</div>
      </div>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="chart-area">
      <div id="main-chart"></div>
      <div id="vol-chart"></div>
    </div>
    <div class="modal-footer">
      <span>週K線（還原權值）｜布林通道 MA20 ± {BB_STD}σ</span>
      <a id="yahooLink" href="#" target="_blank" rel="noopener">在 Yahoo Finance 查看 ↗</a>
    </div>
  </div>
</div>

<footer>
  資料來源：<strong>Yahoo Finance</strong> 週K線（auto_adjust 還原權值）、台灣證券交易所（TWSE）上市股票清單<br>
  圖表：<strong>TradingView Lightweight Charts</strong>｜注意：資料僅供參考，不構成任何投資建議。
</footer>

<script>
// ── 圖表資料 ─────────────────────────────────────────────────────────────────
const CHART_DATA = {chart_data_json};

// ── 排序 ──────────────────────────────────────────────────────────────────────
let sortDir = {{}};
function sortTable(col) {{
  const tb   = document.querySelector("#mainTable tbody");
  const rows = Array.from(tb.querySelectorAll("tr")).filter(r => r.style.display !== "none");
  const dir  = (sortDir[col] = !sortDir[col]);
  rows.sort((a, b) => {{
    let av = a.cells[col].innerText.replace(/[▲▼週前%+,]/g,"").trim();
    let bv = b.cells[col].innerText.replace(/[▲▼週前%+,]/g,"").trim();
    let an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return dir ? an - bn : bn - an;
    return dir ? av.localeCompare(bv,"zh") : bv.localeCompare(av,"zh");
  }});
  rows.forEach(r => tb.appendChild(r));
}}

// ── 搜尋 ──────────────────────────────────────────────────────────────────────
function filterTable() {{
  const q = document.getElementById("search").value.toLowerCase();
  document.querySelectorAll("#mainTable tbody tr").forEach(row => {{
    row.style.display = row.innerText.toLowerCase().includes(q) ? "" : "none";
  }});
}}

// ── 圖表 ──────────────────────────────────────────────────────────────────────
let mainChart = null, volChart = null;

function openChart(code, name) {{
  const data = CHART_DATA[code];
  if (!data) return;

  // 設定標題
  document.getElementById("modalTitle").innerHTML = `<span>${{code}}</span>${{name}}`;
  document.getElementById("yahooLink").href = `https://finance.yahoo.com/chart/${{code}}.TW`;
  document.getElementById("modalOverlay").classList.add("open");

  // 清除舊圖
  document.getElementById("main-chart").innerHTML = "";
  document.getElementById("vol-chart").innerHTML  = "";

  const commonOpts = {{
    layout:     {{ background: {{ color: "#0d1826" }}, textColor: "#94a3b8" }},
    grid:       {{ vertLines: {{ color: "#1a2d42" }}, horzLines: {{ color: "#1a2d42" }} }},
    timeScale:  {{ borderColor: "#2d3f55", timeVisible: true }},
    rightPriceScale: {{ borderColor: "#2d3f55" }},
    crosshair:  {{ mode: 1 }},
    handleScroll: true,
    handleScale:  true,
  }};

  // ── 主圖（K線 + BB + MA20）
  mainChart = LightweightCharts.createChart(
    document.getElementById("main-chart"),
    {{ ...commonOpts, height: 340 }}
  );

  const candleSeries = mainChart.addCandlestickSeries({{
    upColor:        "#f87171",
    downColor:      "#34d399",
    borderUpColor:  "#f87171",
    borderDownColor:"#34d399",
    wickUpColor:    "#f87171",
    wickDownColor:  "#34d399",
  }});
  candleSeries.setData(data.candles);

  // BB 填充區域（上下軌之間的帶狀）
  const bbUpperSeries = mainChart.addLineSeries({{
    color: "rgba(251,146,60,0.7)",
    lineWidth: 1,
    lineStyle: 2,       // dashed
    priceLineVisible: false,
    lastValueVisible: false,
    title: "BB上軌",
  }});
  bbUpperSeries.setData(data.bb_upper);

  const bbLowerSeries = mainChart.addLineSeries({{
    color: "rgba(167,139,250,0.7)",
    lineWidth: 1,
    lineStyle: 2,
    priceLineVisible: false,
    lastValueVisible: false,
    title: "BB下軌",
  }});
  bbLowerSeries.setData(data.bb_lower);

  const ma20Series = mainChart.addLineSeries({{
    color: "#3b82f6",
    lineWidth: 2,
    priceLineVisible: false,
    lastValueVisible: true,
    title: "MA20",
  }});
  ma20Series.setData(data.ma20);

  mainChart.timeScale().fitContent();

  // ── 成交量圖
  volChart = LightweightCharts.createChart(
    document.getElementById("vol-chart"),
    {{
      ...commonOpts,
      height: 100,
      rightPriceScale: {{ borderColor: "#2d3f55", scaleMargins: {{ top: 0.1, bottom: 0 }} }},
    }}
  );

  const volSeries = volChart.addHistogramSeries({{
    priceFormat: {{ type: "volume" }},
    priceScaleId: "right",
  }});
  volSeries.setData(data.volume);
  volChart.timeScale().fitContent();

  // 同步十字線
  mainChart.subscribeCrosshairMove(param => {{
    if (param.time) volChart.setCrosshairPosition(0, param.time, volSeries);
    else volChart.clearCrosshairPosition();
  }});
  volChart.subscribeCrosshairMove(param => {{
    if (param.time) mainChart.setCrosshairPosition(0, param.time, candleSeries);
    else mainChart.clearCrosshairPosition();
  }});
}}

function closeModal() {{
  document.getElementById("modalOverlay").classList.remove("open");
  document.querySelectorAll("#mainTable tbody tr").forEach(r => r.classList.remove("active"));
  if (mainChart) {{ mainChart.remove(); mainChart = null; }}
  if (volChart)  {{ volChart.remove();  volChart  = null; }}
}}

function closeModalOnBg(e) {{
  if (e.target === document.getElementById("modalOverlay")) closeModal();
}}

// 點擊列開圖
document.querySelectorAll("#mainTable tbody tr").forEach(row => {{
  row.addEventListener("click", () => {{
    document.querySelectorAll("#mainTable tbody tr").forEach(r => r.classList.remove("active"));
    row.classList.add("active");
    openChart(row.dataset.code, row.dataset.name);
  }});
}});

// ESC 關閉
document.addEventListener("keydown", e => {{ if (e.key === "Escape") closeModal(); }});
</script>
</body>
</html>
"""
    return html

# ── 主程式 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("  台股週K MA20 掃描工具 v4（大戶散戶週變化版）", flush=True)
    print("=" * 60, flush=True)

    # ① 取得上市股票清單
    stocks = get_twse_stock_list()
    if not stocks:
        print("無法取得股票清單，請確認網路連線後重試")
        sys.exit(1)

    # ② 掃描週K線篩選
    print(f"開始掃描（共 {len(stocks)} 檔）...", flush=True)
    results = scan_stocks(stocks, batch_size=150)
    print(f"初步篩選：{len(results)} 檔符合條件", flush=True)

    # ③ 取得三大法人週資料（大戶）
    print("取得大戶/散戶週變化資料...", flush=True)
    trading_days = get_trading_days(n=10)
    print(f"   交易日：{trading_days}", flush=True)
    institutional = fetch_institutional_weekly(trading_days)

    # ④ 取得融資週增減（散戶）
    margin_weekly = get_margin_weekly_change()

    # ⑤ 合併到 results
    for r in results:
        code = r["code"]
        r["institutional"] = institutional.get(code)
        r["margin_chg"]    = margin_weekly.get(code)

    # ⑥ 產生 HTML
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = gen_html(results, scan_time)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"完成！共 {len(results)} 檔，輸出：{out_path}", flush=True)
