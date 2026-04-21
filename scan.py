"""
台股週K線過濾工具 v2
條件：
  1. 週 MA20 向上（近期 MA20 斜率為正）
  2. 過去 N 週內曾有大正乖離（股價顯著高於 MA20）
  3. 目前股價已修正到 MA20 附近（乖離率很小）

資料來源：TWSE 上市股票清單 + Yahoo Finance 週K線
"""

import yfinance as yf
import pandas as pd
import requests
import sys
import os
from datetime import datetime

# ── 參數設定 ──────────────────────────────────────────────────────────────────
LOOKBACK_WEEKS    = 16    # 往回看幾週判斷「曾有大正乖離」
BIG_DEV_THRESHOLD = 0.15  # 大正乖離門檻：收盤/MA20 超過此值算「乖離過大」(15%)
NEAR_MA20_MAX     = 0.05  # 修正後接近 MA20：乖離率絕對值 <= 5%
NEAR_MA20_MIN     = -0.03 # 下限（允許略低於 MA20 但不能太深）
MA20_SLOPE_WEEKS  = 4     # 用幾週前的 MA20 計算斜率（向上判斷）

# ── 1. 取得台股上市清單（TWSE OpenAPI）────────────────────────────────────────
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

# ── 2. 批次下載週K線並套用篩選條件 ───────────────────────────────────────────
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
                if len(tickers) == 1:
                    close = raw["Close"]
                else:
                    close = raw[ticker]["Close"]

                close = close.dropna()
                if len(close) < 25:
                    continue

                ma20 = close.rolling(20).mean()
                ma20_clean = ma20.dropna()
                if len(ma20_clean) < MA20_SLOPE_WEEKS + 2:
                    continue

                last_close = float(close.iloc[-1])
                last_ma20  = float(ma20_clean.iloc[-1])
                old_ma20   = float(ma20_clean.iloc[-1 - MA20_SLOPE_WEEKS])

                # 條件 1：MA20 向上
                if not (last_ma20 > old_ma20):
                    continue

                # 條件 2：過去 LOOKBACK_WEEKS 週內曾有大正乖離
                recent_close = close.iloc[-LOOKBACK_WEEKS:]
                recent_ma20  = ma20.iloc[-LOOKBACK_WEEKS:]
                dev_series   = (recent_close - recent_ma20) / recent_ma20
                max_dev = float(dev_series.max())
                if max_dev < BIG_DEV_THRESHOLD:
                    continue

                # 條件 3：目前已修正到 MA20 附近
                current_dev = (last_close - last_ma20) / last_ma20
                if not (NEAR_MA20_MIN <= current_dev <= NEAR_MA20_MAX):
                    continue

                ma20_slope_pct = (last_ma20 - old_ma20) / old_ma20 * 100
                peak_idx = int(dev_series.argmax())
                weeks_since_peak = LOOKBACK_WEEKS - 1 - peak_idx

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
                })

            except Exception:
                continue

    return results

# ── 3. 產生 HTML ──────────────────────────────────────────────────────────────
def gen_html(results, scan_time):
    results.sort(key=lambda x: abs(x["current_dev"]))

    rows = ""
    for r in results:
        dev_color = "pos-green" if r["current_dev"] >= 0 else "neg-red"
        dev_str   = f"{r['current_dev']:+.2f}%"
        peak_str  = f"{r['peak_dev']:+.2f}%"
        slope_str = f"{'▲' if r['ma20_slope_pct'] > 0 else '▼'} {r['ma20_slope_pct']:.2f}%"
        yahoo_url = f"https://finance.yahoo.com/chart/{r['code']}.TW"
        rows += f"""
        <tr>
          <td class="code"><a href="{yahoo_url}" target="_blank" rel="noopener">{r['code']}</a></td>
          <td>{r['name']}</td>
          <td class="num">{r['close']}</td>
          <td class="num">{r['ma20']}</td>
          <td class="num {dev_color}">{dev_str}</td>
          <td class="num peak">{peak_str}</td>
          <td class="num">{r['weeks_since_peak']} 週前</td>
          <td class="num up">{slope_str}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>台股週K MA20 回測清單｜正乖離修正</title>
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
    tbody tr {{ border-top: 1px solid #1a2d42; transition: background 0.15s; }}
    tbody tr:hover {{ background: #131f2e; }}
    td {{ padding: 10px 16px; white-space: nowrap; }}
    .code a {{
      color: #fb923c; font-weight: 600; font-family: monospace;
      font-size: 0.95rem; text-decoration: none;
    }}
    .code a:hover {{ color: #fdba74; text-decoration: underline; }}
    .num       {{ text-align: right; }}
    .pos-green {{ color: #34d399; }}
    .neg-red   {{ color: #f87171; }}
    .peak      {{ color: #fb923c; }}
    .up        {{ color: #3b82f6; }}
    footer {{
      margin-top: 36px; padding-top: 16px;
      border-top: 1px solid #2d3f55; font-size: 0.78rem; color: #94a3b8;
    }}
  </style>
</head>
<body>

<h1>台股週K MA20 ｜正乖離修正回測清單</h1>
<p class="subtitle">掃描時間：{scan_time}　｜　資料來源：Yahoo Finance 週K線（還原權值）</p>

<div class="stats">
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
  ② <strong>曾有大正乖離（Positive Deviation）</strong>：過去 {LOOKBACK_WEEKS} 週內，收盤價曾超越 MA20 達 <strong>+{int(BIG_DEV_THRESHOLD*100)}% 以上</strong>，代表曾過熱<br>
  ③ <strong>修正到 MA20 附近（Pullback to MA）</strong>：目前乖離率介於 <strong>{int(NEAR_MA20_MIN*100)}% 至 +{int(NEAR_MA20_MAX*100)}%</strong>，股價已回落至均線支撐區
</div>

<div class="params">
  <span class="param-tag">回看 {LOOKBACK_WEEKS} 週</span>
  <span class="param-tag">大乖離門檻 +{int(BIG_DEV_THRESHOLD*100)}%</span>
  <span class="param-tag">MA20 附近 {int(NEAR_MA20_MIN*100)}%～+{int(NEAR_MA20_MAX*100)}%</span>
  <span class="param-tag">MA20 斜率參考 {MA20_SLOPE_WEEKS} 週</span>
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
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</div>

<footer>
  資料來源：<strong>Yahoo Finance</strong> 週K線（auto_adjust 還原權值）、台灣證券交易所（TWSE）上市股票清單<br>
  注意：資料僅供參考，不構成任何投資建議。投資有風險，操作前請自行評估。
</footer>

<script>
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
  function filterTable() {{
    const q = document.getElementById("search").value.toLowerCase();
    document.querySelectorAll("#mainTable tbody tr").forEach(row => {{
      const txt = row.innerText.toLowerCase();
      row.style.display = txt.includes(q) ? "" : "none";
    }});
  }}
</script>
</body>
</html>
"""
    return html

# ── 主程式 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("  台股週K MA20 掃描工具 v2（正乖離修正版）", flush=True)
    print("=" * 60, flush=True)

    stocks = get_twse_stock_list()
    if not stocks:
        print("無法取得股票清單，請確認網路連線後重試")
        sys.exit(1)

    print(f"開始掃描（共 {len(stocks)} 檔）...", flush=True)
    results = scan_stocks(stocks, batch_size=150)

    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = gen_html(results, scan_time)

    # 輸出到腳本同目錄的 index.html
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"掃描完成！共找到 {len(results)} 檔符合條件的股票", flush=True)
    print(f"輸出：{out_path}", flush=True)
