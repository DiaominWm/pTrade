# pip install akshare pandas numpy openpyxl
"""159915 创业板 ETF —— 底仓日线择时 + 日内正T 回测。

与 backtest.py 是两套完全独立的策略，互不影响。

数据：
  日线  akshare.fund_etf_hist_em          （底仓择时 B、ATR、MA20/MA60）
  5分钟 akshare.fund_etf_hist_min_em      （T+0 高抛低吸）

成交假设：
  1. 底仓：日线收盘后定 B，次日按 5 分钟第一根开盘价调仓（先卖后买，多退少补）；
  2. 做T：5 分钟收盘确认信号，下一根开盘成交；T 份额记在"待买回"里，
     与底仓合并计入总持仓，14:50 未买回则强制回补；
  3. 成本：佣金 0.02%（双边）+ 滑点 0.03% + 印花税 0（ETF 免），往返 ≈ 0.05%；
  4. ETF 无 T+1 限制，当日买入可当日卖出（正T 得以成立）。

数据限制（务必先读）：
  东财 fund_etf_hist_min_em / index_zh_a_hist_min_em 只返回最近约 5 个交易日，
  无法覆盖 2024-01-01 起的完整区间。取不到真实 5 分钟数据时会用日线合成的
  5 分钟 bar 跑通流程，此时结果**不可用于实盘决策**，仅验证代码正确性。
  要得到真实结论，请用本地 5 分钟 CSV 或换付费数据源替换 fetch_5min。
"""
import itertools
import os

import akshare as ak
import numpy as np
import pandas as pd

# ======================================================================
# 一、参数
# ======================================================================

SYMBOL = "159915"
START_DATE = "2024-01-01"
END_DATE = "2026-10-30"

CAPITAL = 500000

# ---------- 交易成本 ----------
COMMISSION = 0.0002            # 佣金，双边各一次
SLIPPAGE_PCT = 0.0003          # 滑点，按成交价百分比
STAMP_TAX = 0.0                # ETF 免印花税

LOT = 100                      # 最小交易单位（份）

# ---------- 底仓比例 B（日线，收盘定、次日生效） ----------
B_DROP_TH = -0.03              # 大盘单日跌 <= -3%
B_DROP_VOL_MULT = 1.5          # 且成交量 >= 20 日均量的 1.5 倍
B_RULES = {
    "暴跌放量": 0.0,
    "跌破MA60": 0.0,
    "MA20下方": 0.30,           # 收盘 < MA20 但 >= MA60
    "MA20上方转暖": 0.60,       # 收盘 >= MA20 但 MA20 < MA60
    "多头排列": 1.00,           # 收盘 >= MA20 且 MA20 >= MA60
}
# 盘中止损触发的降档（长期生效，直到下次日线择时给出更低/更高的比例）
B_CAP_AFTER_DD15 = 0.50        # 单标的回撤 > 15% → 底仓减半，B 上限 50%
B_CAP_AFTER_DD20 = 0.30        # 组合回撤 > 20% → B 强制 <= 30%

# ---------- 正T ----------
T_RATIO = 0.30                 # 单次做T量 = 底仓 * 30%
T_MAX_PER_DAY = 2              # 每日最多 2 次（早盘 1 + 尾盘 1）
T_WINDOWS = [("09:45", "11:00"), ("14:00", "14:45")]

RSI_N = 14
RSI_SELL_TH = 72
RSI_BUY_TH = 40
ATR_N = 20
BARS_PER_DAY_EST = 48          # 5 分钟一天 48 根，用于把 ATR 还原到单根尺度
VWAP_ATR_MULT = 1.0

VOL_DIVERGE = 0.8              # 卖出：近 3 根均量 / 前 20 根均量 < 0.8
VOL_SHRINK = 0.7               # 买回：当前量 / 前 5 根均量 < 0.7
BUYBACK_DROP = 0.007           # 买回：价格 <= 卖出价 * (1 - 0.7%)

FORCE_BUYBACK_TIME = "14:50"
STOP_LOSS_PCT = 0.015          # 价格 > 卖出价 * 1.015 → 立即买回认亏
DAILY_STOP_LOSS = -0.03        # 单日亏损 > 3% → 当日停手，次日 B 降档
LOSE_STREAK_PAUSE = 3          # 连亏 3 次 → 暂停做T 5 日
PAUSE_DAYS = 5

# ---------- 市场择时（"大盘"代理） ----------
MARKET_INDEX = "sh000300"      # 沪深300，用于暴跌判断与 5 分钟跌破 VWAP 过滤
MARKET_VWAP_FILTER = True      # 沪深300 跌破当日 VWAP 时，禁止高抛（避免卖在下跌初段）

ENABLE_GRID = True
GRID_RSI_SELL = [68, 72, 76]
GRID_VWAP_ATR = [0.8, 1.0, 1.2]
GRID_BUYBACK_DROP = [0.005, 0.007, 0.010]

REPORT_FILE = "backtest_etf_t_report.xlsx"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
# 缓存保留原始 5 分钟数据；信号列在 prepare_5min 里计算，方便随时改参数重算

# ======================================================================
# 二、数据获取（带本地缓存，避免反复请求被限流）
# ======================================================================

def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def _normalize(df, date_col):
    """统一成 datetime/open/high/low/close/volume，按时间升序。"""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df[date_col])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    return df.sort_values("datetime").reset_index(drop=True)


def fetch_daily(symbol=SYMBOL, start=START_DATE, end=END_DATE, index=False):
    """日线：优先新浪源（限流宽松），失败再退回东财。均带本地缓存。"""
    tag = f"{'idx' if index else 'etf'}_{symbol}_daily.pkl"
    path = _cache_path(tag)
    if os.path.exists(path):
        df = pd.read_pickle(path)
    else:
        raw, date_col = None, None
        sina_symbol = symbol if symbol[:2] in ("sh", "sz") else "sz" + symbol
        if index:
            try:
                raw = ak.stock_zh_index_daily(symbol=sina_symbol)
                date_col = "date"
            except Exception as e:
                print(f"  新浪指数接口失败（{type(e).__name__}），改用东财")
        else:
            try:
                raw = ak.fund_etf_hist_sina(symbol=sina_symbol)
                date_col = "date"
            except Exception as e:
                print(f"  新浪 ETF 接口失败（{type(e).__name__}），改用东财")
        if raw is None:
            if index:
                raw = ak.stock_zh_index_daily_em(symbol=symbol)
            else:
                raw = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                          start_date=start.replace("-", ""),
                                          end_date=end.replace("-", ""), adjust="qfq")
            date_col = "日期" if "日期" in raw.columns else "date"
        df = _normalize(raw, date_col)
        df.to_pickle(path)
    mask = (df["datetime"] >= pd.Timestamp(start)) & (df["datetime"] <= pd.Timestamp(end))
    return df[mask].reset_index(drop=True)


def fetch_5min(symbol=SYMBOL, start=START_DATE, end=END_DATE, index=False):
    """取 5 分钟线。

    先读本地 cache/ 下的 CSV（用户自备），没有再请求东财。
    东财 fund_etf_hist_min_em 只返回最近约 5 个交易日，无法覆盖长区间历史。
    本地 CSV 需包含列：时间(或datetime), open, high, low, close, volume。
    """
    tag = f"{'idx' if index else 'etf'}_{symbol}_5min"
    csv_path = _cache_path(tag + ".csv")
    pkl_path = _cache_path(tag + ".pkl")
    if os.path.exists(csv_path):
        raw = pd.read_csv(csv_path)
        date_col = "时间" if "时间" in raw.columns else "datetime"
        df = _normalize(raw, date_col)
        print(f"  5 分钟数据来自本地文件: {csv_path}（{len(df)} 行）")
    elif os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
    else:
        if index:
            raw = ak.index_zh_a_hist_min_em(symbol=symbol.replace("sh", "").replace("sz", ""),
                                            period="5")
        else:
            raw = ak.fund_etf_hist_min_em(symbol=symbol, period="5", adjust="qfq")
        df = _normalize(raw, "时间" if "时间" in raw.columns else "datetime")
        df.to_pickle(pkl_path)
    mask = (df["datetime"] >= pd.Timestamp(start)) & (df["datetime"] <= pd.Timestamp(end))
    return df[mask].reset_index(drop=True)


def build_synthetic_5min(df_daily, path):
    """用日线 OHLC 合成 5 分钟 bar，仅用于在无 5 分钟历史时跑通流程（结果不可用于决策）。

    ⚠ 合成数据必须单调递增地穿越当日区间，若像原来那样"来回震荡"，
      会生成真实市场罕见的日内形态，反而制造出虚假信号。
    """
    if os.path.exists(path):
        return pd.read_pickle(path)
    rng = np.random.default_rng(0)
    times = ["09:35", "09:45", "10:00", "10:30", "11:00", "11:30",
             "13:30", "14:00", "14:30", "14:50"]
    w = np.array([0.08, 0.10, 0.12, 0.15, 0.12, 0.10, 0.10, 0.10, 0.10, 0.13])
    k = len(times)
    rows = []
    for _, d in df_daily.iterrows():
        o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
        swing = h - l
        noise = rng.normal(0, 0.02, k) * swing          # 2% 日振幅量级的抖动
        path_px = np.clip(np.linspace(o, c, k) + noise, l, h)
        for i, t in enumerate(times):
            px = float(path_px[i])
            rows.append({
                "datetime": pd.Timestamp(f"{d['datetime'].date()} {t}"),
                "open": px, "high": px, "low": px, "close": px,
                "volume": max(v * w[i] / w.sum(), 1.0),
            })
    df = pd.DataFrame(rows)
    df.to_pickle(path)
    return df

# ======================================================================
# 三、指标
# ======================================================================

def rsi(close, n=RSI_N):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(high, low, close, n=ATR_N):
    """日线口径 ATR 在 5 分钟上会累积全天幅度，先除以每日 bar 数还原到单根尺度。"""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean() / BARS_PER_DAY_EST


def macd_hist(close, fast=12, slow=26, signal=9):
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, (dif - dea) * 2


def daily_vwap_proxy(df5):
    """当日累计 VWAP，按典型价（(H+L+C)/3）加权，模拟真实成交流。"""
    tp = (df5["high"] + df5["low"] + df5["close"]) / 3.0
    pv = (tp * df5["volume"]).groupby(df5["date"]).cumsum()
    cum_v = df5["volume"].groupby(df5["date"]).cumsum().replace(0, np.nan)
    return pv / cum_v


def prepare_5min(df5):
    df = df5.copy()
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    df["RSI"] = rsi(df["close"], RSI_N)
    df["ATR"] = atr(df["high"], df["low"], df["close"], ATR_N)
    df["VWAP"] = daily_vwap_proxy(df)
    df["VOL_MA3_PREV"] = df["volume"].shift(1).rolling(3).mean()
    df["VOL_MA20_PREV"] = df["volume"].shift(1).rolling(20).mean()
    df["VOL_MA5_PREV"] = df["volume"].shift(1).rolling(5).mean()
    df["VOL_DIVERGE"] = df["VOL_MA3_PREV"] / df["VOL_MA20_PREV"]
    dif, dea, _ = macd_hist(df["close"])
    df["DIF"], df["DEA"] = dif, dea
    # 沪深300 的 "跌破 VWAP" 由调用方合并进来
    return df


def base_ratio_signal(df_daily, df_index=None):
    """日线收盘后计算次日的底仓比例 B。返回 DataFrame: datetime, B, reason。"""
    d = df_daily.copy()
    d["MA20"] = d["close"].rolling(20).mean()
    d["MA60"] = d["close"].rolling(60).mean()
    d["RET"] = d["close"].pct_change()
    d["VOL_MA20"] = d["volume"].rolling(20).mean()

    # 大盘暴跌判断（用指数，缺数据时退回 ETF 自身）
    ref = df_index[["datetime", "close", "volume"]].copy() if df_index is not None else \
        d[["datetime", "close", "volume"]].copy()
    ref["MA20"] = ref["close"].rolling(20).mean()
    ref["VOL_MA20"] = ref["volume"].rolling(20).mean()
    ref["RET"] = ref["close"].pct_change()
    ref["CRASH"] = (ref["RET"] <= B_DROP_TH) & (ref["volume"] >= ref["VOL_MA20"] * B_DROP_VOL_MULT)
    crash_map = dict(zip(ref["datetime"].dt.date, ref["CRASH"]))

    rows = []
    for _, r in d.iterrows():
        if pd.isna(r["MA60"]):
            b, reason = 0.0, "数据不足"
        elif crash_map.get(r["datetime"].date(), False):
            b, reason = B_RULES["暴跌放量"], "暴跌放量"
        elif r["close"] < r["MA60"]:
            b, reason = B_RULES["跌破MA60"], "跌破MA60"
        elif r["close"] < r["MA20"]:
            b, reason = B_RULES["MA20下方"], "MA20下方"
        elif r["MA20"] < r["MA60"]:
            b, reason = B_RULES["MA20上方转暖"], "MA20上方转暖"
        else:
            b, reason = B_RULES["多头排列"], "多头排列"
        rows.append({"datetime": r["datetime"], "B": b, "reason": reason})
    return pd.DataFrame(rows)


# ======================================================================
# 四、回测核心
# ======================================================================

def _close_t_round(t_trades, buy_time, buy_price, pnl, sell_cost, reason):
    """回填最近一笔未平仓的做T记录。"""
    if t_trades and t_trades[-1]["buy_time"] is None:
        t_trades[-1].update({
            "buy_time": buy_time, "buy_price": buy_price, "pnl": pnl,
            "ret_pct": pnl / sell_cost * 100 if sell_cost else 0.0, "reason": reason,
        })


def _in_window(time_str, windows):
    return any(lo <= time_str <= hi for lo, hi in windows)


def _sell_t(qty, fill, core_shares, cash):
    """做T卖出：返回 (core_shares, cash, 卖出净额)。"""
    gross = qty * fill
    fee = gross * (COMMISSION + STAMP_TAX)
    return core_shares - qty, cash + gross - fee, gross - fee


def _buy_back(t_shares, fill, core_shares, cash):
    """做T买回：返回 (core_shares, cash, 买入总成本)。"""
    gross = t_shares * fill
    fee = gross * COMMISSION
    return core_shares + t_shares, cash - gross - fee, gross + fee


def run_backtest(df5, b_signal, rsi_sell=RSI_SELL_TH, vwap_atr=VWAP_ATR_MULT,
                 buyback_drop=BUYBACK_DROP, df_mkt=None, quiet=False):
    """底仓择时 + 正T 回测。

    df5      : 已 prepare_5min 的 5 分钟数据
    b_signal : base_ratio_signal 的输出（日线，收盘定）
    df_mkt   : 沪深300 5 分钟（可选，用于 VWAP 过滤）
    """
    b_map = dict(zip(b_signal["datetime"].dt.date, b_signal["B"]))
    reason_map = dict(zip(b_signal["datetime"].dt.date, b_signal["reason"]))

    if df_mkt is None and MARKET_VWAP_FILTER and not quiet:
        print("[提示] 未启用 VWAP 过滤（缺沪深300 数据），高抛条件少 1 个维度")

    mkt_vwap = {}
    if df_mkt is not None and MARKET_VWAP_FILTER:
        m = df_mkt.copy()
        m["date"] = m["datetime"].dt.date
        m["time"] = m["datetime"].dt.strftime("%H:%M")
        tp = (m["high"] + m["low"] + m["close"]) / 3.0
        cum_pv = (tp * m["volume"]).groupby(m["date"]).cumsum()
        cum_v = m["volume"].groupby(m["date"]).cumsum().replace(0, np.nan)
        m["VWAP"] = cum_pv / cum_v
        m["BELOW"] = m["close"] < m["VWAP"]
        mkt_vwap = {(r.date, r.time): r.BELOW for r in m.itertuples()}
        has_mkt = set(zip(df5["date"], df5["time"])) & set(mkt_vwap)
        if not has_mkt:
            print("[提示] 沪深300 时间戳与 ETF 无交集，VWAP 过滤已忽略\n")
            mkt_vwap = {}

    cash = float(CAPITAL)
    core_shares = 0                 # 底仓
    t_shares = 0                    # 做T 卖出后待买回的份额
    t_sell_price = 0.0
    t_sell_cost = 0.0
    t_count_today = 0
    t_pause_until = None
    lose_streak = 0
    day_start_equity = CAPITAL
    day_blocked = False
    peak_equity = CAPITAL
    b_cap = 1.0

    trades = []                     # 所有成交
    t_trades = []                   # 仅做T 的完整来回（用于统计）
    equity_rows = []
    current_date = None

    for i in range(1, len(df5)):
        row = df5.iloc[i]
        prev = df5.iloc[i - 1]
        bar_date, bar_time = row["date"], row["time"]
        price = row["open"]

        # ---------- 新的一天 ----------
        if bar_date != current_date:
            current_date = bar_date
            t_count_today = 0
            day_blocked = False
            day_start_equity = cash + (core_shares + t_shares) * price

            # 底仓调仓：用当天第一根开盘价，B 来自上一个交易日收盘。
            # 先卖后买，且卖出量不超过实际底仓，避免与做T份额重复计数。
            target_b = b_map.get(bar_date)
            if target_b is not None:
                target_b = min(target_b, b_cap)
                total_shares = core_shares + t_shares
                equity = cash + total_shares * price
                target_shares = int(equity * target_b / (price * (1 - COMMISSION)) / LOT) * LOT
                # 可动用资源 = 全部现金 + 当前底仓；做T卖出中待买回的份额不能重复使用
                max_holdable = int(cash / (price * (1 + COMMISSION)) / LOT) * LOT + core_shares
                target_shares = min(target_shares, max_holdable)
                if target_shares < total_shares:
                    delta = min(total_shares - target_shares, core_shares)
                    if delta > 0:
                        gross = delta * price
                        fee = gross * (COMMISSION + STAMP_TAX)
                        cash += gross - fee
                        core_shares -= delta
                        trades.append({"datetime": row["datetime"], "action": "减底仓",
                                       "price": price, "shares": delta,
                                       "reason": reason_map.get(bar_date, ""), "pnl": 0.0})
                elif target_shares > total_shares:
                    delta = min(target_shares - total_shares,
                                int(cash / (price * (1 + COMMISSION)) / LOT) * LOT)
                    if delta > 0:
                        cost = delta * price
                        fee = cost * COMMISSION
                        cash -= cost + fee
                        core_shares += delta
                        trades.append({"datetime": row["datetime"], "action": "加底仓",
                                       "price": price, "shares": delta,
                                       "reason": reason_map.get(bar_date, ""), "pnl": 0.0})

        # ---------- 13:57 之后禁止新开T；14:50 强制买回 ----------
        if t_shares > 0 and bar_time >= FORCE_BUYBACK_TIME:
            fill = price * (1 + SLIPPAGE_PCT)
            qty = min(t_shares, int(cash / (fill * (1 + COMMISSION)) / LOT) * LOT)
            if qty <= 0:                    # 现金不足（底仓切换占用了资金），按剩余份额补足基准仓
                core_shares += t_shares
                t_trades[-1].update({"buy_time": row["datetime"], "buy_price": np.nan,
                                     "pnl": -t_sell_cost, "ret_pct": -100.0,
                                     "reason": "现金不足无法买回"})
                t_shares = 0
                t_sell_price = 0.0
                t_sell_cost = 0.0
                qty = 0
            if qty > 0:
                core_shares, cash, buy_cost = _buy_back(qty, fill, core_shares, cash)
                pnl = t_sell_cost * (qty / t_shares) - buy_cost
                trades.append({"datetime": row["datetime"], "action": "买回",
                               "price": fill, "shares": qty,
                               "reason": "14:50强制买回", "pnl": pnl})
                _close_t_round(t_trades, row["datetime"], fill, pnl,
                               t_sell_cost * (qty / t_shares), "14:50强制买回")
                lose_streak = lose_streak + 1 if pnl < 0 else 0
                if lose_streak >= LOSE_STREAK_PAUSE:
                    t_pause_until = bar_date + pd.Timedelta(days=PAUSE_DAYS)
                    lose_streak = 0
                t_shares = 0
                t_sell_price = 0.0
                t_sell_cost = 0.0

        can_do_t = (
            t_shares == 0
            and not day_blocked
            and t_count_today < T_MAX_PER_DAY
            and (t_pause_until is None or bar_date > t_pause_until)
            and _in_window(bar_time, T_WINDOWS)
            and core_shares >= LOT
        )

        # ---------- 卖出（高抛） ----------
        if can_do_t:
            mkt_below = mkt_vwap.get((bar_date, bar_time), False)
            vol_diverge_ok = (not pd.isna(prev["VOL_DIVERGE"])) and prev["VOL_DIVERGE"] < VOL_DIVERGE
            price_ok = (
                (not pd.isna(prev["VWAP"]))
                and (not pd.isna(prev["ATR"]))
                and prev["close"] > prev["VWAP"] + vwap_atr * prev["ATR"]
            )
            rsi_ok = (not pd.isna(prev["RSI"])) and prev["RSI"] > rsi_sell
            macd_dead = (prev["DIF"] < prev["DEA"]) and (df5.iloc[i - 2]["DIF"] >= df5.iloc[i - 2]["DEA"])
            trigger = macd_dead or mkt_below

            if rsi_ok and price_ok and vol_diverge_ok and trigger:
                sell_qty = int(core_shares * T_RATIO / LOT) * LOT
                if sell_qty >= LOT:
                    fill = price * (1 - SLIPPAGE_PCT)
                    core_shares, cash, net = _sell_t(sell_qty, fill, core_shares, cash)
                    t_shares = sell_qty
                    t_sell_price = fill
                    t_sell_cost = net
                    t_count_today += 1
                    trades.append({"datetime": row["datetime"], "action": "做T卖出",
                                   "price": fill, "shares": sell_qty,
                                   "reason": "RSI超买+高于VWAP+量能背离", "pnl": 0.0})
                    t_trades.append({"sell_time": row["datetime"], "buy_time": None,
                                     "shares": sell_qty, "sell_price": fill,
                                     "buy_price": 0.0, "pnl": 0.0, "ret_pct": 0.0,
                                     "reason": ""})

        # ---------- 买回（低吸） ----------
        elif t_shares > 0:
            price_hit = (prev["close"] <= t_sell_price * (1 - buyback_drop)) or \
                        (not pd.isna(prev["VWAP"]) and prev["close"] <= prev["VWAP"])
            confirm = (
                (not pd.isna(prev["RSI"])) and prev["RSI"] < RSI_BUY_TH
                and (not pd.isna(prev["VOL_MA5_PREV"]))
                and prev["volume"] / prev["VOL_MA5_PREV"] < VOL_SHRINK
            )
            stop = prev["close"] > t_sell_price * (1 + STOP_LOSS_PCT)
            if stop or (price_hit and confirm):
                fill = price * (1 + SLIPPAGE_PCT)
                # 现金不足以原量买回时按可买数量部分回补
                sold_net = t_sell_cost
                affordable = int(cash / (fill * (1 + COMMISSION)) / LOT) * LOT
                qty = min(t_shares, affordable)
                if qty > 0:
                    core_shares, cash, buy_cost = _buy_back(qty, fill, core_shares, cash)
                    pnl = sold_net * (qty / t_shares) - buy_cost
                    trades.append({"datetime": row["datetime"], "action": "买回",
                                   "price": fill, "shares": qty,
                                   "reason": "止损认亏" if stop else "低吸买回", "pnl": pnl})
                    _close_t_round(t_trades, row["datetime"], fill, pnl,
                                   sold_net * (qty / t_shares), "止损" if stop else "低吸")
                    lose_streak = lose_streak + 1 if pnl < 0 else 0
                    if lose_streak >= LOSE_STREAK_PAUSE:
                        t_pause_until = bar_date + pd.Timedelta(days=PAUSE_DAYS)
                        lose_streak = 0
                else:
                    _close_t_round(t_trades, row["datetime"], fill, sold_net - t_shares * fill,
                                   sold_net, "现金不足无法买回")
                t_shares = 0
                t_sell_price = 0.0
                t_sell_cost = 0.0

        # ---------- 当日亏损超限：停手 ----------
        equity = cash + (core_shares + t_shares) * row["close"]
        peak_equity = max(peak_equity, equity)

        if equity <= day_start_equity * (1 + DAILY_STOP_LOSS):
            day_blocked = True          # 当日停手

        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd > 0.20:
            b_cap = 0.30
        elif dd > 0.15:
            b_cap = min(b_cap, B_CAP_AFTER_DD15)
        elif dd < 0.05:
            b_cap = 1.0                 # 回撤修复后恢复满档（B 仍由日线择时决定）

        equity_rows.append({"datetime": row["datetime"], "equity": equity,
                            "B_target": b_map.get(bar_date, np.nan), "B_cap": b_cap,
                            "core_shares": core_shares + t_shares, "cash": cash})

    return _performance(equity_rows, trades, t_trades)


def _performance(equity_rows, trades, t_trades):
    if not equity_rows:
        return {"final": CAPITAL, "profit_pct": 0.0, "max_dd": 0.0, "sharpe": 0.0,
                "t_rounds": 0, "t_win_rate": 0.0, "t_avg_ret": 0.0, "t_pnl": 0.0,
                "equity": pd.DataFrame(), "trades": trades, "t_trades": t_trades}

    eq = pd.DataFrame(equity_rows)
    series = eq["equity"]
    daily = series.groupby(pd.to_datetime(eq["datetime"]).dt.date).last()
    ret = daily.pct_change().dropna()
    sharpe = np.sqrt(252) * ret.mean() / ret.std() if len(ret) > 1 and ret.std() > 0 else 0.0
    max_dd = ((series.cummax() - series) / series.cummax()).max()

    done = [t for t in t_trades if t["buy_time"] is not None]
    wins = [t for t in done if t["pnl"] > 0]

    return {
        "final": float(series.iloc[-1]),
        "profit_pct": (float(series.iloc[-1]) - CAPITAL) / CAPITAL * 100,
        "max_dd": float(max_dd) * 100,
        "sharpe": float(sharpe),
        "t_rounds": len(done),
        "t_win_rate": len(wins) / len(done) * 100 if done else 0.0,
        "t_avg_ret": float(np.mean([t["ret_pct"] for t in done])) if done else 0.0,
        "t_pnl": float(sum(t["pnl"] for t in done)),
        "equity": eq,
        "trades": trades,
        "t_trades": t_trades,
    }


# ======================================================================
# 五、输出
# ======================================================================

def print_perf(title, r):
    print(f"===== {title} =====")
    print(f"期末权益: {r['final']:,.2f}  总收益: {r['profit_pct']:.2f}%  "
          f"最大回撤: {r['max_dd']:.2f}%  夏普: {r['sharpe']:.2f}")
    print(f"做T来回: {r['t_rounds']} 次  胜率: {r['t_win_rate']:.1f}%  "
          f"平均收益率: {r['t_avg_ret']:.3f}%  累计盈亏: {r['t_pnl']:,.2f}")
    print()


def export_excel(result, grid_df=None, filename=REPORT_FILE):
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("未安装 openpyxl，无法导出 Excel。请执行: pip install openpyxl")
        return

    sheets = {"绩效汇总": pd.DataFrame([{
        "期末权益": round(result["final"], 2),
        "总收益%": round(result["profit_pct"], 2),
        "最大回撤%": round(result["max_dd"], 2),
        "夏普": round(result["sharpe"], 3),
        "做T来回": result["t_rounds"],
        "做T胜率%": round(result["t_win_rate"], 2),
        "做T均收益%": round(result["t_avg_ret"], 3),
        "做T累计盈亏": round(result["t_pnl"], 2),
    }])}

    if not result["equity"].empty:
        sheets["权益曲线"] = result["equity"]
    if result["trades"]:
        sheets["成交明细"] = pd.DataFrame(result["trades"])
    if result["t_trades"]:
        sheets["做T明细"] = pd.DataFrame(result["t_trades"])
    if grid_df is not None and not grid_df.empty:
        sheets["参数网格"] = grid_df

    out_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(out_dir, exist_ok=True)
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"报告已导出: {os.path.abspath(filename)}")


def grid_search(df5, b_signal, df_mkt=None):
    rows = []
    for rsi_sell, vwap_atr, bb in itertools.product(
        GRID_RSI_SELL, GRID_VWAP_ATR, GRID_BUYBACK_DROP
    ):
        r = run_backtest(df5, b_signal, rsi_sell=rsi_sell,
                         vwap_atr=vwap_atr, buyback_drop=bb, df_mkt=df_mkt, quiet=True)
        rows.append({
            "rsi_sell": rsi_sell, "vwap_atr": vwap_atr, "buyback_drop": bb,
            "profit_pct": r["profit_pct"], "max_dd": r["max_dd"],
            "sharpe": r["sharpe"], "t_rounds": r["t_rounds"],
            "t_win_rate": r["t_win_rate"], "t_avg_ret": r["t_avg_ret"],
            "t_pnl": r["t_pnl"],
        })
    return pd.DataFrame(rows).sort_values("t_pnl", ascending=False)


# ======================================================================
# 五、信号诊断（不参与回测，用来定位"为什么没交易"）
# ======================================================================

def diagnose_signals(df5, rsi_sell=RSI_SELL_TH, vwap_atr=VWAP_ATR_MULT):
    """逐条统计卖出/买回条件的满足次数，方便判断哪个条件过严。"""
    prev = df5.shift(1)
    prev2 = df5.shift(2)
    win = df5["time"].apply(lambda t: _in_window(t, T_WINDOWS))

    conds = {
        "在可做T时间窗": win,
        "① RSI>阈值": prev["RSI"] > rsi_sell,
        "② 价>VWAP+N×ATR": prev["close"] > prev["VWAP"] + vwap_atr * prev["ATR"],
        "③ 量能背离<0.8": prev["VOL_DIVERGE"] < VOL_DIVERGE,
        "④a 5分钟MACD死叉": (prev["DIF"] < prev["DEA"]) & (prev2["DIF"] >= prev2["DEA"]),
    }
    print("===== 卖出条件逐项命中（窗口内，共 "
          f"{int(win.sum())} 根） =====")
    for name, s in conds.items():
        print(f"  {name:22s}: {int((s & win).sum()):5d}")
    all_sell = conds["① RSI>阈值"] & conds["② 价>VWAP+N×ATR"] & conds["③ 量能背离<0.8"] \
        & conds["④a 5分钟MACD死叉"]
    print(f"  {'①&②&③&④a 全部满足':22s}: {int((all_sell & win).sum()):5d}")
    print("  注: 实际触发是 ④a 或 沪深300跌破VWAP；缺指数数据时只剩 ④a，信号会偏少\n")

    print("\n===== 买回条件逐项命中（当日全部 bar） =====")
    buy_price = (prev["close"] <= prev["VWAP"])
    buy_confirm = (prev["RSI"] < RSI_BUY_TH) & (
        prev["volume"] / prev["VOL_MA5_PREV"] < VOL_SHRINK)
    print(f"  {'价<=VWAP':22s}: {int(buy_price.sum()):5d}")
    print(f"  {'RSI<40 且 缩量<0.7':22s}: {int(buy_confirm.sum()):5d}")
    print(f"  {'两者同时':22s}: {int((buy_price & buy_confirm).sum()):5d}\n")


# ======================================================================
# 六、主流程
# ======================================================================

def main():
    print(f"标的: {SYMBOL}  区间: {START_DATE} ~ {END_DATE}  资金: {CAPITAL:,.0f}\n")

    try:
        df_daily = fetch_daily()
        if df_daily.empty:
            raise RuntimeError("日线为空")
    except Exception as e:
        print(f"日线数据获取失败：{type(e).__name__}: {e}")
        print("akshare 免费接口有频率限制（push2 限流），请等待几分钟后重试，")
        print("或先运行一次把数据缓存到 cache/ 目录，再离线回测。")
        return

    # 5 分钟数据：东财接口只给最近几个交易日，历史区间取不到时用合成数据跑通流程
    try:
        df5 = fetch_5min()
        if len(df5) < 100:
            raise RuntimeError(f"仅取到 {len(df5)} 根 5 分钟数据")
        print("5 分钟数据来源: akshare（真实行情）")
    except Exception as e:
        print(f"5 分钟真实数据不可用（{type(e).__name__}: {e}）")
        print("→ 改用日线合成的 5 分钟 bar 跑通流程，**结果不可用于实盘决策**\n")
        df5 = build_synthetic_5min(df_daily, _cache_path("synth_5min.pkl"))

    df5 = prepare_5min(df5)

    try:
        df_idx_d = fetch_daily(MARKET_INDEX, index=True)
        df_idx_5 = fetch_5min(MARKET_INDEX, index=True)
        if len(df_idx_5) < 100:
            df_idx_5 = None
    except Exception as e:
        print(f"沪深300 数据获取失败（{type(e).__name__}），改用 ETF 自身作为大盘代理\n")
        df_idx_d, df_idx_5 = None, None

    print(f"日线 {len(df_daily)} 行  5分钟 {len(df5)} 行\n")

    b_signal = base_ratio_signal(df_daily, df_idx_d)
    result = run_backtest(df5, b_signal, df_mkt=df_idx_5)
    print_perf("底仓+正T 回测", result)
    diagnose_signals(df5)

    grid_df = grid_search(df5, b_signal, df_idx_5) if ENABLE_GRID else None
    if grid_df is not None and not grid_df.empty:
        print("===== 参数网格（按做T累计盈亏排序） =====")
        print(grid_df.head(10).to_string(index=False))
        print()

    export_excel(result, grid_df)


if __name__ == "__main__":
    main()
