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
  东财 fund_etf_hist_min_em / index_zh_a_hist_min_em 只返回最近约 5 个交易日且极易被
  push2 限流，**已不再作为主数据源**。fetch_5min 现在按
      本地 cache/*.pkl|csv  >  baostock  >  新浪  >  东财
  的顺序多源拼合（均免 token、不依赖东财），合并结果回写 cache/ 供下次复用。

  各源覆盖情况（实测）：
    baostock  159915 覆盖 2024-01~2024-09 与 2026-01/03~10，但不支持指数分钟线；
              2025 全年及 2026-02 无数据（上游服务端即缺失，不是参数问题）。
    新浪      只给最近约 1500 根（≈31 个交易日），无分页参数，ETF 与指数都可用。
    → 两者拼合后仍有缺口。缺口不会被静默填补：程序会打印缺失的交易日区间，
      并且**不再自动用合成数据顶替真实区间**（见 SYNTH_FALLBACK）。
      要拿到连续历史，请每天运行一次本脚本，让新浪的近 31 日窗口逐日滚进 cache/。
"""
import itertools
import json
import os

import akshare as ak
import numpy as np
import pandas as pd

# ======================================================================
# 一、参数
# ======================================================================

SYMBOL = "600584"  # 159915 创业板 ETF，600584 长江电力（测试用）
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
MARKET_INDEX = "sh000300"      # 大盘代理，可改成任意指数或个股：sh000300 / sz000001 / sh600519
MARKET_VWAP_FILTER = True      # 大盘跌破当日 VWAP 时，禁止高抛（避免卖在下跌初段）


def _is_index(symbol):
    """判断代码是指数还是个股票。

    上证指数段：000xxx（上证指数/深证成指等，sh000300、sz399001…）
    深证指数段：399xxx
    个股：60xxxx(沪主板) / 68xxxx(科创板) / 00xxxx(深主板) / 30xxxx(创业板) / 8xxxxx(北交所)
    """
    code = symbol[2:] if symbol[:2] in ("sh", "sz") else symbol
    return code.startswith("399") or (code.startswith("000") and symbol.startswith("sh"))


MARKET_IS_INDEX = _is_index(MARKET_INDEX)
MARKET_TAG = "指数" if MARKET_IS_INDEX else "个股"

ENABLE_GRID = True
GRID_RSI_SELL = [68, 72, 76]
GRID_VWAP_ATR = [0.8, 1.0, 1.2]
GRID_BUYBACK_DROP = [0.005, 0.007, 0.010]

# 真实 5 分钟数据一根都取不到时，是否用日线合成 bar 顶替跑通流程。
# 合成数据只验证代码路径，结果不可用于决策——默认关闭，避免误读收益曲线。
SYNTH_FALLBACK = os.environ.get("PTRADE_SYNTH_FALLBACK", "0") == "1"

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
    """日线：优先新浪源（限流宽松），失败再退回东财。均带本地缓存。

    index=True 时按指数取（沪深300 等）；个股/ETF 自动按代码段判断，无需手动切换。
    缓存文件名带 kind，避免"指数 000001"和"个股 000001"互相覆盖。
    """
    kind = "idx" if index else ("stock" if _is_index(symbol) else "etf")
    tag = f"{kind}_{symbol}_daily.pkl"
    path = _cache_path(tag)
    if os.path.exists(path):
        df = pd.read_pickle(path)
    else:
        raw, date_col = None, None
        sina_symbol = _sina_prefix(symbol)
        if index:
            try:
                raw = ak.stock_zh_index_daily(symbol=sina_symbol)
                date_col = "date"
            except Exception as e:
                print(f"  新浪指数接口失败（{type(e).__name__}），改用东财")
        else:
            # 新浪 fund_etf_hist_sina 对个股也能返回正确行情，ETF 与个股共用
            try:
                raw = ak.fund_etf_hist_sina(symbol=sina_symbol)
                date_col = "date"
            except Exception as e:
                print(f"  新浪行情接口失败（{type(e).__name__}），改用东财")
        if raw is None:
            if index:
                raw = ak.stock_zh_index_daily_em(symbol=symbol)
            elif _is_index(symbol):
                # 个股：按个股口径取，避免落到 ETF 接口
                raw = ak.stock_zh_a_daily(symbol=sina_symbol, adjust="qfq")
            else:
                raw = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                          start_date=start.replace("-", ""),
                                          end_date=end.replace("-", ""), adjust="qfq")
            date_col = "日期" if "日期" in raw.columns else "date"
        df = _normalize(raw, date_col)
        df.to_pickle(path)
    return _in_period(df, start, end).reset_index(drop=True)


def _sina_prefix(symbol):
    """补全交易所前缀。6 开头→sh，其余→sz；已有 sh/sz 前缀则原样返回。

    不带前缀时不能一律补 sz：600584/688xxx 是上交所，补错前缀会取到别的标的。
    """
    if symbol[:2] in ("sh", "sz"):
        return symbol
    return ("sh" if symbol.startswith(("6", "9")) else "sz") + symbol


def _bao_code(symbol):
    """159915 -> sz.159915；sh000300 -> sh.000300。"""
    s = _sina_prefix(symbol)
    return f"{s[:2]}.{s[2:]}"


def fetch_sina_5min(symbol, end, datalen=1500):
    """新浪分钟线：免 token、不限流，单次最多 1500 根（约 31 个交易日），只能取到最新一段。

    这里是增量采集的关键——每天跑一次就能把历史一段段攒进 cache。
    """
    import requests

    code = _sina_prefix(symbol)
    url = ("https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_x=/CN_MarketDataService.getKLineData"
           f"?symbol={code}&scale=5&ma=no&datalen={datalen}")
    r = requests.get(url, timeout=30,
                     headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"})
    r.raise_for_status()
    text = r.text
    payload = text[text.find("(") + 1: text.rfind(")")]
    rows = json.loads(payload)
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["day"])
    out = _normalize(df, "datetime")
    return out[out["datetime"] < pd.Timestamp(end).normalize() + pd.Timedelta(days=1)] \
        .reset_index(drop=True)


def fetch_baostock_5min(symbol, start, end, index=False):
    """baostock 分钟线：免 token、不限流，覆盖 2024-01~2024-09 与 2026 年。

    个股与 ETF 都支持；**指数不支持**（query 会静默返回 0 行）。
    判定依据是代码本身：个股/ETF 走这里，指数由新浪兜底。
    """
    import baostock as bs

    if index or _is_index(symbol):
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    logged_in = bs.login().error_code == "0"
    try:
        # 个股/ETF 用后复权（adjustflag=2），与日线 qfq 口径一致
        rs = bs.query_history_k_data_plus(
            _bao_code(symbol), "date,time,open,high,low,close,volume",
            start_date=start, end_date=end, frequency="5", adjustflag="2")
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
    finally:
        if logged_in:
            bs.logout()
    if not rows:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["date", "time", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["time"].str[:14], format="%Y%m%d%H%M%S")
    # 09:30 那根是集合竞价残根，不是真实的 5 分钟 bar
    df = df[df["datetime"].dt.strftime("%H:%M") != "09:30"]
    return _normalize(df, "datetime")


def _in_period(df, start, end):
    """按 [start, end] 全日闭区间裁剪（end 当天 15:00 的 bar 也应包含）。"""
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
    return df[(df["datetime"] >= lo) & (df["datetime"] < hi)]


def _is_etf(symbol):
    """ETF 代码段：15xxxx/16xxxx(深) 与 50xxxx/51xxxx/56xxxx/58xxxx(沪)。"""
    code = symbol[2:] if symbol[:2] in ("sh", "sz") else symbol
    return code.startswith(("15", "16", "50", "51", "56", "58"))


def fetch_akshare_5min(symbol, index=False):
    """东财分钟线：只给最近约 5 个交易日，且极易被 push2 限流——放在最后兜底。"""
    code = symbol[2:] if symbol[:2] in ("sh", "sz") else symbol
    if index or _is_index(symbol):
        raw = ak.index_zh_a_hist_min_em(symbol=code, period="5")
    elif _is_etf(symbol):
        raw = ak.fund_etf_hist_min_em(symbol=code, period="5", adjust="qfq")
    else:
        raw = ak.stock_zh_a_hist_min_em(symbol=code, period="5", adjust="qfq")
    return _normalize(raw, "时间" if "时间" in raw.columns else "datetime")


def _kind(symbol, index=False):
    """缓存分类前缀：指数 / 个股 / ETF。000001 既要区分上证指数和个股，靠它隔离缓存。"""
    if index or _is_index(symbol):
        return "idx"
    return "stock" if not _is_etf(symbol) else "etf"


def fetch_5min(symbol=SYMBOL, start=START_DATE, end=END_DATE, index=False):
    """取 5 分钟线，按以下顺序取数并用 (symbol, datetime) 合并去重。

    1. 本地 cache/*_5min.csv、*.pkl（用户自备或历次累积，优先级最高）
    2. baostock  —— 无需 token，覆盖 2024-01~2024-09 与 2026 全年（个股/ETF 有效，指数无效）
    3. 新浪      —— 无需 token，始终给最近约 31 个交易日（个股/ETF/指数都有效）
    4. 东财      —— 只给最近约 5 个交易日且极易限流，仅作最后兜底

    多源拼合后仍缺的区间会明确告警，而不是静默用合成数据。
    """
    tag = f"{_kind(symbol, index)}_{symbol}_5min"
    csv_path = _cache_path(tag + ".csv")
    pkl_path = _cache_path(tag + ".pkl")

    sources = []

    def add(name, loader):
        try:
            part = loader()
        except Exception as e:
            msg = str(e).split("\n")[0]
            print(f"  [数据源] {name} 失败（{type(e).__name__}: {msg[:60]}）")
            return
        part = _in_period(part, start, end)
        if part.empty:
            print(f"  [数据源] {name} 该区间无数据")
            return
        print(f"  [数据源] {name}: {len(part)} 根 "
              f"{part['datetime'].min()} ~ {part['datetime'].max()}")
        sources.append(part)

    if os.path.exists(csv_path):
        raw_csv = pd.read_csv(csv_path, nrows=5)
        add(f"本地CSV {os.path.basename(csv_path)}",
            lambda: _normalize(pd.read_csv(csv_path),
                               "时间" if "时间" in raw_csv.columns else "datetime"))
    if os.path.exists(pkl_path):
        add(f"本地缓存 {os.path.basename(pkl_path)}", lambda: pd.read_pickle(pkl_path))

    add("baostock", lambda: fetch_baostock_5min(symbol, start, end, index))
    add("新浪", lambda: fetch_sina_5min(symbol, end))
    # 东财极易限流且只有最近几天，上面两个源已覆盖最近交易日时直接跳过
    recent_days = {d for d in (pd.concat(sources, ignore_index=True)
                               ["datetime"].dt.date.unique() if sources else [])}
    newest = max(recent_days) if recent_days else None
    if newest is not None and pd.Timestamp(end).normalize() - pd.Timestamp(newest) <= pd.Timedelta("5D"):
        print("  [数据源] 东财已跳过（已有源覆盖到最新交易日，避免触发 push2 限流）")
    else:
        add("东财", lambda: fetch_akshare_5min(symbol, index))

    if not sources:
        raise RuntimeError("所有 5 分钟数据源均不可用")

    # 合并优先级：本地缓存 > baostock > 新浪 > 东财
    # （东财可能返回被截断/口径不同的数据，只能最后考虑）
    df = (pd.concat(sources, ignore_index=True)
          .drop_duplicates(subset=["datetime"], keep="first")
          .sort_values("datetime")
          .reset_index(drop=True))

    # 合并结果回写缓存（含历次增量，下次可直接复用），CSV 同步一份便于人工查看
    df.to_pickle(pkl_path)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    _report_coverage(df, fetch_daily(symbol, start, end, index), symbol)
    return df


def _report_coverage(df5, df_daily, symbol):
    """打印 5 分钟数据对日线交易日的覆盖情况，并指出缺口落在哪些连续区间。"""
    covered = set(df5["datetime"].dt.date)
    expected = sorted(set(df_daily["datetime"].dt.date))
    missing = [d for d in expected if d not in covered]
    span = f"{df5['datetime'].min()} ~ {df5['datetime'].max()}"
    if not missing:
        print(f"  [数据源] 合并后 {len(df5)} 根，{span}，覆盖全部 {len(expected)} 个交易日")
        return

    # 把缺失日合并成连续区间，方便直接看出哪些月份没数据
    gaps, run = [], [missing[0]]
    for prev, cur in zip(missing, missing[1:]):
        if (pd.Timestamp(cur) - pd.Timestamp(prev)).days <= 7:
            run.append(cur)
        else:
            gaps.append((run[0], run[-1]))
            run = [cur]
    gaps.append((run[0], run[-1]))
    shown = "、".join(f"{a}~{b}" for a, b in gaps[:4])
    more = "" if len(gaps) <= 4 else f" 等 {len(gaps)} 段"

    print(f"  [数据源] 合并后 {len(df5)} 根，{span}，"
          f"覆盖 {len(expected) - len(missing)}/{len(expected)} 个交易日")
    print(f"  [数据源] 无 5 分钟数据的交易日：{shown}{more}（共 {len(missing)} 天），回测会跳过这些天")
    print("  [数据源] 免费源拿不到这些区间；要补齐请把本脚本加进每日定时任务，"
          "让新浪近 31 个交易日的窗口逐日沉淀进 cache/")




def fetch_5min_cached(symbol=SYMBOL, index=False):
    """只读本地 5 分钟缓存，不发任何网络请求（网格搜索等重复调用用它）。"""
    tag = f"{_kind(symbol, index)}_{symbol}_5min"
    pkl_path = _cache_path(tag + ".pkl")
    if not os.path.exists(pkl_path):
        return None
    return pd.read_pickle(pkl_path)


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
    # 大盘代理的 "跌破 VWAP" 由调用方合并进来
    return df


def base_ratio_signal(df_daily, df_index=None):
    """日线收盘后计算次日的底仓比例 B。返回 DataFrame: datetime, B, reason。"""
    d = df_daily.copy()
    d["MA20"] = d["close"].rolling(20).mean()
    d["MA60"] = d["close"].rolling(60).mean()
    d["RET"] = d["close"].pct_change()
    d["VOL_MA20"] = d["volume"].rolling(20).mean()

    # 大盘暴跌判断（用大盘代理，缺数据时退回标的自身）
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
    df_mkt   : 大盘代理 5 分钟（可选，用于 VWAP 过滤；指数或个股均可）
    """
    b_map = dict(zip(b_signal["datetime"].dt.date, b_signal["B"]))
    reason_map = dict(zip(b_signal["datetime"].dt.date, b_signal["reason"]))

    if df_mkt is None and MARKET_VWAP_FILTER and not quiet:
        print(f"[提示] 未启用 VWAP 过滤（缺 {MARKET_INDEX} 数据），高抛条件少 1 个维度")

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
            print(f"[提示] {MARKET_INDEX} 时间戳与 {SYMBOL} 无交集，VWAP 过滤已忽略\n")
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
    print(f"  注: 实际触发是 ④a 或 {MARKET_INDEX}跌破VWAP；缺大盘数据时只剩 ④a，信号会偏少\n")

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
        print("新浪日线接口偶发失败，请稍后重试。")
        return

    # 5 分钟数据：多源拼合（本地缓存 / baostock / 新浪 / 东财）
    try:
        df5 = fetch_5min()
        if len(df5) < 100:
            raise RuntimeError(f"仅取到 {len(df5)} 根 5 分钟数据")
    except Exception as e:
        print(f"5 分钟真实数据全部不可用（{type(e).__name__}: {e}）")
        if not SYNTH_FALLBACK:
            print("→ 已中止。合成 bar 只能验证代码路径、不能作为结论；")
            print("  如仍要用合成数据跑通流程，设置环境变量 PTRADE_SYNTH_FALLBACK=1")
            return
        print("→ PTRADE_SYNTH_FALLBACK=1：改用日线合成的 5 分钟 bar，"
              "**结果不可用于实盘决策**\n")
        df5 = build_synthetic_5min(df_daily, _cache_path("synth_5min.pkl"))

    df5 = prepare_5min(df5)

    try:
        df_idx_d = fetch_daily(MARKET_INDEX, index=MARKET_IS_INDEX)
    except Exception as e:
        print(f"大盘代理 {MARKET_INDEX} 日线获取失败（{type(e).__name__}），"
              f"改用 {SYMBOL} 自身作为大盘代理\n")
        df_idx_d = None

    # 大盘代理 5 分钟：指数分钟线 baostock 不支持，只能靠新浪 + 本地缓存；
    # 个股/ETF 代理则可吃到 baostock 的长历史。缺失不影响回测主流程。
    df_idx_5 = fetch_5min_cached(MARKET_INDEX, index=MARKET_IS_INDEX)
    try:
        fresh = fetch_5min(MARKET_INDEX, index=MARKET_IS_INDEX)
        df_idx_5 = fresh if df_idx_5 is None else pd.concat([df_idx_5, fresh], ignore_index=True)
        df_idx_5 = (df_idx_5.drop_duplicates(subset=["datetime"], keep="first")
                    .sort_values("datetime").reset_index(drop=True))
    except Exception as e:
        print(f"大盘代理 {MARKET_INDEX} 5 分钟获取失败（{type(e).__name__}），"
              f"{'沿用本地缓存' if df_idx_5 is not None else '本次不启用 VWAP 过滤'}\n")
    if df_idx_5 is not None and len(df_idx_5) < 100:
        df_idx_5 = None

    print(f"日线 {len(df_daily)} 行  5分钟 {len(df5)} 行"
          f"  大盘代理 {MARKET_INDEX}({MARKET_TAG}) 5分钟 {0 if df_idx_5 is None else len(df_idx_5)} 行\n")

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
