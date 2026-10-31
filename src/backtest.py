# pip install baostock "pandas<2.1" "numpy<1.25" openpyxl
import baostock as bs
import pandas as pd
import numpy as np
import itertools
import os

# ======================================================================
# 一、回测参数（全部集中在这里，按需修改）
# ======================================================================

# ---------- 标的与数据区间 ----------
SYMBOL = "300115"
START_DATE = "2024-09-01"
END_DATE = "2026-10-28"        # 分钟线建议避开最近 1~2 个交易日，避免数据未入库
ADJUST_FLAG = "2"              # 1=后复权，2=前复权，3=不复权

# 频率："d"=日线，"60"/"30"/"15"/"5"=分钟线
# 注意：baostock 不支持 1 分钟线（"1"），如需 1 分钟请换数据源
FREQUENCY = "15"

# ---------- 资金与仓位 ----------
CAPITAL = 300000
POSITION = 30000

# ---------- 交易成本 ----------
COMMISSION = 0.0001
STAMP_TAX = 0.001

# ---------- 阶段开关（随时启用/屏蔽） ----------
ENABLE_BASIC = True            # 是否跑基础回测
ENABLE_GRID = True             # 是否跑网格搜索
ENABLE_OOS = True              # 是否跑样本外验证

# ---------- 训练/测试切分 ----------
TRAIN_END = "2025-06-01"
MIN_TRADES = 60                # 网格搜索过滤交易过少的参数

# ---------- Excel 报告 ----------
REPORT_FILE = "backtest_report.xlsx"

# baostock 支持的频率集合
BAOSTOCK_SUPPORTED_FREQ = {"d", "w", "m", "5", "15", "30", "60"}

# ======================================================================
# 二、按频率自动适配的默认参数
# ======================================================================

def get_period_config(freq):
    """根据频率返回指标参数、止损止盈、网格范围、年化因子、滑点。"""
    if freq == "d":
        return {
            "kdj_n": 11, "kdj_m1": 2, "kdj_m2": 2,
            "macd_fast": 10, "macd_slow": 20, "macd_signal": 7,
            "rsi_period": 6, "rsi_buy_th": 50,
            "stop_loss": -0.03, "take_profit": 0.06,
            "use_trend_filter": True, "ma_trend": 60,
            "slippage_pct": 0.0002,
            "annual_factor": 252,
            "grid_stop_loss": [-0.02, -0.03, -0.05],
            "grid_take_profit": [0.04, 0.06, 0.08],
            "grid_kdj_n": [9, 11, 14],
            "grid_rsi_th": [30, 40, 50],
        }
    elif freq == "60":
        return {
            "kdj_n": 9, "kdj_m1": 2, "kdj_m2": 2,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "rsi_period": 6, "rsi_buy_th": 40,
            "stop_loss": -0.02, "take_profit": 0.04,
            "use_trend_filter": True, "ma_trend": 60,
            "slippage_pct": 0.0004,
            "annual_factor": 252 * 4,
            "grid_stop_loss": [-0.01, -0.02, -0.03],
            "grid_take_profit": [0.02, 0.03, 0.04],
            "grid_kdj_n": [9, 14, 21],
            "grid_rsi_th": [20, 30, 40],
        }
    elif freq == "30":
        return {
            "kdj_n": 9, "kdj_m1": 2, "kdj_m2": 2,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "rsi_period": 6, "rsi_buy_th": 35,
            "stop_loss": -0.02, "take_profit": 0.035,
            "use_trend_filter": True, "ma_trend": 80,
            "slippage_pct": 0.0004,
            "annual_factor": 252 * 8,
            "grid_stop_loss": [-0.01, -0.02, -0.03],
            "grid_take_profit": [0.015, 0.025, 0.035],
            "grid_kdj_n": [9, 14, 21],
            "grid_rsi_th": [20, 30, 40],
        }
    elif freq == "15":
        return {
            "kdj_n": 9, "kdj_m1": 2, "kdj_m2": 2,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "rsi_period": 6, "rsi_buy_th": 30,
            "stop_loss": -0.02, "take_profit": 0.03,
            "use_trend_filter": False, "ma_trend": 120,
            "slippage_pct": 0.0005,
            "annual_factor": 252 * 16,
            "grid_stop_loss": [-0.01, -0.02, -0.03],
            "grid_take_profit": [0.01, 0.02, 0.03],
            "grid_kdj_n": [9, 14, 21],
            "grid_rsi_th": [20, 30, 40],
        }
    elif freq == "5":
        return {
            "kdj_n": 9, "kdj_m1": 2, "kdj_m2": 2,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "rsi_period": 6, "rsi_buy_th": 30,
            "stop_loss": -0.02, "take_profit": 0.02,
            "use_trend_filter": False, "ma_trend": 120,
            "slippage_pct": 0.0005,
            "annual_factor": 252 * 48,
            "grid_stop_loss": [-0.01, -0.02, -0.03],
            "grid_take_profit": [0.01, 0.02, 0.03],
            "grid_kdj_n": [9, 14, 21],
            "grid_rsi_th": [20, 30, 40],
        }
    elif freq == "1":
        # baostock 不支持 1 分钟，这里仅保留配置供参考；实际取数会报错
        return {
            "kdj_n": 9, "kdj_m1": 2, "kdj_m2": 2,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "rsi_period": 6, "rsi_buy_th": 30,
            "stop_loss": -0.01, "take_profit": 0.01,
            "use_trend_filter": False, "ma_trend": 240,
            "slippage_pct": 0.0008,
            "annual_factor": 252 * 240,
            "grid_stop_loss": [-0.005, -0.01, -0.02],
            "grid_take_profit": [0.005, 0.01, 0.02],
            "grid_kdj_n": [9, 14, 21],
            "grid_rsi_th": [20, 30, 40],
        }
    else:
        raise ValueError(f"不支持的频率: {freq}，可选 d/60/30/15/5")

CFG = get_period_config(FREQUENCY)

KDJ_N = CFG["kdj_n"]
KDJ_M1 = CFG["kdj_m1"]
KDJ_M2 = CFG["kdj_m2"]
MACD_FAST = CFG["macd_fast"]
MACD_SLOW = CFG["macd_slow"]
MACD_SIGNAL = CFG["macd_signal"]
RSI_PERIOD = CFG["rsi_period"]
RSI_BUY_TH = CFG["rsi_buy_th"]
STOP_LOSS = CFG["stop_loss"]
TAKE_PROFIT = CFG["take_profit"]
USE_TREND_FILTER = CFG["use_trend_filter"]
MA_TREND = CFG["ma_trend"]
SLIPPAGE_PCT = CFG["slippage_pct"]
ANNUAL_FACTOR = CFG["annual_factor"]
GRID_STOP_LOSS = CFG["grid_stop_loss"]
GRID_TAKE_PROFIT = CFG["grid_take_profit"]
GRID_KDJ_N = CFG["grid_kdj_n"]
GRID_RSI_TH = CFG["grid_rsi_th"]

USE_MA10_EXIT = True

# ======================================================================
# 三、数据获取（统一返回 datetime 列）
# ======================================================================

def fetch_data(symbol, start_date, end_date, adjustflag, frequency="d"):
    """获取日线或分钟线，统一返回 datetime/open/high/low/close/volume。"""
    if frequency not in BAOSTOCK_SUPPORTED_FREQ:
        raise ValueError(
            f"baostock 不支持频率 '{frequency}'，"
            f"可选: {sorted(BAOSTOCK_SUPPORTED_FREQ)}。"
            f"如需 1 分钟线，请换用其他数据源（如 akshare/tushare）。"
        )

    bs_code = ("sh." if symbol.startswith("6") else "sz.") + symbol
    is_minute = frequency != "d"

    bs.login()
    fields = "date,time,open,high,low,close,volume" if is_minute else \
             "date,open,high,low,close,volume"
    rs = bs.query_history_k_data_plus(
        bs_code, fields,
        start_date=start_date, end_date=end_date,
        frequency=frequency, adjustflag=adjustflag,
    )
    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())

    # 诊断信息：能区分是接口报错还是数据确实为空
    print(f"[fetch_data] code={bs_code} freq={frequency} "
          f"区间={start_date}~{end_date} "
          f"状态码={rs.error_code} 消息={rs.error_msg} 返回行数={len(data_list)}")

    df = pd.DataFrame(data_list, columns=rs.fields)
    bs.logout()

    if df.empty:
        hint = (
            "未获取到数据。可能原因：\n"
            "  1) 分钟线当日数据需等收盘后 20:30 左右入库，END_DATE 请往前挪 1~2 个交易日；\n"
            "  2) 频率不受支持（baostock 只支持 d/w/m/5/15/30/60，无 1 分钟）；\n"
            "  3) 股票代码、日期区间或复权参数有误。"
        )
        raise RuntimeError(hint)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if is_minute:
        df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S%f")
    else:
        df["datetime"] = pd.to_datetime(df["date"])

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    return df

# ======================================================================
# 四、指标计算
# ======================================================================

def kdj(high, low, close, n=11, m1=2, m2=2):
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    return k, d

def macd(close, fast=10, slow=20, signal=7):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist

def rsi(close, period=6):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def attach_indicators(data, kdj_n, macd_fast, macd_slow, macd_signal, ma_trend):
    data = data.copy()
    k, d = kdj(data["high"], data["low"], data["close"], n=kdj_n)
    dif, dea, hist = macd(data["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    data["K"], data["D"] = k, d
    data["DIF"], data["DEA"], data["HIST"] = dif, dea, hist
    data["RSI6"] = rsi(data["close"], RSI_PERIOD)
    data["MA10"] = data["close"].rolling(10).mean()
    data["MA_TREND"] = data["close"].rolling(ma_trend).mean()
    return data

# ======================================================================
# 五、交易信号
# ======================================================================

def is_buy_signal(prev, row, rsi_th, use_trend_filter):
    kdj_gold = (prev["K"] <= prev["D"]) and (row["K"] > row["D"]) and (row["K"] < 80)
    macd_gold = (prev["DIF"] <= prev["DEA"]) and (row["DIF"] > row["DEA"])
    rsi_ok = prev["RSI6"] < rsi_th
    trend_ok = (prev["close"] > prev["MA_TREND"]) if use_trend_filter else True
    return kdj_gold and macd_gold and rsi_ok and trend_ok

def sell_reason(prev, row, entry_price, stop_loss, take_profit, use_ma10_exit):
    pnl_pct = (prev["close"] - entry_price) / entry_price

    if pnl_pct <= stop_loss:
        return True, "止损"
    if pnl_pct >= take_profit:
        return True, "止盈"
    if use_ma10_exit and prev["close"] < prev["MA10"]:
        return True, "跌破MA10"
    if prev["K"] >= prev["D"] and row["K"] < row["D"]:
        return True, "KDJ死叉"
    if prev["RSI6"] > 80:
        return True, "RSI超买"
    return False, None

# ======================================================================
# 六、回测核心
# ======================================================================

def run_backtest(data, stop_loss, take_profit, kdj_n, macd_fast, macd_slow, macd_signal,
                 rsi_th, use_trend_filter=True, use_ma10_exit=True):
    if data is None or len(data) < 2:
        return {"final": CAPITAL, "profit_pct": 0.0, "ann_ret": 0.0, "max_dd": 0.0,
                "sharpe": 0.0, "calmar": 0.0, "n_trades": 0, "win_rate": 0.0, "trades": []}

    # 防御：DataFrame 取出的数值可能是 float，rolling 要求 int
    kdj_n = int(kdj_n)
    macd_fast = int(macd_fast)
    macd_slow = int(macd_slow)
    macd_signal = int(macd_signal)

    data = attach_indicators(data, kdj_n, macd_fast, macd_slow, macd_signal, MA_TREND)

    cash = CAPITAL
    shares = 0
    entry_price = 0.0
    trades = []
    equity_curve = []

    for i in range(1, len(data)):
        row = data.iloc[i]
        prev = data.iloc[i-1]
        open_p, close_p = row["open"], row["close"]

        # ---------- 卖出 ----------
        if shares > 0:
            should_sell, reason = sell_reason(prev, row, entry_price,
                                              stop_loss, take_profit, use_ma10_exit)
            if should_sell:
                fill_price = open_p * (1 - SLIPPAGE_PCT)
                gross = shares * fill_price
                fee = gross * (COMMISSION + STAMP_TAX)
                cash += gross - fee
                trades.append({"datetime": row["datetime"], "action": "SELL", "price": fill_price,
                               "shares": shares, "reason": reason,
                               "pnl": (fill_price - entry_price) * shares})
                shares = 0
                entry_price = 0.0

        # ---------- 买入 ----------
        if shares == 0 and cash >= POSITION:
            if is_buy_signal(prev, row, rsi_th, use_trend_filter):
                fill_price = open_p * (1 + SLIPPAGE_PCT)
                buy_shares = int(POSITION / fill_price / 100) * 100
                if buy_shares > 0:
                    cost = buy_shares * fill_price
                    fee = cost * COMMISSION
                    if cash >= cost + fee:
                        cash -= cost + fee
                        shares = buy_shares
                        entry_price = fill_price
                        trades.append({"datetime": row["datetime"], "action": "BUY", "price": fill_price,
                                       "shares": buy_shares, "reason": "信号共振", "pnl": 0})

        equity_curve.append(cash + shares * close_p)

    return compute_performance(equity_curve, trades)

def compute_performance(equity_curve, trades):
    equity = pd.Series(equity_curve)
    final_value = equity.iloc[-1]
    daily_ret = equity.pct_change().dropna()
    sharpe = np.sqrt(ANNUAL_FACTOR) * daily_ret.mean() / daily_ret.std() if daily_ret.std() > 0 else 0
    max_dd = ((equity.cummax() - equity) / equity.cummax()).max()
    ann_ret = (final_value / CAPITAL) ** (ANNUAL_FACTOR / len(equity)) - 1
    calmar = ann_ret / max_dd if max_dd > 0 else np.inf

    sell_trades = [t for t in trades if t["action"] == "SELL"]
    wins = [t for t in sell_trades if t["pnl"] > 0]
    win_rate = len(wins) / len(sell_trades) if sell_trades else 0

    return {
        "final": final_value,
        "profit_pct": (final_value - CAPITAL) / CAPITAL * 100,
        "ann_ret": ann_ret,
        "max_dd": max_dd * 100,
        "sharpe": sharpe,
        "calmar": calmar,
        "n_trades": len(sell_trades),
        "win_rate": win_rate * 100,
        "trades": trades,
    }

# ======================================================================
# 七、切分、网格搜索、样本外验证
# ======================================================================

def split_train_test(df, train_end):
    train = df[df["datetime"] < train_end].reset_index(drop=True)
    test = df[df["datetime"] >= train_end].reset_index(drop=True)
    return train, test

def split_with_fallback(df, train_end):
    train, test = split_train_test(df, train_end)
    if len(train) < 60 or len(test) < 60:
        mid = df["datetime"].iloc[len(df) // 2]
        print(f"数据不足以按 {train_end} 切分（训练集 {len(train)} 行），"
              f"改用中点 {mid} 切分\n")
        train, test = split_train_test(df, mid)
    return train, test

def grid_search(train_data, param_grid):
    results = []
    for sl, tp, kn, rt in itertools.product(
        param_grid["stop_loss"], param_grid["take_profit"],
        param_grid["kdj_n"], param_grid["rsi_th"]
    ):
        r = run_backtest(
            train_data,
            stop_loss=sl, take_profit=tp, kdj_n=kn,
            macd_fast=MACD_FAST, macd_slow=MACD_SLOW, macd_signal=MACD_SIGNAL,
            rsi_th=rt,
            use_trend_filter=USE_TREND_FILTER, use_ma10_exit=USE_MA10_EXIT,
        )
        if r["n_trades"] >= MIN_TRADES:
            results.append({
                "stop_loss": sl, "take_profit": tp, "kdj_n": kn, "rsi_th": rt,
                "final": r["final"], "profit_pct": r["profit_pct"],
                "ann_ret": r["ann_ret"], "max_dd": r["max_dd"],
                "sharpe": r["sharpe"], "calmar": r["calmar"],
                "n_trades": r["n_trades"], "win_rate": r["win_rate"],
            })
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("calmar", ascending=False)

def validate_out_of_sample(test_data, best_params):
    return run_backtest(
        test_data,
        stop_loss=best_params["stop_loss"],
        take_profit=best_params["take_profit"],
        kdj_n=best_params["kdj_n"],
        macd_fast=MACD_FAST, macd_slow=MACD_SLOW, macd_signal=MACD_SIGNAL,
        rsi_th=best_params["rsi_th"],
        use_trend_filter=USE_TREND_FILTER, use_ma10_exit=USE_MA10_EXIT,
    )

# ======================================================================
# 八、输出辅助
# ======================================================================

def print_perf(title, r):
    print(f"===== {title} =====")
    print(f"期末权益: {r['final']:,.2f}  总收益: {r['profit_pct']:.2f}%")
    print(f"年化: {r['ann_ret']:.2%}  最大回撤: {r['max_dd']:.2f}%  "
          f"夏普: {r['sharpe']:.2f}  Calmar: {r['calmar']:.2f}")
    print(f"交易次数: {r['n_trades']}  胜率: {r['win_rate']:.1f}%")
    print()

def print_trades(trades):
    print("===== 样本外交易明细 =====")
    if not trades:
        print("（无交易）")
        return
    for t in trades:
        print(f"{t['datetime']} {t['action']:4s} 价格={t['price']:.2f} "
              f"股数={t['shares']} 原因={t['reason']} 盈亏={t['pnl']:.2f}")

# ======================================================================
# 九、Excel 报告导出
# ======================================================================

def _summary_row(stage_name, r):
    return {
        "阶段": stage_name,
        "期末权益": round(r["final"], 2),
        "总收益%": round(r["profit_pct"], 2),
        "年化%": round(r["ann_ret"] * 100, 2),
        "最大回撤%": round(r["max_dd"], 2),
        "夏普": round(r["sharpe"], 3),
        "Calmar": round(r["calmar"], 3),
        "交易次数": r["n_trades"],
        "胜率%": round(r["win_rate"], 2),
    }

def _trades_df(trades):
    if not trades:
        return pd.DataFrame(columns=["datetime", "action", "price", "shares", "reason", "pnl"])
    df = pd.DataFrame(trades)
    df["datetime"] = df["datetime"].astype(str)
    return df

def export_excel(basic_result=None, grid_df=None, oos_result=None,
                 extra_summary=None, filename=REPORT_FILE):
    """把各阶段结果写入多个 Sheet。

    参数:
        basic_result : dict 或 None，基础回测结果
        grid_df      : DataFrame 或 None，网格搜索结果
        oos_result   : dict 或 None，样本外结果
        extra_summary: dict 或 None，参数配置等附加信息
        filename     : 输出文件名
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("未安装 openpyxl，无法导出 Excel。请执行: pip install openpyxl")
        return

    sheets = {}

    # --- 参数配置 ---
    if extra_summary:
        cfg_df = pd.DataFrame(
            [{"项目": k, "值": v} for k, v in extra_summary.items()]
        )
        sheets["参数配置"] = cfg_df

    # --- 摘要汇总 ---
    summary_rows = []
    if basic_result:
        summary_rows.append(_summary_row("基础回测（默认参数）", basic_result))
    if oos_result:
        summary_rows.append(_summary_row("样本外验证", oos_result))
    if summary_rows:
        sheets["绩效汇总"] = pd.DataFrame(summary_rows)

    # --- 基础回测明细 ---
    if basic_result and basic_result.get("trades"):
        sheets["基础回测交易明细"] = _trades_df(basic_result["trades"])

    # --- 网格搜索结果 ---
    if grid_df is not None and not grid_df.empty:
        sheets["网格搜索"] = grid_df

    # --- 样本外明细 ---
    if oos_result and oos_result.get("trades"):
        sheets["样本外交易明细"] = _trades_df(oos_result["trades"])

    if not sheets:
        print("没有可导出的结果，跳过 Excel 报告")
        return

    try:
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)  # Sheet 名上限 31 字符
        print(f"报告已导出: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"导出 Excel 失败: {e}")

# ======================================================================
# 十、三个可开关的回测阶段 API
# ======================================================================

def run_basic_backtest(df):
    """阶段 1：基础回测（默认参数）。"""
    print(f"===== 基础回测（默认参数，频率={FREQUENCY}） =====")
    r = run_backtest(
        df, STOP_LOSS, TAKE_PROFIT, KDJ_N,
        MACD_FAST, MACD_SLOW, MACD_SIGNAL, RSI_BUY_TH,
        USE_TREND_FILTER, USE_MA10_EXIT,
    )
    print_perf("基础回测结果", r)
    return r

def run_grid_search_stage(df):
    """阶段 2：网格搜索（训练集），返回 (train, test, res_df, best)。"""
    train, test = split_with_fallback(df, TRAIN_END)
    print(f"训练集: {train['datetime'].min()} ~ {train['datetime'].max()} ({len(train)} 行)")
    print(f"测试集: {test['datetime'].min()} ~ {test['datetime'].max()} ({len(test)} 行)\n")

    param_grid = {
        "stop_loss": GRID_STOP_LOSS,
        "take_profit": GRID_TAKE_PROFIT,
        "kdj_n": GRID_KDJ_N,
        "rsi_th": GRID_RSI_TH,
    }
    print("===== 网格搜索（训练集） =====")
    res_df = grid_search(train, param_grid)
    if res_df.empty:
        print(f"没有参数满足 n_trades >= {MIN_TRADES}，请放宽网格范围或延长训练区间")
        return train, test, res_df, None

    print("按 Calmar 排序 Top 10：")
    print(res_df.head(10).to_string(index=False))
    print()

    best = res_df.iloc[0].to_dict()
    best["kdj_n"] = int(best["kdj_n"])   # 关键：转 int，避免 rolling 报错
    return train, test, res_df, best

def run_oos_stage(test, best):
    """阶段 3：样本外验证（测试集）。"""
    if best is None:
        print("没有可用的最优参数，跳过样本外验证")
        return None

    print("===== 样本外验证（测试集） =====")
    print(f"最优参数: stop_loss={best['stop_loss']}, take_profit={best['take_profit']}, "
          f"kdj_n={best['kdj_n']}, rsi_th={best['rsi_th']}\n")
    oos = validate_out_of_sample(test, best)
    print_perf("样本外结果", oos)
    print_trades(oos["trades"])
    return oos

# ======================================================================
# 十一、主流程（只做串联，按开关调用）
# ======================================================================

def main():
    df = fetch_data(SYMBOL, START_DATE, END_DATE, ADJUST_FLAG, FREQUENCY)
    print(f"频率: {FREQUENCY}  数据行数: {len(df)}，"
          f"区间: {df['datetime'].min()} ~ {df['datetime'].max()}\n")

    basic_result = None
    grid_df = None
    oos_result = None

    # 阶段 1：基础回测
    if ENABLE_BASIC:
        basic_result = run_basic_backtest(df)

    # 阶段 2：网格搜索
    train = test = res_df = best = None
    if ENABLE_GRID:
        train, test, grid_df, best = run_grid_search_stage(df)

    # 阶段 3：样本外验证
    if ENABLE_OOS:
        # 如果没跑网格搜索，但需要样本外，就用默认参数或先切分
        if best is None and res_df is None:
            train, test = split_with_fallback(df, TRAIN_END)
            best = {
                "stop_loss": STOP_LOSS, "take_profit": TAKE_PROFIT,
                "kdj_n": KDJ_N, "rsi_th": RSI_BUY_TH,
            }
        if test is not None:
            oos_result = run_oos_stage(test, best)

    # 导出 Excel 报告
    extra_summary = {
        "股票代码": SYMBOL,
        "开始日期": START_DATE,
        "结束日期": END_DATE,
        "频率": FREQUENCY,
        "复权方式": ADJUST_FLAG,
        "初始资金": CAPITAL,
        "单笔仓位": POSITION,
        "佣金率": COMMISSION,
        "印花税": STAMP_TAX,
        "滑点": SLIPPAGE_PCT,
        "训练截止": TRAIN_END,
        "最小交易次数": MIN_TRADES,
        "默认止损": STOP_LOSS,
        "默认止盈": TAKE_PROFIT,
        "KDJ_N": KDJ_N,
        "MACD": f"{MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}",
        "RSI周期": RSI_PERIOD,
        "RSI买入阈值": RSI_BUY_TH,
        "趋势过滤": USE_TREND_FILTER,
        "MA趋势": MA_TREND,
    }
    export_excel(
        basic_result=basic_result,
        grid_df=grid_df,
        oos_result=oos_result,
        extra_summary=extra_summary,
        filename=REPORT_FILE,
    )

# ======================================================================
# 十二、入口
# ======================================================================

if __name__ == "__main__":
    main()