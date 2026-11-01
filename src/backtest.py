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
SYMBOL = "159915"
START_DATE = "2024-01-01"
END_DATE = "2026-10-30"        # 分钟线建议避开最近 1~2 个交易日，避免数据未入库
ADJUST_FLAG = "2"              # 1=后复权，2=前复权，3=不复权

# 频率："d"=日线，"60"/"30"/"15"/"5"=分钟线
# 注意：baostock 不支持 1 分钟线（"1"），如需 1 分钟请换数据源
FREQUENCY = "60"

# ---------- 资金与仓位 ----------
CAPITAL = 300000
FULL_POSITION = True           # True=每次满仓买入
POSITION = 30000               # FULL_POSITION=False 时的单笔金额

# ---------- 交易成本 ----------
COMMISSION = 0.0001            # 佣金（买卖各收一次）
STAMP_TAX = 0.001              # 印花税（仅卖出）
SLIPPAGE_YUAN = 0.02           # 滑点，按每股绝对价格计：买入 +，卖出 -

# ---------- 交易制度 ----------
ENFORCE_T1 = True              # A 股 T+1：当日买入的股票当日不可卖出

# ---------- 阶段开关（随时启用/屏蔽） ----------
ENABLE_BASIC = True            # 是否跑基础回测
ENABLE_GRID = True             # 是否跑网格搜索
ENABLE_OOS = True              # 是否跑样本外验证

# ---------- 训练/测试切分 ----------
TRAIN_END = "2026-01-01"
MIN_TRADES = 5                 # 网格搜索过滤交易过少的参数（B/S 条件较严，交易天然偏少）

# ---------- Excel 报告 ----------
REPORT_FILE = "backtest_report.xlsx"

# baostock 支持的频率集合
BAOSTOCK_SUPPORTED_FREQ = {"d", "w", "m", "5", "15", "30", "60"}

# ======================================================================
# 二、按频率自动适配的默认参数
# ======================================================================

# 策略只用到指标周期与年化因子；止损/止盈/趋势过滤已被 B 点/S 点规则取代
PERIOD_CONFIGS = {
    "d":  {"kdj_n": 11, "macd": (10, 20, 7), "annual_factor": 252},
    "60": {"kdj_n": 11, "macd": (10, 20, 7), "annual_factor": 252 * 4},
    "30": {"kdj_n": 11, "macd": (10, 20, 7), "annual_factor": 252 * 8},
    "15": {"kdj_n": 11, "macd": (10, 20, 7), "annual_factor": 252 * 16},
    "5":  {"kdj_n": 11, "macd": (10, 20, 7), "annual_factor": 252 * 48},
}

def get_period_config(freq):
    """根据频率返回 KDJ 周期、MACD 参数、年化因子。"""
    if freq not in PERIOD_CONFIGS:
        raise ValueError(f"不支持的频率: {freq}，可选 d/60/30/15/5")
    cfg = dict(PERIOD_CONFIGS[freq])
    cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"] = cfg["macd"]
    return cfg

CFG = get_period_config(FREQUENCY)

KDJ_N = CFG["kdj_n"]
KDJ_M1 = 2
KDJ_M2 = 2
MACD_FAST = CFG["macd_fast"]
MACD_SLOW = CFG["macd_slow"]
MACD_SIGNAL = CFG["macd_signal"]
CCI_N = 14
RSI_PERIODS = (7, 14, 28)          # 仅计算与展示，不参与信号
ANNUAL_FACTOR = CFG["annual_factor"]

# ---------- 信号参数（B 点 / S 点） ----------
HIST_TH = -0.05                    # MACD 柱（2*(DIF-DEA)）下限
CCI_TH = -75                       # 买入要求 CCI <= 该值
VOL_MULT = 1.9                     # 放量出逃：成交量 >= 前一根的 N 倍
DROP_PCT = 0.025                   # 急跌出逃：较前一根下跌 >= 2.5%

# ---------- 网格搜索范围（只扫信号阈值） ----------
GRID_HIST_TH = [-0.20, -0.10, -0.05, 0.0]
GRID_CCI_TH = [-100, -85, -70, -55]
GRID_VOL_MULT = [1.5, 1.9, 2.5]
GRID_DROP_PCT = [0.02, 0.025, 0.03]

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

def cci(high, low, close, n=14):
    """CCI = (TP - MA(TP,n)) / (0.015 * 平均绝对偏差)。"""
    tp = (high + low + close) / 3.0
    ma = tp.rolling(n).mean()
    md = (tp - ma).abs().rolling(n).mean()
    return (tp - ma) / (0.015 * md)

def attach_indicators(data, kdj_n=None, macd_fast=None, macd_slow=None, macd_signal=None):
    """一次算出全部指标：K/D/J、MACD、CCI、RSI7/14/28、量比、均线。"""
    kdj_n = KDJ_N if kdj_n is None else int(kdj_n)
    macd_fast = MACD_FAST if macd_fast is None else int(macd_fast)
    macd_slow = MACD_SLOW if macd_slow is None else int(macd_slow)
    macd_signal = MACD_SIGNAL if macd_signal is None else int(macd_signal)

    data = data.copy()
    k, d = kdj(data["high"], data["low"], data["close"], n=kdj_n, m1=KDJ_M1, m2=KDJ_M2)
    data["K"], data["D"] = k, d
    data["J"] = 3 * k - 2 * d

    dif, dea, hist = macd(data["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    data["DIF"], data["DEA"], data["HIST"] = dif, dea, hist
    data["HIST_PCT"] = hist / data["close"] * 100      # 归一化柱值（%），便于跨价位比较

    data["CCI"] = cci(data["high"], data["low"], data["close"], CCI_N)
    for p in RSI_PERIODS:
        data[f"RSI{p}"] = rsi(data["close"], p)

    data["VOL_RATIO"] = data["volume"] / data["volume"].shift(1)
    data["VOL_MA20"] = data["volume"].rolling(20).mean()
    data["MA10"] = data["close"].rolling(10).mean()
    data["MA20"] = data["close"].rolling(20).mean()
    data["MA60"] = data["close"].rolling(60).mean()
    data["RET"] = data["close"].pct_change()
    return data

# ======================================================================
# 五、交易信号
# ======================================================================

def is_buy_signal(prev, row, hist_th, cci_th):
    """B 点：MACD 柱 >= hist_th  且  J 上穿 D  且  CCI <= cci_th。

    说明：J = 3K - 2D，故 J - D = 3 * (K - D)，"J 上穿 D"与"K 上穿 D"是同一事件。
    """
    if pd.isna(row["J"]) or pd.isna(row["D"]) or pd.isna(row["CCI"]) or pd.isna(row["HIST"]):
        return False
    j_gold = (prev["J"] <= prev["D"]) and (row["J"] > row["D"])
    macd_ok = row["HIST"] >= hist_th
    cci_ok = row["CCI"] <= cci_th
    return bool(j_gold and macd_ok and cci_ok)

def sell_signal(prev, row, vol_mult, drop_pct):
    """S 点：J 下穿 D，且满足下列任一条件。
        1) 当根成交量 >= 前一根的 vol_mult 倍
        2) 当根收盘价较前一根下跌 >= drop_pct
    """
    if pd.isna(row["J"]) or pd.isna(row["D"]) or pd.isna(row["VOL_RATIO"]) or pd.isna(row["RET"]):
        return False, None
    if not ((prev["J"] >= prev["D"]) and (row["J"] < row["D"])):
        return False, None
    if row["VOL_RATIO"] >= vol_mult:
        return True, f"放量{row['VOL_RATIO']:.2f}倍"
    if row["RET"] <= -drop_pct:
        return True, f"急跌{row['RET'] * 100:.2f}%"
    return False, None

# ======================================================================
# 六、回测核心
# ======================================================================

def _empty_result():
    return {"final": CAPITAL, "profit_pct": 0.0, "ann_ret": 0.0, "max_dd": 0.0,
            "sharpe": 0.0, "calmar": 0.0, "n_trades": 0, "win_rate": 0.0,
            "avg_ret_pct": 0.0, "profit_factor": 0.0, "avg_hold_bars": 0.0,
            "trades": []}

def run_backtest(data, hist_th=None, cci_th=None, vol_mult=None,
                 drop_pct=None, prepared=False):
    """满仓、单标的回测。

    执行约定：
      1. 信号在"当根收盘后"确认，在"下一根开盘"成交，避免前视偏差；
      2. 滑点按绝对值计：买入价 = 开盘价 + SLIPPAGE_YUAN，卖出价 = 开盘价 - SLIPPAGE_YUAN；
      3. T+1：当日买入不可当日卖出，S 点信号顺延到下一个可交易日的第一根开盘执行。
    """
    if data is None or len(data) < 3:
        return _empty_result()

    hist_th = HIST_TH if hist_th is None else hist_th
    cci_th = CCI_TH if cci_th is None else cci_th
    vol_mult = VOL_MULT if vol_mult is None else vol_mult
    drop_pct = DROP_PCT if drop_pct is None else drop_pct

    if not prepared:
        data = attach_indicators(data)

    cash = float(CAPITAL)
    shares = 0
    entry_cost = 0.0
    entry_date = None
    entry_idx = 0
    trades = []
    equity_curve = []
    pending = None                       # ("BUY"/"SELL", reason)，下一根开盘执行

    for i in range(1, len(data)):
        row = data.iloc[i]
        open_p = row["open"]
        bar_date = row["datetime"].date()

        # ---------- 1) 执行上一根收盘挂下的单 ----------
        if pending is not None:
            side, reason = pending
            if side == "SELL":
                if shares == 0:
                    pending = None
                elif ENFORCE_T1 and bar_date == entry_date:
                    pass                     # 当日买入不可卖，挂单顺延到下一根
                else:
                    fill_price = open_p - SLIPPAGE_YUAN
                    gross = shares * fill_price
                    fee = gross * (COMMISSION + STAMP_TAX)
                    cash += gross - fee
                    pnl = gross - fee - entry_cost
                    trades.append({
                        "datetime": row["datetime"], "action": "SELL",
                        "price": fill_price, "shares": shares, "reason": reason,
                        "pnl": pnl,
                        "ret_pct": (pnl / entry_cost * 100) if entry_cost else 0.0,
                        "hold_bars": i - entry_idx,
                    })
                    shares = 0
                    entry_cost = 0.0
                    entry_date = None
                    pending = None
            else:  # BUY
                if shares == 0 and cash > 0:
                    fill_price = open_p + SLIPPAGE_YUAN
                    max_shares = int(cash / (fill_price * (1 + COMMISSION)) / 100) * 100
                    if FULL_POSITION:
                        buy_shares = max_shares
                    else:
                        budget_shares = int(POSITION / (fill_price * (1 + COMMISSION)) / 100) * 100
                        buy_shares = min(budget_shares, max_shares)
                    if buy_shares > 0:
                        cost = buy_shares * fill_price
                        fee = cost * COMMISSION
                        cash -= cost + fee
                        shares = buy_shares
                        entry_cost = cost + fee
                        entry_date = bar_date
                        entry_idx = i
                        trades.append({
                            "datetime": row["datetime"], "action": "BUY",
                            "price": fill_price, "shares": shares, "reason": reason,
                            "pnl": 0.0, "ret_pct": 0.0, "hold_bars": 0,
                        })
                pending = None

        # ---------- 2) 本根收盘后判定信号（最后一根不再挂单） ----------
        if i < len(data) - 1:
            prev = data.iloc[i - 1]
            if shares > 0:
                should_sell, why = sell_signal(prev, row, vol_mult, drop_pct)
                if should_sell:
                    pending = ("SELL", why)
            elif is_buy_signal(prev, row, hist_th, cci_th):
                pending = ("BUY", "B点")

        equity_curve.append(cash + shares * row["close"])

    return compute_performance(equity_curve, trades)

def compute_performance(equity_curve, trades):
    if not equity_curve:
        return _empty_result()

    equity = pd.Series(equity_curve)
    final_value = equity.iloc[-1]
    bar_ret = equity.pct_change().dropna()
    sharpe = (np.sqrt(ANNUAL_FACTOR) * bar_ret.mean() / bar_ret.std()
              if len(bar_ret) > 1 and bar_ret.std() > 0 else 0.0)
    max_dd = ((equity.cummax() - equity) / equity.cummax()).max()
    ann_ret = (final_value / CAPITAL) ** (ANNUAL_FACTOR / len(equity)) - 1
    calmar = ann_ret / max_dd if max_dd > 1e-9 else 0.0

    sell_trades = [t for t in trades if t["action"] == "SELL"]
    wins = [t for t in sell_trades if t["pnl"] > 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in sell_trades if t["pnl"] <= 0))
    win_rate = len(wins) / len(sell_trades) if sell_trades else 0.0

    return {
        "final": final_value,
        "profit_pct": (final_value - CAPITAL) / CAPITAL * 100,
        "ann_ret": ann_ret,
        "max_dd": max_dd * 100,
        "sharpe": sharpe,
        "calmar": calmar,
        "n_trades": len(sell_trades),
        "win_rate": win_rate * 100,
        "avg_ret_pct": float(np.mean([t["ret_pct"] for t in sell_trades])) if sell_trades else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else 0.0,
        "avg_hold_bars": float(np.mean([t["hold_bars"] for t in sell_trades])) if sell_trades else 0.0,
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
    prepared = attach_indicators(train_data)   # 指标只算一次，网格只扫信号阈值
    results = []
    for hist_th, cci_th, vol_mult, drop_pct in itertools.product(
        param_grid["hist_th"], param_grid["cci_th"],
        param_grid["vol_mult"], param_grid["drop_pct"]
    ):
        r = run_backtest(prepared, hist_th=hist_th, cci_th=cci_th,
                         vol_mult=vol_mult, drop_pct=drop_pct, prepared=True)
        if r["n_trades"] >= MIN_TRADES:
            results.append({
                "hist_th": hist_th, "cci_th": cci_th,
                "vol_mult": vol_mult, "drop_pct": drop_pct,
                "final": r["final"], "profit_pct": r["profit_pct"],
                "ann_ret": r["ann_ret"], "max_dd": r["max_dd"],
                "sharpe": r["sharpe"], "calmar": r["calmar"],
                "n_trades": r["n_trades"], "win_rate": r["win_rate"],
                "avg_ret_pct": r["avg_ret_pct"], "profit_factor": r["profit_factor"],
            })
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("calmar", ascending=False)

def validate_out_of_sample(test_data, best_params):
    return run_backtest(
        test_data,
        hist_th=best_params["hist_th"],
        cci_th=best_params["cci_th"],
        vol_mult=best_params["vol_mult"],
        drop_pct=best_params["drop_pct"],
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
    print(f"每笔均收益: {r['avg_ret_pct']:.2f}%  盈亏因子: {r['profit_factor']:.2f}  "
          f"平均持仓: {r['avg_hold_bars']:.1f} 根")
    print()

def print_trades(trades):
    print("===== 样本外交易明细 =====")
    if not trades:
        print("（无交易）")
        return
    for t in trades:
        print(f"{t['datetime']} {t['action']:4s} 价格={t['price']:.2f} "
              f"股数={t['shares']} 原因={t['reason']} 盈亏={t['pnl']:.2f} "
              f"收益率={t['ret_pct']:.2f}% 持仓={t['hold_bars']}根")

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
        "每笔均收益%": round(r["avg_ret_pct"], 2),
        "盈亏因子": round(r["profit_factor"], 3),
        "平均持仓根": round(r["avg_hold_bars"], 1),
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
    print(f"B点: MACD柱>={HIST_TH} 且 J上穿D 且 CCI<={CCI_TH}")
    print(f"S点: J下穿D 且 (量比>={VOL_MULT} 或 跌幅>={DROP_PCT * 100:.1f}%)")
    print(f"资金 {CAPITAL:,.0f} 元，全仓={FULL_POSITION}，滑点 {SLIPPAGE_YUAN} 元/股，T+1={ENFORCE_T1}\n")
    r = run_backtest(df)
    print_perf("基础回测结果", r)
    return r

def run_grid_search_stage(df):
    """阶段 2：网格搜索（训练集），返回 (train, test, res_df, best)。"""
    train, test = split_with_fallback(df, TRAIN_END)
    print(f"训练集: {train['datetime'].min()} ~ {train['datetime'].max()} ({len(train)} 行)")
    print(f"测试集: {test['datetime'].min()} ~ {test['datetime'].max()} ({len(test)} 行)\n")

    param_grid = {
        "hist_th": GRID_HIST_TH,
        "cci_th": GRID_CCI_TH,
        "vol_mult": GRID_VOL_MULT,
        "drop_pct": GRID_DROP_PCT,
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
    return train, test, res_df, best

def run_oos_stage(test, best):
    """阶段 3：样本外验证（测试集）。"""
    if best is None:
        print("没有可用的最优参数，跳过样本外验证")
        return None

    print("===== 样本外验证（测试集） =====")
    print(f"最优参数: hist_th={best['hist_th']}, cci_th={best['cci_th']}, "
          f"vol_mult={best['vol_mult']}, drop_pct={best['drop_pct']}\n")
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
                "hist_th": HIST_TH, "cci_th": CCI_TH,
                "vol_mult": VOL_MULT, "drop_pct": DROP_PCT,
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
        "仓位模式": "全仓" if FULL_POSITION else f"定额 {POSITION}",
        "佣金率": COMMISSION,
        "印花税": STAMP_TAX,
        "训练截止": TRAIN_END,
        "最小交易次数": MIN_TRADES,
        "KDJ": f"{KDJ_N},{KDJ_M1},{KDJ_M2}",
        "MACD": f"{MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}",
        "CCI周期": CCI_N,
        "RSI周期": "/".join(str(p) for p in RSI_PERIODS),
        "MACD柱阈值": HIST_TH,
        "CCI阈值": CCI_TH,
        "放量倍数": VOL_MULT,
        "急跌幅度": DROP_PCT,
        "全仓": FULL_POSITION,
        "滑点(元/股)": SLIPPAGE_YUAN,
        "T+1": ENFORCE_T1,
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