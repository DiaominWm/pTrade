#encoding:gbk
'''
策略：均胜电子(600699.SH) 底仓 + 日内5分钟正T
择时：纯中证1000指数(000852.SH)，个股自身均线不参与择时
风控：个股移动止损（成本-10%减半、高点-15%清仓）为纯指数模式下唯一兜底

修正记录：
P0-1  core_shares 符号错 -> 本地账本 core_shares_local
P0-2  ATR 多除48 -> 直接5分钟ATR
P1-1  T+1 可卖校验 sellable_shares
P1-2  连亏3次暂停5交易日
P1-3  pos_value 只算目标标的
P2-1  尾盘窗口14:00-14:25，强制买回14:55
P2-2  5m取数 count=200
P2-3  底仓死区5%
P2-4  卖出条件：核心2条+辅助≥1
P2-5  capital 从账户读
A模式 择时=纯中证1000（min(指数B, MAX_B)），个股均线不参与
'''

import pandas as pd
import numpy as np

DEBUG_T = True


# ======================= 初始化 =======================
def init(ContextInfo):
    # ===== 标的：均胜电子 =====
    stock_code = getattr(ContextInfo, 'stockcode', '600658') #代码不影响 由界面输入
    market = getattr(ContextInfo, 'market', 'SH')
    ContextInfo.symbol = f"{stock_code}.{market}"          # 600699.SH

    # ===== 基准：中证1000（纯指数择时）=====
    ContextInfo.benchmark = getattr(ContextInfo, 'benchmark', '000852.SH')
    ContextInfo.benchmark_alt = getattr(ContextInfo, 'benchmark_alt', '399852.SZ')

    ContextInfo.set_universe([ContextInfo.symbol])

    ContextInfo.account_id = getattr(ContextInfo, 'accID', '6000000058')
    ContextInfo.account_type = 'STOCK'
    ContextInfo.set_account(ContextInfo.account_id)

    # ---------- 状态 ----------
    ContextInfo.t_shares = 0
    ContextInfo.t_sell_price = 0.0
    ContextInfo.t_count_today = 0
    ContextInfo.current_date = None

    ContextInfo.core_shares_local = None   # 本地底仓账本
    ContextInfo.sellable_shares = 0        # T+1可卖

    ContextInfo.t_loss_streak = 0
    ContextInfo.t_pause_until_date = None
    ContextInfo.t_total_profit = 0.0
    ContextInfo.t_trade_count = 0

    # 底仓调仓控制
    ContextInfo.last_rebalance_date = None
    ContextInfo.last_b = None
    ContextInfo.pending_b = None
    ContextInfo.pending_b_count = 0
    ContextInfo.REBALANCE_COOLDOWN_DAYS = 3
    ContextInfo.B_CONFIRM_DAYS = getattr(ContextInfo, 'b_confirm_days', 3)  # 中证1000波动大，3天

    # 个股止损
    ContextInfo.core_cost = 0.0
    ContextInfo.core_peak_price = 0.0
    ContextInfo.STOP_LOSS_HALF = getattr(ContextInfo, 'stop_loss_half', 0.10)   # 成本-10%减半
    ContextInfo.STOP_LOSS_CLEAR = getattr(ContextInfo, 'stop_loss_clear', 0.15) # 高点-15%清仓

    # ---------- 参数 ----------
    ContextInfo.LOT = 100
    ContextInfo.T_RATIO = getattr(ContextInfo, 't_ratio', 0.30)
    ContextInfo.MAX_T_TIMES = getattr(ContextInfo, 'max_t_times', 2)
    ContextInfo.RSI_SELL = getattr(ContextInfo, 'rsi_sell', 65)
    ContextInfo.RSI_BUY = getattr(ContextInfo, 'rsi_buy', 40)
    ContextInfo.VOL_DIVERGE = getattr(ContextInfo, 'vol_diverge', 0.9)
    ContextInfo.VOL_SHRINK = getattr(ContextInfo, 'vol_shrink', 0.7)
    ContextInfo.BUYBACK_DROP = getattr(ContextInfo, 'buyback_drop', 0.007)
    ContextInfo.STOP_LOSS = getattr(ContextInfo, 'stop_loss', 0.015)
    ContextInfo.MAX_LOSS_STREAK = getattr(ContextInfo, 'max_loss_streak', 3)
    ContextInfo.PAUSE_DAYS = getattr(ContextInfo, 'pause_days', 5)
    ContextInfo.PRICE_ATR_MULT = getattr(ContextInfo, 'price_atr_mult', 0.5)

    # 时间窗用分钟整数（避免字符串比较陷阱）
    ContextInfo.T_WINDOWS_MIN = [(9*60+45, 11*60), (14*60, 14*60+25)]
    ContextInfo.FORCE_BUYBACK_MIN = 14*60+55

    ContextInfo.MAX_B = getattr(ContextInfo, 'max_b', 0.80)

    ContextInfo.capital = _read_initial_equity(ContextInfo)
    print(f"【初始化】标的 {ContextInfo.symbol} | 基准 {ContextInfo.benchmark}(纯指数) | "
          f"权益 {ContextInfo.capital:,.2f} | MAX_B={ContextInfo.MAX_B} | B确认{ContextInfo.B_CONFIRM_DAYS}天")

    _precompute_timing(ContextInfo)


def _read_initial_equity(ContextInfo):
    fallback = getattr(ContextInfo, 'capital', 500000.0)
    try:
        acc_list = get_trade_detail_data(ContextInfo.account_id, ContextInfo.account_type, 'ACCOUNT')
        if acc_list:
            for fname in ('m_dBalance', 'm_dTotalAsset', 'm_dAvailable'):
                val = getattr(acc_list[0], fname, None)
                if val is not None and val > 0:
                    return float(val)
    except Exception:
        pass
    return float(fallback)


def _fetch_daily(ContextInfo, code):
    """取日线，取不到返回空 DataFrame。"""
    try:
        d = ContextInfo.get_market_data_ex(
            ['close', 'high', 'low', 'volume'],
            [code], period='1d', count=600)
        if d and code in d and not d[code].empty:
            df = d[code].copy()
            # 【修复】强制转换为 datetime，解决 QMT 返回字符串导致的 .date() 报错
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _precompute_timing(ContextInfo):
    """纯中证1000择时：B 只由基准指数决定，个股自身均线不参与。"""
    def _calc_b_map(df):
        df = df.copy()
        if isinstance(df.index[0], str):
            df.index = pd.to_datetime(df.index)
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA60'] = df['close'].rolling(60).mean()
        df['RET'] = df['close'].pct_change()
        df['VOL_MA20'] = df['volume'].rolling(20).mean()
        df['CRASH'] = (df['RET'] <= -0.03) & (df['volume'] >= df['VOL_MA20'] * 1.5)
        m = {}
        for dt, row in df.iterrows():
            day = dt.strftime('%Y-%m-%d')
            if pd.isna(row['MA60']):
                b = 0.0
            elif row['CRASH'] or row['close'] < row['MA60']:
                b = 0.0
            elif row['close'] < row['MA20']:
                b = 0.30
            elif row['MA20'] < row['MA60']:
                b = 0.60
            else:
                b = 1.00
            m[day] = b
        return m

    # 基准取数带 fallback
    df_bench = _fetch_daily(ContextInfo, ContextInfo.benchmark)
    if df_bench.empty and ContextInfo.benchmark_alt:
        print(f"?? 基准 {ContextInfo.benchmark} 取不到，尝试 {ContextInfo.benchmark_alt}")
        df_bench = _fetch_daily(ContextInfo, ContextInfo.benchmark_alt)
        if not df_bench.empty:
            ContextInfo.benchmark = ContextInfo.benchmark_alt

    if df_bench.empty:
        print(f"【严重错误】基准 {ContextInfo.benchmark} 日线取不到！")
        ContextInfo.b_map = {}
        return

    b_bench = _calc_b_map(df_bench)
    print(f"【基准】{ContextInfo.benchmark} 日线 {len(df_bench)} 根，"
          f"区间 {df_bench.index[0].date()} ~ {df_bench.index[-1].date()}")

    # ===== 纯中证1000：B 只由基准决定 =====
    ContextInfo.b_map = {d: min(bb, ContextInfo.MAX_B) for d, bb in b_bench.items()}
    print(f"【择时·纯指数】{ContextInfo.benchmark} 信号 {len(ContextInfo.b_map)} 条，"
          f"上限 {ContextInfo.MAX_B}（个股自身均线不参与）")


# ======================= 主循环 =======================
def handlebar(ContextInfo):
    timetag = ContextInfo.get_bar_timetag(ContextInfo.barpos)
    current_dt = pd.to_datetime(timetag, unit='ms')
    bar_date = current_dt.strftime('%Y-%m-%d')
    bar_time = current_dt.strftime('%H:%M')
    try:
        bar_min = _time_to_min(bar_time)
    except Exception:
        bar_min = -1

    # ---------- 跨日 ----------
    if ContextInfo.current_date != bar_date:
        ContextInfo.current_date = bar_date
        ContextInfo.t_count_today = 0
        _cross_day_update(ContextInfo)
        _maybe_rebalance(ContextInfo, bar_date, current_dt)

    # ---------- 个股止损（纯指数模式下唯一兜底）----------
    _check_core_stop(ContextInfo, current_dt)

    # ---------- 基准5分钟（做T过滤）----------
    df_mkt = _get_5m_data(ContextInfo, ContextInfo.benchmark, current_dt)
    if df_mkt.empty or len(df_mkt) < 30:
        # 基准5m取不到时降级为"无大盘过滤"，但仍允许做T
        mkt_below, mkt_macd_dead = False, False
    else:
        df_mkt = _calc_indicators(df_mkt)
        prev_mkt = df_mkt.iloc[-2]
        mkt_below = prev_mkt['close'] < prev_mkt['VWAP']
        mkt_macd_dead = (prev_mkt['DIF'] < prev_mkt['DEA']) and \
                        (df_mkt.iloc[-3]['DIF'] >= df_mkt.iloc[-3]['DEA'])

    _process_single_stock(ContextInfo, current_dt, bar_min, bar_time, mkt_below, mkt_macd_dead)


# ======================= 止损 =======================
def _check_core_stop(ContextInfo, current_dt):
    """个股移动止损：成本回撤10%减半，高点回撤15%清仓。"""
    if ContextInfo.core_shares_local is None or ContextInfo.core_shares_local <= 0:
        return
    df = ContextInfo.get_market_data(['close'], [ContextInfo.symbol], count=1)
    if df.empty:
        return
    price = df.iloc[-1]['close']

    # 更新持仓期最高价
    if ContextInfo.core_peak_price > 0:
        ContextInfo.core_peak_price = max(ContextInfo.core_peak_price, price)

    # 成本回撤减半
    if ContextInfo.core_cost > 0:
        dd = price / ContextInfo.core_cost - 1
        if dd <= -ContextInfo.STOP_LOSS_HALF:
            cash, total_pos, _ = _get_account_info(ContextInfo)
            half = int(ContextInfo.core_shares_local / 2 / ContextInfo.LOT) * ContextInfo.LOT
            half = min(half, ContextInfo.sellable_shares)
            if half >= ContextInfo.LOT:
                passorder(24, 1101, ContextInfo.account_id, ContextInfo.symbol, 5, -1, half, ContextInfo)
                ContextInfo.core_shares_local -= half
                ContextInfo.sellable_shares -= half
                print(f"  ?? [{current_dt.strftime('%Y-%m-%d %H:%M')}] 个股止损减半 "
                      f"(成本回撤{dd:.1%}) 卖{half}股 | 剩余底仓 {ContextInfo.core_shares_local}")

    # 高点回撤清仓
    if ContextInfo.core_peak_price > 0:
        pdd = price / ContextInfo.core_peak_price - 1
        if pdd <= -ContextInfo.STOP_LOSS_CLEAR:
            cash, total_pos, _ = _get_account_info(ContextInfo)
            qty = min(ContextInfo.core_shares_local, ContextInfo.sellable_shares)
            if qty >= ContextInfo.LOT:
                passorder(24, 1101, ContextInfo.account_id, ContextInfo.symbol, 5, -1, qty, ContextInfo)
                ContextInfo.core_shares_local -= qty
                ContextInfo.sellable_shares -= qty
                print(f"  ?? [{current_dt.strftime('%Y-%m-%d %H:%M')}] 个股清仓 "
                      f"(高点回撤{pdd:.1%}) 卖{qty}股")
                ContextInfo.core_cost = 0.0
                ContextInfo.core_peak_price = 0.0


# ======================= 底仓调仓 =======================
def _maybe_rebalance(ContextInfo, bar_date, current_dt):
    target_b = ContextInfo.b_map.get(bar_date, 0.0)

    # B 变化需连续确认
    if target_b != ContextInfo.last_b:
        if ContextInfo.pending_b == target_b:
            ContextInfo.pending_b_count += 1
        else:
            ContextInfo.pending_b = target_b
            ContextInfo.pending_b_count = 1
        if ContextInfo.pending_b_count < ContextInfo.B_CONFIRM_DAYS:
            if DEBUG_T:
                print(f"  [B待确认 {bar_date}] target={target_b:.2f} "
                      f"count={ContextInfo.pending_b_count}/{ContextInfo.B_CONFIRM_DAYS}")
            return
        ContextInfo.last_b = target_b
        ContextInfo.pending_b_count = 0
    else:
        return

    # 冷却期
    if ContextInfo.last_rebalance_date:
        gap = (current_dt - pd.to_datetime(ContextInfo.last_rebalance_date)).days
        if gap < ContextInfo.REBALANCE_COOLDOWN_DAYS:
            return

    _rebalance_core(ContextInfo, bar_date, target_b)
    ContextInfo.last_rebalance_date = bar_date


def _cross_day_update(ContextInfo):
    _, total_pos, _ = _get_account_info(ContextInfo)
    if ContextInfo.core_shares_local is None:
        ContextInfo.core_shares_local = total_pos
    else:
        expected = ContextInfo.core_shares_local + ContextInfo.t_shares
        if abs(expected - total_pos) > ContextInfo.LOT:
            print(f"?? 账本对账：本地{expected} vs 账户{total_pos}，以账户为准")
            ContextInfo.core_shares_local = max(0, total_pos - ContextInfo.t_shares)
    ContextInfo.sellable_shares = ContextInfo.core_shares_local


def _get_account_info(ContextInfo):
    parts = ContextInfo.symbol.split('.')
    target_code = parts[0]
    target_exch = parts[1] if len(parts) > 1 else ''
    cash, total_pos = 0.0, 0
    try:
        acc = get_trade_detail_data(ContextInfo.account_id, ContextInfo.account_type, 'ACCOUNT')
        if acc:
            cash = getattr(acc[0], 'm_dAvailable', 0.0)
        pos_list = get_trade_detail_data(ContextInfo.account_id, ContextInfo.account_type, 'POSITION')
        if pos_list:
            for pos in pos_list:
                if getattr(pos, 'm_strInstrumentID', '') == target_code and \
                   (not target_exch or getattr(pos, 'm_strExchangeID', '') == target_exch):
                    total_pos = getattr(pos, 'm_nVolume', 0)
                    break
    except Exception:
        pass
    return cash, total_pos, None


def _rebalance_core(ContextInfo, bar_date, target_b):
    df = ContextInfo.get_market_data(['close'], [ContextInfo.symbol], count=1)
    if df.empty:
        return
    price = df.iloc[-1]['close']
    cash, total_pos, _ = _get_account_info(ContextInfo)
    core_shares = ContextInfo.core_shares_local
    real_equity = cash + total_pos * price
    target_shares = int(real_equity * target_b / (price * 1.02) / ContextInfo.LOT) * ContextInfo.LOT
    diff = target_shares - core_shares

    if abs(diff) < max(3000, core_shares * 0.05):
        return

    if diff > 0:
        cost = diff * price * 1.0005
        if cost > cash:
            print(f"?? [{bar_date}] 资金不足买 {diff} 股，需{cost:.0f}，可用{cash:.0f}")
            return
        passorder(23, 1101, ContextInfo.account_id, ContextInfo.symbol, 5, -1, diff, ContextInfo)
        old_val = ContextInfo.core_cost * core_shares
        ContextInfo.core_shares_local += diff
        ContextInfo.core_cost = (old_val + diff * price) / ContextInfo.core_shares_local
        ContextInfo.core_peak_price = max(ContextInfo.core_peak_price, price)
        equity = (cash - cost) + (total_pos + diff) * price
        print(f"  >>> [{bar_date}] 买 {diff} | 底仓 {ContextInfo.core_shares_local} | "
              f"B={target_b:.2f} | 成本 {ContextInfo.core_cost:.3f} | "
              f"资产 {equity:,.0f} | 盈亏 {equity-ContextInfo.capital:,.0f}")

    elif diff < 0:
        sell_qty = min(abs(diff), core_shares, ContextInfo.sellable_shares)
        if sell_qty <= 0:
            return
        revenue = sell_qty * price * 0.9995
        passorder(24, 1101, ContextInfo.account_id, ContextInfo.symbol, 5, -1, sell_qty, ContextInfo)
        ContextInfo.core_shares_local -= sell_qty
        ContextInfo.sellable_shares -= sell_qty
        if ContextInfo.core_shares_local <= 0:
            ContextInfo.core_cost = 0.0
            ContextInfo.core_peak_price = 0.0
        equity = (cash + revenue) + (total_pos - sell_qty) * price
        print(f"  >>> [{bar_date}] 卖 {sell_qty} | 底仓 {ContextInfo.core_shares_local} | "
              f"B={target_b:.2f} | 资产 {equity:,.0f} | 盈亏 {equity-ContextInfo.capital:,.0f}")


# ======================= 日内做T =======================
def _process_single_stock(ContextInfo, current_dt, bar_min, bar_time, mkt_below, mkt_macd_dead):
    df = _get_5m_data(ContextInfo, ContextInfo.symbol, current_dt)
    if df.empty or len(df) < 30:
        if DEBUG_T and _in_window_min(bar_min, ContextInfo.T_WINDOWS_MIN):
            print(f"[跳过 {bar_time}] 个股5m不足 {len(df)}")
        return

    df = _calc_indicators(df)
    curr_bar = df.iloc[-1]
    prev_bar = df.iloc[-2]
    price = curr_bar['close']
    cash, total_pos, _ = _get_account_info(ContextInfo)
    in_win = _in_window_min(bar_min, ContextInfo.T_WINDOWS_MIN)

    # ===== 1) 做T买回 =====
    if ContextInfo.t_shares > 0:
        is_force = bar_min >= ContextInfo.FORCE_BUYBACK_MIN
        is_stop = price > ContextInfo.t_sell_price * (1 + ContextInfo.STOP_LOSS)
        price_hit = (prev_bar['close'] <= ContextInfo.t_sell_price * (1 - ContextInfo.BUYBACK_DROP)) \
                    or (prev_bar['close'] <= prev_bar['VWAP'])
        confirm = (prev_bar['RSI'] < ContextInfo.RSI_BUY) and \
                  (prev_bar['volume'] / prev_bar['VOL_MA5_PREV'] < ContextInfo.VOL_SHRINK)

        if DEBUG_T and in_win:
            print(f"  [买回 {bar_time}] force={is_force} stop={is_stop} "
                  f"hit={price_hit} confirm={confirm}")

        if is_force or is_stop or (price_hit and confirm):
            qty = ContextInfo.t_shares
            buy_cost = qty * price * 1.0005
            if buy_cost > cash:
                qty = int(cash / (price * 1.0005) / ContextInfo.LOT) * ContextInfo.LOT
                if qty < ContextInfo.LOT:
                    print(f"?? [{bar_time}] 买回资金不足，剩余 {ContextInfo.t_shares} 股待次日")
                    return
                buy_cost = qty * price * 1.0005

            passorder(23, 1101, ContextInfo.account_id, ContextInfo.symbol, 5, -1, qty, ContextInfo)
            ContextInfo.core_shares_local += qty
            ContextInfo.t_shares -= qty

            sell_rev = ContextInfo.t_sell_price * qty * 0.9995
            t_profit = sell_rev - buy_cost
            ContextInfo.t_total_profit += t_profit
            ContextInfo.t_trade_count += 1
            ContextInfo.t_loss_streak = ContextInfo.t_loss_streak + 1 if t_profit < 0 else 0

            if ContextInfo.t_loss_streak >= ContextInfo.MAX_LOSS_STREAK:
                ContextInfo.t_pause_until_date = _add_trading_days(current_dt, ContextInfo.PAUSE_DAYS)
                print(f"  ?? [{bar_time}] 做T连亏{ContextInfo.t_loss_streak}次，"
                      f"暂停至 {ContextInfo.t_pause_until_date}")
                ContextInfo.t_loss_streak = 0

            equity = (cash - buy_cost) + (total_pos + qty) * price
            print(f"  >>> [{bar_time}] 【做T买回】{qty} 股 {'(强制)' if is_force else ''} | "
                  f"本笔 {t_profit:,.0f} | 累计做T {ContextInfo.t_total_profit:,.0f} | "
                  f"总盈亏 {equity-ContextInfo.capital:,.0f}")

            if ContextInfo.t_shares <= 0:
                ContextInfo.t_shares = 0
                ContextInfo.t_sell_price = 0.0
        return

    # ===== 2) 做T卖出 =====
    if ContextInfo.t_pause_until_date and ContextInfo.current_date <= ContextInfo.t_pause_until_date:
        return
    if not in_win:
        return

    core_shares = ContextInfo.core_shares_local
    if core_shares < ContextInfo.LOT or ContextInfo.sellable_shares < ContextInfo.LOT:
        return
    if ContextInfo.t_count_today >= ContextInfo.MAX_T_TIMES:
        return

    # 核心2条：RSI + 价>VWAP+ATR
    rsi_ok = prev_bar['RSI'] > ContextInfo.RSI_SELL
    price_ok = prev_bar['close'] > (prev_bar['VWAP'] + ContextInfo.PRICE_ATR_MULT * prev_bar['ATR'])
    core_ok = rsi_ok and price_ok

    # 辅助条件：量背离 或 MACD死叉/大盘走弱
    aux1 = prev_bar['VOL_DIVERGE'] < ContextInfo.VOL_DIVERGE
    macd_dead = (prev_bar['DIF'] < prev_bar['DEA']) and (df.iloc[-3]['DIF'] >= df.iloc[-3]['DEA'])
    aux2 = macd_dead or mkt_macd_dead or mkt_below
    aux_ok = aux1 or aux2

    if DEBUG_T:
        print(f"  [卖出 {bar_time}] RSI={prev_bar['RSI']:.1f}({rsi_ok}) "
              f"close={prev_bar['close']:.2f} "
              f"VWAP+ATR={prev_bar['VWAP']+ContextInfo.PRICE_ATR_MULT*prev_bar['ATR']:.2f}({price_ok}) "
              f"volDiv={prev_bar['VOL_DIVERGE']:.2f}({aux1}) "
              f"macd={macd_dead}/mkt={mkt_macd_dead}/{mkt_below}({aux2}) "
              f"-> core={core_ok} aux={aux_ok}")

    if not (core_ok and aux_ok):
        return

    sell_qty = int(core_shares * ContextInfo.T_RATIO / ContextInfo.LOT) * ContextInfo.LOT
    sell_qty = min(sell_qty, ContextInfo.sellable_shares)
    if sell_qty < 1000:
        return

    revenue = sell_qty * price * 0.9995
    passorder(24, 1101, ContextInfo.account_id, ContextInfo.symbol, 5, -1, sell_qty, ContextInfo)
    ContextInfo.core_shares_local -= sell_qty
    ContextInfo.sellable_shares -= sell_qty
    ContextInfo.t_shares = sell_qty
    ContextInfo.t_sell_price = price
    ContextInfo.t_count_today += 1

    equity = (cash + revenue) + (total_pos - sell_qty) * price
    print(f"  >>> [{bar_time}] 【做T卖出】{sell_qty} 股 @ {price:.2f} | "
          f"剩余底仓 {ContextInfo.core_shares_local} | 总盈亏 {equity-ContextInfo.capital:,.0f}")


# ======================= 指标 =======================
def _calc_indicators(df):
    df = df.copy()

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)

    # 5分钟 ATR（不再除48）
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/20, adjust=False).mean()

    # VWAP（按日重置）
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    pv = (tp * df['volume']).groupby(df.index.date).cumsum()
    cv = df['volume'].groupby(df.index.date).cumsum().replace(0, np.nan)
    df['VWAP'] = pv / cv

    # 量能
    df['VOL_MA3_PREV'] = df['volume'].shift(1).rolling(3).mean()
    df['VOL_MA20_PREV'] = df['volume'].shift(1).rolling(20).mean()
    df['VOL_MA5_PREV'] = df['volume'].shift(1).rolling(5).mean()
    df['VOL_DIVERGE'] = df['VOL_MA3_PREV'] / df['VOL_MA20_PREV']

    # MACD
    dif = df['close'].ewm(span=12, adjust=False).mean() - \
          df['close'].ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    df['DIF'], df['DEA'] = dif, dea
    return df


# ======================= 辅助 =======================
def _time_to_min(t):
    h, m = t.split(':')
    return int(h) * 60 + int(m)


def _in_window_min(tmin, windows_min):
    return any(lo <= tmin <= hi for lo, hi in windows_min)


def _get_5m_data(ContextInfo, code, current_dt):
    d = ContextInfo.get_market_data_ex(
        ['open', 'high', 'low', 'close', 'volume'],
        [code], period='5m', count=200)
    if not d or code not in d or d[code].empty:
        return pd.DataFrame()
    df = d[code].copy()
    # 【修复】强制转换为 datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df[df.index <= current_dt]


def _add_trading_days(current_dt, n):
    return (current_dt + pd.Timedelta(days=int(n * 1.5) + 2)).strftime('%Y-%m-%d')