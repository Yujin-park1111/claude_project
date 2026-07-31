#!/usr/bin/env python3
"""
rules.md v1의 "v1-lite" 버전을 과거 데이터로 백테스트한다.

중요한 한계 (반드시 report에도 남길 것):
- rules.md v1의 신호 2번(뉴스 톤)은 과거 날짜별 뉴스 텍스트가 없으면 계산 불가능하다.
  이 백테스트는 가격 데이터만으로 계산 가능한 신호 1(추세)·3(변동성)·4(해외/24h연동)만 적용한다.
  즉 이건 rules.md 전체가 아니라 "가격 기반 부분집합"의 검증이다. 뉴스 신호는 실전(포워드 테스트)에서만 검증된다.
- 룩어헤드 편향 방지: N일차 콜을 만들 때는 N-1일까지의 데이터만 쓰고, 그 콜의 성과는 N일 종가로 판정한다.

실행 전 준비:
  pip install finance-datareader yfinance pandas numpy
  (가상환경 권장: python -m venv venv && venv\\Scripts\\activate 후 위 명령)

실행:
  python backtest_script.py

결과:
  backtest/raw_results.csv  — 날짜별 콜/성과 원본
  backtest/report_v1.md     — 집계 리포트 (마크다운)

기간을 바꾸고 싶으면 아래 YEARS_BACK 값을 수정하면 된다. 데이터가 그만큼 없으면 있는 만큼만 쓴다.
"""
import os
import sys
import math
import datetime
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

YEARS_BACK = 10  # 더 예전 데이터를 쓰고 싶으면 늘려도 된다 (KOSPI는 1980년대까지, S&P/BTC는 소스가 제공하는 만큼)
TREND_THRESHOLD = 1.0     # rules.md 신호1: +-1% 기준
VOL_MULTIPLIER = 2.0      # rules.md 신호3: 평균 대비 2배 이상이면 변동성 급확대로 판단
VOL_WINDOW = 20           # 최근 20일 평균 변동폭

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = HERE


def fetch_kospi(start, end):
    import FinanceDataReader as fdr
    df = fdr.DataReader("KS11", start, end)
    return df["Close"].rename("close")


def fetch_us(start, end):
    import yfinance as yf
    df = yf.Ticker("^GSPC").history(start=start, end=end)
    s = df["Close"].rename("close")
    s.index = s.index.tz_localize(None)
    return s


def fetch_btc(start, end):
    import yfinance as yf
    df = yf.Ticker("BTC-USD").history(start=start, end=end)
    s = df["Close"].rename("close")
    s.index = s.index.tz_localize(None)
    return s


def trend_signal(pct):
    if pct >= TREND_THRESHOLD:
        return "up"
    if pct <= -TREND_THRESHOLD:
        return "down"
    return None  # 중립 = 신호 없음(투표에 참여 안 함)


RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30


def compute_rsi(close, period=RSI_PERIOD):
    """Wilder's RSI. 과매수(70+)/과매도(30-) 구간에서 되돌림(반전) 신호로 쓴다.
    출처: RSI는 횡보장에서 특히 유효하고 다른 신호와 결합할 때 신뢰도가 올라간다는 것이
    일반적으로 알려진 한계다 (강한 추세장에서는 오탐이 늘어남) — 그래서 단독이 아니라
    신호1/3/4와 함께 다수결에 참여시킨다."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(close, fast=12, slow=26, signal=9):
    """표준 MACD(12,26,9). MACD선이 시그널선을 상향/하향 돌파하면 추세 전환(신호1과 같은 방향) 신호로 쓴다.
    RSI(반전)와 반대로 MACD는 추세추종형이라, 강한 추세장에서 신호1(전일 추세)을 보강하는 역할을 기대한다."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


MAGNITUDE_MIN_HISTORY = 60   # 티어 구분(3분위) 계산에 필요한 최소 과거 표본 수
MAGNITUDE_MAX_HISTORY = 252  # 최근 1년치만 사용 (오래된 변동성 체제는 배제)
MAGNITUDE_TIERS = ("완만", "보통", "큰폭")


def magnitude_tier(value, pool):
    """pool(과거 값 목록, 오늘 값은 미포함)의 3분위 경계로 value를 완만/보통/큰폭으로 분류.
    표본이 부족하면 None(판정 보류)을 반환 — 룩어헤드 방지를 위해 pool은 항상 '오늘 이전' 값만 담는다."""
    if len(pool) < MAGNITUDE_MIN_HISTORY:
        return None
    arr = np.array(pool[-MAGNITUDE_MAX_HISTORY:])
    low, high = np.percentile(arr, [33.33, 66.67])
    if value < low:
        return "완만"
    if value < high:
        return "보통"
    return "큰폭"


def run_market(name, close, overseas_close=None, self_lag_for_24h=False, use_rsi=False, use_macd=False):
    """
    close: 이 시장의 종가 시계열 (pandas Series, index=날짜)
    overseas_close: KR인 경우 미국 종가 시계열(신호4용). None이면 신호4 미적용(US 시장).
    self_lag_for_24h: 코인용. True면 신호4를 '자기 자신의 전일 방향'(24h 연동)으로 근사.
    use_rsi: True면 신호5(RSI 과매수/과매도 반전)를 추가로 투표에 참여시킨다.
    use_macd: True면 신호6(MACD 골든/데드크로스)를 추가로 투표에 참여시킨다.
    """
    daily_ret = close.pct_change() * 100  # %
    rsi = compute_rsi(close) if use_rsi else None
    if use_macd:
        macd_line, signal_line = compute_macd(close)
        macd_hist = macd_line - signal_line
    else:
        macd_hist = None
    rows = []

    # 규모(변동폭) 신호용: 오늘 이전 값만 쌓이는 과거 표본 풀 (룩어헤드 방지)
    avg_abs_pool = []   # 예측 규모 분류용 (최근 20일 평균 변동폭의 과거 분포)
    abs_ret_pool = []   # 실제 규모 분류용 (하루 실제 변동폭의 과거 분포)

    for i in range(VOL_WINDOW + 2, len(close)):
        date_t = close.index[i]
        prev_close = close.iloc[i - 1]
        prev_prev_close = close.iloc[i - 2]

        # 신호1: 전일 추세 (전일 종가 vs 전전일 종가)
        trend_pct = (prev_close / prev_prev_close - 1) * 100
        s1 = trend_signal(trend_pct)

        # 신호3: 변동성 급확대 -> 반대방향에 가중치(되돌림 기대)
        recent_abs = daily_ret.iloc[i - 1 - VOL_WINDOW:i - 1].abs()
        avg_abs = recent_abs.mean()
        prev_day_abs_ret = abs(daily_ret.iloc[i - 1])
        s3 = None
        if pd.notna(avg_abs) and avg_abs > 0 and prev_day_abs_ret >= VOL_MULTIPLIER * avg_abs and s1 is not None:
            s3 = "down" if s1 == "up" else "up"  # 추세와 반대 방향(되돌림)

        # 신호4: 해외연동(KR) 또는 24h 연동 근사(코인). US는 미적용.
        s4 = None
        if overseas_close is not None:
            # date_t 이전(<=) 가장 최근 미국 종가 시점 두 개를 찾아 방향 계산
            us_hist = overseas_close[overseas_close.index < date_t]
            if len(us_hist) >= 2:
                us_pct = (us_hist.iloc[-1] / us_hist.iloc[-2] - 1) * 100
                s4 = trend_signal(us_pct)
        elif self_lag_for_24h:
            s4 = s1  # 데이터가 일봉뿐이라 24h 연동은 신호1과 사실상 동일 (report에 한계로 명시)

        # 신호5 (v2, 옵션): RSI 과매수/과매도 반전. 전일(i-1) 시점 RSI만 사용 (룩어헤드 방지)
        s5 = None
        if use_rsi and rsi is not None:
            rsi_prev = rsi.iloc[i - 1]
            if pd.notna(rsi_prev):
                if rsi_prev >= RSI_OVERBOUGHT:
                    s5 = "down"
                elif rsi_prev <= RSI_OVERSOLD:
                    s5 = "up"

        # 신호6 (v3, 옵션): MACD 골든/데드크로스. i-2->i-1 구간에서 히스토그램 부호가 바뀌면 크로스 발생 (룩어헤드 방지)
        s6 = None
        if use_macd and macd_hist is not None and i - 2 >= 0:
            h_prev, h_prev2 = macd_hist.iloc[i - 1], macd_hist.iloc[i - 2]
            if pd.notna(h_prev) and pd.notna(h_prev2):
                if h_prev2 <= 0 and h_prev > 0:
                    s6 = "up"    # 골든크로스
                elif h_prev2 >= 0 and h_prev < 0:
                    s6 = "down"  # 데드크로스

        signals = [s for s in [s1, s3, s4, s5, s6] if s is not None]
        up_votes = signals.count("up")
        down_votes = signals.count("down")

        if not signals:
            call = "보합"
        elif up_votes > down_votes:
            call = "상승"
        elif down_votes > up_votes:
            call = "하락"
        else:
            call = "혼조"

        entry = prev_close
        exit_ = close.iloc[i]
        actual_move_pct = (exit_ - entry) / entry * 100

        if call == "상승":
            position_return_pct = actual_move_pct
            correct = position_return_pct > 0
        elif call == "하락":
            position_return_pct = -actual_move_pct
            correct = position_return_pct > 0
        else:  # 보합/혼조
            position_return_pct = 0.0
            correct = abs(actual_move_pct) <= 1.0

        # 규모(변동폭) 신호 (실험용, rules.md 미반영): 예측 규모 = 어제 시점 최근 20일 평균 변동폭(avg_abs)의
        # 과거 분포 3분위, 실제 규모 = 오늘 실제 변동폭의 과거 분포 3분위. 두 풀 모두 '오늘 이전' 값만 담아 룩어헤드 방지.
        magnitude_predicted = magnitude_tier(avg_abs, avg_abs_pool) if pd.notna(avg_abs) else None
        magnitude_actual = magnitude_tier(abs(actual_move_pct), abs_ret_pool)
        magnitude_match = (magnitude_predicted == magnitude_actual) if (magnitude_predicted and magnitude_actual) else None
        if pd.notna(avg_abs):
            avg_abs_pool.append(float(avg_abs))
        abs_ret_pool.append(abs(float(actual_move_pct)))

        rows.append({
            "market": name,
            "date": date_t.date().isoformat(),
            "signal_trend": s1, "signal_vol": s3, "signal_overseas": s4, "signal_rsi": s5, "signal_macd": s6,
            "call": call,
            "entry": round(float(entry), 4),
            "exit": round(float(exit_), 4),
            "actual_move_pct": round(actual_move_pct, 3),
            "position_return_pct": round(position_return_pct, 3),
            "correct": correct,
            "avg_abs_pct": round(float(avg_abs), 3) if pd.notna(avg_abs) else None,
            "magnitude_predicted": magnitude_predicted,
            "magnitude_actual": magnitude_actual,
            "magnitude_match": magnitude_match,
        })

    return pd.DataFrame(rows)


def summarize_magnitude(df):
    """규모(변동폭) 신호 검증 (실험용, rules.md 미반영).
    예측 규모(avg_abs_pct 기반 완만/보통/큰폭)와 실제 규모(당일 변동폭 기반 완만/보통/큰폭)를 비교한다.
    3분위 분류이므로 무작위 추측의 기대 적중률은 33.3%다 — 이걸 못 넘으면 이 신호는 쓸모가 없다는 뜻."""
    sub = df.dropna(subset=["avg_abs_pct"])
    corr = None
    if len(sub) >= 2:
        corr = float(np.corrcoef(sub["avg_abs_pct"], sub["actual_move_pct"].abs())[0, 1])

    tier_df = df[df["magnitude_predicted"].notna() & df["magnitude_actual"].notna()]
    n_tier = len(tier_df)
    if n_tier == 0:
        return {"n_tier": 0, "corr": round(corr, 3) if corr is not None else None}

    exact_hit_rate = round(tier_df["magnitude_match"].mean() * 100, 1)
    majority_tier = tier_df["magnitude_actual"].mode().iloc[0]
    majority_baseline_pct = round((tier_df["magnitude_actual"] == majority_tier).mean() * 100, 1)
    return {
        "n_tier": n_tier,
        "corr": round(corr, 3) if corr is not None else None,
        "exact_hit_rate_pct": exact_hit_rate,
        "random_baseline_pct": round(100 / 3, 1),
        "majority_baseline_pct": majority_baseline_pct,
        "majority_tier": majority_tier,
    }


def run_magnitude_horizon(close, horizon):
    """N거래일(horizon) 뒤까지의 누적 변동폭 등급을 예측할 수 있는지 검증 (신호7의 7일/1개월 확장 검증).
    예측 변수는 신호7과 동일: 어제까지의 최근 20일 평균 일간 변동폭. 실제값은 어제 종가 대비 horizon거래일 뒤
    종가까지의 누적 변동폭(절대값). 두 분포 모두 '오늘 이전' 값만으로 3분위 분류해 룩어헤드를 방지한다."""
    daily_ret = close.pct_change() * 100
    avg_abs_pool = []
    fwd_pool = []
    rows = []
    for i in range(VOL_WINDOW + 2, len(close) - horizon + 1):
        recent_abs = daily_ret.iloc[i - 1 - VOL_WINDOW:i - 1].abs()
        avg_abs = recent_abs.mean()
        if pd.isna(avg_abs):
            continue
        entry = close.iloc[i - 1]
        exit_ = close.iloc[i - 1 + horizon]
        fwd_move_pct = abs((exit_ - entry) / entry * 100)
        predicted = magnitude_tier(avg_abs, avg_abs_pool)
        actual = magnitude_tier(fwd_move_pct, fwd_pool)
        match = (predicted == actual) if (predicted and actual) else None
        avg_abs_pool.append(float(avg_abs))
        fwd_pool.append(float(fwd_move_pct))
        rows.append({
            "avg_abs_pct": round(float(avg_abs), 3),
            "actual_move_pct": round(float(fwd_move_pct), 3),
            "magnitude_predicted": predicted,
            "magnitude_actual": actual,
            "magnitude_match": match,
        })
    return pd.DataFrame(rows)


def summarize(df):
    n = len(df)
    if n == 0:
        return {"n": 0}
    correct_n = int(df["correct"].sum())
    avg_ret = df["position_return_pct"].mean()
    cum = df["position_return_pct"].cumsum()
    peak = cum.cummax()
    mdd = (cum - peak).min()
    std = df["position_return_pct"].std()
    sharpe_like = (avg_ret / std * math.sqrt(252)) if std and std > 0 else None

    # 방향성 콜(상승/하락)만 따로 집계 — "보합=±1%면 무조건 적중" 규칙 때문에
    # 전체 적중률이 실제 수익성과 따로 노는 문제(2026-07-26 리포트에서 발견)를 보완한다.
    dir_df = df[df["call"].isin(["상승", "하락"])]
    dir_n = len(dir_df)
    dir_hit_rate = round(dir_df["correct"].mean() * 100, 1) if dir_n > 0 else None
    dir_avg_ret = round(dir_df["position_return_pct"].mean(), 3) if dir_n > 0 else None

    return {
        "n": n,
        "hit_rate_pct": round(correct_n / n * 100, 1),
        "avg_position_return_pct": round(avg_ret, 3),
        "cumulative_return_pct": round(float(cum.iloc[-1]), 2),
        "mdd_pct": round(float(mdd), 2),
        "sharpe_like": round(sharpe_like, 2) if sharpe_like is not None else None,
        "direction_only_n": dir_n,
        "direction_only_hit_rate_pct": dir_hit_rate,
        "direction_only_avg_return_pct": dir_avg_ret,
    }


def buy_and_hold(close):
    total = (close.iloc[-1] / close.iloc[0] - 1) * 100
    return round(float(total), 2)


def main():
    end = datetime.date.today()
    start = end - datetime.timedelta(days=365 * YEARS_BACK)

    print("데이터 수집 중...")
    kospi = fetch_kospi(start, end)
    sp500 = fetch_us(start, end)
    btc = fetch_btc(start, end)
    print(f"KOSPI {len(kospi)}행, S&P500 {len(sp500)}행, BTC {len(btc)}행 확보")

    # v1: 기존 신호(추세/변동성/해외연동)만. v2: 여기에 RSI 반전 신호(신호5) 추가.
    v1 = {
        "kr": run_market("kr", kospi, overseas_close=sp500),
        "us": run_market("us", sp500, overseas_close=None),
        "coin": run_market("coin", btc, overseas_close=None, self_lag_for_24h=True),
    }
    v2 = {
        "kr": run_market("kr", kospi, overseas_close=sp500, use_rsi=True),
        "us": run_market("us", sp500, overseas_close=None, use_rsi=True),
        "coin": run_market("coin", btc, overseas_close=None, self_lag_for_24h=True, use_rsi=True),
    }
    v3 = {
        "kr": run_market("kr", kospi, overseas_close=sp500, use_macd=True),
        "us": run_market("us", sp500, overseas_close=None, use_macd=True),
        "coin": run_market("coin", btc, overseas_close=None, self_lag_for_24h=True, use_macd=True),
    }

    all_df = pd.concat(list(v1.values()), ignore_index=True)
    raw_path = os.path.join(OUT_DIR, "raw_results.csv")
    all_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"저장: {raw_path} ({len(all_df)}행)")

    all_v2_df = pd.concat(list(v2.values()), ignore_index=True)
    raw_v2_path = os.path.join(OUT_DIR, "raw_results_v2_rsi.csv")
    all_v2_df.to_csv(raw_v2_path, index=False, encoding="utf-8-sig")
    print(f"저장: {raw_v2_path} ({len(all_v2_df)}행)")

    all_v3_df = pd.concat(list(v3.values()), ignore_index=True)
    raw_v3_path = os.path.join(OUT_DIR, "raw_results_v3_macd.csv")
    all_v3_df.to_csv(raw_v3_path, index=False, encoding="utf-8-sig")
    print(f"저장: {raw_v3_path} ({len(all_v3_df)}행)")

    stats_v1 = {m: summarize(df) for m, df in v1.items()}
    stats_v2 = {m: summarize(df) for m, df in v2.items()}
    stats_v3 = {m: summarize(df) for m, df in v3.items()}
    stats_mag = {m: summarize_magnitude(df) for m, df in v1.items()}
    closes = {"kr": kospi, "us": sp500, "coin": btc}
    stats_mag_h5 = {m: summarize_magnitude(run_magnitude_horizon(c, 5)) for m, c in closes.items()}
    stats_mag_h20 = {m: summarize_magnitude(run_magnitude_horizon(c, 20)) for m, c in closes.items()}
    bh = {
        "kr": buy_and_hold(kospi),
        "us": buy_and_hold(sp500),
        "coin": buy_and_hold(btc),
    }

    report_path = os.path.join(OUT_DIR, "report_v1.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 백테스트 리포트 (v1: 가격 신호만 vs v2: +RSI 반전 신호)\n\n")
        f.write("실제 투자 자문이 아니라 규칙 기반 실험 시스템 검증용입니다.\n\n")
        f.write("## 방법론 한계\n\n")
        f.write("- rules.md v1의 신호2(뉴스 톤)는 과거 뉴스 데이터가 없어 이 백테스트에서 **제외**했습니다. "
                "신호1(추세)·신호3(변동성)·신호4(해외/24h연동)만 반영한 '부분집합' 검증입니다. "
                "뉴스 신호의 효과는 실전(매일 아침 브리핑) 결과로만 검증됩니다.\n")
        f.write("- 코인의 신호4('최근 24시간 방향성')는 일봉 데이터만 있어 신호1(추세)과 사실상 동일하게 계산됐습니다 — 중복 신호일 수 있습니다.\n")
        f.write("- 룩어헤드 편향 방지: N일 콜은 N-1일까지 데이터로만 만들고, N일 종가로 결과를 판정했습니다.\n")
        f.write("- **적중률 지표의 함정**: '보합/혼조' 콜은 실제 변동이 ±1% 이내면 무조건 '적중'으로 처리됩니다 — "
                "지수는 대부분의 날이 이 범위 안에서 움직이므로 전체 적중률이 부풀려질 수 있습니다 "
                "(2026-07-26 실제 리포트에서 미국 시장 적중률 74.7%인데 수익은 -18.5%로 나온 사례로 발견됨). "
                "그래서 아래에 **방향성 콜(상승/하락)만 따로 집계한 적중률**을 추가했습니다 — 이게 더 정직한 지표입니다.\n")
        f.write("- **v2 신호5(RSI)**: RSI(14) 과매수(70+)/과매도(30-) 구간에서 반전을 기대하는 신호를 추가했습니다. "
                "룩어헤드 방지를 위해 전일 RSI만 사용합니다. RSI는 횡보장에서 특히 유효하고 강한 추세장에서는 "
                "오탐이 늘어난다는 게 일반적으로 알려진 한계입니다 — 그래서 단독이 아니라 다수결 투표에만 참여시켰습니다.\n")
        f.write("- **v3 신호6(MACD)**: 표준 MACD(12,26,9) 골든/데드크로스를 추가했습니다. RSI와 반대로 추세추종형 지표라 "
                "강한 추세장에서 신호1(전일 추세)을 보강하는 역할을 기대하고 넣었습니다. 룩어헤드 방지를 위해 "
                "i-2→i-1 구간의 크로스만 봅니다.\n")
        f.write(f"- 데이터 기간: {start.isoformat()} ~ {end.isoformat()} (요청 {YEARS_BACK}년, 실제 확보량은 소스별로 다를 수 있음)\n\n")

        f.write("## 시장별 성과 (v1 vs v2:RSI vs v3:MACD)\n\n")
        for m, label in [("kr", "국내(코스피)"), ("us", "미국(S&P500)"), ("coin", "코인(BTC)")]:
            s1_, s2_, s3_ = stats_v1[m], stats_v2[m], stats_v3[m]
            f.write(f"### {label}\n\n")
            if s1_["n"] == 0:
                f.write("- 데이터 부족으로 결과 없음\n\n")
                continue
            f.write("| 지표 | v1 (가격 신호만) | v2 (+RSI) | v3 (+MACD) |\n")
            f.write("|---|---|---|---|\n")
            f.write(f"| 콜 수 | {s1_['n']}건 | {s2_['n']}건 | {s3_['n']}건 |\n")
            f.write(f"| 전체 적중률 | {s1_['hit_rate_pct']}% | {s2_['hit_rate_pct']}% | {s3_['hit_rate_pct']}% |\n")
            f.write(f"| **방향성 콜만 적중률** (n={s1_['direction_only_n']}/{s2_['direction_only_n']}/{s3_['direction_only_n']}) "
                    f"| {s1_['direction_only_hit_rate_pct']}% | {s2_['direction_only_hit_rate_pct']}% | {s3_['direction_only_hit_rate_pct']}% |\n")
            f.write(f"| 방향성 콜 평균 수익률 | {s1_['direction_only_avg_return_pct']}% | {s2_['direction_only_avg_return_pct']}% | {s3_['direction_only_avg_return_pct']}% |\n")
            f.write(f"| 누적 수익률(단순 합산) | {s1_['cumulative_return_pct']}% | {s2_['cumulative_return_pct']}% | {s3_['cumulative_return_pct']}% |\n")
            f.write(f"| MDD(최대낙폭) | {s1_['mdd_pct']}% | {s2_['mdd_pct']}% | {s3_['mdd_pct']}% |\n")
            f.write(f"| Sharpe 유사 지표 | {s1_['sharpe_like']} | {s2_['sharpe_like']} | {s3_['sharpe_like']} |\n")
            f.write(f"| 같은 기간 buy&hold | {bh[m]}% | {bh[m]}% | {bh[m]}% |\n")
            variants = {"v1(기존)": s1_["cumulative_return_pct"], "v2(RSI)": s2_["cumulative_return_pct"], "v3(MACD)": s3_["cumulative_return_pct"]}
            best = max(variants, key=variants.get)
            f.write(f"\n- **가장 나은 조합**: {best} (누적 수익률 {variants[best]}% 기준)\n\n")

        f.write("## 규모(변동폭) 예측 신호 검증 (실험, rules.md 미반영)\n\n")
        f.write("방향(상승/하락/보합)뿐 아니라 다음날 움직임의 **대략적인 규모**(완만/보통/큰폭)도 예측할 수 있는지 검증합니다. "
                "예측 규모는 '어제 시점 최근 20일 평균 변동폭'을 그 시장의 과거 분포(3분위)에 대입해 완만/보통/큰폭으로 분류하고, "
                "실제 규모는 오늘 실제 변동폭을 같은 방식으로 분류해 비교합니다. 3분위 분류이므로 **무작위 추측은 33.3% 적중**이 기대값입니다 — "
                "이걸 못 넘으면 이 신호는 쓸모가 없다는 뜻입니다. 룩어헤드 방지를 위해 두 분포 모두 '오늘 이전' 값만 사용했습니다.\n\n")
        f.write("| 시장 | 표본수 | 예측↔실제 상관계수 | 티어 정확 적중률 | 무작위 기준(33.3%) | 최빈값만 찍기 기준 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for m, label in [("kr", "국내(코스피)"), ("us", "미국(S&P500)"), ("coin", "코인(BTC)")]:
            sm = stats_mag[m]
            if sm.get("n_tier", 0) == 0:
                f.write(f"| {label} | 0 | {sm.get('corr')} | 표본 부족 | - | - |\n")
                continue
            f.write(f"| {label} | {sm['n_tier']}건 | {sm['corr']} | {sm['exact_hit_rate_pct']}% | "
                    f"{sm['random_baseline_pct']}% | {sm['majority_baseline_pct']}% ({sm['majority_tier']}) |\n")
        f.write("\n- **상관계수**: '최근 변동성이 높았다'는 사실과 '오늘 실제로 크게 움직였다'는 사실이 얼마나 같이 가는지(0에 가까우면 무관, "
                "1에 가까우면 강한 연관). 변동성 군집(volatility clustering)이 실제로 존재하는지의 직접적인 증거입니다.\n")
        f.write("- **티어 정확 적중률**이 무작위 기준(33.3%)과 '최빈값만 찍기' 기준을 동시에 못 넘으면, 이 신호는 규칙에 추가할 근거가 없다고 판단합니다 "
                "— 최빈값 기준을 넘지 못한다는 건 '그냥 항상 가장 흔한 티어를 찍는 것'보다도 못하다는 뜻이기 때문입니다.\n")
        f.write("- 이 신호는 아직 rules.md에 반영되지 않았습니다. 사용자 확인 후에만 changelog.md → rules.md 순으로 정식 반영합니다 "
                "(CLAUDE.md 원칙: 규칙은 점진적으로, 검증된 것만).\n\n")

        f.write("### 7일/1개월 규모 확장 검증\n\n")
        f.write("같은 예측 변수(어제까지 최근 20일 평균 변동폭)로 5거래일(약 7일)·20거래일(약 1개월) 뒤까지의 **누적** 변동폭 등급도 "
                "맞힐 수 있는지 확인했습니다.\n\n")
        f.write("| 시장 | 5거래일(≈7일) 적중률 | 표본 | 최빈값 기준 | 20거래일(≈1개월) 적중률 | 표본 | 최빈값 기준 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for m, label in [("kr", "국내(코스피)"), ("us", "미국(S&P500)"), ("coin", "코인(BTC)")]:
            h5, h20 = stats_mag_h5[m], stats_mag_h20[m]
            h5_txt = f"{h5['exact_hit_rate_pct']}%" if h5.get("n_tier", 0) > 0 else "표본부족"
            h5_maj = f"{h5['majority_baseline_pct']}%" if h5.get("n_tier", 0) > 0 else "-"
            h20_txt = f"{h20['exact_hit_rate_pct']}%" if h20.get("n_tier", 0) > 0 else "표본부족"
            h20_maj = f"{h20['majority_baseline_pct']}%" if h20.get("n_tier", 0) > 0 else "-"
            f.write(f"| {label} | {h5_txt} | {h5.get('n_tier', 0)}건 | {h5_maj} | {h20_txt} | {h20.get('n_tier', 0)}건 | {h20_maj} |\n")
        f.write("\n- 무작위 기준은 여기서도 33.3%입니다.\n")
        f.write("- 기간이 길어질수록 등급 예측은 원래 더 어려워지는 게 정상입니다(먼 미래일수록 다른 변수의 영향이 커짐) — "
                "1일 대비 정확도가 떨어지는 건 신호 실패가 아니라 예상된 현상입니다.\n\n")

        f.write("## 해석 시 주의\n\n")
        f.write("- 방향성 콜만 집계한 적중률이 50%를 크게 넘지 않으면 이 신호 조합은 랜덤과 큰 차이가 없다는 뜻입니다.\n")
        f.write("- 여기서 좋은 성과가 나와도, 실전 브리핑은 뉴스 신호가 추가되므로 결과가 달라질 수 있습니다.\n")
        f.write("- 시장별로 가장 나은 조합에서만 changelog.md에 반영을 제안합니다 — 모든 시장에 일괄 적용하지 않습니다 "
                "(CLAUDE.md 원칙: 규칙은 점진적으로, 검증된 것만).\n")
        f.write("- raw_results.csv(v1), raw_results_v2_rsi.csv(v2), raw_results_v3_macd.csv(v3)로 개별 신호가 켜졌을 때만 따로 적중률을 더 뜯어볼 수 있습니다.\n")

    print(f"저장: {report_path}")
    print("완료.")
    return stats_v1, stats_v2, bh


if __name__ == "__main__":
    main()
