#!/usr/bin/env python3
"""
통계적 참고치 계산기 — 예측이 아니라 "과거 데이터 기반 참고 범위"를 계산한다.

- KR/US 주식: 최근 종가로 일간 변동성(realized volatility)을 구하고,
  랜덤워크 가정(sqrt(t) 스케일링)으로 1일/7일/1개월 변동폭 레인지를 만든다.
  + 최근 5일/20일 모멘텀(%), US는 yfinance의 목표주가 컨센서스(있으면)도 같이 낸다.
- 코인: CoinGecko 공개 API가 이미 제공하는 24h/7d/30d 변동률을 그대로 사용한다
  (직접 계산 안 함 — 제공되는 값을 쓰는 게 더 정확하고 단순함).

사용법:
  python quant_stats.py --market kr 005930 000660
  python quant_stats.py --market us NVDA MSFT
  python quant_stats.py --market coin bitcoin ethereum solana

출력: JSON (stdout). 실패한 티커는 결과에서 error 필드로 표시하고 전체는 죽지 않는다.
"""
import sys
import json
import argparse
import math

sys.stdout.reconfigure(encoding="utf-8")


def kr_stats(tickers):
    import FinanceDataReader as fdr
    import datetime

    out = {}
    end = datetime.date.today()
    start = end - datetime.timedelta(days=120)
    for t in tickers:
        try:
            df = fdr.DataReader(t, start, end)
            if len(df) < 21:
                out[t] = {"error": "데이터 부족 (상장일이 최근이거나 코드 오류 가능)"}
                continue
            close = df["Close"]
            daily_ret = close.pct_change().dropna()
            sigma = daily_ret.tail(20).std()  # 최근 20거래일 일간 변동성
            last = float(close.iloc[-1])
            mom5 = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 6 else None
            mom20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 21 else None
            out[t] = {
                "last_price": last,
                "daily_vol_pct": round(float(sigma) * 100, 2),
                "range_1d_pct": round(float(sigma) * 100, 2),
                "range_7d_pct": round(float(sigma) * math.sqrt(5) * 100, 2),
                "range_1m_pct": round(float(sigma) * math.sqrt(20) * 100, 2),
                "momentum_5d_pct": round(mom5, 2) if mom5 is not None else None,
                "momentum_20d_pct": round(mom20, 2) if mom20 is not None else None,
                "source": "FinanceDataReader (KRX 종가 기준)",
            }
        except Exception as e:
            out[t] = {"error": str(e)}
    return out


def us_stats(tickers):
    import yfinance as yf
    import math as _m

    out = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            df = tk.history(period="4mo")
            if len(df) < 21:
                out[t] = {"error": "데이터 부족"}
                continue
            close = df["Close"]
            daily_ret = close.pct_change().dropna()
            sigma = daily_ret.tail(20).std()
            last = float(close.iloc[-1])
            mom5 = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 6 else None
            mom20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 21 else None
            target = None
            try:
                info = tk.get_info()
                target = info.get("targetMeanPrice")
            except Exception:
                pass
            entry = {
                "last_price": last,
                "daily_vol_pct": round(float(sigma) * 100, 2),
                "range_1d_pct": round(float(sigma) * 100, 2),
                "range_7d_pct": round(float(sigma) * _m.sqrt(5) * 100, 2),
                "range_1m_pct": round(float(sigma) * _m.sqrt(20) * 100, 2),
                "momentum_5d_pct": round(mom5, 2) if mom5 is not None else None,
                "momentum_20d_pct": round(mom20, 2) if mom20 is not None else None,
                "source": "yfinance (Yahoo Finance 종가 기준)",
            }
            if target:
                entry["analyst_target_price"] = target
                entry["analyst_target_upside_pct"] = round((target / last - 1) * 100, 2)
                entry["analyst_target_note"] = "통상 12개월 관점 컨센서스 — 7일/1개월 예측과 혼동 금지"
            out[t] = entry
        except Exception as e:
            out[t] = {"error": str(e)}
    return out


def coin_stats(ids):
    import urllib.request
    import urllib.parse

    out = {}
    ids_param = ",".join(ids)
    url = (
        "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
        f"&ids={urllib.parse.quote(ids_param)}"
        "&price_change_percentage=24h,7d,30d"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        found = {d["id"]: d for d in data}
        for cid in ids:
            d = found.get(cid)
            if not d:
                out[cid] = {"error": "CoinGecko에서 id를 찾지 못함 (coins/list로 정확한 id 확인 필요)"}
                continue
            out[cid] = {
                "last_price_usd": d.get("current_price"),
                "change_24h_pct": d.get("price_change_percentage_24h_in_currency"),
                "change_7d_pct": d.get("price_change_percentage_7d_in_currency"),
                "change_30d_pct": d.get("price_change_percentage_30d_in_currency"),
                "source": "CoinGecko public API",
            }
    except Exception as e:
        for cid in ids:
            out[cid] = {"error": str(e)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True, choices=["kr", "us", "coin"])
    p.add_argument("tickers", nargs="+")
    args = p.parse_args()

    if args.market == "kr":
        result = kr_stats(args.tickers)
    elif args.market == "us":
        result = us_stats(args.tickers)
    else:
        result = coin_stats(args.tickers)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
