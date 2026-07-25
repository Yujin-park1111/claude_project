#!/usr/bin/env python3
"""
모의 포트폴리오 원장(ledger) 관리 스크립트. 실제 매매 없음 — 가상 추적용.
가격 조회는 하지 않는다 (그건 quant_stats.py나 그날 브리핑 작업이 WebSearch로 이미 구한 값을 넘겨받는다).
이 스크립트는 CSV 기록/집계만 담당한다 (단일 책임, portfolio/policy.md 참고).

사용법:
  # 오늘 콜을 오픈 (진입)
  python portfolio_log.py open --market kr --direction 상승 --call_date 2026-07-27 \
      --instrument KOSPI --entry_price 6690.62 --entry_note "전일 종가"

  # 열려있는 콜을 청산 (market + call_date로 매칭되는 가장 최근 open 행을 닫는다)
  python portfolio_log.py close --market kr --call_date 2026-07-27 \
      --exit_price 6700.00 --exit_note "당일 종가"

  # 누적 성과 요약 (마크다운 텍스트로 출력)
  python portfolio_log.py summary
"""
import argparse
import csv
import os
import sys
from datetime import date

sys.stdout.reconfigure(encoding="utf-8")

LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "portfolio", "ledger.csv")
LEDGER_PATH = os.path.normpath(LEDGER_PATH)
FIELDS = [
    "id", "market", "direction", "call_date", "instrument", "entry_price", "entry_note",
    "status", "exit_price", "exit_date", "exit_note", "actual_move_pct", "position_return_pct", "correct",
]


def read_rows():
    if not os.path.exists(LEDGER_PATH):
        return []
    with open(LEDGER_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(rows):
    with open(LEDGER_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def cmd_open(args):
    rows = read_rows()
    new_id = str(len(rows) + 1)
    rows.append({
        "id": new_id,
        "market": args.market,
        "direction": args.direction,
        "call_date": args.call_date,
        "instrument": args.instrument,
        "entry_price": args.entry_price,
        "entry_note": args.entry_note or "",
        "status": "open",
        "exit_price": "",
        "exit_date": "",
        "exit_note": "",
        "actual_move_pct": "",
        "position_return_pct": "",
        "correct": "",
    })
    write_rows(rows)
    print(f"오픈 완료: id={new_id} market={args.market} direction={args.direction} call_date={args.call_date} entry={args.entry_price}")


def cmd_close(args):
    rows = read_rows()
    target = None
    for r in reversed(rows):
        if r["market"] == args.market and r["call_date"] == args.call_date and r["status"] == "open":
            target = r
            break
    if target is None:
        print(f"청산 실패: market={args.market} call_date={args.call_date} 에 해당하는 open 행을 못 찾음", file=sys.stderr)
        sys.exit(1)

    entry = float(target["entry_price"])
    exit_price = float(args.exit_price)
    actual_move_pct = (exit_price - entry) / entry * 100

    direction = target["direction"]
    if direction == "상승":
        position_return_pct = actual_move_pct
        correct = position_return_pct > 0
    elif direction == "하락":
        position_return_pct = -actual_move_pct
        correct = position_return_pct > 0
    else:  # 보합 / 혼조
        position_return_pct = 0.0
        correct = abs(actual_move_pct) <= 1.0

    target["status"] = "closed"
    target["exit_price"] = str(exit_price)
    target["exit_date"] = args.exit_date or date.today().isoformat()
    target["exit_note"] = args.exit_note or ""
    target["actual_move_pct"] = f"{actual_move_pct:.2f}"
    target["position_return_pct"] = f"{position_return_pct:.2f}"
    target["correct"] = "1" if correct else "0"

    write_rows(rows)
    print(f"청산 완료: id={target['id']} 실제변동={actual_move_pct:.2f}% 포지션수익={position_return_pct:.2f}% 적중={correct}")


def cmd_summary(args):
    rows = [r for r in read_rows() if r["status"] == "closed"]
    if not rows:
        print("아직 청산된 기록이 없습니다.")
        return

    by_market = {}
    for r in rows:
        by_market.setdefault(r["market"], []).append(r)

    print("# 모의 포트폴리오 성과 요약\n")
    overall_returns = []
    for market, rs in by_market.items():
        n = len(rs)
        correct_n = sum(1 for r in rs if r["correct"] == "1")
        returns = [float(r["position_return_pct"]) for r in rs]
        overall_returns.extend(returns)
        avg_ret = sum(returns) / n
        # MDD: 순서대로 누적합의 최대 낙폭
        cum = 0.0
        peak = 0.0
        mdd = 0.0
        for ret in returns:
            cum += ret
            peak = max(peak, cum)
            mdd = min(mdd, cum - peak)
        note = "" if n >= 20 else " (표본 20건 미만 — 참고만)"
        print(f"## {market}{note}")
        print(f"- 콜 수: {n}건, 적중률: {correct_n/n*100:.1f}%")
        print(f"- 평균 포지션 수익률: {avg_ret:.2f}%, 누적: {cum:.2f}%, MDD: {mdd:.2f}%")
        print()

    if overall_returns:
        n = len(overall_returns)
        print(f"## 전체")
        print(f"- 총 콜 수: {n}건" + ("" if n >= 20 else " (표본 20건 미만 — 참고만)"))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    po = sub.add_parser("open")
    po.add_argument("--market", required=True, choices=["kr", "us", "coin"])
    po.add_argument("--direction", required=True, choices=["상승", "하락", "보합", "혼조"])
    po.add_argument("--call_date", required=True)
    po.add_argument("--instrument", required=True)
    po.add_argument("--entry_price", required=True)
    po.add_argument("--entry_note", default="")
    po.set_defaults(func=cmd_open)

    pc = sub.add_parser("close")
    pc.add_argument("--market", required=True, choices=["kr", "us", "coin"])
    pc.add_argument("--call_date", required=True, help="닫을 open 행의 call_date (연다고 기록했던 그 날짜)")
    pc.add_argument("--exit_price", required=True)
    pc.add_argument("--exit_date", default="")
    pc.add_argument("--exit_note", default="")
    pc.set_defaults(func=cmd_close)

    ps = sub.add_parser("summary")
    ps.set_defaults(func=cmd_summary)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
