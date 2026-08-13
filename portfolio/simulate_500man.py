#!/usr/bin/env python3
"""
"클대리가 진짜 500만원을 투자했다면" 시뮬레이션.

portfolio/ledger.csv의 실제 콜(상승/하락/보합/혼조)과 실제 결과(position_return_pct)를
그대로 사용해서, 500만원을 국내/미국/코인 3개 시장에 균등 분할(각 1,666,667원)했을 때
매일 콜에 따라 전액 매수/매도(청산 후 재투자)했다고 가정하고 잔고를 복리로 추적한다.

포지션 매핑은 portfolio/policy.md와 동일하게 따른다:
- 상승 콜 -> 매수(롱). 하락 콜 -> 인버스 상품 매수(숏, 국내 인버스 ETF/미국 인버스 ETF/코인 선물 숏 등
  실제 리테일 투자자도 접근 가능한 상품을 가정). 보합/혼조 콜 -> 현금 보유(포지션 없음).
- 매일 계좌 잔고 전액을 그날 콜에 베팅하고(몰빵), 청산 후 다음날 다시 그날 콜에 베팅하는
  가장 단순한 정책이다. 분산투자나 부분 매수 같은 리스크 관리는 하지 않는다 - "규칙을 그대로
  따랐을 때 실제로 얼마를 벌거나 잃었을지"를 보여주는 게 목적이다.

실행:
  python simulate_500man.py

결과:
  portfolio/500만원_시뮬레이션.md - 거래 로그 + 시장별/전체 최종 잔고
  portfolio/simulation_report.html - 그래프+포지션 현황 시각화 리포트 (GitHub Pages로 열람)

전략 노트: portfolio/strategy_note.txt 파일이 있으면(자동화 루틴이 매일 갱신) 그 내용을
리포트 상단 "오늘의 전략 노트"에 그대로 반영한다. 없으면 그 섹션은 생략된다.
"""
import csv
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER_PATH = os.path.join(HERE, "ledger.csv")
OUT_PATH = os.path.join(HERE, "500만원_시뮬레이션.md")
TEMPLATE_PATH = os.path.join(HERE, "simulation_report_template.html")
HTML_OUT_PATH = os.path.join(HERE, "simulation_report.html")
NOTE_PATH = os.path.join(HERE, "strategy_note.txt")

START_CAPITAL = 5_000_000
MARKETS = {"kr": "국내", "us": "미국", "coin": "코인"}
START_PER_MARKET = START_CAPITAL // len(MARKETS)


def load_all_rows():
    """market -> 전체 행(open 포함) 리스트, call_date/id 순 정렬"""
    rows = {m: [] for m in MARKETS}
    with open(LEDGER_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = row["market"]
            if m not in rows:
                continue
            rows[m].append(row)
    for m in rows:
        rows[m].sort(key=lambda r: (r["call_date"], int(r["id"])))
    return rows


def load_closed_rows(all_rows):
    return {m: [r for r in rows if r["status"] == "closed"] for m, rows in all_rows.items()}


def current_status(all_rows):
    """시장별 가장 최근 행 기준 현재 포지션 상태(진입 중/청산 완료)."""
    status = {}
    for m, rows in all_rows.items():
        if not rows:
            status[m] = None
            continue
        last = rows[-1]
        status[m] = {
            "status": last["status"],
            "direction": last["direction"],
            "call_date": last["call_date"],
            "instrument": last["instrument"],
            "entry": last["entry_price"],
        }
    return status


def simulate(rows):
    results = {}
    for m, label in MARKETS.items():
        balance = START_PER_MARKET
        log = []
        for r in rows[m]:
            ret_pct = float(r["position_return_pct"])
            before = balance
            gain = before * (ret_pct / 100)
            balance = before + gain
            log.append({
                "date": r["call_date"],
                "direction": r["direction"],
                "instrument": r["instrument"],
                "entry": r["entry_price"],
                "exit": r["exit_price"],
                "actual_move_pct": float(r["actual_move_pct"]),
                "position_return_pct": ret_pct,
                "balance_before": before,
                "gain_krw": gain,
                "balance_after": balance,
            })
        results[m] = {"label": label, "log": log, "final_balance": balance}
    return results


def won(n):
    return f"{round(n):,}원"


def build_html(results, current, note):
    payload = {
        "markets": {
            m: {
                "label": r["label"],
                "log": [
                    {
                        "date": t["date"],
                        "direction": t["direction"],
                        "instrument": t["instrument"],
                        "entry": t["entry"],
                        "exit": t["exit"],
                        "actual_move_pct": t["actual_move_pct"],
                        "position_return_pct": t["position_return_pct"],
                        "gain_krw": round(t["gain_krw"]),
                        "balance_after": round(t["balance_after"]),
                    }
                    for t in r["log"]
                ],
                "final_balance": round(r["final_balance"]),
            }
            for m, r in results.items()
        },
        "current": current,
        "note": note,
    }
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()
    html = template.replace("__SIM_DATA__", json.dumps(payload, ensure_ascii=False))
    with open(HTML_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"저장: {HTML_OUT_PATH}")


def main():
    all_rows = load_all_rows()
    rows = load_closed_rows(all_rows)
    current = current_status(all_rows)
    note = ""
    if os.path.exists(NOTE_PATH):
        with open(NOTE_PATH, encoding="utf-8") as f:
            note = f.read().strip()
    results = simulate(rows)

    total_final = sum(r["final_balance"] for r in results.values())
    total_profit = total_final - START_CAPITAL

    lines = []
    lines.append("# 클대리의 500만원 실전 투자 시뮬레이션\n")
    lines.append("> 실제 돈이 아닙니다. `portfolio/ledger.csv`에 이미 기록된 실제 콜과 실제 결과를 그대로 재생해, "
                  "'만약 500만원을 진짜로 이 규칙대로 굴렸다면 얼마가 됐을까'를 계산한 사후 시뮬레이션입니다. "
                  "실제 투자 자문이 아니며, 매일 전액을 그날 콜에 베팅하는(몰빵) 가장 단순하고 공격적인 정책을 "
                  "가정했다는 점에 유의하세요 — 실제 투자라면 이렇게 하지 않는 게 정상입니다.\n")
    lines.append(f"**시작 자본**: {won(START_CAPITAL)} (국내/미국/코인 각 {won(START_PER_MARKET)}씩 균등 분할)\n")
    lines.append(f"**기준 기간**: {min(r['call_date'] for m in rows.values() for r in m)} ~ "
                 f"{max(r['call_date'] for m in rows.values() for r in m)} (아직 청산 안 된 진행 중인 포지션은 제외)\n")

    lines.append("## 최종 결과\n")
    lines.append("| 시장 | 시작 자본 | 최종 잔고 | 손익 | 수익률 |")
    lines.append("|---|---|---|---|---|")
    for m, label in MARKETS.items():
        r = results[m]
        profit = r["final_balance"] - START_PER_MARKET
        pct = profit / START_PER_MARKET * 100
        sign = "+" if profit >= 0 else ""
        lines.append(f"| {label} | {won(START_PER_MARKET)} | {won(r['final_balance'])} | "
                     f"{sign}{won(profit)} | {sign}{pct:.1f}% |")
    total_pct = total_profit / START_CAPITAL * 100
    sign = "+" if total_profit >= 0 else ""
    lines.append(f"| **합계** | **{won(START_CAPITAL)}** | **{won(total_final)}** | "
                 f"**{sign}{won(total_profit)}** | **{sign}{total_pct:.1f}%** |\n")

    for m, label in MARKETS.items():
        r = results[m]
        log = r["log"]
        wins = sum(1 for t in log if t["gain_krw"] > 0)
        losses = sum(1 for t in log if t["gain_krw"] < 0)
        flat = sum(1 for t in log if t["gain_krw"] == 0)
        best = max(log, key=lambda t: t["gain_krw"]) if log else None
        worst = min(log, key=lambda t: t["gain_krw"]) if log else None

        lines.append(f"## {label} 거래 로그 ({len(log)}건: 이익 {wins} / 손실 {losses} / 보합·관망 {flat})\n")
        if best:
            lines.append(f"- 최고의 하루: {best['date']} {best['direction']} 콜, "
                         f"실제 변동 {best['actual_move_pct']:+.2f}% -> {won(best['gain_krw'])} 이익\n")
        if worst and worst["gain_krw"] < 0:
            lines.append(f"- 최악의 하루: {worst['date']} {worst['direction']} 콜, "
                         f"실제 변동 {worst['actual_move_pct']:+.2f}% -> {won(worst['gain_krw'])} 손실\n")
        lines.append("| 날짜 | 콜 | 실제 변동 | 포지션 수익률 | 손익 | 잔고 |")
        lines.append("|---|---|---|---|---|---|")
        for t in log:
            sign_g = "+" if t["gain_krw"] >= 0 else ""
            lines.append(f"| {t['date']} | {t['direction']} | {t['actual_move_pct']:+.2f}% | "
                         f"{t['position_return_pct']:+.2f}% | {sign_g}{won(t['gain_krw'])} | {won(t['balance_after'])} |")
        lines.append("")

    lines.append("## 해석 시 주의\n")
    lines.append("- 매일 전액 몰빵을 가정한 가장 공격적인 정책입니다. 실제로 이렇게 투자하면 변동성(MDD)이 "
                 "매우 커집니다 - 위 표의 잔고가 도중에 크게 오르내리는 걸 그대로 보여줍니다.\n")
    lines.append("- 하락 콜은 인버스 상품(숏) 매수를 가정합니다. 국내는 인버스 ETF, 코인은 선물 숏 등으로 "
                 "실제 접근 가능하지만, 레버리지/수수료/슬리피지는 반영하지 않았습니다.\n")
    lines.append("- 표본이 아직 20건(시장별)이 안 되는 시장도 있어 `portfolio/policy.md` 기준으로는 "
                 "이 결과를 '규칙이 좋다/나쁘다'로 해석하기엔 이릅니다 - 단지 '지금까지 실제로 이랬다'는 기록입니다.\n")
    lines.append("- 실험/연구 목적이며 투자 자문이 아닙니다.\n")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    build_html(results, current, note)

    print(f"저장: {OUT_PATH}")
    print(f"최종 합계: {won(total_final)} ({sign}{won(total_profit)}, {sign}{total_pct:.1f}%)")
    for m, label in MARKETS.items():
        r = results[m]
        profit = r["final_balance"] - START_PER_MARKET
        print(f"  {label}: {won(r['final_balance'])} ({'+' if profit>=0 else ''}{won(profit)})")


if __name__ == "__main__":
    main()
