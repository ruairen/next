"""CLI: python -m nxt_predictor [--ticker NXT] [--demo] [--prices-csv FILE] [--news-csv FILE]"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from . import data
from .features import build_features, intraday_summary, score_news
from . import events
from .model import calibrated_range, predict_next, replay
from .report import events_html, events_text, html_report, text_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Прогноз следующей цены акции по ценам, сделкам и новостям")
    ap.add_argument("--ticker", default="NXT")
    ap.add_argument("--company", default="Nextracker", help="название компании для поиска новостей")
    ap.add_argument("--years", type=int, default=10, help="сколько лет истории загрузить")
    ap.add_argument("--context", default=",".join(data.CONTEXT_TICKERS),
                    help="рынок/конкуренты через запятую; 'none' — не использовать")
    ap.add_argument("--backtest-days", type=int, default=250, help="длина честной проверки на истории")
    ap.add_argument("--fast", action="store_true", help="без подбора гиперпараметров (быстрее)")
    ap.add_argument("-v", "--verbose", action="store_true", help="показывать ход обучения")
    ap.add_argument("--demo", action="store_true", help="синтетические данные, без интернета")
    ap.add_argument("--prices-csv", help="свои дневные цены (Date,Open,High,Low,Close,Volume)")
    ap.add_argument("--context-csv", help="свои цены рынка/конкурентов (Date + колонка на тикер), без загрузки")
    ap.add_argument("--news-csv", help="свои новости (time,title[,summary])")
    ap.add_argument("--out", default="output", help="папка для отчётов")
    ap.add_argument("--replay", type=int, metavar="N",
                    help="переиграть последние N дней: что модель предсказала бы каждый вечер, и сравнить с фактом")
    ap.add_argument("--earnings-day", action="store_true",
                    help="следующая сессия — день после отчёта компании (определяется и автоматически)")
    ap.add_argument("--after-hours", type=float, metavar="PRICE",
                    help="цена в after hours / премаркете после отчёта — уточняет сценарий открытия и закрытия")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    context_tickers = [] if args.context.lower() == "none" else [t.strip().upper() for t in args.context.split(",") if t.strip()]
    if args.demo:
        daily, intraday, news = data.synthetic_data()
        context = data.synthetic_context(daily) if context_tickers else pd.DataFrame()
    else:
        daily = data.load_prices_csv(args.prices_csv) if args.prices_csv else data.fetch_daily_prices(args.ticker, args.years)
        news = data.load_news_csv(args.news_csv) if args.news_csv else data.fetch_news(args.ticker, args.company)
        try:
            intraday = data.fetch_intraday_trades(args.ticker)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Минутные сделки недоступны: %s", exc)
            intraday = pd.DataFrame()
        if args.context_csv:
            context = pd.read_csv(args.context_csv, index_col=0, parse_dates=True)
            context = context[[c for c in context.columns if c in context_tickers]] if context_tickers else pd.DataFrame()
        else:
            context = data.fetch_context(context_tickers, args.years) if context_tickers else pd.DataFrame()

    news = score_news(news)
    # Даты отчётов известны только для NXT: эти дни исключаются из обучения обычной модели.
    skip = events.EARNINGS_REACTION_DAYS if args.ticker.upper() == "NXT" else None
    features = build_features(daily, news, context, skip_target_days=skip)
    open_features = build_features(daily, news, context, target="open", skip_target_days=skip)
    if args.replay:
        return run_replay(args, daily, features, open_features, skip)
    print(f"Обучение на {len(features.dropna())} днях, {features.shape[1] - 1} признаков"
          f"{' (быстрый режим)' if args.fast else ', подбор гиперпараметров'}...", flush=True)
    last_close = float(daily["Close"].iloc[-1])
    pred = predict_next(features, last_close=last_close, backtest_days=args.backtest_days, fast=args.fast,
                        interval=calibrated_range(daily, "close", skip_target_days=skip))
    print("Прогноз цены открытия...", flush=True)
    pred_open = predict_next(open_features, last_close=last_close, backtest_days=args.backtest_days, fast=args.fast,
                             interval=calibrated_range(daily, "open", skip_target_days=skip))
    intra = intraday_summary(intraday)

    next_day = daily.index[-1].normalize() + pd.offsets.BDay()
    cal, qturn = events.calendar_stats(daily), events.quarter_turn(daily)
    # Даты отчётов известны только для NXT; для других тикеров раздел про отчёты пустой.
    reactions = events.earnings_reactions(daily) if args.ticker.upper() == "NXT" else pd.DataFrame()
    scen = None
    want = args.earnings_day or args.after_hours or str(next_day.date()) in events.EARNINGS_REACTION_DAYS
    if want and len(reactions) < 5:
        logging.warning("Мало данных об отчётах (%d) — сценарий дня после отчёта не построен", len(reactions))
    elif want:
        scen = events.earnings_scenarios(reactions, last_close, events.current_setup(daily), args.after_hours)

    print(text_report(args.ticker, pred, intra, news, pred_open))
    print(events_text(next_day, cal, scen))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.ticker}_report.html").write_text(html_report(args.ticker, pred, daily, intra, news, pred_open,
                                                                           events_html(cal, reactions, qturn, scen)), encoding="utf-8")
    summary = {k: v for k, v in pred.__dict__.items() if k != "backtest"}
    summary["backtest"] = {k: v for k, v in pred.backtest.items() if k != "series"}
    summary["intraday"] = intra
    summary["open"] = {
        "predicted_price": pred_open.predicted_price,
        "low_price": pred_open.low_price,
        "high_price": pred_open.high_price,
        "predicted_return": pred_open.predicted_return,
        "backtest": {k: v for k, v in pred_open.backtest.items() if k != "series"},
    }
    if scen:
        summary["earnings_day"] = {
            "similar_open": scen["similar_open"], "similar_close": scen["similar_close"],
            "history_close_median": scen["history"]["close_q"][0.5],
            **({"after_hours": {k: scen["after_hours"][k] for k in ("price", "pct", "open", "close")}}
               if scen.get("after_hours") else {}),
        }
    (out / f"{args.ticker}_prediction.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nОтчёт: {out / f'{args.ticker}_report.html'}")
    return 0


def run_replay(args, daily: pd.DataFrame, close_feats: pd.DataFrame, open_feats: pd.DataFrame, skip=None) -> int:
    print(f"Переигрываю последние {args.replay} дней (модель каждый вечер учится только на прошлом)...", flush=True)
    r = replay(daily, close_feats, open_feats, days=args.replay, fast=args.fast, skip_target_days=skip)
    print(f"\n=== {args.ticker}: переигровка последних {len(r)} дней ===")
    print(f"{'день':10s} {'закр.накануне':>13s} | {'откр.прогн':>10s} {'факт':>7s} {'ошибка':>7s} | "
          f"{'закр.прогн':>10s} {'факт':>7s} {'ошибка':>7s} {'P(рост)':>7s} {'напр.':>5s}")
    for d, x in r.iterrows():
        print(f"{d:%Y-%m-%d} {x.prev_close:13.2f} | {x.pred_open:10.2f} {x.open:7.2f} {x.open_err * 100:+6.1f}%"
              f"{'' if x.open_in_range else '!'} | {x.pred_close:10.2f} {x.close:7.2f} {x.close_err * 100:+6.1f}%"
              f"{'' if x.close_in_range else '!'} {x.prob_up:7.0%} {'✓' if x.dir_hit else '✗':>5s}")
    print(f"\nСредняя ошибка: открытие {r.open_err.abs().mean() * 100:.2f}%, закрытие {r.close_err.abs().mean() * 100:.2f}%"
          f" (у «цена не изменится» {r.naive_close_err.abs().mean() * 100:.2f}%)")
    print(f"Факт внутри диапазона 80%: открытие {r.open_in_range.mean():.0%}, закрытие {r.close_in_range.mean():.0%};"
          f" направление угадано {r.dir_hit.mean():.0%} ({int(r.dir_hit.sum())} из {len(r)})")
    print("! — факт вне диапазона 80%")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / f"{args.ticker}_replay.csv")
    print(f"\nТаблица: {out / f'{args.ticker}_replay.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
