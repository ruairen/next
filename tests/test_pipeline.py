import json

import numpy as np
import pandas as pd

from nxt_predictor import data
from nxt_predictor.__main__ import main
from nxt_predictor.features import build_features, context_features, daily_news_features, intraday_summary, sentiment
from nxt_predictor.model import predict_next, tune


def test_finance_sentiment_direction():
    assert sentiment("Nextracker beats earnings estimates, raises guidance") > 0.3
    assert sentiment("Nextracker shares fall after downgrade") < -0.3


def test_news_after_close_maps_to_same_day_and_weekend_to_friday():
    days = pd.bdate_range("2026-09-21", "2026-09-25")  # пн–пт
    news = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-09-22 21:30", "2026-09-26 15:00"], utc=True),  # вт после закрытия, сб
            "title": ["a", "b"],
            "summary": ["", ""],
            "sentiment": [0.5, -0.5],
        }
    )
    f = daily_news_features(news, days)
    assert f.loc["2026-09-22", "news_sent"] == 0.5
    assert f.loc["2026-09-25", "news_sent"] == -0.5
    assert f["news_count"].sum() == 2


def test_features_have_no_lookahead():
    daily, _, news = data.synthetic_data(days=300)
    f1 = build_features(daily, news)
    cut = daily.iloc[:-10]
    known = news[news["time"] < pd.Timestamp(cut.index[-1] + pd.Timedelta(days=1), tz="UTC")]
    f2 = build_features(cut, known)
    common = f2.drop(columns="target").dropna().index
    pd.testing.assert_frame_equal(
        f1.loc[common].drop(columns="target"), f2.loc[common].drop(columns="target")
    )
    assert np.isnan(f1["target"].iloc[-1])


def test_prediction_is_sane():
    daily, _, news = data.synthetic_data(days=400)
    from nxt_predictor.features import score_news

    feats = build_features(daily, score_news(news), data.synthetic_context(daily))
    p = predict_next(feats, float(daily["Close"].iloc[-1]), backtest_days=60, fast=True)
    assert p.low_price <= p.predicted_price <= p.high_price
    assert 0 <= p.prob_up <= 1
    assert abs(p.predicted_return) < 0.2
    assert abs(sum(p.weights.values()) - 1) < 1e-9
    assert set(p.weights) == {"ridge", "gbm", "hgb", "rf", "et"}
    assert 0 <= p.backtest["classifier_accuracy"] <= 1


def test_tuning_picks_from_grid():
    daily, _, news = data.synthetic_data(days=300)
    f = build_features(daily, news).dropna()
    params, weights, mae = tune(f.drop(columns="target"), f["target"], n_splits=2)
    assert params["gbm"]["max_depth"] in (2, 3)
    assert all(v > 0 for v in mae.values())
    assert abs(sum(weights.values()) - 1) < 1e-9


def test_context_features_no_lookahead():
    daily, _, _ = data.synthetic_data(days=300)
    ctx = data.synthetic_context(daily)
    f1 = context_features(daily, ctx)
    f2 = context_features(daily.iloc[:-5], ctx.iloc[:-5])
    common = f2.dropna().index
    pd.testing.assert_frame_equal(f1.loc[common], f2.loc[common])
    assert "beta_tan_60" in f1 and "rel_spy_5" in f1


def test_intraday_summary():
    _, intraday, _ = data.synthetic_data(days=50)
    s = intraday_summary(intraday)
    assert s["minutes"] == 390
    assert 0 <= s["buy_pressure"] <= 1


def test_cli_demo_writes_reports(tmp_path):
    assert main(["--demo", "--fast", "--backtest-days", "60", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "NXT_report.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    summary = json.loads((tmp_path / "NXT_prediction.json").read_text(encoding="utf-8"))
    assert summary["predicted_price"] > 0


def test_csv_inputs(tmp_path):
    daily, _, news = data.synthetic_data(days=300)
    daily.to_csv(tmp_path / "p.csv")
    news.to_csv(tmp_path / "n.csv", index=False)
    loaded = data.load_prices_csv(str(tmp_path / "p.csv"))
    assert len(loaded) == 300
    assert len(data.load_news_csv(str(tmp_path / "n.csv"))) == len(news)


def test_nasdaq_csv_format(tmp_path):
    p = tmp_path / "nasdaq.csv"
    p.write_text(
        "Date,Close/Last,Volume,Open,High,Low\n"
        "09/29/2026,$77.71,2345678,$76.50,$78.10,$76.20\n"
        "09/26/2026,$76.40,1987654,$75.90,$77.00,$75.10\n"
    )
    df = data.load_prices_csv(str(p))
    assert list(df.columns) == data.PRICE_COLUMNS
    assert df.index[0] == pd.Timestamp("2026-09-26")
    assert df["Close"].iloc[-1] == 77.71


def test_open_target_is_gap_to_next_open():
    daily, _, _ = data.synthetic_data(days=100)
    f = build_features(daily, target="open")
    expected = np.log(daily["Open"].iloc[1] / daily["Close"].iloc[0])
    assert abs(f["target"].iloc[0] - expected) < 1e-12
    assert np.isnan(f["target"].iloc[-1])


def _nxt_like_daily():
    """Синтетика с датами реальных отчётов NXT и сильными реакциями в эти дни."""
    from nxt_predictor import events

    idx = pd.bdate_range("2023-02-09", "2026-10-28")
    rng = np.random.default_rng(3)
    ret = rng.normal(0.0005, 0.03, len(idx))
    for i, x in enumerate(idx):
        if str(x.date()) in events.EARNINGS_REACTION_DAYS:
            ret[i] = 0.10
    close = 40 * np.exp(np.cumsum(ret))
    open_ = close / np.exp(ret) * np.exp(np.where(ret == 0.10, 0.08, 0.0))
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.01,
                         "Low": np.minimum(open_, close) * 0.99, "Close": close,
                         "Volume": rng.lognormal(14, 0.3, len(idx))}, index=idx.rename("Date"))


def test_calendar_ignores_unfinished_month_at_end():
    from nxt_predictor import events

    daily = _nxt_like_daily().loc[:"2026-09-29"]  # сентябрь не закончен
    qt = events.quarter_turn(daily)
    assert pd.Timestamp("2026-09-29") not in qt.index
    assert pd.Timestamp("2026-06-30") in qt.index
    cal = events.calendar_stats(daily)
    assert {"30-е число", "конец квартала", "начало квартала"} <= set(cal.index)
    assert ((cal["p (случайность)"] > 0) & (cal["p (случайность)"] <= 1)).all()


def test_earnings_reactions_and_scenarios():
    from nxt_predictor import events

    daily = _nxt_like_daily()
    rx = events.earnings_reactions(daily)
    assert pd.Timestamp("2024-08-02") in rx.index and pd.Timestamp("2024-07-31") not in rx.index
    assert np.allclose(rx["ret"], np.expm1(0.10))
    assert rx["after_hours"].notna().sum() >= 10
    pc = float(daily["Close"].iloc[-1])
    scen = events.earnings_scenarios(rx, pc, events.current_setup(daily), after_hours=pc * 1.10)
    assert abs(scen["history"]["close_q"][0.5] / pc - np.exp(0.10)) < 1e-9
    assert len(scen["similar"]) == 6
    a = scen["after_hours"]
    assert a["open"]["lo"] <= a["open"]["mid"] <= a["open"]["hi"]
    assert abs(a["pct"] - 0.10) < 1e-12


def test_cli_earnings_day_with_after_hours(tmp_path):
    daily = _nxt_like_daily()
    daily.to_csv(tmp_path / "p.csv")
    news = pd.DataFrame({"time": [], "title": []})
    news.to_csv(tmp_path / "n.csv", index=False)
    assert main(["--prices-csv", str(tmp_path / "p.csv"), "--news-csv", str(tmp_path / "n.csv"),
                 "--context", "none", "--fast", "--backtest-days", "40",
                 "--after-hours", str(float(daily["Close"].iloc[-1]) * 1.05), "--out", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "NXT_prediction.json").read_text(encoding="utf-8"))
    assert summary["earnings_day"]["after_hours"]["pct"] > 0.04
    assert "Реакция на отчёты компании" in (tmp_path / "NXT_report.html").read_text(encoding="utf-8")


def test_sparse_news_not_used_as_features():
    daily, _, news = data.synthetic_data(days=300)
    assert "news_sent" in build_features(daily, news).columns
    assert "news_sent" not in build_features(daily, news.tail(3)).columns


def test_skip_target_days_removes_those_targets():
    daily, _, _ = data.synthetic_data(days=120)
    skip = [daily.index[50], daily.index[80]]
    f = build_features(daily, skip_target_days=skip)
    assert np.isnan(f["target"].iloc[49]) and np.isnan(f["target"].iloc[79])
    assert not np.isnan(f["target"].iloc[50])


def test_calibrated_range_reaches_target_coverage():
    from nxt_predictor.model import calibrated_range

    daily, _, _ = data.synthetic_data(days=900)
    r = calibrated_range(daily, "close")
    assert r["lo"] < 0 < r["hi"]
    assert r["coverage"] >= 0.78 and r["n"] == 250
    ro = calibrated_range(daily, "open")
    assert ro["lo"] < 0 < ro["hi"]


def test_replay_uses_only_past_and_reports_errors():
    from nxt_predictor.model import replay

    daily, _, _ = data.synthetic_data(days=220)
    r = replay(daily, build_features(daily), build_features(daily, target="open"), days=2, fast=True)
    assert list(r.index) == list(daily.index[-2:])
    assert np.allclose(r["close"], daily["Close"].iloc[-2:])
    assert (r["close_lo"] <= r["pred_close"]).all() and (r["pred_close"] <= r["close_hi"]).all()
    # прогноз на предпоследний день не меняется, если убрать из данных последний день
    r2 = replay(daily.iloc[:-1], build_features(daily.iloc[:-1]), build_features(daily.iloc[:-1], target="open"),
                days=1, fast=True)
    assert abs(r2["pred_close"].iloc[0] - r["pred_close"].iloc[0]) < 1e-9
