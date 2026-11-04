"""Отчёт: текст в консоль и самодостаточный HTML с графиком."""
from __future__ import annotations

import html

import pandas as pd

from .model import Prediction


def verdict(bt: dict) -> str:
    gain = 1 - bt["mae_model"] / bt["mae_naive"]
    if gain > 0.02 and bt["direction_accuracy"] > 0.53:
        return f"модель лучше наивного прогноза на {gain:.1%} — прогноз полезен"
    if gain > 0:
        return f"модель лишь немного лучше наивного прогноза ({gain:.1%}) — относитесь осторожно"
    return "модель НЕ лучше прогноза «цена не изменится» — прогнозу не доверять"


def text_report(ticker: str, p: Prediction, intraday: dict, news: pd.DataFrame, po: Prediction | None = None) -> str:
    bt = p.backtest
    arrow = "▲" if p.predicted_return > 0 else "▼"
    lines = [
        f"=== {ticker}: прогноз на следующую торговую сессию ===",
        f"Последнее закрытие ({p.last_date}): ${p.last_close:.2f}",
    ]
    if intraday:
        lines.append(
            f"Последняя сделка ({intraday['last_trade_time']}): ${intraday['last_price']:.2f}, "
            f"VWAP ${intraday['vwap']:.2f}, давление покупок {intraday['buy_pressure']:.0%}"
        )
    lines += [
        *([f"Прогноз открытия: ${po.predicted_price:.2f}  {po.predicted_return * 100:+.2f}%  "
           f"(80%: ${po.low_price:.2f} – ${po.high_price:.2f}; на истории ошибка {po.backtest['mae_model'] * 100:.2f}%"
           f" vs {po.backtest['mae_naive'] * 100:.2f}% у «открытие = вчерашнее закрытие»)"] if po else []),
        f"Прогноз закрытия: ${p.predicted_price:.2f}  {arrow} {p.predicted_return * 100:+.2f}%",
        f"Диапазон 80%:     ${p.low_price:.2f} – ${p.high_price:.2f}",
        f"Вероятность роста: {p.prob_up:.0%}",
        "",
        f"Проверка на истории ({bt['days']} дней, walk-forward):",
        f"  средняя ошибка модели {bt['mae_model'] * 100:.2f}% vs «цена не изменится» {bt['mae_naive'] * 100:.2f}%",
        f"  угадано направление: регрессия {bt['direction_accuracy']:.0%}, классификатор {bt['classifier_accuracy']:.0%}"
        f" (доля дней роста {bt['share_up_days']:.0%})",
        f"  Brier вероятности роста: {bt['brier']:.4f} vs {bt['brier_naive']:.4f} у константы (меньше — лучше)",
        f"  вердикт: {verdict(bt)}",
        "",
        f"Обучено на {p.n_train_days} днях, {p.n_features} признаков. Модели (вес · прогноз):",
        *[f"  {k:6s} {p.weights[k]:5.1%} · {p.per_model[k] * 100:+.2f}%  параметры {p.params[k] or 'по умолчанию'}"
          for k in p.weights],
        "",
        "Самые важные признаки: " + ", ".join(k for k, _ in p.top_features[:5]),
    ]
    if not news.empty:
        lines += ["", "Последние новости:"]
        for _, r in news.tail(8).iloc[::-1].iterrows():
            lines.append(f"  [{r['sentiment']:+.2f}] {r['time']:%Y-%m-%d} {r['title'][:110]}")
    lines += ["", "Это статистическая оценка, а не инвестиционная рекомендация."]
    return "\n".join(lines)


def _svg_chart(closes: pd.Series, p: Prediction, w: int = 760, h: int = 300) -> str:
    vals = list(closes.values) + [p.low_price, p.high_price]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    n = len(closes) + 1

    def x(i):
        return 40 + i * (w - 60) / (n - 1)

    def y(v):
        return h - 25 - (v - lo) * (h - 45) / (hi - lo)

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(closes.values))
    xl, xp = x(n - 2), x(n - 1)
    color = "var(--up)" if p.predicted_return > 0 else "var(--down)"
    ticks = "".join(
        f'<text x="4" y="{y(v) + 4:.1f}" class="t">{v:.0f}</text>'
        f'<line x1="36" x2="{w - 16}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="g"/>'
        for v in [lo + (hi - lo) * k / 4 for k in range(5)]
    )
    return f"""<svg viewBox="0 0 {w} {h}" role="img" aria-label="График цены и прогноз">
{ticks}
<polyline points="{pts}" fill="none" stroke="var(--line)" stroke-width="2"/>
<polygon points="{xl:.1f},{y(p.last_close):.1f} {xp:.1f},{y(p.high_price):.1f} {xp:.1f},{y(p.low_price):.1f}" fill="{color}" opacity=".18"/>
<line x1="{xl:.1f}" y1="{y(p.last_close):.1f}" x2="{xp:.1f}" y2="{y(p.predicted_price):.1f}" stroke="{color}" stroke-width="2" stroke-dasharray="5 4"/>
<circle cx="{xp:.1f}" cy="{y(p.predicted_price):.1f}" r="5" fill="{color}"/>
<text x="{xp - 6:.1f}" y="{y(p.predicted_price) - 10:.1f}" text-anchor="end" class="t b">${p.predicted_price:.2f}</text>
<text x="40" y="{h - 6}" class="t">{closes.index[0]:%Y-%m-%d}</text>
<text x="{xl:.1f}" y="{h - 6}" text-anchor="end" class="t">{closes.index[-1]:%Y-%m-%d}</text>
</svg>"""


def html_report(
    ticker: str, p: Prediction, daily: pd.DataFrame, intraday: dict, news: pd.DataFrame, po: Prediction | None = None,
    extra: str = "",
) -> str:
    bt = p.backtest
    sign = "up" if p.predicted_return > 0 else "down"
    news_rows = "".join(
        f"<tr><td>{r['time']:%Y-%m-%d %H:%M}</td><td class='{'up' if r['sentiment'] > 0.05 else 'down' if r['sentiment'] < -0.05 else ''}'>"
        f"{r['sentiment']:+.2f}</td><td>{html.escape(str(r['title']))}</td></tr>"
        for _, r in news.tail(25).iloc[::-1].iterrows()
    ) or "<tr><td colspan=3>Новостей нет</td></tr>"
    trade = (
        f"<div class='card'><div class='k'>Последняя сделка</div><div class='v'>${intraday['last_price']:.2f}</div>"
        f"<div class='s'>VWAP ${intraday['vwap']:.2f} · покупки {intraday['buy_pressure']:.0%} · {intraday['minutes']} мин</div></div>"
        if intraday else ""
    )
    open_card = (
        f"<div class='card'><div class='k'>Прогноз открытия</div><div class='v'>${po.predicted_price:.2f}</div>"
        f"<div class='s'>{po.predicted_return * 100:+.2f}% · 80%: ${po.low_price:.2f}–{po.high_price:.2f}</div></div>"
        if po else ""
    )
    models = "".join(
        f"<tr><td>{k}</td><td>вес {p.weights[k]:.0%}</td><td>{p.per_model[k] * 100:+.2f}%</td>"
        f"<td class='s'>{html.escape(str(p.params[k] or 'по умолчанию'))}</td></tr>"
        for k in p.weights
    )
    feats = "".join(f"<li>{html.escape(k)} <span class='s'>{v:.1%}</span></li>" for k, v in p.top_features)
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{ticker} прогноз цены</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--border:#e4e3df;--line:#3a5a8c;--up:#1d8a5a;--down:#c2412d}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161616;--fg:#ecebe7;--muted:#9a9993;--card:#1f1f1f;--border:#333;--line:#8fb0e6;--up:#4cc28c;--down:#f07a64}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:820px;margin:auto}} h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}}
.s,.t{{color:var(--muted);font-size:13px}} .t{{fill:var(--muted);font-size:11px}} .b{{font-weight:600;fill:var(--fg)}}
.g{{stroke:var(--border)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:16px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 14px}}
.k{{color:var(--muted);font-size:13px}} .v{{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}}
.up{{color:var(--up)}} .down{{color:var(--down)}} svg{{width:100%;height:auto;margin-top:12px}}
table{{width:100%;border-collapse:collapse;font-size:14px}} td{{padding:6px 4px;border-top:1px solid var(--border);vertical-align:top}}
td:first-child{{white-space:nowrap;color:var(--muted)}} ul{{padding-left:18px}}
</style></head><body><main>
<h1>{ticker}: следующая цена</h1>
<div class="s">Данные по {p.last_date}. Статистическая оценка, не инвестиционная рекомендация.</div>
<div class="grid">
{open_card}
<div class="card"><div class="k">Прогноз закрытия</div><div class="v {sign}">${p.predicted_price:.2f}</div><div class="s {sign}">{p.predicted_return * 100:+.2f}% к ${p.last_close:.2f}</div></div>
<div class="card"><div class="k">Диапазон 80%</div><div class="v">${p.low_price:.0f}–{p.high_price:.0f}</div><div class="s">${p.low_price:.2f} – ${p.high_price:.2f}</div></div>
<div class="card"><div class="k">Вероятность роста</div><div class="v">{p.prob_up:.0%}</div><div class="s">по ошибкам модели</div></div>
{trade}
</div>
{_svg_chart(daily["Close"].tail(90), p)}
<h2>Насколько модели можно верить</h2>
<p>Walk-forward проверка на последних {bt['days']} днях: средняя ошибка <b>{bt['mae_model'] * 100:.2f}%</b>
против <b>{bt['mae_naive'] * 100:.2f}%</b> у наивного прогноза «цена не изменится».
Направление угадано в <b>{bt['direction_accuracy']:.0%}</b> случаев регрессией и в <b>{bt['classifier_accuracy']:.0%}</b>
классификатором (доля дней роста — {bt['share_up_days']:.0%}).</p>
<p><b>Вердикт:</b> {html.escape(verdict(bt))}.</p>
<h2>Модели ансамбля</h2>
<div class="s">Обучено на {p.n_train_days} днях, {p.n_features} признаков. Гиперпараметры подобраны на данных до периода проверки.</div>
<table>{models}</table>
<h2>Что влияет сильнее всего</h2><ul>{feats}</ul>
{extra}
<h2>Новости и тональность</h2><table>{news_rows}</table>
</main></body></html>"""


def _pct(v: float) -> str:
    return "—" if pd.isna(v) else f"{v * 100:+.1f}%"


def next_day_tags(next_day: pd.Timestamp) -> list[str]:
    """К каким календарным группам относится прогнозируемый день."""
    tags = []
    if next_day.day == 30:
        tags.append("30-е число")
    if next_day.day == 1:
        tags.append("1-е число")
    nb = next_day + pd.offsets.BDay()
    if nb.month != next_day.month:
        tags.append("последний день месяца")
        if next_day.month in (3, 6, 9, 12):
            tags.append("конец квартала")
    pb = next_day - pd.offsets.BDay()
    if pb.month != next_day.month:
        tags.append("первый день месяца")
        if next_day.month in (1, 4, 7, 10):
            tags.append("начало квартала")
    return tags


def events_text(next_day: pd.Timestamp, cal: pd.DataFrame, scen: dict | None) -> str:
    lines = ["", f"=== Особенности дня {next_day:%Y-%m-%d} ==="]
    tags = next_day_tags(next_day)
    for t in tags:
        r = cal.loc[t]
        lines.append(f"  {t}: рост в {r['доля роста']:.0%} из {r['дней']} дней, медиана {_pct(r['медиана'])}"
                     f" (обычный день {_pct(r['прочие дни'])}), p≈{r['p (случайность)']:.2f}")
    if not tags:
        lines.append("  обычный день календаря")
    if any(r["p (случайность)"] < 0.05 for _, r in cal.iterrows()):
        lines.append("  есть статистически значимые календарные эффекты (p<0.05)")
    else:
        lines.append("  календарные эффекты NXT статистически не доказаны (все p ≥ 0.05) — модель их не учитывает")
    if scen:
        h = scen["history"]
        pc = scen["prev_close"]
        lines += ["", f"!!! Следующий день — ПОСЛЕ ОТЧЁТА КОМПАНИИ. Обычная модель тут не работает.",
                  f"  История ({h['n']} отчётов): рост в {h['up_share']:.0%}, к закрытию откат от открытия в {h['fade_share']:.0%}",
                  f"  Открытие (25/50/75%): " + " / ".join(f"${v:.2f}" for v in h["open_q"].values()),
                  f"  Закрытие (25/50/75%): " + " / ".join(f"${v:.2f}" for v in h["close_q"].values()),
                  f"  Похожие по картине перед отчётом: открытие ~${scen['similar_open']:.2f}, закрытие ~${scen['similar_close']:.2f}"]
        for x, r in scen["similar"].iterrows():
            lines.append(f"    {x:%Y-%m-%d}: гэп {_pct(r['gap'])}, итог {_pct(r['ret'])} → открытие ${pc * (1 + r['gap']):.2f},"
                         f" закрытие ${pc * (1 + r['ret']):.2f}")
        a = scen.get("after_hours")
        if a:
            lines += [f"  After hours ${a['price']:.2f} ({_pct(a['pct'])}), по {a['n']} известным случаям:",
                      f"    открытие ~${a['open']['mid']:.2f} (${a['open']['lo']:.2f}–{a['open']['hi']:.2f}), связь r={a['open']['corr']:.2f}",
                      f"    закрытие ~${a['close']['mid']:.2f} (${a['close']['lo']:.2f}–{a['close']['hi']:.2f}), связь r={a['close']['corr']:.2f}"]
            for x, r in a["nearest"].iterrows():
                lines.append(f"    похожий after hours {x:%Y-%m-%d}: AH {_pct(r['after_hours'])} → гэп {_pct(r['gap'])}, итог {_pct(r['ret'])}")
    return "\n".join(lines)


def events_html(cal: pd.DataFrame, reactions: pd.DataFrame, qturn: pd.DataFrame, scen: dict | None) -> str:
    cal_rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{int(r['дней'])}</td><td>{r['доля роста']:.0%}</td><td>{_pct(r['медиана'])}</td>"
        f"<td>{_pct(r['прочие дни'])}</td><td class='s'>{r['p (случайность)']:.2f}</td></tr>"
        for n, r in cal.iterrows()
    )
    if reactions.empty:
        reactions = pd.DataFrame(columns=["after_hours", "gap", "intra", "ret", "week"])
    rx_rows = "".join(
        f"<tr><td>{x:%Y-%m-%d}</td><td>{_pct(r['after_hours'])}</td><td>{_pct(r['gap'])}</td><td>{_pct(r['intra'])}</td>"
        f"<td class='{'up' if r['ret'] > 0 else 'down'}'>{_pct(r['ret'])}</td><td>{_pct(r['week'])}</td></tr>"
        for x, r in reactions.iloc[::-1].iterrows()
    )
    q_rows = "".join(
        f"<tr><td>{x:%Y-%m-%d}</td><td class='{'up' if r['итог дня'] > 0 else 'down'}'>{_pct(r['итог дня'])}</td>"
        f"<td>{_pct(r['след. день'])}</td></tr>"
        for x, r in qturn.iloc[::-1].iterrows()
    )
    block = ""
    if scen:
        h = scen["history"]
        block = (f"<div class='card' style='margin-top:16px'><b>Следующий день — после отчёта компании.</b> "
                 f"По {h['n']} прошлым отчётам: открытие ~${h['open_q'][0.5]:.2f}, закрытие ~${h['close_q'][0.5]:.2f}; "
                 f"похожие случаи: ${scen['similar_open']:.2f} / ${scen['similar_close']:.2f}.")
        if scen.get("after_hours"):
            a = scen["after_hours"]
            block += (f" С учётом after hours ${a['price']:.2f}: открытие ~${a['open']['mid']:.2f}, "
                      f"закрытие ~${a['close']['mid']:.2f} (${a['close']['lo']:.2f}–{a['close']['hi']:.2f}).")
        block += "</div>"
    return f"""{block}
<h2>Календарь</h2><div class="s">p — вероятность получить такую долю роста случайно; эффект надёжен при p&lt;0.05.</div>
<table><tr><td></td><td>дней</td><td>рост</td><td>медиана</td><td>обычный день</td><td>p</td></tr>{cal_rows}</table>
<h2>Реакция на отчёты компании</h2><div class="s">День после отчёта: after hours, гэп на открытии, движение внутри дня, итог, неделя.</div>
<table><tr><td></td><td>AH</td><td>гэп</td><td>внутри дня</td><td>итог</td><td>неделя</td></tr>{rx_rows}</table>
<h2>Конец квартала → следующий день</h2><table>{q_rows}</table>"""
