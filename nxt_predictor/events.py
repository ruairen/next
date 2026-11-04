"""События: календарные эффекты и реакция акции на квартальные отчёты компании.

Обычная модель (model.py) не знает о датах отчётов, а в день после отчёта NXT
двигается в 5–10 раз сильнее обычного. Здесь собрана статистика по этим дням.
"""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

# Дни ПОСЛЕ отчёта (Nextracker/Nextpower отчитывается после закрытия торгов).
# Даты сверены по пресс-релизам и всплескам объёма. Добавляйте новые по мере выхода.
EARNINGS_REACTION_DAYS = [
    "2023-05-11", "2023-07-27", "2023-10-26", "2024-02-01", "2024-05-15",
    "2024-08-02",  # отчёт 1 августа 2024, не 30 июля
    "2024-10-31", "2025-01-29", "2025-05-15", "2025-07-30", "2025-10-24",
    "2026-01-28", "2026-05-13", "2026-07-31", "2026-10-29",
]

# Движение в after hours вечером в день отчёта, % к закрытию (из новостей:
# Benzinga, CNBC, Investing.com). Неизвестные пропущены, спорные помечены в README.
AFTER_HOURS = {
    "2024-02-01": 18.2, "2024-05-15": 19.0, "2024-08-02": -4.0, "2025-01-29": 16.6,
    "2025-05-15": 3.1, "2025-07-30": -1.0, "2025-10-24": 9.3, "2026-01-28": 11.8,
    "2026-05-13": 10.5, "2026-07-31": 10.3,
    "2026-10-29": 0.0,  # вечером 28.10 скачок до $87, к концу after hours назад к $82
}


def _p_at_least(k: int, n: int) -> float:
    """Вероятность получить ≥k «успехов» из n при монетке 50/50."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2**n


def _with_returns(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["ret"] = d["Close"].pct_change()
    d["gap"] = d["Open"] / d["Close"].shift() - 1
    d["intra"] = d["Close"] / d["Open"] - 1
    return d


def _period_edges(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Маски первого и последнего торгового дня месяца. Незавершённые месяцы
    на краях данных не считаются (иначе последний день файла выглядит концом месяца)."""
    ym = pd.Series(index.to_period("M"), index=index)
    first = (~ym.duplicated(keep="first")).to_numpy(copy=True)
    last = (~ym.duplicated(keep="last")).to_numpy(copy=True)
    if len(index):
        first[(ym == ym.iloc[0]).to_numpy()] = False
        month_end = index[-1] + pd.offsets.BMonthEnd(0)
        if index[-1].normalize() < month_end.normalize():
            last[(ym == ym.iloc[-1]).to_numpy()] = False
    return first, last


def calendar_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Как акция ведёт себя в особые дни календаря по сравнению с остальными."""
    d = _with_returns(daily).dropna(subset=["ret"])
    first, last = _period_edges(d.index)
    q = d.index.month.isin([3, 6, 9, 12])
    qs = d.index.month.isin([1, 4, 7, 10])
    groups = {
        "30-е число": d.index.day == 30,
        "1-е число": d.index.day == 1,
        "последний день месяца": last,
        "первый день месяца": first,
        "конец квартала": last & q,
        "начало квартала": first & qs,
    }
    rows = []
    for name, m in groups.items():
        s, o = d[m], d[~m]
        n, k = len(s), int((s["ret"] > 0).sum())
        rows.append({
            "группа": name, "дней": n, "доля роста": k / n if n else np.nan,
            "средн.": s["ret"].mean(), "медиана": s["ret"].median(), "прочие дни": o["ret"].mean(),
            "гэп": s["gap"].mean(), "внутри дня": s["intra"].mean(),
            "p (случайность)": _p_at_least(max(k, n - k), n) if n else np.nan,
        })
    return pd.DataFrame(rows).set_index("группа")


def quarter_turn(daily: pd.DataFrame) -> pd.DataFrame:
    """Конец квартала и первый день следующего: исторически NXT часто разворачивается."""
    d = _with_returns(daily).dropna(subset=["ret"])
    _, last = _period_edges(d.index)
    ends = d.index[last & d.index.month.isin([3, 6, 9, 12])]
    rows = []
    for e in ends:
        i = d.index.get_loc(e)
        nxt = d["ret"].iloc[i + 1] if i + 1 < len(d) else np.nan
        rows.append({"конец квартала": e, "итог дня": d["ret"].iloc[i], "след. день": nxt})
    return pd.DataFrame(rows).set_index("конец квартала")


def _setup(close: pd.Series, p: int) -> dict:
    """Картина перед отчётом на день p: разгон за неделю/месяц/3 мес. и отставание от максимума за год."""
    hi = close.iloc[max(p - 251, 0): p + 1].max()
    return {
        "pre5": close.iloc[p] / close.iloc[max(p - 5, 0)] - 1,
        "pre20": close.iloc[p] / close.iloc[max(p - 20, 0)] - 1,
        "pre60": close.iloc[p] / close.iloc[max(p - 60, 0)] - 1,
        "dd": close.iloc[p] / hi - 1,
    }


def earnings_reactions(daily: pd.DataFrame) -> pd.DataFrame:
    """Таблица дней после отчётов: гэп, движение внутри дня, итог, неделя, after hours."""
    d = _with_returns(daily)
    rows = []
    for x in pd.to_datetime(EARNINGS_REACTION_DAYS):
        if x not in d.index:
            continue
        k = d.index.get_loc(x)
        if k < 6:
            continue
        p = k - 1
        ah = AFTER_HOURS.get(str(x.date()))
        rows.append({
            "date": x, "prev_close": d["Close"].iloc[p], "open": d["Open"].iloc[k], "close": d["Close"].iloc[k],
            "gap": d["gap"].iloc[k], "intra": d["intra"].iloc[k], "ret": d["ret"].iloc[k],
            "week": d["Close"].iloc[min(k + 4, len(d) - 1)] / d["Close"].iloc[p] - 1,
            "after_hours": ah / 100 if ah is not None else np.nan,
            **_setup(d["Close"], p),
        })
    return pd.DataFrame(rows).set_index("date")


def current_setup(daily: pd.DataFrame) -> dict:
    """Картина на последний день данных (если сегодня день отчёта — это «перед отчётом»)."""
    return _setup(daily["Close"], len(daily) - 1)


def similar_reactions(table: pd.DataFrame, setup: dict, k: int = 6) -> pd.DataFrame:
    """k прошлых отчётов с самой похожей картиной перед ними."""
    cols = list(setup)
    sd = table[cols].std().replace(0, 1)
    dist = np.sqrt((((table[cols] - pd.Series(setup)) / sd) ** 2).sum(axis=1))
    return table.assign(dist=dist).sort_values("dist").head(k)


def earnings_scenarios(table: pd.DataFrame, prev_close: float, setup: dict,
                       after_hours: float | None = None) -> dict:
    """Сценарии открытия и закрытия на день после отчёта.

    Без after hours — по распределению прошлых реакций и похожим случаям.
    С after hours (цена) — по линейной связи after hours → гэп/итог на известных
    случаях; разброс задаётся остатками этой связи.
    """
    sim = similar_reactions(table, setup)
    out = {
        "prev_close": prev_close,
        "history": {
            "n": len(table), "up_share": float((table["ret"] > 0).mean()),
            "open_q": {q: prev_close * (1 + table["gap"].quantile(q)) for q in (0.25, 0.5, 0.75)},
            "close_q": {q: prev_close * (1 + table["ret"].quantile(q)) for q in (0.25, 0.5, 0.75)},
            "fade_share": float((table["intra"] < 0).mean()),
        },
        "similar": sim,
        "similar_open": prev_close * (1 + sim["gap"].median()),
        "similar_close": prev_close * (1 + sim["ret"].median()),
    }
    if after_hours is not None:
        ah = after_hours / prev_close - 1
        known = table.dropna(subset=["after_hours"])
        res = {}
        for col in ("gap", "ret"):
            b, a = np.polyfit(known["after_hours"], known[col], 1)
            resid = known[col] - (a + b * known["after_hours"])
            mid = a + b * ah
            res[col] = {
                "mid": prev_close * (1 + mid),
                "lo": prev_close * (1 + mid + resid.quantile(0.2)),
                "hi": prev_close * (1 + mid + resid.quantile(0.8)),
                "corr": float(np.corrcoef(known["after_hours"], known[col])[0, 1]),
            }
        # Самые близкие случаи по величине after hours.
        near = known.assign(ah_dist=(known["after_hours"] - ah).abs()).sort_values("ah_dist").head(3)
        out["after_hours"] = {"price": after_hours, "pct": ah, "open": res["gap"], "close": res["ret"],
                              "n": len(known), "nearest": near}
    return out
