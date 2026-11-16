"""Модель: подбор гиперпараметров на временных разбиениях, взвешенный ансамбль
регрессий на доходность следующего дня, классификатор направления и квантили."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

QUANTILES = (0.1, 0.9)
WINSOR = 0.02  # доля хвостов цели, обрезаемых при обучении


def _clip(y: pd.Series) -> pd.Series:
    """Обрезает редкие экстремальные дни, чтобы они не перетягивали обучение."""
    return y.clip(y.quantile(WINSOR), y.quantile(1 - WINSOR))


def calibrated_range(daily: pd.DataFrame, target: str = "close", level: float = 0.8,
                     skip_target_days: list | None = None, window: int = 250) -> dict:
    """Диапазон доходности следующего дня с честным покрытием.

    Доходности делятся на текущую волатильность (20 дней), берутся квантили за
    последние `window` обычных дней и умножаются на сегодняшнюю волатильность.
    Ширина подстраивается: выбирается самый узкий вариант, который на последних
    `window` днях (без заглядывания в будущее) накрывал факт не реже `level`.
    """
    c = daily["Close"]
    vol = np.log(c).diff().rolling(20).std()
    y = np.log((daily["Open"] if target == "open" else c).shift(-1) / c)
    z = y / vol
    if skip_target_days is not None:
        nxt = pd.Series(daily.index, index=daily.index).shift(-1)
        z[nxt.isin(pd.to_datetime(list(skip_target_days))).to_numpy()] = np.nan
    hist = z.dropna()
    best = None
    for q in np.arange((1 - level) / 2, 0.019, -0.01):
        lo = hist.rolling(window, min_periods=60).quantile(q).shift()
        hi = hist.rolling(window, min_periods=60).quantile(1 - q).shift()
        ok = ((hist >= lo) & (hist <= hi))[lo.notna()].iloc[-window:]
        best = {"q": float(q), "coverage": float(ok.mean()) if len(ok) else float("nan"), "n": int(len(ok))}
        if len(ok) < 60 or ok.mean() >= level:
            break
    recent, v = hist.iloc[-window:], float(vol.iloc[-1])
    return {**best, "lo": float(recent.quantile(best["q"]) * v), "hi": float(recent.quantile(1 - best["q"]) * v),
            "level": level}

# Модель и сетка гиперпараметров, которые перебираются при обучении.
CANDIDATES = {
    "ridge": (
        make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 30))),
        {},
    ),
    "gbm": (
        GradientBoostingRegressor(loss="huber", subsample=0.8, random_state=0),
        {"n_estimators": [150, 400], "max_depth": [2, 3], "learning_rate": [0.01, 0.03]},
    ),
    "hgb": (
        HistGradientBoostingRegressor(loss="absolute_error", random_state=0),
        {"max_depth": [2, 4], "learning_rate": [0.02, 0.06], "max_iter": [150, 400], "l2_regularization": [0.0, 1.0]},
    ),
    "rf": (
        RandomForestRegressor(n_estimators=300, max_features=0.4, n_jobs=-1, random_state=0),
        {"max_depth": [4, 8], "min_samples_leaf": [20, 60]},
    ),
    "et": (
        ExtraTreesRegressor(n_estimators=300, max_features=0.5, n_jobs=-1, random_state=0),
        {"max_depth": [4, 8], "min_samples_leaf": [20, 60]},
    ),
}


def _classifiers():
    return {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
        "hgb_clf": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=200, l2_regularization=1.0, random_state=0),
    }


def _quantile_model(q: float):
    return GradientBoostingRegressor(
        loss="quantile", alpha=q, n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0
    )


@dataclass
class Ensemble:
    """Обученные на одном наборе данных модели с весами."""
    params: dict
    weights: dict
    models: dict = field(default_factory=dict)
    classifiers: dict = field(default_factory=dict)

    def fit(self, X, y):
        for name, p in self.params.items():
            self.models[name] = clone(CANDIDATES[name][0]).set_params(**p).fit(X, _clip(y))
        for name, c in _classifiers().items():
            self.classifiers[name] = c.fit(X, (y > 0).astype(int))
        return self

    def predict_all(self, X) -> dict[str, np.ndarray]:
        out = {k: m.predict(X) for k, m in self.models.items()}
        out["ensemble"] = sum(self.weights[k] * out[k] for k in self.models)
        return out

    def prob_up(self, X) -> np.ndarray:
        return np.mean([c.predict_proba(X)[:, 1] for c in self.classifiers.values()], axis=0)


def tune(X: pd.DataFrame, y: pd.Series, n_splits: int = 4, fast: bool = False) -> tuple[dict, dict, dict]:
    """Перебирает гиперпараметры на TimeSeriesSplit (обучение всегда в прошлом,
    проверка — в будущем). Возвращает лучшие параметры, веса ансамбля и MAE моделей."""
    cv = TimeSeriesSplit(n_splits=n_splits)
    best_params, best_mae = {}, {}
    for name, (base, grid) in CANDIDATES.items():
        combos = [{}] if fast or not grid else list(ParameterGrid(grid))
        scores = []
        for p in combos:
            errs = []
            for tr, va in cv.split(X):
                m = clone(base).set_params(**p).fit(X.iloc[tr], _clip(y.iloc[tr]))
                errs.append(np.mean(np.abs(m.predict(X.iloc[va]) - y.iloc[va])))
            scores.append(np.mean(errs))
        i = int(np.argmin(scores))
        best_params[name], best_mae[name] = combos[i], float(scores[i])
        log.info("%s: MAE %.4f, параметры %s", name, scores[i], combos[i])

    # Веса ∝ 1/MAE² — точные модели весят больше, но слабые не выкидываются полностью.
    inv = {k: 1 / v**2 for k, v in best_mae.items()}
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}
    return best_params, weights, best_mae


@dataclass
class Prediction:
    last_date: str
    last_close: float
    predicted_return: float
    predicted_price: float
    low_price: float
    high_price: float
    prob_up: float
    per_model: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    cv_mae: dict = field(default_factory=dict)
    backtest: dict = field(default_factory=dict)
    top_features: list = field(default_factory=list)
    n_features: int = 0
    n_train_days: int = 0


def walk_forward(features: pd.DataFrame, params: dict, weights: dict, test_days: int = 250, step: int = 20) -> dict:
    """Честная проверка: модель учится только на прошлом и предсказывает следующие дни."""
    data = features.dropna()
    X, y = data.drop(columns="target"), data["target"]
    test_days = min(test_days, len(data) // 3)
    start = len(data) - test_days
    preds, probs, actual, dates = [], [], [], []
    for i in range(start, len(data), step):
        end = min(i + step, len(data))
        ens = Ensemble(params, weights).fit(X.iloc[:i], y.iloc[:i])
        preds.extend(ens.predict_all(X.iloc[i:end])["ensemble"])
        probs.extend(ens.prob_up(X.iloc[i:end]))
        actual.extend(y.iloc[i:end])
        dates.extend(data.index[i:end])
    preds, probs, actual = np.array(preds), np.array(probs), np.array(actual)
    nonzero = actual != 0
    up = (actual > 0).astype(float)
    return {
        "days": int(len(actual)),
        "mae_model": float(np.mean(np.abs(preds - actual))),
        "mae_naive": float(np.mean(np.abs(actual))),  # «завтра цена = сегодня»
        "direction_accuracy": float(np.mean(np.sign(preds[nonzero]) == np.sign(actual[nonzero]))),
        "classifier_accuracy": float(np.mean((probs > 0.5) == (actual > 0))),
        "brier": float(np.mean((probs - up) ** 2)),
        "brier_naive": float(np.mean((up.mean() - up) ** 2)),
        "share_up_days": float(np.mean(actual > 0)),
        "series": pd.DataFrame({"pred": preds, "prob_up": probs, "actual": actual}, index=pd.DatetimeIndex(dates)),
    }


def predict_next(features: pd.DataFrame, last_close: float, backtest_days: int = 250, fast: bool = False,
                 interval: dict | None = None) -> Prediction:
    train = features.dropna()
    X, y = train.drop(columns="target"), train["target"]
    x_last = features.drop(columns="target").iloc[[-1]].ffill().fillna(0)

    # Параметры подбираются только на данных ДО периода проверки, чтобы бэктест был честным.
    bt_days = min(backtest_days, len(train) // 3)
    params, weights, cv_mae = tune(X.iloc[:-bt_days], y.iloc[:-bt_days], fast=fast)

    ens = Ensemble(params, weights).fit(X, y)
    point = ens.predict_all(x_last)
    r = float(point["ensemble"][0])
    if interval:  # откалиброванный диапазон (calibrated_range)
        lo, hi = min(interval["lo"], r), max(interval["hi"], r)
    else:
        q = {qq: float(_quantile_model(qq).fit(X, y).predict(x_last)[0]) for qq in QUANTILES}
        lo, hi = min(q[QUANTILES[0]], r), max(q[QUANTILES[1]], r)
    prob_up = float(ens.prob_up(x_last)[0])

    imp = pd.Series(0.0, index=X.columns)
    for name in ("gbm", "rf", "et"):
        imp += weights[name] * pd.Series(ens.models[name].feature_importances_, index=X.columns)
    imp = (imp / imp.sum()).sort_values(ascending=False).head(10)

    bt = walk_forward(features, params, weights, test_days=backtest_days)
    if interval:
        bt["interval_coverage"], bt["interval_days"] = interval["coverage"], interval["n"]
    return Prediction(
        last_date=str(features.index[-1].date()),
        last_close=last_close,
        predicted_return=r,
        predicted_price=last_close * np.exp(r),
        low_price=last_close * np.exp(lo),
        high_price=last_close * np.exp(hi),
        prob_up=prob_up,
        per_model={k: float(v[0]) for k, v in point.items()},
        weights=weights,
        params=params,
        cv_mae=cv_mae,
        backtest=bt,
        top_features=[(k, float(v)) for k, v in imp.items()],
        n_features=X.shape[1],
        n_train_days=len(X),
    )


def replay(daily: pd.DataFrame, close_feats: pd.DataFrame, open_feats: pd.DataFrame,
           days: int = 10, fast: bool = False, skip_target_days: list | None = None) -> pd.DataFrame:
    """Модель «как будто» запускали каждый вечер последние `days` торговых дней.

    Для каждого дня t модель учится только на том, что было известно к закрытию t,
    и предсказывает открытие и закрытие t+1. Гиперпараметры подбираются один раз
    на данных до начала окна — без заглядывания в будущее.
    """
    idx = daily.index
    days = min(days, len(idx) - 2)
    start = idx[len(idx) - days - 1]
    setups = {}
    for name, f in (("close", close_feats), ("open", open_feats)):
        data = f.loc[:start].iloc[:-1].dropna()
        X, y = data.drop(columns="target"), data["target"]
        setups[name] = (f,) + tune(X, y, fast=fast)[:2]

    rows = []
    for i in range(len(idx) - days - 1, len(idx) - 1):
        t, nxt = idx[i], idx[i + 1]
        prev = float(daily["Close"].iloc[i])
        row = {"date": nxt, "prev_close": prev,
               "open": float(daily["Open"].iloc[i + 1]), "close": float(daily["Close"].iloc[i + 1])}
        for name, (f, params, weights) in setups.items():
            # Строка t содержит цель t+1 (будущее) — в обучение не берём.
            train = f.loc[:t].iloc[:-1].dropna()
            X, y = train.drop(columns="target"), train["target"]
            x_t = f.drop(columns="target").loc[:t].iloc[[-1]].ffill().fillna(0)
            ens = Ensemble(params, weights).fit(X, y)
            r = float(ens.predict_all(x_t)["ensemble"][0])
            rng = calibrated_range(daily.iloc[: i + 1], name, skip_target_days=skip_target_days)
            lo, hi = rng["lo"], rng["hi"]
            row[f"pred_{name}"] = prev * np.exp(r)
            row[f"{name}_lo"] = prev * np.exp(min(lo, r))
            row[f"{name}_hi"] = prev * np.exp(max(hi, r))
            if name == "close":
                row["prob_up"] = float(ens.prob_up(x_t)[0])
        rows.append(row)
    out = pd.DataFrame(rows).set_index("date")
    for name in ("open", "close"):
        out[f"{name}_err"] = out[name] / out[f"pred_{name}"] - 1
        out[f"{name}_in_range"] = (out[name] >= out[f"{name}_lo"]) & (out[name] <= out[f"{name}_hi"])
    out["naive_close_err"] = out["close"] / out["prev_close"] - 1
    out["dir_hit"] = np.sign(out["pred_close"] - out["prev_close"]) == np.sign(out["close"] - out["prev_close"])
    return out
