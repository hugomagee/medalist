"""Shared building blocks for playground-series-s6e10 (airline satisfaction, AUC).

Experiments import from here so each run.py states only its hypothesis-specific
logic. Every helper takes the harness folds; nothing is fit on full train for
the OOF view. Functions are only ever ADDED or extended with defaults that keep
earlier behaviour, so frozen run.py files of earlier experiments still mean
what they meant when they ran.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

FloatArray = NDArray[np.float64]
Fold = tuple[NDArray[np.int64], NDArray[np.int64]]

COMP_DIR = Path(__file__).resolve().parents[1] / "competitions" / "playground-series-s6e10"
TARGET = "satisfaction"
ID = "id"

CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATINGS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]
NUMS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
FEATS = ["Age", "Flight Distance", *RATINGS, NUMS[2], NUMS[3], *CATS]

CAT_MAPS: dict[str, dict[str, int]] = {
    "Gender": {"Female": 0, "Male": 1},
    "Customer Type": {"Loyal Customer": 0, "disloyal Customer": 1},
    "Type of Travel": {"Business travel": 0, "Personal Travel": 1},
    "Class": {"Eco": 0, "Eco Plus": 1, "Business": 2},
}


def threads() -> int:
    return int(os.environ.get("S6E10_THREADS", "10"))


def clean(name: str) -> str:
    """LightGBM/XGBoost-safe column name."""
    return name.replace(" ", "_").replace("/", "_").replace("-", "_")


def load_original() -> pd.DataFrame:
    return pd.read_csv(COMP_DIR / "data" / "original.csv")


def base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All 22 raw features as numbers (categoricals ordinal-coded, NaN kept)."""
    out = pd.DataFrame(index=df.index)
    for c in FEATS:
        if c in CAT_MAPS:
            mapped = df[c].map(CAT_MAPS[c])
            if mapped.isna().any():
                raise ValueError(f"unmapped values in {c}: {df.loc[mapped.isna(), c].unique()[:5]}")
            out[c] = mapped.astype(np.int64)
        else:
            out[c] = df[c].astype(np.float64)
    return out


def generic_fe(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features (no target, no cross-row stats): safe on any frame."""
    out = X.copy()
    r = X[RATINGS].to_numpy(dtype=np.float64)
    out["r_sum"] = r.sum(axis=1)
    out["r_std"] = r.std(axis=1)
    out["r_min"] = r.min(axis=1)
    out["r_max"] = r.max(axis=1)
    out["r_n0"] = (r == 0).sum(axis=1)
    out["r_n5"] = (r == 5).sum(axis=1)
    out["r_n_le2"] = (r <= 2).sum(axis=1)
    dep = X["Departure Delay in Minutes"].to_numpy(dtype=np.float64)
    arr = X["Arrival Delay in Minutes"].to_numpy(dtype=np.float64)
    out["delay_diff"] = arr - dep
    out["log_dep"] = np.log1p(dep)
    out["log_arr"] = np.log1p(arr)
    out["delayed"] = ((dep > 0) | (np.nan_to_num(arr) > 0)).astype(np.float64)
    out["arr_na"] = np.isnan(arr).astype(np.float64)
    out["log_dist"] = np.log1p(X["Flight Distance"].to_numpy(dtype=np.float64))
    return out


# --------------------------------------------------------------------------- models


def lgb_fit_predict(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xva: pd.DataFrame,
    yva: np.ndarray,
    Xte: pd.DataFrame,
    params: dict[str, Any],
    num_boost_round: int = 20000,
    early_stopping: int = 300,
    weight: np.ndarray | None = None,
    categorical: list[str] | None = None,
) -> tuple[FloatArray, FloatArray, int, dict[str, float]]:
    import lightgbm as lgb

    rename = {c: clean(c) for c in Xtr.columns}
    cat = [rename[c] for c in (categorical or [])]
    p = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 42,
        "num_threads": threads(),
        "deterministic": True,
        "force_row_wise": True,
    }
    p.update(params)
    dtr = lgb.Dataset(
        Xtr.rename(columns=rename), ytr, weight=weight, categorical_feature=cat or "auto",
        free_raw_data=True,
    )
    dva = lgb.Dataset(Xva.rename(columns=rename), yva, reference=dtr, categorical_feature=cat or "auto")
    booster = lgb.train(
        p,
        dtr,
        num_boost_round=num_boost_round,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )
    it = booster.best_iteration
    va = booster.predict(Xva.rename(columns=rename), num_iteration=it)
    te = booster.predict(Xte.rename(columns=rename), num_iteration=it)
    imp = dict(zip(Xtr.columns, booster.feature_importance("gain").astype(float), strict=True))
    return va, te, it, imp


def xgb_fit_predict(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xva: pd.DataFrame,
    yva: np.ndarray,
    Xte: pd.DataFrame,
    params: dict[str, Any],
    num_boost_round: int = 20000,
    early_stopping: int = 300,
    weight: np.ndarray | None = None,
    enable_categorical: bool = False,
) -> tuple[FloatArray, FloatArray, int, dict[str, float]]:
    import xgboost as xgb

    rename = {c: clean(c) for c in Xtr.columns}
    p = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "seed": 42,
        "nthread": threads(),
    }
    p.update(params)
    dtr = xgb.DMatrix(Xtr.rename(columns=rename), ytr, weight=weight, enable_categorical=enable_categorical)
    dva = xgb.DMatrix(Xva.rename(columns=rename), yva, enable_categorical=enable_categorical)
    dte = xgb.DMatrix(Xte.rename(columns=rename), enable_categorical=enable_categorical)
    booster = xgb.train(
        p, dtr, num_boost_round=num_boost_round, evals=[(dva, "va")],
        early_stopping_rounds=early_stopping, verbose_eval=False,
    )
    it = booster.best_iteration + 1
    va = booster.predict(dva, iteration_range=(0, it))
    te = booster.predict(dte, iteration_range=(0, it))
    gain = booster.get_score(importance_type="gain")
    inv = {v: k for k, v in rename.items()}
    imp = {inv.get(k, k): float(v) for k, v in gain.items()}
    return va, te, it, imp


def cat_fit_predict(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xva: pd.DataFrame,
    yva: np.ndarray,
    Xte: pd.DataFrame,
    params: dict[str, Any],
    cat_features: list[str] | None = None,
    weight: np.ndarray | None = None,
    early_stopping: int = 300,
) -> tuple[FloatArray, FloatArray, int, dict[str, float]]:
    from catboost import CatBoostClassifier, Pool

    p = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 20000,
        "random_seed": 42,
        "thread_count": threads(),
        "verbose": 0,
        "allow_writing_files": False,
    }
    p.update(params)
    cf = cat_features or []
    model = CatBoostClassifier(**p)
    model.fit(
        Pool(Xtr, ytr, cat_features=cf, weight=weight),
        eval_set=Pool(Xva, yva, cat_features=cf),
        early_stopping_rounds=early_stopping,
        use_best_model=True,
    )
    va = model.predict_proba(Pool(Xva, cat_features=cf))[:, 1]
    te = model.predict_proba(Pool(Xte, cat_features=cf))[:, 1]
    imp = dict(zip(Xtr.columns, model.get_feature_importance().astype(float), strict=True))
    return va, te, int(model.get_best_iteration() or 0) + 1, imp


# --------------------------------------------------------------------------- CV driver


FoldFn = Callable[[int, NDArray[np.int64], NDArray[np.int64]], tuple[FloatArray, FloatArray, dict[str, Any]]]


def cv_loop(
    y: np.ndarray, folds: list[Fold], n_test: int, fold_fn: FoldFn, log_name: str = "run"
) -> dict[str, Any]:
    """Runs fold_fn(k, train_idx, valid_idx) -> (valid_pred, test_pred, info) over the
    harness folds; returns the run.py result dict (test = mean over fold models)."""
    oof = np.zeros(len(y), dtype=np.float64)
    test = np.zeros(n_test, dtype=np.float64)
    fold_scores: list[float] = []
    infos: list[dict[str, Any]] = []
    imp_sum: dict[str, float] = {}
    for k, (tr, va) in enumerate(folds):
        t0 = time.time()
        pv, pt, info = fold_fn(k, tr, va)
        oof[va] = pv
        test += np.asarray(pt, dtype=np.float64) / len(folds)
        s = float(roc_auc_score(y[va], pv))
        fold_scores.append(s)
        for f, v in (info.pop("importances", None) or {}).items():
            imp_sum[f] = imp_sum.get(f, 0.0) + float(v) / len(folds)
        info["seconds"] = round(time.time() - t0, 1)
        info["auc"] = s
        infos.append(info)
        line = f"[{log_name}] fold {k}: auc={s:.6f} {info}"
        print(line, flush=True)
        with open("log.txt", "a") as fh:  # cwd is the experiment dir
            fh.write(line + "\n")
    return {
        "oof_predictions": oof,
        "test_predictions": test,
        "fold_scores": fold_scores,
        "feature_importances": imp_sum or None,
        "extra": {"folds": infos, "oof_auc": float(roc_auc_score(y, oof))},
    }


# --------------------------------------------------------------------------- encodings


def original_target_stats(
    orig: pd.DataFrame, cols: list[str], apply: list[pd.DataFrame], prior: float = 20.0
) -> list[pd.DataFrame]:
    """For a column (or tuple of columns), the ORIGINAL dataset's smoothed target mean
    and log-count per value, merged onto each frame in `apply`. Uses only the
    original's labels, never Kaggle train labels, so it is fold-safe as is."""
    y = orig[TARGET].astype(np.float64)
    g_mean = float(y.mean())
    key_o = _key(orig, cols)
    stats = y.groupby(key_o).agg(["sum", "count"])
    te = (stats["sum"] + prior * g_mean) / (stats["count"] + prior)
    name = "og_" + "__".join(clean(c) for c in cols)
    out = []
    for df in apply:
        k = _key(df, cols)
        frame = pd.DataFrame(index=df.index)
        frame[name + "_te"] = k.map(te).fillna(g_mean).to_numpy(dtype=np.float64)
        frame[name + "_n"] = np.log1p(k.map(stats["count"]).fillna(0).to_numpy(dtype=np.float64))
        out.append(frame)
    return out


def _key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    s = df[cols[0]].astype(str)
    for c in cols[1:]:
        s = s + "|" + df[c].astype(str)
    return s


def te_fold(
    key_tr: pd.Series, y_tr: np.ndarray, key_apply: list[pd.Series], prior: float = 20.0,
    inner_folds: int = 5, seed: int = 42,
) -> tuple[FloatArray, list[FloatArray]]:
    """Target encoding for one outer fold: train rows get an INNER out-of-fold
    encoding (so the GBM never sees a row's own label in its feature), the
    apply frames (outer-valid, test) get the full fold-train mapping."""
    y_tr = np.asarray(y_tr, dtype=np.float64)
    g = float(y_tr.mean())
    key_tr = key_tr.reset_index(drop=True)
    enc_tr = np.empty(len(key_tr), dtype=np.float64)
    rng = np.random.RandomState(seed)
    part = rng.randint(0, inner_folds, len(key_tr))
    ys = pd.Series(y_tr)
    for f in range(inner_folds):
        fit = part != f
        st = ys[fit].groupby(key_tr[fit].to_numpy()).agg(["sum", "count"])
        m = (st["sum"] + prior * g) / (st["count"] + prior)
        enc_tr[~fit] = key_tr[~fit].map(m).fillna(g).to_numpy(dtype=np.float64)
    st = ys.groupby(key_tr.to_numpy()).agg(["sum", "count"])
    m = (st["sum"] + prior * g) / (st["count"] + prior)
    outs = [k.reset_index(drop=True).map(m).fillna(g).to_numpy(dtype=np.float64) for k in key_apply]
    return enc_tr, outs


def auc_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil standard error of an AUC estimate."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (
        n_pos * n_neg
    )
    return float(np.sqrt(var))


def int_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Fast collision-resistant int64 key over integer-valued columns (NaN -> -1)."""
    k = np.zeros(len(df), dtype=np.int64)
    for c in cols:
        v = df[c].fillna(-1).to_numpy().astype(np.int64)
        k = k * np.int64(1000003) + (v + 7)
    return pd.Series(k, index=df.index)


def key_features_fold(
    keys_tr: dict[str, pd.Series],
    y_tr: np.ndarray,
    keys_apply: list[dict[str, pd.Series]],
    prior: float = 10.0,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Fold-safe Kaggle-train target encodings for several keys (nested inner OOF on
    fold-train rows; full fold-train mapping for the apply frames)."""
    tr_cols: dict[str, FloatArray] = {}
    ap_cols: list[dict[str, FloatArray]] = [{} for _ in keys_apply]
    for name, ktr in keys_tr.items():
        enc_tr, outs = te_fold(ktr, y_tr, [ka[name] for ka in keys_apply], prior=prior)
        tr_cols["te_" + name] = enc_tr
        for i, o in enumerate(outs):
            ap_cols[i]["te_" + name] = o
    return pd.DataFrame(tr_cols), [pd.DataFrame(a) for a in ap_cols]


# --------------------------------------------------------------------------- neural net


def nn_codes(frames: list[pd.DataFrame], dist_bins: int | None = 64, delay_cap: int = 60) -> tuple[
    list[NDArray[np.int64]], list[NDArray[np.float32]], list[int]
]:
    """Integer codes per feature (shared vocabulary over all frames; unsupervised) and a
    small standardized numeric block. Flight Distance is quantile-binned; delays capped."""
    allf = pd.concat(frames, ignore_index=True)
    sizes = [len(f) for f in frames]
    cols: list[NDArray[np.int64]] = []
    card: list[int] = []
    for c in FEATS:
        v = allf[c]
        if c == "Flight Distance" and dist_bins is None:
            code = pd.factorize(v, sort=True)[0].astype(np.int64)
        elif c == "Flight Distance":
            q = np.unique(np.quantile(v, np.linspace(0, 1, dist_bins + 1)[1:-1]))
            code = np.searchsorted(q, v.to_numpy()).astype(np.int64)
        elif c in ("Departure Delay in Minutes", "Arrival Delay in Minutes"):
            code = (np.clip(v.fillna(-1).to_numpy(), -1, delay_cap) + 1).astype(np.int64)
        else:
            code = pd.factorize(v.fillna(-1), sort=True)[0].astype(np.int64)
        cols.append(code)
        card.append(int(code.max()) + 1)
    codes = np.stack(cols, axis=1)
    num = np.stack(
        [
            allf["Age"].to_numpy(),
            np.log1p(allf["Flight Distance"].to_numpy()),
            np.log1p(allf["Departure Delay in Minutes"].to_numpy()),
            np.log1p(allf["Arrival Delay in Minutes"].fillna(allf["Departure Delay in Minutes"]).to_numpy()),
            allf[RATINGS].to_numpy().sum(axis=1),
        ],
        axis=1,
    ).astype(np.float64)
    num = ((num - num.mean(0)) / (num.std(0) + 1e-9)).astype(np.float32)
    out_c, out_n, s = [], [], 0
    for n in sizes:
        out_c.append(codes[s : s + n])
        out_n.append(num[s : s + n])
        s += n
    return out_c, out_n, card


def mlp_fit_predict(
    Ctr: NDArray[np.int64], Ntr: NDArray[np.float32], ytr: np.ndarray,
    Cva: NDArray[np.int64], Nva: NDArray[np.float32], yva: np.ndarray,
    Cte: NDArray[np.int64], Nte: NDArray[np.float32], card: list[int],
    emb_dim: int = 12, hidden: tuple[int, ...] = (512, 256, 128), dropout: float = 0.2,
    epochs: int = 12, batch: int = 4096, lr: float = 2e-3, wd: float = 1e-5, seed: int = 42,
    patience: int = 3,
) -> tuple[FloatArray, FloatArray, int]:
    """Per-feature embedding MLP on MPS (falls back to CPU). Best epoch by valid AUC."""
    import torch
    from torch import nn

    # LightGBM and torch each ship libomp on macOS; torch CPU-parallel ops (randperm) can
    # deadlock in an OMP barrier once LightGBM has run in the same process. Keep torch on
    # one CPU thread and draw permutations with numpy.
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(int(sum(card)), emb_dim)
            self.register_buffer("offs", torch.tensor(np.cumsum([0] + card[:-1]), dtype=torch.long))
            d = emb_dim * len(card) + Ntr.shape[1]
            layers: list[nn.Module] = []
            for h in hidden:
                layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
                d = h
            layers.append(nn.Linear(d, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, c: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            e = self.emb(c + self.offs).flatten(1)
            return self.mlp(torch.cat([e, n], dim=1)).squeeze(1)

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(ytr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    lossf = nn.BCEWithLogitsLoss()
    tc, tn = torch.from_numpy(Ctr).to(dev), torch.from_numpy(Ntr).to(dev)
    ty = torch.from_numpy(np.asarray(ytr, dtype=np.float32)).to(dev)

    def predict(C: NDArray[np.int64], N: NDArray[np.float32]) -> FloatArray:
        net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(C), 16384):
                c = torch.from_numpy(C[s : s + 16384]).to(dev)
                n = torch.from_numpy(N[s : s + 16384]).to(dev)
                out.append(torch.sigmoid(net(c, n)).cpu().numpy())
        net.train()
        return np.concatenate(out).astype(np.float64)

    best, best_ep, best_va, best_te, bad = -1.0, 0, None, None, 0
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(len(ytr))).to(dev)
        for s in range(0, len(ytr), batch):
            idx = perm[s : s + batch]
            opt.zero_grad()
            loss = lossf(net(tc[idx], tn[idx]), ty[idx])
            loss.backward()
            opt.step()
            sched.step()
        pv = predict(Cva, Nva)
        a = float(roc_auc_score(yva, pv))
        if a > best:
            best, best_ep, best_va, best_te, bad = a, ep + 1, pv, predict(Cte, Nte), 0
        else:
            bad += 1
            if bad >= patience:
                break
    assert best_va is not None and best_te is not None
    return best_va, best_te, best_ep


# --------------------------------------------------------------------------- feature builder

TOP10 = ["Online boarding", "Inflight wifi service", "Type of Travel", "Class", "Customer Type",
         "Inflight entertainment", "Seat comfort", "Leg room service", "Ease of Online booking", "Age"]
HIGH_KEYS = {
    "R": RATINGS,
    "RC": RATINGS + CATS,
    "RCT": RATINGS + ["Class", "Type of Travel"],
    "RCA": RATINGS + CATS + ["Age"],
}
TEACHER_PARAMS = {"learning_rate": 0.03, "num_leaves": 127, "min_data_in_leaf": 20,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                  "lambda_l2": 1.0}


def teacher_logits(O: pd.DataFrame, yo: np.ndarray, targets: list[pd.DataFrame],
                   params: dict[str, Any] | None = None, seed: int = 42) -> tuple[list[FloatArray], FloatArray]:
    """LightGBM trained only on original rows (5-fold, ES on its own folds); returns the
    mean fold-model logit for each target frame, plus the original OOF probability."""
    from sklearn.model_selection import StratifiedKFold

    p = dict(TEACHER_PARAMS if params is None else params, seed=seed)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    T = pd.concat(targets, ignore_index=True)
    pred = np.zeros(len(T))
    oof = np.zeros(len(O))
    for tr, va in skf.split(O, yo):
        pv, pt, _, _ = lgb_fit_predict(O.iloc[tr], yo[tr], O.iloc[va], yo[va], T, p,
                                       early_stopping=200)
        oof[va] = pv
        pred += pt / 5
    eps = 1e-6
    logit = np.log((pred + eps) / (1 - pred + eps))
    out, s = [], 0
    for t in targets:
        out.append(logit[s : s + len(t)])
        s += len(t)
    return out, oof


class FeatureBuilder:
    """Static (label-free or original-label-only) features computed once; Kaggle-label
    key encodings computed per fold (nested). Flags select feature groups."""

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, og_te: bool = True,
                 og_pairs: bool = True, teacher: bool = True, key_te: bool = True,
                 og_key_te: bool = True, freq: bool = True, keys: dict[str, list[str]] | None = None,
                 prior: float = 10.0):
        self.X = base_frame(train_df).reset_index(drop=True)
        self.Xt = base_frame(test_df).reset_index(drop=True)
        self.y = train_df[TARGET].to_numpy()
        self.keys = HIGH_KEYS if keys is None else keys
        self.key_te = key_te
        self.prior = prior
        og = load_original()
        O = base_frame(og)
        yo = og[TARGET].to_numpy()
        add_tr: list[pd.DataFrame] = []
        add_te: list[pd.DataFrame] = []
        if og_te:
            Ot = O.copy()
            Ot[TARGET] = yo
            from itertools import combinations
            groups = [[c] for c in FEATS] + ([list(q) for q in combinations(TOP10, 2)] if og_pairs else [])
            for cols in groups:
                a, b = original_target_stats(Ot, cols, [self.X, self.Xt])
                add_tr.append(a)
                add_te.append(b)
        if teacher:
            (lt, lte), self.teacher_oof = teacher_logits(O, yo, [self.X, self.Xt])
            add_tr.append(pd.DataFrame({"og_model": lt}))
            add_te.append(pd.DataFrame({"og_model": lte}))
        self.k_tr = {n: int_key(self.X, c) for n, c in self.keys.items()}
        self.k_te = {n: int_key(self.Xt, c) for n, c in self.keys.items()}
        g = float(yo.mean())
        for n, cols in self.keys.items():
            st_tr, st_te = {}, {}
            if og_key_te:
                ko = int_key(O, cols)
                st = pd.Series(yo.astype(float)).groupby(ko.to_numpy()).agg(["sum", "count"])
                te = (st["sum"] + 10 * g) / (st["count"] + 10)
                st_tr["og_te_" + n] = self.k_tr[n].map(te).fillna(g).to_numpy()
                st_te["og_te_" + n] = self.k_te[n].map(te).fillna(g).to_numpy()
                st_tr["og_n_" + n] = self.k_tr[n].map(st["count"]).fillna(0).to_numpy()
                st_te["og_n_" + n] = self.k_te[n].map(st["count"]).fillna(0).to_numpy()
            if freq:
                fr = pd.concat([self.k_tr[n], self.k_te[n]]).value_counts()
                st_tr["freq_" + n] = self.k_tr[n].map(fr).to_numpy()
                st_te["freq_" + n] = self.k_te[n].map(fr).to_numpy()
            if st_tr:
                add_tr.append(pd.DataFrame(st_tr))
                add_te.append(pd.DataFrame(st_te))
        self.S = pd.concat([self.X] + add_tr, axis=1)
        self.St = pd.concat([self.Xt] + add_te, axis=1)

    def fold(self, tr: NDArray[np.int64], va: NDArray[np.int64]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Xtr = self.S.iloc[tr].reset_index(drop=True)
        Xva = self.S.iloc[va].reset_index(drop=True)
        Xte = self.St
        if self.key_te and self.keys:
            ftr, (fva, fte) = key_features_fold(
                {n: s.iloc[tr] for n, s in self.k_tr.items()}, self.y[tr],
                [{n: s.iloc[va] for n, s in self.k_tr.items()}, self.k_te], prior=self.prior)
            Xtr = pd.concat([Xtr, ftr], axis=1)
            Xva = pd.concat([Xva, fva], axis=1)
            Xte = pd.concat([Xte, fte], axis=1)
        return Xtr, Xva, Xte


def og_stats_fast(O: pd.DataFrame, yo: np.ndarray, groups: list[list[str]],
                  apply: list[pd.DataFrame], prior: float = 20.0,
                  counts: bool = True) -> list[pd.DataFrame]:
    """original_target_stats over many column groups using int64 keys (same smoothing)."""
    yo = np.asarray(yo, dtype=np.float64)
    g = float(yo.mean())
    outs: list[dict[str, FloatArray]] = [{} for _ in apply]
    for cols in groups:
        ko = int_key(O, cols)
        st = pd.Series(yo).groupby(ko.to_numpy()).agg(["sum", "count"])
        te = (st["sum"] + prior * g) / (st["count"] + prior)
        name = "og_" + "__".join(clean(c) for c in cols)
        for i, df in enumerate(apply):
            k = int_key(df, cols)
            outs[i][name + "_te"] = k.map(te).fillna(g).to_numpy(dtype=np.float64)
            if counts:
                outs[i][name + "_n"] = np.log1p(k.map(st["count"]).fillna(0).to_numpy(dtype=np.float64))
    return [pd.DataFrame(o, index=df.index) for o, df in zip(outs, apply, strict=True)]


def teacher_bank(O: pd.DataFrame, yo: np.ndarray, targets: list[pd.DataFrame],
                 which: tuple[str, ...] = ("lgb_full", "lgb_nodelay", "lgb_rc", "lgb_r", "cat_full", "lr_onehot"),
                 ) -> tuple[list[pd.DataFrame], dict[str, float]]:
    """Several teachers trained ONLY on original rows (5-fold on the original for early
    stopping); returns one logit column per teacher for each target frame, plus each
    teacher's OOF AUC on the original."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import OneHotEncoder

    subsets = {
        "lgb_full": FEATS,
        "lgb_nodelay": [c for c in FEATS if c not in ("Departure Delay in Minutes", "Arrival Delay in Minutes", "Flight Distance")],
        "lgb_rc": RATINGS + CATS,
        "lgb_r": RATINGS,
    }
    T = pd.concat(targets, ignore_index=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    cols_out: dict[str, FloatArray] = {}
    aucs: dict[str, float] = {}
    eps = 1e-6
    for name in which:
        pred = np.zeros(len(T))
        oof = np.zeros(len(O))
        for tr, va in skf.split(O, yo):
            if name.startswith("lgb_"):
                c = subsets[name]
                pv, pt, _, _ = lgb_fit_predict(O.iloc[tr][c], yo[tr], O.iloc[va][c], yo[va], T[c],
                                               TEACHER_PARAMS, early_stopping=200)
            elif name == "cat_full":
                pv, pt, _, _ = cat_fit_predict(O.iloc[tr], yo[tr], O.iloc[va], yo[va], T,
                                               {"learning_rate": 0.1, "depth": 7, "iterations": 2500},
                                               cat_features=CATS, early_stopping=200)
            elif name == "lr_onehot":
                disc = RATINGS + CATS
                enc = OneHotEncoder(handle_unknown="ignore")
                Atr = enc.fit_transform(O.iloc[tr][disc].astype(int))
                lr = LogisticRegression(C=1.0, max_iter=2000)
                lr.fit(Atr, yo[tr])
                pv = lr.predict_proba(enc.transform(O.iloc[va][disc].astype(int)))[:, 1]
                pt = lr.predict_proba(enc.transform(T[disc].astype(int)))[:, 1]
            else:
                raise ValueError(name)
            oof[va] = pv
            pred += pt / 5
        aucs[name] = float(roc_auc_score(yo, oof))
        cols_out["t_" + name] = np.log((pred + eps) / (1 - pred + eps))
    full = pd.DataFrame(cols_out)
    out, s = [], 0
    for t in targets:
        out.append(full.iloc[s : s + len(t)].reset_index(drop=True))
        s += len(t)
    return out, aucs


D_, AD_, DD_ = "Flight Distance", "Arrival Delay in Minutes", "Departure Delay in Minutes"
KEYS_V2 = {
    "D": [D_], "DD": [DD_], "AD": [AD_],
    "D_T": [D_, "Type of Travel"], "D_C": [D_, "Class"], "D_CT": [D_, "Customer Type"],
    "D_G": [D_, "Gender"], "D_CT_T": [D_, "Customer Type", "Type of Travel"],
    "D_T_C": [D_, "Type of Travel", "Class"],
    "AD_LR": [AD_, "Leg room service"], "AD_OB": [AD_, "Online boarding"],
    "AD_CK": [AD_, "Checkin service"], "AD_SC": [AD_, "Seat comfort"], "AD_A": [AD_, "Age"],
}


# --------------------------------------------------------------------------- ensembling


def load_members(ids: list[str], base: str = "..") -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """OOF and test prediction matrices (rows x members) from sibling experiment dirs."""
    oof = np.column_stack([pd.read_parquet(f"{base}/{i}/oof.parquet")["prediction"].to_numpy() for i in ids])
    test = np.column_stack([pd.read_parquet(f"{base}/{i}/test_pred.parquet")["prediction"].to_numpy() for i in ids])
    return oof, test


def logit(p: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def stack_logistic(oof: NDArray[np.float64], test: NDArray[np.float64], y: np.ndarray,
                   folds: list[Fold], C: float = 1.0) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fold-safe LR stack on member logits: fold k is predicted by an LR fit on the other
    folds' OOF rows; test by an LR fit on all OOF rows. Returns (oof, test, full-fit coefs)."""
    from sklearn.linear_model import LogisticRegression

    Z, Zt = logit(oof), logit(test)
    out = np.zeros(len(y))
    for tr, va in folds:
        m = LogisticRegression(C=C, max_iter=1000).fit(Z[tr], y[tr])
        out[va] = m.predict_proba(Z[va])[:, 1]
    m = LogisticRegression(C=C, max_iter=1000).fit(Z, y)
    return out, m.predict_proba(Zt)[:, 1], np.append(m.coef_.ravel(), m.intercept_)


def rank_mean(M: NDArray[np.float64]) -> FloatArray:
    from scipy.stats import rankdata

    return np.mean([rankdata(M[:, j]) / len(M) for j in range(M.shape[1])], axis=0)


def og_same_distance_knn(O: pd.DataFrame, yo: np.ndarray, targets: list[pd.DataFrame],
                         taus: tuple[float, ...] = (1.0, 2.0), cols: list[str] | None = None,
                         prior: float = 2.0) -> list[pd.DataFrame]:
    """For each target row: among ORIGINAL rows with the same Flight Distance, the label
    mean weighted by exp(-mismatches/tau) over `cols` (default 13 ratings + 4 cats), shrunk
    toward the original prior with `prior` pseudo-weight; also the max similarity (fewest
    mismatches) and the label mean of the closest original rows. Only original labels used."""
    cols = cols or (RATINGS + CATS)
    yo = np.asarray(yo, dtype=np.float64)
    g = float(yo.mean())
    T = pd.concat(targets, ignore_index=True)
    dt = T["Flight Distance"].to_numpy()
    do = O["Flight Distance"].to_numpy()
    At = T[cols].to_numpy(dtype=np.int16)
    Ao = O[cols].to_numpy(dtype=np.int16)
    res = {f"knnD_t{t:g}": np.full(len(T), g) for t in taus}
    res["knnD_minmis"] = np.full(len(T), float(len(cols)))
    res["knnD_best_y"] = np.full(len(T), g)
    order_o = np.argsort(do, kind="stable")
    uo, start_o = np.unique(do[order_o], return_index=True)
    end_o = np.append(start_o[1:], len(do))
    pos = {v: (s, e) for v, s, e in zip(uo, start_o, end_o, strict=True)}
    order_t = np.argsort(dt, kind="stable")
    ut, start_t = np.unique(dt[order_t], return_index=True)
    end_t = np.append(start_t[1:], len(dt))
    for v, s, e in zip(ut, start_t, end_t, strict=True):
        if v not in pos:
            continue
        ti = order_t[s:e]
        so, eo = pos[v]
        oi = order_o[so:eo]
        for c0 in range(0, len(ti), 4096):
            tc = ti[c0 : c0 + 4096]
            mis = (At[tc][:, None, :] != Ao[oi][None, :, :]).sum(axis=2).astype(np.float64)
            yy = yo[oi]
            for t in taus:
                w = np.exp(-mis / t)
                res[f"knnD_t{t:g}"][tc] = (w @ yy + prior * g) / (w.sum(axis=1) + prior)
            mm = mis.min(axis=1)
            res["knnD_minmis"][tc] = mm
            best = mis == mm[:, None]
            res["knnD_best_y"][tc] = (best * yy).sum(axis=1) / best.sum(axis=1)
    full = pd.DataFrame(res)
    out, s0 = [], 0
    for t in targets:
        out.append(full.iloc[s0 : s0 + len(t)].reset_index(drop=True))
        s0 += len(t)
    return out


def digit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Digit-chunk codes of Flight Distance (and Age's last digit) as integers, mimicking
    how a token-based generator writes numbers: last 3 / last 2 digits, first 3 digits,
    first 2 digits with length, middle two digits with length, units, tens, tail after 3."""
    d = df["Flight Distance"].astype(np.int64)
    s = d.astype(str)
    ln = s.str.len().astype(np.int64)
    out = pd.DataFrame(index=df.index)
    out["dg_last3"] = d % 1000
    out["dg_last2"] = d % 100
    out["dg_first3"] = s.str[:3].astype(np.int64)
    out["dg_first2_len"] = s.str[:2].astype(np.int64) * 10 + ln
    out["dg_mid_len"] = ln * 100 + s.str[1:3].astype(np.int64)
    out["dg_units"] = d % 10
    out["dg_tens"] = (d // 10) % 10
    out["dg_tail_len"] = ln * 10 + s.str[3:].replace("", "-1").astype(np.int64).clip(lower=-1) + 1
    out["dg_len"] = ln
    out["age_units"] = df["Age"].astype(np.int64) % 10
    return out


DIGIT_KEYS = {c: [c] for c in ["dg_last3", "dg_last2", "dg_first3", "dg_first2_len", "dg_mid_len",
                               "dg_units", "dg_tens", "dg_tail_len", "age_units"]}


V3_DIST_KEYS = {
    "D": ["Flight Distance"], "DD": ["Departure Delay in Minutes"], "AD": ["Arrival Delay in Minutes"],
    "D_T": ["Flight Distance", "Type of Travel"], "D_C": ["Flight Distance", "Class"],
    "D_CT": ["Flight Distance", "Customer Type"],
}


def fb_v3(train_df: pd.DataFrame, test_df: pd.DataFrame) -> FeatureBuilder:
    """The e0068 feature set: raw + original single/pair TE + teacher logit + digit codes
    (+ their original TE) + per-fold Kaggle TE of distance/delay/digit keys + key freqs."""
    fb = FeatureBuilder(train_df, test_df, key_te=False, og_key_te=False, freq=False)
    dtr, dte = digit_frame(fb.X), digit_frame(fb.Xt)
    fb.X = pd.concat([fb.X, dtr], axis=1)
    fb.Xt = pd.concat([fb.Xt, dte], axis=1)
    fb.S = pd.concat([fb.S, dtr], axis=1)
    fb.St = pd.concat([fb.St, dte], axis=1)
    og = load_original()
    O = base_frame(og)
    O = pd.concat([O, digit_frame(O)], axis=1)
    a, b = og_stats_fast(O, og[TARGET].to_numpy(), list(DIGIT_KEYS.values()), [fb.X, fb.Xt])
    fb.S = pd.concat([fb.S, a], axis=1)
    fb.St = pd.concat([fb.St, b], axis=1)
    keys = {**V3_DIST_KEYS, **DIGIT_KEYS}
    fb.keys, fb.key_te = keys, True
    fb.k_tr = {n: int_key(fb.X, c) for n, c in keys.items()}
    fb.k_te = {n: int_key(fb.Xt, c) for n, c in keys.items()}
    for n in V3_DIST_KEYS:
        fr = pd.concat([fb.k_tr[n], fb.k_te[n]]).value_counts()
        fb.S["freq_" + n] = fb.k_tr[n].map(fr).to_numpy()
        fb.St["freq_" + n] = fb.k_te[n].map(fr).to_numpy()
    return fb


def original_rows_v3(fb: FeatureBuilder) -> tuple[pd.DataFrame, dict[str, pd.Series], np.ndarray]:
    """Features for the ORIGINAL rows aligned with fb_v3's static columns, built so no
    original row ever sees its own label: original-label stats and the teacher logit are
    out-of-fold within the original (5-fold, seed 42 -- the teacher's own split), Kaggle
    frequencies come from Kaggle train+test. Returns (static frame incl. is_original=1,
    key series for the per-fold Kaggle TE, original labels)."""
    from itertools import combinations
    from sklearn.model_selection import StratifiedKFold

    og = load_original()
    O = base_frame(og).reset_index(drop=True)
    O = pd.concat([O, digit_frame(O)], axis=1)
    yo = og[TARGET].to_numpy()
    static = pd.DataFrame(index=O.index)
    groups = [[c] for c in FEATS] + [list(q) for q in combinations(TOP10, 2)]
    parts = []
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for a, b in skf.split(O, yo):
        Ofit = O.iloc[a].copy()
        Ofit[TARGET] = yo[a]
        Oapp = O.iloc[b]
        cols = [original_target_stats(Ofit, g, [Oapp])[0] for g in groups]
        dg = og_stats_fast(O.iloc[a], yo[a], list(DIGIT_KEYS.values()), [Oapp])[0]
        parts.append(pd.concat(cols + [dg], axis=1))
    oof_feats = pd.concat(parts).sort_index()
    eps = 1e-6
    t = np.clip(fb.teacher_oof, eps, 1 - eps)
    oof_feats["og_model"] = np.log(t / (1 - t))
    k_o = {n: int_key(O, c) for n, c in fb.keys.items()}
    for n in V3_DIST_KEYS:
        fr = pd.concat([fb.k_tr[n], fb.k_te[n]]).value_counts()
        oof_feats["freq_" + n] = k_o[n].map(fr).fillna(0).to_numpy()
    full = pd.concat([O, oof_feats], axis=1)
    missing = [c for c in fb.S.columns if c not in full.columns]
    if missing:
        raise ValueError(f"original rows missing columns: {missing[:10]}")
    static = full[list(fb.S.columns)].astype(np.float32)
    static["is_original"] = 1
    return static, k_o, yo


def fold_with_original(fb: FeatureBuilder, O_static: pd.DataFrame, k_o: dict[str, pd.Series],
                       yo: np.ndarray, tr: NDArray[np.int64], va: NDArray[np.int64],
                       ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """fb.fold plus the original rows appended to the training frame (Kaggle TE for the
    original rows uses the Kaggle fold-train mapping). Returns (Xtr, ytr, Xva, Xte)."""
    Xtr, Xva, Xte = fb.fold(tr, va)
    _, (fo,) = key_features_fold({n: s.iloc[tr] for n, s in fb.k_tr.items()}, fb.y[tr], [k_o],
                                 prior=fb.prior)
    Ot = pd.concat([O_static.reset_index(drop=True), fo], axis=1)
    for M in (Xtr, Xva, Xte):
        M["is_original"] = 0
    Ot = Ot[list(Xtr.columns)]
    return pd.concat([Xtr, Ot], ignore_index=True), np.concatenate([fb.y[tr], yo]), Xva, Xte


def fb_v3_32(train_df: pd.DataFrame, test_df: pd.DataFrame) -> FeatureBuilder:
    """fb_v3 with the static feature frames stored as float32 (about half the memory; two
    fb_v3 processes plus desktop apps exhausted 16 GB and thrashed swap on 2026-10-29).
    Values are unchanged up to float32 rounding, which does not move LightGBM bin edges
    in any meaningful way."""
    fb = fb_v3(train_df, test_df)
    fb.S = fb.S.astype(np.float32)
    fb.St = fb.St.astype(np.float32)
    return fb


TOP8 = ["Online boarding", "Inflight wifi service", "Type of Travel", "Class", "Customer Type",
        "Inflight entertainment", "Seat comfort", "Leg room service"]


def fb_v4(train_df: pd.DataFrame, test_df: pd.DataFrame) -> FeatureBuilder:
    """fb_v3_32 + original-label TE (no counts) for all 210 column pairs not already present
    and all 56 triples of the top-8 columns, + the 6-teacher bank logits. float32."""
    from itertools import combinations

    fb = fb_v3_32(train_df, test_df)
    og = load_original()
    O = base_frame(og)
    yo = og[TARGET].to_numpy()
    have = set(fb.S.columns)
    pairs = [list(p) for p in combinations(FEATS, 2)
             if "og_" + "__".join(clean(c) for c in p) + "_te" not in have]
    triples = [list(t) for t in combinations(TOP8, 3)]
    a, b = og_stats_fast(O, yo, pairs + triples, [fb.X[FEATS], fb.Xt[FEATS]], counts=False)
    (t_tr, t_te), _ = teacher_bank(O, yo, [fb.X[FEATS], fb.Xt[FEATS]])
    fb.S = pd.concat([fb.S, a.astype(np.float32), t_tr.astype(np.float32)], axis=1)
    fb.St = pd.concat([fb.St, b.astype(np.float32), t_te.astype(np.float32)], axis=1)
    return fb


def kaggle_pair_te_fold(fb: FeatureBuilder, groups: list[list[str]], tr: NDArray[np.int64],
                        va: NDArray[np.int64], prior: float = 20.0
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fold-safe Kaggle-label TE for many column groups (nested inner OOF on fold-train rows,
    full fold-train mapping for valid/test), float32."""
    ytr = fb.y[tr].astype(np.float64)
    g = float(ytr.mean())
    rng = np.random.RandomState(42)
    part = rng.randint(0, 5, len(tr))
    out_tr, out_va, out_te = {}, {}, {}
    for cols in groups:
        name = "kte_" + "__".join(clean(c) for c in cols)
        k_all = int_key(fb.X, cols).to_numpy()
        k_te = int_key(fb.Xt, cols).to_numpy()
        ktr, kva = k_all[tr], k_all[va]
        enc = np.empty(len(tr), dtype=np.float32)
        for f in range(5):
            fit = part != f
            st = pd.DataFrame({"k": ktr[fit], "y": ytr[fit]}).groupby("k")["y"].agg(["sum", "count"])
            m = (st["sum"] + prior * g) / (st["count"] + prior)
            enc[~fit] = pd.Series(ktr[~fit]).map(m).fillna(g).to_numpy(dtype=np.float32)
        st = pd.DataFrame({"k": ktr, "y": ytr}).groupby("k")["y"].agg(["sum", "count"])
        m = (st["sum"] + prior * g) / (st["count"] + prior)
        out_tr[name] = enc
        out_va[name] = pd.Series(kva).map(m).fillna(g).to_numpy(dtype=np.float32)
        out_te[name] = pd.Series(k_te).map(m).fillna(g).to_numpy(dtype=np.float32)
    return pd.DataFrame(out_tr), pd.DataFrame(out_va), pd.DataFrame(out_te)
