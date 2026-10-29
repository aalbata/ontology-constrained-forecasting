"""Repeated cross-validation of four regressors with SHACL-compiled feasibility projection."""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

import semantics
import snapshots

MODELS = ("Ridge", "RandomForest", "XGBoost", "CatBoost")
REPEATS = 5
FOLDS = 5
ALPHA = 0.10
BOOTSTRAP = 10_000
THREADS = 1
WORKERS = max(1, min(12, (os.cpu_count() or 2) - 2))


def split_seed(r):
    return r + 1


def model_seed(r, f):
    return 1000 * (r + 1) + f


def make_model(name, seed, categorical):
    """Fixed configurations, applied identically to every design and prediction point."""
    numeric = lambda X: [c for c in X.columns if c not in categorical]
    if name == "Ridge":
        return ("sk", lambda X: make_pipeline(
            ColumnTransformer([("num", StandardScaler(), numeric(X)),
                               ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)]),
            Ridge(alpha=5.0)))
    if name == "RandomForest":
        return ("sk", lambda X: make_pipeline(
            ColumnTransformer([("num", "passthrough", numeric(X)),
                               ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)]),
            RandomForestRegressor(n_estimators=350, min_samples_leaf=2, max_features=0.7,
                                  random_state=seed, n_jobs=THREADS)))
    if name == "XGBoost":
        return ("sk", lambda X: make_pipeline(
            ColumnTransformer([("num", "passthrough", numeric(X)),
                               ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)]),
            XGBRegressor(n_estimators=350, max_depth=3, learning_rate=0.035, subsample=0.85,
                         colsample_bytree=0.85, reg_lambda=2.0, random_state=seed, n_jobs=THREADS,
                         tree_method="hist")))
    if name == "CatBoost":
        return ("cb", lambda X: CatBoostRegressor(iterations=450, depth=5, learning_rate=0.035, l2_leaf_reg=4.0,
                                                  random_seed=seed, thread_count=THREADS, verbose=0,
                                                  allow_writing_files=False))
    raise ValueError(name)


def fit_predict(name, seed, Xtr, ytr, Xte, categorical):
    kind, factory = make_model(name, seed, categorical)
    m = factory(Xtr)
    if kind == "cb":
        m.fit(Xtr, ytr, cat_features=[Xtr.columns.get_loc(c) for c in categorical])
    else:
        m.fit(Xtr, ytr)
    return m.predict(Xte)


def fold_metrics(y, pred, lower):
    ly, lp = np.log1p(y), np.log1p(np.maximum(pred, 0.0))
    res = ly - lp
    return {
        "log_mae": float(np.mean(np.abs(res))),
        "log_rmse": float(np.sqrt(np.mean(res ** 2))),
        "log_r2": float(1 - np.sum(res ** 2) / np.sum((ly - ly.mean()) ** 2)),
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "wape": float(np.sum(np.abs(y - pred)) / np.sum(y)),
        "violation_rate": float(np.mean(pred < lower - 1e-9)),
    }


def conformal_quantile(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(np.sort(scores)[min(k, n) - 1])


def bootstrap_ci(diff, rng):
    n = len(diff)
    idx = rng.integers(0, n, size=(BOOTSTRAP, n))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(table, outdir, log=print):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ontology, shapes = semantics.load_ontology(), semantics.load_shapes()
    specs = semantics.compile_bounds(shapes, ontology)
    y = table["price"].to_numpy(dtype=float)
    strata = table["item"].to_numpy()
    started = time.perf_counter()

    fold_rows, oof_rows, interval_rows = [], [], []
    labels_by_t = {}
    for t in snapshots.PREDICTION_POINTS:
        X = snapshots.features(table, t)
        lab = snapshots.state_labels(table, t)
        labels_by_t[t] = lab
        Xlab = pd.concat([X, lab], axis=1)
        lower, upper = semantics.bounds_at(specs, table, t)
        tasks = []
        for r in range(REPEATS):
            skf = StratifiedKFold(FOLDS, shuffle=True, random_state=split_seed(r))
            tasks += [(r, f, tr, te) for f, (tr, te) in enumerate(skf.split(X, strata), start=1)]
        parts = Parallel(n_jobs=WORKERS, backend="loky")(
            delayed(_fold_task)(t, r, f, tr, te, X, Xlab, list(lab.columns), y, strata, lower, upper, table.index)
            for r, f, tr, te in tasks)
        for a, b, c in parts:
            fold_rows += a
            oof_rows += b
            interval_rows += c
        log(f"t={t:g} done, {time.perf_counter() - started:.0f} s")

    folds = pd.DataFrame(fold_rows)
    intervals = pd.DataFrame(interval_rows)
    oof = pd.DataFrame(oof_rows, columns=["t", "repeat", "model", "output", "auctionid", "prediction"])
    folds.to_csv(outdir / "fold_metrics.csv", index=False, float_format="%.10g", lineterminator="\n")
    intervals.to_csv(outdir / "interval_metrics.csv", index=False, float_format="%.10g", lineterminator="\n")
    oof.to_csv(outdir / "oof_predictions.csv", index=False, float_format="%.6f", lineterminator="\n")
    cv_seconds = time.perf_counter() - started

    results = {"settings": {"repeats": REPEATS, "folds": FOLDS, "alpha": ALPHA, "bootstrap": BOOTSTRAP,
                            "prediction_points": list(snapshots.PREDICTION_POINTS), "models": list(MODELS),
                            "split_seeds": [split_seed(r) for r in range(REPEATS)]},
               "compiled_bounds": [{"property": str(s.prop).split("#")[1], "observable_at": s.observable_at,
                                    "kind": s.kind, "name": s.name} for s in specs]}
    results["data"] = data_summary(table)
    results["fold_means"] = summarise_folds(folds)
    results["repeat_sd"] = repeat_variability(oof, table)
    results["paired"] = paired_analysis(oof, table, specs)
    results["intervals"] = summarise_intervals(intervals)
    results["shap"] = shap_groups(table, 0.5)
    results["knowledge_graph"] = knowledge_graph(table, labels_by_t, oof, specs, outdir)
    with open(outdir / "results.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    # Wall-clock times vary between runs, so they are kept apart from the deterministic results.
    runtime = {"cross_validation": round(cv_seconds), "total": round(time.perf_counter() - started),
               "workers": WORKERS}
    with open(outdir / "runtime.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(runtime, fh, indent=2, sort_keys=True)
    return results


def _fold_task(t, r, f, tr, te, X, Xlab, label_cols, y, strata, lower, upper, index):
    """Fits every model and design on one outer fold."""
    ly = np.log1p(y)
    lz = np.log1p(lower)
    seed = model_seed(r, f)
    cat_plain, cat_lab = ["item"], ["item"] + label_cols
    fold_rows, oof_rows, interval_rows = [], [], []
    for name in MODELS:
        raw = np.expm1(fit_predict(name, seed, X.iloc[tr], ly[tr], X.iloc[te], cat_plain))
        proj = semantics.project(raw, lower[te], upper[te])
        z = fit_predict(name, seed, X.iloc[tr], ly[tr] - lz[tr], X.iloc[te], cat_plain)
        aware = np.expm1(lz[te] + np.maximum(z, 0.0))
        labelled = np.expm1(fit_predict(name, seed, Xlab.iloc[tr], ly[tr], Xlab.iloc[te], cat_lab))
        outputs = {"raw": raw, "projected": proj, "bound_aware": aware, "labels_raw": labelled,
                   "labels_projected": semantics.project(labelled, lower[te], upper[te])}
        for out, pred in outputs.items():
            fold_rows.append({"t": t, "repeat": r, "fold": f, "model": name, "output": out,
                              **fold_metrics(y[te], pred, lower[te])})
            oof_rows += [(t, r, name, out, int(index[i]), float(p)) for i, p in zip(te, pred)]
        # Split-conformal interval on the log scale, then projected onto [L, U].
        fit_idx, cal_idx = train_test_split(tr, test_size=0.2, random_state=seed, stratify=strata[tr])
        pc = fit_predict(name, seed, X.iloc[fit_idx], ly[fit_idx],
                         pd.concat([X.iloc[cal_idx], X.iloc[te]]), cat_plain)
        q = conformal_quantile(np.abs(ly[cal_idx] - pc[:len(cal_idx)]), ALPHA)
        centre = pc[len(cal_idx):]
        lo, hi = np.expm1(centre - q), np.expm1(centre + q)
        plo, phi = semantics.project(lo, lower[te], upper[te]), semantics.project(hi, lower[te], upper[te])
        for kind, a, b in (("raw", lo, hi), ("projected", plo, phi)):
            interval_rows.append({"t": t, "repeat": r, "fold": f, "model": name, "interval": kind,
                                  "coverage": float(np.mean((y[te] >= a - 1e-9) & (y[te] <= b + 1e-9))),
                                  "width": float(np.mean(b - a)), "n": int(len(te))})
    return fold_rows, oof_rows, interval_rows


def data_summary(table):
    t3 = []
    for item, g in [*table.groupby("item", sort=True), ("All", table)]:
        t3.append({"item": item, "auctions": int(len(g)), "mean_close": float(g["price"].mean()),
                   "median_close": float(g["price"].median()), "mean_open": float(g["open"].mean()),
                   "bids_at_50": float(g["bids@0.5"].mean()), "bidders_at_50": float(g["bidders@0.5"].mean()),
                   "no_bid_at_50": int((g["bids@0.5"] == 0).sum())})
    bound = {f"{t:g}": {"standing_equals_price": int(np.isclose(table[f"standing@{t:g}"], table["price"]).sum()),
                        "standing_exceeds_price": int((table[f"standing@{t:g}"] > table["price"] + 1e-9).sum())}
             for t in snapshots.PREDICTION_POINTS}
    return {"bid_records": int(table["n_records"].sum()), "auctions": int(len(table)),
            "by_item": t3,
            "max_record_equals_price": int(np.isclose(table["max_record"], table["price"]).sum()),
            "max_record_exceeds_price": int((table["max_record"] > table["price"] + 1e-9).sum()),
            "standing_vs_price": bound}


def summarise_folds(folds):
    cols = ["log_mae", "log_rmse", "log_r2", "mae", "rmse", "wape", "violation_rate"]
    out = {}
    for (t, model, output), g in folds.groupby(["t", "model", "output"]):
        out.setdefault(f"{t:g}", {}).setdefault(model, {})[output] = {
            **{c: float(g[c].mean()) for c in cols}, **{c + "_sd": float(g[c].std(ddof=1)) for c in cols}}
    return out


def _per_auction(oof, t, model, output, index):
    s = oof[(oof.t == t) & (oof.model == model) & (oof.output == output)]
    wide = s.pivot(index="auctionid", columns="repeat", values="prediction").reindex(index)
    return wide.to_numpy()


def repeat_variability(oof, table):
    y = table["price"].to_numpy(dtype=float)
    out = {}
    for t in snapshots.PREDICTION_POINTS:
        for model in MODELS:
            for output in ("raw", "projected", "bound_aware"):
                P = _per_auction(oof, t, model, output, table.index)
                per_rep = np.abs(P - y[:, None]).mean(axis=0)
                out.setdefault(f"{t:g}", {}).setdefault(model, {})[output] = {
                    "mae_by_repeat": [float(v) for v in per_rep], "mean": float(per_rep.mean()),
                    "sd": float(per_rep.std(ddof=1))}
    return out


def paired_analysis(oof, table, specs):
    y = table["price"].to_numpy(dtype=float)
    ly = np.log1p(y)
    out = {}
    for t in snapshots.PREDICTION_POINTS:
        lower, _ = semantics.bounds_at(specs, table, t)
        for k, model in enumerate(MODELS):
            rng = np.random.default_rng(100 + 10 * k + int(t * 100))
            P = {o: _per_auction(oof, t, model, o, table.index) for o in ("raw", "projected", "bound_aware")}
            err = {o: np.abs(P[o] - y[:, None]).mean(axis=1) for o in P}
            lerr = {o: np.abs(np.log1p(P[o]) - ly[:, None]).mean(axis=1) for o in P}
            d = err["raw"] - err["projected"]
            ld = lerr["raw"] - lerr["projected"]
            da = err["bound_aware"] - err["projected"]
            repaired = (P["raw"] < lower[:, None] - 1e-9).any(axis=1)
            out.setdefault(f"{t:g}", {})[model] = {
                "mae_raw": float(err["raw"].mean()), "mae_projected": float(err["projected"].mean()),
                "mae_bound_aware": float(err["bound_aware"].mean()),
                "reduction_pct": float(100 * d.mean() / err["raw"].mean()),
                "mean_decrease": float(d.mean()), "ci": bootstrap_ci(d, rng),
                "log_mae_raw": float(lerr["raw"].mean()), "log_mae_projected": float(lerr["projected"].mean()),
                "log_reduction_pct": float(100 * ld.mean() / lerr["raw"].mean()),
                "log_mean_decrease": float(ld.mean()), "log_ci": bootstrap_ci(ld, rng),
                "aware_minus_projected": float(da.mean()), "aware_minus_projected_ci": bootstrap_ci(da, rng),
                "auctions_repaired": int(repaired.sum()),
                "auctions_worsened": int((d < -1e-9).sum()),
                "gain_share_repaired": float(d[repaired].sum() / d.sum()) if d.sum() > 0 else 0.0,
                "repaired_mae_raw": float(err["raw"][repaired].mean()) if repaired.any() else 0.0,
                "repaired_mae_projected": float(err["projected"][repaired].mean()) if repaired.any() else 0.0,
            }
    return out


def summarise_intervals(intervals):
    out = {}
    for (t, model, kind), g in intervals.groupby(["t", "model", "interval"]):
        w = g["n"] / g["n"].sum()
        out.setdefault(f"{t:g}", {}).setdefault(model, {})[kind] = {
            "coverage": float((g["coverage"] * w).sum()), "width": float((g["width"] * w).sum())}
    return out


def shap_groups(table, t):
    import shap
    X = snapshots.features(table, t)
    Xd = pd.get_dummies(X, columns=["item"], dtype=float)
    m = RandomForestRegressor(n_estimators=350, min_samples_leaf=2, max_features=0.7, random_state=7,
                              n_jobs=THREADS).fit(Xd, np.log1p(table["price"].to_numpy(dtype=float)))
    sv = shap.TreeExplainer(m).shap_values(Xd)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=Xd.columns)
    groups = {}
    for gname, feats in snapshots.FEATURE_GROUPS.items():
        cols = [c for c in Xd.columns if c in feats or any(c.startswith(f + "_") and f == "item" for f in feats)]
        groups[gname] = float(imp[cols].sum())
    total = sum(groups.values())
    assert abs(total - imp.sum()) < 1e-9 * max(1.0, total)
    top = imp.sort_values(ascending=False).head(5)
    return {"t": t, "group_share_pct": {k: 100 * v / total for k, v in groups.items()},
            "top_features": [{"feature": k, "mean_abs_shap": float(v)} for k, v in top.items()]}


def knowledge_graph(table, labels_by_t, oof, specs, outdir):
    shapes = semantics.load_shapes()
    data_graph = semantics.build_graph(table, labels_by_t)
    conforms, n_data, _ = semantics.shacl_violations(data_graph, shapes)
    validation = {"observed_data": {"conforms": conforms, "violations": n_data}, "predictions": {}}
    for t in snapshots.PREDICTION_POINTS:
        pshapes = semantics.prediction_shapes(specs, t)
        for model in MODELS:
            for output in ("raw", "projected"):
                P = _per_auction(oof, t, model, output, table.index)
                total, nodes, n_bad = 0, 0, 0
                for r in range(P.shape[1]):
                    g = semantics.prediction_graph(specs, table, P[:, r], t)
                    _, nv, nf = semantics.shacl_violations(g, pshapes)
                    total += nv
                    nodes += nf
                    lower, upper = semantics.bounds_at(specs, table, t)
                    n_bad += int(((P[:, r] < lower - 0.005) | (P[:, r] > upper + 0.005)).sum())
                validation["predictions"].setdefault(f"{t:g}", {}).setdefault(model, {})[output] = {
                    "shacl_results": total, "shacl_focus_nodes": nodes, "numeric_violations": n_bad}
    mean_preds = {}
    for output in ("raw", "projected"):
        P = _per_auction(oof, 0.5, "RandomForest", output, table.index)
        mean_preds[("RandomForest", output, 0.5)] = P.mean(axis=1)
    kg = semantics.build_graph(table, labels_by_t, mean_preds)
    kg.serialize(outdir / "auction_kg.ttl", format="turtle")
    return {"statistics": semantics.graph_statistics(kg), "validation": validation,
            "queries": semantics.run_queries(kg)}
