"""Rolling-origin evaluation with monthly refits for the electricity, bike-sharing, and weather tasks.

Every test month is forecast by models fitted on all earlier units. Forecasts are made at 06:00,
12:00 and 18:00 from the information public at that time and projected onto the bounds compiled
from the task's SHACL shapes and ontology.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import domains
import experiment
import semantics

SEEDS = 3
CAL_SHARE = 0.2
BLOCK_DAYS = 7
STOCHASTIC = ("RandomForest", "XGBoost", "CatBoost")
OUTPUTS = ("raw", "projected", "bound_aware")
INTERVAL_KINDS = ("raw", "projected", "bound_aware")
TOL = 1e-9


def scale(kind):
    """Forward and inverse transforms of the modelling scale."""
    if kind == "log":
        return np.log1p, np.expm1
    return (lambda v: np.asarray(v, dtype=float)), (lambda v: np.asarray(v, dtype=float))


def model_seed(s, month_index):
    return 1000 * (s + 1) + month_index


def seeds_for(learner):
    return range(SEEDS) if learner in STOCHASTIC else range(1)


def impute(Xfit, others, categorical):
    """Fills numeric gaps with the medians of Xfit and categorical gaps with a 'missing' level."""
    numeric = [c for c in Xfit.columns if c not in categorical]
    med = Xfit[numeric].median()
    out = []
    for X in (Xfit, *others):
        X = X.copy()
        X[numeric] = X[numeric].fillna(med)
        for c in categorical:
            X[c] = X[c].fillna("missing")
        out.append(X)
    return out


class Margin:
    """Bound-aware target: distance from the single compiled bound on the modelling scale."""

    def __init__(self, side, fwd, inv, lower, upper):
        self.side, self.inv = side, inv
        self.ref = fwd(lower) if side == "lower" else fwd(upper)

    def target(self, sy, idx):
        return sy - self.ref[idx] if self.side == "lower" else self.ref[idx] - sy

    def value(self, z, idx):
        z = np.maximum(z, 0.0)
        return self.inv(self.ref[idx] + z) if self.side == "lower" else self.inv(self.ref[idx] - z)

    def interval(self, zlo, zhi, idx):
        a, b = self.value(zlo, idx), self.value(zhi, idx)
        return (a, b) if self.side == "lower" else (b, a)


def _interval_row(kind, learner, seed, y, a, b):
    return {"learner": learner, "seed": seed, "interval": kind, "n": int(len(y)),
            "covered": int(np.sum((y >= a - TOL) & (y <= b + TOL))), "width_sum": float(np.sum(b - a))}


def _month_job(t, mi, tr, te, X, categorical, sy, y, lower, upper, dates, kind, side):
    """Fits every learner, seed, and design for one test month at one prediction point."""
    fwd, inv = scale(kind)
    margin = Margin(side, fwd, inv, lower, upper)
    Xtr, Xte = impute(X.iloc[tr], [X.iloc[te]], categorical)
    train_dates = np.unique(dates[tr])
    cut = train_dates[int(math.floor(len(train_dates) * (1 - CAL_SHARE)))]
    fit_idx, cal_idx = tr[dates[tr] < cut], tr[dates[tr] >= cut]
    Xfit, Xcal, Xte_c = impute(X.iloc[fit_idx], [X.iloc[cal_idx], X.iloc[te]], categorical)
    Xcal_te = pd.concat([Xcal, Xte_c])
    ncal = len(cal_idx)
    yte, lo_b, hi_b = y[te], lower[te], upper[te]
    preds, rows = {}, []
    for learner in experiment.MODELS:
        for s in seeds_for(learner):
            seed = model_seed(s, mi)

            def fit(Xa, ya, Xb):
                return experiment.fit_predict(learner, seed, Xa, ya, Xb, categorical, catboost_one_hot=True)

            raw = inv(fit(Xtr, sy[tr], Xte))
            proj = semantics.project(raw, lo_b, hi_b)
            aware = margin.value(fit(Xtr, margin.target(sy[tr], tr), Xte), te)
            assert np.all(np.abs(proj - yte) <= np.abs(raw - yte) + TOL), "projection increased an error"
            for out, p in (("raw", raw), ("projected", proj), ("bound_aware", aware)):
                preds[(learner, s, out)] = p
            pc = fit(Xfit, sy[fit_idx], Xcal_te)
            q = experiment.conformal_quantile(np.abs(sy[cal_idx] - pc[:ncal]), experiment.ALPHA)
            a, b = inv(pc[ncal:] - q), inv(pc[ncal:] + q)
            pa, pb = semantics.project(a, lo_b, hi_b), semantics.project(b, lo_b, hi_b)
            inside = (yte >= a - TOL) & (yte <= b + TOL)
            assert np.all(~inside | ((yte >= pa - TOL) & (yte <= pb + TOL))), "projection lost coverage"
            assert np.all(pb - pa <= b - a + TOL), "projection widened an interval"
            pz = fit(Xfit, margin.target(sy[fit_idx], fit_idx), Xcal_te)
            qz = experiment.conformal_quantile(np.abs(margin.target(sy[cal_idx], cal_idx) - pz[:ncal]),
                                               experiment.ALPHA)
            za, zb = margin.interval(pz[ncal:] - qz, pz[ncal:] + qz, te)
            for k, (u, v) in zip(INTERVAL_KINDS, ((a, b), (pa, pb), (za, zb))):
                rows.append(_interval_row(k, learner, s, yte, u, v))
    qa = experiment.quantile_fit_predict(Xfit, sy[fit_idx], Xcal_te, categorical, experiment.ALPHA / 2)
    qb = experiment.quantile_fit_predict(Xfit, sy[fit_idx], Xcal_te, categorical, 1 - experiment.ALPHA / 2)
    a, b = experiment.cqr_interval(qa, qb, sy[cal_idx], ncal, experiment.ALPHA)
    a, b = inv(a), inv(b)
    pa, pb = semantics.project(a, lo_b, hi_b), semantics.project(b, lo_b, hi_b)
    rows.append(_interval_row("raw", "CQR", 0, yte, a, b))
    rows.append(_interval_row("projected", "CQR", 0, yte, pa, pb))
    for r in rows:
        r.update({"t": t, "month": mi})
    return t, te, preds, rows


def block_bootstrap_ci(diff, unit_dates, rng):
    """Percentile interval of the mean of diff from a circular moving-block bootstrap of test dates."""
    agg = pd.DataFrame({"d": diff, "date": unit_dates}).groupby("date")["d"].agg(["sum", "count"]).sort_index()
    s, c = agg["sum"].to_numpy(), agg["count"].to_numpy()
    n = len(s)
    nb = math.ceil(n / BLOCK_DAYS)
    starts = rng.integers(0, n, size=(experiment.BOOTSTRAP, nb), dtype=np.int64)
    idx = ((starts[:, :, None] + np.arange(BLOCK_DAYS)) % n).reshape(experiment.BOOTSTRAP, -1)[:, :n]
    stat = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def run_task(name, log=print):
    """Rolling-origin evaluation of one task; returns results, predictions, interval rows, and timings."""
    task = domains.TASKS[name]
    table = domains.read_table(task.dataset)
    t0 = time.perf_counter()
    ontology = semantics.load_ontology(domains.ONTOLOGY_DIR / task.ontology)
    shapes = semantics.load_shapes(domains.ONTOLOGY_DIR / task.shapes)
    t1 = time.perf_counter()
    specs = semantics.compile_bounds(shapes, ontology, task.target)
    t2 = time.perf_counter()
    sides = {s.kind for s in specs}
    assert len(sides) == 1, sides
    side = sides.pop()
    y = table[task.target_column].to_numpy(dtype=float)
    fwd, _ = scale(task.scale)
    sy = fwd(y)
    dates = table["date"].to_numpy()
    months = table["date"].dt.to_period("M")
    test_months = sorted(m for m in months.unique() if str(m) >= task.first_test_month)
    jobs = []
    for t in domains.POINTS:
        X, categorical = domains.features(task.dataset, table, t)
        lower, upper = semantics.bounds_at(specs, table, t, task.columns)
        assert np.all(y >= lower - TOL) and np.all(y <= upper + TOL), "a compiled bound fails on observed data"
        for mi, month in enumerate(test_months):
            te = np.where((months == month).to_numpy())[0]
            tr = np.where(table["date"].to_numpy() < month.start_time.to_datetime64())[0]
            jobs.append((t, mi, tr, te, X, categorical, lower, upper))
    started = time.perf_counter()
    parts = Parallel(n_jobs=experiment.WORKERS, backend="loky")(
        delayed(_month_job)(t, mi, tr, te, X, cat, sy, y, lo, hi, dates, task.scale, side)
        for t, mi, tr, te, X, cat, lo, hi in jobs)
    fit_seconds = time.perf_counter() - started
    log(f"{name}: {len(jobs)} month jobs, {fit_seconds:.0f} s")
    return task, table, specs, test_months, parts, {"load_rules_s": t1 - t0, "compile_s": t2 - t1,
                                                    "fit_s": fit_seconds}


def collect(task, table, specs, parts):
    """Per-unit prediction matrices (units x seeds) for each point, learner, and output."""
    P, units, intervals = {}, {}, []
    for t, te, preds, rows in parts:
        units.setdefault(t, []).append(te)
        for (learner, s, out), p in preds.items():
            P.setdefault((t, learner, out), {}).setdefault(s, []).append(p)
        intervals += rows
    mats, order = {}, {}
    for t, tes in units.items():
        order[t] = np.concatenate(tes)
    for (t, learner, out), by_seed in P.items():
        mats[(t, learner, out)] = np.column_stack([np.concatenate(by_seed[s]) for s in sorted(by_seed)])
    return mats, order, pd.DataFrame(intervals)


def paired(task, table, specs, mats, order, rng_base):
    y_all = table[task.target_column].to_numpy(dtype=float)
    out = {}
    for ti, t in enumerate(domains.POINTS):
        te = order[t]
        y = y_all[te]
        lower, upper = semantics.bounds_at(specs, table, t, task.columns)
        lo, hi = lower[te], upper[te]
        bound = lo if np.isfinite(lo).all() else hi
        unit_dates = table["date"].to_numpy()[te]
        for li, learner in enumerate(experiment.MODELS):
            rng = np.random.default_rng(rng_base + 10 * li + ti)
            M = {o: mats[(t, learner, o)] for o in OUTPUTS}
            A = {o: np.abs(M[o] - y[:, None]) for o in OUTPUTS}
            err = {o: A[o].mean(axis=1) for o in OUTPUTS}
            d = err["raw"] - err["projected"]
            da = err["bound_aware"] - err["projected"]
            outside = (M["raw"] < lo[:, None] - TOL) | (M["raw"] > hi[:, None] + TOL)
            per_seed = {o: A[o].mean(axis=0) for o in OUTPUTS}
            out.setdefault(f"{t:g}", {})[learner] = {
                "mae_raw": float(err["raw"].mean()), "mae_projected": float(err["projected"].mean()),
                "mae_bound_aware": float(err["bound_aware"].mean()),
                "rmse_raw": float(np.mean(np.sqrt(np.mean((M["raw"] - y[:, None]) ** 2, axis=0)))),
                "rmse_projected": float(np.mean(np.sqrt(np.mean((M["projected"] - y[:, None]) ** 2, axis=0)))),
                "rmse_bound_aware": float(np.mean(np.sqrt(np.mean((M["bound_aware"] - y[:, None]) ** 2, axis=0)))),
                "reduction_pct": float(100 * d.mean() / err["raw"].mean()),
                "mean_decrease": float(d.mean()), "ci": block_bootstrap_ci(d, unit_dates, rng),
                "aware_minus_projected": float(da.mean()),
                "aware_minus_projected_ci": block_bootstrap_ci(da, unit_dates, rng),
                "violation_rate": float(outside.mean()),
                "units_repaired": int(outside.any(axis=1).sum()),
                "units_worsened": int((A["projected"] > A["raw"] + TOL).any(axis=1).sum()),
                "seeds": int(M["raw"].shape[1]),
                "reduction_pct_by_seed": [float(100 * (r - p) / r) for r, p in
                                          zip(per_seed["raw"], per_seed["projected"])],
                "test_units": int(len(te)),
                "mae_bound": float(np.mean(np.abs(bound - y))),
                "share_on_bound": float(np.mean(np.abs(bound - y) < TOL)),
            }
    return out


def summarise_intervals(intervals):
    out = {}
    for (t, learner, kind), g in intervals.groupby(["t", "learner", "interval"]):
        out.setdefault(f"{t:g}", {}).setdefault(learner, {})[kind] = {
            "coverage": float(g["covered"].sum() / g["n"].sum()), "width": float(g["width_sum"].sum() / g["n"].sum())}
    return out


def validate(task, table, specs, mats, order):
    """pySHACL validation of every unconstrained and projected forecast, seed by seed."""
    out, nodes, seconds = {}, 0, 0.0
    for t in domains.POINTS:
        sub = table.iloc[order[t]]
        lower, upper = semantics.bounds_at(specs, sub, t, task.columns)
        lo2, hi2 = semantics.stored(lower), semantics.stored(upper)
        shapes = semantics.prediction_shapes(specs, t, task.prediction_class, task.predicted, task.namespace)
        for learner in experiment.MODELS:
            for output in ("raw", "projected"):
                M = mats[(t, learner, output)]
                total, focus, numeric = 0, 0, 0
                for s in range(M.shape[1]):
                    g = semantics.prediction_graph(specs, sub, M[:, s], t, task.columns, task.prediction_class,
                                                   task.predicted, task.namespace)
                    started = time.perf_counter()
                    _, nv, nf = semantics.shacl_violations(g, shapes)
                    seconds += time.perf_counter() - started
                    nodes += len(sub)
                    total += nv
                    focus += nf
                    pred = semantics.stored(M[:, s])
                    numeric += int(((pred < lo2) | (pred > hi2)).sum())
                out.setdefault(f"{t:g}", {}).setdefault(learner, {})[output] = {
                    "shacl_results": total, "shacl_focus_nodes": focus, "numeric_violations": numeric}
    return out, seconds / nodes * 1000


def validate_observed(dataset, table):
    """pySHACL validation of the observed values against every rule of the dataset's shapes file."""
    tasks = [task for task in domains.TASKS.values() if task.dataset == dataset]
    task = tasks[0]
    columns = {}
    for tk in tasks:
        columns.update(tk.columns)
    g = semantics.observation_graph(table, task.unit_class, columns, task.namespace)
    shapes = semantics.load_shapes(domains.ONTOLOGY_DIR / task.shapes)
    conforms, n, _ = semantics.shacl_violations(g, shapes)
    return {"conforms": conforms, "violations": n, "units": int(len(table))}


def projection_cost():
    """Nanoseconds per forecast for projecting one million forecasts onto their intervals."""
    rng = np.random.default_rng(0)
    p, lo = rng.normal(size=1_000_000), rng.normal(size=1_000_000)
    hi = lo + rng.random(1_000_000)
    best = math.inf
    for _ in range(5):
        started = time.perf_counter()
        semantics.project(p, lo, hi)
        best = min(best, time.perf_counter() - started)
    return best / 1_000_000 * 1e9


def run(outdir, names=tuple(domains.TASKS), log=print):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = {"settings": {"seeds": SEEDS, "calibration_share": CAL_SHARE, "block_days": BLOCK_DAYS,
                            "alpha": experiment.ALPHA, "bootstrap": experiment.BOOTSTRAP,
                            "prediction_hours": [domains.HOURS[t] for t in domains.POINTS],
                            "learners": list(experiment.MODELS)},
               "tasks": {}, "observed": {}}
    runtime = {"tasks": {}, "workers": experiment.WORKERS}
    pred_frames, month_frames = [], []
    for dataset in dict.fromkeys(domains.TASKS[n].dataset for n in names):
        results["observed"][dataset] = validate_observed(dataset, domains.read_table(dataset))
    for k, name in enumerate(names):
        task, table, specs, test_months, parts, timing = run_task(name, log)
        mats, order, intervals = collect(task, table, specs, parts)
        v, shacl_s = validate(task, table, specs, mats, order)
        results["tasks"][name] = {
            "unit": task.unit, "scale": task.scale,
            "data": {"units": int(len(table)), "first_date": str(table["date"].min().date()),
                     "last_date": str(table["date"].max().date()), "test_months": len(test_months),
                     "first_test_month": str(test_months[0]), "last_test_month": str(test_months[-1]),
                     "test_units": int(len(order[domains.POINTS[0]])),
                     "mean_target": float(table[task.target_column].mean())},
            "compiled_bounds": [{"property": str(s.prop).split("#")[1], "observable_at": s.observable_at,
                                 "kind": s.kind, "name": s.name} for s in specs],
            "point": paired(task, table, specs, mats, order, 500 + 100 * k),
            "intervals": summarise_intervals(intervals),
            "validation": v,
        }
        runtime["tasks"][name] = {**{key: round(value, 4) for key, value in timing.items()},
                                  "shacl_s_per_1000_forecasts": round(shacl_s, 2)}
        for (t, learner, out), M in mats.items():
            pred_frames.append(pd.DataFrame({"task": name, "t": t, "learner": learner, "output": out,
                                             "unit": table.index[order[t]], "prediction": M.mean(axis=1)}))
        intervals.insert(0, "task", name)
        month_frames.append(intervals)
        log(f"{name}: done, {time.perf_counter() - started:.0f} s")
    runtime["projection_ns_per_forecast"] = round(projection_cost(), 2)
    runtime["total_s"] = round(time.perf_counter() - started)
    with open(outdir / "periods.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    with open(outdir / "periods_runtime.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(runtime, fh, indent=2, sort_keys=True)
    pd.concat(month_frames).to_csv(outdir / "periods_interval_metrics.csv", index=False, float_format="%.10g",
                                   lineterminator="\n")
    pd.concat(pred_frames).to_csv(outdir / "periods_predictions.csv.gz", index=False, float_format="%.6f",
                                  lineterminator="\n", compression={"method": "gzip", "mtime": 0})
    return results
