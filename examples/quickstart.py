"""Corrects forecasts of the daily maximum temperature made at 18:00 with bounds compiled from the weather rules.

Run from the repository root:  python examples/quickstart.py
The model is fitted on the station-days before 2016 and evaluated on the station-days from 2016 on.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import domains  # noqa: E402
import semantics  # noqa: E402

task = domains.TASKS["weather_max"]
table = domains.read_table(task.dataset)

# 1. Compile the rules: which public values bound the target, and from what time on.
ontology = semantics.load_ontology(domains.ONTOLOGY_DIR / task.ontology)
shapes = semantics.load_shapes(domains.ONTOLOGY_DIR / task.shapes)
specs = semantics.compile_bounds(shapes, ontology, task.target)
for s in specs:
    print(f"{s.kind} bound from {s.prop.split('#')[1]}, usable from fraction {s.observable_at:g}")

# 2. Fit any regressor on the information public at 18:00 (fraction 0.75).
t = 0.75
X, categorical = domains.features(task.dataset, table, t)
X = X.drop(columns=categorical)
X = X.fillna(X.median())
y = table[task.target_column].to_numpy(dtype=float)
train = (table["date"] < "2016-01-01").to_numpy()
test = ~train
model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0, n_jobs=-1)
model.fit(X[train], y[train])
raw = model.predict(X[test])

# 3. Project every forecast onto the interval compiled for its station-day at 18:00.
lower, upper = semantics.bounds_at(specs, table, t, task.columns)
corrected = semantics.project(raw, lower[test], upper[test])

outside = np.mean((raw < lower[test]) | (raw > upper[test]))
mae_raw, mae_corrected = np.mean(np.abs(raw - y[test])), np.mean(np.abs(corrected - y[test]))
print(f"forecasts outside the bounds before correction: {100 * outside:.1f}%")
print(f"mean absolute error (deg C): {mae_raw:.3f} before, {mae_corrected:.3f} after correction")

# 4. Check the corrected forecasts with a standard SHACL engine.
sub = table[test]
graph = semantics.prediction_graph(specs, sub, corrected, t, task.columns, task.prediction_class,
                                   task.predicted, task.namespace)
check = semantics.prediction_shapes(specs, t, task.prediction_class, task.predicted, task.namespace)
conforms, results, _ = semantics.shacl_violations(graph, check)
print(f"pySHACL: conforms={conforms}, validation results={results}")
