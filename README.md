# Compiling Knowledge Graph Rules into Feasibility Bounds for Forecasts of Partially Observed Quantities

Code, ontologies, rule files, and data for the article "Compiling Knowledge Graph Rules into Feasibility Bounds for Forecasts of Partially Observed Quantities" by Snwr Jamal Mohammed, Mohammad F. Al-Hammouri, Lucine Gharibian, Samia Kouki, and Ali Al Bataineh.

Repository: https://github.com/aalbata/ontology-constrained-forecasting

Some forecasts are made while the quantity being forecast is still being determined, such as the energy a household will use today, the rentals of a bicycle-sharing system, the day's maximum or minimum temperature, or the price at which an online auction will close. The part already observed bounds the final value. This package states such rules once, as SHACL property-pair shapes with an OWL annotation of when each value becomes public, compiles them into a feasible interval for every forecast, projects the point forecasts and prediction-interval end points of any regressor onto that interval, and checks the corrected forecasts with pySHACL.

## Quick start

```
python -m pip install -r requirements.txt
python examples/quickstart.py
```

The example compiles the weather rules, fits a Random Forest to the daily maximum temperature at 18:00, corrects its forecasts, and validates them with pySHACL in about a minute. The same four calls apply to any table and any model:

```python
import semantics
ontology = semantics.load_ontology("ontology/weather_ontology.ttl")
shapes = semantics.load_shapes("ontology/weather_shapes.ttl")
specs = semantics.compile_bounds(shapes, ontology, target)      # rules -> bound specifications
lower, upper = semantics.bounds_at(specs, table, t, columns)    # feasible interval of every case at time t
corrected = semantics.project(forecasts, lower, upper)          # nearest feasible value
```

`columns` maps each bounding property of the ontology to the table column that holds its value. To use the compiler in a new domain, write a small ontology that imports `ontology/core.ttl` (the target property and the `fc:observableAtFraction` of every public value) and a shapes file with one `sh:lessThanOrEquals` relation per rule; the compiler itself needs no changes. All computation is local, and no data leave the machine.

## Contents

| Path | Description |
|---|---|
| `reproduce.py` | Entry point. Runs every experiment and draws every figure. |
| `semantics.py` | Rule compiler, projection, knowledge graph, pySHACL validation, SPARQL queries. |
| `snapshots.py` | Public auction snapshots, features, and ontology state classes. |
| `domains.py` | Daily tables, public values, targets, and features of the energy, rental, and weather tasks. |
| `experiment.py` | Auction study: repeated cross-validation, designs, intervals, paired bootstrap. |
| `rolling.py` | Daily tasks: rolling origin with monthly refits, designs, intervals, block bootstrap, validation. |
| `figures.py` | Figures 1 to 5 of the article from the results files and the ontology. |
| `examples/quickstart.py` | Short end-to-end use of the compiler on the weather data. |
| `ontology/` | Shared vocabulary (`core.ttl`), one ontology and one shapes file per domain. |
| `data/` | Derived tables: auction snapshots and the daily tables of the three UCI data sets. |
| `results/` | Reference outputs: `results.json` (auction study), `periods.json` (daily tasks), per-fold and per-month interval metrics, out-of-fold and seed-averaged predictions, and the auction knowledge graph. |
| `figures/` | Reference figures in PDF and PNG. |

## Requirements

Python 3.12 and the pinned packages in `requirements.txt`. The reference outputs were produced on Windows 11 with an Intel Core Ultra 7 155U processor (14 logical cores), 16 GB of memory, and Python 3.12.10; the complete run took 2.6 hours with 12 worker processes and no graphics processor.

## Reproduction

Fast path, from the shipped tables: `python reproduce.py`.

Full path, downloading the raw archives from Zenodo and the UCI Machine Learning Repository, rebuilding every table, and checking that each rebuilt table is identical to the shipped one: `python reproduce.py --from-raw`. Local copies can be supplied with `--zip` (auction archive) and `--raw-dir` (folder with `electricity.zip`, `bike.zip`, and `weather.zip`); their SHA-256 checksums are verified.

Figures only, from existing results: `python reproduce.py --figures-only`.

## Results

Point and interval forecasts for every task, forecast time, and model. Auction errors average each auction's five out-of-fold forecasts; daily-task errors average each case's forecasts over three seeds (one fit for ridge regression).

| Task | Time | Model | Outside (%) | MAE unconstrained | MAE projected | Reduction (%) | 95% CI of decrease | MAE bound-aware | Coverage projected (%) | Width reduction (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| Auction closing price (US$) | 25% | Ridge | 4.5 | 135.17 | 109.13 | 19.26 | [12.24, 44.04] | 109.50 | 90.6 | 12.5 |
| Auction closing price (US$) | 25% | Random Forest | 0.6 | 86.96 | 82.00 | 5.70 | [0.02, 14.58] | 90.49 | 89.1 | 8.3 |
| Auction closing price (US$) | 25% | XGBoost | 3.6 | 88.74 | 83.24 | 6.20 | [0.25, 15.76] | 91.09 | 89.1 | 8.1 |
| Auction closing price (US$) | 25% | CatBoost | 4.2 | 88.78 | 82.86 | 6.67 | [0.49, 16.21] | 89.89 | 89.1 | 9.0 |
| Auction closing price (US$) | 50% | Ridge | 5.0 | 119.92 | 95.02 | 20.76 | [12.15, 41.62] | 95.26 | 90.9 | 15.4 |
| Auction closing price (US$) | 50% | Random Forest | 1.0 | 78.19 | 72.97 | 6.68 | [0.10, 14.82] | 75.42 | 90.1 | 10.3 |
| Auction closing price (US$) | 50% | XGBoost | 3.2 | 80.21 | 73.98 | 7.77 | [0.46, 16.77] | 78.05 | 89.9 | 10.5 |
| Auction closing price (US$) | 50% | CatBoost | 3.9 | 81.06 | 74.15 | 8.53 | [1.04, 17.20] | 77.44 | 90.1 | 11.1 |
| Auction closing price (US$) | 75% | Ridge | 5.7 | 99.37 | 76.76 | 22.75 | [11.73, 36.67] | 76.71 | 90.1 | 20.1 |
| Auction closing price (US$) | 75% | Random Forest | 1.6 | 62.45 | 57.86 | 7.35 | [0.20, 13.12] | 60.13 | 90.2 | 13.1 |
| Auction closing price (US$) | 75% | XGBoost | 4.2 | 64.64 | 58.91 | 8.88 | [0.71, 15.16] | 60.14 | 90.9 | 13.5 |
| Auction closing price (US$) | 75% | CatBoost | 4.7 | 66.08 | 60.06 | 9.11 | [0.91, 15.43] | 61.13 | 89.9 | 14.3 |
| Daily household energy (kWh) | 06:00 | Ridge | 0.0 | 4.684 | 4.684 | 0.00 | [0.000, 0.000] | 4.695 | 89.9 | 0.0 |
| Daily household energy (kWh) | 06:00 | Random Forest | 0.0 | 4.663 | 4.663 | 0.00 | [0.000, 0.000] | 4.778 | 89.2 | 0.0 |
| Daily household energy (kWh) | 06:00 | XGBoost | 0.0 | 4.661 | 4.661 | 0.00 | [0.000, 0.000] | 4.755 | 88.8 | 0.0 |
| Daily household energy (kWh) | 06:00 | CatBoost | 0.0 | 4.599 | 4.599 | 0.00 | [0.000, 0.000] | 4.737 | 88.9 | 0.0 |
| Daily household energy (kWh) | 12:00 | Ridge | 0.0 | 3.746 | 3.746 | 0.00 | [0.000, 0.000] | 3.752 | 89.3 | 0.0 |
| Daily household energy (kWh) | 12:00 | Random Forest | 0.0 | 3.936 | 3.936 | 0.00 | [0.000, 0.000] | 3.745 | 89.4 | 2.5 |
| Daily household energy (kWh) | 12:00 | XGBoost | 0.0 | 3.958 | 3.958 | 0.00 | [0.000, 0.000] | 3.773 | 89.1 | 2.6 |
| Daily household energy (kWh) | 12:00 | CatBoost | 0.0 | 3.812 | 3.812 | 0.00 | [0.000, 0.000] | 3.657 | 89.6 | 1.7 |
| Daily household energy (kWh) | 18:00 | Ridge | 0.0 | 2.560 | 2.560 | 0.00 | [0.000, 0.000] | 2.560 | 89.5 | 0.1 |
| Daily household energy (kWh) | 18:00 | Random Forest | 0.0 | 2.649 | 2.649 | 0.00 | [0.000, 0.000] | 2.499 | 88.8 | 4.1 |
| Daily household energy (kWh) | 18:00 | XGBoost | 0.0 | 2.666 | 2.666 | 0.00 | [0.000, 0.000] | 2.546 | 89.4 | 5.3 |
| Daily household energy (kWh) | 18:00 | CatBoost | 0.0 | 2.653 | 2.653 | 0.00 | [0.000, 0.000] | 2.502 | 89.7 | 4.7 |
| Daily bicycle rentals | 06:00 | Ridge | 0.0 | 989.5 | 989.5 | 0.00 | [0.0, 0.0] | 1014.5 | 86.9 | 0.0 |
| Daily bicycle rentals | 06:00 | Random Forest | 0.0 | 1044.2 | 1044.2 | 0.00 | [0.0, 0.0] | 1060.3 | 85.1 | 0.0 |
| Daily bicycle rentals | 06:00 | XGBoost | 0.0 | 1107.2 | 1107.2 | 0.00 | [0.0, 0.0] | 1026.4 | 82.1 | 0.0 |
| Daily bicycle rentals | 06:00 | CatBoost | 0.0 | 1094.0 | 1094.0 | 0.00 | [0.0, 0.0] | 1000.9 | 87.6 | 0.0 |
| Daily bicycle rentals | 12:00 | Ridge | 0.3 | 733.7 | 733.5 | 0.04 | [0.0, 0.8] | 758.4 | 82.8 | 0.1 |
| Daily bicycle rentals | 12:00 | Random Forest | 0.0 | 515.2 | 515.2 | 0.00 | [0.0, 0.0] | 454.3 | 86.4 | 0.0 |
| Daily bicycle rentals | 12:00 | XGBoost | 0.0 | 558.0 | 558.0 | 0.00 | [0.0, 0.0] | 476.1 | 82.8 | 0.0 |
| Daily bicycle rentals | 12:00 | CatBoost | 0.0 | 608.1 | 608.1 | 0.00 | [0.0, 0.0] | 442.0 | 78.1 | 0.0 |
| Daily bicycle rentals | 18:00 | Ridge | 0.0 | 179.0 | 179.0 | 0.00 | [0.0, 0.0] | 170.7 | 89.3 | 0.3 |
| Daily bicycle rentals | 18:00 | Random Forest | 0.4 | 277.2 | 276.2 | 0.36 | [0.0, 3.0] | 188.0 | 80.4 | 5.9 |
| Daily bicycle rentals | 18:00 | XGBoost | 0.5 | 267.3 | 265.9 | 0.49 | [0.0, 4.0] | 183.0 | 81.1 | 6.2 |
| Daily bicycle rentals | 18:00 | CatBoost | 0.6 | 304.1 | 302.5 | 0.52 | [0.0, 4.7] | 184.7 | 75.7 | 7.2 |
| Daily maximum temperature (deg C) | 06:00 | Ridge | 0.8 | 2.342 | 2.332 | 0.44 | [0.006, 0.015] | 2.331 | 90.0 | 3.8 |
| Daily maximum temperature (deg C) | 06:00 | Random Forest | 0.2 | 2.501 | 2.500 | 0.03 | [0.000, 0.002] | 2.390 | 90.3 | 4.3 |
| Daily maximum temperature (deg C) | 06:00 | XGBoost | 0.3 | 2.368 | 2.366 | 0.11 | [0.001, 0.005] | 2.402 | 90.2 | 4.4 |
| Daily maximum temperature (deg C) | 06:00 | CatBoost | 0.4 | 2.338 | 2.335 | 0.11 | [0.001, 0.005] | 2.349 | 90.6 | 4.1 |
| Daily maximum temperature (deg C) | 12:00 | Ridge | 1.9 | 1.088 | 1.071 | 1.49 | [0.009, 0.025] | 1.085 | 88.9 | 8.7 |
| Daily maximum temperature (deg C) | 12:00 | Random Forest | 1.8 | 1.197 | 1.175 | 1.79 | [0.010, 0.036] | 1.148 | 89.5 | 11.1 |
| Daily maximum temperature (deg C) | 12:00 | XGBoost | 2.3 | 1.148 | 1.123 | 2.21 | [0.014, 0.040] | 1.132 | 89.4 | 10.2 |
| Daily maximum temperature (deg C) | 12:00 | CatBoost | 2.1 | 1.122 | 1.105 | 1.56 | [0.010, 0.027] | 1.124 | 89.6 | 9.7 |
| Daily maximum temperature (deg C) | 18:00 | Ridge | 43.9 | 0.065 | 0.032 | 50.62 | [0.028, 0.040] | 0.011 | 96.1 | 49.9 |
| Daily maximum temperature (deg C) | 18:00 | Random Forest | 42.9 | 0.059 | 0.042 | 29.59 | [0.011, 0.027] | 0.013 | 93.0 | 46.4 |
| Daily maximum temperature (deg C) | 18:00 | XGBoost | 45.2 | 0.136 | 0.081 | 39.95 | [0.047, 0.063] | 0.013 | 94.4 | 47.7 |
| Daily maximum temperature (deg C) | 18:00 | CatBoost | 49.1 | 0.207 | 0.108 | 47.72 | [0.087, 0.112] | 0.012 | 95.2 | 50.7 |
| Daily minimum temperature (deg C) | 06:00 | Ridge | 10.9 | 0.718 | 0.689 | 4.14 | [0.022, 0.038] | 0.690 | 89.7 | 28.6 |
| Daily minimum temperature (deg C) | 06:00 | Random Forest | 2.2 | 0.703 | 0.698 | 0.75 | [0.001, 0.012] | 0.695 | 90.6 | 32.9 |
| Daily minimum temperature (deg C) | 06:00 | XGBoost | 12.1 | 0.724 | 0.688 | 5.00 | [0.024, 0.055] | 0.672 | 90.1 | 30.7 |
| Daily minimum temperature (deg C) | 06:00 | CatBoost | 17.6 | 0.768 | 0.698 | 9.16 | [0.055, 0.091] | 0.675 | 90.5 | 32.2 |
| Daily minimum temperature (deg C) | 12:00 | Ridge | 23.3 | 0.545 | 0.488 | 10.43 | [0.048, 0.066] | 0.481 | 90.1 | 32.9 |
| Daily minimum temperature (deg C) | 12:00 | Random Forest | 10.5 | 0.467 | 0.451 | 3.35 | [0.006, 0.032] | 0.453 | 90.9 | 36.6 |
| Daily minimum temperature (deg C) | 12:00 | XGBoost | 25.1 | 0.553 | 0.483 | 12.65 | [0.054, 0.093] | 0.440 | 90.8 | 35.9 |
| Daily minimum temperature (deg C) | 12:00 | CatBoost | 27.3 | 0.603 | 0.511 | 15.39 | [0.073, 0.120] | 0.434 | 90.8 | 36.8 |
| Daily minimum temperature (deg C) | 18:00 | Ridge | 30.8 | 0.470 | 0.386 | 17.91 | [0.074, 0.095] | 0.383 | 90.5 | 36.9 |
| Daily minimum temperature (deg C) | 18:00 | Random Forest | 15.1 | 0.381 | 0.361 | 5.23 | [0.011, 0.035] | 0.377 | 91.0 | 38.1 |
| Daily minimum temperature (deg C) | 18:00 | XGBoost | 31.5 | 0.501 | 0.401 | 20.06 | [0.083, 0.124] | 0.363 | 91.3 | 38.8 |
| Daily minimum temperature (deg C) | 18:00 | CatBoost | 35.1 | 0.525 | 0.404 | 23.03 | [0.102, 0.147] | 0.355 | 91.6 | 40.0 |

Column definitions:

- Time: forecast time, as a share of the scheduled auction duration or as the hour of the day.
- Outside: percentage of unconstrained forecasts outside the compiled interval.
- MAE unconstrained / projected / bound-aware: mean absolute error in the unit of the task; projected applies the compiled projection; bound-aware fits the model to the margin from the bound.
- Reduction: percentage decrease in MAE from projection.
- 95% CI of decrease: percentile bootstrap interval (10,000 resamples) of the mean per-case decrease in absolute error, over auctions, or over seven-day blocks of test dates for the daily tasks.
- Coverage projected: empirical coverage of the projected 90% split-conformal interval (never lower than before projection).
- Width reduction: percentage decrease in mean interval width from projection.

## Data

- Online Auctions Dataset, Zenodo, https://doi.org/10.5281/zenodo.11259259, CC BY 4.0. The record identifies the companion website of the book "Modeling Online Auctions" by W. Jank and G. Shmueli (Wiley, 2010) as the original source. The derived table keeps no bidder identifiers.
- Individual Household Electric Power Consumption, UCI Machine Learning Repository, https://doi.org/10.24432/C58K54, CC BY 4.0.
- Bike Sharing, UCI Machine Learning Repository, https://doi.org/10.24432/C5W894, CC BY 4.0.
- Beijing Multi-Site Air Quality, UCI Machine Learning Repository, https://doi.org/10.24432/C5RK5G, CC BY 4.0. The data set takes the meteorological readings of each of its 12 air-quality sites from the nearest weather station, so several sites share one temperature series; the weather table keeps one site per distinct series (8 sites).

Users of the data should cite the original records.

## License

The code, ontologies, and shapes are released under the MIT License (see `LICENSE`).

## Citation

See `CITATION.cff`.
