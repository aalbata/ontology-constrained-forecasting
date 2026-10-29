# Ontology-constrained forecasting of online auction prices

Code, ontology, shapes, and data for the article "Compiling Semantic Web Constraints into
Feasibility Guarantees for Machine-Learning Forecasts" by Snwr Mohammad, Lucine Gharibian,
Samia Kouki, and Ali Al Bataineh.

Repository: https://github.com/aalbata/ontology-constrained-forecasting

The pipeline compiles SHACL property-pair shapes and OWL observability annotations into
time-indexed feasible intervals, projects the predictions of four regressors onto those
intervals, validates observed data and predictions with pySHACL, answers SPARQL competency
queries, builds split-conformal intervals, and groups SHAP attributions by ontology concept.

## Contents

| Path | Description |
|---|---|
| `reproduce.py` | Entry point. Runs every experiment and draws every figure. |
| `snapshots.py` | Public-information snapshots, features, and ontology state classes. |
| `semantics.py` | Knowledge graph, SHACL-to-projection compiler, pySHACL validation, SPARQL queries. |
| `experiment.py` | Repeated cross-validation, designs, metrics, paired bootstrap, conformal intervals, SHAP. |
| `figures.py` | Figures 1 to 6 of the article from the results files and the ontology. |
| `ontology/auction_ontology.ttl` | OWL ontology with `observableAtFraction` annotations. |
| `ontology/auction_shapes.ttl` | SHACL shapes relating public prices to the closing price. |
| `data/auction_snapshots.csv` | Auction-level snapshot table derived from the bid records (no bidder identifiers). |
| `results/` | Reference outputs: `results.json` (all reported numbers), per-fold metrics, interval metrics, out-of-fold predictions, and the knowledge graph in Turtle. |
| `figures/` | Reference figures in PDF and PNG. |

## Requirements

Python 3.12 and the pinned packages in `requirements.txt`. The reference outputs were produced on
Windows 11 with an Intel Core Ultra 7 155U processor (14 logical cores), 16 GB of memory, and
Python 3.12.10; the complete run took 22 minutes with 12 worker processes.

```
python -m pip install -r requirements.txt
```

## Reproduction

Fast path, from the shipped snapshot table:

```
python reproduce.py
```

Full path, downloading the bid-level archive from Zenodo, rebuilding the snapshot table, and
checking that it is identical to the shipped table before running the experiments:

```
python reproduce.py --from-raw
```

A local copy of the archive can be supplied with `--zip path/to/Online_Auctions_Dataset.zip`;
its SHA-256 checksum is verified. Outputs are written to `results/` and `figures/`.

## Results

Paired per-auction results over five repetitions of five-fold cross-validation (628 auctions). MAE values
here average each auction's five out-of-fold errors first, as in Table 5 of the article; the fold-mean values
of Table 4 can differ in the second decimal because folds differ slightly in size.

| Point | Model | MAE unconstrained ($) | MAE projected ($) | Reduction (%) | 95% CI of decrease ($) | Repaired | MAE bound-aware ($) | Violation (%) | Coverage projected (%) | Width reduction (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| 25% | Ridge | 135.17 | 109.13 | 19.26 | [12.24, 44.04] | 31 | 109.50 | 4.49 | 90.57 | 12.48 |
| 25% | Random Forest | 86.96 | 82.00 | 5.70 | [0.02, 14.58] | 6 | 90.49 | 0.64 | 89.14 | 8.28 |
| 25% | XGBoost | 88.74 | 83.24 | 6.20 | [0.25, 15.76] | 31 | 91.09 | 3.57 | 89.11 | 8.06 |
| 25% | CatBoost | 88.78 | 82.86 | 6.67 | [0.49, 16.21] | 36 | 89.89 | 4.21 | 89.08 | 9.00 |
| 50% | Ridge | 119.92 | 95.02 | 20.76 | [12.15, 41.62] | 36 | 95.26 | 5.03 | 90.89 | 15.39 |
| 50% | Random Forest | 78.19 | 72.97 | 6.68 | [0.10, 14.82] | 13 | 75.42 | 0.99 | 90.13 | 10.29 |
| 50% | XGBoost | 80.21 | 73.98 | 7.77 | [0.46, 16.77] | 31 | 78.05 | 3.19 | 89.87 | 10.49 |
| 50% | CatBoost | 81.06 | 74.15 | 8.53 | [1.04, 17.20] | 35 | 77.44 | 3.89 | 90.10 | 11.08 |
| 75% | Ridge | 99.37 | 76.76 | 22.75 | [11.73, 36.67] | 41 | 76.71 | 5.73 | 90.13 | 20.11 |
| 75% | Random Forest | 62.45 | 57.86 | 7.35 | [0.20, 13.12] | 16 | 60.13 | 1.63 | 90.16 | 13.09 |
| 75% | XGBoost | 64.64 | 58.91 | 8.88 | [0.71, 15.16] | 39 | 60.14 | 4.24 | 90.89 | 13.48 |
| 75% | CatBoost | 66.08 | 60.06 | 9.11 | [0.91, 15.43] | 45 | 61.13 | 4.75 | 89.94 | 14.32 |

Column definitions:

- Point: prediction time as a share of the scheduled auction duration.
- MAE unconstrained / projected / bound-aware: mean absolute error in US dollars of the closing-price forecast, with each auction's error averaged over its five out-of-fold predictions; projected applies the compiled feasibility projection; bound-aware fits the model to the log margin above the lower bound.
- Reduction: percentage decrease in MAE from projection.
- 95% CI of decrease: percentile bootstrap interval (10,000 resamples of auctions) of the mean per-auction decrease in absolute error.
- Repaired: auctions with at least one unconstrained prediction below the compiled lower bound.
- Violation: mean over the 25 outer folds of the percentage of unconstrained predictions below the bound.
- Coverage projected: empirical coverage of the projected 90% split-conformal interval (identical to the coverage before projection).
- Width reduction: percentage decrease in mean interval width from projection.

The knowledge graph written to `results/auction_kg.ttl` contains 32,326 triples,
32 classes, and 631 named individuals. pySHACL reports no violation on the observed
data and none on any projected prediction.

## Data

The bid-level data are the Online Auctions Dataset on Zenodo (https://doi.org/10.5281/zenodo.11259259),
which the uploader released under the Creative Commons Attribution 4.0 license and whose description
identifies the companion website of the book "Modeling Online Auctions" by W. Jank and G. Shmueli
(Wiley, 2010) as the original source. The derived table `data/auction_snapshots.csv` keeps the auction
identifier, product and product superclass, scheduled duration, opening and closing prices, the number of
bid records and the largest recorded bid (used only for the data audit), and, for each snapshot fraction,
the bid and bidder counts, standing price, last-bid time, and mean feedback rating; bidder identifiers are
not included. Users should cite
the Zenodo record and the book when using the data.

## License

The code, ontology, and shapes are released under the MIT License (see `LICENSE`).
