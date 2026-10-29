"""Draws every figure of the study from the results files and the ontology."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from rdflib import RDF, RDFS  # noqa: E402
from rdflib.namespace import OWL  # noqa: E402

import semantics  # noqa: E402

SINGLE, DOUBLE = 3.54, 7.48  # inches, Elsevier single and double column
MODELS = ("Ridge", "RandomForest", "XGBoost", "CatBoost")
MODEL_LABEL = {"Ridge": "Ridge", "RandomForest": "Random Forest", "XGBoost": "XGBoost", "CatBoost": "CatBoost"}
MODEL_COLOR = {"Ridge": "#CC79A7", "RandomForest": "#0072B2", "XGBoost": "#D55E00", "CatBoost": "#009E73"}
MODEL_MARKER = {"Ridge": "s", "RandomForest": "o", "XGBoost": "^", "CatBoost": "D"}
OUTPUT_COLOR = {"raw": "#8C8C8C", "projected": "#E69F00", "bound_aware": "#56B4E9"}
OUTPUT_LABEL = {"raw": "Unconstrained", "projected": "Projected", "bound_aware": "Bound-aware"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "lines.linewidth": 0.9, "patch.linewidth": 0.5, "pdf.fonttype": 42, "savefig.dpi": 300,
})
META = {"Creator": "Matplotlib", "Producer": None, "Author": None, "Title": None, "Subject": None,
        "Keywords": None, "CreationDate": None}


def _save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, metadata=META)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300,
                metadata={"Software": None})
    plt.close(fig)


PAD = 0.01  # box padding in data units; arrows attach to the padded outline


def _box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD},rounding_size=0.015",
                                fc=color, ec="#333333", lw=0.5))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=7.5)
    ax.text(x + w / 2, y + h / 2 - 0.03, body, ha="center", va="center", fontsize=6.3, linespacing=1.25)


def _arrow(ax, a, b, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8, lw=0.6, color="#333333",
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=3))


def architecture(path):
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h = 0.205, 0.33
    top, bot = 0.6, 0.08
    xs = [0.01, 0.262, 0.514, 0.766]
    _box(ax, xs[0], top, w, h, "Bid records", "10,681 timestamped\nbid records, 628 auctions", "#EAF2FA")
    _box(ax, xs[1], top, w, h, "OWL ontology", "Auction, Product, Snapshot,\nstate classes, observability\nannotations", "#EEF6EA")
    _box(ax, xs[2], top, w, h, "SHACL shapes", "sh:lessThanOrEquals\nrelations to the\nclosing price", "#F3EEF8")
    _box(ax, xs[3], top, w, h, "Validation", "pySHACL reports on\nobserved data and on\nprediction graphs", "#F4F4F4")
    _box(ax, xs[0], bot, w, h, "Public snapshots", "Standing price, bid and\nbidder counts at $t/2$ and\n$t$, for $t$ = 25%, 50%, 75%", "#FDF2E6")
    _box(ax, xs[1], bot, w, h, "Regressors", "Ridge, Random Forest,\nXGBoost, CatBoost", "#FBEAEA")
    _box(ax, xs[2], bot, w, h, "Compiled projection", "Interval $[L_i(t), U_i(t)]$ from\nshapes observable by $t$;\n$\\hat{y}_i\\mapsto\\min\\{\\max\\{\\hat{y}_i, L_i(t)\\}, U_i(t)\\}$", "#E6F0FA")
    _box(ax, xs[3], bot, w, h, "Analytics", "SPARQL competency\nqueries, concept-level\nSHAP, intervals", "#F4F4F4")
    mid_t, mid_b = top + h / 2, bot + h / 2
    right = lambda i: xs[i] + w + PAD
    left = lambda i: xs[i] - PAD
    top_edge, bot_edge = top - PAD, bot + h + PAD  # lower edge of upper row, upper edge of lower row
    for i in range(3):
        _arrow(ax, (right(i), mid_t), (left(i + 1), mid_t))  # data flow along the upper row
        _arrow(ax, (right(i), mid_b), (left(i + 1), mid_b))  # predictive path along the lower row
    _arrow(ax, (xs[0] + w / 2, top_edge), (xs[0] + w / 2, bot_edge))  # bid records to snapshots
    _arrow(ax, (xs[2] + w / 2, top_edge), (xs[2] + w / 2, bot_edge))  # shapes compiled into the projection
    _arrow(ax, (xs[1] + 0.8 * w, top_edge), (xs[2] + 0.2 * w, bot_edge))  # observability annotations
    _arrow(ax, (xs[2] + 0.85 * w, bot_edge), (xs[3] + 0.3 * w, top_edge), ls="--")  # validation of predictions
    _save(fig, path)


def ontology_diagram(path):
    g = semantics.load_ontology()
    oa = semantics.OA
    label = lambda c: str(c).split("#")[1]
    children = {}
    for c in g.subjects(RDF.type, OWL.Class):
        for p in g.objects(c, RDFS.subClassOf):
            children.setdefault(p, []).append(c)
    for v in children.values():
        v.sort(key=label)

    fig, ax = plt.subplots(figsize=(DOUBLE, 3.1))
    ax.axis("off")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    bw, bh = 0.118, 0.07

    def node(x, y, text, fc, fs=6.2, w=bw):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - bh / 2), w, bh, boxstyle="round,pad=0.004,rounding_size=0.01",
                                    fc=fc, ec="#333333", lw=0.5))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def edge(a, b, text=None, dashed=False, off=0.0):
        _arrow(ax, a, b, style="-|>", ls="--" if dashed else "-")
        if text:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.025 + off, text, ha="center", va="bottom",
                    fontsize=6, style="italic")

    # Core relations between individuals
    yc = 0.9
    pos = {"Auction": (0.19, yc), "Snapshot": (0.5, yc), "PricePrediction": (0.8, yc)}
    node(*pos["Auction"], "Auction", "#FDF2E6")
    node(*pos["Snapshot"], "Snapshot", "#FDF2E6")
    node(*pos["PricePrediction"], "PricePrediction", "#FDF2E6", w=0.14)
    edge((pos["Auction"][0] + bw / 2, yc), (pos["Snapshot"][0] - bw / 2, yc), "hasSnapshot")
    edge((pos["Snapshot"][0] + bw / 2, yc), (pos["PricePrediction"][0] - 0.07, yc), "hasPrediction")

    # Product hierarchy
    px = 0.19
    node(px, 0.66, "Product", "#EAF2FA")
    _arrow(ax, (px, yc - bh / 2), (px, 0.66 + bh / 2))
    ax.text(px + 0.008, (yc + 0.66) / 2, "hasProduct", ha="left", va="center", fontsize=6, style="italic")
    xs = {oa.Electronics: 0.12, oa.LuxuryGoods: 0.27}
    for c in children[oa.Product]:
        node(xs[c], 0.45, label(c), "#EAF2FA", w=0.11)
        edge((px, 0.66 - bh / 2), (xs[c], 0.45 + bh / 2), dashed=True)
    leaves = [(oa.Electronics, oa.PalmPilotM515), (oa.Electronics, oa.XboxGameConsole),
              (oa.LuxuryGoods, oa.CartierWristwatch)]
    for (p, c), x in zip(leaves, [0.068, 0.194, 0.32]):
        node(x, 0.22, label(c), "#EAF2FA", fs=5.5, w=0.112)
        edge((xs[p], 0.45 - bh / 2), (x, 0.22 + bh / 2), dashed=True)

    # State hierarchy
    sx = 0.712
    node(sx, 0.66, "AuctionState", "#EEF6EA", w=0.13)
    _arrow(ax, (pos["Snapshot"][0], yc - bh / 2), (sx - 0.04, 0.66 + bh / 2))
    ax.text((pos["Snapshot"][0] + sx - 0.04) / 2 + 0.012, (yc + 0.66) / 2 + 0.01, "rdf:type", ha="left",
            va="center", fontsize=6, style="italic")
    order = {"DurationState": ["ShortAuction", "MediumAuction", "LongAuction"],
             "CompetitionState": ["LowCompetition", "ModerateCompetition", "HighCompetition"],
             "ReputationState": ["UnratedBidders", "EmergingReputation", "EstablishedReputation", "HighReputation"],
             "MomentumState": ["DormantAuction", "ActivatedAuction", "AcceleratingAuction", "SteadyAuction"],
             "UpliftState": ["LowUplift", "ModerateUplift", "HighUplift"]}
    for name, x in zip(order, [0.44, 0.576, 0.712, 0.848, 0.984]):
        c = oa[name]
        assert sorted(label(k) for k in children[c]) == sorted(order[name]), name
        node(x, 0.45, name, "#EEF6EA", fs=5.5, w=0.118)
        edge((sx, 0.66 - bh / 2), (x, 0.45 + bh / 2), dashed=True)
        for j, k in enumerate(order[name]):
            ax.text(x, 0.36 - 0.052 * j, k, ha="center", va="top", fontsize=5.4)
        ax.plot([x - bw / 2 + 0.005, x + bw / 2 - 0.005], [0.375, 0.375], lw=0.4, color="#666666")
    ax.plot([], [], ls="--", color="#333333", lw=0.6, label="rdfs:subClassOf")
    ax.plot([], [], ls="-", color="#333333", lw=0.6, label="object property or typing")
    ax.legend(loc="lower left", frameon=False, bbox_to_anchor=(0.0, -0.02), ncol=2)
    _save(fig, path)


def mae_bars(res, path, t="0.5"):
    rv = res["repeat_sd"][t]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    width = 0.26
    for k, out in enumerate(("raw", "projected", "bound_aware")):
        xs = [i + (k - 1) * width for i in range(len(MODELS))]
        vals = [rv[m][out]["mean"] for m in MODELS]
        sds = [rv[m][out]["sd"] for m in MODELS]
        ax.bar(xs, vals, width, yerr=sds, color=OUTPUT_COLOR[out], label=OUTPUT_LABEL[out],
               error_kw={"elinewidth": 0.5, "capsize": 1.5, "capthick": 0.5}, edgecolor="#333333", linewidth=0.3)
    ax.set_xticks(range(len(MODELS)), [MODEL_LABEL[m] for m in MODELS])
    ax.set_ylabel("Mean absolute error (US$)")
    ax.set_ylim(0, max(rv[m]["raw"]["mean"] for m in MODELS) * 1.18)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def reduction_by_time(res, path):
    fig, ax = plt.subplots(figsize=(SINGLE, 2.1))
    ts = sorted(res["paired"], key=float)
    offs = {m: (i - 1.5) * 0.012 for i, m in enumerate(MODELS)}
    for m in MODELS:
        x = [float(t) * 100 + offs[m] * 100 for t in ts]
        p = [res["paired"][t][m] for t in ts]
        yv = [q["reduction_pct"] for q in p]
        lo = [100 * q["ci"][0] / q["mae_raw"] for q in p]
        hi = [100 * q["ci"][1] / q["mae_raw"] for q in p]
        ax.errorbar(x, yv, yerr=[[a - b for a, b in zip(yv, lo)], [b - a for a, b in zip(yv, hi)]],
                    color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ms=3, capsize=1.5, elinewidth=0.5,
                    label=MODEL_LABEL[m])
    ax.set_xticks([float(t) * 100 for t in ts], [f"{float(t) * 100:.0f}%" for t in ts])
    ax.set_xlabel("Prediction point (share of scheduled duration)")
    ax.set_ylabel("Reduction in MAE (%)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.5,
              columnspacing=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def interval_panels(res, path):
    ts = sorted(res["intervals"], key=float)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.72, 1.95))
    for m in MODELS:
        x = [float(t) * 100 for t in ts]
        cov_r = [100 * res["intervals"][t][m]["raw"]["coverage"] for t in ts]
        cov_p = [100 * res["intervals"][t][m]["projected"]["coverage"] for t in ts]
        red = [100 * (1 - res["intervals"][t][m]["projected"]["width"] / res["intervals"][t][m]["raw"]["width"])
               for t in ts]
        axes[0].plot(x, cov_r, color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ms=2.8, ls=":", lw=0.7)
        axes[0].plot(x, cov_p, color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ms=2.8, label=MODEL_LABEL[m])
        axes[1].plot(x, red, color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ms=2.8, label=MODEL_LABEL[m])
    axes[0].axhline(90, color="#333333", lw=0.5, ls="--")
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[1].set_ylabel("Reduction in mean width (%)")
    axes[1].set_ylim(bottom=0)
    for a, tag in zip(axes, "ab"):
        a.set_xticks([float(t) * 100 for t in ts], [f"{float(t) * 100:.0f}%" for t in ts])
        a.set_xlabel("Prediction point (share of scheduled duration)")
        a.spines[["top", "right"]].set_visible(False)
        a.text(-0.2, 1.02, f"({tag})", transform=a.transAxes, fontsize=7)
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout(w_pad=1.5)
    _save(fig, path)


def shap_bars(res, path):
    s = res["shap"]["group_share_pct"]
    order = sorted(s, key=s.get)
    fig, ax = plt.subplots(figsize=(SINGLE, 1.55))
    ax.barh(order, [s[k] for k in order], color="#0072B2", height=0.6, edgecolor="#333333", linewidth=0.3)
    for i, k in enumerate(order):
        ax.text(s[k] + 0.8, i, f"{s[k]:.1f}", va="center", fontsize=6.3)
    ax.set_xlabel("Share of total mean |SHAP| (%)")
    ax.set_xlim(0, max(s.values()) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def make_all(results_dir, fig_dir):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(results_dir) / "results.json", encoding="utf-8") as fh:
        res = json.load(fh)
    architecture(fig_dir / "fig1_architecture")
    ontology_diagram(fig_dir / "fig2_ontology")
    mae_bars(res, fig_dir / "fig3_mae")
    reduction_by_time(res, fig_dir / "fig4_reduction_by_time")
    interval_panels(res, fig_dir / "fig5_intervals")
    shap_bars(res, fig_dir / "fig6_shap")
