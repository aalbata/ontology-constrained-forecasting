"""Draws every figure of the study from the results files and the ontology."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
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
# Vertical gradients (top, bottom) and borders of diagram shapes, one role per color.
PALETTE = {"blue": ("#DAE8FC", "#7EA6E0", "#6C8EBF"), "green": ("#D5E8D4", "#97D077", "#82B366"),
           "yellow": ("#FFF2CC", "#FFD966", "#D6B656"), "red": ("#F8CECC", "#EA6B66", "#B85450")}


def _gradient(ax, patch, x0, y0, x1, y1, role, strips=48):
    """Fills a patch with a vertical gradient made of thin vector strips clipped to its outline."""
    top, bottom, edge = PALETTE[role]
    a, b = to_rgb(top), to_rgb(bottom)
    patch.set_facecolor("none")
    patch.set_edgecolor(edge)
    ax.add_patch(patch)
    dy = (y1 - y0) / strips
    for i in range(strips):
        f = (i + 0.5) / strips  # 0 at the bottom, 1 at the top
        color = tuple(bb + (aa - bb) * f for aa, bb in zip(a, b))
        strip = Rectangle((x0, y0 + i * dy), x1 - x0, dy * 1.02, fc=color, ec="none", lw=0, zorder=patch.get_zorder() - 0.1)
        ax.add_patch(strip)
        strip.set_clip_path(patch)


def _box(ax, x, y, w, h, title, body, role):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD},rounding_size=0.015", lw=0.6, zorder=2)
    _gradient(ax, patch, x - PAD, y - PAD, x + w + PAD, y + h + PAD, role)
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=7.5, zorder=3)
    ax.text(x + w / 2, y + h / 2 - 0.03, body, ha="center", va="center", fontsize=6.3, linespacing=1.25, zorder=3)


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
    _box(ax, xs[0], top, w, h, "Period records", "Bids of online auctions;\nhourly energy use, rentals,\nand temperatures", "blue")
    _box(ax, xs[1], top, w, h, "Domain ontology", "Target property and the\ntime at which each value\nbecomes public", "green")
    _box(ax, xs[2], top, w, h, "SHACL shapes", "sh:lessThanOrEquals\nbetween public values\nand the target", "green")
    _box(ax, xs[3], top, w, h, "Validation", "pySHACL reports on\nobserved data and on\nforecast graphs", "green")
    _box(ax, xs[0], bot, w, h, "Public state at $t$", "Values known at 25%, 50%,\nand 75% of the period", "blue")
    _box(ax, xs[1], bot, w, h, "Any regressor", "Ridge, Random Forest,\nXGBoost, CatBoost, and\nconformal intervals", "blue")
    _box(ax, xs[2], bot, w, h, "Compiled projection", "Interval $[L_i(t), U_i(t)]$ from\nshapes observable by $t$;\n$\\hat{y}_i\\mapsto\\min\\{\\max\\{\\hat{y}_i, L_i(t)\\}, U_i(t)\\}$", "yellow")
    _box(ax, xs[3], bot, w, h, "Corrected outputs", "Point forecasts and\ninterval end points\ninside the bounds", "red")
    mid_t, mid_b = top + h / 2, bot + h / 2
    right = lambda i: xs[i] + w + PAD
    left = lambda i: xs[i] - PAD
    top_edge, bot_edge = top - PAD, bot + h + PAD  # lower edge of upper row, upper edge of lower row
    for i in range(3):
        _arrow(ax, (right(i), mid_t), (left(i + 1), mid_t))  # data flow along the upper row
        _arrow(ax, (right(i), mid_b), (left(i + 1), mid_b))  # predictive path along the lower row
    _arrow(ax, (xs[0] + w / 2, top_edge), (xs[0] + w / 2, bot_edge))  # records to the public state
    _arrow(ax, (xs[2] + w / 2, top_edge), (xs[2] + w / 2, bot_edge))  # shapes compiled into the projection
    _arrow(ax, (xs[1] + 0.8 * w, top_edge), (xs[2] + 0.2 * w, bot_edge))  # observability annotations
    _arrow(ax, (xs[2] + 0.85 * w, bot_edge), (xs[3] + 0.3 * w, top_edge), ls="--")  # validation of forecasts
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

    def node(x, y, text, role, fs=6.2, w=bw):
        patch = FancyBboxPatch((x - w / 2, y - bh / 2), w, bh, boxstyle="round,pad=0.004,rounding_size=0.01",
                               lw=0.6, zorder=2)
        _gradient(ax, patch, x - w / 2 - 0.004, y - bh / 2 - 0.004, x + w / 2 + 0.004, y + bh / 2 + 0.004, role)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3)

    def edge(a, b, text=None, dashed=False, off=0.0):
        _arrow(ax, a, b, style="-|>", ls="--" if dashed else "-")
        if text:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.025 + off, text, ha="center", va="bottom",
                    fontsize=6, style="italic")

    # Core relations between individuals
    yc = 0.9
    pos = {"Auction": (0.19, yc), "Snapshot": (0.5, yc), "PricePrediction": (0.8, yc)}
    node(*pos["Auction"], "Auction", "yellow")
    node(*pos["Snapshot"], "Snapshot", "yellow")
    node(*pos["PricePrediction"], "PricePrediction", "yellow", w=0.14)
    edge((pos["Auction"][0] + bw / 2, yc), (pos["Snapshot"][0] - bw / 2, yc), "hasSnapshot")
    edge((pos["Snapshot"][0] + bw / 2, yc), (pos["PricePrediction"][0] - 0.07, yc), "hasPrediction")

    # Product hierarchy
    px = 0.19
    node(px, 0.66, "Product", "blue")
    _arrow(ax, (px, yc - bh / 2), (px, 0.66 + bh / 2))
    ax.text(px + 0.008, (yc + 0.66) / 2, "hasProduct", ha="left", va="center", fontsize=6, style="italic")
    xs = {oa.Electronics: 0.12, oa.LuxuryGoods: 0.27}
    for c in children[oa.Product]:
        node(xs[c], 0.45, label(c), "blue", w=0.11)
        edge((px, 0.66 - bh / 2), (xs[c], 0.45 + bh / 2), dashed=True)
    leaves = [(oa.Electronics, oa.PalmPilotM515), (oa.Electronics, oa.XboxGameConsole),
              (oa.LuxuryGoods, oa.CartierWristwatch)]
    for (p, c), x in zip(leaves, [0.068, 0.194, 0.32]):
        node(x, 0.22, label(c), "blue", fs=5.5, w=0.112)
        edge((xs[p], 0.45 - bh / 2), (x, 0.22 + bh / 2), dashed=True)

    # State hierarchy
    sx = 0.712
    node(sx, 0.66, "AuctionState", "green", w=0.13)
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
        node(x, 0.45, name, "green", fs=5.5, w=0.118)
        edge((sx, 0.66 - bh / 2), (x, 0.45 + bh / 2), dashed=True)
        for j, k in enumerate(order[name]):
            ax.text(x, 0.36 - 0.052 * j, k, ha="center", va="top", fontsize=5.4)
        ax.plot([x - bw / 2 + 0.005, x + bw / 2 - 0.005], [0.375, 0.375], lw=0.4, color="#666666")
    ax.plot([], [], ls="--", color="#333333", lw=0.6, label="rdfs:subClassOf")
    ax.plot([], [], ls="-", color="#333333", lw=0.6, label="object property or typing")
    ax.legend(loc="lower left", frameon=False, bbox_to_anchor=(0.0, -0.02), ncol=2)
    _save(fig, path)


TASKS = ("auction", "electricity", "bike", "weather_max", "weather_min")
TASK_LABEL = {"auction": "Auction closing price", "electricity": "Daily household energy",
              "bike": "Daily bicycle rentals", "weather_max": "Daily maximum temperature",
              "weather_min": "Daily minimum temperature"}
TS = ("0.25", "0.5", "0.75")
POINT_LABEL = {"auction": ["25%", "50%", "75%"], "default": ["06:00", "12:00", "18:00"]}


def task_cells(res, per):
    """Per task, prediction point and learner: reduction (%), its 95% interval (%), and violation rate (%)."""
    cells = {}
    for t in TS:
        for m in MODELS:
            q = res["paired"][t][m]
            cells[("auction", t, m)] = (q["reduction_pct"], 100 * q["ci"][0] / q["mae_raw"],
                                        100 * q["ci"][1] / q["mae_raw"],
                                        100 * res["fold_means"][t][m]["raw"]["violation_rate"])
    for name, task in per["tasks"].items():
        for t in TS:
            for m in MODELS:
                q = task["point"][t][m]
                cells[(name, t, m)] = (q["reduction_pct"], 100 * q["ci"][0] / q["mae_raw"],
                                       100 * q["ci"][1] / q["mae_raw"], 100 * q["violation_rate"])
    return cells


def _point_axis(ax, task):
    ax.set_xticks(range(3), POINT_LABEL.get(task, POINT_LABEL["default"]))
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))  # a panel without any change keeps a readable 0 to 1 axis
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(TASK_LABEL[task], fontsize=6.6)


def _panel_layout(fig):
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.13, top=0.76, wspace=0.42)


def reduction_panels(res, per, path):
    cells = task_cells(res, per)
    fig, axes = plt.subplots(1, len(TASKS), figsize=(DOUBLE, 1.95))
    offs = {m: (i - 1.5) * 0.12 for i, m in enumerate(MODELS)}
    for ax, task in zip(axes, TASKS):
        for m in MODELS:
            v = [cells[(task, t, m)] for t in TS]
            x = [i + offs[m] for i in range(3)]
            ax.errorbar(x, [c[0] for c in v], yerr=[[c[0] - c[1] for c in v], [c[2] - c[0] for c in v]],
                        color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ms=2.6, capsize=1.2, elinewidth=0.5,
                        lw=0.7, label=MODEL_LABEL[m])
        ax.set_ylim(bottom=0)
        _point_axis(ax, task)
    axes[0].set_ylabel("Reduction in MAE (%)")
    axes[0].legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.13), handlelength=1.5,
                   columnspacing=1.0)
    _panel_layout(fig)
    _save(fig, path)


def violation_scatter(res, per, path):
    cells = task_cells(res, per)
    colors = {"auction": "#333333", "electricity": "#0072B2", "bike": "#009E73", "weather_max": "#D55E00",
              "weather_min": "#CC79A7"}
    fig, ax = plt.subplots(figsize=(SINGLE * 1.15, 2.3))
    for task in TASKS:
        for m in MODELS:
            pts = [cells[(task, t, m)] for t in TS]
            ax.scatter([c[3] for c in pts], [c[0] for c in pts], s=9, marker=MODEL_MARKER[m], facecolor="none",
                       edgecolor=colors[task], linewidth=0.6)
    for task in TASKS:
        ax.scatter([], [], s=9, marker="o", facecolor="none", edgecolor=colors[task], linewidth=0.6,
                   label=TASK_LABEL[task])
    for m in MODELS:
        ax.scatter([], [], s=9, marker=MODEL_MARKER[m], facecolor="none", edgecolor="#666666", linewidth=0.6,
                   label=MODEL_LABEL[m])
    ax.set_xlabel("Unconstrained forecasts outside the bounds (%)")
    ax.set_ylabel("Reduction in MAE (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=5.6, loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def interval_cells(res, per):
    """Per task, point and learner: width reduction (%) from projecting split-conformal and quantile intervals."""
    def red(iv, m):
        return 100 * (1 - iv[m]["projected"]["width"] / iv[m]["raw"]["width"])
    out = {}
    for t in TS:
        for m in (*MODELS, "CQR"):
            out[("auction", t, m)] = red(res["intervals"][t], m)
    for name, task in per["tasks"].items():
        for t in TS:
            for m in (*MODELS, "CQR"):
                out[(name, t, m)] = red(task["intervals"][t], m)
    return out


def width_panels(res, per, path):
    cells = interval_cells(res, per)
    fig, axes = plt.subplots(1, len(TASKS), figsize=(DOUBLE, 1.95))
    for ax, task in zip(axes, TASKS):
        for m in MODELS:
            ax.plot(range(3), [cells[(task, t, m)] for t in TS], color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
                    ms=2.6, lw=0.7, label=MODEL_LABEL[m])
        ax.plot(range(3), [cells[(task, t, "CQR")] for t in TS], color="#333333", ls="--", marker="x", ms=2.6,
                lw=0.7, label="Conformalized quantile regression")
        ax.set_ylim(bottom=0)
        _point_axis(ax, task)
    axes[0].set_ylabel("Reduction in mean width (%)")
    axes[0].legend(frameon=False, ncol=5, loc="lower left", bbox_to_anchor=(0.0, 1.13), handlelength=1.5,
                   columnspacing=1.0)
    _panel_layout(fig)
    _save(fig, path)


def make_all(results_dir, fig_dir):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(results_dir) / "results.json", encoding="utf-8") as fh:
        res = json.load(fh)
    with open(Path(results_dir) / "periods.json", encoding="utf-8") as fh:
        per = json.load(fh)
    architecture(fig_dir / "fig1_architecture")
    ontology_diagram(fig_dir / "fig2_ontology")
    reduction_panels(res, per, fig_dir / "fig3_reduction")
    violation_scatter(res, per, fig_dir / "fig4_violation_gain")
    width_panels(res, per, fig_dir / "fig5_interval_width")
