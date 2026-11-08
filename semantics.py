"""Knowledge graph construction, SHACL-to-projection compilation, validation, and queries."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyshacl import validate
from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD
from rdflib.namespace import OWL, SH

FC = Namespace("http://example.org/feasibility#")
OA = Namespace("http://example.org/ontoauction#")
HERE = Path(__file__).resolve().parent
CORE = HERE / "ontology" / "core.ttl"
ONTOLOGY = HERE / "ontology" / "auction_ontology.ttl"
SHAPES = HERE / "ontology" / "auction_shapes.ttl"

# Column of the auction table holding the value of each ontology datatype property.
PROPERTY_COLUMNS = {
    OA.openingPrice: "open",
    OA.standingPriceAt25: "standing@0.25",
    OA.standingPriceAt50: "standing@0.5",
    OA.standingPriceAt75: "standing@0.75",
    OA.finalPrice: "price",
}


@dataclass(frozen=True)
class BoundSpec:
    """One feasibility relation compiled from a SHACL property shape."""
    prop: URIRef
    observable_at: float
    kind: str  # "lower": prop <= target; "upper": target <= prop
    name: str


def load_ontology(path=ONTOLOGY):
    """Domain ontology together with the shared vocabulary it imports."""
    g = Graph()
    g.parse(CORE, format="turtle")
    g.parse(path, format="turtle")
    return g


def load_shapes(path=SHAPES):
    g = Graph()
    g.parse(path, format="turtle")
    return g


def target_property(ontology, target=None):
    """The property flagged as prediction target; `target` selects one when several are flagged."""
    targets = sorted(ontology.subjects(FC.isPredictionTarget, Literal(True)))
    if target is not None:
        assert target in targets, (target, targets)
        return target
    assert len(targets) == 1, targets
    return targets[0]


def compile_bounds(shapes, ontology, target=None):
    """Reads property-pair shapes that relate a public value to the prediction target.

    sh:lessThanOrEquals with the target on the right yields a lower bound; with the target
    as sh:path and another property on the right it yields an upper bound. The time at
    which the bound becomes usable is taken from fc:observableAtFraction in the ontology.
    """
    target = target_property(ontology, target)
    specs = []
    for ps in shapes.subjects(SH.lessThanOrEquals, None):
        path = shapes.value(ps, SH.path)
        other = shapes.value(ps, SH.lessThanOrEquals)
        name = str(shapes.value(ps, SH.name) or "")
        if other == target and path != target:
            prop, kind = path, "lower"
        elif path == target and other != target:
            prop, kind = other, "upper"
        else:
            continue
        frac = ontology.value(prop, FC.observableAtFraction)
        assert frac is not None, f"no observableAtFraction for {prop}"
        specs.append(BoundSpec(prop, float(frac), kind, name))
    return sorted(specs, key=lambda s: (s.kind, s.observable_at, str(s.prop)))


def bounds_at(specs, table, t, columns=None):
    """Feasible interval [L, U] for each unit from the relations observable by fraction t.

    columns maps each bounding property to the table column holding its value.
    """
    columns = PROPERTY_COLUMNS if columns is None else columns
    n = len(table)
    lower = np.full(n, -np.inf)
    upper = np.full(n, np.inf)
    for s in specs:
        if s.observable_at > t:
            continue
        v = table[columns[s.prop]].to_numpy(dtype=float)
        if s.kind == "lower":
            lower = np.maximum(lower, v)
        else:
            upper = np.minimum(upper, v)
    return lower, upper


def project(pred, lower, upper):
    """Euclidean projection of each prediction onto its feasible interval."""
    return np.minimum(np.maximum(pred, lower), upper)


def _dec(x):
    return Literal(f"{float(x):.2f}", datatype=XSD.decimal)


def stored(values):
    """Values as written into a graph, rounded to two decimals, for comparisons that mirror the SHACL engine."""
    return np.array([float(f"{float(v):.2f}") for v in values])


def auction_iri(aid):
    return OA[f"auction_{aid}"]


def build_graph(table, labels_by_t, predictions=None):
    """Instantiates auctions, product individuals, snapshots with state classes, and predictions.

    predictions: optional dict {(model, output, t): array aligned with table rows}.
    """
    g = load_ontology()
    for item in sorted(table["item"].unique()):
        g.add((OA[f"product_{item}"], RDF.type, OA[item]))
        g.add((OA[f"product_{item}"], RDF.type, OWL.NamedIndividual))
    for aid, row in table.iterrows():
        a = auction_iri(aid)
        g.add((a, RDF.type, OA.Auction))
        g.add((a, RDF.type, OWL.NamedIndividual))
        g.add((a, OA.hasProduct, OA[f"product_{row['item']}"]))
        g.add((a, OA.scheduledDurationDays, Literal(int(row["duration"]), datatype=XSD.integer)))
        for prop, col in PROPERTY_COLUMNS.items():
            g.add((a, prop, _dec(row[col])))
    for t, labels in labels_by_t.items():
        for aid, lab in labels.iterrows():
            snap = OA[f"snapshot_{aid}_{int(round(t * 100))}"]
            g.add((auction_iri(aid), OA.hasSnapshot, snap))
            g.add((snap, RDF.type, OA.Snapshot))
            g.add((snap, OA.observationFraction, Literal(f"{t:g}", datatype=XSD.decimal)))
            row = table.loc[aid]
            g.add((snap, OA.bidCount, Literal(int(row[f"bids@{t:g}"]), datatype=XSD.integer)))
            g.add((snap, OA.bidderCount, Literal(int(row[f"bidders@{t:g}"]), datatype=XSD.integer)))
            fb = row[f"feedback@{t:g}"]
            if not np.isnan(fb):
                g.add((snap, OA.meanBidderFeedback, _dec(fb)))
            for dim in ("DurationState", "CompetitionState", "ReputationState", "MomentumState", "UpliftState"):
                g.add((snap, RDF.type, OA[lab[dim]]))
    if predictions:
        for (model, output, t), pred in predictions.items():
            for aid, value in zip(table.index, pred):
                p = OA[f"prediction_{aid}_{int(round(t * 100))}_{model}_{output}"]
                g.add((OA[f"snapshot_{aid}_{int(round(t * 100))}"], OA.hasPrediction, p))
                g.add((p, RDF.type, OA.PricePrediction))
                g.add((p, OA.modelName, Literal(model)))
                g.add((p, OA.outputType, Literal(output)))
                g.add((p, OA.predictedFinalPrice, _dec(value)))
    return g


def prediction_shapes(specs, t, prediction_class=OA.PricePrediction, predicted=OA.predictedFinalPrice,
                      namespace=OA):
    """Shapes graph requiring each prediction at fraction t to respect the compiled bounds."""
    sg = Graph()
    shape = namespace[f"PredictionFeasibilityShape_{int(round(t * 100))}"]
    sg.add((shape, RDF.type, SH.NodeShape))
    sg.add((shape, SH.targetClass, prediction_class))
    for s in specs:
        if s.observable_at > t:
            continue
        ps = BNode()
        sg.add((shape, SH.property, ps))
        if s.kind == "lower":
            sg.add((ps, SH.path, s.prop))
            sg.add((ps, SH.lessThanOrEquals, predicted))
        else:
            sg.add((ps, SH.path, predicted))
            sg.add((ps, SH.lessThanOrEquals, s.prop))
    return sg


def prediction_graph(specs, table, pred, t, columns=None, prediction_class=OA.PricePrediction,
                     predicted=OA.predictedFinalPrice, namespace=OA):
    """Prediction nodes carrying the bound values public at t and the predicted value."""
    columns = PROPERTY_COLUMNS if columns is None else columns
    g = Graph()
    cols = {s.prop: columns[s.prop] for s in specs if s.observable_at <= t}
    values = {prop: table[col].to_numpy(dtype=float) for prop, col in cols.items()}
    for i, uid in enumerate(table.index):
        p = namespace[f"p_{uid}"]
        g.add((p, RDF.type, prediction_class))
        for prop, v in values.items():
            g.add((p, prop, _dec(v[i])))
        g.add((p, predicted, _dec(pred[i])))
    return g


def observation_graph(table, unit_class, columns, namespace):
    """One node per observed unit carrying the values of the mapped datatype properties."""
    g = Graph()
    values = {prop: table[col].to_numpy(dtype=float) for prop, col in columns.items()}
    for i, uid in enumerate(table.index):
        u = namespace[f"u_{uid}"]
        g.add((u, RDF.type, unit_class))
        for prop, v in values.items():
            g.add((u, prop, _dec(v[i])))
    return g


def shacl_violations(data_graph, shapes_graph):
    """Runs pySHACL; returns (conforms, validation results, distinct focus nodes in violation)."""
    conforms, report, _ = validate(data_graph, shacl_graph=shapes_graph, inference="none",
                                   abort_on_first=False, allow_warnings=False)
    results = list(report.subjects(RDF.type, SH.ValidationResult))
    nodes = {report.value(r, SH.focusNode) for r in results}
    return bool(conforms), len(results), len(nodes)


QUERIES = {
    "CQ1": ("Auction individuals", """
        SELECT (COUNT(DISTINCT ?a) AS ?n) WHERE { ?a a oa:Auction . }"""),
    "CQ2": ("Auctions whose product is typed under Electronics", """
        SELECT (COUNT(DISTINCT ?a) AS ?n) WHERE {
          ?a oa:hasProduct ?p . ?p a ?c . ?c rdfs:subClassOf* oa:Electronics . }"""),
    "CQ3": ("Auctions in the high-competition state at 50%", """
        SELECT (COUNT(DISTINCT ?a) AS ?n) WHERE {
          ?a oa:hasSnapshot ?s . ?s oa:observationFraction 0.5 ; a oa:HighCompetition . }"""),
    "CQ4": ("Auctions whose standing price at 50% exceeds the closing price", """
        SELECT (COUNT(DISTINCT ?a) AS ?n) WHERE {
          ?a oa:standingPriceAt50 ?b ; oa:finalPrice ?y . FILTER(?b > ?y) }"""),
    "CQ5": ("Random Forest mean predictions at 50% below the standing price (raw)", """
        SELECT (COUNT(DISTINCT ?a) AS ?n) WHERE {
          ?a oa:standingPriceAt50 ?b ; oa:openingPrice ?o ; oa:hasSnapshot ?s .
          ?s oa:hasPrediction ?p . ?p oa:modelName "RandomForest" ; oa:outputType "raw" ;
             oa:predictedFinalPrice ?yh . FILTER(?yh < ?b || ?yh < ?o) }"""),
    "CQ6": ("Random Forest mean predictions at 50% below the standing price (projected)", """
        SELECT (COUNT(DISTINCT ?a) AS ?n) WHERE {
          ?a oa:standingPriceAt50 ?b ; oa:openingPrice ?o ; oa:hasSnapshot ?s .
          ?s oa:hasPrediction ?p . ?p oa:modelName "RandomForest" ; oa:outputType "projected" ;
             oa:predictedFinalPrice ?yh . FILTER(?yh < ?b || ?yh < ?o) }"""),
    "CQ7": ("Raw repair candidates of CQ5 under LuxuryGoods", """
        SELECT (COUNT(DISTINCT ?a) AS ?n) WHERE {
          ?a oa:standingPriceAt50 ?b ; oa:openingPrice ?o ; oa:hasProduct ?pr ; oa:hasSnapshot ?s .
          ?pr a ?c . ?c rdfs:subClassOf* oa:LuxuryGoods .
          ?s oa:hasPrediction ?p . ?p oa:modelName "RandomForest" ; oa:outputType "raw" ;
             oa:predictedFinalPrice ?yh . FILTER(?yh < ?b || ?yh < ?o) }"""),
}


def run_queries(g):
    out = {}
    for key, (label, body) in QUERIES.items():
        res = list(g.query(body, initNs={"oa": OA, "rdfs": RDFS}))
        out[key] = {"label": label, "result": int(res[0][0])}
    return out


def graph_statistics(g):
    ont = load_ontology()
    return {
        "triples": len(g),
        "classes": len(set(ont.subjects(RDF.type, OWL.Class))),
        "object_properties": len(set(ont.subjects(RDF.type, OWL.ObjectProperty))),
        "datatype_properties": len(set(ont.subjects(RDF.type, OWL.DatatypeProperty))),
        "named_individuals": len(set(g.subjects(RDF.type, OWL.NamedIndividual))),
        "snapshots": len(set(g.subjects(RDF.type, OA.Snapshot))),
        "prediction_nodes": len(set(g.subjects(RDF.type, OA.PricePrediction))),
    }
