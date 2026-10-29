"""Knowledge graph construction, SHACL-to-projection compilation, validation, and queries."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyshacl import validate
from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD
from rdflib.namespace import OWL, SH

OA = Namespace("http://example.org/ontoauction#")
HERE = Path(__file__).resolve().parent
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


def load_ontology():
    g = Graph()
    g.parse(ONTOLOGY, format="turtle")
    return g


def load_shapes():
    g = Graph()
    g.parse(SHAPES, format="turtle")
    return g


def target_property(ontology):
    targets = list(ontology.subjects(OA.isPredictionTarget, Literal(True)))
    assert len(targets) == 1, targets
    return targets[0]


def compile_bounds(shapes, ontology):
    """Reads property-pair shapes that relate a public value to the prediction target.

    sh:lessThanOrEquals with the target on the right yields a lower bound; with the target
    as sh:path and another property on the right it yields an upper bound. The time at
    which the bound becomes usable is taken from oa:observableAtFraction in the ontology.
    """
    target = target_property(ontology)
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
        frac = ontology.value(prop, OA.observableAtFraction)
        assert frac is not None, f"no observableAtFraction for {prop}"
        specs.append(BoundSpec(prop, float(frac), kind, name))
    return sorted(specs, key=lambda s: (s.kind, s.observable_at, str(s.prop)))


def bounds_at(specs, table, t):
    """Feasible interval [L, U] for each auction from the relations observable by fraction t."""
    n = len(table)
    lower = np.full(n, -np.inf)
    upper = np.full(n, np.inf)
    for s in specs:
        if s.observable_at > t:
            continue
        v = table[PROPERTY_COLUMNS[s.prop]].to_numpy(dtype=float)
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


def prediction_shapes(specs, t):
    """Shapes graph requiring each prediction at fraction t to respect the compiled bounds."""
    sg = Graph()
    shape = OA[f"PredictionFeasibilityShape_{int(round(t * 100))}"]
    sg.add((shape, RDF.type, SH.NodeShape))
    sg.add((shape, SH.targetClass, OA.PricePrediction))
    for s in specs:
        if s.observable_at > t:
            continue
        ps = BNode()
        sg.add((shape, SH.property, ps))
        if s.kind == "lower":
            sg.add((ps, SH.path, s.prop))
            sg.add((ps, SH.lessThanOrEquals, OA.predictedFinalPrice))
        else:
            sg.add((ps, SH.path, OA.predictedFinalPrice))
            sg.add((ps, SH.lessThanOrEquals, s.prop))
    return sg


def prediction_graph(specs, table, pred, t):
    """Prediction nodes carrying the bound values public at t and the predicted price."""
    g = Graph()
    cols = {s.prop: PROPERTY_COLUMNS[s.prop] for s in specs if s.observable_at <= t}
    for i, (aid, row) in enumerate(table.iterrows()):
        p = OA[f"p_{aid}"]
        g.add((p, RDF.type, OA.PricePrediction))
        for prop, col in cols.items():
            g.add((p, prop, _dec(row[col])))
        g.add((p, OA.predictedFinalPrice, _dec(pred[i])))
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
