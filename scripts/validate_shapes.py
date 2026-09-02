"""
Semantic validation of the graph with SHACL (pySHACL).
Fetches a SAMPLE of observations from GraphDB and checks their conformance
against the shapes defined in shapes.ttl.
Read-only: it never modifies the repository — it runs completely on its own.
"""
import os
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, XSD
from pyshacl import validate

SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT", "http://localhost:7200/repositories/thesis")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
EX = Namespace("http://example.org/thesis/")

# Observations to sample per metric (a representative sample, not all ~100M rows).
SAMPLE_PER_PROPERTY = 500

def fetch_sample_graph():
    """Fetch a representative sample of observations into a local rdflib graph."""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setReturnFormat(JSON)
    g = Graph()

    # Metrics we want to validate
    properties = [
        "Fatigue", "Mood", "Readiness", "SleepQuality", "Soreness", "Stress",
        "SleepScore", "SleepEfficiency",
        "DeepSleepMinutes", "RemSleepMinutes", "LightSleepMinutes",
        "WakeStageMinutes", "MinutesAsleep", "MinutesAwake",
        "HeartRate", "RestingHeartRate",
        "StepCount", "Calories", "WaterGlasses",
        "VeryActiveMinutes", "SedentaryMinutes", "BodyWeight"
    ]

    for prop in properties:
        query = f"""
        PREFIX sosa: <http://www.w3.org/ns/sosa/>
        PREFIX ex: <http://example.org/thesis/>
        SELECT ?obs ?user ?value ?time
        WHERE {{
            ?obs a sosa:Observation ;
                 sosa:hasFeatureOfInterest ?user ;
                 sosa:observedProperty ex:{prop} ;
                 sosa:hasSimpleResult ?value ;
                 sosa:resultTime ?time .
        }}
        LIMIT {SAMPLE_PER_PROPERTY}
        """
        sparql.setQuery(query)
        try:
            results = sparql.query().convert()
        except Exception as e:
            print(f"[WARNING] Failed to fetch {prop}: {e}")
            continue

        count = 0
        for r in results["results"]["bindings"]:
            obs = URIRef(r["obs"]["value"])
            g.add((obs, RDF.type, SOSA.Observation))
            g.add((obs, SOSA.hasFeatureOfInterest, URIRef(r["user"]["value"])))
            g.add((obs, SOSA.observedProperty, EX[prop]))
            g.add((obs, SOSA.resultTime, Literal(r["time"]["value"])))

            # Rebuild the value with its proper numeric datatype, otherwise SHACL's
            # min/max range checks (sh:minInclusive/maxInclusive) would not fire.
            val_raw = r["value"]["value"]
            try:
                if "." in val_raw:
                    g.add((obs, SOSA.hasSimpleResult, Literal(float(val_raw), datatype=XSD.float)))
                else:
                    g.add((obs, SOSA.hasSimpleResult, Literal(int(val_raw), datatype=XSD.integer)))
            except ValueError:
                g.add((obs, SOSA.hasSimpleResult, Literal(val_raw, datatype=XSD.string)))
            count += 1

        print(f"[INFO] {prop}: {count} observations in the sample.")

    print(f"\n[INFO] Total sample: {len(g)} triples.\n")
    return g

def main():
    print("=== Semantic graph validation with SHACL ===\n")

    data_graph = fetch_sample_graph()

    # shapes.ttl lives next to this script, so resolve it relative to the file.
    shapes_path = os.path.join(os.path.dirname(__file__), "shapes.ttl")
    shapes_graph = Graph().parse(shapes_path, format="turtle")

    print("[INFO] Running SHACL validation...\n")
    conforms, results_graph, results_text = validate(
        data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        advanced=True,     # required so pySHACL evaluates the SPARQL-based targets
        debug=False
    )

    if conforms:
        print("[RESULT] The (sample) graph fully CONFORMS.")
        print("   No shape violations were found.")
    else:
        # Count violations by scanning the human-readable report.
        violations = results_text.count("Constraint Violation")
        print(f"[RESULT] Found {violations} violations.")
        print(results_text[:3000])  # first violations, for a quick overview

if __name__ == "__main__":
    main()