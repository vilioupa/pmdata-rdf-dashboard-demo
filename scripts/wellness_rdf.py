import os
import pandas as pd
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD

def process_wellness_data(user_id):
    """Parses wellness CSV data and generates an RDF graph in Turtle format."""

    # 1. INITIALIZATION & CONFIGURATION
    print(f"[INFO] Initializing RDF Graph Generation for Wellness Data (User: {user_id})...")
    g = Graph()

    # Namespaces
    SOSA = Namespace("http://www.w3.org/ns/sosa/")
    EX = Namespace("http://example.org/thesis/")

    # Bind namespaces for clean Turtle serialization
    g.bind("sosa", SOSA)
    g.bind("ex", EX)
    g.bind("xsd", XSD)

    # Dynamic User Entity
    user_uri = URIRef(EX[f"user_{user_id.lower()}"])

    # Helper Function
    def add_observation(date_str, name, value, datatype):
        """Constructs a SOSA Observation, skipping NaNs and standardizing the resultTime."""
        if pd.isna(value):
            return

        # Key the URI by the date part only, keeping it clean and one node per metric per day.
        try:
            str_date_id = date_str.split('T')[0].split(' ')[0].replace('/', '-')
        except Exception:
            str_date_id = "unknown"

        observation_id = URIRef(EX[f"Obs_{user_id}_{name}_{str_date_id}"])

        g.add((observation_id, RDF.type, SOSA.Observation))
        g.add((observation_id, SOSA.hasFeatureOfInterest, user_uri))
        g.add((observation_id, SOSA.observedProperty, URIRef(EX[name])))
        g.add((observation_id, SOSA.hasSimpleResult, Literal(value, datatype=datatype)))

        # Tag the time as dateTime only when it carries a time component, otherwise plain string.
        time_datatype = XSD.dateTime if ('T' in str(date_str) or ':' in str(date_str)) else XSD.string
        g.add((observation_id, SOSA.resultTime, Literal(date_str, datatype=time_datatype)))

    # 2. DATA LOADING
    INPUT_FILE = f'data/{user_id}/wellness.csv'

    try:
        wellness_data = pd.read_csv(INPUT_FILE)
        print(f"[SUCCESS] {len(wellness_data)} wellness records found for {user_id}.")
    except FileNotFoundError:
        print(f"[ERROR] File '{INPUT_FILE}' not found. Skipping user {user_id}.")
        return

    # 3. DATA TRANSFORMATION (ETL)
    print(f"[PROCESS] Generating RDF triples from wellness metrics for {user_id}...")

    for index, row in wellness_data.iterrows():
        # Self-reported daily wellness survey; one timestamp per row.
        date = str(row['effective_time_frame'])

        # Map each survey column to its own observation.
        add_observation(date, "Fatigue", row['fatigue'], XSD.integer)
        add_observation(date, "Mood", row['mood'], XSD.integer)
        add_observation(date, "Readiness", row['readiness'], XSD.integer)
        add_observation(date, "SleepDurationHours", row['sleep_duration_h'], XSD.float)
        add_observation(date, "SleepQuality", row['sleep_quality'], XSD.integer)
        add_observation(date, "Soreness", row['soreness'], XSD.integer)
        add_observation(date, "SorenessArea", row['soreness_area'], XSD.string)
        add_observation(date, "Stress", row['stress'], XSD.integer)

    # 4. SERIALIZATION
    OUTPUT_DIR = f'rdf_output/{user_id}/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OUTPUT_FILE = f"{OUTPUT_DIR}wellness_graph.ttl"

    print(f"[STATUS] Serializing graph to {OUTPUT_FILE}...")
    g.serialize(destination=OUTPUT_FILE, format='turtle')

    print(f"[SUCCESS] Semantic graph generation complete for {user_id}.")
    print(f"Total Triples: {len(g)}\n")