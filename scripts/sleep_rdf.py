import os
import json
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD

def process_sleep_data(user_id):
    """Parses sleep-session JSON data and generates an RDF graph in Turtle format."""

    # 1. INITIALIZATION & CONFIGURATION
    print(f"[INFO] Initializing RDF Graph Generation for Sleep Data (User: {user_id})...")
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
    def create_observation(log_id, time_str, prop_name, value, datatype=XSD.integer):
        """Constructs a SOSA Observation node for a single sleep-log metric."""
        # Key the URI by the sleep-session log ID (unique per night) plus the property,
        # so every metric of the same session shares one stable, collision-free prefix.
        obs_id = URIRef(EX[f"Obs_{user_id}_Sleep_{log_id}_{prop_name}"])

        g.add((obs_id, RDF.type, SOSA.Observation))
        g.add((obs_id, SOSA.hasFeatureOfInterest, user_uri))
        g.add((obs_id, SOSA.observedProperty, URIRef(EX[prop_name])))
        g.add((obs_id, SOSA.hasSimpleResult, Literal(value, datatype=datatype)))
        g.add((obs_id, SOSA.resultTime, Literal(time_str, datatype=XSD.dateTime)))

    # 2. DATA LOADING
    INPUT_FILE = f'data/{user_id}/sleep.json'

    try:
        print(f"[PROCESS] Loading {INPUT_FILE}...")
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
        print(f"[SUCCESS] {len(data)} sleep records loaded successfully for {user_id}.")
    except FileNotFoundError:
        print(f"[ERROR] File '{INPUT_FILE}' not found. Skipping user {user_id}.")
        print()
        return

    # 3. DATA TRANSFORMATION (ETL)
    print(f"[PROCESS] Transforming JSON data to SOSA Triples for {user_id}...")

    for entry in data:
        log_id = str(entry['logId'])
        start_time = entry['startTime']  # ISO string, e.g. "2019-11-01T23:00:00"

        # Top-level metrics, each guarded since not every record reports all of them.
        if 'minutesAsleep' in entry:
            create_observation(log_id, start_time, "MinutesAsleep", entry['minutesAsleep'])

        if 'minutesAwake' in entry:
            create_observation(log_id, start_time, "MinutesAwake", entry['minutesAwake'])

        if 'efficiency' in entry:
            create_observation(log_id, start_time, "SleepEfficiency", entry['efficiency'])

        # Per-stage minutes live nested under levels.summary (Deep, REM, Light, Wake).
        if 'levels' in entry and 'summary' in entry['levels']:
            summary = entry['levels']['summary']

            if 'deep' in summary:
                create_observation(log_id, start_time, "DeepSleepMinutes", summary['deep']['minutes'])

            if 'rem' in summary:
                create_observation(log_id, start_time, "RemSleepMinutes", summary['rem']['minutes'])

            if 'light' in summary:
                create_observation(log_id, start_time, "LightSleepMinutes", summary['light']['minutes'])

            if 'wake' in summary:
                create_observation(log_id, start_time, "WakeStageMinutes", summary['wake']['minutes'])

    # 4. SERIALIZATION
    OUTPUT_DIR = f'rdf_output/{user_id}/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OUTPUT_FILE = f"{OUTPUT_DIR}sleep_graph.ttl"

    print(f"[STATUS] Serializing graph to {OUTPUT_FILE}...")
    g.serialize(destination=OUTPUT_FILE, format="turtle")

    print(f"[SUCCESS] Semantic graph generation complete for {user_id}.")
    print(f"Total Triples: {len(g)}")
    print(f"File saved as: {OUTPUT_FILE}\n")