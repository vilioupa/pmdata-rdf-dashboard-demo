import json
import os
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD

def process_steps_data(user_id):
    """Parses steps JSON data and generates an RDF graph in N-Triples format using batching."""

    # 1. INITIALIZATION & CONFIGURATION
    print(f"\n[INFO] Initializing Batch RDF Processor for Steps Data (User: {user_id})...")

    # Namespaces
    SOSA = Namespace("http://www.w3.org/ns/sosa/")
    EX = Namespace("http://example.org/thesis/")

    # Dynamic User Entity
    user_uri = URIRef(EX[f"user_{user_id.lower()}"])  # Same user URI scheme as the other parsers
    step_property = URIRef(EX["StepCount"])

    # Output setup
    INPUT_FILE = f'data/{user_id}/steps.json'
    OUTPUT_DIR = f'rdf_output/{user_id}/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OUTPUT_FILE = f"{OUTPUT_DIR}steps_graph.nt"  # N-Triples for large files
    BATCH_SIZE = 50000

    # 2. DATA LOADING
    try:
        print(f"[PROCESS] Loading {INPUT_FILE}...")
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
        print(f"[SUCCESS] {len(data)} step records loaded successfully for {user_id}.")
    except FileNotFoundError:
        print(f"[ERROR] File '{INPUT_FILE}' not found. Skipping user {user_id}.")
        return

    # Clean up a previous run only after a successful load, so a missing input leaves old output intact.
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # 3. BATCH PROCESSING LOOP
    print(f"[PROCESS] Processing records in batches of {BATCH_SIZE}...")

    g = Graph()
    counter = 0
    total_triples = 0

    for item in data:
        # 1. Extraction
        raw_time = item['dateTime']  # e.g. "2019-11-01 00:00:00"
        steps_value = int(item['value'])

        # 2. Sanitize the timestamp into a URI-safe ID ("2019-11-01 00:00:00" -> "2019-11-01_000000")
        clean_time_id = raw_time.replace(" ", "_").replace(":", "")
        obs_id = URIRef(EX[f"Obs_{user_id}_Steps_{clean_time_id}"])

        # 3. Triple Generation
        g.add((obs_id, RDF.type, SOSA.Observation))
        g.add((obs_id, SOSA.hasFeatureOfInterest, user_uri))
        g.add((obs_id, SOSA.observedProperty, step_property))
        g.add((obs_id, SOSA.hasSimpleResult, Literal(steps_value, datatype=XSD.integer)))

        # Convert timestamp to ISO 8601 format
        iso_time = raw_time.replace(" ", "T")
        g.add((obs_id, SOSA.resultTime, Literal(iso_time, datatype=XSD.dateTime)))

        counter += 1

        # 4. Batch flush: write and clear the buffer to keep memory bounded on large files
        if counter % BATCH_SIZE == 0:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(g.serialize(format='nt'))

            total_triples += len(g)
            g = Graph()  # Clear memory buffer
            print(f"[STATUS] Saved batch! Processed {counter} records...", end="\r")

    # 5. FINALIZATION
    # Write any remaining triples in the final (partial) batch
    if len(g) > 0:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(g.serialize(format='nt'))
        total_triples += len(g)

    print(f"\n[SUCCESS] Semantic graph generation complete for {user_id}.")
    print(f"Total Records Processed: {counter}")
    print(f"Total Triples Created:   {total_triples}")
    print(f"File saved as:           {OUTPUT_FILE}\n")