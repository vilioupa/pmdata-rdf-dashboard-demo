import json
import os
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD

def process_heart_rate_data(user_id):
    """Parses high-frequency heart rate JSON data and generates an RDF graph in N-Triples format using batching."""

    # 1. INITIALIZATION & CONFIGURATION
    print(f"[INFO] Initializing Batch RDF Processor for High-Frequency HR Data (User: {user_id})")

    # Namespaces
    SOSA = Namespace("http://www.w3.org/ns/sosa/")
    EX = Namespace("http://example.org/thesis/")

    # Dynamic User Entity
    user_uri = URIRef(EX[f"user_{user_id.lower()}"])
    hr_property = URIRef(EX["HeartRate"])

    # Output setup
    INPUT_FILE = f'data/{user_id}/heart_rate.json'
    OUTPUT_DIR = f'rdf_output/{user_id}/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_FILE = f"{OUTPUT_DIR}heart_rate_graph.nt" # N-Triples for large files

    # Cleanup previous run
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # Batch configuration
    BATCH_SIZE = 50000
    g = Graph()        
    counter = 0
    total_triples = 0

    # 2. DATA LOADING
    try:
        print(f"[PROCESS] Loading {INPUT_FILE}...")
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
        print(f"[SUCCESS] {len(data)} records loaded successfully for {user_id}.")
    except FileNotFoundError:
        print(f"[WARNING] File '{INPUT_FILE}' not found. Skipping user {user_id}.\n")
        return

    # 3. BATCH PROCESSING LOOP
    print(f"[PROCESS] Processing records in batches of {BATCH_SIZE}...")

    for item in data:
        # 1. Extraction
        raw_time = item['dateTime']
        hr_value = int(item['value']['bpm'])
        
        # 2. Remove spaces and colons to construct a valid URI
        safe_time_id = raw_time.replace(" ", "_").replace(":", "")
        obs_id = URIRef(EX[f"Obs_{user_id}_HeartRate_{safe_time_id}"])

        # 3. Triple Generation
        g.add((obs_id, RDF.type, SOSA.Observation))
        g.add((obs_id, SOSA.hasFeatureOfInterest, user_uri))
        g.add((obs_id, SOSA.observedProperty, hr_property))
        g.add((obs_id, SOSA.hasSimpleResult, Literal(hr_value, datatype=XSD.integer)))
        
        # Convert timestamp to ISO 8601 format
        iso_time = raw_time.replace(" ", "T")
        g.add((obs_id, SOSA.resultTime, Literal(iso_time, datatype=XSD.dateTime)))

        counter += 1

        # 4. Batch Writing to Disk
        if counter % BATCH_SIZE == 0:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(g.serialize(format='nt'))
            
            total_triples += len(g)
            g = Graph() # Clear memory buffer
            print(f"[STATUS] Saved batch. Processed {counter} records...", end="\r")


    # 4. FINALIZATION
    # Write any remaining triples in the final batch
    if len(g) > 0:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(g.serialize(format='nt'))
        total_triples += len(g)

    print(f"\n[SUCCESS] Semantic graph generation complete for {user_id}.")
    print(f"Total Triples: {total_triples}")
    print(f"File saved as: {OUTPUT_FILE}\n")