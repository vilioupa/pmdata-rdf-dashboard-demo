import json
import os
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD

def process_activity_data(user_id):
    """Parses activity JSON files for a user and generates an RDF graph in N-Triples format using batching."""

    # 1. INITIALIZATION & CONFIGURATION
    print(f"\n[INFO] Initializing Batch RDF Processor for Activity Data (User: {user_id})")

    # Namespaces
    SOSA = Namespace("http://www.w3.org/ns/sosa/")
    EX = Namespace("http://example.org/thesis/")

    # Dynamic User Entity
    user_uri = URIRef(EX[f"user_{user_id.lower()}"])

    # Output setup
    OUTPUT_DIR = f'rdf_output/{user_id}/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_FILE = f"{OUTPUT_DIR}activity_graph.nt" # N-Triples for large files

    # Cleanup previous run
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    BATCH_SIZE = 50000
    g = Graph()        
    counter = 0
    total_triples = 0

    # Dictionary mapping JSON files to their corresponding SOSA properties
    activity_files = {
        'steps.json': 'StepCount',
        'calories.json': 'Calories',
        'very_active_minutes.json': 'VeryActiveMinutes',
        'sedentary_minutes.json': 'SedentaryMinutes'
    }

    DATA_DIR = f'data/{user_id}/'

    # 2. DATA EXTRACTION & BATCH PROCESSING
    for filename, prop_name in activity_files.items():
        input_file = os.path.join(DATA_DIR, filename)
        property_uri = URIRef(EX[prop_name])

        try:
            print(f"\n[PROCESS] Loading {input_file}...")
            with open(input_file, 'r') as f:
                data = json.load(f)
            print(f"[SUCCESS] {len(data)} records loaded for {prop_name}.")
        except FileNotFoundError:
            # Skip to the next metric if the specific file is missing
            print(f"[WARNING] File '{input_file}' not found. Skipping {prop_name}.")
            continue

        # 3. TRIPLE GENERATION (SOSA Ontology)
        for item in data:
            raw_time = item['dateTime'] # e.g. "2019-11-01 00:00:00"
            val = item['value']

            # Remove spaces and colons to construct a valid URI
            clean_time_id = raw_time.replace(" ", "_").replace(":", "")
            obs_id = URIRef(EX[f"Obs_{user_id}_{prop_name}_{clean_time_id}"])

            # Triple Generation
            g.add((obs_id, RDF.type, SOSA.Observation))
            g.add((obs_id, SOSA.hasFeatureOfInterest, user_uri))
            g.add((obs_id, SOSA.observedProperty, property_uri))
            
            # Dynamic datatype casting (values can be either floats or integers)
            try:
                num_val = float(val)
                if num_val.is_integer():
                    g.add((obs_id, SOSA.hasSimpleResult, Literal(int(num_val), datatype=XSD.integer)))
                else:
                    g.add((obs_id, SOSA.hasSimpleResult, Literal(num_val, datatype=XSD.float)))
            except ValueError:
                g.add((obs_id, SOSA.hasSimpleResult, Literal(str(val), datatype=XSD.string)))
            
            # Convert timestamp to ISO 8601 format
            iso_time = raw_time.replace(" ", "T")
            g.add((obs_id, SOSA.resultTime, Literal(iso_time, datatype=XSD.dateTime)))

            counter += 1

            # 4. BATCH FLUSH TO DISK
            # Write in batches to prevent RAM overflow for large datasets
            if counter % BATCH_SIZE == 0:
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
                    out_f.write(g.serialize(format='nt'))
                
                total_triples += len(g)
                g = Graph() # Clear memory buffer
                print(f"[STATUS] Saved batch. Processed {counter} overall records...", end="\r")

    # 5. FINALIZATION
    # Write any remaining triples in the final batch
    if len(g) > 0:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
            out_f.write(g.serialize(format='nt'))
        total_triples += len(g)

    print(f"\n\n[SUCCESS] Activity graph generation complete for {user_id}.")
    print(f"Total Records Processed: {counter}")
    print(f"Total Triples Created:   {total_triples}")
    print(f"File saved as:           {OUTPUT_FILE}\n")