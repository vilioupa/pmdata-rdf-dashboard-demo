import os
import pandas as pd
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD

def process_context_data(user_id):
    """ parses context data (sleep, injury, reporting) from CSVs and generates an RDF graph in Turtle format. """

    # 1. INITIALIZATION & CONFIGURATION
    print(f"[INFO] Initializing RDF Graph Generation for Context Data (User: {user_id})")
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
    def create_observation(date_str, prop_name, value, datatype):
        """
        Constructs a SOSA Observation node and standardizes the timestamp to XSD.dateTime where possible."""
        if pd.isna(value):
            return
        
        # Create a URI-safe date ID (e.g. extracting "2019-11-01")
        try:
            clean_date_id = str(date_str).split('T')[0].split(' ')[0].replace('/', '-')
        except Exception:
            clean_date_id = "unknown"
        
        obs_id = URIRef(EX[f"Obs_{user_id}_{prop_name}_{clean_date_id}"])
        
        # Check if date_str is suitable for dateTime or just string
        time_datatype = XSD.string
        if 'T' in str(date_str) or ':' in str(date_str):
            time_datatype = XSD.dateTime

        # Triple Generation
        g.add((obs_id, RDF.type, SOSA.Observation))
        g.add((obs_id, SOSA.hasFeatureOfInterest, user_uri))
        g.add((obs_id, SOSA.observedProperty, URIRef(EX[prop_name])))
        g.add((obs_id, SOSA.hasSimpleResult, Literal(value, datatype=datatype)))
        g.add((obs_id, SOSA.resultTime, Literal(str(date_str), datatype=time_datatype)))

    DATA_DIR = f'data/{user_id}/'

    # 2. DATA TRANSFORMATION (ETL)

    # --- DATASET 1: SLEEP METRICS ---
    sleep_file = f'{DATA_DIR}sleep_score.csv'
    try:
        df_sleep = pd.read_csv(sleep_file)
        print(f"[PROCESS] Processing sleep_score.csv for {user_id}: Found {len(df_sleep)} entries.")
        
        for index, row in df_sleep.iterrows():
            ts = row['timestamp']
            create_observation(ts, "SleepScore", row['overall_score'], XSD.integer)
            create_observation(ts, "DeepSleepMinutes", row['deep_sleep_in_minutes'], XSD.integer)
            create_observation(ts, "RestingHeartRate", row['resting_heart_rate'], XSD.integer)
            create_observation(ts, "Restlessness", row['restlessness'], XSD.float)

    except FileNotFoundError:
        print("[ERROR] sleep_score.csv not found.")

    # --- DATASET 2: INJURY LOGS ---
    injury_file = f'{DATA_DIR}injury.csv'
    try:
        df_injury = pd.read_csv(injury_file)
        print(f"[PROCESS] Processing injury.csv for {user_id}: Found {len(df_injury)} entries.")
        
        for _, row in df_injury.iterrows():
            ts = row['effective_time_frame']
            injuries = str(row['injuries'])

            # Ignore empty arrays or missing values
            if injuries != "{}" and injuries != "nan":
                create_observation(ts, "InjuryLog", injuries, XSD.string)
    except FileNotFoundError:
        print(f"[WARNING] {injury_file} not found. Skipping injury logs.")

    # --- DATASET 3: DAILY REPORTING ---
    reporting_file = f'{DATA_DIR}reporting.csv'
    try:
        df_report = pd.read_csv(reporting_file)
        print(f"[PROCESS] Processing reporting.csv for {user_id}: Found {len(df_report)} entries.")
        
        for index, row in df_report.iterrows():
            ts_raw = row['timestamp']
            
            # Standardize timestamp to ISO 8601 format
            try:
                ts_iso = pd.to_datetime(ts_raw, dayfirst=True).isoformat()
            except Exception:
                ts_iso = str(ts_raw)

            if not pd.isna(row['weight']):
                create_observation(ts_iso, "BodyWeight", row['weight'], XSD.float)
            if not pd.isna(row['glasses_of_fluid']):
                create_observation(ts_iso, "WaterGlasses", row['glasses_of_fluid'], XSD.integer)
            if not pd.isna(row['alcohol_consumed']):
                create_observation(ts_iso, "AlcoholConsumed", row['alcohol_consumed'], XSD.string)
            if not pd.isna(row['meals']):
                create_observation(ts_iso, "MealsEaten", row['meals'], XSD.string)

    except FileNotFoundError:
        print(f"[WARNING] {reporting_file} not found. Skipping daily reporting data.")


    # 3. SERIALIZATION
    OUTPUT_DIR = f'rdf_output/{user_id}/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    OUTPUT_FILE = f"{OUTPUT_DIR}context_graph.ttl"
    
    print(f"[STATUS] Serializing graph to {OUTPUT_FILE}...")
    g.serialize(destination=OUTPUT_FILE, format="turtle")

    print(f"[SUCCESS] Semantic graph generation complete for {user_id}. Total Triples: {len(g)}")
    print(f"File saved as: {OUTPUT_FILE}\n")