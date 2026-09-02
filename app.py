# Streamlit dashboard for the thesis: reads wellness data as RDF (SOSA/SSN) from
# GraphDB, visualizes it, runs Isolation Forest anomaly detection, writes anomalies
# back to the graph with PROV-O provenance, and derives ex:HighRiskDay via a SPARQL rule.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import re
import os

# PAGE CONFIGURATION
st.set_page_config(page_title="Thesis Dashboard", layout="wide", page_icon="📊")
st.title("Διαδραστικό Dashboard Ανάλυσης Δεδομένων Ευεξίας")

# Endpoint is read from the environment so the repo can be run against a different
# GraphDB without editing the code; falls back to the local default.
SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT", "http://localhost:7200/repositories/thesis")

# DATA-ACCESS FUNCTIONS
# All read queries are cached with @st.cache_data so repeated reruns (which Streamlit
# triggers on every widget interaction) don't re-hit GraphDB for unchanged data.

@st.cache_data
def get_average_steps():
    """Fetch every raw StepCount reading (one row per observation) for all users.
    Despite the name, averaging is done later at the page level, not here."""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = """
    PREFIX sosa: <http://www.w3.org/ns/sosa/>
    PREFIX ex: <http://example.org/thesis/>
    SELECT ?user ?stepVal
    WHERE {
        ?obs a sosa:Observation ;
             sosa:hasFeatureOfInterest ?user ;
             sosa:observedProperty ex:StepCount ;
             sosa:hasSimpleResult ?stepVal .
    }
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    data = []
    try:
        results = sparql.query().convert()
        for r in results["results"]["bindings"]:
            user_uri = r["user"]["value"]
            # Regex pulls the participant ID (P01, P02, ...) from anywhere in the URI,
            # so the code doesn't depend on a fixed URI layout.            
            match = re.search(r'P\d+', user_uri, re.IGNORECASE)
            user_id = match.group(0).upper() if match else "UNKNOWN"
            
            data.append({
                "User": user_id,
                "stepVal": float(r["stepVal"]["value"])
            })
    except Exception as e:
        st.error(f"Σφάλμα στα βήματα: {e}")
    
    return pd.DataFrame(data)

@st.cache_data
def get_user_fusion_data(user_id):
    """Fetch time-synchronized heart-rate and step readings for one user,
    joined on a shared resultTime so each row is a single moment in time."""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query_fusion = f"""
    PREFIX sosa: <http://www.w3.org/ns/sosa/>
    PREFIX ex: <http://example.org/thesis/>
    SELECT ?time ?heartRate ?steps
    WHERE {{
        ?hrObs a sosa:Observation ;
               sosa:hasFeatureOfInterest ex:user_{user_id.lower()} ;
               sosa:observedProperty ex:HeartRate ;
               sosa:resultTime ?time ;
               sosa:hasSimpleResult ?heartRate .

        ?stepObs a sosa:Observation ;
                 sosa:hasFeatureOfInterest ex:user_{user_id.lower()} ;
                 sosa:observedProperty ex:StepCount ;
                 sosa:resultTime ?time ;
                 sosa:hasSimpleResult ?steps .
    }}
    ORDER BY ?time
    LIMIT 1000
    """
    sparql.setQuery(query_fusion)
    sparql.setReturnFormat(JSON)
    data = []
    try:
        results = sparql.query().convert()
        for r in results["results"]["bindings"]:
            data.append({
                "Time": r["time"]["value"],
                "HeartRate": float(r["heartRate"]["value"]),
                "Steps": float(r["steps"]["value"])
            })
    except Exception: pass
    
    df = pd.DataFrame(data)
    if not df.empty:
        # Parse timestamps up front so downstream plotting/sorting is chronological.
        df['Time'] = pd.to_datetime(df['Time'])
    return df

@st.cache_data
def get_lifestyle_data():
    """Fetch lifestyle context (water, alcohol, meals) and normalize each value to a float."""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = """
    PREFIX sosa: <http://www.w3.org/ns/sosa/>
    PREFIX ex: <http://example.org/thesis/>
    SELECT ?user ?property ?value
    WHERE {
        ?obs a sosa:Observation ;
             sosa:hasFeatureOfInterest ?user ;
             sosa:observedProperty ?property ;
             sosa:hasSimpleResult ?value .
        FILTER(?property IN (ex:WaterGlasses, ex:AlcoholConsumed, ex:MealsEaten))
    }
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Collapse mixed logging formats (yes/no, counts, comma-separated meal lists)
    # onto one numeric scale so they can be aggregated and plotted uniformly.
    def smart_parser(prop_name, val):
        val_str = str(val).strip().lower()
        if val_str in ['yes', 'true']: return 1.0
        if val_str in ['no', 'false']: return 0.0
        if 'meal' in prop_name.lower():
            if val_str == 'nan' or val_str == '': return 0.0
            # Meals are logged as a comma-separated string, so item count = number of meals.
            return float(len(val_str.split(',')))
        try: return float(val)
        except ValueError: return None
            
    results = sparql.query().convert()
    data = []
    for r in results["results"]["bindings"]:
        prop = r["property"]["value"].split('/')[-1]
        val = smart_parser(prop, r["value"]["value"])
        
        user_uri = r["user"]["value"]
        match = re.search(r'P\d+', user_uri, re.IGNORECASE)
        user_id = match.group(0).upper() if match else "UNKNOWN"

        if val is not None:
            data.append({
                "User": user_id, 
                "Property": prop,
                "Value": val
            })
    return pd.DataFrame(data)

@st.cache_data
def get_daily_metrics_data():
    """Fetch daily wellness metrics and pivot to one row per (Date, User),
    so each property (sleep, stress, HR, ...) becomes its own column for correlation."""    
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = """
    PREFIX sosa: <http://www.w3.org/ns/sosa/>
    PREFIX ex: <http://example.org/thesis/>
    SELECT ?user ?property ?value ?dateStr
    WHERE {
        ?obs a sosa:Observation ;
             sosa:hasFeatureOfInterest ?user ;
             sosa:observedProperty ?property ;
             sosa:hasSimpleResult ?value ;
             sosa:resultTime ?time .
        FILTER(?property IN (ex:SleepScore, ex:WaterGlasses, ex:AlcoholConsumed, ex:DeepSleepMinutes, ex:RestingHeartRate, ex:Stress))
        BIND(SUBSTR(STR(?time), 1, 10) AS ?dateStr)
    }
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    def smart_parser(prop_name, val):
        val_str = str(val).strip().lower()
        if val_str in ['yes', 'true']: return 1.0
        if val_str in ['no', 'false']: return 0.0
        try: return float(val)
        except ValueError: return None
            
    results = sparql.query().convert()
    data = []
    for r in results["results"]["bindings"]:
        prop = r["property"]["value"].split('/')[-1]
        val = smart_parser(prop, r["value"]["value"])
        
        user_uri = r["user"]["value"]
        match = re.search(r'P\d+', user_uri, re.IGNORECASE)
        user_id = match.group(0).upper() if match else "UNKNOWN"
        
        if val is not None:
            data.append({
                "User": user_id, 
                "Date": r["dateStr"]["value"],
                "Property": prop,
                "Value": val
            })
    df = pd.DataFrame(data)
    if not df.empty:
        return df.pivot_table(index=['Date', 'User'], columns='Property', values='Value', aggfunc='mean').reset_index()
    return pd.DataFrame()


def insert_anomaly_to_graphdb(user_id, date, score, algorithm="IsolationForest"):
    """Write a detected anomaly back to GraphDB with PROV-O provenance.
    Idempotent: it first deletes any previous version of THIS SAME anomaly, then
    re-inserts it, so repeated saves never create duplicate or conflicting triples."""
    update_endpoint = SPARQL_ENDPOINT + "/statements"
    sparql = SPARQLWrapper(update_endpoint)

    # Deterministic URI (user + date) => the same anomaly always maps to the same URI.
    anomaly_id = f"Anomaly_{user_id}_{date.replace('-', '')}"
    # Provenance timestamp (when the detection actually ran).
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")

    # 1. Remove any previous version of THIS specific anomaly
    delete_query = f"""
    PREFIX ex: <http://example.org/thesis/>
    DELETE WHERE {{
        ex:{anomaly_id} ?p ?o .
    }}
    """
    sparql.setQuery(delete_query)
    sparql.setMethod('POST')
    try:
        sparql.query()
    except Exception as e:
        print(f"Σφάλμα κατά τον καθαρισμό ανωμαλίας: {e}")
        return False

    # 2. Insert the anomaly plus its PROV-O provenance chain
    # The entity records who/when/how it was generated (Activity + SoftwareAgent),
    # making each finding traceable back to the algorithm that produced it.
    query = f"""
    PREFIX ex: <http://example.org/thesis/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX prov: <http://www.w3.org/ns/prov#>

    INSERT DATA {{
        ex:{anomaly_id} a ex:AnomalyDetected , prov:Entity ;
                        ex:refersToUser ex:user_{user_id.lower()} ;
                        ex:anomalyDate "{date}"^^xsd:date ;
                        ex:anomalyScore "{score}"^^xsd:float ;
                        ex:detectedByAlgorithm "{algorithm}" ;
                        prov:wasGeneratedBy ex:detection_{algorithm} ;
                        prov:generatedAtTime "{generated_at}"^^xsd:dateTime .

        ex:detection_{algorithm} a prov:Activity ;
                        prov:used ex:algorithm_{algorithm} .

        ex:algorithm_{algorithm} a prov:SoftwareAgent ;
                        ex:algorithmName "{algorithm}" .
    }}
    """
    sparql.setQuery(query)
    sparql.setMethod('POST')

    try:
        sparql.query()
        return True
    except Exception as e:
        print(f"Σφάλμα κατά την εγγραφή στο GraphDB: {e}")
        return False

# --- SEMANTIC RULE: HighRiskDay (reasoning performed inside the graph) ---
# A day is labeled "high risk" only when three conditions co-occur on the SAME day:
#   (1) a detected sleep anomaly (ex:AnomalyDetected, written by page 5)
#   (2) high stress that day (ex:Stress >= threshold)
#   (3) alcohol consumption that day (ex:AlcoholConsumed = "Yes")

def _high_risk_where_pattern(stress_threshold):
    """The shared WHERE graph pattern of the rule, reused by both preview and insert
    so the two can never drift apart in their definition of 'high risk'."""
    return f"""
        # (1) A sleep anomaly already stored in the graph
        ?anomaly a ex:AnomalyDetected ;
                 ex:refersToUser ?user ;
                 ex:anomalyDate ?anomalyDate .
        BIND(STR(?anomalyDate) AS ?date)
 
        # (2) High stress on the SAME day (scale 1-5, neutral = 3)
        ?stressObs a sosa:Observation ;
                   sosa:hasFeatureOfInterest ?user ;
                   sosa:observedProperty ex:Stress ;
                   sosa:resultTime ?stressTime ;
                   sosa:hasSimpleResult ?stressVal .
        FILTER(SUBSTR(STR(?stressTime), 1, 10) = ?date)
        FILTER(xsd:decimal(?stressVal) >= {stress_threshold})
 
        # (3) Alcohol consumed on the SAME day
        ?alcObs a sosa:Observation ;
                sosa:hasFeatureOfInterest ?user ;
                sosa:observedProperty ex:AlcoholConsumed ;
                sosa:resultTime ?alcTime ;
                sosa:hasSimpleResult ?alcVal .
        FILTER(SUBSTR(STR(?alcTime), 1, 10) = ?date)
        FILTER(LCASE(STR(?alcVal)) IN ("yes", "true"))
    """

def preview_high_risk_days(stress_threshold):
    """Show WHICH days the rule would label, WITHOUT writing anything to the graph."""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = f"""
    PREFIX sosa: <http://www.w3.org/ns/sosa/>
    PREFIX ex: <http://example.org/thesis/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT DISTINCT ?user ?date ?stressVal ?anomaly
    WHERE {{
        {_high_risk_where_pattern(stress_threshold)}
    }}
    ORDER BY ?user ?date
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    data = []
    try:
        results = sparql.query().convert()
        for r in results["results"]["bindings"]:
            match = re.search(r'P\d+', r["user"]["value"], re.IGNORECASE)
            user_id = match.group(0).upper() if match else "UNKNOWN"
            data.append({
                "User": user_id,
                "Date": r["date"]["value"],
                "StressLevel": float(r["stressVal"]["value"]),
                "Anomaly": r["anomaly"]["value"].split('/')[-1]
            })
    except Exception as e:
        st.error(f"Σφάλμα στο preview κανόνα: {e}")
    return pd.DataFrame(data)

def insert_high_risk_days(stress_threshold):
    """Run the reasoning INSIDE the graph: derive and store ex:HighRiskDay entities.
    Delete-then-insert, so the graph always reflects the latest chosen threshold
    instead of accumulating stale HighRiskDay entities from previous runs."""
    update_endpoint = SPARQL_ENDPOINT + "/statements"
    sparql = SPARQLWrapper(update_endpoint)

    # 1. Clear previous HighRiskDay entities (and only those)
    delete_query = """
    PREFIX ex: <http://example.org/thesis/>
    DELETE {
        ?rd ?p ?o .
    }
    WHERE {
        ?rd a ex:HighRiskDay ;
            ?p ?o .
    }
    """
    sparql.setQuery(delete_query)
    sparql.setMethod('POST')
    try:
        sparql.query()
    except Exception as e:
        print(f"Σφάλμα κατά τον καθαρισμό HighRiskDay: {e}")
        return False

    # 2. Derive and insert the current HighRiskDay entities
    # A deterministic IRI (user + date) is built inline so re-running the rule
    # overwrites rather than duplicates each day's node    
    insert_query = f"""
    PREFIX sosa: <http://www.w3.org/ns/sosa/>
    PREFIX ex: <http://example.org/thesis/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    INSERT {{
        ?riskDay a ex:HighRiskDay ;
                 ex:refersToUser ?user ;
                 ex:riskDate ?anomalyDate ;
                 ex:derivedFromAnomaly ?anomaly ;
                 ex:stressLevel ?stressVal ;
                 ex:stressThreshold {stress_threshold} ;
                 ex:hadAlcohol true .
    }}
    WHERE {{
        {_high_risk_where_pattern(stress_threshold)}
        BIND(IRI(CONCAT("http://example.org/thesis/HighRisk_",
                        STRAFTER(STR(?user), "user_"),
                        "_", REPLACE(?date, "-", ""))) AS ?riskDay)
    }}
    """
    sparql.setQuery(insert_query)
    sparql.setMethod('POST')
    try:
        sparql.query()
        return True
    except Exception as e:
        print(f"Σφάλμα κατά την εγγραφή HighRiskDay: {e}")
        return False

def get_high_risk_days():
    """Read back the ex:HighRiskDay entities already materialized in the graph."""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = """
    PREFIX ex: <http://example.org/thesis/>
    SELECT ?user ?date ?stress ?anomaly
    WHERE {
        ?rd a ex:HighRiskDay ;
            ex:refersToUser ?user ;
            ex:riskDate ?date .
        OPTIONAL { ?rd ex:stressLevel ?stress . }
        OPTIONAL { ?rd ex:derivedFromAnomaly ?anomaly . }
    }
    ORDER BY ?user ?date
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    data = []
    try:
        results = sparql.query().convert()
        for r in results["results"]["bindings"]:
            match = re.search(r'P\d+', r["user"]["value"], re.IGNORECASE)
            user_id = match.group(0).upper() if match else "UNKNOWN"
            data.append({
                "User": user_id,
                "Date": r["date"]["value"],
                # OPTIONAL fields may be missing, so guard before reading them.
                "StressLevel": float(r["stress"]["value"]) if "stress" in r else None,
                "Anomaly": r["anomaly"]["value"].split('/')[-1] if "anomaly" in r else None
            })
    except Exception as e:
        st.error(f"Σφάλμα ανάκτησης HighRiskDay: {e}")
    return pd.DataFrame(data)

# HELPER FUNCTIONS
def filter_df(df_to_filter, selected_user):
    """Apply the sidebar user filter; "Όλοι" (All) means no filtering."""
    if selected_user != "Όλοι":
        return df_to_filter[df_to_filter['User'] == selected_user]
    return df_to_filter

# SIDEBAR & DYNAMIC DATA
# Steps are loaded once here because they also drive the user list and color map below.
df_steps_raw = get_average_steps()

st.sidebar.header("Πίνακας Ελέγχου")
page = st.sidebar.radio("Επιλέξτε Σελίδα Ανάλυσης:", [
    "1. Δραστηριότητα & Παλμοί",
    "2. Προφίλ Lifestyle", 
    "3. Επίδραση Συνηθειών στον Ύπνο", 
    "4. Ψυχολογία & Φυσιολογία", 
    "5. Εντοπισμός Ανωμαλιών (ML)",
    "6. Σημασιολογικά Συμπεράσματα (Risk)"
])

st.sidebar.markdown("---")

# Build the user dropdown dynamically from whoever is actually in the data
# (UNKNOWN IDs from failed regex matches are excluded).
if not df_steps_raw.empty:
    raw_users = df_steps_raw['User'].dropna().unique().astype(str).tolist()
    user_list = ["Όλοι"] + sorted([u for u in raw_users if u != "UNKNOWN"])
    user_filter = st.sidebar.selectbox("Φιλτράρισμα Χρήστη:", user_list)
else:
    user_filter = st.sidebar.selectbox("Φιλτράρισμα Χρήστη:", ["Όλοι"])

# Assign one stable color per user so a given user looks the same on every chart/page.
USER_COLORS = {}
if not df_steps_raw.empty:
    available_colors = px.colors.qualitative.Safe
    for i, user in enumerate(sorted(df_steps_raw['User'].unique())):
        USER_COLORS[user] = available_colors[i % len(available_colors)]

# --- DASHBOARD PAGES ---

if page == "1. Δραστηριότητα & Παλμοί":
    st.subheader("Κινητική Δραστηριότητα και Καρδιακή Λειτουργία")
    if not df_steps_raw.empty:
        # Benchmark line = overall mean across all raw step readings.
        total_avg = df_steps_raw['stepVal'].mean()
        df_to_plot = filter_df(df_steps_raw, user_filter)
        df_compare = df_to_plot.groupby('User')['stepVal'].mean().reset_index()

        fig1 = px.bar(df_compare, x='User', y='stepVal', color='User', text_auto='.2f', color_discrete_map=USER_COLORS)
        fig1.add_hline(y=total_avg, line_dash="dash", line_color="red")
        st.plotly_chart(fig1, use_container_width=True)

    # The dual-axis HR/steps timeline only makes sense for a single user.
    if user_filter != "Όλοι":
        df_fusion = get_user_fusion_data(user_filter)
        if not df_fusion.empty:
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(df_fusion['Time'], df_fusion['HeartRate'], color='tab:red', label='Heart Rate')
            ax3 = ax2.twinx()
            ax3.fill_between(df_fusion['Time'], df_fusion['Steps'], color='tab:blue', alpha=0.3, label='Steps')
            st.pyplot(fig2)

elif page == "2. Προφίλ Lifestyle":
    st.subheader("Lifestyle Προφίλ (Νερό, Αλκοόλ, Γεύματα)")
    df_life = get_lifestyle_data()
    if not df_life.empty:
        df_life = filter_df(df_life, user_filter)
        df_grouped = df_life.groupby(['User', 'Property'])['Value'].mean().reset_index()
        fig = px.bar(df_grouped, x='Property', y='Value', color='User', barmode='group', text_auto='.2f', color_discrete_map=USER_COLORS)
        st.plotly_chart(fig, use_container_width=True)

elif page == "3. Επίδραση Συνηθειών στον Ύπνο":
    st.subheader("Επίδραση Αλκοόλ και Υδάτωσης στον Ύπνο")
    df_daily = get_daily_metrics_data()
    if not df_daily.empty and 'SleepScore' in df_daily.columns:
        df_habits = filter_df(df_daily.copy(), user_filter)
        col1, col2 = st.columns(2)
        with col1:
            if 'AlcoholConsumed' in df_habits.columns:
                # Reduce alcohol volume to a yes/no category for the boxplot split.
                df_habits['Alcohol_Category'] = df_habits['AlcoholConsumed'].apply(lambda x: 'Με Αλκοόλ' if x > 0 else 'Χωρίς Αλκοόλ')
                st.plotly_chart(px.box(df_habits, x='Alcohol_Category', y='SleepScore', color='User', color_discrete_map=USER_COLORS), use_container_width=True)
        with col2:
            if 'WaterGlasses' in df_habits.columns:
                st.plotly_chart(px.scatter(df_habits, x='WaterGlasses', y='SleepScore', color='User', color_discrete_map=USER_COLORS), use_container_width=True)

elif page == "4. Ψυχολογία & Φυσιολογία":
    st.subheader("Στρες & Καρδιακή Λειτουργία")
    df_daily = get_daily_metrics_data()
    if not df_daily.empty:
        df_physio = filter_df(df_daily.copy(), user_filter)
        col1, col2 = st.columns(2)
        with col1:
            if 'Stress' in df_physio.columns:
                st.plotly_chart(px.scatter(df_physio, x='Stress', y='SleepScore', color='User', color_discrete_map=USER_COLORS), use_container_width=True)
        with col2:
            if 'DeepSleepMinutes' in df_physio.columns:
                st.plotly_chart(px.scatter(df_physio, x='DeepSleepMinutes', y='RestingHeartRate', color='User', color_discrete_map=USER_COLORS), use_container_width=True)

elif page == "5. Εντοπισμός Ανωμαλιών (ML)":
    st.subheader("Ανάλυση 5.1: Εντοπισμός Κινδύνου στον Ύπνο (Isolation Forest)")
    st.markdown("Μοντέλο Μηχανικής Μάθησης που εντοπίζει επικίνδυνες αποκλίσεις στον Βαθύ Ύπνο και τους Παλμούς Ηρεμίας.")
    
    df_daily = get_daily_metrics_data()
    if not df_daily.empty and 'DeepSleepMinutes' in df_daily.columns and 'RestingHeartRate' in df_daily.columns:
        df_ml = df_daily.dropna(subset=['DeepSleepMinutes', 'RestingHeartRate']).copy()
        
        features = ['DeepSleepMinutes', 'RestingHeartRate']

        # Train one model PER USER: each person's "normal" baseline differs, so a
        # per-user fit flags deviations relative to that individual, not the group.
        results = []
        for user_id, user_group in df_ml.groupby('User'):
            if len(user_group) > 10:                     # enough days for a reliable per-user model
                user_group = user_group.copy()
                model = IsolationForest(contamination=0.05, random_state=42)
                user_group['Anomaly_Score'] = model.fit_predict(user_group[features])
                user_group['Status'] = user_group['Anomaly_Score'].map({1: 'Φυσιολογικό', -1: 'Ανωμαλία'})
                results.append(user_group)

        if results:
            df_ml = pd.concat(results, ignore_index=True)
            df_display = filter_df(df_ml, user_filter)
            
            fig = px.scatter(df_display, x='DeepSleepMinutes', y='RestingHeartRate', 
                             color='Status', symbol='Status',
                             color_discrete_map={'Φυσιολογικό': '#00CC96', 'Ανωμαλία': '#EF553B'},
                             hover_data=['Date', 'User'])
            st.plotly_chart(fig, use_container_width=True)

            anomalies = df_display[df_display['Status'] == 'Ανωμαλία']
            if not anomalies.empty:
                st.error(f"Εντοπίστηκαν {len(anomalies)} ύποπτες καταγραφές ύπνου/παλμών:")
                st.dataframe(anomalies[['Date', 'User', 'DeepSleepMinutes', 'RestingHeartRate']].reset_index(drop=True), use_container_width=True)
                if st.button("Αποθήκευση Ανωμαλιών στον Γράφο Γνώσης (GraphDB)"):
                    # Persist each detected anomaly back to the graph (with provenance).
                    success_count = 0
                    for index, row in anomalies.iterrows():
                        success = insert_anomaly_to_graphdb(row['User'], row['Date'], row['Anomaly_Score'])
                        if success:
                            success_count += 1
                    
                    if success_count == len(anomalies):
                        st.success(f"Επιτυχής εγγραφή {success_count} νέων τριπλετών στο GraphDB.")
                    else:
                        st.warning("Ολοκληρώθηκε με κάποια σφάλματα. Ελέγξτε το terminal.")
            else:
                st.success("Δεν εντοπίστηκαν ανωμαλίες ύπνου για τον επιλεγμένο χρήστη.")
                
            # Real-time detection simulation
            # Lets the user test a hypothetical sleep/HR reading against the trained model.
            st.markdown("---")
            st.subheader("Προσομοίωση Νέας Μέτρησης Ύπνου (Real-time Detection)")

            if user_filter == "Όλοι":
                st.info("Επιλέξτε έναν συγκεκριμένο χρήστη από το αριστερό μενού για να δοκιμάσετε μια μέτρηση στο δικό του προφίλ.")
            else:
                df_user_sim = df_ml[df_ml['User'] == user_filter]

                # Retrain on just this user so the simulated point is judged against their own history.
                if len(df_user_sim) > 10:
                    model_sim = IsolationForest(contamination=0.05, random_state=42)
                    model_sim.fit(df_user_sim[features])

                    with st.form("simulation_form"):
                        col1, col2 = st.columns(2)
                        with col1:
                            new_sleep = st.slider("Λεπτά Βαθύ Ύπνου (Deep Sleep)", min_value=0, max_value=200, value=50)
                        with col2:
                            new_hr = st.slider("Μέσοι Παλμοί Ηρεμίας (Resting HR)", min_value=40, max_value=130, value=65)
                        submit_button = st.form_submit_button("Έλεγχος Μέτρησης Ύπνου")

                    if submit_button:
                        new_data = pd.DataFrame([[new_sleep, new_hr]], columns=features)
                        prediction = model_sim.predict(new_data)[0]
                        if prediction == -1:
                            st.error(f"**ΠΡΟΣΟΧΗ - ΑΝΩΜΑΛΙΑ!** Οι τιμές (Ύπνος: {new_sleep} λεπτά, Παλμοί: {new_hr} bpm) αποκλίνουν από το ιστορικό.")
                        else:
                            st.success("**ΦΥΣΙΟΛΟΓΙΚΗ ΜΕΤΡΗΣΗ**")
                else:
                    st.warning(f"Ο χρήστης {user_filter} δεν έχει αρκετές εγγραφές (>10) για να εκπαιδευτεί το μοντέλο προσομοίωσης.")
        else:
            st.warning("Το μοντέλο ύπνου απαιτεί περισσότερα δεδομένα (>5 εγγραφές) για να λειτουργήσει.")

    # --- SECOND DETECTOR: activity-vs-heart-rate anomalies on the time series ---
    st.markdown("---")
    st.subheader("Analyse 5.2: Ανίχνευση Στρες / Ασθένειας (Παλμοί vs Βήματα)")
    st.markdown("Αυτός ο αλγόριθμος εξετάζει τα δεδομένα υψηλής συχνότητας (time-series). Ψάχνει για χρονικές στιγμές όπου οι παλμοί είναι αδικαιολόγητα υψηλοί σε σχέση με τη φυσική δραστηριότητα.")

    if user_filter == "Όλοι":
        st.info("Παρακαλώ επιλέξτε έναν συγκεκριμένο χρήστη από το αριστερό μενού για να αναλυθούν τα δεδομένα δραστηριότητάς του.")
    else:
        df_fusion = get_user_fusion_data(user_filter)
        if not df_fusion.empty and len(df_fusion) > 10:
            # Features for this detector are the Steps/HeartRate pair per timestamp.
            fusion_features = ['Steps', 'HeartRate']
            df_fusion_ml = df_fusion.dropna(subset=fusion_features).copy()
            
            # Separate model tuned for the steps/HR relationship (lower 3% contamination).
            model_fusion = IsolationForest(contamination=0.03, random_state=42)
            df_fusion_ml['Anomaly_Score'] = model_fusion.fit_predict(df_fusion_ml[fusion_features])
            df_fusion_ml['Status'] = df_fusion_ml['Anomaly_Score'].map({1: 'Φυσιολογικό', -1: 'Ανωμαλία'})
            
            # Create interactive Scatter Plot
            fig_fusion = px.scatter(df_fusion_ml, x='Steps', y='HeartRate',
                                    color='Status', symbol='Status',
                                    color_discrete_map={'Φυσιολογικό': '#636EFA', 'Ανωμαλία': '#EF553B'},
                                    hover_data=['Time'],
                                    labels={'Steps': 'Βήματα (ανά λεπτό/ώρα)', 'HeartRate': 'Καρδιακοί Παλμοί (BPM)'},
                                    title=f"Σχέση Βημάτων και Παλμών για τον χρήστη {user_filter}")
            st.plotly_chart(fig_fusion, use_container_width=True)
            
            fusion_anomalies = df_fusion_ml[df_fusion_ml['Status'] == 'Ανωμαλία']
            if not fusion_anomalies.empty:
                st.warning(f"Εντοπίστηκαν {len(fusion_anomalies)} χρονικές στιγμές με περίεργη καρδιακή συμπεριφορά:")
                # Sort by heart rate (descending) so the most extreme moments surface first.
                fusion_anomalies_sorted = fusion_anomalies.sort_values(by='HeartRate', ascending=False)
                st.dataframe(fusion_anomalies_sorted[['Time', 'Steps', 'HeartRate']].reset_index(drop=True), use_container_width=True)
            else:
                st.success("Όλες οι καταγραφές παλμών-βημάτων βρίσκονται εντός των αναμενόμενων ορίων.")
        else:
            st.warning(f"Δεν υπάρχουν επαρκή συγχρονισμένα δεδομένα (Βήματα & Παλμοί) στο GraphDB για τον χρήστη {user_filter}.")

elif page == "6. Σημασιολογικά Συμπεράσματα (Risk)":
    st.subheader("Σημασιολογική Σύνθεση: Ημέρες Υψηλού Κινδύνου")
    st.markdown(
        "Σε αντίθεση με τους στατιστικούς ανιχνευτές (σελ. 5) που κοιτούν **ένα σήμα**, "
        "εδώ ο **γράφος συμπεραίνει νέα γνώση**: συνδυάζει μια ανιχνευμένη ανωμαλία ύπνου "
        "με **υψηλό στρες** και **κατανάλωση αλκοόλ** την ίδια μέρα, και παράγει την έννοια "
        "`ex:HighRiskDay`. Ο κανόνας τρέχει μέσα στο GraphDB ως `INSERT ... WHERE`."
    )

    # Stress threshold is user-tunable (scale 1-5, neutral = 3); a design choice, not a standard.

    stress_threshold = st.slider(
        "Κατώφλι υψηλού στρες (κλίμακα 1-5, όπου 3 = ουδέτερη κατάσταση):",
        min_value=1, max_value=5, value=4,
        help="Ως 'υψηλό' ορίζονται οι τιμές πάνω από το ουδέτερο 3. Σχεδιαστική επιλογή, όχι πρότυπο."
    )

    st.info(
        "Ο κανόνας βασίζεται στις ανωμαλίες της σελίδας 5. "
        "Αν δεν βλέπεις αποτελέσματα, πήγαινε πρώτα στη σελίδα 5 και πάτησε "
        "«Αποθήκευση Ανωμαλιών στον Γράφο Γνώσης»."
    )

    # 1) PREVIEW: show what would be labeled, without writing to the graph.
    st.markdown("#### 1. Υποψήφιες ημέρες (preview του κανόνα)")
    df_preview = preview_high_risk_days(stress_threshold)
    df_preview_show = filter_df(df_preview, user_filter) if not df_preview.empty else df_preview

    if df_preview_show.empty:
        st.write("Δεν βρέθηκαν ημέρες που να πληρούν και τις τρεις συνθήκες για την τρέχουσα επιλογή.")
    else:
        st.write(f"Βρέθηκαν **{len(df_preview_show)}** ημέρες που πληρούν τον κανόνα:")
        st.dataframe(df_preview_show.reset_index(drop=True), use_container_width=True, height=430)

    # 2) Execute the rule (reasoning + write-back to the graph).
    st.markdown("#### 2. Εκτέλεση κανόνα & εμπλουτισμός γράφου")
    if st.button("Παραγωγή & Αποθήκευση HighRiskDay στον γράφο"):
        if insert_high_risk_days(stress_threshold):
            st.success("Ο κανόνας εκτελέστηκε. Οι νέες οντότητες ex:HighRiskDay προστέθηκαν στον γράφο.")
        else:
            st.warning("Παρουσιάστηκε σφάλμα. Έλεγξε το terminal.")

    # 3) Read back the HighRiskDay entities now stored in the graph.
    st.markdown("#### 3. Ημέρες Υψηλού Κινδύνου αποθηκευμένες στον γράφο")
    df_risk = get_high_risk_days()
    if not df_risk.empty:
        df_risk_show = filter_df(df_risk, user_filter)
        if df_risk_show.empty:
            st.write("Καμία αποθηκευμένη HighRiskDay για τον επιλεγμένο χρήστη.")
        else:
            st.error(f"{len(df_risk_show)} ημέρες υψηλού κινδύνου στον γράφο:")
            st.dataframe(df_risk_show.reset_index(drop=True), use_container_width=True)
    else:
        st.write("Δεν υπάρχουν ακόμη οντότητες ex:HighRiskDay στον γράφο.")

    # Transparency: display the actual rule (useful when defending the thesis).
    with st.expander("Δες τον σημασιολογικό κανόνα (SPARQL)"):
        st.code(f"""INSERT {{
    ?riskDay a ex:HighRiskDay ;
             ex:refersToUser ?user ;
             ex:riskDate ?anomalyDate ;
             ex:derivedFromAnomaly ?anomaly ;
             ex:stressLevel ?stressVal ;
             ex:hadAlcohol true .
}}
WHERE {{
    ?anomaly a ex:AnomalyDetected ; ex:refersToUser ?user ; ex:anomalyDate ?anomalyDate .
    ?s a sosa:Observation ; sosa:observedProperty ex:Stress ;
       sosa:hasFeatureOfInterest ?user ; sosa:resultTime ?st ; sosa:hasSimpleResult ?stressVal .
    FILTER(xsd:decimal(?stressVal) >= {stress_threshold})
    ?a a sosa:Observation ; sosa:observedProperty ex:AlcoholConsumed ;
       sosa:hasFeatureOfInterest ?user ; sosa:hasSimpleResult ?alcVal .
    FILTER(LCASE(STR(?alcVal)) IN ("yes","true"))
    # Ταύτιση ημερομηνίας μέσω SUBSTR(resultTime,1,10)
}}""", language="sparql")