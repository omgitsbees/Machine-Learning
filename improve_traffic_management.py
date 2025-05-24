import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx

# --- Example Data Loading ---
# Assume traffic_df has columns: ['timestamp', 'location_id', 'speed', 'volume', 'incident', 'weather', 'event']
traffic_df = pd.read_csv('traffic_data.csv', parse_dates=['timestamp'])

# --- Feature Engineering ---
traffic_df['hour'] = traffic_df['timestamp'].dt.hour
traffic_df['dayofweek'] = traffic_df['timestamp'].dt.dayofweek

features = ['location_id', 'hour', 'dayofweek', 'weather', 'event']
target_speed = 'speed'
target_volume = 'volume'

# --- Real-time Traffic Prediction ---
X = pd.get_dummies(traffic_df[features])
y_speed = traffic_df[target_speed]
y_volume = traffic_df[target_volume]

speed_model = RandomForestRegressor(n_estimators=100)
speed_model.fit(X, y_speed)

volume_model = RandomForestRegressor(n_estimators=100)
volume_model.fit(X, y_volume)

# --- Incident Detection (Anomaly Detection) ---
anomaly_model = IsolationForest(contamination=0.01)
anomaly_features = traffic_df[['speed', 'volume']]
anomaly_model.fit(anomaly_features)
traffic_df['anomaly_score'] = anomaly_model.decision_function(anomaly_features)
traffic_df['is_incident'] = anomaly_model.predict(anomaly_features) == -1

# --- Congestion Hotspot Detection (Clustering) ---
scaler = StandardScaler()
loc_features = scaler.fit_transform(traffic_df[['speed', 'volume']])
dbscan = DBSCAN(eps=0.5, min_samples=10)
traffic_df['congestion_cluster'] = dbscan.fit_predict(loc_features)

# --- Route Optimization (Shortest/Fastest Path) ---
# Example: Build a road network graph
G = nx.DiGraph()
for _, row in traffic_df.iterrows():
    # Add edges with travel time as weight (distance/speed)
    # For demo, assume location_id encodes edges as (from, to)
    from_node, to_node = map(int, str(row['location_id']).split('-'))
    travel_time = 1.0 / max(row['speed'], 1)
    G.add_edge(from_node, to_node, weight=travel_time)

def get_fastest_route(G, source, target):
    path = nx.shortest_path(G, source=source, target=target, weight='weight')
    return path

# --- Event Impact Analysis ---
def event_impact(traffic_df, event_name):
    before = traffic_df[traffic_df['event'] != event_name]['speed'].mean()
    during = traffic_df[traffic_df['event'] == event_name]['speed'].mean()
    impact = before - during
    return impact

# --- Traffic Signal Optimization (RL-ready stub) ---
def optimize_signals(traffic_df):
    # Placeholder for RL or simulation-based optimization
    # Could use SUMO, CityFlow, or custom RL agent
    pass

# --- Example Usage ---
if __name__ == "__main__":
    # Predict speed and volume for new data
    new_data = pd.DataFrame([{'location_id': 101, 'hour': 8, 'dayofweek': 1, 'weather': 'clear', 'event': 'none'}])
    new_X = pd.get_dummies(new_data).reindex(columns=X.columns, fill_value=0)
    pred_speed = speed_model.predict(new_X)[0]
    pred_volume = volume_model.predict(new_X)[0]
    print(f"Predicted speed: {pred_speed:.2f}, Predicted volume: {pred_volume:.2f}")

    # Detect incidents
    incidents = traffic_df[traffic_df['is_incident']]
    print("Detected incidents:", incidents[['timestamp', 'location_id', 'speed', 'volume']])

    # Find congestion hotspots
    hotspots = traffic_df[traffic_df['congestion_cluster'] != -1]
    print("Congestion hotspots:", hotspots[['location_id', 'congestion_cluster']].drop_duplicates())

    # Route optimization example
    try:
        route = get_fastest_route(G, source=1, target=5)
        print("Fastest route:", route)
    except Exception as e:
        print("Route not found:", e)

    # Event impact analysis
    impact = event_impact(traffic_df, event_name='concert')
    print(f"Impact of 'concert' event on speed: {impact:.2f}")

    # Traffic signal optimization (stub)
    optimize_signals(traffic_df)