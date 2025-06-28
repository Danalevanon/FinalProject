"""
Simulates spatial- treatment-based
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


### Settings

np.random.seed(42)

# Sampling rate for each predefined region
smart_sampling_by_region = {
    'Northern Center': 0.15,
    'Northern Margins': 0.15,
    'Northern Pathways': 0.1,
    'Southern Center': 0.05,
    'Southern Margins': 0.05,
    'Southern Pathways': 0.1
}

num_simulations = 500

# Input files with planting dates
base_path = "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/חדש"
files_and_planting = [
    {"file": os.path.join(base_path, "margal11_2015_new.csv"), "planting_date": "25/07/2015"},
    {"file": os.path.join(base_path, "margal4_2015_new.csv"), "planting_date": "25/07/2015"},
    {"file": os.path.join(base_path, "margal12_2016_new.csv"), "planting_date": "20/07/2016"},
    {"file": os.path.join(base_path, "margal11_2016_new.csv"), "planting_date": "20/07/2016"},
    {"file": os.path.join(base_path, "margal7_2016_new.csv"), "planting_date": "20/07/2016"},
]


### Helper functions


# Assign region based on plant count
def classify_region(plant_coun):
    if 1 <= plant_coun <= 3000:
        return 'Southern Margins'
    elif 3001 <= plant_coun <= 10000:
        return 'Southern Center'
    elif 10001 <= plant_coun <= 13000:
        return 'Southern Pathways'
    elif 13001 <= plant_coun <= 16000:
        return 'Northern Pathways'
    elif 16001 <= plant_coun <= 23000:
        return 'Northern Center'
    elif 23001 <= plant_coun <= 26000:
        return 'Northern Margins'
    else:
        return 'Undefined'

# Return nearby plant IDs within defined neighborhood
def get_cluster_square(df, row_id, col_id, assigned, cluster_members):
    members = set()
    if row_id == 100:
        rows_to_check = [99, 100]
    elif row_id == 101:
        rows_to_check = [101, 102]
    else:
        rows_to_check = [row_id - 1, row_id, row_id + 1]

    for r in rows_to_check:
        for c in range(col_id - 3, col_id + 4):
            n = df[(df['row_id'] == r) & (df['col_id'] == c)]
            if not n.empty:
                n_id = n.iloc[0]['new_id']
                if n_id not in assigned and n_id not in cluster_members:
                    members.add(n_id)
    return members

# Run cluster detection based on spatial sampling
def run_cluster_detection(data):
    data = data.copy()
    data['cluster_id'] = np.nan
    assigned = set()
    tested_ids = set()
    cluster_counter = 1

    for region, rate in smart_sampling_by_region.items():
        region_data = data[data['Region'] == region]
        if region_data.empty:
            continue
        sampled = region_data.sample(frac=rate)
        sampled_ids = sampled['new_id'].tolist()

        for plant_id in sampled_ids:
            if plant_id in assigned:
                continue
            row = data[data['new_id'] == plant_id].iloc[0]
            tested_ids.add(plant_id)
            if row['severity'] == 0:
                assigned.add(plant_id)
                continue
            row_id, col_id = row['row_id'], row['col_id']
            cluster_members = get_cluster_square(data, row_id, col_id, assigned, set())
            cluster_members.update([plant_id])
            queue = [(row_id, col_id)]
            checked_edges = set()

            while queue:
                r_id, c_id = queue.pop(0)
                for direction in [-4, 4]:
                    edge = data[(data['row_id'] == r_id) & (data['col_id'] == c_id + direction)]
                    if not edge.empty:
                        edge = edge.iloc[0]
                        n_id = edge['new_id']
                        if (r_id, c_id + direction) not in checked_edges:
                            tested_ids.add(n_id)
                            if n_id not in assigned and n_id not in cluster_members and edge['severity'] > 0:
                                new_members = get_cluster_square(data, r_id, c_id + direction, assigned, cluster_members)
                                cluster_members.update(new_members)
                                queue.append((r_id, c_id + direction))
                            checked_edges.add((r_id, c_id + direction))

            for member in cluster_members:
                data.loc[data['new_id'] == member, 'cluster_id'] = cluster_counter
            assigned.update(cluster_members)
            cluster_counter += 1

    # Calculate detection metrics
    detected = data[(data['cluster_id'].notna()) & (data['severity'] > 0)]
    total_infected = data[data['severity'] > 0].shape[0]
    detection_rate = 100 * len(detected) / total_infected if total_infected > 0 else np.nan
    num_clusters = data['cluster_id'].dropna().nunique()
    avg_cluster_size = data['cluster_id'].dropna().groupby(data['cluster_id']).size().mean() if num_clusters > 0 else 0
    avg_infected_per_cluster = detected['cluster_id'].value_counts().mean() if num_clusters > 0 else 0

    return len(tested_ids), len(detected), total_infected, detection_rate, num_clusters, avg_cluster_size, avg_infected_per_cluster


### Run simulation


results = []

for item in files_and_planting:
    file_path = item["file"]
    planting_date = pd.to_datetime(item["planting_date"], format='%d/%m/%Y')
    field_name = os.path.basename(file_path).replace(".csv", "")

    print(f"\n📁 Running on field: {field_name}")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Week'] = ((df['Date'] - planting_date).dt.days // 7)
    df['Region'] = df['plant_coun'].apply(classify_region)

    for date_val, group in df.groupby('Date'):
        print(f"→ {date_val.strftime('%d/%m/%Y')}")
        for i in range(num_simulations):
            tested, detected, total, rate, clusters, avg_size, avg_infected = run_cluster_detection(group)
            print(f"   → Sim {i+1}: Tested={tested}, Detected={detected}, Infected={total}, Clusters={clusters}, AvgSize={avg_size:.2f}, AvgInfected={avg_infected:.2f}")
            results.append({
                'Field': field_name,
                'Date': date_val.strftime('%Y-%m-%d'),
                'Week': ((date_val - planting_date).days) // 7,
                'Simulation': i + 1,
                'Tested': tested,
                'Detected': detected,
                'Total Infected': total,
                'Detection Rate': rate,
                'Clusters Detected': clusters,
                'Avg Cluster Size': avg_size,
                'Avg Infected Per Cluster': avg_infected
            })

# Save results to file
results_df = pd.DataFrame(results)
results_df.to_csv("spatial_cluster_detection_results.csv", index=False)
print("\n Spatial detection simulation saved to spatial_cluster_detection_results.csv")
