"""
Simulation of spatiotemporal treatment-based sampling
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime


### Settings

np.random.seed(42)

# Manual sampling rates per region and stage
manual_sampling = pd.DataFrame({
    'Region': [
        'Northern Center', 'Northern Center', 'Northern Center', 'Northern Center',
        'Northern Margins', 'Northern Margins', 'Northern Margins', 'Northern Margins',
        'Northern Pathways', 'Northern Pathways', 'Northern Pathways', 'Northern Pathways',
        'Southern Center', 'Southern Center', 'Southern Center', 'Southern Center',
        'Southern Margins', 'Southern Margins', 'Southern Margins', 'Southern Margins',
        'Southern Pathways', 'Southern Pathways', 'Southern Pathways', 'Southern Pathways'
    ],
    'Season Stage': [
        'Early', 'Rising', 'Peak', 'Decline',
        'Early', 'Rising', 'Peak', 'Decline',
        'Early', 'Rising', 'Peak', 'Decline',
        'Early', 'Rising', 'Peak', 'Decline',
        'Early', 'Rising', 'Peak', 'Decline',
        'Early', 'Rising', 'Peak', 'Decline'
    ],
    'Sampling Rate (%)': [
        5, 10, 15, 10,
        30, 15, 5, 15,
        10, 30, 10, 15,
        2, 2, 10, 10,
        2, 2, 10, 10,
        2, 2, 10, 35
    ]
})

# File paths and planting dates
base_path = "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/חדש"
files_and_planting = [
    {"file": os.path.join(base_path, "margal11_2015_new.csv"), "planting_date": "25/07/2015"},
    {"file": os.path.join(base_path, "margal4_2015_new.csv"), "planting_date": "25/07/2015"},
    {"file": os.path.join(base_path, "margal12_2016_new.csv"), "planting_date": "20/07/2016"},
    {"file": os.path.join(base_path, "margal11_2016_new.csv"), "planting_date": "20/07/2016"},
    {"file": os.path.join(base_path, "margal7_2016_new.csv"), "planting_date": "20/07/2016"},
]


### Helper functions


# Classify plant to region based on plant_coun
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

# Get stage from week number
def get_stage(week):
    if week <= 7:
        return 'Early'
    elif 8 <= week <= 11:
        return 'Rising'
    elif 12 <= week <= 17:
        return 'Peak'
    else:
        return 'Decline'

# Get square area (3 rows × 7 cols) around a plant
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

# Simulate sampling and cluster detection
def run_spatiotemporal_cluster_detection(data):
    data = data.copy()
    data['cluster_id'] = np.nan
    assigned = set()
    tested_ids = set()
    cluster_counter = 1
    sampled_infected_ids = set()

    for region, region_group in data.groupby('Region'):
        if region == 'Undefined':
            continue
        stage = region_group['Stage'].iloc[0]
        row = manual_sampling[(manual_sampling['Region'] == region) & (manual_sampling['Season Stage'] == stage)]
        if row.empty:
            continue
        frac = row['Sampling Rate (%)'].values[0] / 100
        sampled = region_group.sample(frac=frac)

        for _, row in sampled.iterrows():
            plant_id = row['new_id']
            if plant_id in assigned:
                continue
            tested_ids.add(plant_id)
            if row['severity'] == 0:
                assigned.add(plant_id)
                continue
            sampled_infected_ids.add(plant_id)
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

    detected = data[(data['cluster_id'].notna()) & (data['severity'] > 0)]
    total_infected = data[data['severity'] > 0].shape[0]
    detection_rate = 100 * len(detected) / total_infected if total_infected > 0 else np.nan
    num_clusters = data['cluster_id'].dropna().nunique()
    avg_cluster_size = data['cluster_id'].dropna().groupby(data['cluster_id']).size().mean() if num_clusters > 0 else 0
    avg_infected_per_cluster = detected['cluster_id'].value_counts().mean() if num_clusters > 0 else 0

    sampled_infected = data[data['new_id'].isin(sampled_infected_ids) & (data['severity'] > 0)]

    return len(tested_ids), len(detected), total_infected, detection_rate, num_clusters, avg_cluster_size, avg_infected_per_cluster, len(sampled_infected)


### Run simulation


num_simulations = 500
results = []

# Loop over all fields
for item in files_and_planting:
    file_path = item["file"]
    planting_date = pd.to_datetime(item["planting_date"], format='%d/%m/%Y')
    field_name = os.path.basename(file_path).replace(".csv", "")

    print(f"\n📁 Running on field: {field_name}")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Week'] = ((df['Date'] - planting_date).dt.days // 7)
    df['Region'] = df['plant_coun'].apply(classify_region)
    df['Stage'] = df['Week'].apply(get_stage)

    # Loop over all dates in the field
    for date_val, group in df.groupby('Date'):
        print(f"→ {date_val.strftime('%d/%m/%Y')}")
        for i in range(num_simulations):
            tested, detected, total, rate, clusters, avg_size, avg_infected, sampled_infected = run_spatiotemporal_cluster_detection(group)
            print(f"   → Sim {i+1}: Tested={tested}, Detected={detected}, Infected={total}, Sampled Infected={sampled_infected}, Clusters={clusters}, AvgSize={avg_size:.2f}, AvgInfected={avg_infected:.2f}")
            results.append({
                'Field': field_name,
                'Date': date_val.strftime('%Y-%m-%d'),
                'Week': ((date_val - planting_date).days) // 7,
                'Stage': get_stage(((date_val - planting_date).days) // 7),
                'Simulation': i + 1,
                'Tested': tested,
                'Detected': detected,
                'Total Infected': total,
                'Sampled Infected': sampled_infected,
                'Detection Rate': rate,
                'Clusters Detected': clusters,
                'Avg Cluster Size': avg_size,
                'Avg Infected Per Cluster': avg_infected
            })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("spatiotemporal_cluster_detection_results.csv", index=False)
print("\n Spatiotemporal detection simulation saved to spatiotemporal_cluster_detection_results.csv")
