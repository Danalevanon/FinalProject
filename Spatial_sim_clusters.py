"""
Simulation of spatial sampling  (cluster-based detection).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

### Settings ###
smart_sampling_by_region = {
    'Northern Center': 0.15,
    'Northern Margins': 0.15,
    'Northern Pathways': 0.1,
    'Southern Center': 0.05,
    'Southern Margins': 0.05,
    'Southern Pathways': 0.1
}
num_simulations = 500

files_and_planting = [
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal11_2015.csv", "planting_date": "25/07/2015"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal4_2015.csv", "planting_date": "25/07/2015"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal12_2016.csv", "planting_date": "20/07/2016"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal11_2016.csv", "planting_date": "20/07/2016"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal7_2016.csv", "planting_date": "20/07/2016"},
]

### Helper Functions ###

# Map plant_coun values to region names
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

# Apply DBSCAN clustering
def identify_clusters(data, eps=1.5, min_samples=2):
    if data.empty:
        data['cluster'] = pd.Series(dtype='int')
        return data
    coords = data[['X', 'Y']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    data['cluster'] = clustering.labels_
    return data

# Calculate detection rate from a sampled subset
def simulate_detection_from_sample(sample, full_clusters):
    if sample.empty or full_clusters.empty:
        return 0.0

    infected_sample = sample[sample['severity'] > 0].copy()
    if infected_sample.empty:
        return 0.0

    infected_sample = identify_clusters(infected_sample, eps=1.5, min_samples=1)
    merged = infected_sample.merge(
        full_clusters[['X', 'Y', 'cluster']],
        on=['X', 'Y'],
        how='left',
        suffixes=('_sampled', '_full')
    )
    merged['final_cluster'] = merged['cluster_full'].fillna(merged['cluster_sampled'])
    detected_clusters = set(merged['final_cluster'].unique()) - {-1}
    true_clusters = set(full_clusters['cluster'].unique()) - {-1}

    return (len(true_clusters & detected_clusters) / len(true_clusters)) * 100 if true_clusters else 100.0

### Run Simulation ###
all_rows = []

for item in files_and_planting:
    file_path = item["file"]
    planting_date = pd.to_datetime(item["planting_date"], format='%d/%m/%Y')
    year = planting_date.year
    field_name = file_path.split("/")[-1].replace(".csv", "")

    print(f"\n🚜 Loading file: {field_name}")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Week_From_Planting'] = ((df['Date'] - planting_date).dt.days // 7)
    df['Year'] = year
    df['Region'] = df['plant_coun'].apply(classify_region)
    df['Field'] = field_name

    for (date, field), group in df.groupby(['Date', 'Field']):
        week = group['Week_From_Planting'].iloc[0]
        infected = group[group['severity'] > 0].copy()
        full_clusters = identify_clusters(infected, eps=1.5, min_samples=2)

        if full_clusters.empty:
            continue

        detection_rates = []
        for _ in range(num_simulations):
            sampled_parts = []
            for region, rate in smart_sampling_by_region.items():
                region_data = group[group['Region'] == region]
                if not region_data.empty:
                    sampled_parts.append(region_data.sample(frac=rate))
            full_sample = pd.concat(sampled_parts)
            det_rate = simulate_detection_from_sample(full_sample, full_clusters)
            detection_rates.append(det_rate)

        mean_detection = np.mean(detection_rates)

        all_rows.append({
            'Field': field,
            'Date': date,
            'Week': week,
            'Detection Rate': mean_detection
        })

### Save and Plot ###
summary_df = pd.DataFrame(all_rows)
summary_df.to_csv("spatial_detection_only.csv", index=False)

for field in summary_df['Field'].unique():
    field_data = summary_df[summary_df['Field'] == field]
    plt.figure(figsize=(12, 4))
    plt.plot(field_data['Week'], field_data['Detection Rate'], marker='o', label='Detection Rate')
    plt.title(f"Detection Rate by Week - {field}")
    plt.xlabel("Week From Planting")
    plt.ylabel("Detection Rate (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
