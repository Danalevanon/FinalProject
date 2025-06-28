"""
Simulation of spatial sampling (cluster-based approach).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

### Settings ###
np.random.seed(42)
sampling_rates = [0.02, 0.05, 0.1, 0.15, 0.2]
num_simulations = 500
threshold = 80

files_and_planting = [
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal11_2015.csv", "planting_date": "25/07/2015"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal4_2015.csv", "planting_date": "25/07/2015"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal12_2016.csv", "planting_date": "20/07/2016"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal11_2016.csv", "planting_date": "20/07/2016"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal7_2016.csv", "planting_date": "20/07/2016"},
]


### Helper Functions ###

# Assign region based on plant_coun value
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


# DBSCAN clustering
def identify_clusters(data, eps=1.5, min_samples=2):
    if data.empty:
        data['cluster'] = pd.Series(dtype='int')
        return data
    coords = data[['X', 'Y']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    data['cluster'] = clustering.labels_
    return data


# Simulate sampling and compute detection rate
def simulate_detection(data, full_clusters, rate):
    original_clusters = set(full_clusters['cluster'].unique()) - {-1}
    if len(original_clusters) == 0:
        return 100.0
    detection_rates = []
    for _ in range(num_simulations):
        sample = data.sample(frac=rate)
        infected_sample = sample[sample['severity'] > 0].copy()
        infected_sample = identify_clusters(infected_sample, eps=1.5, min_samples=1)
        merged = infected_sample.merge(
            full_clusters[['X', 'Y', 'cluster']],
            on=['X', 'Y'], how='left', suffixes=('_sampled', '_full')
        )
        merged['final_cluster'] = merged['cluster_full'].fillna(merged['cluster_sampled'])
        detected = set(merged['final_cluster'].unique()) - {-1}
        rate_detected = (len(original_clusters & detected) / len(original_clusters)) * 100
        detection_rates.append(rate_detected)
    return np.mean(detection_rates)


### Run Simulation ###
all_results = []

for item in files_and_planting:
    file_path = item["file"]
    planting_date = pd.to_datetime(item["planting_date"], format='%d/%m/%Y')
    year = planting_date.year
    field_name = file_path.split("/")[-1].replace(".csv", "")
    print(f"\n Loading file: {field_name}")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Week'] = ((df['Date'] - planting_date).dt.days // 7)
    df['Year'] = year
    df['Field'] = field_name
    df['Region'] = df['plant_coun'].apply(classify_region)

    for (region, week), group in df.groupby(['Region', 'Week']):
        if region == 'Undefined':
            continue
        for rate in sampling_rates:
            infected = group[group['severity'] > 0].copy()
            full_clusters = identify_clusters(infected, eps=1.5, min_samples=2)
            mean_det = simulate_detection(group, full_clusters, rate)
            all_results.append({
                'Region': region,
                'Field': field_name,
                'Week': week,
                'Year': year,
                'Sampling Rate': rate,
                'Mean Detection Rate': mean_det,
                'Above 80': mean_det >= threshold
            })

### Save Results ###
results_df = pd.DataFrame(all_results)
results_df.to_csv("spatial_detection.csv", index=False)
print(" Simulation results saved to 'spatial_detection.csv'")
