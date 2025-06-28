"""
Simulation of naive sampling using the cluster-based detection approach.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import defaultdict

### Setup ###
sampling_rates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
num_simulations = 500
np.random.seed(42)

files_and_planting = [
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal11_2015.csv", "planting_date": "25/07/2015"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal4_2015.csv", "planting_date": "25/07/2015"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal12_2016.csv", "planting_date": "20/07/2016"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal11_2016.csv", "planting_date": "20/07/2016"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal7_2016.csv", "planting_date": "20/07/2016"},
]

### Clustering ###
def identify_clusters(data, eps=1.5, min_samples=2):
    if data.empty:
        data['cluster'] = pd.Series(dtype='int')
        return data
    coords = data[['X', 'Y']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    data['cluster'] = clustering.labels_
    return data

### Weekly simulation ###
def simulate_weekly_detection(df, planting_date, rate):
    results = {}
    df['Week'] = ((df['Date'] - planting_date).dt.days // 7)
    weeks = sorted(df['Week'].unique())

    for week in weeks:
        week_data = df[df['Week'] == week]
        infected = week_data[week_data['severity'] > 0].copy()
        full_clusters = identify_clusters(infected)

        original_clusters = set(full_clusters['cluster'].unique()) - {-1}
        if len(original_clusters) == 0:
            results[week] = 100.0
            continue

        detection_rates = []
        for i in range(num_simulations):
            np.random.seed(42 + i)
            sample = week_data.sample(frac=rate)
            infected_sample = sample[sample['severity'] > 0].copy()
            infected_sample = identify_clusters(infected_sample, eps=1.5, min_samples=1)

            merged = infected_sample.merge(
                full_clusters[['X', 'Y', 'cluster']],
                on=['X', 'Y'],
                how='left',
                suffixes=('_sampled', '_full')
            )
            merged['final_cluster'] = merged['cluster_full'].fillna(merged['cluster_sampled'])
            detected = set(merged['final_cluster'].unique()) - {-1}
            rate_detected = (len(original_clusters & detected) / len(original_clusters)) * 100
            detection_rates.append(rate_detected)

        results[week] = np.mean(detection_rates)

    return results

### Run simulations ###
years_aggregated = {2015: defaultdict(list), 2016: defaultdict(list)}

for item in files_and_planting:
    file_path = item["file"]
    planting_date = pd.to_datetime(item["planting_date"], format='%d/%m/%Y')
    year = planting_date.year

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    for rate in sampling_rates:
        weekly_results = simulate_weekly_detection(df.copy(), planting_date, rate)
        for week, value in weekly_results.items():
            years_aggregated[year][(rate, week)].append(value)

### Plot results ###
for year in years_aggregated:
    plt.figure(figsize=(10, 6))
    rate_week_dict = defaultdict(list)

    for (rate, week), values in years_aggregated[year].items():
        avg = np.mean(values)
        rate_week_dict[rate].append((week, avg))

    for rate in sorted(rate_week_dict):
        sorted_data = sorted(rate_week_dict[rate])
        weeks = [w for w, _ in sorted_data]
        detections = [d for _, d in sorted_data]
        plt.plot(weeks, detections, marker='o', label=f"{int(rate * 100)}% Sampling")

    plt.title(f"Cluster Detection Rate by Sampling Rate - Season {year}", fontsize=16)
    plt.xlabel("Week From Planting", fontsize=14)
    plt.ylabel("Average Detection Rate (%)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Sampling Rate", fontsize=11)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()
