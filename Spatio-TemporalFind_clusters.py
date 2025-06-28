"""
Spatio-temporal simulation for identifying minimal sampling rates by region and seasonal stage.
The simulation estimates cluster detection rates under different sampling intensities,
grouped by spatial regions and temporal periods.
"""


import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

### Settings ###
sampling_rates = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
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

# Assign region based on plant_coun
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

# Assign seasonal period based on week number
def assign_period(week):
    if week <= 7:
        return 'Early'
    elif 8 <= week <= 11:
        return 'Rising'
    elif 12 <= week <= 17:
        return 'Peak'
    else:
        return 'Decline'

# Cluster detection using DBSCAN
def identify_clusters(data, eps=1.5, min_samples=2):
    if data.empty:
        data['cluster'] = pd.Series(dtype='int')
        return data
    coords = data[['X', 'Y']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    data['cluster'] = clustering.labels_
    return data

# Run detection simulation for a specific sample
def simulate_detection(data, full_clusters, rate, field_name, region, date):
    original_clusters = set(full_clusters['cluster'].unique()) - {-1}
    n_clusters_true = len(original_clusters)

    if n_clusters_true == 0:
        print(f"\U0001F4C5 {field_name} | {region} | {date.date()} | Sampling {rate*100:.1f}% ➤ No clusters to detect.")
        return 100.0

    detection_rates = []
    for _ in range(num_simulations):
        sample = data.sample(frac=rate)
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
        rate_detected = (len(original_clusters & detected) / n_clusters_true) * 100
        detection_rates.append(rate_detected)

    mean_rate_detected = np.mean(detection_rates)

    print(f"\U0001F4C5 Detection Summary")
    print(f"  ➤ Field: {field_name}")
    print(f"  ➤ Region: {region}")
    print(f"  ➤ Date: {date.date()}")
    print(f"  ➤ Sampling rate: {rate*100:.1f}%")
    print(f"  ➤ True clusters in full data: {n_clusters_true}")
    print(f"  ➤ Mean Detection Rate over 500 runs: {mean_rate_detected:.2f}%\n")

    return mean_rate_detected

### Run Simulation ###
all_results = []

for item in files_and_planting:
    file_path = item["file"]
    planting_date = pd.to_datetime(item["planting_date"], format='%d/%m/%Y')
    field_name = file_path.split("/")[-1].replace(".csv", "")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Week'] = ((df['Date'] - planting_date).dt.days // 7)
    df['Region'] = df['plant_coun'].apply(classify_region)
    df['Period'] = df['Week'].apply(assign_period)

    for (region, period, date), group in df.groupby(['Region', 'Period', 'Date']):
        if region == 'Undefined':
            continue
        for rate in sampling_rates:
            infected = group[group['severity'] > 0].copy()
            full_clusters = identify_clusters(infected, eps=1.5, min_samples=2)

            mean_det = simulate_detection(
                data=group,
                full_clusters=full_clusters,
                rate=rate,
                field_name=field_name,
                region=region,
                date=date
            )

            all_results.append({
                'Field': field_name,
                'Region': region,
                'Period': period,
                'Date': date,
                'Sampling Rate': rate,
                'Mean Detection Rate': mean_det
            })

### Save Results ###
results_df = pd.DataFrame(all_results)
results_df.to_csv("spatiotemporal_detection_results.csv", index=False)
