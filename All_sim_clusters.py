"""
Combined simulation comparing four sampling strategies:
naive_spatial, naive_temporal, spatial, and spatio_temporal.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from datetime import datetime

np.random.seed(42)


### Input Sampling Strategies

spatial_rates = {
    'Southern Margins': 0.05,
    'Southern Center': 0.05,
    'Southern Pathways': 0.1,
    'Northern Pathways': 0.1,
    'Northern Center': 0.15,
    'Northern Margins': 0.15
}

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


### Helper Functions

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

def get_stage(week):
    if week <= 7:
        return 'Early'
    elif 8 <= week <= 11:
        return 'Rising'
    elif 12 <= week <= 17:
        return 'Peak'
    else:
        return 'Decline'

def identify_clusters(data, eps=1.5, min_samples=2):
    data = data.copy()
    if data.empty:
        data['cluster'] = pd.Series(dtype='int')
        return data
    coords = data[['X', 'Y']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    data.loc[:, 'cluster'] = clustering.labels_
    return data

def simulate_detection(full_data, sampled_data, num_simulations=500):
    full_clusters = identify_clusters(full_data[full_data['severity'] > 0], eps=1.5, min_samples=2)
    original_clusters = set(full_clusters['cluster'].unique()) - {-1}

    if len(original_clusters) == 0:
        return 100.0

    detection_rates = []
    for _ in range(num_simulations):
        infected_sample = sampled_data[sampled_data['severity'] > 0].copy()
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


### Load Data and Run Combined Simulation


files_and_planting = [
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal11_2015.csv", "planting_date": "25/07/2015"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal4_2015.csv", "planting_date": "25/07/2015"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal12_2016.csv", "planting_date": "20/07/2016"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal11_2016.csv", "planting_date": "20/07/2016"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal7_2016.csv", "planting_date": "20/07/2016"}
]

all_results = []

for item in files_and_planting:
    df = pd.read_csv(item['file'])
    planting_date = pd.to_datetime(item['planting_date'], dayfirst=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Week'] = ((df['Date'] - planting_date).dt.days // 7)
    df['Field'] = item['file'].split('/')[-1].replace('.csv', '')
    df['Region'] = df['plant_coun'].apply(classify_region)
    df['Period'] = df['Week'].apply(get_stage)

    for (date, week), group in df.groupby(['Date', 'Week']):
        full_data = group.copy()
        current_period = full_data['Period'].iloc[0]

        # Simulate spatio-temporal to get exact sampled count
        temp_sample = pd.DataFrame()
        for region, region_data in group.groupby('Region'):
            period = region_data['Period'].iloc[0]
            rate_row = manual_sampling[(manual_sampling['Region'] == region) & (manual_sampling['Season Stage'] == period)]
            if not rate_row.empty:
                rate = rate_row['Sampling Rate (%)'].values[0] / 100
                temp_sample = pd.concat([temp_sample, region_data.sample(frac=rate)], ignore_index=True)
        actual_rate = len(temp_sample) / len(group) if len(group) > 0 else 0

        for algo in ['naive_spatial', 'naive_temporal', 'spatial', 'spatio_temporal']:
            sample = pd.DataFrame()

            if algo == 'naive_spatial':
                total_rate = np.mean(list(spatial_rates.values()))
                sample = group.sample(frac=total_rate)

            elif algo == 'naive_temporal':
                sample = group.sample(frac=actual_rate)

            elif algo == 'spatial':
                for region, region_data in group.groupby('Region'):
                    rate = spatial_rates.get(region, 0)
                    if rate > 0:
                        sample = pd.concat([sample, region_data.sample(frac=rate)], ignore_index=True)

            elif algo == 'spatio_temporal':
                for region, region_data in group.groupby('Region'):
                    period = region_data['Period'].iloc[0]
                    rate_row = manual_sampling[(manual_sampling['Region'] == region) & (manual_sampling['Season Stage'] == period)]
                    if not rate_row.empty:
                        rate = rate_row['Sampling Rate (%)'].values[0] / 100
                        sample = pd.concat([sample, region_data.sample(frac=rate)], ignore_index=True)

            mean_det = simulate_detection(full_data, sample)
            all_results.append({
                'Algorithm': algo,
                'Field': group['Field'].iloc[0],
                'Week': week,
                'Date': date,
                'Sampling Rate': round(len(sample) / len(full_data), 3),
                'Mean Detection Rate': round(mean_det, 2)
            })

# Save to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv("comparison_simulation_all.csv", index=False)
print("comparison_simulation_all.csv saved.")
