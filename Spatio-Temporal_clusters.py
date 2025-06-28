"""
Spatio-temporal simulation using a manually defined sampling strategy per region and season stage.

"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

### Helper Functions ###
np.random.seed(42)

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

def identify_clusters(data, eps=1.5, min_samples=2):
    if data.empty:
        data['cluster'] = pd.Series(dtype='int')
        return data
    coords = data[['X', 'Y']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    data['cluster'] = clustering.labels_
    return data

def get_stage(week):
    if week <= 7:
        return 'Early'
    elif 8 <= week <= 11:
        return 'Rising'
    elif 12 <= week <= 17:
        return 'Peak'
    else:
        return 'Decline'

### 1. Define Manual Spatio-Temporal Sampling Strategy ###
manual_sampling = pd.DataFrame({
    'Region': [
        'Northern Center', 'Northern Center', 'Northern Center','Northern Center',
        'Northern Margins', 'Northern Margins', 'Northern Margins','Northern Margins',
        'Northern Pathways', 'Northern Pathways', 'Northern Pathways','Northern Pathways',
        'Southern Center', 'Southern Center', 'Southern Center','Southern Center',
        'Southern Margins', 'Southern Margins', 'Southern Margins','Southern Margins',
        'Southern Pathways', 'Southern Pathways', 'Southern Pathways', 'Southern Pathways'
    ],
    'Season Stage': [
        'Early', 'Rising', 'Peak','Decline',
        'Early', 'Rising', 'Peak','Decline',
        'Early', 'Rising', 'Peak','Decline',
        'Early', 'Rising', 'Peak','Decline',
        'Early', 'Rising', 'Peak','Decline',
        'Early', 'Rising', 'Peak','Decline'
    ],
    'Sampling Rate (%)': [
        5, 10, 15,10,
        30, 15, 5,15,
        10, 30, 10,15,
        2, 2, 10,10,
        2, 2, 10,10,
        2, 2, 10,35
    ]
})

### 2. Load Data and Run Detection Simulation ###
files_and_planting = [
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal11_2015.csv", "planting_date": "25/07/2015"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal4_2015.csv", "planting_date": "25/07/2015"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal12_2016.csv", "planting_date": "20/07/2016"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal11_2016.csv", "planting_date": "20/07/2016"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal7_2016.csv", "planting_date": "20/07/2016"},
]

num_simulations = 500
all_detection_results = []

for item in files_and_planting:
    file_path = item["file"]
    planting_date = pd.to_datetime(item["planting_date"], dayfirst=True)
    field_name = file_path.split("/")[-1].replace(".csv", "")

    print(f"\n🚜 Running on field: {field_name}")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Week'] = ((df['Date'] - planting_date).dt.days // 7)
    df['Region'] = df['plant_coun'].apply(classify_region)
    df['Season Stage'] = df['Week'].apply(get_stage)

    for (date, week), group in df.groupby(['Date', 'Week']):
        combined_sample = pd.DataFrame()

        for region, region_data in group.groupby('Region'):
            if region == 'Undefined':
                continue

            stage = region_data['Season Stage'].iloc[0]
            matching_row = manual_sampling[
                (manual_sampling['Region'] == region) &
                (manual_sampling['Season Stage'] == stage)
            ]
            if matching_row.empty:
                continue

            sampling_rate = matching_row['Sampling Rate (%)'].values[0] / 100
            if sampling_rate > 0:
                sampled_points = region_data.sample(frac=sampling_rate)
                combined_sample = pd.concat([combined_sample, sampled_points], ignore_index=True)

        if combined_sample.empty:
            continue

        infected_all = group[group['severity'] > 0].copy()
        full_clusters = identify_clusters(infected_all, eps=1.5, min_samples=2)
        original_clusters = set(full_clusters['cluster'].unique()) - {-1}

        if len(original_clusters) == 0:
            all_detection_results.append({
                'Field': field_name,
                'Week': week,
                'Date': date,
                'Mean Detection Rate': 100.0
            })
            continue

        detection_rates = []
        detected_cluster_counts = []

        for _ in range(num_simulations):
            infected_sample = combined_sample[combined_sample['severity'] > 0].copy()
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
            detected_cluster_counts.append(len(original_clusters & detected))

        mean_detection = np.mean(detection_rates)
        avg_detected_clusters = np.mean(detected_cluster_counts)

        all_detection_results.append({
            'Field': field_name,
            'Week': week,
            'Date': date,
            'Stage': stage,
            'Num Clusters': len(original_clusters),
            'Avg Detected Clusters': round(avg_detected_clusters, 2),
            'Sampled Plants': len(combined_sample),
            'Mean Detection Rate': round(mean_detection, 2)
        })

### 3. Summary: Detection Rate Over Time ###
detection_df = pd.DataFrame(all_detection_results)

plt.figure(figsize=(12, 6))
for field in detection_df['Field'].unique():
    field_data = detection_df[detection_df['Field'] == field]
    plt.plot(field_data['Week'], field_data['Mean Detection Rate'], marker='o', label=field)

plt.axhline(80, color='red', linestyle='--', label='Target 80%')
plt.xlabel('Week From Planting')
plt.ylabel('Detection Rate (%)')
plt.title('Detection Rate Per Field (Manual Spatio-Temporal Sampling)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

### 4. Average Detection Rate by Stage ###
def get_stage_label(week):
    if week <= 7:
        return 'Early'
    elif 8 <= week <= 11:
        return 'Rising'
    elif 12 <= week <= 17:
        return 'Peak'
    else:
        return 'Decline'

detection_df['Stage'] = detection_df['Week'].apply(get_stage_label)

avg_by_stage = (
    detection_df.groupby('Stage')['Mean Detection Rate']
    .mean()
    .reset_index()
    .rename(columns={'Mean Detection Rate': 'Avg Detection Rate'})
)

print("\n📊 Average Detection Rate by Stage:")
print(avg_by_stage)

plt.figure(figsize=(8, 5))
plt.bar(avg_by_stage['Stage'], avg_by_stage['Avg Detection Rate'], color='teal')
plt.axhline(80, color='red', linestyle='--', label='Target 80%')
plt.ylabel("Average Detection Rate (%)")
plt.title("Mean Detection Rate per Season Stage")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Save final results
detection_df.to_csv("summary_detection_with_clusters.csv", index=False)
