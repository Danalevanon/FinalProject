"""
Monitoring disease spread over time using cluster-based and individual infection analysis.
This script segments the season into stages, calculates weekly infection/clustering rates,
and visualizes trends at the field and global levels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter1d

### 1. Load files and planting dates ###
files_and_planting = [
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal11_2015.csv", "planting_date": "25/07/2015"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal4_2015.csv", "planting_date": "25/07/2015"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal12_2016.csv", "planting_date": "20/07/2016"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal11_2016.csv", "planting_date": "20/07/2016"},
    {"file": "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/margal7_2016.csv", "planting_date": "20/07/2016"},
]

### 2. Read and preprocess all data ###
all_data = []
plt.rcParams.update({'font.size': 16})

for item in files_and_planting:
    file_path = item["file"]
    planting_date = pd.to_datetime(item["planting_date"], dayfirst=True)
    field_name = os.path.basename(file_path).replace('.csv', '')

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Week_From_Planting'] = ((df['Date'] - planting_date).dt.days // 7)
    df['Infected'] = (df['severity'] > 0).astype(int)
    df['Field'] = field_name

    all_data.append(df)

full_data = pd.concat(all_data, ignore_index=True)

### 3. Cluster detection using DBSCAN ###
def identify_clusters(data, eps=1.5, min_samples=2):
    if data.empty:
        return pd.Series(dtype='int')
    coords = data[['X', 'Y']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    return pd.Series(clustering.labels_)

### 4. Calculate infection and cluster stats per field per date ###
per_sample_stats = []

for (field, date), group in full_data.groupby(['Field', 'Date']):
    week = group['Week_From_Planting'].iloc[0]
    total = group.shape[0]
    infected = group['Infected'].sum()
    percent_infected = (infected / total) * 100 if total > 0 else 0

    infected_points = group[group['Infected'] == 1]
    if infected_points.shape[0] >= 2:
        cluster_labels = identify_clusters(infected_points)
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels.values else 0)
    else:
        num_clusters = 0

    print(f"{field} | {date.date()} | Week {week} | % Infected: {percent_infected:.2f}% | Clusters: {num_clusters}")

    per_sample_stats.append({
        'Field': field,
        'Date': date,
        'Week_From_Planting': week,
        'percent_infected': percent_infected,
        'num_clusters': num_clusters
    })

sample_df = pd.DataFrame(per_sample_stats)

### 5. Weekly averages across all fields ###
weekly_summary = sample_df.groupby('Week_From_Planting').agg({
    'percent_infected': 'mean',
    'num_clusters': 'mean'
}).reset_index()

### 6. Plot smoothed global trends ###
fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = 'tab:orange'
smoothed_infection = gaussian_filter1d(weekly_summary['percent_infected'], sigma=1)
ax1.set_xlabel('Weeks From Planting')
ax1.set_ylabel('Percent Infected (%)', color=color1)
ax1.plot(weekly_summary['Week_From_Planting'], smoothed_infection, marker='o', color=color1, label='Smoothed % Infected')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.6)

# Seasonal phase borders
ax1.axvline(x=7, color='purple', linestyle='--', label='Early-Rising Border (Week 7)')
ax1.axvline(x=11, color='brown', linestyle='--', label='Rising-Peak Border (Week 11)')
ax1.axvline(x=17, color='gray', linestyle='--', label='Peak-Decline Border (Week 17)')

color2 = 'tab:green'
ax2 = ax1.twinx()
ax2.set_ylabel('Average Clusters', color=color2)
smoothed_clusters = gaussian_filter1d(weekly_summary['num_clusters'], sigma=1)
ax2.plot(weekly_summary['Week_From_Planting'], smoothed_clusters, marker='s', color=color2, label='Smoothed Clusters')
ax2.tick_params(axis='y', labelcolor=color2)

fig.suptitle('Smoothed % Infected and Avg Clusters (All Fields)')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
plt.show()

### 7. Plot per-field trends ###
for field in sample_df['Field'].unique():
    field_data = sample_df[sample_df['Field'] == field].groupby('Week_From_Planting').agg(
        avg_percent_infected=('percent_infected', 'mean'),
        avg_clusters=('num_clusters', 'mean')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Weeks From Planting')
    ax1.set_ylabel('Percent Infected (%)', color='tab:orange')
    smoothed_field_infection = gaussian_filter1d(field_data['avg_percent_infected'], sigma=1)
    ax1.plot(field_data['Week_From_Planting'], smoothed_field_infection, marker='o', color='tab:orange', label='Smoothed % Infected')
    ax1.tick_params(axis='y', labelcolor='tab:orange')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Seasonal phase borders
    ax1.axvline(x=7, color='purple', linestyle='--', label='Early-Emergence Border')
    ax1.axvline(x=11, color='brown', linestyle='--', label='Emergence-Peak Border')
    ax1.axvline(x=17, color='gray', linestyle='--', label='Peak-Decline Border')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Clusters', color='tab:green')
    smoothed_field_clusters = gaussian_filter1d(field_data['avg_clusters'], sigma=1)
    ax2.plot(field_data['Week_From_Planting'], smoothed_field_clusters, marker='s', color='tab:green', label='Smoothed Clusters')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.suptitle(f'Smoothed Trends – {field}')
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
    plt.show()
