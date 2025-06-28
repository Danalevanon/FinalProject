# This script analyzes detection rate improvements between naive, spatial, and spatio-temporal sampling strategies.
# It compares results by field and stage, and visualizes detection rates, improvements, stability, and sampling rates.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load data ===
base_path = "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/חדש"

naive = pd.read_csv(f"{base_path}/naive_cluster_detection_stage_based.csv")
spatial = pd.read_csv(f"{base_path}/spatial_cluster_detection_results.csv")
spatiotemporal = pd.read_csv(f"{base_path}/spatiotemporal_cluster_detection_results.csv")


# === Convert Date columns ===
for df in [naive, spatial, spatiotemporal]:
    df['Date'] = pd.to_datetime(df['Date'])

# === Merge Stage from spatiotemporal to other datasets ===
stage_df = spatiotemporal[['Field', 'Date', 'Stage']].drop_duplicates()
naive = naive.merge(stage_df, on=['Field', 'Date'], how='left')
spatial = spatial.merge(stage_df, on=['Field', 'Date'], how='inner')

# === Use Stage column from the merge ===
naive['Stage'] = naive['Stage_y'] if 'Stage_y' in naive.columns else naive['Stage']
spatial['Stage'] = spatial['Stage_y'] if 'Stage_y' in spatial.columns else spatial['Stage']

# === Label algorithms ===
naive['Algorithm'] = 'Naive'
spatial['Algorithm'] = 'Spatial'
spatiotemporal['Algorithm'] = 'Spatio-Temporal'

# === Ensure 'Detection Rate' column exists ===
for df in [naive, spatial, spatiotemporal]:
    if 'Detection Rate' not in df.columns and 'Mean Detection Rate' in df.columns:
        df['Detection Rate'] = df['Mean Detection Rate']

# === Combine all data into one DataFrame ===
all_df = pd.concat([naive, spatial, spatiotemporal], ignore_index=True)

# === Format stage and field order ===
stage_order = ['Early', 'Rising', 'Peak', 'Decline']
all_df['Stage'] = pd.Categorical(all_df['Stage'], categories=stage_order, ordered=True)
field_order = sorted(all_df['Field'].unique())
stage_palette = {
    'Early': '#66c2a5',
    'Rising': '#fc8d62',
    'Peak': '#8da0cb',
    'Decline': '#e78ac3'
}

# === Filter data by algorithms and drop missing stage values ===
filtered_spatial = all_df[all_df['Algorithm'].isin(['Spatial', 'Spatio-Temporal'])].dropna(subset=['Stage'])
filtered_naive = all_df[all_df['Algorithm'].isin(['Naive', 'Spatio-Temporal'])].dropna(subset=['Stage'])

# === Create pivot tables to compare detection rates ===
pivot_spatial = filtered_spatial.pivot_table(index=['Field', 'Stage'], columns='Algorithm',
                                             values='Detection Rate', aggfunc='mean').reset_index()
pivot_naive = filtered_naive.pivot_table(index=['Field', 'Stage'], columns='Algorithm',
                                         values='Detection Rate', aggfunc='mean').reset_index()

# === Calculate improvement (Spatio-Temporal minus other) ===
pivot_spatial['Improvement'] = pivot_spatial.get('Spatio-Temporal') - pivot_spatial.get('Spatial')
pivot_naive['Improvement'] = pivot_naive.get('Spatio-Temporal') - pivot_naive.get('Naive')

# === Overall improvement per field (averaged over dates) ===
overall_spatial = filtered_spatial.pivot_table(index=['Field', 'Date'], columns='Algorithm',
                                               values='Detection Rate').reset_index()
overall_spatial['Improvement'] = overall_spatial.get('Spatio-Temporal') - overall_spatial.get('Spatial')
spatial_avg = overall_spatial.groupby('Field')['Improvement'].mean().reset_index()

overall_naive = filtered_naive.pivot_table(index=['Field', 'Date'], columns='Algorithm',
                                           values='Detection Rate').reset_index()
overall_naive['Improvement'] = overall_naive.get('Spatio-Temporal') - overall_naive.get('Naive')
naive_avg = overall_naive.groupby('Field')['Improvement'].mean().reset_index()

# === Prepare for boxplot and stability analysis ===
all_df['Algorithm Label'] = all_df['Algorithm']
boxplot_df = all_df.dropna(subset=['Stage'])
stability_df = boxplot_df.groupby(['Stage', 'Algorithm Label'])['Detection Rate'].std().reset_index()
stability_df['Stage'] = pd.Categorical(stability_df['Stage'], categories=stage_order, ordered=True)

# === Set seaborn style ===
sns.set(style="whitegrid")

# === Set common Y axis limits for bar plots ===
combined_improvement = pd.concat([
    pivot_spatial['Improvement'],
    pivot_naive['Improvement']
], ignore_index=True)
y_min = combined_improvement.min() - 5
y_max = combined_improvement.max() + 5

# === GRAPH 1: Spatio-Temporal vs Spatial per Field and Stage ===
plt.figure(figsize=(14, 6))
sns.barplot(data=pivot_spatial, x='Field', y='Improvement', hue='Stage', palette=stage_palette,
            order=field_order, hue_order=stage_order)
for i, field in enumerate(field_order):
    row = spatial_avg[spatial_avg['Field'] == field]
    if not row.empty:
        overall = row['Improvement'].values[0]
        plt.plot([i - 0.4, i + 0.4], [overall, overall], color='black', linestyle='--')
        plt.text(i, overall + 1.5, f"{overall:.2f}", ha='center', fontsize=11, fontweight='bold')
plt.axhline(0, color='gray', linestyle=':')
plt.ylim(y_min, y_max)
plt.title("Spatio-Temporal vs Spatial: Detection Improvement per Field and Stage", fontsize=16)
plt.ylabel("Detection Improvement (%)")
plt.xlabel("Field")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === GRAPH 2: Spatio-Temporal vs Naive per Field and Stage ===
plt.figure(figsize=(14, 6))
sns.barplot(data=pivot_naive, x='Field', y='Improvement', hue='Stage', palette=stage_palette,
            order=field_order, hue_order=stage_order)
for i, field in enumerate(field_order):
    row = naive_avg[naive_avg['Field'] == field]
    if not row.empty:
        overall = row['Improvement'].values[0]
        plt.plot([i - 0.4, i + 0.4], [overall, overall], color='black', linestyle='--')
        plt.text(i, overall + 1.5, f"{overall:.2f}", ha='center', fontsize=11, fontweight='bold')
plt.axhline(0, color='gray', linestyle=':')
plt.ylim(y_min, y_max)
plt.title("Spatio-Temporal vs Naive: Detection Improvement per Field and Stage", fontsize=16)
plt.ylabel("Detection Improvement (%)")
plt.xlabel("Field")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === GRAPH 3: Boxplot of Detection Rate by Stage ===
plt.figure(figsize=(10, 6))
sns.boxplot(data=boxplot_df, x='Stage', y='Detection Rate', hue='Algorithm Label', palette='Set2', order=stage_order)
plt.axhline(80, color='red', linestyle='--', label='Target 80%')
plt.title("Detection Rate by Season Stage – Algorithm Comparison", fontsize=16)
plt.ylabel("Mean Detection Rate (%)")
plt.xlabel("Season Stage")
plt.legend(title='Algorithm')
plt.tight_layout()
plt.show()

# === GRAPH 4: Stability (Standard Deviation) of Detection Rate ===
plt.figure(figsize=(10, 6))
sns.barplot(data=stability_df, x='Stage', y='Detection Rate', hue='Algorithm Label', palette='Set2', order=stage_order)
plt.title("Stability of Detection Rates by Algorithm and Season Stage", fontsize=16)
plt.ylabel("Standard Deviation (%)")
plt.xlabel("Season Stage")
plt.legend(title='Algorithm')
plt.tight_layout()
plt.show()

# === GRAPH 5: Average Sampling Rate per Stage per Algorithm ===
sample_rate_df = all_df.copy()
sample_rate_df = sample_rate_df.dropna(subset=['Stage', 'Tested'])
sample_rate_df['Sampling Rate'] = sample_rate_df['Tested'] / 26000

sampling_stage_avg = sample_rate_df.groupby(['Stage', 'Algorithm'])['Sampling Rate'].mean().reset_index()
sampling_stage_avg['Stage'] = pd.Categorical(sampling_stage_avg['Stage'], categories=stage_order, ordered=True)

plt.figure(figsize=(10, 6))
sns.barplot(data=sampling_stage_avg, x='Stage', y='Sampling Rate', hue='Algorithm', palette='Set2', order=stage_order)
plt.title("Average Sampling Rate per Stage and Algorithm (Treatment-Oriented)", fontsize=16)
plt.ylabel("Average Sampling Rate")
plt.xlabel("Season Stage")
plt.ylim(0, 0.15)
plt.legend(title='Algorithm')
plt.tight_layout()
plt.show()
