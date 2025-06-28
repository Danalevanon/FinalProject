"""
Analysis simulation of all
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load detection simulation results
df = pd.read_csv("/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/השוואות/comparison_simulation_all.csv")


# Plot settings
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12
})

# Assign season stage by week
def get_stage_label(week):
    if week <= 7:
        return 'Early'
    elif 8 <= week <= 11:
        return 'Rising'
    elif 12 <= week <= 17:
        return 'Peak'
    else:
        return 'Decline'

df['Stage'] = df['Week'].apply(get_stage_label)
stage_order = ['Early', 'Rising', 'Peak', 'Decline']
df['Stage'] = pd.Categorical(df['Stage'], categories=stage_order, ordered=True)


stage_palette = {
    'Early': '#66c2a5',
    'Rising': '#fc8d62',
    'Peak': '#8da0cb',
    'Decline': '#e78ac3'
}

# --- Barplot 1: Spatio-temporal vs Spatial improvement ---
filtered = df[df['Algorithm'].isin(['spatial', 'spatio_temporal'])]
pivot = filtered.pivot_table(index=['Field', 'Stage'], columns='Algorithm', values='Mean Detection Rate', aggfunc='mean').reset_index()
pivot['Improvement'] = pivot['spatio_temporal'] - pivot['spatial']
pivot = pivot.sort_values(by=['Field', 'Stage'])

# Calculate overall improvement by field
overall_df = filtered.pivot_table(index=['Field', 'Date'], columns='Algorithm', values='Mean Detection Rate').reset_index()
overall_df['Improvement'] = overall_df['spatio_temporal'] - overall_df['spatial']
pivot_overall = overall_df.groupby('Field')['Improvement'].mean().reset_index()
pivot_overall.rename(columns={'Improvement': 'Overall Improvement'}, inplace=True)

#  Barplot 2: Spatio-temporal vs Naive improvement
filtered_naive = df[df['Algorithm'].isin(['naive_temporal', 'spatio_temporal'])]
pivot_naive = filtered_naive.pivot_table(index=['Field', 'Stage'], columns='Algorithm', values='Mean Detection Rate', aggfunc='mean').reset_index()
pivot_naive['Improvement'] = pivot_naive['spatio_temporal'] - pivot_naive['naive_temporal']
pivot_naive = pivot_naive.sort_values(by=['Field', 'Stage'])

# Calculate overall improvement for naive
overall_naive_df = filtered_naive.pivot_table(index=['Field', 'Date'], columns='Algorithm', values='Mean Detection Rate').reset_index()
overall_naive_df['Improvement'] = overall_naive_df['spatio_temporal'] - overall_naive_df['naive_temporal']
pivot_naive_overall = overall_naive_df.groupby('Field')['Improvement'].mean().reset_index()
pivot_naive_overall.rename(columns={'Improvement': 'Overall Improvement'}, inplace=True)

# Y-axis limits for both plots
combined_improvements = pd.concat([pivot['Improvement'], pivot_naive['Improvement']])
y_min = combined_improvements.min() - 5
y_max = combined_improvements.max() + 5

#  Plot 1: Spatio-temporal vs Spatial
plt.figure(figsize=(14, 6))
sns.barplot(data=pivot, x='Field', y='Improvement', hue='Stage', palette=stage_palette)
for i, field in enumerate(pivot['Field'].unique()):
    overall = pivot_overall.loc[pivot_overall['Field'] == field, 'Overall Improvement'].values[0]
    plt.plot([i - 0.4, i + 0.4], [overall, overall], color='black', linestyle='--', linewidth=2)
    plt.text(i, overall + 1.5, f"{overall:.2f}", ha='center', color='black', fontsize=9, fontweight='bold')
plt.axhline(0, color='gray', linestyle=':')
plt.ylim(y_min, y_max)
plt.ylabel("Detection Improvement (%)")
plt.title("Spatio-Temporal vs Spatial: Detection Improvement per Field and Stage")
plt.legend(title='Season Stage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#  Plot 2: Spatio-temporal vs Naive
plt.figure(figsize=(14, 6))
sns.barplot(data=pivot_naive, x='Field', y='Improvement', hue='Stage', palette=stage_palette)
for i, field in enumerate(pivot_naive['Field'].unique()):
    overall = pivot_naive_overall.loc[pivot_naive_overall['Field'] == field, 'Overall Improvement'].values[0]
    plt.plot([i - 0.4, i + 0.4], [overall, overall], color='black', linestyle='--', linewidth=2)
    plt.text(i, overall + 1.5, f"{overall:.2f}", ha='center', color='black', fontsize=9, fontweight='bold')
plt.axhline(0, color='gray', linestyle=':')
plt.ylim(y_min, y_max)
plt.ylabel("Detection Improvement (%)")
plt.title("Spatio-Temporal vs Naive: Detection Improvement per Field and Stage")
plt.legend(title='Season Stage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Boxplot: detection rate by stage and algorithm
algorithm_labels = {
    'naive_temporal': 'Naive',
    'spatial': 'Spatial',
    'spatio_temporal': 'Spatio-Temporal'
}
df_box = df[df['Algorithm'].isin(algorithm_labels.keys())].copy()
df_box['Algorithm Label'] = df_box['Algorithm'].map(algorithm_labels)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_box, x='Stage', y='Mean Detection Rate', hue='Algorithm Label', palette='Set2')
plt.axhline(80, color='red', linestyle='--', label='Target 80%')
plt.title("Detection Rate by Season Stage – Algorithm Comparison")
plt.ylabel("Mean Detection Rate (%)")
plt.xlabel("Season Stage")
plt.legend(title='Algorithm')
plt.tight_layout()
plt.show()

# Barplot: standard deviation = detection stability
stability_df = (
    df[df['Algorithm'].isin(['naive_temporal', 'spatial', 'spatio_temporal'])]
    .groupby(['Stage', 'Algorithm'])['Mean Detection Rate']
    .std()
    .reset_index()
)
stability_df['Algorithm Label'] = stability_df['Algorithm'].map(algorithm_labels)
stability_df['Stage'] = pd.Categorical(stability_df['Stage'], categories=stage_order, ordered=True)

plt.figure(figsize=(10, 6))
sns.barplot(data=stability_df, x='Stage', y='Mean Detection Rate', hue='Algorithm Label', palette='Set2')
plt.title("Stability of Detection Rates by Algorithm and Season Stage")
plt.ylabel("Standard Deviation (%)")
plt.xlabel("Season Stage")
plt.legend(title='Algorithm')
plt.tight_layout()
plt.show()

# Barplot: average sampling rate by stage
df_sample = df[df['Algorithm'].isin(algorithm_labels.keys())].copy()
df_sample['Algorithm Label'] = df_sample['Algorithm'].map(algorithm_labels)

sampling_summary = (
    df_sample.groupby(['Stage', 'Algorithm Label'])['Sampling Rate']
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 6))
sns.barplot(data=sampling_summary, x='Stage', y='Sampling Rate', hue='Algorithm Label', palette='Set2')
plt.title("Average Sampling Rate by Algorithm and Spread Stage")
plt.ylabel("Average Sample Rate (%)")
plt.xlabel("Spread Stage")
plt.legend(title='Algorithm')
plt.tight_layout()
plt.show()
