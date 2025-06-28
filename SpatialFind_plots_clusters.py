"""
Analysis of spatial find sampling results (output from SpatialFind_Clusters).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### Load Data ###
file_path = "/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/השוואות/spatial_detection.csv"
df = pd.read_csv(file_path)


plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 13
})

### Boxplot: Detection Rate by Region and Sampling Rate ###
plt.figure(figsize=(14, 7))
sns.boxplot(data=df, x='Region', y='Mean Detection Rate', hue='Sampling Rate')
plt.axhline(80, color='red', linestyle='--', label='Target 80%')
plt.title('Detection Rate by Region and Sampling Rate')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sampling Rate')
plt.tight_layout()
plt.show()

### Compute Summary Metrics ###
df['Above_80'] = df['Mean Detection Rate'] >= 80

summary = (
    df.groupby(['Region', 'Sampling Rate'])
    .agg(
        Avg_Detection=('Mean Detection Rate', 'mean'),
        Total_Weeks=('Week', 'count'),
        Weeks_Above_80=('Above_80', 'sum')
    )
    .reset_index()
)

summary['% Weeks ≥ 80'] = (summary['Weeks_Above_80'] / summary['Total_Weeks']) * 100

### Weighted Score & Filtering ###
summary['Combined Score'] = 0.5 * summary['Avg_Detection'] + 0.5 * summary['% Weeks ≥ 80']
qualified = summary[summary['Combined Score'] >= 80]

### Select Minimal Sampling Rate per Region ###
min_required = qualified.sort_values('Sampling Rate').groupby('Region').first().reset_index()

# Save to file
min_required.to_csv("minimal_sampling_rate_combined_score.csv", index=False)

### Barplot: Minimal Sampling Rate per Region ###
plt.figure(figsize=(10, 5))
sns.barplot(data=min_required, x='Region', y='Sampling Rate', hue='Region', palette='viridis', legend=False)
plt.title('Minimal Sampling Rate per Region ')
plt.ylabel('Minimal Sampling Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
