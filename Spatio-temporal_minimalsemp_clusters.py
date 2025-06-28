"""
Analysis of spatio-temporal detection results:
Shows detection trends and identifies the minimal sampling rate per region and period (score ≥ 80).
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### Setup ###
sns.set_context("talk")  # Larger font size for presentation

# Load detection results
df = pd.read_csv("/Users/danalevanon/Desktop/לימודים/תואר שני/נתונים/השוואות/spatiotemporal_detection.csv")

# Define custom order for the periods
period_order = ["Early", "Rising", "Peak", "Decline"]
df["Period"] = pd.Categorical(df["Period"], categories=period_order, ordered=True)

### Line Plot: Detection Trends by Period and Region ###
grouped = df.groupby(['Period', 'Region', 'Sampling Rate'])['Mean Detection Rate'].mean().reset_index()

g = sns.FacetGrid(grouped, col="Period", col_order=period_order, col_wrap=2, height=6, sharey=True)
g.map_dataframe(sns.lineplot, x="Sampling Rate", y="Mean Detection Rate", hue="Region", marker="o")

g.set_titles(col_template="{col_name} Period")
g.set_axis_labels("Sampling Rate", "Mean Detection Rate (%)")
g.set(ylim=(0, 100))

# Add shared legend
g.add_legend(title="Region")
g._legend.set_bbox_to_anchor((1, 0.25))  # Position legend to the right
g._legend.set_frame_on(False)

plt.tight_layout()
plt.show()

### Compute Minimal Sampling Rate per Region-Period ###
results = []
grouped_df = df.groupby(['Region', 'Period'])

for (region, period), group in grouped_df:
    found = False
    for rate in sorted(group['Sampling Rate'].unique()):
        subset = group[group['Sampling Rate'] == rate]
        mean_det = subset['Mean Detection Rate'].mean()
        prop_above = (subset['Mean Detection Rate'] >= 80).mean()
        score = 0.5 * mean_det + 0.5 * (prop_above * 100)

        if score >= 80:
            results.append({
                "Region": region,
                "Period": period,
                "Min Sampling Rate (Score ≥ 80)": rate,
                "Mean Detection Rate": round(mean_det, 2),
                "% Above 80": round(prop_above, 2),
                "Combined Score": round(score, 2)
            })
            found = True
            break

    # If no rate meets the threshold, record the best available
    if not found:
        mean_det = group['Mean Detection Rate'].mean()
        prop_above = (group['Mean Detection Rate'] >= 80).mean()
        score = 0.5 * mean_det + 0.5 * (prop_above * 100)

        results.append({
            "Region": region,
            "Period": period,
            "Min Sampling Rate (Score ≥ 80)": None,
            "Mean Detection Rate": round(mean_det, 2),
            "% Above 80": round(prop_above, 2),
            "Combined Score": round(score, 2)
        })

### Bar Plot: Minimal Sampling Rate per Region-Period ###
result_df = pd.DataFrame(results)

# Prepare data for plotting
plot_df = result_df.dropna(subset=["Min Sampling Rate (Score ≥ 80)"]).copy()
plot_df["Label"] = plot_df["Region"] + " – " + plot_df["Period"]
plot_df["Period"] = pd.Categorical(plot_df["Period"], categories=period_order, ordered=True)
plot_df = plot_df.sort_values(by=["Region", "Period"])

plt.figure(figsize=(14, 10))
sns.barplot(data=plot_df, y="Label", x="Min Sampling Rate (Score ≥ 80)", palette="viridis")

plt.title("Minimal Sampling Rate by Region and Period")
plt.xlabel("Minimal Sampling Rate")
plt.ylabel("Region – Period")
plt.xlim(0, 0.35)
plt.tight_layout()
plt.show()
