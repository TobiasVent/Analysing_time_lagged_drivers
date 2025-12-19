# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import calendar


# -------------------------------------------------------------------------
# Settings & columns
# -------------------------------------------------------------------------
feature_columns = [
    "SST", "SAL", "ice_frac", "mixed_layer_depth", "heat_flux_down", "water_flux_up",
    "stress_X", "stress_Y", "currents_X", "currents_Y", "co2flux_pre"
]
feature_columns_with_time = feature_columns + ["month"]






# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def get_monthly_mean(paths):
    """Reads one or more pickle files, adds month column, normalizes, and returns monthly means."""
    all_month_means = []

    for path in paths:
        print(f"Loading {path}")
        df = pd.read_pickle(path)
        print(df.head())
        df = df.drop_duplicates()
        print(df.shape)
        feature_means = df[feature_columns].mean()
        feature_stds = df[feature_columns].std()
        # Extract month from time column
        df["month"] = pd.to_datetime(df["time_counter"]).dt.month

        # Keep only relevant columns
        df = df[feature_columns_with_time]

        # Z-score normalization
        df[feature_columns] = (df[feature_columns] - feature_means) / feature_stds

        # Compute monthly mean
        month_mean = df.groupby("month").mean()
        all_month_means.append(month_mean)

    # Combine all monthly means
    if len(all_month_means) == 0:
        return pd.DataFrame()
    return pd.concat(all_month_means).groupby("month").mean()


def plot_monthly(ax, data, title, colors=None):
    """
    Plots monthly mean values for each feature with dynamic color control.
    """
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H']

    lines = []
    for i, col in enumerate(feature_columns):
        line, = ax.plot(
            data.index, data[col],
            label=col,
            color=colors[i],
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            markersize=6
        )
        lines.append(line)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Month", fontsize=16)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True)

    return lines


# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------
monthly_means_NA_1 = get_monthly_mean([
    "data/data_exploration/concatenated_years/zone_North_Atlantc_1958_2018_experiment_1.pkl"
])

monthly_means_SO_1 = get_monthly_mean([
    "data/data_exploration/concatenated_years/zone_Southern_Ocean_1958_2018_experiment_1.pkl",
])

monthly_means_NA_2 = get_monthly_mean([
    "data/data_exploration/concatenated_years/zone_North_Atlantc_1958_2018_experiment_5.pkl"
])

monthly_means_SO_2 = get_monthly_mean([
    "data/data_exploration/concatenated_years/zone_Southern_Ocean_1958_2018_experiment_5.pkl",
])


# -------------------------------------------------------------------------
# (Optional) save / reload monthly means (kept as in your code, just grouped)
# -------------------------------------------------------------------------
# monthly_means_NA_1.to_pickle("/media/stu231428/1120 7818/data_exploration/monthly_means_NA_1.pkl")
# monthly_means_SO_1.to_pickle("/media/stu231428/1120 7818/data_exploration/monthly_means_SO_1.pkl")
# monthly_means_NA_2.to_pickle("/media/stu231428/1120 7818/data_exploration/monthly_means_NA_2.pkl")
# monthly_means_SO_2.to_pickle("/media/stu231428/1120 7818/data_exploration/monthly_means_SO_2.pkl")

# monthly_means_NA_1 = pd.read_pickle("/media/stu231428/1120 7818/data_exploration/monthly_means_NA_1.pkl")
# monthly_means_SO_1 = pd.read_pickle("/media/stu231428/1120 7818/data_exploration/monthly_means_SO_1.pkl")
# monthly_means_NA_2 = pd.read_pickle("/media/stu231428/1120 7818/data_exploration/monthly_means_NA_2.pkl")
# monthly_means_SO_2 = pd.read_pickle("/media/stu231428/1120 7818/data_exploration/monthly_means_SO_2.pkl")


# -------------------------------------------------------------------------
# Plot settings
# -------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
month_names = [calendar.month_abbr[i] for i in range(1, 13)]

custom_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#000000", "#d62728"
]

pretty_labels = {
    "SST": "SST",
    "SAL": "SAL",
    "ice_frac": "Ice fraction",
    "mixed_layer_depth": "Mixed layer depth",
    "heat_flux_down": "Heat flux down",
    "water_flux_up": "Water flux up",
    "stress_X": "Wind stress X",
    "stress_Y": "Wind stress Y",
    "currents_X": "Currents X",
    "currents_Y": "Currents Y",
    "co2flux_pre": r"$CO_{2}$ flux pre",
}
legend_labels = [pretty_labels[col] for col in feature_columns]


# -------------------------------------------------------------------------
# Plot panels
# -------------------------------------------------------------------------
lines = plot_monthly(axs[0, 0], monthly_means_NA_1, "North Atlantic – Simulation 1", custom_colors)
plot_monthly(axs[0, 1], monthly_means_SO_1, "Southern Ocean – Simulation 1", custom_colors)
plot_monthly(axs[1, 0], monthly_means_NA_2, "North Atlantic – Simulation 2", custom_colors)
plot_monthly(axs[1, 1], monthly_means_SO_2, "Southern Ocean – Simulation 2", custom_colors)

fig.supylabel("Normalized Value (Z-Score)", fontsize=16)

fig.legend(
    handles=lines,
    labels=legend_labels,
    loc="lower center",
    ncol=6,
    fontsize=16,
    frameon=False,
    bbox_to_anchor=(0.5, -0.03)
)

plt.tight_layout(rect=[0, 0.07, 1, 0.95])
plt.savefig(
    "/data/stu231428/Master_Thesis/data_exploration/monthly_mean/combined_monthly_mean.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
