# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# -------------------------------------------------------------------------
# Feature definitions
# -------------------------------------------------------------------------
feature_columns = [
    "SST", "SAL", "ice_frac", "mixed_layer_depth", "heat_flux_down", "water_flux_up",
    "stress_X", "stress_Y", "currents_X", "currents_Y", "co2flux_pre"
]
feature_columns_with_time = feature_columns + ["month"]


# -------------------------------------------------------------------------
# Pretty labels for plotting
# -------------------------------------------------------------------------
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

custom_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#000000", "#d62728",
]


# -------------------------------------------------------------------------
# Helper function: load data, normalize, compute monthly means
# -------------------------------------------------------------------------
def get_monthly_mean(paths):
    """
    Reads one or more pickle files, extracts month from time column,
    applies z-score normalization (per file), and returns monthly means.
    """
    all_month_means = []

    for path in paths:
        print(f"Loading {path}")
        df = pd.read_pickle(path)

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Compute normalization statistics from the data itself
        feature_means = df[feature_columns].mean()
        feature_stds = df[feature_columns].std()

        # Extract month from time column
        df["month"] = pd.to_datetime(df["time_counter"]).dt.month

        # Keep only relevant columns
        df = df[feature_columns_with_time]

        # Z-score normalization
        df[feature_columns] = (df[feature_columns] - feature_means) / feature_stds

        # Monthly mean
        month_mean = df.groupby("month").mean()
        all_month_means.append(month_mean)

    if len(all_month_means) == 0:
        return pd.DataFrame()

    # Combine results from all input files
    return pd.concat(all_month_means).groupby("month").mean()


# -------------------------------------------------------------------------
# Experiment configuration
# -------------------------------------------------------------------------
experiments = {
    "experiment_1": {
        "label": "Simulation 1",
        "suffix": "exp_1",
    },
    "experiment_5": {
        "label": "Simulation 2",
        "suffix": "exp_5",
    },
}


# -------------------------------------------------------------------------
# Main loop: run once per experiment
# -------------------------------------------------------------------------
for experiment_name, exp_cfg in experiments.items():

    print(f"\n=== Running {experiment_name} ===")

    # ---------------------------------------------------------------------
    # Load monthly means for each region
    # ---------------------------------------------------------------------
    monthly_means_NA = get_monthly_mean([
        f"data/data_exploration/concatenated_years/zone_North_Atlantc_1958_2018_{experiment_name}.pkl"
    ])

    monthly_means_SO = get_monthly_mean([
        f"data/data_exploration/concatenated_years/zone_Southern_Ocean_1958_2018_{experiment_name}.pkl"
    ])

    # ---------------------------------------------------------------------
    # Select January and July
    # ---------------------------------------------------------------------
    df_month_North_Atlantic_january = monthly_means_NA[monthly_means_NA["month"] == 1]
    df_month_South_Ocean_january = monthly_means_SO[monthly_means_SO["month"] == 1]
    df_month_North_Atlantic_july = monthly_means_NA[monthly_means_NA["month"] == 7]
    df_month_South_Ocean_july = monthly_means_SO[monthly_means_SO["month"] == 7]    

    matrix_na_jan = df_month_North_Atlantic_january.corr()
    matrix_so_jan = df_month_South_Ocean_january.corr()
    matrix_na_jul = df_month_North_Atlantic_july.corr()
    matrix_so_jul = df_month_South_Ocean_july.corr()

    matrix_na_jan = matrix_na_jan.rename(index=pretty_labels, columns=pretty_labels)
    matrix_na_jul = matrix_na_jul.rename(index=pretty_labels, columns=pretty_labels)
    matrix_so_jan = matrix_so_jan.rename(index=pretty_labels, columns=pretty_labels)
    matrix_so_jul = matrix_so_jul.rename(index=pretty_labels, columns=pretty_labels)
    # === Plot Setup ===
    fig, axes = plt.subplots(2, 2, figsize=(18, 18))
    sns.set(font_scale=1.3)

    # === Plot ===
    sns.heatmap(matrix_na_jan, ax=axes[0, 0], cmap='coolwarm', annot=True, vmin=-1, vmax=1, 
                cbar=False, fmt=".2f")
    axes[0, 0].set_title("North Atlantic - January", fontsize=18)

    sns.heatmap(matrix_na_jul, ax=axes[0, 1], cmap='coolwarm', annot=True, vmin=-1, vmax=1, 
                cbar=False, fmt=".2f")
    axes[0, 1].set_title("North Atlantic - July", fontsize=18)

    sns.heatmap(matrix_so_jan, ax=axes[1, 0], cmap='coolwarm', annot=True, vmin=-1, vmax=1, 
                cbar=False, fmt=".2f")
    axes[1, 0].set_title("Southern Ocean - January", fontsize=18)

    sns.heatmap(matrix_so_jul, ax=axes[1, 1], cmap='coolwarm', annot=True, vmin=-1, vmax=1, 
                cbar=False, fmt=".2f")
    axes[1, 1].set_title("Southern Ocean - July", fontsize=18)

    # Make tick labels larger
    for ax in axes.flat:
        ax.tick_params(axis='both', labelsize=16)

    # Layout optimieren
    plt.tight_layout()
    plt.savefig(f"data/data_exploration/combined_correlation_matrix_{experiment_name}.png", dpi=150)
    plt.show()
