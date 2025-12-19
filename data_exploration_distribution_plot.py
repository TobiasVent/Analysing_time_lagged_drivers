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

    # Rename columns for plotting
    df_na_jan_plot = df_month_North_Atlantic_january.rename(columns=pretty_labels)
    df_na_jul_plot = df_month_North_Atlantic_july.rename(columns=pretty_labels)
    df_so_jan_plot = df_month_South_Ocean_january.rename(columns=pretty_labels)
    df_so_jul_plot = df_month_South_Ocean_july.rename(columns=pretty_labels)

    # ---------------------------------------------------------------------
    # Plot KDE distributions
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    sns.set(font_scale=1.2)

    # North Atlantic
    sns.kdeplot(df_na_jan_plot, palette=custom_colors, fill=True,
                ax=axes[0, 0], clip=(-4, 4))
    axes[0, 0].set_title(f"North Atlantic – January ({exp_cfg['label']})", fontsize=18)

    sns.kdeplot(df_na_jul_plot, palette=custom_colors, fill=True,
                ax=axes[0, 1], clip=(-4, 4))
    axes[0, 1].set_title(f"North Atlantic – July ({exp_cfg['label']})", fontsize=18)

    # Southern Ocean
    sns.kdeplot(df_so_jan_plot, palette=custom_colors, fill=True,
                ax=axes[1, 0], clip=(-4, 4))
    axes[1, 0].set_title(f"Southern Ocean – January ({exp_cfg['label']})", fontsize=18)

    sns.kdeplot(df_so_jul_plot, palette=custom_colors, fill=True,
                ax=axes[1, 1], clip=(-4, 4))
    axes[1, 1].set_title(f"Southern Ocean – July ({exp_cfg['label']})", fontsize=18)

    # ---------------------------------------------------------------------
    # Axis formatting
    # ---------------------------------------------------------------------
    for ax in axes.flat:
        ax.set_ylim(0, 0.6)
        ax.set_xlabel("Normalized Value (Z-Score)", fontsize=16)
        ax.set_ylabel("Density", fontsize=16)
        ax.grid(True)
        ax.tick_params(axis="both", labelsize=16)

    # ---------------------------------------------------------------------
    # Save and show figure
    # ---------------------------------------------------------------------
    plt.tight_layout()
    output_path = (
        f"data/data_exploration/combined_displots_monthly_mean_{experiment_name}.png"
    )
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"Saved figure: {output_path}")
