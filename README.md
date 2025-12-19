# Analysing Time-Lagged Drivers of Oceanic CO₂ Flux Using Deep Learning Models for Multivariate Time Series Prediction
Autor: Tobias Vent
Email: venttobias@gmail.com
## Abstract

The Earth system model FOCI-MOPS integrates physical and biogeochemical components of the ocean to simulate key processes of the global carbon cycle, including oceanic CO₂ fluxes. However, its biogeochemical module is computationally demanding, which limits simulation efficiency. This thesis investigates whether these CO₂ fluxes can be reconstructed solely from time-lagged physical ocean parameters using deep learning methods, while maintaining reliable predictive performance.

Four models were implemented and compared: XGBoost, a Multi-Layer Perceptron (MLP), a standard Long Short-Term Memory (LSTM) network, and an Attention-LSTM incorporating both input and temporal attention mechanisms. The results demonstrate that oceanic CO₂ fluxes can be successfully reconstructed from lagged physical parameters using deep learning models. Among the tested approaches, the standard LSTM achieved the best overall performance. Additionally, explainable AI methods were applied to assess the influence of lagged physical variables on model predictions. The feature importance analysis revealed that recent time steps contribute more strongly to the predictions than variables with longer lags. Although the Attention-LSTM did not outperform the standard LSTM in accuracy, it provides improved interpretability by leveraging attention mechanisms to highlight the most relevant time steps and enhance model transparency.

This thesis presents a promising approach to accelerate climate simulations while enhancing model transparency.



## Python Environment Setup

Create and activate a virtual Python environment from the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```



## Data Exploration

As a first step in the data exploration workflow, execute:

- **`data_preprocessing_dataexploration.py`**

This script creates **regional, concatenated multi-year datasets** for:

- **North Atlantic** – `experiment_1`
- **Southern Ocean** – `experiment_1`
- **North Atlantic** – `experiment_5`
- **Southern Ocean** – `experiment_5`

It loads the yearly `.pkl` files, concatenates them over the specified
time period, and stores the results in:

- `data/data_exploration/concatenated_years/`

All subsequent data exploration and analysis scripts operate on these
concatenated datasets.

---

### Monthly Mean Seasonal Cycles

Execute:

- **`data_exploration_monthly_mean.py`**

This script computes and visualizes **monthly mean seasonal cycles** for all input
features in the **North Atlantic** and **Southern Ocean**, for both
**`experiment_1`** and **`experiment_5`**.

---

### Feature Distribution Plots

Execute:

- **`data_exploration_distribution_plot.py`**

This script generates **feature distribution plots** for the **North Atlantic**
and **Southern Ocean**, for both experiments.

---

### Correlation Matrix Analysis

Execute:

- **`data_exploration_corr_matrix_plot.py`**

This script computes and visualizes **correlation matrices** for all input features
in the **North Atlantic** and **Southern Ocean**, for both experiments.

---

## Data Preparation and Model Training

### Training and Validation Data

To generate the training and validation datasets, execute:

- **`data_preprocessing_training_validation_sets.py`**

Within this script:

- The **fraction used for spatial subsampling** can be specified directly in the function calls.
- The **temporal range** defining the training and validation periods can be configured.
- The **file paths** for the training and validation datasets must be specified in  
  **`config/data_paths.py`**.

---

### Test Data Generation

To generate the test datasets, execute:

- **`data_preprocessing_test_set.py`**

In this script:

- The **target region** and **temporal range** and **experiment_name** are specified.
- The script generates **one test dataset per selected year**, which is later used
  for reconstruction and evaluation.

---

### Hyperparameter Optimization and Model Training

Hyperparameter optimization is performed using:

- **`optuna_pipeline.py`**

The resulting optimal hyperparameters are stored in the corresponding
model configuration files.

Each model has a dedicated configuration file
(e.g. **`configs/lstm_config.py`**), in which the final hyperparameters
obtained during the thesis work are already defined.
These configuration files also specify:

- the directory where the trained model is saved,
- the location where training statistics (e.g. loss curves, metrics) are stored.

Model training is then carried out by executing:

- **`train_lstm.py`**
- **`train_attention_lstm.py`**
- **`train_mlp.py`**
- **`train_xgboost.py`**

These scripts load the predefined hyperparameters from the corresponding
configuration files and train the final models using the prepared
training and validation datasets.

---

## CO₂ Flux Reconstruction from Test Data

The reconstruction of CO₂ flux fields is performed using the generated test datasets.
For this purpose, the reconstruction scripts

- **`reconstruct_test_set_cache_lstm.py`**
- **`reconstruct_test_set_cache_xgboost.py`**
- *(and analogous scripts for other models)*

are executed.

These scripts apply the trained models to the test data and create
**year-wise cache files** containing spatially and temporally resolved
CO₂ flux information.

---

### Reconstruction Procedure

- The user specifies the **experiment** (`experiment_1` or `experiment_5`).
- The **target region** (`North_Atlantic`, `Southern_Ocean`, or `global`) is selected.
- A **start year** and **end year** define the temporal range of the reconstruction.
- For **each model** and **each year** within this range, predictions are computed independently.
- The results are stored **year-wise**, such that **one cache file is created per model and per year**.

---

### Cache Contents

Each yearly cache file contains the following variables:

- latitude  
- longitude  
- timestep  
- simulated CO₂ flux  
- reconstructed CO₂ flux  

These caches provide a standardized, model-specific and year-resolved
representation of the reconstruction results.

---

### Usage of Cached Data

The generated cache files serve as the basis for subsequent analysis steps.

---

## Available Analysis Scripts

- **`reconstruction_experiment.py`**  
  Compares reconstructed and simulated CO₂ fluxes on spatial maps.

- **`plot_difference.py`**  
  Visualizes the spatial difference between reconstructed and simulated CO₂ flux.

- **`annual_seasonal.py`**  
  Generates annual mean and seasonal mean CO₂ flux plots.  
  The experiment to be analyzed must be selected within the script.

- **`scatter_plots.py`**  
  Produces scatter plots comparing reconstructed versus simulated CO₂ flux values.

- **`feature_importance_shap.py`**  
  Computes feature importance for **MLP** and **XGBoost** models using SHAP values.

- **`feature_importance_timeshap.py`**  
  Computes feature importance for **LSTM** and **Attention LSTM** models using TimeSHAP.

- **`feature_importance_attention_scores.py`**  
  Derives feature importance from attention scores produced by the **Attention LSTM** model.
