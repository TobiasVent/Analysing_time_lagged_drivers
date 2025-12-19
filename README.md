## Data Preparation and Model Training




### Training and Validation Data

To generate the training and validation samples, execute:

- **`data_preprocessing_training_validation_sets.py`**

Within this script:
- The **fraction used for spatial subsampling** can be specified directly in the function calls.
- The **temporal range** defining the training and validation periods can be configured.
- The **file paths** for the training and validation datasets must be set in  
  **`config.data_paths.py`**.

---

### Test Data Generation

To generate the test datasets, open:

- **`data_preprocessing_test_set.py`**

In this script:
- The **desired temporal range** and **region** must be specified.
- The script generates **one test set per year that was chosen,**, which is later

  used for reconstruction and evaluation.

---

### Hyperparameter Optimization and Model Training

Hyperparameter optimization is performed using:

- **`optuna_pipeline.py`**

The resulting optimal hyperparameters are stored in the corresponding
model configuration files.

Model training is then carried out by executing:

- **`train_lstm.py`**
- **`train_attention_lstm.py`**
- **`train_mlp.py`**
- **`train_xgboost.py`**

These scripts train the final models using the prepared training and validation datasets.


## CO₂ Flux Reconstruction from Test Data

The reconstruction of CO₂ flux fields is performed using the generated test datasets.
For this purpose, the reconstruction scripts

- `reconstruct_test_set_cache_lstm.py`
- `reconstruct_test_set_cache_xgboost.py`
- (and analogous scripts for other models)

are executed.

These scripts apply the trained models to the test data and create **yearly cache files**
containing spatially and temporally resolved CO₂ flux information.

### Reconstruction Procedure

- The user specifies the 
-The **experiment** (`experiment_1` or `experiment_2`) 

- the **target region**(`North_Atlantic`, `Southern_Ocean`, or `global`).
- A **start year** and **end year** define the temporal range over which reconstructions are performed.
- For **each model** and **each year** within this range, model predictions are computed independently.
- The results are **stored year-wise**, such that **one cache file is created per model and per year**.

### Cache Contents

Each yearly cache file contains the following variables:

- latitude  
- longitude  
- timestep  
- simulated CO₂ flux  
- reconstructed CO₂ flux  

These caches provide a standardized, model-specific and year-resolved representation of the reconstruction results.

### Usage of Cached Data

The generated cache files serve as the basis for subsequent analysis steps, including:

- temporal aggregation (e.g. seasonal or annual means),
- spatial visualization and mapping,
- quantitative comparison between simulated and reconstructed CO₂ fluxes,
- inter-model comparisons across regions and time periods.


### Available Experiments

- **`reconstruction_experiment.py`**  
  Compares the reconstructed CO₂ flux with the simulated CO₂ flux on a spatial map.

- **`plot_difference.py`**  
  Visualizes the spatial difference between reconstructed and simulated CO₂ flux.

- **`annual_seasonal.py`**  
  Generates an annual mean plot and a seasonal mean plot of CO₂ flux.
  Here you have to select the experiment you want to analyse in the script

- **`scatter_plots.py`**  
  Produces scatter plots comparing reconstructed versus simulated CO₂ flux values.

- **`feature_importance_shap.py`**  
  Creates feature importance visualizations for **MLP** and **XGBoost** models using SHAP values.

- **`feature_importance_timeshap.py`**  
  Generates feature importance plots for **LSTM** and **Attention LSTM** models using TimeSHAP.
- **`feature_importance_attention_scores.py`**  
  Computes feature importance based on the attention scores produced by the **Attention LSTM** model.

