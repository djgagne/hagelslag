.. title:: Machine Learning

.. machine_learning:

Machine Learning
================

Once storm data are extracted to CSV files, Hagelslag can train and execute sets of machine learning files for
predicting the probability of hail and the hail size distribution. Machine learning modeling is performed through the
*hsforecast* program. Like *hsdata*, *hsforecast* utilizes a Python config file with many arguments in order to set
up the models and data sources.

Config Options
--------------
The config object for *hsforecast* should contain the following keys:

:ensemble_name: Name of the Ensemble forecast system. Supports "SSEF", "NCAR", and others.
:ensemble_members: List of ensemble member names.
:num_procs: Integer number of processors
:start_dates:
    Dictionary containing datetime objects associated with the start date for "train" and "forecast" modes.
:end_dates:
    Dictionary containing datetime objects associated with the end date (inclusive) for "train" and "forecast" modes.
:start_hour: First forecast hour extracted for training/evaluation.
:end_hour: Last forecast hour (inclusive).
:train_data_path: Path to directory of csv training data files.
:forecast_data_path: Path to directory of csv forecast data files.
:member_files:
    Dictionary of paths to csv files containing configuration information about each ensemble member.
    This information is used to group ensemble members into similar subsets for training.
:group_col: Column in the member csv file used to group ensemble members.
:data_format: Currently only "csv" is supported. Additional file formats supported by pandas could be added if there was interest.
:condition_model_names: List of long names for each machine learning model that predicts the probability of hail occurring.
:condition_model_objs: List of scikit-learn model objects for each probability of hail machine learning model.
:condition_input_columns: List of input variables used for probability of hail machine learning models.
:condition_output_column: Column in data files used as a binary label of whether hail is occurring or not. Should contain 1s and 0s.
:condition_threshold: Threshold on the "condition_output_column" data used to split storms into hail and no-hail events.
:size_distribution_model_names: List of long names for each machine learning model that predicts the hail size distribution parameters.
:size_distribution_model_objs:
    List of scikit-learn model objects for the size distribution hail models.
:size_distribution_input_columns: List of variable names used as input to the size distribution models.
:size_distribution_output_columns: List of output columns used to fit the size distribution model.
:size_distribution_loc: Specified value for location parameter of gamma distribution.
:load_models: Whether to load machine learning models from disk or use existing model output.
:model_path: Path to directory containing machine learning model pickle files
:metadata_columns: List of columns to be included in prediction output files
:data_json_path: Path to track data json files
:forecast_json_path: Path where track forecast files are output
:forecast_csv_path: Path where forecast csv files are output
:netcdf_path: Path where track data netCDF files are stored
:ensemble_variables: Forecast variables from ensemble system used to generate storm surrogate probabilities.
:ensemble_variable_thresholds: Dictionary where keys are ensemble variables and values are lists of thresholds for the hail forecasts.
:ml_grid_method: Currently only supports "gamma". Other methods could be added in the future.
:neighbor_condition_model: Specifies which hail condition model is used to generate neighborhood probabilities
:neighbor_radius: List of radii in grid points over which events are aggregated
:neighbor_sigma: List of Gaussian filter standard deviations that are applied to neighborhood probability fields
:ensemble_consensus_path: Path to directory where ensemble consensus netCDF files are stored.
:ensemble_data_path: Path to top level directory of ensemble model output
:model_map_file: Path to map projection file for the ensemble, which should be in ``hagelslag/mapfiles``.
:ml_grid_percentiles: List of percentiles from 1 to 99 or "mean" that are extracted from the sampled machine learning hail sizes.
:grib_path: Path to where machine learning grib2 files are output.
:single_step: Whether raw model output is stored in a single file per hour (True) or all hours are in one file (False).

Running hsforecast
------------------
*hsforecast* features four operational modes as detailed below:
-t, --train   Trains all machine learning models and saves the models to pickle files
-f, --fore    Generates forecasts from the machine learning models
-e, --ens     Generates ensemble neighborhood probabilities from machine learning and raw ensemble output
-g, --grid    Generates gridded machine learning forecasts and writes the grids to GRIB2 files.

When running the model in training mode, none of the other modes should be activated. It is recommended to train
all of the machine learning models offline and not in real-time operations. Please have all paths specified in the
config file created.

The fore, ens, and grid options can be run simultaneously to produce the ML forecasts and the resulting other products.
If you are running only the machine learning forecasts, then --fore and --grid are the only options needed.

Currently, machine learning forecasts are output to a CSV file. Older versions of hagelslag output the forecasts to
geoJSON files, but the process was very time consuming.

*hseval* also has the ability to generate coarse neighborhood probabilities for both machine learning and raw ensemble
variables.

Machine Learning Model Specification
------------------------------------
Hagelslag uses the scikit-learn machine learning model object format and can support any of the classifiers and
regressors from scikit-learn as well as custom objects following the scikit-learn conventions. Scikit-learn model objects
contain a set of keyword arguments as model hyperparameters and have fit, predict, and predict_proba
methods. If one wishes to perform a hyperparameter search as part of the fitting process, one can wrap the model
in a GridSearchCV object and provide a dictionary of hyperparameter value options to search through. All machine
learning models should be declared in the config file.