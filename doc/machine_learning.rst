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
:map_filename: Path to map projection file for the ensemble, which should be in ``hagelslag/mapfiles``.
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
:condition_threshold: Probability threshold between 0 and 1