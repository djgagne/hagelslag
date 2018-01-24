.. title:: Data Processing

.. data_processing:

Data Processing
===============
The Hagelslag library can read gridded model and observation data with the purpose of identifying storm objects and
tracks. Once these tracks are identified, data can be extracted from within the bounds of the tracks. All data
processing tasks are performed with the *hsdata* program. Individual data processing steps can also be imported into
a custom processing script.

hsdata Config Options
-----------------------
*hsdata* takes a config file as input. The config file should be written in valid Python and
contain a dictionary object called config. The config object should contain the following key/value pairs:

:dates: List of model run dates in datetime.datetime format
:start_hour: Starting forecast hour for data extraction
:end_hour: Ending forecast hour for data extraction
:watershed_variable: Model variable used for extracting storm objects
:ensemble_name: Name of the ensemble system being used. CAPS and NCAR are currently supported.
:ensemble_members: List of the names of the different ensemble members
:single_step: Set to True if ensemble output if each forecast hour is stored in a single file. False if all hours are in one file.
:model_path: Path to the top level of the ensemble model directory
:model_watershed_params: Tuple of configuration values for enhanced watershed (min intensity, increment, max intensity, max area, max intensity range)
:size_filter: minimum area of a storm object in number of grid points
:gaussian_window: standard deviation of Gaussian smoother applied to watershed_variable field before watershed
:mrms_path: Path to top level of MRMS data archive
:mrms_variable: Name of MRMS variable being used for observations
:mrms_watershed_params: Tuple of watershed parameters for the MRMS data
:object_matcher_params: ([distance functions], np.array([weights]), np.array([maximum distance values])),
:track_matcher_params: ([track distance functions], np.array([maximum distances])),
:storm_variables: List of variables extracted at the same forecast hour as the storm object
:potential_variables: List of variables extracted from the previous forecast hour
:tendency_variables: List of variables that have their difference between current and previous hours extracted
:shape_variables: List of variables describing shape of object. Must be property of scikit-image regionprops
:variable_statistics: List of statistics calculated for every variable, includes min, max, mean, std, and percentile_value (such as percentile_90 for the 90th percentile)
:csv_path: Path to where CSV files are output
:geojson_path: Path to where geoJSON files are output
:nc_path: Path to where netCDF patch files are output
:patch_radius: Radius of square storm patches in number of grid points
:match_steps: If True, then individual track steps are matched with all nearby observed track steps. If False, full track matching is used.
:unique_matches: If match_steps is False, then this determines whether each forecast storm is matched with only one storm or not
:closest_matches: If True, matches with closest storm. If False, Hungarian method is used to find optimal match.
:train: If True, then forecasts and observations are extracted. If False, only forecast tracks are extracted.
:label_type: Set to "gamma"
:model_map_file: Path to text file containing map projection information for the model
:mask_file: Path to netCDF file containing masking information

Model Output
------------
Hagelslag supports reading model output in netCDF format with the netCDF4-python library. NetCDF files from the
CAPS Storm-Scale Ensemble and the NCAR Ensemble are currently supported. Other model output could be supported by
creating a file handler that extends the ModelGrid class and then adding it to ModelOutput as an option.

The ModelGrid subclasses contain ways to specify the file being opened in their __init__ method. The ModelOutput class
calls the appropriate ModelGrid object, as well as loading data and map projection information.

Object Finding
--------------
Storm objects are found with the enhanced watershed method. The enhanced watershed finds local maxima in a given
2D field and then grows objects from that local maximum until they meet or exceed a specified area or intensity range
threshold.

Object Tracking
---------------
Storm objects are tracked.

Data Extraction
---------------

