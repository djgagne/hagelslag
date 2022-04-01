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

Adding a New Modeling System to Hagelslag
-----------------------------------------
Hagelslag contains support for multiple convection allowing model frameworks, but you may need to add support
for others in the future. Follow these instructions to do so and make your model output archive
accessible by hsdata:
1. Create a derived ModelGrid class, such as HRRRModelGrid, that inherits from the ModelGrid, GribModelGrid, or
ZarrModelGrid base class. The `__init__` method should generate a list of CAM filenames based on the NWP initial date,
requested forecast period, and ensemble member name. Then, that information should be passed to the
`super(NewModelGrid, self).__init__` call to initialize the parent class with the list of filenames
and other arguments. Assuming the model output is storred in netCDF, grib2, or zarr format, the parent
class load_data method should handle other operations from there.
2. Add your new subclass to ModelOutput.py in ModelOutput.load_data. There are a series of if statements for each CAM
type. Add yours to the list. It should resemble the following::

        elif self.ensemble_name.upper() == "NCAR":
            mg = NCARModelGrid(self.member_name,
                               self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path,
                               single_step=self.single_step)
            self.data, self.units = mg.load_data()
            mg.close()

3. Create a map projection text file. The current ones are stored in the mapfiles directory. Most of the entries are
arguments for pyproj projections except for dx, dy, sw_lon, sw_lat, ne_lon, and ne_lat. The map file will be used in hsdata
to reconstruct the CAM grid. If you do not specify the map projection correctly, the created lat-lon and x-y grids
may have a different number of grid dimensions than the data. Double check to be sure. The contents of an example
map file are below::
    proj=lcc # map projection
    a=6370000 # Major Radius of Earth in m. Note that WRF uses a spherical Earth.
    b=6370000 # Minor radius of Earth in m
    lat_0=38.33643 # Center latitude of projection
    lon_0=-101. # Center longitude of projection
    lat_1=32.0 # First standard parallel
    lat_2=46.0 # Second standard parallel
    units=m # units of x-y grid and a, b
    dx=3000 # grid spacing in x direction
    dy=3000 # grid spacing in y direction
    sw_lon=-120.81058 # lon coordinates of SW corner of grid
    sw_lat=23.159264 # lat coordinates of SW corner of grid
    ne_lon=-65.02124 # lon coordinates of NE corner of grid
    ne_lat=46.88567 # lat coordinates of NE corner of grid

Object Finding
--------------
Storm objects are found with the enhanced watershed method. The enhanced watershed finds local maxima in a given
2D field and then grows objects from that local maximum until they meet or exceed a specified area
threshold. The enhanced watershed is sensitive to a fair number of its tuning parameters, but the
most important ones are related to how the input data are discretized and the max_size threshold.

If your enhanced watershed results are suboptimal, you should consider the following changes:

* Apply a Gaussian filter to smooth the data. A standard deviation of 1 or 2 works well.
* Use a larger increment when discretizing your data.
* Change the max_size parameter. For a 3 km grid, 100 is a good starting point.
* A small max_size will cause more objects to be found, but they will be small. A larger max_size will result in fewer, larger objects.
* Setting delta to 0 or 1 can also be helpful.

If none of these tweaks are helping, then you should consider using a simpler object finding scheme, such as
scipy.ndimage.label or Hysteresis.

Once objects are extracted, they can be stored in STObjects, which are designed to contain scalar
and field metadata.

Object Tracking
---------------
Storm objects are tracked using ObjectMatcher class. Currently we support a large number of different
distances that can be used to detect centroid differences and overlap between objects at different times.

If you have created tracks from your initial object time steps, then you can use TrackMatcher to match tracks
or TrackStepMatcher to match the steps within one track to the steps in another track.

Data Extraction
---------------
STObjects can extract patches from a given ModelOutput grid and attach it to each object. This
procedure enables object-based data analysis.
