from hagelslag.data.ModelOutput import ModelOutput
from hagelslag.data.MRMSGrid import MRMSGrid
from hagelslag.processing.EnhancedWatershedSegmenter import EnhancedWatershed, rescale_data
from hagelslag.processing.Watershed import Watershed
from hagelslag.processing.Hysteresis import Hysteresis
from hagelslag.processing.tracker import label_storm_objects, extract_storm_patches, track_storms
from .ObjectMatcher import ObjectMatcher, TrackMatcher, TrackStepMatcher
from scipy.ndimage import find_objects, gaussian_filter
from .STObject import STObject, read_geojson
import numpy as np
from scipy.interpolate import interp1d
from glob import glob
import pandas as pd
from datetime import timedelta
from scipy.stats import gamma
from netCDF4 import Dataset


class TrackProcessor(object):
    """
    TrackProcessor identifies local maxima in a convection-allowing model run and links them in time to form
    storm tracks. A similar procedure is applied to the observations, and the two sets of tracks are matched.
    Storm and environmental attributes are extracted from within the identified track areas.
    
    Args:
        run_date: Datetime model run was initialized
        start_date: Datetime for the beginning of storm extraction.
        end_date: Datetime for the ending of storm extraction.
        ensemble_name: Name of the ensemble being used.
        ensemble_member: name of the ensemble member being used.
        variable: model variable being used for extraction.
        model_path: path to the ensemble output.
        model_map_file: File containing model map projection information.
        model_watershed_params: tuple of parameters used for segmentation,
        object_matcher_params: tuple of parameters used for ObjectMatcher.
        track_matcher_params: tuple of parameters for TrackMatcher or TrackStepMatcher.
        size_filter: minimum size of model objects
        gaussian_window: number of grid points
        segmentation_approach: Select the segmentation algorithm. "ew" for enhanced watershed (default), "ws" for
            regular watershed, and "hyst" for hysteresis.
        match_steps: If True, match individual steps in tracks instead of matching whole tracks
        mrms_path: Path to MRMS netCDF files
        mrms_variable: MRMS variable being used
        mrms_watershed_params: tuple of parameters for segmentation applied to MESH data. If None, then model
            segmentation parameters are used.
        single_step: Whether model timesteps are in separate files or aggregated into one file.
        mask_file: netCDF filename containing a mask of valid grid points on the model domain.
    """
    def __init__(self,
                 run_date,
                 start_date,
                 end_date,
                 ensemble_name,
                 ensemble_member,
                 variable,
                 model_path,
                 model_map_file,
                 model_watershed_params,
                 object_matcher_params,
                 track_matcher_params,
                 size_filter,
                 gaussian_window,
                 segmentation_approach="ew",
                 match_steps=True,
                 mrms_path=None,
                 mrms_variable=None,
                 mrms_watershed_params=None,
                 single_step=True,
                 mask_file=None,
                 patch_radius=32):
        self.run_date = run_date
        self.start_date = start_date
        self.end_date = end_date
        self.start_hour = int((self.start_date - self.run_date).total_seconds()) // 3600
        self.end_hour = int((self.end_date - self.run_date).total_seconds()) // 3600
        self.hours = np.arange(int(self.start_hour), int(self.end_hour) + 1)
        self.ensemble_name = ensemble_name
        self.ensemble_member = ensemble_member
        self.variable = variable
        self.segmentation_approach = segmentation_approach
        if self.segmentation_approach == "ws":
            self.model_ew = Watershed(*model_watershed_params)
        elif self.segmentation_approach == "hyst":
            self.model_ew = Hysteresis(*model_watershed_params)
        else:
            self.model_ew = EnhancedWatershed(*model_watershed_params)
        self.object_matcher = ObjectMatcher(*object_matcher_params)
        if match_steps:
            self.track_matcher = None
            self.track_step_matcher = TrackStepMatcher(*track_matcher_params)
        else:
            self.track_matcher = TrackMatcher(*track_matcher_params)
            self.track_step_matcher = None
        self.size_filter = size_filter
        self.gaussian_window = gaussian_window
        self.model_path = model_path
        self.model_map_file = model_map_file
        self.mrms_path = mrms_path
        self.single_step = single_step
        self.model_grid = ModelOutput(self.ensemble_name, self.ensemble_member, self.run_date, self.variable,
                                      self.start_date, self.end_date, self.model_path, self.model_map_file,
                                        single_step=self.single_step)
        self.model_grid.load_map_info(self.model_map_file)
        if self.mrms_path is not None:
            self.mrms_variable = mrms_variable
            self.mrms_grid = MRMSGrid(self.start_date, self.end_date, self.mrms_variable, self.mrms_path)
            if mrms_watershed_params is None:
                mrms_watershed_params = model_watershed_params
            if self.segmentation_approach == "ws":
                self.mrms_ew = Watershed(*mrms_watershed_params)
            elif self.segmentation_approach == "hyst":
                self.mrms_ew = Hysteresis(*mrms_watershed_params)
            else:
                self.mrms_ew = EnhancedWatershed(*mrms_watershed_params)
        else:
            self.mrms_grid = None
            self.mrms_ew = None
        self.mask_file = mask_file
        self.mask = None
        if self.mask_file is not None:
            mask_data = Dataset(self.mask_file)
            self.mask = mask_data.variables["usa_mask"][:]
            mask_data.close()
        self.patch_radius = patch_radius
        return

    def find_model_patch_tracks(self):
        """
        Identify storms in gridded model output and extract uniform sized patches around the storm centers of mass.

        Returns:

        """
        self.model_grid.load_data()
        tracked_model_objects = []
        model_objects = []
        if self.model_grid.data is None:
            print("No model output found")
            return tracked_model_objects
        if self.segmentation_approach == "ew":
            min_orig = self.model_ew.min_intensity
            max_orig = self.model_ew.max_intensity
            data_increment_orig = self.model_ew.data_increment
            self.model_ew.min_intensity = 0
            self.model_ew.data_increment = 1
            self.model_ew.max_intensity = 100
        else:
            min_orig = 0
            max_orig = 1
            data_increment_orig = 1
        for h, hour in enumerate(self.hours):
            # Identify storms at each time step and apply size filter
            print("Finding {0} objects for run {1} Hour: {2:02d}".format(self.ensemble_member,
                                                                         self.run_date.strftime("%Y%m%d%H"), hour))
            if self.mask is not None:
                model_data = self.model_grid.data[h] * self.mask
            else:
                model_data = self.model_grid.data[h]
            model_data[:self.patch_radius] = 0
            model_data[-self.patch_radius:] = 0
            model_data[:, :self.patch_radius] = 0
            model_data[:, -self.patch_radius:] = 0
            if self.segmentation_approach == "ew":
                scaled_data = np.array(rescale_data(model_data, min_orig, max_orig))
                hour_labels = label_storm_objects(scaled_data, self.segmentation_approach,
                                                  self.model_ew.min_intensity, self.model_ew.max_intensity,
                                                  min_area=self.size_filter, max_area=self.model_ew.max_size,
                                                  max_range=self.model_ew.delta, increment=self.model_ew.data_increment,
                                                  gaussian_sd=self.gaussian_window)
                del scaled_data
            else:
                hour_labels = label_storm_objects(model_data, self.segmentation_approach,
                                                  self.model_ew.min_intensity, self.model_ew.max_intensity,
                                                  min_area=self.size_filter, gaussian_sd=self.gaussian_window)
            model_objects.extend(extract_storm_patches(hour_labels, model_data, self.model_grid.x,
                                                       self.model_grid.y, [hour],
                                                       dx=self.model_grid.dx,
                                                       patch_radius=self.patch_radius))
            for model_obj in model_objects[-1]:
                slices = list(find_objects(model_obj.masks[-1]))
                if len(slices) > 0:
                    dims = (slices[0][0].stop - slices[0][0].start, slices[0][1].stop - slices[0][1].start)
                    if h > 0:
                        model_obj.estimate_motion(hour, self.model_grid.data[h-1], dims[1], dims[0])

            del model_data
            del hour_labels
        tracked_model_objects.extend(track_storms(model_objects, self.hours,
                                                  self.object_matcher.cost_function_components,
                                                  self.object_matcher.max_values,
                                                  self.object_matcher.weights))
        if self.segmentation_approach == "ew":
            self.model_ew.min_intensity = min_orig
            self.model_ew.max_intensity = max_orig
            self.model_ew.data_increment = data_increment_orig
        return tracked_model_objects

    def find_model_tracks(self):
        """
        Identify storms at each model time step and link them together with object matching.

        Returns:
            List of STObjects containing model track information.
        """
        self.model_grid.load_data()
        model_objects = []
        tracked_model_objects = []
        if self.model_grid.data is None:
            print("No model output found")
            return tracked_model_objects
        for h, hour in enumerate(self.hours):
            # Identify storms at each time step and apply size filter
            print("Finding {0} objects for run {1} Hour: {2:02d}".format(self.ensemble_member,
                                                                         self.run_date.strftime("%Y%m%d%H"), hour))
            if self.mask is not None:
                model_data = self.model_grid.data[h] * self.mask
            else:
                model_data = self.model_grid.data[h]

            # remember orig values

            # scale to int 0-100.
            if self.segmentation_approach == "ew":
                min_orig = self.model_ew.min_intensity
                max_orig = self.model_ew.max_intensity
                data_increment_orig = self.model_ew.data_increment
                scaled_data = np.array(rescale_data(self.model_grid.data[h], min_orig, max_orig))
                self.model_ew.min_intensity = 0
                self.model_ew.data_increment = 1
                self.model_ew.max_intensity = 100
            else:
                min_orig = 0
                max_orig = 1
                data_increment_orig = 1
                scaled_data = self.model_grid.data[h]
            hour_labels = self.model_ew.label(gaussian_filter(scaled_data, self.gaussian_window))
            hour_labels[model_data < self.model_ew.min_intensity] = 0
            if self.size_filter > 1:
                hour_labels = self.model_ew.size_filter(hour_labels, self.size_filter)
            # Return to orig values
            if self.segmentation_approach == "ew":
                self.model_ew.min_intensity = min_orig
                self.model_ew.max_intensity = max_orig
                self.model_ew.data_increment = data_increment_orig
            obj_slices = find_objects(hour_labels)

            num_slices = len(list(obj_slices))
            model_objects.append([])
            if num_slices > 0:
                for s, sl in enumerate(obj_slices):
                    model_objects[-1].append(STObject(self.model_grid.data[h][sl],
                                                      np.where(hour_labels[sl] == s + 1, 1, 0),
                                                      self.model_grid.x[sl], 
                                                      self.model_grid.y[sl], 
                                                      self.model_grid.i[sl], 
                                                      self.model_grid.j[sl],
                                                      hour,
                                                      hour,
                                                      dx=self.model_grid.dx))
                    if h > 0:
                        dims = model_objects[-1][-1].timesteps[0].shape
                        model_objects[-1][-1].estimate_motion(hour, self.model_grid.data[h-1], dims[1], dims[0])
            del hour_labels
            del scaled_data
            del model_data
        for h, hour in enumerate(self.hours):
            past_time_objs = []
            for obj in tracked_model_objects:
                # Potential trackable objects are identified
                if obj.end_time == hour - 1:
                    past_time_objs.append(obj)
            # If no objects existed in the last time step, then consider objects in current time step all new
            if len(past_time_objs) == 0:
                tracked_model_objects.extend(model_objects[h])
            # Match from previous time step with current time step
            elif len(past_time_objs) > 0 and len(model_objects[h]) > 0:
                assignments = self.object_matcher.match_objects(past_time_objs, model_objects[h], hour - 1, hour)
                unpaired = list(range(len(model_objects[h])))
                for pair in assignments:
                    past_time_objs[pair[0]].extend(model_objects[h][pair[1]])
                    unpaired.remove(pair[1])
                if len(unpaired) > 0:
                    for up in unpaired:
                        tracked_model_objects.append(model_objects[h][up])
            print("Tracked Model Objects: {0:03d} Hour: {1:02d}".format(len(tracked_model_objects), hour))

        return tracked_model_objects

    def load_model_tracks(self, json_path):
        model_track_files = sorted(glob(json_path + "{0}/{1}/{2}_*.json".format(self.run_date.strftime("%Y%m%d"),
                                                                                self.ensemble_member,
                                                                                self.ensemble_name)))
        model_tracks = []
        for model_track_file in model_track_files:
            model_tracks.append(read_geojson(model_track_file))
        return model_tracks

    def load_mrms_tracks(self, json_path, mrms_name="mesh"):
        mrms_track_files = sorted(glob(json_path + "{0}/{1}/{2}_*.json".format(self.run_date.strftime("%Y%m%d"),
                                                                               self.ensemble_member,
                                                                               mrms_name)))
        mrms_tracks = []
        for mrms_track_file in mrms_track_files:
            mrms_tracks.append(read_geojson(mrms_track_file))
        return mrms_tracks

    def find_mrms_tracks(self):
        """
        Identify objects from MRMS timesteps and link them together with object matching.

        Returns:
            List of STObjects containing MESH track information.
        """
        obs_objects = []
        tracked_obs_objects = []
        if self.mrms_ew is not None:
            self.mrms_grid.load_data()
            
            if len(self.mrms_grid.data) != len(self.hours):
                print('Less than 24 hours of observation data found')
                
                return tracked_obs_objects
         
            for h, hour in enumerate(self.hours):
                mrms_data = np.zeros(self.mrms_grid.data[h].shape)
                mrms_data[:] = np.array(self.mrms_grid.data[h])
                mrms_data[mrms_data < 0] = 0
                hour_labels = self.mrms_ew.size_filter(self.mrms_ew.label(gaussian_filter(mrms_data,
                                                                                      self.gaussian_window)),
                                                       self.size_filter)
                hour_labels[mrms_data < self.mrms_ew.min_intensity] = 0
                obj_slices = find_objects(hour_labels)
                num_slices = len(list(obj_slices))
                obs_objects.append([])
                if num_slices > 0:
                    for sl in obj_slices:
                        obs_objects[-1].append(STObject(mrms_data[sl],
                                                        np.where(hour_labels[sl] > 0, 1, 0),
                                                        self.model_grid.x[sl],
                                                        self.model_grid.y[sl],
                                                        self.model_grid.i[sl],
                                                        self.model_grid.j[sl],
                                                        hour,
                                                        hour,
                                                        dx=self.model_grid.dx))
                        if h > 0:
                            dims = obs_objects[-1][-1].timesteps[0].shape
                            obs_objects[-1][-1].estimate_motion(hour, self.mrms_grid.data[h-1], dims[1], dims[0])
        
            for h, hour in enumerate(self.hours):
                past_time_objs = []
                for obj in tracked_obs_objects:
                    if obj.end_time == hour - 1:
                        past_time_objs.append(obj)
                if len(past_time_objs) == 0:
                    tracked_obs_objects.extend(obs_objects[h])
                elif len(past_time_objs) > 0 and len(obs_objects[h]) > 0:
                    assignments = self.object_matcher.match_objects(past_time_objs, obs_objects[h], hour - 1, hour)
                    unpaired = list(range(len(obs_objects[h])))
                    for pair in assignments:
                        past_time_objs[pair[0]].extend(obs_objects[h][pair[1]])
                        unpaired.remove(pair[1])
                    if len(unpaired) > 0:
                        for up in unpaired:
                            tracked_obs_objects.append(obs_objects[h][up])
                print("Tracked Obs Objects: {0:03d} Hour: {1:02d}".format(len(tracked_obs_objects), hour))
        
        return tracked_obs_objects

    def match_tracks(self, model_tracks, obs_tracks, unique_matches=True, closest_matches=False):
        """
        Match forecast and observed tracks.

        Args:
            model_tracks:
            obs_tracks:
            unique_matches:
            closest_matches:

        Returns:

        """
        if unique_matches:
            pairings = self.track_matcher.match_tracks(model_tracks, obs_tracks, closest_matches=closest_matches)
        else:
            pairings = self.track_matcher.neighbor_matches(model_tracks, obs_tracks)
        return pairings

    def match_track_steps(self, model_tracks, obs_tracks):
        return self.track_step_matcher.match(model_tracks, obs_tracks)

    def extract_model_attributes(self, tracked_model_objects, storm_variables, potential_variables,
                                 tendency_variables=None, future_variables=None):
        """
        Extract model attribute data for each model track. Storm variables are those that describe the model storm
        directly, such as radar reflectivity or updraft helicity. Potential variables describe the surrounding
        environmental conditions of the storm, and should be extracted from the timestep before the storm arrives to
        reduce the chance of the storm contaminating the environmental values. Examples of potential variables include
        CAPE, shear, temperature, and dewpoint. Future variables are fields that occur in the hour after the extracted
        field.

        Args:
            tracked_model_objects: List of STObjects describing each forecasted storm
            storm_variables: List of storm variable names
            potential_variables: List of potential variable names.
            tendency_variables: List of tendency variables
        """
        if tendency_variables is None:
            tendency_variables = []
        if future_variables is None:
            future_variables = []
        model_grids = {}
        for l_var in ["lon", "lat"]:
            for model_obj in tracked_model_objects:
                model_obj.extract_attribute_array(getattr(self.model_grid, l_var), l_var)
        for storm_var in storm_variables:
            print("Storm {0} {1} {2}".format(storm_var,self.ensemble_member, self.run_date.strftime("%Y%m%d")))
            model_grids[storm_var] = ModelOutput(self.ensemble_name, self.ensemble_member,
                                                 self.run_date, storm_var, self.start_date - timedelta(hours=1),
                                                 self.end_date + timedelta(hours=1),
                                                 self.model_path,self.model_map_file, 
                                                 self.single_step)
            model_grids[storm_var].load_data()
            for model_obj in tracked_model_objects:
                model_obj.extract_attribute_grid(model_grids[storm_var])
            if storm_var not in potential_variables + tendency_variables + future_variables:
                del model_grids[storm_var]
        for potential_var in potential_variables:
            print("Potential {0} {1} {2}".format(potential_var,self.ensemble_member, self.run_date.strftime("%Y%m%d")))
            if potential_var not in model_grids.keys():
                model_grids[potential_var] = ModelOutput(self.ensemble_name, self.ensemble_member,
                                                         self.run_date, potential_var,
                                                         self.start_date - timedelta(hours=1),
                                                         self.end_date + timedelta(hours=1),
                                                         self.model_path, self.model_map_file, 
                                                         self.single_step)
                model_grids[potential_var].load_data()
            for model_obj in tracked_model_objects:
                model_obj.extract_attribute_grid(model_grids[potential_var], potential=True)
            if potential_var not in tendency_variables + future_variables:
                del model_grids[potential_var]
        for future_var in future_variables:
            print("Future {0} {1} {2}".format(future_var, self.ensemble_member, self.run_date.strftime("%Y%m%d")))
            if future_var not in model_grids.keys():
                model_grids[future_var] = ModelOutput(self.ensemble_name, self.ensemble_member,
                                                         self.run_date, future_var,
                                                         self.start_date - timedelta(hours=1),
                                                         self.end_date + timedelta(hours=1),
                                                         self.model_path, self.model_map_file,
                                                         self.single_step)
                model_grids[future_var].load_data()
            for model_obj in tracked_model_objects:
                model_obj.extract_attribute_grid(model_grids[future_var], future=True)
            if future_var not in tendency_variables:
                del model_grids[future_var]
        for tendency_var in tendency_variables:
            print("Tendency {0} {1} {2}".format(tendency_var, self.ensemble_member, self.run_date.strftime("%Y%m%d")))
            if tendency_var not in model_grids.keys():
                model_grids[tendency_var] = ModelOutput(self.ensemble_name, self.ensemble_member,
                                                        self.run_date, tendency_var,
                                                        self.start_date - timedelta(hours=1),
                                                        self.end_date,
                                                        self.model_path, self.model_map_file, 
                                                        self.single_step)
            for model_obj in tracked_model_objects:
                model_obj.extract_tendency_grid(model_grids[tendency_var])
            del model_grids[tendency_var]


    @staticmethod
    def match_hail_sizes(model_tracks, obs_tracks, track_pairings):
        """
        Given forecast and observed track pairings, maximum hail sizes are associated with each paired forecast storm
        track timestep. If the duration of the forecast and observed tracks differ, then interpolation is used for the
        intermediate timesteps.

        Args:
            model_tracks: List of model track STObjects
            obs_tracks: List of observed STObjects
            track_pairings: list of tuples containing the indices of the paired (forecast, observed) tracks
        """
        unpaired = list(range(len(model_tracks)))
        for p, pair in enumerate(track_pairings):
            model_track = model_tracks[pair[0]]
            unpaired.remove(pair[0])
            obs_track = obs_tracks[pair[1]]
            obs_hail_sizes = np.array([step[obs_track.masks[t] == 1].max()
                                       for t, step in enumerate(obs_track.timesteps)])
            if obs_track.times.size > 1 and model_track.times.size > 1:
                normalized_obs_times = 1.0 / (obs_track.times.max() - obs_track.times.min())\
                    * (obs_track.times - obs_track.times.min())
                normalized_model_times = 1.0 / (model_track.times.max() - model_track.times.min())\
                    * (model_track.times - model_track.times.min())
                hail_interp = interp1d(normalized_obs_times, obs_hail_sizes, kind="nearest",
                                       bounds_error=False, fill_value=0)
                model_track.observations = hail_interp(normalized_model_times)
            elif obs_track.times.size == 1:
                model_track.observations = np.ones(model_track.times.shape) * obs_hail_sizes[0]
            elif model_track.times.size == 1:
                model_track.observations = np.array([obs_hail_sizes.max()])
            print(pair[0], "obs",  obs_hail_sizes)
            print(pair[0], "model", model_track.observations)
        for u in unpaired:
            model_tracks[u].observations = np.zeros(model_tracks[u].times.shape)

    def match_size_distributions(self, model_tracks, obs_tracks, track_pairings):
        def match_single_track_dist(model_track, obs_track):
            label_columns = ["Max_Hail_Size", "Shape", "Location", "Scale"]
            obs_hail_dists = pd.DataFrame(index=obs_track.times,
                                          columns=label_columns)
            model_hail_dists = pd.DataFrame(index=model_track.times,
                                            columns=label_columns)
            for t, step in enumerate(obs_track.timesteps):
                step_vals = step[(obs_track.masks[t] == 1) & (obs_track.timesteps[t] > self.mrms_ew.min_intensity)]
                min_hail = step_vals.min() - 0.1
                obs_hail_dists.loc[obs_track.times[t], ["Shape", "Location", "Scale"]] = gamma.fit(step_vals,
                                                                                                   floc=min_hail)
                obs_hail_dists.loc[obs_track.times[t], "Max_Hail_Size"] = step_vals.max()
            if obs_track.times.size > 1 and model_track.times.size > 1:
                normalized_obs_times = 1.0 / (obs_track.times.max() - obs_track.times.min()) \
                                       * (obs_track.times - obs_track.times.min())
                normalized_model_times = 1.0 / (model_track.times.max() - model_track.times.min()) \
                                         * (model_track.times - model_track.times.min())
                for col in label_columns:
                    interp_func = interp1d(normalized_obs_times, obs_hail_dists[col], kind="linear",
                                           bounds_error=False, fill_value=0)
                    model_hail_dists.loc[model_track.times, col] = interp_func(normalized_model_times)
            else:
                for param in obs_hail_dists.columns:
                    model_hail_dists.loc[model_track.times, param] = obs_hail_dists.loc[obs_track.times[0], param]
            return model_hail_dists
        unpaired = list(range(len(model_tracks)))
        for p, pair in enumerate(track_pairings):
            unpaired.remove(pair[0])
            if type(pair[1]) in [int, np.int64, np.int32]:
                interp_hail_dists = match_single_track_dist(model_tracks[pair[0]], obs_tracks[pair[1]])
                model_tracks[pair[0]].observations = interp_hail_dists
            else:
                model_tracks[pair[0]].observations = []
                for op in pair[1]:
                    interp_hail_dists = match_single_track_dist(model_tracks[pair[0]], obs_tracks[op])
                    model_tracks[pair[0]].observations.append(interp_hail_dists)
        return

    def match_hail_size_step_distributions(self, model_tracks, obs_tracks, track_pairings):
        """
        Given a matching set of observed tracks for each model track, 
        
        Args:
            model_tracks: 
            obs_tracks: 
            track_pairings: 

        Returns:

        """
        label_columns = ["Matched", "Max_Hail_Size", "Num_Matches", "Shape", "Location", "Scale"]
        s = 0
        for m, model_track in enumerate(model_tracks):
            model_track.observations = pd.DataFrame(index=model_track.times, columns=label_columns, dtype=np.float64)
            model_track.observations.loc[:, :] = 0
            model_track.observations["Matched"] = model_track.observations["Matched"].astype(np.int32)
            for t, time in enumerate(model_track.times):
                model_track.observations.loc[time, "Matched"] = track_pairings.loc[s, "Matched"]
                if model_track.observations.loc[time, "Matched"] > 0:
                    all_hail_sizes = []
                    step_pairs = track_pairings.loc[s, "Pairings"]
                    for step_pair in step_pairs:
                        obs_step = obs_tracks[step_pair[0]].timesteps[step_pair[1]].ravel()
                        obs_mask = obs_tracks[step_pair[0]].masks[step_pair[1]].ravel()
                        all_hail_sizes.append(obs_step[(obs_mask == 1) & (obs_step >= self.mrms_ew.min_intensity)])
                    combined_hail_sizes = np.concatenate(all_hail_sizes)
                    min_hail = combined_hail_sizes.min() - 0.1
                    model_track.observations.loc[time, "Max_Hail_Size"] = combined_hail_sizes.max()
                    model_track.observations.loc[time, "Num_Matches"] = step_pairs.shape[0]
                    model_track.observations.loc[time, ["Shape", "Location", "Scale"]] = gamma.fit(combined_hail_sizes,
                                                                                                   floc=min_hail)
                s += 1

    @staticmethod
    def calc_track_errors(model_tracks, obs_tracks, track_pairings):
        """
        Calculates spatial and temporal translation errors between matched
        forecast and observed tracks.

        Args:
            model_tracks: List of model track STObjects
            obs_tracks: List of observed track STObjects
            track_pairings: List of tuples pairing forecast and observed tracks.

        Returns:
            pandas DataFrame containing different track errors
        """
        columns = ['obs_track_id',
                   'translation_error_x',
                   'translation_error_y',
                   'start_time_difference',
                   'end_time_difference',
                   ]
        track_errors = pd.DataFrame(index=list(range(len(model_tracks))),
                                    columns=columns)
        for p, pair in enumerate(track_pairings):
            model_track = model_tracks[pair[0]]
            if type(pair[1]) in [int, np.int64]:
                obs_track = obs_tracks[pair[1]]
            else:
                obs_track = obs_tracks[pair[1][0]]
            model_com = model_track.center_of_mass(model_track.start_time)
            obs_com = obs_track.center_of_mass(obs_track.start_time)
            track_errors.loc[pair[0], 'obs_track_id'] = pair[1] if type(pair[1]) in [int, np.int64] else pair[1][0]
            track_errors.loc[pair[0], 'translation_error_x'] = model_com[0] - obs_com[0]
            track_errors.loc[pair[0], 'translation_error_y'] = model_com[1] - obs_com[1]
            track_errors.loc[pair[0], 'start_time_difference'] = model_track.start_time - obs_track.start_time
            track_errors.loc[pair[0], 'end_time_difference'] = model_track.end_time - obs_track.end_time 
        return track_errors

