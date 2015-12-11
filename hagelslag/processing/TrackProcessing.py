from hagelslag.data.ModelOutput import ModelOutput
from hagelslag.data.MRMSGrid import MRMSGrid
from EnhancedWatershedSegmenter import EnhancedWatershed
from ObjectMatcher import ObjectMatcher, TrackMatcher
from scipy.ndimage import find_objects, gaussian_filter
from STObject import STObject, read_geojson
import numpy as np
from scipy.interpolate import interp1d
from glob import glob
import pandas as pd
from datetime import timedelta
from scipy.stats import gamma


class TrackProcessor(object):
    """
    TrackProcessor identifies local maxima in a convection-allowing model run and links them in time to form
    storm tracks. A similar procedure is applied to the observations, and the two sets of tracks are matched.
    Storm and environmental attributes are extracted from within the identified track areas.

    :param run_date: Datetime model run was initialized
    :param start_date: Datetime for the beginning of storm extraction.
    :param end_date: Datetime for the ending of storm extraction.
    :param ensemble_name: Name of the ensemble being used.
    :param ensemble_member: name of the ensemble member being used.
    :param variable: model variable being used for extraction.
    :param model_path: path to the ensemble output.
    :param model_map_file: File containing model map projection information.
    :param model_watershed_params: tuple of parameters used for EnhancedWatershed
    :param object_matcher_params: tuple of parameters used for ObjectMatcher.
    :param track_matcher_params: tuple of parameters for TrackMatcher.
    :param size_filter: minimum size of model objects
    :param gaussian_window: number of grid points
    :param mrms_path: Path to MRMS netCDF files
    :param mrms_variable: MRMS variable being used
    :param mrms_watershed_params: tuple of parameters for Enhanced Watershed applied to MESH data.
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
                 mrms_path=None,
                 mrms_variable=None,
                 mrms_watershed_params=None,
                 single_step=True):
        self.run_date = run_date
        self.start_date = start_date
        self.end_date = end_date
        self.start_hour = int((self.start_date - self.run_date).total_seconds()) / 3600
        self.end_hour = int((self.end_date - self.run_date).total_seconds()) / 3600
        self.hours = range(self.start_hour, self.end_hour + 1)
        self.ensemble_name = ensemble_name
        self.ensemble_member = ensemble_member
        self.variable = variable
        self.model_ew = EnhancedWatershed(*model_watershed_params)
        self.object_matcher = ObjectMatcher(*object_matcher_params)
        self.track_matcher = TrackMatcher(*track_matcher_params)
        self.size_filter = size_filter
        self.gaussian_window = gaussian_window
        self.model_path = model_path
        self.mrms_path = mrms_path
        self.single_step = single_step
        self.model_grid = ModelOutput(self.ensemble_name, self.ensemble_member, self.run_date, self.variable,
                                      self.start_date, self.end_date, self.model_path, single_step=self.single_step)
        self.model_grid.load_data()
        self.model_grid.load_map_info(model_map_file)
        if self.mrms_path is not None:
            self.mrms_variable = mrms_variable
            self.mrms_grid = MRMSGrid(self.start_date, self.end_date, self.mrms_variable, self.mrms_path)
            self.mrms_grid.load_data()
            self.mrms_ew = EnhancedWatershed(*mrms_watershed_params)
        else:
            self.mrms_grid = None
            self.mrms_ew = None
        return
    
    def find_model_tracks(self):
        """
        Identify storms at each model time step and link them together with object matching.

        :return: list of STObjects containing model track information.
        """
        model_objects = []
        tracked_model_objects = []
        for h, hour in enumerate(self.hours):
            # Identify storms at each time step and apply size filter
            print("Finding {0} objects for run {1} Hour: {2:02d}".format(self.ensemble_member, self.run_date.strftime("%Y%m%d%H"), hour))
            hour_labels = self.model_ew.size_filter(self.model_ew.label(gaussian_filter(self.model_grid.data[h],
                                                                                        self.gaussian_window)), 
                                                    self.size_filter)
            hour_labels[self.model_grid.data[h] < self.model_ew.min_thresh] = 0
            obj_slices = find_objects(hour_labels)
            num_slices = len(obj_slices)
            model_objects.append([])
            if num_slices > 0:
                for sl in obj_slices:   
                    model_objects[-1].append(STObject(self.model_grid.data[h][sl],
                                                      np.where(hour_labels[sl] > 0, 1, 0),
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
                unpaired = range(len(model_objects[h]))
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

        :return: list of STObjects containing MESH track information.
        """
        obs_objects = []
        tracked_obs_objects = []
        if self.mrms_ew is not None:
            for h, hour in enumerate(self.hours):
                mrms_data = np.zeros(self.mrms_grid.data[h].shape)
                mrms_data[:] = np.array(self.mrms_grid.data[h])
                mrms_data[mrms_data < 0] = 0
                hour_labels = self.mrms_ew.size_filter(self.mrms_ew.label(gaussian_filter(mrms_data,
                                                                                          self.gaussian_window)),
                                                       self.size_filter)
                hour_labels[mrms_data < self.mrms_ew.min_thresh] = 0
                obj_slices = find_objects(hour_labels)
                num_slices = len(obj_slices)
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
                    unpaired = range(len(obs_objects[h]))
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

    def extract_model_attributes(self, tracked_model_objects, storm_variables, potential_variables,
                                 tendency_variables=None):
        """
        Extract model attribute data for each model track. Storm variables are those that describe the model storm
        directly, such as radar reflectivity or updraft helicity. Potential variables describe the surrounding
        environmental conditions of the storm, and should be extracted from the timestep before the storm arrives to
        reduce the chance of the storm contaminating the environmental values. Examples of potential variables include
        CAPE, shear, temperature, and dewpoint.

        :param tracked_model_objects: List of STObjects describing each forecasted storm
        :param storm_variables: List of storm variable names
        :param potential_variables: List of potential variable names.
        :param tendency_variables: List of tendency variables
        """
        model_grids = {}
        for storm_var in storm_variables:
            print("Storm {0} {1} {2}".format(storm_var,self.ensemble_member, self.run_date.strftime("%Y%m%d")))
            model_grids[storm_var] = ModelOutput(self.ensemble_name, self.ensemble_member,
                                                 self.run_date, storm_var, self.start_date - timedelta(hours=1),
                                                 self.end_date,
                                                 self.model_path, self.single_step)
            model_grids[storm_var].load_data()
            for model_obj in tracked_model_objects:
                model_obj.extract_attribute_grid(model_grids[storm_var])
            if storm_var not in potential_variables and storm_var not in tendency_variables:
                del model_grids[storm_var]

        for potential_var in potential_variables:
            print("Potential {0} {1} {2}".format(potential_var,self.ensemble_member, self.run_date.strftime("%Y%m%d")))
            if potential_var not in model_grids.keys():
                model_grids[potential_var] = ModelOutput(self.ensemble_name, self.ensemble_member,
                                                         self.run_date, potential_var,
                                                         self.start_date - timedelta(hours=1),
                                                         self.end_date,
                                                         self.model_path, self.single_step)
                model_grids[potential_var].load_data()
            for model_obj in tracked_model_objects:
                model_obj.extract_attribute_grid(model_grids[potential_var], potential=True)
            if potential_var not in tendency_variables:
                del model_grids[potential_var]
        for tendency_var in tendency_variables:
            print("Tendency {0} {1} {2}".format(tendency_var, self.ensemble_member, self.run_date.strftime("%Y%m%d")))
            if tendency_var not in model_grids.keys():
                model_grids[tendency_var] = ModelOutput(self.ensemble_name, self.ensemble_member,
                                                        self.run_date, tendency_var,
                                                        self.start_date - timedelta(hours=1),
                                                        self.end_date,
                                                        self.model_path, self.single_step)
            for model_obj in tracked_model_objects:
                model_obj.extract_tendency_grid(model_grids[tendency_var])
            del model_grids[tendency_var]


    @staticmethod
    def match_hail_sizes(model_tracks, obs_tracks, track_pairings):
        """
        Given forecast and observed track pairings, maximum hail sizes are associated with each paired forecast storm
        track timestep. If the duration of the forecast and observed tracks differ, then interpolation is used for the
        intermediate timesteps.

        :param model_tracks: List of model track STObjects
        :param obs_tracks: List of observed STObjects
        :param track_pairings: list of tuples containing the indices of the paired (forecast, observed) tracks
        :return:
        """
        unpaired = range(len(model_tracks))
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
            print pair[0], "obs",  obs_hail_sizes
            print pair[0], "model", model_track.observations
        for u in unpaired:
            model_tracks[u].observations = np.zeros(model_tracks[u].times.shape)
        return

    def match_size_distributions(self, model_tracks, obs_tracks, track_pairings):
        def match_single_track_dist(model_track, obs_track):
            label_columns = ["Max_Hail_Size", "Shape", "Location", "Scale"]
            obs_hail_dists = pd.DataFrame(index=obs_track.times,
                                          columns=label_columns)
            model_hail_dists = pd.DataFrame(index=model_track.times,
                                            columns=label_columns)
            for t, step in enumerate(obs_track.timesteps):
                step_vals = step[(obs_track.masks[t] == 1) & (obs_track.timesteps[t] > self.mrms_ew.min_thresh)]
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
        unpaired = range(len(model_tracks))
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

    @staticmethod
    def calc_track_errors(model_tracks, obs_tracks, track_pairings):
        """
        Calculates spatial and temporal translation errors between matched
        forecast and observed tracks.

        :param model_tracks: List of model track STObjects
        :param obs_tracks: List of observed track STObjects
        :param track_pairings: List of tuples pairing forecast and observed tracks.
        :return: pandas DataFrame containing different track errors
        """
        columns = ['obs_track_id',
                   'translation_error_x',
                   'translation_error_y',
                   'start_time_difference',
                   'end_time_difference',
                   ]
        track_errors = pd.DataFrame(index=range(len(model_tracks)),
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

