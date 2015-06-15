from data.SSEFModelGrid import SSEFModelGrid
from data.MESHGrid import MESHInterpolatedGrid
from EnhancedWatershedSegmenter import EnhancedWatershed
from ObjectMatcher import ObjectMatcher, TrackMatcher
from scipy.ndimage import find_objects, gaussian_filter
from STObject import STObject
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd


def main():
    return


class TrackProcessor(object):
    """
    TrackProcessor identifies local maxima in a convection-allowing model run and links them in time to form
    storm tracks. A similar procedure is applied to the observations, and the two sets of tracks are matched.
    Storm and environmental attributes are extracted from within the identified track areas.

    :param run_date: Datetime model run was initialized
    :param start_date: Datetime for the beginning of storm extraction.
    :param end_date: Datetime for the ending of storm extraction.
    :param ensemble_member: name of the ensemble member being used.
    :param variable: model variable being used for extraction.
    :param model_path: path to the ensemble output.
    :param model_watershed_params: tuple of parameters used for EnhancedWatershed
    :param object_matcher_params: tuple of parameters used for ObjectMatcher.
    :param track_matcher_params: tuple of parameters for TrackMatcher.
    :param size_filter: minimum size of model objects
    :param gaussian_window: number of grid points
    :param mesh_path: Path to MESH files
    :param mesh_watershed_params: tuple of parameters for Enhanced Watershed applied to MESH data.
    """
    def __init__(self,
                 run_date,
                 start_date,
                 end_date,
                 ensemble_member,
                 variable,
                 model_path,
                 model_watershed_params,
                 object_matcher_params,
                 track_matcher_params,
                 size_filter,
                 gaussian_window,
                 mesh_path=None,
                 mesh_watershed_params=None
                 ):
        self.run_date = run_date
        self.start_date = start_date
        self.end_date = end_date
        self.start_hour = int((self.start_date - self.run_date).total_seconds()) / 3600
        self.end_hour = int((self.end_date - self.run_date).total_seconds()) / 3600
        self.hours = range(self.start_hour, self.end_hour + 1)
        self.ensemble_member = ensemble_member
        self.variable = variable
        self.model_ew = EnhancedWatershed(*model_watershed_params)
        self.object_matcher = ObjectMatcher(*object_matcher_params)
        self.track_matcher = TrackMatcher(*track_matcher_params)
        self.size_filter = size_filter
        self.gaussian_window = gaussian_window
        self.model_path = model_path
        self.mesh_path = mesh_path
        self.model_grid = SSEFModelGrid(self.model_path,
                                        self.ensemble_member,
                                        self.run_date.strftime("%Y%m%d"),
                                        self.start_hour,   
                                        self.end_hour,
                                        self.variable
                                        )
        if self.mesh_path is not None:
            self.mesh_grid = MESHInterpolatedGrid(self.start_date, self.end_date, self.mesh_path)
            self.mesh_ew = EnhancedWatershed(*mesh_watershed_params)
        else:
            self.mesh_grid = None
            self.mesh_ew = None
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
            hour_labels = self.model_ew.size_filter(self.model_ew.label(gaussian_filter(self.model_grid.data[h],
                                                                                        self.gaussian_window)), 
                                                    self.size_filter)
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
                                                      hour))
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
            print "Tracked Model Objects: ", len(tracked_model_objects), " Hour ", hour

        return tracked_model_objects

    def find_mesh_tracks(self):
        """
        Identify objects from MESH timesteps and link them together with object matching.

        :return: list of STObjects containing MESH track information.
        """
        obs_objects = []
        tracked_obs_objects = []
        if self.mesh_ew is not None:
            for h, hour in enumerate(self.hours):
                hour_labels = self.mesh_ew.size_filter(self.mesh_ew.label(gaussian_filter(self.mesh_grid.MESH[h],
                                                                                          self.gaussian_window)),
                                                       self.size_filter)
                obj_slices = find_objects(hour_labels)
                num_slices = len(obj_slices)
                obs_objects.append([])
                if num_slices > 0:
                    for sl in obj_slices:
                        obs_objects[-1].append(STObject(self.mesh_grid.MESH[h][sl],
                                                        np.where(hour_labels[sl] > 0, 1, 0),
                                                        self.model_grid.x[sl],
                                                        self.model_grid.y[sl],
                                                        self.model_grid.i[sl],
                                                        self.model_grid.j[sl],
                                                        hour,
                                                        hour))
        
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
                print "Tracked Obs Objects: ", len(tracked_obs_objects), " Hour ", hour
        return tracked_obs_objects

    def match_tracks(self, model_tracks, obs_tracks):
        """
        Match forecast and observed tracks.

        :param model_tracks: list of STObjects describing forecast tracks
        :param obs_tracks: list of STObjects describing observed tracks
        :return:
        """
        return self.track_matcher.match_tracks(model_tracks, obs_tracks)

    def extract_model_attributes(self, tracked_model_objects, storm_variables, potential_variables):
        """
        Extract model attribute data for each model track. Storm variables are those that describe the model storm
        directly, such as radar reflectivity or updraft helicity. Potential variables describe the surrounding
        environmental conditions of the storm, and should be extracted from the timestep before the storm arrives to
        reduce the chance of the storm contaminating the environmental values. Examples of potential variables include
        CAPE, shear, temperature, and dewpoint.

        :param tracked_model_objects: List of STObjects describing each forecasted storm
        :param storm_variables: List of storm variable names
        :param potential_variables: List of potential variable names.
        """
        for storm_var in storm_variables:
            print storm_var, self.ensemble_member, self.run_date
            storm_grid = SSEFModelGrid(self.model_path, 
                                       self.ensemble_member,
                                       self.run_date.strftime("%Y%m%d"),
                                       self.start_hour,
                                       self.end_hour,
                                       storm_var,
                                       loadMap=False)
            storm_grid.close()
            for model_obj in tracked_model_objects:
                model_obj.extract_attribute_grid(storm_grid)

        for potential_var in potential_variables:
            print potential_var, self.ensemble_member, self.run_date
            potential_grid = SSEFModelGrid(self.model_path,
                                           self.ensemble_member,
                                           self.run_date.strftime("%Y%m%d"),
                                           self.start_hour - 1,
                                           self.end_hour - 1,
                                           potential_var,
                                           loadMap=False)
            potential_grid.close()
            for model_obj in tracked_model_objects:
                model_obj.extract_attribute_grid(potential_grid, potential=True)

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
            obs_hail_sizes = np.array([step[obs_track.masks[t]].max() for t, step in enumerate(obs_track.timesteps)])
            if obs_track.times.size > 1 and model_track.times.size > 1:
                normalized_obs_times = 1.0 / (obs_track.times.max() - obs_track.times.min())\
                    * (obs_track.times - obs_track.times.min())
                normalized_model_times = 1.0 / (model_track.times.max() - model_track.times.min())\
                    * (model_track.times - model_track.times.min())
                hail_interp = interp1d(normalized_obs_times, obs_hail_sizes, bounds_error=False, fill_value=0)
                model_track.observations = hail_interp(normalized_model_times)
            elif obs_track.times.size == 1:
                model_track.observations = np.ones(model_track.times.shape) * obs_hail_sizes[0]
            elif model_track.times.size == 1:
                model_track.observations = np.array([obs_hail_sizes.max()])
        for u in unpaired:
            model_tracks[u].observations = np.zeros(model_tracks[u].times.shape)
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
            obs_track = obs_tracks[pair[1]]
            model_com = model_track.center_of_mass(model_track.start_time)
            obs_com = obs_track.center_of_mass(obs_track.start_time)
            track_errors.loc[pair[0], 'obs_track_id'] = pair[1]
            track_errors.loc[pair[0], 'translation_error_x'] = model_com[0] - obs_com[0]
            track_errors.loc[pair[0], 'translation_error_y'] = model_com[1] - obs_com[1]
            track_errors.loc[pair[0], 'start_time_difference'] = model_track.start_time - obs_track.start_time
            track_errors.loc[pair[0], 'end_time_difference'] = model_track.end_time - obs_track.end_time 
        return track_errors

if __name__ == "__main__":
    main()
